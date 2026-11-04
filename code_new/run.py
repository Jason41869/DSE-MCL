# #!/usr/bin/python3

#

import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
from data_helper import TrainDataset, construct_kg, get_kg
import hydra
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
import pickle
from os.path import join
from model_helper import train_step, evaluate
from model import DSE-MCL
import shutil
from umls_case_study import print_stream_top3
from umls_case_study import print_top5

def get_unique_checkpoint_name(cfg):
    """
    生成唯一的 Checkpoint 文件名（仅保留核心标识，简化命名）
    """
    # 简化命名：只保留数据集+时间戳（减少长度，避免路径问题）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"best_model_{cfg.dataset}_{timestamp}.torch"
    config_filename = f"best_config_{cfg.dataset}_{timestamp}.pickle"

    # 确保保存目录存在且有写入权限
    save_dir = "DSE-MCL-main/DSE-MCL/model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 检查目录可写性
    if not os.access(save_dir, os.W_OK):
        raise PermissionError(f"目录 {save_dir} 无写入权限！")

    checkpoint_path = os.path.join(save_dir, filename)
    config_path = os.path.join(save_dir, config_filename)
    return checkpoint_path, config_path


def save_model(model, save_variables, old_ckpt_path=None, old_cfg_path=None):
    """
    保存最优模型（只保留最后一次最佳）
    :param old_ckpt_path: 旧的最优模型路径（用于删除）
    :param old_cfg_path: 旧的配置文件路径（用于删除）
    """
    cfg = utils.get_global_config()

    # 1. 删除旧的最优模型（如果存在）
    if old_ckpt_path and os.path.exists(old_ckpt_path):
        try:
            os.remove(old_ckpt_path)
        except Exception as e:
            logging.warning(f"删除旧模型失败：{e}")
    if old_cfg_path and os.path.exists(old_cfg_path):
        try:
            os.remove(old_cfg_path)
        except Exception as e:
            logging.warning(f"删除旧配置失败：{e}")

    # 2. 生成新路径
    new_ckpt_path, new_cfg_path = get_unique_checkpoint_name(cfg)

    # 3. 保存配置
    try:
        pickle.dump(cfg, open(new_cfg_path, 'wb'))
    except Exception as e:
        raise RuntimeError(f"保存配置文件失败：{e}")

    # 4. 保存模型（捕获写入异常）
    state_dict = {
        'model_state_dict': model.state_dict(),
        **save_variables
    }
    try:
        torch.save(state_dict, new_ckpt_path)
    except Exception as e:
        raise RuntimeError(f"保存模型失败：{e}")

    return new_ckpt_path, new_cfg_path


def get_linear_scheduler_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    """
    Create scheduler with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        """
        Compute a ratio according to current step,
        by which the optimizer's lr will be mutiplied.
        :param current_step:
        :return:
        """
        assert current_step <= max_steps

        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return (max_steps - current_step) / (max_steps - warmup_steps)

    assert max_steps >= warmup_steps

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def format_metrics(name, h_metric, t_metric):
    msg_h = name + ' (head) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_t = name + ' (tail) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_avg = name + ' (avg) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'

    msg_h = msg_h.format(h_metric['MRR'], h_metric['MR'],
                         h_metric['HITS@1'], h_metric['HITS@3'], h_metric['HITS@10'])
    msg_t = msg_t.format(t_metric['MRR'], t_metric['MR'],
                         t_metric['HITS@1'], t_metric['HITS@3'], t_metric['HITS@10'])
    msg_avg = msg_avg.format(
        (h_metric['MRR'] + t_metric['MRR']) / 2,
        (h_metric['MR'] + t_metric['MR']) / 2,
        (h_metric['HITS@1'] + t_metric['HITS@1']) / 2,
        (h_metric['HITS@3'] + t_metric['HITS@3']) / 2,
        (h_metric['HITS@10'] + t_metric['HITS@10']) / 2
    )
    return msg_h, msg_t, msg_avg


@hydra.main(config_path=join('..', 'config'), config_name="config")
def main(config: DictConfig):
    utils.set_global_config(config)
    cfg = utils.get_global_config()
    assert cfg.dataset in cfg.dataset_list

    utils.remove_randomness()

    logging.info('\n------Config------\n {}'.format(utils.filter_config(cfg)))

    code_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(code_dir_path)
    config_dir_path = os.path.join(project_dir_path, 'config')
    hydra_current_dir = os.getcwd()
    logging.info('Code dir path: {}'.format(code_dir_path))
    logging.info('Config dir path: {}'.format(config_dir_path))
    logging.info('Model save path: {}'.format(hydra_current_dir))
    os.system('cp -r {} {}'.format(code_dir_path, hydra_current_dir))
    os.system('cp -r {} {}'.format(config_dir_path, hydra_current_dir))

    device = torch.device(cfg.device)
    model = DSE-MCL(cfg.h_dim)
    model = model.to(device)

    src, dst, rel, hr2eid, rt2eid = construct_kg('train', directed=False)
    kg = get_kg(src, dst, rel, device)


    #in_degrees = kg.in_degrees(range(kg.number_of_nodes()))
    #nodes_with_indegree_lt_2 = [node for node, in_degree in enumerate(in_degrees) if in_degree > 170.8]

    kg_out_deg = kg.out_degrees(torch.arange(kg.number_of_nodes(), device=device))
    kg_zero_deg_num = torch.sum(kg_out_deg < 1).to(torch.int).item()
    logging.info('kg # node: {}'.format(kg.number_of_nodes()))
    logging.info('kg # edge: {}'.format(kg.number_of_edges()))
    logging.info('kg # zero deg node: {}'.format(kg_zero_deg_num))

    train_loader = DataLoader(
        TrainDataset('train', hr2eid, rt2eid),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.cpu_worker_num,
        collate_fn=TrainDataset.collate_fn
    )

    logging.info('-----Model Parameter Configuration-----')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))

    n_epoch = cfg.epoch
    single_epoch_step = len(train_loader)
    max_steps = n_epoch * single_epoch_step
    warm_up_steps = int(single_epoch_step * cfg.warmup_epoch)
    init_lr = cfg.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=init_lr,
    )
    scheduler = get_linear_scheduler_with_warmup(optimizer, warm_up_steps, max_steps)

    logging.info('Training... total epoch: {0}, step: {1}'.format(n_epoch, max_steps))
    last_improve_epoch = 0
    best_test_mrr = 0.0
    best_val_mrr = 0.0
    best_test_full_result = None
    best_test_metrics = None
    old_best_ckpt = None
    old_best_cfg = None

    cfg = utils.get_global_config()


    for epoch in range(n_epoch):
        loss_list = []
        cl_loss_list = []
        model.train()
        for batch_data in train_loader:
            train_log = train_step(model, batch_data, kg, optimizer, scheduler)
            loss_list.append(train_log['loss'])
            cl_loss_list.append(train_log['cl_loss'])
            # get a new kg, since in previous kg some edges are removed.
            if cfg.rm_rate > 0:
                kg = get_kg(src, dst, rel, device)

        model.eval()



        with torch.no_grad():
            val_metrics = evaluate(model, set_flag='valid', kg=kg)['average']
            test_metrics = evaluate(model, set_flag='test', kg=kg)['average']
        if val_metrics['MRR'] > best_val_mrr:
            best_val_mrr = val_metrics['MRR']

            save_variables = {
                'best_val_metrics': val_metrics,
                'best_test_metrics': test_metrics,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_mrr': best_val_mrr,
                'best_test_mrr': best_test_mrr
            }
            old_best_ckpt, old_best_cfg = save_model(
                model, save_variables,
                old_ckpt_path=old_best_ckpt,
                old_cfg_path=old_best_cfg
            )
            improvement_flag = '*'
            last_improve_epoch = epoch
        else:
            improvement_flag = ''
        current_test_mrr_for_log = 0.0
        if improvement_flag == '*':
            # 跑完整测试集
            full_test_result = evaluate(model, set_flag='test', kg=kg)
            current_test_mrr_for_log = full_test_result['average']['MRR']

            # 如果是历史最高 TEST MRR，则保存
            if current_test_mrr_for_log > best_test_mrr:
                best_test_mrr = current_test_mrr_for_log
                best_test_full_result = full_test_result  # 保存完整结果


        val_msg = 'Val - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f} | '.format(
            val_metrics['MRR'], val_metrics['MR'],
            val_metrics['HITS@1'], val_metrics['HITS@3'], val_metrics['HITS@10']
        )

        if improvement_flag == '*':
            val_msg += 'Test - MRR: {:5.4f} | '.format(current_test_mrr_for_log)

        val_msg += improvement_flag

        msg = 'Epoch: {:3d} | Loss: {:5.4f} | CL_Loss: {:5.4f} | '.format(
            epoch, np.mean(loss_list), np.mean(cl_loss_list)
        )
        msg += val_msg
        logging.info(msg)

        # whether early stopping
        if epoch - last_improve_epoch > cfg.max_no_improve:
            logging.info("Long time no improvenment, stop training...")
            break

    logging.info('Training end...')

    # evaluate train and test set
    # load best model
    if old_best_ckpt and os.path.exists(old_best_ckpt):
        try:
            checkpoint = torch.load(old_best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"✅ 成功加载最优模型：{old_best_ckpt}")
            logging.info(
                f"📌 最优模型来自Epoch：{checkpoint.get('epoch', '未知')}，最优MRR：{checkpoint.get('best_test_mrr', 0):.4f}")

            src, dst, rel, _, _ = construct_kg("train", directed=False)
            case_kg = get_kg(src, dst, rel, device)

            print_stream_top3(
                model,
                case_kg,
                entity="disease_or_syndrome",
                layer_index=-1,
            )
            print_top5(
                model,
                case_kg,
                head="disease_or_syndrome",
                relation="affects",
                target="organ_or_tissue_function"
            )
        except Exception as e:
            logging.error(f"❌ 加载最优模型失败：{e}")
    else:
        logging.warning("⚠️ 未找到最优模型文件，使用最后一轮模型评估！")

    model.eval()

    logging.info('Train metrics ...')
    train_metrics = evaluate(model, 'train', kg=kg)
    train_msg = format_metrics('Train', train_metrics['head_batch'], train_metrics['tail_batch'])
    logging.info(train_msg[0])
    logging.info(train_msg[1])
    logging.info(train_msg[2] + '\n')

    logging.info('Valid metrics ...')
    valid_metrics = evaluate(model, 'valid', kg=kg)
    valid_msg = format_metrics('Valid', valid_metrics['head_batch'], valid_metrics['tail_batch'])
    logging.info(valid_msg[0])
    logging.info(valid_msg[1])
    logging.info(valid_msg[2] + '\n')

    logging.info('Test metrics...')
    if best_test_full_result is not None:
        test_msg = format_metrics('Test', best_test_full_result['head_batch'], best_test_full_result['tail_batch'])
    else:
        full_test_result = evaluate(model, set_flag='test', kg=kg)
        test_msg = format_metrics('Test', full_test_result['head_batch'], full_test_result['tail_batch'])
    logging.info(test_msg[0])
    logging.info(test_msg[1])
    logging.info(test_msg[2] + '\n')

    logging.info('Model save path: {}'.format(os.getcwd()))


if __name__ == '__main__':
    main()
