"""Print stream attention Top-3 and filtered predictions for UMLS, FB15k-237 or WN18RR."""

import argparse
import pickle
import sys
from pathlib import Path


def print_stream_top3(model, kg, entity="disease_or_syndrome", layer_index=-1):
    """Reproduce CompLayer's edge selection and attention, then rank neighbors.

    Scores are mean-over-head attention, summed over parallel incoming edges.
    Uses the current model.py's S/L inputs (unchanged between layers in eval).
    These are encoder attention weights, not relation-conditioned predictions.
    """
    import dgl
    import torch
    import utils
    from data_helper import construct_dict

    cfg = utils.get_global_config()
    if cfg.dataset not in ("UMLS", "FB15k_237", "WN18RR"):
        raise ValueError("Expected a UMLS, FB15k_237 or WN18RR model configuration.")
    ent2id, _ = construct_dict(str(Path(cfg.dataset_dir) / cfg.dataset))
    id2ent = {idx: name for name, idx in ent2id.items()}
    target = ent2id[entity]
    layer = model.comp_layers[layer_index]
    device = next(model.parameters()).device
    was_training = model.training
    result = {}
    try:
        model.eval()
        with torch.no_grad():
            rel_emb = model.rel_drop(model.rel_embs[layer_index])
            streams = {
                "common": model.com_drop(model.S(model.ent_emb)),
                "private": model.pri_drop(model.L(model.ent_emb)),
            }
            for stream, emb in streams.items():
                # Match CompLayer: first attention head selects incoming top-k edges.
                with kg.local_scope():
                    src, dst = kg.edges(order="eid")
                    rid = kg.edata["rel_id"]
                    projected = emb @ layer.W[0]
                    rel_projected = rel_emb[rid] @ layer.W_r[0]
                    attn_input = torch.cat(
                        [projected[src], projected[dst], rel_projected], dim=1)
                    kg.edata["score_samp"] = layer.LeakyReLU(
                        (attn_input @ layer.a[0]).squeeze(-1))
                    sampled = dgl.sampling.select_topk(
                        kg.to("cpu"), layer.topk, "score_samp", edge_dir="in").to(device)
                src, dst = sampled.edges(order="eid")
                mask = dst == target
                src, dst = src[mask], dst[mask]
                rid = sampled.edata["rel_id"][mask]
                if src.numel() == 0:
                    print(f"{stream}: no selected incoming neighbors")
                    result[stream] = []
                    continue
                weights = []
                for head in range(layer.n_heads):
                    projected = emb @ layer.W[head]
                    rel_projected = rel_emb[rid] @ layer.W_r[head]
                    attn_input = torch.cat(
                        [projected[src], projected[dst], rel_projected], dim=1)
                    logits = layer.LeakyReLU((attn_input @ layer.a[head]).squeeze(-1))
                    # All edges here enter the same target: this is edge_softmax.
                    weights.append(torch.softmax(logits, dim=0))
                edge_weights = torch.stack(weights).mean(dim=0)
                neighbors, inverse = torch.unique(src, sorted=True, return_inverse=True)
                scores = torch.zeros(neighbors.numel(), device=device)
                scores.scatter_add_(0, inverse, edge_weights)
                values, positions = torch.topk(scores, k=min(10, neighbors.numel()))
                rows = [(id2ent[idx], value) for idx, value in
                        zip(neighbors[positions].tolist(), values.tolist())]
                result[stream] = rows
                print(f"\n{stream.upper()} | entity={entity} | "
                      f"layer={layer_index % len(model.comp_layers) + 1} | "
                      f"selected edges={src.numel()} | unique neighbors={neighbors.numel()}")
                print("Rank\tNeighbor entity\tAttention score")
                for rank, (name, score) in enumerate(rows, 1):
                    print(f"{rank}\t{name}\t{score:.8f}")
    finally:
        model.train(was_training)
    return result


def print_top5(model, kg, head="disease_or_syndrome", relation="affects",
               target="organ_or_tissue_function"):
    """Print target ranks and filtered Top-3; return (name, original score).

    Mask other known tails for this exact (head, relation) across all splits.
    The target remains eligible. Neither the graph nor attention is modified.
    """
    import torch
    import utils
    from data_helper import construct_dict, read_data

    cfg = utils.get_global_config()
    if cfg.dataset not in ("UMLS", "FB15k_237", "WN18RR"):
        raise ValueError("Expected a UMLS, FB15k_237 or WN18RR model configuration.")
    ent2id, rel2id = construct_dict(str(Path(cfg.dataset_dir) / cfg.dataset))
    id2ent = {idx: name for name, idx in ent2id.items()}
    h, r = ent2id[head], rel2id[relation]
    target_id = ent2id[target]
    split_tails = {}
    for split in ("train", "valid", "test"):
        split_tails[split] = read_data(split)["pos_tails"].get((h, r), set())
    known_tails = set().union(*split_tails.values())
    if target_id not in known_tails:
        raise ValueError(f"Target triple is not a known answer: ({head}, {relation}, {target})")
    device = next(model.parameters()).device
    was_training = model.training
    # Current HDRL computes an auxiliary contrastive loss during aggregation,
    # even in eval mode, and requires these IDs to be initialized.
    saved_ids = {name: getattr(model, name) for name in
                 ("current_h_ids", "current_t_ids") if hasattr(model, name)}
    try:
        model.eval()
        for name in saved_ids:
            # These IDs only feed the unused auxiliary contrastive loss.
            # Avoid an all-entity quadratic loss matrix on FB15k-237.
            setattr(model, name, torch.tensor([h, target_id], device=device))
        with torch.no_grad():
            ent_emb, rel_emb, *_ = model.aggragate_emb(kg)
            scores = model.predictor(ent_emb[h:h + 1], rel_emb[r:r + 1], ent_emb)[0]
            if scores.numel() != len(ent2id) or not torch.isfinite(scores).all():
                raise ValueError("Invalid candidate score vector.")
    finally:
        for name, value in saved_ids.items():
            setattr(model, name, value)
        model.train(was_training)

    filtered_scores = scores.clone()
    excluded = sorted(known_tails - {target_id})
    if excluded:
        filtered_scores[excluded] = -float("inf")

    def average_rank(values):
        target_score = values[target_id]
        rank_min = (values > target_score).sum().item() + 1
        rank_max = (values >= target_score).sum().item()
        return (rank_min + rank_max) / 2

    print(f"\nQuery: ({head}, {relation}, ?)")
    print(f"Target: {target}")
    print(f"Target score: {scores[target_id].item():.8f}")
    print(f"Raw rank: {average_rank(scores):.1f}")
    print(f"Filtered rank: {average_rank(filtered_scores):.1f}")
    print(f"Filtered out: {len(excluded)} other known answers")
    print("Ranking: filtered / target retained; Score: unchanged predictor output")
    k = min(5, int(torch.isfinite(filtered_scores).sum().item()))
    _, indices = torch.topk(filtered_scores, k=k)
    print("Rank\tCandidate entity\tScore\tKnown in split\tTarget")
    results = []
    for rank, idx in enumerate(indices.tolist(), 1):
        value = scores[idx].item()
        name = id2ent[idx]
        known = ",".join(s for s, tails in split_tails.items() if idx in tails) or "unobserved"
        mark = "*" if idx == target_id else ""
        print(f"{rank}\t{name}\t{value:.8f}\t{known}\t{mark}")
        results.append((name, value))
    print("Test-set answers: " + ", ".join(sorted(id2ent[t] for t in split_tails["test"])))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path,
                        help="Matching best_config_*.pickle saved by run.py (trusted file only)")
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Model source directory used for training this checkpoint")
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--entity")
    parser.add_argument("--relation")
    parser.add_argument("--target",
                        help="Known target tail to retain during filtered ranking")
    parser.add_argument("--test-index", type=int,
                        help="Zero-based test triple index; FB15k_237 and WN18RR default to 0")
    parser.add_argument("--layer-index", type=int, default=-1,
                        help="Zero-based encoder layer index; -1 selects the last layer")
    args = parser.parse_args()
    for path in (args.checkpoint, args.config):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    sys.path.insert(0, str(args.code_dir.resolve()))
    import torch
    import utils
    from data_helper import construct_dict, construct_kg, get_kg, read_data
    from model import HDRL

    with args.config.open("rb") as stream:
        cfg = pickle.load(stream)
    if cfg.dataset not in ("UMLS", "FB15k_237", "WN18RR"):
        parser.error("The saved configuration must use UMLS, FB15k_237 or WN18RR.")
    cfg.dataset_dir = str(args.dataset_dir.resolve())
    cfg.device = args.device
    utils.set_global_config(cfg)
    utils.remove_randomness()
    supplied = (args.entity, args.relation, args.target)
    if any(x is not None for x in supplied):
        if not all(x is not None for x in supplied) or args.test_index is not None:
            parser.error("Supply entity/relation/target together, or use --test-index.")
    elif args.test_index is not None or cfg.dataset in ("FB15k_237", "WN18RR"):
        index = args.test_index if args.test_index is not None else 0
        test = read_data("test")
        if not 0 <= index < len(test["src_list"]):
            parser.error("--test-index is outside the test set.")
        ent2id, rel2id = construct_dict(str(Path(cfg.dataset_dir) / cfg.dataset))
        id2ent = {v: k for k, v in ent2id.items()}
        id2rel = {v: k for k, v in rel2id.items()}
        args.entity = id2ent[test["src_list"][index]]
        args.relation = id2rel[test["rel_list"][index]]
        args.target = id2ent[test["dst_list"][index]]
        print(f"Selected test triple index: {index}")
    else:
        args.entity, args.relation, args.target = (
            "disease_or_syndrome", "affects", "organ_or_tissue_function")
    print(f"Dataset: {cfg.dataset}")
    print(f"Case: ({args.entity}, {args.relation}, {args.target})")
    device = torch.device(args.device)
    model = HDRL(cfg.h_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Only training edges participate in message passing.
    src, dst, rel, *_ = construct_kg("train", directed=False)
    kg = get_kg(src, dst, rel, device)
    print_stream_top3(model, kg, entity=args.entity, layer_index=args.layer_index)
    print_top5(model, kg, head=args.entity, relation=args.relation, target=args.target)


if __name__ == "__main__":
    main()
