import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, default_data_collator


# -----------------------------
# Labels + minimal metadata + features
# -----------------------------
@dataclass
class ExampleResult:
    # identifiers / metadata (not used as features)
    ex_id: str
    question: str
    context_len: int
    gold_has_answer: int   # keep for analysis only (DO NOT use as feature)
    pred_text: str
    pred_is_no_answer: int

    # eval targets
    exact: int
    f1: float
    trustworthy_em: int
    trustworthy_f1_80: int

    # -------- Feature set A (minimal+strong) --------
    best_span_score: float
    null_score: float
    margin: float  # best_span_score - null_score

    start_entropy: float
    end_entropy: float
    start_top2_gap: float
    end_top2_gap: float

    answer_len_tokens: int
    answer_len_frac_of_context: float

    # -------- Feature set B (richer, still sane) --------
    p_start_max: float
    p_end_max: float
    start_top5_mass: float
    end_top5_mass: float
    is_margin_positive: int


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=4)

    # Instead of a single max_samples, we support (offset, n) so you can carve train/val/test
    p.add_argument("--max_samples", type=int, default=None, help="Number of examples to include (after offset).")
    p.add_argument("--offset", type=int, default=0, help="Start index into the split before taking max_samples.")

    p.add_argument("--out_csv", type=str, default="artifacts/reliability/reliability_dataset.csv")
    p.add_argument(
        "--null_threshold",
        type=float,
        default=0.0,
        help="Predict no-answer if (null_score - best_span_score) > this threshold (SQuADv2 style)."
    )
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exps = np.exp(x)
    denom = np.sum(exps)
    if denom == 0:
        return np.ones_like(x) / len(x)
    return exps / denom


def entropy_from_logits(logits: np.ndarray) -> float:
    p = stable_softmax(logits)
    return float(-(p * np.log(p + 1e-12)).sum())


def top2_gap(logits: np.ndarray) -> float:
    vals = np.sort(logits)[::-1]
    if len(vals) < 2:
        return 0.0
    return float(vals[0] - vals[1])


def topk_mass_from_logits(logits: np.ndarray, k: int) -> float:
    p = stable_softmax(logits)
    if len(p) == 0:
        return 0.0
    k = min(k, len(p))
    topk = np.sort(p)[::-1][:k]
    return float(np.sum(topk))


def pmax_from_logits(logits: np.ndarray) -> float:
    p = stable_softmax(logits)
    return float(np.max(p)) if len(p) else 0.0


def f1_score_simple(pred: str, gold: str) -> float:
    pred_toks = pred.strip().split()
    gold_toks = gold.strip().split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0

    counts: Dict[str, int] = {}
    for t in pred_toks:
        counts[t] = counts.get(t, 0) + 1

    overlap = 0
    for t in gold_toks:
        if counts.get(t, 0) > 0:
            overlap += 1
            counts[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return float(2 * precision * recall / (precision + recall))


def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip() == gold.strip())


# -----------------------------
# Tokenization -> features
# -----------------------------
def prepare_features(examples, tokenizer, max_length, doc_stride):
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = []
    new_offset_mapping = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        # keep offsets only for context tokens
        new_offsets = []
        for offset, seq_id in zip(offsets, sequence_ids):
            new_offsets.append(offset if seq_id == 1 else None)
        new_offset_mapping.append(new_offsets)

    tokenized["offset_mapping"] = new_offset_mapping
    return tokenized


# -----------------------------
# Postprocess best span + null
# -----------------------------
def postprocess_best_span_for_example(
    example,
    features_for_ex,
    start_logits_all,
    end_logits_all,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
    null_score_diff_threshold=0.0
) -> Tuple[str, float, float, int]:
    """
    Returns:
      (pred_text, best_span_score, best_null_score, best_window_index)
    best_window_index is the feature window that produced the best non-null span.
    If we abstain, best_window_index is still set based on best span window (for uncertainty features).
    """
    context = example["context"]
    best_answer = {"score": -1e18, "text": "", "win": 0}
    best_null_score = -1e18

    for fi, feat in enumerate(features_for_ex):
        start_logits = start_logits_all[fi]
        end_logits = end_logits_all[fi]
        offsets = feat["offset_mapping"]
        input_ids = feat["input_ids"]

        cls_index = input_ids.index(tokenizer.cls_token_id)
        null_score = float(start_logits[cls_index] + end_logits[cls_index])
        if null_score > best_null_score:
            best_null_score = null_score

        start_idxs = np.argsort(start_logits)[-n_best_size:][::-1]
        end_idxs = np.argsort(end_logits)[-n_best_size:][::-1]

        for s in start_idxs:
            for e in end_idxs:
                if offsets[s] is None or offsets[e] is None:
                    continue
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue

                start_char = offsets[s][0]
                end_char = offsets[e][1]
                text = context[start_char:end_char]
                score = float(start_logits[s] + end_logits[e])

                if score > best_answer["score"]:
                    best_answer = {"score": score, "text": text, "win": fi}

    best_text = best_answer["text"]
    best_span_score = float(best_answer["score"])
    best_null_score = float(best_null_score)
    best_window_index = int(best_answer["win"])

    # abstain if null beats best span by threshold:
    # (null - span) > threshold => no-answer
    if (best_null_score - best_span_score) > null_score_diff_threshold:
        return "", best_span_score, best_null_score, best_window_index

    return best_text, best_span_score, best_null_score, best_window_index


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    ds = load_dataset("squad_v2")
    split = ds[args.split]

    # apply offset + cap
    if args.offset < 0 or args.offset >= len(split):
        raise ValueError(f"--offset {args.offset} is out of range for split size {len(split)}")

    if args.max_samples is None:
        split = split.select(range(args.offset, len(split)))
    else:
        end = min(args.offset + args.max_samples, len(split))
        split = split.select(range(args.offset, end))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)

    features = split.map(
        lambda x: prepare_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=split.column_names,
        desc=f"Preparing {args.split} features",
    )

    trainer = Trainer(model=model, data_collator=default_data_collator)

    print("Predicting logits...")
    preds = trainer.predict(features)
    start_logits_all, end_logits_all = preds.predictions  # (num_features, seq_len)

    feat_example_ids = features["example_id"]
    feat_input_ids = features["input_ids"]
    feat_offsets = features["offset_mapping"]

    ex_to_feat_idxs = defaultdict(list)
    for i, ex_id in enumerate(feat_example_ids):
        ex_to_feat_idxs[ex_id].append(i)

    rows: List[ExampleResult] = []

    for ex in split:
        ex_id = ex["id"]
        q = ex["question"]
        context = ex["context"]
        gold_answers = ex["answers"]["text"]
        gold_has_answer = 1 if len(gold_answers) > 0 else 0

        feat_idxs = ex_to_feat_idxs.get(ex_id, [])
        if not feat_idxs:
            continue

        feats_for_ex = []
        start_logits_for_ex = []
        end_logits_for_ex = []

        for idx in feat_idxs:
            feats_for_ex.append({
                "input_ids": feat_input_ids[idx],
                "offset_mapping": feat_offsets[idx],
            })
            start_logits_for_ex.append(start_logits_all[idx])
            end_logits_for_ex.append(end_logits_all[idx])

        start_logits_for_ex = np.asarray(start_logits_for_ex)
        end_logits_for_ex = np.asarray(end_logits_for_ex)

        pred_text, best_span_score, null_score, best_w = postprocess_best_span_for_example(
            ex,
            feats_for_ex,
            start_logits_for_ex,
            end_logits_for_ex,
            tokenizer,
            null_score_diff_threshold=args.null_threshold,
        )

        pred_is_no_answer = int(pred_text.strip() == "")

        # Metrics against gold
        if gold_has_answer:
            em = max(exact_match(pred_text, ga) for ga in gold_answers)
            f1 = max(f1_score_simple(pred_text, ga) for ga in gold_answers)
        else:
            em = int(pred_is_no_answer == 1)
            f1 = float(em)

        # Labeling logic (your spec)
        if gold_has_answer and pred_is_no_answer:
            trustworthy_em = 0
            trustworthy_f1_80 = 0
        elif (not gold_has_answer) and pred_is_no_answer:
            trustworthy_em = 1
            trustworthy_f1_80 = 1
        else:
            trustworthy_em = int(em == 1)
            trustworthy_f1_80 = int(f1 >= 0.8)

        # Representative window for uncertainty features:
        # Use the window that produced the best span (best_w)
        start_logits_w = start_logits_for_ex[best_w]
        end_logits_w = end_logits_for_ex[best_w]

        # ---- Feature Set A ----
        margin = float(best_span_score - null_score)  # IMPORTANT: fixed sign

        start_ent = entropy_from_logits(start_logits_w)
        end_ent = entropy_from_logits(end_logits_w)
        start_gap = top2_gap(start_logits_w)
        end_gap = top2_gap(end_logits_w)

        answer_len_tokens = int(len(pred_text.strip().split())) if pred_text.strip() else 0
        context_len_chars = max(1, len(context))
        answer_len_frac_of_context = float(len(pred_text) / context_len_chars) if pred_text else 0.0

        # ---- Feature Set B ----
        p_start_max = pmax_from_logits(start_logits_w)
        p_end_max = pmax_from_logits(end_logits_w)
        start_top5_mass = topk_mass_from_logits(start_logits_w, k=5)
        end_top5_mass = topk_mass_from_logits(end_logits_w, k=5)
        is_margin_positive = int(margin > 0.0)

        rows.append(ExampleResult(
            ex_id=ex_id,
            question=q,
            context_len=len(context),
            gold_has_answer=gold_has_answer,
            pred_text=pred_text,
            pred_is_no_answer=pred_is_no_answer,
            exact=int(em),
            f1=float(f1),
            trustworthy_em=int(trustworthy_em),
            trustworthy_f1_80=int(trustworthy_f1_80),

            best_span_score=float(best_span_score),
            null_score=float(null_score),
            margin=margin,

            start_entropy=start_ent,
            end_entropy=end_ent,
            start_top2_gap=start_gap,
            end_top2_gap=end_gap,

            answer_len_tokens=answer_len_tokens,
            answer_len_frac_of_context=answer_len_frac_of_context,

            p_start_max=p_start_max,
            p_end_max=p_end_max,
            start_top5_mass=start_top5_mass,
            end_top5_mass=end_top5_mass,
            is_margin_positive=is_margin_positive,
        ))

    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}  (rows={len(df)})")
    for col in ["trustworthy_em", "trustworthy_f1_80"]:
        if col in df.columns:
            print(f"{col} positive rate: {df[col].mean():.3f}")
    print("Gold has answer rate:", df["gold_has_answer"].mean())
    print("Predicted no-answer rate:", df["pred_is_no_answer"].mean())


if __name__ == "__main__":
    main()
