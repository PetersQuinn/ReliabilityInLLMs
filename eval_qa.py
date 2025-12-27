import argparse
import numpy as np
from collections import defaultdict

from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, default_data_collator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True, help="Path to a checkpoint folder or final model folder")
    p.add_argument("--model_name_fallback", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def prepare_validation_features(examples, tokenizer, max_length, doc_stride):
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
        return_tensors=None,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = []
    new_offset_mapping = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        # Only keep offsets for context tokens
        new_offsets = []
        for offset, seq_id in zip(offsets, sequence_ids):
            new_offsets.append(offset if seq_id == 1 else None)
        new_offset_mapping.append(new_offsets)

    tokenized["offset_mapping"] = new_offset_mapping
    return tokenized


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)

    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        best_answer = {"score": -1e9, "text": ""}
        best_null_score = None

        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offsets = features[fi]["offset_mapping"]
            input_ids = features[fi]["input_ids"]

            cls_index = input_ids.index(tokenizer.cls_token_id)
            null_score = start_logits[cls_index] + end_logits[cls_index]
            if best_null_score is None or null_score > best_null_score:
                best_null_score = null_score

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]

                    if score > best_answer["score"]:
                        best_answer = {"score": score, "text": text}

        # SQuAD v2: choose no-answer if null is better
        if best_null_score is not None and best_null_score > best_answer["score"]:
            predictions[example["id"]] = ""
        else:
            predictions[example["id"]] = best_answer["text"]

    return predictions


def main():
    args = parse_args()
    dataset = load_dataset("squad_v2")
    eval_examples = dataset["validation"]

    # Load tokenizer/model from the fine-tuned checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_dir)

    eval_features = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Preparing validation features",
    )

    trainer = Trainer(
        model=model,
        data_collator=default_data_collator,
    )

    print("Running prediction on validation features...")
    preds = trainer.predict(eval_features)
    predictions = postprocess_qa_predictions(eval_examples, eval_features, preds.predictions, tokenizer)

    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    metric = evaluate.load("squad_v2")
    final_metrics = metric.compute(predictions=formatted_predictions, references=references)
    print("SQuAD v2 metrics:", final_metrics)


if __name__ == "__main__":
    main()
