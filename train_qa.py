import os
import argparse
from collections import defaultdict

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

# ----------------------------
# Utilities
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output_dir", type=str, default="artifacts/models/qa_distilbert_squadv2")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")  # enable if you have an NVIDIA GPU that supports fp16
    return p.parse_args()


def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    # Some questions are padded on the left; strip leading whitespace
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

    # Map each tokenized feature back to its original example
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answer, label CLS (no-answer)
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Answer span in character indices
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find tokens that are part of the context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer not fully inside this window, label CLS
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        # Otherwise, find start token
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        # Find end token
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


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
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    # For evaluation, keep example_id and adjust offset mapping so only context tokens have offsets
    tokenized["example_id"] = []
    new_offset_mapping = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        # Set offsets to None for non-context tokens
        new_offsets = []
        for offset, seq_id in zip(offsets, sequence_ids):
            if seq_id == 1:
                new_offsets.append(offset)
            else:
                new_offsets.append(None)
        new_offset_mapping.append(new_offsets)

    tokenized["offset_mapping"] = new_offset_mapping
    return tokenized


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    """
    Convert start/end logits into text answers per example.
    This is a standard HF-style postprocess for extractive QA.
    """
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

        # Also track best null (CLS) score for SQuAD v2 (no answer)
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
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]

                    if score > best_answer["score"]:
                        best_answer = {"score": score, "text": text}

        # SQuAD v2 decision: compare best span vs best null
        # If null score is higher, predict empty string
        if best_null_score is not None and best_null_score > best_answer["score"]:
            predictions[example["id"]] = ""
        else:
            predictions[example["id"]] = best_answer["text"]

    return predictions


def compute_metrics_fn(tokenizer, eval_examples, eval_features):
    squad_metric = evaluate.load("squad_v2")

    def compute_metrics(p):
        # p.predictions is a tuple: (start_logits, end_logits)
        predictions = postprocess_qa_predictions(
            eval_examples,
            eval_features,
            p.predictions,
            tokenizer=tokenizer,
        )

        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                                 for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

        return squad_metric.compute(predictions=formatted_predictions, references=references)

    return compute_metrics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading dataset: squad_v2")
    raw_datasets = load_dataset("squad_v2")

    print(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # Preprocess
    print("Tokenizing train split...")
    train_dataset = raw_datasets["train"].map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Preparing train features",
    )

    print("Tokenizing validation split...")
    eval_examples = raw_datasets["validation"]
    eval_features = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Preparing validation features",
    )

    # Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=args.fp16,
        report_to="none",
        load_best_model_at_end=False,   # <-- important
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_features,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_fn(tokenizer, eval_examples, eval_features),
    )

    print("Training...")
    trainer.train()

    print("Running prediction on validation features...")
    preds = trainer.predict(eval_features)

    print("Post-processing predictions into text answers...")
    predictions = postprocess_qa_predictions(
        eval_examples,
        eval_features,
        preds.predictions,
        tokenizer=tokenizer,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
        for k, v in predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad_v2")
    final_metrics = squad_metric.compute(predictions=formatted_predictions, references=references)
    print("Final SQuAD v2 metrics:", final_metrics)

    print("Evaluating best checkpoint...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
