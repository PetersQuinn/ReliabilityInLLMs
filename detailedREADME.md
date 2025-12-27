# Extended README (Outline) — Inference-Time Reliability Estimation for LLM Answers

> **Goal of this document:** a detailed, end-to-end walkthrough of the full pipeline (QA model → dataset → features → trust head training → calibration → selective evaluation), written for ML internship reviewers.

---

## 0) TL;DR (what a reviewer should know in 60 seconds)

* We estimate LLM answer reliability **using only inference-time signals** from a *single forward pass*.
* We **rank** answers by trustworthiness (selective prediction), rather than trying to perfectly classify every answer.
* A lightweight **Logistic Regression** trust head matches tree models in selective performance (signal-limited regime).
* Calibration (sigmoid/isotonic) improves probability alignment (Brier/ECE) but changes ranking minimally.

**Figures to include (final doc):**

1. Selective curve (Oracle vs baseline vs trust head)
2. Reliability diagram (raw vs calibrated)
3. Optional ROC curve

---

## 1) Project Motivation and Problem Setup

### 1.1 Why reliability estimation matters

* LLMs can be fluent but wrong (hallucinations / overconfident errors).
* In production, you often need to **prioritize** what to trust, not make an all-or-nothing call.

### 1.2 Reframing: from classification to ranking

* Classic: “Is this answer correct?”
* Our framing: “Among many answers, **which should we trust first**?”
* This maps to **coverage vs answered-trust-rate** (selective prediction curves).

### 1.3 Definitions

* **Coverage:** fraction of answers you choose to “accept/answer.”
* **Answered trust-rate:** among accepted answers, proportion that are actually trustworthy.
* **Oracle:** perfect ranker (sort by ground truth label).

## 2. Base Question Answering Model

### 2.1 Dataset

The base question answering system is trained on the **SQuAD v2** dataset using the HuggingFace `datasets` library. SQuAD v2 extends the original SQuAD task by including unanswerable questions, which is critical for reliability research since the model must explicitly learn to abstain.

The dataset is loaded as:

```
raw_datasets = load_dataset("squad_v2")
```

Training is performed on the `train` split, while all reliability analysis is derived from the `validation` split.

---

### 2.2 Model Architecture

The QA backbone is a fine-tuned **DistilBERT** extractive question answering model:

```
model_name = "distilbert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```

DistilBERT is selected for the following reasons:

* Lightweight and fast inference
* Well-calibrated token-level probabilities
* Strong baseline performance on extractive QA tasks

---

### 2.3 Tokenization & Sliding Window Handling

Contexts are truncated using a sliding window mechanism to support long passages:

* `max_length = 384`
* `doc_stride = 128`

Each question–context pair is split into overlapping windows. Token–character offset mappings are retained to enable reconstruction of answer spans after inference.

Unanswerable examples are labeled by assigning both start and end positions to the CLS token.

---

### 2.4 Training Objective

For each tokenized window, the model predicts start and end token positions. If the answer is not fully contained in the current window, the CLS token is used as the label.

This allows the model to jointly learn:

* Span extraction
* No-answer detection

---

### 2.5 Training Configuration

The QA model is trained using HuggingFace `Trainer` with the following configuration:

| Parameter           | Value |
| ------------------- | ----- |
| Batch size          | 8     |
| Learning rate       | 3e-5  |
| Epochs              | 1     |
| Weight decay        | 0.01  |
| Max sequence length | 384   |
| Doc stride          | 128   |

Mixed precision (`fp16`) is optionally enabled for GPU training.

---

### 2.6 Postprocessing Predictions

At inference time, the model produces start and end logits per token. These are converted into text answers using a standard SQuAD-style postprocessing routine:

* Top-N start/end candidates are selected.
* Span pairs are scored by `start_logit + end_logit`.
* The highest scoring valid span is selected.
* A null (CLS) score is computed for no-answer detection.

If the null score exceeds the best span score, the model abstains.

---

### 2.7 Evaluation Metrics

Performance is evaluated using the official SQuAD v2 metrics:

* Exact Match (EM)
* Token-level F1

Metrics are computed by reconstructing predicted spans and comparing them against ground truth references.

---

### 2.8 Output Artifacts

After training, the following artifacts are saved:

* Fine-tuned QA model weights
* Tokenizer
* Validation predictions
* Raw token-level logits

These outputs form the foundation for constructing the reliability dataset and extracting inference-time uncertainty features in subsequent stages.

## 3. Reliability Dataset Construction (build_reliability_dataset.py)

This section describes how the **reliability dataset** is built: a tabular dataset where each row corresponds to a QA example, containing (1) the model’s prediction, (2) correctness targets, and (3) **inference-time uncertainty features** computed from the model’s raw token logits. This dataset is the foundation for training the downstream **trust head**.

---

### 3.1 Objective

The goal is to convert standard extractive QA outputs into a **supervised learning problem**:

* **Input**: uncertainty signals available at inference time (single forward pass).
* **Output**: a binary label indicating whether the answer is “trustworthy.”

Importantly, this dataset is designed to be:

* **Leakage-safe**: no features use gold answers, ground-truth metadata, or evaluation-only signals.
* **Deployment-realistic**: every feature can be computed from logits produced during inference.
* **Aligned to selective prediction**: the resulting model is used to **rank** answers by reliability.

---

### 3.2 Dataset and Splitting Strategy

The script operates on **SQuAD v2** and supports building train/validation/test CSVs by slicing a chosen split.

* `--split {train, validation}` selects the source split.
* `--offset` and `--max_samples` allow carving contiguous segments from that split.

This approach makes it straightforward to construct:

* a trust-head training set,
* a calibration validation set,
* and a final held-out test set,

without mixing data across stages.

---

### 3.3 Forward Pass: Obtaining Start/End Logits

The core data source is the model’s **start and end logits** for each token position.

1. The script tokenizes each example using the same sliding-window approach as evaluation:

   * `max_length` controls the total token budget.
   * `doc_stride` controls overlap between windows.
   * long contexts produce **multiple feature windows** per example.

2. A Hugging Face `Trainer` is used to run batched inference over tokenized features:

* Outputs:

  * `start_logits_all`: shape `(num_features, seq_len)`
  * `end_logits_all`: shape `(num_features, seq_len)`

These logits become the raw material for both:

* span selection (the predicted answer), and
* uncertainty feature computation.

---

### 3.4 Postprocessing: Selecting the Best Span vs No-Answer

Each SQuAD v2 example may yield multiple windows. The script aggregates windows per example using `example_id`, then selects the final answer via a standard span search.

**Candidate span scoring**

For each window:

* Compute a `null_score` using the CLS token:

  * `null_score = start_logits[CLS] + end_logits[CLS]`

* Search for the best non-null span:

  * Take top-`n_best_size` start indices and end indices.
  * Filter invalid spans (non-context offsets, end < start, too long).
  * Score spans by:

    * `span_score = start_logits[s] + end_logits[e]`

Across all windows, track:

* `best_span_score` and its `best_window_index`
* `best_null_score`

**No-answer decision**

A SQuAD v2-style threshold is applied:

* predict no-answer if:

  * `(best_null_score - best_span_score) > null_threshold`

This yields:

* `pred_text` (empty string indicates abstention)
* `pred_is_no_answer` (binary indicator)

---

### 3.5 Correctness Targets and Trust Labels

The dataset includes both raw correctness metrics and binarized trust labels.

**Evaluation targets**

* `exact` (EM):

  * for answerable questions: exact string match against any gold answer.
  * for unanswerable questions: correct if the model predicts no-answer.

* `f1`:

  * token-overlap F1 against any gold answer.
  * for unanswerable questions: defined as 1.0 if no-answer is predicted, else 0.0.

**Trust label definitions**

Two trust targets are derived:

* `trustworthy_em`:

  * 1 if EM is correct under the rules above

* `trustworthy_f1_80`:

  * 1 if F1 ≥ 0.80 for answerable questions
  * 1 if the model correctly abstains on unanswerable questions

A key design choice is that abstaining on an answerable question is **always untrustworthy**:

* if `gold_has_answer == 1` and `pred_is_no_answer == 1`:

  * trust = 0

This encodes an explicit preference:

* *Correct abstention on unanswerable questions is good.*
* *Abstention when an answer exists is treated as failure.*

---

### 3.6 Inference-Time Feature Engineering

Each row contains uncertainty features computed from the model’s logits. Features are computed using the window that produced the **best span** (`best_window_index`), ensuring features reflect the window responsible for the final prediction.

#### 3.6.1 Utility Functions

Several helper functions convert logits into probabilities and dispersion statistics:

* **Stable softmax**: subtract max logit before exponentiation.
* **Entropy**: Shannon entropy of the softmax distribution.
* **Top-2 gap**: difference between the highest and second-highest logits.
* **Top-k mass**: probability mass contained in the top-k softmax values.
* **pmax**: maximum softmax probability.

These provide different views of uncertainty:

* entropy captures distribution spread,
* gaps capture decisiveness between top candidates,
* top-k mass and pmax capture confidence concentration.

#### 3.6.2 Feature Set A (Minimal + Strong)

These are the primary features used throughout experiments:

* **Span vs null competition**

  * `best_span_score`
  * `null_score`
  * `margin = best_span_score - null_score`

* **Uncertainty dispersion**

  * `start_entropy`, `end_entropy`

* **Token competition**

  * `start_top2_gap`, `end_top2_gap`

* **Length priors**

  * `answer_len_tokens`
  * `answer_len_frac_of_context`

The `margin` sign convention is important:

* positive margin ⇒ best span beats null ⇒ model is leaning toward answering.

#### 3.6.3 Feature Set B (Richer, Still Inference-Safe)

Additional probability-based features are computed from the same logits:

* `p_start_max`, `p_end_max`
* `start_top5_mass`, `end_top5_mass`
* `is_margin_positive`

These features provide more granular information about probability concentration beyond entropy and logit gaps.

---

### 3.7 Metadata Included for Analysis (Not as Features)

The CSV also stores metadata fields that help interpret results but are not used during trust-head training:

* `question` (text)
* `pred_text` (model output)
* `context_len`
* `gold_has_answer` (explicitly marked as analysis-only)

The training pipeline intentionally restricts features to the pre-defined inference-safe sets.

---

### 3.8 Output Artifact

The script writes a CSV where each row is one QA example:

* identifiers + metadata
* prediction + correctness
* trust labels
* uncertainty features

It also prints basic dataset statistics:

* positive rates for `trustworthy_em` and `trustworthy_f1_80`
* proportion of answerable examples (`gold_has_answer`)
* overall predicted no-answer rate

These diagnostics serve as sanity checks and help validate that the dataset composition matches expectations.


## 4) Feature Engineering (A + B + derived)

> What signals we compute and why they correlate with correctness.

### 4.1 Base features (A)

* Span vs null competition: best_span_score, null_score, margin = best_span_score - null_score
* Uncertainty shape: start_entropy, end_entropy
* Token competition: start_top2_gap, end_top2_gap
* Length priors: answer_len_tokens, answer_len_frac_of_context

### 4.2 Additional safe signals (B)

* Peak token confidence: p_start_max, p_end_max
* Distribution concentration: start_top5_mass, end_top5_mass
* Sign indicator: is_margin_positive
* Normalization: context_len

### 4.3 Derived features (safe transforms)

* margin_abs
* entropy_sum, entropy_diff
* top2gap_min, top2gap_sum
* pmax_min, pmax_prod
* answer_len_log1p, answer_len_is_zero

### 4.4 Feature pruning / collinearity drops

* Drop highly collinear raw signals in favor of more stable derived summaries (e.g., drop p_start_max/p_end_max when using pmax_min/pmax_prod).
* Explicit forced-drop list (kept consistent across runs for reproducibility and interpretability).

---

## 5) Trust Head Training

> Predict a probability-like trust score from inference-time features.

This stage trains a lightweight supervised model (“trust head”) that maps the inference-time feature vector to a scalar score intended to rank answers from most to least trustworthy. The trust head is trained *after* the QA model is fixed. At no point does this stage modify the underlying LLM/QA model; it only learns a post-hoc mapping from uncertainty signals to empirical correctness.

### 5.1 Model families evaluated

Multiple model families were benchmarked to test whether additional model capacity yields better reliability ranking. All models were trained on the same feature set (A + B + derived) and the same target labels.

**Families evaluated**

* **Logistic Regression (final)** — linear model with L2 regularization; fast, stable, interpretable.
* **Histogram Gradient Boosting (HGB)** — non-linear boosted trees; can capture interactions.
* **Random Forest (RF)** — bagged decision trees; robust but heavier.
* **Extra Trees (ET)** — randomized tree splits; often strong tabular baseline.

**Observed outcome (empirical)**

Across both label definitions (EM-based and F1-based) and across calibrations, model families were extremely close in ranking-based performance on TEST. For example, under the **trustworthy_em** label, TEST **AP** clustered tightly:

* LR (raw): **0.7549**
* HGB (raw): **0.7565**
* RF (raw): **0.7570**
* ET (raw): **0.7574**

Despite the tree models having substantially higher capacity, the improvements over LR were small and inconsistent relative to the variance expected from dataset slicing and threshold choices.

### 5.2 Why the primary objective is Average Precision (AP)

The trust head is trained and tuned for **ranking**, not classification. In deployment terms, the goal is:

> *Surface the most trustworthy answers first, especially at low coverage (answering fewer questions).*

Two factors make **Average Precision (AP)** an appropriate optimization target:

1. **Class imbalance and base trust-rate**

The positive class (“trustworthy”) is not rare in absolute terms but is meaningfully imbalanced and label-dependent:

* For **trustworthy_em** on TEST, the positive rate is **0.525**.
* For **trustworthy_f1_80** on TEST, the positive rate is **0.557**.

Metrics that reflect precision under varying thresholds are better aligned with selective answering than plain accuracy.

2. **AP aligns with “retrieve trustworthy first”**

AP is equivalent to the area under the precision–recall curve. It emphasizes early retrieval quality—exactly the behavior required when selecting only the top fraction of answers.

This aligns tightly with the selective prediction evaluation later (Section 7), where answers are sorted by the trust score and only the top fraction are kept.

### 5.3 Hyperparameter tuning protocol

Hyperparameter tuning was designed to preserve strict split boundaries and prevent any leakage from validation/test into training.

**Protocol**

* **RandomizedSearchCV on TRAIN only**

  * For Logistic Regression, the principal tuned parameter was **C** (inverse L2 regularization strength).
  * Example best values:

    * **EM label:** best C = **0.006020894493336131** (best CV AP ≈ **0.7966**)
    * **F1 label:** best C = **0.0011689518164985776** (best CV AP ≈ **0.8079**)

* **Stratified K-fold cross-validation**

  * Stratification preserves the trustworthy / non-trustworthy ratio per fold.

* **No validation/test leakage**

  * Hyperparameters are selected solely by CV performance on TRAIN.
  * The VAL split is reserved for calibration (Section 6) and model selection sanity checks.
  * The TEST split is reserved for final reporting of metrics and selective curves.

### 5.4 Metrics reported (raw probabilities)

Even though the core objective is ranking, multiple metrics are logged to characterize the trust head from different angles.

**Ranking metrics**

* **ROC-AUC** — probability that a random positive is ranked above a random negative.
* **Average Precision (AP)** — emphasizes high precision at high confidence.

**Probability quality metrics**

* **Brier score** — mean squared error between predicted probability and binary outcome.
* **ECE (Expected Calibration Error)** — discrepancy between predicted confidence and empirical accuracy across bins.

**Representative results (Logistic Regression, derived features)**

**Label: trustworthy_em (raw)**

* train: AUC **0.7498**, AP **0.7969**, Brier **0.2030**, ECE **0.0566**, pos_rate **0.570**
* val: AUC **0.7553**, AP **0.7891**, Brier **0.2006**, ECE **0.0319**, pos_rate **0.541**
* test: AUC **0.7331**, AP **0.7549**, Brier **0.2087**, ECE **0.0257**, pos_rate **0.525**

**Label: trustworthy_f1_80 (raw)**

* train: AUC **0.7334**, AP **0.8081**, Brier **0.2095**, ECE **0.0924**, pos_rate **0.610**
* val: AUC **0.7391**, AP **0.8020**, Brier **0.2071**, ECE **0.0702**, pos_rate **0.583**
* test: AUC **0.7093**, AP **0.7571**, Brier **0.2177**, ECE **0.0569**, pos_rate **0.557**

**Interpretable coefficient structure (Logistic Regression)**

The learned weights are consistent with the intuition that high-confidence, sharply peaked distributions correlate with correctness.

For **trustworthy_em**, the strongest positive coefficients were:

* `pmax_min` (+0.3273), `pmax_prod` (+0.2504)
* `top2gap_min` (+0.2321), `margin` (+0.1392)

The strongest negative coefficients were primarily length priors:

* `answer_len_log1p` (−0.2463)
* `answer_len_frac_of_context` (−0.1224)

This pattern is consistent with the idea that overly long extracted spans (relative to context) are more likely to be incorrect or over-generated.

### 5.5 Key finding: signal-limited regime

A consistent observation was that stronger model families did not translate into meaningfully better selective prediction curves. For **trustworthy_em**, the top-coverage trust rates on TEST were essentially identical across model families:

* **LR:** 10% → **0.883**, 40% → **0.735**, 80% → **0.594**
* **HGB:** 10% → **0.881**, 40% → **0.734**, 80% → **0.594**
* **RF:** 10% → **0.890**, 40% → **0.734**, 80% → **0.593**
* **ET:** 10% → **0.883**, 40% → **0.735**, 80% → **0.595**

The differences are small relative to the overall gains vs the baseline margin heuristic. This supports the conclusion that performance is **signal-limited**: the inference-time uncertainty features carry only a limited amount of separable information, and additional model capacity cannot manufacture new signal.

---

## 6) Calibration (Sigmoid vs Isotonic)

> Make predicted probabilities match empirical frequencies.

Calibration is treated as a second-stage transformation of the trust head’s output. The ranking objective is already optimized in Section 5; calibration is only required when the trust score is consumed as a probability-like quantity (e.g., thresholding at a chosen risk level).

### 6.1 What calibration means here

Calibration is the alignment between predicted trust scores and empirical correctness rates.

If the model outputs **0.80** trust on a set of examples, then approximately **80%** of those examples should be trustworthy under the chosen label definition.

This is distinct from ranking: a model can rank well but still be miscalibrated.

### 6.2 Methods

Two standard post-hoc calibration methods were evaluated.

* **Sigmoid (Platt scaling)**

  * Fits a logistic mapping from raw scores to probabilities.
  * Low variance and typically stable.

* **Isotonic regression**

  * Fits a monotonic, piecewise-constant mapping.
  * Flexible but can overfit on smaller validation sets.

### 6.3 How calibration is fit

To avoid leakage, calibration is trained on **VAL only** and then applied to **TEST**.

**Protocol**

1. Train the trust head on TRAIN (with CV inside TRAIN for hyperparameters).
2. Fit calibrator on VAL predictions and VAL labels.
3. Apply the fitted calibrator to TEST predictions.

This ensures the final calibrated probability metrics reflect generalization rather than in-sample fitting.

### 6.4 What changed and what didn’t

Calibration primarily improved probability-alignment metrics (ECE, sometimes Brier), while leaving ranking metrics (AUC, AP) largely unchanged.

**EM label (LR)**

* Raw (TEST): Brier **0.2087**, ECE **0.0257**
* Sigmoid (TEST): Brier **0.2083**, ECE **0.0179**
* Isotonic (TEST): Brier **0.2085**, ECE **0.0163**

**F1 label (LR)**

* Raw (TEST): Brier **0.2177**, ECE **0.0569**
* Sigmoid (TEST): Brier **0.2147**, ECE **0.0236**
* Isotonic (TEST): Brier **0.2148**, ECE **0.0207**

In contrast, the selective curves (Section 7) were nearly identical between raw and calibrated variants, indicating that calibration does not materially change the ordering of examples.

---

## 7) Evaluation: Selective Prediction Curves

> The core evaluation—how reliability improves as coverage decreases.

The main deployment objective is to answer only when confidence is high. Selective prediction curves quantify how the empirical trust rate changes as fewer answers are returned.

### 7.1 Curve definition

For a fixed dataset split (typically TEST):

1. Compute a trust score for every example.
2. Sort examples by trust score (descending).
3. For each coverage level **c** (e.g., 0.10, 0.20, …, 1.00):

   * Take the top **c · N** examples.
   * Compute the mean of the trust label among those examples.

This produces a curve mapping **coverage → answered trust-rate**.

### 7.2 Baselines

To ensure improvements are attributable to learned structure rather than trivial transformations, the trust head is compared against simple inference-time heuristics:

* **margin** (best_span_score − null_score)
* **pmax_min** and **pmax_prod** (derived from start/end max probabilities)

The margin baseline is particularly strong because it encodes the span-vs-null competition used for SQuAD v2 style abstention.

### 7.3 What “good” looks like

A strong selective curve has:

* High answered trust-rate at low coverage (e.g., top 10%).
* A large gap vs baselines across a wide range of coverages.
* A curve that approaches the oracle curve at low coverage (rare in practice).

**Example (EM label, TEST)**

* **Oracle:** 10% → 1.000, 40% → 1.000
* **Baseline margin:** 10% → 0.631, 40% → 0.536
* **Trust head (LR):** 10% → 0.883, 40% → 0.735

This demonstrates that the trust head substantially improves early precision relative to naive confidence measures.

A compact table is used to summarize selective performance at fixed coverages:

* 10%, 20%, 40%, 60%, 80%, 100% answered trust-rate

**EM label (TEST, LR)**

| Coverage | Oracle | Baseline: margin | Trust head (LR) |
| -------: | -----: | ---------------: | --------------: |
|      10% |  1.000 |            0.631 |           0.883 |
|      20% |  1.000 |            0.596 |           0.823 |
|      40% |  1.000 |            0.536 |           0.735 |
|      60% |  0.870 |            0.462 |           0.656 |
|      80% |  0.660 |            0.472 |           0.594 |
|     100% |  0.525 |            0.525 |           0.525 |

**F1 label (TEST, LR)**

| Coverage | Oracle | Baseline: margin | Trust head (LR) |
| -------: | -----: | ---------------: | --------------: |
|      10% |  1.000 |            0.690 |           0.882 |
|      20% |  1.000 |            0.653 |           0.824 |
|      40% |  1.000 |            0.595 |           0.742 |
|      60% |  0.923 |            0.515 |           0.673 |
|      80% |  0.700 |            0.512 |           0.617 |
|     100% |  0.557 |            0.557 |           0.557 |

Notably, raw vs calibrated curves were indistinguishable at this resolution.

---

## 8) Results Summary (What was concluded)

### 8.1 Final model family choice

**Logistic Regression** was selected as the final trust head because it achieved selective performance that matched tree-based models while remaining simplest to deploy.

Under **trustworthy_em**, LR achieved TEST AP **0.7549** and a top-10% answered trust-rate of **0.883**, compared to **0.631** for the margin baseline. Tree models produced nearly identical selective curves despite higher complexity.

### 8.2 EM vs F1 label behavior

Two labeling schemes were used to define “trustworthy”:

* **EM-based**: strict correctness (exact match).
* **F1≥0.8-based**: more forgiving partial-match correctness.

As expected, the F1-based label has a higher base trust-rate (TEST pos_rate **0.557** vs **0.525** for EM). However, ranking behavior remained broadly similar: the trust head produced strong improvements over the margin baseline under both label definitions.

### 8.3 Calibration conclusion

Calibration is useful when the trust score must be interpreted probabilistically, but it does not materially affect the ranking objective.

* **Sigmoid** is a safe default due to stability and low variance.
* **Isotonic** can achieve extremely low ECE on VAL (often near 0.0) but may slightly reduce AP on TEST in some settings.

Given that selective curves are nearly unchanged, calibration should be treated as optional for ranking-only deployment and recommended for threshold-based decision policies.


## 9) Limitations and Future Work

* Dataset domain limits (extractive QA)
* Label noise (F1 thresholding)
* Next: apply to generative LLM answers (NLP/LLM setting)
* Next: selective abstention policy + cost/utility tradeoffs
* Next: generalization tests across datasets/models


