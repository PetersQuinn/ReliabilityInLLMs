# Inference-Time Reliability Estimation for LLM Answers

This repository presents a lightweight framework for estimating the trustworthiness of Large Language Model (LLM) answers using only inference-time uncertainty signals. The goal is not to classify answers as correct or incorrect, but to **rank answers by reliability** such that users or downstream systems can selectively trust the most dependable outputs.

---

## Problem

LLMs frequently generate plausible but incorrect responses. While confidence scores are available at inference time, raw probabilities are poorly calibrated and unreliable as indicators of correctness. Most existing methods rely on ensembles, repeated sampling, or external verification — all of which are costly or infeasible in real-time systems.

We instead focus on the following question:

> *Given a set of answers, which ones should we trust first?*

This reframes the problem as **selective prediction** rather than binary classification.

---

## Approach

We train a small "trust head" on top of inference-time features derived directly from the model's token-level outputs. All features are leakage-safe and computable in a single forward pass.

### Feature Categories

* **Token confidence aggregates**
  `pmax_min`, `pmax_prod`, top-k mass, margin

* **Entropy dynamics**
  `start_entropy`, `end_entropy`, `entropy_sum`, `entropy_diff`

* **Token competition**
  `start_top2_gap`, `end_top2_gap`, `top2gap_min`, `top2gap_sum`

* **Answer length priors**
  `answer_len_log1p`, `answer_len_frac_of_context`, `answer_len_is_zero`

These features encode internal model uncertainty patterns without requiring multiple model calls.

---

## Model Family Selection

We evaluated Logistic Regression, Histogram Gradient Boosting, Random Forests, and Extra Trees across both Exact Match (EM) and F1-based trust labels, with and without probability calibration.

**Finding:**

> All model families converged to nearly identical selective performance curves.

This indicates that the problem is **signal-limited**, not model-limited. Increasing model complexity yields negligible gains.

### Final Choice: Logistic Regression

We select **Logistic Regression** as the final model family because it:

* Matches tree-based models in selective trust performance
* Is fast, stable, and interpretable
* Requires minimal compute overhead

---

## Evaluation: Selective Prediction

Rather than accuracy, we evaluate using **selective trust curves**:

> For the top *k%* most confident answers, what fraction are truly trustworthy?

This reflects the real deployment use-case: maximizing reliability under limited coverage.

### Example (EM Task)

| Coverage | Baseline Margin | Trust Head (LR) | Oracle |
| -------- | --------------- | --------------- | ------ |
| 10%      | 0.63            | **0.88**        | 1.00   |
| 40%      | 0.54            | **0.74**        | 1.00   |
| 80%      | 0.47            | **0.59**        | 0.66   |

The trust head dramatically outperforms naive confidence baselines across all coverage levels.

---

## Calibration

We evaluated raw, sigmoid (Platt), and isotonic calibration. While calibration improves probability alignment (Brier, ECE), it has **minimal impact on ranking performance**, which is our primary objective.

This further supports using Logistic Regression without heavy post-processing.

---

## Key Insight

> Trustworthiness is not a binary decision — it is a ranking problem.

By learning how internal uncertainty signals correlate with correctness, we can surface the most reliable answers first without modifying the underlying LLM.

---

## Conclusion

A simple logistic trust head trained on inference-time uncertainty features is sufficient to construct a strong reliability ranking system for LLM outputs. This framework enables selective answering, improves safety, and operates with virtually no runtime overhead.

This project demonstrates that meaningful reliability estimation is achievable today — not by scaling models, but by understanding their uncertainty.
