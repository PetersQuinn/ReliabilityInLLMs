# src/train_trust_head.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression


# -----------------------------
# A + B (inference-safe, no leakage)
# -----------------------------
FEATURE_SETS = {
    "A_plus_B": [
        # --- A ---
        "best_span_score",
        "null_score",
        "margin",
        "start_entropy",
        "end_entropy",
        "start_top2_gap",
        "end_top2_gap",
        "answer_len_tokens",
        "answer_len_frac_of_context",

        # --- B ---
        "p_start_max",
        "p_end_max",
        "start_top5_mass",
        "end_top5_mass",
        "is_margin_positive",

        # normalization
        "context_len",
    ],
    "debug_with_metadata": [
        "best_span_score",
        "null_score",
        "margin",
        "start_entropy",
        "end_entropy",
        "start_top2_gap",
        "end_top2_gap",
        "answer_len_tokens",
        "answer_len_frac_of_context",
        "p_start_max",
        "p_end_max",
        "start_top5_mass",
        "end_top5_mass",
        "is_margin_positive",
        "context_len",
    ],
}

# -----------------------------
# Drop redundant / collinear features
# -----------------------------
DROP_COLS = {
    "answer_len_tokens",
    "p_start_max",
    "p_end_max",
    "is_margin_positive",
    "margin_sigmoid",
    "best_span_score",
    "null_score",
}

# -----------------------------
# Derived features (inference-safe)
# -----------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "margin" in out.columns:
        m = out["margin"].astype(float).values
        out["margin_abs"] = np.abs(m)

    if "start_entropy" in out.columns and "end_entropy" in out.columns:
        se = out["start_entropy"].astype(float)
        ee = out["end_entropy"].astype(float)
        out["entropy_sum"] = se + ee
        out["entropy_diff"] = np.abs(se - ee)

    if "start_top2_gap" in out.columns and "end_top2_gap" in out.columns:
        sg = out["start_top2_gap"].astype(float)
        eg = out["end_top2_gap"].astype(float)
        out["top2gap_min"] = np.minimum(sg, eg)
        out["top2gap_sum"] = sg + eg

    if "p_start_max" in out.columns and "p_end_max" in out.columns:
        ps = out["p_start_max"].astype(float)
        pe = out["p_end_max"].astype(float)
        out["pmax_min"] = np.minimum(ps, pe)
        out["pmax_prod"] = ps * pe

    if "answer_len_tokens" in out.columns:
        al = out["answer_len_tokens"].astype(float).values
        out["answer_len_log1p"] = np.log1p(al)
        out["answer_len_is_zero"] = (al <= 0).astype(int)

    return out


def resolve_feature_cols(feature_set: str, add_derived: bool):
    base = FEATURE_SETS[feature_set].copy()
    derived = []
    if add_derived:
        derived = [
            "margin_abs",
            "entropy_sum",
            "entropy_diff",
            "top2gap_min",
            "top2gap_sum",
            "pmax_min",
            "pmax_prod",
            "answer_len_log1p",
            "answer_len_is_zero",
        ]
    return base, derived


def load_xy(path, label, base_cols, derived_cols, add_derived, drop_cols_set):
    df = pd.read_csv(path)

    if add_derived:
        df = add_derived_features(df)

    feature_cols = base_cols + [c for c in derived_cols if c in df.columns]
    feature_cols = [c for c in feature_cols if c not in drop_cols_set]

    missing = [c for c in feature_cols + [label] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    X = df[feature_cols].astype(float).values
    y = df[label].astype(int).values
    return df, X, y, feature_cols


# -----------------------------
# Calibration helpers
# -----------------------------
def fit_sigmoid_calibrator(p_val: np.ndarray, y_val: np.ndarray):
    """
    Platt scaling on probabilities p -> calibrated p
    We fit a logistic regression on the logit of p.
    """
    eps = 1e-8
    p = np.clip(p_val, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(logit, y_val)
    return {"type": "sigmoid", "model": lr}

def apply_sigmoid_calibrator(cal, p: np.ndarray):
    eps = 1e-8
    p2 = np.clip(p, eps, 1 - eps)
    logit = np.log(p2 / (1 - p2)).reshape(-1, 1)
    return cal["model"].predict_proba(logit)[:, 1]

def fit_isotonic_calibrator(p_val: np.ndarray, y_val: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return {"type": "isotonic", "model": iso}

def apply_isotonic_calibrator(cal, p: np.ndarray):
    return cal["model"].predict(p)

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE with equal-width bins on [0,1].
    """
    y = y.astype(int)
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


# -----------------------------
# Evaluation
# -----------------------------
def eval_split(name, y, p):
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    brier = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p, n_bins=15)
    print(f"{name:>8} | AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}  ECE={ece:.4f}  pos_rate={y.mean():.3f}")


def print_top_coeffs_if_lr(best_estimator: Pipeline, feature_cols, topk=15):
    try:
        clf = best_estimator.named_steps["clf"]
    except Exception:
        return
    if not isinstance(clf, LogisticRegression):
        return
    coefs = clf.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1][:topk]
    print("\nTop coefficients (by |coef|):")
    for i in order:
        print(f"  {feature_cols[i]:<28} coef={coefs[i]: .4f}")


# -----------------------------
# Model factories + parameter spaces
# -----------------------------
def make_search(model_name: str, seed: int, max_iter_lr: int):
    if model_name == "lr":
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=max_iter_lr,
                class_weight="balanced",
                solver="lbfgs",
            ))
        ])
        Cs = np.logspace(-4, 3, 60)
        param_dist = {"clf__C": Cs}
        return pipe, param_dist

    if model_name == "rf":
        pipe = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=800,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ))
        ])
        param_dist = {
            "clf__n_estimators": [400, 800, 1200, 1600],
            "clf__max_depth": [None, 4, 6, 8, 10, 14, 18, 24],
            "clf__min_samples_split": [2, 4, 8, 16, 32],
            "clf__min_samples_leaf": [1, 2, 4, 8, 16, 32],
            "clf__max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7],
            "clf__bootstrap": [True, False],
        }
        return pipe, param_dist

    if model_name == "extratrees":
        pipe = Pipeline([
            ("clf", ExtraTreesClassifier(
                n_estimators=1200,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            ))
        ])
        param_dist = {
            "clf__n_estimators": [600, 1200, 1800, 2400],
            "clf__max_depth": [None, 4, 6, 8, 10, 14, 18, 24],
            "clf__min_samples_split": [2, 4, 8, 16, 32],
            "clf__min_samples_leaf": [1, 2, 4, 8, 16, 32],
            "clf__max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7],
        }
        return pipe, param_dist

    if model_name == "hgb":
        pipe = Pipeline([
            ("clf", HistGradientBoostingClassifier(
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
            ))
        ])
        param_dist = {
            "clf__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
            "clf__max_depth": [None, 2, 3, 4, 5, 6, 8],
            "clf__max_leaf_nodes": [15, 31, 63, 127],
            "clf__min_samples_leaf": [10, 20, 40, 80, 160],
            "clf__l2_regularization": [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "clf__max_bins": [64, 128, 255],
        }
        return pipe, param_dist

    raise ValueError(f"Unknown model: {model_name}")


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--label", type=str, required=True, choices=["trustworthy_em", "trustworthy_f1_80"])
    p.add_argument("--out_dir", type=str, default="artifacts/reliability/models")

    p.add_argument("--feature_set", type=str, default="A_plus_B", choices=list(FEATURE_SETS.keys()))
    p.add_argument("--add_derived", action="store_true")
    p.add_argument("--drop_cols", type=str, default=None)

    p.add_argument("--model", type=str, default="hgb",
                   choices=["lr", "rf", "extratrees", "hgb"])
    p.add_argument("--n_iter", type=int, default=200)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--scoring", type=str, default="average_precision",
                   choices=["average_precision", "roc_auc"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--max_iter", type=int, default=4000)

    # Calibration
    p.add_argument("--calibrate", type=str, default="none",
                   choices=["none", "sigmoid", "isotonic"],
                   help="Fit a calibrator on VAL set probabilities, then apply to VAL/TEST.")
    p.add_argument("--show_topk", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_cols, derived_cols = resolve_feature_cols(args.feature_set, args.add_derived)

    drop_cols_set = set(DROP_COLS)
    if args.drop_cols:
        extra = [x.strip() for x in args.drop_cols.split(",") if x.strip()]
        drop_cols_set.update(extra)

    _, Xtr, ytr, feature_cols = load_xy(
        args.train_csv, args.label, base_cols, derived_cols, args.add_derived, drop_cols_set
    )
    _, Xva, yva, _ = load_xy(
        args.val_csv, args.label, base_cols, derived_cols, args.add_derived, drop_cols_set
    )
    _, Xte, yte, _ = load_xy(
        args.test_csv, args.label, base_cols, derived_cols, args.add_derived, drop_cols_set
    )

    print("Using features:", len(feature_cols))
    print("Force-dropped (configured):", sorted(list(drop_cols_set)))

    pipe, param_dist = make_search(args.model, seed=args.seed, max_iter_lr=args.max_iter)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        cv=cv,
        verbose=2,
        random_state=args.seed,
        refit=True,
    )

    print(f"\nTuning model={args.model} with RandomizedSearchCV:")
    print(f"  n_iter={args.n_iter} | cv_folds={args.cv_folds} | scoring={args.scoring}")
    search.fit(Xtr, ytr)

    best_model = search.best_estimator_
    print("\nBest CV score:", search.best_score_)
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Raw probabilities
    ptr_raw = best_model.predict_proba(Xtr)[:, 1]
    pva_raw = best_model.predict_proba(Xva)[:, 1]
    pte_raw = best_model.predict_proba(Xte)[:, 1]

    print(f"\nBest tuned trust head (RAW) | label={args.label} | model={args.model} | derived={args.add_derived}")
    eval_split("train_raw", ytr, ptr_raw)
    eval_split("  val_raw", yva, pva_raw)
    eval_split(" test_raw", yte, pte_raw)

    print_top_coeffs_if_lr(best_model, feature_cols, topk=args.show_topk)

    calibrator = None
    if args.calibrate != "none":
        # Fit calibrator ONLY on validation predictions to avoid leakage.
        if args.calibrate == "sigmoid":
            calibrator = fit_sigmoid_calibrator(pva_raw, yva)
            pva_cal = apply_sigmoid_calibrator(calibrator, pva_raw)
            pte_cal = apply_sigmoid_calibrator(calibrator, pte_raw)
        elif args.calibrate == "isotonic":
            calibrator = fit_isotonic_calibrator(pva_raw, yva)
            pva_cal = apply_isotonic_calibrator(calibrator, pva_raw)
            pte_cal = apply_isotonic_calibrator(calibrator, pte_raw)
        else:
            raise ValueError("Unknown calibration method.")

        print(f"\nCALIBRATED probs using method={args.calibrate} (fit on VAL only)")
        eval_split("  val_cal", yva, pva_cal)
        eval_split(" test_cal", yte, pte_cal)

    out_path = os.path.join(
        args.out_dir,
        f"trust_head_{args.label}_{args.model}_{args.feature_set}{'_derived' if args.add_derived else ''}_tuned"
        + (f"_cal_{args.calibrate}" if args.calibrate != "none" else "")
        + ".joblib"
    )

    joblib.dump({
        "model": best_model,
        "feature_cols": feature_cols,
        "label": args.label,
        "feature_set": args.feature_set,
        "add_derived": args.add_derived,
        "drop_cols": sorted(list(drop_cols_set)),
        "calibration": {
            "method": args.calibrate,
            "calibrator": calibrator,  # None if not calibrated
            "fit_on": "val_only",
        },
        "search": {
            "model_family": args.model,
            "n_iter": args.n_iter,
            "cv_folds": args.cv_folds,
            "scoring": args.scoring,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        }
    }, out_path)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
