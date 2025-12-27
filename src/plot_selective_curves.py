# src/plot_selective_curves.py
import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Usually test_reliability_*.csv or trust_test.csv")
    p.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="Comma-separated list of joblib bundles from train_trust_head.py (raw + calibrated).",
    )
    p.add_argument(
        "--names",
        type=str,
        default=None,
        help="Optional comma-separated display names aligned to --model_paths. If omitted, names are inferred.",
    )
    p.add_argument("--out_dir", type=str, default="artifacts/reliability/plots")
    p.add_argument("--n_points", type=int, default=80, help="Points on coverage axis")
    p.add_argument("--min_cov", type=float, default=0.05, help="Minimum coverage shown")
    p.add_argument("--max_cov", type=float, default=1.0, help="Maximum coverage shown")

    p.add_argument(
        "--baselines",
        type=str,
        default="margin,pmax_prod,pmax_min",
        help="Comma list of baseline rankers to include if present in CSV",
    )
    p.add_argument(
        "--also_plot_roc",
        action="store_true",
        help="Optional: save ROC curve plot(s) too (only if predict_proba exists).",
    )
    p.add_argument(
        "--also_plot_reliability",
        action="store_true",
        help="Optional: save reliability diagram(s) too (calibration curves).",
    )
    p.add_argument("--reliability_bins", type=int, default=15)
    return p.parse_args()


# -----------------------------
# Derived features (must match trainer)
# -----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must match the derived features used in train_trust_head.py (current version).
    Only uses inference-safe columns.
    """
    out = df.copy()

    if "margin" in out.columns:
        m = out["margin"].astype(float).values
        out["margin_abs"] = np.abs(m)

    if "start_entropy" in out.columns and "end_entropy" in out.columns:
        se = out["start_entropy"].astype(float).values
        ee = out["end_entropy"].astype(float).values
        out["entropy_sum"] = se + ee
        out["entropy_diff"] = np.abs(se - ee)

    if "start_top2_gap" in out.columns and "end_top2_gap" in out.columns:
        sg = out["start_top2_gap"].astype(float).values
        eg = out["end_top2_gap"].astype(float).values
        out["top2gap_min"] = np.minimum(sg, eg)
        out["top2gap_sum"] = sg + eg

    if "p_start_max" in out.columns and "p_end_max" in out.columns:
        ps = out["p_start_max"].astype(float).values
        pe = out["p_end_max"].astype(float).values
        out["pmax_min"] = np.minimum(ps, pe)
        out["pmax_prod"] = ps * pe

    if "answer_len_tokens" in out.columns:
        al = out["answer_len_tokens"].astype(float).values
        out["answer_len_log1p"] = np.log1p(al)
        out["answer_len_is_zero"] = (al <= 0).astype(int)
    elif "answer_len_chars" in out.columns:
        al = out["answer_len_chars"].astype(float).values
        out["answer_len_log1p"] = np.log1p(al)
        out["answer_len_is_zero"] = (al <= 0).astype(int)

    return out


# -----------------------------
# Metrics + curves
# -----------------------------
def selective_curve(scores: np.ndarray, y: np.ndarray, coverages: np.ndarray):
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]
    n = len(y_sorted)
    acc = np.empty_like(coverages, dtype=float)
    for i, c in enumerate(coverages):
        k = max(1, int(round(c * n)))
        acc[i] = float(y_sorted[:k].mean())
    return acc


def model_scores_from_estimator(model, X: np.ndarray):
    """
    Returns "score" used for ranking. Prefer prob(class=1) if available.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    return np.asarray(model.predict(X)).ravel().astype(float)


def apply_calibrator(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    """
    Robustly apply a calibration mapping trained on probabilities.

    Supports:
      - IsotonicRegression: .transform(...)
      - LogisticRegression (Platt): .predict_proba(raw.reshape(-1,1))[:,1]
      - Any estimator with .predict_proba(...)
      - Fallback to .predict(...)
    """
    raw_probs = np.asarray(raw_probs).ravel().astype(float)
    x2 = raw_probs.reshape(-1, 1)

    # IsotonicRegression style
    if hasattr(calibrator, "transform"):
        out = calibrator.transform(raw_probs)
        return np.clip(np.asarray(out).ravel().astype(float), 0.0, 1.0)

    # Estimator style
    if hasattr(calibrator, "predict_proba"):
        proba = calibrator.predict_proba(x2)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.clip(proba[:, 1].astype(float), 0.0, 1.0)
        return np.clip(proba.ravel().astype(float), 0.0, 1.0)

    if hasattr(calibrator, "predict"):
        out = calibrator.predict(x2)
        return np.clip(np.asarray(out).ravel().astype(float), 0.0, 1.0)

    raise ValueError("Calibrator exists but has no transform/predict_proba/predict interface.")


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins.
    """
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def reliability_curve_points(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    """
    Returns (bin_centers, empirical_acc, mean_conf, counts)
    """
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    accs = np.full(n_bins, np.nan, dtype=float)
    confs = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        counts[i] = int(mask.sum())
        if counts[i] == 0:
            continue
        accs[i] = float(y[mask].mean())
        confs[i] = float(p[mask].mean())

    return centers, accs, confs, counts


def pretty_points_table(coverages, curves, points=(0.10, 0.20, 0.40, 0.60, 0.80, 1.00)):
    idxs = [int(np.argmin(np.abs(coverages - p))) for p in points]
    header = ["model"] + [f"{coverages[i]:.2f}" for i in idxs]
    rows = []
    for name, accs in curves.items():
        rows.append([name] + [f"{accs[i]:.3f}" for i in idxs])

    widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]

    def fmt(r):
        return " | ".join(str(x).ljust(w) for x, w in zip(r, widths))

    print("\nAnswered trust-rate table (approx):")
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def infer_display_name(model_path: str, bundle: dict):
    """
    Try to infer a helpful label from:
      - bundle["search"]["model_family"]
      - bundle["calibration"]["method"] or bundle["calibration_method"]
      - filename hints (_cal_sigmoid/_cal_isotonic)
    """
    base = os.path.basename(model_path)

    model_family = None
    cal_method = None

    # from bundle
    if isinstance(bundle, dict):
        if "search" in bundle and isinstance(bundle["search"], dict):
            model_family = bundle["search"].get("model_family", None)
        model_family = model_family or bundle.get("model_family", None) or bundle.get("model", None)

        cal_method = bundle.get("calibration_method", None)
        if "calibration" in bundle and isinstance(bundle["calibration"], dict):
            cal_method = cal_method or bundle["calibration"].get("method", None)
        if "calibrator" in bundle:
            cal_method = cal_method or "calibrated"

    # from filename
    if cal_method is None:
        if "_cal_sigmoid" in base or "sigmoid" in base:
            cal_method = "sigmoid"
        elif "_cal_isotonic" in base or "isotonic" in base:
            cal_method = "isotonic"
        else:
            cal_method = "raw"

    if model_family is None:
        # grab common tokens from filename
        # e.g. trust_head_trustworthy_em_extratrees_A_plus_B_derived_tuned_cal_sigmoid.joblib
        m = re.search(r"trust_head_[^_]+_([a-zA-Z0-9]+)_", base)
        model_family = m.group(1) if m else "model"

    return f"{model_family}:{cal_method}"


def compute_binary_metrics(y: np.ndarray, p: np.ndarray):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    brier = brier_score_loss(y, p)
    ece = ece_score(y, p, n_bins=15)
    return {"AUC": float(auc), "AP": float(ap), "Brier": float(brier), "ECE": float(ece), "pos_rate": float(y.mean())}


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    model_paths = [s.strip() for s in args.model_paths.split(",") if s.strip()]
    if not model_paths:
        raise ValueError("No --model_paths provided.")

    if args.names:
        names = [s.strip() for s in args.names.split(",")]
        if len(names) != len(model_paths):
            raise ValueError("--names must have same count as --model_paths")
    else:
        names = None

    # Determine label from first bundle or CSV
    first_bundle = joblib.load(model_paths[0])
    label = first_bundle.get("label", None)
    if label is None:
        if "trustworthy_em" in df.columns:
            label = "trustworthy_em"
        elif "trustworthy_f1_80" in df.columns:
            label = "trustworthy_f1_80"
        else:
            raise ValueError("Could not infer label column from CSV or bundle.")

    if label not in df.columns:
        raise ValueError(f"CSV missing label column: {label}")

    y = df[label].astype(int).values

    coverages = np.linspace(args.min_cov, args.max_cov, args.n_points)

    # Baselines
    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]

    curves = {}
    metrics_rows = []

    # Oracle curve
    oracle_score = y.astype(float)
    curves["Oracle (perfect)"] = selective_curve(oracle_score, y, coverages)

    # Baseline curves (ranking-only)
    for b in baseline_names:
        if b in df.columns:
            curves[f"Baseline:{b}"] = selective_curve(df[b].astype(float).values, y, coverages)

    # Each model curve + metrics
    all_model_probs_for_roc = {}          # name -> probs (for roc)
    all_model_probs_for_reliability = {}  # name -> probs (for reliability)

    for i, mp in enumerate(model_paths):
        bundle = joblib.load(mp)

        model = bundle.get("model", None)
        if model is None:
            raise ValueError(f"Bundle missing 'model': {mp}")

        feature_cols = bundle.get("feature_cols", None)
        if feature_cols is None:
            raise ValueError(f"Bundle missing 'feature_cols': {mp}")

        add_derived = bool(bundle.get("add_derived", False))

        # Ensure we have required features; add derived if needed.
        missing_feats = [c for c in feature_cols if c not in df.columns]
        if missing_feats and add_derived:
            df_work = add_derived_features(df)
        else:
            df_work = df

        missing_feats = [c for c in feature_cols if c not in df_work.columns]
        if missing_feats:
            raise ValueError(f"CSV is missing model feature columns for {mp}: {missing_feats}")

        X = df_work[feature_cols].astype(float).values

        # Determine display name
        disp = names[i] if names is not None else infer_display_name(mp, bundle)

        # Get probabilities for policy (thresholding) + ranking
        # If this bundle has a calibrator, prefer calibrated probs for policy plots/metrics.
        # For ranking curves, calibrated vs raw should be almost identical (monotone), but we’ll still use the bundle’s “final” probs.
        p_raw = model_scores_from_estimator(model, X)

        # If bundle stores a separate calibrator for mapping raw probs -> calibrated probs, apply it.
        # Otherwise, assume model already outputs its final probs.
        p_final = p_raw
        if "calibrator" in bundle and bundle["calibrator"] is not None:
            try:
                p_final = apply_calibrator(bundle["calibrator"], p_raw)
            except Exception as e:
                raise RuntimeError(f"Failed to apply calibrator for {mp}: {e}")

        p_final = np.clip(np.asarray(p_final).ravel().astype(float), 0.0, 1.0)

        # Selective curve uses scores for ranking
        curves[disp] = selective_curve(p_final, y, coverages)

        # Metrics (classification/probability quality)
        m = compute_binary_metrics(y, p_final)
        m["model"] = disp
        metrics_rows.append(m)

        all_model_probs_for_roc[disp] = p_final
        all_model_probs_for_reliability[disp] = p_final

    # -----------------------------
    # Plot: selective curves
    # -----------------------------
    plt.figure()
    for name, accs in curves.items():
        plt.plot(coverages, accs, label=name)
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel(f"Answered trust rate ({label})")
    plt.title("Selective prediction curves (Oracle / baselines / models)")
    plt.legend(fontsize=8)

    tag = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = os.path.join(args.out_dir, f"selective_multi_{label}_{tag}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print("Saved:", out_path)

    pretty_points_table(coverages, curves)

    # -----------------------------
    # Print metrics table
    # -----------------------------
    if metrics_rows:
        print("\nModel probability metrics on this CSV:")
        # stable ordering: best Brier then best ECE as tie-break
        metrics_rows = sorted(metrics_rows, key=lambda r: (r["Brier"], r["ECE"]))
        header = ["model", "AUC", "AP", "Brier", "ECE", "pos_rate"]
        widths = {h: max(len(h), max(len(f"{r[h]:.4f}") if h != "model" else len(r["model"]) for r in metrics_rows)) for h in header}
        widths["model"] = max(widths["model"], max(len(r["model"]) for r in metrics_rows))

        def fmt_row(r):
            return (
                r["model"].ljust(widths["model"])
                + " | "
                + f"{r['AUC']:.4f}".rjust(widths["AUC"])
                + " | "
                + f"{r['AP']:.4f}".rjust(widths["AP"])
                + " | "
                + f"{r['Brier']:.4f}".rjust(widths["Brier"])
                + " | "
                + f"{r['ECE']:.4f}".rjust(widths["ECE"])
                + " | "
                + f"{r['pos_rate']:.3f}".rjust(widths["pos_rate"])
            )

        print(
            "model".ljust(widths["model"])
            + " | "
            + "AUC".rjust(widths["AUC"])
            + " | "
            + "AP".rjust(widths["AP"])
            + " | "
            + "Brier".rjust(widths["Brier"])
            + " | "
            + "ECE".rjust(widths["ECE"])
            + " | "
            + "pos_rate".rjust(widths["pos_rate"])
        )
        print("-" * (sum(widths.values()) + 3 * (len(header) - 1) + 2))
        for r in metrics_rows:
            print(fmt_row(r))

    # -----------------------------
    # Optional: ROC curves (one plot, multiple lines)
    # -----------------------------
    if args.also_plot_roc:
        from sklearn.metrics import roc_curve, auc

        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

        for name, p in all_model_probs_for_roc.items():
            fpr, tpr, _ = roc_curve(y, p)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curves ({label})")
        plt.legend(fontsize=8)

        roc_path = os.path.join(args.out_dir, f"roc_multi_{label}_{tag}.png")
        plt.savefig(roc_path, dpi=180, bbox_inches="tight")
        print("Saved:", roc_path)

    # -----------------------------
    # Optional: Reliability diagrams (one plot, multiple lines)
    # -----------------------------
    if args.also_plot_reliability:
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")

        for name, p in all_model_probs_for_reliability.items():
            centers, accs, confs, counts = reliability_curve_points(y, p, n_bins=args.reliability_bins)
            # plot only bins with data
            mask = ~np.isnan(accs) & ~np.isnan(confs)
            plt.plot(confs[mask], accs[mask], marker="o", linewidth=1.2, label=name)

        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical accuracy")
        plt.title(f"Reliability diagram ({label})")
        plt.legend(fontsize=8)

        rel_path = os.path.join(args.out_dir, f"reliability_multi_{label}_{tag}.png")
        plt.savefig(rel_path, dpi=180, bbox_inches="tight")
        print("Saved:", rel_path)


if __name__ == "__main__":
    main()
