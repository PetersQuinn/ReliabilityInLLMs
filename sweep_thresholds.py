import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="CSV produced by build_reliability_dataset.py")
    ap.add_argument("--label", type=str, default="trustworthy_em", choices=["trustworthy_em", "trustworthy_f1_80"])
    ap.add_argument("--n", type=int, default=60, help="number of thresholds to sweep")
    ap.add_argument("--out", type=str, default="artifacts/reliability/threshold_sweep.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # We want margin = null_score - best_span_score
    if "null_score" not in df.columns or "best_span_score" not in df.columns:
        raise ValueError("CSV must include null_score and best_span_score columns.")

    margin = df["null_score"].astype(float) - df["best_span_score"].astype(float)

    # Sweep thresholds over the empirical margin range
    lo, hi = float(margin.quantile(0.01)), float(margin.quantile(0.99))
    thresholds = np.linspace(lo, hi, args.n)

    rows = []
    gold_has = df["gold_has_answer"].astype(int).to_numpy()
    exact = df["exact"].astype(int).to_numpy()
    f1 = df["f1"].astype(float).to_numpy()

    for t in thresholds:
        abstain = (margin.to_numpy() > t)  # predict no-answer
        answered = ~abstain

        # Recompute trust labels under your rules
        # If gold has answer and abstain -> untrust
        # If gold has no answer and abstain -> trust
        # Else trust based on EM or F1>=0.8
        if args.label == "trustworthy_em":
            trust = np.where(
                abstain & (gold_has == 1), 0,
                np.where(abstain & (gold_has == 0), 1,
                         (exact == 1).astype(int))
            )
        else:
            trust = np.where(
                abstain & (gold_has == 1), 0,
                np.where(abstain & (gold_has == 0), 1,
                         (f1 >= 0.8).astype(int))
            )

        coverage = answered.mean()
        abstain_rate = abstain.mean()
        trust_rate_overall = trust.mean()

        # Trust among answered only (avoid division by zero)
        if answered.sum() > 0:
            trust_rate_answered = trust[answered].mean()
        else:
            trust_rate_answered = np.nan

        rows.append({
            "threshold": t,
            "coverage": coverage,
            "abstain_rate": abstain_rate,
            "trust_rate_overall": trust_rate_overall,
            "trust_rate_answered": trust_rate_answered,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Saved sweep: {args.out}")

    # Plot: coverage vs trustworthiness
    plt.figure()
    plt.plot(out_df["coverage"], out_df["trust_rate_answered"])
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel(f"{args.label} among answered")
    plt.title("Selective prediction curve (answered accuracy vs coverage)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: plot abstain_rate vs overall trust
    plt.figure()
    plt.plot(out_df["abstain_rate"], out_df["trust_rate_overall"])
    plt.xlabel("Abstain rate")
    plt.ylabel(f"{args.label} overall")
    plt.title("Overall trust vs abstention")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
