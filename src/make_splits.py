import argparse
import os
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--holdout_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="artifacts/reliability/splits")
    p.add_argument("--val_frac", type=float, default=0.5, help="fraction of holdout to use as val (rest is test)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    holdout_df = pd.read_csv(args.holdout_csv)

    # Shuffle holdout and split into val/test
    holdout_df = holdout_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(holdout_df) * args.val_frac)

    val_df = holdout_df.iloc[:n_val].copy()
    test_df = holdout_df.iloc[n_val:].copy()

    train_path = os.path.join(args.out_dir, "trust_train.csv")
    val_path = os.path.join(args.out_dir, "trust_val.csv")
    test_path = os.path.join(args.out_dir, "trust_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved splits:")
    print(" ", train_path, len(train_df))
    print(" ", val_path, len(val_df))
    print(" ", test_path, len(test_df))

if __name__ == "__main__":
    main()
