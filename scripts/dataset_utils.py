import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type = int, default = 41, help = "Random seed for reproducibility.")
    subparsers = parser.add_subparsers(dest = "command")

    split_parser = subparsers.add_parser("split", help = "Perform ordinary train/val split of the given dataset.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    split_parser.add_argument("--dataset_path", type = str, help = "Path to the dataset.", default = "../data/HateSpeechDatasetBalanced.csv")
    split_parser.add_argument("--q", type = float, default = 0.9, help = "Portion of samples to keep in the training set, should be a float in [0, 1].")
    split_parser.add_argument("--output_dir", type = str, help = "Output directory for the new datasets.", default = "../data/")
    
    return parser.parse_args()

def tt_split(dataset_path : str, q : float, output_dir : str, seed : int):
    df = pd.read_csv(dataset_path)
    y = df["Label"].to_numpy()

    # Preserve class distribution in obtained splits.
    X_train, X_test, _, _ = train_test_split(df, y, train_size = q, random_state = seed, stratify = y)
    print(len(X_train))
    print(len(X_test))

    print(X_train["Label"].value_counts())
    print(X_test["Label"].value_counts())

    filename = os.path.basename(dataset_path).split(".")[0]
    X_train.to_csv(os.path.join(output_dir, filename + "_train.csv"), index = False)
    X_test.to_csv(os.path.join(output_dir, filename + "_test.csv"), index = False)

def test(dataset_path : str):
    df = pd.read_csv(dataset_path)
    df.to_csv("../test.csv", index = False, lineterminator = "\n")

if __name__ == "__main__":
    args = parse_args()

    if args.command == "split":
        tt_split(args.dataset_path, args.q, args.output_dir, args.seed)
        test(args.dataset_path)
    