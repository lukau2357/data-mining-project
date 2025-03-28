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
    split_parser.add_argument("--q", type = float, default = 0.8, help = "Portion of samples to keep in the training set, should be a float in [0, 1].")
    split_parser.add_argument("--output_dir", type = str, help = "Output directory for the new datasets.", default = "../data/")
    
    return parser.parse_args()

def tt_split(dataset_path : str, q : float, output_dir : str, seed : int):
    df = pd.read_csv(dataset_path)
    df = df[["Content", "Label"]]
    # Some rows contain invalid values for Label column, disregard such rows
    df = df[df["Label"] != "Label"]
    y = df["Label"].to_numpy()

    # Preserve class distribution in obtained splits.
    X_train, X_test, _, _ = train_test_split(df, y, train_size = q, random_state = seed, stratify = y)

    train_label_dis = X_train["Label"].value_counts()
    test_label_dis = X_test["Label"].value_counts()

    train_dis_0 = train_label_dis.iloc[0]
    train_dis_1 = train_label_dis.iloc[1]
    train_total = train_dis_0 + train_dis_1

    test_dis_0 = test_label_dis.iloc[0]
    test_dis_1 = test_label_dis.iloc[1]
    test_total = test_dis_0 + test_dis_1

    print(f"0 train samples: {train_dis_0} 1 train samples: {train_dis_1} Total train samples: {train_total}")
    print(f"0 ratio: {(train_dis_0 / train_total):.4f} 1 ratio: {(train_dis_1 / train_total):.4f}")

    print(f"0 test samples: {test_dis_0} 1 test samples: {test_dis_1} Total test samples: {test_total}")
    print(f"0 ratio: {(test_dis_0 / test_total):.4f} 1 ratio: {(test_dis_1 / test_total):.4f}")

    filename = os.path.basename(dataset_path).split(".")[0]
    X_train.to_csv(os.path.join(output_dir, filename + "_train.csv"), index = False)
    X_test.to_csv(os.path.join(output_dir, filename + "_test.csv"), index = False)

if __name__ == "__main__":
    args = parse_args()

    if args.command == "split":
        tt_split(args.dataset_path, args.q, args.output_dir, args.seed)