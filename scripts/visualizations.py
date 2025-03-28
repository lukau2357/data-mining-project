import argparse
import matplotlib.pyplot as plt
import pandas as pd
import langid
import langcodes
import json
import os
import numpy as np

from tqdm import tqdm
from scipy.stats import gaussian_kde

plt.style.use("ggplot")
tqdm.pandas()

def identify_language(sentence : str):
    return langcodes.Language.get((langid.classify(sentence))[0]).display_name()

def sentence_word_length(sentence : str):
    return len(sentence.split(" "))

def word_length_distribution(dataset_path : str, results_dir : str, figures_dir : str):
    df = pd.read_csv(dataset_path)
    print("Calculating sentence length over the given dataset.")
    df["WordLength"] = df["Content"].progress_apply(sentence_word_length)
    print("Done with sentence length calculation.")

    value_counts = df["WordLength"].value_counts().to_dict()
    lengths = []

    for key, value in value_counts.items():
        lengths = lengths + [key] * value

    print("Creating the sentence length distribution plot:")
    kde = gaussian_kde(lengths)
    fig, ax = plt.subplots()

    ax.set_xlabel("Comment length")
    ax.set_ylabel("Frequency")
    ax.set_title("Comment length distribution")

    x = np.linspace(min(lengths), max(lengths), 1000)
    bin_values = np.array(list(range(0, 301, 30)))
    ax.plot(x, kde(x), label = "Estimated PDF")
    ax.hist(lengths, alpha = 0.6, edgecolor = "black", density = True, bins = bin_values)
    ax.legend()
    ax.set_xticks(bin_values)
    fig_file = os.path.join(figures_dir, "sentence_length_distribution.png")
    fig.savefig(fig_file, bbox_inches = "tight")
    print("Sentence length distribution plot created and saved.")

def transform_language_counts(value_counts, threshold : float = 0.1):
    total = sum(v for v in value_counts.values())
    new_dict = {}

    for key, value in value_counts.items():
        if value >= int(total * threshold):
            new_dict[key] = value
        
        else:
            new_dict["Other"] = new_dict.get("Other", 0) + value
    
    return new_dict, total

def language_distribution(dataset_path : str, results_dir : str, figures_dir : str, threshold : float = 0.1):
    df = pd.read_csv(dataset_path)
    print("Performing language classification over the given dataset:")
    df["Language"] = df["Content"].progress_apply(identify_language)
    print("Done with language classification.")
    
    value_counts = df["Language"].value_counts().to_dict()
    value_counts, total = transform_language_counts(value_counts, threshold)

    out_file = os.path.join(results_dir, "language_value_counts.json")
    with open(out_file, "w+") as f:
        json.dump(value_counts, f)
    
    fig, ax = plt.subplots()
    bars = ax.barh(value_counts.keys(), value_counts.values(), edgecolor = "black")
    ax.set_xlabel("Language")
    ax.set_ylabel("Number of comments")
    ax.set_title("Language distribution")

    for bar in bars:
        width = bar.get_width()
        y_position = bar.get_y() + bar.get_height() / 2  # Positioning text vertically centered on the bar

        # Add number on top of the bar
        ax.text(width, y_position, f"{(width / total) * 100:.2f}%", ha = "left", va = "center", fontsize = 10, color = "black")

        # Draw a line from the label to the bar
        ax.plot([width, width + 0.2], [y_position, y_position], color = "black", linewidth = 0.7)

    plt.savefig(os.path.join(figures_dir, "language_distribution.png"), bbox_inches = "tight")

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type = int, default = 41, help = "Random seed for reproducibility.")
    subparsers = parser.add_subparsers(dest = "command")

    split_parser = subparsers.add_parser("word_length_distribution", help = "Creates the word length distribution graph over the given dataset.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    split_parser.add_argument("--dataset_path", type = str, help = "Path to the dataset.", default = "../data/HateSpeechDatasetBalanced.csv")
    split_parser.add_argument("--results_dir", type = str, help = "Directory in which the results will be saved.", default = "../results")
    split_parser.add_argument("--figures_dir", type = str, help = "Directory in which resulting figures will be saved.", default = "../figures")

    language_parser = subparsers.add_parser("language_distribution", help = "Creates language distribution graph over the given dataset", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    language_parser.add_argument("--dataset_path", type = str, help = "Path to the dataset.", default = "../data/HateSpeechDatasetBalanced.csv")
    language_parser.add_argument("--results_dir", type = str, help = "Directory in which the results will be saved - language distribution as an object", default = "../results")
    language_parser.add_argument("--figures_dir", type = str, help = "Directory in which resulting figures will be saved.", default = "../figures")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.command == "language_distribution":
        language_distribution(args.dataset_path, args.results_dir, args.figures_dir)
    
    if args.command == "word_length_distribution":
        word_length_distribution(args.dataset_path, args.results_dir, args.figures_dir)