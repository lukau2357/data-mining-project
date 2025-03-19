import pandas as pd
import argparse
import os
import json
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type = str, help = "Path to evaluation dataset.", default = "../data/HateSpeechDatasetBalanced_test.csv")
    args.add_argument("--predictions_path", type = str, help = "Path to model predictions file.", default = "../results/all-distilroberta-v1.npy")
    args.add_argument("--missclassified_sentences_limit", type = int, help = "Limit the number of missclassified sentences in the output.", default = 200)
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    predictions_path = args.predictions_path
    missclassified_sentences_limit = args.missclassified_sentences_limit
    model_label = os.path.basename(predictions_path).split(".")[0]

    df = pd.read_csv(dataset_path)
    sentence_limit = missclassified_sentences_limit

    true = df["Label"].to_numpy()
    results = np.load(predictions_path)

    # Second column contains actual predictions
    predictions = results[:, 2]
    miss = true != predictions

    if sum(miss) > 0:
        counter = 0
        examples = []

        for i in range(len(miss)):
            if miss[i]:
                counter += 1
                sentence = df.iloc[i]['Content']
                true_label = true[i]
                predicted_label = predictions[i]

                examples.append({
                    "sentence": sentence,
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "similarity_0": results[i, 0],
                    "similarity_1": results[i, 1]
                })

                if counter > sentence_limit:
                    break
            
            with open(f"./missclassified_{model_label}.json", "w+") as f:
                json.dump(examples, f, indent = 4)

    precision = precision_score(true, predictions)
    recall = recall_score(true, predictions)
    f1 = f1_score(true, predictions)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")