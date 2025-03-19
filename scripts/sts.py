import torch
import numpy as np
import argparse
import pandas as pd
import os
import tqdm

from sentence_transformers import SentenceTransformer
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def embed_dataset(df : pd.DataFrame, model : SentenceTransformer, output_dir : str, encode_batch_size : int = 1024):
    print(f"Precomputing dataset sentence embeddings. Encoding batch size: {encode_batch_size}")
    N = len(df)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in tqdm.tqdm(range(0, N, encode_batch_size)):
        sentences = df[i : i + encode_batch_size]["Content"].tolist()
        embeddings = model.encode(sentences, normalize_embeddings = True)
        np.save(f"{output_dir}/{i // encode_batch_size}.npy", embeddings)

@torch.no_grad()
def predict(embeddings_dir : str, model : SentenceTransformer, model_label : str, target_labels : List[str], output_dir : str, threshold : float = 0):
    files = os.listdir(embeddings_dir)
    files = sorted(files, key = lambda x: int(x.split(".")[0]))
    target_embeddings = model.encode(target_labels, normalize_embeddings = True)

    res = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Target labels:")
    for label in target_labels:
        print(label)

    print("Performing predictions using precomputed sentence embeddings.")

    for file in tqdm.tqdm(files):
        embeddings = np.load(f"{embeddings_dir}/{file}")
        # sims = embeddings @ target_embeddings.T
        # Should be equivalent to the following line, but for safety use the model function.
        sims = model.similarity(embeddings, target_embeddings).cpu().numpy()
        # Take most similar label as prediction.
        predictions = np.argmax(sims, axis = 1, keepdims = True)
        # predictions using cosine similairty threshold on hateful label only, did not produce good results.
        # predictions = np.where(sims[:, 1] >= threshold, 1, 0).reshape((-1, 1))
        sims = np.concatenate((sims, predictions), axis = 1)
        res.append(sims)

    res = np.concatenate(res, axis = 0)
    np.save(f"{output_dir}/{model_label}.npy", res)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest = "command")

    embed_parser = subparsers.add_parser("embed", help = "Compute sentence embeddings over the given dataset.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    embed_parser.add_argument("--model_url", type = str, help = "Specify HuggingFace SentenceTransformer model URL, for example: sentence-transformers/all-distilroberta-v1", default = "sentence-transformers/all-distilroberta-v1")
    embed_parser.add_argument("--dataset_path", type = str, help = "Specify path to file that contains dataset in .csv format.", default = "../data/HateSpeechDatasetBalanced_test.csv")
    embed_parser.add_argument("--encode_batch_size", type = int, help = "Specify batch size used for generating dataset embeddings", default = 1024)
    embed_parser.add_argument("--output_dir", default = "../data/sts_embeddings", type = str, help = "Output directory that will contain sentence embeddings. In this directory, a new directory will be created, named by the model, and this inner directory will contain the actual embeddings.")

    predict_parser = subparsers.add_parser("predict", help = "Use computed embeddings to make predictions.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    predict_parser.add_argument("--model_url", type = str, help = "Specify HuggingFace SentenceTransformer model URL, for example: sentence-transformers/all-distilroberta-v1", default = "sentence-transformers/all-distilroberta-v1")
    predict_parser.add_argument("--embeddings_dir", type = str, default = "../data/sts_embeddings/all-distilroberta-v1", help = "Override path to embeddings file optionally.")
    predict_parser.add_argument("--target_labels_path", type = str, help = "File that contains target labels. Assumption is that each label is seperated by a new line.", default = "../data/target_labels.txt")
    predict_parser.add_argument("--output_dir", type = str, default = "../results", help = "Override output directory optionally.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_url = args.model_url
    output_dir = args.output_dir

    model = SentenceTransformer(model_url, device = DEVICE, token = False)
    model.eval()
    model_label = model_url.split("/")[-1]

    if args.command == "embed":
        df = pd.read_csv(args.dataset_path)
        output_dir = os.path.join(output_dir, model_label)
        embed_dataset(df, model, output_dir, encode_batch_size = args.encode_batch_size)
    
    if args.command == "predict":
        with open(args.target_labels_path, "r") as f:
            target_labels = f.read().split("\n")

        predict(args.embeddings_dir, model, model_label, target_labels, output_dir)