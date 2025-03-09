import torch
import numpy as np
import argparse
import pandas as pd
import os
import tqdm

from transformers import AutoTokenizer, AutoModel
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
        embeddings = model.encode(sentences)
        np.save(f"{output_dir}/{i // encode_batch_size}.npy", embeddings)

@torch.no_grad()
def predict(embeddings_dir : str, model : SentenceTransformer, model_label : str, target_labels : List[str], output_dir : str):
    files = os.listdir(embeddings_dir)
    files = sorted(files, key = lambda x: int(x.split(".")[0]))
    target_embeddings = model.encode(target_labels)
    res = []

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Target labels:")
    for label in target_labels:
        print(label)

    print("Performing predictions using precomputed sentence embeddings.")

    for file in tqdm.tqdm(files):
        embeddings = np.load(f"{embeddings_dir}/{file}")
        sims = embeddings @ target_embeddings.T
        # sims = model.similarity(embeddings, target_embeddings)
        predictions = np.argmax(sims, axis = 1, keepdims = True)
        sims = np.concatenate((sims, predictions), axis = 1)
        res.append(sims)

    res = np.concatenate(res, axis = 0)
    np.save(f"{output_dir}/{model_label}.npy", res)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_url", type = str, help = "Specify HuggingFace SentenceTransformer model URL, for example: sentence-transformers/all-distilroberta-v1", default = "sentence-transformers/all-distilroberta-v1")
    args.add_argument("--mode", type = str, help = "Specify operating mode. embed - Compute embeddings over the given dataset.")
    args.add_argument("--dataset_path", type = str, help = "Specify path to file that contains dataset in .csv format.", default = "../data/HateSpeechDatasetBalanced.csv")
    args.add_argument("--encode_batch_size", type = int, help = "Specify batch size used for generating dataset embeddings", default = 1024)
    args.add_argument("--output_dir", type = str, help = "Override output directory optionally.")
    args.add_argument("--embeddings_dir", type = str, help = "Override path to embeddings file optionally.")
    args.add_argument("--target_labels_path", type = str, help = "File that contains target labels. Assumption is that each label is seperated by a new line.", default = "../data/target_labels.txt")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_url = args.model_url
    mode = args.mode
    dataset_path = args.dataset_path
    encode_batch_size = args.encode_batch_size
    output_dir = args.output_dir
    embeddings_dir = args.embeddings_dir
    target_labels_path = args.target_labels_path

    assert mode in ["embed", "predict"], f"Invalid mode {mode} selected. Supported modes: embed, predict"

    model = SentenceTransformer(model_url, device = DEVICE, token = False)
    model_label = model_url.split("/")[-1]

    if not output_dir:
        output_dir = f"../data/sts_embeddings/{model_label}" if mode == "embed" else "../results"

    if not embeddings_dir:
        embeddings_dir = f"../data/sts_embeddings/{model_label}"

    if mode == "embed":
        df = pd.read_csv(dataset_path)
        embed_dataset(df, model, output_dir, encode_batch_size = encode_batch_size)
    
    if mode == "predict":
        with open(target_labels_path, "r") as f:
            target_labels = f.read().split("\n")

        predict(embeddings_dir, model, model_label, target_labels, output_dir)