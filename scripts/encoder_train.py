import argparse
import os
import torch
import pandas as pd
import evaluate
import yaml

from transformers import AutoModelForSequenceClassification, RobertaTokenizerFast, PreTrainedTokenizerFast, TrainingArguments, Trainer
from torch.utils.data import Dataset
from typing import List
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from functools import partial

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("../secrets.yaml", "r") as f:
    data = yaml.safe_load(f)
    os.environ["HF_TOKEN"] = data["hf_token"]

def collate_fn(batch : List[str], tokenizer : PreTrainedTokenizerFast):
    comments = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    res = tokenizer(
        comments,
        padding = "max_length",
        return_tensors = "pt",
        return_attention_mask = True
    )
    
    res["labels"] = labels
    return res

def compute_f1(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')

    return {
        "f1": f1
    }

class CommentClassificationDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.head(1000)
        self.N = len(self.df)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        res = self.df.iloc[idx]
        return res["Content"], int(res["Label"])
    
def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type = int, default = 41, help = "Random seed for reproducibility.")
    subparsers = parser.add_subparsers(dest = "command")

    split_parser = subparsers.add_parser("train", help = "Train a transformer encoder model for comment classification", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    split_parser.add_argument("--train_dataset_path", type = str, help = "Path to the CSV file containing the training dataset.", default = "../data/HateSpeechDatasetBalanced_train.csv")
    split_parser.add_argument("--test_dataset_path", type = str, help = "Path to the CSV file containing the test dataset.", default = "../data/HateSpeechDatasetBalanced_test.csv")
    split_parser.add_argument("--model_url", type = str, help = "HuggingFace URL of the desired base model.", default = "roberta-base")
    split_parser.add_argument("--classifier_dropout", type = float, help = "Dropout for final classification head.", default = 0.1)
    split_parser.add_argument("--batch_size", type = int, help = "Batch size to be used during training.", default = 64)
    split_parser.add_argument("--grad_accum", type = int, help = "Number of gradient accumulation steps during training, making the effective batch size grad_accum * batch_size", default = 4)
    split_parser.add_argument("--lora_rank", type = int, help = "Which LoRA rank to use for training.", default = 8)
    split_parser.add_argument("--checkpoint_dir", type = str, help = "Directory that will contain the model checkpoint.", default = "../encoder_checkpoints/roberta_test")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed = args.seed

    # TODO: Be careful with this in multi-GPU setting!
    torch.manual_seed(seed)

    # Most RoBERTa parameters can be passed to from_pretrained.
    # Documentation: https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/roberta#transformers.RobertaForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(args.model_url, num_labels = 2, 
                                                               id2label = {0: "Non-hateful", 1: "Hateful"}, 
                                                               classifier_dropout = args.classifier_dropout,
                                                               token = False).to(DEVICE)
    
    trainable_model_params = sum((p.numel() for p in model.parameters()))

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_url, token = False)
    train_dataset = CommentClassificationDataset(args.train_dataset_path)
    test_dataset = CommentClassificationDataset(args.test_dataset_path)

    # Documentation: https://huggingface.co/docs/peft/en/package_reference/lora
    lora_config = LoraConfig(
        r = args.lora_rank, 
        lora_dropout = 0.1,
        bias = "none",
        lora_alpha = 16,
        task_type = "SEQ_CLS" 
    )
    
    model = get_peft_model(model, lora_config)
    lora_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of model trainable parameters: {trainable_model_params}")
    print(f"Number of LoRA trainable parameters: {lora_trainable_parameters}")
    print(f"Conservation ratio: {trainable_model_params / lora_trainable_parameters:.4f}")

    trainer_args = TrainingArguments(
        output_dir = args.checkpoint_dir,
        overwrite_output_dir = True,
        fp16 = False,
        report_to = "none",
        gradient_accumulation_steps = args.grad_accum,
        per_device_train_batch_size = args.batch_size // args.grad_accum,
        seed = args.seed,
        data_seed = args.seed,
        # learning_rate = hp["learning_rate"],
        weight_decay = 1e-4,
        save_strategy = "epoch",
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        num_train_epochs = 2,
        # Combination of save_total_limit and load_best_model_at_end ensures that model with best loss on dataset
        # and most recent model are checkpointed. 
        # https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments.save_total_limit
        # https://github.com/huggingface/transformers/issues/19041
        load_best_model_at_end = True,
        save_total_limit = 2,
        greater_is_better = True,
        metric_for_best_model = "f1"
    )

    trainer = Trainer(
        model = model,
        args = trainer_args,
        data_collator = partial(collate_fn, tokenizer = tokenizer),
        # Evaluate models based on loss achieved on entire training dataset
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = compute_f1
    )

    trainer.train()