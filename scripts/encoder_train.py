import argparse
import torch
import pandas as pd
import os
import yaml
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
from typing import List
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from functools import partial

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "secrets.yaml"), "r") as f:
   data = yaml.safe_load(f)
   os.environ["HF_TOKEN"] = data["hf_token"]

def collate_fn(batch : List[str], tokenizer : AutoTokenizer):
    comments = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    res = tokenizer(
        comments,
        padding = True,
        truncation = True,
        return_tensors = "pt",
        return_attention_mask = True
    )
    
    res["labels"] = labels
    return res

def compute_f1(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis = -1)
    f1 = f1_score(labels, preds, average = "weighted")

    return {
        "eval_f1": f1
    }

class CommentClassificationDataset(Dataset):
    def __init__(self, dataset_path, data_ratio):
        self.data_ratio = data_ratio
        self.df = pd.read_csv(dataset_path)
        self.N = int(len(self.df) * data_ratio)
        self.df = self.df[:self.N]

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
    split_parser.add_argument("--lora_dropout", type = float, help = "Dropout for LoRA layers.", default = 0.1)
    split_parser.add_argument("--lora_alpha", type = float, help = "Scaling factor for lora layers, is actually computed as lora_alpha / lora_rank.", default = 8)
    split_parser.add_argument("--learning_rate", type = float, help = "Optimizer learning rate.", default = 5e-5)
    split_parser.add_argument("--weight_decay", type = float, help = "Optimizer weight decay.", default = 1e-3)
    split_parser.add_argument("--batch_size", type = int, help = "Batch size to be used during training. In distributed enviroments, this should be interpreted as per-GPU batch size.", default = 64)
    split_parser.add_argument("--grad_accum", type = int, help = "Number of gradient accumulation steps during training, making the effective batch size grad_accum * batch_size", default = 4)
    split_parser.add_argument("--lora_rank", type = int, help = "Which LoRA rank to use for training.", default = 8)
    split_parser.add_argument("--checkpoint_dir", type = str, help = "Directory that will contain the model checkpoint, or specifies the directory that continues model checkpoint to continue training.", default = "../encoder_checkpoints/roberta_test")
    split_parser.add_argument("--from_checkpoint", type = bool, default = None, nargs = "?", const = True, help = "Continue training from latest checkpoint in the given checkpoint directory.")
    split_parser.add_argument("--train_epochs", type = int, help = "Number of model training epochs.", default = 2)
    split_parser.add_argument("--warmup_ratio", type = float, help = "Warmup ratio for learning rate scheduler.", default = 0.1)
    split_parser.add_argument("--data_ratio", type = float, help = "Ratio of data to take for trainig, mainly used for testing purposes.", default = 1)
    split_parser.add_argument("--disable_tqdm", type = bool, default = False, nargs = "?", const = True, help = "Disables TQDM progress bars, they are problematic when run from Jupyter notebook it seems.")
    # split_parser.add_argument("--tpu_num_cores", type = int, default = None, help = "Number of TPU cores to use when training with TPU.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed = args.seed

    # TODO: Be careful with this in multi-GPU setting!
    # torch.manual_seed(seed)

    # Most RobertaConfig parameters can be passed to from_pretrained.
    # Documentation: https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/roberta#transformers.RobertaForSequenceClassification
    id2label = {0: "Non-hateful", 1: "Hateful"}
    label2id = {"Hateful": 1, "Non-hateful": 0}
    model = AutoModelForSequenceClassification.from_pretrained(args.model_url, num_labels = 2, 
                                                               classifier_dropout = args.classifier_dropout,
                                                               id2label = id2label,
                                                               label2id = label2id,
                                                               token = False).to(DEVICE)
    trainable_model_params = sum((p.numel() for p in model.parameters()))

    tokenizer = AutoTokenizer.from_pretrained(args.model_url, token = False)
    train_dataset = CommentClassificationDataset(args.train_dataset_path, args.data_ratio)
    test_dataset = CommentClassificationDataset(args.test_dataset_path, args.data_ratio)

    # Documentation: https://huggingface.co/docs/peft/en/package_reference/lora
    lora_config = LoraConfig(
        r = args.lora_rank, 
        lora_dropout = args.lora_dropout,
        # Try without lora biases.
        bias = "none",
        lora_alpha = args.lora_alpha,
        task_type = "SEQ_CLS",
        # Which layers are affected by LoRA, should be modified depending on the chosen model!
        target_modules = ["query", "key", "value", "dense"],
        layers_to_transform = list(range(12)),
        use_rslora = True
    )
    
    model = get_peft_model(model, lora_config)
    lora_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model trainable parameters: {trainable_model_params}")
    print(f"Number of LoRA trainable parameters: {lora_trainable_parameters}")
    print(f"Conservation ratio: {trainable_model_params / lora_trainable_parameters:.4f}")

    # Cosine learning rate scheduling with linear warmup
    # Documentation: https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup
    trainer_args = TrainingArguments(
        output_dir = args.checkpoint_dir,
        overwrite_output_dir = True,
        fp16 = False,
        report_to = "none",
        gradient_accumulation_steps = args.grad_accum,
        per_device_train_batch_size = args.batch_size,
        seed = args.seed,
        data_seed = args.seed,
        # defaults to 5e-5
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        save_strategy = "epoch",
        eval_strategy = "epoch",
        logging_strategy = "epoch",
        num_train_epochs = args.train_epochs,
        # Combination of save_total_limit and load_best_model_at_end ensures that model with best loss on dataset
        # and most recent model are checkpointed. 
        # https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments.save_total_limit
        # https://github.com/huggingface/transformers/issues/19041
        save_total_limit = 2,
        load_best_model_at_end = True,
        greater_is_better = True,
        metric_for_best_model = "eval_f1",
        lr_scheduler_type = "cosine",
        warmup_ratio = args.warmup_ratio,
        log_level = "info",
        disable_tqdm = args.disable_tqdm,
        ddp_find_unused_parameters = False, # Fixes Kaggle warning, prevents one additional forward pass for the model.
        # tpu_num_cores = args.tpu_num_cores
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

    trainer.train(resume_from_checkpoint = args.from_checkpoint)