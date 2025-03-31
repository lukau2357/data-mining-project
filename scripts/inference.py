import torch
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftConfig, PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading LoRA for inference
# https://huggingface.co/docs/peft/v0.6.0/en/task_guides/token-classification-lora#train
checkpoint_path = r"C:\Users\Korisnik\Desktop\pmf_master_new\year_1\semester_2\data_mining\data-mining-project\encoder_checkpoints\roberta_base_initial\checkpoint-108918"
lora_config = PeftConfig.from_pretrained(checkpoint_path)
id2label = {0: "Non-hateful", 1: "Hateful"}
label2id = {"Hateful": 1, "Non-hateful": 0}
base_model = AutoModelForSequenceClassification.from_pretrained(lora_config.base_model_name_or_path, id2label = id2label, label2id = label2id).to(device)
tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, checkpoint_path).to(device)
model.eval()

with torch.no_grad():
    sentences = ["Thank you very much.", "I hate you!"]
    enc = tokenizer(sentences, padding = True, truncation = True, return_tensors = "pt").to(device)
    res = model(**enc)

    logits = res.logits.cpu()
    probs = torch.nn.functional.softmax(logits, dim = -1)
    preds = torch.argmax(probs, dim = -1)
    counter = 0

    for pred, sentence in zip(preds, sentences):
        print(f"{sentence} Prediction: {model.config.id2label[int(pred.item())]} Probabilities: {probs[counter][0]:.4f} {probs[counter][1]:.4f}")
        counter += 1