Project for Data Mining course at University of Montenegro, Computer Science, master studies - Hate speech detection with transformer neural networks.

Used dataset: [hate speech detection dataset](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset?select=HateSpeechDatasetBalanced.csv)

Some scripts require HuggingFace API token, so create `secrets.yaml` file and add a single field, `hf_token` with your HuggingFace token:
```yaml
# secrets.yaml
hf_token: <your_token>
```

Alternatively, you can inspect the main notebook directly from Kaggle: [notebook](https://www.kaggle.com/code/lukautjesinovic/roberta-lora-fine-tunning)

3 models were tested:
* [all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1) from [sentence-transformers](https://www.sbert.net/), classification treated as a retrieval problem. To each class we assign a descriptive textual label, and each sentence is classified based on sentence-to-label compatibility. The descriptive labels are of course hyperparameters, and we found that this algorithm is very sensitive to the choice of these labels (more details in the seminar report). Best achieved F1 with this model was 0.77, albeit without any fine-tuning, just using off-the-shelf sentence embeddings. You can find sentence-transformers related scripts in the `/scripts` directory.

* LoRAs for [roberta-base](https://huggingface.co/FacebookAI/roberta-base). Different LoRAs for this model were trained, and the best one achieved 0.91 macro-averaged F1. You can find train and inference code in the given notebook, or visit the previous Kaggle link. 

* LoRAs for [flan-t5-base](https://huggingface.co/google/flan-t5-base). Only one LoRA was trained for this model, and it achieved 0.86 macro-averaged F1.