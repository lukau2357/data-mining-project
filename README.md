# data-mining-project
Project for Data Mining course at University of Montenegro, Computer Science, master studies

Dataset used: https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset?select=HateSpeechDatasetBalanced.csv

Some scripts require HuggingFace API token, so create `secrets.yaml` file and add a single field, `hf_token` with your HuggingFace token:
```yaml
# secrets.yaml
hf_token: <your_token>
```

# TODOS
* Solve weights_only = False when working with checkpoints