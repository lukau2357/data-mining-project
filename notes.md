## Dataset properties
* HateSpeechDataset.csv - 417561 sentences, 0.82 of them are non-hateful, 0.18 are hateful
* HateSpeechDatasetBalanced.csv - 700000 sentences, near equal class distribution achieved with data augmentation. First, non-hateful sentences were undersampled. Afterwards, hateful sentences were augmented using BERT masking and Word2Vec synonym augmentation.

## STS Zero shot classification
For distilroberta-v1, depending on the prompts used, following results were obtained, on HateSpeechDatasetBalanced_test:
* **This is not hateful speech.**, **This is hateful speech.** - 0.64 F1
* **This is not hateful and non toxic content**, **This is hateful and toxic content** - 0.71 F1
* **Friendly, pleasantry, civility, gesture, levity.**, **Hatred, profanity, racial slurs.** - 0.63 F1

For sentence-transformers/all-mpnet-base-v2:
* **This is not hateful speech.**, **This is hateful speech.** - 0.56 F1
* **This is not hateful and non toxic content**, **This is hateful and toxic content** - 0.64 F1
* **Friendly, pleasantry, civility, gesture, levity.**, **Hatred, profanity, racial slurs.** - 0.64 F1

Results are decent considering no training was performed, but in order to improve, fine-tuned models will need to be produced. Potentially, try fine-tuning sentence transformer models as well.

With HateSpeechDataset, which is the original non-augmented dataset, we get significantly worse results - for mpnet and distilroberta F1 is in range 0.3 - 0.4, which is very underwhelming. **Investigate what is causing this!**

We can see that "prompts" significantlly impact the performance of classification.

## Encoder-only models
roberta-base 125M parameters
roberta-large 355M parameters

roberta-X tokenizer seems to be cased, which is what we want for our problem!

- Posluziti se nekim od datasetova za Hate Speech detection na Engleskom. Za to koristiti RoBERT-a, ili neki drugi slican encoder-based model za Engleski.
  Eksperimentisati i sa encoder-decoder arhitekturama. Moguce je naravno i koriscenje decoder-only modela, BART?
  
- Alternativno, iskoristiti Bertic: https://huggingface.co/classla/bcms-bertic-frenk-hate, CroSloEn: https://huggingface.co/EMBEDDIA/crosloengual-bert ili neki 
  slican model za fine tunning/adapter nad hate speech dataset-u koji je na nasem jeziku. Tavki dataset-ovi:
    - https://www.clarin.si/repository/xmlui/handle/11356/1433

- Sentences are of fairly low quality due to data augmentation, and furthermore it seems that the dataset contains multiple languages. Maybe use a multi-lingual model here?

## LoRA notes:
Let's try using rank stabilised LoRA, the final lora weight is divided by sqrt(rank) instead of rank.
RoBERTa base:
  * First no target_modules and layers_to_transform specification was tried, defaults to applying LoRA only to keys and values.
  * `target_modules = ["query", "key", "value", "dense"], layers_to_transform = list(range(12))` applies LoRA to all linear transformations within the model, EXCLUDING the classification head!

- Tokenizing input when creating batches costs arround 25 milliseconds for batch size of 128. I don't think it's worth the effort to pre-tokenize the entire dataset.

- Try working with TPUs on Kaggle, see if it speeds up the training.

124647170 - Number of RoBERTa-base parameters
247577856 - Number of T5 v1.1-base parameters. Roughly 2 times higher than RoBERTa, which is expected. To compensate, use lower LoRA rank for T5 fine tunning
to truly compare these models.

alllora model of rank 8 (currently best with 0.9 F1) has 1919234 parameters (counting only LoRA parameters), while the original model has 124647170, which is roughly 65 times less. Training time: 26 hours.

t5v1_1 small model of rank 32 has 1376256 parameters, while the original model has 76961152, which is roughly 55 times less. T5 models take significantlly longer to train, due to label generation! Training time: 64 hours if let for 8 epochs, which is way too long for current hardware.