## STS Zero shot classification
For distilroberta-v1, depending on the prompts used, following results were obtained:
* **This is not hateful speech.**, **This is hateful speech.** - 0.64 F1
* **This is not hateful and non toxic content**, **This is hateful and toxic content** - 0.71 F1
* **Friendly, pleasantry, civility, gesture, levity.**, **Hatred, profanity, racial slurs.** - 0.63 F1

We can see that "prompts" significantlly impact the performance of classification.

- Posluziti se nekim od datasetova za Hate Speech detection na Engleskom. Za to koristiti RoBERT-a, ili neki drugi slican encoder-based model za Engleski.
  Eksperimentisati i sa encoder-decoder arhitekturama. Moguce je naravno i koriscenje decoder-only modela, BART?
  
- Alternativno, iskoristiti Bertic: https://huggingface.co/classla/bcms-bertic-frenk-hate, CroSloEn: https://huggingface.co/EMBEDDIA/crosloengual-bert ili neki 
  slican model za fine tunning/adapter nad hate speech dataset-u koji je na nasem jeziku. Tavki dataset-ovi:
    - https://www.clarin.si/repository/xmlui/handle/11356/1433

- Sentences are of fairly low quality due to data augmentation, and furthermore it seems that the dataset contains multiple languages. Maybe use a multi-lingual model here?