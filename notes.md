## STS Zero shot classification
For distilroberta-v1, depending on the prompts used, following results were obtained:
* **This is not hateful speech.**, **This is hateful speech.** - 0.64 F1
* **This is not hateful and non toxic content**, **This is hateful and toxic content** - 0.71 F1
* **Friendly, pleasantry, civility, gesture, levity.**, **Hatred, profanity, racial slurs.** - 0.63 F1

We can see that "prompts" significantlly impact the performance of classification.