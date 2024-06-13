"""ngram

Same as 'language.py', but relying more on nltk
"""
from pathlib import Path

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import KneserNeyInterpolated, LanguageModel
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

def genz() -> list[str]:
    filepath = Path("./data/input/genz/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")
    return df["phrase"].values

detokenize = TreebankWordDetokenizer().detokenize

def generate_sentence(model: LanguageModel, num_words: int, random_seed=None) -> str:
    content = []
    for token in model.generate(num_words=num_words, random_seed=random_seed):
        if token == "<s>":
            continue
        if token == "</s>":
            break
        content.append(token)
    return detokenize(content)

samples = [s.split() for s in genz()]

train, vocab = padded_everygram_pipeline(2, samples)

model = KneserNeyInterpolated(2)
model.fit(train, vocab)

print(generate_sentence(model, 100))