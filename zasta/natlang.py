"""natlang"""

from pathlib import Path
import pandas as pd
import nltk

def debug_tokenization(phrase: str) -> None:
    """Prints debug information about tokenization"""
    tagged_words = nltk.pos_tag(nltk.word_tokenize(phrase), tagset="universal")
    justified_words = [word.ljust(max(len(pos), len(word))) for word, pos, in tagged_words]
    justified_tags = [pos.ljust(max(len(pos), len(word))) for word, pos, in tagged_words]
    print(" ".join(justified_words))
    print(" ".join(justified_tags))
    print("-"*3)


if __name__ == "__main__":
    filepath = Path("./data/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")

    for phrase in df["phrase"]:
        debug_tokenization(phrase)
