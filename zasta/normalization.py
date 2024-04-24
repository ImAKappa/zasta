"""normalization

Module for text normalization of Gen Z language


1. Tokenizing (segmenting) words
2. Normalizing word forms
3. Segmenting sentences
"""
from collections import Counter

def freq_map(s: str) -> Counter:
    """Constructs a frequency map of words in a corpus.
    Assumes that words are separated by a space.
    """
    s = s.lower()
    words = s.split()
    return Counter(words)

if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint

    df = pd.read_csv("./data/input/genz/genz.csv", encoding="utf-8")
    genz = " ".join(df["phrase"].values)
    frequencies = freq_map(genz)
    pprint(frequencies)