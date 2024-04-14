"""app.py

zasta is a gen-z conversation generator.
It uses a Markov chain-based approach. 
"""
from pathlib import Path
from zasta.markov import Markov
import pandas as pd

if __name__ == "__main__":
    # Gen-Z
    # filepath = Path("./data/genz.csv")
    # df = pd.read_csv(filepath, encoding="utf-8")
    # content = "\n".join(df["phrase"].values)

    # Copypasta
    # content = Path("./data/copypasta.txt").read_text(encoding="utf-8")

    # Drake
    content = Path("./data/drake/drake_lyrics.txt").read_text(encoding="utf-8")

    print("Zasta".center(15, "="))
    print()
    mark = Markov(content)
    num_sentences = 10
    for _ in range(num_sentences):
        print(mark.new_sentence(min_words=15))

    