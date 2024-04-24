"""app

Module for playing around with MarkovGenerator
"""

from pathlib import Path
import pandas as pd
import logging
from zasta.language import LanguageModel
from zasta.tokenizer import Tokenizer
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
log = logging.getLogger("zasta")
# log.level = logging.INFO

from nltk.tokenize import word_tokenize, WhitespaceTokenizer

def genz() -> pd.DataFrame:
    filepath = Path("./data/input/genz/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")
    return df["phrase"].values

def drake() -> list[str]:
    content = Path("./data/input/drake/drake_lyrics.txt").read_text(encoding="utf-8")
    return content.splitlines()

def shakespeare() -> list[str]:
    content = Path("./data/input/shortstory/shakespeare.txt").read_text(encoding="utf-8")
    return filter(None, content.splitlines())

if __name__ == "__main__":
    pd.set_option('display.precision', 2)
    # pd.set_option('display.max_colwidth', None)

    title = "Zasta: Gen(erative) Z text"
    print(title)
    print("="*len(title))

    # --- Load data
    data = genz()

    # --- Preprocess
    t = Tokenizer()
    # samples = [t.word_non_word(text) for text in data]
    samples = [word_tokenize(text) for text in data]
    ws_tokenizer = WhitespaceTokenizer()
    whitespace = [ws_tokenizer.span_tokenize(text) for text in data]

    # --- Configure Model
    mark = LanguageModel(order=3, temperature=5)
    log.info(mark)
    
    # --- Train
    mark.batch_train(samples)
    log.info("\tTraning complete!")
    mark.normalize()
    
    # --- Test

    # --- Measure performance

    # --- Reconfigure Model

    # profiler = MarkovProfiler(samples, mark, k=30)
    # profile = profiler.profile()
    # log.info("\tProfiling complete!")

    print()
    print("Generating phraZes...")
    print()
    # for sentence, novelty in profile[["sentence", "novel"]].drop_duplicates().itertuples(index=False):
    #     novelty_tag = "NEW" if novelty else "OLD"
    #     log.info(f"{novelty_tag} {sentence}")

    #     if novelty:
    #         print(sentence)

    for sentence in mark.generate(k=5, delimiter=" "):
        novelty = "OLD" if sentence in data else "NEW"
        print(f"{novelty} {sentence}")