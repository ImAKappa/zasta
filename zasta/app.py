"""app

Module for playing around with MarkovGenerator
"""

from pathlib import Path
import pandas as pd
import logging
from zasta.markov import MarkovGenerator, MarkovProfiler
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
log = logging.getLogger("zasta")
# log.level = logging.INFO

def genz() -> pd.DataFrame:
    filepath = Path("./data/input/genz/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")
    return df["phrase"].values

if __name__ == "__main__":
    pd.set_option('display.precision', 2)
    # pd.set_option('display.max_colwidth', None)

    samples = genz()
    # samples = drake()

    title = "Zasta: Gen(erative) Z text"
    print(title)
    print("="*len(title))

    mark = MarkovGenerator(context=2, temperature=5)
    log.info(mark)
    mark.train(samples)
    log.info("\tTraning complete!")
    
    profiler = MarkovProfiler(samples, mark, k=30)
    profile = profiler.profile()
    log.info("\tProfiling complete!")

    print()
    print("Generating phraZes...")
    print()
    for sentence, novelty in profile[["sentence", "novel"]].drop_duplicates().itertuples(index=False):
        novelty_tag = "NEW" if novelty else "OLD"
        log.info(f"{novelty_tag} {sentence}")

        if novelty:
            print(sentence)