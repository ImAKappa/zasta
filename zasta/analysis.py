"""app.py

zasta is a gen-z conversation generator.
It uses a Markov chain-based approach. 
"""
from pathlib import Path
from zasta.markov import MarkovGenerator, new_sentences, MarkovProfiler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def genz() -> pd.DataFrame:
    filepath = Path("./data/input/genz/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")
    return df["phrase"].values

def drake() -> list[str]:
    content = Path("./data/input/drake/drake_lyrics.txt").read_text(encoding="utf-8")
    return content.splitlines()

if __name__ == "__main__":
    samples = genz()
    # samples = drake()

    print("Zasta".center(15, "="))
    print()

    generators = [
        MarkovGenerator(context=4, temperature=1),
        MarkovGenerator(context=2, temperature=3),
        MarkovGenerator(context=1, temperature=4),
    ]

    pd.set_option('display.precision', 2)
    pd.set_option('display.max_colwidth', None)
    sns.set_theme()

    profiles = []

    for mg in generators:
        print(mg)
        mg.train(samples)
        print("\tTraning complete")
        profiler = MarkovProfiler(samples, mg, k=400)
        profile = profiler.profile()
        profiles.append(profile)
        results = profiler.summarize(profile)
        print("\tProfiling complete")
        report_path = Path(f"./data/reports/{mg}.txt")
        results = report_path.write_text(results, encoding="utf-8")
        print(f"\tSaved report to '{report_path}'")

    profiles = pd.concat(profiles)
    
    # --- Novelty compared to training data
    pd.crosstab(profiles["model"], profiles["novel"]).plot(kind="bar", stacked=True, rot=0.75)
    plt.show()

    # --- Duplication
    pd.crosstab(profiles["model"], profiles["is_duplicate"]).plot(kind="bar", stacked=True, rot=0.75)
    plt.show()

    # --- Lexical diversity
    sns.kdeplot(profiles, x="lexical_diversity", hue="model")
    plt.show()

    # --- Lexical diversity
    sns.scatterplot(profiles, x="num_words", y="lexical_diversity", hue="model", alpha=0.6)
    plt.show()