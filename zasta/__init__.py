# SPDX-FileCopyrightText: 2024-present ImAKappa <imaninconsp1cuouskappa@gmail.com>
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zasta.language import LanguageModel

@dataclass
class Args:
    """
    Args:
        phrases - the number of phrases to generate
    """
    phrases: int

def get_args() -> Args:
    """Get arguments from command-line"""
    parser = ArgumentParser()
    parser.add_argument("phrases", help="Number of phrases to generate", type=int)
    args = parser.parse_args()
    return Args(phrases=args.phrases)

def genz() -> pd.DataFrame:
    filepath = Path("./data/input/genz/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")
    phrases = list(df["phrase"].values)

    gospel_chapters = Path("./data/input/zgospel").glob("*.txt")
    for ch in gospel_chapters:
        verses = ch.read_text(encoding="utf-8").splitlines()
        phrases.extend(verses[1:])
    return phrases

def preprocess(samples: list[str]) -> list[list[str]]:
    """Preprocess samples"""
    return [s.split() for s in samples]

def init_model() -> LanguageModel:
    """Initializes a zoomer LanguageModel"""
    # TODO: Figure out why order 1 only outpus one word
    # Based on tests, order=2 seems to produce the best results; order=1 doesn't work
    # and order > 2 just parrots the training the data bc the training sentences are so short
    lm = LanguageModel(order=2, temperature=5)
    print(lm)
    print()
    samples = genz()
    samples = preprocess(samples)
    lm.batch_train(samples)
    return lm

def chat() -> None:
    """Interactive chat with Zasta language model"""
    raise NotImplementedError()

def main() -> None:
    print("Zasta")
    args = get_args()

    lm = init_model()
    phrases = lm.generate(k=args.phrases)
    print(*phrases, sep="\n")

    