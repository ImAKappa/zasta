"""language.py

Module for N-Gram language model.
"""
from collections import defaultdict, Counter
from typing import Self
import random
import pandas as pd
import numpy as np
import logging
logging.basicConfig(format="%(levelname)s:%(funcName)s():%(lineno)d\t%(message)s", level=logging.ERROR)

from nltk.util import ngrams
from nltk import ConditionalFreqDist

from zasta.tokenizer import Tokenizer




# TODO: Support more complex Token types (str + part-of-speech tag)

# TODO: Incorporate Kneser-Ney Smoothing for new words
# TODO (Feature) preserve whitespace (non-word) characters in the tokenization step

class LanguageModel:
    """An N-Gram language model"""

    def __init__(self, order=2, temperature=1) -> Self:
        self._order = order
        self._temperature = temperature

        self._stats: defaultdict[tuple[str, ...], Counter] = defaultdict(Counter)
        self._rng = np.random.default_rng()

        self._padding_start = ["<s>"] * (self._order - 1)
        self._padding_end = ["</s>"] * (self._order - 1)

        self._is_normalized = False
        self._words = 0

        # Number of distinct words
        self.vocabulary_size = 0

        self.cfd = ConditionalFreqDist()

    def __repr__(self) -> str:
        return f"LanguageModel(context={self._order}, temperature={self._temperature})"

    def batch_train(self, samples: list[list[str]]) -> None:
        """Trains the language model on many samples.
        Note, that the samples should be pre-processed into tokens first.
        """
        for sample in samples:
            self.train(sample)

    def train(self, sample: list[str]) -> None:
        """Accept one sample of training data.
        Note, the sample should be tokenized first.
        """
        self.vocabulary_size += len(set(sample))
        ngram_iter = list(ngrams(
            sample,
            self._order,
            pad_left=True,
            left_pad_symbol="<s>",
            pad_right=True,
            right_pad_symbol="</s>"
        )) 
        for ngram in ngram_iter:
            *history, current_token = ngram
            self._stats[tuple(history)][current_token] += 1
            self.cfd[tuple(history)][current_token] += 1

    def normalize(self) -> None:
        if self._is_normalized:
            return
        """Normalizes a language model stats object to have probabilities"""
        d = {}
        for ngram, counter in self._stats.items():
            N = sum(counter.values())
            d[ngram] = {token:count/N for token, count in counter.items()}
        self._stats = d
        self._is_normalized = True

    def generate(self, k: int = 1, delimiter: str = " ") -> list[list[str]]:
        """Generate multiple ngrams"""
        return [self._generate_sample(delimiter) for _ in range(k)]

    def _generate_sample(self, delimiter: str) -> list[str]:
        """Generate sample of tokens"""
        tokens = tuple(self._padding_start)
        while True:
            history = tokens[-self._order+1:]
            if history not in self._stats:
                break
            tokens = tokens + (self._choose_token(self._stats[history]),)
        return delimiter.join(tokens).strip("<s> ").strip(" </s>")

    def _choose_token(self, c: Counter) -> str:
        """Chooses a token based on frequency count"""
        weights = np.array(list(c.values()))
        new_weights = np.exp(weights / self._temperature)
        new_weights /= np.sum(new_weights)
        token = random.choices(list(c.keys()), new_weights)[0] 
        return token

if __name__ == "__main__":
    # from zasta.profiler import MarkovProfiler
    from pprint import pprint
    from nltk import word_tokenize

    # pd.set_option('display.precision', 2)
    samples = [
        "We're no strangers to love 🥰",
        "You know the rules and so do I",
        "A full commitment's what I'm thinking of",
        "You wouldn't get this from any other guy",
    ]

    # --- Preprocess
    # t = Tokenizer()
    # samples = [t.word_non_word(text) for text in samples]
    samples = [word_tokenize(text) for text in samples]

    # --- Train
    m = LanguageModel(order=2)
    m.batch_train(samples)
    pprint(m._stats)
    pprint(m.cfd.conditions())
    # m.normalize()
    # pprint(m._stats)

    # -- Metrics
    print(f"{m.vocabulary_size=}")
    print()

    for sentence in m.generate(k=5):
        print(sentence)
    # profiler = MarkovProfiler(samples, m, k=5)
    # profile = profiler.profile()
    # print(profiler.summarize(profile))

# TODO (Performance) Use `nltk.pos_tag_sents` instead of `nltk.pos_tag`
# TODO (Feature) Allow user to export/compile a model
# TODO (Feature) Allow users to treat newlines as important
#   list mode (just list of sentences) vs plain-text mode (don't ignore spaces)
# TODO (Ideate) Figure out if the training process can be parallized
#   like by chunking the training data and constructing independent groups of chains, then combining chains with some well-defined addition operator for chains