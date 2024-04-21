"""markov.py

Module for generating markov chains
"""
from collections import defaultdict
from typing import Self, Optional, NamedTuple
import random
from dataclasses import dataclass
import re
from pprint import pprint
from io import StringIO
import pandas as pd
import numpy as np
import logging
logging.basicConfig(format="%(levelname)s:%(funcName)s():%(lineno)d\t%(message)s", level=logging.ERROR)

import nltk

# TODO: Incorporate Kneser-Ney Smoothing for new words

class Unigram(NamedTuple):
    """A unit of text"""
    token: str
    part_of_speech: str

    def __repr__(self) -> str:
        return f"Ugram({self.token}<{self.part_of_speech}>)"
    
    def is_punctuation(self):
        return self.part_of_speech == "."

class NGram:
    """An ordered sequence of unigrams"""

    def __init__(self, *args: Unigram) -> Self:
        self.unigrams = tuple(args)

    def __repr__(self) -> str:
        return f"Ngram[{self.unigrams}]"
    
    def __str__(self) -> str:
        return " ".join([u.token for u in self.unigrams])

    def __eq__(self, other: Self) -> bool:
        return self.unigrams == other.unigrams

    def __add__(self, other: Unigram) -> Self:
        return NGram(*self.unigrams, other)
    
    def __getitem__(self, slice: int|slice) -> Self:
        return NGram(*self.unigrams[slice])
    
    def __len__(self) -> int:
        return len(self.unigrams)

type Model = dict[NGram, dict[Unigram, int]]

class Chain:
    """class representing a Markov chain"""

    def __init__(self, prior_ngram: NGram) -> Self:
        self.prior_ngram = prior_ngram
        self.length = 0
        self.next_unigrams = {}

    def add_unigram(self, unigram: Unigram) -> None:
        self.length += 1
        count = self.next_unigrams.get(unigram, 0)
        count += 1
        self.next_unigrams[unigram] = count

    def __repr__(self) -> str:
        entries = []
        for key, value in self.next_unigrams.items():
            entries.append(f"{key}:{value}")
        all_entries = ", ".join(entries)
        return f"Chain({all_entries})"
    
class FmtNGram:
    """Class for formatting ngrams"""

    NON_SPACE = ("n't", "'")

    def __init__(self, ngram: NGram) -> Self:
        self.ngram = ngram
        self.current_pos = 0

    def peek(self, look_ahead: int = 1) -> Optional[Unigram]:
        """Peaks ahead to next unigram in sequence"""
        if self.current_pos + look_ahead > len(self.ngram) - 1:
            return None
        return self.ngram.unigrams[self.current_pos + look_ahead]
    
    def _should_not_have_space_before(self, u: Unigram) -> bool:
        """Determines if unigram should not have a space before it"""
        CONTRACTIONS = r"'\w+"
        return u.is_punctuation() or u.token in self.NON_SPACE or re.match(CONTRACTIONS, u.token)

    def fmt(self) -> str:
        """Formats an ngram as a sentence"""
        s = StringIO()
        for i, u in enumerate(self.ngram.unigrams):
            self.current_pos = i

            if u.token in ("__START__", "__END__"):
                continue

            s.write(u.token)

            # Decide whether to append a space or not
            next_unigram = self.peek()
            if next_unigram is not None and not self._should_not_have_space_before(next_unigram):
                s.write(" ")
        return s.getvalue().strip()

# TODO (Feature) preserve whitespace (non-word) characters in the tokenization step
class MarkovGenerator:

    def __init__(self, context=2, temperature=1) -> Self:
        self._startgram = Unigram("__START__", "N/A")
        self._endgram = Unigram("__END__", "N/A")
        self._context = context
        self._stats: dict[NGram, Chain] = {}

        self._temperature = temperature

        self._rng = np.random.default_rng()

    def __repr__(self) -> str:
        return f"MarkovGenerator(context={self._context}, temperature={self._temperature})"

    def _text_to_unigrams(self, text: str) -> list[Unigram]:
        """Convert text to a list of unigrams"""
        tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text), tagset="universal")
        return [Unigram(token, pos) for token, pos in tagged_tokens]

    def _train_one_sample(self, sample: str) -> None:
        """Accept one sample of training data"""
        unigrams = self._text_to_unigrams(sample)
        unigrams = [*unigrams, self._endgram]
        prev_ngram = NGram(self._startgram)
        for current_unigram in unigrams:
            logging.debug(current_unigram)

            # If new, add to table
            chain = self._stats.get(prev_ngram.unigrams)
            if not chain:
                chain = Chain(prev_ngram)
                self._stats[prev_ngram.unigrams] = chain
            chain.add_unigram(current_unigram)

            prev_ngram = prev_ngram + current_unigram
            # Keep only the last N characters
            prev_ngram = prev_ngram[-self._context:]

    def train(self, samples: list[str]) -> None:
        for sample in samples:
            self._train_one_sample(sample)

    # TODO (Feature) Allow user to configure 'Temperature' (tendency to pick less likely parameters)
    def _choose_unigram(self, c: Chain) -> Unigram:
        """Chooses a unigram based """
        values = list(c.next_unigrams.keys())
        weights = np.array(list(c.next_unigrams.values()))
        new_weights = np.exp(weights / self._temperature)
        new_weights /= np.sum(new_weights)
        logging.debug(f"{weights=} {new_weights=}")
        return random.choices(values, weights)[0]

    def _generate_one_ngram(self) -> NGram:
        """Generate one ngram example"""
        generated_gram = NGram(self._startgram)
        context_ngram = NGram(self._startgram)
        unigram = None
        while unigram != self._endgram:
            chain = self._stats[context_ngram.unigrams]
            unigram = self._choose_unigram(chain)
            generated_gram += unigram
            context_ngram = generated_gram[-self._context:]
        return generated_gram
    
    def _generate_ngrams(self, k: int = 1) -> list[NGram]:
        """Generate multiple ngrams"""
        return [self._generate_one_ngram() for _ in range(k)]
    
def new_sentence(mg: MarkovGenerator) -> str:
    """Generate one sentence from a MarkovGenerator"""
    ngram = mg._generate_one_ngram()
    return FmtNGram(ngram).fmt()

def new_sentences(mg: MarkovGenerator, k: int = 1) -> list[str]:
    """Generate multiple sentences"""
    if k < 1:
        raise ValueError(f"Expected k >= 1, got {k=}")
    return [new_sentence(mg) for _ in range(k)]

@dataclass
class SentenceMetrics:
    sentence: str
    novel: bool
    num_characters: int
    num_words: int
    lexical_diversity: float

class MarkovProfiler:
    """Profiler for Markov Generators
    Assumes the generator is trained
    """
    
    def __init__(self, samples: list[str], generator: MarkovGenerator, k: int = 50) -> Self:
        self.generator = generator
        self.samples = samples
        self.k = k

        if not self.generator._stats:
            raise ValueError(f"Expected {generator} to be trained")

    def analyze_sentence(self, sentence: str) -> str:
        metric = SentenceMetrics(
            sentence,
            sentence not in self.samples,
            len(sentence),
            len(sentence.split()),
            len(set(sentence)) / len(sentence),
        )
        return metric
    
    def profile(self) -> pd.DataFrame:
        sentences = new_sentences(self.generator, self.k)
        metrics = [self.analyze_sentence(s) for s in sentences]
        df = pd.DataFrame.from_dict(metrics)
        df["is_duplicate"] = df.duplicated()
        df["context_size"] = self.generator._context
        df["temperature"] = self.generator._temperature
        df["model"] = repr(self.generator)
        return df
    
    def summarize(self, df: pd.DataFrame) -> str:
        report = StringIO()
        model_params = repr(self.generator)
        report.write(f"{model_params}\n{'='*len(model_params)}\n")

        sections = {
            "Training Data": f"{len(self.samples)} lines",
            "Numeric Characteristics": df[["num_characters", "num_words", "lexical_diversity"]].describe(),
            "Novelty": df["novel"].value_counts(dropna=False, normalize=True),
            "Duplicate Sentences": df["sentence"].duplicated().value_counts(dropna=False, normalize=True),
            "Details": df,
        }

        for title, results in sections.items():
            report.write("\n")
            report.write(f"{title}\n{'-'*len(title)}\n")
            report.write(str(results))
            report.write("\n")

        return report.getvalue()

if __name__ == "__main__":
    pd.set_option('display.precision', 2)
    samples = [
        "We're no strangers to love 🥰",
        "You know the rules and so do I",
        "A full commitment's what I'm thinking of",
        "You wouldn't get this from any other guy",
    ]
    m = MarkovGenerator(context=2)
    m.train(samples)
    profiler = MarkovProfiler(samples, m, k=5)
    profile = profiler.profile()
    print(profiler.summarize(profile))

# TODO (Performance) Use `nltk.pos_tag_sents` instead of `nltk.pos_tag`
# TODO (Feature) Allow user to export/compile a model
# TODO (Feature) Allow users to treat newlines as important
#   list mode (just list of sentences) vs plain-text mode (don't ignore spaces)
# TODO (Ideate) Figure out if the training process can be parallized
#   like by chunking the training data and constructing independent groups of chains, then combining chains with some well-defined addition operator for chains