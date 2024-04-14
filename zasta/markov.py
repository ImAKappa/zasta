"""markov.py

Module for generating markov chains
"""
from collections import defaultdict
from typing import Self, Optional, NamedTuple
import random
from dataclasses import dataclass
import re
from pprint import pprint
import logging
logging.basicConfig(format="%(levelname)s:%(funcName)s():%(lineno)d\t%(message)s", level=logging.DEBUG)

import nltk

class Unigram(NamedTuple):
    """A unit of text"""
    token: str
    part_of_speech: str

    def __repr__(self) -> str:
        return f"Ugram({self.token}<{self.part_of_speech}>)"

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
    
class MarkovGenerator:

    def __init__(self, lookback_size=2) -> Self:
        self._startgram = Unigram("__START__", "N/A")
        self._endgram = Unigram("__END__", "N/A")
        self._lookback_size = lookback_size
        self._stats: dict[NGram, Chain] = {}

    def _text_to_unigrams(self, text: str) -> list[Unigram]:
        """Convert text to a list of unigrams"""
        tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text), tagset="universal")
        return [Unigram(token, pos) for token, pos in tagged_tokens]

    def train(self, sample: str) -> None:
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
            prev_ngram = prev_ngram[-self._lookback_size:]

    def _choose_unigram(self, c: Chain) -> Unigram:
        """Chooses a unigram based """
        values = list(c.next_unigrams.keys())
        weights = list(c.next_unigrams.values())
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
            context_ngram = generated_gram[-self._lookback_size:]
        return generated_gram
    
    def _generate_ngrams(self, k: int = 1) -> list[NGram]:
        """Generate multiple ngrams"""
        return [self._generate_one_ngram() for _ in range(k)]
    
    def new_sentence(self) -> str:
        """Generate one sentence"""
        return str(self._generate_one_ngram()[1:-1])

    def new_sentences(self, k: int = 1) -> list[str]:
        """Generate multiple sentences"""
        if k < 1:
            raise ValueError(f"Expected k >= 1, got {k=}")
        return [self.new_sentence() for _ in range(k)]

if __name__ == "__main__":
    m = MarkovGenerator(lookback_size=2)
    m.train("you is dark you is black you is crack and you is faggot")
    print(m.new_sentences(k=3))

# TODO (Feature) Parameterize context. Currently it is one token long, but allow user to set 2 to 4 token context-size
# TODO (Performance) Use `nltk.pos_tag_sents` instead of `nltk.pos_tag`
# TODO (Feature) Allow user to export/compile a model
# TODO (Feature) Allow users to treat newlines as important
#   list mode (just list of sentences) vs plain-text mode (don't ignore spaces)
# TODO (Ideate) Figure out if the training process can be parallized
#   like by chunking the training data and constructing independent groups of chains, then combining chains with some well-defined addition operator for chains
class Markov:

    def __init__(self, text: str, model: Optional[Model] = None, context = 1) -> Self:
        self.text = text
        self.model = model if model else self.build()
        self.context = context

    def _tidy(self, text: str) -> str:
        """Tidies up a sentence"""
        punctuation = list(".,?!;'")
        for p in punctuation:
            text = text.replace(f" {p}", p)

        # FIXME (Bug) Handle case with "n't" which gets chunked into a separate token by `nltk`
        #   For example, the output usually has "does n't" instead of the expected "doesn't"
        return text
    
    def _tokenize(self) -> list[tuple[str, str]]:
        """tokenizes the text (splits into tokens and tags with parts of speech)"""
        raise NotImplementedError()
    
    def _chain(self) -> Model:
        """constructs a Markov chain based on input data and context"""
        raise NotImplementedError()

    # TODO (Refactor) Split `self.build` into several smaller functions
    #   self._tokenize, self._chain
    def build(self) -> Model:
        """Builds a markov chain of words"""
        start = defaultdict(int)
        end = defaultdict(int)
        model: Model = defaultdict(int)

        for line in self.text.splitlines():
            if not line:
                continue
            text = nltk.pos_tag(nltk.word_tokenize(line), tagset="universal")
            if not text:
                continue

            for i, word in enumerate(text):
                if i == len(text) - 1:
                    end[text[-1]] += 1
                else:
                    if i == 0:
                        start[text[0]] += 1
                    if word in model:
                        model[word][text[i+1]] += 1
                    else:
                        model[word] = defaultdict(int)
                        model[word][text[i+1]] += 1
        model["__START__"] = start
        model["__END__"] = end
        return model
    
    def _assemble_sequence(self, seq: list[str]) -> str:
        """Assembles sentence based on a generated sequence"""
        raise NotImplementedError
    
    def _new_sequence(self) -> list[str]:
        """Generates a new sequence of tokens based on a Markov chain.
        Note that whitespace is explicitly represented as a token to avoid incorrect spacing issues
        E.g. 'does n't' vs 'doesn't'
        """

    # TODO (Refactor) split into several functions
    #   self._assemble_sequence, self._new_sequence
    def new_sentence(self, min_words: int = 10) -> str:
        """Generates a sample Gen Z sentence"""
        generate: list[tuple[str, str]] = []

        while True:
            words_dict = None
            if not generate:
                words_dict = self.model["__START__"]
                logging.debug(f"START: {words_dict}")
            # elif generate[-1] in self.model["END"] and len(generate) > min_words:
            elif generate[-1] in self.model["__END__"]:
                logging.debug(f"END")
                if len(generate) < min_words:
                    words_dict = self.model["__START__"]
                    logging.debug(f"Sentence too short, drawing from start: {words_dict=}")
                else:
                    break
            else:
                words_dict = self.model[generate[-1]]
                logging.debug(f"ELSE: last word='{generate[-1]}' {words_dict=}")
            next_word = random.choices(
                list(words_dict.keys()),
                weights=list(words_dict.values())
            )[0]
            generate.append(next_word)
        sentence = [tagged_word[0] for tagged_word in generate]
        logging.debug(f"{sentence=}")
        return self._tidy(" ".join(sentence))