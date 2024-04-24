"""tokens

Module for natural language tokens
"""
from abc import ABC
from typing import Self, NamedTuple, Optional
import re
from io import StringIO
import nltk

class Tokenizer:
    """Tokenizes text"""

    def __init__(self) -> Self:
        pass

    def word_non_word(self, text: str) -> list[str]:
        """Tokenizes text based on alternating word and non-word characters"""
        return re.findall(r"\w+|\W+", text)

class Unigram(NamedTuple):
    """A unit of text"""
    token: str
    part_of_speech: str

    def __repr__(self) -> str:
        return f"Ugram({self.token}<{self.part_of_speech}>)"
    
    def is_punctuation(self):
        return self.part_of_speech == "."
    
def _text_to_unigrams(text: str) -> list[Unigram]:
    """Convert text to a list of unigrams"""
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text), tagset="universal")
    return [Unigram(token, pos) for token, pos in tagged_tokens]

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
