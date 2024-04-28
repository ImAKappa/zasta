"""tokens

Module for natural language tokens
"""
from abc import ABC
from typing import Self, NamedTuple, Optional
import re
from io import StringIO
import nltk

class TaggedText(NamedTuple):
    """A token of text tagged with part-of-speech"""
    token: str
    part_of_speech: str

    def __repr__(self) -> str:
        return f"TaggedText({self.token}<{self.part_of_speech}>)"

class Tokenizer:
    """Tokenizes text"""

    def __init__(self) -> Self:
        pass

    def word_non_word(self, text: str) -> list[str]:
        """Tokenizes text based on alternating word and non-word characters"""
        return re.findall(r"\w+|\W+", text)
    
    def tagged_word_non_word(self, text: str) -> list[TaggedText]:
        """Convert text to a list of tagged text"""
        tagged_tokens = nltk.pos_tag(self.word_non_word(text), tagset="universal")
        return [TaggedText(token, pos) for token, pos in tagged_tokens]
