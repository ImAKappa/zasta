"""preprocessor

Module for preprocessing data
"""
from typing import Generator

def text_to_word_tuples(s: str) -> Generator[tuple[str], None, None]:
    """Converts a string to a generator of tuple-wrapped strings
    Assumes words are separated by spaces
    """
    for word in s.split():
        yield (word,)