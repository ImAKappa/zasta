"""language

Module for language models
"""
from typing import Self, Iterable, TypeVar, Protocol
from collections import defaultdict, Counter
from abc import ABC
import random

T = TypeVar("T")

class LanguageModel:

    def __init__(self, order: int = 2) -> Self:
        self.order = order
        self.model = defaultdict(Counter)

    def load_model(self, model: dict[T, Counter]) -> None:
        self.model = model

    def train(self, data: Iterable[T], initial_history: T) -> None:
        history = initial_history
        for t in data:
            self.model[history][t] += 1
            history = (history + t)[-self.order:]

    def batch_train(self, data: list[Iterable[T]], initial_history: T) -> None:
        for sample in data:
            self.train(sample, initial_history)

    def generate(self, initial_text: T, k: int = 1, length: int = 100) -> list[Iterable[T]]:
        return [self._generate_one(initial_text, length) for _ in range(k)]

    def _generate_one(self, initial_text: T, length: int = 100) -> Iterable[T]:
        text = initial_text
        while len(text) < length:
            history = text[-self.order:]
            if history not in self.model:
                break
            text = text + self._choose(self.model[history])
        return text
    
    def _choose(self, counter: Counter) -> T:
        choice = random.choices(list(counter.keys()), weights=list(counter.values()))[0]
        return choice

if __name__ == "__main__":
    from pprint import pprint
    from zasta.utils import preprocessor as pre
    samples = [
        "We're no strangers to love 🥰",
        "You know the rules and so do I",
        "A full commitment's what I'm thinking of",
        "You wouldn't get this from any other guy",
    ]

    langchar = LanguageModel(order=3)
    langchar.train(samples[0], initial_history="")
    pprint(langchar.model)
    print()

    samples = [pre.text_to_word_tuples(s) for s in samples]
    wordchar = LanguageModel(order=3)
    wordchar.batch_train(samples, initial_history=("",))
    pprint(wordchar.model)
    print()