from dataclasses import dataclass
from typing import Self
from io import StringIO
import math
import pandas as pd

from zasta.language import LanguageModel
from zasta.fmt import new_sentence


@dataclass
class SentenceMetrics:
    sentence: str
    novel: bool
    num_characters: int
    num_words: int
    lexical_diversity: float

class LangProfiler:
    """Profiler for n-gram language model"""
    
    def __init__(self) -> Self:
        pass
        
    def profile(self, model: LanguageModel,
            samples: list[str],
            train_percent: float,
            validation_percent: float,
            seed: int
        ) -> None:
        """Profiles an n-gram language model"""
        # Shuffle data
        # Split into train and test set
        # Train model on training set
        # Calculate perplexity of test set
        
    # def perplexity(self, model: LanguageModel, text: list[str]) -> Self:
    #     """The inverse probability of the test set, normalized by number of words.
    #     Used as a metric for evaluating probability models

    #     The model should be normalized.

    #     text should be tokenized.

    #     Note: to avoid floating-point issues, we use log-probabilities
    #     """
    #     if not model._is_normalized:
    #         raise ValueError("Model should be normalized")
    #     print()
    #     print(model._stats)
    #     N = len(text) + len(model._padding_end)
    #     p = 0.0
    #     text = tuple([*model._padding_start, *text, *model._padding_end])

    #     for i in range(len(text) - model._context):
    #         ngram = text[i:i + model._context]
    #         current_token = text[i + model._context]
    #         prob = model._stats[ngram][current_token]

    #         print(f"{ngram=} {current_token=} {prob=}")
    #         p += math.log(prob)
    #         ngram = (ngram + (current_token,))[-model._context:]
    #     return math.exp(-(1/N) * p)

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
