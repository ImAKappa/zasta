from zasta.language import LanguageModel
from zasta.tokenizer import FmtNGram

def new_sentence(mg: LanguageModel) -> str:
    """Generate one sentence from a MarkovGenerator"""
    ngram = mg._generate_sample()
    return FmtNGram(ngram).fmt()

def new_sentences(mg: LanguageModel, k: int = 1) -> list[str]:
    """Generate multiple sentences"""
    if k < 1:
        raise ValueError(f"Expected k >= 1, got {k=}")
    return [new_sentence(mg) for _ in range(k)]