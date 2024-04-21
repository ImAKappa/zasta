import pytest
import zasta.language as zm


class TestNGram:

    @pytest.fixture()
    def ngram(self) -> zm.NGram:
        return zm.NGram(zm.Unigram("Hi", "INTERJECT"), zm.Unigram("world", "NOUN"), zm.Unigram("Run", "VERB"))

    def test_ngram_init(self, ngram: zm.NGram):
        assert ngram.unigrams == (zm.Unigram("Hi", "INTERJECT"), zm.Unigram("world", "NOUN"), zm.Unigram("Run", "VERB"))

    def test_ngram_add_unigram(self, ngram: zm.NGram):
        new_ngram = ngram + zm.Unigram("away", "ADVERB")
        expected = (
            zm.Unigram("Hi", "INTERJECT"),
            zm.Unigram("world", "NOUN"),
            zm.Unigram("Run", "VERB"),
            zm.Unigram("away", "ADVERB")
        )
        assert new_ngram.unigrams == expected
        assert ngram != new_ngram

    def test_ngram_slice(self, ngram: zm.NGram):
        assert ngram[0].unigrams == zm.Unigram("Hi", "INTERJECT")
        assert ngram[:2].unigrams == (zm.Unigram("Hi", "INTERJECT"), zm.Unigram("world", "NOUN"))

