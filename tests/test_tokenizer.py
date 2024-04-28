import pytest

from zasta.tokenizer import Tokenizer

class TestTokenizer:

    @pytest.fixture()
    def s(self) -> str:
        return "Hello, World!"

    def test_word_non_word(self, s: str):
        t = Tokenizer()
        actual = t.word_non_word(s)
        expected = ["Hello", ", ", "World", "!"]
        assert actual == expected
