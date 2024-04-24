from zasta.language import LanguageModel
from zasta.profiler import LangProfiler

import math

class TestProfiler:
    pass

    # def test_perplexity(self):

    #     training = [
    #         "0 1 2 3 4",
    #         "0 1 3 4 2",
    #     ]
    #     training = [t.split() for t in training]
    #     test = "0 1 3 4 2"

    #     model = LanguageModel(context=2)
    #     model.batch_train(training)
    #     model.normalize()

    #     profiler = LangProfiler()
    #     actual = profiler.perplexity(model, test.split())
    #     print(actual)

    #     expected = 10.0

    #     assert math.isclose(actual, expected)