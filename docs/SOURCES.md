# Sources

## Markov Chains


### [Markov Basics](https://github.com/unoti/markov-basics/blob/main/markov-basics.ipynb)

Implementation: Split between `Generator` and `Cell` classes (the cell class is the same as a 'chain')

```python
class MarkovGenerator:

    def __init__(self):
        self.end_char = ''

    def train(self, sample: str) -> None:
        """Accept one sample of training data"""
        pass

    def generate(self) -> str:
        """Generate one example"""
        pass
```

```python
class MarkovCell:

    def __init__(self, prior_letters: str):
        self.prior_letters = prior_letters
        # How many training examples have we seen?
        self.total_count = 0
        self.next_letters = {}

    def add_letter(self, letter: str) -> str:
        pass
```

### [Generative Character-Level Language Model (Peter Norvig)](https://colab.research.google.com/github/norvig/pytudes/blob/main/ipynb/Goldberg.ipynb#scrollTo=WSuIHhK_DR9W)


#### [The unreasonable effectiveness of Character-level Language Models](https://nbviewer.org/gist/yoavg/d76121dfde2618422139)


[Introducing Grammar Rules into Markov Chain](https://www2.hawaii.edu/~chin/661/Projects/AdvAI_Project_Report_Meek.pdf)


TODO: Study Norvig's work on ngrams


TODO: Study this Shakespeare example


- [Kneser-Ney Smoothing](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)

TODO: Study the short code example
[17 Line Markov Chain](https://theorangeduck.com/page/17-line-markov-chain)

[Usenet 'bot' messages generated with Markov Chains (Mark V. Shaney)](https://en.wikipedia.org/wiki/Mark_V._Shaney)

TODO: study this meta-analysis of applications of Markov Chains to nautral language processing

[Survey of markov model applications in natural language processing](https://www.researchgate.net/profile/Farrukh-Nadeem-2/publication/363000243_Markov_Models_Applications_in_Natural_Language_Processing_A_Survey/links/64c7e6c5902de670aa16fbac/Markov-Models-Applications-in-Natural-Language-Processing-A-Survey.pdf)

## Parts of Speech Tagging

[Part of Speech Tagging (Stanford)](https://web.stanford.edu/~jurafsky/slp3/old_oct19/8.pdf)

[Natural Language Processing Lecture](https://hannibunny.github.io/nlpbook/03postagging/01tagsetsAndAlgorithms.html)

## Kinds of Markov Models


### Hidden Markov Models 

[Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/A.pdf)

### Partially Observable Markov Decision Process Models

[Introduction to POMDPs](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13692)


# Alternative models for text generation

[Makemore (Andrej Karpathy)](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4)