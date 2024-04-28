# TODO

## Stage 1: Prototype

### Phase A: Design

To design Zasta well, I need to learn more.
Fortunately, I've found a pretty good book on NLP:

[Zasta](https://web.stanford.edu/~jurafsky/slp3/ed3book_jan72023.pdf)

At minimum, I should learn the following skills:

1. Chapter 1
   1. How to compose regular expressions
2. Chapter 2
   1. How to normalize text
      1. How to tokenize text
      2. How to case-fold text
      3. How to segment sentences
   2. How to compute the edit distance of strings
3. Chapter 3
   1. n-grams
   2. Probability theory
      1. Chain rule
      2. Conditional probability
      3. Markov assumption
      4. Log probability
   3. Evaluating language models
      1. Intrinsic evaluation
      2. Minimizing perplexity
      3. Sampling
   4. Vocabulary
      1. Dealing with unknown/out-of-vocabulary words
      2. Smoothing algorithms
   5. Implementation
      1. Use hashing functions to save space
      2. Use reverse tries
      3. Use smaller-bit numbers
      4. Prune low-frequency ngrams

### Phase B: Implementation

- [ ] Tokenizer: Create a tokenizer using `nltk.word_tokenizer`