from collections import Counter
from gensim.models.phrases import Phrases, Phraser

def create_vocabulary(tokenized_texts, special_tokens=None, min_freq=10):
    vocab = ["UNK"]
    if special_tokens is not None:
        vocab += special_tokens
    print("creating vocabulary")
    words = []
    for t in tokenized_texts:
        words += t
    counter = Counter(words)
    for word in counter:
        if counter[word] > min_freq:
            vocab.append(word)

    reverse_vocab = {}
    _vocab = {}
    for i, word in enumerate(vocab):
        _vocab[word] = i
        reverse_vocab[i] = word
    print("finished building vocab. total words %s" % len(vocab))
    return ({"word2id":_vocab, "id2word":reverse_vocab},
            [[(_vocab[word] if word in _vocab else 0) for word in text] for text in tokenized_texts])


# Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
# Distributed Representations of Words and Phrases and their Compositionality.
# In Proceedings of NIPS, 2013.
# https://radimrehurek.com/gensim/models/phrases.html
def find_phrases(tokenized_texts, min_count=5, threshold=10.0):
    print("finding frequent bigrams")
    phrases = Phrases(tokenized_texts, min_count=min_count, threshold=threshold)
    bigram = Phraser(phrases)
    print("found. now tokenizing")
    bigram_tokenized_list = list(bigram[tokenized_texts])
    return bigram_tokenized_list

