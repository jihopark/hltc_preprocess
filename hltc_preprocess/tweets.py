"""
Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import sys
import re
import string

from nltk import TweetTokenizer

FLAGS = re.MULTILINE | re.DOTALL


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    return hashtag_body

def num_words(sent):
    return len(re.findall(r'\w+',sent))

def filter_tweets(texts, remove_url=True, remove_quotation=True, min_words=5):
    assert isinstance(texts, list)

    texts = list(filter(lambda x:
                            (not remove_url or not ("http://" in x or
                                "https://" in x or "<url>" in x))
                            and
                            (not remove_quotation or "\"" not in x)
                            and
                            (num_words(x) >= min_words),
                            texts))
    return texts

def tokenize_tweets(texts, bigram=False):
    tknzr = TweetTokenizer()
    return [tknzr.tokenize(t) for t in texts]

def clean_tweet(text,
                remove_nonalphanumeric=False,
                use_number_special_token=True,
                use_user_special_token=True,
                use_emoticon_special_token=True,
                make_hashtag_as_word=True,
                deal_repetition=True,
                preserve_case=False,
                remove_hashtag_at_end=False):
   # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    if remove_hashtag_at_end:
        text = re.sub(r"(^.*?)(#[\S]+\s+)*#[\S]+$", r"\1", text.strip())

    if remove_nonalphanumeric:
        text = re_sub(r'([^\s\w\@]|_)+', "")

    # removes url
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    if use_user_special_token:
        text = re_sub(r"@\w+", "<user>")
    if use_emoticon_special_token:
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes,
                                                      nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes,
                                                nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"<3", "<heart>")
    if use_number_special_token:
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    if make_hashtag_as_word:
        text = re_sub(r"#\S+", hashtag)
    if deal_repetition:
        text = re_sub(r"([!?.]){2,}", r"\1")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2")
    if preserve_case:
        return text
    return text.lower()
