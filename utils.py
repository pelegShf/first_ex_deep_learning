# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
import os
import re
import string
import loglinear as ll
from train_loglin import feats_to_vec

def clean_data(text):
    #remove capitalization
    text = text.lower()
    #remove links
    text = re.sub(r'http\S+', '', text)
    #remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    #Removing numbers
    text = re.sub(r'\w*\d\w*', '', text)
    #remove unicode characters
    return text

def read_data(fname):
    data = []
    with open(fname, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        label, text = line.strip().lower().split("\t", 1)
        text = clean_data(text)
        data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]
TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]



from collections import Counter

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(1000)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
print(L2I)