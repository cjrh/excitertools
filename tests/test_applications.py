import excitertools
from collections import Counter
from excitertools import apply


def stem(word):
    """Stem word to primitive form

    This example taken from toolz: https://github.com/pytoolz/toolz

    There are several different ways to split a string, stem the words,
    and calculate their frequency.

    """
    return word.lower().rstrip(",.!:;'-\"").lstrip("'\"")


def test_stem():
    sentence = "This cat jumped over this other cat!"
    wc = excitertools.Iter(sentence.split()).map(stem).collect(Counter)
    assert wc == Counter({"this": 2, "cat": 2, "jumped": 1, "over": 1, "other": 1})


def test_stem2():
    sentence = "This cat jumped over this other cat!"
    wc = apply(str.split, sentence).map(stem).collect(Counter)
    assert wc == Counter({"this": 2, "cat": 2, "jumped": 1, "over": 1, "other": 1})


def test_stem3():
    sentence = "This cat jumped over this other cat!"
    wc = excitertools.splititer_regex(r"\s+", sentence).map(stem).collect(Counter)
    assert wc == Counter({"this": 2, "cat": 2, "jumped": 1, "over": 1, "other": 1})
