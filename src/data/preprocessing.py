"""
PoseStitch-style text normalization for sign language translation.
"""

import re
from pathlib import Path

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "it's": "it is", "that's": "that is",
    "he's": "he is", "she's": "she is", "i'm": "i am",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "let's": "let us", "ain't": "am not",
    "mightn't": "might not", "mustn't": "must not", "needn't": "need not",
    "shan't": "shall not", "there's": "there is", "here's": "here is",
    "who's": "who is", "what's": "what is", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
}
NUMBER_RE = re.compile(r'^\d+$')
STOPWORDS = set("a an the and or but in on at to for of with by from up about into through during before after above below between out off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t will just don should now is was are were been be have has had do does did can could would may might must shall should will would".split())


def posestitch_preprocess(text, low_freq_words=None):
    text = text.strip().lower()
    words = text.split()
    words = [CONTRACTIONS.get(w, w) for w in words]
    text = ' '.join(words)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    expanded = []
    for w in words:
        if NUMBER_RE.match(w):
            expanded.extend(list(w))
        else:
            expanded.append(w)
    words = expanded
    if low_freq_words is not None:
        words = [w if w in low_freq_words else "unknown" for w in words]
    return ' '.join(words)
