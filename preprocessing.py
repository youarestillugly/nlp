import html
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

IMPORTANT_WORDS = {
    "no", "not", "never", "nor", "but", "however", "although",
    "very", "too", "more", "most", "less", "least"
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - IMPORTANT_WORDS

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "didn't": "did not", "doesn't": "does not", "isn't": "is not",
    "wasn't": "was not", "weren't": "were not", "aren't": "are not",
    "don't": "do not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "mustn't": "must not", "mightn't": "might not",
    "shan't": "shall not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will"
}

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()

    for contraction, replacement in CONTRACTIONS.items():
        text = text.replace(contraction, replacement)

    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)

def clean_series(texts):
    return np.array([clean_text(text) for text in texts])
