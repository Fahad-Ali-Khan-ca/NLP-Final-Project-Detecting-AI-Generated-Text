# src/features.py
"""Linguistic and stylometric feature extraction for AI-text detection.

Extracts writing-style signals that complement bag-of-words and transformer
embeddings: vocabulary richness, sentence structure, readability, and
punctuation patterns.
"""

import re
import math
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sentence / token helpers
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _syllable_count(word: str) -> int:
    """Rough syllable count using vowel-group heuristic."""
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiouy]+", word))
    return max(count, 1)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def vocabulary_richness(words: List[str]) -> Dict[str, float]:
    """Type-token ratio and hapax legomena ratio."""
    n = len(words)
    if n == 0:
        return {"ttr": 0.0, "hapax_ratio": 0.0}
    types = set(words)
    hapax = sum(1 for w in types if words.count(w) == 1)
    return {
        "ttr": len(types) / n,
        "hapax_ratio": hapax / n,
    }


def sentence_stats(sentences: List[str]) -> Dict[str, float]:
    """Mean / std of sentence lengths (in words)."""
    lens = [len(s.split()) for s in sentences]
    if not lens:
        return {"sent_len_mean": 0.0, "sent_len_std": 0.0, "num_sentences": 0}
    return {
        "sent_len_mean": float(np.mean(lens)),
        "sent_len_std": float(np.std(lens)),
        "num_sentences": len(lens),
    }


def readability_scores(text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
    """Flesch Reading Ease and Flesch-Kincaid Grade Level."""
    n_words = len(words)
    n_sents = max(len(sentences), 1)
    n_syllables = sum(_syllable_count(w) for w in words) if words else 0

    if n_words == 0:
        return {"flesch_reading_ease": 0.0, "flesch_kincaid_grade": 0.0}

    asl = n_words / n_sents           # average sentence length
    asw = n_syllables / n_words       # average syllables per word

    fre = 206.835 - 1.015 * asl - 84.6 * asw
    fkg = 0.39 * asl + 11.8 * asw - 15.59

    return {
        "flesch_reading_ease": round(fre, 2),
        "flesch_kincaid_grade": round(fkg, 2),
    }


def punctuation_features(text: str) -> Dict[str, float]:
    """Normalized counts for punctuation patterns."""
    n = max(len(text), 1)
    return {
        "comma_rate": text.count(",") / n,
        "period_rate": text.count(".") / n,
        "question_rate": text.count("?") / n,
        "exclamation_rate": text.count("!") / n,
        "semicolon_rate": text.count(";") / n,
        "colon_rate": text.count(":") / n,
        "quote_rate": (text.count('"') + text.count("'")) / n,
    }


def word_length_features(words: List[str]) -> Dict[str, float]:
    """Statistics on word lengths."""
    if not words:
        return {"word_len_mean": 0.0, "word_len_std": 0.0, "long_word_ratio": 0.0}
    lens = [len(w) for w in words]
    return {
        "word_len_mean": float(np.mean(lens)),
        "word_len_std": float(np.std(lens)),
        "long_word_ratio": sum(1 for l in lens if l > 6) / len(lens),
    }


def entropy(words: List[str]) -> float:
    """Shannon entropy of the word distribution."""
    if not words:
        return 0.0
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    n = len(words)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ---------------------------------------------------------------------------
# Main extraction API
# ---------------------------------------------------------------------------

def extract_features(text: str) -> Dict[str, float]:
    """Extract all linguistic features from a single text sample."""
    words = _words(text)
    sents = _sentences(text)

    feats: Dict[str, float] = {}
    feats["char_count"] = float(len(text))
    feats["word_count"] = float(len(words))
    feats.update(vocabulary_richness(words))
    feats.update(sentence_stats(sents))
    feats.update(readability_scores(text, words, sents))
    feats.update(punctuation_features(text))
    feats.update(word_length_features(words))
    feats["word_entropy"] = entropy(words)

    return feats


def extract_features_df(texts: pd.Series) -> pd.DataFrame:
    """Extract features for a Series of texts, returns a DataFrame."""
    records = [extract_features(str(t)) for t in texts]
    return pd.DataFrame(records)
