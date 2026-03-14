# tests/test_features.py
import pytest
import pandas as pd

from src.features import (
    extract_features,
    extract_features_df,
    vocabulary_richness,
    sentence_stats,
    readability_scores,
    punctuation_features,
    word_length_features,
    entropy,
    _words,
    _sentences,
)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

class TestTokenHelpers:
    def test_words_basic(self):
        assert _words("Hello World") == ["hello", "world"]

    def test_words_strips_punctuation(self):
        assert _words("Hello, world!") == ["hello", "world"]

    def test_words_empty(self):
        assert _words("") == []

    def test_sentences_basic(self):
        sents = _sentences("First. Second! Third?")
        assert len(sents) == 3

    def test_sentences_no_delimiter(self):
        sents = _sentences("No period at end")
        assert len(sents) == 1


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

class TestVocabularyRichness:
    def test_all_unique(self):
        result = vocabulary_richness(["a", "b", "c", "d"])
        assert result["ttr"] == 1.0
        assert result["hapax_ratio"] == 1.0

    def test_all_same(self):
        result = vocabulary_richness(["a", "a", "a"])
        assert result["ttr"] == pytest.approx(1 / 3, rel=1e-3)
        assert result["hapax_ratio"] == 0.0

    def test_empty(self):
        result = vocabulary_richness([])
        assert result["ttr"] == 0.0


class TestSentenceStats:
    def test_basic(self):
        result = sentence_stats(["one two three", "four five"])
        assert result["num_sentences"] == 2
        assert result["sent_len_mean"] == 2.5

    def test_empty(self):
        result = sentence_stats([])
        assert result["num_sentences"] == 0


class TestReadability:
    def test_returns_floats(self):
        text = "This is a simple test sentence. It has two sentences."
        words = _words(text)
        sents = _sentences(text)
        result = readability_scores(text, words, sents)
        assert "flesch_reading_ease" in result
        assert isinstance(result["flesch_reading_ease"], float)

    def test_empty(self):
        result = readability_scores("", [], [])
        assert result["flesch_reading_ease"] == 0.0


class TestPunctuationFeatures:
    def test_counts(self):
        result = punctuation_features("Hello, world! How are you?")
        assert result["comma_rate"] > 0
        assert result["exclamation_rate"] > 0
        assert result["question_rate"] > 0


class TestWordLengthFeatures:
    def test_basic(self):
        result = word_length_features(["hi", "there", "everyone"])
        assert result["word_len_mean"] > 0

    def test_empty(self):
        result = word_length_features([])
        assert result["word_len_mean"] == 0.0


class TestEntropy:
    def test_uniform(self):
        # 4 unique words → entropy = 2.0 bits
        e = entropy(["a", "b", "c", "d"])
        assert e == pytest.approx(2.0, rel=1e-3)

    def test_single_word(self):
        e = entropy(["a", "a", "a"])
        assert e == 0.0

    def test_empty(self):
        assert entropy([]) == 0.0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_dict(self):
        feats = extract_features("Hello world. This is a test.")
        assert isinstance(feats, dict)
        assert "word_count" in feats
        assert "ttr" in feats
        assert "flesch_reading_ease" in feats
        assert "word_entropy" in feats

    def test_word_count(self):
        feats = extract_features("one two three four five")
        assert feats["word_count"] == 5

    def test_empty_string(self):
        feats = extract_features("")
        assert feats["word_count"] == 0


class TestExtractFeaturesDf:
    def test_returns_dataframe(self):
        texts = pd.Series(["Hello world.", "Another test sentence here."])
        df = extract_features_df(texts)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "ttr" in df.columns
