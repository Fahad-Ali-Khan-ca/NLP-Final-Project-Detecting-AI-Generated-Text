# tests/test_data.py
import os
import pytest
import pandas as pd
import tempfile

from src.data import normalize_schema, make_splits


class TestNormalizeSchema:
    def test_standard_columns(self):
        df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
        result = normalize_schema(df)
        assert list(result.columns) == ["text", "label"]
        assert result["label"].tolist() == [0, 1]

    def test_alternate_column_names(self):
        df = pd.DataFrame({"content": ["hi", "bye"], "generated": [0, 1]})
        result = normalize_schema(df)
        assert "text" in result.columns
        assert "label" in result.columns

    def test_string_labels(self):
        df = pd.DataFrame({"text": ["a", "b", "c"], "label": ["human", "ai", "gpt"]})
        result = normalize_schema(df)
        assert result["label"].tolist() == [0, 1, 1]

    def test_missing_text_raises(self):
        df = pd.DataFrame({"foo": [1], "label": [0]})
        with pytest.raises(ValueError, match="text column"):
            normalize_schema(df)

    def test_missing_label_raises(self):
        df = pd.DataFrame({"text": ["hello"], "bar": [1]})
        with pytest.raises(ValueError, match="label column"):
            normalize_schema(df)

    def test_drops_nan(self):
        df = pd.DataFrame({"text": ["hello", None], "label": [0, 1]})
        result = normalize_schema(df)
        assert len(result) == 1


class TestMakeSplits:
    def test_creates_csvs(self):
        df = pd.DataFrame({
            "text": [f"sample {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50,
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            make_splits(df, tmpdir, test_size=0.2, valid_size=0.2, seed=42)
            assert os.path.exists(os.path.join(tmpdir, "train.csv"))
            assert os.path.exists(os.path.join(tmpdir, "valid.csv"))
            assert os.path.exists(os.path.join(tmpdir, "test.csv"))

            train = pd.read_csv(os.path.join(tmpdir, "train.csv"))
            valid = pd.read_csv(os.path.join(tmpdir, "valid.csv"))
            test = pd.read_csv(os.path.join(tmpdir, "test.csv"))
            assert len(train) + len(valid) + len(test) == 100
