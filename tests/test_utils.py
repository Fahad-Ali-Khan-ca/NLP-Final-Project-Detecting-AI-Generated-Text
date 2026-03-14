# tests/test_utils.py
import os
import json
import tempfile

import pytest
import numpy as np

from src.utils import (
    set_seed,
    ensure_dir,
    compute_metrics_from_preds,
    save_json,
)


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


class TestEnsureDir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "nested", "dir")
            ensure_dir(new_dir)
            assert os.path.isdir(new_dir)

    def test_existing_dir_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_dir(tmpdir)  # should not raise


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        m = compute_metrics_from_preds(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = compute_metrics_from_preds(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_partial(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        m = compute_metrics_from_preds(y_true, y_pred)
        assert 0 < m["accuracy"] < 1.0
        assert "precision" in m
        assert "recall" in m


class TestSaveJson:
    def test_saves_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            data = {"accuracy": 0.95, "f1": 0.93}
            save_json(data, path)

            with open(path) as f:
                loaded = json.load(f)
            assert loaded == data
