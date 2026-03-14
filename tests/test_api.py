# tests/test_api.py
"""Tests for the FastAPI app schemas and endpoints (without model loading).

These tests validate request/response schemas and edge cases.
Integration tests with actual models require trained artifacts.
"""
import pytest
from unittest.mock import patch, MagicMock

from api.app import PredictRequest, PredictResponse, BatchRequest


class TestSchemas:
    def test_predict_request_valid(self):
        req = PredictRequest(text="Hello world")
        assert req.text == "Hello world"

    def test_predict_request_empty_rejected(self):
        with pytest.raises(Exception):
            PredictRequest(text="")

    def test_batch_request_valid(self):
        req = BatchRequest(texts=["hello", "world"])
        assert len(req.texts) == 2

    def test_predict_response_schema(self):
        resp = PredictResponse(
            label="Human-written",
            confidence=0.95,
            prob_human=0.95,
            prob_ai=0.05,
            features={"word_count": 10},
            latency_ms=12.5,
        )
        assert resp.label == "Human-written"
        assert resp.confidence == 0.95
