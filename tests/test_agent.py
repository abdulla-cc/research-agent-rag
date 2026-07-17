"""Unit tests for the agentic pipeline's parsing/fallback logic.
Uses mocking to test without real LLM calls (no Gemini quota needed)."""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent


def test_decompose_parses_json_array():
    with patch("agent._call_llm", return_value='["q one", "q two"]'):
        result = agent.decompose_question("complex question")
    assert result == ["q one", "q two"]


def test_decompose_falls_back_on_garbage():
    with patch("agent._call_llm", return_value="not json at all"):
        result = agent.decompose_question("original question")
    assert result == ["original question"]


def test_decompose_strips_prose_around_json():
    # LLMs often wrap JSON in explanation — we should still extract it
    with patch("agent._call_llm", return_value='Here you go: ["a", "b"] hope that helps'):
        result = agent.decompose_question("q")
    assert result == ["a", "b"]


def test_decompose_ignores_non_string_items():
    with patch("agent._call_llm", return_value='["valid", 123, null, "also valid"]'):
        result = agent.decompose_question("q")
    assert result == ["valid", "also valid"]


def test_critique_parses_json_object():
    fake = '{"supported": true, "notes": "All claims supported."}'
    with patch("agent._call_llm", return_value=fake):
        result = agent.critique_answer("some answer", [{"text": "src", "title": "T", "distance": 0.1}])
    assert result["supported"] is True
    assert "supported" in result["notes"].lower()


def test_critique_handles_unparseable():
    with patch("agent._call_llm", return_value="totally not json"):
        result = agent.critique_answer("answer", [{"text": "s", "title": "T", "distance": 0.1}])
    assert result["supported"] is None
