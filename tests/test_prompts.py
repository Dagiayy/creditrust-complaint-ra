from __future__ import annotations

from creditrust.rag.prompts import build_prompt, postprocess_answer


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt(question="What is X?", context="Some context here.")
    assert "What is X?" in prompt
    assert "Some context here." in prompt


def test_postprocess_answer_strips_answer_marker():
    raw = "Some reasoning...\nAnswer: The actual answer text."
    assert postprocess_answer(raw) == "The actual answer text."


def test_postprocess_answer_passthrough_when_no_marker():
    raw = "Just the answer, no marker."
    assert postprocess_answer(raw) == raw
