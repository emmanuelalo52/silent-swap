"""Tests for the parts that decide what participants see and what gets logged.

They use a small hand-made plan, so they run without the model or the recorded runs:
    python3 -m pytest -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from server import check_answer, served, session_for  # noqa: E402


def make_plan(n_items=4, swap_at=3):
    items = []
    for n in range(1, n_items + 1):
        flipped = n == swap_at  # the first post-swap item changes answer
        items.append({
            "n": n, "id": f"q{n}", "subject": "geo", "question": f"Q{n}?", "choices": ["a", "b", "c", "d"],
            "answer": "A",
            "before": {"pred": "A", "confidence": 0.7, "correct": True},
            "after": {"pred": "B" if flipped else "A", "confidence": 0.55, "correct": not flipped},
            "flipped": flipped,
        })
    return {"swap_at": swap_at, "subjects": ["geo"], "items": items,
            "before": {"name": "standard 16-bit build"}, "after": {"name": "compressed 8-bit build"},
            "notice": {"title": "t", "changed": "c", "unchanged": "u", "watch": "w"}}


def test_control_never_swaps():
    plan = make_plan()
    assert all(served(plan, it, "control") == ("before", False) for it in plan["items"])


@pytest.mark.parametrize("condition", ["silent", "notice"])
def test_swap_happens_at_swap_point(condition):
    plan = make_plan()
    sides = [served(plan, it, condition)[0] for it in plan["items"]]
    assert sides == ["before", "before", "after", "after"]


def test_session_hides_correct_answers_and_shows_served_answer():
    plan = make_plan()
    view = session_for(plan, "silent")
    assert all("answer" not in it for it in view["items"])
    assert view["items"][2]["ai"]["letter"] == "B"  # after-build answer on the flipped item
    assert session_for(plan, "control")["items"][2]["ai"]["letter"] == "A"


def test_notice_only_in_notice_condition():
    plan = make_plan()
    assert session_for(plan, "notice")["notice"] is not None
    assert session_for(plan, "silent")["notice"] is None
    assert session_for(plan, "control")["notice"] is None


def test_check_answer_logs_reliance_fields():
    plan = make_plan()
    rec, fb = check_answer(plan, {"pid": "P01", "condition": "silent", "n": 3, "initial": "A", "final": "B"})
    assert rec["swapped"] and rec["flipped_item"]
    assert rec["ai"] == "B" and not rec["ai_correct"]
    assert rec["followed_ai"] and rec["switched_to_ai"] and not rec["final_correct"]
    assert fb == {"answer": "A", "ai": "B", "ai_correct": False, "you_correct": False}


@pytest.mark.parametrize("body", [
    {"pid": "../etc", "condition": "silent", "n": 1, "initial": "A", "final": "A"},  # path traversal
    {"pid": "P01", "condition": "other", "n": 1, "initial": "A", "final": "A"},
    {"pid": "P01", "condition": "silent", "n": 99, "initial": "A", "final": "A"},
    {"pid": "P01", "condition": "silent", "n": 1, "initial": "E", "final": "A"},
])
def test_check_answer_rejects_bad_input(body):
    with pytest.raises(ValueError):
        check_answer(make_plan(), body)


def test_fake_quantize_puts_weights_on_the_grid():
    torch = pytest.importorskip("torch")
    from run_configs import fake_quantize_
    w = torch.randn(8, 256, dtype=torch.float32)
    q = w.clone()
    fake_quantize_(q, "int", 4, 128)
    groups = q.reshape(8, 2, 128)
    scale = w.reshape(8, 2, 128).abs().amax(-1, keepdim=True) / 7
    steps = groups / scale
    assert torch.allclose(steps, steps.round(), atol=1e-3)  # every weight is an integer step
    assert steps.abs().max() <= 8
    assert (q - w).abs().max() <= scale.max() / 2 + 1e-6  # round-to-nearest error bound
