# tests/test_eval_client.py
import json
from unittest.mock import patch, MagicMock
import pytest
from eval_client import EvalClient, EvalResult
from eval_store import EvalReference


def _make_ref(is_good: bool) -> EvalReference:
    ref = EvalReference(
        image_path="dummy.png",
        description="test ref",
        is_good=is_good,
        score=9.0 if is_good else 3.0,
        issues=[] if is_good else ["bad quality"],
    )
    ref.load_bytes = lambda: b"ref-image"
    return ref


MOCK_PASS_RESPONSE = {
    "hard_gates": {
        "background_black": True,
        "heart_shape_clean": True,
    },
    "dimensions": {
        "character_likeness": 8.5,
        "heart_quality": 9.0,
        "decoration_harmony": 8.0,
        "composition": 8.5,
    },
    "issues": [],
    "suggestion": "",
}

MOCK_FAIL_SCORE_RESPONSE = {
    "hard_gates": {
        "background_black": True,
        "heart_shape_clean": True,
    },
    "dimensions": {
        "character_likeness": 5.0,
        "heart_quality": 7.0,
        "decoration_harmony": 6.0,
        "composition": 7.0,
    },
    "issues": ["Character does not resemble anchor photo"],
    "suggestion": "Increase image weight for better likeness",
}

MOCK_FAIL_GATE_RESPONSE = {
    "hard_gates": {
        "background_black": False,
        "heart_shape_clean": True,
    },
    "dimensions": {
        "character_likeness": 9.0,
        "heart_quality": 9.0,
        "decoration_harmony": 9.0,
        "composition": 9.0,
    },
    "issues": ["Background has blue gradient, not pure black"],
    "suggestion": "Add 'pure black background #000000' emphasis",
}

INPUT_DATA = {
    "text_output": "Wells",
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "photo_analysis": {"gender": "male", "hair": "dark hair"},
    "anchor_characterization": "interactive streamer",
}


def _make_eval_client(mock_response: dict):
    """Create EvalClient with mocked OpenAI returning given response."""
    with patch("eval_client.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_response)))]
        )
        ec = EvalClient()
    ec.client = mock_client
    return ec


class TestEvalPass:
    def test_all_gates_pass_and_score_above_threshold(self):
        ec = _make_eval_client(MOCK_PASS_RESPONSE)
        result = ec.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
            anchor_photo=b"anchor-photo",
        )
        assert result.passed is True
        assert result.total_score == 8.5
        assert "character_likeness" in result.dimensions
        assert "heart_quality" in result.dimensions
        assert len(result.issues) == 0


class TestEvalFailScore:
    def test_gates_pass_but_score_below_threshold(self):
        ec = _make_eval_client(MOCK_FAIL_SCORE_RESPONSE)
        result = ec.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
            anchor_photo=b"anchor-photo",
        )
        assert result.passed is False
        assert result.total_score == 6.2
        assert len(result.issues) > 0


class TestEvalFailGate:
    def test_gate_fail_overrides_high_scores(self):
        ec = _make_eval_client(MOCK_FAIL_GATE_RESPONSE)
        result = ec.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
            anchor_photo=b"anchor-photo",
        )
        assert result.passed is False
        assert result.total_score == 0.0
        assert "background" in result.issues[0].lower()


class TestEvalPromptContent:
    def test_system_prompt_has_new_dimensions(self):
        ec = _make_eval_client(MOCK_PASS_RESPONSE)
        ec.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
            anchor_photo=b"anchor-photo",
        )
        call_args = ec.client.chat.completions.create.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "character_likeness" in system_msg
        assert "heart_quality" in system_msg
        assert "decoration_harmony" in system_msg
        assert "background_black" in system_msg
        assert "heart_shape_clean" in system_msg
        assert "color_match" not in system_msg
        assert "text_render" not in system_msg

    def test_anchor_photo_in_user_message(self):
        ec = _make_eval_client(MOCK_PASS_RESPONSE)
        ec.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
            anchor_photo=b"anchor-photo",
        )
        call_args = ec.client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        texts = [p["text"] for p in user_content if p.get("type") == "text"]
        full_text = " ".join(texts)
        assert "Anchor photo" in full_text

    def test_no_anchor_photo_still_works(self):
        ec = _make_eval_client(MOCK_PASS_RESPONSE)
        result = ec.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[_make_ref(True)],
            bad_refs=[_make_ref(False)],
        )
        assert result.passed is True
