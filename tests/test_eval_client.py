# tests/test_eval_client.py
import json
from unittest.mock import patch, MagicMock
import pytest
from eval_client import EvalClient, EvalResult
from eval_store import EvalReference


def _make_ref(is_good: bool) -> EvalReference:
    return EvalReference(
        image_path="dummy.png",
        description="test ref",
        is_good=is_good,
        score=9.0 if is_good else 3.0,
        issues=[] if is_good else ["bad quality"],
    )


MOCK_EVAL_RESPONSE = {
    "passed": True,
    "total_score": 8.5,
    "dimensions": {
        "heart_carrier": 9,
        "character": 8,
        "decorations": 8,
        "text_render": 9,
        "color_match": 8,
        "composition": 9,
        "quality": 8.5,
    },
    "issues": [],
    "suggestion": "",
}

MOCK_FAIL_RESPONSE = {
    "passed": False,
    "total_score": 5.5,
    "dimensions": {
        "heart_carrier": 7,
        "character": 6,
        "decorations": 5,
        "text_render": 3,
        "color_match": 6,
        "composition": 7,
        "quality": 4.5,
    },
    "issues": ["Text 'Wells' rendered as 'Wels'", "Artifacts on heart edge"],
    "suggestion": "Emphasize exact text spelling in quotes, add 'clean edges' to prompt",
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


class TestEvalClient:
    @patch("eval_client.openai.OpenAI")
    def test_evaluate_pass(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_EVAL_RESPONSE)))]
        )

        eval_client = EvalClient()
        # Mock load_bytes on refs to avoid file I/O
        good_ref = _make_ref(True)
        bad_ref = _make_ref(False)
        good_ref.load_bytes = lambda: b"good-image"
        bad_ref.load_bytes = lambda: b"bad-image"

        result = eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[good_ref],
            bad_refs=[bad_ref],
        )

        assert isinstance(result, EvalResult)
        assert result.passed is True
        assert result.total_score == 8.5
        assert result.dimensions["text_render"] == 9
        assert len(result.issues) == 0

    @patch("eval_client.openai.OpenAI")
    def test_evaluate_fail(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_FAIL_RESPONSE)))]
        )

        eval_client = EvalClient()
        good_ref = _make_ref(True)
        bad_ref = _make_ref(False)
        good_ref.load_bytes = lambda: b"good-image"
        bad_ref.load_bytes = lambda: b"bad-image"

        result = eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[good_ref],
            bad_refs=[bad_ref],
        )

        assert result.passed is False
        assert result.total_score == 5.5
        assert "Wels" in result.issues[0]
        assert result.suggestion != ""

    @patch("eval_client.openai.OpenAI")
    def test_system_prompt_contains_dimensions(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_EVAL_RESPONSE)))]
        )

        eval_client = EvalClient()
        good_ref = _make_ref(True)
        bad_ref = _make_ref(False)
        good_ref.load_bytes = lambda: b"good-image"
        bad_ref.load_bytes = lambda: b"bad-image"

        eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[good_ref],
            bad_refs=[bad_ref],
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        system_msg = messages[0]["content"]
        assert "heart_carrier" in system_msg
        assert "text_render" in system_msg
        assert "color_match" in system_msg

    @patch("eval_client.openai.OpenAI")
    def test_input_data_in_user_message(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(MOCK_EVAL_RESPONSE)))]
        )

        eval_client = EvalClient()
        good_ref = _make_ref(True)
        bad_ref = _make_ref(False)
        good_ref.load_bytes = lambda: b"good-image"
        bad_ref.load_bytes = lambda: b"bad-image"

        eval_client.evaluate(
            generated_image=b"fake-image",
            input_data=INPUT_DATA,
            good_refs=[good_ref],
            bad_refs=[bad_ref],
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        user_texts = []
        for msg in messages:
            if msg["role"] == "user":
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_texts.append(part["text"])
        full_text = " ".join(user_texts)
        assert "Wells" in full_text
        assert "Gold" in full_text
