# tests/test_orchestrator.py
from unittest.mock import MagicMock, patch
import pytest
from orchestrator import Orchestrator, OrchestrationResult
from eval_client import EvalResult
from prompt_builder import PromptValidationError
from seedream_sdk import SeedreamResponse


SAMPLE_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": "interactive streamer with signature blue hoodie",
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Vibrant Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "photo_analysis": {
        "gender": "male",
        "hair": "dark black side-parted short hair",
        "eyes": "deep brown eyes",
        "expression": "gentle smile",
        "skin_tone": "medium olive",
    },
}

MOCK_KEYWORDS = {"character": {"gender": "male"}, "text": {"content": "Wells"}}
MOCK_PROMPT = 'C4D Badge, 3D Pixar realistic cartoon style. heart "Wells" test prompt'


def _make_eval_result(passed: bool, score: float, suggestion: str = "") -> EvalResult:
    return EvalResult(
        passed=passed,
        total_score=score,
        dimensions={"heart_carrier": score, "character": score, "decorations": score,
                     "text_render": score, "color_match": score, "composition": score,
                     "quality": score},
        issues=[] if passed else ["some issue"],
        suggestion=suggestion,
    )


def _make_seedream_response() -> SeedreamResponse:
    return SeedreamResponse(
        images=[b"fake-jpeg-data"],
        llm_result="",
        request_id="test-123",
    )


def _patch_prompt_steps():
    """Patch the 3 LLM prompt-building steps to return mock data."""
    return [
        patch("orchestrator.assemble_keywords", return_value=MOCK_KEYWORDS),
        patch("orchestrator.expand_prompt", return_value=MOCK_PROMPT),
        patch("orchestrator.validate_prompt"),
    ]


class TestOrchestratorPassOnFirstTry:
    @patch("orchestrator.openai.OpenAI")
    def test_returns_on_first_pass(self, mock_openai_cls):
        with patch("orchestrator.assemble_keywords", return_value=MOCK_KEYWORDS), \
             patch("orchestrator.expand_prompt", return_value=MOCK_PROMPT), \
             patch("orchestrator.validate_prompt"):

            seedream = MagicMock()
            seedream.generate.return_value = _make_seedream_response()

            eval_client = MagicMock()
            eval_client.evaluate.return_value = _make_eval_result(True, 8.5)

            orch = Orchestrator(seedream_client=seedream, eval_client=eval_client)
            result = orch.run(SAMPLE_INPUT)

            assert isinstance(result, OrchestrationResult)
            assert result.passed is True
            assert result.rounds == 1
            assert len(result.eval_history) == 1
            seedream.generate.assert_called_once()


class TestOrchestratorRetry:
    @patch("orchestrator.openai.OpenAI")
    def test_retries_on_failure_then_passes(self, mock_openai_cls):
        mock_adj_client = MagicMock()
        mock_openai_cls.return_value = mock_adj_client
        mock_adj_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="adjusted prompt text"))]
        )

        with patch("orchestrator.assemble_keywords", return_value=MOCK_KEYWORDS), \
             patch("orchestrator.expand_prompt", return_value=MOCK_PROMPT), \
             patch("orchestrator.validate_prompt"):

            seedream = MagicMock()
            seedream.generate.return_value = _make_seedream_response()

            eval_client = MagicMock()
            eval_client.evaluate.side_effect = [
                _make_eval_result(False, 6.0, "fix text rendering"),
                _make_eval_result(True, 8.5),
            ]

            orch = Orchestrator(seedream_client=seedream, eval_client=eval_client)
            result = orch.run(SAMPLE_INPUT)

            assert result.passed is True
            assert result.rounds == 2
            assert len(result.prompt_history) == 2

    @patch("orchestrator.openai.OpenAI")
    def test_rerolls_after_max_retries(self, mock_openai_cls):
        mock_adj_client = MagicMock()
        mock_openai_cls.return_value = mock_adj_client
        mock_adj_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="rerolled prompt text"))]
        )

        with patch("orchestrator.assemble_keywords", return_value=MOCK_KEYWORDS), \
             patch("orchestrator.expand_prompt", return_value=MOCK_PROMPT), \
             patch("orchestrator.validate_prompt"):

            seedream = MagicMock()
            seedream.generate.return_value = _make_seedream_response()

            eval_client = MagicMock()
            eval_client.evaluate.side_effect = [
                _make_eval_result(False, 5.0, "fix A"),
                _make_eval_result(False, 5.5, "fix B"),
                _make_eval_result(False, 6.0, "fix C"),
                _make_eval_result(True, 8.5),
            ]

            orch = Orchestrator(
                seedream_client=seedream, eval_client=eval_client,
                max_retries=2, max_rerolls=1,
            )
            result = orch.run(SAMPLE_INPUT)

            assert result.passed is True
            assert result.rounds == 4


class TestOrchestratorFallback:
    @patch("orchestrator.openai.OpenAI")
    def test_returns_best_on_all_failures(self, mock_openai_cls):
        mock_adj_client = MagicMock()
        mock_openai_cls.return_value = mock_adj_client
        mock_adj_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="adjusted prompt"))]
        )

        with patch("orchestrator.assemble_keywords", return_value=MOCK_KEYWORDS), \
             patch("orchestrator.expand_prompt", return_value=MOCK_PROMPT), \
             patch("orchestrator.validate_prompt"):

            seedream = MagicMock()
            seedream.generate.return_value = _make_seedream_response()

            eval_client = MagicMock()
            eval_client.evaluate.side_effect = [
                _make_eval_result(False, 5.0, "fix A"),
                _make_eval_result(False, 7.0, "fix B"),
                _make_eval_result(False, 6.0, "fix C"),
                _make_eval_result(False, 6.5, "fix D"),
            ]

            orch = Orchestrator(
                seedream_client=seedream, eval_client=eval_client,
                max_retries=2, max_rerolls=1,
            )
            result = orch.run(SAMPLE_INPUT)

            assert result.passed is False
            assert result.score == 7.0


class TestOrchestratorValidation:
    @patch("orchestrator.openai.OpenAI")
    def test_invalid_input_raises(self, mock_openai_cls):
        with patch("orchestrator.assemble_keywords", return_value=MOCK_KEYWORDS), \
             patch("orchestrator.expand_prompt", return_value=MOCK_PROMPT), \
             patch("orchestrator.validate_prompt", side_effect=PromptValidationError("text_output is empty")):

            orch = Orchestrator(
                seedream_client=MagicMock(), eval_client=MagicMock()
            )
            with pytest.raises(PromptValidationError):
                orch.run({"text_output": ""})
