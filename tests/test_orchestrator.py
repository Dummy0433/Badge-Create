# tests/test_orchestrator.py
from unittest.mock import MagicMock, patch, call
import pytest
from orchestrator import Orchestrator, UnitResult, BatchResult
from eval_client import EvalResult
from generator import GenerateResult
from prompt_builder import PromptValidationError


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

MOCK_KEYWORDS = {
    "character": {"gender": "male", "hair": "dark hair", "eyes": "brown",
                  "expression": "smile", "clothing": "blue hoodie"},
    "heart_carrier": {"color_name": "Gold", "color_hex": "#D4AF37",
                      "material": "soft candy"},
    "text": {"content": "Wells", "color_tint": "#007BFF"},
    "decorations": {"elements": ["music note"]},
    "background_color": "#000000",
}
MOCK_PROMPT = 'C4D Badge, 3D Pixar realistic cartoon style. heart "Wells" test'


def _eval(passed: bool, score: float) -> EvalResult:
    return EvalResult(
        passed=passed, total_score=score,
        dimensions={"heart_carrier": score, "character": score,
                     "decorations": score, "text_render": score,
                     "color_match": score, "composition": score,
                     "quality": score},
        issues=[] if passed else ["some issue"],
    )


def _gen_result(seed: int = 1) -> GenerateResult:
    return GenerateResult(
        image=b"fake-jpeg", seed=seed, request_id=f"req-{seed}",
    )


def _make_orchestrator(generator, eval_client, llm_client=None, **kwargs):
    return Orchestrator(
        llm_client=llm_client or MagicMock(),
        generator=generator,
        eval_client=eval_client,
        **kwargs,
    )


class TestSingleUnitPassFirst:
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_pass_on_first_try(self, mock_process, mock_build, mock_pick_refs):
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.return_value = _eval(True, 8.5)

        orch = _make_orchestrator(gen, ev)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert isinstance(result, BatchResult)
        assert len(result.results) == 1
        assert result.results[0].passed is True
        assert result.results[0].score == 8.5
        assert result.results[0].rounds == 1
        gen.generate.assert_called_once()


class TestSingleUnitRetryEscalation:
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_level1_retry_new_seed(self, mock_process, mock_build, mock_pick_refs):
        """Fail first, pass on Level 1 retry (same prompt, new seed)."""
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.side_effect = [_eval(False, 6.0), _eval(True, 8.5)]

        orch = _make_orchestrator(gen, ev, max_retries=2)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert result.results[0].passed is True
        assert result.results[0].rounds == 2
        assert gen.generate.call_count == 2

    @patch("orchestrator.expand_prompt", return_value="re-expanded prompt")
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_level2_reexpand(self, mock_process, mock_build, mock_pick_refs, mock_expand):
        """Fail all Level 1 retries, pass on Level 2 re-expand."""
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        # initial + 2 retries fail, re-expand passes
        ev.evaluate.side_effect = [
            _eval(False, 5.0), _eval(False, 5.5), _eval(False, 6.0),
            _eval(True, 8.5),
        ]

        orch = _make_orchestrator(gen, ev, max_retries=2, max_reexpands=1)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert result.results[0].passed is True
        assert result.results[0].rounds == 4
        mock_expand.assert_called_once()

    @patch("orchestrator.expand_prompt", return_value="re-expanded prompt")
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_all_fail_returns_best(self, mock_process, mock_build, mock_pick_refs, mock_expand):
        """All retries and re-expands fail — return highest scoring result."""
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.side_effect = [
            _eval(False, 5.0), _eval(False, 7.0), _eval(False, 6.0),
            _eval(False, 6.5),
        ]

        orch = _make_orchestrator(gen, ev, max_retries=2, max_reexpands=1)
        result = orch.run_batch(SAMPLE_INPUT, count=1)

        assert result.results[0].passed is False
        assert result.results[0].score == 7.0


class TestBatch:
    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_batch_runs_n_units(self, mock_process, mock_build, mock_pick_refs):
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.return_value = _eval(True, 9.0)

        orch = _make_orchestrator(gen, ev)
        result = orch.run_batch(SAMPLE_INPUT, count=3)

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert len(result.results) == 3
        assert all(r.passed for r in result.results)

    @patch("orchestrator.pick_eval_references")
    @patch("orchestrator.build_prompt", return_value=(MOCK_KEYWORDS, MOCK_PROMPT))
    @patch("orchestrator.process")
    def test_batch_results_sorted_by_score(self, mock_process, mock_build, mock_pick_refs):
        mock_process.return_value = SAMPLE_INPUT
        mock_pick_refs.return_value = ([], [])

        gen = MagicMock()
        gen.generate.return_value = _gen_result()
        ev = MagicMock()
        ev.evaluate.side_effect = [_eval(True, 7.0), _eval(True, 9.0), _eval(True, 8.0)]

        orch = _make_orchestrator(gen, ev, max_retries=0, max_reexpands=0)
        result = orch.run_batch(SAMPLE_INPUT, count=3)

        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)
