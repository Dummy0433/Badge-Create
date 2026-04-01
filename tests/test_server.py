# tests/test_server.py
import json
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from server import app
from orchestrator import BatchResult, UnitResult
from eval_client import EvalResult


def _make_unit_result(score: float = 8.5) -> UnitResult:
    return UnitResult(
        image=b"\xff\xd8\xff\xe0fake-jpeg",
        score=score,
        passed=score >= 8.0,
        seed=12345,
        prompt="test prompt",
        eval_result=EvalResult(
            passed=score >= 8.0, total_score=score,
            dimensions={"heart_carrier": score}, issues=[],
        ),
        rounds=1,
        request_id="req-001",
    )


@pytest.mark.asyncio
async def test_pipeline_delegates_to_orchestrator():
    mock_batch_result = BatchResult(
        results=[_make_unit_result(9.0), _make_unit_result(7.5)],
        prompt="generated prompt",
        keywords={},
        total=2,
        success=1,
        failed=0,
    )

    with patch("server.orchestrator") as mock_orch:
        mock_orch.run_batch.return_value = mock_batch_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/pipeline",
                data={
                    "input_json": json.dumps({"slogan": "Test", "anchor_info": {
                        "anchor": {}, "anchor_characterization": "test",
                        "brand_palette": {"primary": {"name": "Red", "hex": "#F00"},
                                          "secondary": {"name": "Blue", "hex": "#00F"},
                                          "tertiary": {"name": "White", "hex": "#FFF"}},
                    }}),
                    "count": "2",
                },
                files=[("anchor_photo", ("test.jpg", b"\xff\xd8fake", "image/jpeg"))],
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["success"] == 1
        assert len(body["results"]) == 2
        assert body["results"][0]["eval"]["score"] == 9.0
        assert body["prompt"] == "generated prompt"
        mock_orch.run_batch.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_invalid_json():
    with patch("server.orchestrator"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/pipeline",
                data={"input_json": "not-json", "count": "1"},
            )
        assert resp.status_code == 400
