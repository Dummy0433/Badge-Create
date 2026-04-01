# tests/test_server.py
import json
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from server import app
from seedream_sdk import SeedreamResponse


@pytest.fixture
def mock_sdk_response():
    return SeedreamResponse(
        images=[b"\xff\xd8\xff\xe0fake-jpeg-bytes"],
        llm_result="A cute orange cat sitting on a windowsill",
        image_prompts=[],
        request_id="test-req-001",
        raw_response={"status_code": 0},
    )


@pytest.mark.asyncio
async def test_generate_t2i_success(mock_sdk_response):
    with patch("server.client") as mock_client:
        mock_client.generate.return_value = mock_sdk_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/generate",
                data={"prompt": "a cute cat", "width": "1024", "height": "1024"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["request_id"] == "test-req-001"
        assert body["llm_result"] == "A cute orange cat sitting on a windowsill"
        assert len(body["images"]) == 1
        assert body["images"][0].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_generate_with_files(mock_sdk_response):
    with patch("server.client") as mock_client:
        mock_client.generate.return_value = mock_sdk_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/generate",
                data={"prompt": "edit this image"},
                files=[("files", ("test.jpg", b"\xff\xd8fake", "image/jpeg"))],
            )

        assert resp.status_code == 200
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs["images"] is not None
        assert len(call_kwargs["images"]) == 1


@pytest.mark.asyncio
async def test_generate_missing_prompt():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/generate", data={})

    assert resp.status_code == 422
