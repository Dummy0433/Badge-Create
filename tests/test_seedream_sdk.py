# tests/test_seedream_sdk.py
import base64
import json
from unittest.mock import patch, MagicMock
import pytest
import requests as requests_lib
from seedream_sdk import SeedreamClient, SeedreamResponse, SeedreamAPIError


def _mock_success_response(image_bytes=b"fake-jpeg-data", llm_result="", request_id="test-123"):
    """Create a mock requests.Response for a successful Seedream API call."""
    resp = MagicMock()
    resp.status_code = 200
    resp_body = {
        "status_code": 0,
        "data": {
            "afr_data": [],
            "resp_json": json.dumps({
                "llm_result": llm_result,
                "image_prompt": [],
                "request_id": request_id,
            }),
            "binary_data": [],
        },
        "extra": {"log_id": "log-abc"},
    }
    resp.json.return_value = resp_body
    resp.content = image_bytes
    resp.headers = {"Content-Type": "application/json"}
    return resp


def _mock_error_response(algo_code=100199, algo_msg="algoError"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status_code": 1,
        "algo_status_code": algo_code,
        "algo_status_message": algo_msg,
        "data": {"afr_data": []},
        "message": "Couldn't process image.",
    }
    return resp


class TestSeedreamClientBuildRequest:
    """Test that the client builds correct HTTP requests."""

    @patch("seedream_sdk.requests.post")
    def test_t2i_request_format(self, mock_post):
        mock_post.return_value = _mock_success_response()
        client = SeedreamClient()
        client.generate(prompt="a cute cat", width=1024, height=1024)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args

        # Check endpoint
        assert call_kwargs.args[0] == "https://api2.musical.ly/media/api/pic/afr"

        # Check form data contains algorithms and conf
        data = call_kwargs.kwargs.get("data", {})
        assert data["algorithms"] == "tt_vlm_high_aes_scheduler"

        conf = json.loads(data["conf"])
        assert conf["prompt"] == "a cute cat"
        assert conf["width"] == 1024
        assert conf["height"] == 1024
        assert conf["model_version"] == "general_v4.5"
        assert conf["pre_vlm_version"] == "tt_seed_x2i_40l_pe_20b_T2_18"
        assert conf["force_single"] is True

    @patch("seedream_sdk.requests.post")
    def test_i2i_request_includes_files(self, mock_post):
        mock_post.return_value = _mock_success_response()
        client = SeedreamClient()
        fake_image = b"\xff\xd8\xff\xe0fake-jpeg"
        client.generate(prompt="edit this", images=[fake_image])

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", {})
        assert data["input_img_type"] == "multiple_files"

        files = call_kwargs.kwargs.get("files", [])
        assert len(files) == 1
        assert files[0][0] == "files[]"

    @patch("seedream_sdk.requests.post")
    def test_kwargs_passthrough(self, mock_post):
        mock_post.return_value = _mock_success_response()
        client = SeedreamClient()
        client.generate(prompt="test", guidance_scale=5.0, num_inference_timesteps=20)

        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data", {})
        conf = json.loads(data["conf"])
        assert conf["guidance_scale"] == 5.0
        assert conf["num_inference_timesteps"] == 20


class TestSeedreamClientErrors:

    @patch("seedream_sdk.requests.post")
    def test_algo_error_raises(self, mock_post):
        mock_post.return_value = _mock_error_response(100199, "JSON parse error")
        client = SeedreamClient()

        with pytest.raises(SeedreamAPIError) as exc_info:
            client.generate(prompt="bad request")
        assert "100199" in str(exc_info.value)

    @patch("seedream_sdk.requests.post")
    def test_http_error_raises(self, mock_post):
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests_lib.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = resp
        client = SeedreamClient()

        with pytest.raises(requests_lib.exceptions.HTTPError):
            client.generate(prompt="test")


class TestSeedreamClientResponseParsing:

    @patch("seedream_sdk.requests.post")
    def test_parse_response_success(self, mock_post):
        """Test that response parsing correctly extracts images, llm_result, and request_id."""
        fake_image = b"\xff\xd8\xff\xe0fake-jpeg"
        encoded_image = base64.b64encode(fake_image).decode()

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "status_code": 0,
            "data": {
                "afr_data": [encoded_image],
                "resp_json": json.dumps({
                    "llm_result": "A cute orange cat",
                    "image_prompt": ["detailed cat prompt"],
                    "request_id": "req-456",
                }),
                "binary_data": [],
            },
            "extra": {"log_id": "log-abc"},
        }
        mock_post.return_value = resp

        client = SeedreamClient()
        result = client.generate(prompt="a cute cat")

        assert len(result.images) == 1
        assert result.images[0] == fake_image
        assert result.llm_result == "A cute orange cat"
        assert result.image_prompts == ["detailed cat prompt"]
        assert result.request_id == "req-456"
