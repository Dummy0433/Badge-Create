# tests/test_generator.py
import json
from unittest.mock import MagicMock, patch
import pytest
from generator import Generator, GenerateResult
from reference_store import ReferenceImage
from seedream_sdk import SeedreamResponse


def _mock_ref(desc: str = "test ref description") -> ReferenceImage:
    ref = MagicMock(spec=ReferenceImage)
    ref.description = desc
    ref.load_bytes.return_value = b"fake-ref-image"
    return ref


def _make_seedream_response() -> SeedreamResponse:
    return SeedreamResponse(
        images=[b"fake-jpeg-data"],
        llm_result="rewritten prompt",
        request_id="req-123",
    )


class TestGeneratorBasic:
    def test_generate_t2i_no_images(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        result = gen.generate(prompt="test prompt", negative_prompt="nsfw", seed=42)

        assert isinstance(result, GenerateResult)
        assert result.image == b"fake-jpeg-data"
        assert result.seed == 42
        assert result.request_id == "req-123"
        sdk.generate.assert_called_once()
        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["prompt"] == "test prompt"
        assert call_kw["seed"] == 42
        assert call_kw["images"] is None

    def test_generate_with_photo(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        result = gen.generate(
            prompt="test", negative_prompt="nsfw",
            photo_bytes=b"photo-data",
        )

        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["images"] == [b"photo-data"]


class TestGeneratorRefInjection:
    def test_refs_prepended_before_photo(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        refs = [_mock_ref("ref1 desc"), _mock_ref("ref2 desc")]
        gen.generate(
            prompt="test prompt", negative_prompt="nsfw",
            photo_bytes=b"photo-data", refs=refs,
        )

        call_kw = sdk.generate.call_args.kwargs
        images = call_kw["images"]
        assert len(images) == 3  # 2 refs + 1 photo
        assert images[0] == b"fake-ref-image"
        assert images[1] == b"fake-ref-image"
        assert images[2] == b"photo-data"

    def test_refs_build_pe_kwargs(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        refs = [_mock_ref("desc A"), _mock_ref("desc B")]
        gen.generate(
            prompt="my prompt", negative_prompt="nsfw",
            photo_bytes=b"photo", refs=refs,
        )

        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["use_pre_llm"] is False
        pe = json.loads(call_kw["pre_llm_result"])
        assert pe["input1"] == "desc A"
        assert pe["input2"] == "desc B"
        assert pe["output"] == "my prompt"
        assert pe["ratio"] == "1:1"

    def test_no_refs_no_pe(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        gen.generate(prompt="test", negative_prompt="nsfw")

        call_kw = sdk.generate.call_args.kwargs
        assert "use_pre_llm" not in call_kw
        assert "pre_llm_result" not in call_kw


class TestGeneratorKwargsPassthrough:
    def test_extra_kwargs_forwarded(self):
        sdk = MagicMock()
        sdk.generate.return_value = _make_seedream_response()
        gen = Generator(sdk)

        gen.generate(
            prompt="test", negative_prompt="nsfw",
            guidance_scale=8.0, cfg_rescale_factor=0.0,
        )

        call_kw = sdk.generate.call_args.kwargs
        assert call_kw["guidance_scale"] == 8.0
        assert call_kw["cfg_rescale_factor"] == 0.0
