# tests/test_prompt_builder.py
import pytest
from prompt_builder import build_prompt, build_negative_prompt, PromptValidationError


SAMPLE_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": (
        "This Jordanian anchor hosts an energetic and highly interactive "
        "'Just Chatting' stream. He engages directly with his audience, "
        "responding to song requests, comments, and playful banter. "
        "Visually, he presents a casual style, frequently seen in a "
        "signature blue hoodie."
    ),
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


class TestBuildPrompt:
    def test_contains_fixed_style(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "C4D Badge" in prompt
        assert "3D Pixar realistic cartoon style" in prompt
        assert "candy color palette" in prompt

    def test_contains_text_output(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert '"Wells"' in prompt

    def test_contains_brand_colors(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "Gold" in prompt or "#D4AF37" in prompt
        assert "#000000" in prompt or "Black" in prompt

    def test_contains_character_traits(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "dark black side-parted short hair" in prompt
        assert "deep brown eyes" in prompt
        assert "gentle smile" in prompt

    def test_contains_lighting(self):
        prompt = build_prompt(SAMPLE_INPUT)
        assert "Warm side light from left" in prompt

    def test_missing_text_output_raises(self):
        bad_input = {**SAMPLE_INPUT, "text_output": ""}
        with pytest.raises(PromptValidationError, match="text_output"):
            build_prompt(bad_input)

    def test_missing_photo_analysis_raises(self):
        bad_input = {**SAMPLE_INPUT}
        del bad_input["photo_analysis"]
        with pytest.raises(PromptValidationError, match="photo_analysis"):
            build_prompt(bad_input)

    def test_missing_brand_palette_raises(self):
        bad_input = {**SAMPLE_INPUT}
        del bad_input["brand_palette"]
        with pytest.raises(PromptValidationError, match="brand_palette"):
            build_prompt(bad_input)


class TestBuildNegativePrompt:
    def test_contains_key_terms(self):
        neg = build_negative_prompt()
        assert "nsfw" in neg
        assert "balloon" in neg
        assert "inflatable" in neg
