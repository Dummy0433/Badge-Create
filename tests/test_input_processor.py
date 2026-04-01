# tests/test_input_processor.py
import pytest
from input_processor import process, InputValidationError


NESTED_INPUT = {
    "slogan": "Diedriht",
    "community_type": "host",
    "slogan_lang": "Spanish",
    "anchor_info": {
        "anchor": {
            "nick_name": "Diedriht Mayorga",
            "bio_description": "Bienvenidos ami live",
        },
        "anchor_characterization": "Spanish-speaking anchor with playful style",
        "brand_palette": {
            "primary": {"name": "Black", "hex": "#000000"},
            "secondary": {"name": "Red", "hex": "#FF0000"},
            "tertiary": {"name": "White", "hex": "#FFFFFF"},
        },
    },
}

FLAT_INPUT = {
    "text_output": "Wells",
    "anchor_characterization": "interactive streamer",
    "brand_palette": {
        "primary": {"name": "Gold", "hex": "#D4AF37"},
        "secondary": {"name": "Blue", "hex": "#007BFF"},
        "tertiary": {"name": "Black", "hex": "#000000"},
    },
    "anchor_nickname": "Wells",
    "anchor_bio": "Just chatting",
    "community_type": "host",
    "slogan_lang": "English",
}


class TestProcessNestedInput:
    def test_flattens_nested_format(self):
        result = process(NESTED_INPUT)
        assert result["text_output"] == "Diedriht"
        assert result["anchor_characterization"] == "Spanish-speaking anchor with playful style"
        assert result["brand_palette"]["secondary"]["hex"] == "#FF0000"
        assert result["anchor_nickname"] == "Diedriht Mayorga"
        assert result["anchor_bio"] == "Bienvenidos ami live"
        assert result["community_type"] == "host"
        assert result["slogan_lang"] == "Spanish"

    def test_passes_through_flat_format(self):
        result = process(FLAT_INPUT)
        assert result["text_output"] == "Wells"
        assert result["anchor_characterization"] == "interactive streamer"
        assert result["anchor_nickname"] == "Wells"


class TestProcessValidation:
    def test_missing_slogan_and_text_output_raises(self):
        bad = {"anchor_info": {"anchor": {}, "anchor_characterization": "x", "brand_palette": {}}}
        with pytest.raises(InputValidationError, match="text_output"):
            process(bad)

    def test_missing_brand_palette_raises(self):
        bad = {"slogan": "Test", "anchor_info": {"anchor": {}, "anchor_characterization": "x"}}
        with pytest.raises(InputValidationError, match="brand_palette"):
            process(bad)

    def test_empty_slogan_raises(self):
        bad = {**NESTED_INPUT, "slogan": "  "}
        with pytest.raises(InputValidationError, match="text_output"):
            process(bad)
