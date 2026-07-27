import pytest

from data_processing.pipeline import _filter_common_codepoints


def test_auto_dataset_range_keeps_detected_common_glyphs_without_index_table():
    common = {0x4E00, 0x20000}

    filtered, index_map = _filter_common_codepoints(common, "auto")

    assert filtered == common
    assert index_map == {}


def test_unknown_dataset_range_is_rejected():
    with pytest.raises(ValueError, match="Unsupported charset"):
        _filter_common_codepoints({0x4E00}, "not-a-range")
