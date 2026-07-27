from data_processing.charsets import get_charset_codepoints


def test_gbk_is_larger_than_gb2312_and_all_cjk_contains_extensions():
    gb2312 = get_charset_codepoints("gb2312")
    gbk = get_charset_codepoints("gbk")
    all_cjk = get_charset_codepoints("all-cjk")
    assert len(gbk) == 20923
    assert gb2312 < gbk
    assert 0x20000 in all_cjk


def test_auto_range_covers_all_cjk_candidates():
    assert get_charset_codepoints("auto") == get_charset_codepoints("all-cjk")
