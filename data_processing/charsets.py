from typing import FrozenSet

from .charsets_data import (
    GB2312_CODEPOINTS,
    BIG5_CODEPOINTS,
    JISX0208_CODEPOINTS,
    KSX1001_CODEPOINTS,
)


SUPPORTED_CHARSETS = frozenset([
    "auto",
    "gb2312",
    "gbk",
    "big5",
    "jisx0208",
    "ksx1001",
    "all-cjk",
])

_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2EBEF),
    (0x30000, 0x3134F),
)


def _is_cjk(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in _CJK_RANGES)


def _decode_gbk_codepoints() -> FrozenSet[int]:
    values = set()
    for lead in range(0x81, 0xFF):
        for trail in range(0x40, 0xFF):
            if trail == 0x7F:
                continue
            try:
                text = bytes((lead, trail)).decode("gbk")
            except UnicodeDecodeError:
                continue
            if len(text) == 1 and _is_cjk(ord(text)):
                values.add(ord(text))
    return frozenset(values)


GBK_CODEPOINTS = _decode_gbk_codepoints()
ALL_CJK_CODEPOINTS = frozenset(
    codepoint
    for start, end in _CJK_RANGES
    for codepoint in range(start, end + 1)
)


def get_charset_codepoints(charset_name: str) -> FrozenSet[int]:
    charset_lower = charset_name.lower()
    if charset_lower not in SUPPORTED_CHARSETS:
        raise ValueError(
            f"Unknown charset: {charset_name}. "
            f"Available: {', '.join(sorted(SUPPORTED_CHARSETS))}"
        )

    if charset_lower == "gb2312":
        return GB2312_CODEPOINTS
    if charset_lower == "gbk":
        return GBK_CODEPOINTS
    if charset_lower == "big5":
        return BIG5_CODEPOINTS
    if charset_lower == "jisx0208":
        return JISX0208_CODEPOINTS
    if charset_lower == "ksx1001":
        return KSX1001_CODEPOINTS
    if charset_lower in {"auto", "all-cjk"}:
        return ALL_CJK_CODEPOINTS

    raise ValueError(f"Unknown charset: {charset_name}")
