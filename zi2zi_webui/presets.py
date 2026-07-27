from __future__ import annotations

import codecs
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).with_name("preset_data")


@dataclass(frozen=True)
class CharsetPreset:
    id: str
    name_zh: str
    name_en: str
    region: str
    description_zh: str
    description_en: str
    source: str
    kind: str
    value: str
    expected_count: int | None = None
    includes: tuple[str, ...] = ()


PRESETS = {
    item.id: item
    for item in (
        CharsetPreset(
            "latin-ascii", "拉丁 ASCII", "Latin ASCII", "BASE",
            "可打印 ASCII U+0020–U+007E。", "Printable ASCII U+0020–U+007E.",
            "Unicode", "ranges", "0020-007E", 95,
        ),
        CharsetPreset(
            "latin-western", "西欧拉丁", "Latin Western", "BASE",
            "Windows-1252 可打印字符。", "Printable Windows-1252 characters.",
            "Python cp1252 codec", "encoding", "cp1252",
        ),
        CharsetPreset(
            "latin-extended", "拉丁扩展", "Latin Extended", "BASE",
            "Basic Latin、Latin-1 与 Latin Extended-A。", "Basic Latin, Latin-1 and Latin Extended-A.",
            "Unicode blocks", "ranges", "0020-017F", 352,
        ),
        CharsetPreset(
            "latin-full-european", "完整欧洲拉丁", "Full European Latin", "BASE",
            "扩展至 Latin Extended-B 与组合附加符号。", "Adds Latin Extended-B and combining marks.",
            "Unicode blocks", "ranges", "0020-024F,0300-036F",
        ),
        CharsetPreset(
            "cjk-punctuation", "CJK 标点", "CJK punctuation", "BASE",
            "CJK 符号、标点和全角形式。", "CJK symbols, punctuation and full-width forms.",
            "Unicode blocks", "ranges", "3000-303F,FE30-FE4F,FF01-FF60",
        ),
        CharsetPreset(
            "symbols-common", "常用符号", "Common symbols", "BASE",
            "常用货币、箭头、数学与几何符号。", "Common currency, arrows, math and geometric symbols.",
            "Unicode blocks", "ranges", "20A0-20CF,2190-21FF,2200-22FF,25A0-25FF",
        ),
        CharsetPreset(
            "zh-Hans-3500", "简体常用 3500", "Simplified Chinese 3500", "SC",
            "《通用规范汉字表》一级字。", "Level 1 of the Table of General Standard Chinese Characters.",
            "PRC Ministry of Education, 2013", "file", "zh-Hans-8105.txt", 3500,
        ),
        CharsetPreset(
            "zh-Hans-6500", "简体标准 6500", "Simplified Chinese 6500", "SC",
            "《通用规范汉字表》一级与二级字。", "Levels 1 and 2 of the general standard table.",
            "PRC Ministry of Education, 2013", "file", "zh-Hans-8105.txt", 6500,
        ),
        CharsetPreset(
            "zh-Hans-8105", "简体完整 8105", "Simplified Chinese 8105", "SC",
            "完整《通用规范汉字表》三级 8105 字。", "All 8,105 entries of the general standard table.",
            "PRC Ministry of Education, 2013", "file", "zh-Hans-8105.txt", 8105,
        ),
        CharsetPreset(
            "zh-Hans-GBK", "GBK 兼容", "GBK compatibility", "SC",
            "Python GBK 映射中的 Unicode 字符。", "Unicode characters mapped by Python's GBK codec.",
            "Python gbk codec", "encoding", "gbk",
        ),
        CharsetPreset(
            "zh-Hant-TW-4808", "台湾常用 4808", "Taiwan common 4808", "TC",
            "台湾教育部常用国字 4,808 字。", "4,808 common characters from Taiwan's Ministry of Education.",
            "Taiwan Ministry of Education", "file", "zh-Hant-TW-4808.txt", 4808,
        ),
        CharsetPreset(
            "zh-Hant-TW-extended", "台湾常用＋次常用", "Taiwan common + less common", "TC",
            "实用扩展集：教育部 4,808 常用字与 Big5 映射的并集。",
            "Practical extended set: union of the official 4,808 list and Big5 mapping.",
            "Taiwan Ministry of Education + Python Big5 codec", "composite", "",
            includes=("zh-Hant-TW-4808", "zh-Hant-Big5"),
        ),
        CharsetPreset(
            "zh-Hant-Big5", "Big5 兼容", "Big5 compatibility", "TC",
            "Python Big5 映射字符。", "Characters mapped by Python's Big5 codec.",
            "Python big5 codec", "encoding", "big5",
        ),
        CharsetPreset(
            "zh-Hant-HK-HKSCS", "香港 HKSCS", "Hong Kong HKSCS", "TC",
            "Big5-HKSCS 映射，覆盖香港粤语、人名和地名用字。", "Big5-HKSCS mapping for Hong Kong usage.",
            "Python big5hkscs codec", "encoding", "big5hkscs",
        ),
        CharsetPreset(
            "ja-basic", "日语常用", "Japanese basic", "JP",
            "平假名、片假名、日文标点与 2,136 常用汉字。", "Kana, Japanese punctuation and 2,136 Joyo kanji.",
            "Japan Agency for Cultural Affairs, 2010", "composite", "",
            includes=("ja-joyo", "ja-kana", "cjk-punctuation"),
        ),
        CharsetPreset(
            "ja-joyo", "日本常用汉字 2136", "Joyo kanji 2136", "JP",
            "2010 年常用汉字表。", "2010 Joyo kanji table.",
            "Japan Agency for Cultural Affairs, 2010", "file", "ja-joyo-2136.txt", 2136,
        ),
        CharsetPreset(
            "ja-kana", "日语假名", "Japanese kana", "JP",
            "平假名、片假名及片假名语音扩展。", "Hiragana, katakana and phonetic extensions.",
            "Unicode blocks", "ranges", "3040-30FF,31F0-31FF",
        ),
        CharsetPreset(
            "ja-jis0208", "JIS X 0208", "JIS X 0208", "JP",
            "JIS X 0208 兼容字符。", "JIS X 0208 compatibility characters.",
            "Python shift_jis codec", "encoding", "shift_jis",
        ),
        CharsetPreset(
            "ja-jis0213", "JIS X 0213", "JIS X 0213", "JP",
            "JIS X 0213 兼容字符。", "JIS X 0213 compatibility characters.",
            "Python shift_jis_2004 codec", "encoding", "shift_jis_2004",
        ),
        CharsetPreset(
            "ja-cp932", "Windows CP932", "Windows CP932", "JP",
            "Windows 日语兼容字符。", "Windows Japanese compatibility characters.",
            "Python cp932 codec", "encoding", "cp932",
        ),
        CharsetPreset(
            "ko-ksx1001", "韩文 KS X 1001", "Korean KS X 1001", "KR",
            "KS X 1001/EUC-KR 兼容字符。", "KS X 1001/EUC-KR compatibility characters.",
            "Python euc_kr codec", "encoding", "euc_kr",
        ),
        CharsetPreset(
            "east-asia-complete", "完整东亚兼容", "Complete East Asian compatibility", "MULTI",
            "GBK、Big5-HKSCS、JIS X 0213 与 KS X 1001 的去重并集。",
            "Deduplicated union of GBK, Big5-HKSCS, JIS X 0213 and KS X 1001.",
            "Deterministic Python codec mappings", "composite", "",
            includes=("zh-Hans-GBK", "zh-Hant-HK-HKSCS", "ja-jis0213", "ko-ksx1001"),
        ),
    )
}


def _parse_ranges(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        start, separator, end = part.strip().partition("-")
        first = int(start, 16)
        last = int(end, 16) if separator else first
        result.update(range(first, last + 1))
    return result


@lru_cache(maxsize=None)
def _encoding_codepoints(encoding: str) -> frozenset[int]:
    codecs.lookup(encoding)
    result: set[int] = set()
    for value in range(256):
        try:
            text = bytes([value]).decode(encoding)
        except UnicodeDecodeError:
            continue
        result.update(ord(char) for char in text if char.isprintable())
    for first in range(256):
        for second in range(256):
            try:
                text = bytes([first, second]).decode(encoding)
            except UnicodeDecodeError:
                continue
            result.update(ord(char) for char in text if char.isprintable())
    return frozenset(result)


def _load_codepoint_file(filename: str) -> list[int]:
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Preset data is missing: {path}. Reinstall the package or use an encoding preset."
        )
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.split("#", 1)[0].strip()
        if not item:
            continue
        values.append(int(item.removeprefix("U+"), 16))
    return values


@lru_cache(maxsize=None)
def resolve_preset(preset_id: str) -> frozenset[int]:
    preset = PRESETS[preset_id]
    if preset.kind == "ranges":
        values = _parse_ranges(preset.value)
    elif preset.kind == "encoding":
        values = set(_encoding_codepoints(preset.value))
    elif preset.kind == "file":
        ordered = _load_codepoint_file(preset.value)
        if preset.expected_count is not None:
            ordered = ordered[: preset.expected_count]
        values = set(ordered)
    elif preset.kind == "composite":
        values = set()
        for child in preset.includes:
            values.update(resolve_preset(child))
    else:
        raise ValueError(f"Unsupported preset kind: {preset.kind}")
    if preset.expected_count is not None and len(values) != preset.expected_count:
        raise ValueError(
            f"{preset.id} expected {preset.expected_count} unique codepoints, got {len(values)}"
        )
    return frozenset(values)


def resolve_selection(
    preset_ids: Iterable[str],
    custom_text: str = "",
    custom_file: str | Path | None = None,
) -> tuple[list[int], dict[str, int]]:
    selected: set[int] = set()
    increments: dict[str, int] = {}
    for preset_id in preset_ids:
        before = len(selected)
        selected.update(resolve_preset(preset_id))
        increments[preset_id] = len(selected) - before
    selected.update(ord(char) for char in custom_text if not char.isspace())
    if custom_file:
        for line in Path(custom_file).read_text(encoding="utf-8").splitlines():
            item = line.split("#", 1)[0].strip()
            if not item:
                continue
            if item.upper().startswith("U+"):
                selected.add(int(item[2:], 16))
            else:
                selected.update(ord(char) for char in item if not char.isspace())
    return sorted(selected), increments


def resolve_selection_by_region(
    preset_ids: Iterable[str],
    *,
    primary_region: str = "SC",
    custom_text: str = "",
    custom_file: str | Path | None = None,
) -> dict[str, list[int]]:
    """Split a selection into regional rendering groups.

    Region-specific composites stay together so, for example, Japanese
    punctuation included by ``ja-basic`` is rendered with the Japanese source
    font. Top-level BASE presets and custom input use the selected primary
    region. MULTI composites are expanded into their regional children.
    """
    if primary_region not in {"SC", "TC", "JP", "KR"}:
        raise ValueError(f"Unsupported primary region: {primary_region}")
    grouped: dict[str, set[int]] = {}

    def add_preset(preset_id: str, inherited_region: str | None = None) -> None:
        preset = PRESETS[preset_id]
        region = inherited_region or preset.region
        if region == "MULTI":
            for child in preset.includes:
                add_preset(child)
            return
        destination = primary_region if region == "BASE" else region
        grouped.setdefault(destination, set()).update(resolve_preset(preset_id))

    for preset_id in preset_ids:
        add_preset(preset_id)
    custom, _ = resolve_selection([], custom_text=custom_text, custom_file=custom_file)
    if custom:
        grouped.setdefault(primary_region, set()).update(custom)
    return {region: sorted(values) for region, values in grouped.items() if values}


def preset_catalog(language: str = "zh") -> list[dict]:
    items = []
    for preset in PRESETS.values():
        value = asdict(preset)
        value["label"] = preset.name_en if language == "en" else preset.name_zh
        value["description"] = (
            preset.description_en if language == "en" else preset.description_zh
        )
        items.append(value)
    return items


def write_selection_manifest(
    output: str | Path,
    preset_ids: list[str],
    codepoints: list[int],
    increments: dict[str, int],
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "presets": preset_ids,
                "increments": increments,
                "count": len(codepoints),
                "codepoints": [f"U+{value:04X}" for value in codepoints],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
