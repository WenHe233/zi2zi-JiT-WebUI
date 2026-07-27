from zi2zi_webui.presets import (
    resolve_preset,
    resolve_selection,
    resolve_selection_by_region,
)


def test_authoritative_preset_counts():
    assert len(resolve_preset("zh-Hans-3500")) == 3500
    assert len(resolve_preset("zh-Hans-6500")) == 6500
    assert len(resolve_preset("zh-Hans-8105")) == 8105
    assert len(resolve_preset("zh-Hant-TW-4808")) == 4808
    assert len(resolve_preset("ja-joyo")) == 2136


def test_selection_deduplicates_and_reports_increment():
    values, increments = resolve_selection(
        ["latin-ascii", "latin-extended"],
        custom_text="ABC中中",
    )
    assert len(values) == len(set(values))
    assert ord("中") in values
    assert increments["latin-ascii"] == 95
    assert increments["latin-extended"] == len(resolve_preset("latin-extended")) - 95


def test_basic_japanese_contains_kana_and_joyo():
    values = resolve_preset("ja-basic")
    assert ord("あ") in values
    assert ord("ア") in values
    assert resolve_preset("ja-joyo").issubset(values)


def test_region_split_uses_primary_for_base_and_custom_text():
    grouped = resolve_selection_by_region(
        ["latin-ascii", "zh-Hans-3500", "ja-basic"],
        primary_region="TC",
        custom_text="測",
    )
    assert ord("A") in grouped["TC"]
    assert ord("測") in grouped["TC"]
    assert ord("一") in grouped["SC"]
    assert ord("あ") in grouped["JP"]


def test_multi_region_composite_expands_children():
    grouped = resolve_selection_by_region(["east-asia-complete"])
    assert {"SC", "TC", "JP", "KR"}.issubset(grouped)
