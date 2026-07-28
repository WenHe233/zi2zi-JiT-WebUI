import json
import os
import time
from pathlib import Path

from zi2zi_webui.generation_batches import (
    batch_choices,
    candidates_for_codepoint,
    list_generation_batches,
    load_and_prune_selections,
    preferred_generation_batch,
    review_page,
    scan_generation_batch,
)


def _request_batch(
    root: Path,
    request_id: str,
    *,
    regions=("SC",),
    complete=True,
) -> Path:
    batch = root / f"request-{request_id}"
    batch.mkdir(parents=True)
    (batch / "generation-plan.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "request_id": request_id,
                "candidates": 1,
                "regions": {region: 1 for region in regions},
            }
        ),
        encoding="utf-8",
    )
    for index, region in enumerate(regions):
        region_dir = batch / region
        generated = region_dir / "run" / "generated"
        generated.mkdir(parents=True)
        (region_dir / "request.json").write_text(
            json.dumps({"count": 1}),
            encoding="utf-8",
        )
        (generated / f"0000_U+{0x4E00 + index:04X}.png").touch()
        if complete:
            (region_dir / "generation-result.json").write_text(
                json.dumps({"status": "succeeded", "generated_count": 1}),
                encoding="utf-8",
            )
    return batch


def test_latest_fully_successful_batch_is_preferred_over_newer_incomplete(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    complete = _request_batch(generation, "complete")
    incomplete = _request_batch(generation, "incomplete", complete=False)
    now = time.time_ns()
    os.utime(
        incomplete / "generation-plan.json",
        ns=(now + 2_000_000_000, now + 2_000_000_000),
    )

    batches = list_generation_batches(generation)
    assert batches[0].batch_id == "incomplete"
    assert preferred_generation_batch(batches).batch_id == "complete"
    choices, preferred = batch_choices(generation)
    assert choices
    assert preferred == str(complete.resolve())


def test_batch_scan_prefers_manual_then_primary_region_then_c00(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    batch = _request_batch(
        generation,
        "regional",
        regions=("SC", "TC", "JP"),
    )
    for region in ("SC", "TC", "JP"):
        generated = batch / region / "run" / "generated"
        (generated / "0001_U+4E10_c01_s42.png").touch()
        (generated / "0001_U+4E10_c00_s42.png").touch()

    glyphs = scan_generation_batch(batch, primary_region="TC")
    assert glyphs[0x4E10].parts[-4] == "TC"
    assert "_c00_" in glyphs[0x4E10].name

    manual = batch / "SC" / "run" / "generated" / "0001_U+4E10_c01_s42.png"
    glyphs = scan_generation_batch(
        batch,
        primary_region="TC",
        selections={"U+4E10": str(manual)},
    )
    assert glyphs[0x4E10] == manual.resolve()
    assert candidates_for_codepoint(batch, 0x4E10)[0].name.endswith(
        "_c00_s42.png"
    )


def test_legacy_batch_uses_newest_duplicate_by_mtime(tmp_path):
    legacy = tmp_path / "legacy"
    first = legacy / "first" / "0000_U+4E00.png"
    second = legacy / "second" / "0000_U+4E00.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    now = time.time_ns()
    os.utime(first, ns=(now, now))
    os.utime(
        second,
        ns=(now + 2_000_000_000, now + 2_000_000_000),
    )

    assert scan_generation_batch(legacy)[0x4E00] == second


def test_review_page_has_search_and_no_global_500_item_truncation(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    for index in range(525):
        (legacy / f"{index:04d}_U+{0x3400 + index:04X}.png").touch()

    images, codepoints, page, total_pages = review_page(
        legacy,
        page=6,
        page_size=100,
    )
    assert len(images) == 25
    assert len(codepoints) == 25
    assert (page, total_pages) == (6, 6)

    images, codepoints, _page, _total = review_page(
        legacy,
        search=f"{0x3400 + 500:04X}",
        page=1,
    )
    assert len(images) == 1
    assert codepoints == [0x3400 + 500]


def test_invalid_manual_selections_are_pruned(tmp_path):
    valid = tmp_path / "valid.png"
    valid.touch()
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "U+4E00": str(valid),
                "U+4E01": str(tmp_path / "missing.png"),
            }
        ),
        encoding="utf-8",
    )

    assert load_and_prune_selections(manifest) == {"U+4E00": str(valid)}
    assert "U+4E01" not in manifest.read_text(encoding="utf-8")
