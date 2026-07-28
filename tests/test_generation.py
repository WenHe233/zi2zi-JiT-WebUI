from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from generate_chars import ManifestInputSource, expanded_sample_indices
from zi2zi_webui.font_builder import FontMetadata, build_ttf
from zi2zi_webui.inference import (
    build_generation_request,
    write_generation_request,
)


def _glyph(path: Path) -> None:
    image = Image.new("L", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 35, 210, 220), fill="black")
    draw.rectangle((90, 80, 165, 175), fill="white")
    image.save(path)


def _request_fixture(tmp_path):
    glyph = tmp_path / "glyph.png"
    _glyph(glyph)
    source = tmp_path / "source.ttf"
    build_ttf(
        {0x4E00: glyph, 0x4E01: glyph},
        source,
        FontMetadata(family_name="Request source"),
    )
    references = []
    for index in range(8):
        reference = tmp_path / f"reference-{index}.png"
        _glyph(reference)
        references.append(reference)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.touch()
    return source, references, checkpoint


def test_generation_request_is_small_and_lazily_renders_samples(tmp_path):
    source, references, checkpoint = _request_fixture(tmp_path)
    payload = build_generation_request(
        [0x4E00, 0x4E01],
        [source],
        references,
        project_id="project",
        request_id="request",
        region="SC",
        checkpoint=checkpoint,
        seed=42,
        candidates=4,
    )
    manifest = write_generation_request(payload, tmp_path / "request.json")

    assert payload["schema_version"] == 2
    assert "content_images" not in payload
    assert payload["count"] == 2
    request = ManifestInputSource(manifest, pairwise=None)
    sample = request.sample(0)
    assert sample["content_image"].shape == (3, 256, 256)
    assert sample["style_image"].shape == (3, 128, 128)
    assert sample["unicode_label"] == 0x4E00


def test_generation_request_rejects_missing_source_coverage_before_submission(
    tmp_path,
):
    source, references, checkpoint = _request_fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot render"):
        build_generation_request(
            [0x4E02],
            [source],
            references,
            project_id="project",
            request_id="request",
            region="SC",
            checkpoint=checkpoint,
        )


def test_candidate_expansion_uses_indices_without_copying_full_arrays():
    assert expanded_sample_indices(0, 6, 6, 3) == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert expanded_sample_indices(4, 4, 6, 3)[-1] == (1, 2)
