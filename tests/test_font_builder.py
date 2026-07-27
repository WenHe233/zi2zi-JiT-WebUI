from pathlib import Path

from PIL import Image, ImageDraw
from fontTools.ttLib import TTFont

from data_processing.font_utils import GlyphRendererPool
from zi2zi_webui.font_builder import (
    ATTRIBUTION,
    FontMetadata,
    build_export_package,
    build_ttf,
    codepoint_from_filename,
)
from zi2zi_webui.inference import render_style_references


def _glyph(path: Path):
    image = Image.new("L", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 35, 210, 220), fill="black")
    draw.rectangle((90, 80, 165, 175), fill="white")
    image.convert("RGB").save(path)


def test_filename_codepoint_parser():
    assert codepoint_from_filename("0000_U+4E00_c00_s42.png") == 0x4E00
    assert codepoint_from_filename("uni0041.png") == 0x41
    assert codepoint_from_filename("not-a-glyph.png") is None


def test_build_ttf_with_hole(tmp_path):
    source = tmp_path / "U+4E00.png"
    _glyph(source)
    output = tmp_path / "Demo.ttf"
    report = build_ttf(
        {0x4E00: source},
        output,
        FontMetadata(family_name="Demo"),
    )
    assert output.is_file()
    assert report["glyph_count"] == 2
    assert report["included"][0]["qa"]["has_holes"] is True
    font = TTFont(output)
    assert font.getBestCmap()[0x4E00] == "uni4E00"
    assert font["head"].unitsPerEm == 1000
    package = build_export_package(output, {0x4E00: source})
    assert package.is_file()
    import zipfile

    with zipfile.ZipFile(package) as archive:
        assert {"font/Demo.ttf", "README.txt", "glyph-manifest.json"}.issubset(
            archive.namelist()
        )


def test_target_font_creates_eight_style_references(tmp_path):
    source = tmp_path / "shape.png"
    _glyph(source)
    output = tmp_path / "References.ttf"
    build_ttf(
        {0x4E00 + index: source for index in range(8)},
        output,
        FontMetadata(family_name="References"),
    )
    references = render_style_references(output, tmp_path / "references")
    assert len(references) == 8
    assert all(Path(item).is_file() for item in references)


def test_source_font_pool_uses_ordered_coverage_fallback(tmp_path):
    source = tmp_path / "shape.png"
    _glyph(source)
    first = tmp_path / "Jigmo-1.ttf"
    second = tmp_path / "Jigmo-2.ttf"
    build_ttf(
        {0x4E00: source, 0x4E01: source},
        first,
        FontMetadata(family_name="Jigmo 1"),
    )
    build_ttf(
        {0x4E01: source, 0x4E02: source},
        second,
        FontMetadata(family_name="Jigmo 2"),
    )
    pool = GlyphRendererPool([first, second], 256)
    assert pool.source_for(0x4E00) == first.resolve()
    assert pool.source_for(0x4E01) == first.resolve()
    assert pool.source_for(0x4E02) == second.resolve()
    assert pool.render(0x4E02) is not None


def test_attribution_over_200_glyphs(tmp_path, monkeypatch):
    # Exercise metadata policy without tracing 201 real images.
    from zi2zi_webui import font_builder

    source = tmp_path / "U+0021.png"
    _glyph(source)
    original = font_builder.vectorize_png

    # A single SVG can safely back the metadata-policy fixture.
    svg = tmp_path / "shape.svg"
    original(source, svg)

    def reuse_svg(_source, destination, _options=None):
        Path(destination).write_text(svg.read_text(encoding="utf-8"), encoding="utf-8")
        return {"contours": 1, "has_holes": False, "ink_ratio": 0.5, "bbox": [45, 35, 210, 220]}

    monkeypatch.setattr(font_builder, "vectorize_png", reuse_svg)
    output = tmp_path / "Attributed.ttf"
    report = build_ttf(
        {0x3400 + index: source for index in range(201)},
        output,
        FontMetadata(family_name="Attributed"),
    )
    assert report["attribution_required"] is True
    font = TTFont(output)
    descriptions = font["name"].names
    assert any(
        ATTRIBUTION in item.toUnicode()
        for item in descriptions
        if item.nameID in {10, 13}
    )
