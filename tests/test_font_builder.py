from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from data_processing.font_utils import GlyphRendererPool, get_preserved_codepoints
from zi2zi_webui.font_builder import (
    ATTRIBUTION,
    FontMetadata,
    _glyph_from_svg,
    build_export_package,
    build_ttf,
    codepoint_from_filename,
    scan_glyph_directory,
    vectorize_png,
)
from zi2zi_webui.inference import (
    exclude_existing_target_glyphs,
    render_style_references,
)


def _glyph(path: Path, outer=(45, 35, 210, 220), inner=(90, 80, 165, 175)):
    image = Image.new("L", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(outer, fill="black")
    draw.rectangle(inner, fill="white")
    image.convert("RGB").save(path)


def test_filename_codepoint_parser():
    assert codepoint_from_filename("0000_U+4E00_c00_s42.png") == 0x4E00
    assert codepoint_from_filename("uni0041.png") == 0x41
    assert codepoint_from_filename("not-a-glyph.png") is None


def test_scan_glyph_directory_finds_nested_generator_outputs(tmp_path):
    generated = tmp_path / "ab2-steps20" / "generated"
    generated.mkdir(parents=True)
    first = generated / "0000_U+3400.png"
    second = generated / "0001_U+3401.png"
    _glyph(first)
    _glyph(second)

    assert scan_glyph_directory(tmp_path) == {
        0x3400: first,
        0x3401: second,
    }


def test_scan_prefers_generated_glyph_over_pairwise_preview(tmp_path):
    generated = tmp_path / "run" / "generated"
    pairs = tmp_path / "run" / "pairs"
    generated.mkdir(parents=True)
    pairs.mkdir(parents=True)
    canonical = generated / "0000_U+3400.png"
    preview = pairs / "0000_U+3400.png"
    _glyph(canonical)
    _glyph(preview)

    assert scan_glyph_directory(tmp_path)[0x3400] == canonical


def test_vectorization_supports_unicode_output_paths(tmp_path):
    source = tmp_path / "glyph.png"
    _glyph(source)
    svg = tmp_path / "中文字体" / "U+3400.svg"

    report = vectorize_png(source, svg)

    assert svg.is_file()
    assert report["contours"] > 0


def test_vectorization_falls_back_when_vtracer_panics(tmp_path, monkeypatch):
    import vtracer

    class SimulatedPanic(BaseException):
        pass

    def panic(*_args, **_kwargs):
        raise SimulatedPanic("simulated Rust panic")

    monkeypatch.setattr(vtracer, "convert_image_to_svg_py", panic)
    source = tmp_path / "glyph.png"
    _glyph(source)
    svg = tmp_path / "fallback.svg"

    report = vectorize_png(source, svg)

    assert svg.is_file()
    assert report["vectorizer"] == "opencv-fallback"
    assert "SimulatedPanic" in report["vectorizer_warning"]


def test_svg_path_and_group_transforms_are_preserved(tmp_path):
    svg = tmp_path / "transformed.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">'
        '<g transform="translate(40, 30)">'
        '<path d="M0 0 L20 0 L20 10 L0 10 Z" '
        'transform="translate(60, 20)"/>'
        "</g></svg>",
        encoding="utf-8",
    )

    glyph = _glyph_from_svg(svg, (1, 0, 0, 1, 0, 0))
    glyph.recalcBounds(None)

    assert (glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax) == (100, 50, 120, 60)


def test_mapped_whitespace_is_preserved_without_an_outline(tmp_path):
    source = tmp_path / "shape.png"
    _glyph(source)
    base_path = tmp_path / "Base.ttf"
    build_ttf(
        {0x4E00: source},
        base_path,
        FontMetadata(family_name="Base"),
    )
    font = TTFont(base_path)
    empty_name = "space"
    empty_pen = TTGlyphPen(None)
    font["glyf"].glyphs[empty_name] = empty_pen.glyph()
    font["hmtx"].metrics[empty_name] = (500, 0)
    font.setGlyphOrder([*font.getGlyphOrder(), empty_name])
    for table in font["cmap"].tables:
        if table.isUnicode() and hasattr(table, "cmap") and table.format != 14:
            table.cmap[0x20] = empty_name
    font.save(base_path)
    font.close()

    reopened = TTFont(base_path)
    try:
        assert 0x20 in get_preserved_codepoints(reopened)
    finally:
        reopened.close()

    generated, skipped = exclude_existing_target_glyphs([0x20, 0x4E01], base_path)
    assert generated == [0x4E01]
    assert skipped == [0x20]


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
    stale_svg = output.with_suffix("") / "svg" / "U+4E01.svg"
    stale_svg.write_text("<svg/>", encoding="utf-8")
    package = build_export_package(output, {0x4E00: source})
    assert package.is_file()
    import zipfile

    with zipfile.ZipFile(package) as archive:
        assert {"font/Demo.ttf", "README.txt", "glyph-manifest.json"}.issubset(
            archive.namelist()
        )
        assert "svg/U+4E00.svg" in archive.namelist()
        assert "svg/U+4E01.svg" not in archive.namelist()


def test_target_font_existing_glyphs_are_skipped_and_missing_glyphs_are_merged(tmp_path):
    source = tmp_path / "shape.png"
    _glyph(source)
    base_path = tmp_path / "Base.ttf"
    build_ttf(
        {0x4E00: source},
        base_path,
        FontMetadata(
            family_name="Base",
            license_description="Original target license",
        ),
    )
    base_font = TTFont(base_path)
    base_order = base_font.getGlyphOrder()
    base_tables = set(base_font.keys())
    original_name = base_font.getBestCmap()[0x4E00]
    original_glyph = base_font["glyf"][original_name].compile(base_font["glyf"])
    original_upm = base_font["head"].unitsPerEm
    base_font.close()

    generated, skipped = exclude_existing_target_glyphs(
        [0x4E00, 0x4E01],
        base_path,
    )
    assert generated == [0x4E01]
    assert skipped == [0x4E00]

    output = tmp_path / "Extended.ttf"
    report = build_ttf(
        {0x4E00: source, 0x4E01: source},
        output,
        FontMetadata(
            family_name="Extended",
            license_description="Additional generated-glyph notice",
        ),
        base_font_path=base_path,
    )

    assert report["base_glyph_count"] == len(base_order)
    assert report["added_glyph_count"] == 1
    assert report["skipped_existing_count"] == 1
    merged = TTFont(output)
    merged_cmap = merged.getBestCmap()
    assert merged_cmap[0x4E00] == original_name
    assert 0x4E01 in merged_cmap
    assert merged["glyf"][original_name].compile(merged["glyf"]) == original_glyph
    assert merged["head"].unitsPerEm == original_upm
    assert merged["OS/2"].usLastCharIndex == 0x4E01
    assert set(base_tables) <= set(merged.keys())
    assert len(merged.getGlyphOrder()) == len(base_order) + 1
    merged_license = "\n".join(
        record.toUnicode()
        for record in merged["name"].names
        if record.nameID == 13
    )
    assert "Original target license" in merged_license
    assert "Additional generated-glyph notice" in merged_license
    merged.close()


def test_generated_cjk_layout_matches_base_font_size_center_and_advance(tmp_path):
    base_image = tmp_path / "base.png"
    generated_image = tmp_path / "generated.png"
    _glyph(base_image, outer=(40, 40, 215, 215), inner=(90, 90, 165, 165))
    _glyph(generated_image, outer=(75, 75, 180, 180), inner=(105, 105, 150, 150))
    base_path = tmp_path / "Base.ttf"
    base_codepoints = range(0x4E00, 0x4E08)
    generated_codepoints = range(0x4E10, 0x4E18)
    build_ttf(
        {codepoint: base_image for codepoint in base_codepoints},
        base_path,
        FontMetadata(family_name="Base"),
    )

    output = tmp_path / "Extended.ttf"
    report = build_ttf(
        {codepoint: generated_image for codepoint in generated_codepoints},
        output,
        FontMetadata(family_name="Extended"),
        base_font_path=base_path,
    )

    assert report["cjk_layout"] is not None
    font = TTFont(output)
    cmap = font.getBestCmap()

    def layout(codepoints):
        rows = []
        for codepoint in codepoints:
            name = cmap[codepoint]
            glyph = font["glyf"][name]
            rows.append(
                (
                    glyph.xMax - glyph.xMin,
                    glyph.yMax - glyph.yMin,
                    (glyph.xMin + glyph.xMax) / 2,
                    (glyph.yMin + glyph.yMax) / 2,
                    font["hmtx"][name][0],
                )
            )
        return tuple(median(item[index] for item in rows) for index in range(5))

    base_layout = layout(base_codepoints)
    generated_layout = layout(generated_codepoints)
    assert generated_layout[:2] == base_layout[:2]
    assert abs(generated_layout[2] - base_layout[2]) <= 4
    assert abs(generated_layout[3] - base_layout[3]) <= 4
    assert generated_layout[4] == base_layout[4]
    font.close()


def test_incremental_packaging_can_export_an_unchanged_base_font(tmp_path):
    source = tmp_path / "shape.png"
    _glyph(source)
    base_path = tmp_path / "Base.ttf"
    build_ttf(
        {0x4E00: source},
        base_path,
        FontMetadata(family_name="Base"),
    )

    output = tmp_path / "Renamed.ttf"
    report = build_ttf(
        {},
        output,
        FontMetadata(family_name="Renamed"),
        base_font_path=base_path,
    )

    assert report["added_glyph_count"] == 0
    unchanged = TTFont(output)
    assert unchanged.getBestCmap()[0x4E00]
    unchanged.close()


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
