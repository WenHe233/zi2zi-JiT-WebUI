from __future__ import annotations

import html
import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from fontTools.fontBuilder import FontBuilder
from fontTools.misc.transform import Transform
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import parse_path
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable

from data_processing.font_utils import get_preserved_codepoints


ATTRIBUTION = "Created using zi2zi-JiT artifacts"
UPSTREAM_URL = "https://github.com/kaonashi-tyc/zi2zi-JiT"


@dataclass
class VectorizeOptions:
    threshold: int = 180
    despeckle_area: int = 4
    smooth: bool = True


@dataclass
class FontMetadata:
    family_name: str
    style_name: str = "Regular"
    version: str = "1.000"
    designer: str = ""
    copyright: str = ""
    license_description: str = ""


def codepoint_from_filename(path: str | Path) -> int | None:
    match = re.search(r"(?:U\+|uni|u)([0-9A-Fa-f]{4,6})", Path(path).stem)
    return int(match.group(1), 16) if match else None


def preprocess_glyph(
    source: str | Path,
    destination: str | Path,
    options: VectorizeOptions | None = None,
) -> dict:
    options = options or VectorizeOptions()
    try:
        with Image.open(source) as image:
            gray = np.asarray(image.convert("L"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read image: {source}") from exc
    if gray.mean() < 127:
        gray = 255 - gray
    _, bitmap = cv2.threshold(gray, options.threshold, 255, cv2.THRESH_BINARY_INV)
    if options.despeckle_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bitmap, 8)
        cleaned = np.zeros_like(bitmap)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= options.despeckle_area:
                cleaned[labels == label] = 255
        bitmap = cleaned
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(255 - bitmap).save(destination, format="PNG")
    contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ink = np.where(bitmap > 0)
    bbox = None
    if len(ink[0]):
        bbox = [int(ink[1].min()), int(ink[0].min()), int(ink[1].max()), int(ink[0].max())]
    return {
        "contours": len(contours),
        "has_holes": bool(hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0)),
        "ink_ratio": round(float((bitmap > 0).mean()), 5),
        "bbox": bbox,
        "empty": not contours,
    }


def vectorize_png(
    source: str | Path,
    svg_path: str | Path,
    options: VectorizeOptions | None = None,
) -> dict:
    svg_path = Path(svg_path)
    processed = svg_path.with_suffix(".processed.png")
    qa = preprocess_glyph(source, processed, options)
    if qa["empty"]:
        raise ValueError(f"No glyph outline found in {source}")
    try:
        import vtracer

        # The Rust vtracer binding cannot reliably open Unicode paths on
        # Windows. Bridge through a short ASCII-only temporary directory and
        # copy the result back with Python's Unicode-safe filesystem APIs.
        with tempfile.TemporaryDirectory(prefix="zi2zi-vtracer-") as temporary:
            temporary_root = Path(temporary)
            temporary_input = temporary_root / "glyph.png"
            temporary_output = temporary_root / "glyph.svg"
            shutil.copyfile(processed, temporary_input)
            vtracer.convert_image_to_svg_py(
                str(temporary_input),
                str(temporary_output),
                colormode="binary",
                mode="spline",
            )
            shutil.copyfile(temporary_output, svg_path)
        qa["vectorizer"] = "vtracer"
    except ImportError:
        _opencv_svg(processed, svg_path)
        qa["vectorizer"] = "opencv-fallback"
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        _opencv_svg(processed, svg_path)
        qa["vectorizer"] = "opencv-fallback"
        qa["vectorizer_warning"] = f"{type(exc).__name__}: {exc}"
    return qa


def _opencv_svg(source: str | Path, destination: str | Path) -> None:
    with Image.open(source) as image:
        gray = np.asarray(image.convert("L"))
    bitmap = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
    contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for contour in contours:
        points = contour[:, 0, :]
        if len(points) < 3:
            continue
        commands = [f"M {points[0][0]} {points[0][1]}"]
        commands.extend(f"L {x} {y}" for x, y in points[1:])
        commands.append("Z")
        parts.append(" ".join(commands))
    Path(destination).write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
        f'<path d="{" ".join(parts)}" fill="black" fill-rule="evenodd"/></svg>',
        encoding="utf-8",
    )


_SVG_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
_SVG_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
)


def _svg_transform(value: str) -> Transform:
    result = Transform()
    position = 0
    for match in _SVG_TRANSFORM_RE.finditer(value):
        if value[position : match.start()].strip(" ,\t\r\n"):
            raise ValueError(f"Invalid SVG transform: {value}")
        name = match.group(1).lower()
        values = [float(item) for item in _SVG_NUMBER_RE.findall(match.group(2))]
        if name == "matrix" and len(values) == 6:
            operation = Transform(*values)
        elif name == "translate" and len(values) in {1, 2}:
            operation = Transform().translate(
                values[0],
                values[1] if len(values) == 2 else 0,
            )
        elif name == "scale" and len(values) in {1, 2}:
            operation = Transform().scale(
                values[0],
                values[1] if len(values) == 2 else None,
            )
        elif name == "rotate" and len(values) in {1, 3}:
            angle = math.radians(values[0])
            if len(values) == 3:
                operation = (
                    Transform()
                    .translate(values[1], values[2])
                    .rotate(angle)
                    .translate(-values[1], -values[2])
                )
            else:
                operation = Transform().rotate(angle)
        elif name == "skewx" and len(values) == 1:
            operation = Transform().skew(math.radians(values[0]), 0)
        elif name == "skewy" and len(values) == 1:
            operation = Transform().skew(0, math.radians(values[0]))
        else:
            raise ValueError(f"Unsupported SVG transform: {match.group(0)}")
        result = result.transform(operation)
        position = match.end()
    if not position or value[position:].strip(" ,\t\r\n"):
        raise ValueError(f"Invalid SVG transform: {value}")
    return result


def _glyph_from_svg(
    svg_path: Path,
    transform: tuple[float, float, float, float, float, float],
):
    root = ET.parse(svg_path).getroot()
    pen = TTGlyphPen(None)
    quadratic = Cu2QuPen(pen, max_err=1.0, all_quadratic=True)
    transformed = TransformPen(quadratic, transform)
    # VTracer places each contour in its own local coordinate system and
    # restores its canvas position with SVG transform attributes. Parsing only
    # the path's ``d`` value collapses every contour around the origin.
    found = False

    def draw(element: ET.Element, inherited: Transform) -> None:
        nonlocal found
        current = inherited
        if element.attrib.get("transform"):
            current = inherited.transform(_svg_transform(element.attrib["transform"]))
        if element.tag.endswith("path") and element.attrib.get("d"):
            parse_path(element.attrib["d"], TransformPen(transformed, current))
            found = True
        for child in element:
            draw(child, current)

    draw(root, Transform())
    if not found:
        raise ValueError(f"No SVG path in {svg_path}")
    return pen.glyph()


def _glyph_name(codepoint: int, used_names: set[str]) -> str:
    base = f"uni{codepoint:04X}" if codepoint <= 0xFFFF else f"u{codepoint:06X}"
    if base not in used_names:
        return base
    suffix = 1
    while f"{base}.zi2zi{suffix}" in used_names:
        suffix += 1
    return f"{base}.zi2zi{suffix}"


def _update_cmap(font: TTFont, additions: dict[int, str]) -> None:
    if not additions:
        return
    unicode_tables = [
        table
        for table in font["cmap"].tables
        if table.isUnicode() and hasattr(table, "cmap")
    ]
    if any(codepoint > 0xFFFF for codepoint in additions) and not any(
        table.format in {12, 13} for table in unicode_tables
    ):
        table = CmapSubtable.newSubtable(12)
        table.platformID = 3
        table.platEncID = 10
        table.language = 0
        table.cmap = dict(font.getBestCmap() or {})
        font["cmap"].tables.append(table)
        unicode_tables.append(table)
    for table in unicode_tables:
        for codepoint, glyph_name in additions.items():
            if codepoint <= 0xFFFF or table.format in {12, 13}:
                table.cmap[codepoint] = glyph_name


def _set_name(font: TTFont, name_id: int, value: str) -> None:
    if not value:
        return
    updated = False
    for record in font["name"].names:
        if record.nameID != name_id:
            continue
        try:
            record.string = value.encode(record.getEncoding())
            updated = True
        except (LookupError, UnicodeEncodeError):
            continue
    if not updated:
        font["name"].setName(value, name_id, 3, 1, 0x409)


def _update_base_font_names(
    font: TTFont,
    metadata: FontMetadata,
    *,
    attributed: bool,
) -> None:
    family = metadata.family_name.strip()
    style = metadata.style_name.strip() or "Regular"
    ps_name = re.sub(r"[^A-Za-z0-9-]", "", f"{family}-{style}")[:63] or "Zi2ZiFont"
    values = {
        1: family,
        2: style,
        3: f"{family}-{style}-{metadata.version}",
        4: f"{family} {style}".strip(),
        5: f"Version {metadata.version}",
        6: ps_name,
    }
    if metadata.copyright:
        values[0] = metadata.copyright
    if metadata.designer:
        values[9] = metadata.designer
    for name_id, value in values.items():
        _set_name(font, name_id, value)
    if metadata.license_description:
        existing_licenses = [
            record.toUnicode()
            for record in font["name"].names
            if record.nameID == 13
        ]
        combined_license = "\n".join(
            dict.fromkeys([*existing_licenses, metadata.license_description])
        )
        _set_name(font, 13, combined_license)
    if attributed:
        descriptions = [
            record.toUnicode()
            for record in font["name"].names
            if record.nameID == 10
        ]
        description = "\n".join(dict.fromkeys([*descriptions, ATTRIBUTION, UPSTREAM_URL]))
        _set_name(font, 10, description)


def build_ttf(
    glyph_images: dict[int, str | Path],
    output_path: str | Path,
    metadata: FontMetadata,
    *,
    metrics_profile: str = "proportional",
    vectorize_options: VectorizeOptions | None = None,
    svg_dir: str | Path | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    base_font_path: str | Path | None = None,
) -> dict:
    if not metadata.family_name.strip():
        raise ValueError("Font family name is required")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_dir = Path(svg_dir or output_path.with_suffix("")) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    base_font = None
    existing_outlines: set[int] = set()
    if base_font_path:
        candidate_font = TTFont(str(base_font_path))
        try:
            if "glyf" not in candidate_font or "hmtx" not in candidate_font:
                raise ValueError(
                    "Incremental packaging requires a static TrueType font with glyf/hmtx "
                    "tables; CFF-outline OTF fonts are not supported"
                )
            if "fvar" in candidate_font or "gvar" in candidate_font:
                raise ValueError(
                    "Incremental packaging does not support variable-font targets"
                )
            existing_outlines = get_preserved_codepoints(candidate_font)
            glyphs = candidate_font["glyf"].glyphs
            metrics = candidate_font["hmtx"].metrics
            character_map = dict(candidate_font.getBestCmap() or {})
            glyph_order = list(candidate_font.getGlyphOrder())
            upm = int(candidate_font["head"].unitsPerEm)
        except Exception:
            candidate_font.close()
            raise
        base_font = candidate_font
    else:
        glyphs = {}
        metrics = {}
        character_map = {}
        glyph_order = [".notdef"]
        empty_pen = TTGlyphPen(None)
        glyphs[".notdef"] = empty_pen.glyph()
        metrics[".notdef"] = (1000, 50)
        upm = 1000

    skipped_existing = sorted(set(glyph_images) & existing_outlines)
    ordered_images = [
        (codepoint, image_path)
        for codepoint, image_path in sorted(glyph_images.items())
        if codepoint not in existing_outlines
    ]
    if len(glyph_order) + len(ordered_images) > 65535:
        if base_font is not None:
            base_font.close()
        raise ValueError("The font would exceed the TrueType 65,535 glyph limit")
    report = {
        "included": [],
        "failed": [],
        "skipped_existing": [
            {
                "codepoint": f"U+{codepoint:04X}",
                "glyph": character_map.get(codepoint, ""),
            }
            for codepoint in skipped_existing
        ],
        "metrics_profile": metrics_profile,
        "base_font": str(Path(base_font_path).resolve()) if base_font_path else "",
        "base_glyph_count": len(glyph_order) if base_font is not None else 0,
    }

    used_names = set(glyph_order)
    additions: dict[int, str] = {}
    for index, (codepoint, image_path) in enumerate(ordered_images, start=1):
        glyph_name = _glyph_name(codepoint, used_names)
        svg_path = svg_dir / f"U+{codepoint:04X}.svg"
        try:
            options = vectorize_options or VectorizeOptions()
            cache_path = svg_path.with_suffix(".source.json")
            digest = hashlib.sha256()
            digest.update(Path(image_path).read_bytes())
            digest.update(json.dumps(asdict(options), sort_keys=True).encode("utf-8"))
            source_hash = digest.hexdigest()
            if svg_path.is_file() and cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                cached = {}
            if cached.get("source_hash") == source_hash:
                qa = cached["qa"]
            else:
                qa = vectorize_png(image_path, svg_path, options)
                cache_path.write_text(
                    json.dumps({"source_hash": source_hash, "qa": qa}, indent=2),
                    encoding="utf-8",
                )
            bbox = qa.get("bbox") or [0, 0, 255, 255]
            base_scale = (upm * 0.9) / 256
            margin = upm * 0.05
            if codepoint < 0x0250 and metrics_profile == "monospace-2to1":
                advance = max(1, int(round(upm / 2)))
                side_margin = upm * 0.04
                x_scale = min(
                    (advance - 2 * side_margin)
                    / max(1, bbox[2] - bbox[0] + 1),
                    base_scale,
                )
                x_offset = side_margin - bbox[0] * x_scale
            elif codepoint < 0x0250:
                x_scale = base_scale
                ink_width = (bbox[2] - bbox[0] + 1) * x_scale
                advance = int(
                    max(upm * 0.3, min(upm * 0.9, ink_width + upm * 0.12))
                )
                x_offset = upm * 0.06 - bbox[0] * x_scale
            else:
                x_scale = base_scale
                advance = upm
                x_offset = margin
            transform = (
                x_scale,
                0,
                0,
                -base_scale,
                x_offset,
                upm * 0.85,
            )
            glyphs[glyph_name] = _glyph_from_svg(svg_path, transform)
            metrics[glyph_name] = (advance, max(0, int(x_offset)))
            character_map[codepoint] = glyph_name
            glyph_order.append(glyph_name)
            used_names.add(glyph_name)
            additions[codepoint] = glyph_name
            report["included"].append(
                {"codepoint": f"U+{codepoint:04X}", "glyph": glyph_name, "qa": qa}
            )
        except Exception as exc:
            report["failed"].append(
                {"codepoint": f"U+{codepoint:04X}", "error": f"{type(exc).__name__}: {exc}"}
            )
        if progress_callback is not None:
            progress_callback(index, len(ordered_images), codepoint)

    attributed = len(report["included"]) > 200
    if base_font is not None:
        base_font.setGlyphOrder(glyph_order)
        _update_cmap(base_font, additions)
        base_font["maxp"].numGlyphs = len(glyph_order)
        if "OS/2" in base_font:
            base_font["OS/2"].updateFirstAndLastCharIndex(base_font)
            base_font["OS/2"].recalcUnicodeRanges(base_font)
        if "DSIG" in base_font:
            del base_font["DSIG"]
        _update_base_font_names(base_font, metadata, attributed=attributed)
        try:
            base_font.save(output_path)
        finally:
            base_font.close()
    else:
        font = FontBuilder(1000, isTTF=True)
        font.setupGlyphOrder(glyph_order)
        font.setupCharacterMap(character_map)
        font.setupGlyf(glyphs)
        font.setupHorizontalMetrics(metrics)
        font.setupHorizontalHeader(ascent=880, descent=-120)
        font.setupOS2(
            sTypoAscender=880,
            sTypoDescender=-120,
            sTypoLineGap=0,
            usWinAscent=880,
            usWinDescent=120,
            sxHeight=500,
            sCapHeight=700,
        )
        description = metadata.license_description
        if attributed:
            description = f"{description}\n{ATTRIBUTION}\n{UPSTREAM_URL}".strip()
        family = metadata.family_name.strip()
        style = metadata.style_name.strip() or "Regular"
        font.setupNameTable(
            {
                "familyName": family,
                "styleName": style,
                "uniqueFontIdentifier": f"{family}-{style}-{metadata.version}",
                "fullName": f"{family} {style}".strip(),
                "psName": re.sub(r"[^A-Za-z0-9-]", "", f"{family}-{style}")[:63] or "Zi2ZiFont",
                "version": f"Version {metadata.version}",
                "copyright": metadata.copyright,
                "designer": metadata.designer,
                "description": ATTRIBUTION if attributed else "Generated with zi2zi-JiT",
                "licenseDescription": description,
                "licenseInfoURL": UPSTREAM_URL if attributed else "",
            }
        )
        font.setupPost()
        font.setupMaxp()
        font.setupHead()
        font.save(output_path)

    report.update(
        {
            "font_path": str(output_path),
            "glyph_count": len(glyph_order),
            "added_glyph_count": len(report["included"]),
            "skipped_existing_count": len(report["skipped_existing"]),
            "attribution_required": attributed,
            "attribution": ATTRIBUTION if attributed else "",
        }
    )
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html_report(report, output_path.with_suffix(".report.html"))
    return report


def _write_html_report(report: dict, output: Path) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(item['codepoint'])}</td><td>{html.escape(item['glyph'])}</td>"
        f"<td>{item['qa']['contours']}</td><td>{item['qa']['ink_ratio']}</td></tr>"
        for item in report["included"]
    )
    failures = "".join(
        f"<li>{html.escape(item['codepoint'])}: {html.escape(item['error'])}</li>"
        for item in report["failed"]
    )
    skipped = "".join(
        f"<li>{html.escape(item['codepoint'])}: retained "
        f"{html.escape(item.get('glyph', ''))}</li>"
        for item in report.get("skipped_existing", [])
    )
    base = html.escape(report.get("base_font", "")) or "None"
    output.write_text(
        "<!doctype html><meta charset='utf-8'><title>Font build report</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem}</style>"
        f"<h1>Font build report</h1><p>Glyphs: {report['glyph_count']}</p>"
        f"<p>Base font: {base}</p>"
        f"<p>Retained base glyphs: {report.get('base_glyph_count', 0)}; "
        f"added glyphs: {report.get('added_glyph_count', len(report['included']))}</p>"
        f"<p>Profile: {html.escape(report['metrics_profile'])}</p>"
        f"<h2>Failures</h2><ul>{failures or '<li>None</li>'}</ul>"
        f"<h2>Existing glyphs retained</h2><ul>{skipped or '<li>None</li>'}</ul>"
        f"<h2>Glyphs</h2><table><tr><th>Codepoint</th><th>Name</th><th>Contours</th>"
        f"<th>Ink ratio</th></tr>{rows}</table>",
        encoding="utf-8",
    )


def scan_glyph_directory(path: str | Path) -> dict[int, Path]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Generated glyph directory does not exist: {root}")
    candidates: list[tuple[int, str, int, Path]] = []
    for item in root.rglob("*"):
        if item.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        codepoint = codepoint_from_filename(item)
        if codepoint is not None:
            # Prefer the generator's canonical `generated` directory over
            # pairwise previews, then use a stable path order. User-reviewed
            # selections are applied separately and override this default.
            priority = 0 if item.parent.name.lower() == "generated" else 1
            candidates.append((priority, item.as_posix(), codepoint, item))
    result: dict[int, Path] = {}
    for _priority, _path_key, codepoint, item in sorted(candidates):
        result.setdefault(codepoint, item)
    return result


def build_export_package(
    font_path: str | Path,
    glyph_images: dict[int, str | Path],
    *,
    project_dir: str | Path | None = None,
) -> Path:
    """Bundle the installable draft and its editable/reproducibility artifacts."""
    font_path = Path(font_path).resolve()
    report_path = font_path.with_suffix(".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    archive_path = font_path.with_suffix(".zip")
    svg_dir = font_path.with_suffix("") / "svg"
    included = {item["codepoint"] for item in report.get("included", [])}
    base_note = (
        "The installable font retains the target TrueType font's original glyphs "
        "and OpenType tables, and adds only generated glyphs that were missing. "
        f"Base font: {report['base_font']}\n\n"
        if report.get("base_font")
        else ""
    )
    readme = (
        "zi2zi-JiT generated font draft\n\n"
        f"{base_note}"
        "This package contains a TrueType draft, editable SVG outlines, "
        "the selected generated PNG glyphs, build quality reports, and available "
        "generation/training manifests. Failed glyphs are listed in the report and "
        "were not replaced with source-font outlines.\n\n"
        f"Upstream: {UPSTREAM_URL}\n"
    )
    if report.get("attribution_required"):
        readme += f"\nRequired attribution: {ATTRIBUTION}\n"
    manifest = {
        "schema_version": 1,
        "font": font_path.name,
        "glyph_count": report.get("glyph_count"),
        "base_font": report.get("base_font", ""),
        "base_glyph_count": report.get("base_glyph_count", 0),
        "added_glyph_count": report.get("added_glyph_count", 0),
        "skipped_existing": report.get("skipped_existing", []),
        "attribution_required": report.get("attribution_required", False),
        "glyphs": [
            {
                "codepoint": f"U+{codepoint:04X}",
                "png": Path(path).name,
                "source_path": str(Path(path)),
            }
            for codepoint, path in sorted(glyph_images.items())
            if f"U+{codepoint:04X}" in included
        ],
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(font_path, f"font/{font_path.name}")
        for suffix in (".report.json", ".report.html"):
            artifact = font_path.with_suffix(suffix)
            if artifact.is_file():
                archive.write(artifact, f"reports/{artifact.name}")
        for codepoint, source in sorted(glyph_images.items()):
            if f"U+{codepoint:04X}" in included:
                archive.write(Path(source), f"png/U+{codepoint:04X}{Path(source).suffix.lower()}")
        if svg_dir.is_dir():
            for svg in sorted(svg_dir.glob("*.svg")):
                codepoint = codepoint_from_filename(svg)
                if (
                    codepoint is not None
                    and f"U+{codepoint:04X}" in included
                ):
                    archive.write(svg, f"svg/{svg.name}")
        if project_dir:
            project_root = Path(project_dir).resolve()
            artifacts = [
                *project_root.glob("generation/**/charset.json"),
                *project_root.glob("generation/**/*.json"),
                *project_root.glob("training/**/training-report.html"),
            ]
            for artifact in dict.fromkeys(item.resolve() for item in artifacts if item.is_file()):
                if project_root in artifact.parents:
                    archive.write(
                        artifact,
                        f"project-artifacts/{artifact.relative_to(project_root).as_posix()}",
                    )
        archive.writestr(
            "glyph-manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        archive.writestr("README.txt", readme)
    return archive_path
