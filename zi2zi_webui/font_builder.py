from __future__ import annotations

import html
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import parse_path


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
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read image: {source}")
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
    cv2.imwrite(str(destination), 255 - bitmap)
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

        vtracer.convert_image_to_svg_py(
            str(processed),
            str(svg_path),
            colormode="binary",
            mode="spline",
        )
        qa["vectorizer"] = "vtracer"
    except ImportError:
        _opencv_svg(processed, svg_path)
        qa["vectorizer"] = "opencv-fallback"
    return qa


def _opencv_svg(source: str | Path, destination: str | Path) -> None:
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
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


def _glyph_from_svg(svg_path: Path, transform: tuple[float, float, float, float, float, float]):
    root = ET.parse(svg_path).getroot()
    pen = TTGlyphPen(None)
    quadratic = Cu2QuPen(pen, max_err=1.0, all_quadratic=True)
    transformed = TransformPen(quadratic, transform)
    found = False
    for element in root.iter():
        if element.tag.endswith("path") and element.attrib.get("d"):
            parse_path(element.attrib["d"], transformed)
            found = True
    if not found:
        raise ValueError(f"No SVG path in {svg_path}")
    return pen.glyph()


def build_ttf(
    glyph_images: dict[int, str | Path],
    output_path: str | Path,
    metadata: FontMetadata,
    *,
    metrics_profile: str = "proportional",
    vectorize_options: VectorizeOptions | None = None,
    svg_dir: str | Path | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict:
    if not metadata.family_name.strip():
        raise ValueError("Font family name is required")
    if len(glyph_images) + 1 > 65535:
        raise ValueError("The font would exceed the TrueType 65,535 glyph limit")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_dir = Path(svg_dir or output_path.with_suffix("")) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    glyphs = {}
    metrics = {}
    character_map = {}
    glyph_order = [".notdef"]
    empty_pen = TTGlyphPen(None)
    glyphs[".notdef"] = empty_pen.glyph()
    metrics[".notdef"] = (1000, 50)
    report = {"included": [], "failed": [], "metrics_profile": metrics_profile}

    ordered_images = sorted(glyph_images.items())
    for index, (codepoint, image_path) in enumerate(ordered_images, start=1):
        glyph_name = f"uni{codepoint:04X}" if codepoint <= 0xFFFF else f"u{codepoint:06X}"
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
            base_scale = 900 / 256
            if codepoint < 0x0250 and metrics_profile == "monospace-2to1":
                x_scale = min((500 - 80) / max(1, bbox[2] - bbox[0] + 1), base_scale)
                advance = 500
                x_offset = 40 - bbox[0] * x_scale
            elif codepoint < 0x0250:
                x_scale = base_scale
                ink_width = (bbox[2] - bbox[0] + 1) * x_scale
                advance = int(max(300, min(900, ink_width + 120)))
                x_offset = 60 - bbox[0] * x_scale
            else:
                x_scale = base_scale
                advance = 1000
                x_offset = 50
            transform = (x_scale, 0, 0, -base_scale, x_offset, 850)
            glyphs[glyph_name] = _glyph_from_svg(svg_path, transform)
            metrics[glyph_name] = (advance, max(0, int(x_offset)))
            character_map[codepoint] = glyph_name
            glyph_order.append(glyph_name)
            report["included"].append(
                {"codepoint": f"U+{codepoint:04X}", "glyph": glyph_name, "qa": qa}
            )
        except Exception as exc:
            report["failed"].append(
                {"codepoint": f"U+{codepoint:04X}", "error": f"{type(exc).__name__}: {exc}"}
            )
        if progress_callback is not None:
            progress_callback(index, len(ordered_images), codepoint)

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
    attributed = len(report["included"]) > 200
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
    output.write_text(
        "<!doctype html><meta charset='utf-8'><title>Font build report</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem}</style>"
        f"<h1>Font build report</h1><p>Glyphs: {report['glyph_count']}</p>"
        f"<p>Profile: {html.escape(report['metrics_profile'])}</p>"
        f"<h2>Failures</h2><ul>{failures or '<li>None</li>'}</ul>"
        f"<h2>Glyphs</h2><table><tr><th>Codepoint</th><th>Name</th><th>Contours</th>"
        f"<th>Ink ratio</th></tr>{rows}</table>",
        encoding="utf-8",
    )


def scan_glyph_directory(path: str | Path) -> dict[int, Path]:
    result = {}
    for item in Path(path).iterdir():
        if item.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        codepoint = codepoint_from_filename(item)
        if codepoint is not None:
            result[codepoint] = item
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
    readme = (
        "zi2zi-JiT generated font draft\n\n"
        "This package contains an unhinted TrueType draft, editable SVG outlines, "
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
