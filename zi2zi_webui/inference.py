from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from data_processing.font_utils import GlyphRenderer, get_cjk_codepoints, load_font


def _style_image(path: str | Path, size: int = 128) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.width >= 512 and image.height >= 128:
        image = image.crop((512, 0, 640, 128))
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)


def select_reference(reference_paths: list[str], codepoint: int, seed: int) -> str:
    if not reference_paths:
        raise ValueError("At least one style reference is required")
    digest = hashlib.sha256(f"{codepoint}:{seed}".encode("ascii")).digest()
    return reference_paths[int.from_bytes(digest[:8], "big") % len(reference_paths)]


def build_inference_npz(
    codepoints: Iterable[int],
    source_font: str | Path,
    reference_paths: list[str],
    output_path: str | Path,
    *,
    seed: int = 42,
    resolution: int = 256,
) -> Path:
    codepoints = list(dict.fromkeys(int(value) for value in codepoints))
    if not codepoints:
        raise ValueError("No characters were selected")
    _, validated_font = load_font(str(source_font))
    renderer = GlyphRenderer(str(validated_font), resolution)

    valid: list[int] = []
    missing: list[int] = []
    contents: list[np.ndarray] = []
    styles: list[np.ndarray] = []
    references: list[str] = []
    for codepoint in codepoints:
        content = renderer.render(codepoint)
        if content is None or codepoint not in renderer._cmap:
            missing.append(codepoint)
            continue
        reference = select_reference(reference_paths, codepoint, seed)
        valid.append(codepoint)
        references.append(reference)
        contents.append(np.asarray(content, dtype=np.uint8).transpose(2, 0, 1))
        styles.append(_style_image(reference))
    if not valid:
        raise ValueError("The source font cannot render any selected character")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        font_labels=np.zeros(len(valid), dtype=np.int64),
        char_labels=np.zeros(len(valid), dtype=np.int64),
        unicode_labels=np.asarray(valid, dtype=np.int64),
        content_images=np.stack(contents),
        style_images=np.stack(styles),
        num_original_samples=np.int64(len(valid)),
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": seed,
                "source_font": str(validated_font),
                "count": len(valid),
                "missing": [f"U+{value:04X}" for value in missing],
                "samples": [
                    {"codepoint": f"U+{cp:04X}", "reference": ref}
                    for cp, ref in zip(valid, references)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def choose_diverse_references(paths: Iterable[str | Path], count: int = 8) -> list[str]:
    scored = []
    for item in paths:
        path = Path(item)
        try:
            gray = np.asarray(Image.open(path).convert("L").resize((128, 128)))
        except OSError:
            continue
        ink = gray < 220
        if not ink.any():
            continue
        ys, xs = np.where(ink)
        density = float(ink.mean())
        aspect = float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
        scored.append((density, aspect, str(path)))
    if len(scored) <= count:
        return [item[2] for item in sorted(scored)]
    scored.sort()
    selected = []
    for index in np.linspace(0, len(scored) - 1, count, dtype=int):
        selected.append(scored[index][2])
    return list(dict.fromkeys(selected))


def render_style_references(
    font_path: str | Path,
    output_dir: str | Path,
    *,
    count: int = 8,
    candidate_count: int = 96,
    resolution: int = 256,
) -> list[str]:
    """Render a deterministic, visually varied reference pool from a font."""
    font, validated_font = load_font(str(font_path))
    codepoints = sorted(get_cjk_codepoints(font))
    if len(codepoints) < count:
        raise ValueError(
            f"Target font needs at least {count} renderable CJK glyphs; found {len(codepoints)}"
        )
    sample_indexes = np.linspace(
        0, len(codepoints) - 1, min(candidate_count, len(codepoints)), dtype=int
    )
    renderer = GlyphRenderer(str(validated_font), resolution)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index in sample_indexes:
        codepoint = codepoints[int(index)]
        image = renderer.render(codepoint)
        if image is None:
            continue
        path = destination / f"U+{codepoint:04X}.png"
        image.save(path)
        rendered.append(path)
    selected = choose_diverse_references(rendered, count)
    if len(selected) < count:
        raise ValueError(
            f"Could only create {len(selected)} usable style references; {count} are required"
        )
    return selected
