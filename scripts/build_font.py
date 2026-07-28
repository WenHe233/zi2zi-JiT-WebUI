#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from zi2zi_webui.font_builder import (
    FontMetadata,
    VectorizeOptions,
    build_export_package,
    build_ttf,
    scan_glyph_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize zi2zi-JiT PNG glyphs into a TTF.")
    parser.add_argument("--glyph-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--style", default="Regular")
    parser.add_argument("--version", default="1.000")
    parser.add_argument("--designer", default="")
    parser.add_argument("--copyright", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--profile", choices=["proportional", "monospace-2to1"], default="proportional")
    parser.add_argument("--threshold", type=int, default=180)
    parser.add_argument("--despeckle-area", type=int, default=4)
    parser.add_argument("--selection-manifest", default="")
    parser.add_argument("--project-dir", default="")
    parser.add_argument(
        "--base-font",
        default="",
        help="Static TrueType target font whose existing glyphs and tables are retained.",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if args.project_dir:
        fonts_root = (Path(args.project_dir).resolve() / "fonts").resolve()
        if output.parent != fonts_root:
            raise SystemExit(
                f"--output must be directly inside the project fonts directory: {fonts_root}"
            )
    glyphs = scan_glyph_directory(args.glyph_dir)
    if args.selection_manifest:
        selections = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
        for label, path in selections.items():
            glyphs[int(label.removeprefix("U+"), 16)] = Path(path)
    if not glyphs:
        raise SystemExit(
            "No U+XXXX-named glyph images were found recursively under "
            f"{Path(args.glyph_dir).resolve()}. Refusing to export an unchanged base font."
        )
    report = build_ttf(
        glyphs,
        output,
        FontMetadata(
            family_name=args.family,
            style_name=args.style,
            version=args.version,
            designer=args.designer,
            copyright=args.copyright,
            license_description=args.license,
        ),
        metrics_profile=args.profile,
        vectorize_options=VectorizeOptions(
            threshold=args.threshold,
            despeckle_area=args.despeckle_area,
        ),
        progress_callback=lambda current, total, codepoint: print(
            "WEBUI_PROGRESS "
            + json.dumps(
                {
                    "current": current,
                    "total": total,
                    "message": f"Vectorizing U+{codepoint:04X} ({current}/{total})",
                }
            ),
            flush=True,
        ),
        base_font_path=args.base_font or None,
    )
    archive = build_export_package(
        output,
        glyphs,
        project_dir=args.project_dir or None,
    )
    report["export_package"] = str(archive)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
