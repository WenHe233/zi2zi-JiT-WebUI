#!/usr/bin/env python3
"""Generate FontSrcTarget train and/or test dataset from target TTF/OTF fonts.

By default generates both train/ and test/ under --output-dir.

Usage:
    # Both train and test
    python scripts/generate_font_dataset.py \
        --source-font /path/to/source.ttf \
        --font-dir /path/to/fonts/ \
        --output-dir /path/to/dataset

    # Train only
    python scripts/generate_font_dataset.py \
        --source-font /path/to/source.ttf \
        --font-dir /path/to/fonts/ \
        --output-dir /path/to/dataset \
        --train-only

    # Test only (requires existing train dir)
    python scripts/generate_font_dataset.py \
        --source-font /path/to/source.ttf \
        --font-dir /path/to/fonts/ \
        --output-dir /path/to/dataset \
        --test-only --train-dir /path/to/existing_train
"""
import argparse
import json
import logging
from pathlib import Path

from data_processing.pipeline import generate_train_dataset, generate_test_dataset, create_test_npz


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FontSrcTarget dataset (train + test by default)."
    )
    parser.add_argument(
        "--source-font",
        required=True,
        nargs="+",
        help="Ordered source/reference font paths; later fonts provide missing glyphs.",
    )
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument(
        "--font-dir",
        help="Directory containing target fonts (legacy multi-font mode).",
    )
    targets.add_argument(
        "--target-font",
        action="append",
        help="Exact target font to process. Repeat for multiple explicit fonts.",
    )
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory. Train goes to <output-dir>/train, test to <output-dir>/test.")

    # Mode flags
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="Generate train dataset only.")
    mode.add_argument("--test-only", action="store_true", help="Generate test dataset only (requires --train-dir).")

    # Train args
    parser.add_argument("--num-fonts", type=int, default=None, help="Number of target fonts to process.")
    parser.add_argument("--start-index", type=int, default=1, help="Starting font index for folder names.")
    parser.add_argument("--train-chars-per-font", type=int, default=500,
                        help="Characters per font for training (default: 500).")
    parser.add_argument("--train-seed", type=int, default=42, help="Random seed for training (default: 42).")

    # Test args
    parser.add_argument("--train-dir", type=str, default=None,
                        help="Existing train directory (required for --test-only, ignored otherwise).")
    parser.add_argument("--test-chars-per-font", type=int, default=8,
                        help="Unseen characters per font for testing (default: 8).")
    parser.add_argument("--test-seed", type=int, default=99999, help="Random seed for testing (default: 99999).")

    # Shared
    parser.add_argument("--charset", type=str, default="auto",
                        choices=["auto", "gb2312", "gbk", "big5", "jisx0208", "ksx1001", "all-cjk"],
                        help=(
                            "Optional range filter. 'auto' detects all outlined CJK glyphs "
                            "shared by the ordered source-font collection and target font."
                        ))
    parser.add_argument("--resolution", type=int, default=256, help="Glyph resolution (default: 256).")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers for font processing (default: 4).")
    parser.add_argument(
        "--build-marker",
        default=None,
        help="Optional WebUI dataset build-state marker.",
    )

    args = parser.parse_args()

    if args.test_only and args.train_dir is None:
        parser.error("--test-only requires --train-dir")

    if args.test_only and not Path(args.train_dir).is_dir():
        raise FileNotFoundError(f"--train-dir does not exist: {args.train_dir}")

    return args


def write_build_marker(path: str | None, status: str, **details) -> None:
    if not path:
        return
    marker = Path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "status": status, **details}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(marker)


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label} generation completed: "
          f"total={summary['total']} success={summary['success']} failed={summary['failed']}")
    for result in summary["results"]:
        tag = "OK" if result.get("success") else "SKIP"
        font_file = result.get("font_file", "<unknown>")
        info = result.get("error", f"extracted={result.get('extracted', 0)} failed={result.get('failed', 0)}")
        print(f"  [{tag}] {font_file}: {info}")


def emit_progress(current: int, total: int, message: str) -> None:
    print(
        "WEBUI_PROGRESS "
        + json.dumps(
            {"current": current, "total": total, "message": message},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(args.output_dir)
    write_build_marker(args.build_marker, "building")
    do_train = not args.test_only
    do_test = not args.train_only
    target_fonts = (
        [Path(item).resolve() for item in args.target_font]
        if args.target_font
        else None
    )
    font_dir = Path(args.font_dir).resolve() if args.font_dir else None

    train_dir = Path(args.train_dir) if args.train_dir else output_dir / "train"
    total_stages = int(do_train) + int(do_test) * 2
    completed_stages = 0
    emit_progress(completed_stages, total_stages, "Preparing dataset build")

    if do_train:
        train_out = output_dir / "train"
        print(f"=== Generating train dataset -> {train_out} ===")
        train_summary = generate_train_dataset(
            source_font=[Path(item) for item in args.source_font],
            font_dir=font_dir,
            output_dir=train_out,
            target_fonts=target_fonts,
            num_fonts=args.num_fonts,
            chars_per_font=args.train_chars_per_font,
            charset=args.charset,
            resolution=args.resolution,
            seed=args.train_seed,
            start_index=args.start_index,
            num_workers=args.num_workers,
        )
        print_summary("Train", train_summary)
        if train_summary["failed"] or train_summary["success"] == 0:
            raise RuntimeError("Training dataset generation failed")
        if any(
            result.get("extracted") != args.train_chars_per_font
            for result in train_summary["results"]
        ):
            raise RuntimeError(
                "Training dataset did not produce the requested number of glyphs"
            )
        train_dir = train_out
        completed_stages += 1
        emit_progress(completed_stages, total_stages, "Training split complete")

    if do_test:
        test_out = output_dir / "test"
        print(f"\n=== Generating test dataset -> {test_out} ===")
        test_summary = generate_test_dataset(
            source_font=[Path(item) for item in args.source_font],
            font_dir=font_dir,
            train_dir=train_dir,
            output_dir=test_out,
            target_fonts=target_fonts,
            chars_per_font=args.test_chars_per_font,
            charset=args.charset,
            resolution=args.resolution,
            seed=args.test_seed,
            num_workers=args.num_workers,
        )
        print_summary("Test", test_summary)
        if test_summary["failed"] or test_summary["success"] == 0:
            raise RuntimeError("Validation dataset generation failed")
        if any(
            result.get("extracted") != args.test_chars_per_font
            for result in test_summary["results"]
        ):
            raise RuntimeError(
                "Validation dataset did not produce the requested number of glyphs"
            )
        completed_stages += 1
        emit_progress(completed_stages, total_stages, "Validation split complete")

        # Convert test set to NPZ
        npz_path = output_dir / "test.npz"
        print(f"\n=== Creating test NPZ -> {npz_path} ===")
        result = create_test_npz(test_out, npz_path)
        print(f"  {result['samples']} samples ({result['file_size_mb']:.1f} MB)")
        expected = args.test_chars_per_font * test_summary["success"]
        if result["samples"] != expected or not npz_path.is_file():
            raise RuntimeError(
                f"Validation NPZ contains {result['samples']} of {expected} expected samples"
            )
        completed_stages += 1
        emit_progress(completed_stages, total_stages, "Validation NPZ complete")
    write_build_marker(
        args.build_marker,
        "complete",
        train_count=args.train_chars_per_font if do_train else None,
        test_count=args.test_chars_per_font if do_test else None,
    )
    emit_progress(total_stages, total_stages, "Dataset build complete")


if __name__ == "__main__":
    parsed_args = get_args()
    try:
        main(parsed_args)
    except BaseException as exc:
        write_build_marker(
            parsed_args.build_marker,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
