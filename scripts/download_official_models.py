#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a verified-by-name official zi2zi-JiT model.")
    parser.add_argument("--folder-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("gdown is required. Recreate the environment from environment.yaml.") from exc

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zi2zi-model-") as temporary:
        downloaded = gdown.download_folder(
            url=args.folder_url,
            output=temporary,
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
        matches = [Path(item) for item in downloaded or [] if Path(item).name == args.expected]
        if not matches:
            raise SystemExit(f"Expected model was not found in official folder: {args.expected}")
        destination = output_dir / args.expected
        shutil.copy2(matches[0], destination)
        print(f"Downloaded {destination}", flush=True)


if __name__ == "__main__":
    main()
