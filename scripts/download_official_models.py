#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a named model from the official zi2zi-JiT folder."
    )
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
        from zi2zi_webui.checkpoints import write_checkpoint_sidecar
        from zi2zi_webui.services import validate_checkpoint

        metadata = validate_checkpoint(destination)
        metadata["source"] = "official-folder-download"
        metadata["provenance_note"] = (
            "SHA-256 was calculated after download for local auditing only; "
            "no trusted upstream digest was available for authenticity verification."
        )
        sidecar = write_checkpoint_sidecar(destination, metadata)
        print(
            f"Downloaded {destination}; audit metadata: {sidecar}; "
            f"SHA-256: {metadata['sha256']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
