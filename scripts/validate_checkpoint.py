#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from zi2zi_webui.checkpoints import write_checkpoint_sidecar
from zi2zi_webui.services import validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trusted local checkpoint and write its metadata sidecar."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--provenance-note", default="")
    args = parser.parse_args()

    metadata = validate_checkpoint(args.checkpoint)
    metadata["source"] = "trusted-shared-library"
    metadata["provenance_note"] = args.provenance_note
    sidecar = write_checkpoint_sidecar(args.checkpoint, metadata)
    print(
        json.dumps(
            {
                "checkpoint": metadata["path"],
                "architecture": metadata["architecture"],
                "file_size": metadata["file_size"],
                "sha256": metadata["sha256"],
                "sidecar": str(sidecar),
                "provenance_note": metadata["provenance_note"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
