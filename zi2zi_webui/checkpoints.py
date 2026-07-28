from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_ARCHITECTURES = {"JiT-B/16", "JiT-L/16"}


def checkpoint_sidecar_path(path: str | Path) -> Path:
    checkpoint = Path(path)
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_checkpoint_metadata(
    path: str | Path,
    *,
    architecture: str,
    num_fonts: int | None,
    num_chars: int | None,
    img_size: int | None = None,
    lora: bool,
    state_keys: int,
    source: str,
    trusted_required: bool,
    provenance_note: str = "",
) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if architecture not in CHECKPOINT_ARCHITECTURES:
        raise ValueError(f"Unsupported or unknown checkpoint architecture: {architecture}")
    stat = checkpoint.stat()
    return {
        "schema_version": 1,
        "path": str(checkpoint),
        "filename": checkpoint.name,
        "architecture": architecture,
        "model": architecture,
        "img_size": img_size,
        "num_fonts": num_fonts,
        "num_chars": num_chars,
        "lora": bool(lora),
        "state_keys": int(state_keys),
        "file_size": stat.st_size,
        "sha256": sha256_file(checkpoint),
        "source": source,
        "trusted_required": bool(trusted_required),
        "provenance_note": provenance_note,
    }


def write_checkpoint_sidecar(
    path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    sidecar = checkpoint_sidecar_path(path)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(sidecar)
    return sidecar


def read_checkpoint_sidecar(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    sidecar = checkpoint_sidecar_path(checkpoint)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Checkpoint metadata sidecar is missing: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint metadata sidecar: {sidecar}") from exc
    architecture = str(
        metadata.get("architecture") or metadata.get("model") or ""
    )
    if architecture not in CHECKPOINT_ARCHITECTURES:
        raise ValueError(f"Checkpoint sidecar has unknown architecture: {architecture}")
    if int(metadata.get("file_size", -1)) != checkpoint.stat().st_size:
        raise ValueError(
            "Checkpoint size no longer matches its metadata; validate it again"
        )
    digest = str(metadata.get("sha256") or "")
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest.lower()
    ):
        raise ValueError("Checkpoint sidecar does not contain a valid SHA-256")
    metadata["architecture"] = architecture
    metadata["model"] = architecture
    return metadata
