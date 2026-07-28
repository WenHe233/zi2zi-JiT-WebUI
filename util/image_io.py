from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imwrite_unicode(
    path: str | Path,
    image: np.ndarray,
    params: list[int] | None = None,
) -> Path:
    """Write an OpenCV image through imencode/tofile for Unicode Windows paths."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image, params or [])
    if not success or encoded.size == 0:
        raise OSError(f"OpenCV could not encode image for: {destination}")
    try:
        encoded.tofile(str(destination))
    except OSError as exc:
        raise OSError(f"Could not write image: {destination}") from exc
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"Image write produced no data: {destination}")
    return destination
