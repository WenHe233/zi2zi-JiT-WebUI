from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .font_builder import codepoint_from_filename


REGION_ORDER = ("SC", "TC", "JP", "KR")
_CANDIDATE_RE = re.compile(r"_c(\d{2})(?:_|$)", re.IGNORECASE)


@dataclass(frozen=True)
class GenerationBatch:
    path: Path
    batch_id: str
    status: str
    regions: tuple[str, ...]
    expected_count: int
    generated_count: int
    legacy: bool
    modified_ns: int


def _candidate_index(path: Path) -> int:
    match = _CANDIDATE_RE.search(path.stem)
    return int(match.group(1)) if match else 0


def _image_candidates(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and codepoint_from_filename(path) is not None
        and path.parent.name.lower() not in {"compare", "pairs"}
    ]


def _request_batch(path: Path) -> GenerationBatch:
    plan_path = path / "generation-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    requested_regions = tuple(str(item) for item in (plan.get("regions") or {}))
    regions = requested_regions or tuple(
        region for region in REGION_ORDER if (path / region).is_dir()
    )
    expected = 0
    generated = 0
    statuses = []
    candidates = max(int(plan.get("candidates", 1)), 1)
    for region in regions:
        request_path = path / region / "request.json"
        result_path = path / region / "generation-result.json"
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            expected += int(request.get("count", 0)) * candidates
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            statuses.append("invalid")
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            generated += int(result.get("generated_count", 0))
            statuses.append(str(result.get("status", "invalid")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            statuses.append("pending")
    status = (
        "succeeded"
        if regions
        and all(value == "succeeded" for value in statuses)
        and generated == expected
        else "incomplete"
    )
    return GenerationBatch(
        path=path.resolve(),
        batch_id=str(plan.get("request_id") or path.name.removeprefix("request-")),
        status=status,
        regions=regions,
        expected_count=expected,
        generated_count=generated,
        legacy=False,
        modified_ns=max(
            [plan_path.stat().st_mtime_ns]
            + [
                item.stat().st_mtime_ns
                for item in path.glob("*/generation-result.json")
            ]
        ),
    )


def list_generation_batches(root: str | Path) -> list[GenerationBatch]:
    root = Path(root)
    if not root.is_dir():
        return []
    batches = []
    claimed_roots: set[Path] = set()
    for path in root.glob("request-*"):
        if not path.is_dir() or not (path / "generation-plan.json").is_file():
            continue
        try:
            batches.append(_request_batch(path))
            claimed_roots.add(path.resolve())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    for path in root.iterdir():
        if not path.is_dir() or path.resolve() in claimed_roots:
            continue
        images = _image_candidates(path)
        if not images:
            continue
        batches.append(
            GenerationBatch(
                path=path.resolve(),
                batch_id=path.name,
                status="legacy",
                regions=(),
                expected_count=len(images),
                generated_count=len(images),
                legacy=True,
                modified_ns=max(item.stat().st_mtime_ns for item in images),
            )
        )
    return sorted(batches, key=lambda item: item.modified_ns, reverse=True)


def preferred_generation_batch(
    batches: list[GenerationBatch],
) -> GenerationBatch | None:
    return next(
        (batch for batch in batches if batch.status == "succeeded"),
        batches[0] if batches else None,
    )


def batch_choices(
    root: str | Path,
) -> tuple[list[tuple[str, str]], str | None]:
    batches = list_generation_batches(root)
    choices = [
        (
            (
                f"{batch.batch_id} · {batch.status} · "
                f"{batch.generated_count}/{batch.expected_count}"
            ),
            str(batch.path),
        )
        for batch in batches
    ]
    preferred = preferred_generation_batch(batches)
    return choices, str(preferred.path) if preferred else None


def validate_batch_path(root: str | Path, selected: str | Path) -> Path:
    root = Path(root).resolve()
    selected = Path(selected).resolve()
    allowed = {batch.path for batch in list_generation_batches(root)}
    if selected not in allowed:
        raise ValueError("The selected generation batch is unavailable")
    return selected


def load_and_prune_selections(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid glyph selection manifest: {exc}") from exc
    cleaned = {
        str(label): str(image_path)
        for label, image_path in raw.items()
        if isinstance(label, str)
        and isinstance(image_path, str)
        and Path(image_path).is_file()
    }
    if cleaned != raw:
        path.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return cleaned


def _preferred_region_order(primary_region: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys([primary_region, *REGION_ORDER]))


def scan_generation_batch(
    batch_path: str | Path,
    *,
    primary_region: str = "SC",
    selections: dict[str, str] | None = None,
) -> dict[int, Path]:
    batch = Path(batch_path).resolve()
    result: dict[int, Path] = {}
    plan_path = batch / "generation-plan.json"
    if plan_path.is_file():
        for region in _preferred_region_order(primary_region):
            region_root = batch / region
            if not region_root.is_dir():
                continue
            grouped: dict[int, list[Path]] = {}
            for path in _image_candidates(region_root):
                codepoint = codepoint_from_filename(path)
                if codepoint is not None:
                    grouped.setdefault(codepoint, []).append(path)
            for codepoint, paths in grouped.items():
                chosen = min(
                    paths,
                    key=lambda path: (
                        _candidate_index(path),
                        -path.stat().st_mtime_ns,
                        path.as_posix(),
                    ),
                )
                result.setdefault(codepoint, chosen)
    else:
        for path in sorted(
            _image_candidates(batch),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        ):
            codepoint = codepoint_from_filename(path)
            if codepoint is not None:
                result.setdefault(codepoint, path)

    for label, selected_path in (selections or {}).items():
        try:
            codepoint = int(label.removeprefix("U+"), 16)
        except ValueError:
            continue
        candidate = Path(selected_path).resolve()
        if candidate.is_file() and batch in candidate.parents:
            result[codepoint] = candidate
    return result


def review_page(
    batch_path: str | Path,
    *,
    search: str = "",
    page: int = 1,
    page_size: int = 100,
    primary_region: str = "SC",
) -> tuple[list[Path], list[int], int, int]:
    glyphs = scan_generation_batch(
        batch_path,
        primary_region=primary_region,
    )
    query = search.strip().upper().removeprefix("U+")
    codepoints = sorted(glyphs)
    if query:
        codepoints = [
            codepoint
            for codepoint in codepoints
            if query in f"{codepoint:04X}" or search in chr(codepoint)
        ]
    total_pages = max((len(codepoints) + page_size - 1) // page_size, 1)
    page = min(max(int(page), 1), total_pages)
    selected = codepoints[(page - 1) * page_size : page * page_size]
    return [glyphs[codepoint] for codepoint in selected], selected, page, total_pages


def candidates_for_codepoint(
    batch_path: str | Path,
    codepoint: int,
) -> list[Path]:
    batch = Path(batch_path)
    paths = [
        path
        for path in _image_candidates(batch)
        if codepoint_from_filename(path) == codepoint
    ]
    return sorted(
        paths,
        key=lambda path: (
            _candidate_index(path),
            -path.stat().st_mtime_ns,
            path.as_posix(),
        ),
    )
