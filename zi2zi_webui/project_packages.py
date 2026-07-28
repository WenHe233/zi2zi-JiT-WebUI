from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .models import ProjectManifest, TrainingRun, utc_now
from .storage import PROJECT_FOLDERS, Storage


PACKAGE_FORMAT = "zi2zi-jit-project"
PACKAGE_SCHEMA_VERSION = 1
PROJECT_TOKEN = "${PROJECT}/"
PROJECT_ROOT_TOKEN = "${PROJECT}"
MODEL_TOKEN = "${MODEL}/"
EXTERNAL_TOKEN = "${EXTERNAL}/"
MAX_PACKAGE_FILES = 200_000
MAX_UNCOMPRESSED_BYTES = 500 * 1024**3
MAX_METADATA_BYTES = 16 * 1024**2
ProgressCallback = Callable[[int, int, str], None]


def _noop_progress(_current: int, _total: int, _message: str) -> None:
    return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"Unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    if re.match(r"^[A-Za-z]:", path.parts[0]):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    normalized = path.as_posix()
    if normalized != name.rstrip("/"):
        raise ValueError(f"Non-canonical archive member path: {name!r}")
    return normalized


def _safe_output_path(root: Path, relative: str) -> Path:
    relative_path = PurePosixPath(_safe_archive_name(relative))
    destination = (root / Path(*relative_path.parts)).resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Archive path escaped the import root: {relative}")
    return destination


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _read_json_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    maximum: int = MAX_METADATA_BYTES,
) -> Any:
    info = archive.getinfo(name)
    if info.file_size > maximum:
        raise ValueError(f"Package metadata is too large: {name}")
    with archive.open(info) as handle:
        return json.loads(handle.read(maximum + 1).decode("utf-8"))


def _write_bytes(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
    checksums: dict[str, str],
) -> None:
    safe_name = _safe_archive_name(name)
    archive.writestr(safe_name, payload)
    checksums[safe_name] = hashlib.sha256(payload).hexdigest()


def _write_file(
    archive: zipfile.ZipFile,
    name: str,
    source: Path,
    checksums: dict[str, str],
) -> None:
    safe_name = _safe_archive_name(name)
    digest = hashlib.sha256()
    with source.open("rb") as input_handle, archive.open(
        safe_name,
        "w",
        force_zip64=True,
    ) as output_handle:
        for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
            digest.update(chunk)
            output_handle.write(chunk)
    checksums[safe_name] = digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _is_checkpoint_sidecar(path: PurePosixPath) -> bool:
    lowered = path.name.lower()
    return lowered.endswith((".pth.json", ".pt.json", ".ckpt.json"))


def _lightweight_file(relative: PurePosixPath) -> bool:
    lowered = relative.as_posix().lower()
    suffix = relative.suffix.lower()
    if lowered.startswith("training/"):
        return relative.name in {"metrics.jsonl", "training-report.html"} or suffix in {
            ".csv",
            ".json",
            ".html",
        }
    if lowered.startswith(("generation/", "glyphs/")):
        return suffix in {".json", ".csv", ".txt", ".md", ".html"}
    if lowered.startswith(("fonts/", "exports/")):
        return suffix in {".json", ".csv", ".txt", ".md", ".html"}
    return False


def _project_files(project_root: Path, mode: str) -> list[tuple[PurePosixPath, Path]]:
    values: list[tuple[PurePosixPath, Path]] = []
    for path in sorted(project_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Project packages do not support symbolic links: {path}")
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(project_root).as_posix())
        if relative == PurePosixPath("project.json"):
            continue
        if path.name.endswith((".tmp", ".part")):
            continue
        if mode == "lightweight" and not _lightweight_file(relative):
            continue
        values.append((relative, path))
    return values


def _model_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        return {}
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _collect_models(
    manifest: ProjectManifest,
    project_root: Path,
    include_shared_models: bool,
) -> tuple[list[dict[str, Any]], dict[Path, str]]:
    models: list[dict[str, Any]] = []
    path_to_key: dict[Path, str] = {}
    digest_to_key: dict[str, str] = {}
    for raw_path in (manifest.base_model, manifest.active_checkpoint):
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if path == project_root or project_root in path.parents:
            continue
        if path in path_to_key:
            continue
        if not path.is_file():
            key = f"missing-{uuid4().hex}"
            models.append(
                {
                    "key": key,
                    "filename": path.name,
                    "sha256": "",
                    "file_size": 0,
                    "included": False,
                    "available_at_export": False,
                }
            )
            path_to_key[path] = key
            continue
        digest = sha256_file(path)
        if digest in digest_to_key:
            path_to_key[path] = digest_to_key[digest]
            continue
        sidecar = _model_sidecar(path)
        key = digest
        digest_to_key[digest] = key
        entry = {
            "key": key,
            "filename": path.name,
            "sha256": digest,
            "file_size": path.stat().st_size,
            "included": bool(include_shared_models),
            "available_at_export": True,
            "architecture": sidecar.get("architecture", ""),
            "num_fonts": sidecar.get("num_fonts"),
            "num_chars": sidecar.get("num_chars"),
        }
        if include_shared_models:
            entry["package_path"] = f"shared-models/{digest}/{path.name}"
        models.append(entry)
        path_to_key[path] = key
    return models, path_to_key


def _normalize_value(
    value: Any,
    project_root: Path,
    model_paths: dict[Path, str],
    external_references: set[str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_value(item, project_root, model_paths, external_references)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_value(item, project_root, model_paths, external_references)
            for item in value
        ]
    if not isinstance(value, str) or not value:
        return value
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            return value
        resolved = path.resolve()
    except (OSError, ValueError):
        return value
    try:
        relative = resolved.relative_to(project_root)
        if relative == Path("."):
            return PROJECT_ROOT_TOKEN
        return PROJECT_TOKEN + PurePosixPath(relative.as_posix()).as_posix()
    except ValueError:
        pass
    if resolved in model_paths:
        return MODEL_TOKEN + model_paths[resolved]
    reference = resolved.name or "external-path"
    external_references.add(reference)
    return EXTERNAL_TOKEN + reference


def _export_job_records(storage: Storage, project_id: str) -> list[dict[str, Any]]:
    with storage._connect() as db:
        rows = db.execute(
            "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at",
            (project_id,),
        ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        try:
            record["command"] = json.loads(record.pop("command_json"))
        except (TypeError, ValueError, json.JSONDecodeError):
            record["command"] = []
            record.pop("command_json", None)
        record.pop("env_json", None)
        record["env"] = {}
        record["pid"] = None
        records.append(record)
    return records


def export_project_package(
    storage: Storage,
    project_id: str,
    output_path: str | Path,
    *,
    mode: str = "lightweight",
    include_shared_models: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if mode not in {"lightweight", "full"}:
        raise ValueError("Project export mode must be 'lightweight' or 'full'")
    callback = progress or _noop_progress
    manifest = storage.get_project(project_id)
    with storage._connect() as db:
        active = db.execute(
            """
            SELECT id FROM jobs
            WHERE project_id=? AND status IN ('queued', 'running')
            ORDER BY created_at
            """,
            (project_id,),
        ).fetchall()
    if active:
        raise ValueError(
            "Cancel active project jobs before export: "
            + ", ".join(row["id"] for row in active)
        )

    project_root = storage.project_dir(project_id)
    project_files = _project_files(project_root, mode)
    models, model_paths = _collect_models(
        manifest,
        project_root,
        include_shared_models,
    )
    external_references: set[str] = set()
    normalized_manifest = _normalize_value(
        manifest.to_dict(),
        project_root,
        model_paths,
        external_references,
    )
    normalized_runs = _normalize_value(
        storage.list_training_runs(project_id),
        project_root,
        model_paths,
        external_references,
    )
    normalized_jobs = _normalize_value(
        _export_job_records(storage, project_id),
        project_root,
        model_paths,
        external_references,
    )

    included_models = [
        (entry, next(path for path, key in model_paths.items() if key == entry["key"]))
        for entry in models
        if entry.get("included")
    ]
    payload_bytes = sum(path.stat().st_size for _relative, path in project_files)
    payload_bytes += sum(path.stat().st_size for _entry, path in included_models)
    metadata = {
        "format": PACKAGE_FORMAT,
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "export_mode": mode,
        "source_project_id": project_id,
        "project_name": manifest.name,
        "include_shared_models": bool(include_shared_models),
        "payload_file_count": len(project_files) + len(included_models),
        "payload_bytes": payload_bytes,
        "training_run_count": len(normalized_runs),
        "job_count": len(normalized_jobs),
        "models": models,
        "external_references": sorted(external_references),
        "warnings": [
            "Package checksums detect corruption but do not prove publisher authenticity.",
            "Imported PyTorch checkpoints must be revalidated before use.",
            "Confirm redistribution rights for fonts, models, and training assets.",
        ],
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{uuid4().hex}.tmp")
    total_items = 5 + len(project_files) + len(included_models)
    completed = 0
    checksums: dict[str, str] = {}
    callback(completed, total_items, "Preparing project package")
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, value in (
                ("package.json", metadata),
                ("records/project.json", normalized_manifest),
                ("records/training-runs.json", normalized_runs),
                ("records/jobs.json", normalized_jobs),
            ):
                _write_bytes(archive, name, _json_bytes(value), checksums)
                completed += 1
                callback(completed, total_items, f"Writing {name}")
            readme = (
                "zi2zi-JiT project package\n"
                "Checksums verify transport integrity, not publisher authenticity.\n"
                "Checkpoint files are not loaded during import and require validation before use.\n\n"
                "zi2zi-JiT 项目包\n"
                "校验和仅用于检查传输完整性，不能证明发布者身份。\n"
                "导入时不会加载 checkpoint，使用前必须重新校验。\n"
            ).encode("utf-8")
            _write_bytes(archive, "README.txt", readme, checksums)
            completed += 1
            callback(completed, total_items, "Writing package README")
            for relative, source in project_files:
                name = f"project/{relative.as_posix()}"
                _write_file(archive, name, source, checksums)
                completed += 1
                callback(completed, total_items, f"Packing {relative.as_posix()}")
            for entry, source in included_models:
                _write_file(archive, entry["package_path"], source, checksums)
                completed += 1
                callback(completed, total_items, f"Packing model {source.name}")
            checksum_payload = "".join(
                f"{digest}  {name}\n"
                for name, digest in sorted(checksums.items())
            ).encode("utf-8")
            archive.writestr("checksums.sha256", checksum_payload)
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    callback(total_items, total_items, "Project package complete")
    return {
        **metadata,
        "package_path": str(output),
        "package_size": output.stat().st_size,
        "checksum_file_count": len(checksums),
    }


def _validated_archive(
    package_path: str | Path,
) -> tuple[zipfile.ZipFile, list[zipfile.ZipInfo]]:
    package = Path(package_path).expanduser().resolve()
    if not package.is_file():
        raise FileNotFoundError(package)
    try:
        archive = zipfile.ZipFile(package, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a valid ZIP project package") from exc
    infos = [info for info in archive.infolist() if not info.is_dir()]
    if len(infos) > MAX_PACKAGE_FILES:
        archive.close()
        raise ValueError(f"Project package contains too many files: {len(infos)}")
    names: set[str] = set()
    total_size = 0
    try:
        for info in infos:
            name = _safe_archive_name(info.filename)
            if name in names:
                raise ValueError(f"Duplicate archive member: {name}")
            names.add(name)
            if _is_zip_symlink(info):
                raise ValueError(f"Symbolic links are not allowed in project packages: {name}")
            total_size += info.file_size
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Project package is too large after decompression")
            if (
                info.file_size > 100 * 1024**2
                and info.compress_size > 0
                and info.file_size / info.compress_size > 10_000
            ):
                raise ValueError(f"Suspicious compression ratio in archive member: {name}")
        required = {
            "package.json",
            "records/project.json",
            "records/training-runs.json",
            "records/jobs.json",
            "checksums.sha256",
        }
        missing = required - names
        if missing:
            raise ValueError(
                "Project package is missing required files: " + ", ".join(sorted(missing))
            )
    except BaseException:
        archive.close()
        raise
    return archive, infos


def inspect_project_package(
    package_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    callback = progress or _noop_progress
    archive, infos = _validated_archive(package_path)
    try:
        metadata = _read_json_member(archive, "package.json")
        if not isinstance(metadata, dict):
            raise ValueError("package.json must contain an object")
        if metadata.get("format") != PACKAGE_FORMAT:
            raise ValueError("Unsupported project package format")
        if metadata.get("schema_version") != PACKAGE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project package schema: {metadata.get('schema_version')}"
            )
        checksum_info = archive.getinfo("checksums.sha256")
        if checksum_info.file_size > MAX_METADATA_BYTES:
            raise ValueError("checksums.sha256 is too large")
        raw_checksums = archive.read("checksums.sha256").decode("utf-8")
        expected: dict[str, str] = {}
        for line in raw_checksums.splitlines():
            if not line.strip():
                continue
            try:
                digest, name = line.split("  ", 1)
            except ValueError as exc:
                raise ValueError("Malformed checksums.sha256") from exc
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid SHA-256 digest for {name}")
            safe_name = _safe_archive_name(name)
            if safe_name in expected:
                raise ValueError(f"Duplicate checksum entry: {safe_name}")
            expected[safe_name] = digest
        actual_names = {
            info.filename
            for info in infos
            if info.filename != "checksums.sha256"
        }
        if set(expected) != actual_names:
            missing = actual_names - set(expected)
            extra = set(expected) - actual_names
            raise ValueError(
                "Checksum manifest does not match package members"
                f"; missing={sorted(missing)}; extra={sorted(extra)}"
            )
        total = max(len(actual_names), 1)
        for index, name in enumerate(sorted(actual_names), start=1):
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected[name]:
                raise ValueError(f"Checksum mismatch: {name}")
            callback(index, total, f"Verifying {name}")
        project = _read_json_member(archive, "records/project.json")
        runs = _read_json_member(archive, "records/training-runs.json")
        jobs = _read_json_member(archive, "records/jobs.json")
        if not isinstance(project, dict) or not isinstance(runs, list) or not isinstance(jobs, list):
            raise ValueError("Project package record files have invalid JSON types")
        if not all(isinstance(item, dict) and item.get("id") for item in runs):
            raise ValueError("Every imported training run must be an object with an id")
        if not all(isinstance(item, dict) and item.get("id") for item in jobs):
            raise ValueError("Every imported job must be an object with an id")
        run_ids = [str(item["id"]) for item in runs]
        job_ids = [str(item["id"]) for item in jobs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("Project package contains duplicate training run ids")
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("Project package contains duplicate job ids")
        return {
            **metadata,
            "package_path": str(Path(package_path).resolve()),
            "package_size": Path(package_path).stat().st_size,
            "archive_file_count": len(infos),
            "uncompressed_bytes": sum(info.file_size for info in infos),
            "checksum_verified": True,
            "training_run_count": len(runs),
            "job_count": len(jobs),
        }
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"Project package is corrupted: {exc}") from exc
    finally:
        archive.close()


def _remap_relative(
    relative: str,
    run_ids: dict[str, str],
    job_ids: dict[str, str],
) -> PurePosixPath:
    path = PurePosixPath(_safe_archive_name(relative))
    parts = list(path.parts)
    if len(parts) >= 2 and parts[0] == "training" and parts[1] in run_ids:
        parts[1] = run_ids[parts[1]]
    if len(parts) >= 2 and parts[0] == "logs":
        stem = PurePosixPath(parts[-1]).stem
        suffix = PurePosixPath(parts[-1]).suffix
        if stem in job_ids:
            parts[-1] = job_ids[stem] + suffix
    remapped = PurePosixPath(*parts)
    if _is_checkpoint_sidecar(remapped):
        remapped = remapped.with_name(
            remapped.name.removesuffix(".json") + ".imported-metadata.json"
        )
    return remapped


def _restore_value(
    value: Any,
    project_root: Path,
    run_ids: dict[str, str],
    job_ids: dict[str, str],
    model_paths: dict[str, str],
    missing_models: set[str],
    missing_external: set[str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _restore_value(
                item,
                project_root,
                run_ids,
                job_ids,
                model_paths,
                missing_models,
                missing_external,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _restore_value(
                item,
                project_root,
                run_ids,
                job_ids,
                model_paths,
                missing_models,
                missing_external,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    if value == PROJECT_ROOT_TOKEN:
        return str(project_root)
    if value.startswith(PROJECT_TOKEN):
        relative = _remap_relative(
            value.removeprefix(PROJECT_TOKEN),
            run_ids,
            job_ids,
        )
        return str(_safe_output_path(project_root, relative.as_posix()))
    if value.startswith(MODEL_TOKEN):
        key = value.removeprefix(MODEL_TOKEN)
        resolved = model_paths.get(key, "")
        if not resolved:
            missing_models.add(key)
        return resolved
    if value.startswith(EXTERNAL_TOKEN):
        missing_external.add(value.removeprefix(EXTERNAL_TOKEN))
        return ""
    return value


def _find_existing_model(storage: Storage, entry: dict[str, Any]) -> Path | None:
    digest = str(entry.get("sha256") or "")
    filename = str(entry.get("filename") or "")
    if not digest:
        return None
    for path in storage.models_dir.iterdir():
        if not path.is_file() or path.name.endswith(".json"):
            continue
        sidecar = _model_sidecar(path)
        if sidecar.get("sha256") == digest:
            return path.resolve()
    candidate = storage.models_dir / filename
    if candidate.is_file() and sha256_file(candidate) == digest:
        return candidate.resolve()
    return None


def _prune_unavailable_manifest_paths(
    value: dict[str, Any],
    final_project: Path,
    staged_project: Path,
    planned_models: set[str],
) -> list[str]:
    missing: list[str] = []

    def available(raw_path: Any) -> bool:
        text = str(raw_path or "")
        if not text:
            return False
        path = Path(text).resolve()
        try:
            relative = path.relative_to(final_project)
        except ValueError:
            exists = path.exists() or str(path) in planned_models
        else:
            exists = (staged_project / relative).exists()
        if not exists:
            missing.append(text)
        return exists

    value["global_source_fonts"] = [
        path for path in value.get("global_source_fonts") or [] if available(path)
    ]
    regional = value.get("regional_source_fonts") or {}
    value["regional_source_fonts"] = {
        region: [path for path in regional.get(region, []) if available(path)]
        for region in ("SC", "TC", "JP", "KR")
    }
    value["target_assets"] = [
        path for path in value.get("target_assets") or [] if available(path)
    ]
    value["style_reference_pool"] = [
        path for path in value.get("style_reference_pool") or [] if available(path)
    ]
    for key in ("base_model", "active_checkpoint"):
        if value.get(key) and not available(value[key]):
            value[key] = ""
    training = dict(value.get("training") or {})
    if training.get("dataset_path") and not available(training["dataset_path"]):
        training["dataset_path"] = ""
    value["training"] = training
    return sorted(set(missing))


def _model_destination(storage: Storage, entry: dict[str, Any]) -> Path:
    filename = Path(str(entry.get("filename") or "checkpoint.pth")).name
    destination = (storage.models_dir / filename).resolve()
    if destination.parent != storage.models_dir:
        raise ValueError("Invalid shared model filename in project package")
    if destination.exists():
        digest = str(entry.get("sha256") or "")
        if digest and sha256_file(destination) == digest:
            return destination
        destination = storage.models_dir / f"{digest[:12]}-{filename}"
        counter = 2
        while destination.exists() and sha256_file(destination) != digest:
            destination = storage.models_dir / f"{digest[:12]}-{counter}-{filename}"
            counter += 1
    return destination.resolve()


def import_project_package(
    storage: Storage,
    package_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    callback = progress or _noop_progress
    inspection = inspect_project_package(package_path, progress=callback)
    archive, infos = _validated_archive(package_path)
    stage_root = (storage.root / ".imports" / uuid4().hex).resolve()
    staged_project = stage_root / "project"
    staged_models = stage_root / "models"
    new_project_id = str(uuid4())
    final_project = storage.project_dir(new_project_id)
    created_models: list[Path] = []
    project_moved = False
    try:
        raw_project = _read_json_member(archive, "records/project.json")
        raw_runs = _read_json_member(archive, "records/training-runs.json")
        raw_jobs = _read_json_member(archive, "records/jobs.json")
        run_ids = {
            str(record["id"]): str(uuid4())
            for record in raw_runs
            if isinstance(record, dict) and record.get("id")
        }
        job_ids = {
            str(record["id"]): str(uuid4())
            for record in raw_jobs
            if isinstance(record, dict) and record.get("id")
        }
        staged_project.mkdir(parents=True)
        staged_models.mkdir(parents=True)
        for folder in PROJECT_FOLDERS:
            (staged_project / folder).mkdir(exist_ok=True)

        model_paths: dict[str, str] = {}
        included_model_sources: dict[str, Path] = {}
        models = inspection.get("models") or []
        model_entries = {
            str(entry.get("key")): entry
            for entry in models
            if isinstance(entry, dict) and entry.get("key")
        }
        total = max(len(infos), 1)
        for index, info in enumerate(infos, start=1):
            name = info.filename
            if name.startswith("project/"):
                old_relative = name.removeprefix("project/")
                relative = _remap_relative(old_relative, run_ids, job_ids)
                destination = _safe_output_path(
                    staged_project,
                    relative.as_posix(),
                )
            elif name.startswith("shared-models/"):
                parts = PurePosixPath(name).parts
                if len(parts) < 3:
                    raise ValueError(f"Invalid shared model member: {name}")
                key = parts[1]
                destination = _safe_output_path(
                    staged_models,
                    f"{key}/{Path(parts[-1]).name}",
                )
                included_model_sources[key] = destination
            else:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            callback(index, total, f"Extracting {name}")

        for key, entry in model_entries.items():
            existing = _find_existing_model(storage, entry)
            if existing:
                model_paths[key] = str(existing)
                continue
            source = included_model_sources.get(key)
            if source and source.is_file():
                digest = sha256_file(source)
                if digest != entry.get("sha256"):
                    raise ValueError(f"Included model checksum mismatch: {entry.get('filename')}")
                model_paths[key] = str(_model_destination(storage, entry))
            else:
                model_paths[key] = ""

        missing_models: set[str] = set()
        missing_external: set[str] = set()
        restored_project = _restore_value(
            raw_project,
            final_project,
            run_ids,
            job_ids,
            model_paths,
            missing_models,
            missing_external,
        )
        restored_project["id"] = new_project_id
        restored_project["created_at"] = utc_now()
        restored_project["updated_at"] = utc_now()
        export_settings = dict(restored_project.get("export") or {})
        export_settings["import_provenance"] = {
            "source_project_id": inspection.get("source_project_id"),
            "package_created_at": inspection.get("created_at"),
            "imported_at": utc_now(),
            "export_mode": inspection.get("export_mode"),
        }
        restored_project["export"] = export_settings
        missing_project_references = _prune_unavailable_manifest_paths(
            restored_project,
            final_project,
            staged_project,
            {path for path in model_paths.values() if path},
        )
        manifest = ProjectManifest.from_dict(restored_project)
        manifest.id = new_project_id

        restored_runs: list[TrainingRun] = []
        for raw_run in raw_runs:
            restored = _restore_value(
                raw_run,
                final_project,
                run_ids,
                job_ids,
                model_paths,
                missing_models,
                missing_external,
            )
            old_id = str(raw_run.get("id") or "")
            restored["id"] = run_ids.get(old_id, str(uuid4()))
            restored["project_id"] = new_project_id
            if restored.get("parent_run_id"):
                restored["parent_run_id"] = run_ids.get(
                    str(restored["parent_run_id"]),
                    "",
                )
            parameters = dict(restored.get("parameters") or {})
            if parameters.get("job_id"):
                parameters["job_id"] = job_ids.get(str(parameters["job_id"]), "")
            restored["parameters"] = parameters
            if restored.get("status") in {"queued", "running"}:
                restored["status"] = "interrupted"
            fields = TrainingRun.__dataclass_fields__
            restored_runs.append(
                TrainingRun(
                    **{
                        key: value
                        for key, value in restored.items()
                        if key in fields
                    }
                )
            )

        restored_jobs: list[dict[str, Any]] = []
        for raw_job in raw_jobs:
            restored = _restore_value(
                raw_job,
                final_project,
                run_ids,
                job_ids,
                model_paths,
                missing_models,
                missing_external,
            )
            old_id = str(raw_job.get("id") or "")
            restored["id"] = job_ids.get(old_id, str(uuid4()))
            restored["project_id"] = new_project_id
            restored["pid"] = None
            restored["env"] = {}
            if restored.get("status") in {"queued", "running"}:
                restored["status"] = "interrupted"
                restored["finished_at"] = utc_now()
            restored_jobs.append(restored)

        (staged_project / "project.json").write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if final_project.exists():
            raise FileExistsError(final_project)
        staged_project.replace(final_project)
        project_moved = True
        for key, destination_text in model_paths.items():
            source = included_model_sources.get(key)
            destination = Path(destination_text) if destination_text else None
            if not source or not source.is_file() or not destination:
                continue
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            created_models.append(destination)

        with storage._lock, storage._connect() as db:
            manifest_path = final_project / "project.json"
            db.execute(
                """
                INSERT INTO projects(id, name, manifest_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    manifest.id,
                    manifest.name,
                    str(manifest_path),
                    manifest.created_at,
                    manifest.updated_at,
                ),
            )
            for run in restored_runs:
                db.execute(
                    """
                    INSERT INTO training_runs(
                        id, project_id, status, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        new_project_id,
                        run.status,
                        json.dumps(run.to_dict(), ensure_ascii=False),
                        run.created_at,
                        run.updated_at,
                    ),
                )
            for job in restored_jobs:
                db.execute(
                    """
                    INSERT INTO jobs(
                        id, project_id, job_type, status, command_json, cwd, gpu,
                        env_json, pid, return_code, log_path, error, schema_version,
                        progress, progress_text, resume_point, created_at, started_at,
                        finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job["id"],
                        new_project_id,
                        str(job.get("job_type") or "imported"),
                        str(job.get("status") or "interrupted"),
                        json.dumps(job.get("command") or []),
                        str(job.get("cwd") or final_project),
                        job.get("gpu"),
                        "{}",
                        None,
                        job.get("return_code"),
                        str(job.get("log_path") or ""),
                        job.get("error"),
                        int(job.get("schema_version") or 1),
                        float(job.get("progress") or 0),
                        str(job.get("progress_text") or ""),
                        str(job.get("resume_point") or ""),
                        str(job.get("created_at") or utc_now()),
                        job.get("started_at"),
                        job.get("finished_at"),
                    ),
                )
        callback(total, total, "Project import complete")
        missing_model_names = [
            str(model_entries[key].get("filename") or key)
            for key in sorted(missing_models)
            if key in model_entries
        ]
        return {
            "project_id": new_project_id,
            "project_name": manifest.name,
            "project_path": str(final_project),
            "source_project_id": inspection.get("source_project_id"),
            "training_run_count": len(restored_runs),
            "job_count": len(restored_jobs),
            "missing_models": missing_model_names,
            "missing_external_references": sorted(missing_external),
            "missing_project_references": missing_project_references,
            "checkpoint_validation_required": True,
        }
    except BaseException:
        if project_moved and final_project.exists():
            shutil.rmtree(final_project)
        for model in created_models:
            if model.is_file():
                model.unlink()
        raise
    finally:
        archive.close()
        if stage_root.exists():
            shutil.rmtree(stage_root, ignore_errors=True)
