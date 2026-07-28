from __future__ import annotations

import json
import hashlib
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from .devices import detect_devices, training_preset
from .jobs import JobManager
from .models import ProjectManifest, TrainingRun
from .storage import Storage


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MODEL_FOLDER = "https://drive.google.com/drive/folders/1QJi2ihxDBK2NF-jCE07g59YwuUTAd-iY"
OFFICIAL_MODELS = {
    "JiT-B/16": "zi2zi-JiT-B-16.pth",
    "JiT-L/16": "zi2zi-JiT-L-16.pth",
}
DATASET_SIZE_PRESETS = {
    "quick": (500, 8),
    "balanced": (3000, 64),
    "coverage": (6000, 128),
}
GENERATION_CHECKPOINTS = (
    ("checkpoint-best-ssim.pth", "best SSIM"),
    ("checkpoint-best-lpips.pth", "best LPIPS"),
    ("checkpoint-last.pth", "last"),
)


def dataset_size_preset(name: str) -> tuple[int, int]:
    try:
        return DATASET_SIZE_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset size preset: {name}") from exc


def font_dataset_capacity(manifest: ProjectManifest, charset: str) -> int:
    if manifest.input_mode != "font":
        if not manifest.target_assets:
            raise ValueError("A rendered glyph directory is required")
        from data_processing.font_utils import GlyphRendererPool, scan_rendered_glyphs

        source_fonts = manifest.source_fonts_for_region(
            {
                "gb2312": "SC",
                "gbk": "SC",
                "big5": "TC",
                "jisx0208": "JP",
                "ksx1001": "KR",
            }.get(charset)
        )
        if not source_fonts:
            raise ValueError("At least one source font is required")
        glyphs = scan_rendered_glyphs(manifest.target_assets[0])
        source_pool = GlyphRendererPool(source_fonts, 256)
        return sum(
            1 for codepoint in glyphs if source_pool.source_for(codepoint) is not None
        )
    source_fonts = manifest.source_fonts_for_region(
        {
            "gb2312": "SC",
            "gbk": "SC",
            "big5": "TC",
            "jisx0208": "JP",
            "ksx1001": "KR",
        }.get(charset)
    )
    if not source_fonts or not manifest.target_assets:
        raise ValueError("At least one source font and one target font are required")

    from data_processing.charsets import get_charset_codepoints
    from data_processing.font_utils import (
        GlyphRendererPool,
        has_valid_outline,
        is_cjk_codepoint,
        load_font,
    )

    allowed = (
        None
        if charset.lower() in {"auto", "all-cjk"}
        else get_charset_codepoints(charset)
    )
    target_font, _ = load_font(manifest.target_assets[0])
    target_cmap = target_font.getBestCmap() or {}
    target_candidates = {
        codepoint
        for codepoint in target_cmap
        if (allowed is None or codepoint in allowed) and is_cjk_codepoint(codepoint)
    }
    target_valid = {
        codepoint
        for codepoint in target_candidates
        if has_valid_outline(target_font, codepoint)
    }
    source_pool = GlyphRendererPool(source_fonts, 256)
    return sum(
        1 for codepoint in target_valid if source_pool.source_for(codepoint) is not None
    )


def dataset_max_chars_per_font(dataset_dir: str | Path) -> int:
    counts = []
    for metadata_path in (Path(dataset_dir) / "train").glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            counts.append(int(metadata.get("extracted_count") or 0))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return max(counts, default=0)


def clamp_training_retry_command(
    command: list[str],
    device_id: str,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Apply current hardware-safe limits to a historical training command."""
    updated = list(command)

    def option(name: str, default: int = 0) -> int:
        try:
            return int(updated[updated.index(name) + 1])
        except (ValueError, IndexError, TypeError):
            return default

    try:
        model_variant = updated[updated.index("--model") + 1]
    except (ValueError, IndexError):
        return updated, {}
    device = next((item for item in detect_devices() if item.id == str(device_id)), None)
    if not device or not device.training_supported:
        return updated, {}
    rank = option("--lora_r", 32)
    quality = "quality" if rank >= 64 else "economy" if rank <= 16 else "balanced"
    safe = training_preset(device, quality, model_variant)
    changes: dict[str, tuple[int, int]] = {}
    for argument, preset_key in (
        ("--batch_size", "batch_size"),
        ("--gen_bsz", "gen_bsz"),
        ("--num_workers", "num_workers"),
    ):
        try:
            value_index = updated.index(argument) + 1
            current = int(updated[value_index])
        except (ValueError, IndexError, TypeError):
            continue
        limit = int(safe[preset_key])
        if current > limit:
            updated[value_index] = str(limit)
            changes[argument] = (current, limit)
    return updated, changes


def infer_checkpoint_model(path: str | Path, metadata: dict[str, Any] | None = None) -> str | None:
    metadata = metadata or {}
    configured = str(metadata.get("model") or "")
    if configured in OFFICIAL_MODELS:
        return configured
    name = Path(path).name.lower()
    if re.search(r"(?:jit|^)[-_]?l(?:[-_.]|$)", name):
        return "JiT-L/16"
    if re.search(r"(?:jit|^)[-_]?b(?:[-_.]|$)", name):
        return "JiT-B/16"
    return None


def copy_project_input(
    storage: Storage,
    project_id: str,
    source: str | Path,
    category: str,
) -> Path:
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir = storage.project_dir(project_id) / "inputs" / category
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.is_dir():
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


def import_model(storage: Storage, source: str | Path, trusted: bool = False) -> dict[str, Any]:
    if not trusted:
        raise ValueError("Checkpoint import requires explicit trusted-file confirmation")
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
        raise ValueError("Expected a .pth, .pt, or .ckpt checkpoint")
    destination = storage.models_dir / source.name
    if source != destination:
        shutil.copy2(source, destination)
    metadata = validate_checkpoint(destination)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def validate_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    state_dict = None
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_ema1") or checkpoint.get("model")
        if state_dict is None and all(isinstance(key, str) for key in checkpoint):
            state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a recognizable model state")
    model = getattr(args, "model", None) if args is not None else None
    if model not in OFFICIAL_MODELS:
        position = state_dict.get("net.pos_embed")
        width = int(position.shape[-1]) if hasattr(position, "shape") else 0
        model = "JiT-L/16" if width == 1024 else "JiT-B/16" if width == 768 else None
    model = model or infer_checkpoint_model(path)
    return {
        "path": str(Path(path).resolve()),
        "filename": Path(path).name,
        "model": model or "unknown",
        "img_size": getattr(args, "img_size", None),
        "num_fonts": getattr(args, "num_fonts", None),
        "num_chars": getattr(args, "num_chars", None),
        "lora": any("lora_" in key for key in state_dict),
        "state_keys": len(state_dict),
        "trusted_required": True,
    }


def queue_official_model_download(
    jobs: JobManager,
    storage: Storage,
    model_name: str,
) -> str:
    expected = OFFICIAL_MODELS[model_name]
    helper = ROOT / "scripts" / "download_official_models.py"
    return jobs.submit(
        "model_download",
        [
            sys.executable,
            str(helper),
            "--folder-url",
            OFFICIAL_MODEL_FOLDER,
            "--output-dir",
            str(storage.models_dir),
            "--expected",
            expected,
        ],
        cwd=ROOT,
    )


def dataset_command(
    manifest: ProjectManifest,
    storage: Storage,
    *,
    train_count: int,
    test_count: int,
    charset: str,
    workers: int,
) -> tuple[list[str], Path]:
    project_dir = storage.project_dir(manifest.id)
    output = (
        project_dir
        / "datasets"
        / (
            f"dataset-{charset}-train{train_count}-test{test_count}"
            f"-{uuid4().hex}"
        )
    )
    build_marker = output / ".webui-dataset.json"
    charset_region = {
        "gb2312": "SC",
        "gbk": "SC",
        "big5": "TC",
        "jisx0208": "JP",
        "ksx1001": "KR",
    }.get(charset)
    source_fonts = manifest.source_fonts_for_region(charset_region)
    if manifest.input_mode == "font":
        if not source_fonts or not manifest.target_assets:
            raise ValueError("At least one source font and one target font are required")
        target = Path(manifest.target_assets[0])
        command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_font_dataset.py"),
            "--source-font",
            *source_fonts,
            "--target-font",
            str(target),
            "--output-dir",
            str(output),
            "--num-fonts",
            "1",
            "--train-chars-per-font",
            str(train_count),
            "--test-chars-per-font",
            str(test_count),
            "--charset",
            charset,
            "--num-workers",
            str(workers),
            "--build-marker",
            str(build_marker),
        ]
    else:
        if not source_fonts or not manifest.target_assets:
            raise ValueError("At least one source font and a glyph directory are required")
        command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_glyph_dataset.py"),
            "--source-font",
            *source_fonts,
            "--glyph-dir",
            manifest.target_assets[0],
            "--output-dir",
            str(output),
            "--train-count",
            str(train_count),
            "--test-count",
            str(test_count),
            "--build-marker",
            str(build_marker),
        ]
    return command, output


def resolve_training_dataset(
    manifest: ProjectManifest,
    storage: Storage,
    requested: str | Path | None = None,
) -> Path:
    """Return a complete dataset, falling back to the project's newest dataset."""
    candidates: list[Path] = []
    requested_text = str(requested or "").strip()
    stored_text = str(manifest.training.get("dataset_path") or "").strip()
    for value in (requested_text, stored_text):
        if value:
            candidate = Path(value).expanduser().resolve()
            if candidate not in candidates:
                candidates.append(candidate)

    datasets_root = storage.project_dir(manifest.id) / "datasets"
    if datasets_root.is_dir():
        discovered = sorted(
            (path for path in datasets_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        candidates.extend(path.resolve() for path in discovered if path.resolve() not in candidates)

    for candidate in candidates:
        marker = candidate / ".webui-dataset.json"
        marker_complete = True
        if marker.exists():
            try:
                marker_complete = (
                    json.loads(marker.read_text(encoding="utf-8")).get("status")
                    == "complete"
                )
            except (OSError, ValueError, json.JSONDecodeError):
                marker_complete = False
        if (
            marker_complete
            and (candidate / "train").is_dir()
            and (candidate / "test.npz").is_file()
        ):
            return candidate

    checked = ", ".join(str(path) for path in candidates) or str(datasets_root)
    raise ValueError(
        "No complete training dataset was found. Generate the dataset first; "
        f"expected both train/ and test.npz under: {checked}"
    )


def training_command(
    manifest: ProjectManifest,
    storage: Storage,
    dataset_dir: str | Path,
    device_id: str,
    quality: str,
    overrides: dict[str, Any] | None = None,
    resume_checkpoint: str | Path | None = None,
    parent_run_id: str = "",
) -> tuple[TrainingRun, list[str]]:
    device = next((item for item in detect_devices() if item.id == device_id), None)
    if not device or not device.training_supported:
        raise ValueError("LoRA training requires an NVIDIA CUDA device")
    model_variant = str((overrides or {}).get("model") or "") or None
    preset = training_preset(device, quality, model_variant)
    preset.update(overrides or {})
    dataset_dir = resolve_training_dataset(manifest, storage, dataset_dir)
    manifest.training["dataset_path"] = str(dataset_dir)
    storage.save_project(manifest)
    preset["dataset_path"] = str(dataset_dir)
    extracted_per_font = dataset_max_chars_per_font(dataset_dir)
    if extracted_per_font:
        preset["max_chars_per_font"] = extracted_per_font
    fingerprint = hashlib.sha256()
    for path in sorted(dataset_dir.rglob("metadata.json")):
        fingerprint.update(str(path.relative_to(dataset_dir)).encode("utf-8"))
        fingerprint.update(path.read_bytes())
    test_npz = dataset_dir / "test.npz"
    if test_npz.exists():
        stat = test_npz.stat()
        fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    run = TrainingRun(
        project_id=manifest.id,
        parameters=preset,
        device={"id": device.id, "name": device.name, "memory_total_gb": device.memory_total_gb},
        dataset_hash=fingerprint.hexdigest(),
        parent_run_id=parent_run_id,
    )
    run_dir = storage.project_dir(manifest.id) / "training" / run.id
    run_dir.mkdir(parents=True, exist_ok=False)
    run.metrics_path = str(run_dir / "metrics.jsonl")
    run.tensorboard_path = str(run_dir / "tensorboard")
    storage.save_training_run(run)

    command = [
        sys.executable,
        str(ROOT / "lora_single_gpu_finetune_jit.py"),
        "--data_path",
        str(dataset_dir / "train"),
        "--test_npz_path",
        str(dataset_dir / "test.npz"),
        "--output_dir",
        str(run_dir),
        "--base_checkpoint",
        manifest.base_model,
        "--model",
        preset.get("model", "JiT-B/16"),
        "--num_fonts",
        str(preset.get("num_fonts", 1000)),
        "--num_chars",
        str(preset.get("num_chars", 20000)),
        "--max_chars_per_font",
        str(preset.get("max_chars_per_font", 500)),
        "--epochs",
        str(preset.get("epochs", 200)),
        "--batch_size",
        str(preset["batch_size"]),
        "--blr",
        str(preset.get("blr", 8e-4)),
        "--warmup_epochs",
        str(preset.get("warmup_epochs", 1)),
        "--save_last_freq",
        str(preset.get("save_last_freq", 10)),
        "--lora_r",
        str(preset["lora_r"]),
        "--lora_alpha",
        str(preset["lora_alpha"]),
        "--gen_bsz",
        str(preset["gen_bsz"]),
        "--num_workers",
        str(preset["num_workers"]),
        "--metrics_jsonl",
        run.metrics_path,
        "--run_id",
        run.id,
        "--snapshot_freq",
        str(preset["snapshot_freq"]),
        "--eval_step_folders",
    ]
    if preset.get("full_eval"):
        command += ["--online_eval", "--eval_freq", str(preset["eval_freq"])]
    if preset.get("early_stop"):
        command += [
            "--early_stop",
            "--early_stop_metric",
            str(preset.get("early_stop_metric", "ssim")),
            "--early_stop_start",
            str(preset.get("early_stop_start", 40)),
            "--early_stop_patience",
            str(preset.get("early_stop_patience", 4)),
            "--early_stop_min_delta",
            str(preset.get("early_stop_min_delta", 0.001)),
        ]
    if resume_checkpoint:
        command += ["--resume", str(Path(resume_checkpoint).resolve())]
    return run, command


def generation_command(
    manifest: ProjectManifest,
    storage: Storage,
    input_path: str | Path,
    device_id: str,
    *,
    seed: int,
    candidates: int = 1,
    output_name: str | None = None,
    output_dir: str | Path | None = None,
    request_manifest: bool = False,
    checkpoint_path: str | Path | None = None,
) -> tuple[list[str], Path]:
    output = Path(output_dir) if output_dir else (
        storage.project_dir(manifest.id)
        / "generation"
        / (output_name or f"seed-{seed}")
    )
    command = [
        sys.executable,
        str(ROOT / "generate_chars.py"),
        "--checkpoint",
        str(checkpoint_path or manifest.active_checkpoint or manifest.base_model),
        "--request-manifest" if request_manifest else "--test_npz",
        str(input_path),
        "--output_dir",
        str(output),
        "--device",
        "cuda" if device_id.isdigit() else device_id,
        "--sampling_method",
        "ab2",
        "--seed",
        str(seed),
        "--num_candidates",
        str(candidates),
    ]
    return command, output


def generation_checkpoint_options(
    manifest: ProjectManifest,
    storage: Storage,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return project-owned inference checkpoints and the recommended selection."""
    options: list[tuple[str, str]] = []
    seen: set[Path] = set()
    training_paths: list[Path] = []

    for run in storage.list_training_runs(manifest.id):
        run_id = str(run["id"])
        run_dir = storage.project_dir(manifest.id) / "training" / run_id
        for filename, kind in GENERATION_CHECKPOINTS:
            checkpoint = (run_dir / filename).resolve()
            if not checkpoint.is_file() or checkpoint in seen:
                continue
            options.append(
                (
                    f"LoRA · run {run_id[:8]} · {kind}",
                    str(checkpoint),
                )
            )
            seen.add(checkpoint)
            training_paths.append(checkpoint)

    active = (
        Path(manifest.active_checkpoint).resolve()
        if manifest.active_checkpoint
        else None
    )
    base = Path(manifest.base_model).resolve() if manifest.base_model else None
    if active and active.is_file() and active not in seen and active != base:
        options.append((f"Active checkpoint · {active.name}", str(active)))
        seen.add(active)
    if base and base.is_file() and base not in seen:
        options.append((f"Base model · {base.name}", str(base)))
        seen.add(base)

    if active and active.is_file() and active != base:
        preferred = active
    elif training_paths:
        preferred = training_paths[0]
    elif base and base.is_file():
        preferred = base
    elif active and active.is_file():
        preferred = active
    else:
        preferred = None
    return options, str(preferred) if preferred else None


def resolve_generation_checkpoint(
    manifest: ProjectManifest,
    storage: Storage,
    selected: str | Path | None,
) -> str:
    options, preferred = generation_checkpoint_options(manifest, storage)
    requested = Path(selected or preferred or "").resolve()
    allowed = {Path(value).resolve() for _label, value in options}
    if not str(selected or preferred or "").strip():
        raise ValueError(
            "No generation checkpoint is available. Attach a base model or finish LoRA training."
        )
    if requested not in allowed:
        raise ValueError(
            "The selected checkpoint is unavailable or does not belong to this project."
        )
    if not requested.is_file():
        raise ValueError(f"Generation checkpoint not found: {requested}")
    return str(requested)
