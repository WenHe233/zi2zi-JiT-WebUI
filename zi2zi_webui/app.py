from __future__ import annotations

import json
import html
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib.parse import quote

from .charts import (
    latest_summary_from_records,
    run_parameter_rows,
    training_figures,
    training_figures_from_records,
)
from .checkpoints import checkpoint_sidecar_path, read_checkpoint_sidecar
from .devices import detect_devices, disk_free_gb
from .font_builder import (
    FontMetadata,
    validate_font_metadata,
)
from .generation_batches import (
    batch_choices,
    candidates_for_codepoint,
    load_and_prune_selections,
    review_page as generation_review_page,
    scan_generation_batch,
    validate_batch_path,
)
from .i18n import translator
from .inference import (
    build_generation_request,
    choose_diverse_references,
    exclude_existing_target_glyphs,
    render_style_references,
    write_generation_request,
)
from .jobs import JobManager
from .presets import (
    PRESETS,
    preset_catalog,
    resolve_selection,
    resolve_selection_by_region,
    write_selection_manifest,
)
from .services import (
    ROOT,
    clamp_training_retry_command,
    copy_project_input,
    dataset_command,
    dataset_size_preset,
    font_dataset_capacity,
    generation_command,
    generation_checkpoint_options,
    import_model,
    infer_checkpoint_model,
    queue_checkpoint_validation,
    queue_official_model_download,
    resolve_generation_checkpoint,
    resolve_training_dataset,
    training_command,
)
from .storage import Storage
from .telemetry import (
    IncrementalMetricsCache,
    export_metrics_csv,
    export_metrics_json,
)


def training_snapshot_items(
    records: list[tuple[str, list[dict[str, Any]]]],
    language: str = "zh",
    limit: int = 24,
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for run_id, events in records:
        for event in events:
            if event.get("phase") != "evaluation":
                continue
            try:
                epoch_number = int(event.get("epoch", 0)) + 1
            except (TypeError, ValueError):
                epoch_number = 1
            snapshot_root = Path(str(event.get("snapshot_path") or ""))
            if not snapshot_root.is_dir():
                continue
            for path in sorted(snapshot_root.glob("*.png")):
                key = (run_id, epoch_number, str(path.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    group_number = int(path.stem.rsplit("_", 1)[-1]) + 1
                except ValueError:
                    group_number = 1
                caption = (
                    f"Run {run_id} · 第 {epoch_number} 轮 · 第 {group_number} 组"
                    f"（每对：目标真值 → 当前生成）"
                    if language != "en"
                    else f"Run {run_id} · Epoch {epoch_number} · Group {group_number} "
                    f"(each pair: target → generated)"
                )
                items.append((str(path), caption))
    return items[-limit:]


def build_app(
    storage: Storage,
    jobs: JobManager,
    *,
    language: str = "zh",
    tensorboard_url: str = "http://127.0.0.1:6006",
):
    import gradio as gr

    t = translator(language)
    b = lambda zh, en: en if language == "en" else zh
    metrics_cache = IncrementalMetricsCache()
    dashboard_status_cache: dict[str, str] = {}
    job_log_cache: dict[str, Any] = {"job_id": None, "signature": None}
    catalog = preset_catalog("en" if language == "en" else "zh")
    preset_choices = [
        (
            f"{item['label']} ⓘ — {item['description']} [{item['source']}]",
            item["id"],
        )
        for item in catalog
        if not item["id"] in {"ja-joyo", "ja-kana"}
    ]
    devices = detect_devices()
    device_choices = [
        (
            f"{item.id}: {item.name}"
            + (f" · {item.memory_total_gb} GB" if item.memory_total_gb else ""),
            item.id,
        )
        for item in devices
    ]

    def project_choices():
        return [(f"{item.name} · {item.id[:8]}", item.id) for item in storage.list_projects()]

    def create_project(name):
        project = storage.create_project(name)
        return (
            gr.update(choices=project_choices(), value=project.id),
            f"Created: {project.name} ({project.id})",
        )

    def refresh_projects():
        choices = project_choices()
        return gr.update(choices=choices, value=choices[0][1] if choices else None)

    def delete_project_action(project_id, confirmed):
        if not project_id:
            raise gr.Error(b("请先选择项目", "Select a project first"))
        if not confirmed:
            raise gr.Error(
                b("请先确认永久删除项目", "Confirm permanent project deletion first")
            )
        try:
            deleted = storage.delete_project(project_id)
        except KeyError as exc:
            raise gr.Error(
                b("所选项目已不存在", "The selected project no longer exists")
            ) from exc
        except (ValueError, OSError) as exc:
            raise gr.Error(
                f"{b('删除项目失败', 'Project deletion failed')}: {exc}"
            ) from exc
        choices = project_choices()
        next_project_id = choices[0][1] if choices else None
        next_manifest = (
            project_summary(next_project_id) if next_project_id else "{}"
        )
        return (
            gr.update(choices=choices, value=next_project_id),
            next_manifest,
            (
                f"{b('已永久删除项目及其文件', 'Permanently deleted project and its files')}: "
                f"{deleted.name} ({deleted.id})"
            ),
            False,
        )

    def project_summary(project_id):
        if not project_id:
            return "{}"
        return json.dumps(storage.get_project(project_id).to_dict(), ensure_ascii=False, indent=2)

    def save_assets(
        project_id,
        input_mode,
        source_fonts,
        target_files,
        sc_font,
        tc_font,
        jp_font,
        kr_font,
    ):
        if not project_id:
            raise gr.Error(b("请先选择项目", "Select a project first"))
        project = storage.get_project(project_id)
        project.input_mode = input_mode
        if source_fonts:
            project.global_source_fonts = [
                str(copy_project_input(storage, project_id, item, "source-font"))
                for item in source_fonts
            ]
        copied_targets = []
        for item in target_files or []:
            category = "target-font" if input_mode == "font" else "target-glyphs"
            copied_targets.append(copy_project_input(storage, project_id, item, category))
        if copied_targets:
            if input_mode == "glyphs":
                project.target_assets = [str(copied_targets[0].parent)]
            else:
                project.target_assets = [str(item) for item in copied_targets[:1]]
        for region, values in zip(
            ("SC", "TC", "JP", "KR"),
            (sc_font, tc_font, jp_font, kr_font),
        ):
            if values:
                project.regional_source_fonts[region] = [
                    str(
                        copy_project_input(
                            storage,
                            project_id,
                            item,
                            f"source-{region.lower()}",
                        )
                    )
                    for item in values
                ]
        if input_mode == "font" and project.target_assets:
            project.style_reference_pool = render_style_references(
                project.target_assets[0],
                storage.project_dir(project_id) / "inputs" / "style-references",
            )
        else:
            from data_processing.font_utils import scan_rendered_glyphs

            candidates = []
            for asset in project.target_assets:
                path = Path(asset)
                if path.is_dir():
                    try:
                        candidates.extend(scan_rendered_glyphs(path).values())
                    except ValueError as exc:
                        raise gr.Error(
                            f"{b('字形图片校验失败', 'Glyph image validation failed')}: {exc}"
                        ) from exc
                elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    candidates.append(path)
            project.style_reference_pool = choose_diverse_references(candidates, 8)
            if len(project.style_reference_pool) < 8:
                raise gr.Error(
                    b(
                        "风格参考至少需要 8 张有效目标字形图片",
                        "At least 8 valid target glyph images are required for style references",
                    )
                )
        storage.save_project(project)
        return "Assets saved.\n" + project_summary(project_id)

    def attach_model(project_id, upload, trusted):
        if not project_id:
            raise gr.Error(b("请先选择项目", "Select a project first"))
        if not upload:
            raise gr.Error(b("请选择 checkpoint", "Choose a checkpoint"))
        metadata = import_model(storage, upload, trusted)
        project = storage.get_project(project_id)
        project.base_model = metadata["path"]
        project.active_checkpoint = metadata["path"]
        storage.save_project(project)
        return json.dumps(metadata, ensure_ascii=False, indent=2)

    def download_model(name):
        job_id = queue_official_model_download(jobs, storage, name)
        return f"Queued model download: {job_id}"

    def shared_model_choices():
        choices = [
            (item.name, str(item))
            for item in sorted(storage.models_dir.glob("*"))
            if item.suffix.lower() in {".pth", ".pt", ".ckpt"}
        ]
        return gr.update(choices=choices, value=choices[0][1] if choices else None)

    def attach_shared_model(project_id, model_path):
        if not project_id or not model_path:
            raise gr.Error(
                b("请选择项目和共享模型", "Select a project and shared model")
            )
        project = storage.get_project(project_id)
        project.base_model = model_path
        project.active_checkpoint = model_path
        storage.save_project(project)
        try:
            metadata = read_checkpoint_sidecar(model_path)
        except (FileNotFoundError, OSError, ValueError):
            job_id = ensure_checkpoint_validation(model_path, project_id)
            return (
                f"Attached trusted shared model: {model_path}\n"
                f"Metadata validation queued as job {job_id}. "
                "Training remains disabled until the sidecar is ready."
            )
        return (
            f"Attached shared model: {model_path}\n"
            f"Architecture: {metadata['architecture']}\n"
            f"SHA-256: {metadata['sha256']}"
        )

    def ensure_checkpoint_validation(model_path, project_id=None):
        resolved = str(Path(model_path).resolve())
        for job in jobs.list_jobs(project_id=project_id, limit=200):
            if (
                job["job_type"] == "checkpoint_validation"
                and job["status"] in {"queued", "running"}
                and resolved in json.loads(job["command_json"])
            ):
                return job["id"]
        return queue_checkpoint_validation(
            jobs,
            resolved,
            project_id=project_id,
            provenance_note=(
                "Shared model directory is trusted by local single-user policy. "
                "Digest is recorded for audit/change detection, not provenance proof."
            ),
        )

    def queue_dataset(project_id, train_count, test_count, charset, workers):
        project = storage.get_project(project_id)
        train_count = int(train_count)
        test_count = int(test_count)
        if train_count < 9:
            raise gr.Error(
                b("训练至少需要 9 个不同字形", "Training needs at least 9 distinct glyphs")
            )
        if test_count < 1:
            raise gr.Error(
                b("验证字形数量至少为 1", "Validation glyph count must be at least 1")
            )
        capacity = font_dataset_capacity(project, charset)
        if train_count + test_count > capacity:
            raise gr.Error(
                b(
                    f"所选字体和范围筛选共有 {capacity} 个可用公共字形；训练与验证请求了 "
                    f"{train_count + test_count} 个，请将训练数量降至 "
                    f"{max(capacity - test_count, 0)} 或更少。",
                    f"The selected fonts and optional range filter have {capacity} usable common glyphs. "
                    f"Training + validation requested {train_count + test_count}; reduce training "
                    f"to at most {max(capacity - test_count, 0)}.",
                )
            )
        command, output = dataset_command(
            project,
            storage,
            train_count=train_count,
            test_count=test_count,
            charset=charset,
            workers=int(workers),
        )
        project.training["dataset_path"] = str(output)
        storage.save_project(project)
        job_id = jobs.submit("dataset", command, project_id=project_id, cwd=ROOT)
        return (
            str(output),
            f"Queued dataset job: {job_id}. Usable glyph capacity: {capacity}; "
            f"training: {train_count}; held-out validation: {test_count}.",
        )

    def analyze_dataset_capacity(project_id, charset, test_count):
        if not project_id:
            raise gr.Error(b("请选择项目", "Select a project"))
        capacity = font_dataset_capacity(storage.get_project(project_id), charset)
        test_count = max(int(test_count), 1)
        return (
            f"**{b('可用公共字形', 'Usable common glyphs')}：{capacity}**  \n"
            f"{b('在保留验证集后，训练字形最多可设为', 'Maximum training glyphs after holding out validation')}"
            f"：{max(capacity - test_count, 0)}"
        )

    def apply_dataset_size_preset(name):
        return dataset_size_preset(name)

    def project_dataset_path(project_id):
        if not project_id:
            return ""
        project = storage.get_project(project_id)
        stored = str(project.training.get("dataset_path") or "").strip()
        if stored:
            return stored
        try:
            return str(resolve_training_dataset(project, storage))
        except ValueError:
            return ""

    def queue_training(
        project_id,
        dataset_path,
        device,
        quality,
        epochs,
        horizontal_flip_prob,
    ):
        project = storage.get_project(project_id)
        if not project.base_model:
            raise gr.Error(
                b(
                    "请先导入或下载并挂接基础模型",
                    "Import or download and attach a base model first",
                )
            )
        flip_probability = float(horizontal_flip_prob)
        if not 0.0 <= flip_probability <= 0.5:
            raise gr.Error(
                b(
                    "水平镜像概率必须在 0.0 到 0.5 之间",
                    "Horizontal flip probability must be between 0.0 and 0.5",
                )
            )
        overrides: dict[str, Any] = {
            "epochs": int(epochs),
            "horizontal_flip_prob": flip_probability,
        }
        try:
            metadata = read_checkpoint_sidecar(project.base_model)
        except (FileNotFoundError, OSError, ValueError) as exc:
            job_id = ensure_checkpoint_validation(project.base_model, project_id)
            raise gr.Error(
                b(
                    f"Checkpoint 元数据缺失或已失效；校验任务 {job_id} 正在运行，"
                    "成功后请重试训练。",
                    f"Checkpoint metadata is missing or stale. Validation job {job_id} "
                    "is running; retry training after it succeeds.",
                )
            ) from exc
        for key in ("num_fonts", "num_chars"):
            if metadata.get(key):
                overrides[key] = metadata[key]
        model_variant = infer_checkpoint_model(project.base_model, metadata)
        if not model_variant:
            raise gr.Error(
                b(
                    "无法判断 checkpoint 是 JiT-B/16 还是 JiT-L/16；"
                    "请重新导入以生成校验后的元数据。",
                    "Cannot determine whether the checkpoint is JiT-B/16 or JiT-L/16. "
                    "Re-import it to generate validated metadata.",
                )
            )
        overrides["model"] = model_variant
        run, command = training_command(
            project, storage, dataset_path, device, quality, overrides
        )
        job_id = jobs.submit(
            "training",
            command,
            project_id=project_id,
            cwd=ROOT,
            gpu=device,
            resume_point=Path(run.metrics_path).parent / "checkpoint-last.pth",
        )
        run.parameters["job_id"] = job_id
        storage.save_training_run(run)
        return (
            run.id,
            f"Queued training run {run.id}; job {job_id}; architecture {model_variant}",
        )

    def resume_training(
        project_id,
        run_ids,
        dataset_path,
        device,
        epochs,
        horizontal_flip_prob,
    ):
        selected_id = (run_ids or [None])[0]
        parent = next(
            (item for item in project_runs(project_id) if item["id"] == selected_id),
            None,
        )
        if not parent:
            raise gr.Error(
                b("请选择要续训的训练 run", "Select a training run to resume")
            )
        checkpoint = (
            storage.project_dir(project_id)
            / "training"
            / parent["id"]
            / "checkpoint-last.pth"
        )
        if not checkpoint.is_file():
            raise gr.Error(
                b(
                    "所选 run 没有 checkpoint-last.pth",
                    "The selected run has no checkpoint-last.pth",
                )
            )
        source_dataset = parent.get("parameters", {}).get("dataset_path") or dataset_path
        if not source_dataset:
            raise gr.Error(
                b("原始数据集路径不可用", "The original dataset path is unavailable")
            )
        overrides = dict(parent.get("parameters", {}))
        overrides["epochs"] = int(epochs)
        flip_probability = float(horizontal_flip_prob)
        if not 0.0 <= flip_probability <= 0.5:
            raise gr.Error(
                b(
                    "水平镜像概率必须在 0.0 到 0.5 之间",
                    "Horizontal flip probability must be between 0.0 and 0.5",
                )
            )
        overrides["horizontal_flip_prob"] = flip_probability
        run, command = training_command(
            storage.get_project(project_id),
            storage,
            source_dataset,
            device,
            "balanced",
            overrides,
            resume_checkpoint=checkpoint,
            parent_run_id=parent["id"],
        )
        job_id = jobs.submit(
            "training",
            command,
            project_id=project_id,
            cwd=ROOT,
            gpu=device,
            resume_point=Path(run.metrics_path).parent / "checkpoint-last.pth",
        )
        run.parameters["job_id"] = job_id
        storage.save_training_run(run)
        return run.id, f"Resuming {parent['id']} as child run {run.id}; job {job_id}"

    def delete_training(project_id, run_ids, confirmed):
        selected_id = (run_ids or [None])[0]
        if not selected_id:
            raise gr.Error(b("请选择训练 run", "Select a training run"))
        if not confirmed:
            raise gr.Error(b("请先确认永久删除", "Confirm permanent deletion first"))
        selected = next(
            (item for item in project_runs(project_id) if item["id"] == selected_id),
            None,
        )
        if not selected:
            raise gr.Error(b("找不到训练 run", "Training run was not found"))
        if selected.get("status") in {"queued", "running"}:
            raise gr.Error(
                b(
                    "删除 run 前请先取消活动任务",
                    "Cancel the active job before deleting its run",
                )
            )
        storage.delete_training_run(project_id, selected_id)
        return run_choices(project_id), f"Deleted training run {selected_id}"

    def project_runs(project_id):
        runs = storage.list_training_runs(project_id) if project_id else []
        for item in runs:
            job_id = item.get("parameters", {}).get("job_id")
            if job_id:
                try:
                    job = jobs.get_job(job_id)
                    item["status"] = job["status"]
                    item["error"] = job.get("error") or ""
                    item["progress"] = job.get("progress") or 0
                except KeyError:
                    pass
        return runs

    def run_choices(project_id):
        runs = project_runs(project_id)
        choices = [
            (
                f"{item['id'][:8]} · {item['status']} · "
                f"{float(item.get('progress', 0)) * 100:.1f}%",
                item["id"],
            )
            for item in runs
        ]
        return gr.update(choices=choices, value=[item[1] for item in choices[:1]])

    def generation_checkpoint_choices(project_id):
        if not project_id:
            return gr.update(choices=[], value=None)
        project = storage.get_project(project_id)
        choices, preferred = generation_checkpoint_options(project, storage)
        for _label, checkpoint in choices:
            if not checkpoint_sidecar_path(checkpoint).is_file():
                ensure_checkpoint_validation(checkpoint, project_id)
        return gr.update(choices=choices, value=preferred)

    def refresh_charts(project_id, run_ids):
        runs = project_runs(project_id)
        selected = [item for item in runs if item["id"] in (run_ids or [])][:5]
        headers = ["parameter", *[item["id"][:8] for item in selected]]
        records = [
            (item["id"][:8], metrics_cache.read(item["metrics_path"]))
            for item in selected
        ]
        figures = (
            training_figures_from_records(records)
            if selected
            else (None, None, None, None)
        )
        checkpoints = [
            [
                run_id,
                event.get("epoch"),
                event.get("kind"),
                event.get("metric", ""),
                event.get("path", ""),
            ]
            for run_id, events in records
            for event in events
            if event.get("phase") == "checkpoint"
        ]
        alerts = [
            [
                run_id,
                event.get("epoch", ""),
                event.get("global_step", ""),
                event.get("alert") or event.get("error"),
            ]
            for run_id, events in records
            for event in events
            if event.get("alert") or event.get("phase") == "fatal"
        ]
        snapshots = training_snapshot_items(records, language)
        if selected and selected[0].get("error"):
            summary = (
                f"**{b('训练失败原因', 'Training failure reason')}：** "
                f"{selected[0]['error']}\n\n"
                "```\n"
                + latest_summary_from_records(records[0][1])
                + "\n```"
            )
        elif selected:
            summary = (
                "```\n"
                + latest_summary_from_records(records[0][1])
                + "\n```"
            )
        else:
            summary = "No run selected."
        return (
            *figures,
            gr.update(headers=headers, value=run_parameter_rows(selected)),
            summary,
            checkpoints,
            alerts,
            snapshots,
        )

    def poll_charts(project_id, run_ids, paused):
        selected = [
            item
            for item in project_runs(project_id)
            if item["id"] in (run_ids or [])
        ][:5]
        if paused or not selected:
            return (*[gr.skip() for _ in range(9)], gr.skip())
        active = any(item["status"] in {"queued", "running"} for item in selected)
        changed_to_terminal = any(
            dashboard_status_cache.get(item["id"]) in {"queued", "running"}
            and item["status"] not in {"queued", "running"}
            for item in selected
        )
        for item in selected:
            dashboard_status_cache[item["id"]] = item["status"]
        if not active and not changed_to_terminal:
            return (*[gr.skip() for _ in range(9)], gr.skip())
        return (
            *refresh_charts(project_id, run_ids),
            generation_checkpoint_choices(project_id),
        )

    def export_run_metrics(project_id, run_ids):
        runs = project_runs(project_id)
        selected = next((item for item in runs if item["id"] in (run_ids or [])), None)
        if not selected:
            raise gr.Error(b("请选择 run", "Select a run"))
        destination = storage.project_dir(project_id) / "exports"
        csv_path = export_metrics_csv(
            selected["metrics_path"], destination / f"{selected['id']}-metrics.csv"
        )
        json_path = export_metrics_json(
            selected["metrics_path"], destination / f"{selected['id']}-metrics.json"
        )
        figure_paths = []
        for name, figure in zip(
            ("loss", "learning-rate", "resources", "evaluation"),
            training_figures([selected["metrics_path"]]),
        ):
            path = destination / f"{selected['id']}-{name}.png"
            figure.savefig(path, dpi=160)
            figure_paths.append(str(path))
        return [str(csv_path), str(json_path), *figure_paths]

    def build_request(
        project_id,
        text,
        preset_ids,
        custom_file,
        split_regions,
        primary_region,
        device,
        generation_checkpoint,
        seed,
        candidates,
        allow_large_candidates,
    ):
        project = storage.get_project(project_id)
        try:
            selected_checkpoint = resolve_generation_checkpoint(
                project,
                storage,
                generation_checkpoint,
            )
        except ValueError as exc:
            raise gr.Error(
                f"{b('生成模型无效', 'Invalid generation model')}: {exc}"
            ) from exc
        requested_codepoints, increments = resolve_selection(
            preset_ids or [],
            custom_text=text or "",
            custom_file=custom_file,
        )
        target_font_path = (
            project.target_assets[0]
            if project.input_mode == "font" and project.target_assets
            else None
        )
        codepoints, skipped_existing = exclude_existing_target_glyphs(
            requested_codepoints, target_font_path
        )
        if int(candidates) > 1 and len(codepoints) > 64 and not allow_large_candidates:
            raise gr.Error(
                b(
                    "候选生成默认最多 64 个字形，请显式解除限制后再继续",
                    "Candidate generation is limited to 64 glyphs unless explicitly unlocked",
                )
            )
        request_id = uuid4().hex[:10]
        if split_regions:
            regional = resolve_selection_by_region(
                preset_ids or [],
                primary_region=primary_region,
                custom_text=text or "",
                custom_file=custom_file,
            )
            skipped_set = set(skipped_existing)
            regional = {
                region: [
                    codepoint
                    for codepoint in region_codepoints
                    if codepoint not in skipped_set
                ]
                for region, region_codepoints in regional.items()
            }
            regional = {
                region: region_codepoints
                for region, region_codepoints in regional.items()
                if region_codepoints
            }
        else:
            regional = {primary_region: codepoints}
        if not regional:
            return (
                "",
                b(
                    f"无需生成：请求的 {len(requested_codepoints)} 个字符均已存在于目标字体中。",
                    f"No generation required: all {len(requested_codepoints)} requested "
                    "characters already exist in the target font.",
                ),
            )
        if len(project.style_reference_pool) < 8:
            raise gr.Error(
                b(
                    "项目至少需要 8 个风格参考",
                    "The project needs at least 8 style references",
                )
            )
        request_dir = (
            storage.project_dir(project_id)
            / "generation"
            / f"request-{request_id}"
        )
        payloads = {}
        fallbacks = []
        try:
            for region, region_codepoints in regional.items():
                source_fonts = project.source_fonts_for_region(region)
                if not source_fonts:
                    raise ValueError(
                        f"No source font is configured for region {region}"
                    )
                if not project.regional_source_fonts.get(region):
                    fallbacks.append(region)
                payloads[region] = build_generation_request(
                    region_codepoints,
                    source_fonts,
                    project.style_reference_pool,
                    project_id=project_id,
                    request_id=request_id,
                    region=region,
                    checkpoint=selected_checkpoint,
                    seed=int(seed),
                    candidates=int(candidates),
                )
        except (OSError, ValueError) as exc:
            raise gr.Error(
                f"{b('生成请求校验失败', 'Generation request validation failed')}: {exc}"
            ) from exc

        request_dir.mkdir(parents=True, exist_ok=False)
        write_selection_manifest(
            request_dir / "charset.json",
            list(preset_ids or []),
            requested_codepoints,
            increments,
        )
        (request_dir / "generation-plan.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "zi2zi-generation-plan",
                    "request_id": request_id,
                    "target_font": (
                        project.target_assets[0]
                        if project.input_mode == "font" and project.target_assets
                        else ""
                    ),
                    "checkpoint": selected_checkpoint,
                    "seed": int(seed),
                    "candidates": int(candidates),
                    "requested_count": len(requested_codepoints),
                    "skipped_existing_count": len(skipped_existing),
                    "generation_count": len(codepoints),
                    "regions": {
                        region: payload["count"]
                        for region, payload in payloads.items()
                    },
                    "skipped_existing": [
                        f"U+{value:04X}" for value in skipped_existing
                    ],
                    "generation_codepoints": [
                        f"U+{value:04X}" for value in codepoints
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        specifications = []
        outputs = []
        for region, payload in payloads.items():
            region_dir = request_dir / region
            request_manifest = write_generation_request(
                payload,
                region_dir / "request.json",
            )
            command, output = generation_command(
                project,
                storage,
                request_manifest,
                device,
                seed=int(seed),
                candidates=int(candidates),
                output_dir=region_dir,
                request_manifest=True,
                checkpoint_path=selected_checkpoint,
            )
            outputs.append(output)
            specifications.append(
                {
                    "job_type": "generation",
                    "command": command,
                    "project_id": project_id,
                    "cwd": ROOT,
                    "gpu": device if str(device).isdigit() else None,
                }
            )
        job_ids = jobs.submit_many(specifications)
        warning = (
            f" Source-font fallback used for {', '.join(sorted(set(fallbacks)))}; "
            "review regional glyph variants."
            if fallbacks
            else ""
        )
        project.active_checkpoint = selected_checkpoint
        project.inference["checkpoint"] = selected_checkpoint
        project.charset_presets = list(preset_ids or [])
        project.split_regions = bool(split_regions)
        project.primary_region = primary_region
        storage.save_project(project)
        return (
            "\n".join(str(item) for item in outputs),
            (
                f"Queued {len(codepoints)} missing glyphs in {len(job_ids)} regional job(s): "
                f"{', '.join(job_ids)}. Skipped {len(skipped_existing)} glyphs already "
                f"present in the target font. Checkpoint: {Path(selected_checkpoint).name}."
                f"{warning}"
            ),
        )

    def generation_batch_options(project_id):
        if not project_id:
            return gr.update(choices=[], value=None)
        choices, preferred = batch_choices(
            storage.project_dir(project_id) / "generation"
        )
        return gr.update(choices=choices, value=preferred)

    def refresh_review_page(project_id, batch_path, search, page):
        if not project_id or not batch_path:
            return [], gr.update(choices=[], value=None), 1, "No generation batch."
        project = storage.get_project(project_id)
        batch = validate_batch_path(
            storage.project_dir(project_id) / "generation",
            batch_path,
        )
        images, codepoints, current_page, total_pages = generation_review_page(
            batch,
            search=str(search or ""),
            page=int(page or 1),
            page_size=100,
            primary_region=project.primary_region,
        )
        choices = [f"U+{codepoint:04X}" for codepoint in codepoints]
        selection_path = (
            storage.project_dir(project_id) / "glyphs" / "selection.json"
        )
        load_and_prune_selections(selection_path)
        return (
            [str(path) for path in images],
            gr.update(choices=choices, value=choices[0] if choices else None),
            current_page,
            (
                f"{len(codepoints)} glyphs on page {current_page}/{total_pages}; "
                "100 glyphs per page."
            ),
        )

    def candidate_gallery(project_id, batch_path, codepoint_label):
        if not codepoint_label:
            return [], gr.update(choices=[])
        batch = validate_batch_path(
            storage.project_dir(project_id) / "generation",
            batch_path,
        )
        images = [
            str(path)
            for path in candidates_for_codepoint(
                batch,
                int(codepoint_label[2:], 16),
            )
        ]
        return images, gr.update(
            choices=images,
            value=images[0] if images else None,
        )

    def save_candidate(project_id, batch_path, codepoint_label, image_path):
        if not codepoint_label or not image_path:
            raise gr.Error(b("请选择字形和候选", "Choose a glyph and candidate"))
        batch = validate_batch_path(
            storage.project_dir(project_id) / "generation",
            batch_path,
        )
        candidate = Path(image_path).resolve()
        if not candidate.is_file() or batch not in candidate.parents:
            raise gr.Error(
                b(
                    "所选候选不属于当前批次",
                    "The selected candidate is not part of this batch",
                )
            )
        path = storage.project_dir(project_id) / "glyphs" / "selection.json"
        values = load_and_prune_selections(path)
        values[codepoint_label] = str(candidate)
        path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"Selected {candidate.name} for {codepoint_label}"

    def clear_candidate(project_id, codepoint_label):
        if not project_id or not codepoint_label:
            raise gr.Error(b("请选择字形", "Choose a glyph"))
        path = storage.project_dir(project_id) / "glyphs" / "selection.json"
        values = load_and_prune_selections(path)
        removed = values.pop(codepoint_label, None)
        path.write_text(
            json.dumps(values, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return (
            f"Cleared selection for {codepoint_label}"
            if removed
            else f"No manual selection existed for {codepoint_label}"
        )

    def queue_font(
        project_id,
        generation_batch,
        family,
        style,
        version,
        designer,
        license_text,
        profiles,
        threshold,
        despeckle,
    ):
        project = storage.get_project(project_id)
        project_dir = storage.project_dir(project_id)
        try:
            validate_font_metadata(
                FontMetadata(
                    family_name=str(family or ""),
                    style_name=str(style or ""),
                    version=str(version or ""),
                )
            )
        except ValueError as exc:
            raise gr.Error(
                f"{b('字体元数据无效', 'Invalid font metadata')}: {exc}"
            ) from exc
        selection = project_dir / "glyphs" / "selection.json"
        if not str(generation_batch or "").strip():
            raise gr.Error(b("请选择生成批次", "Choose a generation batch"))
        try:
            batch = validate_batch_path(
                project_dir / "generation",
                generation_batch,
            )
            selected_values = load_and_prune_selections(selection)
            discovered_glyphs = scan_generation_batch(
                batch,
                primary_region=project.primary_region,
                selections=selected_values,
            )
        except (OSError, ValueError) as exc:
            raise gr.Error(
                f"{b('生成批次校验失败', 'Generation batch validation failed')}: {exc}"
            ) from exc
        effective_glyph_count = len(discovered_glyphs)
        if not effective_glyph_count:
            raise gr.Error(
                b(
                    "未递归找到以 U+XXXX 命名的字形图片。字体导出已停止，"
                    "以避免静默复制未变化的目标字体。",
                    "No U+XXXX-named glyph images were found recursively. "
                    "Font export was stopped to avoid silently copying the target font unchanged.",
                )
            )
        export_selection = project_dir / "glyphs" / f"export-{uuid4().hex}.json"
        export_selection.write_text(
            json.dumps(
                {
                    f"U+{codepoint:04X}": str(path)
                    for codepoint, path in sorted(discovered_glyphs.items())
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        base_font_path = ""
        if project.input_mode == "font":
            if not project.target_assets:
                raise gr.Error(
                    b("项目没有目标字体", "The project does not have a target font")
                )
            base_font_path = project.target_assets[0]
            from fontTools.ttLib import TTFont

            base_font = TTFont(base_font_path)
            try:
                if "glyf" not in base_font or "hmtx" not in base_font:
                    raise gr.Error(
                        b(
                            "增量封装需要 TrueType 轮廓目标字体；不支持 CFF 轮廓 OTF",
                            "Incremental packaging requires a TrueType-outline target; "
                            "CFF-outline OTF fonts are not supported",
                        )
                    )
                if "fvar" in base_font or "gvar" in base_font:
                    raise gr.Error(
                        b(
                            "增量封装不支持可变字体目标",
                            "Incremental packaging does not support variable-font targets",
                        )
                    )
            finally:
                base_font.close()
        job_ids = []
        for profile in profiles or ["proportional"]:
            suffix = "Text" if profile == "proportional" else "Mono"
            fonts_root = (project_dir / "fonts").resolve()
            output = (fonts_root / f"{family}-{suffix}-{version}.ttf").resolve()
            if output.parent != fonts_root:
                raise gr.Error(
                    b(
                        "字体输出路径越出了项目 fonts 目录",
                        "Font output path escaped the project fonts directory",
                    )
                )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_font.py"),
                "--glyph-dir",
                str(batch),
                "--output",
                str(output),
                "--family",
                f"{family} {suffix}",
                "--style",
                style,
                "--version",
                version,
                "--designer",
                designer,
                "--license",
                license_text,
                "--profile",
                profile,
                "--threshold",
                str(int(threshold)),
                "--despeckle-area",
                str(int(despeckle)),
                "--project-dir",
                str(project_dir),
            ]
            if base_font_path:
                command += ["--base-font", base_font_path]
            command += [
                "--selection-manifest",
                str(export_selection),
                "--selection-only",
            ]
            job_ids.append(
                jobs.submit("font_build", command, project_id=project_id, cwd=ROOT)
            )
        return (
            f"Queued font build jobs: {', '.join(job_ids)}. "
            f"Discovered {effective_glyph_count} generated glyphs recursively."
        )

    def refresh_font_artifacts(project_id, sample_text):
        project_dir = storage.project_dir(project_id)
        artifacts = sorted(
            [
                *project_dir.glob("fonts/*.ttf"),
                *project_dir.glob("fonts/*.zip"),
                *project_dir.glob("fonts/*.report.json"),
                *project_dir.glob("fonts/*.report.html"),
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        svgs = sorted(
            project_dir.glob("fonts/*/svg/*.svg"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        fonts = [item for item in artifacts if item.suffix.lower() == ".ttf"]
        if not fonts:
            preview = b("尚未构建字体。", "No font has been built yet.")
        else:
            font_url = "/gradio_api/file=" + quote(str(fonts[0]), safe="")
            preview = (
                "<style>@font-face{font-family:zi2ziPreview;src:url('"
                + font_url
                + "')} .zi2zi-preview{font-family:zi2ziPreview;font-size:48px;"
                "line-height:1.8;overflow-wrap:anywhere}</style>"
                f"<div class='zi2zi-preview'>{html.escape(sample_text or '永字八法 ABC 123')}</div>"
            )
        return [str(item) for item in artifacts], [str(item) for item in svgs[-100:]], preview

    def monitor_jobs(_project_id, selected_id=None, follow_latest=False):
        rows = jobs.list_jobs()
        table = [
            [
                item["id"],
                (item["project_id"] or "")[:8],
                item["job_type"],
                item["status"],
                round(float(item["progress"] or 0) * 100, 1),
                item.get("progress_text") or "",
                item["gpu"] or "",
                item["created_at"],
                item.get("error") or "",
            ]
            for item in rows
        ]
        choices = [
            (
                f"{item['job_type']} · {item['status']} · "
                f"{float(item['progress'] or 0) * 100:.1f}% · {item['id'][:8]}",
                item["id"],
            )
            for item in rows
        ]
        valid_ids = {item["id"] for item in rows}
        selected = (
            rows[0]["id"]
            if rows and follow_latest
            else selected_id
            if selected_id in valid_ids
            else rows[0]["id"]
            if rows
            else None
        )
        if not selected:
            return table, gr.update(choices=[], value=None), 0, "No tasks.", ""
        job = next(item for item in rows if item["id"] == selected)
        reason = job.get("error") or ""
        status = (
            f"**{job['job_type']} · {job['status']}**  \n"
            f"`{job['id']}`  \n"
            f"{job.get('progress_text') or ''}"
        )
        if reason:
            status += f"\n\n**{b('失败原因', 'Failure reason')}：** {reason}"
        log_path = Path(job["log_path"])
        signature = (
            log_path.stat().st_mtime_ns,
            log_path.stat().st_size,
        ) if log_path.is_file() else None
        if (
            job_log_cache["job_id"] == selected
            and job_log_cache["signature"] == signature
        ):
            log_value = gr.skip()
        else:
            job_log_cache.update(job_id=selected, signature=signature)
            log_value = jobs.read_log(selected)
        return (
            table,
            gr.update(choices=choices, value=selected),
            round(float(job["progress"] or 0) * 100, 1),
            status,
            log_value,
        )

    def retry_selected_job(job_id):
        job = jobs.get_job(job_id)
        if job["status"] not in {"failed", "cancelled", "interrupted"}:
            raise gr.Error(
                b(
                    "只能重试失败、已取消或已中断的任务",
                    "Only failed, cancelled, or interrupted jobs can be retried",
                )
            )
        if job["job_type"] != "training":
            return f"Retried as: {jobs.retry(job_id)}"
        command = json.loads(job["command_json"])
        try:
            checkpoint = command[command.index("--base_checkpoint") + 1]
            model_index = command.index("--model") + 1
            data_index = command.index("--data_path") + 1
            test_index = command.index("--test_npz_path") + 1
        except (ValueError, IndexError) as exc:
            raise gr.Error(
                b(
                    "训练命令缺少 checkpoint、模型或数据集参数",
                    "Training command is missing checkpoint/model/dataset arguments",
                )
            ) from exc
        variant = infer_checkpoint_model(checkpoint)
        if not variant:
            raise gr.Error(
                b(
                    "无法推断重试所需的 checkpoint 架构",
                    "Cannot infer checkpoint architecture for retry",
                )
            )
        command[model_index] = variant
        command, safety_changes = clamp_training_retry_command(
            command,
            str(job.get("gpu") or ""),
        )
        requested_dataset = Path(command[data_index]).parent
        project = storage.get_project(job["project_id"])
        dataset = resolve_training_dataset(project, storage, requested_dataset)
        command[data_index] = str(dataset / "train")
        command[test_index] = str(dataset / "test.npz")
        project.training["dataset_path"] = str(dataset)
        storage.save_project(project)
        resume_point = Path(job.get("resume_point") or "")
        if resume_point.is_file() and "--resume" not in command:
            command += ["--resume", str(resume_point)]
        new_id = jobs.submit(
            "training",
            command,
            project_id=job["project_id"],
            cwd=job["cwd"],
            gpu=job["gpu"],
            env=json.loads(job["env_json"]),
            resume_point=job.get("resume_point"),
        )
        storage.replace_training_job(job["project_id"], job_id, new_id)
        safety_note = (
            "; safety limits "
            + ", ".join(
                f"{name} {before}→{after}"
                for name, (before, after) in safety_changes.items()
            )
            if safety_changes
            else ""
        )
        return (
            f"Retried training as {new_id} with architecture {variant}; "
            f"dataset {dataset}{safety_note}"
        )

    initial_project_choices = project_choices()
    initial_project_id = (
        initial_project_choices[0][1] if initial_project_choices else None
    )
    if initial_project_id:
        initial_project = storage.get_project(initial_project_id)
        initial_checkpoint_choices, initial_checkpoint = (
            generation_checkpoint_options(initial_project, storage)
        )
        initial_batch_choices, initial_batch = batch_choices(
            storage.project_dir(initial_project_id) / "generation"
        )
    else:
        initial_checkpoint_choices, initial_checkpoint = [], None
        initial_batch_choices, initial_batch = [], None

    with gr.Blocks(title=t("app_title"), theme=gr.themes.Soft()) as app:
        gr.Markdown(
            f"# {t('app_title')}\n"
            f"{b('本地优先 · CUDA 感知 · 项目目录', 'Local-first · CUDA-aware · Project storage')}: "
            f"`{storage.root}`"
        )
        with gr.Row():
            project_selector = gr.Dropdown(
                choices=initial_project_choices,
                value=initial_project_id,
                label=t("projects"),
                scale=5,
            )
            refresh_projects_btn = gr.Button(t("refresh"), scale=1)
            language_display = gr.Textbox(
                value="English" if language == "en" else "简体中文",
                label=b("当前服务语言", "Current service language"),
                interactive=False,
                scale=1,
            )
        project_json = gr.Code(label=b("项目清单", "Project manifest"), language="json", lines=8)
        with gr.Accordion(b("实时任务监控", "Live task monitor"), open=True):
            jobs_table = gr.Dataframe(
                headers=[
                    "id",
                    "project",
                    "type",
                    "status",
                    "progress %",
                    "activity",
                    "gpu",
                    "created",
                    "error",
                ],
                interactive=False,
            )
            with gr.Row():
                job_selector = gr.Dropdown(
                    label=b("当前任务", "Current task"),
                    allow_custom_value=True,
                    scale=5,
                )
                follow_latest_job = gr.Checkbox(
                    value=False,
                    label=b("自动跟随最新任务", "Follow latest task"),
                    scale=1,
                )
            job_progress = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                interactive=False,
                label=b("进度（%）", "Progress (%)"),
            )
            job_monitor_status = gr.Markdown()
            job_log = gr.Textbox(
                label=b("实时日志", "Live log"),
                lines=16,
                max_lines=32,
                autoscroll=True,
            )

        with gr.Tab(t("projects")):
            with gr.Row():
                new_project_name = gr.Textbox(label=t("project_name"))
                create_project_btn = gr.Button(t("create_project"), variant="primary")
            project_status = gr.Markdown()
            gr.Markdown(
                f"**{b('数据根目录', 'Data root')}:** `{storage.root}` · "
                f"**{b('磁盘可用', 'Free disk')}:** {disk_free_gb(storage.root)} GB"
            )
            with gr.Accordion(b("危险操作", "Danger zone"), open=False):
                gr.Markdown(
                    b(
                        "永久删除当前项目、训练运行、任务记录，以及项目目录中的素材、"
                        "数据集、checkpoint、生成结果和字体。共享模型库不会被删除。"
                        "若项目仍有运行中或排队中的任务，必须先取消。",
                        "Permanently delete the current project, training runs, job records, "
                        "and all project assets, datasets, checkpoints, generated results, "
                        "and fonts. Shared models are retained. Active jobs must be cancelled first.",
                    )
                )
                with gr.Row():
                    confirm_delete_project = gr.Checkbox(
                        label=b(
                            "确认永久删除当前项目及全部文件",
                            "Confirm permanent deletion of the current project and all files",
                        )
                    )
                    delete_project_btn = gr.Button(
                        b("删除项目及文件", "Delete project and files"),
                        variant="stop",
                    )

        with gr.Tab(t("model_assets")):
            gr.Markdown(
                b(
                    "⚠️ 只导入你信任的 checkpoint；PyTorch checkpoint 可能包含可执行对象。",
                    "⚠️ Only import checkpoints you trust. PyTorch checkpoints may contain executable objects.",
                )
            )
            with gr.Row():
                official_model = gr.Dropdown(
                    choices=["JiT-B/16", "JiT-L/16"], value="JiT-B/16",
                    label=b("官方模型", "Official model"),
                )
                download_model_btn = gr.Button(b("下载官方模型", "Download official model"))
            model_download_status = gr.Markdown()
            with gr.Row():
                shared_model = gr.Dropdown(label=b("共享模型库", "Shared model library"))
                refresh_shared_models_btn = gr.Button(b("刷新模型库", "Refresh model library"))
                attach_shared_model_btn = gr.Button(b("关联共享模型", "Attach shared model"))
            shared_model_status = gr.Markdown()
            with gr.Row():
                checkpoint_upload = gr.File(label=b("本地 checkpoint", "Local checkpoint"), type="filepath")
                checkpoint_trusted = gr.Checkbox(label=b("我信任此 checkpoint", "I trust this checkpoint"))
                attach_model_btn = gr.Button(b("导入并关联", "Import and attach"))
            checkpoint_status = gr.Code(label=b("Checkpoint 元数据", "Checkpoint metadata"), language="json")
            input_mode = gr.Radio(
                choices=[
                    (b("目标字体", "Target font"), "font"),
                    (b("已渲染字形", "Rendered glyphs"), "glyphs"),
                ],
                value="font",
                label=b("目标风格素材", "Target style source"),
            )
            source_font = gr.File(
                label=b(
                    "全局内容/源字体（按顺序回退，可多选）",
                    "Global content/source fonts (ordered fallback)",
                ),
                type="filepath",
                file_count="multiple",
            )
            target_files = gr.File(label=b("目标字体或字形图片", "Target font or glyph images"), type="filepath", file_count="multiple")
            with gr.Accordion(b("地区源字体（可选）", "Regional source fonts (optional)"), open=False):
                with gr.Row():
                    sc_font = gr.File(label="SC", type="filepath", file_count="multiple")
                    tc_font = gr.File(label="TC", type="filepath", file_count="multiple")
                    jp_font = gr.File(label="JP", type="filepath", file_count="multiple")
                    kr_font = gr.File(label="KR", type="filepath", file_count="multiple")
            save_assets_btn = gr.Button(b("保存项目素材", "Save project assets"), variant="primary")
            asset_status = gr.Code(label=b("素材状态", "Asset status"), language="json")

        with gr.Tab(t("dataset")):
            gr.Markdown(
                b(
                    "默认自动扫描有真实轮廓的公共 CJK 字形：源字体集合中任一字体可提供、"
                    "且目标字体也包含的字形会进入候选池，再以固定 seed 拆分为训练集和"
                    "不重叠的验证集。无需先指定字符集。",
                    "By default, the app detects outlined CJK glyphs shared by the ordered "
                    "source-font collection and target font, then uses fixed seeds to create "
                    "disjoint training and validation splits. No charset selection is required.",
                )
            )
            dataset_scale = gr.Dropdown(
                choices=[
                    (b("快速验证：500 / 8", "Quick check: 500 / 8"), "quick"),
                    (b("均衡训练：3000 / 64", "Balanced: 3000 / 64"), "balanced"),
                    (b("高覆盖：6000 / 128", "High coverage: 6000 / 128"), "coverage"),
                ],
                value="balanced",
                label=b("数据集规模预设（训练 / 验证）", "Dataset size preset (train / validation)"),
            )
            with gr.Row():
                train_count = gr.Number(value=3000, precision=0, label=b("训练字形数", "Training glyphs"))
                test_count = gr.Number(value=64, precision=0, label=b("验证字形数", "Validation glyphs"))
                workers = gr.Number(value=4, precision=0, label=b("工作进程", "Workers"))
            with gr.Accordion(
                b("高级：可选字符范围过滤", "Advanced: optional character-range filter"),
                open=False,
            ):
                dataset_charset = gr.Dropdown(
                    choices=[
                        (b("自动检测全部公共字形（推荐）", "Auto-detect all common glyphs (recommended)"), "auto"),
                        ("GB2312", "gb2312"),
                        ("GBK", "gbk"),
                        ("Big5", "big5"),
                        ("JIS X 0208", "jisx0208"),
                        ("KS X 1001", "ksx1001"),
                        (b("全部 CJK 范围（兼容旧项目）", "All CJK ranges (legacy alias)"), "all-cjk"),
                    ],
                    value="auto",
                    label=b("字形范围过滤", "Glyph range filter"),
                )
                gr.Markdown(
                    b(
                        "过滤器只用于限制候选范围；无论选择哪项，实际样本仍必须同时存在于"
                        "源字体集合和目标字体，并包含有效轮廓。",
                        "A filter only narrows the candidate range. Every sample must still "
                        "have a valid outline in both the source-font collection and target font.",
                    )
                )
            analyze_capacity_btn = gr.Button(
                b("分析当前字体可用容量", "Analyze usable glyph capacity")
            )
            dataset_capacity_status = gr.Markdown()
            gr.Markdown(
                b(
                    "样本数应增加“不同字符”，而不是复制同一图片。训练会自动使用数据集"
                    "实际生成的全部字形；训练耗时约随样本数线性增加。",
                    "Increase distinct glyphs rather than duplicating images. Training now uses "
                    "all glyphs actually generated; runtime grows roughly linearly with sample count.",
                )
            )
            queue_dataset_btn = gr.Button(b("构建数据集", "Build dataset"), variant="primary")
            dataset_path = gr.Textbox(label=b("数据集路径", "Dataset path"))
            dataset_status = gr.Markdown()

        with gr.Tab(t("training")):
            with gr.Row():
                training_device = gr.Dropdown(
                    choices=device_choices,
                    value=next((item.id for item in devices if item.training_supported), "cpu"),
                    label=b("训练 GPU", "Training GPU"),
                )
                quality = gr.Dropdown(
                    choices=[("Economy", "economy"), ("Balanced", "balanced"), ("Quality", "quality")],
                    value="balanced",
                    label=b("训练预设", "Preset"),
                )
                epochs = gr.Number(value=200, precision=0, label="Epoch")
            with gr.Accordion(
                b("高级训练设置", "Advanced training settings"),
                open=False,
            ):
                horizontal_flip_prob = gr.Slider(
                    0.0,
                    0.5,
                    value=0.0,
                    step=0.05,
                    label=b(
                        "水平镜像概率（汉字通常保持 0）",
                        "Horizontal flip probability (normally 0 for CJK)",
                    ),
                )
            queue_training_btn = gr.Button(b("开始 LoRA 训练", "Start LoRA training"), variant="primary")
            resume_training_btn = gr.Button(
                b("从所选 run 的 last checkpoint 续训", "Resume selected run from last checkpoint")
            )
            training_run_id = gr.Textbox(label="Run ID")
            training_status = gr.Markdown()
            run_selector = gr.Dropdown(
                label=b("对比 run（2–5 个）", "Compare runs (2–5)"),
                multiselect=True,
                allow_custom_value=True,
                max_choices=5,
            )
            with gr.Row():
                refresh_runs_btn = gr.Button(b("刷新 run", "Refresh runs"))
                refresh_charts_btn = gr.Button(b("刷新图表", "Refresh charts"))
                export_metrics_btn = gr.Button(b("导出 CSV / JSON / 图表", "Export CSV / JSON / charts"))
                pause_dashboard = gr.Checkbox(
                    value=False,
                    label=b("暂停自动刷新", "Pause auto-refresh"),
                )
            with gr.Row():
                confirm_delete_run = gr.Checkbox(
                    label=b("确认永久删除所选 run", "Confirm permanent deletion of selected run")
                )
                delete_run_btn = gr.Button(b("删除所选 run", "Delete selected run"))
            exported_metrics = gr.File(label=b("指标导出", "Metrics export"), file_count="multiple")
            dashboard_summary = gr.Markdown(b("未选择 run。", "No run selected."))
            with gr.Row():
                loss_plot = gr.Plot(label="Loss")
                lr_plot = gr.Plot(label=b("学习率", "Learning rate"))
            with gr.Row():
                resource_plot = gr.Plot(label=b("资源", "Resources"))
                evaluation_plot = gr.Plot(label=b("评估", "Evaluation"))
            parameter_table = gr.Dataframe(label=b("Run 参数对比", "Run parameter comparison"), interactive=False)
            checkpoint_table = gr.Dataframe(
                headers=["run", "epoch", "kind", "metric", "path"],
                label=b("Checkpoint 时间轴", "Checkpoint timeline"),
                interactive=False,
            )
            alert_table = gr.Dataframe(
                headers=["run", "epoch", "step", "alert"],
                label=b("告警", "Alerts"),
                interactive=False,
            )
            gr.Markdown(
                b(
                    "使用相同验证字符和 Seed 观察训练进展；对比图中每一对左侧为目标真值，"
                    "右侧为该轮模型的生成结果。",
                    "The same validation glyphs and seed are reused across epochs. In each pair, "
                    "the target is on the left and that epoch's generated glyph is on the right.",
                )
            )
            snapshot_gallery = gr.Gallery(
                label=b(
                    "训练字形演变（固定字符与 Seed）",
                    "Training glyph evolution (fixed glyphs and seed)",
                ),
                columns=2,
                height="auto",
                object_fit="contain",
            )
            gr.HTML(
                f'<a href="{tensorboard_url}" target="_blank">'
                f"{b('打开项目 TensorBoard', 'Open project TensorBoard')} ↗</a>"
            )

        with gr.Tab(t("generation")):
            gr.Markdown(
                b(
                    "目标字体中已有有效轮廓的字符会自动跳过，只生成缺失字形；"
                    "请求、跳过和生成清单会写入 generation-plan.json。",
                    "Characters already backed by valid outlines in the target font are "
                    "skipped automatically; only missing glyphs are generated. The request, "
                    "skipped, and generation lists are recorded in generation-plan.json.",
                )
            )
            generation_text = gr.Textbox(
                label=b("自定义文字", "Custom text"), lines=4,
                placeholder=b("输入需要生成的字符…", "Enter characters to generate…"),
            )
            generation_presets = gr.CheckboxGroup(
                choices=preset_choices,
                value=["latin-extended", "cjk-punctuation", "zh-Hans-6500"],
                label=b(
                    "字符预设（多选；选项含来源与说明）",
                    "Character presets (multi-select; each option includes provenance)",
                ),
            )
            custom_charset_file = gr.File(label=b("自定义 U+XXXX/文本清单", "Custom U+XXXX/text list"), type="filepath")
            with gr.Row():
                split_regions = gr.Checkbox(
                    value=True, label=b("自动拆分 SC / TC / JP / KR", "Auto-split SC / TC / JP / KR")
                )
                primary_region = gr.Dropdown(
                    choices=["SC", "TC", "JP", "KR"],
                    value="SC",
                    label=b("拉丁、符号和自定义文本的主地区", "Primary region for Latin, symbols, and custom text"),
                )
            with gr.Row():
                generation_checkpoint = gr.Dropdown(
                    choices=initial_checkpoint_choices,
                    value=initial_checkpoint,
                    label=b(
                        "生成模型（LoRA / checkpoint）",
                        "Generation model (LoRA / checkpoint)",
                    ),
                    info=b(
                        "首次默认推荐最新训练 run 的 best-SSIM，之后记住所选模型；"
                        "也可选择 best-LPIPS、last 或基础模型。",
                        "The latest run's best-SSIM is initially recommended and later "
                        "selections are remembered; best-LPIPS, last, and the base model "
                        "remain selectable.",
                    ),
                    allow_custom_value=True,
                )
                refresh_generation_checkpoints_btn = gr.Button(
                    b("刷新训练模型", "Refresh trained models")
                )
            with gr.Row():
                generation_device = gr.Dropdown(
                    choices=device_choices, value=device_choices[0][1], label=b("设备", "Device")
                )
                generation_seed = gr.Number(value=42, precision=0, label="Seed")
                candidate_count = gr.Slider(1, 8, value=1, step=1, label=b("每字候选数", "Candidates per glyph"))
                unlock_candidates = gr.Checkbox(label=b("解除 64 字候选生成限制", "Unlock 64-glyph candidate limit"))
            queue_generation_btn = gr.Button(b("构建请求并生成", "Build request and generate"), variant="primary")
            generation_output = gr.Textbox(label=b("生成输出目录", "Generation output root"))
            generation_status = gr.Markdown()

        with gr.Tab(t("review")):
            with gr.Row():
                review_batch = gr.Dropdown(
                    choices=initial_batch_choices,
                    value=initial_batch,
                    label=b("生成批次", "Generation batch"),
                    allow_custom_value=True,
                )
                refresh_review_batches_btn = gr.Button(
                    b("刷新批次", "Refresh batches")
                )
            with gr.Row():
                review_search = gr.Textbox(
                    label=b("搜索字符或码位", "Search character or codepoint")
                )
                review_page_number = gr.Number(
                    value=1,
                    precision=0,
                    minimum=1,
                    label=b("页码", "Page"),
                )
                refresh_gallery_btn = gr.Button(
                    b("刷新当前页", "Refresh current page")
                )
            review_page_status = gr.Markdown()
            generated_gallery = gr.Gallery(label=b("生成字形", "Generated glyphs"), columns=8, height=500)
            with gr.Row():
                review_codepoint = gr.Dropdown(label=b("码位", "Codepoint"))
                review_candidate = gr.Dropdown(label=b("所选候选文件", "Selected candidate file"))
            candidate_gallery_component = gr.Gallery(label=b("候选", "Candidates"), columns=4, height=300)
            with gr.Row():
                save_candidate_btn = gr.Button(b("采用此候选", "Use this candidate"))
                clear_candidate_btn = gr.Button(
                    b("清除人工选择", "Clear manual selection")
                )
            candidate_status = gr.Markdown()

        with gr.Tab(t("export")):
            gr.Markdown(
                b(
                    "字体导出以目标 TTF 为底稿，保留原字形、字宽和 OpenType 表，仅追加"
                    "生成的缺失字形。当前不支持将 CFF 轮廓 OTF 或可变字体作为增量封装底稿。",
                    "Export starts from the target TTF, retaining its original glyphs, metrics, "
                    "and OpenType tables while adding only generated missing glyphs. "
                    "CFF-outline OTF and variable-font bases are not supported for "
                    "incremental packaging.",
                )
            )
            with gr.Row():
                export_batch = gr.Dropdown(
                    choices=initial_batch_choices,
                    value=initial_batch,
                    label=b("用于导出的生成批次", "Generation batch to export"),
                    allow_custom_value=True,
                )
                refresh_export_batches_btn = gr.Button(
                    b("刷新批次", "Refresh batches")
                )
            with gr.Row():
                family_name = gr.Textbox(label=b("字体家族名", "Family name"))
                style_name = gr.Textbox(value="Regular", label=b("样式", "Style"))
                font_version = gr.Textbox(value="1.000", label=b("版本", "Version"))
            designer = gr.Textbox(label=b("设计者", "Designer"))
            license_text = gr.Textbox(label=b("附加许可证文本", "Additional license text"), lines=3)
            metric_profiles = gr.CheckboxGroup(
                choices=[
                    (
                        b("CJK 全角 + 拉丁比例宽度", "CJK full-width + proportional Latin"),
                        "proportional",
                    ),
                    (
                        b("CJK/拉丁 2:1 等宽", "2:1 CJK/Latin monospace"),
                        "monospace-2to1",
                    ),
                ],
                value=["proportional"],
                label=b("输出指标配置", "Output profiles"),
            )
            with gr.Accordion(b("矢量化", "Vectorization"), open=False):
                threshold = gr.Slider(1, 254, value=180, step=1, label=b("阈值", "Threshold"))
                despeckle = gr.Number(value=4, precision=0, label=b("最小连通域面积", "Minimum component area"))
            build_font_btn = gr.Button(b("加入 TTF 构建队列", "Queue TTF build"), variant="primary")
            font_status = gr.Markdown()
            with gr.Row():
                preview_text = gr.Textbox(
                    value=b("永字八法 天地玄黄 ABC 123", "The quick brown fox ABC 123"),
                    label=b("字体预览文字", "Font preview text"),
                )
                refresh_fonts_btn = gr.Button(b("刷新字体产物", "Refresh font artifacts"))
            font_artifacts = gr.File(
                label=b("字体与报告", "Fonts and reports"), file_count="multiple"
            )
            svg_gallery = gr.Gallery(label=b("SVG 预览", "SVG preview"), columns=8, height=320)
            final_font_preview = gr.HTML()

        with gr.Tab(t("advanced")):
            refresh_jobs_btn = gr.Button(b("刷新任务", "Refresh jobs"))
            with gr.Row():
                refresh_log_btn = gr.Button(b("刷新日志", "Refresh log"))
                cancel_job_btn = gr.Button(b("取消", "Cancel"))
                retry_job_btn = gr.Button(b("重试", "Retry"))
            job_action_status = gr.Markdown()

        create_project_btn.click(
            create_project, new_project_name, [project_selector, project_status]
        )
        delete_project_btn.click(
            delete_project_action,
            [project_selector, confirm_delete_project],
            [
                project_selector,
                project_json,
                project_status,
                confirm_delete_project,
            ],
        )
        refresh_projects_btn.click(refresh_projects, outputs=project_selector)
        project_selector.change(project_summary, project_selector, project_json)
        project_selector.change(project_dataset_path, project_selector, dataset_path)
        project_selector.change(run_choices, project_selector, run_selector)
        project_selector.change(
            generation_checkpoint_choices,
            project_selector,
            generation_checkpoint,
        )
        project_selector.change(
            generation_batch_options,
            project_selector,
            review_batch,
        )
        project_selector.change(
            generation_batch_options,
            project_selector,
            export_batch,
        )
        project_selector.change(
            lambda: False,
            outputs=confirm_delete_project,
        )
        save_assets_btn.click(
            save_assets,
            [project_selector, input_mode, source_font, target_files, sc_font, tc_font, jp_font, kr_font],
            asset_status,
        )
        attach_model_btn.click(
            attach_model,
            [project_selector, checkpoint_upload, checkpoint_trusted],
            checkpoint_status,
        )
        download_model_btn.click(download_model, official_model, model_download_status)
        refresh_shared_models_btn.click(shared_model_choices, outputs=shared_model)
        attach_shared_model_btn.click(
            attach_shared_model,
            [project_selector, shared_model],
            shared_model_status,
        )
        dataset_scale.change(
            apply_dataset_size_preset,
            dataset_scale,
            [train_count, test_count],
        )
        analyze_capacity_btn.click(
            analyze_dataset_capacity,
            [project_selector, dataset_charset, test_count],
            dataset_capacity_status,
        )
        queue_dataset_btn.click(
            queue_dataset,
            [project_selector, train_count, test_count, dataset_charset, workers],
            [dataset_path, dataset_status],
        )
        queue_training_btn.click(
            queue_training,
            [
                project_selector,
                dataset_path,
                training_device,
                quality,
                epochs,
                horizontal_flip_prob,
            ],
            [training_run_id, training_status],
        )
        resume_training_btn.click(
            resume_training,
            [
                project_selector,
                run_selector,
                dataset_path,
                training_device,
                epochs,
                horizontal_flip_prob,
            ],
            [training_run_id, training_status],
        )
        delete_run_btn.click(
            delete_training,
            [project_selector, run_selector, confirm_delete_run],
            [run_selector, training_status],
        )
        refresh_runs_btn.click(run_choices, project_selector, run_selector)
        refresh_generation_checkpoints_btn.click(
            generation_checkpoint_choices,
            project_selector,
            generation_checkpoint,
        )
        refresh_charts_btn.click(
            refresh_charts,
            [project_selector, run_selector],
            [
                loss_plot,
                lr_plot,
                resource_plot,
                evaluation_plot,
                parameter_table,
                dashboard_summary,
                checkpoint_table,
                alert_table,
                snapshot_gallery,
            ],
        )
        dashboard_timer = gr.Timer(10)
        dashboard_timer.tick(
            poll_charts,
            [project_selector, run_selector, pause_dashboard],
            [
                loss_plot,
                lr_plot,
                resource_plot,
                evaluation_plot,
                parameter_table,
                dashboard_summary,
                checkpoint_table,
                alert_table,
                snapshot_gallery,
                generation_checkpoint,
            ],
        )
        export_metrics_btn.click(
            export_run_metrics, [project_selector, run_selector], exported_metrics
        )
        queue_generation_btn.click(
            build_request,
            [
                project_selector,
                generation_text,
                generation_presets,
                custom_charset_file,
                split_regions,
                primary_region,
                generation_device,
                generation_checkpoint,
                generation_seed,
                candidate_count,
                unlock_candidates,
            ],
            [generation_output, generation_status],
        )
        refresh_review_batches_btn.click(
            generation_batch_options,
            project_selector,
            review_batch,
        )
        refresh_export_batches_btn.click(
            generation_batch_options,
            project_selector,
            export_batch,
        )
        refresh_gallery_btn.click(
            refresh_review_page,
            [project_selector, review_batch, review_search, review_page_number],
            [
                generated_gallery,
                review_codepoint,
                review_page_number,
                review_page_status,
            ],
        )
        review_batch.change(
            refresh_review_page,
            [project_selector, review_batch, review_search, review_page_number],
            [
                generated_gallery,
                review_codepoint,
                review_page_number,
                review_page_status,
            ],
        )
        review_codepoint.change(
            candidate_gallery,
            [project_selector, review_batch, review_codepoint],
            [candidate_gallery_component, review_candidate],
        )
        save_candidate_btn.click(
            save_candidate,
            [
                project_selector,
                review_batch,
                review_codepoint,
                review_candidate,
            ],
            candidate_status,
        )
        clear_candidate_btn.click(
            clear_candidate,
            [project_selector, review_codepoint],
            candidate_status,
        )
        build_font_btn.click(
            queue_font,
            [
                project_selector,
                export_batch,
                family_name,
                style_name,
                font_version,
                designer,
                license_text,
                metric_profiles,
                threshold,
                despeckle,
            ],
            font_status,
        )
        refresh_fonts_btn.click(
            refresh_font_artifacts,
            [project_selector, preview_text],
            [font_artifacts, svg_gallery, final_font_preview],
        )
        refresh_jobs_btn.click(
            monitor_jobs,
            [project_selector, job_selector, follow_latest_job],
            [jobs_table, job_selector, job_progress, job_monitor_status, job_log],
        )
        refresh_log_btn.click(
            monitor_jobs,
            [project_selector, job_selector, follow_latest_job],
            [jobs_table, job_selector, job_progress, job_monitor_status, job_log],
        )
        job_selection = job_selector.input(
            lambda: False,
            outputs=follow_latest_job,
        )
        job_selection.then(
            monitor_jobs,
            [project_selector, job_selector, follow_latest_job],
            [jobs_table, job_selector, job_progress, job_monitor_status, job_log],
        )
        job_timer = gr.Timer(1)
        job_timer.tick(
            monitor_jobs,
            [project_selector, job_selector, follow_latest_job],
            [jobs_table, job_selector, job_progress, job_monitor_status, job_log],
        )
        cancel_job_btn.click(
            lambda job_id: f"Cancelled: {jobs.cancel(job_id)}",
            job_selector,
            job_action_status,
        )
        retry_job_btn.click(
            retry_selected_job,
            job_selector,
            job_action_status,
        )

        def initialize_app(project_id, selected_job_id):
            if not project_id:
                choices = project_choices()
                project_id = choices[0][1] if choices else None
            if project_id:
                manifest = project_summary(project_id)
                dataset = project_dataset_path(project_id)
                runs = run_choices(project_id)
                checkpoints = generation_checkpoint_choices(project_id)
                batches = generation_batch_options(project_id)
            else:
                manifest = "{}"
                dataset = ""
                runs = gr.update(choices=[], value=[])
                checkpoints = gr.update(choices=[], value=None)
                batches = gr.update(choices=[], value=None)
            return (
                manifest,
                dataset,
                runs,
                checkpoints,
                batches,
                batches,
                *monitor_jobs(project_id, selected_job_id, False),
            )

        app.load(
            initialize_app,
            [project_selector, job_selector],
            [
                project_json,
                dataset_path,
                run_selector,
                generation_checkpoint,
                review_batch,
                export_batch,
                jobs_table,
                job_selector,
                job_progress,
                job_monitor_status,
                job_log,
            ],
        )

    return app
