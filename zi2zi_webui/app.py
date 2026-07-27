from __future__ import annotations

import json
import html
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib.parse import quote

from .charts import latest_summary, run_parameter_rows, training_figures
from .devices import detect_devices, disk_free_gb
from .font_builder import codepoint_from_filename
from .i18n import translator
from .inference import (
    build_inference_npz,
    choose_diverse_references,
    render_style_references,
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
    copy_project_input,
    dataset_command,
    generation_command,
    import_model,
    queue_official_model_download,
    training_command,
)
from .storage import Storage
from .telemetry import export_metrics_csv, export_metrics_json, read_metrics


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
            raise gr.Error("Select a project first")
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
            candidates = []
            for asset in project.target_assets:
                path = Path(asset)
                if path.is_dir():
                    candidates.extend(
                        item
                        for item in path.iterdir()
                        if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    )
                elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    candidates.append(path)
            project.style_reference_pool = choose_diverse_references(candidates, 8)
            if len(project.style_reference_pool) < 8:
                raise gr.Error(
                    "At least 8 valid target glyph images are required for style references"
                )
        storage.save_project(project)
        return "Assets saved.\n" + project_summary(project_id)

    def attach_model(project_id, upload, trusted):
        if not project_id:
            raise gr.Error("Select a project first")
        if not upload:
            raise gr.Error("Choose a checkpoint")
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
            raise gr.Error("Select a project and shared model")
        project = storage.get_project(project_id)
        project.base_model = model_path
        project.active_checkpoint = model_path
        storage.save_project(project)
        return f"Attached shared model: {model_path}"

    def queue_dataset(project_id, train_count, test_count, charset, workers):
        project = storage.get_project(project_id)
        command, output = dataset_command(
            project,
            storage,
            train_count=int(train_count),
            test_count=int(test_count),
            charset=charset,
            workers=int(workers),
        )
        job_id = jobs.submit("dataset", command, project_id=project_id, cwd=ROOT)
        return str(output), f"Queued dataset job: {job_id}"

    def queue_training(project_id, dataset_path, device, quality, epochs):
        project = storage.get_project(project_id)
        if not project.base_model:
            raise gr.Error("Import or download and attach a base model first")
        metadata_path = Path(project.base_model).with_suffix(Path(project.base_model).suffix + ".json")
        overrides: dict[str, Any] = {"epochs": int(epochs)}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("num_fonts", "num_chars"):
                if metadata.get(key):
                    overrides[key] = metadata[key]
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
        return run.id, f"Queued training run {run.id}; job {job_id}"

    def resume_training(project_id, run_ids, dataset_path, device, epochs):
        selected_id = (run_ids or [None])[0]
        parent = next(
            (item for item in project_runs(project_id) if item["id"] == selected_id),
            None,
        )
        if not parent:
            raise gr.Error("Select a training run to resume")
        checkpoint = (
            storage.project_dir(project_id)
            / "training"
            / parent["id"]
            / "checkpoint-last.pth"
        )
        if not checkpoint.is_file():
            raise gr.Error("The selected run has no checkpoint-last.pth")
        source_dataset = parent.get("parameters", {}).get("dataset_path") or dataset_path
        if not source_dataset:
            raise gr.Error("The original dataset path is unavailable")
        overrides = dict(parent.get("parameters", {}))
        overrides["epochs"] = int(epochs)
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
            raise gr.Error("Select a training run")
        if not confirmed:
            raise gr.Error("Confirm permanent deletion first")
        selected = next(
            (item for item in project_runs(project_id) if item["id"] == selected_id),
            None,
        )
        if not selected:
            raise gr.Error("Training run was not found")
        if selected.get("status") in {"queued", "running"}:
            raise gr.Error("Cancel the active job before deleting its run")
        storage.delete_training_run(project_id, selected_id)
        return run_choices(project_id), f"Deleted training run {selected_id}"

    def project_runs(project_id):
        runs = storage.list_training_runs(project_id) if project_id else []
        for item in runs:
            job_id = item.get("parameters", {}).get("job_id")
            if job_id:
                try:
                    item["status"] = jobs.get_job(job_id)["status"]
                except KeyError:
                    pass
        return runs

    def run_choices(project_id):
        runs = project_runs(project_id)
        choices = [(f"{item['id'][:8]} · {item['status']}", item["id"]) for item in runs]
        return gr.update(choices=choices, value=[item[1] for item in choices[:1]])

    def refresh_charts(project_id, run_ids):
        runs = project_runs(project_id)
        selected = [item for item in runs if item["id"] in (run_ids or [])][:5]
        figures = training_figures([item["metrics_path"] for item in selected])
        headers = ["parameter", *[item["id"][:8] for item in selected]]
        records = [
            (item["id"][:8], read_metrics(item["metrics_path"]))
            for item in selected
        ]
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
        snapshots = [
            str(path)
            for _run_id, events in records
            for event in events
            if event.get("phase") == "evaluation"
            for path in Path(event.get("snapshot_path", "")).glob("*.png")
        ]
        summary = (
            "```\n" + latest_summary(selected[0]["metrics_path"]) + "\n```"
            if selected
            else "No run selected."
        )
        return (
            *figures,
            gr.update(headers=headers, value=run_parameter_rows(selected)),
            summary,
            checkpoints,
            alerts,
            snapshots[-24:],
        )

    def export_run_metrics(project_id, run_ids):
        runs = project_runs(project_id)
        selected = next((item for item in runs if item["id"] in (run_ids or [])), None)
        if not selected:
            raise gr.Error("Select a run")
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
        seed,
        candidates,
        allow_large_candidates,
    ):
        project = storage.get_project(project_id)
        codepoints, increments = resolve_selection(
            preset_ids or [],
            custom_text=text or "",
            custom_file=custom_file,
        )
        if int(candidates) > 1 and len(codepoints) > 64 and not allow_large_candidates:
            raise gr.Error("Candidate generation is limited to 64 glyphs unless explicitly unlocked")
        if not project.style_reference_pool:
            raise gr.Error("The project needs at least one style reference")
        project.charset_presets = list(preset_ids or [])
        project.split_regions = bool(split_regions)
        project.primary_region = primary_region
        storage.save_project(project)
        request_id = uuid4().hex[:10]
        request_dir = storage.project_dir(project_id) / "generation" / f"request-{request_id}"
        request_dir.mkdir(parents=True, exist_ok=True)
        write_selection_manifest(
            request_dir / "charset.json", list(preset_ids or []), codepoints, increments
        )
        if split_regions:
            regional = resolve_selection_by_region(
                preset_ids or [],
                primary_region=primary_region,
                custom_text=text or "",
                custom_file=custom_file,
            )
        else:
            regional = {primary_region: codepoints}
        job_ids = []
        outputs = []
        fallbacks = []
        for region, region_codepoints in regional.items():
            source_fonts = project.source_fonts_for_region(region)
            if not source_fonts:
                raise gr.Error(f"No source font is configured for region {region}")
            if not project.regional_source_fonts.get(region):
                fallbacks.append(region)
            region_dir = request_dir / region
            npz = build_inference_npz(
                region_codepoints,
                source_fonts,
                project.style_reference_pool,
                region_dir / "request.npz",
                seed=int(seed),
            )
            command, output = generation_command(
                project,
                storage,
                npz,
                device,
                seed=int(seed),
                candidates=int(candidates),
                output_name=f"{request_id}-{region}-seed-{int(seed)}",
            )
            outputs.append(output)
            job_ids.append(
                jobs.submit(
                    "generation",
                    command,
                    project_id=project_id,
                    cwd=ROOT,
                    gpu=device if str(device).isdigit() else None,
                )
            )
        warning = (
            f" Source-font fallback used for {', '.join(sorted(set(fallbacks)))}; "
            "review regional glyph variants."
            if fallbacks
            else ""
        )
        return (
            "\n".join(str(item) for item in outputs),
            f"Queued {len(codepoints)} unique glyphs in {len(job_ids)} regional job(s): "
            f"{', '.join(job_ids)}.{warning}",
        )

    def find_generated(project_id):
        root = storage.project_dir(project_id) / "generation"
        images = sorted(root.rglob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)
        return [str(item) for item in images[:500]]

    def candidate_groups(project_id):
        groups: dict[str, list[str]] = {}
        for image in find_generated(project_id):
            codepoint = codepoint_from_filename(image)
            if codepoint is not None:
                groups.setdefault(f"U+{codepoint:04X}", []).append(image)
        choices = sorted(groups)
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def candidate_gallery(project_id, codepoint_label):
        if not codepoint_label:
            return [], gr.update(choices=[])
        images = [
            item
            for item in find_generated(project_id)
            if codepoint_from_filename(item) == int(codepoint_label[2:], 16)
        ]
        return images, gr.update(choices=images, value=images[0] if images else None)

    def save_candidate(project_id, codepoint_label, image_path):
        if not codepoint_label or not image_path:
            raise gr.Error("Choose a glyph and candidate")
        path = storage.project_dir(project_id) / "glyphs" / "selection.json"
        values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        values[codepoint_label] = image_path
        path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"Selected {Path(image_path).name} for {codepoint_label}"

    def queue_font(
        project_id,
        glyph_dir,
        family,
        style,
        version,
        designer,
        license_text,
        profiles,
        threshold,
        despeckle,
    ):
        project_dir = storage.project_dir(project_id)
        selection = project_dir / "glyphs" / "selection.json"
        job_ids = []
        for profile in profiles or ["proportional"]:
            suffix = "Text" if profile == "proportional" else "Mono"
            output = project_dir / "fonts" / f"{family}-{suffix}-{version}.ttf"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "build_font.py"),
                "--glyph-dir",
                glyph_dir,
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
            if selection.exists():
                command += ["--selection-manifest", str(selection)]
            job_ids.append(
                jobs.submit("font_build", command, project_id=project_id, cwd=ROOT)
            )
        return f"Queued font build jobs: {', '.join(job_ids)}"

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

    def job_table(project_id):
        rows = jobs.list_jobs(project_id)
        table = [
            [
                item["id"],
                item["job_type"],
                item["status"],
                item["gpu"] or "",
                item["created_at"],
                item["return_code"],
            ]
            for item in rows
        ]
        choices = [(f"{item['job_type']} · {item['status']} · {item['id'][:8]}", item["id"]) for item in rows]
        return table, gr.update(choices=choices, value=choices[0][1] if choices else None)

    with gr.Blocks(title=t("app_title"), theme=gr.themes.Soft()) as app:
        gr.Markdown(
            f"# {t('app_title')}\n"
            f"{b('本地优先 · CUDA 感知 · 项目目录', 'Local-first · CUDA-aware · Project storage')}: "
            f"`{storage.root}`"
        )
        with gr.Row():
            project_selector = gr.Dropdown(
                choices=project_choices(), label=t("projects"), scale=5
            )
            refresh_projects_btn = gr.Button(t("refresh"), scale=1)
            language_display = gr.Dropdown(
                choices=[("简体中文", "zh"), ("English", "en")],
                value=language,
                label="Language / 语言",
                interactive=False,
                scale=1,
            )
        project_json = gr.Code(label=b("项目清单", "Project manifest"), language="json", lines=8)

        with gr.Tab(t("projects")):
            with gr.Row():
                new_project_name = gr.Textbox(label=t("project_name"))
                create_project_btn = gr.Button(t("create_project"), variant="primary")
            project_status = gr.Markdown()
            gr.Markdown(
                f"**{b('数据根目录', 'Data root')}:** `{storage.root}` · "
                f"**{b('磁盘可用', 'Free disk')}:** {disk_free_gb(storage.root)} GB"
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
            with gr.Row():
                train_count = gr.Number(value=500, precision=0, label=b("训练字形数", "Training glyphs"))
                test_count = gr.Number(value=8, precision=0, label=b("测试字形数", "Test glyphs"))
                dataset_charset = gr.Dropdown(
                    choices=["gb2312", "gbk", "big5", "jisx0208", "ksx1001"],
                    value="gb2312",
                    label=b("数据集字符集", "Dataset charset"),
                )
                workers = gr.Number(value=4, precision=0, label=b("工作进程", "Workers"))
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
            queue_training_btn = gr.Button(b("开始 LoRA 训练", "Start LoRA training"), variant="primary")
            resume_training_btn = gr.Button(
                b("从所选 run 的 last checkpoint 续训", "Resume selected run from last checkpoint")
            )
            training_run_id = gr.Textbox(label="Run ID")
            training_status = gr.Markdown()
            run_selector = gr.Dropdown(
                label=b("对比 run（2–5 个）", "Compare runs (2–5)"), multiselect=True, max_choices=5
            )
            with gr.Row():
                refresh_runs_btn = gr.Button(b("刷新 run", "Refresh runs"))
                refresh_charts_btn = gr.Button(b("刷新图表", "Refresh charts"))
                export_metrics_btn = gr.Button(b("导出 CSV / JSON / 图表", "Export CSV / JSON / charts"))
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
            snapshot_gallery = gr.Gallery(
                label=b("固定字形快照时间轴", "Fixed-glyph snapshot timeline"), columns=4, height=360
            )
            gr.HTML(
                f'<a href="{tensorboard_url}" target="_blank">'
                f"{b('打开项目 TensorBoard', 'Open project TensorBoard')} ↗</a>"
            )

        with gr.Tab(t("generation")):
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
            refresh_gallery_btn = gr.Button(b("刷新生成字形", "Refresh generated glyphs"))
            generated_gallery = gr.Gallery(label=b("生成字形", "Generated glyphs"), columns=8, height=500)
            with gr.Row():
                review_codepoint = gr.Dropdown(label=b("码位", "Codepoint"))
                review_candidate = gr.Dropdown(label=b("所选候选文件", "Selected candidate file"))
            candidate_gallery_component = gr.Gallery(label=b("候选", "Candidates"), columns=4, height=300)
            save_candidate_btn = gr.Button(b("采用此候选", "Use this candidate"))
            candidate_status = gr.Markdown()

        with gr.Tab(t("export")):
            glyph_dir = gr.Textbox(label=b("生成字形目录", "Generated glyph directory"))
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
            jobs_table = gr.Dataframe(
                headers=["id", "type", "status", "gpu", "created", "return_code"],
                interactive=False,
            )
            job_selector = gr.Dropdown(label=b("任务", "Job"))
            with gr.Row():
                refresh_log_btn = gr.Button(b("刷新日志", "Refresh log"))
                cancel_job_btn = gr.Button(b("取消", "Cancel"))
                retry_job_btn = gr.Button(b("重试", "Retry"))
            job_log = gr.Textbox(label=b("日志", "Log"), lines=24, max_lines=40)
            job_action_status = gr.Markdown()

        create_project_btn.click(
            create_project, new_project_name, [project_selector, project_status]
        )
        refresh_projects_btn.click(refresh_projects, outputs=project_selector)
        project_selector.change(project_summary, project_selector, project_json)
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
        queue_dataset_btn.click(
            queue_dataset,
            [project_selector, train_count, test_count, dataset_charset, workers],
            [dataset_path, dataset_status],
        )
        queue_training_btn.click(
            queue_training,
            [project_selector, dataset_path, training_device, quality, epochs],
            [training_run_id, training_status],
        )
        resume_training_btn.click(
            resume_training,
            [project_selector, run_selector, dataset_path, training_device, epochs],
            [training_run_id, training_status],
        )
        delete_run_btn.click(
            delete_training,
            [project_selector, run_selector, confirm_delete_run],
            [run_selector, training_status],
        )
        refresh_runs_btn.click(run_choices, project_selector, run_selector)
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
        dashboard_timer = gr.Timer(5)
        dashboard_timer.tick(
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
                generation_seed,
                candidate_count,
                unlock_candidates,
            ],
            [generation_output, generation_status],
        )
        refresh_gallery_btn.click(find_generated, project_selector, generated_gallery)
        refresh_gallery_btn.click(candidate_groups, project_selector, review_codepoint)
        review_codepoint.change(
            candidate_gallery,
            [project_selector, review_codepoint],
            [candidate_gallery_component, review_candidate],
        )
        save_candidate_btn.click(
            save_candidate,
            [project_selector, review_codepoint, review_candidate],
            candidate_status,
        )
        build_font_btn.click(
            queue_font,
            [
                project_selector,
                glyph_dir,
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
            job_table, project_selector, [jobs_table, job_selector]
        )
        refresh_log_btn.click(jobs.read_log, job_selector, job_log)
        cancel_job_btn.click(
            lambda job_id: f"Cancelled: {jobs.cancel(job_id)}",
            job_selector,
            job_action_status,
        )
        retry_job_btn.click(
            lambda job_id: f"Retried as: {jobs.retry(job_id)}",
            job_selector,
            job_action_status,
        )

    return app
