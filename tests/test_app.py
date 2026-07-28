from zi2zi_webui.app import build_app, training_snapshot_items
from zi2zi_webui.jobs import JobManager
from zi2zi_webui.models import TrainingRun
from zi2zi_webui.storage import Storage


def test_dynamic_timer_dropdowns_accept_restored_browser_values(tmp_path):
    storage = Storage(tmp_path / "state")
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        dropdowns = {
            block.label: block
            for block in app.blocks.values()
            if hasattr(block, "allow_custom_value")
        }
        assert dropdowns["Current task"].allow_custom_value
        assert dropdowns["Compare runs (2–5)"].allow_custom_value
        assert dropdowns["Current task"].preprocess("restored-job-id") == "restored-job-id"
        assert dropdowns["Compare runs (2–5)"].preprocess(
            ["restored-run-id"]
        ) == ["restored-run-id"]
    finally:
        jobs.stop()


def test_training_snapshots_include_run_epoch_and_pair_description(tmp_path):
    snapshot_dir = tmp_path / "compare"
    snapshot_dir.mkdir()
    image = snapshot_dir / "grid_00002.png"
    image.touch()
    items = training_snapshot_items(
        [
            (
                "abc12345",
                [
                    {
                        "phase": "evaluation",
                        "epoch": 9,
                        "snapshot_path": str(snapshot_dir),
                    }
                ],
            )
        ],
        language="zh",
    )
    assert items == [
        (
            str(image),
            "Run abc12345 · 第 10 轮 · 第 3 组（每对：目标真值 → 当前生成）",
        )
    ]


def test_training_snapshots_deduplicate_repeated_metric_events(tmp_path):
    snapshot_dir = tmp_path / "compare"
    snapshot_dir.mkdir()
    image = snapshot_dir / "grid_00000.png"
    image.touch()
    event = {
        "phase": "evaluation",
        "epoch": 19,
        "snapshot_path": str(snapshot_dir),
    }

    items = training_snapshot_items(
        [("run12345", [event, dict(event)])],
        language="en",
    )

    assert items == [
        (
            str(image),
            "Run run12345 · Epoch 20 · Group 1 (each pair: target → generated)",
        )
    ]


def test_training_snapshot_gallery_uses_page_scrolling(tmp_path):
    storage = Storage(tmp_path / "state")
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        gallery = next(
            block
            for block in app.blocks.values()
            if getattr(block, "label", "")
            == "Training glyph evolution (fixed glyphs and seed)"
        )
        assert gallery.height == "auto"
        assert gallery.columns == 2
    finally:
        jobs.stop()


def test_project_page_has_confirmed_danger_zone_deletion(tmp_path):
    storage = Storage(tmp_path / "state")
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        delete_button = next(
            block
            for block in app.blocks.values()
            if getattr(block, "value", "") == "Delete project and files"
        )
        confirmation = next(
            block
            for block in app.blocks.values()
            if getattr(block, "label", "")
            == "Confirm permanent deletion of the current project and all files"
        )
        assert delete_button.variant == "stop"
        assert confirmation.value is False
    finally:
        jobs.stop()


def test_generation_page_has_dynamic_lora_checkpoint_selector(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("LoRA")
    base = storage.models_dir / "base.pth"
    base.touch()
    project.base_model = str(base)
    project.active_checkpoint = str(base)
    storage.save_project(project)
    run = TrainingRun(project_id=project.id, parameters={})
    storage.save_training_run(run)
    run_dir = storage.project_dir(project.id) / "training" / run.id
    run_dir.mkdir(parents=True)
    best = run_dir / "checkpoint-best-ssim.pth"
    best.touch()
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        selector = next(
            block
            for block in app.blocks.values()
            if getattr(block, "label", "")
            == "Generation model (LoRA / checkpoint)"
        )
        assert selector.allow_custom_value
        assert selector.value == str(best.resolve())
        assert any(
            value == str(best.resolve())
            for _label, value in selector.choices
        )
        assert selector.preprocess("restored-checkpoint.pth") == "restored-checkpoint.pth"
    finally:
        jobs.stop()


def test_review_and_export_pages_select_generation_batches(tmp_path):
    storage = Storage(tmp_path / "state")
    storage.create_project("Batches")
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        labels = {
            getattr(block, "label", "")
            for block in app.blocks.values()
        }
        assert "Generation batch" in labels
        assert "Generation batch to export" in labels
        assert "Search character or codepoint" in labels
        assert "Page" in labels
        assert "Generated glyph directory" not in labels
    finally:
        jobs.stop()


def test_training_page_defaults_horizontal_flip_off(tmp_path):
    storage = Storage(tmp_path / "state")
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        slider = next(
            block
            for block in app.blocks.values()
            if getattr(block, "label", "")
            == "Horizontal flip probability (normally 0 for CJK)"
        )
        assert slider.value == 0.0
        assert slider.minimum == 0.0
        assert slider.maximum == 0.5
    finally:
        jobs.stop()
