from zi2zi_webui.app import build_app, training_snapshot_items
from zi2zi_webui.jobs import JobManager
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
