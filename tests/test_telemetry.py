import json

from zi2zi_webui.telemetry import (
    IncrementalMetricsCache,
    MetricsWriter,
    export_metrics_csv,
    export_metrics_json,
    read_metrics,
    write_training_report,
)


def test_metrics_jsonl_survives_truncated_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path, "run")
    writer.write({"phase": "train", "loss": 1.0, "global_step": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"broken"')
    records = read_metrics(path)
    assert len(records) == 1
    assert records[0]["run_id"] == "run"
    assert records[0]["loss_ema"] == 1.0


def test_incremental_metrics_cache_waits_for_complete_lines_and_resets(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"global_step": 1}\n{"global_step":')
    cache = IncrementalMetricsCache()

    assert cache.read(path) == [{"global_step": 1}]
    with path.open("ab") as handle:
        handle.write(b" 2}\n")
    assert cache.read(path) == [{"global_step": 1}, {"global_step": 2}]

    path.write_text('{"global_step": 3}\n', encoding="utf-8")
    assert cache.read(path) == [{"global_step": 3}]


def test_training_metrics_emit_explicit_webui_progress(tmp_path, capsys):
    writer = MetricsWriter(tmp_path / "metrics.jsonl", "run")
    writer.write(
        {
            "phase": "train",
            "epoch": 1,
            "step": 2,
            "global_step": 12,
            "total_steps": 100,
            "steps_per_epoch": 10,
        }
    )
    output = capsys.readouterr().out
    assert "WEBUI_PROGRESS" in output
    assert '"current": 12' in output
    assert '"total": 100' in output


def test_metrics_exports_and_report(tmp_path):
    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path)
    writer.write({"phase": "train", "epoch": 1, "loss": 0.5})
    writer.write(
        {"phase": "evaluation", "epoch": 1, "ssim": 0.7, "lpips": 0.2, "l1": 0.1, "fid": 50}
    )
    assert export_metrics_csv(path, tmp_path / "metrics.csv").is_file()
    exported = export_metrics_json(path, tmp_path / "metrics.json")
    assert json.loads(exported.read_text(encoding="utf-8"))[0]["loss"] == 0.5
    report = write_training_report(path, tmp_path / "report.html")
    assert "SSIM" in report.read_text(encoding="utf-8")
