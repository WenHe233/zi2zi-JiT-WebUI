import json

from zi2zi_webui.telemetry import (
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
