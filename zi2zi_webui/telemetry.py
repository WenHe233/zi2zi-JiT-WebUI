from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable


class MetricsWriter:
    def __init__(self, path: str | Path, run_id: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._lock = threading.Lock()
        self._loss_window: deque[float] = deque(maxlen=20)
        self._low_gpu_samples = 0

    def write(self, event: dict[str, Any]) -> dict[str, Any]:
        record = dict(event)
        record.setdefault("schema_version", 1)
        record.setdefault("run_id", self.run_id)
        record.setdefault("timestamp", time.time())
        loss = record.get("loss")
        if isinstance(loss, (int, float)):
            if len(self._loss_window) >= 10:
                median = statistics.median(self._loss_window)
                if median > 0 and float(loss) > median * 3:
                    record.setdefault("alert", "loss_spike")
            self._loss_window.append(float(loss))
            alpha = 0.1
            previous = getattr(self, "_loss_ema", float(loss))
            self._loss_ema = alpha * float(loss) + (1 - alpha) * previous
            record.setdefault("loss_ema", self._loss_ema)
        utilization = record.get("gpu_utilization")
        if isinstance(utilization, (int, float)):
            self._low_gpu_samples = self._low_gpu_samples + 1 if utilization < 10 else 0
            if self._low_gpu_samples >= 5:
                record.setdefault("alert", "sustained_low_gpu_utilization")
        allocated = record.get("gpu_memory_allocated_gb")
        total = record.get("gpu_memory_total_gb")
        if isinstance(allocated, (int, float)) and isinstance(total, (int, float)):
            if total > 0 and allocated / total >= 0.95:
                record.setdefault("alert", "gpu_memory_pressure")
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        return record


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def export_metrics_csv(source: str | Path, destination: str | Path) -> Path:
    records = read_metrics(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    return destination


def export_metrics_json(source: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(read_metrics(source), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def resource_snapshot(device_index: int | None = None, data_path: str | Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil

        result["cpu_percent"] = psutil.cpu_percent()
        result["ram_percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass
    target = Path(data_path or Path.cwd())
    result["disk_free_gb"] = round(shutil.disk_usage(target).free / (1024**3), 2)
    try:
        import torch

        if torch.cuda.is_available():
            index = device_index if device_index is not None else torch.cuda.current_device()
            result["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(index) / (1024**3), 3)
            result["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(index) / (1024**3), 3)
            result["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(index).total_memory / (1024**3), 3
            )
    except (ImportError, RuntimeError):
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index or 0)
        result["gpu_utilization"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        result["gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        result["gpu_power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000, 1)
    except Exception:
        pass
    return result


def write_training_report(metrics_path: str | Path, output_path: str | Path) -> Path:
    records = read_metrics(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluations = [item for item in records if item.get("phase") == "evaluation" and "ssim" in item]
    alerts = [item for item in records if item.get("alert") or item.get("phase") == "early_stop"]
    last_train = next((item for item in reversed(records) if item.get("phase") == "train"), {})
    evaluation_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{item.get(key, '')}</td>"
            for key in ("epoch", "ssim", "lpips", "l1", "fid", "inception_score")
        )
        + "</tr>"
        for item in evaluations
    )
    alert_rows = "".join(
        f"<li>epoch {item.get('epoch', '')}: {item.get('alert') or item.get('phase')}</li>"
        for item in alerts
    )
    output_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>zi2zi-JiT training report</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.45rem}</style>"
        "<h1>zi2zi-JiT training report</h1>"
        f"<p>Events: {len(records)} · Last epoch: {last_train.get('epoch', '')} · "
        f"Last loss: {last_train.get('loss', '')}</p>"
        f"<h2>Alerts</h2><ul>{alert_rows or '<li>None</li>'}</ul>"
        "<h2>Evaluation</h2><table><tr><th>Epoch</th><th>SSIM</th><th>LPIPS</th>"
        f"<th>L1</th><th>FID</th><th>IS</th></tr>{evaluation_rows}</table>",
        encoding="utf-8",
    )
    return output_path


def metrics_to_plot_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        for record in read_metrics(path):
            row = dict(record)
            row["source"] = Path(path).parent.name
            rows.append(row)
    return rows
