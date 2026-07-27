from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .telemetry import read_metrics


def training_figures(metric_paths: Iterable[str | Path]):
    paths = [Path(item) for item in metric_paths if item]
    records_by_run = [(path.parent.name[:8], read_metrics(path)) for path in paths]

    figures = []
    specs = (
        ("Training loss", ("loss", "loss_ema"), "global_step"),
        ("Learning rate", ("lr",), "global_step"),
        (
            "Resources",
            ("gpu_memory_allocated_gb", "gpu_memory_reserved_gb", "gpu_utilization", "ram_percent"),
            "global_step",
        ),
        ("Evaluation", ("ssim", "lpips", "l1", "fid"), "epoch"),
    )
    for title, keys, x_key in specs:
        figure = Figure(figsize=(7.2, 3.2), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        found = False
        for run_name, records in records_by_run:
            for key in keys:
                points = [
                    (item.get(x_key), item.get(key))
                    for item in records
                    if item.get(x_key) is not None and item.get(key) is not None
                ]
                if not points:
                    continue
                found = True
                axis.plot(
                    [item[0] for item in points],
                    [item[1] for item in points],
                    label=f"{run_name} · {key}",
                    linewidth=1.5,
                )
        axis.set_title(title)
        axis.set_xlabel(x_key.replace("_", " "))
        axis.grid(alpha=0.25)
        if found:
            axis.legend(fontsize=8, ncol=2)
        else:
            axis.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=axis.transAxes)
        figures.append(figure)
    return figures


def run_parameter_rows(runs: list[dict]) -> list[list[str]]:
    if not runs:
        return []
    keys = sorted({key for run in runs for key in run.get("parameters", {})})
    rows = []
    for key in keys:
        rows.append(
            [
                key,
                *[
                    json.dumps(run.get("parameters", {}).get(key), ensure_ascii=False)
                    for run in runs
                ],
            ]
        )
    return rows


def latest_summary(metric_path: str | Path) -> str:
    records = read_metrics(metric_path)
    if not records:
        return "No metrics yet."
    last = next((item for item in reversed(records) if item.get("phase") == "train"), records[-1])
    fields = []
    for key in (
        "epoch",
        "step",
        "loss",
        "loss_ema",
        "lr",
        "iterations_per_second",
        "eta_seconds",
        "gpu_memory_allocated_gb",
        "gpu_utilization",
        "ssim",
        "lpips",
        "l1",
        "fid",
    ):
        if key in last:
            fields.append(f"{key}: {last[key]}")
    for key in ("gpu_temperature_c", "gpu_power_w"):
        fields.append(f"{key}: {last.get(key, 'not supported')}")
    return "\n".join(fields)
