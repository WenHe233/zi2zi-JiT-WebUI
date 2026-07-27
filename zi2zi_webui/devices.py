from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DeviceInfo:
    id: str
    name: str
    backend: str
    memory_total_gb: float | None
    training_supported: bool


def detect_devices() -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    DeviceInfo(
                        id=str(index),
                        name=properties.name,
                        backend="cuda",
                        memory_total_gb=round(properties.total_memory / (1024**3), 1),
                        training_supported=True,
                    )
                )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(DeviceInfo("mps", "Apple Metal", "mps", None, False))
    except (ImportError, RuntimeError):
        pass
    devices.append(DeviceInfo("cpu", "CPU", "cpu", None, False))
    return devices


def training_preset(device: DeviceInfo, quality: str = "balanced") -> dict[str, int | float | bool]:
    memory = device.memory_total_gb or 0
    if quality == "economy" or memory < 8:
        return {
            "batch_size": 8,
            "gen_bsz": 4,
            "num_workers": 4,
            "lora_r": 16,
            "lora_alpha": 16,
            "snapshot_freq": 20,
            "full_eval": False,
            "eval_freq": 40,
        }
    if quality == "quality" and memory >= 16:
        return {
            "batch_size": 32,
            "gen_bsz": 16,
            "num_workers": 8,
            "lora_r": 64,
            "lora_alpha": 64,
            "snapshot_freq": 10,
            "full_eval": True,
            "eval_freq": 40,
        }
    return {
        "batch_size": 16,
        "gen_bsz": 8,
        "num_workers": 6,
        "lora_r": 32,
        "lora_alpha": 32,
        "snapshot_freq": 10,
        "full_eval": True,
        "eval_freq": 40,
    }


def disk_free_gb(path: str | Path) -> float:
    return round(shutil.disk_usage(Path(path)).free / (1024**3), 1)


def devices_as_dicts() -> list[dict]:
    return [asdict(item) for item in detect_devices()]
