from zi2zi_webui.devices import DeviceInfo, training_preset
from zi2zi_webui.services import clamp_training_retry_command


def cuda_device(memory_gb):
    return DeviceInfo("0", "Test GPU", "cuda", memory_gb, True)


def test_quality_preset_limits_jit_large_batch_on_16gb_gpu():
    preset = training_preset(cuda_device(16), "quality", "JiT-L/16")
    assert preset["batch_size"] == 12
    assert preset["gen_bsz"] == 8
    assert preset["lora_r"] == 64


def test_quality_preset_keeps_batch_32_for_large_model_on_24gb_gpu():
    preset = training_preset(cuda_device(24), "quality", "JiT-L/16")
    assert preset["batch_size"] == 32


def test_quality_preset_keeps_batch_32_for_base_model_on_16gb_gpu():
    preset = training_preset(cuda_device(16), "quality", "JiT-B/16")
    assert preset["batch_size"] == 32


def test_retry_clamps_legacy_large_model_quality_command(monkeypatch):
    monkeypatch.setattr(
        "zi2zi_webui.services.detect_devices",
        lambda: [cuda_device(16)],
    )
    command = [
        "python",
        "train.py",
        "--model",
        "JiT-L/16",
        "--batch_size",
        "32",
        "--gen_bsz",
        "16",
        "--num_workers",
        "8",
        "--lora_r",
        "64",
    ]
    updated, changes = clamp_training_retry_command(command, "0")
    assert updated[updated.index("--batch_size") + 1] == "12"
    assert updated[updated.index("--gen_bsz") + 1] == "8"
    assert updated[updated.index("--num_workers") + 1] == "6"
    assert changes["--batch_size"] == (32, 12)
