from zi2zi_webui.devices import DeviceInfo, training_preset


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
