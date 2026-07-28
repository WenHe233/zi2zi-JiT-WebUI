from types import SimpleNamespace

import pytest
import torch

from util.misc import save_model_no_ema
from zi2zi_webui.checkpoints import (
    checkpoint_sidecar_path,
    read_checkpoint_sidecar,
    sha256_file,
)


def test_training_checkpoint_writes_versioned_sidecar(tmp_path):
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        model="JiT-B/16",
        num_fonts=4,
        num_chars=321,
        img_size=256,
    )

    save_model_no_ema(args, model, epoch=2, epoch_name="last")

    checkpoint = tmp_path / "checkpoint-last.pth"
    metadata = read_checkpoint_sidecar(checkpoint)
    assert metadata["source"] == "training"
    assert metadata["architecture"] == "JiT-B/16"
    assert metadata["num_fonts"] == 4
    assert metadata["num_chars"] == 321
    assert metadata["sha256"] == sha256_file(checkpoint)


def test_checkpoint_sidecar_detects_replaced_file(tmp_path):
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        model="JiT-L/16",
        num_fonts=1,
        num_chars=10,
        img_size=256,
    )
    save_model_no_ema(args, model, epoch=0, epoch_name="last")
    checkpoint = tmp_path / "checkpoint-last.pth"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="size no longer matches"):
        read_checkpoint_sidecar(checkpoint)
    assert checkpoint_sidecar_path(checkpoint).is_file()
