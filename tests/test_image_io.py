import numpy as np
import pytest
from PIL import Image

from util.image_io import imwrite_unicode
from lora_single_gpu_finetune_jit import get_args_parser as lora_args_parser
from main_jit import get_args_parser as main_args_parser


def test_unicode_safe_image_write_round_trip(tmp_path):
    path = tmp_path / "中文项目" / "训练快照" / "第十轮.png"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 8:24] = 0

    written = imwrite_unicode(path, image)

    assert written == path
    with Image.open(path) as reopened:
        assert reopened.size == (32, 32)
        assert reopened.getpixel((16, 16)) == (0, 0, 0)


def test_unicode_safe_image_write_raises_when_encoding_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "util.image_io.cv2.imencode",
        lambda *_args, **_kwargs: (False, np.asarray([], dtype=np.uint8)),
    )
    with pytest.raises(OSError, match="could not encode"):
        imwrite_unicode(
            tmp_path / "中文.png",
            np.zeros((4, 4, 3), dtype=np.uint8),
        )


def test_training_cli_horizontal_flip_defaults_off():
    assert main_args_parser().parse_args([]).horizontal_flip_prob == 0.0
    assert lora_args_parser().parse_args([]).horizontal_flip_prob == 0.0
