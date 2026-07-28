import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from data_processing.pipeline import create_test_npz
from zi2zi_webui.font_builder import FontMetadata, build_ttf


def _glyph(path: Path) -> None:
    image = Image.new("L", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 8, 53, 55), fill="black")
    image.save(path)


def test_rendered_glyph_dataset_generates_exact_train_and_test_counts(tmp_path):
    glyph_dir = tmp_path / "glyphs"
    glyph_dir.mkdir()
    for index in range(12):
        _glyph(glyph_dir / f"U+{0x4E00 + index:04X}.png")

    source_font = tmp_path / "source.ttf"
    build_ttf(
        {
            0x4E00 + index: glyph_dir / f"U+{0x4E00 + index:04X}.png"
            for index in range(12)
        },
        source_font,
        FontMetadata(family_name="Dataset source"),
    )
    output = tmp_path / "dataset"
    script = Path(__file__).parents[1] / "scripts" / "generate_glyph_dataset.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-font",
            str(source_font),
            "--glyph-dir",
            str(glyph_dir),
            "--output-dir",
            str(output),
            "--train-count",
            "9",
            "--test-count",
            "2",
            "--resolution",
            "64",
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    train_images = list((output / "train").rglob("*.jpg"))
    test_images = list((output / "test").rglob("*.jpg"))
    assert len(train_images) == 9
    assert len(test_images) == 2
    assert (output / "test.npz").is_file()


def test_create_test_npz_rejects_empty_test_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = create_test_npz(empty, tmp_path / "empty.npz")
    assert result["samples"] == 0
