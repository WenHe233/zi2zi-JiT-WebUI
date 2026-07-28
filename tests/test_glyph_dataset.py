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
    marker = output / ".webui-dataset.json"
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
            "--build-marker",
            str(marker),
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
    assert '"status": "complete"' in marker.read_text(encoding="utf-8")


def test_font_dataset_processes_only_the_explicit_target_font(tmp_path):
    glyph_dir = tmp_path / "glyphs"
    glyph_dir.mkdir()
    for index in range(12):
        _glyph(glyph_dir / f"U+{0x4E00 + index:04X}.png")
    source_font = tmp_path / "source.ttf"
    selected_font = tmp_path / "selected.ttf"
    ignored_font = tmp_path / "aaa-ignored.ttf"
    glyphs = {
        0x4E00 + index: glyph_dir / f"U+{0x4E00 + index:04X}.png"
        for index in range(12)
    }
    build_ttf(glyphs, source_font, FontMetadata(family_name="Source"))
    build_ttf(glyphs, selected_font, FontMetadata(family_name="Selected"))
    build_ttf(glyphs, ignored_font, FontMetadata(family_name="Ignored"))
    output = tmp_path / "font-dataset"
    marker = output / ".webui-dataset.json"
    script = Path(__file__).parents[1] / "scripts" / "generate_font_dataset.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-font",
            str(source_font),
            "--target-font",
            str(selected_font),
            "--output-dir",
            str(output),
            "--train-chars-per-font",
            "9",
            "--test-chars-per-font",
            "2",
            "--resolution",
            "64",
            "--num-workers",
            "1",
            "--build-marker",
            str(marker),
        ],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    train_folders = [path.name for path in (output / "train").iterdir()]
    assert train_folders == ["001_selected"]
    assert '"status": "complete"' in marker.read_text(encoding="utf-8")


def test_create_test_npz_rejects_empty_test_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = create_test_npz(empty, tmp_path / "empty.npz")
    assert result["samples"] == 0
