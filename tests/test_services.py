import pytest
from PIL import Image, ImageDraw

from zi2zi_webui.services import (
    font_dataset_capacity,
    dataset_max_chars_per_font,
    dataset_command,
    dataset_size_preset,
    generation_checkpoint_options,
    generation_command,
    infer_checkpoint_model,
    resolve_generation_checkpoint,
    resolve_training_dataset,
)
from zi2zi_webui.font_builder import FontMetadata, build_ttf
from zi2zi_webui.models import TrainingRun
from zi2zi_webui.storage import Storage
from util.training_schedule import periodic_or_final


def test_dataset_command_passes_ordered_regional_and_global_source_fonts(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Jigmo")
    project.input_mode = "font"
    project.global_source_fonts = ["global-1.ttf", "global-2.ttf"]
    project.regional_source_fonts["JP"] = ["jigmo-1.ttf", "jigmo-2.ttf"]
    project.target_assets = ["target.ttf"]

    command, output = dataset_command(
        project,
        storage,
        train_count=100,
        test_count=8,
        charset="jisx0208",
        workers=1,
    )
    start = command.index("--source-font") + 1
    assert command[start : start + 4] == [
        "jigmo-1.ttf",
        "jigmo-2.ttf",
        "global-1.ttf",
        "global-2.ttf",
    ]
    assert command[command.index("--target-font") + 1] == "target.ttf"
    assert "--font-dir" not in command
    assert "train100-test8" in str(output)


def test_auto_dataset_uses_ordered_global_fonts_without_a_region_filter(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Automatic")
    project.input_mode = "font"
    project.global_source_fonts = ["jigmo-1.ttf", "jigmo-2.ttf"]
    project.target_assets = ["target.ttf"]

    command, output = dataset_command(
        project,
        storage,
        train_count=3000,
        test_count=64,
        charset="auto",
        workers=2,
    )

    start = command.index("--source-font") + 1
    assert command[start : start + 2] == ["jigmo-1.ttf", "jigmo-2.ttf"]
    assert command[command.index("--charset") + 1] == "auto"
    assert "dataset-auto-train3000-test64" in str(output)


def test_rendered_glyph_dataset_passes_exact_validation_count(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Rendered")
    project.input_mode = "glyphs"
    project.global_source_fonts = ["source.ttf"]
    project.target_assets = ["glyphs"]

    command, _output = dataset_command(
        project,
        storage,
        train_count=9,
        test_count=3,
        charset="auto",
        workers=1,
    )

    assert command[command.index("--train-count") + 1] == "9"
    assert command[command.index("--test-count") + 1] == "3"


def test_repeated_dataset_commands_use_isolated_output_directories(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Isolated")
    project.input_mode = "font"
    project.global_source_fonts = ["source.ttf"]
    project.target_assets = ["target.ttf"]

    first_command, first_output = dataset_command(
        project,
        storage,
        train_count=9,
        test_count=1,
        charset="auto",
        workers=1,
    )
    second_command, second_output = dataset_command(
        project,
        storage,
        train_count=9,
        test_count=1,
        charset="auto",
        workers=1,
    )

    assert first_output != second_output
    assert first_command[first_command.index("--build-marker") + 1] == str(
        first_output / ".webui-dataset.json"
    )
    assert second_command[second_command.index("--build-marker") + 1] == str(
        second_output / ".webui-dataset.json"
    )


def test_rendered_glyph_capacity_requires_valid_target_and_source_outlines(tmp_path):
    def draw_glyph(path):
        image = Image.new("L", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 210, 210), fill="black")
        image.save(path)

    shape = tmp_path / "shape.png"
    draw_glyph(shape)
    source_font = tmp_path / "source.ttf"
    build_ttf(
        {0x4E00: shape},
        source_font,
        FontMetadata(family_name="Source"),
    )
    glyph_dir = tmp_path / "glyphs"
    glyph_dir.mkdir()
    draw_glyph(glyph_dir / "U+4E00.png")
    draw_glyph(glyph_dir / "U+4E01.png")

    storage = Storage(tmp_path / "state")
    project = storage.create_project("Capacity")
    project.input_mode = "glyphs"
    project.global_source_fonts = [str(source_font)]
    project.target_assets = [str(glyph_dir)]

    assert font_dataset_capacity(project, "auto") == 1


def test_checkpoint_variant_is_inferred_from_official_filename():
    assert infer_checkpoint_model("zi2zi-JiT-L-16.pth") == "JiT-L/16"
    assert infer_checkpoint_model("zi2zi-JiT-B-16.pth") == "JiT-B/16"


def test_training_dataset_falls_back_from_empty_or_invalid_path(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Fallback")
    dataset = storage.project_dir(project.id) / "datasets" / "complete"
    (dataset / "train").mkdir(parents=True)
    (dataset / "test.npz").touch()

    assert resolve_training_dataset(project, storage, "") == dataset.resolve()
    assert resolve_training_dataset(project, storage, tmp_path / "wrong") == dataset.resolve()


def test_training_dataset_requires_train_directory_and_test_npz(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Incomplete")
    dataset = storage.project_dir(project.id) / "datasets" / "incomplete"
    dataset.mkdir(parents=True)

    with pytest.raises(ValueError, match="train/ and test.npz"):
        resolve_training_dataset(project, storage, dataset)


def test_training_dataset_skips_new_build_without_complete_marker(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Build state")
    datasets = storage.project_dir(project.id) / "datasets"
    complete = datasets / "older"
    failed = datasets / "newer"
    for path in (complete, failed):
        (path / "train").mkdir(parents=True)
        (path / "test.npz").touch()
    (complete / ".webui-dataset.json").write_text(
        '{"schema_version": 1, "status": "complete"}',
        encoding="utf-8",
    )
    (failed / ".webui-dataset.json").write_text(
        '{"schema_version": 1, "status": "failed"}',
        encoding="utf-8",
    )

    assert resolve_training_dataset(project, storage, failed) == complete.resolve()


def test_single_epoch_training_always_saves_last_checkpoint():
    assert periodic_or_final(epoch=0, total_epochs=1, frequency=10)
    assert not periodic_or_final(epoch=0, total_epochs=10, frequency=10)
    assert periodic_or_final(epoch=9, total_epochs=200, frequency=10)


def test_dataset_size_presets_and_generated_sample_count(tmp_path):
    assert dataset_size_preset("balanced") == (3000, 64)
    dataset = tmp_path / "dataset"
    metadata = dataset / "train" / "001_font" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"extracted_count": 2875}', encoding="utf-8")
    assert dataset_max_chars_per_font(dataset) == 2875


def test_generation_checkpoints_prefer_latest_run_best_ssim(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("LoRA")
    base = storage.models_dir / "base.pth"
    base.touch()
    project.base_model = str(base)
    project.active_checkpoint = str(base)
    storage.save_project(project)

    older = TrainingRun(project_id=project.id, parameters={})
    storage.save_training_run(older)
    older_dir = storage.project_dir(project.id) / "training" / older.id
    older_dir.mkdir(parents=True)
    (older_dir / "checkpoint-best-ssim.pth").touch()

    latest = TrainingRun(project_id=project.id, parameters={})
    storage.save_training_run(latest)
    latest_dir = storage.project_dir(project.id) / "training" / latest.id
    latest_dir.mkdir(parents=True)
    best_ssim = latest_dir / "checkpoint-best-ssim.pth"
    best_lpips = latest_dir / "checkpoint-best-lpips.pth"
    last = latest_dir / "checkpoint-last.pth"
    for checkpoint in (best_ssim, best_lpips, last):
        checkpoint.touch()

    choices, preferred = generation_checkpoint_options(project, storage)
    values = [value for _label, value in choices]
    assert values[:3] == [
        str(best_ssim.resolve()),
        str(best_lpips.resolve()),
        str(last.resolve()),
    ]
    assert preferred == str(best_ssim.resolve())
    assert resolve_generation_checkpoint(project, storage, preferred) == preferred

    command, _output = generation_command(
        project,
        storage,
        tmp_path / "request.npz",
        "0",
        seed=42,
        checkpoint_path=preferred,
    )
    assert command[command.index("--checkpoint") + 1] == preferred


def test_generation_checkpoint_rejects_paths_outside_project_catalog(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("LoRA")
    outside = tmp_path / "untrusted.pth"
    outside.touch()

    with pytest.raises(ValueError, match="does not belong"):
        resolve_generation_checkpoint(project, storage, outside)
