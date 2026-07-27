import pytest

from zi2zi_webui.services import (
    dataset_command,
    infer_checkpoint_model,
    resolve_training_dataset,
)
from zi2zi_webui.storage import Storage
from util.training_schedule import periodic_or_final


def test_dataset_command_passes_ordered_regional_and_global_source_fonts(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Jigmo")
    project.input_mode = "font"
    project.global_source_fonts = ["global-1.ttf", "global-2.ttf"]
    project.regional_source_fonts["JP"] = ["jigmo-1.ttf", "jigmo-2.ttf"]
    project.target_assets = ["target.ttf"]

    command, _ = dataset_command(
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


def test_single_epoch_training_always_saves_last_checkpoint():
    assert periodic_or_final(epoch=0, total_epochs=1, frequency=10)
    assert not periodic_or_final(epoch=0, total_epochs=10, frequency=10)
    assert periodic_or_final(epoch=9, total_epochs=200, frequency=10)
