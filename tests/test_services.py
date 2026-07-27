from zi2zi_webui.services import dataset_command, infer_checkpoint_model
from zi2zi_webui.storage import Storage


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
