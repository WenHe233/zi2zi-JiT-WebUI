import json
import sqlite3
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from zi2zi_webui.models import TrainingRun
from zi2zi_webui.project_packages import (
    export_project_package,
    import_project_package,
    inspect_project_package,
)
from zi2zi_webui.storage import Storage


def _completed_job(storage, project_id, job_id="old-job"):
    log = storage.project_dir(project_id) / "logs" / f"{job_id}.log"
    log.write_text("completed", encoding="utf-8")
    with storage._connect() as db:
        db.execute(
            """
            INSERT INTO jobs(
                id, project_id, job_type, status, command_json, cwd, gpu,
                env_json, pid, return_code, log_path, error, progress,
                progress_text, resume_point, created_at, started_at, finished_at
            ) VALUES (?, ?, 'training', 'succeeded', ?, ?, '0', ?, NULL, 0, ?,
                      NULL, 1, 'Completed', '', 'created', 'started', 'finished')
            """,
            (
                job_id,
                project_id,
                json.dumps(["python", str(storage.project_dir(project_id) / "train.py")]),
                str(storage.project_dir(project_id)),
                json.dumps({"SECRET": "must-not-export"}),
                str(log),
            ),
        )
    return job_id


def _source_project(tmp_path):
    storage = Storage(tmp_path / "source")
    project = storage.create_project("Portable")
    project_root = storage.project_dir(project.id)
    source_font = project_root / "inputs" / "source.ttf"
    source_font.write_bytes(b"source-font")
    target_font = project_root / "inputs" / "target.ttf"
    target_font.write_bytes(b"target-font")
    project.global_source_fonts = [str(source_font)]
    project.target_assets = [str(target_font)]
    project.style_reference_pool = []
    job_id = _completed_job(storage, project.id)
    run = TrainingRun(
        project_id=project.id,
        parameters={"job_id": job_id, "dataset_path": str(project_root / "datasets" / "one")},
        metrics_path=str(project_root / "training" / "old-run" / "metrics.jsonl"),
        id="old-run",
        status="succeeded",
    )
    run_dir = project_root / "training" / run.id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text('{"loss": 1}\n', encoding="utf-8")
    (run_dir / "checkpoint-last.pth").write_bytes(b"checkpoint")
    (run_dir / "checkpoint-last.pth.json").write_text(
        '{"sha256": "untrusted-sidecar"}',
        encoding="utf-8",
    )
    storage.save_training_run(run)
    storage.save_project(project)
    return storage, project, run, job_id


def test_full_project_package_round_trip_remaps_ids_and_paths(tmp_path):
    source, project, _run, old_job_id = _source_project(tmp_path)
    package = tmp_path / "portable.zi2zi-project.zip"
    exported = export_project_package(
        source,
        project.id,
        package,
        mode="full",
    )
    assert exported["checksum_file_count"] >= 5
    assert inspect_project_package(package)["checksum_verified"]

    target = Storage(tmp_path / "target")
    imported = import_project_package(target, package)
    restored = target.get_project(imported["project_id"])
    restored_root = target.project_dir(restored.id)
    assert restored.id != project.id
    assert restored.name == project.name
    assert restored.global_source_fonts == [str(restored_root / "inputs" / "source.ttf")]
    assert Path(restored.global_source_fonts[0]).read_bytes() == b"source-font"

    runs = target.list_training_runs(restored.id)
    assert len(runs) == 1
    assert runs[0]["id"] != "old-run"
    assert runs[0]["parameters"]["job_id"] != old_job_id
    assert Path(runs[0]["metrics_path"]).is_file()
    imported_run_dir = restored_root / "training" / runs[0]["id"]
    assert (imported_run_dir / "checkpoint-last.pth").is_file()
    assert not (imported_run_dir / "checkpoint-last.pth.json").exists()
    assert (imported_run_dir / "checkpoint-last.pth.imported-metadata.json").is_file()

    with target._connect() as db:
        job = db.execute(
            "SELECT * FROM jobs WHERE project_id=?",
            (restored.id,),
        ).fetchone()
    assert job["id"] != old_job_id
    assert job["status"] == "succeeded"
    assert json.loads(job["env_json"]) == {}
    assert Path(job["log_path"]).name == f"{job['id']}.log"
    assert Path(job["log_path"]).is_file()


def test_same_package_can_be_imported_repeatedly_with_fresh_ids(tmp_path):
    source, project, _run, _job = _source_project(tmp_path)
    package = tmp_path / "portable.zip"
    export_project_package(source, project.id, package, mode="full")
    target = Storage(tmp_path / "target")

    first = import_project_package(target, package)
    second = import_project_package(target, package)

    assert first["project_id"] != second["project_id"]
    first_run = target.list_training_runs(first["project_id"])[0]
    second_run = target.list_training_runs(second["project_id"])[0]
    assert first_run["id"] != second_run["id"]
    assert first_run["parameters"]["job_id"] != second_run["parameters"]["job_id"]


def test_lightweight_package_excludes_binary_assets(tmp_path):
    source, project, _run, _job = _source_project(tmp_path)
    package = tmp_path / "light.zip"
    export_project_package(source, project.id, package, mode="lightweight")

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert "project/inputs/source.ttf" not in names
    assert "project/training/old-run/checkpoint-last.pth" not in names
    assert "project/training/old-run/metrics.jsonl" in names
    target = Storage(tmp_path / "target")
    imported = import_project_package(target, package)
    restored = target.get_project(imported["project_id"])
    assert restored.global_source_fonts == []
    assert restored.target_assets == []
    assert imported["missing_project_references"]


def test_shared_models_are_referenced_by_digest_or_included_explicitly(tmp_path):
    source = Storage(tmp_path / "source")
    project = source.create_project("Model")
    model = source.models_dir / "base.pth"
    model.write_bytes(b"trusted-source-model")
    project.base_model = str(model)
    project.active_checkpoint = str(model)
    source.save_project(project)

    reference_package = tmp_path / "reference.zip"
    export_project_package(source, project.id, reference_package, mode="lightweight")
    target = Storage(tmp_path / "target")
    referenced = import_project_package(target, reference_package)
    referenced_project = target.get_project(referenced["project_id"])
    assert referenced_project.base_model == ""
    assert referenced["missing_models"] == ["base.pth"]

    included_package = tmp_path / "included.zip"
    export_project_package(
        source,
        project.id,
        included_package,
        mode="full",
        include_shared_models=True,
    )
    second_target = Storage(tmp_path / "second-target")
    included = import_project_package(second_target, included_package)
    included_project = second_target.get_project(included["project_id"])
    imported_model = Path(included_project.base_model)
    assert imported_model.read_bytes() == b"trusted-source-model"
    assert imported_model.parent == second_target.models_dir
    assert not imported_model.with_suffix(imported_model.suffix + ".json").exists()


def test_package_inspection_rejects_checksum_tampering(tmp_path):
    source, project, _run, _job = _source_project(tmp_path)
    package = tmp_path / "original.zip"
    export_project_package(source, project.id, package, mode="lightweight")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package) as original, zipfile.ZipFile(tampered, "w") as output:
        for info in original.infolist():
            payload = original.read(info.filename)
            if info.filename == "records/project.json":
                payload += b" "
            output.writestr(info, payload)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        inspect_project_package(tampered)


def test_package_inspection_rejects_path_traversal(tmp_path):
    package = tmp_path / "traversal.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../escape", b"no")
        archive.writestr("package.json", b"{}")
        archive.writestr("records/project.json", b"{}")
        archive.writestr("records/training-runs.json", b"[]")
        archive.writestr("records/jobs.json", b"[]")
        archive.writestr("checksums.sha256", b"")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        inspect_project_package(package)


def test_export_rejects_active_project_jobs(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Busy")
    with storage._connect() as db:
        db.execute(
            """
            INSERT INTO jobs(
                id, project_id, job_type, status, command_json, cwd, gpu,
                env_json, log_path, created_at
            ) VALUES ('active', ?, 'training', 'queued', '[]', '.', NULL, '{}', 'x', 'now')
            """,
            (project.id,),
        )

    with pytest.raises(ValueError, match="active"):
        export_project_package(storage, project.id, tmp_path / "busy.zip")


def test_import_rolls_back_new_project_files_when_database_insert_fails(
    tmp_path,
    monkeypatch,
):
    source, project, _run, _job = _source_project(tmp_path)
    package = tmp_path / "portable.zip"
    export_project_package(source, project.id, package, mode="full")
    target = Storage(tmp_path / "target")

    @contextmanager
    def fail_connect():
        raise sqlite3.OperationalError("simulated import database failure")
        yield

    monkeypatch.setattr(target, "_connect", fail_connect)
    with pytest.raises(sqlite3.OperationalError, match="simulated"):
        import_project_package(target, package)

    assert not any(target.projects_dir.iterdir())
