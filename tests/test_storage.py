import json

from zi2zi_webui.models import ProjectManifest, TrainingRun
from zi2zi_webui.storage import PROJECT_FOLDERS, Storage


def test_project_round_trip(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Demo")
    assert storage.get_project(project.id).name == "Demo"
    assert {item.name for item in storage.project_dir(project.id).iterdir()} >= {
        *PROJECT_FOLDERS,
        "project.json",
    }
    assert storage.list_projects()[0].id == project.id


def test_training_run_round_trip(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Demo")
    run = TrainingRun(project_id=project.id, parameters={"batch_size": 16})
    storage.save_training_run(run)
    loaded = storage.list_training_runs(project.id)[0]
    assert loaded["id"] == run.id
    assert loaded["parameters"]["batch_size"] == 16
    run_dir = storage.project_dir(project.id) / "training" / run.id
    run_dir.mkdir()
    storage.delete_training_run(project.id, run.id)
    assert storage.list_training_runs(project.id) == []
    assert not run_dir.exists()


def test_delete_project_removes_files_runs_and_job_records(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Disposable")
    run = TrainingRun(project_id=project.id, parameters={})
    storage.save_training_run(run)
    artifact = storage.project_dir(project.id) / "generation" / "glyph.png"
    artifact.write_bytes(b"glyph")
    with storage._connect() as db:
        db.execute(
            """
            INSERT INTO jobs(
                id, project_id, job_type, status, command_json, cwd, gpu,
                env_json, log_path, created_at
            ) VALUES ('done-job', ?, 'test', 'succeeded', '[]', '.', NULL, '{}', 'x.log', 'now')
            """,
            (project.id,),
        )

    deleted = storage.delete_project(project.id)

    assert deleted.name == "Disposable"
    assert not storage.project_dir(project.id).exists()
    assert storage.list_projects() == []
    with storage._connect() as db:
        assert db.execute(
            "SELECT COUNT(*) AS count FROM training_runs WHERE project_id=?",
            (project.id,),
        ).fetchone()["count"] == 0
        assert db.execute(
            "SELECT COUNT(*) AS count FROM jobs WHERE project_id=?",
            (project.id,),
        ).fetchone()["count"] == 0


def test_delete_project_rejects_active_jobs(tmp_path):
    storage = Storage(tmp_path / "state")
    project = storage.create_project("Busy")
    with storage._connect() as db:
        db.execute(
            """
            INSERT INTO jobs(
                id, project_id, job_type, status, command_json, cwd, gpu,
                env_json, log_path, created_at
            ) VALUES ('active-job', ?, 'training', 'queued', '[]', '.', NULL, '{}', 'x.log', 'now')
            """,
            (project.id,),
        )

    try:
        storage.delete_project(project.id)
    except ValueError as exc:
        assert "active-job" in str(exc)
    else:
        raise AssertionError("Expected active job deletion to be rejected")

    assert storage.project_dir(project.id).is_dir()
    assert storage.get_project(project.id).name == "Busy"


def test_running_jobs_become_interrupted_on_restart(tmp_path):
    storage = Storage(tmp_path / "state")
    with storage._connect() as db:
        db.execute(
            """
            INSERT INTO jobs(
                id, project_id, job_type, status, command_json, cwd, gpu,
                env_json, log_path, created_at
            ) VALUES ('job', NULL, 'test', 'running', '[]', '.', NULL, '{}', 'x.log', 'now')
            """
        )
    Storage(tmp_path / "state")
    with storage._connect() as db:
        status = db.execute("SELECT status FROM jobs WHERE id='job'").fetchone()["status"]
    assert status == "interrupted"


def test_legacy_single_source_font_manifest_migrates_to_ordered_lists():
    project = ProjectManifest.from_dict(
        {
            "name": "Legacy",
            "global_source_font": "global.ttf",
            "regional_source_fonts": {"SC": "sc.ttf", "JP": ""},
        }
    )
    assert project.global_source_fonts == ["global.ttf"]
    assert project.regional_source_fonts["SC"] == ["sc.ttf"]
    assert project.source_fonts_for_region("SC") == ["sc.ttf", "global.ttf"]
