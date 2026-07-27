from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import ProjectManifest, TrainingRun, utc_now


PROJECT_FOLDERS = (
    "inputs",
    "datasets",
    "training",
    "generation",
    "glyphs",
    "svg",
    "fonts",
    "exports",
    "logs",
)


class Storage:
    """Filesystem artifacts plus a small SQLite index.

    Large or user-editable artifacts remain ordinary files. SQLite stores only
    searchable metadata and job state, making projects portable and inspectable.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.models_dir = self.root / "models"
        self.projects_dir = self.root / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "state.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    gpu TEXT,
                    env_json TEXT NOT NULL,
                    pid INTEGER,
                    return_code INTEGER,
                    log_path TEXT NOT NULL,
                    error TEXT,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    progress REAL NOT NULL DEFAULT 0,
                    progress_text TEXT NOT NULL DEFAULT '',
                    resume_point TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS jobs_project_idx ON jobs(project_id, created_at);
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for name, definition in (
                ("schema_version", "INTEGER NOT NULL DEFAULT 1"),
                ("progress", "REAL NOT NULL DEFAULT 0"),
                ("progress_text", "TEXT NOT NULL DEFAULT ''"),
                ("resume_point", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
            db.execute(
                "UPDATE jobs SET status='interrupted', finished_at=? WHERE status='running'",
                (utc_now(),),
            )

    def project_dir(self, project_id: str) -> Path:
        path = (self.projects_dir / project_id).resolve()
        if self.projects_dir not in path.parents:
            raise ValueError("Invalid project id")
        return path

    def create_project(self, name: str, **overrides: Any) -> ProjectManifest:
        name = name.strip()
        if not name:
            raise ValueError("Project name is required")
        manifest = ProjectManifest(name=name, **overrides)
        project_dir = self.project_dir(manifest.id)
        project_dir.mkdir(parents=True, exist_ok=False)
        for folder in PROJECT_FOLDERS:
            (project_dir / folder).mkdir()
        self.save_project(manifest)
        return manifest

    def save_project(self, manifest: ProjectManifest) -> None:
        manifest.updated_at = utc_now()
        project_dir = self.project_dir(manifest.id)
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / "project.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        with self._lock, self._connect() as db:
            db.execute(
                """
                INSERT INTO projects(id, name, manifest_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    manifest_path=excluded.manifest_path,
                    updated_at=excluded.updated_at
                """,
                (manifest.id, manifest.name, str(path), manifest.created_at, manifest.updated_at),
            )

    def get_project(self, project_id: str) -> ProjectManifest:
        path = self.project_dir(project_id) / "project.json"
        if not path.is_file():
            raise KeyError(project_id)
        return ProjectManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_projects(self) -> list[ProjectManifest]:
        with self._connect() as db:
            rows = db.execute("SELECT id FROM projects ORDER BY updated_at DESC").fetchall()
        projects = []
        for row in rows:
            try:
                projects.append(self.get_project(row["id"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
        return projects

    def save_training_run(self, run: TrainingRun) -> None:
        run.updated_at = utc_now()
        payload = json.dumps(run.to_dict(), ensure_ascii=False)
        with self._lock, self._connect() as db:
            db.execute(
                """
                INSERT INTO training_runs(id, project_id, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (run.id, run.project_id, run.status, payload, run.created_at, run.updated_at),
            )

    def list_training_runs(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload_json FROM training_runs WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def delete_training_run(self, project_id: str, run_id: str) -> None:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT id FROM training_runs WHERE id=? AND project_id=?",
                (run_id, project_id),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            db.execute("DELETE FROM training_runs WHERE id=?", (run_id,))
        training_root = (self.project_dir(project_id) / "training").resolve()
        run_dir = (training_root / run_id).resolve()
        if training_root not in run_dir.parents:
            raise ValueError("Invalid training run id")
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def replace_training_job(
        self,
        project_id: str,
        old_job_id: str,
        new_job_id: str,
    ) -> None:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT id, payload_json FROM training_runs WHERE project_id=?",
                (project_id,),
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload_json"])
                parameters = payload.get("parameters", {})
                if parameters.get("job_id") != old_job_id:
                    continue
                parameters["job_id"] = new_job_id
                payload["parameters"] = parameters
                payload["status"] = "queued"
                db.execute(
                    """
                    UPDATE training_runs
                    SET status='queued', payload_json=?, updated_at=?
                    WHERE id=?
                    """,
                    (json.dumps(payload, ensure_ascii=False), utc_now(), row["id"]),
                )
