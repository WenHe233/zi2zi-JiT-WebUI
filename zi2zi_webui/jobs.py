from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import utc_now
from .storage import Storage


class JobManager:
    """Persistent, single-worker subprocess queue."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._worker = threading.Thread(target=self._run, name="zi2zi-job-worker", daemon=True)
        self._worker.start()
        self._restore_queued()

    def _restore_queued(self) -> None:
        with self.storage._connect() as db:
            rows = db.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"
            ).fetchall()
        for row in rows:
            self._queue.put(row["id"])

    def submit(
        self,
        job_type: str,
        command: list[str],
        *,
        project_id: str | None = None,
        cwd: str | Path | None = None,
        gpu: str | int | None = None,
        env: dict[str, str] | None = None,
        resume_point: str | Path | None = None,
    ) -> str:
        job_id = str(uuid4())
        workdir = str(Path(cwd or Path.cwd()).resolve())
        log_dir = self.storage.project_dir(project_id) / "logs" if project_id else self.storage.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        with self.storage._connect() as db:
            db.execute(
                """
                INSERT INTO jobs(
                    id, project_id, job_type, status, command_json, cwd, gpu,
                    env_json, log_path, resume_point, created_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    job_type,
                    json.dumps(command),
                    workdir,
                    None if gpu is None else str(gpu),
                    json.dumps(env or {}),
                    str(log_path),
                    str(resume_point or ""),
                    utc_now(),
                ),
            )
        self._queue.put(job_id)
        return job_id

    def list_jobs(self, project_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        values: list[Any] = []
        if project_id:
            query += " WHERE project_id=?"
            values.append(project_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self.storage._connect() as db:
            rows = db.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.storage._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return dict(row)

    def cancel(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job["status"] == "queued":
            self._update(job_id, status="cancelled", finished_at=utc_now())
            return True
        with self._lock:
            process = self._processes.get(job_id)
        if not process:
            return False
        self._terminate_tree(process)
        return True

    def retry(self, job_id: str) -> str:
        job = self.get_job(job_id)
        if job["status"] not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("Only failed, cancelled, or interrupted jobs can be retried")
        return self.submit(
            job["job_type"],
            json.loads(job["command_json"]),
            project_id=job["project_id"],
            cwd=job["cwd"],
            gpu=job["gpu"],
            env=json.loads(job["env_json"]),
            resume_point=job.get("resume_point"),
        )

    def read_log(self, job_id: str, max_chars: int = 40000) -> str:
        path = Path(self.get_job(job_id)["log_path"])
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            self._terminate_tree(process, final_status="interrupted")

    def _run(self) -> None:
        while not self._stopping.is_set():
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                job = self.get_job(job_id)
                if job["status"] != "queued":
                    continue
                self._execute(job)
            except Exception as exc:
                self._update(
                    job_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=utc_now(),
                )

    def _execute(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        environment = os.environ.copy()
        environment.update(json.loads(job["env_json"]))
        if job["gpu"] not in (None, ""):
            environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])

        creationflags = 0
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"[{utc_now()}] $ {' '.join(json.loads(job['command_json']))}\n")
            process = subprocess.Popen(
                json.loads(job["command_json"]),
                cwd=job["cwd"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                **popen_kwargs,
            )
            with self._lock:
                self._processes[job_id] = process
            self._update(job_id, status="running", pid=process.pid, started_at=utc_now())
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
            return_code = process.wait()

        with self._lock:
            self._processes.pop(job_id, None)
        current = self.get_job(job_id)
        if current["status"] in {"cancelled", "interrupted"}:
            self._update(job_id, return_code=return_code, finished_at=utc_now())
        elif return_code == 0:
            self._update(
                job_id,
                status="succeeded",
                progress=1.0,
                return_code=return_code,
                finished_at=utc_now(),
            )
        else:
            self._update(
                job_id,
                status="failed",
                return_code=return_code,
                error=f"Process exited with code {return_code}",
                finished_at=utc_now(),
            )

    def _terminate_tree(
        self,
        process: subprocess.Popen[str],
        *,
        final_status: str = "cancelled",
    ) -> None:
        try:
            import psutil

            root = psutil.Process(process.pid)
            children = root.children(recursive=True)
            for child in children:
                child.terminate()
            root.terminate()
            _, alive = psutil.wait_procs([root, *children], timeout=5)
            for item in alive:
                item.kill()
        except (ImportError, OSError):
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        self._update(
            next((key for key, value in self._processes.items() if value is process), ""),
            status=final_status,
            finished_at=utc_now(),
        )

    def _update(self, job_id: str, **values: Any) -> None:
        if not job_id or not values:
            return
        columns = ", ".join(f"{key}=?" for key in values)
        with self.storage._connect() as db:
            db.execute(
                f"UPDATE jobs SET {columns} WHERE id=?",
                [*values.values(), job_id],
            )
