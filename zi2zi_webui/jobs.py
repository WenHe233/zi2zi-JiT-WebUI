from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import utc_now
from .storage import Storage


def _command_option(command: list[str], name: str, default: int = 0) -> int:
    try:
        return int(command[command.index(name) + 1])
    except (ValueError, IndexError, TypeError):
        return default


def parse_job_progress(
    line: str,
    job_type: str,
    command: list[str],
) -> tuple[float | None, str]:
    text = line.strip().replace("\r", "")
    if not text:
        return None, ""
    if text.startswith("WEBUI_PROGRESS "):
        try:
            event = json.loads(text.removeprefix("WEBUI_PROGRESS "))
            current = float(event.get("current", 0))
            total = max(float(event.get("total", 1)), 1)
            return min(current / total, 0.99), str(event.get("message", text))[:500]
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if job_type == "training":
        match = re.search(
            r"Epoch:\s*\[(\d+)\].*?\[\s*(\d+)\s*/\s*(\d+)\s*\]",
            text,
        )
        if match:
            epoch, step, steps = (int(value) for value in match.groups())
            epochs = max(_command_option(command, "--epochs", 1), 1)
            progress = (epoch + step / max(steps, 1)) / epochs
            return min(progress, 0.99), f"Epoch {epoch + 1}/{epochs} · step {step}/{steps}"
    percentage = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)%", text)
    if percentage:
        value = min(float(percentage.group(1)) / 100, 0.99)
        return value, text[:500]
    ratio = re.search(r"[\[(]?\s*(\d+)\s*/\s*(\d+)\s*[\])]?", text)
    if ratio:
        current, total = (int(value) for value in ratio.groups())
        if total > 0 and current <= total:
            return min(current / total, 0.99), text[:500]
    return None, text[:500]


def summarize_failure(log_path: str | Path, return_code: int) -> str:
    path = Path(log_path)
    text = (
        path.read_text(encoding="utf-8", errors="replace")[-120000:]
        if path.exists()
        else ""
    )
    if "size mismatch for net.pos_embed" in text or (
        "Unexpected key(s) in state_dict" in text and "size mismatch" in text
    ):
        return (
            "Checkpoint architecture does not match the selected model variant "
            "(JiT-B/16 vs JiT-L/16). Reattach the matching model and retry."
        )
    if "CUDA out of memory" in text:
        return "CUDA out of memory. Choose a lower-memory preset or reduce batch size."
    if "No space left on device" in text:
        return "Disk is full while writing task artifacts."
    if "FileNotFoundError" in text and re.search(
        r"[\\/]+train(?:['\"]|\s|$)",
        text,
        flags=re.IGNORECASE,
    ):
        return (
            "Training dataset directory is missing. Generate or select a dataset "
            "containing train/ and test.npz, then retry."
        )
    if "Can't pickle local object" in text:
        return (
            "A DataLoader worker could not start on Windows because the training "
            "transform is not serializable. Update the training code and retry; "
            "the trailing EOFError is only a secondary worker-process error."
        )
    if "Cannot find a working triton installation" in text:
        return (
            "torch.compile tried to use Triton, but no working Triton installation "
            "is available. The model must fall back to eager execution on this system."
        )
    exception_lines = re.findall(
        r"^(?:[A-Za-z_][\w.]*)(?:Error|Exception):\s+.+$",
        text,
        flags=re.MULTILINE,
    )
    if exception_lines:
        return exception_lines[-1][:2000]
    meaningful = [line.strip() for line in text.splitlines() if line.strip()]
    if meaningful:
        return meaningful[-1][:2000]
    return f"Process exited with code {return_code}"


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
        return self.submit_many(
            [
                {
                    "job_type": job_type,
                    "command": command,
                    "project_id": project_id,
                    "cwd": cwd,
                    "gpu": gpu,
                    "env": env,
                    "resume_point": resume_point,
                }
            ]
        )[0]

    def submit_many(self, specifications: list[dict[str, Any]]) -> list[str]:
        """Persist a batch of jobs in one transaction, then enqueue all of them."""
        if not specifications:
            return []
        records = []
        for specification in specifications:
            job_id = str(uuid4())
            project_id = specification.get("project_id")
            workdir = str(
                Path(specification.get("cwd") or Path.cwd()).resolve()
            )
            log_dir = (
                self.storage.project_dir(project_id) / "logs"
                if project_id
                else self.storage.root / "logs"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            records.append(
                (
                    job_id,
                    project_id,
                    str(specification["job_type"]),
                    json.dumps(specification["command"]),
                    workdir,
                    (
                        None
                        if specification.get("gpu") is None
                        else str(specification["gpu"])
                    ),
                    json.dumps(specification.get("env") or {}),
                    str(log_dir / f"{job_id}.log"),
                    str(specification.get("resume_point") or ""),
                    utc_now(),
                )
            )
        with self._lock:
            with self.storage._connect() as db:
                db.executemany(
                    """
                    INSERT INTO jobs(
                        id, project_id, job_type, status, command_json, cwd, gpu,
                        env_json, log_path, resume_point, created_at
                    ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
        job_ids = [record[0] for record in records]
        for job_id in job_ids:
            self._queue.put(job_id)
        return job_ids

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
        return [self._enrich_failure(dict(row)) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.storage._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._enrich_failure(dict(row))

    def _enrich_failure(self, job: dict[str, Any]) -> dict[str, Any]:
        if job.get("status") != "failed":
            return job
        error = str(job.get("error") or "")
        if error and not error.startswith("Process exited with code"):
            return job
        reason = summarize_failure(job["log_path"], int(job.get("return_code") or 1))
        self._update(
            job["id"],
            error=reason,
            progress_text=f"Failed: {reason}"[:2000],
        )
        job["error"] = reason
        job["progress_text"] = f"Failed: {reason}"[:2000]
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job["status"] == "queued":
            return self._transition(
                job_id,
                {"queued"},
                status="cancelled",
                progress_text="Cancelled before start",
                finished_at=utc_now(),
            )
        with self._lock:
            process = self._processes.get(job_id)
            if not process or process.poll() is not None:
                return False
            if not self._transition(
                job_id,
                {"running"},
                status="cancelled",
                progress_text="Cancelled by user",
                finished_at=utc_now(),
            ):
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
            with self.storage._connect() as db:
                db.execute(
                    """
                    UPDATE jobs
                    SET status='interrupted',
                        progress_text='Interrupted during application shutdown',
                        finished_at=?
                    WHERE status='queued'
                    """,
                    (utc_now(),),
                )
        with self._lock:
            processes = list(self._processes.items())
            for job_id, _process in processes:
                self._transition(
                    job_id,
                    {"running"},
                    status="interrupted",
                    progress_text="Interrupted during application shutdown",
                    finished_at=utc_now(),
                )
        for _job_id, process in processes:
            self._terminate_tree(process)

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
                try:
                    status = self.get_job(job_id)["status"]
                except KeyError:
                    status = ""
                if status not in {"cancelled", "interrupted"}:
                    self._update(
                        job_id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        progress_text=f"Failed: {type(exc).__name__}: {exc}"[:2000],
                        finished_at=utc_now(),
                    )

    def _execute(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        environment = os.environ.copy()
        environment.update(json.loads(job["env_json"]))
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
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
            started = self._transition(
                job_id,
                {"queued"},
                status="running",
                pid=process.pid,
                progress_text="Process started",
                started_at=utc_now(),
            )
            if not started:
                self._terminate_tree(process)
                with self._lock:
                    self._processes.pop(job_id, None)
                return
            assert process.stdout is not None
            command = json.loads(job["command_json"])
            last_progress = float(job.get("progress") or 0)
            last_update = 0.0
            for line in process.stdout:
                log.write(line)
                progress, progress_text = parse_job_progress(
                    line,
                    job["job_type"],
                    command,
                )
                now = time.monotonic()
                values: dict[str, Any] = {}
                if progress is not None and progress >= last_progress:
                    last_progress = progress
                    values["progress"] = progress
                if progress_text and (
                    progress is not None or now - last_update >= 1.0
                ):
                    values["progress_text"] = progress_text
                if values:
                    self._update(job_id, **values)
                    last_update = now
            return_code = process.wait()

        with self._lock:
            self._processes.pop(job_id, None)
        current = self.get_job(job_id)
        if current["status"] in {"cancelled", "interrupted"}:
            self._update(job_id, return_code=return_code, finished_at=utc_now())
        elif return_code == 0:
            self._transition(
                job_id,
                {"running"},
                status="succeeded",
                progress=1.0,
                progress_text="Completed",
                return_code=return_code,
                finished_at=utc_now(),
            )
        else:
            reason = summarize_failure(log_path, return_code)
            self._transition(
                job_id,
                {"running"},
                status="failed",
                return_code=return_code,
                error=reason,
                progress_text=f"Failed: {reason}"[:2000],
                finished_at=utc_now(),
            )

    def _terminate_tree(
        self,
        process: subprocess.Popen[str],
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
        except Exception:
            if process.poll() is not None:
                return
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    def _update(self, job_id: str, **values: Any) -> None:
        if not job_id or not values:
            return
        columns = ", ".join(f"{key}=?" for key in values)
        with self._lock:
            with self.storage._connect() as db:
                db.execute(
                    f"UPDATE jobs SET {columns} WHERE id=?",
                    [*values.values(), job_id],
                )

    def _transition(
        self,
        job_id: str,
        expected: set[str],
        **values: Any,
    ) -> bool:
        if not job_id or not values or not expected:
            return False
        columns = ", ".join(f"{key}=?" for key in values)
        placeholders = ", ".join("?" for _ in expected)
        with self._lock:
            with self.storage._connect() as db:
                cursor = db.execute(
                    f"UPDATE jobs SET {columns} "
                    f"WHERE id=? AND status IN ({placeholders})",
                    [*values.values(), job_id, *sorted(expected)],
                )
                return cursor.rowcount == 1
