import sys
import time

from zi2zi_webui.jobs import JobManager
from zi2zi_webui.storage import Storage


def test_single_worker_executes_job(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    try:
        job_id = manager.submit(
            "test",
            [sys.executable, "-c", "print('hello from worker')"],
            cwd=tmp_path,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            job = manager.get_job(job_id)
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert manager.get_job(job_id)["status"] == "succeeded"
        assert "hello from worker" in manager.read_log(job_id)
    finally:
        manager.stop()
