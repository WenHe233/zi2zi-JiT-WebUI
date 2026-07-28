import sys
import threading
import time

import pytest

from zi2zi_webui.jobs import JobManager, parse_job_progress, summarize_failure
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


def test_progress_parser_supports_training_and_generic_tasks():
    progress, text = parse_job_progress(
        "Epoch: [2]  [ 5/10] eta: 0:01:00",
        "training",
        ["python", "train.py", "--epochs", "10"],
    )
    assert progress == 0.25
    assert "Epoch 3/10" in text
    assert parse_job_progress("Downloading 40%", "model_download", [])[0] == 0.4


def test_failure_summary_explains_checkpoint_architecture_mismatch(tmp_path):
    log = tmp_path / "failed.log"
    log.write_text(
        "RuntimeError: Error(s) in loading state_dict\n"
        "Unexpected key(s) in state_dict\n"
        "size mismatch for net.pos_embed\n",
        encoding="utf-8",
    )
    assert "JiT-B/16 vs JiT-L/16" in summarize_failure(log, 1)


def test_failure_summary_explains_missing_training_dataset(tmp_path):
    log = tmp_path / "failed.log"
    log.write_text(
        "FileNotFoundError: [WinError 3] path not found: "
        "'C:\\\\repo\\\\train'\n",
        encoding="utf-8",
    )
    assert "Training dataset directory is missing" in summarize_failure(log, 1)


def test_failure_summary_prefers_windows_pickle_root_cause_over_eof(tmp_path):
    log = tmp_path / "failed.log"
    log.write_text(
        "AttributeError: Can't pickle local object 'main.<locals>.<lambda>'\n"
        "EOFError: Ran out of input\n",
        encoding="utf-8",
    )
    summary = summarize_failure(log, 1)
    assert "DataLoader worker could not start on Windows" in summary
    assert "secondary" in summary


def test_failure_summary_explains_missing_triton_fallback(tmp_path):
    log = tmp_path / "failed.log"
    log.write_text(
        "RuntimeError: Cannot find a working triton installation.\n",
        encoding="utf-8",
    )
    assert "fall back to eager execution" in summarize_failure(log, 1)


def test_live_job_progress_is_persisted(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    try:
        job_id = manager.submit(
            "test",
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('Downloading 25%', flush=True); time.sleep(1)",
            ],
            cwd=tmp_path,
        )
        deadline = time.time() + 5
        observed = False
        while time.time() < deadline:
            job = manager.get_job(job_id)
            if float(job["progress"]) >= 0.25:
                observed = True
                break
            time.sleep(0.05)
        assert observed
        assert "25%" in manager.get_job(job_id)["progress_text"]
    finally:
        manager.stop()


def test_submit_many_persists_all_jobs_before_enqueuing(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    try:
        job_ids = manager.submit_many(
            [
                {
                    "job_type": "generation",
                    "command": [sys.executable, "-c", "print('SC')"],
                    "cwd": tmp_path,
                },
                {
                    "job_type": "generation",
                    "command": [sys.executable, "-c", "print('TC')"],
                    "cwd": tmp_path,
                },
            ]
        )
        assert len(job_ids) == 2
        assert {job["id"] for job in manager.list_jobs()} == set(job_ids)
    finally:
        manager.stop()


def test_submit_many_does_not_persist_a_partial_invalid_batch(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    try:
        with pytest.raises(KeyError):
            manager.submit_many(
                [
                    {
                        "job_type": "generation",
                        "command": [sys.executable, "-c", "print('valid')"],
                        "cwd": tmp_path,
                    },
                    {"job_type": "generation", "cwd": tmp_path},
                ]
            )
        assert manager.list_jobs() == []
    finally:
        manager.stop()


def _wait_for_status(manager, job_id, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job["status"] in statuses:
            return job
        time.sleep(0.05)
    return manager.get_job(job_id)


def test_running_job_cancellation_is_not_overwritten_by_worker_exit(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    try:
        job_id = manager.submit(
            "test",
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('started', flush=True); time.sleep(30)",
            ],
            cwd=tmp_path,
        )
        assert _wait_for_status(manager, job_id, {"running"}).get("status") == "running"
        assert manager.cancel(job_id)
        job = _wait_for_status(manager, job_id, {"cancelled"})
        time.sleep(0.2)
        assert job["status"] == "cancelled"
        assert manager.get_job(job_id)["status"] == "cancelled"
    finally:
        manager.stop()


def test_application_stop_interrupts_running_and_queued_jobs(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    running = manager.submit(
        "test",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
    )
    queued = manager.submit(
        "test",
        [sys.executable, "-c", "print('should not start')"],
        cwd=tmp_path,
    )
    assert _wait_for_status(manager, running, {"running"}).get("status") == "running"
    manager.stop()
    assert manager.get_job(running)["status"] == "interrupted"
    assert manager.get_job(queued)["status"] == "interrupted"


def test_concurrent_job_updates_are_serialized(tmp_path):
    storage = Storage(tmp_path / "state")
    manager = JobManager(storage)
    try:
        job_id = manager.submit(
            "test",
            [sys.executable, "-c", "import time; time.sleep(2)"],
            cwd=tmp_path,
        )
        threads = [
            threading.Thread(
                target=lambda index=index: [
                    manager._update(job_id, progress_text=f"update-{index}-{step}")
                    for step in range(20)
                ]
            )
            for index in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert manager.get_job(job_id)["id"] == job_id
    finally:
        manager.stop()
