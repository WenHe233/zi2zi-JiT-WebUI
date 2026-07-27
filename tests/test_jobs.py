import sys
import time

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
