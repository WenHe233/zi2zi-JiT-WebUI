from zi2zi_webui.app import build_app
from zi2zi_webui.jobs import JobManager
from zi2zi_webui.storage import Storage


def test_dynamic_timer_dropdowns_accept_restored_browser_values(tmp_path):
    storage = Storage(tmp_path / "state")
    jobs = JobManager(storage)
    try:
        app = build_app(storage, jobs, language="en")
        dropdowns = {
            block.label: block
            for block in app.blocks.values()
            if hasattr(block, "allow_custom_value")
        }
        assert dropdowns["Current task"].allow_custom_value
        assert dropdowns["Compare runs (2–5)"].allow_custom_value
        assert dropdowns["Current task"].preprocess("restored-job-id") == "restored-job-id"
        assert dropdowns["Compare runs (2–5)"].preprocess(
            ["restored-run-id"]
        ) == ["restored-run-id"]
    finally:
        jobs.stop()
