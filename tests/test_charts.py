from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import matplotlib

from zi2zi_webui.charts import training_figures


def _render_empty_dashboard() -> list[int]:
    sizes = []
    for figure in training_figures([]):
        output = BytesIO()
        figure.savefig(output, format="png")
        sizes.append(len(output.getvalue()))
    return sizes


def test_dashboard_uses_thread_safe_non_gui_backend():
    assert matplotlib.get_backend().lower() == "agg"
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: _render_empty_dashboard(), range(12)))
    assert all(len(sizes) == 4 and all(size > 0 for size in sizes) for sizes in results)


def test_dashboard_does_not_register_pyplot_figures():
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    _render_empty_dashboard()
    assert set(plt.get_fignums()) == before
