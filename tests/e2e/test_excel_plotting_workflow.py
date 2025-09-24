import io
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # noqa: F401
    MATPLOT = True
except Exception:
    MATPLOT = False

try:
    import xlsxwriter  # noqa: F401
    XLSX = True
except Exception:
    XLSX = False


@pytest.mark.e2e
@pytest.mark.slow
def test_export_metrics_and_plot_to_excel(tmp_path):
    if not (MATPLOT and XLSX):
        pytest.skip("matplotlib/xlsxwriter not available; skipping Excel workflow.")

    # Fake metrics you might produce in an earlier E2E
    metrics = {
        "less_greedy": {"rmse": 12.3, "mae": 9.1, "r2": 0.74},
        "cart": {"rmse": 14.2, "mae": 10.0, "r2": 0.69},
    }
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"})
    xlsx = tmp_path / "metrics.xlsx"

    # Write table + an image
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="metrics")
        workbook = writer.book
        sheet = workbook.add_worksheet("plot")
        writer.sheets["plot"] = sheet

        # Simple plot to memory
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        ax.bar(df["model"], df["r2"])
        ax.set_title("R2 by model")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        sheet.insert_image("B2", "r2.png", {"image_data": buf})

    assert xlsx.exists() and xlsx.stat().st_size > 0