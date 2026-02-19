from pathlib import Path

import pandas as pd

from distraction_tool.reporting import save_events_csv, save_summary


def test_save_events_csv_empty(tmp_path: Path):
    out = save_events_csv(pd.DataFrame(), tmp_path)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "start_time" in content


def test_save_summary_contains_metrics(tmp_path: Path):
    frame_df = pd.DataFrame([{"risk": 1.0}, {"risk": 3.0}])
    events = pd.DataFrame(
        [
            {"start_time": 1.0, "end_time": 4.0, "duration": 3.0, "reason": "looking_side"},
            {"start_time": 7.0, "end_time": 8.0, "duration": 1.0, "reason": "looking_down"},
        ]
    )
    out = save_summary(frame_df, events, tmp_path, top_n=1)
    content = out.read_text(encoding="utf-8")

    assert "分心事件总时长: 4.00 秒" in content
    assert "Top 1" in content
