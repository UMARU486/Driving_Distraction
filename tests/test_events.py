import pandas as pd

from distraction_tool.analyzer import detect_events


def test_detect_events_groups_continuous_risk_segments():
    frame_df = pd.DataFrame(
        [
            {"time_sec": 0.0, "risk": 0.1, "reason": "on_road"},
            {"time_sec": 1.0, "risk": 2.5, "reason": "looking_side"},
            {"time_sec": 2.0, "risk": 2.9, "reason": "looking_side"},
            {"time_sec": 3.0, "risk": 0.2, "reason": "on_road"},
            {"time_sec": 4.0, "risk": 2.1, "reason": "looking_down"},
            {"time_sec": 5.0, "risk": 0.3, "reason": "on_road"},
        ]
    )

    events = detect_events(frame_df, threshold=2.0, min_duration=0.5)

    assert len(events) == 2
    assert events.iloc[0]["start_time"] == 1.0
    assert events.iloc[0]["end_time"] == 3.0
    assert events.iloc[0]["reason"] == "looking_side"


def test_detect_events_respects_min_duration():
    frame_df = pd.DataFrame(
        [
            {"time_sec": 0.0, "risk": 2.2, "reason": "looking_side"},
            {"time_sec": 0.2, "risk": 0.1, "reason": "on_road"},
        ]
    )

    events = detect_events(frame_df, threshold=2.0, min_duration=1.0)

    assert events.empty
