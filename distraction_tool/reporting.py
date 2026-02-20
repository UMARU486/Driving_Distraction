from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_events_csv(events: pd.DataFrame, out_dir: Path) -> Path:
    out = out_dir / "distraction_events.csv"
    if events.empty:
        pd.DataFrame(columns=["start_time", "end_time", "duration", "reason"]).to_csv(out, index=False)
    else:
        events.to_csv(out, index=False)
    return out


def save_timeline(frame_df: pd.DataFrame, threshold: float, out_dir: Path) -> Path:
    out = out_dir / "timeline.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_df["time_sec"], frame_df["risk"], color="tab:blue", linewidth=1)
    ax.axhline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold}")
    ax.set_title("Driver Distraction Risk Timeline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Risk score")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_summary(frame_df: pd.DataFrame, events: pd.DataFrame, out_dir: Path, top_n: int = 5) -> Path:
    out = out_dir / "summary.md"
    total = float(events["duration"].sum()) if not events.empty else 0.0
    longest = float(events["duration"].max()) if not events.empty else 0.0
    count = int(len(events))
    avg_risk = float(frame_df["risk"].mean()) if not frame_df.empty else 0.0

    reason_block = "- 无\n"
    if not events.empty:
        reason_dist = events["reason"].value_counts()
        reason_block = "\n".join([f"- {k}: {v}" for k, v in reason_dist.items()])

    top_block = "- 无\n"
    if not events.empty:
        tops = events.sort_values("duration", ascending=False).head(top_n)
        top_block = "\n".join(
            [
                f"- [{i+1}] {r.start_time:.2f}s ~ {r.end_time:.2f}s, {r.duration:.2f}s, reason={r.reason}"
                for i, r in enumerate(tops.itertuples(index=False))
            ]
        )

    content = f"""# 驾驶分心风险分析摘要

- 平均风险分数: {avg_risk:.3f}
- 分心事件总时长: {total:.2f} 秒
- 最长单次分心: {longest:.2f} 秒
- 分心事件次数: {count}

## 原因分布
{reason_block}

## Top {top_n} 高风险片段
{top_block}
"""

    out.write_text(content, encoding="utf-8")
    return out
