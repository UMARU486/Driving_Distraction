from __future__ import annotations

from pathlib import Path

import typer

from distraction_tool.analyzer import AnalyzeConfig, analyze_video
from distraction_tool.reporting import save_events_csv, save_summary, save_timeline

app = typer.Typer(add_completion=False, help="Driver Distraction Risk MVP")


@app.command()
def analyze(
    video_path: Path = typer.Argument(..., exists=True, readable=True, help="Input video path (.mp4/.mov)"),
    out: Path = typer.Option(Path("outputs"), "--out", help="Output directory"),
    threshold: float = typer.Option(2.0, "--threshold", help="Risk threshold for event extraction"),
    clip: bool = typer.Option(False, "--clip", help="Export high-risk clips to outputs/clips/ (requires ffmpeg)"),
) -> None:
    cfg = AnalyzeConfig(video_path=video_path, out_dir=out, threshold=threshold, clip=clip)
    frame_df, events = analyze_video(cfg)
    out.mkdir(parents=True, exist_ok=True)

    save_events_csv(events, out)
    save_timeline(frame_df, threshold, out)
    save_summary(frame_df, events, out)

    typer.echo(f"Analysis completed. Outputs saved to: {out}")


if __name__ == "__main__":
    app()
