from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd


@dataclass
class AnalyzeConfig:
    video_path: Path
    out_dir: Path
    threshold: float = 2.0
    clip: bool = False
    clip_padding: float = 1.0
    min_event_duration: float = 1.0


@dataclass
class FrameRisk:
    time_sec: float
    risk: float
    yaw_deg: float
    pitch_deg: float
    face_visible: int
    reason: str


def _estimate_head_pose(frame: np.ndarray, face_landmarks, image_w: int, image_h: int) -> tuple[float, float]:
    lm = face_landmarks.landmark
    image_points = np.array(
        [
            (lm[1].x * image_w, lm[1].y * image_h),
            (lm[152].x * image_w, lm[152].y * image_h),
            (lm[33].x * image_w, lm[33].y * image_h),
            (lm[263].x * image_w, lm[263].y * image_h),
            (lm[61].x * image_w, lm[61].y * image_h),
            (lm[291].x * image_w, lm[291].y * image_h),
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ]
    )
    focal_length = image_w
    center = (image_w / 2, image_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))
    success, rot_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rot_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = float(euler[0])
    yaw = float(euler[1])
    return yaw, pitch


def _risk_from_pose(yaw: float, pitch: float, face_visible: bool) -> tuple[float, str]:
    if not face_visible:
        return 4.0, "face_not_visible"

    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw < 20 and abs_pitch < 15:
        return 0.2, "on_road"

    yaw_risk = max(0.0, (abs_yaw - 20) / 15)
    pitch_risk = max(0.0, (abs_pitch - 15) / 12)
    risk = min(5.0, yaw_risk + pitch_risk)

    if abs_yaw >= 35:
        reason = "looking_side"
    elif pitch <= -20:
        reason = "looking_up"
    elif pitch >= 20:
        reason = "looking_down"
    else:
        reason = "minor_off_road"

    return risk, reason


def analyze_video(config: AnalyzeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {config.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    frame_risks: list[FrameRisk] = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            face_visible = bool(result.multi_face_landmarks)
            yaw = 0.0
            pitch = 0.0
            if face_visible:
                yaw, pitch = _estimate_head_pose(frame, result.multi_face_landmarks[0], w, h)
            risk, reason = _risk_from_pose(yaw, pitch, face_visible)
            frame_risks.append(
                FrameRisk(
                    time_sec=frame_idx / fps,
                    risk=risk,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    face_visible=int(face_visible),
                    reason=reason,
                )
            )
            frame_idx += 1

    cap.release()

    frame_df = pd.DataFrame([r.__dict__ for r in frame_risks])
    event_df = detect_events(frame_df, config.threshold, config.min_event_duration)

    if config.clip and not event_df.empty:
        clip_dir = config.out_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        export_clips(config.video_path, event_df, clip_dir, config.clip_padding)

    return frame_df, event_df


def detect_events(frame_df: pd.DataFrame, threshold: float, min_duration: float) -> pd.DataFrame:
    events: list[dict] = []
    in_event = False
    start = 0.0
    reasons: list[str] = []

    for row in frame_df.itertuples(index=False):
        high = row.risk >= threshold
        if high and not in_event:
            in_event = True
            start = row.time_sec
            reasons = [row.reason]
        elif high and in_event:
            reasons.append(row.reason)
        elif (not high) and in_event:
            end = row.time_sec
            duration = end - start
            if duration >= min_duration:
                reason = pd.Series(reasons).value_counts().index[0]
                events.append({"start_time": start, "end_time": end, "duration": duration, "reason": reason})
            in_event = False

    if in_event:
        end = float(frame_df.iloc[-1]["time_sec"])
        duration = end - start
        if duration >= min_duration:
            reason = pd.Series(reasons).value_counts().index[0]
            events.append({"start_time": start, "end_time": end, "duration": duration, "reason": reason})

    return pd.DataFrame(events)


def export_clips(video_path: Path, event_df: pd.DataFrame, clip_dir: Path, padding: float = 1.0) -> None:
    for idx, event in event_df.iterrows():
        start = max(0.0, float(event["start_time"]) - padding)
        duration = float(event["duration"]) + padding * 2
        clip_path = clip_dir / f"event_{idx:03d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(clip_path),
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
