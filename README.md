# Driver Distraction Risk MVP

基于可解释视觉代理指标（head pose / face orientation）的驾驶分心风险检测最小可用版本。

> 设计原则：
> - 输入应来自公开视频或你自录且不含个人敏感信息的数据
> - 输出用于行为风险分析，不用于个人身份判断

## 功能

输入视频（`.mp4` / `.mov`），输出到 `outputs/`：

- `distraction_events.csv`：分心事件段（`start_time, end_time, duration, reason`）
- `timeline.png`：逐帧风险分数时间轴图
- `summary.md`：总分心时长、最长一次、原因分布、Top N 高风险片段
- `clips/`（可选）：自动切出高风险片段（需要 ffmpeg + `--clip`）

## 环境

- Python 3.11+
- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `matplotlib`
- `typer`
- `pytest`

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 一键运行

```bash
python -m distraction_tool analyze path/to/video.mp4 --out outputs/ --threshold 2.0 --clip
```

参数：
- `--threshold`：事件阈值，默认 `2.0`
- `--clip`：启用高风险片段切出（依赖 ffmpeg）

## 方法说明
1. 使用 MediaPipe Face Mesh 提取人脸关键点。
2. 用 `solvePnP` 估计头部姿态（yaw/pitch）。
3. 将 `yaw/pitch + face_visible` 映射为风险分数：
   - 正常朝前：低分
   - 明显偏航（看侧方）、俯仰偏移（看下/看上）：中高分
   - 人脸不可见：高分（可能遮挡或偏离视线）
4. 对连续超过阈值的时间段做事件合并，导出 CSV/图/摘要。

## 测试

```bash
pytest
```

## 局限性

- 这是 MVP：风险是“代理指标”而非医学级注意力诊断。
- 单目视频下，快速光照变化和强遮挡会影响鲁棒性。
- 建议结合车辆信号（转向、速度）和场景上下文进一步增强。
