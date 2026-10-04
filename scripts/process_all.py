import os, glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "data" / "input" / "reference_videos" / "zaid"
OUT_DIR = ROOT_DIR / "data" / "processed" / "v1"
MODEL_PATH = str(ROOT_DIR / "models" / "pose_landmarker_lite.task")
SHOOTING_ARM = "r"
N_POINTS = 100

os.makedirs(OUT_DIR, exist_ok=True)
IDX = {"shoulder": 12, "elbow": 14, "wrist": 16, "index": 20} if SHOOTING_ARM == "r" \
      else {"shoulder": 11, "elbow": 13, "wrist": 15, "index": 19}

def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def process_video(path):
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, int(i * 1000 / fps))
            row = {"frame": i, "time_s": i / fps}
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                pts = {k: np.array([lms[v].x * w, lms[v].y * h]) for k, v in IDX.items()}
                row["elbow_angle"] = angle(pts["shoulder"], pts["elbow"], pts["wrist"])
                row["wrist_angle"] = angle(pts["elbow"], pts["wrist"], pts["index"])
                arm_len = np.linalg.norm(pts["shoulder"] - pts["elbow"]) + 1e-9
                row["wrist_height"] = (pts["shoulder"][1] - pts["wrist"][1]) / arm_len
            rows.append(row)
            i += 1
    cap.release()
    return pd.DataFrame(rows)

def normalize(df, cols):
    out = {}
    for c in cols:
        s = df[c].interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).mean()
        out[c] = np.interp(np.linspace(0, len(s) - 1, N_POINTS), np.arange(len(s)), s)
    return pd.DataFrame(out)

cols = ["elbow_angle", "wrist_angle", "wrist_height"]
all_shots = []

video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
                     glob.glob(os.path.join(VIDEO_DIR, "*.mov")))
print(f"Found {len(video_files)} videos in '{VIDEO_DIR}'")

for path in video_files:
    name = os.path.splitext(os.path.basename(path))[0]
    print("Processing", name)
    df = process_video(path)
    df.to_csv(os.path.join(OUT_DIR, f"{name}_raw.csv"), index=False)
    if any(c not in df.columns for c in cols) or df[cols].notna().sum().min() < 10:
        print("  skipped (pose barely detected)")
        continue
    norm = normalize(df, cols)
    norm["video"] = name
    norm["pct_of_shot"] = np.arange(N_POINTS)
    all_shots.append(norm)

if not all_shots:
    raise SystemExit("No videos were processed. Check the folder name and file types.")

combined = pd.concat(all_shots)
combined.to_csv(os.path.join(OUT_DIR, "all_shots_normalized.csv"), index=False)

avg = combined.groupby("pct_of_shot")[cols].mean()
std = combined.groupby("pct_of_shot")[cols].std()
avg.to_csv(os.path.join(OUT_DIR, "average_shot.csv"))

fig, axes = plt.subplots(len(cols), 1, figsize=(8, 8), sharex=True)
for ax, c in zip(axes, cols):
    for _, g in combined.groupby("video"):
        ax.plot(g["pct_of_shot"], g[c], color="gray", alpha=0.25)
    ax.plot(avg.index, avg[c], color="C0", linewidth=2.5)
    ax.fill_between(avg.index, avg[c] - std[c], avg[c] + std[c], color="C0", alpha=0.2)
    ax.set_ylabel(c)
axes[-1].set_xlabel("% of shot")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "average_shot.png"), dpi=150)
plt.show()
print("Done. Check the output folder.")