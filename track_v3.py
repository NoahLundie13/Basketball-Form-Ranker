import os, glob, math
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

VIDEO_DIR = "."
OUT_DIR = "output_v3"
POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
USE_HANDS = os.path.exists(HAND_MODEL)
SHOOTING_ARM = "r"
N_POINTS = 100
MIN_VIS = 0.0

os.makedirs(OUT_DIR, exist_ok=True)

SIDES = {
    "l": dict(shoulder=11, elbow=13, wrist=15, index=19, hip=23, knee=25, ankle=27),
    "r": dict(shoulder=12, elbow=14, wrist=16, index=20, hip=24, knee=26, ankle=28),
}
GUIDE_ARM = "l" if SHOOTING_ARM == "r" else "r"
ARMS = {"shoot": SIDES[SHOOTING_ARM], "guide": SIDES[GUIDE_ARM]}

REQUIRED_COLS = []
for arm in ARMS:
    REQUIRED_COLS += [f"{arm}_elbow_angle", f"{arm}_wrist_angle", f"{arm}_shoulder_angle"]
REQUIRED_COLS += ["hip_height", "torso_lean", "nose_x_offset", "nose_y_offset"]
LEG_COLS = []
for side in ARMS:
    LEG_COLS += [f"{side}_knee_angle", f"{side}_thigh_angle"]
HAND_COLS = ["finger_spread", "index_extension"]
ALL_COLS = REQUIRED_COLS + LEG_COLS + (HAND_COLS if USE_HANDS else [])


def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def process_video(path):
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    hand_ctx = None
    if USE_HANDS:
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        hand_ctx = vision.HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows = []
    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i * 1000 / fps)
            res = pose_lm.detect_for_video(mp_img, ts)
            row = {"frame": i, "time_s": i / fps}

            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                P = lambda idx: np.array([lms[idx].x * w, lms[idx].y * h])
                vis = lambda idx: (lms[idx].visibility or 0)

                sh_mid = (P(11) + P(12)) / 2
                hip_mid = (P(23) + P(24)) / 2
                torso = np.linalg.norm(sh_mid - hip_mid) + 1e-9

                for arm, ix in ARMS.items():
                    s, e, wr, idx, hp = (P(ix[k]) for k in ("shoulder", "elbow", "wrist", "index", "hip"))
                    row[f"{arm}_elbow_angle"] = angle(s, e, wr)
                    row[f"{arm}_wrist_angle"] = angle(e, wr, idx)
                    row[f"{arm}_shoulder_angle"] = angle(hp, s, e)

                    # legs (only if the points are actually visible)
                    if vis(ix["hip"]) > MIN_VIS and vis(ix["knee"]) > MIN_VIS:
                        v = P(ix["knee"]) - P(ix["hip"])
                        row[f"{arm}_thigh_angle"] = math.degrees(math.atan2(abs(v[0]), v[1]))  # 0 = vertical
                        if vis(ix["ankle"]) > MIN_VIS:
                            row[f"{arm}_knee_angle"] = angle(P(ix["hip"]), P(ix["knee"]), P(ix["ankle"]))

                row["hip_y_raw"] = hip_mid[1] / torso
                v = sh_mid - hip_mid
                row["torso_lean"] = math.degrees(math.atan2(v[0], -v[1]))

                nose = P(0)
                row["nose_x_offset"] = (nose[0] - sh_mid[0]) / torso
                row["nose_y_offset"] = (sh_mid[1] - nose[1]) / torso

                if hand_ctx is not None:
                    hres = hand_ctx.detect_for_video(mp_img, ts)
                    if hres.hand_landmarks:
                        pose_wrist = P(ARMS["shoot"]["wrist"])
                        best, best_d = None, 1e9
                        for hl in hres.hand_landmarks:
                            d = np.linalg.norm(np.array([hl[0].x * w, hl[0].y * h]) - pose_wrist)
                            if d < best_d:
                                best, best_d = hl, d
                        if best is not None and best_d < torso * 0.6:
                            H = lambda k: np.array([best[k].x * w, best[k].y * h])
                            row["finger_spread"] = angle(H(8), H(0), H(20))
                            palm = np.linalg.norm(H(5) - H(0)) + 1e-9
                            row["index_extension"] = np.linalg.norm(H(8) - H(0)) / palm
            rows.append(row)
            i += 1
    cap.release()
    if hand_ctx is not None:
        hand_ctx.close()

    df = pd.DataFrame(rows)
    if "hip_y_raw" in df and df["hip_y_raw"].notna().any():
        df["hip_height"] = -(df["hip_y_raw"] - df["hip_y_raw"].dropna().iloc[0])
    return df


def normalize(df, cols):
    out = {}
    for c in cols:
        if c not in df or df[c].notna().sum() < 10:
            out[c] = np.full(N_POINTS, np.nan)
            continue
        s = df[c].interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).mean()
        out[c] = np.interp(np.linspace(0, len(s) - 1, N_POINTS), np.arange(len(s)), s)
    return pd.DataFrame(out)


video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
                     glob.glob(os.path.join(VIDEO_DIR, "*.mov")))
print(f"Found {len(video_files)} videos in '{VIDEO_DIR}'")
print("Finger tracking:", "ON" if USE_HANDS else "OFF (hand_landmarker.task not found)")

all_shots = []
for path in video_files:
    name = os.path.splitext(os.path.basename(path))[0]
    print("Processing", name)
    df = process_video(path)
    df.to_csv(os.path.join(OUT_DIR, f"{name}_raw.csv"), index=False)
    if any(c not in df.columns or df[c].notna().sum() < 10 for c in REQUIRED_COLS):
        print("  skipped (pose barely detected)")
        continue
    leg_ok = [c for c in LEG_COLS if c in df.columns and df[c].notna().sum() >= 10]
    if len(leg_ok) < len(LEG_COLS):
        print("  note: legs not clearly visible in this clip (leg graphs may be partial)")
    norm = normalize(df, ALL_COLS)
    norm["video"] = name
    norm["pct_of_shot"] = np.arange(N_POINTS)
    all_shots.append(norm)

if not all_shots:
    raise SystemExit("No videos were processed. Check the folder and model files.")

combined = pd.concat(all_shots)
combined.to_csv(os.path.join(OUT_DIR, "all_shots_normalized.csv"), index=False)

avg = combined.groupby("pct_of_shot")[ALL_COLS].mean()
std = combined.groupby("pct_of_shot")[ALL_COLS].std()
avg.to_csv(os.path.join(OUT_DIR, "average_shot.csv"))

ncols = 2
nrows = math.ceil(len(ALL_COLS) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3 * nrows), sharex=True)
axes = axes.flatten()
for ax, c in zip(axes, ALL_COLS):
    for _, g in combined.groupby("video"):
        ax.plot(g["pct_of_shot"], g[c], color="gray", alpha=0.25)
    ax.plot(avg.index, avg[c], color="C0", linewidth=2.5)
    ax.fill_between(avg.index, avg[c] - std[c], avg[c] + std[c], color="C0", alpha=0.2)
    ax.set_title(c, fontsize=10)
for ax in axes[len(ALL_COLS):]:
    ax.axis("off")
for ax in axes[-ncols:]:
    ax.set_xlabel("% of shot")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "average_shot.png"), dpi=150)
plt.show()
print("Done. Check the output_v3 folder.")