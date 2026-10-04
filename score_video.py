import os, sys, math
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

# ---- settings (keep these the same as track_v3.py) ----
REF_DIR = "output_v3"
POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
USE_HANDS = os.path.exists(HAND_MODEL)
SHOOTING_ARM = "r"
N_POINTS = 100
MIN_VIS = 0.5

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "testvideos/test1.mp4"

SIDES = {
    "l": dict(shoulder=11, elbow=13, wrist=15, index=19, hip=23, knee=25, ankle=27),
    "r": dict(shoulder=12, elbow=14, wrist=16, index=20, hip=24, knee=26, ankle=28),
}
GUIDE_ARM = "l" if SHOOTING_ARM == "r" else "r"
ARMS = {"shoot": SIDES[SHOOTING_ARM], "guide": SIDES[GUIDE_ARM]}


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
            row = {}
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
                    if vis(ix["hip"]) > MIN_VIS and vis(ix["knee"]) > MIN_VIS:
                        v = P(ix["knee"]) - P(ix["hip"])
                        row[f"{arm}_thigh_angle"] = math.degrees(math.atan2(abs(v[0]), v[1]))
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


# ---- load Shai's reference ----
avg = pd.read_csv(os.path.join(REF_DIR, "average_shot.csv"), index_col="pct_of_shot")
shots = pd.read_csv(os.path.join(REF_DIR, "all_shots_normalized.csv"))
std = shots.groupby("pct_of_shot")[list(avg.columns)].std()

# ---- process the new video ----
if not os.path.exists(VIDEO):
    raise SystemExit(f"Video not found: {VIDEO}")
print("Scoring", VIDEO)
new = normalize(process_video(VIDEO), list(avg.columns))

# ---- compare ----
scores = {}
for c in avg.columns:
    ref_ok = avg[c].notna().all() and std[c].notna().all()
    if not ref_ok or new[c].isna().any():
        continue  # skip metrics missing in Shai's data or in the new clip
    sigma = np.maximum(std[c].values, 0.25 * np.nanmean(std[c].values) + 1e-6)
    z = np.abs(new[c].values - avg[c].values) / sigma
    scores[c] = float(np.mean(np.exp(-0.5 * (z / 2) ** 2)) * 100)

if not scores:
    raise SystemExit("No metrics could be compared. Is the person visible in the video?")

overall = float(np.mean(list(scores.values())))
print(f"\nFORM SIMILARITY TO SHAI: {overall:.0f} / 100\n")
print("Per-metric scores (best to worst):")
for c, s in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {c:22s} {s:5.0f}")
skipped = [c for c in avg.columns if c not in scores]
if skipped:
    print("\nNot scored (no reliable data):", ", ".join(skipped))

# ---- plot ----
cols = list(scores)
ncols = 2
nrows = math.ceil(len(cols) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3 * nrows), sharex=True)
axes = np.array(axes).flatten()
for ax, c in zip(axes, cols):
    ax.plot(avg.index, avg[c], color="C0", linewidth=2, label="Shai average")
    ax.fill_between(avg.index, avg[c] - std[c], avg[c] + std[c], color="C0", alpha=0.2)
    ax.plot(range(N_POINTS), new[c], color="red", linewidth=2, label="This shot")
    ax.set_title(f"{c} (score {scores[c]:.0f})", fontsize=10)
axes[0].legend()
for ax in axes[len(cols):]:
    ax.axis("off")
plt.suptitle(f"Overall similarity: {overall:.0f}/100", fontsize=14)
plt.tight_layout()
plt.savefig("score_result.png", dpi=150)
plt.show()