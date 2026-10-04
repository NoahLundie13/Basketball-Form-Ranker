import argparse
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from .local_feedback import generate_feedback

# ---- settings (keep these the same as track_v3.py) ----
ROOT_DIR = Path(__file__).resolve().parents[1]
REF_DIR = ROOT_DIR / "data" / "processed" / "v3"
POSE_MODEL = str(ROOT_DIR / "models" / "pose_landmarker_lite.task")
HAND_MODEL = str(ROOT_DIR / "models" / "hand_landmarker.task")
USE_HANDS = os.path.exists(HAND_MODEL)
SHOOTING_ARM = "r"
N_POINTS = 100
MIN_VIS = 0.5

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


METRIC_LABELS = {
    "shoot_elbow_angle": "Shooting elbow angle",
    "shoot_wrist_angle": "Shooting wrist angle",
    "shoot_shoulder_angle": "Shooting shoulder angle",
    "guide_elbow_angle": "Guide elbow angle",
    "guide_wrist_angle": "Guide wrist angle",
    "guide_shoulder_angle": "Guide shoulder angle",
    "hip_height": "Hip height",
    "torso_lean": "Torso lean",
    "nose_x_offset": "Head side-to-side position",
    "nose_y_offset": "Head vertical position",
    "shoot_knee_angle": "Shooting-side knee angle",
    "shoot_thigh_angle": "Shooting-side thigh angle",
    "guide_knee_angle": "Guide-side knee angle",
    "guide_thigh_angle": "Guide-side thigh angle",
    "finger_spread": "Finger spread",
    "index_extension": "Index finger extension",
}
METRIC_UNITS = {
    "hip_height": "torso lengths",
    "nose_x_offset": "torso lengths",
    "nose_y_offset": "torso lengths",
    "index_extension": "palm lengths",
}
METRIC_MEANINGS = {
    "shoot_elbow_angle": "Angle at the shooting elbow; a larger angle means the arm is straighter.",
    "guide_elbow_angle": "Angle at the guide elbow; a larger angle means the arm is straighter.",
    "shoot_wrist_angle": "Angle from the shooting forearm through the wrist toward the hand.",
    "guide_wrist_angle": "Angle from the guide forearm through the wrist toward the hand.",
    "shoot_shoulder_angle": "Elevation of the shooting upper arm relative to the torso.",
    "guide_shoulder_angle": "Elevation of the guide upper arm relative to the torso.",
    "hip_height": "Upward movement of the hips from the start of the shot.",
    "torso_lean": "Torso orientation relative to upright; the sign depends on camera orientation.",
    "nose_x_offset": "Head position side-to-side relative to the shoulders; the sign depends on camera orientation.",
    "nose_y_offset": "Head height relative to the shoulders.",
    "shoot_knee_angle": "Angle at the shooting-side knee; a larger angle means a straighter leg.",
    "guide_knee_angle": "Angle at the guide-side knee; a larger angle means a straighter leg.",
    "shoot_thigh_angle": "Shooting-side thigh angle from vertical; a larger angle means more sideways tilt.",
    "guide_thigh_angle": "Guide-side thigh angle from vertical; a larger angle means more sideways tilt.",
    "finger_spread": "Angle between the thumb and outer fingers; a larger angle means more spread.",
    "index_extension": "Index-finger extension relative to palm length; a larger value means more extension.",
}
METRIC_COACHING_CONTEXT = {
    "shoot_elbow_angle": "A higher value means the shooting elbow is straighter. If it opens early, keep a comfortable bend through the set, then extend through the release; do not lower the elbow in space or reduce shot force.",
    "guide_elbow_angle": "A higher value means the guide elbow is straighter. The guide hand should support the side of the ball without pushing; opening much earlier can indicate it is moving out of sync.",
    "shoot_wrist_angle": "A higher value means the shooting forearm and hand are more aligned. Cue a relaxed wrist and a consistent finger finish through the release.",
    "guide_wrist_angle": "A higher value means the guide forearm and hand are more aligned. Cue a relaxed guide wrist that steadies the ball without adding a push.",
    "shoot_shoulder_angle": "This angle describes the shooting upper arm relative to the torso. Avoid inferring shoulder height or pain from this measurement alone; cue a comfortable, repeatable set position.",
    "guide_shoulder_angle": "This angle describes the guide upper arm relative to the torso. Cue a relaxed guide side that supports rather than steers the ball.",
    "hip_height": "This measures upward hip movement from the start of the clip. Differences may reflect when the player begins rising; cue smooth leg drive coordinated with release.",
    "torso_lean": "The sign depends on camera orientation. Describe the measured change without saying left or right; cue balance over the base.",
    "nose_x_offset": "The sign depends on camera orientation. Describe a side-to-side head shift without saying left or right; cue a steady head and sightline.",
    "nose_y_offset": "This measures head height relative to the shoulders. Cue a steady, comfortable head position and avoid inferring gaze direction.",
    "shoot_knee_angle": "A higher value means the shooting-side knee is straighter. Cue comfortable bend and smooth extension, not a forced deep squat.",
    "guide_knee_angle": "A higher value means the guide-side knee is straighter. Cue balanced, comfortable leg drive through the release.",
    "shoot_thigh_angle": "Lower values mean the thigh is closer to vertical. If the value is low, suggest a little more hip and knee flexion during the load while keeping the knee aligned over the foot; do not tell the player to straighten the leg. This measure does not preserve which way the thigh tilts.",
    "guide_thigh_angle": "Lower values mean the thigh is closer to vertical. If the value is low, suggest a little more comfortable leg loading while keeping the knee aligned over the foot; do not tell the player to straighten the leg. This measure does not preserve which way the thigh tilts.",
    "finger_spread": "A higher value means the fingers are spread wider. Cue relaxed, comfortable contact rather than forcing a particular hand shape.",
    "index_extension": "A higher value means the index finger extends farther relative to the palm. Cue a relaxed, repeatable finger finish rather than forcing the finger straight.",
}
MIN_FEEDBACK_Z = 1.5
FEEDBACK_LIMIT = 3


def _json_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _max_sustained_deviation(z_values, window=5):
    deviations = np.abs(np.asarray(z_values, dtype=float))
    if len(deviations) < window:
        return float(np.nanmean(deviations)), int(np.nanargmax(deviations))
    means = np.convolve(deviations, np.ones(window) / window, mode="valid")
    start = int(np.nanargmax(means))
    return float(means[start]), start + window // 2


def _opening_onset(values):
    values = np.asarray(values, dtype=float)
    change = values[-1] - values[0]
    if change < 10:
        return None
    threshold = values[0] + change * 0.15
    matches = np.flatnonzero(values >= threshold)
    return int(matches[0]) if len(matches) else None


def _build_feedback(metric_results):
    candidates = []
    for metric in metric_results:
        if metric["score"] is None:
            continue
        peak, phase_index = _max_sustained_deviation(metric["standardized_deviation"])
        metric["largest_sustained_deviation"] = round(peak, 3)
        metric["largest_deviation_phase_percent"] = round(phase_index * 100 / (N_POINTS - 1), 1)
        if peak >= MIN_FEEDBACK_Z:
            candidates.append((peak, metric, phase_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    findings = []
    for peak, metric, phase_index in candidates[:FEEDBACK_LIMIT]:
        observed = metric["shot_values"][phase_index]
        expected = metric["reference_mean"][phase_index]
        difference = observed - expected
        phase_percent = phase_index * 100 / (N_POINTS - 1)
        finding = {
            "metric": metric["key"],
            "label": metric["label"],
            "meaning": METRIC_MEANINGS.get(metric["key"], "Measured body movement during the shot."),
            "coaching_context": METRIC_COACHING_CONTEXT.get(metric["key"], "Use a comfortable adjustment that matches the measured movement."),
            "severity": round(peak, 2),
            "phase_percent": round(phase_percent, 1),
            "observed_value": round(observed, 2),
            "desired_value": round(expected, 2),
            "difference": round(difference, 2),
            "direction": "higher" if difference > 0 else "lower",
            "unit": METRIC_UNITS.get(metric["key"], "degrees"),
        }
        if metric["key"] == "guide_elbow_angle":
            desired_onset = _opening_onset(metric["reference_mean"])
            observed_onset = _opening_onset(metric["shot_values"])
            if desired_onset is not None and observed_onset is not None:
                timing_delta = observed_onset - desired_onset
                if abs(timing_delta) >= 8:
                    finding["opening_timing"] = {
                        "observed_start_percent": round(observed_onset * 100 / (N_POINTS - 1), 1),
                        "desired_start_percent": round(desired_onset * 100 / (N_POINTS - 1), 1),
                        "difference_percentage_points": abs(timing_delta) * 100 / (N_POINTS - 1),
                        "direction": "later" if timing_delta > 0 else "earlier",
                    }
                    finding.update({
                        "label": "Guide elbow opening timing",
                        "meaning": "Shot phase when the guide elbow begins to open; an earlier start means it opens too soon.",
                        "coaching_context": "If the guide elbow opens too early, keep the guide hand supporting the ball until the shooting arm begins extending. If it opens too late, let the guide hand come away sooner without pushing.",
                        "observed_value": finding["opening_timing"]["observed_start_percent"],
                        "desired_value": finding["opening_timing"]["desired_start_percent"],
                        "difference": finding["opening_timing"]["difference_percentage_points"],
                        "direction": finding["opening_timing"]["direction"],
                        "unit": "percentage points",
                    })
        findings.append(finding)

    generated = generate_feedback(findings)
    generated_by_metric = {item["metric"]: item["message"] for item in generated}
    return [
        {
            "metric": finding["metric"],
            "severity": finding["severity"],
            "phase_percent": finding["phase_percent"],
            "message": generated_by_metric[finding["metric"]],
        }
        for finding in findings
    ]


def analyze_video(video_path, reference_dir=REF_DIR, reference_name="Zaid"):
    """Analyze a video and return JSON-compatible scores and graph series."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reference_dir = Path(reference_dir)
    average_path = reference_dir / "average_shot.csv"
    shots_path = reference_dir / "all_shots_normalized.csv"
    if not average_path.is_file() or not shots_path.is_file():
        raise FileNotFoundError(f"Reference data is incomplete: {reference_dir}")

    average = pd.read_csv(average_path, index_col="pct_of_shot")
    shots = pd.read_csv(shots_path)
    std = shots.groupby("pct_of_shot")[list(average.columns)].std()
    shot = normalize(process_video(str(video_path)), list(average.columns))

    metric_results = []
    metric_scores = []
    skipped = []
    for column in average.columns:
        reference_mean = average[column].to_numpy(dtype=float)
        reference_std = std[column].to_numpy(dtype=float)
        shot_values = shot[column].to_numpy(dtype=float)
        if (not np.isfinite(reference_mean).all() or not np.isfinite(reference_std).all()
                or not np.isfinite(shot_values).all()):
            skipped.append(column)
            metric_results.append({
                "key": column,
                "label": METRIC_LABELS.get(column, column.replace("_", " ").title()),
                "unit": METRIC_UNITS.get(column, "degrees"),
                "status": "unavailable",
                "score": None,
                "shot_values": _json_values(shot_values),
                "reference_mean": _json_values(reference_mean),
                "reference_std": _json_values(reference_std),
                "standardized_deviation": [],
            })
            continue

        sigma = np.maximum(reference_std, 0.25 * np.nanmean(reference_std) + 1e-6)
        z = (shot_values - reference_mean) / sigma
        score = float(np.mean(np.exp(-0.5 * (np.abs(z) / 2) ** 2)) * 100)
        metric_scores.append(score)
        metric_results.append({
            "key": column,
            "label": METRIC_LABELS.get(column, column.replace("_", " ").title()),
            "unit": METRIC_UNITS.get(column, "degrees"),
            "status": "scored",
            "score": round(score, 2),
            "shot_values": _json_values(shot_values),
            "reference_mean": _json_values(reference_mean),
            "reference_std": _json_values(reference_std),
            "standardized_deviation": _json_values(z),
        })

    if not metric_scores:
        raise ValueError("No metrics could be compared. Is the person visible in the video?")

    overall = float(np.mean(metric_scores))
    feedback = _build_feedback(metric_results)
    for metric in metric_results:
        metric.pop("standardized_deviation", None)

    return {
        "schema_version": 1,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "video": {"filename": video_path.name},
        "reference": {"key": reference_name.lower(), "name": reference_name},
        "phase_percent": _json_values(np.linspace(0, 100, N_POINTS)),
        "overall_score": round(overall, 2),
        "metrics": metric_results,
        "feedback": feedback,
        "unavailable_metrics": skipped,
    }


def save_plot(result, output_path):
    scored = [metric for metric in result["metrics"] if metric["status"] == "scored"]
    ncols = 2
    nrows = math.ceil(len(scored) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    x_values = result["phase_percent"]
    for axis, metric in zip(axes, scored):
        mean = np.asarray(metric["reference_mean"])
        spread = np.asarray(metric["reference_std"])
        axis.plot(x_values, mean, color="C0", linewidth=2,
                  label=f"{result['reference']['name']} average")
        axis.fill_between(x_values, mean - spread, mean + spread, color="C0", alpha=0.2)
        axis.plot(x_values, metric["shot_values"], color="red", linewidth=2, label="This shot")
        axis.set_title(f"{metric['label']} (score {metric['score']:.0f})", fontsize=10)
        axis.set_xlabel("% of shot")
    if scored:
        axes[0].legend()
    for axis in axes[len(scored):]:
        axis.axis("off")
    plt.suptitle(f"Overall similarity: {result['overall_score']:.0f}/100", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare a basketball shot to a reference profile.")
    parser.add_argument("video", nargs="?", default=ROOT_DIR / "data" / "input" / "test_videos" / "test1.mp4")
    parser.add_argument("--reference-dir", default=REF_DIR)
    parser.add_argument("--reference-name", default="Zaid")
    parser.add_argument("--json-out", default=ROOT_DIR / "data" / "results" / "score_result.json")
    parser.add_argument("--plot-out", default=ROOT_DIR / "data" / "results" / "score_result.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = analyze_video(args.video, args.reference_dir, args.reference_name)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    if not args.no_plot:
        save_plot(result, args.plot_out)

    print(f"\nFORM SIMILARITY TO {result['reference']['name'].upper()}: "
          f"{result['overall_score']:.0f} / 100\n")
    print("Per-metric scores (best to worst):")
    for metric in sorted(result["metrics"], key=lambda item: item["score"] or -1, reverse=True):
        if metric["score"] is not None:
            print(f"  {metric['key']:22s} {metric['score']:5.0f}")
    if result["unavailable_metrics"]:
        print("\nNot scored (no reliable data):", ", ".join(result["unavailable_metrics"]))
    print("\nFeedback:")
    for item in result["feedback"]:
        print(f"  - {item['message']}")
    print(f"\nSaved result data to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())