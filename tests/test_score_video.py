import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.analysis_store import get_analysis, save_analysis
from backend.local_feedback import LocalFeedbackError, _parse_response
from backend.score_video import _build_feedback, _json_values, _max_sustained_deviation, _opening_onset


class ScoreVideoFeedbackTests(unittest.TestCase):
    def test_opening_onset_uses_fifteen_percent_of_reference_change(self):
        self.assertEqual(_opening_onset([0, 1, 3, 6, 10, 14, 18]), 2)

    def test_sustained_deviation_suppresses_single_point_spike(self):
        sustained, phase = _max_sustained_deviation([0, 0, 2, 2, 2, 0, 0])
        self.assertAlmostEqual(sustained, 1.2)
        self.assertEqual(phase, 2)

    def test_feedback_is_ranked_and_limited_to_three(self):
        metrics = []
        for key, deviation in (
            ("shoot_elbow_angle", 2),
            ("guide_elbow_angle", 4),
            ("torso_lean", 3),
            ("hip_height", 1),
        ):
            reference = np.linspace(100, 180, 100).tolist() if key == "guide_elbow_angle" else [0.0] * 100
            shot = (np.interp(np.arange(100), [0, 5, 99], [100, 125, 180]).tolist()
                    if key == "guide_elbow_angle" else [1.0] * 100)
            metrics.append({
                "key": key,
                "label": key.replace("_", " ").title(),
                "score": 70.0,
                "reference_mean": reference,
                "shot_values": shot,
                "standardized_deviation": [deviation] * 100,
            })

        with patch("backend.score_video.generate_feedback") as generate:
            generate.side_effect = lambda findings: [
                {"metric": finding["metric"], "message": f"Generated coaching for {finding['metric']}"}
                for finding in findings
            ]
            feedback = _build_feedback(metrics)

        self.assertEqual([item["metric"] for item in feedback], [
            "guide_elbow_angle", "torso_lean", "shoot_elbow_angle",
        ])
        generated_findings = generate.call_args.args[0]
        guide_finding = next(item for item in generated_findings if item["metric"] == "guide_elbow_angle")
        self.assertEqual(guide_finding["opening_timing"]["direction"], "earlier")
        self.assertAlmostEqual(guide_finding["opening_timing"]["difference_percentage_points"], 12.1, places=1)
        self.assertEqual(guide_finding["label"], "Guide elbow opening timing")
        self.assertEqual(guide_finding["unit"], "percentage points")
        self.assertAlmostEqual(guide_finding["difference"], 12.1, places=1)
        self.assertNotIn("reference", str(generated_findings).lower())

    def test_large_shooting_elbow_difference_has_actionable_explanation(self):
        response = json.dumps({
            "feedback": [{
                "metric": "shoot_elbow_angle",
                "message": (
                    "Your shooting elbow angle is nearly 90 degrees higher than it should be "
                    "around 47% through the shot. Try keeping the elbow bent through the set "
                    "before extending; this can help keep the release path consistent."
                ),
            }],
        })
        finding = {
            "metric": "shoot_elbow_angle",
            "difference": 83.9,
            "direction": "higher",
            "unit": "degrees",
        }
        parsed = _parse_response(response, [finding])
        message = parsed[0]["message"]
        self.assertIn("nearly 90 degrees higher than it should be", message)
        self.assertIn("Try keeping the elbow bent", message)
        self.assertIn("help keep the release path consistent", message)

    def test_model_feedback_rejects_reference_comparisons(self):
        response = json.dumps({
            "feedback": [{
                "metric": "shoot_elbow_angle",
                "message": (
                    "Your elbow is 20 degrees higher than it should be. Try bending it more; "
                    "this can help, compared to the reference."
                ),
            }],
        })
        finding = {
            "metric": "shoot_elbow_angle",
            "difference": 20.0,
            "direction": "higher",
            "unit": "degrees",
        }
        with self.assertRaises(LocalFeedbackError):
            _parse_response(response, [finding])

    def test_model_feedback_rejects_wrong_deviation_amount(self):
        response = json.dumps({
            "feedback": [{
                "metric": "guide_elbow_angle",
                "message": (
                    "Your guide elbow is nearly 90 degrees higher than it should be. Keep it "
                    "supported until your shooting arm extends; this can help control the release."
                ),
            }],
        })
        finding = {
            "metric": "guide_elbow_angle",
            "difference": 37.0,
            "direction": "higher",
            "unit": "degrees",
        }
        with self.assertRaises(LocalFeedbackError):
            _parse_response(response, [finding])

    def test_model_feedback_accepts_natural_benefit_without_forced_modal(self):
        response = json.dumps({
            "feedback": [{
                "metric": "shoot_elbow_angle",
                "message": (
                    "Your shooting elbow angle is 20 degrees higher than it should be. "
                    "Keep it bent through the set. This helps maintain a repeatable release path."
                ),
            }],
        })
        finding = {
            "metric": "shoot_elbow_angle",
            "difference": 20.0,
            "direction": "higher",
            "unit": "degrees",
        }
        self.assertEqual(_parse_response(response, [finding])[0]["metric"], "shoot_elbow_angle")

    def test_model_feedback_accepts_exact_two_decimal_difference(self):
        response = json.dumps({
            "feedback": [{
                "metric": "shoot_thigh_angle",
                "message": (
                    "Your shooting-side thigh angle is 29.35 degrees lower than it should be. "
                    "Increase hip flexion slightly during the load. This may help keep your base balanced."
                ),
            }],
        })
        finding = {
            "metric": "shoot_thigh_angle",
            "difference": -29.35,
            "direction": "lower",
            "unit": "degrees",
        }
        self.assertEqual(_parse_response(response, [finding])[0]["metric"], "shoot_thigh_angle")

    def test_guide_wrist_feedback_accepts_maintain_adjustment(self):
        response = json.dumps({
            "feedback": [{
                "metric": "guide_wrist_angle",
                "message": (
                    "Your guide wrist angle is 8 degrees lower than it should be around 42% "
                    "through the shot. Maintain a relaxed, neutral wrist while supporting the ball. "
                    "This can help keep the shooting hand in control of the release."
                ),
            }],
        })
        finding = {
            "metric": "guide_wrist_angle",
            "difference": -8.0,
            "direction": "lower",
            "unit": "degrees",
        }
        self.assertEqual(_parse_response(response, [finding])[0]["metric"], "guide_wrist_angle")

    def test_model_feedback_rejects_unqualified_guarantee(self):
        response = json.dumps({
            "feedback": [{
                "metric": "shoot_elbow_angle",
                "message": (
                    "Your shooting elbow is 20 degrees higher than it should be. Keep it bent "
                    "through the set; this will ensure a perfect shot every time."
                ),
            }],
        })
        finding = {
            "metric": "shoot_elbow_angle",
            "difference": 20.0,
            "direction": "higher",
            "unit": "degrees",
        }
        with self.assertRaises(LocalFeedbackError):
            _parse_response(response, [finding])

    def test_model_feedback_allows_negated_guarantee_disclaimer(self):
        response = json.dumps({
            "feedback": [{
                "metric": "shoot_wrist_angle",
                "message": (
                    "Your shooting wrist angle is 10.6 degrees higher than it should be. "
                    "Keeping your wrist relaxed may help maintain alignment and support a smoother "
                    "release, but does not guarantee improved accuracy."
                ),
            }],
        })
        finding = {
            "metric": "shoot_wrist_angle",
            "difference": 10.6,
            "direction": "higher",
            "unit": "degrees",
        }
        self.assertEqual(_parse_response(response, [finding])[0]["metric"], "shoot_wrist_angle")

    def test_model_feedback_accepts_qualified_ensure(self):
        response = json.dumps({
            "feedback": [{
                "metric": "guide_elbow_angle",
                "message": (
                    "Your guide elbow starts opening 37.4 percentage points earlier than it should. "
                    "Keep the guide hand supported until your shooting arm extends. This helps to "
                    "ensure the shooting hand controls the release."
                ),
            }],
        })
        finding = {
            "metric": "guide_elbow_angle",
            "difference": 37.4,
            "direction": "earlier",
            "unit": "percentage points",
            "opening_timing": {
                "direction": "earlier",
                "difference_percentage_points": 37.4,
            },
        }
        self.assertEqual(_parse_response(response, [finding])[0]["metric"], "guide_elbow_angle")

    def test_early_guide_elbow_feedback_rejects_reversed_timing_advice(self):
        response = json.dumps({
            "feedback": [{
                "metric": "guide_elbow_angle",
                "message": (
                    "Your guide elbow starts opening 37 percentage points earlier than it should. Open it earlier "
                    "in the motion; this can help it support the ball better."
                ),
            }],
        })
        finding = {
            "metric": "guide_elbow_angle",
            "opening_timing": {
                "direction": "earlier",
                "difference_percentage_points": 37.0,
            },
        }
        with self.assertRaises(LocalFeedbackError):
            _parse_response(response, [finding])

    def test_early_guide_elbow_accepts_observed_timing_and_delay_cue(self):
        response = json.dumps({
            "feedback": [{
                "metric": "guide_elbow_angle",
                "message": (
                    "Your guide elbow starts opening 37.4 percentage points earlier than it should "
                    "around 31% through the shot. Keep it supported until your shooting arm begins "
                    "extending; this can help keep the release in sync."
                ),
            }],
        })
        finding = {
            "metric": "guide_elbow_angle",
            "difference": 37.4,
            "direction": "earlier",
            "unit": "percentage points",
            "opening_timing": {
                "direction": "earlier",
                "difference_percentage_points": 37.4,
            },
        }
        result = _parse_response(response, [finding])
        self.assertIn("37.4 percentage points earlier", result[0]["message"])

    def test_guide_timing_rejects_awkward_too_soon_than_wording(self):
        response = json.dumps({
            "feedback": [{
                "metric": "guide_elbow_angle",
                "message": (
                    "Your guide elbow opens too soon than it should, nearly 37 percentage points "
                    "earlier. Keep it supported until your shooting arm extends; this can help "
                    "keep the release in sync."
                ),
            }],
        })
        finding = {
            "metric": "guide_elbow_angle",
            "difference": 37.4,
            "direction": "earlier",
            "unit": "percentage points",
            "opening_timing": {
                "direction": "earlier",
                "difference_percentage_points": 37.4,
            },
        }
        with self.assertRaises(LocalFeedbackError):
            _parse_response(response, [finding])

    def test_json_values_converts_nan_to_null(self):
        self.assertEqual(_json_values([1.0, np.nan]), [1.0, None])

    def test_analysis_round_trips_through_metric_tables(self):
        result = {
            "schema_version": 1,
            "analyzed_at": "2026-10-04T12:00:00+00:00",
            "video": {"filename": "shot.mp4"},
            "reference": {"key": "zaid", "name": "Zaid"},
            "phase_percent": [0.0, 100.0],
            "overall_score": 81.5,
            "metrics": [{
                "key": "guide_elbow_angle",
                "label": "Guide elbow angle",
                "status": "scored",
                "score": 74.0,
                "shot_values": [90.0, 120.0],
                "reference_mean": [95.0, 125.0],
                "reference_std": [2.0, 3.0],
                "largest_sustained_deviation": 2.1,
                "largest_deviation_phase_percent": 100.0,
            }],
            "feedback": [],
            "unavailable_metrics": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "analysis.sqlite3"
            analysis_id = save_analysis(result, database_path)
            restored = get_analysis(analysis_id, database_path)

        self.assertEqual(restored["analysis_id"], analysis_id)
        self.assertEqual(restored["metrics"][0]["shot_values"], [90.0, 120.0])
        self.assertEqual(restored["metrics"][0]["reference_mean"], [95.0, 125.0])
        self.assertEqual(restored["overall_score"], result["overall_score"])


if __name__ == "__main__":
    unittest.main()