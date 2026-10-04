import json
import sqlite3
import uuid
from pathlib import Path


def _connect(database_path):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            analyzed_at TEXT NOT NULL,
            video_filename TEXT NOT NULL,
            reference_key TEXT NOT NULL,
            reference_name TEXT NOT NULL,
            phase_percent TEXT NOT NULL,
            overall_score REAL NOT NULL,
            feedback TEXT NOT NULL,
            unavailable_metrics TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metric_results (
            analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            metric_key TEXT NOT NULL,
            label TEXT NOT NULL,
            status TEXT NOT NULL,
            score REAL,
            shot_values TEXT NOT NULL,
            reference_mean TEXT NOT NULL,
            reference_std TEXT NOT NULL,
            largest_sustained_deviation REAL,
            largest_deviation_phase_percent REAL,
            PRIMARY KEY (analysis_id, metric_key)
        );
    """)
    connection.commit()
    return connection


def save_analysis(result, database_path):
    analysis_id = uuid.uuid4().hex
    connection = _connect(database_path)
    try:
        connection.execute(
            """INSERT INTO analyses (
                id, schema_version, analyzed_at, video_filename, reference_key,
                reference_name, phase_percent, overall_score, feedback, unavailable_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id,
                result["schema_version"],
                result["analyzed_at"],
                result["video"]["filename"],
                result["reference"]["key"],
                result["reference"]["name"],
                json.dumps(result["phase_percent"], allow_nan=False),
                result["overall_score"],
                json.dumps(result["feedback"], allow_nan=False),
                json.dumps(result["unavailable_metrics"], allow_nan=False),
            ),
        )
        connection.executemany(
            """INSERT INTO metric_results (
                analysis_id, metric_key, label, status, score, shot_values,
                reference_mean, reference_std, largest_sustained_deviation,
                largest_deviation_phase_percent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    analysis_id,
                    metric["key"],
                    metric["label"],
                    metric["status"],
                    metric["score"],
                    json.dumps(metric["shot_values"], allow_nan=False),
                    json.dumps(metric["reference_mean"], allow_nan=False),
                    json.dumps(metric["reference_std"], allow_nan=False),
                    metric.get("largest_sustained_deviation"),
                    metric.get("largest_deviation_phase_percent"),
                )
                for metric in result["metrics"]
            ],
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return analysis_id


def get_analysis(analysis_id, database_path):
    connection = _connect(database_path)
    try:
        analysis = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if analysis is None:
            return None

        metrics = connection.execute(
            "SELECT * FROM metric_results WHERE analysis_id = ? ORDER BY rowid",
            (analysis_id,),
        ).fetchall()
        return {
            "analysis_id": analysis["id"],
            "schema_version": analysis["schema_version"],
            "analyzed_at": analysis["analyzed_at"],
            "video": {"filename": analysis["video_filename"]},
            "reference": {
                "key": analysis["reference_key"],
                "name": analysis["reference_name"],
            },
            "phase_percent": json.loads(analysis["phase_percent"]),
            "overall_score": analysis["overall_score"],
            "metrics": [
                {
                    "key": metric["metric_key"],
                    "label": metric["label"],
                    "status": metric["status"],
                    "score": metric["score"],
                    "shot_values": json.loads(metric["shot_values"]),
                    "reference_mean": json.loads(metric["reference_mean"]),
                    "reference_std": json.loads(metric["reference_std"]),
                    **({"largest_sustained_deviation": metric["largest_sustained_deviation"]}
                       if metric["largest_sustained_deviation"] is not None else {}),
                    **({"largest_deviation_phase_percent": metric["largest_deviation_phase_percent"]}
                       if metric["largest_deviation_phase_percent"] is not None else {}),
                }
                for metric in metrics
            ],
            "feedback": json.loads(analysis["feedback"]),
            "unavailable_metrics": json.loads(analysis["unavailable_metrics"]),
        }
    finally:
        connection.close()