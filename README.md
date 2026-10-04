# Basketball Shot Analysis

## Project Layout

- `backend/`: API, shot scoring, persistence, and local feedback generation.
- `scripts/`: reference-video processing and API test client.
- `tests/`: unit tests.
- `models/`: MediaPipe model assets.
- `data/input/`: test clips and reference videos.
- `data/processed/`: versioned tracking outputs and reference curves.
- `data/results/`: saved analysis JSON, plots, and SQLite database.

## Run the API

From the repository root:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m uvicorn backend.app:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive docs are at `/docs`.

## Open the Website

With the API running, serve the landing page from a second terminal:

```bash
./.venv/bin/python -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500/`. The page posts the selected MP4/MOV as the `video` field to `/analyze`.
For a deployed site, set `window.BBALLMOTIONS_API_URL` to the API base URL before the page script,
and set the backend's `BASKETBALL_CORS_ORIGINS` to the website origin(s).

Coaching feedback is generated locally with MLX on Apple Silicon. The first analysis downloads
the default 4-bit Hugging Face model (about 2.3 GB); later runs use the local cache. Set
`SHOT_FEEDBACK_MODEL` to select another MLX-compatible Hugging Face model.

## Test an Upload

With the API running in another terminal:

```bash
./.venv/bin/python scripts/test_api.py
```

## Reprocess Reference Videos

Run the desired tracker from the repository root:

```bash
./.venv/bin/python scripts/track_v3.py
```

This reads `data/input/reference_videos/zaid/` and writes `data/processed/v3/`.

## Run Tests

```bash
./.venv/bin/python -m unittest discover -s tests
```