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
./.venv/bin/python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:5500/`. The page posts the selected MP4/MOV as the `video` field to `/analyze`.

## Deploy the Website to Netlify

The repository includes `netlify.toml`, which publishes only `frontend/`; the backend, model files,
videos, and analysis datasets remain outside the public site. Connect the repository in Netlify and
use the default settings from that file (no build command). Add the `.tech` domain in Netlify's
Domain settings and follow the DNS instructions shown there.

The FastAPI service is separate from the static Netlify site. Deploy it to a host that can run the
backend and local model, then set `window.BBALLMOTIONS_API_URL` in `frontend/index.html` to that
public HTTPS API base URL. Configure the backend's `BASKETBALL_CORS_ORIGINS` with both the apex
domain and `www` origin if both are used. The default API URL in the page is only for the prior
Render deployment; local development selects `http://127.0.0.1:8000` automatically.

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