import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .analysis_store import get_analysis, save_analysis
from .local_feedback import LocalFeedbackError
from .score_video import analyze_video


BASE_DIR = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
DATABASE_PATH = Path(os.environ.get(
    "BASKETBALL_DB_PATH", BASE_DIR / "data" / "results" / "analysis_results.sqlite3"))
REFERENCE_PROFILES = {
    "zaid": {"name": "Zaid", "directory": BASE_DIR / "data" / "processed" / "v3"},
}

app = FastAPI(title="Basketball Shot Analysis", version="1.0.0")
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "BASKETBALL_CORS_ORIGINS",
        "null,http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    reference: str = Query(default="zaid"),
):
    profile = REFERENCE_PROFILES.get(reference.lower())
    if profile is None:
        raise HTTPException(status_code=400, detail=f"Unknown reference profile: {reference}")

    filename = Path(video.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(status_code=415, detail="Upload a .mp4 or .mov video")

    temporary_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = temporary_file.name
            while chunk := await video.read(CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 250 MB upload limit")
                temporary_file.write(chunk)

        result = await run_in_threadpool(
            analyze_video, temporary_path, profile["directory"], profile["name"])
        result["video"]["filename"] = filename
        result["analysis_id"] = save_analysis(result, DATABASE_PATH)
        return result
    except LocalFeedbackError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await video.close()
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


@app.get("/analyses/{analysis_id}")
def read_analysis(analysis_id: str):
    result = get_analysis(analysis_id, DATABASE_PATH)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result