"""
FastAPI backend for Image Motion Studio.
Handles image upload, depth estimation, and living cinematic motion video generation.
"""

import asyncio
import os
import sys
import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add backend dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth import estimate_depth, get_model_info, get_device
from renderer import run_pipeline

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
FRONTEND_DIR = BASE_DIR / "frontend"

for d in [UPLOADS_DIR, OUTPUTS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- App ---
app = FastAPI(title="Image Motion Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")

# --- State ---
executor = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict] = {}
image_data: dict[str, dict] = {}
depth_maps: dict[str, np.ndarray] = {}


# --- Models ---
class DepthRequest(BaseModel):
    image_id: str


class GenerateRequest(BaseModel):
    image_id: str
    # Timing & Quality
    duration: float = 2.0
    fps: int = 30
    resolution: str = "1080p"
    aspect_ratio: str = "original"
    # Camera Motion
    push_in: float = 0.0
    horizontal_drift: float = 0.0
    vertical_drift: float = 0.0
    handheld: float = 6.5
    camera_shake: float = 2.0
    zoom_out: float = 5.0
    zoom_in: float = 5.0
    # Parallax / Depth
    depth_strength: float = 15.0
    foreground_separation: float = 10.0
    edge_fill: str = "inpaint"
    # Bio-Motion
    breathing: float = 10.0
    watcher_sway: float = 10.0
    blink: bool = False
    micro_saccades: float = 2.5
    edge_flutter: float = 1.0
    heartbeat_pulse: float = 2.5
    # Atmosphere & Optics
    dust_particles: float = 1.0
    light_shift: float = 2.0
    film_grain: float = 3.0
    rack_focus: float = 2.0
    specular_shimmer: float = 2.0
    motion_blur: float = 1.0


# --- Endpoints ---

@app.get("/")
async def root():
    """Serve frontend index.html."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Image Motion Studio API is running",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/style.css")
async def get_css():
    return FileResponse(str(FRONTEND_DIR / "style.css"))


@app.get("/app.js")
async def get_js():
    return FileResponse(str(FRONTEND_DIR / "app.js"))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    model_info = get_model_info()
    return {
        "status": "ok",
        "device": model_info["device"],
        "model": model_info["model"],
        "model_loaded": model_info["loaded"],
    }


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file."""
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {', '.join(allowed)}")

    image_id = uuid.uuid4().hex[:12]
    filename = f"{image_id}{ext}"
    filepath = UPLOADS_DIR / filename

    try:
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(500, f"Failed to save image: {str(e)}")

    img = cv2.imread(str(filepath))
    if img is None:
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read image. File may be corrupt.")

    h, w = img.shape[:2]
    file_size = len(content)

    image_data[image_id] = {
        "id": image_id,
        "filename": file.filename,
        "stored_filename": filename,
        "filepath": str(filepath),
        "width": w,
        "height": h,
        "file_size": file_size,
    }

    return {
        "id": image_id,
        "filename": file.filename,
        "width": w,
        "height": h,
        "file_size": file_size,
        "url": f"/uploads/{filename}",
    }


@app.post("/api/depth")
async def generate_depth(request: DepthRequest):
    """Generate depth map for an uploaded image."""
    image_id = request.image_id
    if not image_id or image_id not in image_data:
        raise HTTPException(404, "Image not found")

    info = image_data[image_id]
    depth_filename = f"{image_id}_depth.png"
    depth_path = str(TEMP_DIR / depth_filename)

    try:
        loop = asyncio.get_event_loop()
        depth_map = await loop.run_in_executor(
            executor, estimate_depth, info["filepath"], depth_path
        )
        depth_maps[image_id] = depth_map
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Depth estimation failed: {str(e)}")

    return {
        "image_id": image_id,
        "depth_url": f"/temp/{depth_filename}",
        "status": "complete",
        "model": "Depth Anything V2 Small",
        "device": get_device().upper(),
    }


@app.post("/api/generate")
async def generate_motion(request: GenerateRequest):
    """Start motion video generation as a background job."""
    if request.image_id not in image_data:
        raise HTTPException(404, "Image not found")
    if request.image_id not in depth_maps:
        raise HTTPException(400, "Depth map not generated. Call /api/depth first.")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "stage": "Preparing image",
        "result_path": None,
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        _run_generation_job,
        job_id,
        request,
    )

    return {"job_id": job_id}


def _run_generation_job(job_id: str, request: GenerateRequest):
    """Run the generation pipeline in a background thread."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["stage"] = "Estimating depth"
        jobs[job_id]["progress"] = 10

        info = image_data[request.image_id]
        depth_map = depth_maps[request.image_id]

        jobs[job_id]["stage"] = "Building depth map"
        jobs[job_id]["progress"] = 20

        def progress_cb(stage: str, pct: int):
            jobs[job_id]["stage"] = stage
            if stage == "Rendering frames":
                jobs[job_id]["progress"] = 30 + int(pct * 0.55)
            elif stage == "Encoding MP4":
                jobs[job_id]["progress"] = 90
            elif stage == "Complete":
                jobs[job_id]["progress"] = 100

        output_path = run_pipeline(
            image_path=info["filepath"],
            depth_map=depth_map,
            output_dir=str(OUTPUTS_DIR),
            temp_dir=str(TEMP_DIR),
            duration=request.duration,
            fps=request.fps,
            push_in=request.push_in,
            h_drift=request.horizontal_drift,
            v_drift=request.vertical_drift,
            handheld=request.handheld,
            zoom_out=getattr(request, "zoom_out", getattr(request, "zoom_in", 1.0)),
            depth_strength=request.depth_strength,
            foreground_separation=request.foreground_separation,
            edge_fill=request.edge_fill,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            breathing=request.breathing,
            watcher_sway=request.watcher_sway,
            blink=request.blink,
            micro_saccades=request.micro_saccades,
            edge_flutter=request.edge_flutter,
            heartbeat_pulse=request.heartbeat_pulse,
            dust_particles=request.dust_particles,
            light_shift=request.light_shift,
            film_grain=request.film_grain,
            rack_focus=request.rack_focus,
            specular_shimmer=request.specular_shimmer,
            motion_blur=request.motion_blur,
            camera_shake=request.camera_shake,
            progress_callback=progress_cb,
        )

        result_filename = os.path.basename(output_path)
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["stage"] = "Complete"
        jobs[job_id]["result_path"] = f"/outputs/{result_filename}"
        jobs[job_id]["result_filename"] = result_filename

    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["stage"] = "Failed"


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    """Check the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "result_path": job.get("result_path"),
        "error": job.get("error"),
    }


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Download the generated MP4 file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(400, "Job not complete yet")

    filename = job.get("result_filename")
    filepath = OUTPUTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Result file not found")

    return FileResponse(
        str(filepath),
        media_type="video/mp4",
        filename=filename,
    )


# --- Serve Frontend UI ---
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
