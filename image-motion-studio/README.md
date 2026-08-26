# Image Motion Studio

Turn a static image into a subtle cinematic motion clip using depth-based parallax — no AI video generation model required.

## How It Works

```
Static Image → Depth Estimation → Depth-Based Parallax → Camera Motion → Video Frames → MP4
```

1. **Upload** a static image (JPG, PNG, WEBP)
2. **Depth Anything V2 Small** estimates a monocular depth map
3. **Parallax warping** displaces pixels based on depth — foreground moves more than background
4. **Camera trajectory** with cubic easing simulates a real camera push-in with subtle handheld drift
5. **FFmpeg/H.264** encodes the frames into a smooth MP4

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React + TypeScript + Vite + Tailwind CSS v4 |
| **Backend** | Python 3.11 + FastAPI |
| **Depth Model** | Depth Anything V2 Small (HuggingFace) |
| **Computer Vision** | OpenCV + NumPy + PyTorch |
| **Video Encoding** | imageio-ffmpeg + H.264 |
| **GPU** | MPS (Apple Silicon) / CUDA / CPU fallback |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- FFmpeg (`brew install ffmpeg`)

### Install

```bash
# Backend
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt

# Frontend
cd ../frontend
npm install
```

### Run

```bash
# Terminal 1: Backend (from image-motion-studio/)
source backend/.venv/bin/activate
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend (from image-motion-studio/)
cd frontend
npm run dev
```

Open **http://localhost:5173**

Or use the convenience script:
```bash
./run.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend status + device info |
| POST | `/api/upload` | Upload image (multipart) |
| POST | `/api/depth` | Generate depth map |
| POST | `/api/generate` | Start motion generation job |
| GET | `/api/status/{job_id}` | Check job progress |
| GET | `/api/result/{job_id}` | Download generated MP4 |

## Project Structure

```
image-motion-studio/
├── frontend/          # React + Vite + TypeScript + Tailwind
│   └── src/
│       ├── App.tsx              # Main layout + state management
│       ├── api.ts               # API client
│       └── components/
│           ├── TopBar.tsx        # Title bar + status indicators
│           ├── ControlsSidebar.tsx  # All parameters + upload + generate
│           ├── PreviewPanel.tsx  # Image/video preview + player controls
│           ├── DepthSidebar.tsx  # Depth map visualization
│           └── ProgressOverlay.tsx  # Generation progress pipeline
│
├── backend/
│   ├── main.py        # FastAPI app + endpoints + job runner
│   ├── depth.py       # Depth Anything V2 model loader + inference
│   ├── parallax.py    # Depth-based displacement + image warping
│   └── renderer.py    # Camera trajectory + frame gen + MP4 encoding
│
├── uploads/           # Uploaded images
├── outputs/           # Generated MP4 files
├── temp/              # Depth maps + temp files
├── requirements.txt
├── run.sh
└── README.md
```

## Motion Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Duration | 1–5 sec | 2 sec | Clip length |
| FPS | 24/30/60 | 30 | Frame rate |
| Push-In | 0–10 | 2 | Camera zoom intensity |
| Horizontal Drift | -10 to +10 | 1 | Lateral camera movement |
| Vertical Drift | -10 to +10 | 0 | Vertical camera movement |
| Handheld | 0–10 | 1 | Organic camera shake |
| Depth Strength | 0–10 | 4 | Parallax depth effect |
| Foreground Separation | 0–10 | 5 | FG/BG motion contrast |
| Edge Fill | Mirror/Blur/Inpaint | Mirror | Exposed edge handling |

## Limitations

- **No inpainting of disoccluded regions**: When foreground objects move, the revealed areas behind them are filled with mirror/blur/inpaint, not AI-generated content
- **2D warping only**: The parallax effect is a 2D displacement based on estimated depth, not true 3D re-projection
- **Monocular depth estimation**: Depth maps are estimates and may have artifacts at depth discontinuities (e.g., hair, thin objects)
- **No temporal consistency**: Each frame is independently warped from the source image (this is actually a feature — no AI hallucination or flickering)
- **Large images**: Very high-resolution images (4K+) may be slow to process on CPU

## Architecture Notes

- The depth model is loaded once and cached in memory for fast subsequent calls
- Generation runs in a background thread pool so the API stays responsive
- Progress is reported via polling (500ms intervals) — no WebSocket needed for this scale
- The frontend proxies API calls through Vite's dev server to avoid CORS issues
