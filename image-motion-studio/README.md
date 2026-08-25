# Image Motion Studio (100% JavaScript)

Turn a static image into a subtle cinematic motion clip using depth-based parallax and biological motion dynamics — built entirely in **JavaScript (Node.js)**.

## How It Works

```
Static Image → Depth Estimation (ONNX) → Parallax Displacements → Living Bio-Motion → Frame Rendering → MP4
```

1. **Upload** a static image (JPG, PNG, WEBP).
2. **Depth Anything V2** estimates monocular depth map natively in JavaScript via `@huggingface/transformers` ONNX.
3. **Adaptive Parallax Engine** displaces pixels using depth-aware transfer curves and multi-layer reflection blending.
4. **Living Physiological Dynamics** adds natural breathing deformation, ocular micro-saccades, fabric flutter, and 3D dust particles.
5. **Optical Physics Pipeline** applies rack focus bokeh, specular catchlight shimmer, ambient light breathing, and 35mm film grain.
6. **FFmpeg / H.264** encodes the rendered frames into a high quality MP4 video.

---

## Tech Stack (100% JavaScript)

| Layer | Technology |
|---|---|
| **Runtime** | Node.js (v18+) |
| **Backend Framework** | Express.js |
| **Depth Estimation** | `@huggingface/transformers` (Depth Anything V2 ONNX) |
| **Image & Pixel Manipulation** | `sharp`, `@napi-rs/canvas`, typed `Float32Array` |
| **Video Encoding** | `ffmpeg` CLI (H.264 / libx264) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript with live HUD & canvas preview loop |

---

## Quick Start

### Prerequisites

- **Node.js** (v18 or higher)
- **FFmpeg** (`brew install ffmpeg` on macOS or `apt install ffmpeg` on Linux)

### Installation

```bash
cd image-motion-studio
npm install
```

---

## Running the Application

### 1. Web Studio (UI + Backend)

Start the full web application:

```bash
npm start
# Or
./run.sh
```

Then open **[http://localhost:8000](http://localhost:8000)** in your web browser.

### 2. JavaScript CLI

Generate cinematic motion clips directly from the command line:

```bash
node generate.js <path_to_image> [output_directory]
```

Example:
```bash
node generate.js sample.jpg ./outputs
```

---

## API Endpoints

- `GET  /api/health` — Check server and ONNX model status
- `POST /api/upload` — Upload image file (`multipart/form-data`)
- `POST /api/depth` — Compute monocular depth map (`{ "image_id": "..." }`)
- `POST /api/generate` — Start cinematic motion video generation
- `GET  /api/status/:jobId` — Check progress of video generation job
- `GET  /api/result/:jobId` — Download the completed MP4 video

---

## Project Structure

```
image-motion-studio/
├── backend/
│   ├── depth.js           # Pure JavaScript Depth Anything V2 (ONNX)
│   ├── parallax.js        # Depth-based parallax & bio-motion displacement fields
│   ├── renderer.js        # Optical physics frame renderer & MP4 encoder
│   └── server.js          # Express.js REST API server & static host
├── frontend/
│   ├── index.html         # Studio interface & control panel
│   ├── style.css          # Glassmorphism dark UI styling
│   ├── app.js             # Client controller & real-time animation preview
│   └── src/presets.js     # Default preset configuration
├── generate.js            # JavaScript CLI tool
├── package.json           # Project dependencies and npm scripts
└── run.sh                 # One-click startup script
```
