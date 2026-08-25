/**
 * server.js — Image Motion Studio Node.js Backend
 * Express.js server replacing Python FastAPI main.py
 *
 * Routes:
 *   GET  /api/health
 *   POST /api/upload
 *   POST /api/depth
 *   POST /api/generate
 *   GET  /api/status/:jobId
 *   GET  /api/result/:jobId
 *   GET  /              → serves frontend index.html
 */

'use strict';

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const depth = require('./depth');
const { runPipeline } = require('./renderer');

// ─────────────────────────────────────────────
// Paths
// ─────────────────────────────────────────────
const BASE_DIR = path.resolve(__dirname, '..');
const UPLOADS_DIR = path.join(BASE_DIR, 'uploads');
const OUTPUTS_DIR = path.join(BASE_DIR, 'outputs');
const TEMP_DIR = path.join(BASE_DIR, 'temp');
const FRONTEND_DIR = path.join(BASE_DIR, 'frontend');

[UPLOADS_DIR, OUTPUTS_DIR, TEMP_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// ─────────────────────────────────────────────
// State (in-memory)
// ─────────────────────────────────────────────
const imageData = new Map();   // imageId → metadata
const depthMaps = new Map();   // imageId → { depthMap: Float32Array, h, w }
const jobs = new Map();        // jobId → { status, progress, stage, resultPath, error }

// ─────────────────────────────────────────────
// App
// ─────────────────────────────────────────────
const app = express();
app.use(express.json());

// CORS — restrict to the server's own origin; loosen via CORS_ORIGIN env var
app.use((req, res, next) => {
  const allowedOrigin = process.env.CORS_ORIGIN || `http://localhost:${process.env.PORT || 8000}`;
  res.header('Access-Control-Allow-Origin', allowedOrigin);
  res.header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

// Static file serving
app.use('/uploads', express.static(UPLOADS_DIR));
app.use('/outputs', express.static(OUTPUTS_DIR));
app.use('/temp', express.static(TEMP_DIR));

// ─────────────────────────────────────────────
// Multer upload config
// ─────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const imageId = uuidv4().replace(/-/g, '').slice(0, 12);
    const ext = path.extname(file.originalname).toLowerCase();
    req.imageId = imageId;
    cb(null, `${imageId}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
  fileFilter: (req, file, cb) => {
    const allowedExts = ['.jpg', '.jpeg', '.png', '.webp'];
    const allowedMimes = ['image/jpeg', 'image/png', 'image/webp'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (!allowedExts.includes(ext) || !allowedMimes.includes(file.mimetype)) {
      return cb(new Error(`Unsupported format: ${ext} (${file.mimetype})`));
    }
    cb(null, true);
  },
});

// ─────────────────────────────────────────────
// Routes
// ─────────────────────────────────────────────

/** GET /api/health */
app.get('/api/health', async (req, res) => {
  const info = depth.getModelInfo();
  return res.json({
    status: 'ok',
    device: info.device || 'onnx-js',
    model: info.model || 'Depth Anything V2 Small (JS)',
    model_loaded: info.loaded,
    runtime: 'node.js',
  });
});

/** POST /api/upload */
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ detail: 'No file uploaded' });

    const imageId = req.imageId || path.basename(req.file.filename, path.extname(req.file.filename));
    const filePath = req.file.path;
    const filename = req.file.filename;

    // Get image dimensions via sharp
    const sharp = require('sharp');
    const meta = await sharp(filePath).metadata();
    const { width: w, height: h } = meta;
    const fileSize = req.file.size;

    imageData.set(imageId, {
      id: imageId,
      filename: req.file.originalname,
      storedFilename: filename,
      filepath: filePath,
      width: w,
      height: h,
      fileSize,
    });

    return res.json({
      id: imageId,
      filename: req.file.originalname,
      width: w,
      height: h,
      file_size: fileSize,
      url: `/uploads/${filename}`,
    });
  } catch (err) {
    console.error('[Upload Error]', err);
    return res.status(500).json({ detail: `Upload failed: ${err.message}` });
  }
});

/** POST /api/depth */
app.post('/api/depth', async (req, res) => {
  const { image_id: imageId } = req.body;
  if (!imageId || !imageData.has(imageId)) {
    return res.status(404).json({ detail: 'Image not found' });
  }

  const info = imageData.get(imageId);
  const depthFilename = `${imageId}_depth.png`;
  const depthPath = path.join(TEMP_DIR, depthFilename);

  try {
    const { depthMap, h, w, device } = await depth.estimateDepth(info.filepath, depthPath);
    depthMaps.set(imageId, { depthMap, h, w });

    return res.json({
      image_id: imageId,
      depth_url: `/temp/${depthFilename}`,
      status: 'complete',
      model: 'Depth Anything V2 Small',
      device: (device || 'ONNX-JS').toUpperCase(),
    });
  } catch (err) {
    console.error('[Depth Error]', err);
    return res.status(500).json({ detail: `Depth estimation failed: ${err.message}` });
  }
});

/** POST /api/generate */
app.post('/api/generate', (req, res) => {
  const body = req.body;
  const imageId = body.image_id;

  if (!imageData.has(imageId)) {
    return res.status(404).json({ detail: 'Image not found' });
  }
  if (!depthMaps.has(imageId)) {
    return res.status(400).json({ detail: 'Depth map not generated. Call /api/depth first.' });
  }

  const jobId = uuidv4().replace(/-/g, '').slice(0, 12);
  jobs.set(jobId, {
    status: 'queued',
    progress: 0,
    stage: 'Preparing image',
    resultPath: null,
    resultFilename: null,
    error: null,
  });

  // Run generation in background (same thread, async)
  setImmediate(() => runGenerationJob(jobId, imageId, body));

  return res.json({ job_id: jobId });
});

/** GET /api/status/:jobId */
app.get('/api/status/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ detail: 'Job not found' });

  return res.json({
    job_id: req.params.jobId,
    status: job.status,
    progress: job.progress,
    stage: job.stage,
    result_path: job.resultPath,
    error: job.error,
  });
});

/** GET /api/result/:jobId */
app.get('/api/result/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) return res.status(404).json({ detail: 'Job not found' });
  if (job.status !== 'complete') return res.status(400).json({ detail: 'Job not complete yet' });

  const filePath = path.join(OUTPUTS_DIR, job.resultFilename);
  if (!fs.existsSync(filePath)) return res.status(404).json({ detail: 'Result file not found' });

  res.setHeader('Content-Type', 'video/mp4');
  res.setHeader('Content-Disposition', `attachment; filename="${job.resultFilename}"`);
  fs.createReadStream(filePath).pipe(res);
});

// ─────────────────────────────────────────────
// Background generation job
// ─────────────────────────────────────────────

async function runGenerationJob(jobId, imageId, params) {
  const job = jobs.get(jobId);
  job.status = 'running';
  job.stage = 'Building depth map';
  job.progress = 20;

  try {
    const info = imageData.get(imageId);
    const { depthMap } = depthMaps.get(imageId);

    const outputPath = await runPipeline({
      imagePath: info.filepath,
      depthMap,
      outputDir: OUTPUTS_DIR,
      tempDir: TEMP_DIR,
      duration: params.duration ?? 2.0,
      fps: parseInt(params.fps ?? 30),
      pushIn: params.push_in ?? 0.0,
      hDrift: params.horizontal_drift ?? 0.0,
      vDrift: params.vertical_drift ?? 0.0,
      handheld: params.handheld ?? 6.5,
      zoomOut: params.zoom_out ?? 5.0,
      depthStrength: params.depth_strength ?? 15.0,
      foregroundSeparation: params.foreground_separation ?? 10.0,
      edgeFill: params.edge_fill ?? 'inpaint',
      aspectRatio: params.aspect_ratio ?? 'original',
      resolution: params.resolution ?? '1080p',
      breathing: params.breathing ?? 10.0,
      watcherSway: params.watcher_sway ?? 10.0,
      blink: params.blink ?? false,
      microSaccades: params.micro_saccades ?? 2.5,
      edgeFlutter: params.edge_flutter ?? 1.0,
      rackFocus: params.rack_focus ?? 2.0,
      specularShimmer: params.specular_shimmer ?? 2.0,
      heartbeatPulse: params.heartbeat_pulse ?? 2.5,
      motionBlur: params.motion_blur ?? 1.0,
      cameraShake: params.camera_shake ?? 0.0,
      dustParticles: params.dust_particles ?? 1.0,
      lightShift: params.light_shift ?? 2.0,
      filmGrain: params.film_grain ?? 3.0,
      progressCallback: (stage, pct) => {
        job.stage = stage;
        if (stage === 'Rendering frames') {
          job.progress = 30 + Math.round(pct * 0.55);
        } else if (stage === 'Encoding MP4') {
          job.progress = 90;
        } else if (stage === 'Complete') {
          job.progress = 100;
        }
      },
    });

    const resultFilename = path.basename(outputPath);
    job.status = 'complete';
    job.progress = 100;
    job.stage = 'Complete';
    job.resultPath = `/outputs/${resultFilename}`;
    job.resultFilename = resultFilename;

  } catch (err) {
    console.error('[Generation Error]', err);
    job.status = 'error';
    job.error = err.message;
    job.stage = 'Failed';
  }
}

// ─────────────────────────────────────────────
// Frontend static serving
// ─────────────────────────────────────────────
if (fs.existsSync(FRONTEND_DIR)) {
  app.use(express.static(FRONTEND_DIR));
  app.get('/', (req, res) => {
    res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
  });
}

// ─────────────────────────────────────────────
// Start
// ─────────────────────────────────────────────
const PORT = process.env.PORT || 8000;

if (require.main === module) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[ImageMotionStudio] 100% JavaScript Node.js server running on http://0.0.0.0:${PORT}`);
    console.log(`[ImageMotionStudio] Frontend: ${FRONTEND_DIR}`);
  });
}

module.exports = app;
