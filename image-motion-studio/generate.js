#!/usr/bin/env node
/**
 * Image Motion Studio — JavaScript CLI (generate.js)
 * Usage:
 *     node generate.js <image_path> [output_dir]
 * 
 * Output:
 *     <name>_2.0s.mp4 — living cinematic clip with full optical & biological dynamics
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { estimateDepth } = require('./backend-node/depth');
const { runPipeline } = require('./backend-node/renderer');

// ─────────────────────────────────────────────
// DEFAULT SETTINGS (Living Cinematic Engine)
// ─────────────────────────────────────────────
const DEFAULTS = {
  duration: 2.0,
  fps: 30,
  resolution: '1080p',
  aspectRatio: 'original',
  edgeFill: 'inpaint',

  // Camera Motion
  pushIn: 0.0,
  hDrift: 0.0,
  vDrift: 0.0,
  handheld: 6.5,
  zoomOut: 5.0,
  cameraShake: 0.0,

  // Parallax / Depth
  depthStrength: 15.0,
  foregroundSeparation: 10.0,

  // Bio-Motion & Physiological Dynamics
  breathing: 10.0,
  watcherSway: 10.0,
  blink: false,
  microSaccades: 2.5,
  edgeFlutter: 1.0,
  heartbeatPulse: 2.5,

  // Atmosphere & Optical Physics
  dustParticles: 1.0,
  lightShift: 2.0,
  filmGrain: 3.0,
  rackFocus: 2.0,
  specularShimmer: 2.0,
  motionBlur: 1.0,
};

function progressBar(label) {
  return (stage, pct) => {
    const barLen = 30;
    const filled = Math.floor((barLen * pct) / 100);
    const bar = '█'.repeat(filled) + '░'.repeat(barLen - filled);
    process.stdout.write(`\r  ${label} [${bar}] ${String(pct).padStart(3)}%  ${stage.padEnd(30)}`);
    if (pct >= 100) {
      process.stdout.write('\n');
    }
  };
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    console.log(`
Image Motion Studio CLI (JavaScript) — generates living cinematic clips

Usage:
  node generate.js <image_path> [output_dir]

Examples:
  node generate.js photo.jpg
  node generate.js photo.jpg ./outputs
`);
    process.exit(0);
  }

  const imagePath = path.resolve(args[0]);
  if (!fs.existsSync(imagePath)) {
    console.error(`❌ Image not found: ${imagePath}`);
    process.exit(1);
  }

  const outputDir = path.resolve(args[1] || 'outputs');
  const tempDir = path.join(outputDir, 'temp');
  fs.mkdirSync(outputDir, { recursive: true });
  fs.mkdirSync(tempDir, { recursive: true });

  const stem = path.basename(imagePath, path.extname(imagePath));

  console.log(`\n🎬 Image Motion Studio — Living Cinematic Engine (100% JavaScript)`);
  console.log(`   Image     : ${path.basename(imagePath)}`);
  console.log(`   Output dir: ${outputDir}`);
  console.log();

  // ── Step 1: Depth Estimation ──────────────────────────────────────────────
  console.log('1/2  Estimating depth in pure JavaScript (Transformers.js ONNX)...');
  const depthPath = path.join(tempDir, `${stem}_depth.png`);
  const { depthMap, w, h, device } = await estimateDepth(imagePath, depthPath);
  console.log(`     ✅ Depth map: ${w}x${h} (${device})`);

  // ── Step 2: Render clip ─────────────────────────────────────────
  console.log(`\n2/2  Rendering ${DEFAULTS.duration} sec video clip...`);
  const cb = progressBar('     Render');

  const outputFile = await runPipeline({
    imagePath,
    depthMap,
    outputDir,
    tempDir,
    ...DEFAULTS,
    progressCallback: cb,
  });

  const finalPath = path.join(outputDir, `${stem}_${DEFAULTS.duration}s.mp4`);
  if (fs.existsSync(outputFile) && outputFile !== finalPath) {
    fs.renameSync(outputFile, finalPath);
  }

  const stats = fs.statSync(finalPath);
  console.log(`     ✅ ${path.basename(finalPath)} (${Math.round(stats.size / 1024)} KB)`);

  console.log(`
╔══════════════════════════════════╗
║  ✅  Done!                       ║
╠══════════════════════════════════╣
║  📹  ${path.basename(finalPath)}
║  🗂   ${outputDir}
╚══════════════════════════════════╝
`);

  // Clean up temp
  try {
    fs.rmSync(tempDir, { recursive: true, force: true });
  } catch {}
}

main().catch(err => {
  console.error('\n❌ Fatal error:', err);
  process.exit(1);
});
