/**
 * renderer.js — Video renderer (Node.js port of renderer.py)
 * Orchestrates camera motion, parallax, bio-motion, atmospheric particles,
 * ambient light, film grain, rack focus, specular shimmer, motion blur,
 * camera shake, and MP4 encoding via ffmpeg.
 *
 * Image data is handled as flat RGBA Uint8ClampedArray buffers.
 * Depth maps are Float32Array [h*w] normalised 0..1.
 */

'use strict';

const path = require('path');
const os = require('os');
const fs = require('fs');
const { execSync, spawnSync } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const sharp = require('sharp');
const { createCanvas } = require('@napi-rs/canvas');

const {
  detectSubjectRegion,
  createDisplacementField,
  warpImage,
} = require('./parallax');

// ─────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────
function cubicEaseOut(t) { return 1.0 - (1.0 - t) ** 3; }
function clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function sin(x) { return Math.sin(x); }
function cos(x) { return Math.cos(x); }
function exp(x) { return Math.exp(x); }
function atan2(y, x) { return Math.atan2(y, x); }
function sqrt(x) { return Math.sqrt(x); }

// ─────────────────────────────────────────────
// Sharp helpers — load image to RGBA buffer
// ─────────────────────────────────────────────

async function loadImageRGBA(imagePath) {
  const { data, info } = await sharp(imagePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { pixels: new Uint8ClampedArray(data.buffer), w: info.width, h: info.height };
}

function rgbaToBuffer(pixels, w, h) {
  return Buffer.from(pixels.buffer);
}

// Convert RGBA Uint8ClampedArray to PNG Buffer via @napi-rs/canvas
async function pixelsToPng(pixels, w, h) {
  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(w, h);
  imageData.data.set(pixels);
  ctx.putImageData(imageData, 0, 0);
  return canvas.toBuffer('image/png');
}

// ─────────────────────────────────────────────
// Camera Trajectory
// ─────────────────────────────────────────────

function generateCameraTrajectory({ numFrames, pushIn, hDrift, vDrift, handheld, fps, cameraShake = 0.0 }) {
  const trajectory = [];
  const duration = numFrames / fps;
  const intensity = Math.max(cameraShake / 10.0, 0.0);

  for (let i = 0; i < numFrames; i++) {
    const timeSec = i / fps;

    let phaseT;
    if (timeSec < 0.5) {
      const settleT = timeSec / 0.5;
      phaseT = settleT * 0.1;
    } else {
      const mainT = (timeSec - 0.5) / Math.max(duration - 0.5, 0.1);
      phaseT = 0.1 + 0.9 * cubicEaseOut(clip(mainT, 0, 1));
    }

    const tShake = timeSec * (0.15 + intensity * 2.05);
    const nx = sin(tShake * 1.0) * 0.6 + sin(tShake * 1.7 + 1.3) * 0.4;
    const ny = sin(tShake * 1.3 + 0.7) * 0.6 + sin(tShake * 2.1 + 2.4) * 0.4;
    const nr = sin(tShake * 0.55 + 0.4);

    const txPct = nx * 3.0 * intensity;
    const tyPct = ny * 3.0 * intensity;
    const rollAngle = nr * (0.25 * Math.PI / 180.0) * intensity;

    const handheldScale = handheld * 0.3;
    const hhNoiseX = handheldScale * (
      sin(timeSec * 2.3) * 0.3 + sin(timeSec * 5.7) * 0.15 + sin(timeSec * 11.1) * 0.05
    );
    const hhNoiseY = handheldScale * (
      sin(timeSec * 1.9 + 0.7) * 0.3 + sin(timeSec * 4.3 + 1.2) * 0.15 + sin(timeSec * 9.7 + 2.1) * 0.05
    );

    const baseOverscan = 1.0 + (pushIn / 10.0) * 0.12;
    const zoomTime = timeSec * (0.06 + (pushIn / 10.0) * 0.49);
    const zoomProgress = 0.5 - 0.5 * cos(zoomTime);
    const maxZoomExtra = 0.04 + (pushIn / 10.0) * 0.51;
    const scale = baseOverscan * (1.0 + zoomProgress * maxZoomExtra * phaseT);

    trajectory.push({
      t: phaseT,
      timeSec,
      noiseX: hhNoiseX + txPct * 0.4,
      noiseY: hhNoiseY + tyPct * 0.4,
      txPct,
      tyPct,
      scale,
      rollAngle,
      frameIndex: i,
    });
  }
  return trajectory;
}

// ─────────────────────────────────────────────
// Aspect Ratio & Resolution Helpers
// ─────────────────────────────────────────────

function applyAspectRatioCrop(pixels, w, h, aspectRatio) {
  if (!aspectRatio || aspectRatio === 'original') return { pixels, w, h };

  const ratios = { '16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1.0 };
  const targetRatio = ratios[aspectRatio] || (w / h);
  const currentRatio = w / h;

  let newW = w, newH = h, offX = 0, offY = 0;
  if (currentRatio > targetRatio) {
    newW = Math.floor(h * targetRatio);
    offX = Math.floor((w - newW) / 2);
  } else {
    newH = Math.floor(w / targetRatio);
    offY = Math.floor((h - newH) / 2);
  }

  const out = new Uint8ClampedArray(newW * newH * 4);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const srcI = ((y + offY) * w + (x + offX)) * 4;
      const dstI = (y * newW + x) * 4;
      out[dstI] = pixels[srcI];
      out[dstI + 1] = pixels[srcI + 1];
      out[dstI + 2] = pixels[srcI + 2];
      out[dstI + 3] = 255;
    }
  }
  return { pixels: out, w: newW, h: newH };
}

function applyResolutionScale(pixels, depthMap, w, h, resolution) {
  if (!resolution || resolution === 'original') return { pixels, depthMap, w, h };

  const targetHeights = { '1080p': 1080, '720p': 720 };
  const targetH = targetHeights[resolution] || h;
  if (h === targetH) return { pixels, depthMap, w, h };

  const scale = targetH / h;
  const targetW = Math.floor(w * scale) % 2 === 0 ? Math.floor(w * scale) : Math.floor(w * scale) + 1;
  const newH = targetH % 2 === 0 ? targetH : targetH + 1;

  // Bilinear resize for image pixels
  const outPixels = bilinearResize(pixels, w, h, targetW, newH);
  const outDepth = bilinearResizeFloat(depthMap, w, h, targetW, newH);
  return { pixels: outPixels, depthMap: outDepth, w: targetW, h: newH };
}

function bilinearResize(src, sw, sh, dw, dh) {
  const out = new Uint8ClampedArray(dw * dh * 4);
  const xRatio = sw / dw;
  const yRatio = sh / dh;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const srcX = x * xRatio, srcY = y * yRatio;
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, sw - 1), y1 = Math.min(y0 + 1, sh - 1);
      const fx = srcX - x0, fy = srcY - y0;
      const di = (y * dw + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v = (1 - fx) * (1 - fy) * src[(y0 * sw + x0) * 4 + c]
                + fx       * (1 - fy) * src[(y0 * sw + x1) * 4 + c]
                + (1 - fx) * fy       * src[(y1 * sw + x0) * 4 + c]
                + fx       * fy       * src[(y1 * sw + x1) * 4 + c];
        out[di + c] = clip(v, 0, 255);
      }
      out[di + 3] = 255;
    }
  }
  return out;
}

function bilinearResizeFloat(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const xRatio = sw / dw, yRatio = sh / dh;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const srcX = x * xRatio, srcY = y * yRatio;
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, sw - 1), y1 = Math.min(y0 + 1, sh - 1);
      const fx = srcX - x0, fy = srcY - y0;
      out[y * dw + x] =
        (1 - fx) * (1 - fy) * src[y0 * sw + x0]
        + fx     * (1 - fy) * src[y0 * sw + x1]
        + (1 - fx) * fy     * src[y1 * sw + x0]
        + fx     * fy       * src[y1 * sw + x1];
    }
  }
  return out;
}

// ─────────────────────────────────────────────
// Background Parallax Layer (soft blurred)
// ─────────────────────────────────────────────

function renderParallaxBackground({ pixels, w, h, scale, txPct, tyPct, rotRad }) {
  const PARALLAX_MOTION_FACTOR = 0.35;
  const pxScale = (scale * PARALLAX_MOTION_FACTOR + (1.0 - PARALLAX_MOTION_FACTOR)) * 1.15;
  const totalScale = pxScale;

  // Gaussian blur (box blur approximation, 3 passes)
  let blurred = boxBlur(pixels, w, h, 11);
  blurred = adjustBrightnessSaturation(blurred, w, h, 0.55, 0.85);

  // Affine transform: scale + translate
  const pTx = (txPct * PARALLAX_MOTION_FACTOR / 100.0) * w;
  const pTy = (tyPct * PARALLAX_MOTION_FACTOR / 100.0) * h;
  const pRotDeg = (rotRad * 180.0 / Math.PI) * PARALLAX_MOTION_FACTOR;

  return affineTransform(blurred, w, h, totalScale, pTx, pTy, pRotDeg);
}

function affineTransform(src, w, h, scale, tx, ty, rotDeg) {
  const out = new Uint8ClampedArray(w * h * 4);
  const rad = (rotDeg * Math.PI) / 180.0;
  const cosR = cos(rad), sinR = sin(rad);
  const cx = w / 2, cy = h / 2;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // Inverse transform
      const dx = x - cx - tx, dy = y - cy - ty;
      const srcX = (cosR * dx + sinR * dy) / scale + cx;
      const srcY = (-sinR * dx + cosR * dy) / scale + cy;

      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = x0 + 1, y1 = y0 + 1;
      const fx = srcX - x0, fy = srcY - y0;

      const getC = (px, py, c) => {
        px = clip(mirrorCoord(px, w), 0, w - 1);
        py = clip(mirrorCoord(py, h), 0, h - 1);
        return src[(py * w + px) * 4 + c];
      };

      const di = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v = (1 - fx) * (1 - fy) * getC(x0, y0, c)
                + fx       * (1 - fy) * getC(x1, y0, c)
                + (1 - fx) * fy       * getC(x0, y1, c)
                + fx       * fy       * getC(x1, y1, c);
        out[di + c] = clip(v, 0, 255);
      }
      out[di + 3] = 255;
    }
  }
  return out;
}

function mirrorCoord(v, size) {
  v = v % (2 * size);
  if (v < 0) v += 2 * size;
  if (v >= size) v = 2 * size - 1 - v;
  return clip(v, 0, size - 1);
}

function boxBlur(src, w, h, radius) {
  let buf = new Float32Array(src);
  // Horizontal pass
  const tmp = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let rr = 0, gg = 0, bb = 0, count = 0;
      for (let k = -radius; k <= radius; k++) {
        const sx = clip(x + k, 0, w - 1);
        rr += buf[(y * w + sx) * 4];
        gg += buf[(y * w + sx) * 4 + 1];
        bb += buf[(y * w + sx) * 4 + 2];
        count++;
      }
      tmp[(y * w + x) * 4] = rr / count;
      tmp[(y * w + x) * 4 + 1] = gg / count;
      tmp[(y * w + x) * 4 + 2] = bb / count;
      tmp[(y * w + x) * 4 + 3] = 255;
    }
  }
  // Vertical pass
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let rr = 0, gg = 0, bb = 0, count = 0;
      for (let k = -radius; k <= radius; k++) {
        const sy = clip(y + k, 0, h - 1);
        rr += tmp[(sy * w + x) * 4];
        gg += tmp[(sy * w + x) * 4 + 1];
        bb += tmp[(sy * w + x) * 4 + 2];
        count++;
      }
      out[(y * w + x) * 4] = rr / count;
      out[(y * w + x) * 4 + 1] = gg / count;
      out[(y * w + x) * 4 + 2] = bb / count;
      out[(y * w + x) * 4 + 3] = 255;
    }
  }
  return out;
}

function adjustBrightnessSaturation(pixels, w, h, brightness, saturation) {
  const out = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < w * h; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    // Convert to HSL-like: desaturate
    const avg = (r + g + b) / 3;
    const nr = avg + (r - avg) * saturation;
    const ng = avg + (g - avg) * saturation;
    const nb = avg + (b - avg) * saturation;
    out[i * 4] = clip(nr * brightness, 0, 255);
    out[i * 4 + 1] = clip(ng * brightness, 0, 255);
    out[i * 4 + 2] = clip(nb * brightness, 0, 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

// ─────────────────────────────────────────────
// Dust Particle System
// ─────────────────────────────────────────────

class DustParticleSystem {
  constructor(numParticles, w, h) {
    this.w = w;
    this.h = h;
    this.particles = [];
    // Deterministic seed equivalent via simple LCG
    let seed = 42;
    const rand = () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; };

    for (let k = 0; k < numParticles; k++) {
      this.particles.push({
        x: 0.05 * w + rand() * 0.9 * w,
        y: 0.05 * h + rand() * 0.9 * h,
        z: 0.1 + rand() * 0.8,
        radius: 1.2 + rand() * 2.3,
        speedX: -15.0 + rand() * 35.0,
        speedY: -8.0 + rand() * 20.0,
        phase: rand() * 2 * Math.PI,
        brightness: 0.4 + rand() * 0.5,
      });
    }
  }

  render(pixels, w, h, timeSec, intensity) {
    if (intensity <= 0) return pixels;
    const out = new Uint8ClampedArray(pixels);
    const alphaScale = Math.min(intensity / 10.0, 1.0);

    for (const p of this.particles) {
      const curX = ((p.x + p.speedX * timeSec * p.z + sin(timeSec * 1.5 + p.phase) * 10 * p.z) % w + w) % w;
      const curY = ((p.y + p.speedY * timeSec * p.z + cos(timeSec * 1.2 + p.phase) * 8 * p.z) % h + h) % h;
      const size = p.radius * (0.8 + 1.8 * p.z);
      const rad = Math.max(Math.round(size), 1);
      const flicker = 0.85 + 0.15 * sin(timeSec * 3.5 + p.phase);
      const br = p.brightness * flicker * alphaScale;
      const cr = clip(200 * br, 0, 255);
      const cg = clip(230 * br, 0, 255);
      const cb = clip(255 * br, 0, 255);

      // Draw soft circle
      const ix = Math.round(curX), iy = Math.round(curY);
      for (let dy = -rad; dy <= rad; dy++) {
        for (let dx = -rad; dx <= rad; dx++) {
          const dist = sqrt(dx * dx + dy * dy);
          if (dist > rad) continue;
          const px = clip(ix + dx, 0, w - 1);
          const py = clip(iy + dy, 0, h - 1);
          const alpha = (1 - dist / rad) * 0.6;
          const idx = (py * w + px) * 4;
          out[idx] = clip(out[idx] + cr * alpha, 0, 255);
          out[idx + 1] = clip(out[idx + 1] + cg * alpha, 0, 255);
          out[idx + 2] = clip(out[idx + 2] + cb * alpha, 0, 255);
        }
      }
    }
    return out;
  }
}

// ─────────────────────────────────────────────
// Per-frame effects
// ─────────────────────────────────────────────

function applyAmbientLightPulse(pixels, w, h, timeSec, intensity) {
  if (intensity <= 0) return pixels;
  const pulse = sin(timeSec * 2.2) * 0.5 + sin(timeSec * 4.1) * 0.2;
  const scale = (intensity / 10.0) * pulse * 0.08;
  const out = new Uint8ClampedArray(pixels.length);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const grad = (1.0 - (y / h) * 0.4) * (0.6 + 0.4 * (x / w));
      out[i] = clip(pixels[i] * (1 + grad * scale * 1.3), 0, 255);     // R (warm)
      out[i + 1] = clip(pixels[i + 1] * (1 + grad * scale * 1.0), 0, 255); // G
      out[i + 2] = clip(pixels[i + 2] * (1 + grad * scale * 0.6), 0, 255); // B
      out[i + 3] = 255;
    }
  }
  return out;
}

function applySubsurfaceVascularPulse(pixels, w, h, depthMap, timeSec, intensity, subjectDepth) {
  if (intensity <= 0) return pixels;
  const bpmFreq = 66.0 / 60.0 * 2.0 * Math.PI;
  const pulse = sin(timeSec * bpmFreq) * 0.65 + sin(timeSec * bpmFreq * 2.0 - 0.5) * 0.35;
  const pulseAmp = (intensity / 10.0) * 0.012 * Math.max(pulse, 0);
  if (pulseAmp <= 0) return pixels;

  const out = new Uint8ClampedArray(pixels);
  const n = w * h;

  for (let i = 0; i < n; i++) {
    const d = depthMap[i];
    const subjMask = exp(-((d - subjectDepth) ** 2) / (2 * (0.20 ** 2)));
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];

    // Simple skin detection: high R, medium G, low B
    const skinScore = clip((r - 100) / 60, 0, 1) * clip((200 - b) / 80, 0, 1);
    const skinMask = skinScore * subjMask;

    if (skinMask > 0.01) {
      out[i * 4] = clip(r * (1.0 + skinMask * 1.2 * pulseAmp), 0, 255);
      out[i * 4 + 1] = clip(g * (1.0 + skinMask * 0.5 * pulseAmp), 0, 255);
      out[i * 4 + 2] = clip(b * (1.0 + skinMask * 0.2 * pulseAmp), 0, 255);
    }
  }
  return out;
}

function applySpecularShimmer(pixels, w, h, timeSec, intensity) {
  if (intensity <= 0) return pixels;
  const shimmer = sin(timeSec * 4.5) * 0.4 + sin(timeSec * 9.2) * 0.2;
  const flareAmp = (intensity / 10.0) * (0.8 + shimmer * 0.5);
  const out = new Uint8ClampedArray(pixels);
  const n = w * h;

  for (let i = 0; i < n; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    const brightness = (r + g + b) / 3;
    if (brightness > 225) {
      const boost = flareAmp * 0.4 * ((brightness - 225) / 30);
      out[i * 4] = clip(r + boost * 0.8 * 255, 0, 255);
      out[i * 4 + 1] = clip(g + boost * 0.95 * 255, 0, 255);
      out[i * 4 + 2] = clip(b + boost * 255, 0, 255);
    }
  }
  return out;
}

function applyFilmGrain(pixels, w, h, frameIndex, intensity) {
  if (intensity <= 0) return pixels;
  const grainScale = (intensity / 10.0) * 12.0;
  const out = new Uint8ClampedArray(pixels);
  const n = w * h;

  // Deterministic per-frame noise
  let seed = (frameIndex * 1337 + 42) >>> 0;
  const rand = () => {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return ((seed >>> 0) / 0xffffffff) * 2 - 1; // -1 to 1
  };

  for (let i = 0; i < n; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    const lum = (r + g + b) / (3 * 255);
    const midtones = 4.0 * lum * (1.0 - lum);
    const noise = rand() * grainScale * midtones;
    out[i * 4] = clip(r + noise, 0, 255);
    out[i * 4 + 1] = clip(g + noise, 0, 255);
    out[i * 4 + 2] = clip(b + noise, 0, 255);
  }
  return out;
}

function applyDynamicDofBokeh(pixels, w, h, depthMap, timeSec, intensity, subjectDepth) {
  if (intensity <= 0) return pixels;
  const focalDepth = subjectDepth + 0.03 * sin(timeSec * 0.8);

  // Create blur level 1 (kernel 9) and level 2 (kernel 21)
  const blur1 = boxBlurPixels(pixels, w, h, 4);
  const blur2 = boxBlurPixels(pixels, w, h, 10);
  const out = new Uint8ClampedArray(pixels.length);

  for (let i = 0; i < w * h; i++) {
    const coc = Math.abs(depthMap[i] - focalDepth) * (intensity / 10.0) * 1.8;
    const w1 = clip((coc - 0.15) / 0.25, 0, 1);
    const w2 = clip((coc - 0.40) / 0.30, 0, 1);
    for (let c = 0; c < 3; c++) {
      out[i * 4 + c] = clip(
        pixels[i * 4 + c] * (1 - w1) +
        blur1[i * 4 + c] * (w1 * (1 - w2)) +
        blur2[i * 4 + c] * w2,
        0, 255
      );
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

function boxBlurPixels(pixels, w, h, r) {
  return boxBlur(pixels, w, h, r);
}

function applyPostParallaxCameraMotion({ pixels, w, h, timeSec, duration, zoomOut, cameraShake, handheld }) {
  const t = clip(timeSec / Math.max(duration, 0.01), 0, 1);
  const easedT = cubicEaseOut(t);

  const zoomAmp = (zoomOut / 10.0) * 0.20;
  const zoomScale = 1.04 + zoomAmp * (1.0 - easedT);

  const shakeIntensity = cameraShake / 10.0;
  const tShake = timeSec * (0.15 + shakeIntensity * 2.05);
  const nx = sin(tShake * 1.0) * 0.6 + sin(tShake * 1.7 + 1.3) * 0.4;
  const ny = sin(tShake * 1.3 + 0.7) * 0.6 + sin(tShake * 2.1 + 2.4) * 0.4;
  const nr = sin(tShake * 0.55 + 0.4);

  const hhX = (handheld / 10.0) * (sin(timeSec * 2.1) * 0.4 + sin(timeSec * 4.3) * 0.15);
  const hhY = (handheld / 10.0) * (sin(timeSec * 1.7 + 0.5) * 0.4 + sin(timeSec * 3.8 + 1.1) * 0.15);

  const txPx = (nx * 3.0 * shakeIntensity + hhX) * (w / 100.0);
  const tyPx = (ny * 3.0 * shakeIntensity + hhY) * (h / 100.0);
  const rotDeg = (nr * 0.35 * shakeIntensity) * (180.0 / Math.PI);
  const totalScale = Math.max(zoomScale, 1.04);

  return affineTransform(pixels, w, h, totalScale, txPx, tyPx, rotDeg);
}

// ─────────────────────────────────────────────
// Main Render Pipeline
// ─────────────────────────────────────────────

async function renderFrames({
  pixels, depthMap, origW, origH,
  duration, fps,
  pushIn, hDrift, vDrift, handheld,
  depthStrength, foregroundSeparation, edgeFill,
  aspectRatio, resolution,
  zoomOut = 5.0,
  breathing = 10.0, watcherSway = 10.0, blink = false,
  dustParticles = 1.0, lightShift = 2.0, filmGrain = 3.0,
  microSaccades = 2.5, edgeFlutter = 1.0,
  rackFocus = 2.0, specularShimmer = 2.0,
  heartbeatPulse = 2.5, motionBlur = 1.0,
  cameraShake = 0.0,
  progressCallback = null,
}) {
  // Apply aspect ratio crop
  let { pixels: p, w, h } = applyAspectRatioCrop(pixels, origW, origH, aspectRatio);
  let dm = applyAspectRatioDepth(depthMap, origW, origH, w, h, aspectRatio);

  // Apply resolution scale
  ({ pixels: p, depthMap: dm, w, h } = applyResolutionScale(p, dm, w, h, resolution));

  const numFrames = Math.round(duration * fps);
  const subjectInfo = detectSubjectRegion(dm, h, w);
  const particles = new DustParticleSystem(45, w, h);

  const trajectory = generateCameraTrajectory({
    numFrames, pushIn, hDrift, vDrift, handheld, fps, cameraShake,
  });

  const frames = [];

  for (let fi = 0; fi < numFrames; fi++) {
    const cam = trajectory[fi];

    // 1. Background parallax layer
    const bgLayer = renderParallaxBackground({
      pixels: p, w, h,
      scale: cam.scale,
      txPct: cam.txPct,
      tyPct: cam.tyPct,
      rotRad: cam.rollAngle,
    });

    // 2. Displacement field
    const { dx, dy } = createDisplacementField({
      depthMap: dm, h, w,
      pushIn, hDrift: hDrift + cam.noiseX, vDrift: vDrift + cam.noiseY,
      depthStrength, foregroundSeparation,
      frameT: cam.t, rawTimeSec: cam.timeSec,
      breathing, watcherSway, blink, microSaccades, edgeFlutter,
      rollAngle: cam.rollAngle,
    });

    // 3. Warp image
    let frame = warpImage(p, dx, dy, h, w, edgeFill, bgLayer);

    // 4. Rack focus / bokeh
    if (rackFocus > 0)
      frame = applyDynamicDofBokeh(frame, w, h, dm, cam.timeSec, rackFocus, subjectInfo.subjectDepth);

    // 5. Specular shimmer
    if (specularShimmer > 0)
      frame = applySpecularShimmer(frame, w, h, cam.timeSec, specularShimmer);

    // 6. Heartbeat pulse
    if (heartbeatPulse > 0)
      frame = applySubsurfaceVascularPulse(frame, w, h, dm, cam.timeSec, heartbeatPulse, subjectInfo.subjectDepth);

    // 7. Ambient light pulse
    if (lightShift > 0)
      frame = applyAmbientLightPulse(frame, w, h, cam.timeSec, lightShift);

    // 8. Dust particles
    if (dustParticles > 0)
      frame = particles.render(frame, w, h, cam.timeSec, dustParticles);

    // 9. Film grain
    if (filmGrain > 0)
      frame = applyFilmGrain(frame, w, h, fi, filmGrain);

    // 10. Post-parallax camera motion (zoom + shake)
    frame = applyPostParallaxCameraMotion({
      pixels: frame, w, h,
      timeSec: cam.timeSec, duration, zoomOut, cameraShake, handheld,
    });

    frames.push({ data: frame, w, h });

    if (progressCallback) progressCallback(fi + 1, numFrames);
  }

  return frames;
}

// Depth map cropping helper
function applyAspectRatioDepth(depthMap, origW, origH, newW, newH, aspectRatio) {
  if (!aspectRatio || aspectRatio === 'original') return depthMap;

  const ratios = { '16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1.0 };
  const targetRatio = ratios[aspectRatio] || (origW / origH);
  const currentRatio = origW / origH;

  let offX = 0, offY = 0;
  if (currentRatio > targetRatio) {
    offX = Math.floor((origW - newW) / 2);
  } else {
    offY = Math.floor((origH - newH) / 2);
  }

  const out = new Float32Array(newW * newH);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      out[y * newW + x] = depthMap[(y + offY) * origW + (x + offX)];
    }
  }
  return out;
}

// ─────────────────────────────────────────────
// MP4 Encoding via ffmpeg
// ─────────────────────────────────────────────

async function encodeMp4(frames, outputPath, fps, tempDir) {
  if (!frames.length) throw new Error('No frames to encode');

  const { w, h } = frames[0];
  const frameDir = path.join(tempDir, `frames_${uuidv4().slice(0, 8)}`);
  fs.mkdirSync(frameDir, { recursive: true });

  try {
    // Save each frame as PNG
    for (let i = 0; i < frames.length; i++) {
      const framePath = path.join(frameDir, `frame_${String(i).padStart(5, '0')}.png`);
      const canvas = createCanvas(w, h);
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(w, h);
      imageData.data.set(frames[i].data);
      ctx.putImageData(imageData, 0, 0);
      const pngBuf = canvas.toBuffer('image/png');
      fs.writeFileSync(framePath, pngBuf);
    }

    // Find ffmpeg
    let ffmpegPath = 'ffmpeg';
    try {
      const { stdout } = require('child_process').spawnSync('which', ['ffmpeg'], { encoding: 'utf8' });
      if (stdout.trim()) ffmpegPath = stdout.trim();
    } catch {}

    // Encode with ffmpeg
    const result = spawnSync(ffmpegPath, [
      '-y',
      '-framerate', String(fps),
      '-i', path.join(frameDir, 'frame_%05d.png'),
      '-c:v', 'libx264',
      '-pix_fmt', 'yuv420p',
      '-crf', '17',
      '-preset', 'medium',
      outputPath,
    ], { encoding: 'utf8', timeout: 300000 });

    if (result.status !== 0) {
      throw new Error(`ffmpeg failed: ${result.stderr}`);
    }
  } finally {
    // Cleanup frame images
    try {
      fs.rmSync(frameDir, { recursive: true, force: true });
    } catch {}
  }

  return outputPath;
}

// ─────────────────────────────────────────────
// Main pipeline entry point
// ─────────────────────────────────────────────

async function runPipeline({
  imagePath, depthMap, outputDir, tempDir,
  duration = 2.0, fps = 30,
  pushIn = 0.0, hDrift = 0.0, vDrift = 0.0,
  handheld = 6.5, zoomOut = 5.0,
  depthStrength = 15.0, foregroundSeparation = 10.0,
  edgeFill = 'inpaint', aspectRatio = 'original', resolution = '1080p',
  breathing = 10.0, watcherSway = 10.0, blink = false,
  dustParticles = 1.0, lightShift = 2.0, filmGrain = 3.0,
  microSaccades = 2.5, edgeFlutter = 1.0,
  rackFocus = 2.0, specularShimmer = 2.0,
  heartbeatPulse = 2.5, motionBlur = 1.0,
  cameraShake = 0.0,
  progressCallback = null,
}) {
  if (progressCallback) progressCallback('Loading image', 0);

  const { pixels, w: origW, h: origH } = await loadImageRGBA(imagePath);

  if (progressCallback) progressCallback('Rendering frames', 0);

  const frames = await renderFrames({
    pixels, depthMap, origW, origH,
    duration, fps, pushIn, hDrift, vDrift, handheld,
    depthStrength, foregroundSeparation, edgeFill, aspectRatio, resolution,
    zoomOut, breathing, watcherSway, blink,
    dustParticles, lightShift, filmGrain,
    microSaccades, edgeFlutter, rackFocus, specularShimmer,
    heartbeatPulse, motionBlur, cameraShake,
    progressCallback: (cur, total) => {
      if (progressCallback) progressCallback('Rendering frames', Math.round((cur / total) * 100));
    },
  });

  if (progressCallback) progressCallback('Encoding MP4', 0);

  const outputFilename = `motion_${uuidv4().slice(0, 8)}.mp4`;
  const outputPath = path.join(outputDir, outputFilename);
  await encodeMp4(frames, outputPath, fps, tempDir);

  if (progressCallback) progressCallback('Complete', 100);

  return outputPath;
}

module.exports = { runPipeline };
