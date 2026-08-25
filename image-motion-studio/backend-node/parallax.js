/**
 * parallax.js — Depth-based parallax warping and organic living motion fields.
 * Direct port of parallax.py using Float32Array typed arrays instead of NumPy.
 *
 * All functions:
 * - computeAdaptiveDepthWeight
 * - detectSubjectRegion
 * - createDisplacementField
 * - computeBreathingField
 * - computeMicroSaccadesField
 * - computeEdgeFlutterField
 * - computeForegroundSwayField
 * - computeEyeMicroBlinkField
 * - warpImage
 */

'use strict';

// ─────────────────────────────────────────────
// Utility math helpers
// ─────────────────────────────────────────────

function exp(x) { return Math.exp(x); }
function sin(x) { return Math.sin(x); }
function cos(x) { return Math.cos(x); }
function clip(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function f32zeros(size) { return new Float32Array(size); }

/**
 * computeAdaptiveDepthWeight
 * Universal non-linear depth transfer function.
 * @param {Float32Array} depthMap - flat float32 array [h*w]
 * @param {number} foregroundSeparation
 * @param {number} h
 * @param {number} w
 * @returns {Float32Array}
 */
function computeAdaptiveDepthWeight(depthMap, foregroundSeparation, h, w) {
  const n = h * w;
  const fgFactor = foregroundSeparation / 10.0;
  const k = 4.5 + fgFactor * 2.0;
  const x0 = 0.40 - fgFactor * 0.10;

  // sigmoid min value
  const sigMin = 1.0 / (1.0 + exp(k * x0));

  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = depthMap[i];
    const sig = 1.0 / (1.0 + exp(-k * (d - x0)));
    const sigNorm = (sig - sigMin) / (1.0 - sigMin + 1e-6);

    // Linear blend
    let dw = 0.45 * d + 0.55 * sigNorm;

    // Soft foreground saturation
    const fgMask = clip((d - 0.82) / 0.18, 0.0, 1.0);
    const clampedW = 0.85 + 0.15 * Math.tanh((dw - 0.85) * 3.0);
    dw = dw * (1.0 - fgMask) + clampedW * fgMask;

    result[i] = clip(dw, 0.0, 1.0);
  }
  return result;
}

/**
 * detectSubjectRegion
 * @param {Float32Array} depthMap
 * @param {number} h
 * @param {number} w
 * @returns {{ cx, cy, subjectDepth, sigmaX, sigmaY }}
 */
function detectSubjectRegion(depthMap, h, w) {
  let totalWeight = 0;
  let sumX = 0, sumY = 0;
  let hasSubject = false;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = depthMap[y * w + x];
      if (d >= 0.20 && d <= 0.80) {
        hasSubject = true;
        totalWeight += d;
        sumX += x * d;
        sumY += y * d;
      }
    }
  }

  if (!hasSubject || totalWeight === 0) {
    // Fallback: general depth-weighted centre
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const d = depthMap[y * w + x];
        totalWeight += d;
        sumX += x * d;
        sumY += y * d;
      }
    }
  }

  let cx, cy, subjectDepth;
  if (totalWeight > 0) {
    cx = sumX / totalWeight;
    cy = sumY / totalWeight;
    const ix = clip(Math.round(cx), 0, w - 1);
    const iy = clip(Math.round(cy), 0, h - 1);
    subjectDepth = depthMap[iy * w + ix];
  } else {
    cx = w * 0.5;
    cy = h * 0.5;
    subjectDepth = 0.5;
  }

  return {
    cx,
    cy,
    subjectDepth,
    sigmaX: Math.max(0.18 * w, 40),
    sigmaY: Math.max(0.22 * h, 40),
  };
}

/**
 * createDisplacementField
 * Create combined X and Y displacement fields.
 * @returns {{ dx: Float32Array, dy: Float32Array }}
 */
function createDisplacementField({
  depthMap, h, w,
  pushIn, hDrift, vDrift,
  depthStrength, foregroundSeparation,
  frameT, rawTimeSec = 0,
  breathing = 3.0, watcherSway = 3.0,
  blink = true, microSaccades = 2.0, edgeFlutter = 2.0,
  rollAngle = 0.0,
  image = null,
}) {
  const n = h * w;
  const subjectInfo = detectSubjectRegion(depthMap, h, w);
  const depthWeight = computeAdaptiveDepthWeight(depthMap, foregroundSeparation, h, w);

  const pushScale = pushIn * 10.0;
  const hDriftScale = hDrift * 8.0;
  const vDriftScale = vDrift * 8.0;
  const depthScale = depthStrength / 10.0;

  const cx = w / 2.0;
  const cy = h / 2.0;

  const totalDx = new Float32Array(n);
  const totalDy = new Float32Array(n);

  // --- 1. Camera displacement field ---
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const xRel = (x - cx) / cx;
      const yRel = (y - cy) / cy;
      const dw = depthWeight[i];

      const pushDx = -xRel * pushScale * dw * depthScale * frameT;
      const pushDy = -yRel * pushScale * dw * depthScale * frameT;

      const driftDepthFactor = 0.35 + 0.65 * dw * depthScale;
      const driftDx = hDriftScale * driftDepthFactor * frameT;
      const driftDy = vDriftScale * driftDepthFactor * frameT;

      totalDx[i] = pushDx + driftDx;
      totalDy[i] = pushDy + driftDy;

      // 1b. Camera rotational shake / roll
      if (rollAngle !== 0.0) {
        const cosR = cos(rollAngle);
        const sinR = sin(rollAngle);
        totalDx[i] += (xRel * (cosR - 1.0) - yRel * sinR) * cx;
        totalDy[i] += (xRel * sinR + yRel * (cosR - 1.0)) * cy;
      }
    }
  }

  // --- 2. Breathing field ---
  if (breathing > 0) {
    const { dx: bDx, dy: bDy } = computeBreathingField({
      depthMap, h, w, timeSec: rawTimeSec, intensity: breathing, subjectInfo,
    });
    for (let i = 0; i < n; i++) { totalDx[i] += bDx[i]; totalDy[i] += bDy[i]; }
  }

  // --- 3. Micro-saccades ---
  if (microSaccades > 0) {
    const { dx: sDx, dy: sDy } = computeMicroSaccadesField({
      depthMap, h, w, timeSec: rawTimeSec, intensity: microSaccades, subjectInfo,
    });
    for (let i = 0; i < n; i++) { totalDx[i] += sDx[i]; totalDy[i] += sDy[i]; }
  }

  // --- 4. Edge flutter ---
  if (edgeFlutter > 0) {
    const { dx: efDx, dy: efDy } = computeEdgeFlutterField({
      depthMap, h, w, timeSec: rawTimeSec, intensity: edgeFlutter,
    });
    for (let i = 0; i < n; i++) { totalDx[i] += efDx[i]; totalDy[i] += efDy[i]; }
  }

  // --- 5. Foreground sway ---
  if (watcherSway > 0) {
    const { dx: wDx, dy: wDy } = computeForegroundSwayField({
      depthMap, h, w, timeSec: rawTimeSec, intensity: watcherSway,
    });
    for (let i = 0; i < n; i++) { totalDx[i] += wDx[i]; totalDy[i] += wDy[i]; }
  }

  // --- 6. Eye micro-blink ---
  if (blink) {
    const { dx: eDx, dy: eDy } = computeEyeMicroBlinkField({
      depthMap, h, w, timeSec: rawTimeSec, subjectInfo,
    });
    for (let i = 0; i < n; i++) { totalDx[i] += eDx[i]; totalDy[i] += eDy[i]; }
  }

  return { dx: totalDx, dy: totalDy };
}

function computeBreathingField({ depthMap, h, w, timeSec, intensity, subjectInfo }) {
  const n = h * w;
  const cyclePeriod = 3.5;
  const tPhase = (timeSec % cyclePeriod) / cyclePeriod;
  let breathCycle;

  if (tPhase < 0.38) {
    const u = tPhase / 0.38;
    breathCycle = 0.5 * (1.0 - cos(u * Math.PI));
  } else if (tPhase < 0.48) {
    breathCycle = 1.0 - 0.03 * sin((tPhase - 0.38) / 0.10 * Math.PI);
  } else {
    const u = (tPhase - 0.48) / 0.52;
    breathCycle = 0.5 * (1.0 + cos(u * Math.PI));
  }

  const bAmp = (intensity / 10.0) * 3.6 * (breathCycle - 0.5);
  const { cx: chestX, sigmaX, sigmaY, subjectDepth: sDepth } = subjectInfo;
  const chestY = Math.min(subjectInfo.cy + 0.08 * h, 0.90 * h);

  const dx = new Float32Array(n);
  const dy = new Float32Array(n);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const d = depthMap[i];
      const subjMask = exp(-((d - sDepth) ** 2) / (2 * (0.22 ** 2)));
      const torsoW = exp(
        -(((x - chestX) ** 2) / (2 * sigmaX ** 2) + ((y - chestY) ** 2) / (2 * sigmaY ** 2))
      );
      const combined = subjMask * torsoW;
      const dxRel = (x - chestX) / Math.max(sigmaX, 1.0);
      const dyRel = (y - chestY) / Math.max(sigmaY, 1.0);
      dx[i] = dxRel * bAmp * combined * 0.8;
      dy[i] = (dyRel * bAmp - Math.abs(bAmp) * 0.4) * combined;
    }
  }
  return { dx, dy };
}

function computeMicroSaccadesField({ depthMap, h, w, timeSec, intensity, subjectInfo }) {
  const n = h * w;
  const scale = (intensity / 10.0) * 0.85;

  const saccadeX = (
    sin(timeSec * 6.3) * 0.4 +
    sin(timeSec * 17.1 + 0.3) * 0.25 +
    sin(timeSec * 29.7 + 1.1) * 0.15
  ) * scale;
  const saccadeY = (
    cos(timeSec * 5.7 + 0.8) * 0.35 +
    sin(timeSec * 15.3 + 0.5) * 0.2 +
    cos(timeSec * 31.2 + 2.0) * 0.1
  ) * scale;

  const faceX = subjectInfo.cx;
  const faceY = Math.max(subjectInfo.cy - 0.10 * h, 0.15 * h);
  const sigFaceX = Math.max(subjectInfo.sigmaX * 0.5, 25.0);
  const sigFaceY = Math.max(subjectInfo.sigmaY * 0.4, 25.0);
  const sDepth = subjectInfo.subjectDepth;

  const dx = new Float32Array(n);
  const dy = new Float32Array(n);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const faceW = exp(
        -(((x - faceX) ** 2) / (2 * sigFaceX ** 2) + ((y - faceY) ** 2) / (2 * sigFaceY ** 2))
      );
      const subjDepth = exp(-((depthMap[i] - sDepth) ** 2) / (2 * (0.18 ** 2)));
      const weight = faceW * subjDepth;
      dx[i] = weight * saccadeX;
      dy[i] = weight * saccadeY;
    }
  }
  return { dx, dy };
}

function computeEdgeFlutterField({ depthMap, h, w, timeSec, intensity }) {
  const n = h * w;
  const scale = (intensity / 10.0) * 1.5;
  const dx = new Float32Array(n);
  const dy = new Float32Array(n);

  if (scale <= 0) return { dx, dy };

  // Simple edge approximation via gradient (no OpenCV needed)
  const edges = new Float32Array(n);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = depthMap[i + 1] - depthMap[i - 1];
      const gy = depthMap[i + w] - depthMap[i - w];
      edges[i] = clip(Math.sqrt(gx * gx + gy * gy) * 5.0, 0, 1);
    }
  }

  const freqY = 0.02, freqX = 0.02;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const waveX = sin(y * freqY + timeSec * 3.2) * cos(x * freqX + timeSec * 2.1);
      const waveY = cos(y * freqY * 1.2 + timeSec * 2.7) * sin(x * freqX * 0.9 + timeSec * 1.8);
      const depthFactor = clip(depthMap[i], 0.2, 0.9);
      dx[i] = edges[i] * waveX * scale * depthFactor;
      dy[i] = edges[i] * waveY * (scale * 0.6) * depthFactor;
    }
  }
  return { dx, dy };
}

function computeForegroundSwayField({ depthMap, h, w, timeSec, intensity }) {
  const n = h * w;
  const swayAmp = (intensity / 10.0) * 3.5;
  const swayX = sin(timeSec * 1.8) * 0.7 + sin(timeSec * 3.4) * 0.3;
  const swayY = cos(timeSec * 1.5) * 0.5 + sin(timeSec * 2.7) * 0.2;

  const dx = new Float32Array(n);
  const dy = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const fgMask = clip((depthMap[i] - 0.78) / 0.15, 0.0, 1.0);
    dx[i] = fgMask * (swayX * swayAmp);
    dy[i] = fgMask * (swayY * swayAmp * 0.6);
  }
  return { dx, dy };
}

function computeEyeMicroBlinkField({ depthMap, h, w, timeSec, subjectInfo }) {
  const n = h * w;
  const dx = new Float32Array(n);
  const dy = new Float32Array(n);

  const blinkStart = 1.0, blinkDuration = 0.22;
  if (timeSec < blinkStart || timeSec > blinkStart + blinkDuration) return { dx, dy };

  const relT = (timeSec - blinkStart) / blinkDuration;
  const blinkWeight = sin(relT * Math.PI) ** 2;

  const eyeX = subjectInfo.cx;
  const eyeY = Math.max(subjectInfo.cy - 0.10 * h, 0.15 * h);
  const sigX = Math.max(subjectInfo.sigmaX * 0.25, 15.0);
  const sigY = Math.max(subjectInfo.sigmaY * 0.15, 10.0);
  const sDepth = subjectInfo.subjectDepth;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const subjDepth = exp(-((depthMap[i] - sDepth) ** 2) / (2 * (0.18 ** 2)));
      const eyeW = exp(
        -(((x - eyeX) ** 2) / (2 * sigX ** 2) + ((y - eyeY) ** 2) / (2 * sigY ** 2))
      ) * subjDepth;
      dy[i] = eyeW * (blinkWeight * 2.2);
    }
  }
  return { dx, dy };
}

/**
 * warpImage - Remap pixels using displacement fields.
 * @param {Uint8ClampedArray} imageData - RGBA flat pixel array
 * @param {Float32Array} dx
 * @param {Float32Array} dy
 * @param {number} h
 * @param {number} w
 * @param {string} edgeFill - 'mirror' | 'blur' | 'inpaint'
 * @param {Uint8ClampedArray|null} bgLayer - RGBA
 * @returns {Uint8ClampedArray}
 */
function warpImage(imageData, dx, dy, h, w, edgeFill = 'mirror', bgLayer = null) {
  const out = new Uint8ClampedArray(h * w * 4);
  const oobMask = new Uint8Array(h * w);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let srcX = x + dx[i];
      let srcY = y + dy[i];
      let isOob = false;

      if (edgeFill === 'mirror') {
        // Mirror / reflect
        srcX = mirrorCoord(srcX, w);
        srcY = mirrorCoord(srcY, h);
      } else {
        if (srcX < 0 || srcX >= w || srcY < 0 || srcY >= h) {
          isOob = true;
          oobMask[i] = 1;
          srcX = clip(srcX, 0, w - 1);
          srcY = clip(srcY, 0, h - 1);
        }
      }

      // Bilinear interpolation
      const x0 = Math.floor(srcX), y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
      const fx = srcX - x0, fy = srcY - y0;

      const getPixel = (px, py, ch) => imageData[(py * w + px) * 4 + ch];

      for (let c = 0; c < 3; c++) {
        const v = (1 - fx) * (1 - fy) * getPixel(x0, y0, c)
                + fx       * (1 - fy) * getPixel(x1, y0, c)
                + (1 - fx) * fy       * getPixel(x0, y1, c)
                + fx       * fy       * getPixel(x1, y1, c);
        out[i * 4 + c] = clip(v, 0, 255);
      }
      out[i * 4 + 3] = 255;
    }
  }

  // Blend OOB pixels with bgLayer if provided
  if (bgLayer) {
    for (let i = 0; i < h * w; i++) {
      if (oobMask[i]) {
        // Simple soft blend at boundary
        for (let c = 0; c < 3; c++) {
          out[i * 4 + c] = bgLayer[i * 4 + c];
        }
      }
    }
  }

  return out;
}

function mirrorCoord(v, size) {
  v = v % (2 * size);
  if (v < 0) v += 2 * size;
  if (v >= size) v = 2 * size - 1 - v;
  return v;
}

module.exports = {
  computeAdaptiveDepthWeight,
  detectSubjectRegion,
  createDisplacementField,
  warpImage,
};
