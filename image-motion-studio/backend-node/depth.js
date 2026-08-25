/**
 * depth.js — Pure JavaScript Depth Estimation using @huggingface/transformers
 * Runs Depth Anything V2 directly in Node.js via ONNX Runtime Web/Node.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

let _depthPipeline = null;
let _loadingPromise = null;

async function getDepthPipeline() {
  if (_depthPipeline) return _depthPipeline;
  if (_loadingPromise) return _loadingPromise;

  _loadingPromise = (async () => {
    console.log('[Depth JS] Loading Depth Anything V2 model in pure JavaScript...');
    const { pipeline, env } = await import('@huggingface/transformers');
    
    // Configure cache directory
    env.cacheDir = path.join(process.cwd(), '.cache');
    
    _depthPipeline = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
      device: 'auto',
    });
    console.log('[Depth JS] Model loaded successfully in JavaScript!');
    return _depthPipeline;
  })();

  return _loadingPromise;
}

/**
 * Estimate depth map from image using pure JavaScript.
 * @param {string} imagePath 
 * @param {string} outputPath 
 * @returns {Promise<{ depthMap: Float32Array, h: number, w: number, device: string }>}
 */
async function estimateDepth(imagePath, outputPath) {
  const pipe = await getDepthPipeline();
  
  // Get image original dimensions
  const metadata = await sharp(imagePath).metadata();
  const origW = metadata.width;
  const origH = metadata.height;

  // Run pipeline
  const result = await pipe(imagePath);
  
  // result.depth is a RawImage or image tensor with { data, width, height, channels }
  let depthRawData, depthW, depthH;
  if (result.depth && result.depth.data) {
    depthRawData = result.depth.data;
    depthW = result.depth.width;
    depthH = result.depth.height;
  } else if (result.data) {
    depthRawData = result.data;
    depthW = result.width;
    depthH = result.height;
  } else {
    throw new Error('Unexpected depth result format from transformers.js');
  }

  // Convert raw depth values to normalized Float32Array [0..1]
  const nPixels = depthW * depthH;
  const floatDepth = new Float32Array(nPixels);
  
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < nPixels; i++) {
    const val = depthRawData[i];
    if (val < minVal) minVal = val;
    if (val > maxVal) maxVal = val;
  }

  const range = maxVal - minVal > 1e-6 ? maxVal - minVal : 1.0;
  for (let i = 0; i < nPixels; i++) {
    floatDepth[i] = (depthRawData[i] - minVal) / range;
  }

  // Resize depth map to original image dimensions if needed
  let finalDepthMap = floatDepth;
  if (depthW !== origW || depthH !== origH) {
    finalDepthMap = bilinearResizeFloat(floatDepth, depthW, depthH, origW, origH);
  }

  // Save visualization PNG if outputPath requested
  if (outputPath) {
    const depthU8 = new Uint8ClampedArray(origW * origH);
    for (let i = 0; i < origW * origH; i++) {
      depthU8[i] = Math.round(finalDepthMap[i] * 255);
    }
    await sharp(Buffer.from(depthU8.buffer), {
      raw: { width: origW, height: origH, channels: 1 }
    }).png().toFile(outputPath);
  }

  return {
    depthMap: finalDepthMap,
    h: origH,
    w: origW,
    device: 'onnx-js',
  };
}

function bilinearResizeFloat(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const xRatio = sw / dw;
  const yRatio = sh / dh;
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

function getModelInfo() {
  return {
    loaded: _depthPipeline !== null,
    device: 'onnx-js',
    model: 'Depth Anything V2 Small (Transformers.js ONNX)',
  };
}

module.exports = {
  estimateDepth,
  getModelInfo,
};
