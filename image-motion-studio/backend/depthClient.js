/**
 * depthClient.js — Pure In-Process JavaScript Depth Estimation Interface
 * Directly uses depth.js (Transformers.js ONNX) without any HTTP service or Python backend.
 */

'use strict';

const depth = require('./depth');

/**
 * Estimate depth directly in JavaScript via ONNX Runtime.
 * @param {string} imagePath 
 * @param {string} outputPath 
 * @returns {Promise<{ depthMap: Float32Array, h: number, w: number, device: string }>}
 */
async function estimateDepth(imagePath, outputPath) {
  return await depth.estimateDepth(imagePath, outputPath);
}

/**
 * Health check for the in-process JS depth model.
 */
async function getDepthServiceHealth() {
  return depth.getModelInfo();
}

module.exports = {
  estimateDepth,
  getDepthServiceHealth,
};
