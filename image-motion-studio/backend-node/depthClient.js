/**
 * depthClient.js — Calls the Python depth micro-service (depth_service.py)
 * and returns a Float32Array depth map that Node.js can use directly.
 */

'use strict';

const axios = require('axios');

const DEPTH_SERVICE_URL = process.env.DEPTH_SERVICE_URL || 'http://127.0.0.1:8001';

/**
 * Estimate depth for an image via the Python depth service.
 * @param {string} imagePath - Absolute path to input image
 * @param {string} outputPath - Where to save the depth PNG visualisation
 * @returns {{ depthMap: Float32Array, h: number, w: number, device: string }}
 */
async function estimateDepth(imagePath, outputPath) {
  const response = await axios.post(
    `${DEPTH_SERVICE_URL}/estimate`,
    { image_path: imagePath, output_path: outputPath },
    { timeout: 120_000 }
  );

  const { depth_map_b64, shape, device } = response.data;
  const [h, w] = shape;

  // Decode base64 → Buffer → Float32Array
  const buf = Buffer.from(depth_map_b64, 'base64');
  const depthMap = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

  return { depthMap: new Float32Array(depthMap), h, w, device };
}

/**
 * Health check for the depth service.
 * @returns {{ status, device, model, modelLoaded }}
 */
async function getDepthServiceHealth() {
  const res = await axios.get(`${DEPTH_SERVICE_URL}/health`, { timeout: 5000 });
  return res.data;
}

module.exports = { estimateDepth, getDepthServiceHealth };
