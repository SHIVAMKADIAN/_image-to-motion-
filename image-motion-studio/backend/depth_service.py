"""
Depth Estimation Micro-Service
Minimal Flask wrapper around depth.py so Node.js can call it via HTTP.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64
import numpy as np
from flask import Flask, request, jsonify

from depth import estimate_depth, get_model_info, get_device

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    info = get_model_info()
    return jsonify({
        "status": "ok",
        "device": info["device"],
        "model": info["model"],
        "model_loaded": info["loaded"],
    })


@app.route("/estimate", methods=["POST"])
def estimate():
    data = request.json
    if not data or "image_path" not in data or "output_path" not in data:
        return jsonify({"error": "Missing image_path or output_path"}), 400

    image_path = data["image_path"]
    output_path = data["output_path"]

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 404

    try:
        depth_map = estimate_depth(image_path, output_path)
        depth_bytes = depth_map.astype(np.float32).tobytes()
        depth_b64 = base64.b64encode(depth_bytes).decode("ascii")
        h, w = depth_map.shape[:2]
        return jsonify({
            "depth_map_b64": depth_b64,
            "shape": [h, w],
            "device": get_device().upper(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("DEPTH_SERVICE_PORT", "8001"))
    print(f"[DepthService] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=False)
