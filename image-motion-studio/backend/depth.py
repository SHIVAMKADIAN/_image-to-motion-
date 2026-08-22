"""
Depth estimation using Depth Anything V2 Small.
Loads model once and caches in memory.
Uses MPS (Apple Silicon) when available, falls back to CPU.
"""

import cv2
import numpy as np
import torch
from PIL import Image

# Global singleton
_model = None
_processor = None
_device = None
_device_torch = None


def get_device() -> str:
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    """Load Depth Anything V2 Small model. Called once, cached globally."""
    global _model, _processor, _device, _device_torch

    if _model is not None:
        return

    _device = get_device()
    print(f"[Depth] Loading Depth Anything V2 Small on {_device}...")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = AutoModelForDepthEstimation.from_pretrained(model_name)

    _device_torch = torch.device(_device)
    _model = _model.to(_device_torch)
    _model.eval()

    print(f"[Depth] Model loaded successfully on {_device}")


def estimate_depth(image_path: str, output_path: str) -> np.ndarray:
    """
    Estimate depth from a single image.

    Args:
        image_path: Path to input image
        output_path: Path to save depth map visualization

    Returns:
        Normalized depth map as float32 array (0=far, 1=near)
    """
    load_model()

    img = Image.open(image_path).convert("RGB")
    original_w, original_h = img.size

    # Process image
    inputs = _processor(images=img, return_tensors="pt")
    inputs = {k: v.to(_device_torch) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_np = depth.cpu().numpy().astype(np.float32)

    # Normalize to 0-1 range
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_np)

    # Apply slight Gaussian smoothing
    depth_smooth = cv2.GaussianBlur(depth_normalized, (5, 5), 1.0)

    # Save depth visualization as grayscale PNG
    depth_vis = (depth_smooth * 255).astype(np.uint8)
    cv2.imwrite(output_path, depth_vis)

    print(f"[Depth] Depth map saved to {output_path} (shape: {depth_smooth.shape})")
    return depth_smooth


def get_model_info() -> dict:
    """Return info about the depth model status."""
    device = get_device()
    return {
        "model": "Depth Anything V2 Small",
        "device": device.upper(),
        "loaded": _model is not None,
    }
