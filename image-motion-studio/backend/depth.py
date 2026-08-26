"""
Depth estimation using Depth Anything V2 Small.
Loads model once and caches in memory.
Includes robust percentile normalization, guided edge refinement,
and adaptive dynamic-range enhancement for balanced multi-layer parallax.
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


def refine_and_balance_depth(
    depth_raw: np.ndarray,
    rgb_img: np.ndarray = None,
) -> np.ndarray:
    """
    Universal depth map refinement:
    1. Robust percentile normalization (eliminates outlier compression).
    2. Adaptive CLAHE blending (expands compressed midground subjects like drivers, seated persons).
    3. Joint edge-preserving bilateral filtering to prevent boundary bleeding.
    """
    # 1. Robust Percentile Normalization (1st to 99th percentile)
    p_low, p_high = np.percentile(depth_raw, (1.0, 99.0))
    if p_high > p_low:
        depth_norm = np.clip((depth_raw - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        depth_norm = np.clip(depth_raw, 0.0, 1.0)

    # 2. Adaptive Depth Dynamic Range Expansion (CLAHE)
    # Stretches depth contrast across compressed mid-tones without blowing out background/foreground
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth_clahe = clahe.apply(depth_u8).astype(np.float32) / 255.0

    # Blend 65% linear depth + 35% CLAHE depth for natural tonal distribution
    depth_balanced = 0.65 * depth_norm + 0.35 * depth_clahe

    # 3. Edge-Preserving Bilateral Smoothing
    # Smooths noise while keeping sharp silhouette boundaries
    depth_balanced_u8 = (depth_balanced * 255.0).astype(np.uint8)
    depth_filtered_u8 = cv2.bilateralFilter(depth_balanced_u8, d=7, sigmaColor=35, sigmaSpace=7)
    depth_refined = depth_filtered_u8.astype(np.float32) / 255.0

    return np.clip(depth_refined, 0.0, 1.0)


def estimate_depth(image_path: str, output_path: str) -> np.ndarray:
    """
    Estimate depth from a single image with universal adaptive enhancement.

    Args:
        image_path: Path to input image
        output_path: Path to save depth map visualization

    Returns:
        Normalized refined depth map as float32 array (0=far, 1=near)
    """
    load_model()

    pil_img = Image.open(image_path).convert("RGB")
    original_w, original_h = pil_img.size
    rgb_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Process image through model
    inputs = _processor(images=pil_img, return_tensors="pt")
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

    depth_raw = depth.cpu().numpy().astype(np.float32)

    # Apply universal adaptive enhancement
    depth_smooth = refine_and_balance_depth(depth_raw, rgb_cv)

    # Save depth visualization as grayscale PNG
    depth_vis = (depth_smooth * 255).astype(np.uint8)
    cv2.imwrite(output_path, depth_vis)

    print(f"[Depth] Depth map saved to {output_path} (shape: {depth_smooth.shape})")
    return depth_smooth


def get_model_info() -> dict:
    """Return model status and device info."""
    return {
        "loaded": _model is not None,
        "device": get_device(),
        "model": "Depth Anything V2 Small",
    }
