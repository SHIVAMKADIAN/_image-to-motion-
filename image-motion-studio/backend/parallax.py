"""
Depth-based parallax warping and organic living motion fields.
Includes:
- Camera parallax displacement
- Biological subject breathing deformation field
- Foreground watcher organic sway field
- Eye micro-blink / settle deformation
"""

import math
import cv2
import numpy as np


def create_displacement_field(
    depth_map: np.ndarray,
    push_in: float,
    h_drift: float,
    v_drift: float,
    depth_strength: float,
    foreground_separation: float,
    frame_t: float,
    raw_time_sec: float = 0.0,
    breathing: float = 3.0,
    watcher_sway: float = 3.0,
    blink: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create combined X and Y displacement fields for camera motion + living subject motion.

    Args:
        depth_map: Normalized depth map (0=far, 1=near), shape (H, W)
        push_in: Push-in camera amount (0-10)
        h_drift: Horizontal drift (-10 to +10)
        v_drift: Vertical drift (-10 to +10)
        depth_strength: How much depth affects displacement (0-10)
        foreground_separation: Foreground vs background separation (0-10)
        frame_t: Normalized time for camera motion (0-1, eased)
        raw_time_sec: Current timestamp in seconds (for periodic bio-cycles)
        breathing: Chest/torso breathing intensity (0-10)
        watcher_sway: Foreground watcher independent sway (0-10)
        blink: Enable subtle eye micro-blink
    """
    h, w = depth_map.shape[:2]

    # --- 1. CAMERA DISPLACEMENT FIELD ---
    push_scale = push_in * 10.0
    h_drift_scale = h_drift * 8.0
    v_drift_scale = v_drift * 8.0
    depth_scale = depth_strength / 10.0
    fg_sep = foreground_separation / 10.0

    depth_weight = np.power(depth_map, 1.0 + fg_sep * 2.0)
    if depth_weight.max() > 0:
        depth_weight = depth_weight / depth_weight.max()

    cy, cx = h / 2.0, w / 2.0
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    x_rel = (x_coords - cx) / cx
    y_rel = (y_coords - cy) / cy

    push_dx = -x_rel * push_scale * depth_weight * depth_scale * frame_t
    push_dy = -y_rel * push_scale * depth_weight * depth_scale * frame_t

    drift_depth_factor = 0.3 + 0.7 * depth_weight * depth_scale
    drift_dx = h_drift_scale * drift_depth_factor * frame_t
    drift_dy = v_drift_scale * drift_depth_factor * frame_t

    total_dx = push_dx + drift_dx
    total_dy = push_dy + drift_dy

    # --- 2. SUBJECT BREATHING FIELD (Living Human) ---
    if breathing > 0:
        b_dx, b_dy = compute_breathing_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            intensity=breathing,
            x_coords=x_coords,
            y_coords=y_coords,
            w=w,
            h=h,
        )
        total_dx += b_dx
        total_dy += b_dy

    # --- 3. FOREGROUND WATCHER ORGANIC SWAY ---
    if watcher_sway > 0:
        w_dx, w_dy = compute_foreground_sway_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            intensity=watcher_sway,
            w=w,
            h=h,
        )
        total_dx += w_dx
        total_dy += w_dy

    # --- 4. EYE MICRO-BLINK / SETTLE ---
    if blink:
        eye_dx, eye_dy = compute_eye_micro_blink_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            x_coords=x_coords,
            y_coords=y_coords,
            w=w,
            h=h,
        )
        total_dx += eye_dx
        total_dy += eye_dy

    return total_dx.astype(np.float32), total_dy.astype(np.float32)


def compute_breathing_field(
    depth_map: np.ndarray,
    time_sec: float,
    intensity: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an organic chest/torso expansion and contraction deformation field.
    The breathing cycle uses an asymmetric inhalation/exhalation curve (~3.5 second natural period).
    """
    # Natural breathing cycle: faster inhale, slower relaxed exhale
    cycle_period = 3.2
    phase = (time_sec % cycle_period) / cycle_period  # 0 to 1
    # Smooth biological respiration curve (smooth expansion and gentle drop)
    breath_cycle = math.sin(phase * 2 * math.pi) * 0.7 + math.sin(phase * 4 * math.pi - 0.5) * 0.3

    # Breathing amplitude scaled by user intensity (e.g. 0 to ~4 pixels max displacement)
    b_amp = (intensity / 10.0) * 3.5 * breath_cycle

    # Isolate subject region:
    # 1. Subject depth band (typically mid-depth: 0.35 to 0.75 in normalized depth)
    subject_depth_mask = np.clip((depth_map - 0.3) / 0.25, 0.0, 1.0) * np.clip((0.8 - depth_map) / 0.15, 0.0, 1.0)

    # 2. Chest & Torso anatomical center (approx center-left in normalized coords for seated character)
    # Character chest center is approx at (0.50 * w, 0.60 * h)
    chest_x = 0.50 * w
    chest_y = 0.60 * h
    sigma_x = 0.20 * w
    sigma_y = 0.22 * h

    # 2D Gaussian torso spatial weighting
    torso_weight = np.exp(-(((x_coords - chest_x) ** 2) / (2 * (sigma_x ** 2)) + ((y_coords - chest_y) ** 2) / (2 * (sigma_y ** 2))))

    combined_weight = subject_depth_mask * torso_weight

    # Organic chest displacement: expands outward radially and rises slightly vertically
    dx_rel = (x_coords - chest_x) / sigma_x
    dy_rel = (y_coords - chest_y) / sigma_y

    # Slight upward lift during inhale (-y) + radial expansion
    dx = dx_rel * b_amp * combined_weight * 0.7
    dy = (dy_rel * b_amp - abs(b_amp) * 0.5) * combined_weight

    return dx.astype(np.float32), dy.astype(np.float32)


def compute_foreground_sway_field(
    depth_map: np.ndarray,
    time_sec: float,
    intensity: float,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate independent organic swaying and breathing for the foreground watcher/silhouette.
    """
    # Foreground mask: highest depth values (depth > 0.75)
    fg_mask = np.clip((depth_map - 0.75) / 0.15, 0.0, 1.0)

    # Multi-frequency pendulum swaying (natural human balancing / breathing stance)
    sway_amp = (intensity / 10.0) * 4.0
    sway_x = math.sin(time_sec * 1.8) * 0.7 + math.sin(time_sec * 3.4) * 0.3
    sway_y = math.cos(time_sec * 1.5) * 0.5 + math.sin(time_sec * 2.7) * 0.2

    dx = fg_mask * (sway_x * sway_amp)
    dy = fg_mask * (sway_y * sway_amp * 0.6)

    return dx.astype(np.float32), dy.astype(np.float32)


def compute_eye_micro_blink_field(
    depth_map: np.ndarray,
    time_sec: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a subtle organic micro-blink / eyelid settle around t = 1.0s to 1.25s.
    """
    blink_start = 1.0
    blink_duration = 0.22  # ~220ms natural eyelid closure
    if not (blink_start <= time_sec <= blink_start + blink_duration):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    # Blink bell-curve (rapid close, slightly slower open)
    rel_t = (time_sec - blink_start) / blink_duration
    blink_weight = math.sin(rel_t * math.pi) ** 2

    # Eye region location for the subject (center head approx at 0.51*w, 0.46*h)
    eye_x = 0.51 * w
    eye_y = 0.465 * h
    sigma_x = 0.04 * w
    sigma_y = 0.02 * h

    # Subject depth check
    subject_depth = np.clip((depth_map - 0.4) / 0.2, 0.0, 1.0) * np.clip((0.8 - depth_map) / 0.15, 0.0, 1.0)

    eye_weight = np.exp(-(((x_coords - eye_x) ** 2) / (2 * (sigma_x ** 2)) + ((y_coords - eye_y) ** 2) / (2 * (sigma_y ** 2)))) * subject_depth

    # Downward slight compression of eyelid (max ~1.5 - 2 pixels)
    dy = eye_weight * (blink_weight * 2.2)
    dx = np.zeros_like(dy)

    return dx.astype(np.float32), dy.astype(np.float32)


def warp_image(
    image: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    edge_fill: str = "mirror",
) -> np.ndarray:
    """
    Warp image using displacement fields with edge fill handling.
    """
    h, w = image.shape[:2]

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x_coords + dx
    map_y = y_coords + dy

    if edge_fill == "mirror":
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    elif edge_fill == "blur":
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask = _create_oob_mask(map_x, map_y, w, h)
        if mask.any():
            blurred = cv2.GaussianBlur(warped, (31, 31), 10)
            mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
            mask_3d = cv2.GaussianBlur(mask_3d, (15, 15), 5)
            warped = (warped * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
    elif edge_fill == "inpaint":
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = _create_oob_mask(map_x, map_y, w, h)
        if mask.any():
            inpaint_mask = (mask * 255).astype(np.uint8)
            warped = cv2.inpaint(warped, inpaint_mask, 10, cv2.INPAINT_TELEA)
    else:
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    return warped


def _create_oob_mask(map_x: np.ndarray, map_y: np.ndarray, w: int, h: int) -> np.ndarray:
    """Create a boolean mask of pixels that map outside original image bounds."""
    oob = (map_x < 0) | (map_x >= w - 1) | (map_y < 0) | (map_y >= h - 1)
    return oob.astype(np.float32)
