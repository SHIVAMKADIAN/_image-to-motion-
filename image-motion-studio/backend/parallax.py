"""
Depth-based parallax warping and organic living motion fields.
Includes:
- Universal adaptive subject-aware parallax curve (dynamic mid-depth boost + foreground soft clamping)
- Auto-subject spatial centroid & depth band detection
- Biological subject breathing deformation field (asymmetrical physiological curve)
- Ocular micro-saccades and gaze drift
- Edge & fabric wavelet flutter (micro-breeze dynamics)
- Foreground watcher organic sway field
- Eye micro-blink / settle deformation
"""

import math
import cv2
import numpy as np


def compute_adaptive_depth_weight(
    depth_map: np.ndarray,
    foreground_separation: float = 5.0,
) -> np.ndarray:
    """
    Universal non-linear depth transfer function:
    - Protects midground subjects (humans, seated drivers, animals, objects) from being starved of motion.
    - Applies an adaptive S-curve with mid-depth slope boost.
    - Soft-clamps extreme foreground (>0.85) with hyperbolic saturation to prevent edge tearing.
    """
    fg_factor = foreground_separation / 10.0

    # 1. Base smooth S-curve mapping for natural depth progression
    # Sigmoidal boost centered around mid-depth (0.45)
    k = 4.5 + fg_factor * 2.0
    x0 = 0.40 - fg_factor * 0.10
    sig = 1.0 / (1.0 + np.exp(-k * (depth_map - x0)))
    sig_min = 1.0 / (1.0 + np.exp(k * x0))
    sig_norm = (sig - sig_min) / (1.0 - sig_min + 1e-6)

    # 2. Linear blend to preserve depth gradient linearity
    depth_weight = 0.45 * depth_map + 0.55 * sig_norm

    # 3. Soft foreground saturation (hyperbolic tangent roll-off above 0.82)
    fg_mask = np.clip((depth_map - 0.82) / 0.18, 0.0, 1.0)
    clamped_weight = 0.85 + 0.15 * np.tanh((depth_weight - 0.85) * 3.0)
    final_weight = depth_weight * (1.0 - fg_mask) + clamped_weight * fg_mask

    return np.clip(final_weight, 0.0, 1.0)


def detect_subject_region(depth_map: np.ndarray) -> dict:
    """
    Universal subject detector:
    Analyzes depth distribution to find the primary focal subject's spatial centroid (cx, cy),
    depth band, and spatial dimensions across any random image.
    """
    h, w = depth_map.shape[:2]

    # Find mid-depth range where human subjects typically sit (excluding extreme background < 0.15 and extreme foreground > 0.85)
    subject_mask = (depth_map >= 0.20) & (depth_map <= 0.80)

    if not np.any(subject_mask):
        # Fallback to general depth-weighted center
        weights = depth_map
    else:
        weights = depth_map * subject_mask.astype(np.float32)

    total_weight = np.sum(weights)
    if total_weight > 0:
        y_indices, x_indices = np.mgrid[0:h, 0:w]
        cx = float(np.sum(x_indices * weights) / total_weight)
        cy = float(np.sum(y_indices * weights) / total_weight)

        # Subject depth value at centroid
        int_x = int(np.clip(cx, 0, w - 1))
        int_y = int(np.clip(cy, 0, h - 1))
        subject_depth = float(depth_map[int_y, int_x])
    else:
        cx, cy = 0.5 * w, 0.5 * h
        subject_depth = 0.5

    return {
        "cx": cx,
        "cy": cy,
        "subject_depth": subject_depth,
        "sigma_x": max(0.18 * w, 40.0),
        "sigma_y": max(0.22 * h, 40.0),
    }


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
    micro_saccades: float = 2.0,
    edge_flutter: float = 2.0,
    roll_angle: float = 0.0,
    image: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create combined X and Y displacement fields for camera motion + living subject motion.
    """
    h, w = depth_map.shape[:2]

    # Universal subject analysis for coordinate anchoring
    subject_info = detect_subject_region(depth_map)

    # --- 1. CAMERA DISPLACEMENT FIELD (Adaptive Subject-Aware) ---
    push_scale = push_in * 10.0
    h_drift_scale = h_drift * 8.0
    v_drift_scale = v_drift * 8.0
    depth_scale = depth_strength / 10.0

    # Compute universal adaptive depth weight
    depth_weight = compute_adaptive_depth_weight(depth_map, foreground_separation)

    cy, cx = h / 2.0, w / 2.0
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    x_rel = (x_coords - cx) / cx
    y_rel = (y_coords - cy) / cy

    push_dx = -x_rel * push_scale * depth_weight * depth_scale * frame_t
    push_dy = -y_rel * push_scale * depth_weight * depth_scale * frame_t

    drift_depth_factor = 0.35 + 0.65 * depth_weight * depth_scale
    drift_dx = h_drift_scale * drift_depth_factor * frame_t
    drift_dy = v_drift_scale * drift_depth_factor * frame_t

    total_dx = push_dx + drift_dx
    total_dy = push_dy + drift_dy

    # --- 1b. CAMERA ROTATIONAL SHAKE / ROLL ---
    if roll_angle != 0.0:
        cos_r = math.cos(roll_angle)
        sin_r = math.sin(roll_angle)
        rot_dx = (x_rel * (cos_r - 1.0) - y_rel * sin_r) * cx
        rot_dy = (x_rel * sin_r + y_rel * (cos_r - 1.0)) * cy
        total_dx += rot_dx
        total_dy += rot_dy

    # --- 2. SUBJECT BREATHING FIELD (Living Subject - Anchored Dynamically) ---
    if breathing > 0:
        b_dx, b_dy = compute_breathing_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            intensity=breathing,
            x_coords=x_coords,
            y_coords=y_coords,
            subject_info=subject_info,
            w=w,
            h=h,
        )
        total_dx += b_dx
        total_dy += b_dy

    # --- 3. OCULAR MICRO-SACCADES & GAZE DRIFT (Anchored to Upper Subject Region) ---
    if micro_saccades > 0:
        s_dx, s_dy = compute_micro_saccades_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            intensity=micro_saccades,
            x_coords=x_coords,
            y_coords=y_coords,
            subject_info=subject_info,
            w=w,
            h=h,
        )
        total_dx += s_dx
        total_dy += s_dy

    # --- 4. SECONDARY EDGE & FABRIC WAVELET FLUTTER ---
    if edge_flutter > 0:
        ef_dx, ef_dy = compute_edge_flutter_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            intensity=edge_flutter,
            image=image,
            w=w,
            h=h,
        )
        total_dx += ef_dx
        total_dy += ef_dy

    # --- 5. FOREGROUND WATCHER ORGANIC SWAY ---
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

    # --- 6. EYE MICRO-BLINK / SETTLE ---
    if blink:
        eye_dx, eye_dy = compute_eye_micro_blink_field(
            depth_map=depth_map,
            time_sec=raw_time_sec,
            x_coords=x_coords,
            y_coords=y_coords,
            subject_info=subject_info,
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
    subject_info: dict,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Organic chest/torso breathing deformation field anchored dynamically to the detected subject.
    """
    cycle_period = 3.5
    t_phase = (time_sec % cycle_period) / cycle_period

    if t_phase < 0.38:
        u = t_phase / 0.38
        breath_cycle = 0.5 * (1.0 - math.cos(u * math.pi))
    elif t_phase < 0.48:
        breath_cycle = 1.0 - 0.03 * math.sin((t_phase - 0.38) / 0.10 * math.pi)
    else:
        u = (t_phase - 0.48) / 0.52
        breath_cycle = 0.5 * (1.0 + math.cos(u * math.pi))

    b_amp = (intensity / 10.0) * 3.6 * (breath_cycle - 0.5)

    # Dynamic anchor coordinates
    chest_x = subject_info["cx"]
    chest_y = min(subject_info["cy"] + 0.08 * h, 0.90 * h)  # Torso is slightly below subject center
    sigma_x = subject_info["sigma_x"]
    sigma_y = subject_info["sigma_y"]
    s_depth = subject_info["subject_depth"]

    # Subject depth proximity band
    subject_depth_mask = np.exp(-((depth_map - s_depth) ** 2) / (2 * (0.22 ** 2)))

    torso_weight = np.exp(-(((x_coords - chest_x) ** 2) / (2 * (sigma_x ** 2)) + ((y_coords - chest_y) ** 2) / (2 * (sigma_y ** 2))))
    combined_weight = subject_depth_mask * torso_weight

    dx_rel = (x_coords - chest_x) / max(sigma_x, 1.0)
    dy_rel = (y_coords - chest_y) / max(sigma_y, 1.0)

    dx = dx_rel * b_amp * combined_weight * 0.8
    dy = (dy_rel * b_amp - abs(b_amp) * 0.4) * combined_weight

    return dx.astype(np.float32), dy.astype(np.float32)


def compute_micro_saccades_field(
    depth_map: np.ndarray,
    time_sec: float,
    intensity: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    subject_info: dict,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate ocular micro-saccades dynamically anchored to upper subject head area.
    """
    scale = (intensity / 10.0) * 0.85

    saccade_x = (
        math.sin(time_sec * 6.3) * 0.4 +
        math.sin(time_sec * 17.1 + 0.3) * 0.25 +
        math.sin(time_sec * 29.7 + 1.1) * 0.15
    ) * scale
    saccade_y = (
        math.cos(time_sec * 5.7 + 0.8) * 0.35 +
        math.sin(time_sec * 15.3 + 0.5) * 0.2 +
        math.cos(time_sec * 31.2 + 2.0) * 0.1
    ) * scale

    # Face/head region (upper portion of detected subject)
    face_x = subject_info["cx"]
    face_y = max(subject_info["cy"] - 0.10 * h, 0.15 * h)
    sigma_face_x = max(subject_info["sigma_x"] * 0.5, 25.0)
    sigma_face_y = max(subject_info["sigma_y"] * 0.4, 25.0)
    s_depth = subject_info["subject_depth"]

    face_weight = np.exp(-(((x_coords - face_x) ** 2) / (2 * (sigma_face_x ** 2)) + ((y_coords - face_y) ** 2) / (2 * (sigma_face_y ** 2))))
    subject_depth = np.exp(-((depth_map - s_depth) ** 2) / (2 * (0.18 ** 2)))
    weight = face_weight * subject_depth

    dx = weight * saccade_x
    dy = weight * saccade_y

    return dx.astype(np.float32), dy.astype(np.float32)


def compute_edge_flutter_field(
    depth_map: np.ndarray,
    time_sec: float,
    intensity: float,
    image: np.ndarray,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Secondary motion: subtle organic flutter / micro-breeze on high-contrast silhouette edges & clothing.
    """
    scale = (intensity / 10.0) * 1.5
    if scale <= 0:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    depth_u8 = (depth_map * 255).astype(np.uint8)
    edges = cv2.Canny(depth_u8, 30, 90).astype(np.float32) / 255.0
    edges_blurred = cv2.GaussianBlur(edges, (15, 15), 4.0)

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    freq_y = 0.02
    freq_x = 0.02
    wave_x = np.sin(y_grid * freq_y + time_sec * 3.2) * np.cos(x_grid * freq_x + time_sec * 2.1)
    wave_y = np.cos(y_grid * freq_y * 1.2 + time_sec * 2.7) * np.sin(x_grid * freq_x * 0.9 + time_sec * 1.8)

    depth_factor = np.clip(depth_map, 0.2, 0.9)

    dx = edges_blurred * wave_x * scale * depth_factor
    dy = edges_blurred * wave_y * (scale * 0.6) * depth_factor

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
    fg_mask = np.clip((depth_map - 0.78) / 0.15, 0.0, 1.0)

    sway_amp = (intensity / 10.0) * 3.5
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
    subject_info: dict,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a subtle organic micro-blink / eyelid settle around t = 1.0s to 1.25s.
    """
    blink_start = 1.0
    blink_duration = 0.22
    if not (blink_start <= time_sec <= blink_start + blink_duration):
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    rel_t = (time_sec - blink_start) / blink_duration
    blink_weight = math.sin(rel_t * math.pi) ** 2

    eye_x = subject_info["cx"]
    eye_y = max(subject_info["cy"] - 0.10 * h, 0.15 * h)
    sigma_x = max(subject_info["sigma_x"] * 0.25, 15.0)
    sigma_y = max(subject_info["sigma_y"] * 0.15, 10.0)
    s_depth = subject_info["subject_depth"]

    subject_depth = np.exp(-((depth_map - s_depth) ** 2) / (2 * (0.18 ** 2)))
    eye_weight = np.exp(-(((x_coords - eye_x) ** 2) / (2 * (sigma_x ** 2)) + ((y_coords - eye_y) ** 2) / (2 * (sigma_y ** 2)))) * subject_depth

    dy = eye_weight * (blink_weight * 2.2)
    dx = np.zeros_like(dy)

    return dx.astype(np.float32), dy.astype(np.float32)


def warp_image(
    image: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    edge_fill: str = "mirror",
    bg_layer: np.ndarray = None,
) -> np.ndarray:
    """
    Warp image using displacement fields with edge fill handling and ambient background blending.
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

    # If an ambient parallax background layer is provided, blend boundary edges with it
    if bg_layer is not None:
        mask = _create_oob_mask(map_x, map_y, w, h)
        if mask.any():
            mask_3d = np.stack([mask] * 3, axis=-1).astype(np.float32)
            mask_3d = cv2.GaussianBlur(mask_3d, (21, 21), 7.0)
            warped = (warped.astype(np.float32) * (1.0 - mask_3d) + bg_layer.astype(np.float32) * mask_3d)
            warped = np.clip(warped, 0, 255).astype(np.uint8)

    return warped


def _create_oob_mask(map_x: np.ndarray, map_y: np.ndarray, w: int, h: int) -> np.ndarray:
    """Create a boolean mask of pixels that map outside original image bounds."""
    oob = (map_x < 0) | (map_x >= w - 1) | (map_y < 0) | (map_y >= h - 1)
    return oob.astype(np.float32)
