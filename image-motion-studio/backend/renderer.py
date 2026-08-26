"""
Video renderer: orchestrates camera motion, parallax frame generation,
living subject bio-motion, atmospheric particles, ambient light dynamics,
cinematic film grain, dynamic rack focus / bokeh, specular shimmer,
subsurface vascular pulse, directional motion blur, camera shake, and MP4 encoding.
"""

import math
import os
import uuid
import cv2
import numpy as np
import imageio

from parallax import create_displacement_field, warp_image, detect_subject_region


def cubic_ease_out(t: float) -> float:
    """Cubic ease-out: decelerating to stop."""
    return 1.0 - (1.0 - t) ** 3


def generate_camera_trajectory(
    num_frames: int,
    push_in: float,
    h_drift: float,
    v_drift: float,
    handheld: float,
    fps: int,
    camera_shake: float = 0.0,
) -> list[dict]:
    """
    Generate per-frame camera parameters with easing, organic handheld drift,
    MotionFrame multi-harmonic shake & rotational roll.
    """
    trajectory = []
    duration = num_frames / fps

    for i in range(num_frames):
        time_sec = i / fps

        # Settle phase: 0-0.5s = subtle motion
        # Main phase: 0.5s onwards = full motion
        if time_sec < 0.5:
            settle_t = time_sec / 0.5
            phase_t = settle_t * 0.1
        else:
            main_t = (time_sec - 0.5) / max(duration - 0.5, 0.1)
            phase_t = 0.1 + 0.9 * cubic_ease_out(min(max(main_t, 0.0), 1.0))

        # 1. MotionFrame Multi-Harmonic Kinetic Camera Shake & Roll
        # mapShakeSpeed: lerp(0.15, 2.2, speed)
        intensity = max(camera_shake / 10.0, 0.0)
        t_shake = time_sec * (0.15 + intensity * 2.05)

        # Multi-harmonic sinusoidal wave formulation
        nx = math.sin(t_shake * 1.0) * 0.6 + math.sin(t_shake * 1.7 + 1.3) * 0.4
        ny = math.sin(t_shake * 1.3 + 0.7) * 0.6 + math.sin(t_shake * 2.1 + 2.4) * 0.4
        nr = math.sin(t_shake * 0.55 + 0.4)

        # 3.0% max translate, 0.25 deg max rotation ceiling
        tx_pct = nx * 3.0 * intensity
        ty_pct = ny * 3.0 * intensity
        roll_angle = nr * (0.25 * math.pi / 180.0) * intensity

        # 2. Organic handheld drift (smooth low-frequency wander)
        handheld_scale = handheld * 0.3
        hh_noise_x = handheld_scale * (
            math.sin(time_sec * 2.3) * 0.3 +
            math.sin(time_sec * 5.7) * 0.15 +
            math.sin(time_sec * 11.1) * 0.05
        )
        hh_noise_y = handheld_scale * (
            math.sin(time_sec * 1.9 + 0.7) * 0.3 +
            math.sin(time_sec * 4.3 + 1.2) * 0.15 +
            math.sin(time_sec * 9.7 + 2.1) * 0.05
        )

        # Zoom progression with base overscan
        base_overscan = 1.0 + (push_in / 10.0) * 0.12
        zoom_time = time_sec * (0.06 + (push_in / 10.0) * 0.49)
        zoom_progress = 0.5 - 0.5 * math.cos(zoom_time)
        max_zoom_extra = 0.04 + (push_in / 10.0) * 0.51
        scale = base_overscan * (1.0 + zoom_progress * max_zoom_extra * phase_t)

        noise_x = hh_noise_x + tx_pct * 0.4
        noise_y = hh_noise_y + ty_pct * 0.4

        trajectory.append({
            "t": phase_t,
            "time_sec": time_sec,
            "noise_x": noise_x,
            "noise_y": noise_y,
            "tx_pct": tx_pct,
            "ty_pct": ty_pct,
            "scale": scale,
            "roll_angle": roll_angle,
            "frame_index": i,
        })

    return trajectory


def render_motionframe_parallax_background(
    image: np.ndarray,
    target_shape: tuple[int, int],
    scale: float,
    tx_pct: float,
    ty_pct: float,
    rot_rad: float,
    parallax_intensity: float = 0.25,
) -> np.ndarray:
    """
    Renders the MotionFrame soft ambient blurred parallax background layer:
    - Gaussian blur + 0.55 brightness + 0.85 saturation.
    - Damped motion tracking: 0.35x translation, scale, and rotation.
    - Full-bleed 1.15x cover scale with alpha blending.
    """
    h, w = target_shape[:2]
    img_h, img_w = image.shape[:2]

    PARALLAX_MOTION_FACTOR = 0.35

    # 1. Base blurred, dimmed, desaturated image
    blur_k = max(21, int(min(img_w, img_h) * 0.03) | 1)
    blurred = cv2.GaussianBlur(image, (blur_k, blur_k), 10.0)

    # Convert to HSV to adjust saturation to 85% and brightness to 55%
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255)  # saturate(0.85)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.55, 0, 255)  # brightness(0.55)
    bg_base = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 2. Compute transform with PARALLAX_MOTION_FACTOR
    px_scale = (scale * PARALLAX_MOTION_FACTOR + (1.0 - PARALLAX_MOTION_FACTOR)) * 1.15
    cover_scale = max(w / img_w, h / img_h)
    total_scale = px_scale * cover_scale

    p_tx = (tx_pct * PARALLAX_MOTION_FACTOR / 100.0) * w
    p_ty = (ty_pct * PARALLAX_MOTION_FACTOR / 100.0) * h
    p_rot_deg = (rot_rad * 180.0 / math.pi) * PARALLAX_MOTION_FACTOR

    center = (img_w / 2.0, img_h / 2.0)
    M = cv2.getRotationMatrix2D(center, p_rot_deg, total_scale)
    M[0, 2] += (w / 2.0 - center[0]) + p_tx
    M[1, 2] += (h / 2.0 - center[1]) + p_ty

    bg_layer = cv2.warpAffine(
        bg_base, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return bg_layer


class DustParticleSystem:
    """
    Simulates 3D depth-aware illuminated air dust motes drifting in the room.
    """
    def __init__(self, num_particles: int = 45, width: int = 1080, height: int = 1920):
        self.w = width
        self.h = height
        np.random.seed(42)  # Deterministic seed for reproducible cinematic look

        # Particles (x, y, z, base_radius, drift_speed_x, drift_speed_y, brightness)
        self.particles = []
        for _ in range(num_particles):
            self.particles.append({
                "x": np.random.uniform(0.05 * width, 0.95 * width),
                "y": np.random.uniform(0.05 * height, 0.95 * height),
                "z": np.random.uniform(0.1, 0.9),  # depth: 0.1 (far) to 0.9 (near)
                "radius": np.random.uniform(1.2, 3.5),
                "speed_x": np.random.uniform(-15.0, 20.0),
                "speed_y": np.random.uniform(-8.0, 12.0),
                "phase": np.random.uniform(0, 2 * math.pi),
                "brightness": np.random.uniform(0.4, 0.9),
            })

    def render(self, frame: np.ndarray, time_sec: float, intensity: float) -> np.ndarray:
        if intensity <= 0:
            return frame

        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        alpha_scale = min(intensity / 10.0, 1.0)

        for p in self.particles:
            cur_x = (p["x"] + p["speed_x"] * time_sec * p["z"] + math.sin(time_sec * 1.5 + p["phase"]) * 10 * p["z"]) % w
            cur_y = (p["y"] + p["speed_y"] * time_sec * p["z"] + math.cos(time_sec * 1.2 + p["phase"]) * 8 * p["z"]) % h

            size = p["radius"] * (0.8 + 1.8 * p["z"])
            int_x, int_y = int(cur_x), int(cur_y)
            rad = max(int(size), 1)

            flicker = 0.85 + 0.15 * math.sin(time_sec * 3.5 + p["phase"])
            color = np.array([200, 230, 255], dtype=np.float32) * p["brightness"] * flicker * alpha_scale

            cv2.circle(overlay, (int_x, int_y), rad, color.tolist(), -1)

        overlay = cv2.GaussianBlur(overlay, (7, 7), 2.5)
        out = frame.astype(np.float32) + overlay
        return np.clip(out, 0, 255).astype(np.uint8)


def apply_ambient_light_pulse(
    frame: np.ndarray,
    time_sec: float,
    intensity: float,
) -> np.ndarray:
    """
    Subtle warm ambient light breathing & illumination dynamics.
    """
    if intensity <= 0:
        return frame

    h, w = frame.shape[:2]
    pulse = math.sin(time_sec * 2.2) * 0.5 + math.sin(time_sec * 4.1) * 0.2
    scale = (intensity / 10.0) * pulse * 0.08

    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (1.0 - (y_idx / h) * 0.4) * (0.6 + 0.4 * (x_idx / w))

    light_map = np.ones((h, w, 3), dtype=np.float32)
    light_map[:, :, 0] += grad * scale * 0.6  # Blue
    light_map[:, :, 1] += grad * scale * 1.0  # Green
    light_map[:, :, 2] += grad * scale * 1.3  # Red (warm)

    out = frame.astype(np.float32) * light_map
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_subsurface_vascular_pulse(
    frame: np.ndarray,
    depth_map: np.ndarray,
    time_sec: float,
    intensity: float,
    subject_depth: float = 0.5,
) -> np.ndarray:
    """
    Subconscious physiological realism: 66 BPM (~1.1 Hz) vascular micro-flush on skin-chroma regions.
    """
    if intensity <= 0:
        return frame

    h, w = frame.shape[:2]
    bpm_freq = 66.0 / 60.0 * 2.0 * math.pi  # ~1.1 Hz
    pulse = (math.sin(time_sec * bpm_freq) * 0.65 + math.sin(time_sec * bpm_freq * 2.0 - 0.5) * 0.35)
    pulse_amp = (intensity / 10.0) * 0.012 * max(pulse, 0.0)

    # Subject depth mask centered dynamically around subject_depth
    subject_mask = np.exp(-((depth_map - subject_depth) ** 2) / (2 * (0.20 ** 2)))

    # Skin-chroma tone mask in YCrCb color space
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)
    skin_mask = np.clip((cr - 130) / 15.0, 0.0, 1.0) * np.clip((175 - cr) / 15.0, 0.0, 1.0) * np.clip((cb - 75) / 15.0, 0.0, 1.0) * np.clip((130 - cb) / 15.0, 0.0, 1.0)
    skin_mask = skin_mask * subject_mask

    if skin_mask.max() > 0:
        skin_mask_3d = np.stack([skin_mask * 0.2, skin_mask * 0.5, skin_mask * 1.2], axis=-1)
        out = frame.astype(np.float32) * (1.0 + skin_mask_3d * pulse_amp)
        return np.clip(out, 0, 255).astype(np.uint8)

    return frame


def apply_dynamic_dof_bokeh(
    frame: np.ndarray,
    depth_map: np.ndarray,
    time_sec: float,
    intensity: float,
    subject_depth: float = 0.5,
) -> np.ndarray:
    """
    Dynamic depth-of-field rack focus and background bokeh expansion.
    Focal plane automatically tracks the detected subject's depth.
    """
    if intensity <= 0:
        return frame

    h, w = frame.shape[:2]
    # Focal plane locks dynamically onto the subject depth with subtle breathing
    focal_depth = subject_depth + 0.03 * math.sin(time_sec * 0.8)
    coc = np.abs(depth_map - focal_depth) * (intensity / 10.0) * 1.8

    # Multi-layer progressive blur
    blur_level1 = cv2.GaussianBlur(frame, (9, 9), 3.0)
    blur_level2 = cv2.GaussianBlur(frame, (21, 21), 7.0)

    w1 = np.clip((coc - 0.15) / 0.25, 0.0, 1.0)[:, :, np.newaxis]
    w2 = np.clip((coc - 0.40) / 0.30, 0.0, 1.0)[:, :, np.newaxis]

    out = frame.astype(np.float32) * (1.0 - w1) + blur_level1.astype(np.float32) * (w1 * (1.0 - w2)) + blur_level2.astype(np.float32) * w2
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_specular_shimmer(
    frame: np.ndarray,
    time_sec: float,
    intensity: float,
) -> np.ndarray:
    """
    Glint and flare on catchlights and shiny surfaces responsive to camera drift angle.
    """
    if intensity <= 0:
        return frame

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, highlights = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

    if highlights.max() == 0:
        return frame

    shimmer = math.sin(time_sec * 4.5) * 0.4 + math.sin(time_sec * 9.2) * 0.2
    flare_amp = (intensity / 10.0) * (0.8 + shimmer * 0.5)

    kernel_h = np.zeros((3, 9), dtype=np.float32)
    kernel_h[1, :] = 1.0 / 9.0
    glint = cv2.filter2D(highlights.astype(np.float32), -1, kernel_h)
    glint = cv2.GaussianBlur(glint, (5, 5), 1.5)

    glint_3d = np.stack([glint * 0.8, glint * 0.95, glint * 1.0], axis=-1) * flare_amp * 0.4
    out = frame.astype(np.float32) + glint_3d
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_directional_motion_blur(
    frame: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """
    Subtle velocity-aware motion blur along displacement flow vectors.
    """
    if intensity <= 0:
        return frame

    avg_dx = float(np.mean(np.abs(dx)))
    avg_dy = float(np.mean(np.abs(dy)))
    flow_mag = math.sqrt(avg_dx ** 2 + avg_dy ** 2)

    if flow_mag < 0.8:
        return frame

    ksize = min(int(flow_mag * (intensity / 10.0) * 1.2), 7)
    if ksize % 2 == 0:
        ksize += 1
    if ksize <= 1:
        return frame

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    angle = math.atan2(np.mean(dy), np.mean(dx))
    cx, cy = ksize // 2, ksize // 2

    for r in range(ksize):
        for c in range(ksize):
            dr, dc = r - cy, c - cx
            pt_angle = math.atan2(dr, dc)
            diff = abs(angle - pt_angle)
            if diff < 0.4 or abs(diff - math.pi) < 0.4:
                kernel[r, c] = 1.0

    if kernel.sum() > 0:
        kernel /= kernel.sum()
        blurred = cv2.filter2D(frame, -1, kernel)
        blend_w = min((intensity / 10.0) * 0.4, 0.45)
        out = cv2.addWeighted(frame, 1.0 - blend_w, blurred, blend_w, 0)
        return out

    return frame


def apply_film_grain(
    frame: np.ndarray,
    frame_index: int,
    intensity: float,
) -> np.ndarray:
    """
    Add dynamic cinematic 35mm film grain (per-frame temporal noise).
    Modulated by image luminance so highlights/shadows retain realistic grain response.
    """
    if intensity <= 0:
        return frame

    h, w = frame.shape[:2]
    np.random.seed(frame_index * 1337 + 42)

    grain_scale = (intensity / 10.0) * 12.0
    noise = np.random.normal(0, grain_scale, (h, w, 1)).astype(np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    midtones = 4.0 * gray * (1.0 - gray)
    midtones = np.expand_dims(midtones, axis=-1)

    out = frame.astype(np.float32) + noise * midtones
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_post_parallax_camera_motion(
    frame: np.ndarray,
    time_sec: float,
    duration: float,
    zoom_out: float = 1.0,
    camera_shake: float = 0.0,
    handheld: float = 3.0,
) -> np.ndarray:
    """
    Applies pure optical 2D camera zoom-out and multi-harmonic kinetic camera shake
    AFTER parallax displacement on the final output frame.
    Zero depth warping distortion or layer tearing.
    """
    h, w = frame.shape[:2]
    t = min(max(time_sec / max(duration, 0.01), 0.0), 1.0)
    eased_t = cubic_ease_out(t)

    # 1. Optical Zoom Out calculation (starts at tight punched-in scale and eases out to full frame)
    zoom_amp = (zoom_out / 10.0) * 0.20
    zoom_scale = 1.04 + zoom_amp * (1.0 - eased_t)

    # 2. Multi-Harmonic Kinetic Camera Shake & Roll on the final output
    shake_intensity = camera_shake / 10.0
    t_shake = time_sec * (0.15 + shake_intensity * 2.05)

    nx = math.sin(t_shake * 1.0) * 0.6 + math.sin(t_shake * 1.7 + 1.3) * 0.4
    ny = math.sin(t_shake * 1.3 + 0.7) * 0.6 + math.sin(t_shake * 2.1 + 2.4) * 0.4
    nr = math.sin(t_shake * 0.55 + 0.4)

    # Handheld wander
    hh_x = (handheld / 10.0) * (math.sin(time_sec * 2.1) * 0.4 + math.sin(time_sec * 4.3) * 0.15)
    hh_y = (handheld / 10.0) * (math.sin(time_sec * 1.7 + 0.5) * 0.4 + math.sin(time_sec * 3.8 + 1.1) * 0.15)

    tx_px = (nx * 3.0 * shake_intensity + hh_x) * (w / 100.0)
    ty_px = (ny * 3.0 * shake_intensity + hh_y) * (h / 100.0)
    rot_deg = (nr * 0.35 * shake_intensity) * (180.0 / math.pi)

    total_scale = max(zoom_scale, 1.04)

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rot_deg, total_scale)
    M[0, 2] += tx_px
    M[1, 2] += ty_px

    zoomed = cv2.warpAffine(
        frame, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return zoomed


def render_frames(
    image: np.ndarray,
    depth_map: np.ndarray,
    duration: float,
    fps: int,
    push_in: float,
    h_drift: float,
    v_drift: float,
    handheld: float,
    depth_strength: float,
    foreground_separation: float,
    edge_fill: str,
    aspect_ratio: str,
    resolution: str,
    zoom_out: float = 5.0,
    breathing: float = 10.0,
    watcher_sway: float = 10.0,
    blink: bool = False,
    dust_particles: float = 1.0,
    light_shift: float = 2.0,
    film_grain: float = 3.0,
    micro_saccades: float = 2.5,
    edge_flutter: float = 1.0,
    rack_focus: float = 2.0,
    specular_shimmer: float = 2.0,
    heartbeat_pulse: float = 2.5,
    motion_blur: float = 1.0,
    camera_shake: float = 0.0,
    progress_callback=None,
) -> list[np.ndarray]:
    """
    Render all frames with universal adaptive parallax, living subject motion,
    optical physics, atmospheric dynamics, and camera shake.
    """
    num_frames = int(duration * fps)

    # Apply aspect ratio crop
    image = apply_aspect_ratio(image, aspect_ratio)
    depth_map = apply_aspect_ratio(depth_map, aspect_ratio)

    # Apply resolution scaling
    image, depth_map = apply_resolution(image, depth_map, resolution)
    h, w = image.shape[:2]

    # Detect subject metrics for universal focal plane and bio-motion tracking
    subject_info = detect_subject_region(depth_map)
    subject_depth = subject_info["subject_depth"]

    # Initialize particle system for this resolution
    particles = DustParticleSystem(num_particles=45, width=w, height=h)

    # Generate camera trajectory with camera shake
    trajectory = generate_camera_trajectory(
        num_frames=num_frames,
        push_in=push_in,
        h_drift=h_drift,
        v_drift=v_drift,
        handheld=handheld,
        fps=fps,
        camera_shake=camera_shake,
    )

    frames = []
    for i, cam in enumerate(trajectory):
        # 1. Render MotionFrame soft ambient blurred parallax background layer
        bg_layer = render_motionframe_parallax_background(
            image=image,
            target_shape=(h, w),
            scale=cam.get("scale", 1.12),
            tx_pct=cam.get("tx_pct", 0.0),
            ty_pct=cam.get("ty_pct", 0.0),
            rot_rad=cam.get("roll_angle", 0.0),
            parallax_intensity=foreground_separation / 10.0,
        )

        # 2. Create combined displacement field with universal adaptive curve and rotational shake
        dx, dy = create_displacement_field(
            depth_map=depth_map,
            push_in=push_in,
            h_drift=h_drift + cam["noise_x"],
            v_drift=v_drift + cam["noise_y"],
            depth_strength=depth_strength,
            foreground_separation=foreground_separation,
            frame_t=cam["t"],
            raw_time_sec=cam["time_sec"],
            breathing=breathing,
            watcher_sway=watcher_sway,
            blink=blink,
            micro_saccades=micro_saccades,
            edge_flutter=edge_flutter,
            roll_angle=cam.get("roll_angle", 0.0),
            image=image,
        )

        # 3. Warp image with edge fill and ambient background blend
        frame = warp_image(image, dx, dy, edge_fill, bg_layer=bg_layer)

        # 3. Dynamic Rack Focus & Bokeh Blur (tracks auto-detected subject depth)
        if rack_focus > 0:
            frame = apply_dynamic_dof_bokeh(frame, depth_map, cam["time_sec"], rack_focus, subject_depth)

        # 4. Specular Catchlight Shimmer
        if specular_shimmer > 0:
            frame = apply_specular_shimmer(frame, cam["time_sec"], specular_shimmer)

        # 5. Subsurface Vascular / Heartbeat Pulse
        if heartbeat_pulse > 0:
            frame = apply_subsurface_vascular_pulse(frame, depth_map, cam["time_sec"], heartbeat_pulse, subject_depth)

        # 6. Directional Motion Blur
        if motion_blur > 0:
            frame = apply_directional_motion_blur(frame, dx, dy, motion_blur)

        # 7. Ambient Light Breathing
        if light_shift > 0:
            frame = apply_ambient_light_pulse(frame, cam["time_sec"], light_shift)

        # 8. Render 3D Dust Motes
        if dust_particles > 0:
            frame = particles.render(frame, cam["time_sec"], dust_particles)

        # 9. Apply Living Film Grain
        if film_grain > 0:
            frame = apply_film_grain(frame, i, film_grain)

        # 10. Post-Parallax Pure Optical Camera Zoom-Out & Final Output Camera Shake
        frame = apply_post_parallax_camera_motion(
            frame=frame,
            time_sec=cam["time_sec"],
            duration=duration,
            zoom_out=zoom_out,
            camera_shake=camera_shake,
            handheld=handheld,
        )

        frames.append(frame)

        if progress_callback:
            progress_callback(i + 1, num_frames)

    return frames


def apply_aspect_ratio(img: np.ndarray, aspect_ratio: str) -> np.ndarray:
    """Crop image to target aspect ratio from center."""
    if aspect_ratio == "original" or aspect_ratio is None:
        return img

    h, w = img.shape[:2]
    ratios = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "1:1": 1.0,
    }
    target_ratio = ratios.get(aspect_ratio, w / h)
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        img = img[:, offset:offset + new_w]
    else:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        img = img[offset:offset + new_h, :]

    return img


def apply_resolution(
    image: np.ndarray, depth_map: np.ndarray, resolution: str
) -> tuple[np.ndarray, np.ndarray]:
    """Scale image and depth map to target resolution."""
    if resolution == "original" or resolution is None:
        return image, depth_map

    h, w = image.shape[:2]
    target_heights = {
        "1080p": 1080,
        "720p": 720,
    }
    target_h = target_heights.get(resolution, h)

    if h != target_h:
        scale = target_h / h
        target_w = int(w * scale)
        target_w = target_w if target_w % 2 == 0 else target_w + 1
        target_h = target_h if target_h % 2 == 0 else target_h + 1

        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        depth_map = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return image, depth_map


def encode_mp4(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> str:
    """
    Encode frames to MP4 using imageio-ffmpeg with H.264 codec.
    """
    if not frames:
        raise ValueError("No frames to encode")

    h, w = frames[0].shape[:2]
    if w % 2 != 0 or h % 2 != 0:
        new_w = w if w % 2 == 0 else w + 1
        new_h = h if h % 2 == 0 else h + 1
        frames = [cv2.resize(f, (new_w, new_h)) for f in frames]

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=None,
        output_params=[
            "-pix_fmt", "yuv420p",
            "-crf", "17",
            "-preset", "medium",
        ],
    )

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

    writer.close()
    return output_path


def run_pipeline(
    image_path: str,
    depth_map: np.ndarray,
    output_dir: str,
    temp_dir: str,
    duration: float = 2.0,
    fps: int = 30,
    push_in: float = 0.0,
    h_drift: float = 0.0,
    v_drift: float = 0.0,
    handheld: float = 6.5,
    zoom_out: float = 5.0,
    depth_strength: float = 15.0,
    foreground_separation: float = 10.0,
    edge_fill: str = "inpaint",
    aspect_ratio: str = "original",
    resolution: str = "1080p",
    breathing: float = 10.0,
    watcher_sway: float = 10.0,
    blink: bool = False,
    dust_particles: float = 1.0,
    light_shift: float = 2.0,
    film_grain: float = 3.0,
    micro_saccades: float = 2.5,
    edge_flutter: float = 1.0,
    rack_focus: float = 2.0,
    specular_shimmer: float = 2.0,
    heartbeat_pulse: float = 2.5,
    motion_blur: float = 1.0,
    camera_shake: float = 0.0,
    progress_callback=None,
) -> str:
    """
    Run the full rendering pipeline with living motion dynamics and camera shake.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    if progress_callback:
        progress_callback("Generating motion", 0)

    def frame_progress(current, total):
        if progress_callback:
            pct = int((current / total) * 100)
            progress_callback("Rendering frames", pct)

    frames = render_frames(
        image=image,
        depth_map=depth_map,
        duration=duration,
        fps=fps,
        push_in=push_in,
        h_drift=h_drift,
        v_drift=v_drift,
        handheld=handheld,
        depth_strength=depth_strength,
        foreground_separation=foreground_separation,
        edge_fill=edge_fill,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        zoom_out=zoom_out,
        breathing=breathing,
        watcher_sway=watcher_sway,
        blink=blink,
        dust_particles=dust_particles,
        light_shift=light_shift,
        film_grain=film_grain,
        micro_saccades=micro_saccades,
        edge_flutter=edge_flutter,
        rack_focus=rack_focus,
        specular_shimmer=specular_shimmer,
        heartbeat_pulse=heartbeat_pulse,
        motion_blur=motion_blur,
        camera_shake=camera_shake,
        progress_callback=frame_progress,
    )

    if progress_callback:
        progress_callback("Encoding MP4", 0)

    output_filename = f"motion_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    encode_mp4(frames, output_path, fps)

    if progress_callback:
        progress_callback("Complete", 100)

    return output_path
