"""
Video renderer: orchestrates camera motion, parallax frame generation,
living subject bio-motion, atmospheric particles, ambient light dynamics,
cinematic film grain, and MP4 encoding.
"""

import math
import os
import uuid
import cv2
import numpy as np
import imageio

from parallax import create_displacement_field, warp_image


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
) -> list[dict]:
    """
    Generate per-frame camera parameters with easing and subtle handheld noise.
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

        # Organic handheld noise
        handheld_scale = handheld * 0.3
        noise_x = handheld_scale * (
            math.sin(time_sec * 2.3) * 0.3 +
            math.sin(time_sec * 5.7) * 0.15 +
            math.sin(time_sec * 11.1) * 0.05
        )
        noise_y = handheld_scale * (
            math.sin(time_sec * 1.9 + 0.7) * 0.3 +
            math.sin(time_sec * 4.3 + 1.2) * 0.15 +
            math.sin(time_sec * 9.7 + 2.1) * 0.05
        )

        trajectory.append({
            "t": phase_t,
            "time_sec": time_sec,
            "noise_x": noise_x,
            "noise_y": noise_y,
            "frame_index": i,
        })

    return trajectory


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
            # 3D drift calculation
            cur_x = (p["x"] + p["speed_x"] * time_sec * p["z"] + math.sin(time_sec * 1.5 + p["phase"]) * 10 * p["z"]) % w
            cur_y = (p["y"] + p["speed_y"] * time_sec * p["z"] + math.cos(time_sec * 1.2 + p["phase"]) * 8 * p["z"]) % h

            # Size and bokeh blur scaled by depth (near particles are larger, softer)
            size = p["radius"] * (0.8 + 1.8 * p["z"])
            int_x, int_y = int(cur_x), int(cur_y)
            rad = max(int(size), 1)

            # Golden warm ambient light mote color (BGR: warm yellow/white)
            flicker = 0.85 + 0.15 * math.sin(time_sec * 3.5 + p["phase"])
            color = np.array([200, 230, 255], dtype=np.float32) * p["brightness"] * flicker * alpha_scale

            # Draw soft circular mote
            cv2.circle(overlay, (int_x, int_y), rad, color.tolist(), -1)

        # Soft blur for atmospheric bokeh
        overlay = cv2.GaussianBlur(overlay, (7, 7), 2.5)

        # Additive blend onto frame
        out = frame.astype(np.float32) + overlay
        return np.clip(out, 0, 255).astype(np.uint8)


def apply_ambient_light_pulse(
    frame: np.ndarray,
    time_sec: float,
    intensity: float,
) -> np.ndarray:
    """
    Subtle warm ambient light breathing & doorway illumination dynamics.
    """
    if intensity <= 0:
        return frame

    h, w = frame.shape[:2]
    # Slow organic light pulse (cycle ~2.8s)
    pulse = math.sin(time_sec * 2.2) * 0.5 + math.sin(time_sec * 4.1) * 0.2
    scale = (intensity / 10.0) * pulse * 0.08  # up to ~8% brightness shift

    # Gradient: Light comes predominantly from doorway/window (top-left / right doorway)
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (1.0 - (y_idx / h) * 0.4) * (0.6 + 0.4 * (x_idx / w))

    # Apply warm tone boost (increase Red/Green slightly more than Blue)
    light_map = np.ones((h, w, 3), dtype=np.float32)
    light_map[:, :, 0] += grad * scale * 0.6  # Blue
    light_map[:, :, 1] += grad * scale * 1.0  # Green
    light_map[:, :, 2] += grad * scale * 1.3  # Red (warm)

    out = frame.astype(np.float32) * light_map
    return np.clip(out, 0, 255).astype(np.uint8)


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

    grain_scale = (intensity / 10.0) * 12.0  # max ~12 pixel intensity variance
    noise = np.random.normal(0, grain_scale, (h, w, 1)).astype(np.float32)

    # Grain is more visible in midtones than pure darks or blown highlights
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    midtones = 4.0 * gray * (1.0 - gray)  # Peak at 0.5 luminance
    midtones = np.expand_dims(midtones, axis=-1)

    out = frame.astype(np.float32) + noise * midtones
    return np.clip(out, 0, 255).astype(np.uint8)


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
    breathing: float = 3.0,
    watcher_sway: float = 3.0,
    blink: bool = True,
    dust_particles: float = 4.0,
    light_shift: float = 3.0,
    film_grain: float = 2.0,
    progress_callback=None,
) -> list[np.ndarray]:
    """
    Render all frames with parallax, living subject motion, and atmospheric dynamics.
    """
    num_frames = int(duration * fps)

    # Apply aspect ratio crop
    image = apply_aspect_ratio(image, aspect_ratio)
    depth_map = apply_aspect_ratio(depth_map, aspect_ratio)

    # Apply resolution scaling
    image, depth_map = apply_resolution(image, depth_map, resolution)
    h, w = image.shape[:2]

    # Initialize particle system for this resolution
    particles = DustParticleSystem(num_particles=45, width=w, height=h)

    # Generate camera trajectory
    trajectory = generate_camera_trajectory(
        num_frames, push_in, h_drift, v_drift, handheld, fps
    )

    frames = []
    for i, cam in enumerate(trajectory):
        # 1. Create combined displacement field (Camera + Subject Breathing + Watcher Sway + Blink)
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
        )

        # 2. Warp image with edge fill
        frame = warp_image(image, dx, dy, edge_fill)

        # 3. Apply Ambient Light Breathing
        if light_shift > 0:
            frame = apply_ambient_light_pulse(frame, cam["time_sec"], light_shift)

        # 4. Render 3D Dust Motes (Air volume)
        if dust_particles > 0:
            frame = particles.render(frame, cam["time_sec"], dust_particles)

        # 5. Apply Living Film Grain
        if film_grain > 0:
            frame = apply_film_grain(frame, i, film_grain)

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
    push_in: float = 2.0,
    h_drift: float = 1.0,
    v_drift: float = 0.0,
    handheld: float = 1.0,
    depth_strength: float = 4.0,
    foreground_separation: float = 5.0,
    edge_fill: str = "mirror",
    aspect_ratio: str = "original",
    resolution: str = "1080p",
    breathing: float = 3.0,
    watcher_sway: float = 3.0,
    blink: bool = True,
    dust_particles: float = 4.0,
    light_shift: float = 3.0,
    film_grain: float = 2.0,
    progress_callback=None,
) -> str:
    """
    Run the full rendering pipeline with living motion dynamics.
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
        breathing=breathing,
        watcher_sway=watcher_sway,
        blink=blink,
        dust_particles=dust_particles,
        light_shift=light_shift,
        film_grain=film_grain,
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
