/**
 * Image Motion Studio — Frontend Controller
 */
(function() {
  "use strict";

  // Backend API base URL (relative if served together, or localhost:8000)
  const API_BASE = window.location.origin.includes(":5173") 
    ? "http://localhost:8000" 
    : window.location.origin;

  /* ────────── Default Motion Configuration ────────── */
  const DEFAULT_PARAMS = {
    duration: 2.0, fps: 30, resolution: "1080p", aspectRatio: "original", edgeFill: "inpaint",
    pushIn: 0.0, horizontalDrift: 0.0, verticalDrift: 0.0, handheld: 6.5, cameraShake: 2.0, zoomOut: 5.0,
    depthStrength: 15.0, foregroundSeparation: 10.0,
    breathing: 10.0, watcherSway: 10.0, blink: false,
    microSaccades: 2.5, edgeFlutter: 1.0, heartbeatPulse: 2.5,
    dustParticles: 1.0, lightShift: 2.0, filmGrain: 3.0,
    rackFocus: 2.0, specularShimmer: 2.0, motionBlur: 1.0,
  };

  /* ────────── DOM Elements ────────── */
  const serverDot = document.getElementById("serverDot");
  const serverStatusText = document.getElementById("serverStatusText");
  const deviceBadge = document.getElementById("deviceBadge");

  const viewfinder = document.getElementById("viewfinder");
  const frameEl = document.getElementById("frame");
  const previewImg = document.getElementById("previewImg");
  const parallaxLayer = document.getElementById("parallaxLayer");
  const parallaxImg = document.getElementById("parallaxImg");
  const depthDisplay = document.getElementById("depthDisplay");
  const depthImg = document.getElementById("depthImg");
  const videoContainer = document.getElementById("videoContainer");
  const resultVideo = document.getElementById("resultVideo");
  const dropOverlay = document.getElementById("dropOverlay");
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const changeImgBtn = document.getElementById("changeImgBtn");
  const toast = document.getElementById("toast");

  const tabPreview = document.getElementById("tabPreview");
  const tabDepth = document.getElementById("tabDepth");
  const tabResult = document.getElementById("tabResult");

  const hudZoom = document.getElementById("hudZoom");
  const hudShake = document.getElementById("hudShake");
  const hudDepth = document.getElementById("hudDepth");
  const hudRes = document.getElementById("hudRes");

  const playPauseBtn = document.getElementById("playPauseBtn");
  const playPauseLabel = document.getElementById("playPauseLabel");
  const playIcon = document.getElementById("playIcon");
  const pauseIcon = document.getElementById("pauseIcon");
  const resetAnimBtn = document.getElementById("resetAnimBtn");

  const generateBtn = document.getElementById("generateBtn");
  const generateBtnText = document.getElementById("generateBtnText");
  const downloadBtn = document.getElementById("downloadBtn");

  const progressCard = document.getElementById("progressCard");
  const stageLabel = document.getElementById("stageLabel");
  const progressPct = document.getElementById("progressPct");
  const progressFill = document.getElementById("progressFill");

  /* ────────── State ────────── */
  let currentImageId = null;
  let hasDepth = false;
  let currentJobId = null;
  let statusPollTimer = null;

  const anim = {
    playing: true,
    elapsed: 0,
    lastTs: null,
  };

  /* ────────── Toast Helper ────────── */
  function showToast(msg, isError = false) {
    toast.textContent = msg;
    toast.classList.toggle("error", isError);
    toast.classList.add("visible");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.remove("visible"), 3500);
  }

  /* ────────── Health Check ────────── */
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      serverDot.classList.remove("error");
      serverStatusText.textContent = "Backend Connected";
      deviceBadge.textContent = `${data.device.toUpperCase()}`;
    } catch {
      serverDot.classList.add("error");
      serverStatusText.textContent = "Backend Offline";
      deviceBadge.textContent = "Offline";
    }
  }
  checkHealth();
  setInterval(checkHealth, 8000);

  /* ────────── Sliders & Inputs Binding ────────── */
  const paramKeys = [
    "duration", "fps", "resolution", "aspect_ratio", "edge_fill",
    "zoom_out", "handheld", "camera_shake",
    "depth_strength", "foreground_separation",
    "breathing", "watcher_sway", "micro_saccades", "edge_flutter", "heartbeat_pulse", "blink",
    "dust_particles", "light_shift", "film_grain", "rack_focus", "specular_shimmer", "motion_blur"
  ];

  function getParamValues() {
    const params = {};
    for (const key of paramKeys) {
      const el = document.getElementById(`param_${key}`);
      if (!el) continue;
      if (el.type === "checkbox") {
        params[key] = el.checked;
      } else if (el.tagName === "SELECT") {
        params[key] = el.value;
      } else {
        params[key] = parseFloat(el.value);
      }
    }
    return params;
  }

  function setParamValues(p) {
    const mapKey = {
      aspectRatio: "aspect_ratio",
      edgeFill: "edge_fill",
      zoomOut: "zoom_out",
      zoomIn: "zoom_out",
      cameraShake: "camera_shake",
      depthStrength: "depth_strength",
      foregroundSeparation: "foreground_separation",
      watcherSway: "watcher_sway",
      microSaccades: "micro_saccades",
      edgeFlutter: "edge_flutter",
      heartbeatPulse: "heartbeat_pulse",
      dustParticles: "dust_particles",
      lightShift: "light_shift",
      filmGrain: "film_grain",
      rackFocus: "rack_focus",
      specularShimmer: "specular_shimmer",
      motionBlur: "motion_blur"
    };

    for (const [k, v] of Object.entries(p)) {
      const normalizedKey = mapKey[k] || k;
      const el = document.getElementById(`param_${normalizedKey}`);
      if (!el) continue;
      if (el.type === "checkbox") {
        el.checked = Boolean(v);
      } else {
        el.value = v;
      }
      updateLabel(normalizedKey);
    }
  }

  function updateLabel(key) {
    const el = document.getElementById(`param_${key}`);
    const lbl = document.getElementById(`val_${key}`);
    if (!el || !lbl) return;
    if (key === "duration") {
      lbl.textContent = parseFloat(el.value).toFixed(1) + "s";
    } else {
      lbl.textContent = parseFloat(el.value).toFixed(1);
    }
  }

  paramKeys.forEach(key => {
    const el = document.getElementById(`param_${key}`);
    if (!el) return;
    el.addEventListener("input", () => {
      updateLabel(key);
    });
    updateLabel(key);
  });

  /* ────────── MotionFrame Interactive Preview Loop ────────── */
  function tick(ts) {
    if (anim.lastTs === null) anim.lastTs = ts;
    const dt = Math.min((ts - anim.lastTs) / 1000, 0.05);
    anim.lastTs = ts;

    const params = getParamValues();

    if (anim.playing) {
      anim.elapsed += dt;
      if (anim.elapsed > (params.duration || 2.0)) {
        anim.elapsed = 0;
      }
    }

    const shakeInt = params.camera_shake !== undefined ? params.camera_shake : 0.0;
    const shakeSpeed = 0.15 + shakeInt * 2.05;
    const tShake = anim.elapsed * shakeSpeed;

    // Multi-harmonic kinetic motion equations
    const nx = Math.sin(tShake * 1.0) * 0.6 + Math.sin(tShake * 1.7 + 1.3) * 0.4;
    const ny = Math.sin(tShake * 1.3 + 0.7) * 0.6 + Math.sin(tShake * 2.1 + 2.4) * 0.4;
    const nr = Math.sin(tShake * 0.55 + 0.4);

    const tx = nx * 3.0 * shakeInt;
    const ty = ny * 3.0 * shakeInt;
    const rot = nr * 0.25 * shakeInt;

    // Post-parallax pure optical camera zoom-out progression (pulls back to full frame)
    const zoomVal = params.zoom_out !== undefined ? params.zoom_out : 1.0;
    const progressT = anim.elapsed / (params.duration || 2.0);
    const easedT = 1.0 - Math.pow(1.0 - Math.min(progressT, 1.0), 3);
    const scale = 1.04 + (zoomVal / 10.0) * 0.20 * (1.0 - easedT);

    frameEl.style.transform = `scale(${scale.toFixed(4)}) translate(${tx.toFixed(2)}%, ${ty.toFixed(2)}%) rotate(${rot.toFixed(2)}deg)`;

    // Soft parallax background layer (0.35 motion factor)
    const PARALLAX_FACTOR = 0.35;
    const pxScale = scale * PARALLAX_FACTOR + (1 - PARALLAX_FACTOR);
    const ptx = tx * PARALLAX_FACTOR;
    const pty = ty * PARALLAX_FACTOR;
    const prot = rot * PARALLAX_FACTOR;
    parallaxLayer.style.transform = `scale(${pxScale.toFixed(4)}) translate(${ptx.toFixed(2)}%, ${pty.toFixed(2)}%) rotate(${prot.toFixed(2)}deg)`;

    // Real-time HUD
    hudZoom.textContent = `${Math.round((scale / 1.04) * 100)}%`;
    hudShake.textContent = shakeInt.toFixed(2);
    hudDepth.textContent = (params.depth_strength || 9.0).toFixed(1);
    hudRes.textContent = params.resolution || "1080p";

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  /* ────────── Playback Transport ────────── */
  function setPlaying(playing) {
    anim.playing = playing;
    playIcon.style.display = playing ? "none" : "";
    pauseIcon.style.display = playing ? "" : "none";
    playPauseLabel.textContent = playing ? "Pause" : "Play";
  }

  playPauseBtn.addEventListener("click", () => setPlaying(!anim.playing));
  resetAnimBtn.addEventListener("click", () => {
    anim.elapsed = 0;
  });

  /* ────────── Tab Navigation ────────── */
  function switchTab(tab) {
    tabPreview.classList.toggle("active", tab === "preview");
    tabDepth.classList.toggle("active", tab === "depth");
    tabResult.classList.toggle("active", tab === "result");

    frameEl.hidden = (tab !== "preview");
    depthDisplay.hidden = (tab !== "depth");
    videoContainer.hidden = (tab !== "result");

    if (tab === "result" && resultVideo.src) {
      resultVideo.play().catch(() => {});
    }
  }

  tabPreview.addEventListener("click", () => switchTab("preview"));
  tabDepth.addEventListener("click", () => switchTab("depth"));
  tabResult.addEventListener("click", () => switchTab("result"));

  /* ────────── Image Upload & Depth Trigger ────────── */
  uploadBtn.addEventListener("click", () => fileInput.click());
  changeImgBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleUpload(file);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach(evt => {
    viewfinder.addEventListener(evt, (e) => {
      e.preventDefault();
      viewfinder.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    viewfinder.addEventListener(evt, (e) => {
      e.preventDefault();
      viewfinder.classList.remove("drag-over");
    });
  });

  viewfinder.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });

  async function handleUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    showToast("Uploading photo...");
    dropOverlay.classList.add("hidden");

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      currentImageId = data.id;

      // Set preview images
      const imgUrl = `${API_BASE}${data.url}`;
      previewImg.src = imgUrl;
      parallaxImg.src = imgUrl;

      showToast("Photo loaded. Estimating depth map...");
      generateBtn.disabled = false;
      switchTab("preview");

      // Auto-trigger depth estimation in background
      triggerDepthEstimation(currentImageId);

    } catch (err) {
      showToast(err.message, true);
      dropOverlay.classList.remove("hidden");
    }
  }

  async function triggerDepthEstimation(imageId) {
    try {
      const res = await fetch(`${API_BASE}/api/depth`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: imageId })
      });

      if (!res.ok) throw new Error("Depth estimation failed");
      const data = await res.json();

      depthImg.src = `${API_BASE}${data.depth_url}`;
      tabDepth.disabled = false;
      hasDepth = true;
      showToast("Depth map generated & ready!");

    } catch (err) {
      showToast("Failed to compute depth map.", true);
    }
  }

  /* ────────── Motion Generation & Status Polling ────────── */
  generateBtn.addEventListener("click", async () => {
    if (!currentImageId) return;

    const params = getParamValues();
    const payload = {
      image_id: currentImageId,
      duration: params.duration,
      fps: parseInt(params.fps, 10),
      resolution: params.resolution,
      aspect_ratio: params.aspect_ratio,
      edge_fill: params.edge_fill,
      push_in: 0.0,
      horizontal_drift: 0.0,
      vertical_drift: 0.0,
      zoom_out: params.zoom_out !== undefined ? params.zoom_out : 1.0,
      handheld: params.handheld,
      camera_shake: params.camera_shake,
      depth_strength: params.depth_strength,
      foreground_separation: params.foreground_separation,
      breathing: params.breathing,
      watcher_sway: params.watcher_sway,
      blink: params.blink,
      micro_saccades: params.micro_saccades,
      edge_flutter: params.edge_flutter,
      heartbeat_pulse: params.heartbeat_pulse,
      dust_particles: params.dust_particles,
      light_shift: params.light_shift,
      film_grain: params.film_grain,
      rack_focus: params.rack_focus,
      specular_shimmer: params.specular_shimmer,
      motion_blur: params.motion_blur,
    };

    generateBtn.disabled = true;
    generateBtnText.textContent = "Rendering Video...";
    progressCard.hidden = false;
    progressFill.style.width = "5%";
    progressPct.textContent = "5%";
    stageLabel.textContent = "Initiating motion pipeline...";

    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Generation request failed");
      }

      const data = await res.json();
      currentJobId = data.job_id;
      pollJobStatus(currentJobId);

    } catch (err) {
      showToast(err.message, true);
      generateBtn.disabled = false;
      generateBtnText.textContent = "Generate Motion Video";
      progressCard.hidden = true;
    }
  });

  function pollJobStatus(jobId) {
    clearInterval(statusPollTimer);

    statusPollTimer = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${jobId}`);
        if (!res.ok) throw new Error();
        const data = await res.json();

        progressPct.textContent = `${data.progress}%`;
        progressFill.style.width = `${data.progress}%`;
        stageLabel.textContent = data.stage || "Processing...";

        if (data.status === "complete") {
          clearInterval(statusPollTimer);
          onGenerationComplete(jobId, data.result_path);
        } else if (data.status === "error") {
          clearInterval(statusPollTimer);
          showToast(`Render failed: ${data.error || "Unknown error"}`, true);
          generateBtn.disabled = false;
          generateBtnText.textContent = "Generate Motion Video";
          progressCard.hidden = true;
        }

      } catch {
        // Retry silently on temporary poll error
      }
    }, 600);
  }

  function onGenerationComplete(jobId, resultPath) {
    showToast("Render complete! Loading video...");
    generateBtn.disabled = false;
    generateBtnText.textContent = "Generate Motion Video";
    progressCard.hidden = true;

    const videoUrl = `${API_BASE}${resultPath}`;
    resultVideo.src = videoUrl;
    tabResult.disabled = false;
    switchTab("result");

    downloadBtn.href = `${API_BASE}/api/result/${jobId}`;
    downloadBtn.style.display = "inline-flex";
  }

})();
