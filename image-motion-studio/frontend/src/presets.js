/**
 * Image Motion Studio — Fixed Motion Configuration
 */

const PRESETS = {
  "default": {
    label: "Living Motion Studio (Standard)",
    description: "2.0s 1080p, 2.5 Zoom Out, 5.0 Handheld Drift, 0.0 Camera Shake, 15.0 Depth, 10.0 Separation, 10.0 Breathing, 10.0 Sway.",
    params: {
      duration: 2.0,
      fps: 30,
      resolution: "1080p",
      aspectRatio: "original",
      edgeFill: "inpaint",
      pushIn: 0.0,
      hDrift: 0.0,
      vDrift: 0.0,
      zoomOut: 2.5,
      handheld: 5.0,
      cameraShake: 0.0,
      depthStrength: 15.0,
      foregroundSeparation: 10.0,
      breathing: 10.0,
      watcherSway: 10.0,
      blink: false,
      microSaccades: 2.5,
      edgeFlutter: 1.0,
      heartbeatPulse: 2.5,
      dustParticles: 0.0,
      lightShift: 2.0,
      filmGrain: 3.0,
      rackFocus: 2.0,
      specularShimmer: 10.0,
      motionBlur: 1.0,
    },
  },
};

export default PRESETS;
