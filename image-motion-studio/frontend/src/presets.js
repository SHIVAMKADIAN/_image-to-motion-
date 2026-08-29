/**
 * Image Motion Studio — Fixed Motion Configuration
 */

const PRESETS = {
  "default": {
    label: "Living Motion Studio (Standard)",
    description: "2.0s 1080p, No Zoom, 5.0 Handheld Drift, 0.0 Camera Shake, 25.0 Depth, 20.0 Separation, 10.0 Breathing, 10.0 Sway.",
    params: {
      duration: 2.0,
      fps: 30,
      resolution: "1080p",
      aspectRatio: "original",
      edgeFill: "mirror",
      pushIn: 0.0,
      hDrift: 0.0,
      vDrift: 0.0,
      zoomIn: 0.0,
      zoomOut: 0.0,
      handheld: 6.5,
      cameraShake: 0.0,
      horizontalWiggle: 5.0,
      depthStrength: 25.0,
      foregroundSeparation: 20.0,
      breathing: 9.0,
      watcherSway: 9.0,
      blink: false,
      microSaccades: 2.5,
      edgeFlutter: 1.0,
      heartbeatPulse: 2.5,
      dustParticles: 1.0,
      lightShift: 2.0,
      filmGrain: 3.0,
      rackFocus: 2.0,
      specularShimmer: 2.0,
      motionBlur: 1.0,
    },
  },
};

export default PRESETS;
