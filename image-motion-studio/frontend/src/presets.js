/**
 * Image Motion Studio — Frontend Presets
 *
 * Each preset defines a named set of rendering parameters.
 * The "living-cinematic" preset is the default and matches
 * the backend system defaults.
 */

const PRESETS = {
  // The frontend's default — aligned to backend system defaults.
  "living-cinematic": {
    label: "Living Cinematic (Alive Human + Room)",
    description: "Breathing subject, swaying foreground, dust and warm light.",
    params: {
      pushIn: 1.0, hDrift: 3.0, vDrift: 2.0, handheld: 6.0,
      depthStrength: 9.0, foregroundSeparation: 9.0,
      breathing: 9.0, watcherSway: 9.0, blink: false,
      dustParticles: 2.5, lightShift: 3.0, filmGrain: 5.0,
    },
  },

  "subtle-breathing": {
    label: "Living Portrait (Breathing + Gaze)",
    description: "Breath foregrounded, camera almost static. For a face filling the frame.",
    params: {
      pushIn: 1.0, hDrift: 3.0, vDrift: 2.0, handheld: 6.0,
      depthStrength: 9.0, foregroundSeparation: 9.0,
      breathing: 9.0, watcherSway: 9.0, blink: false,
      dustParticles: 2.5, lightShift: 3.0, filmGrain: 5.0,
    },
  },

  // Hard foreground sway with the blink deliberately off — the silhouette is
  // meant to read as watching, not as alive.
  "voyeur-stalker": {
    label: "Voyeur Watcher (Doorway Silhouette Sway)",
    description: "Heavy foreground sway, high depth separation, no blink.",
    params: {
      pushIn: 1.0, hDrift: 3.0, vDrift: 2.0, handheld: 6.0,
      depthStrength: 9.0, foregroundSeparation: 9.0,
      breathing: 9.0, watcherSway: 9.0, blink: false,
      dustParticles: 2.5, lightShift: 3.0, filmGrain: 5.0,
    },
  },

  // Camera only. Everything organic and atmospheric off, which also makes this
  // the baseline to diff the others against when a render looks wrong.
  "push-in-parallax": {
    label: "Push-In + Parallax (Classic)",
    description: "Camera move only — no breathing, dust, light or grain.",
    params: {
      pushIn: 1.0, hDrift: 3.0, vDrift: 2.0, handheld: 6.0,
      depthStrength: 9.0, foregroundSeparation: 9.0,
      breathing: 9.0, watcherSway: 9.0, blink: false,
      dustParticles: 2.5, lightShift: 3.0, filmGrain: 5.0,
    },
  },

  // The only preset with negative vertical drift: the camera rises slightly.
  "cinematic-drift": {
    label: "Atmospheric Drift",
    description: "Lateral drift with a slow rise, heavy dust and light.",
    params: {
      pushIn: 1.0, hDrift: 3.0, vDrift: 2.0, handheld: 6.0,
      depthStrength: 9.0, foregroundSeparation: 9.0,
      breathing: 9.0, watcherSway: 9.0, blink: false,
      dustParticles: 2.5, lightShift: 3.0, filmGrain: 5.0,
    },
  },
};

export default PRESETS;
