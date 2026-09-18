# OxiScope demo (M2 + M3)

A dark, colorist-tool web app that grades WebCodecs frames with the
`@cooljapan/oximedia-web` WebAssembly modules and shows four live broadcast
scopes reading the **graded** output — waveform (RGB parade, IRE), vectorscope
(skin-tone line, 75% graticule), histogram and false colour.

Pure static ES modules. No build step, no bundler, no network beyond this
origin — nothing is ever uploaded.

## Run

```bash
../scripts/build.sh          # once, if dist/ is missing or stale
../scripts/serve.sh 8080     # python3 -m http.server
# open http://localhost:8080/demo/
```

## What it does

- **Inputs**: drag-drop / pick a video file (`blob:` URL, never uploaded), the
  webcam (`getUserMedia`, prefers `MediaStreamTrackProcessor`), or three
  built-in **procedural** patterns (SMPTE bars, luma-sweep + zone plate,
  out-of-gamut saturation) drawn on a hidden canvas and driven via
  `captureStream` — so a visitor with no footage still gets the moment.
- **Grade** (`ColorPipeline`): exposure (stops), contrast, saturation,
  tone-map operator (off / Reinhard / Filmic / ACES / ACES-ODT) with a peak-nits
  slider, and optional BT.2020→709 / P3→709 gamut mapping. The graded frame is
  painted to the preview **and** fed to all four scopes, so switching a tone-map
  operator visibly rolls the 100 IRE white patch off the top of the waveform.
- **Export**: bakes the current grade to a 33³ `.cube` and downloads it via a
  `Blob` — the counter still reads *uploaded: 0 bytes*.
- **Honesty**: a live `wasm: N kB` badge measured from the Resource Timing API
  (never hard-coded), a per-frame perf line (fps + grade/scope ms breakdown,
  degrading to alternating scopes under load but never dropping the preview),
  and a capabilities readout. Missing wasm SIMD is a hard, actionable error —
  there is no software-codec fallback.

## Files

| File | Role |
| --- | --- |
| `index.html` | Layout: preview + 2×2 scope grid + controls. |
| `style.css` | Dark theme, CSS grid, responsive to ~1024px. |
| `app.js` | Orchestration: capabilities, wasm init, render loop, badges, export. |
| `patterns.js` | The three procedural test-pattern generators. |
| `sources.js` | Input management + frame drivers (rVFC / `MediaStreamTrackProcessor`). |
