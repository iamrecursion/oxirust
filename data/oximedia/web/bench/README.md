# bench/

Benchmark harness for the `oximedia-web` wasm modules (M4a). Measures the
published `dist/*.js` wrappers (`scopes`, `color`, `scale`, `quality`)
against deterministic synthetic frames, alongside honest pure-JS/Canvas2D
baselines for comparison — all in one runnable, reproducible harness.

**Reproducibility is the product.** Every comparative number this project
ever publishes has to come from running this harness, by a stranger, with
one command. There are no placeholder results and no estimated tables
anywhere in this directory; a results table with nothing in it means "run
it locally," not "numbers pending."

## Limitations — read this before trusting a number

- **Single machine, single browser, single run.** Nothing here is
  cross-browser-tested, cross-OS-tested, or averaged over multiple runs.
  `bench/run.sh` reports one run's median/p95/min per suite; run-to-run
  variance on a shared/laptop machine (thermal throttling, background
  processes, Chrome's own tab/process scheduling) can be significant.
  Treat a single `local-latest.json` as "a number from one machine on one
  day," and re-run before drawing any conclusion that matters.
- **No cross-browser matrix.** `run.sh` drives one specific installed
  Google Chrome. Safari and Firefox have different WebCodecs/wasm-SIMD
  support and different JIT characteristics; this harness says nothing
  about them today. The page (`index.html`) itself is plain, dependency-free
  ES modules and works in any browser you open it in manually — only the
  automated `run.sh` path is Chrome-specific.
- **The baselines are best-effort JS, not SOTA.** `lib/baselines.js`'s
  pure-JS luma waveform/histogram and the Canvas2D `drawImage` +
  `getImageData` downscale are straightforward, reasonably-written
  single-threaded scalar implementations — not hand-tuned, not using
  `OffscreenCanvas` workers or manual SIMD.js-style tricks. They exist to
  answer "what would a plain JS implementation of this cost," not "what is
  the fastest possible non-wasm implementation." Labelled results say
  exactly this (`baseline-js: ...`, `baseline-canvas2d: ...`) — never
  mistake them for a claim about the theoretical best case.
- **Frame acquisition is part of what's measured, on purpose.** The
  `scopes`/`color`/`scale` suites call the real wrapper methods
  (`Scopes#waveform`, `ColorPipeline#applyToCanvas`, ...), which internally
  acquire pixels from a `VideoFrame` (via `copyTo`) or `OffscreenCanvas`
  (via `drawImage` + `getImageData`) before running the wasm kernel — see
  `../js/_frame.js`. That acquisition cost is real and is what an
  integrator actually pays, so it is included rather than measuring the
  wasm kernel in isolation. The `quality` suite is the one exception: its
  wrapper has a documented raw-buffer fast path
  (`Quality#compare(Uint8ClampedArray, Uint8ClampedArray)`) that skips
  acquisition entirely, and the benchmark uses it, because that is the
  realistic way to call it when you already have decoded pixels. The
  environment block in every results JSON records which acquisition path
  (`VideoFrame` or `OffscreenCanvas`) was actually used, since that
  materially affects the numbers.
- **Synthetic frames, not real footage.** `lib/rng.js` generates
  deterministic gradient+noise and pure-noise frames from a seeded PRNG —
  see that file for the exact formulas. They are chosen to avoid
  degenerate fast paths (flat color, all-zero) without shipping or
  depending on any binary test asset, not to resemble real footage
  statistically.

## How to run

```sh
./web/bench/run.sh
```

That's the one command. It:

1. Builds `web/dist/` if it looks missing or stale (`../scripts/build.sh`).
2. Starts a local dev server on a free port (`python3 -m http.server`,
   the same one `../scripts/serve.sh` uses).
3. Launches headless Chrome against `index.html?auto=1`, which runs every
   suite automatically and reports completion through Chrome's own
   `--enable-logging=stderr` console output (see the script for why this
   was chosen over `--dump-dom`: on the Chrome version this was built
   against, `--dump-dom` does not wait for the page's own async work to
   finish, so it captured an empty/premature page).
4. Writes the parsed results to `results/local-latest.json` (left
   **untracked** — nothing in this directory commits results for you; see
   `results/README.md`).
5. Prints the results table to your terminal.

If Google Chrome isn't found at the expected location, `run.sh` prints
manual instructions (build, serve, open `bench/index.html` yourself) and
exits cleanly rather than failing — the harness's browser page has no
Chrome-specific dependency, only the automated driver script does.

To run manually in any browser instead:

```sh
./web/scripts/build.sh      # if dist/ is missing or stale
./web/scripts/serve.sh 8080
# open http://127.0.0.1:8080/bench/index.html and click "Run benchmarks",
# or open .../bench/index.html?auto=1 to run automatically.
```

## How to read the results

Each entry in `results` is one suite:

| field        | meaning                                                          |
|--------------|-------------------------------------------------------------------|
| `name`       | Suite name, e.g. `scopes: waveform (1080p)`.                     |
| `n`          | Measured sample count (warmup excluded — see below).             |
| `median_ms`  | Median wall time per call, milliseconds.                          |
| `p95_ms`     | 95th-percentile wall time per call.                                |
| `min_ms`     | Fastest observed call — closest thing to a noise floor.           |

Every suite runs 10 untimed warmup calls (let the JIT tier up and any
grow-once internal buffers reach steady state) followed by 60 timed calls,
each timed individually with `performance.now()`.

Suites, in the order they run:

- `scopes: waveform / vectorscope / histogram / falseColor (1080p)` — each
  scope alone, one 1920x1080 frame, default 512x256 scope output.
- `scopes: all-four combined (1080p)` — all four scopes back-to-back on the
  same frame, timed as one call (this is closer to what a real
  multi-scope UI panel actually costs per incoming frame than summing the
  four individual numbers post hoc, since it excludes the per-suite setup
  each individual entry above amortizes away).
- `scopes: all-four combined (1080p, worst-case noise)` — same, but on the
  pure-noise stress frame instead of the gradient one, to show the spread
  between a realistic and an adversarial input.
- `color: exposure+aces+lut33 (1080p)` — exposure(+0.7 stops) -> ACES tone
  map -> a real (identity-valued, but not identity-*cost*) 33^3 tetrahedral
  LUT, applied via `ColorPipeline#applyToCanvas`.
- `scale: lanczos3 (4K -> 1080p)` — `Scaler#resizeToCanvas` downscaling the
  3840x2160 gradient frame to 1920x1080.
- `quality: ssim+psnr (1080p pair)` — `Quality#compare` on a 1920x1080
  reference/distorted pair (see `lib/rng.js`'s `deriveDistorted`).
- `baseline-js: luma waveform / luma histogram (1080p)` — the honest pure-JS
  comparison points, same 1920x1080 gradient frame as the `scopes` suites.
- `baseline-canvas2d: downscale drawImage+getImageData (4K -> 1080p)` — the
  honest Canvas2D comparison point for the `scale` suite, same 4K frame.

## Files

- `index.html` / `bench.js` — the page. Zero external dependencies, plain
  ES modules, works from any static file server (no COOP/COEP headers
  needed — same guarantee the rest of `oximedia-web` makes).
- `lib/rng.js` — the deterministic synthetic-frame generator.
- `lib/frame-source.js` — wraps a raw RGBA8 buffer into whatever
  `FrameSource` shape the wrapper APIs expect (`VideoFrame`, preferring it
  when available, else `OffscreenCanvas`).
- `lib/harness.js` — the warmup/measure/reduce timing loop.
- `lib/baselines.js` — the pure-JS waveform/histogram baselines.
- `run.sh` — headless-Chrome driver (see "How to run" above).
- `results/` — where `run.sh` writes `local-latest.json`; see
  `results/README.md`.
