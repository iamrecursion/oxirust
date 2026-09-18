# Changelog — oximedia-compat-cv2

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] — 2026-05-07 (Run 5)

### Added
- **Slice A — Farneback dense optical flow**: `calc_optical_flow_farneback` fully implemented
  (Gunnar Farneback 2003 polynomial-expansion algorithm); Gaussian pyramid, separable 6-coefficient
  polynomial expansion, per-pixel 2×2 system solve, spatial box/Gaussian smoothing, iterative
  refinement, coarse-to-fine upsampling. Output is `CV_32FC2` Mat.
- **Slice A — MatType additions**: `CV_32FC2` (two-channel float), `CV_32FC3`, `CV_64FC1`
  (double-precision, used for homography/affine matrices).
- **Slice B — Perspective warp completeness**: `get_perspective_transform` (DLT 8×8 system, outputs
  3×3 `CV_64FC1`), `get_affine_transform` (DLT 6×6, outputs 2×3 `CV_64FC1`), `warp_perspective`
  (inverse-map warp with `WARP_INVERSE_MAP` support, bilinear/nearest sampling, border modes),
  `remap` (generic CV_32FC1 float-map sampling).
- **Slice C — Shape analysis**: `Moments` struct (24 fields, Green's theorem O(n) computation),
  `hu_moments` (Hu 1962 seven invariants), `fit_ellipse` (Fitzgibbon-Pilu-Fisher direct
  least-squares with Jacobi 6×6 eigenvalue solver), `fit_line` (PCA for DIST_L2, IRLS for DIST_L1),
  `min_area_rect` (rotating-calipers on convex hull), `point_polygon_test` (Jordan curve ray-casting
  + signed distance), `RotatedRect` struct.
- **Slice D — Mat reductions and channel ops**: `count_non_zero`, `sum_elems`, `mean_val`,
  `mean_std_dev` (Welford's algorithm), `norm`, `norm_diff`, `min_max_loc`, `split`, `merge`.
  Mat methods: `clone_mat`, `convert_to`, `submat`, `reshape`.
- **Slice E — Drawing completions**: `draw_contours` (outline + FILLED scanline), `draw_keypoints`
  (with `DRAW_RICH_KEYPOINTS` orientation lines), `draw_matches` (side-by-side stitch + match
  lines), `draw_marker` (7 marker types: CROSS, TILTED_CROSS, STAR, DIAMOND, SQUARE,
  TRIANGLE_UP, TRIANGLE_DOWN).
- **Slice E — Hershey vector font**: `put_text_hershey` with `FONT_HERSHEY_SIMPLEX`; 24-glyph
  subset (digits 0–9, uppercase A–N). Full 95-glyph table deferred to Refinement 12.
- **Slice F — BFMatcher norm generalization**: `BFMatcher::new()` now accepts `NORM_L1`, `NORM_L2`,
  `NORM_L2SQR` in addition to `NORM_HAMMING`/`NORM_HAMMING2`. `DMatch.distance` promoted to `f32`
  for cv2 API parity. Dtype validation at match entry (CV_8UC1 for binary norms, CV_32FC1 for
  float norms).
- **New constants**: `NORM_L2SQR`, `WARP_INVERSE_MAP`, `OPTFLOW_FARNEBACK_GAUSSIAN`,
  `OPTFLOW_USE_INITIAL_FLOW`, `DRAW_MATCHES_FLAGS_*`, `MARKER_*` types, `DIST_*` types.

### Tests added (Run 5)
- `tests/farneback_smoke.rs` — 6 tests (Farneback flow correctness + dtype)
- `tests/perspective_smoke.rs` — 14 tests (DLT homography/affine, warp_perspective, remap)
- `tests/shape_analysis.rs` — 17 tests (moments, Hu, fit_ellipse, fit_line, min_area_rect, point_polygon_test)
- `tests/reductions_smoke.rs` — 32 tests (reductions, channel ops, Mat methods)
- `tests/drawing_completions.rs` — 30 tests (draw_contours, draw_keypoints, draw_matches, draw_marker, put_text_hershey)
- `tests/bf_matcher_norms.rs` — 6 tests (NORM_L2/L1/L2SQR identity, Pythagorean distance, dtype mismatch, knn sorting)

## [0.1.7] - 2026-05-06

### Added
- `Mat` type with `MatType` enum (CV_8UC1/3/4, CV_32FC1/3).
- 134 OpenCV constants grouped by category module.
- `imread`, `imwrite`, `imdecode`, `imencode` image I/O functions.
- `Scalar`, `Point`, `Point2f`, `Size`, `Rect` geometry types.
- `Cv2Error`, `Cv2Result` error types.
- **Compile-time constants reflection** — `build.rs` parses
  `src/constants.rs` with `syn` and emits a flat `LIST_CONSTANTS:
  &[(category, name, type, value)]` table to `OUT_DIR/constants_list.rs`.
  `src/lib.rs` re-exports it as `pub mod constants_list`. Eliminates the
  hand-maintained constant list previously duplicated inside the
  `oximedia-cv2` binary; the binary now iterates this auto-generated
  table.
- **Full ORB pipeline** in `src/features.rs` — image-pyramid construction,
  FAST corner detection, Harris re-scoring for stability, intensity-centroid
  orientation estimation, 256-bit rotated BRIEF descriptor extraction with
  Gaussian-smoothed sampling pairs, and Hamming brute-force matching.
  `features.rs` grew 401 → 1227 lines, well under the 2000-line refactor
  ceiling.
- **`bf_match_hamming(query, train)`** — symmetric brute-force descriptor
  matcher returning `Vec<DMatch>`; mirrors the `cv2.BFMatcher` Hamming-norm
  default for ORB.
- **`DMatch` struct** with `query_idx`, `train_idx`, and `distance` fields,
  matching the `cv2.DMatch` shape.
- 7 new feature/match unit tests covering ORB construction, descriptor
  shape, self-match Hamming distance == 0, and rotated-image match recall.

### Changed
- `orb_create()` now returns `Ok(Orb::default())` instead of
  `Err(Cv2Error::FeatureNotImplemented)`. Callers that previously had to
  check for the placeholder error path can drop that branch.

### Validated
- `cargo nextest run -p oximedia-compat-cv2 --all-features` passes
  (153 tests).
- `cargo clippy -p oximedia-compat-cv2 --all-features --all-targets --
  -D warnings` is clean.
- `cargo doc --no-deps -p oximedia-compat-cv2` is clean.

### Run 4 (2026-05-06)

#### Added
- **`dnn` module** (`src/dnn.rs`, ~555 lines, gated `dnn = ["dep:oxionnx"]`) —
  full cv2.dnn API surface: `Net` (wraps `oxionnx::Session`),
  `read_net_from_onnx(path)`, `Net::forward(blob)`/`forward_named(blob, output)`,
  `blob_from_image(image, scale, size, mean, swap_rb, crop)` with planar (CHW)
  output, `nms_boxes(boxes, scores, score_thresh, nms_thresh)` with greedy
  score-descending IoU filtering. Mat representation for blobs is `CV_32FC3`
  planar layout (shape recoverable as `[1, channels, rows, cols]`), documented
  prominently in the module-level doc comment. `Cv2Error::Dnn(String)` variant
  added for module errors. 15 tests in `tests/dnn_smoke.rs` covering blob
  preprocessing, R/B swap, NMS IoU correctness, NMS empty/no-overlap edge
  cases, grayscale promotion, missing-model error path. The real-model
  forward test is `#[ignore]` and gated on `OXIMEDIA_TEST_ONNX_MODEL`.
- **`BFMatcher` struct** (`src/features.rs`) — cv2-idiomatic descriptor matcher
  wrapping `bf_match_hamming`. `BFMatcher::new(NORM_HAMMING)` constructor
  validates the norm type; `.with_cross_check(true)` enables mutual-best
  filtering; `.match_descriptors(query, train)` returns `Vec<DMatch>` (cv2's
  `match` is reserved in Rust); `.knn_match(query, train, k)` returns
  `Vec<Vec<DMatch>>` for Lowe's ratio-test workflow with a bounded
  `BinaryHeap`-of-size-k for O(m·n·log k) cost.
- **Mask support in `Orb::detect_and_compute`** (`src/features.rs`) — mask
  parameter is now respected: keypoints whose centre falls in a zero mask pixel
  are filtered out *before* the `num_features` truncation (matching cv2's
  filter-first semantics). Validates mask shape (must match image WxH) and
  dtype (must be `CV_8UC1`); errors via `Cv2Error::SizeMismatch` /
  `Cv2Error::UnsupportedDtype`.
- **`NORM_HAMMING = 6` and `NORM_HAMMING2 = 7` constants** (`src/constants.rs`)
  in the `pub mod norm_type` namespace; auto-picked up by `LIST_CONSTANTS`
  reflection.

#### Validated
- `cargo nextest run -p oximedia-compat-cv2 --features dnn` — 175 pass, 1
  skipped (the `#[ignore]` real-model test).
- `cargo nextest run -p oximedia-compat-cv2 --all-features` — 175 pass.
- `cargo clippy -p oximedia-compat-cv2 --features dnn --all-targets -- -D warnings`
  — clean.
- `cargo clippy -p oximedia-compat-cv2 --no-default-features --all-targets -- -D warnings`
  — clean.

#### Notes
- `features.rs` now 1496 lines (up from 1227); well under the 2000-line refactor
  ceiling.
- The `oximedia-cv2` binary subcommand wiring (`dnn-forward`, `orb-detect`,
  plus 8 already-impl'd op exposures) is recorded under
  `oximedia-cli/CHANGELOG.md`.
