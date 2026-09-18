# oximedia-compat-cv2 TODO — version 0.2.0

## Milestone: v0.1.8 — Initial release [COMPLETE]

- [x] Mat type (CV_8UC1/3/4, CV_32FC1/3)
- [x] 134 OpenCV constants
- [x] imread / imwrite / imdecode / imencode
- [x] Color conversion (cvt_color) — Slice B (completed 2026-05-05)
- [x] Geometry (resize, flip, rotate) — Slice B (completed 2026-05-05)
- [x] Filters (gaussian_blur, bilateral, median) — Slice B (completed 2026-05-05)
- [x] Edge detection (canny, sobel) — Slice C (completed 2026-05-05)
- [x] Threshold (otsu, adaptive) — Slice C (completed 2026-05-05)
- [x] Morphology (erode, dilate, morph_ex) — Slice C (completed 2026-05-05)
- [x] Features (harris, fast, lk_flow) — Slice D (completed 2026-05-05)
- [x] Contours + Hough + Histogram + Template — Slice D (completed 2026-05-05)
- [x] Drawing + Arithmetic + Connected components — Slice E (completed 2026-05-05)

## Run 5 (planned 2026-05-07)

### Slice A — Farneback dense optical flow
- [x] Implement `calc_optical_flow_farneback` (replace stub in `optical_flow.rs:170-178`)
  - **Goal:** Full Farneback 2003 polynomial-expansion dense flow; output `CV_32FC2` Mat
  - **Files:** `src/optical_flow.rs`, `src/mat.rs` (add `CV_32FC2`), `tests/farneback_smoke.rs`
  - **Tests:** 6 (static frame, horizontal/vertical/diagonal translation, dtype+shape, size mismatch)

### Slice B — Perspective warp completeness
- [x] Implement `warp_perspective`, `get_perspective_transform`, `get_affine_transform`, `remap`
  - **Goal:** Complete geometric-transform surface alongside existing `warp_affine`
  - **Files:** `src/geometry.rs`, `tests/perspective_smoke.rs`
  - **Tests:** 7 (identity H, DLT round-trip, warpPerspective corners, affine identity, remap)

### Slice C — Shape analysis
- [x] Implement `moments`, `hu_moments`, `fit_ellipse`, `fit_line`, `min_area_rect`, `point_polygon_test`
  - **Goal:** Six cv2 shape-analysis functions; Fitzgibbon-Pilu-Fisher ellipse fit; rotating calipers
  - **Files:** `src/contour.rs`, `tests/shape_analysis.rs`
  - **Tests:** 9 (moments square, Hu invariants, ellipse fit, fit_line L2/L1, min_area_rect, pointPolygonTest)

### Slice D — Mat reductions + channel ops
- [x] Implement `count_non_zero`, `sum_elems`, `mean`, `mean_std_dev`, `norm`, `norm_diff`, `min_max_loc`, `split`, `merge`; add `Mat::clone_mat`, `Mat::convert_to`, `Mat::submat`, `Mat::reshape`
  - **Files:** `src/arithmetic.rs`, `src/mat.rs`, `tests/reductions_smoke.rs`
  - **Tests:** 8

### Slice E — Drawing completions + Hershey vector font
- [x] Implement `draw_contours`, `draw_keypoints`, `draw_matches`, `draw_marker`; add Hershey font (24 glyphs)
  - **Files:** `src/drawing.rs`, `src/hershey_font.rs` (new), `src/lib.rs`, `tests/drawing_completions.rs`
  - **Tests:** 7

### Slice F — BFMatcher NORM_L2 / NORM_L1 generalization
- [x] Generalize `BFMatcher::new` to accept `NORM_L1`, `NORM_L2`, `NORM_L2SQR`
  - **Files:** `src/features.rs`, `tests/bf_matcher_norms.rs`
  - **Tests:** 5

### Slice G — Bookkeeping
- [x] Update `CHANGELOG.md` and `README.md` for Run 5 deliverables
  - **Files:** `CHANGELOG.md`, `README.md`

---

### Proposed follow-ups

### Refinement 12 — Hershey font full glyph table (planned 2026-05-07)
Slice E shipped 24 glyphs (digits 0-9, uppercase A-N) sufficient to
demonstrate the vector-font renderer. Future pass adds the remaining 71
printable-ASCII glyphs and wires FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
variants.

### Refinement 13 — fitLine DIST_HUBER / DIST_FAIR variants (planned 2026-05-07)
Slice C shipped DIST_L2 and DIST_L1 (covers ~90% of real cv2 use). DIST_HUBER
(= 7) and DIST_FAIR (= 5 in our dist_type module) require IRLS with
M-estimator weight functions that differ from L1's 1/|r|. Future pass
implements both with their full weight tables.
