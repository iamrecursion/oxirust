# oximedia-colormgmt TODO

## Current Status
- 60 source files; professional color management with ICC profiles, ACES workflows, HDR processing
- Color spaces: sRGB, Adobe RGB, ProPhoto RGB, Display P3, Rec.709, Rec.2020, DCI-P3
- ACES: full pipeline (IDT, RRT, ODT, LMT) with AP0 and AP1 primaries
- ICC: v2/v4 profile parsing and validation
- HDR: PQ/HLG transfer functions, tone mapping, gamut mapping
- Modules: colorspaces, aces/aces_config/aces_pipeline, icc/icc_profile, hdr (gamut, tonemapping, transfer, metadata), gamut/gamut_clip/gamut_mapping/gamut_ops, transforms (LUT, parametric), pipeline, delta_e (1976, 2000), ciecam02, color_appearance, spectral_data/spectral_locus, chromatic_adaptation, grading, curves

## Enhancements
- [x] Implement CIEDE2000 weighting parameters (k_L, k_C, k_H) as configurable in `delta_e` module
- [x] Add ICC v5 profile support in `icc/parser.rs` (iccMAX) (verified 2026-05-16; src/icc/parser.rs:259 IccVersion::V5, struct IccMaxSpectralTag:308)
- [x] Extend `hdr/tonemapping.rs` with Reinhard, Filmic, and ACES-fitted tone mapping operators (verified 2026-05-16; src/hdr/tonemapping.rs:ToneCurve)
  - **Implemented:** `ToneCurve` enum — `ReinhardSimple` (L/(1+L)), `ReinhardExtended { l_white }` (L*(1+L/lw²)/(1+L)), `FilmicHable` (Hable/Uncharted2 7-param curve, A=0.15 B=0.50 C=0.10 D=0.20 E=0.02 F=0.30 W=11.2), `AcesFitted` (Narkowicz rational, a=2.51 b=0.03 c=2.43 d=0.59 e=0.14).
  - **Tests:** 7 tests — Reinhard simple (1.0→0.5), extended (4.0→1.0 at l_white=4), Filmic (monotone on [0,W]), ACES (1.0→~0.83), monotone property test on [0,16].
- [x] Add Jzazbz and JzCzhz perceptual color space support alongside CIECAM02
- [x] Extend `gamut_mapping.rs` with cusp-based gamut mapping (ACES gamut mapping algorithm) (completed 2026-06-01: Wave 16 Slice C)
  - **Implemented:** `AcesCuspTable::acescg_default()` in `src/aces_gamut.rs` now computes AP1 cusps from real CIECAM02 boundary scan via `find_cusp_at_hue_for_primaries` (new public wrapper in `src/cusp_gamut.rs`); `OnceLock` cache for process-lifetime memoisation; CIECAM02 J/C normalised to [0,1].
  - **Tests added:** `test_acescg_cusps_are_computed_not_hardcoded` (span, plausibility, differs from old round numbers), `test_aces_gamut_compress_roundtrip_infield` (grey in-gamut unchanged), `test_aces_gamut_compress_oob_clamped` (negative-channel color compressed to ≥0).
- [x] Add multi-illuminant support in `chromatic_adaptation.rs` (beyond Bradford/Von Kries) (verified 2026-05-16; src/chromatic_adaptation.rs:296 compute_multi_illuminant_adaptation, MultiIlluminantAdapter:347)
- [x] Implement automatic color space detection from ICC profile in `display_profile.rs` (verified 2026-05-16; src/display_profile.rs:399 detect_color_space_from_primaries, detected_color_space:499)
- [x] Extend `color_quantize.rs` with octree and k-means quantization algorithms (verified 2026-06-01: src/color_quantize.rs:528 full k-means++ seeding + Lloyd iterations, quantize_octree, QuantizerAlgorithm enum, quantize_palette dispatch; 13 tests pass)

## New Features
- [x] Implement OCIO (OpenColorIO) configuration file parser for studio pipeline integration (verified 2026-05-16; src/ocio_parser.rs:1 OcioConfig, OcioColorSpace, transforms)
- [x] Add CTL (Color Transformation Language) interpreter for custom ACES transforms (verified 2026-05-16; src/ctl_interpreter.rs:1269 lines, ctl_lexer.rs, ctl_parser.rs)
- [x] Implement spectral rendering via `spectral_data.rs` for physically accurate color processing (verified 2026-05-16; src/spectral_data.rs:410, src/spectral_upsampling.rs:1514 lines)
- [x] Add display characterization and profiling utilities in `display_profile.rs` (verified 2026-05-16; src/display_profile.rs:729 lines display characterization)
- [x] Implement color grading LUT generation (3D LUT export) in `grading/` (verified 2026-05-16; src/grading/lut_export.rs:498 lines GradingLut3D)
- [x] Add ACES Output Transform 2.0 (candidate) implementation in `aces_pipeline.rs` (verified 2026-05-16; src/aces_output_transform.rs:1 ACES OT 2.0, AcesOutputTransform2:67)
- [x] Implement ICtCp color space for HDR content perceptual analysis
- [x] Add Oklab/Oklch perceptual color space support for uniform perceptual blending

## Performance
- [x] Add SIMD-accelerated 3x3 matrix multiply for bulk pixel transforms in `transforms/` (verified 2026-05-16; src/transforms/batch_matrix.rs:164 invert_matrix_3x3, src/transforms/matrix_bulk.rs:35 apply_matrix_bulk_f64)
- [x] Implement LUT interpolation with tetrahedral interpolation in `transforms/lut.rs` (verified 2026-05-16; src/transforms/lut.rs:91 tetrahedral_interpolate from math::interpolation)
- [x] Cache linearization/delinearization lookup tables in `transfer_function.rs` (implemented 2026-05-30: TransferFunctionLut 1024-entry forward/inverse, OnceLock<Mutex<HashMap>> global cache, linearize_via_lut/delinearize_via_lut on TransferFunction; 4 new tests)
- [x] Parallelize batch pixel conversion using rayon in `pipeline/mod.rs` (verified 2026-05-16; src/pipeline/mod.rs:172 rayon par_iter under #[cfg(feature = "rayon")])
- [x] Optimize `delta_e_2000` with precomputed trigonometric tables for batch color comparisons (implemented 2026-05-30: 360-entry OnceLock sin/cos table with linear interpolation in utils/delta_e.rs; delta_e_2000_f32 kernel with trig-table T/R_T lookups; delta_e_2000_batch rayon par_iter f32 API; color_diff_batch declared in lib.rs; 5 new tests, all 1068 tests pass, 0 clippy warnings)

## Testing
- [x] Add CIE standard illuminant round-trip tests (D50, D65, A, F2) in `white_point.rs` (verified 2026-06-01: src/illuminant_tests.rs:115+ covers all four illuminants)
- [ ] Test ICC profile parsing against real-world profiles (sRGB IEC61966, Adobe RGB, ProPhoto)
- [x] Add accuracy benchmarks: verify sRGB-to-Rec.2020 conversion achieves dE < 1.0
- [x] Test ACES pipeline end-to-end: IDT -> ACEScg -> RRT -> sRGB ODT
- [x] Add edge-case tests for out-of-gamut colors in `gamut_clip.rs` (verified 2026-06-01: src/gamut_clip.rs:351+ has negative/high OOG tests)
- [x] Test color_blindness.rs simulation accuracy against Brettel/Vienot references (completed 2026-06-01: Wave 16 Slice C)
  - **Implemented:** Removed `#![allow(dead_code)]` from `src/color_blindness.rs` (root cause: all items already used).
  - **Tests added:** `test_deuteranopia_preserves_luminance` (10 colours, luma drift < 30%), `test_protanopia_confusion_line` (pure-red R–G separation collapse < 0.70), `test_tritanopia_severity_interpolation` (half-severity ≈ midpoint within 0.01), `test_all_conditions_compile_and_run` (smoke test all three CVD types).

## Documentation
- [ ] Add color pipeline architecture diagram showing transform chain
- [ ] Document supported transfer functions with mathematical definitions
- [ ] Add ACES workflow tutorial with code examples in crate-level docs
