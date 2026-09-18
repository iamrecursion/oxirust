# oximedia-calibrate TODO

## Current Status
- 29 modules covering camera calibration, display calibration, color matching, ICC profiles, LUT generation, white balance, color temperature, gamut mapping, chromatic adaptation
- Sub-directories: `camera/`, `chromatic/`, `display/`, `gamut/`, `icc/`, `lut/`, `match/`, `temp/`, `white/`
- Supports ColorChecker Classic 24, Passport, Datacolor SpyderCheckr, and custom targets
- Dependencies on `ndarray` (policy violation pending)

## Enhancements
- [x] Replace `ndarray` dependency with SciRS2-Core per SCIRS2 policy (ndarray was not present; pure Rust confirmed)
- [x] Add CAT16 chromatic adaptation transform (from CIECAM16) in `chromatic` module (`chromatic/cat16.rs` + `cat16_adapt` free fn)
- [x] Extend `icc` module with ICC v5 (iccMAX) profile support (implemented 2026-05-15: `IccProfileVersion::V5IccMax`, `IccMaxProfile`, `SpectralData`, `IccMaxTag`, `parse_iccmax`; tests `test_iccmax_version_parsed`, `test_iccmax_float_tag_roundtrip`)
- [x] Add DCI-P3 and Rec.2020 working color space support in `color_space` (`ColorSpace` enum with `DciP3`, `Rec2020`, `DciP3D65` variants)
- [x] Implement CIEDE2000 delta-E formula in `delta_e` (`delta_e_2000` + `test_delta_e2000_identical`)
- [x] Extend `camera` profiling with dual-illuminant calibration (D50 + A) for DNG color matrices (implemented 2026-05-15: `DualIlluminantCalibration`, `DngColorProfile::with_dual_illuminant`, `interpolate_matrix_by_cct`; tests `test_dual_illuminant_d50_limit`, `test_dual_illuminant_illum_a_limit`, `test_dual_illuminant_midpoint`)
- [x] Add `display_verify` automated pass/fail report generation against target tolerances (verified 2026-05-16; src/display_verify.rs:114 passes_target fn, VerifyResult:69, DisplayTarget:17)
- [x] Improve `gamut_checker` with per-pixel gamut boundary visualization (implemented 2026-05-15: `GamutPixel`, `GamutVisualization`, `GamutChecker::visualize_gamut_boundary`; tests `test_gamut_visualization_in_gamut_zero`, `test_gamut_visualization_out_of_gamut_positive`)
- [x] Extend `lut` generation with tetrahedral interpolation for 3D LUTs (`trilinear_lookup_3d`, `tetrahedral_lookup_3d` free fns + `test_tetrahedral_identity_lut`)

## New Features
- [x] Add `spectral_reconstruction` module to recover spectral data from RGB camera measurements (verified 2026-05-16; src/spectral_reconstruction.rs:139 WienerSpectralReconstructor, reconstruct:212, 436 lines)
- [x] Implement `aces_calibration` module for ACES (Academy Color Encoding System) IDT generation (verified 2026-05-16; src/aces_calibration.rs:246 AcesIdt, AcesIdtGenerator:326, generate:344, 595 lines)
- [x] Add `hdr_calibration` module for PQ/HLG display calibration with peak luminance mapping (verified 2026-05-16; src/hdr_calibration.rs:210 HdrCalibrator, HdrCalibrationResult:171, 567 lines)
- [x] Implement `printer_calibration` module for CMYK soft-proofing workflows (verified 2026-05-16; src/printer_calibration.rs:349 PrinterCalibrator, PrinterCalibratorConfig:326, 815 lines)
- [x] Add `ambient_compensation` module adjusting display profile based on ambient light measurement (verified 2026-05-16; src/ambient_compensation.rs:161 AmbientCompensationEngine, AmbientCompensationConfig:105, 536 lines)
- [x] Implement `batch_calibrate` module for calibrating multiple cameras from a single shooting session (verified 2026-05-16; src/batch_calibrate.rs:162 BatchCalibrator, BatchCalibrationResult:118, 594 lines)
- [x] Add `calibration_schedule` module tracking when devices are due for re-calibration (verified 2026-05-16; src/calibration_schedule.rs:173 CalibrationSchedule, 472 lines)

## Performance
- [x] Use SIMD for 3x3 matrix multiplication in `chromatic` adaptation transforms (implemented 2026-05-15: `mat3x3_mul_vec3_simd` with AVX2/NEON/scalar dispatch; wired into `ChromaticAdaptation::apply_matrix`; tests `test_mat3x3_simd_matches_scalar`, `test_mat3x3_simd_identity`)
- [x] Parallelize ICC profile application across image tiles using rayon (done — par_chunks_exact_mut at src/icc/apply.rs:63)
- [x] Cache frequently-used illuminant-to-illuminant adaptation matrices in `chromatic` (done — ADAPT_MATRIX_CACHE + cached_transform_matrix at src/chromatic/adapt.rs:160-233)
- [ ] Optimize `patch_extract` with early-exit when sufficient patches detected
- [x] Pre-compute and cache LUT interpolation coefficients in `lut` module (implemented 2026-06-05: `Lut3dCached` + `TetraCoeff` in `lut/cached.rs`; precomputes per-cell base index and axis strides; 4 tests: matches-direct, identity-lut, out-of-range-clamps, perf-10k)

## Testing
- [x] Add test for `camera` profile generation against known ColorChecker reference values (Wave 30, 2026-06-08: real least-squares 3×3 color matrix via normal equations M=B·A⁻¹ + λ=1e-6 Tikhonov + closed-form 3×3 adjugate inverse replacing the identity stub in `CameraCalibrator::compute_color_matrix`; new public `fit_color_matrix`/`apply_color_matrix`; classic24 identity-recovery + planted-transform recovery <1e-6 + rank-deficient graceful error; 5 tests in tests/camera_color_matrix.rs + 6 in-file helper tests)
- [ ] Test ICC profile round-trip: generate from measurements, write, read back, verify TRCs match
- [x] Add delta-E verification test: calibrated output vs reference should be under 2.0 dE (Wave 30, 2026-06-08: new `src/camera/color_space.rs` sRGB→linear→XYZ(D65)→Lab pipeline; `calibration_delta_e_under_2` asserts mean ΔE2000<2.0 after fit on synthetic distorted ColorChecker, `calibration_beats_identity_baseline` asserts ΔE(fit)<ΔE(identity))
- [ ] Test `chromatic` Bradford adaptation against published ASTM E308 reference data
- [ ] Add `gamut_checker` test with known out-of-gamut colors verifying correct mapping
- [ ] Test `white_balance` presets produce expected RGB multiplier ratios

## Documentation
- [ ] Document end-to-end camera calibration workflow with code examples
- [ ] Add color science reference appendix explaining CIE XYZ, Lab, illuminants, observers
- [ ] Document supported ColorChecker targets with patch layout diagrams
