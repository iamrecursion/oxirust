# oximedia-lut TODO

## Current Status
- Core modules: lut1d, lut3d, interpolation, identity_lut, lut_resample
- LUT ops: lut_chain, lut_combine, lut_analysis, lut_validate, lut_stats, lut_dither, lut_fingerprint
- LUT I/O: formats (cube), lut_io, cube_writer, export, lut_gradient, lut_metadata, lut_provenance, lut_version
- Color spaces: colorspace (Rec.709, Rec.2020, DCI-P3, Adobe RGB, sRGB, ProPhoto, ACES AP0/AP1), matrix operations
- Color science: chromatic adaptation (Bradford, Von Kries), temperature (Kelvin to RGB), gamut mapping
- HDR: hdr_lut, hdr_metadata, hdr_pipeline (Reinhard, ACES filmic, Drago, Hejl), tonemap
- Advanced: aces pipeline, baking, builder, color_cube, domain_clamp, gamut_compress_lut

## Enhancements
- [x] Add LUT size validation and automatic resampling when chaining LUTs of different sizes in `lut_chain.rs`
- [x] Implement `lut_validate.rs` monotonicity checks for 1D LUTs and smoothness checks for 3D LUTs
- [x] Add `lut_analysis.rs` gamut coverage analysis (percentage of target gamut covered by LUT transform)
- [x] Implement LUT inversion (analytical for 1D, iterative for 3D) in `lut_combine.rs` (verified 2026-05-16; src/invert.rs:246 lines)
- [x] Extend `hdr_pipeline.rs` with ACES RRT v1.2 reference rendering transform (verified 2026-05-16; src/aces.rs aces_rrt fn used in tests at line 300)

## New Features
- [x] Add a `clf.rs` module for Common LUT Format (CLF/DLP) read/write (Academy/ASC standard) (verified 2026-05-16; src/clf.rs:1141 lines)
- [x] Implement a `creative_grade.rs` module with named film emulation presets (Kodak Vision3, Fuji Eterna) (verified 2026-05-16; src/creative_grade.rs:929 lines)
- [x] Add a `lut_blend.rs` module for blending between two LUTs with a mix factor (crossfade grading) (verified 2026-05-16; src/lut_blend.rs:583 lines)
- [x] Implement a `display_calibration.rs` module for generating display calibration LUTs (verified 2026-05-16; src/display_calibration.rs:637 lines)
- [x] Add a `lut_compress.rs` module for lossy LUT compression (verified 2026-05-16; src/lut_compress.rs:632 lines)

## Testing
- [x] Add round-trip tests for all LUT formats: .cube -> parse -> write -> parse -> compare (Wave 28 Slice 2, 2026-06-06; tests/cube_roundtrip.rs — 3D string round-trip via Lut3d::load_cube + export::export_3d_lut_cube/parse_cube_3d incl. B-major flatten, channel-swap, corner exactness, 8^3 raw; 1D via export_1d_lut_cube/parse_cube_1d; file-backed Lut1d::to_file/from_file)
- [x] Test `lut_chain.rs` with long chains (>10 LUTs) for numerical stability (Wave 28 Slice 2, 2026-06-06; tests/chain_stability.rs — inline seeded LCG, 1000 samples each: 15x identity <1e-4, 8x R<->B self-inverse swap <2e-4, 8x gamma2/gamma0.5 pair (16 stages) bounded <0.30 stability check)

## 0.1.9 Wave 29 — 2026-06-06 (Slice 1)
- [x] Polymorphic `.cube` loader added (formats/cube.rs: `CubeLut` enum + `parse_cube_any` + `CubeLut::from_file`/`is_1d`/`is_3d`/`as_1d`/`as_3d`; re-exported from `formats::` and crate root). Header-driven dispatch: `LUT_1D_SIZE` -> `CubeLut::Lut1d(Lut1d::from_file)`, `LUT_3D_SIZE` -> `CubeLut::Lut3d(Lut3d::from_file)`, neither -> `LutError::UnsupportedFormat`. Closes the prior LATENT GAP (Wave 28 Slice 2): `Lut3d::from_file` cannot load a 1D `.cube` — `parse_cube_any` now accepts both WITHOUT changing the existing `parse_cube_file -> Lut3d` back-compat contract. Tests: tests/cube_polymorphic.rs (1D/3D variant + direct-loader oracles, headerless rejection, `CubeLut::from_file` dispatch). 922/922 crate tests pass, 0 clippy warnings.

## 0.1.9 Wave 28 — 2026-06-06 (Slice 2)
- [x] `.cube` round-trip + chain-stability test suites added (tests/cube_roundtrip.rs, tests/chain_stability.rs); 918/918 crate tests pass, 0 clippy warnings
- [x] 1D-asymmetry bug investigated — NOT reproducible: `Lut1d::to_file`/`from_file` (lut1d.rs:405/323) are a self-contained `LUT_1D_SIZE` writer/reader pair and do NOT route through `formats/cube.rs::parse_cube_file` (the Lut3d reader, which correctly type-rejects `LUT_1D_SIZE` at cube.rs:54). No root fix fabricated; `lut1d_to_file_from_file_roundtrip` pins the 1D path is sound. LATENT GAP noted: `Lut3d::from_file` cannot load a 1D `.cube` (type-correct rejection, no Lut1d fallback) — future enhancement if a polymorphic loader is wanted.

## 0.1.8 Wave 6 — 2026-05-29
- [x] Register 22 orphan modules in lib.rs + hald_clut cfix (Slice D, 2026-05-29)
