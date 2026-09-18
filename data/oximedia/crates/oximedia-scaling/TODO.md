# oximedia-scaling TODO

## Current Status
- 30 modules covering bilinear, bicubic, Lanczos filtering, EWA resampling, content-aware scaling, adaptive scaling, aspect ratio preservation, batch scaling, chroma scaling, crop/pad, deinterlace, field scaling, half-pixel interpolation, perceptual sharpening, quality metrics, resolution ladders, ROI scaling, super resolution, thumbnails, tile-based processing
- Core types: ScalingParams, ScalingMode (Bilinear/Bicubic/Lanczos), AspectRatioMode, VideoScaler
- Dependencies: oximedia-core, rayon, serde, thiserror

## Enhancements
- [x] Add `ScalingMode::NearestNeighbor` for pixel-art and retro content scaling without interpolation (verified 2026-05-16; src/nearest_neighbor.rs:19 NearestNeighborConfig, NearestNeighborScaler:66)
- [x] Extend `VideoScaler::calculate_dimensions` to handle non-square pixel aspect ratios (PAR correction) (verified 2026-05-16; src/lib.rs:446 test_calculate_dimensions_ntsc_par_correction)
- [x] Add configurable Lanczos window size (2/3/4/5 taps) in `lanczos` module (verified 2026-05-16; src/lanczos.rs:58 LanczosWindowSize enum Tap2/Tap3/Tap4/Tap5)
- [x] Implement `content_aware_scale` seam carving with forward energy for better seam selection (verified 2026-05-16; src/seam_carve.rs:59 ForwardEnergy variant, forward_energy fn:337)
- [x] Extend `resolution_ladder` with per-title encoding optimization using VIF/SSIM target thresholds (verified 2026-05-16; src/resolution_ladder.rs:588 PerTitleLadder, PerTitleRung:563, compute:647)
- [x] Add `chroma_scale` support for 4:2:0, 4:2:2, and 4:4:4 chroma subsampling with proper phase alignment
- [x] Implement `deinterlace` with motion-adaptive algorithm (bob for motion areas, weave for static)
- [x] Add `batch_scale` progress reporting callback and cancellation token support

## New Features
- [x] Add `neural_upscale` module with lightweight inference for 2x/4x upscaling (patent-free architecture) (verified 2026-05-16; src/neural_upscale.rs:413 lines)
- [x] Implement `hdr_scaling` module handling PQ/HLG tone mapping during resolution changes (verified 2026-05-16; src/hdr_scaling.rs:806 lines, LanczosWindowSize integration)
- [x] Add `film_grain_scale` module that removes grain before scaling and re-synthesizes at target resolution (verified 2026-05-16; src/film_grain_scale.rs:423 lines)
- [x] Implement `temporal_scaling` module for frame rate conversion (frame blending, motion-compensated interpolation) (verified 2026-05-16; src/temporal_scaling.rs:910 lines)
- [x] Add `watermark_safe_scale` that preserves watermark positions during scaling operations (verified 2026-05-16; src/watermark_safe_scale.rs:277 lines)
- [x] Implement `multi_pass_scale` for extreme scaling ratios (e.g., 8K->480p) with intermediate steps (verified 2026-05-16; src/multi_pass_scale.rs:286 lines)
- [x] Add `scale_preview` for generating fast low-quality previews before committing to full-quality scale (verified 2026-05-16; src/scale_preview.rs:200 lines)
- [x] Implement `edge_directed_interpolation` module (NEDI-like) for improved diagonal edge rendering (verified 2026-05-16; src/edge_directed_interpolation.rs:271 lines)

## Performance
- [x] Add SIMD intrinsics for `bicubic` and `lanczos` horizontal/vertical filter passes (Wave 14, bicubic.rs + lanczos.rs via simd_interp.rs:255/312)
- [x] Implement rayon parallel row processing in `VideoScaler` for multi-core scaling (already implemented at parallel_scale.rs:137/173/212, Wave 14 verified)
- [x] Add tile-based processing in `tile` with cache-line-friendly memory access patterns (Wave 20 Slice B, 2026-06-02; src/tile.rs scale_tiled+scale_reference, bit-exact vs reference, rayon par_iter over tiles, 7 tests: downscale/upscale/RGBA/edge-tiles/zero-size/single-pixel-tile/output-size)
- [x] Optimize `ewa_resample` by precomputing filter weight tables for common scale factors (already implemented at ewa_resample.rs:329 FilterWeightTable, Wave 14 verified)
- [x] Add in-place scaling in `resampler` to avoid intermediate buffer allocations (Wave 14, resampler.rs resize_in_place)
- [x] Implement ring-buffer row caching in vertical filter passes to minimize memory footprint (already implemented at ring_buffer_cache.rs:42 RowRingBuffer, Wave 14 verified)

## Testing
- [ ] Add PSNR/SSIM quality regression tests comparing scaling output against reference implementations
- [x] Test `aspect_preserve` with edge cases: 1:1, ultrawide (32:9), portrait (9:16), anamorphic (2.39:1)
- [ ] Add roundtrip tests: scale down then up, verify quality metric degradation is within bounds
- [ ] Test `batch_scale` with mixed input resolutions and aspect ratios in a single batch
- [x] Add fuzz tests for `ScalingParams` with zero/negative/maximum dimensions

## Documentation
- [ ] Add quality comparison guide for Bilinear vs Bicubic vs Lanczos with visual examples
- [ ] Document the `resolution_ladder` algorithm and how `ContentDifficultyScore` affects rung selection
- [ ] Add guide for choosing between `content_aware_scale` and standard scaling for different content types
- [ ] Document the EWA resampling filter selection (Mitchell, Lanczos, Catmull-Rom) trade-offs
