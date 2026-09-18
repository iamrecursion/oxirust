# oximedia-hdr TODO

## Current Status
- 8 modules covering HDR video processing
- Transfer functions: PQ (ST 2084), HLG, SDR gamma in `transfer_function.rs`
- Gamut handling: Rec.2020, color gamut conversion matrices in `gamut.rs`
- Tone mapping: configurable operators in `tone_mapping.rs`
- HDR10/HDR10+ metadata: CLL, mastering display parsing/encoding in `color_volume.rs`
- Dolby Vision: profile detection, RPU data, cross-version metadata in `dolby_vision_profile.rs`
- Dynamic metadata: per-frame HDR10+ dynamic metadata in `dynamic_metadata.rs`
- HLG advanced: HLG-to-HDR10 and HLG-to-SDR conversion in `hlg_advanced.rs`
- Dependencies: thiserror, serde, log

## Enhancements
- [x] Add scene-referred tone mapping with per-frame luminance analysis in `tone_mapping.rs` (verified 2026-05-11; tone_mapping.rs:~535-650, SceneReferredToneMapper + FrameLuminanceAnalysis)
- [x] Extend `gamut.rs` with soft-clip gamut mapping (perceptual desaturation instead of hard clamp) (verified 2026-05-11; gamut.rs:~304-356, convert_soft_clip + soft_clip_gamut_map; also soft_clip_gamut.rs:~1-490)
- [x] Add BT.2446 Method A and Method C tone mapping operators to `tone_mapping.rs` (verified 2026-05-11; tone_mapping.rs:~30-200, Bt2446MethodA/Bt2446MethodC enum arms, bt2446_method_a_inverse, bt2446_method_c_luminance, BT2446MethodAToneMapper, BT2446MethodCToneMapper)
- [x] Implement Dolby Vision RPU generation (not just parsing) in `dolby_vision_profile.rs` (verified 2026-05-11; dolby_vision_profile.rs:~288-490, RpuGenerationConfig + generate_rpu_nal)
- [x] Extend `hlg_advanced.rs` with system gamma adjustment based on display peak luminance (verified 2026-05-11; hlg_advanced.rs:~240-290, hlg_adapted_system_gamma + hlg_system_gamma per BT.2100-2 Annex 1)
- [x] Add MaxRGB and percentile luminance statistics computation for `color_volume.rs` CLL auto-detect (verified 2026-05-11; luminance_stats.rs:~1-250, FrameLuminanceStats with max_rgb_nits, percentile_95_nits, percentile_99_nits; also color_volume.rs:~204-245 MaxRgbAnalyzer)
- [x] Implement inverse tone mapping (SDR-to-HDR upconversion) in `tone_mapping.rs` (verified 2026-05-11; tone_mapping.rs:~666-730, InverseToneMapper + InverseToneMappingOperator; also sdr_to_hdr.rs:~1-80, SdrToHdrConverter; also tone_mapping_ext.rs SdrToHdrMapper)
- [x] Add validation for HDR10+ dynamic metadata JSON schema in `dynamic_metadata.rs` (verified 2026-05-11; dynamic_metadata_validator.rs:~1-370, DynMetaValidationReport + validate_dynamic_metadata + validate_dynamic_metadata_fn)

## New Features
- [x] Add a `hdr_histogram.rs` module for luminance histogram analysis of HDR frames (verified 2026-05-11; hdr_histogram.rs:~1-810, HdrHistogram, LuminanceHistogram, HdrHistogramAnalyzer with PQ/linear-nits inputs)
- [x] Implement a `display_model.rs` module for target display characterization (peak nits, black level, gamut) (verified 2026-05-11; display_model.rs:~1-200, DisplayModel with peak_luminance_nits, black_level_nits, tone_map_to, contrast_ratio)
- [x] Add a `color_volume_transform.rs` module for ICtCp color space conversions (BT.2100) (verified 2026-05-11; color_volume_transform.rs:~1-240, rgb_to_ictcp, ictcp_to_rgb, ICtCpFrame, delta_ictcp per ITU-R BT.2100-2 Note 5)
- [x] Implement a `vivid_hdr.rs` module for Vivid HDR (SL-HDR1/SL-HDR2) metadata support (verified 2026-05-11; vivid_hdr.rs:~1-400, VividHdrMetadata, VividEotf, VividToneMapParams, parse/serialize 34-byte SL-HDR1 SEI)
- [x] Add a `hdr_scopes.rs` module for waveform/vectorscope analysis of HDR content (verified 2026-05-11; hdr_scopes.rs:~1-530, WaveformMonitor, Vectorscope, LumaStatistics with horizontal/vertical axes)
- [x] Implement a `cuva_metadata.rs` module for CUVA HDR Vivid dynamic metadata (Chinese HDR standard) (verified 2026-05-11; cuva_metadata.rs:~1-530, CuvaMetadata, CuvaToneMapParams, CuvaWhitepoint, parse/serialize SEI per T/UHD 004)
- [x] Add a `st2094.rs` module for SMPTE ST 2094 series dynamic metadata (beyond HDR10+) (verified 2026-05-11; st2094.rs:~1-530, St2094_10Metadata, St2094_40Metadata, ExtBlock, ExtBlockType with round-trip serialization)

## Performance
- [x] Add SIMD-optimized PQ EOTF/OETF computation in `transfer_function.rs` (batch f32 processing) (verified 2026-05-11; pq_simd.rs:~1-200, PqSimdProcessor with 8-wide unrolled EOTF/OETF + runtime CPU tier detection; pq_eotf_batch/pq_oetf_batch in transfer_function.rs)
- [x] Implement LUT-based fast path for PQ and HLG transfer functions (precomputed 1D tables) (verified 2026-05-11; transfer_function.rs:~245-380, PqEotfLut, PqOetfLut, HlgEotfLut with configurable 1D table sizes)
- [x] Add parallel per-row tone mapping in `tone_mapping.rs` using rayon (verified 2026-05-11; tone_mapping.rs:~835-900, map_frame_parallel + tone_map_frame_rayon using into_par_iter)
- [x] Cache gamut conversion matrices in `gamut.rs` to avoid recomputation per pixel (implemented 2026-05-16: process-wide OnceLock<RwLock<GamutCacheMap>> in gamut.rs; ColorGamut derives Copy+Eq+Hash; compute_matrix() extracted; two-phase read/write locking; cache_len() test accessor; test_gamut_cache_deduplicates + test_gamut_cache_thread_hammer (8 threads, 10 pairs) pass; 506 tests green, 0 clippy warnings)

## Testing
- [x] Add reference value tests for `pq_eotf`/`pq_oetf` against ITU-R BT.2100 Table values (verified 2026-05-11; transfer_function.rs:~544-575, test_pq_oetf_bt2100_ref_10_nits, test_pq_oetf_bt2100_ref_1000_nits, test_pq_eotf_bt2100_ref_1000_nits)
- [x] Test `hlg_oetf`/`hlg_eotf` round-trip accuracy across the full luminance range (verified 2026-05-11; transfer_function.rs:~457-495, test_hlg_round_trip_low, test_hlg_round_trip_high, test_hlg_oetf_zero, test_hlg_eotf_zero, test_hlg_eotf_lut_accuracy)
- [x] Add tests for `dolby_vision_profile.rs` with real-world RPU bitstream samples (completed 2026-06-24: added 30 tests (tests 21-50) covering generate_rpu_nal/parse_rpu_nal_header/verify_rpu_crc — SYNTHETIC spec-structured byte vectors, not captured content; all round-trip, error, corruption, edge-case paths covered)
- [x] Implement DV RPU parser (BitReader + DoviRpuParser) as the inverse of DoviRpuBuilder (verified 2026-05-16; src/dovi_rpu.rs:DoviRpuParser — BitReader mirrors BitWriter MSB-first; DoviRpuParser::parse() reads all bit-fields in the same order as build_rpu_payload(); round-trip of bl_max_pq, target_max_pq, trim_slop, trim_offset, trim_power, frame_idx; 14 new tests: test_bitreader_*, test_parser_roundtrip_*, test_parser_rejects_*; 523 tests pass, 0 clippy warnings)
- [x] Test tone mapping operators against ITU-R BT.2446 reference implementations (verified 2026-05-11; tone_mapping.rs:~1093-1450, 30+ BT.2446 Method A/C tests including monotonic, C1 continuity, shadow/highlight, mid-grey, crosstalk)
- [x] Add edge case tests for `color_volume.rs` with zero/negative luminance values (verified 2026-05-11; color_volume.rs:~503-528, test_cll_zero_values (max_cll=0, max_fall=0) + test_luminance_max_roundtrip (min_luminance=0))

## Documentation
- [x] Add a flowchart showing the HDR-to-SDR conversion pipeline (metadata -> gamut map -> tone map -> transfer) (completed 2026-06-24: 5-stage ASCII art + mermaid flowchart added to lib.rs module-level doc comment; stages: metadata extraction → EOTF decode → gamut mapping → tone mapping → SDR OETF)
- [x] Document supported Dolby Vision profiles and their constraints (verified 2026-05-11; dolby_vision_profile.rs:~1-42, module-level //! with Profile Overview table P4/P5/P7/P8/P9 and per-variant doc comments including EL/single-layer constraints)
- [x] Add reference links to SMPTE/ITU standards for each module (verified 2026-05-11; standard names cited inline: SMPTE ST 2084 in transfer_function.rs:~1,9,27,41; SMPTE ST 2086/ST 2094-40 in color_volume.rs:~3,14,80; ITU-R BT.2100-2 in color_volume_transform.rs:~1, hlg_advanced.rs:~244; BT.2446-1 in tone_mapping.rs:~418; T/UHD 004 in cuva_metadata.rs)
