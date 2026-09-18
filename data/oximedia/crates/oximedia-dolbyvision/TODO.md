# oximedia-dolbyvision TODO

## Current Status
- 32 source files covering RPU parsing, writing, tone mapping, metadata blocks, and profile management
- Metadata-only implementation respecting Dolby intellectual property
- Supports Profiles 5, 7, 8, 8.1, 8.4 with full level metadata (L1-L11)
- Features: RPU parse/write (NAL and bitstream), tone mapping, XML metadata, shot boundary detection
- Dependencies: oximedia-core, bitstream-io, bitflags, bytes; optional serde

## Enhancements
- [x] Add Level 4 (global dimming) and Level 7 (source display color volume) metadata support
- [x] Implement RPU data validation for all metadata levels in `validation.rs` (currently only checks header)
- [x] Add round-trip fidelity verification in `parser.rs` / `writer.rs` (parse -> write -> parse should be identical)
- [x] Enhance `profile_convert.rs` with Profile 8 -> Profile 8.4 (HDR10 to HLG) conversion
- [x] Add automatic profile detection from RPU header fields (verified 2026-05-16; src/auto_profile_detect.rs:137 fn detect)
- [x] Improve `scene_trim.rs` with scene-change detection heuristics from L1 metadata (verified 2026-05-16; src/scene_trim.rs:293 fn detect_scene_changes)
- [x] Add `cm_analysis.rs` content mapping analysis with histogram-based scene statistics (verified 2026-05-16; src/cm_analysis.rs:104 struct CmAnalyzer, histogram bins)
- [x] Implement RPU merging for combining metadata from multiple sources in `shot_metadata_ext.rs` (verified 2026-05-16; src/rpu_merge.rs:87 fn merge_rpus)
- [x] Add streaming RPU parser that processes NAL units incrementally without buffering full stream (verified 2026-05-16; src/streaming_parser.rs:53 StreamingRpuParser)

## New Features
- [x] Implement Dolby Vision to HDR10+ metadata conversion bridge (verified 2026-05-16; src/dv_hdr10plus_bridge.rs:113 struct DvToHdr10PlusBridge)
- [x] Add RPU metadata visualization/plotting utilities for debugging (verified 2026-05-16; src/rpu_visualize.rs:82 struct RpuPlotter)
- [x] Implement `ambient_metadata.rs` ambient light adaptation metadata generation (verified 2026-05-16; src/ambient_metadata.rs:17 struct AmbientLight)
- [x] Add batch RPU extraction from HEVC/AVC bitstreams (verified 2026-05-16; src/batch_rpu.rs:209 struct BatchRpuProcessor)
- [x] Implement RPU metadata timeline editing (insert, delete, retime RPU entries) (verified 2026-05-16; src/rpu_timeline.rs:137 struct RpuTimeline)
- [x] Add Profile 10 (AV1-based Dolby Vision) metadata structure support (verified 2026-05-16; src/profile10.rs:34 struct Av1MetadataObuHeader)
- [x] Implement `dv_xml_metadata.rs` full Dolby Vision XML round-trip (parse + generate) (verified 2026-05-16; src/dv_xml_metadata.rs:849 lines)
- [x] Add RPU statistics reporting (min/max/avg luminance per scene) (verified 2026-05-16; src/rpu_stats.rs:118 struct SequenceLuminanceReport)

## Performance
- [x] Optimize `tonemap.rs` PQ/HLG transfer functions with lookup tables (verified 2026-05-16; src/tonemap.rs:41 pq_to_linear, ReshapingLut:236, ColorVolumeLut:299 3D-LUT trilinear)
- [x] Add SIMD-accelerated `ipt_pq.rs` color space conversion (done — ipt_pq_batch_simd at src/ipt_pq_simd.rs:71, registered lib.rs:38)
- [x] Cache parsed RPU structures in `parser.rs` to avoid re-parsing identical NAL units (done — RPU_CACHE + parse_nal_unit_cached at src/parser.rs:11-65)
- [x] Optimize `mapping_curve.rs` polynomial evaluation with Horner's method
- [x] Pre-compute `BilateralGrid` and `ColorVolumeLut` for repeated tone mapping operations (verified 2026-05-16; src/tonemap.rs:667 struct BilateralGrid, :299 ColorVolumeLut with precomputed 3D LUT)

## Testing
- [ ] Add conformance tests with known-good RPU bitstreams from reference tools
- [ ] Test all profile conversions in `profile_convert.rs` with real-world metadata samples
- [x] Add fuzz testing for `parser.rs` to verify robustness against malformed RPU data
- [ ] Test `tone_mapping.rs` output against reference DV display management pipeline
- [x] Add round-trip tests for all Level metadata types (L1 through L11)
- [ ] Test `xml_metadata.rs` parsing against Dolby Vision XML specification samples

## Documentation
- [ ] Document the RPU binary format structure in `parser.rs` and `writer.rs`
- [ ] Add metadata level reference table documenting each level's purpose and fields
- [ ] Document tone mapping pipeline flow from RPU metadata to display output

## 0.1.8 Wave 6 — 2026-05-29
- [x] Register 25 orphan modules in lib.rs + dv_xml_export cfix (verified 2026-05-29; 25 orphans wired, 24 smoke tests, 0 warnings)
