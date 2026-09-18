# oximedia-compat-ffmpeg TODO

## Current Status (Wave 12 — 2026-05-31)
- 26 source files; FFmpeg CLI argument compatibility layer
- `compat_ext.rs` (1,880 lines) split via splitrs into `compat_ext/` (types.rs, functions.rs, functions_2.rs) — all under 600 lines
- Key features: argument parsing, codec mapping (80+ mappings), filter graph parsing, stream specifiers, diagnostics, translator
- Modules: arg_parser, argument_builder, batch_mode, codec_map, codec_mapping, compat_ext/, concat_compat, diagnostics, encoder_options, ffprobe, ffprobe_output, filter_complex, filter_graph, filter_lex, filter_shorthand, hwaccel_compat, lavfi_compat, metadata_compat, pass, preset_translator, real_world_tests, seek, stream_spec, translator, two_pass

## Enhancements
- [x] Extend `codec_map.rs` to cover all common FFmpeg codec aliases (e.g., h264_nvenc, hevc_amf -> av1 equivalents)
- [x] Add support for `-filter_complex` multi-input/output filter graph parsing in `filter_lex.rs` (verified 2026-05-16; src/filter_lex.rs:100 semicolon-split multi-chain parsing, test:850)
- [x] Improve `diagnostics.rs` with suggestion-based error messages ("did you mean..." for mistyped codecs) (verified 2026-05-16; src/diagnostics.rs:335 fn suggest_codec, fuzzy match:243)
- [x] Extend `stream_spec.rs` to handle complex stream specifiers like `0:v:0`, `0:a:#0x1100` (verified 2026-05-16; src/stream_spec.rs:93 StreamSpec, Pid variant:58, program_id:101)
- [x] Add `-map` flag with negative mapping support (e.g., `-map 0 -map -0:s`) in `arg_parser.rs` (verified 2026-05-16; src/arg_parser.rs:104 negative field, test_negative_map:1205)
- [x] Implement `-ss` / `-to` / `-t` seeking/duration options in `arg_parser.rs` (verified 2026-05-16; src/arg_parser.rs:297 -ss parsing, :307 -t, :624 -to)
- [x] Add `-preset` / `-tune` / `-profile` translation in `codec_mapping.rs` (verified 2026-05-16; src/encoder_options.rs:6 EncoderQualityPreset, EncoderQualityOptions:105)
- [x] APV aliases added to codec_map.rs + codec_mapping.rs — Slice A of /ultra Wave 3 (2026-04-17) DONE
- [x] FFmpeg compat Wave 3: filter_complex, -map stream_spec, -ss/-to/-t, ffprobe output — Slice G of /ultra Wave 3 (2026-04-17)

## Wave 4 Progress (2026-04-18)
- [x] codec-map-cache: OnceLock singleton for codec_map + codec_mapping registries — Wave 4 Slice E
- [x] encoder-quality-args: -preset/-tune/-profile:v parsing → EncoderQualityOptions — Wave 4 Slice E
- [x] filter-shorthand: -vf/-af parsing → single-chain FilterGraph (reuses filter_complex AST) — Wave 4 Slice E
- [x] two-pass: -pass 1/-pass 2 → PassPhase::First/Second with JSON stats file — Wave 4 Slice E

## New Features
- [x] Implement `ffprobe`-compatible output mode (JSON/XML/CSV format info) — ffprobe.rs has ProbeOutput, ProbeStream, ProbeFormat; Wave 12 added `translate_ffprobe_args` + `FfprobeQuery` (2026-05-31)
- [x] Add `-vf` / `-af` shorthand filter chain parsing alongside `-filter_complex` (verified 2026-05-16; Wave 4 Slice E)
- [x] Implement batch mode translation for converting multiple files in one invocation — `translate_batch_command` + `BatchJob` + `BatchInputSpec` + `BatchOutputSpec` added to batch_mode.rs (2026-05-31)
- [x] Add `-movflags +faststart` and similar muxer option translation in `translator.rs` — `MuxerAction::FastStart` implemented; rw_25 test passes (verified 2026-05-31)
- [x] Implement `-hwaccel` option translation to OxiMedia GPU pipeline flags (verified 2026-06-06: `translate_hwaccel` implemented in hwaccel_compat.rs:297)
- [x] Add support for concat protocol (`concat:file1|file2`) and concat demuxer syntax (verified 2026-06-06: `parse_concat_protocol` concat_compat.rs:252 + `parse_concat_demuxer` :298)
- [x] Implement two-pass encoding translation (`-pass 1` / `-pass 2`) in `translator.rs` (verified 2026-05-16; Wave 4 Slice E two-pass implementation)
- [x] Add `-metadata` tag translation for title, artist, comment fields — `translate_metadata_args(&[&str]) -> Vec<(String, String)>` added to metadata_compat.rs (2026-05-31)

## Performance
- [x] Cache parsed codec map in `codec_map.rs` to avoid repeated HashMap construction (verified 2026-05-16; Wave 4 Slice E OnceLock singleton)
- [x] Optimize `filter_lex.rs` parser with zero-copy string slicing — `parse_filter_graph_zerocopy` + `FilterToken<'a>` implemented in filter_lex.rs (2026-05-31)
- [ ] Pre-compile regex patterns in `arg_parser.rs` for repeated argument parsing

## Testing
- [x] Add test suite covering 50+ real-world FFmpeg command lines — real_world_tests.rs has 67 tests (rw_01..rw_67) as of Wave 12 (2026-05-31)
- [x] Test `filter_lex.rs` with complex filter graphs (split, overlay, amix chains)
- [x] Add round-trip test: build arguments with `argument_builder.rs`, parse back, verify equivalence (compat_ext tests cover builder)
- [ ] Test diagnostic output formatting matches FFmpeg-style warning/error format
- [ ] Add fuzz testing for `arg_parser.rs` with random argument combinations
- [x] Test codec mapping completeness — rw_61/rw_62/rw_63 verify patent-free codecs and substitutions (2026-05-31)

## Wave 12 Deliverables (2026-05-31)
- [x] `compat_ext.rs` (1,880L) split by splitrs → `compat_ext/` (types.rs 573L, functions.rs 439L, functions_2.rs 519L) + mod.rs; all < 600L
- [x] `translate_ffprobe_args(&[&str]) -> Result<FfprobeQuery, FfprobeArgError>` — parses ffprobe CLI flags (-v, -print_format/-of, -show_format, -show_streams, -show_packets, -select_streams, -i)
- [x] `FfprobeQuery` struct (input_path, print_format, show_format, show_streams, show_packets, verbosity, select_streams)
- [x] `FfprobeArgError` enum (MissingValue, UnknownPrintFormat)
- [x] `translate_batch_command(&[&str]) -> Result<Vec<BatchJob>, BatchError>` — multi-input multi-output parsing
- [x] `BatchJob { inputs: Vec<BatchInputSpec>, outputs: Vec<BatchOutputSpec> }` — structured batch result
- [x] `BatchInputSpec { path, index }` + `BatchOutputSpec { path, video_codec, audio_codec, video_bitrate, audio_bitrate, crf, format, metadata }`
- [x] `translate_metadata_args(&[&str]) -> Vec<(String, String)>` — extracts `-metadata key=value` pairs from raw arg slice
- [x] `parse_filter_graph_zerocopy(input: &str) -> Vec<FilterToken<'_>>` — zero-copy lexer (Label, FilterName, Arg, ChainSep)
- [x] `FilterToken<'a>` enum with borrowed `&'a str` slices (no heap alloc)
- [x] rw_61..rw_67 tests: codec completeness (patent-free video, audio, substitution), zero-copy lexer, batch_command, ffprobe, metadata integration
- [x] All 749 tests pass, 0 clippy warnings

## Documentation
- [ ] Add FFmpeg-to-OxiMedia command translation examples in crate docs
- [ ] Document supported and unsupported FFmpeg options with migration notes
- [ ] Add filter graph syntax reference showing supported filter names
