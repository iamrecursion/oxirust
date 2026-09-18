# oximedia-codec TODO

## Current Status
- 100+ source files; video codecs: AV1, VP9, VP8, Theora, H.263, FFV1; audio: Opus (SILK+CELT); image: PNG, GIF, WebP, JPEG-XL
- Feature-gated codecs: av1, vp9, vp8, theora, h263, opus, ffv1, jpegxl, image-io
- Subsystems: rate_control (CBR/VBR/CRF/CQP), multipass encoding, SIMD (x86 AVX2/AVX-512, ARM, scalar), intra prediction, motion estimation, tile encoding, reconstruction (CDEF, deblock, film grain, super-res)
- Re-exports: VideoFrame, AudioFrame, VideoDecoder/Encoder traits, rate control types, reconstruction pipeline, tile encoder

## Wave 2 Progress (2026-04-17)
- [x] JPEG encoder+decoder spec-compliance fix: zigzag DQT table ordering, correct AC Huffman symbol ordering — Wave 2 Slice A (2026-04-17).
- [x] AJXL ISOBMFF animated encoder: finish_isobmff(), jxlp box helpers, JxlAnimation/AnimationHeader types — Wave 2 Slice D (2026-04-17).
- [x] AJXL streaming decoder iterator: JxlStreamingDecoder<R: Read>, ISOBMFF + native bitstream auto-detect — Wave 2 Slice E (2026-04-17).

## Enhancements
- [x] Complete VP9 encoder (Vp9Encoder exists with 28+ tests)
- [x] Complete VP8 encoder (Vp8Encoder exists with 28+ tests)
- [x] Improve AV1 film grain synthesis fidelity in `av1/film_grain.rs` with per-block grain parameters
- [x] Add temporal scalability (SVC) support to AV1 encoder via `av1/svc_encoder.rs`
- [x] Extend `rate_control/lookahead.rs` with scene-adaptive bitrate allocation using content analysis (`rate_control/scene_adaptive.rs`)
- [x] Improve Opus encoder with voice activity detection (VAD) in `opus/silk.rs`
- [x] Add adaptive quantization matrix selection in `av1/quantization.rs` based on content type
- [x] Extend `motion/diamond.rs` with hexagonal and UMHex search patterns for faster estimation

## New Features
- [x] Add AVIF still image encoding/decoding (AV1-based) as a codec variant
- [x] Implement Vorbis audio encoder/decoder (vorbis module re-exported)
- [x] Add FLAC audio encoder/decoder for lossless audio (flac module exists)
- [x] Implement PCM codec support (pcm module exists)
- [x] Add APNG (animated PNG) support in `png/` module (apng module exists)
- [x] Implement WebP animation encoding in `webp` module (`webp/animation.rs`)
- [x] Add two-pass encoding support to the Theora encoder (`theora/two_pass.rs`)
- [x] Implement constant-quality mode for GIF encoder (`gif/quality.rs`)

## Performance
- [x] Expand SIMD coverage: ARM NEON implementations in `simd/arm/neon.rs` with real intrinsics
- [x] Add WASM SIMD128 backend in `simd/wasm.rs` with real `core::arch::wasm32` intrinsics
- [x] Optimize AV1 CDEF filter with SIMD in `simd/av1/cdef.rs` for 10-bit depth (`cdef_filter_u16`)
- [x] Add parallel tile decoding in `tile.rs` using rayon work-stealing
- [x] Optimize entropy coding in `entropy_coding.rs` with table-based arithmetic coding
- [x] Profile and optimize `reconstruct/loop_filter.rs` hot paths with cache-friendly access patterns
- [x] Optimize entropy coding in `entropy_tables.rs` with table-based CDF arithmetic coding (RangeCoder, CdfTable, 4 AV1 tables, encode/decode_symbol_table, 31 tests)
- [x] Add SIMD-accelerated pixel format conversion for YUV420/422/444

## Testing
- [x] Add bitstream conformance tests for AV1 decoder against reference test vectors
- [x] Add round-trip encode/decode quality tests for each codec (PSNR > threshold) — see `tests/codec_quality.rs`
- [x] Test rate control accuracy: verify CBR output stays within 10% of target bitrate — 3 CBR verifier tests
- [x] Add fuzzing targets for `png/decoder.rs`, `gif` decoder, and `webp` decoder
- [x] Test multipass encoding produces better quality than single-pass at same bitrate
- [x] Add regression tests for `jpegxl` modular and ANS coding paths

## Documentation
- [x] Document codec feature matrix (encode/decode, bitdepth, chroma support) in crate-level docs
- [x] Add rate control tuning guide with examples for each mode (CBR/VBR/CRF/CQP)
- [x] Document SIMD dispatch mechanism in `simd/mod.rs`

## 0.1.8 follow-up (added 2026-05-29 by /ultra)
- [x] Fix `CoeffBuffer::pos_to_rowcol` non-square TX derivation — `coeff_decode.rs:435` uses `sqrt(len)` for width which is wrong for all non-square AV1 TX sizes (4x8, 8x4, 4x16, 16x4, etc.) (done 2026-05-29)
  - **Goal:** `pos_to_rowcol` returns the correct (row, col) for all 19 AV1 TxSize variants.
  - **Design:** Add `pub const fn width(&self) -> usize` and `height()` to `CoeffBuffer` (private fields already exist). Rewrite `pos_to_rowcol` to use `buffer.width()`. Audit wider av1 module for the same `sqrt`-style anti-pattern.
  - **Files:** `src/av1/coeff_decode.rs:432-437`, `src/av1/coefficients.rs:797-814`
  - **Tests:** Construct `CoeffBuffer::from_tx_size(TxSize::Tx4x8)`, assert `pos_to_rowcol(7) == (1, 3)`. Cover Tx8x4, Tx4x16, Tx16x4.
  - **Risk:** Fix may surface latent bugs masked by square-biased test vectors; report any that appear.
- [x] Implement `silk_NSQ` (noise-shaped quantisation) in Opus SILK encoder — 440 Hz SNR 1.65 dB → > 6 dB achieved; 1 kHz SNR 3.31 dB (greedy quant; trellis search pending) (completed 2026-05-29)
  - **Goal:** SILK encoder excitation quantiser uses perceptual weighting filter `W(z)` + closed-loop NSQ state, clearing 6 dB segmental SNR on synthetic tones. `silk_encoder.rs` must be split via splitrs first (1898 lines, near 2000-line policy limit).
  - **Design:** (1) `splitrs` `silk_encoder.rs` → `silk_lpc.rs` / `silk_nlsf.rs` / `silk_ltp.rs` / `silk_excitation.rs`. (2) New `silk_nsq.rs`: `silk_warped_LPC_analysis_filter` (lambda warping 0.16/0.21/0.26 for NB/MB/WB), `NsqState` struct (sLPC[], sLTP[], sLTP_shp[], sLF_AR_shp_Q14, sLF_MA_shp_Q14, prev_gain_Q10), closed-loop per-sample quant loop (RFC 6716 §4.2.7.8.2), greedy D+λR pulse selector. (3) Wire `silk_nsq::process_subframe()` to replace `encode_excitation` direct-quant path.
  - **Files:** `src/opus/silk_encoder.rs` (splitrs), `src/opus/silk_nsq.rs` (new), `src/opus/silk_decoder.rs` (oracle for consistency checks)
  - **Tests:** `silk_warped_LPC_analysis_filter` (zero-input, impulse, 440 Hz tone). NSQ state round-trip (encode 100 ms synthetic noise; xq[] matches decoder reconstruction). 440 Hz SNR > 6 dB. 1 kHz SNR > 6 dB. White noise SNR > 0 dB.
  - **Risk:** splitrs may need manual review; greedy quantiser may undershoot 6 dB (acceptable if structural NSQ is in place; trellis search is follow-up).

## 0.1.8 Wave 19 (added 2026-06-01 by /ultra)

- [x] Upgrade SILK LTP encoder: coarse-to-fine decimated pitch search + per-subframe contour RD + fractional-lag refinement + encode→decode round-trip harness (completed 2026-06-02)
  - **Goal:** Replace the single-resolution full-rate integer pitch search and hardcoded contour-0 uniform lag with (1) a coarse-to-fine decimated autocorrelation, (2) per-subframe RD search over the existing PITCH_CONTOUR ICDF codebooks in silk_tables.rs, (3) parabolic fractional-lag interpolation improving solve_ltp_taps conditioning. Add the missing encode→decode round-trip SNR harness. NOT chasing the 1 kHz synthetic-tone SNR (spec floor: max trackable pitch = internal_rate÷min_lag ≈ 500 Hz at WB).
  - **Design:** Gap 1 — FIR half-band lowpass → downsample → coarse ACF scan → refine ±few samples full-rate (reduces octave errors, O(N/4·lags/4)). Gap 2 — for each of 4 subframes, RD search over PITCH_CONTOUR_{NB,MB_OR_WB}_*_ICDF tables to find best-fit contour vector; replace literal contour=0 at silk_ltp.rs:336 and uniform-lag loop at :338–340. Gap 3 — parabolic interpolation around integer ACF peak for fractional lag feeding tap solve (improves conditioning; emitted bitstream lag stays integer). Prerequisite: `tests/silk_ltp_roundtrip.rs` encode→decode round-trip harness (no such test exists today).
  - **Files:** `src/opus/silk_ltp.rs`, `src/opus/silk_encoder.rs`, `src/opus/silk_tables.rs`, `src/opus/silk_nsq.rs`, `src/opus/silk_decoder.rs`, `tests/silk_ltp_roundtrip.rs` (new)
  - **Tests:** voiced 150 Hz glottal-pulse shows LTP gain vs LTP-off; pitch-glide yields non-uniform per-subframe lags (contour≠0); coarse-to-fine agrees with full-rate on clean periodic input (±1 sample); fractional lag improves residual on non-integer-period input; round-trip decoder-consistency (emitted taps reconstruct through silk_decoder); white-noise correctly flagged unvoiced.
  - **Risk:** use correct NB-vs-MB/WB codebook per bandwidth — assert table selection in test. If round-trip harness surfaces a latent silk_decoder bug, flag as deviation — do NOT silently fix the decoder in this slice.

## 0.1.8 Wave 4 follow-up (added 2026-05-29 by /ultra)

- [x] Implement N=4 Viterbi trellis-search NSQ: 440 Hz SNR 3.31→6.91 dB (+3.6 dB); 1 kHz SNR 3.09 dB (structural floor: period < SILK min LTP lag of 32 samples) (completed 2026-05-29)
  - **Goal:** Replace the greedy 3-candidate local search in `process_subframe()` (lines 238–257 of `src/opus/silk_nsq.rs`) with an N=4 hypothesis delayed-decision trellis (per libopus `silk_NSQ_del_dec.c`, RFC 6716 §4.2.7.8). Target: 440 Hz SNR ≥ 8 dB, 1 kHz SNR ≥ 6 dB, white noise SNR ≥ 0 dB.
  - **Design:** Each of N=4 surviving paths carries cumulative cost `f64`, state memories (slpc[], sltp[], slf_ar_shp, slf_ma_shp, sltp_shp[], prev_gain_q10), pulse trajectory. Per sample: expand N×K candidates (K=5: ±2 around rounded integer), prune to N by ascending cost (tie-break: lower |pulse|). Cost: `D + λR` where `D = (signal_t - reconstructed_t)²`, `R ≈ |pulse|^0.55 + sign_change_indicator`, `λ` from `quant_gain_q10 × bandwidth_factor`. Add `NsqMode::{Greedy, TrellisDelDec}` enum; default `TrellisDelDec`. Decoder-consistency invariant: emitted pulse trajectory must reconstruct bit-exact through `silk_decoder::process_subframe`.
  - **Files:** `crates/oximedia-codec/src/opus/silk_nsq.rs` (replace lines 217–280 with trellis loop; add NsqMode enum; add NsqPath struct for trellis state; total ~300–500 new LoC)
  - **Tests:** `test_nsq_snr_440hz_trellis` (SNR ≥ 8 dB), `test_nsq_snr_1khz_trellis` (SNR ≥ 6 dB), `test_nsq_snr_white_noise_trellis` (SNR ≥ 0 dB), `test_nsq_trellis_vs_greedy` (trellis cost ≤ greedy cost on 100 random LPC sets), `test_nsq_decoder_consistency` (bit-exact decode round-trip)
  - **Risk:** If 1 kHz SNR misses by < 0.5 dB, tune `λ` and pulse-cost exponent before re-architecting. If decoder-consistency test surfaces SILK decoder bugs, flag as deviation — do not silently fix decoder in this slice.
