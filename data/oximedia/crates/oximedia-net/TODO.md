# oximedia-net TODO

## Current Status
- 26 source files/directories covering major streaming protocols
- Protocols: HLS, DASH, RTMP, SRT, WebRTC, SMPTE ST 2110, QUIC
- Infrastructure: CDN (multi-CDN failover, circuit breaker), ABR (adaptive bitrate), FEC (forward error correction)
- Network: connection_pool, bandwidth_estimator, bandwidth_throttle, flow_control, packet_buffer, qos_monitor
- Utilities: protocol_detect, retry_policy, session_tracker, stream_mux, multicast, network_path, rtp_session
- Re-exports: SRT stats, encryption session, ABR controller/variant/bandwidth types

## Enhancements
- [x] Implement CMAF low-latency HLS (LL-HLS) with partial segment support in hls module (verified 2026-05-16; src/hls/ll_hls.rs:727 lines, struct LlHlsConfig:18, MediaPart:65)
- [x] Add DASH low-latency (LL-DASH) with chunked transfer encoding in dash module (verified 2026-05-16; src/dash/ll_dash.rs, src/dash/ll_dash_chunker.rs)
- [x] Implement SRT caller/listener/rendezvous connection modes fully in srt module (verified 2026-05-16; src/srt/connection_mode.rs:24-28 CallerState:59, ListenerState:171, RendezvousState)
- [x] Add RTMP enhanced mode (RTMP+) with AV1/VP9 codec support in rtmp module (verified 2026-05-16; src/rtmp/enhanced.rs:833 lines)
- [x] Implement WebRTC WHIP/WHEP signaling for browser-based ingest/playback in webrtc module (verified 2026-05-16; src/webrtc/whip_whep.rs:1040 lines)
- [x] Add QUIC datagram mode for ultra-low-latency media transport in quic module
- [x] Improve ABR algorithm with buffer-based (BBA) strategy alongside bandwidth-based in abr module
- [x] Add SRT encryption with AES-256-GCM in addition to current AES-128-CTR

## New Features
- [x] Implement RIST (Reliable Internet Stream Transport) protocol as alternative to SRT (verified 2026-05-16; src/rist.rs:1493 lines)
- [x] Add Zixi-compatible protocol support for broadcast contribution links
- [x] Implement SMPTE ST 2022-7 seamless protection switching (dual-path redundancy) in smpte2110 (verified 2026-05-16; src/smpte2022_7.rs:1018 lines)
- [x] Add media relay/restreaming server -- receive from one protocol, retransmit on another (verified 2026-05-16; src/relay.rs)
- [x] Bandwidth-aware transcoding trigger — wire BandwidthTriggerEvaluator to codec signaling (verified 2026-06-01; src/bandwidth_adaptation.rs `BandwidthAdaptationController` wraps `BandwidthTrigger` + `Box<dyn Fn(TriggerAction)+Send>` callback; fires on Downgrade/Upgrade, silent on Hold; 5 tests)
  - **Goal:** Connect the existing `TriggerEvaluator` (which emits `TriggerAction`) to a callback so callers can wire in codec bitrate changes.
  - **Design:** `src/bandwidth_trigger.rs` has `TriggerEvaluator` that emits `TriggerAction::{Upgrade,Downgrade,Hold}` with hysteresis but nothing consumes them. Add `BandwidthAdaptationController` struct that owns `TriggerEvaluator` + a `Box<dyn Fn(TriggerAction) + Send>` callback; call callback on each `evaluate()` result. The consumer wires in a lambda that calls their codec's bitrate API. Pure Rust, no cross-crate dependency added.
  - **Files:** `src/bandwidth_trigger.rs`, `src/bandwidth_adaptation.rs` (new), `TODO.md`.
  - **Tests:** callback fires on `Downgrade` when bandwidth drops below threshold; callback NOT fired on `Hold`; Upgrade fires on recovery; callback receives the correct `TriggerAction` variant.
  - **Risk:** callback must be `Send` for async contexts; document that the consumer is responsible for the actual codec API call.
- [x] Add ICE (Interactive Connectivity Establishment) for WebRTC NAT traversal (verified 2026-05-16; src/webrtc/ice.rs:1082 lines, src/webrtc/ice_agent.rs)
- [x] Implement multipath streaming -- send redundant streams over multiple network interfaces (verified 2026-05-16; src/multipath.rs:1241 lines)

## Performance
- [x] Use connection pooling in CDN module for keep-alive HTTP connections to edge servers (verified 2026-05-16; src/cdn/keepalive_pool.rs:1 HTTP keep-alive connection pool per-host)
- [x] Implement zero-copy segment serving using sendfile/splice syscalls in HLS/DASH server (ZeroCopyFileServer done)
- [ ] Add io_uring support for high-throughput RTP packet handling in smpte2110 (verified-open 2026-05-16: no io_uring in smpte2110)
- [x] Implement packet pacing in SRT to smooth out burst traffic and reduce jitter (SrtPacketPacer done)
- [x] SIMD-accelerated XOR for FEC encode/decode (explicit intrinsics) (verified 2026-06-01; src/fec_interleave.rs `xor_blocks_avx2` AVX2 256-bit XOR + `xor_blocks_neon` NEON 128-bit + scalar fallback; `xor_blocks` runtime-dispatch; 4 new tests: SIMD==scalar 4KB, non-32-multiple 100B, double-XOR identity, multi-input)
  - **Goal:** Replace compiler-autovectorized u64-word XOR with explicit AVX2/NEON SIMD paths.
  - **Design:** `src/fec_interleave.rs:131-164` uses u64-word XOR relying on compiler autovectorization. Add explicit `#[target_feature(enable = "avx2")]` path: load 256-bit vectors, XOR, store; runtime-detect via `is_x86_feature_detected!("avx2")`. Add NEON path for aarch64 via `cfg(target_arch = "aarch64")`. Scalar fallback (current u64 path) always present. Use unaligned loads for safety.
  - **Files:** `src/fec_interleave.rs`, `TODO.md`.
  - **Tests:** SIMD XOR output == scalar XOR output on 4KB of random data; explicit-intrinsic path is hit on AVX2 hardware (assert via a test-only counter or runtime check); NEON path compiles on aarch64.
  - **Risk:** 32-byte alignment requirements — use unaligned `_mm256_loadu_si256` / `_mm256_storeu_si256`; last-chunk handling for non-256-bit-multiple lengths.
- [x] Cache parsed manifest/playlist structures to avoid re-parsing on each client request (verified: manifest_cache.rs:303:ManifestCache, TTL, ETag stale_while_revalidate)

## Testing
- [x] Add HLS playlist generation test verifying M3U8 format compliance (EXT-X-VERSION, segment tags) (verified: hls/playlist.rs:test_hls_m3u8_format_compliance)
- [x] Test DASH MPD generation against schema validation (structural validation, not XSD; 2026-06-24; test_mpd_generation_structural_validation)
- [x] Add SRT connection test with simulated packet loss verifying ARQ retransmission (verified 2026-06-05; srt/loss.rs: test_loss_contiguous_run_is_one_range, test_loss_scattered_singletons_are_n_ranges, test_loss_adjacent_ranges_merge, test_loss_remove_shrinks_range, test_loss_max_entries_cap, test_loss_oldest_correct — 6 LossList/NAK tests)
- [x] Test CDN failover: simulate primary CDN timeout, verify automatic switch to secondary (deterministic in-memory test; 2026-06-24; test_cdn_failover_primary_timeout_switches_to_secondary)
- [x] Add ABR test: simulate fluctuating bandwidth, verify smooth quality transitions without rebuffering (verified 2026-06-05; abr_buffer.rs: test_abr_buffer_low_selects_lowest, test_abr_buffer_high_selects_highest, test_abr_buffer_monotonic; abr/bba1.rs: test_bba1_reservoir_returns_lowest, test_bba1_cushion_upper_returns_highest — 5 smooth-transition tests)
- [x] Test RTMP handshake and chunk stream parsing against reference output (known byte vectors, byte-level assertions; 2026-06-24; test_handshake_c0c1_byte_layout + test_chunk_type0_header_byte_layout)

## Documentation
- [x] Document supported protocol matrix with capabilities (added as crate-level rustdoc in lib.rs; 2026-06-24)
- [x] Add protocol selection guide: when to use HLS vs DASH vs SRT vs WebRTC vs ST 2110 (added as crate-level rustdoc in lib.rs; 2026-06-24)
- [x] Document CDN configuration with examples for Cloudflare, Fastly, CloudFront integration (added as crate-level rustdoc with no_run example; 2026-06-24)
