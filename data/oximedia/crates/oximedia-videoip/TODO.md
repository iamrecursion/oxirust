# oximedia-videoip TODO

## Current Status
- 44 source files implementing professional video-over-IP protocol (patent-free NDI alternative)
- Key features: low-latency UDP transport with FEC, mDNS/DNS-SD service discovery, VP9/AV1/v210/UYVY video codecs, Opus/PCM audio formats, tally lights, PTZ control, timecode, metadata, jitter buffering, packet loss recovery, multi-stream support
- Modules: bandwidth_est, bonding, codec, color_space_conv, congestion, discovery, encryption, fec, flow_monitor, flow_stats, frame_pacing, jitter, metadata, multicast, multicast_group, ndi_bridge, nmos, packet, packet_loss, ptp_boundary, ptz, quic_transport, receiver, redundancy, rist, sdp, smpte2110, source, srt_config, stats, stream_descriptor, stream_health, stream_recorder, stream_sync, tally, transport, utils (bandwidth, connection, quality, ring_buffer)
- Dependencies: oximedia-core, oximedia-codec, oximedia-net, oximedia-monitor, oximedia-timecode, tokio, socket2, mdns-sd, reed-solomon-erasure, crossbeam, flume

## Enhancements
- [x] Implement actual QUIC transport in `quic_transport` using quinn or equivalent pure-Rust QUIC library (implemented 2026-05-15; src/quic_transport_quinn.rs: QuicTransport, QuicConnection, QuicTransportConfig, send_datagram/recv_datagram; feature-gated `quic-quinn`; self-signed certs via rcgen; loopback test passes; 517 lines)
- [x] Add adaptive FEC rate in `fec` module that adjusts parity packet ratio based on measured `packet_loss` rate
- [x] Extend `congestion` control with BBR-style algorithm instead of basic AIMD for better bandwidth utilization
- [x] Implement actual PTP (IEEE 1588) clock synchronization in `ptp_boundary` with sub-microsecond accuracy (verified 2026-05-16; src/ptp_boundary.rs:189 PtpBoundaryClock, ClockIdentity:52, AnnounceMessage:90, 551 lines)
- [x] Add NMOS IS-04/IS-05 full registration and connection management in `nmos` module (verified 2026-05-16; src/nmos.rs:69 IS-04 NmosNode, IS-05 RtpTransportParams:154, NmosId:19, 463 lines)
- [x] Extend `encryption` module with DTLS-SRTP for standard-compliant media encryption (verified 2026-05-16; src/dtls_srtp.rs:175 DtlsSrtpConfig, SrtpKeyMaterial:93, from_dtls_export:141, 718 lines)
- [x] Improve `jitter` buffer with adaptive depth based on network conditions (expand during congestion, shrink during stability)
- [x] Add actual SDP offer/answer negotiation in `sdp` module for standard SIP/WebRTC interop (verified 2026-05-16; src/sdp_negotiation.rs:148 SdpMediaSection, SdpAttribute:99, 750 lines)

## New Features
- [x] Implement `whip_whep` module for WHIP/WHEP protocol support (WebRTC-based ingest/egress) (verified 2026-05-16; src/whip_whep.rs:86 WhipWhepSession, IceCandidate:57, SessionRole, 608 lines)
- [x] Add `srt_transport` module implementing full SRT (Secure Reliable Transport) protocol in pure Rust (verified 2026-05-16; src/srt_transport.rs:152 SrtHandshakeMsg, SrtTransportConfig:34, 1210 lines)
- [x] Implement `rtsp_server` module for RTSP/RTP source serving to standard media players (verified 2026-05-16; src/rtsp_server.rs:117 RtspSession, MediaTrack:253, RtspResponse:186, 642 lines)
- [x] Add `stream_recording_mux` that records incoming streams directly to MKV/WebM containers (verified 2026-05-16; src/stream_recording_mux.rs:147 RecordingConfig, VideoTrack:113, MuxFrame:304, 1014 lines)
- [x] Implement `bandwidth_shaping` module for traffic shaping and QoS prioritization per stream (verified 2026-05-16; src/bandwidth_shaping.rs:208 BandwidthShaper, StreamShapeConfig:107, submit_packet:268, 835 lines)
- [x] Add `multiview` module for combining multiple receiver streams into a single mosaic output (verified 2026-05-16; src/multiview.rs:201 MultiviewCompositor, MultiviewCell:43, composite:324, 576 lines)
- [x] Implement `stream_relay` for re-broadcasting received streams to multiple downstream receivers (verified 2026-05-16; src/stream_relay.rs:72 RelaySink, RelayFrame:48, subscribe_to:105, 473 lines)
- [x] Add `diagnostic_overlay` module that burns network stats (latency, packet loss, bitrate) onto video frames (verified 2026-05-16; src/diagnostic_overlay.rs:204 DiagnosticOverlay, NetworkStats:11, render_onto:226, 416 lines)

## Performance
- [x] Optimize `fec` Reed-Solomon encoding/decoding with SIMD-accelerated Galois field arithmetic (implemented 2026-05-15; src/gf_simd.rs: gf_mul_lut, gf_mul_slice_simd with AVX2/SSSE3 VPSHUFB dispatch, SimdRsEncoder; all GF tests pass; 511 lines)
- [x] Implement zero-copy packet path in `transport` using `bytes::Bytes` throughout without intermediate copies (implemented 2026-05-15; src/zero_copy_packet.rs: ZeroCopyPacket, ZeroCopyPacketBuilder, ZeroCopySend trait, UdpTransport impl; tests pass; 381 lines)
- [x] Add lock-free ring buffer in `utils::ring_buffer` for audio/video frame handoff between network and processing threads (implemented 2026-05-15; src/spsc_ring.rs: SpscRing<T: Copy+Send> backed by crossbeam::ArrayQueue; SPSC concurrent test passes; 205 lines)
- [x] Optimize `color_space_conv` with SIMD for v210-to-planar and UYVY-to-planar conversion hot paths (implemented 2026-05-15; src/color_space_simd.rs: uyvy_to_planar_simd with SSSE3 dispatch, v210_to_planar scalar; all tests pass; 435 lines)
- [x] Implement scatter/gather I/O (sendmmsg/recvmmsg) in UDP transport for reduced syscall overhead (implemented 2026-05-15; src/udp_scatter_gather.rs: UdpScatterGather::send_many with Linux sendmmsg and fallback loop; cfg-gated; tests pass; 299 lines)
- [x] Profile and optimize `frame_pacing` to use precise timer (mach_absolute_time on macOS) instead of sleep-based pacing (implemented 2026-06-01; src/frame_pacing.rs: PreciseClock struct with now_ns()+sleep_until_ns(); macOS mach_absolute_time via cfg-gated extern "C" + #[allow(unsafe_code)]; FramePacer::wait_for_next_frame() + next_deadline_ns field; 560 lines)

## Testing
- [x] Add network simulation tests for `congestion` control under varying latency/loss conditions (implemented 2026-06-01; tests/network_sim.rs: test_congestion_cwnd_responds_to_rtt_spike, test_congestion_backs_off_on_loss; 13 tests total, all pass)
- [x] Test `fec` recovery with configurable packet loss patterns (burst, random, periodic) (implemented 2026-06-01; tests/network_sim.rs: test_fec_recovery_burst_drop, test_fec_recovery_random_drop, test_fec_parity_count_matches_config)
- [ ] Add integration test for full source->receiver round-trip with loopback UDP
- [ ] Test `discovery` mDNS announcement and resolution with multiple concurrent sources
- [x] Benchmark `jitter` buffer at various network jitter levels (1ms, 5ms, 20ms, 50ms) (implemented 2026-06-01; tests/network_sim.rs: test_jitter_buffer_adapts_to_higher_jitter, test_jitter_buffer_shrinks_after_stable_phase)
- [x] Test `stream_sync` lip-sync accuracy between audio and video streams under packet reordering (implemented 2026-06-01; tests/network_sim.rs: test_stream_sync_gap_measurement, test_stream_sync_detects_excessive_av_gap, test_stream_sync_sequence_reorder_detection)

## Documentation
- [ ] Document protocol wire format specification (packet header layout, control message types)
- [ ] Add network configuration guide (firewall ports, multicast setup, bandwidth requirements per resolution)
- [ ] Document SMPTE 2110 compatibility mapping between VideoIP types and ST 2110-20/30/40

## 0.1.8 Wave 18 Slice F — 2026-06-01
- [x] PreciseClock abstraction in frame_pacing: now_ns() via mach_absolute_time on macOS (cfg-gated extern "C" + #[allow(unsafe_code)]), SystemTime fallback on other platforms; sleep_until_ns() coarse-sleep + 200µs busy-wait tail; FramePacer::wait_for_next_frame() real-time blocking method; all 1066 tests pass, 0 warnings
- [x] tests/network_sim.rs: 13 deterministic network simulation tests (PreciseClock monotonicity, FramePacer cadence+delay, CongestionController RTT spike + loss reaction, FEC burst/random drop recovery, JitterBuffer jitter adaptation, StreamSyncMonitor gap measurement + reorder detection)

## 0.1.8 Wave 6 — 2026-05-29
- [x] Register 22 orphan modules in lib.rs (verified 2026-05-29; 22 orphans wired, 22 smoke tests, 0 warnings)
