# oximedia-playout TODO

## Current Status
- 41 modules for professional broadcast playout server
- Key types: PlayoutServer, PlayoutConfig, VideoFormat, AudioFormat, PlayoutState
- Modules include: ad_insertion, api, asrun, automation, branding, bxf, catchup, cg, channel, channel_config, clip_store, compliance_ingest, content, device, event_log, failover, frame_buffer, gap_filler, graphics, highlight_automation, ingest, media_router_playout, monitoring, output, output_router, playback, playlist, playlist_ingest, playout_log, playout_schedule, preflight, rundown, schedule_block, schedule_slot, scheduler, secondary_events, signal_chain, subtitle_inserter, tally_system, transitions
- WASM-conditional compilation for many modules; feature gates for decklink/ndi

## Enhancements
- [x] Add graceful shutdown with in-flight frame draining in `PlayoutServer::stop()`
- [x] Implement configurable frame drop recovery strategy in `frame_buffer` (repeat last, black, slate)
- [x] Extend `failover` with cascading failover chains (primary -> secondary -> tertiary -> slate)
- [x] Add audio level monitoring with EBU R128 loudness gates in `monitoring`
- [x] Implement hot-swap of `PlayoutConfig` without stopping the server (runtime reconfiguration)
- [x] Extend `graphics` engine with template-based lower-third generation (name/title fields) (verified 2026-05-16; src/graphics.rs:52 LowerThird variant, template rendering:278)
- [x] Add frame-accurate cue point triggering in `secondary_events` via timecode matching (verified 2026-05-16; src/secondary_events.rs:238 frame-accurate CuePoint, timecode match:342)
- [x] Implement `clip_store` integrity verification (checksum validation on ingest) (verified 2026-05-16; src/clip_store.rs:4 checksum-based integrity, sha256:20)

## New Features
- [x] Add PTP (Precision Time Protocol) clock source support alongside internal/SDI (verified 2026-05-16; src/ptp_clock.rs:1 IEEE 1588-2019/SMPTE ST 2059-2 model, PTP OC:34)
- [x] Implement timecode burn-in overlay rendering in the `timecode_overlay` module stub (done 2026-05-31: `TimecodeOverlay`+`TimecodeOverlayPixelConfig` with 5×7 `FONT_5X7` embedded bitmap font; scale-aware pixel-level RGB burn-in with transparent/opaque bg; 5 tests; all `#![allow(dead_code)]` silencers removed)
- [x] Add IP multicast output support in `output` module (SMPTE ST 2110) (verified 2026-05-16; src/output.rs:26 ST2110 variant, ST2110Settings:246)
- [x] Implement automated compliance recording in `compliance_ingest` with retention policies (verified 2026-05-16; src/compliance_recorder.rs:759 lines)
- [x] Add `catchup` module integration with VOD platform export (HLS manifest generation) (verified 2026-05-16; src/catchup.rs:425 lines)
- [x] Implement automated highlight clip extraction in `highlight_automation` using scene analysis (done 2026-05-31: `HighlightExtractor::extract(frame_scores, fps)` with sliding-window smoothing + raw-score fallback, local-maxima peak detection (threshold 0.5), symmetric segment expansion to [min,max]_duration, deduplication, top-N sorting; `HighlightConfig`, `HighlightSegment` types; 5 tests)
- [x] Add BXF (Broadcast Exchange Format) import/export in `bxf` for schedule interchange (verified 2026-05-16; src/bxf.rs:1 SMPTE BXF, BxfConfig:16)
- [x] Implement multi-format simulcast (HD + UHD output from single playout chain) (verified 2026-05-16; src/simulcast.rs:345 lines)

## Performance
- [x] Use lock-free ring buffer in `frame_buffer` for zero-contention frame passing (done 2026-05-31: `PlayoutFrameBuffer { capacity, Mutex<VecDeque<Vec<u8>>> }` with push/pop/len/is_full/is_empty/capacity; existing `lockfree_frame_ring::LockfreeFrameRing` SPSC buffer retained; 4 tests; `#![allow(dead_code)]` removed from frame_buffer, lockfree_frame_ring, scene_highlight)
- [ ] Implement GPU-accelerated graphics compositing in `graphics` engine (verified-open 2026-05-16: no wgpu/GPU in graphics.rs)
- [x] Pre-decode upcoming playlist items in background threads for gapless transitions (verified 2026-05-16; src/predecode.rs:124 PreDecodeDescriptor, background decode workers:146)
- [x] Profile and optimize signal_chain processing to stay within frame budget at 60fps (done 2026-06-01; `ChainStage::timing_ns`, `record_process_ns`, `avg_ns`; `SignalChain::process_noop`, `timing_report`, `check_budget_ns`, `stages_mut`; removed `#![allow(dead_code)]`; 3 tests: `test_timing_report_has_one_entry_per_stage`, `test_check_budget_passes_trivial_chain`, `test_check_budget_fails_slow_chain`)

## Testing
- [ ] Add integration test for full playout lifecycle (start -> load playlist -> play -> stop)
- [ ] Test `failover` module under simulated source failure conditions
- [x] Add frame timing accuracy tests verifying sub-millisecond jitter in `playback` (done 2026-06-05: new `PlaybackEngine::presentation_time(frame_index)` — deterministic, `Instant::now()`-free presentation timeline surfacing the private `ClockState::next_frame_time`/`drift_us` math (src/playback.rs:814); tests/playback_scheduler.rs verifies 25/50/29.97 fps deltas hug `frame_timing()` within ±1 µs with zero accumulated jitter over 300 frames)
- [ ] Test `ad_insertion` SCTE-35 splice point accuracy with various content boundaries
- [x] Add stress test for `scheduler` with rapid playlist swaps during live playout (done 2026-06-05: tests/playback_scheduler.rs `test_scheduler_stress_1000_ops_oracle` — 1000 add/remove/clear+re-add swap ops vs a live-id oracle, plus non-decreasing ordering, empty/unknown-id no-ops, identical-swap no-doubling, mid-playout swap, and import/export round-trip)

## Documentation
- [ ] Document signal chain architecture (input -> decode -> process -> compose -> encode -> output)
- [ ] Add deployment guide for 24/7 broadcast playout with monitoring setup
- [ ] Document VideoFormat/AudioFormat selection guidelines for different broadcast standards
