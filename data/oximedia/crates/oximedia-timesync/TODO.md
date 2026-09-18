# oximedia-timesync TODO

## Current Status
- 67 source files across 46+ public modules providing comprehensive time synchronization
- PTP (IEEE 1588-2019): clock (discipline, drift, holdover, offset, selection), ptp (clock, message, port, bmca, slave, transparent, dataset), boundary_clock, gptp
- NTP (RFC 5905): ntp (client, packet, pool, filter, stratum)
- Timecode sync: timecode (mod, ltc, mtc, smpte, jam_sync)
- Clock discipline: clock_discipline, clock_domain, clock_ensemble, clock_error, clock_recovery, clock_steering
- Monitoring: drift_monitor, sync_audit, sync_metrics, sync_monitor, sync_stats, sync_status, sync_window
- Media sync: sync (genlock, audio, video), aes67, dante_clock, frequency_estimator, frequency_sync, holdover_estimator, jitter_buffer, leap_second, offset_correction, offset_filter, phase_lock, reference_clock, sync_protocol, time_reference
- IPC: ipc (socket, shmem), ffi (clock_adjust)
- Integration: integration module with timestamp conversion utilities
- Dependencies: oximedia-core, oximedia-timecode, bytes, bitflags, chrono, crc, tokio, memmap2, libc (non-wasm)

## Enhancements
- [x] Add PTP Announce message timeout handling in `ptp::bmca` for detecting master clock loss and triggering re-election (verified 2026-05-16; src/ptp/bmca.rs:280 AnnounceTimeoutTracker)
- [x] Extend `ntp::client` to support NTS (Network Time Security, RFC 8915) for authenticated time synchronization (verified 2026-05-16; src/ntp/nts.rs:1 NTS RFC 8915, NtsCookie:81, NtsKeyMaterial:115)
- [x] Improve `clock::drift::DriftEstimator` with Allan variance computation for oscillator characterization (implemented 2026-05-31: AllanVarianceEstimator in src/clock/drift.rs; allan_variance, overlapping_allan_variance, add_sample; 6 tests covering white phase/freq noise + capacity + constant phase)
- [x] Add `clock_ensemble` support for weighted averaging of multiple clock sources using Bayesian estimation (verified 2026-05-16; src/clock_ensemble.rs:596 BayesianEnsemble Gaussian conjugate updates)
- [x] Extend `aes67` module with PTP profile compliance checking per AES67-2013 specification (implemented 2026-05-31: Aes67ProfileChecker + Aes67ComplianceReport + PtpConfig + PtpDelayMechanism in src/aes67.rs; §7.2/§7.3 checks; 6 tests)
- [x] Improve `jitter_buffer` with adaptive depth adjustment based on measured network jitter statistics (verified 2026-05-16; src/jitter_buffer.rs:123 adaptive_depth field, JitterStats:118)
- [x] Add `sync_audit` persistent logging to file with rotation for post-production timing analysis (implemented 2026-05-31: FileAuditLogger + AuditEntry in src/sync_audit.rs; JSON-lines format, size-triggered auto-rotate, manual rotate, BufWriter flush; 4 tests)
- [x] Extend `gptp` (gPTP/802.1AS) with neighbor rate ratio measurement for improved accuracy (verified 2026-05-16; src/gptp.rs:184 neighbor_rate_ratio field)

## New Features
- [x] Implement `white_rabbit` module for White Rabbit sub-nanosecond PTP extension (used in broadcast facilities) (verified 2026-05-16; src/white_rabbit.rs:39 WrLinkDelayCoefficients, DdmtdSample:96, 503 lines)
- [x] Add `smpte_2059` module for SMPTE ST 2059 PTP profile for professional media (media-specific PTP profile) (verified 2026-05-16; src/smpte_2059.rs:18 Rational, frame_duration_ns:50, 502 lines)
- [x] Implement `clock_graph` module for visualizing clock hierarchy and sync relationships in multi-device setups (verified 2026-05-16; src/clock_graph.rs:168 ClockGraph, 547 lines)
- [x] Add `ravenna` module for RAVENNA AoIP clock synchronization profile support (verified 2026-05-16; src/ravenna.rs:77 RavennaStreamConfig, RavennaClockDomain:221, 427 lines)
- [x] Implement `sync_test` module with synthetic clock drift/offset injection for testing sync algorithms (verified 2026-05-16; src/sync_test.rs:199 SyntheticClock, SyncTestHarness:292, 547 lines)
- [x] Add `gps_reference` module for GPS-disciplined clock input as ultimate reference source (verified 2026-05-16; src/gps_reference.rs:307 GpsDisciplinedClock, GpsPpsSource:224, 561 lines)
- [x] Implement `ptp_management` module for PTP management messages (GET/SET/COMMAND) per IEEE 1588 (verified 2026-05-16; src/ptp_management.rs:257 ManagementMessage, SimpleManagementResponder:444, 768 lines)
- [x] Add `clock_quality_monitor` that tracks and reports MTIE (Maximum Time Interval Error) and TDEV (verified 2026-05-16; src/clock_quality_monitor.rs:101 ClockQualityMonitor, 418 lines)

## Performance
- [x] Use lock-free shared memory updates in `ipc::shmem` for microsecond-level timestamp distribution (done — seqlock at src/ipc/shmem.rs:52)
- [x] Implement batched PTP message processing in `ptp::port` to handle burst arrival of sync/follow-up messages (done — PtpBatchProcessor at src/ptp/port.rs:204)
- [ ] Add hardware timestamping support detection in `ptp::clock` for sub-microsecond PTP accuracy
- [ ] Use SIMD-accelerated CRC computation in PTP message validation
- [x] Implement zero-allocation PTP message parsing in `ptp::message` using nom or manual byte parsing (implemented 2026-05-31: SyncMessageView<'a>, parse_sync_message_zerocopy, parse_sync_batch + PtpError in src/ptp/message.rs; 5 tests incl. roundtrip vs allocating parse)

## Testing
- [x] Add PTP BMCA election test with multiple clock candidates at different priorities and verify correct grandmaster selection (implemented 2026-05-31: BmcaSelector + PtpClockIdentity in src/ptp/bmca.rs; full IEEE 1588 ordering; 6 tests covering all 5 tiebreak levels + empty + single)
- [ ] Test `clock::holdover` accuracy degradation over time with known oscillator drift model
- [x] Add NTP client test with simulated server responses at various stratum levels and verify correct server selection (implemented 2026-05-31: NtpServerSelector + NtpServerInfo in src/ntp/stratum.rs; stratum-first then RTT selection; 6 tests)
- [ ] Test `timecode::jam_sync` lock acquisition and holdover behavior with intermittent LTC signal
- [ ] Add `genlock` test verifying frame-edge alignment within +/-1 sample at 48kHz for genlock output
- [ ] Test `leap_second` handling at UTC midnight boundary with PTP and NTP sources
- [ ] Add `dante_clock` interop test verifying compatibility with Dante clock domain behavior

## Documentation
- [ ] Add clock hierarchy diagram showing PTP grandmaster -> boundary clock -> ordinary clock -> application
- [ ] Document sync accuracy expectations for each protocol (PTP: <1us, NTP: <10ms, LTC: 1 frame, genlock: sub-sample)
- [ ] Add deployment guide for broadcast facility time synchronization with PTP, genlock, and LTC
