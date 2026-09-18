//! Network simulation tests for oximedia-videoip.
//!
//! Each test exercises a component deterministically — no actual OS sleep is
//! required because we feed synthetic timestamps directly through the public APIs.

use bytes::Bytes;
use oximedia_videoip::{
    congestion::CongestionController,
    fec::{FecDecoder, FecEncoder},
    frame_pacing::{FramePacer, PreciseClock},
    jitter::{NetworkAwareJitterBuffer, NetworkAwareJitterConfig, NetworkCondition},
    packet::PacketBuilder,
    stream_sync::StreamSyncMonitor,
};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// PreciseClock smoke test
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that `PreciseClock::now_ns` is monotonically non-decreasing over a
/// short loop.  We do NOT sleep here; we only confirm the counter never goes
/// backwards.
#[test]
fn test_precise_clock_monotonic() {
    let clock = PreciseClock::new();
    let mut prev = clock.now_ns();
    for _ in 0..1_000 {
        let now = clock.now_ns();
        assert!(now >= prev, "PreciseClock went backwards: {prev} → {now}");
        prev = now;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FramePacer cadence test (deterministic, no actual sleep)
// ─────────────────────────────────────────────────────────────────────────────

/// Run `FramePacer` for 100 simulated frames at 60 fps by feeding synthetic
/// `Instant` values spaced exactly one frame apart.
///
/// Asserts:
/// * every decision is `should_send = true`
/// * each frame is produced with `sequence` incrementing by 1
/// * stats report exactly 100 frames
#[test]
fn test_precise_clock_cadence_simulated() {
    let mut pacer = FramePacer::new(60.0);
    let interval = pacer.frame_interval();
    let t0 = Instant::now();

    for i in 0..100u32 {
        let t = t0 + interval * i;
        let decision = pacer.pace_frame(t);
        assert!(
            decision.should_send,
            "frame {i} should be sendable when supplied at the ideal instant"
        );
        assert_eq!(decision.sequence, i as u64, "sequence must increment");
    }

    let stats = pacer.stats();
    assert_eq!(stats.total_frames, 100, "total_frames must be 100");
    assert_eq!(stats.delayed_frames, 0, "no frame should be delayed");
}

/// Verify `FramePacer` correctly marks a frame as delayed when we call
/// `pace_frame` *before* the scheduled deadline.
#[test]
fn test_frame_pacer_delay_when_early() {
    let mut pacer = FramePacer::new(30.0);
    let t0 = Instant::now();
    // First frame — always immediate.
    let d0 = pacer.pace_frame(t0);
    assert!(d0.should_send);
    // Ask for second frame immediately (no time elapsed) — should need delay.
    let d1 = pacer.pace_frame(t0);
    assert!(
        !d1.should_send,
        "second frame before interval must be delayed"
    );
    assert!(d1.delay > Duration::ZERO, "delay must be positive");
}

// ─────────────────────────────────────────────────────────────────────────────
// CongestionController: cwnd backs off after RTT spike
// ─────────────────────────────────────────────────────────────────────────────

/// Feed a synthetic RTT sequence: 10 ms baseline × 4 probes, then a 50 ms
/// spike at probe 5.
///
/// Asserts: `target_bitrate_bps` at frame 10 is **less than** at frame 4
/// (controller reduced rate after detecting congestion).
#[test]
fn test_congestion_cwnd_responds_to_rtt_spike() {
    let mut cc = CongestionController::new();
    let base_rtt = Duration::from_millis(10);

    // Build up a stable baseline (at least MIN_RTT_SAMPLES = 4).
    for _ in 0..4 {
        cc.report_rtt(base_rtt);
    }
    let bitrate_before = cc.target_bitrate_bps();

    // Inject a 50 ms spike — well above the 1.5× threshold.
    cc.report_rtt(Duration::from_millis(50));
    // Feed a few more high-RTT samples to drive convergence.
    for _ in 0..5 {
        cc.report_rtt(Duration::from_millis(40));
    }
    let bitrate_after = cc.target_bitrate_bps();

    assert!(
        bitrate_after < bitrate_before,
        "bitrate should decrease after RTT spike: {bitrate_after} < {bitrate_before}"
    );
}

/// Verify that loss reporting also triggers a bitrate reduction.
#[test]
fn test_congestion_backs_off_on_loss() {
    let mut cc = CongestionController::new();
    let before = cc.target_bitrate_bps();
    // Report 5 % loss (above default 2 % threshold).
    cc.report_loss(100, 5);
    let after = cc.target_bitrate_bps();
    assert!(
        after < before,
        "bitrate should decrease after loss: {after} < {before}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// FEC: burst and random recovery
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: create a deterministic `Packet` with a given sequence number and
/// 64-byte payload.
fn make_packet(seq: u16) -> oximedia_videoip::packet::Packet {
    let payload: Vec<u8> = (0u8..64).map(|b| b.wrapping_add(seq as u8)).collect();
    PacketBuilder::new(seq)
        .video()
        .with_timestamp(u64::from(seq) * 1000)
        .build(Bytes::from(payload))
        .expect("packet build should succeed")
}

/// Drop 2 consecutive packets (indices 3, 4) out of 5 data packets.
///
/// The FEC group is set up exactly as the existing internal test does:
/// * data packets: seq 0..4 (data_shards = 5)
/// * parity packets: seq 5..7 (base_sequence = data_shards = 5, parity_shards = 3)
/// * total group = data_shards + parity_shards = 8 — within u8 index space
///
/// Dropping 2 out of 5 data packets is within the parity budget of 3.
/// Asserts: recovered packet count >= 2.
#[test]
fn test_fec_recovery_burst_drop() {
    let data_shards = 5_usize;
    let parity_shards = 3_usize;

    let encoder = FecEncoder::new(data_shards, parity_shards).expect("encoder init");
    let mut decoder = FecDecoder::new(data_shards, parity_shards).expect("decoder init");

    // Data packets: seq 0, 1, 2, 3, 4
    let data_packets: Vec<_> = (0u16..data_shards as u16).map(make_packet).collect();

    // Parity packets start at seq = data_shards (seq 5, 6, 7).
    let parity = encoder
        .encode(
            &data_packets,
            data_shards as u16, // base_sequence for parity packets
            0,
            oximedia_videoip::types::StreamType::Program,
        )
        .expect("encode should succeed");
    assert_eq!(
        parity.len(),
        parity_shards,
        "must have {parity_shards} parity packets"
    );

    // Drop consecutive packets at index 3 and 4 (burst drop).
    let drop_seqs = [3u16, 4u16];
    let mut recovered_total = 0usize;
    for pkt in &data_packets {
        if drop_seqs.contains(&pkt.header.sequence) {
            continue;
        }
        let recovered = decoder
            .add_packet(pkt.clone())
            .expect("add_packet should succeed");
        recovered_total += recovered.len();
    }
    // Feed all parity packets — decoder should trigger reconstruction.
    for pkt in parity {
        let recovered = decoder
            .add_packet(pkt)
            .expect("parity add_packet should succeed");
        recovered_total += recovered.len();
    }

    assert!(
        recovered_total >= drop_seqs.len(),
        "FEC must recover at least {n} dropped packets; got {recovered_total}",
        n = drop_seqs.len()
    );
}

/// Drop 2 non-consecutive (random) packets (index 1 and 3) out of 5.
/// Asserts recovery succeeds.
#[test]
fn test_fec_recovery_random_drop() {
    let data_shards = 5_usize;
    let parity_shards = 3_usize;

    let encoder = FecEncoder::new(data_shards, parity_shards).expect("encoder init");
    let mut decoder = FecDecoder::new(data_shards, parity_shards).expect("decoder init");

    let data_packets: Vec<_> = (0u16..data_shards as u16).map(make_packet).collect();
    let parity = encoder
        .encode(
            &data_packets,
            data_shards as u16,
            0,
            oximedia_videoip::types::StreamType::Program,
        )
        .expect("encode should succeed");

    let drop_seqs = [1u16, 3u16];
    let mut recovered_total = 0usize;
    for pkt in &data_packets {
        if drop_seqs.contains(&pkt.header.sequence) {
            continue;
        }
        let recovered = decoder.add_packet(pkt.clone()).expect("add_packet");
        recovered_total += recovered.len();
    }
    for pkt in parity {
        let recovered = decoder.add_packet(pkt).expect("parity add_packet");
        recovered_total += recovered.len();
    }

    assert!(
        recovered_total >= drop_seqs.len(),
        "FEC must recover at least {n} dropped packets; got {recovered_total}",
        n = drop_seqs.len()
    );
}

/// FEC encoder produces parity packets whose count equals `parity_shards`.
/// Parity base sequence is set to `data_shards` so group_id bookkeeping aligns.
#[test]
fn test_fec_parity_count_matches_config() {
    let data_shards = 8_usize;
    let parity_shards = 2_usize;
    let encoder = FecEncoder::new(data_shards, parity_shards).expect("encoder init");
    let pkts: Vec<_> = (0u16..data_shards as u16).map(make_packet).collect();
    let parity = encoder
        .encode(
            &pkts,
            data_shards as u16,
            0,
            oximedia_videoip::types::StreamType::Program,
        )
        .expect("encode");
    assert_eq!(parity.len(), parity_shards);
}

// ─────────────────────────────────────────────────────────────────────────────
// NetworkAwareJitterBuffer: depth adapts to jitter level
// ─────────────────────────────────────────────────────────────────────────────

/// Feed two phases of `NetworkCondition` to `NetworkAwareJitterBuffer::adapt`:
/// * Phase 1 (50 cycles): low jitter (rtt_variance_ms = 1)
/// * Phase 2 (50 cycles): high jitter (rtt_variance_ms = 20)
///
/// Asserts: buffer depth after Phase 2 >= depth after Phase 1.
#[test]
fn test_jitter_buffer_adapts_to_higher_jitter() {
    let config = NetworkAwareJitterConfig {
        initial_depth_ms: 10,
        min_depth_ms: 5,
        max_depth_ms: 200,
        capacity: 512,
        expand_step_ms: 5,
        shrink_step_ms: 2,
        stable_cycles_before_shrink: 5,
        variance_multiplier: 2.0,
        congestion_penalty_ms: 10.0,
        loss_penalty_ms_per_pct: 1.0,
        depth_ema_alpha: 1.0,
    };

    let mut buf = NetworkAwareJitterBuffer::new(config);

    // Phase 1: low jitter — let the buffer stabilise.
    let low_jitter = NetworkCondition {
        rtt_ms: 5.0,
        rtt_variance_ms: 1.0,
        loss_rate: 0.0,
        congested: false,
    };
    for _ in 0..50 {
        buf.adapt(&low_jitter);
    }
    let depth_low = buf.target_depth_ms();

    // Phase 2: high jitter — buffer should grow.
    let high_jitter = NetworkCondition {
        rtt_ms: 30.0,
        rtt_variance_ms: 20.0,
        loss_rate: 0.03,
        congested: true,
    };
    for _ in 0..50 {
        buf.adapt(&high_jitter);
    }
    let depth_high = buf.target_depth_ms();

    assert!(
        depth_high >= depth_low,
        "buffer depth must be >= after high-jitter phase: high={depth_high} >= low={depth_low}"
    );
}

/// After high-jitter then stable recovery cycles, the buffer shrinks back.
#[test]
fn test_jitter_buffer_shrinks_after_stable_phase() {
    let config = NetworkAwareJitterConfig {
        initial_depth_ms: 5,
        min_depth_ms: 5,
        max_depth_ms: 200,
        capacity: 512,
        expand_step_ms: 20,
        shrink_step_ms: 5,
        stable_cycles_before_shrink: 3,
        variance_multiplier: 2.0,
        congestion_penalty_ms: 50.0,
        loss_penalty_ms_per_pct: 0.0,
        depth_ema_alpha: 1.0,
    };

    let mut buf = NetworkAwareJitterBuffer::new(config);

    // Expand under congestion.
    let cong = NetworkCondition {
        rtt_ms: 50.0,
        rtt_variance_ms: 30.0,
        loss_rate: 0.0,
        congested: true,
    };
    for _ in 0..20 {
        buf.adapt(&cong);
    }
    let depth_expanded = buf.target_depth_ms();

    // Now 3 stable cycles → should shrink.
    let stable = NetworkCondition {
        rtt_ms: 5.0,
        rtt_variance_ms: 1.0,
        loss_rate: 0.0,
        congested: false,
    };
    for _ in 0..3 {
        buf.adapt(&stable);
    }
    let depth_shrunk = buf.target_depth_ms();

    assert!(
        depth_shrunk < depth_expanded,
        "buffer should shrink after stable cycles: {depth_shrunk} < {depth_expanded}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamSyncMonitor: reorder tolerance and A/V offset constraint
// ─────────────────────────────────────────────────────────────────────────────

/// Feed 10 video + audio timestamp pairs to `StreamSyncMonitor::measure_gap`.
/// The gap for each pair is computed and recorded.  We check structural
/// properties:
///
/// * `measure_gap` succeeds (returns `Some`) for all 10 pairs
/// * Mean gap magnitude < 33 ms (one video frame period)
/// * Excessive-gap counter reflects truly large offsets
#[test]
fn test_stream_sync_gap_measurement() {
    // video clock rate = 90_000 Hz, audio clock rate = 48_000 Hz
    // max_gap_us = 33_333 (one 30-fps frame)
    let mut mon = StreamSyncMonitor::new(90_000, 48_000, 33_333);

    // Anchor both streams at time 0.
    mon.set_ntp_map_a(0, 0);
    mon.set_ntp_map_b(0, 0);

    // Feed 10 pairs with zero offset — should all be in-sync.
    let mut all_some = true;
    for i in 0..10u32 {
        // advance video by one frame each iteration: 90_000 / 30 = 3_000 ticks
        let v_ts = (i * 3_000) as u32;
        // advance audio proportionally: 48_000 / 30 = 1_600 ticks per frame
        let a_ts = (i * 1_600) as u32;
        if mon.measure_gap(v_ts, a_ts).is_none() {
            all_some = false;
        }
    }

    assert!(all_some, "measure_gap must return Some for all pairs");
    assert_eq!(
        mon.excessive_gap_count(),
        0,
        "no pair should exceed 33 ms when video/audio are in sync"
    );
    // Mean gap should be exactly 0 when perfectly synchronised.
    assert!(
        mon.mean_gap_us().abs() < 1.0,
        "mean gap for in-sync streams should be ~0 µs, got {}",
        mon.mean_gap_us()
    );
}

/// Feed timestamps with a deliberate ≈50 ms offset (> 33 ms threshold).
/// Asserts: at least one excessive-gap event is recorded.
#[test]
fn test_stream_sync_detects_excessive_av_gap() {
    let mut mon = StreamSyncMonitor::new(90_000, 48_000, 33_333);
    mon.set_ntp_map_a(0, 0);
    mon.set_ntp_map_b(0, 0);

    // 4_500 ticks at 90_000 Hz = 50 ms.
    let gap = mon.measure_gap(4_500, 0);
    assert!(gap.is_some(), "measure_gap must return a result");
    assert!(
        gap.unwrap().is_excessive,
        "50 ms gap must exceed the 33 ms threshold"
    );
    assert!(
        mon.excessive_gap_count() >= 1,
        "excessive gap counter must be incremented"
    );
}

/// Feed out-of-order sequence numbers to `SequenceChecker` and verify the gap
/// counter is updated correctly.
#[test]
fn test_stream_sync_sequence_reorder_detection() {
    let mut chk = oximedia_videoip::stream_sync::SequenceChecker::new();

    // In-order: 0,1,2,3 — no gaps.
    for i in 0u16..4 {
        let delta = chk.process(i);
        assert_eq!(delta, 0, "in-order packet {i} must have delta 0");
    }

    // Skip sequence 4 — deliver 5, then 4.
    let gap_delta = chk.process(5);
    assert_eq!(gap_delta, 1, "skipping seq 4 → delta must be 1");
    assert_eq!(chk.gap_count, 1, "gap_count must be 1 after skip");

    // Deliver the out-of-order packet (seq 4) — it is a late arrival.
    let late_delta = chk.process(4);
    assert!(
        late_delta < 0,
        "late (out-of-order) packet must return negative delta"
    );
}
