//! Deterministic playback pacing and scheduler-stress tests.
//!
//! These tests exercise:
//!   * `PlaybackEngine::presentation_time` — the wall-clock-free, reproducible
//!     frame-presentation timeline (no `Instant::now()`, no `tokio::time`).
//!   * `Scheduler` playlist-swap robustness under repeated
//!     clear/add/remove churn, ordering guarantees, and import/export
//!     round-tripping.
//!
//! Everything here is fully deterministic: a fixed base `DateTime<Utc>` is used
//! (never `Utc::now()`), and no wall-clock-derived value (e.g. `current_frame`)
//! is asserted on.

use std::path::PathBuf;

use chrono::{DateTime, TimeZone, Utc};
use oximedia_playout::playback::{PlaybackConfig, PlaybackEngine};
use oximedia_playout::scheduler::{ScheduledEvent, Scheduler, SchedulerConfig, Transition};
use oximedia_playout::{PlayoutConfig, VideoFormat};
use uuid::Uuid;

/// One frame at 25 fps, in nanoseconds.
const FRAME_25FPS_NANOS: u128 = 40_000_000;
/// One frame at 50 fps, in nanoseconds.
const FRAME_50FPS_NANOS: u128 = 20_000_000;

/// Build a `PlaybackEngine` for an explicit video format.
fn engine_for(format: VideoFormat) -> PlaybackEngine {
    let mut playout = PlayoutConfig::default();
    playout.video_format = format;
    let config = PlaybackConfig::from_playout_config(&playout);
    PlaybackEngine::new(config).expect("engine construction should succeed in test")
}

/// A fixed, reproducible base time (2020-01-01T00:00:00Z) — never `Utc::now()`.
fn base_time() -> DateTime<Utc> {
    match Utc.timestamp_opt(1_577_836_800, 0) {
        chrono::LocalResult::Single(t) => t,
        _ => panic!("fixed base timestamp must be valid"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Default config (25 fps) frame timing.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_default_config_frame_timing_is_40ms() {
    let engine = engine_for(VideoFormat::HD1080p25);
    let timing = engine.frame_timing();

    assert_eq!(
        timing.as_nanos(),
        FRAME_25FPS_NANOS,
        "25 fps frame timing must be exactly 40,000,000 ns"
    );
    assert!(
        (timing.as_secs_f64() - 0.040).abs() < 1e-6,
        "25 fps frame timing must be ~0.040 s, got {}",
        timing.as_secs_f64()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. presentation_time deltas at 25 fps are exactly 40 ms (zero drift).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_presentation_time_25fps_consecutive_deltas_exactly_40ms() {
    let engine = engine_for(VideoFormat::HD1080p25);

    // Frame 0 must anchor the timeline at zero.
    assert_eq!(
        engine.presentation_time(0),
        std::time::Duration::ZERO,
        "frame 0 must present at t=0"
    );

    let pts: Vec<std::time::Duration> = (0..100).map(|i| engine.presentation_time(i)).collect();

    for i in 1..pts.len() {
        let delta = pts[i]
            .checked_sub(pts[i - 1])
            .expect("presentation time must be monotonically non-decreasing");
        assert_eq!(
            delta.as_nanos(),
            FRAME_25FPS_NANOS,
            "delta between frame {} and {} must be exactly 40 ms",
            i,
            i - 1
        );
    }

    // Absolute check: frame N == N * 40 ms with zero accumulated drift.
    for (i, pt) in pts.iter().enumerate() {
        assert_eq!(
            pt.as_nanos(),
            FRAME_25FPS_NANOS * (i as u128),
            "frame {i} presentation time must equal i*40ms exactly"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. presentation_time deltas at 50 fps are exactly 20 ms (±1 µs).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_presentation_time_50fps_consecutive_deltas_20ms() {
    let engine = engine_for(VideoFormat::HD1080p50);
    assert_eq!(
        engine.frame_timing().as_nanos(),
        FRAME_50FPS_NANOS,
        "50 fps frame timing must be exactly 20,000,000 ns"
    );

    let pts: Vec<std::time::Duration> = (0..100).map(|i| engine.presentation_time(i)).collect();

    for i in 1..pts.len() {
        let delta = pts[i]
            .checked_sub(pts[i - 1])
            .expect("presentation time must be monotonically non-decreasing");
        let delta_ns = delta.as_nanos() as i128;
        let expected_ns = FRAME_50FPS_NANOS as i128;
        assert!(
            (delta_ns - expected_ns).abs() <= 1_000,
            "delta {delta_ns} ns must be within 1 µs of 20 ms ({expected_ns} ns)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. presentation_time at 29.97 fps: deltas within ±1 µs of frame_timing()
//    over 300 frames, with NO accumulated jitter.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_presentation_time_2997fps_no_accumulated_jitter() {
    let engine = engine_for(VideoFormat::HD1080p2997);
    let period = engine.frame_timing();
    let period_ns = period.as_nanos() as i128;

    let pts: Vec<std::time::Duration> = (0..300).map(|i| engine.presentation_time(i)).collect();

    // Each consecutive delta must hug the true period to within ±1 µs.
    for i in 1..pts.len() {
        let delta = pts[i]
            .checked_sub(pts[i - 1])
            .expect("presentation time must be monotonically non-decreasing");
        let delta_ns = delta.as_nanos() as i128;
        assert!(
            (delta_ns - period_ns).abs() <= 1_000,
            "frame {i} delta {delta_ns} ns deviates >1 µs from period {period_ns} ns"
        );
    }

    // No accumulated jitter: frame N must be within ±1 µs of N * period even
    // after 300 frames (i.e. error does NOT grow with the frame index).
    for (i, pt) in pts.iter().enumerate() {
        let exact_ns = period_ns * (i as i128);
        let actual_ns = pt.as_nanos() as i128;
        assert!(
            (actual_ns - exact_ns).abs() <= 1_000,
            "frame {i}: presentation {actual_ns} ns drifted >1 µs from exact {exact_ns} ns"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scheduler helpers.
// ─────────────────────────────────────────────────────────────────────────────

fn new_scheduler() -> Scheduler {
    Scheduler::new(SchedulerConfig::default())
}

/// Build a content event at `base + offset_ms` milliseconds.
fn content_event_at(base: DateTime<Utc>, offset_ms: i64, path: &str) -> ScheduledEvent {
    let when = base + chrono::Duration::milliseconds(offset_ms);
    ScheduledEvent::new_content(
        when,
        PathBuf::from(path),
        Some(1000),
        Transition::Cut,
        Transition::Cut,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Scheduler stress: 1000 ops (add / remove-one / swap), oracle-checked.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scheduler_stress_1000_ops_oracle() {
    let scheduler = new_scheduler();
    let base = base_time();

    // Oracle of currently-live event ids.
    let mut live: Vec<Uuid> = Vec::new();

    for i in 0..1000_i64 {
        match i % 3 {
            0 => {
                // add one
                let ev = content_event_at(base, i, "/stress/add.mxf");
                live.push(ev.id);
                scheduler.add_event(ev);
            }
            1 => {
                // remove one (the oldest live id), if any
                if !live.is_empty() {
                    let id = live.remove(0);
                    scheduler
                        .remove_event(id)
                        .expect("remove_event must succeed");
                }
            }
            _ => {
                // playlist swap: clear everything, then add a fresh batch of 5
                scheduler.clear_schedule();
                live.clear();
                for k in 0..5_i64 {
                    let ev = content_event_at(base, i * 8 + k, "/stress/swap.mxf");
                    live.push(ev.id);
                    scheduler.add_event(ev);
                }
            }
        }
    }

    // Count must match the oracle exactly.
    assert_eq!(
        scheduler.event_count(),
        live.len(),
        "event_count must equal oracle live-set size"
    );

    // Every live id must be retrievable exactly once over a wide window.
    let window_start = base - chrono::Duration::days(1);
    let window_end = base + chrono::Duration::days(1);
    let fetched = scheduler.get_events_in_range(window_start, window_end);
    assert_eq!(
        fetched.len(),
        live.len(),
        "range query must return exactly the live events"
    );

    for id in &live {
        let hits = fetched.iter().filter(|e| e.id == *id).count();
        assert_eq!(hits, 1, "live id {id} must appear exactly once");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. After swaps, range query is non-decreasing in scheduled_time.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scheduler_range_non_decreasing_after_swaps() {
    let scheduler = new_scheduler();
    let base = base_time();

    // Several swap rounds, each adding events at scattered (but distinct) offsets.
    for round in 0..10_i64 {
        scheduler.clear_schedule();
        // Intentionally add out of chronological insertion order.
        for &off in &[7_i64, 1, 9, 3, 5, 0, 8, 2, 6, 4] {
            scheduler.add_event(content_event_at(base, round * 100 + off, "/order.mxf"));
        }
    }

    let events = scheduler.get_events_in_range(
        base - chrono::Duration::days(1),
        base + chrono::Duration::days(1),
    );
    assert!(!events.is_empty(), "events must be present after swaps");

    for w in events.windows(2) {
        assert!(
            w[0].scheduled_time <= w[1].scheduled_time,
            "range query must be non-decreasing in scheduled_time"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Empty-schedule swap + unknown-id removal are no-ops (no panic).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scheduler_empty_swap_and_unknown_remove() {
    let scheduler = new_scheduler();

    // Clearing an already-empty schedule must not panic and stays at 0.
    scheduler.clear_schedule();
    assert_eq!(scheduler.event_count(), 0);

    // Removing an unknown id must be Ok and leave the count at 0.
    let res = scheduler.remove_event(Uuid::new_v4());
    assert!(res.is_ok(), "removing unknown id must be Ok");
    assert_eq!(scheduler.event_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Swap-to-identical: re-adding structurally identical events does not double.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scheduler_swap_to_identical_no_doubling() {
    let scheduler = new_scheduler();
    let base = base_time();
    let paths = [
        "/id/a.mxf",
        "/id/b.mxf",
        "/id/c.mxf",
        "/id/d.mxf",
        "/id/e.mxf",
    ];

    // First batch.
    for (k, p) in paths.iter().enumerate() {
        scheduler.add_event(content_event_at(base, k as i64, p));
    }
    assert_eq!(scheduler.event_count(), 5);

    // Swap: clear, then add 5 structurally-identical events (fresh Uuids).
    scheduler.clear_schedule();
    for (k, p) in paths.iter().enumerate() {
        scheduler.add_event(content_event_at(base, k as i64, p));
    }
    assert_eq!(
        scheduler.event_count(),
        5,
        "identical swap must not double the count"
    );

    // The content-path set must be preserved across the swap.
    let events = scheduler.get_events_in_range(
        base - chrono::Duration::days(1),
        base + chrono::Duration::days(1),
    );
    let mut got: Vec<String> = events
        .iter()
        .filter_map(|e| e.content_path.as_ref())
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    got.sort();
    let mut want: Vec<String> = paths.iter().map(|p| (*p).to_string()).collect();
    want.sort();
    assert_eq!(got, want, "content-path set must be identical after swap");
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Mid-playout swap + import/export round-trip.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_scheduler_mid_playout_swap() {
    let scheduler = new_scheduler();
    let base = base_time();

    // Add 10 events; record the would-be "next" event before swapping.
    for k in 0..10_i64 {
        scheduler.add_event(content_event_at(base, k, "/old.mxf"));
    }
    assert_eq!(scheduler.event_count(), 10);
    let _next_before = scheduler.get_next_event();

    // Swap to a fresh batch of 10.
    scheduler.clear_schedule();
    let mut new_ids: Vec<Uuid> = Vec::new();
    for k in 0..10_i64 {
        // Schedule these slightly in the future relative to a fixed reference
        // so get_next_event() (which uses the scheduler's internal context time)
        // can resolve one of them. The scheduler context defaults to "now",
        // so events near `base` (in the past) would all be skipped by
        // get_next_event's lower bound. Use a far-future absolute time.
        let far_future = Utc
            .timestamp_opt(4_102_444_800 + k, 0) // 2100-01-01 + k seconds
            .single()
            .expect("far-future timestamp must be valid");
        let ev = ScheduledEvent::new_content(
            far_future,
            PathBuf::from("/new.mxf"),
            Some(1000),
            Transition::Cut,
            Transition::Cut,
        );
        new_ids.push(ev.id);
        scheduler.add_event(ev);
    }
    assert_eq!(
        scheduler.event_count(),
        10,
        "mid-playout swap must yield 10, not 20"
    );

    // get_next_event() must now resolve to one of the new ids.
    let next_after = scheduler
        .get_next_event()
        .expect("a future event must be returned as next");
    assert!(
        new_ids.contains(&next_after.id),
        "next event after swap must be one of the freshly-added events"
    );

    // Import/export round-trip preserves count (build a 12-event schedule).
    let exporter = new_scheduler();
    for k in 0..12_i64 {
        exporter.add_event(content_event_at(base, k, "/roundtrip.mxf"));
    }
    let json = exporter
        .export_schedule(
            base - chrono::Duration::days(1),
            base + chrono::Duration::days(1),
        )
        .expect("export must succeed");

    let importer = new_scheduler();
    importer
        .import_schedule(&json)
        .expect("import must succeed");
    assert_eq!(
        importer.event_count(),
        12,
        "import/export round-trip must preserve the 12-event count"
    );
}
