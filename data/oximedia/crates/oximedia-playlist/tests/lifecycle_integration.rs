//! End-to-end integration tests for the full broadcast playlist lifecycle.
//!
//! Each test drives the **real** crate APIs through the four lifecycle stages —
//! CREATE → SCHEDULE → PLAY → LOG — using a fixed, simulated clock so every
//! assertion is fully deterministic and never touches the wall clock or sleeps.
//!
//! * CREATE   — [`Playlist`] / [`PlaylistItem`] with explicit durations + IDs.
//! * SCHEDULE — per-item timeline from cumulative durations, plus the real
//!   [`ScheduleEngine`] (active-window queries + conflict detection).
//! * PLAY     — [`PlayoutEngine`] state machine + the playlist's own cursor
//!   navigation, advanced over the simulated timeline.
//! * LOG      — the [`AsRunLog`] as-run compliance log and the
//!   [`MetadataTracker`] real-time playback history.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::time::Duration;

use oximedia_playlist::automation::playout::{PlayoutConfig, PlayoutEvent, PlayoutState};
use oximedia_playlist::live::insert::{LiveInsert, LiveInsertManager, LiveSource};
use oximedia_playlist::metadata::asrun::{AsRunEntry, AsRunLog};
use oximedia_playlist::metadata::track::{MetadataTracker, PlaybackEvent};
use oximedia_playlist::playlist::{Playlist, PlaylistItem, PlaylistType};
use oximedia_playlist::schedule::engine::{ScheduleEngine, ScheduleEvent};
use oximedia_playlist::{PlaylistError, PlayoutEngine};

/// Single broadcast channel used by the as-run log assertions.
const CHANNEL_ID: &str = "ch-primetime";

/// Definition of one playlist segment: (item id, title, duration in seconds).
const SEGMENTS: &[(&str, &str, u64)] = &[
    ("seg-00", "Morning News", 1800),
    ("seg-01", "Weather", 300),
    ("seg-02", "Talk Show", 2700),
    ("seg-03", "Ad Break", 120),
    ("seg-04", "Documentary", 3600),
];

/// Fixed wall-clock base — 2026-06-06T12:00:00Z — so the simulated timeline is
/// deterministic and independent of `Utc::now()`.
fn fixed_base() -> DateTime<Utc> {
    DateTime::from_timestamp(1_780_488_000, 0).expect("valid fixed base timestamp")
}

/// CREATE: build a linear playlist from [`SEGMENTS`] with explicit, deterministic
/// item IDs so the LOG-stage identity assertions are unambiguous.
fn create_playlist() -> Playlist {
    let mut playlist = Playlist::new("Prime Time Block", PlaylistType::Linear);
    for (id, title, secs) in SEGMENTS {
        let mut item = PlaylistItem::new(format!("content/{id}.mxf"))
            .with_duration(Duration::from_secs(*secs))
            .with_title(*title);
        item.id = (*id).to_string();
        playlist.add_item(item);
    }
    playlist
}

/// SCHEDULE: compute the per-item timeline (start/end/duration) from the
/// cumulative item durations — the arithmetic a playout scheduler performs.
fn compute_timeline(
    playlist: &Playlist,
    base: DateTime<Utc>,
) -> Vec<(String, DateTime<Utc>, DateTime<Utc>, Duration)> {
    let mut timeline = Vec::with_capacity(playlist.len());
    let mut cursor = base;
    for item in &playlist.items {
        let dur = item.effective_duration();
        let start = cursor;
        let end = start + ChronoDuration::from_std(dur).expect("duration fits chrono range");
        timeline.push((item.id.clone(), start, end, dur));
        cursor = end;
    }
    timeline
}

/// Full create → schedule → play → log walk-through over a deterministic clock.
#[test]
fn full_lifecycle_create_schedule_play_log() {
    let base = fixed_base();

    // ── CREATE ────────────────────────────────────────────────────────────
    let playlist = create_playlist();
    assert_eq!(playlist.len(), SEGMENTS.len(), "CREATE: every item added");
    let expected_total: u64 = SEGMENTS.iter().map(|(_, _, s)| *s).sum();
    assert_eq!(
        playlist.total_duration,
        Duration::from_secs(expected_total),
        "CREATE: total_duration is the sum of item durations",
    );

    // ── SCHEDULE ──────────────────────────────────────────────────────────
    // Per-item timeline derived from cumulative durations: ordering preserved,
    // and item[i].start == base + sum(durations[0..i]).
    let timeline = compute_timeline(&playlist, base);
    assert_eq!(timeline.len(), SEGMENTS.len());
    let mut running = 0u64;
    for (idx, (tl_id, tl_start, tl_end, tl_dur)) in timeline.iter().enumerate() {
        let (seg_id, _title, secs) = SEGMENTS[idx];
        assert_eq!(tl_id.as_str(), seg_id, "SCHEDULE: order preserved at {idx}");
        assert_eq!(
            *tl_start,
            base + ChronoDuration::seconds(running as i64),
            "SCHEDULE: item {idx} start offset is cumulative",
        );
        assert_eq!(*tl_dur, Duration::from_secs(secs));
        running += secs;
        assert_eq!(
            *tl_end,
            base + ChronoDuration::seconds(running as i64),
            "SCHEDULE: item {idx} end offset is cumulative",
        );
    }
    // The last item ends exactly at base + total_duration.
    assert_eq!(
        timeline.last().expect("non-empty timeline").2,
        base + ChronoDuration::seconds(expected_total as i64),
    );

    // Hand the playlist to the real scheduling engine.
    let engine = ScheduleEngine::new();
    let mut sched_events = engine
        .take_event_receiver()
        .expect("receiver available once");
    engine
        .schedule_playlist(playlist.clone(), base, 10)
        .expect("clean schedule succeeds");
    assert_eq!(engine.get_scheduled().expect("read scheduled").len(), 1);

    // The engine reports the playlist active across its window, idle outside it
    // (end-exclusive).
    assert!(
        engine.get_active(base).expect("active query").is_some(),
        "SCHEDULE: active at window start",
    );
    let mid = base + ChronoDuration::seconds((expected_total / 2) as i64);
    assert!(
        engine.get_active(mid).expect("active query").is_some(),
        "SCHEDULE: active mid-window",
    );
    let after = base + ChronoDuration::seconds(expected_total as i64);
    assert!(
        engine.get_active(after).expect("active query").is_none(),
        "SCHEDULE: idle at exact window end",
    );
    assert!(
        engine
            .get_active(base - ChronoDuration::seconds(1))
            .expect("active query")
            .is_none(),
        "SCHEDULE: idle before window",
    );
    assert!(
        sched_events.try_recv().is_err(),
        "SCHEDULE: no events emitted for a conflict-free schedule",
    );

    // ── PLAY ──────────────────────────────────────────────────────────────
    // Real playout state machine: Stopped → Playing → … → Stopped.
    let playout = PlayoutEngine::new(PlayoutConfig::default());
    let mut play_events = playout
        .take_event_receiver()
        .expect("receiver available once");
    playout.load_playlist(playlist.clone()).expect("load ok");
    assert_eq!(playout.get_state().expect("state"), PlayoutState::Stopped);
    playout.play().expect("play ok");
    assert_eq!(playout.get_state().expect("state"), PlayoutState::Playing);

    // Real navigation: start on item 0, advance with `next()` to completion,
    // verifying the cursor steps forward by exactly one each time.
    let mut nav = playlist.clone();
    assert_eq!(nav.current_position, 0, "PLAY: cursor starts at first item");
    let mut play_order = vec![nav.current_item().expect("first item").id.clone()];
    loop {
        let next_id = match nav.next() {
            Some(item) => item.id.clone(),
            None => break,
        };
        play_order.push(next_id);
        assert_eq!(
            nav.current_position,
            play_order.len() - 1,
            "PLAY: cursor advances by exactly one per next()",
        );
    }
    let expected_order: Vec<String> = SEGMENTS
        .iter()
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    assert_eq!(
        play_order, expected_order,
        "PLAY: navigation visits every item once, in create order",
    );
    assert_eq!(
        nav.current_position,
        SEGMENTS.len() - 1,
        "PLAY: cursor rests on the last item at completion",
    );
    assert_eq!(nav.current_item().expect("last item").id, "seg-04");
    assert!(
        nav.next().is_none(),
        "PLAY: a completed non-looping playlist yields no further items",
    );

    // Drive the playout/tracker/log over the simulated clock (no sleeps).
    let tracker = MetadataTracker::new();
    let mut log = AsRunLog::new();
    for (idx, (seg_id, start, end, dur)) in timeline.iter().enumerate() {
        playout.emit_event(PlayoutEvent::ItemStarted {
            item_index: idx,
            timestamp: *start,
        });
        tracker.record_event(PlaybackEvent::ItemStarted {
            item_id: seg_id.clone(),
            timestamp: *start,
        });

        playout.emit_event(PlayoutEvent::ItemFinished {
            item_index: idx,
            timestamp: *end,
        });
        tracker.record_event(PlaybackEvent::ItemFinished {
            item_id: seg_id.clone(),
            timestamp: *end,
            success: true,
        });

        // ── LOG (as-run compliance entry) ──
        let mut entry =
            AsRunEntry::new(SEGMENTS[idx].1, seg_id.as_str(), CHANNEL_ID, *start, *start)
                .with_scheduled_duration(*dur)
                .with_actual_duration(*dur);
        entry.mark_completed();
        log.add_entry(entry);
    }
    playout.emit_event(PlayoutEvent::PlaylistFinished { timestamp: after });
    playout.stop().expect("stop ok");
    assert_eq!(playout.get_state().expect("state"), PlayoutState::Stopped);

    // The playout event stream contains one start+finish per item, two state
    // changes (play + stop), and exactly one playlist-finished event.
    let mut item_started = 0;
    let mut item_finished = 0;
    let mut state_changes = 0;
    let mut playlist_finished = 0;
    while let Ok(ev) = play_events.try_recv() {
        match ev {
            PlayoutEvent::ItemStarted { .. } => item_started += 1,
            PlayoutEvent::ItemFinished { .. } => item_finished += 1,
            PlayoutEvent::StateChanged { .. } => state_changes += 1,
            PlayoutEvent::PlaylistFinished { .. } => playlist_finished += 1,
            PlayoutEvent::Error { message } => panic!("unexpected error event: {message}"),
        }
    }
    assert_eq!(
        item_started,
        SEGMENTS.len(),
        "PLAY: one ItemStarted per item"
    );
    assert_eq!(
        item_finished,
        SEGMENTS.len(),
        "PLAY: one ItemFinished per item"
    );
    assert_eq!(state_changes, 2, "PLAY: StateChanged for play + stop");
    assert_eq!(playlist_finished, 1, "PLAY: a single PlaylistFinished");

    // ── LOG (assertions) ──────────────────────────────────────────────────
    // The as-run log records exactly what was played, in order, with matching
    // timing and durations.
    assert_eq!(
        log.len(),
        SEGMENTS.len(),
        "LOG: one as-run entry per played item"
    );
    let logged_ids: Vec<&str> = log
        .get_all_entries()
        .iter()
        .map(|e| e.content_id.as_str())
        .collect();
    let expected_ids: Vec<&str> = SEGMENTS.iter().map(|(id, _, _)| *id).collect();
    assert_eq!(
        logged_ids, expected_ids,
        "LOG: as-run identities + order match play order"
    );

    for (idx, entry) in log.get_all_entries().iter().enumerate() {
        let (seg_id, title, secs) = SEGMENTS[idx];
        assert_eq!(entry.content_id, seg_id);
        assert_eq!(entry.title, title);
        assert!(entry.completed, "LOG: each played item is marked completed");
        assert_eq!(entry.scheduled_duration, Duration::from_secs(secs));
        assert_eq!(entry.actual_duration, Duration::from_secs(secs));
        assert_eq!(
            entry.duration_variance(),
            Duration::ZERO,
            "LOG: ran to schedule"
        );
        assert_eq!(
            entry.actual_start, timeline[idx].1,
            "LOG: timing matches simulated clock"
        );
    }

    // Channel + time-range queries return the right subset.
    assert_eq!(
        log.get_entries_for_channel(CHANNEL_ID).len(),
        SEGMENTS.len()
    );
    assert_eq!(log.get_entries_for_channel("ch-other").len(), 0);
    assert_eq!(
        log.get_entries_in_range(&base, &(after + ChronoDuration::seconds(1)))
            .len(),
        SEGMENTS.len(),
        "LOG: range covering the window returns every entry",
    );

    // JSON export round-trips and mentions every played content id.
    let json = log.export_json().expect("json export");
    for (id, _, _) in SEGMENTS {
        assert!(json.contains(id), "LOG: json export mentions {id}");
    }

    // The real-time tracker history holds a start+finish per item.
    assert_eq!(
        tracker.event_count(),
        SEGMENTS.len() * 2,
        "LOG: tracker history has start+finish per item",
    );
    assert_eq!(tracker.tracked_item_count(), SEGMENTS.len());
    for (id, _, _) in SEGMENTS {
        let stats = tracker.get_item_stats(id).expect("item tracked");
        assert!(
            stats.start_time.is_some(),
            "LOG: tracked start time for {id}"
        );
        assert!(stats.end_time.is_some(), "LOG: tracked end time for {id}");
        assert_eq!(stats.error_count, 0);
        assert_eq!(stats.event_count, 2, "LOG: start+finish recorded for {id}");
    }
}

/// SCHEDULE: an overlapping playlist is rejected (with a `ConflictDetected`
/// event), while a back-to-back playlist is accepted and selected correctly.
#[test]
fn schedule_rejects_overlap_accepts_back_to_back() {
    let base = fixed_base();
    let engine = ScheduleEngine::new();
    let mut events = engine
        .take_event_receiver()
        .expect("receiver available once");

    let mut first = create_playlist();
    first.id = "pl-first".to_string();
    let total = first.total_duration;
    engine
        .schedule_playlist(first, base, 5)
        .expect("first schedule ok");

    // An overlapping playlist (starts mid-window) must conflict.
    let overlap = create_playlist();
    let mid = base + ChronoDuration::seconds(60);
    let err = engine
        .schedule_playlist(overlap, mid, 5)
        .expect_err("overlap must be rejected");
    assert!(
        matches!(err, PlaylistError::SchedulingConflict(_)),
        "overlap yields a SchedulingConflict",
    );
    assert_eq!(
        engine.get_scheduled().expect("read scheduled").len(),
        1,
        "the rejected schedule is not stored",
    );

    // The engine emitted a ConflictDetected event for the overlap.
    let mut saw_conflict = false;
    while let Ok(ev) = events.try_recv() {
        if let ScheduleEvent::ConflictDetected { .. } = ev {
            saw_conflict = true;
        }
    }
    assert!(saw_conflict, "engine emits ConflictDetected on overlap");

    // A back-to-back playlist starting exactly at the first window's end is OK.
    let mut next = create_playlist();
    next.id = "pl-second".to_string();
    let next_start = base + ChronoDuration::from_std(total).expect("total fits chrono");
    engine
        .schedule_playlist(next, next_start, 5)
        .expect("back-to-back schedule ok");
    assert_eq!(engine.get_scheduled().expect("read scheduled").len(), 2);

    // The engine selects the right playlist in each window.
    let in_first = engine
        .get_active(base + ChronoDuration::seconds(10))
        .expect("active query")
        .expect("a playlist is active in the first window");
    let in_second = engine
        .get_active(next_start + ChronoDuration::seconds(10))
        .expect("active query")
        .expect("a playlist is active in the second window");
    assert_eq!(in_first.playlist.id, "pl-first");
    assert_eq!(in_second.playlist.id, "pl-second");
    assert_eq!(in_first.start_time, base);
    assert_eq!(in_second.start_time, next_start);
}

/// PLAY + LOG: a scheduled live insert interrupts at its trigger time and is
/// recorded in the as-run log at exactly the right position.
#[test]
fn live_insert_appears_in_asrun_at_correct_position() {
    let base = fixed_base();
    let playlist = create_playlist();
    let timeline = compute_timeline(&playlist, base);

    // Breaking-news live insert scheduled to interrupt exactly at seg-02's start.
    let break_at = timeline[2].1;
    let mut manager = LiveInsertManager::new();
    let live = LiveInsert::new(
        "Breaking News",
        LiveSource::Ndi {
            name: "NEWS_CAM".to_string(),
        },
    )
    .with_start_time(break_at)
    .with_max_duration(Duration::from_secs(600))
    .with_priority(100)
    .as_interrupt();
    let live_id = live.id.clone();
    manager.add_insert(live);

    let mut log = AsRunLog::new();
    let mut as_run_order: Vec<String> = Vec::new();
    let mut live_logged = false;

    for (idx, (seg_id, start, _end, dur)) in timeline.iter().enumerate() {
        // Before each scheduled item, fire any live insert due at this clock.
        // Collect owned fields first so the immutable borrow ends before the
        // mutable `activate_insert` call.
        let due: Vec<(String, String, bool)> = manager
            .check_scheduled_inserts(start)
            .iter()
            .map(|i| (i.id.clone(), i.name.clone(), i.interrupt))
            .collect();
        if let Some((insert_id, insert_name, interrupt)) = due.into_iter().next() {
            assert!(interrupt, "the live insert is flagged to interrupt");
            manager.activate_insert(&insert_id);
            let active = manager
                .get_highest_priority_active()
                .expect("an active insert after activation");
            assert_eq!(active.priority, 100, "highest-priority active insert wins");

            let mut live_entry = AsRunEntry::new(
                insert_name.as_str(),
                insert_id.as_str(),
                CHANNEL_ID,
                *start,
                *start,
            )
            .with_actual_duration(Duration::from_secs(600));
            live_entry.mark_completed();
            live_entry.add_metadata("source", "live");
            log.add_entry(live_entry);
            as_run_order.push(format!("LIVE:{insert_id}"));
            live_logged = true;
        }

        let mut entry =
            AsRunEntry::new(SEGMENTS[idx].1, seg_id.as_str(), CHANNEL_ID, *start, *start)
                .with_scheduled_duration(*dur)
                .with_actual_duration(*dur);
        entry.mark_completed();
        log.add_entry(entry);
        as_run_order.push(seg_id.clone());
    }

    assert!(live_logged, "the scheduled live insert fired");
    // The as-run order has the live insert spliced between seg-01 and seg-02.
    let expected_order = vec![
        "seg-00".to_string(),
        "seg-01".to_string(),
        format!("LIVE:{live_id}"),
        "seg-02".to_string(),
        "seg-03".to_string(),
        "seg-04".to_string(),
    ];
    assert_eq!(
        as_run_order, expected_order,
        "live insert appears at the correct as-run position",
    );
    // The log carries one extra entry (the live insert) beyond the 5 segments,
    // sitting at index 2 and tagged as live.
    assert_eq!(log.len(), SEGMENTS.len() + 1);
    let live_entry = &log.get_all_entries()[2];
    assert_eq!(live_entry.content_id, live_id);
    assert_eq!(live_entry.actual_start, break_at);
    assert_eq!(
        live_entry.metadata.get("source").map(String::as_str),
        Some("live")
    );

    // The insert fired exactly once: nothing pending past its window.
    let after = timeline.last().expect("non-empty timeline").2;
    assert!(
        manager.check_scheduled_inserts(&after).is_empty(),
        "no further scheduled inserts remain after the window",
    );
}

/// PLAY (queue manager): the play queue drains a front-inserted priority item
/// first, then the remaining segments in FIFO order, to completion.
#[test]
fn play_queue_drains_segments_in_order() {
    use oximedia_playlist::queue_manager::PlayQueue;

    let mut queue = PlayQueue::new();
    assert!(queue.is_empty(), "queue starts empty");
    for (id, _, _) in SEGMENTS {
        queue
            .enqueue(format!("content/{id}.mxf"))
            .expect("unbounded enqueue always succeeds");
    }
    assert_eq!(queue.len(), SEGMENTS.len());

    // A late-breaking item jumps to the front (high-priority insertion).
    queue
        .enqueue_front("content/breaking.mxf")
        .expect("front enqueue succeeds");
    assert_eq!(
        queue.peek().expect("front entry").uri,
        "content/breaking.mxf",
        "front-inserted item is at the head",
    );

    // Drain the queue to completion, capturing the play order.
    let mut drained = Vec::new();
    while let Some(entry) = queue.dequeue_front() {
        drained.push(entry.uri);
    }
    assert!(queue.is_empty(), "queue is fully drained");

    let mut expected: Vec<String> = vec!["content/breaking.mxf".to_string()];
    expected.extend(
        SEGMENTS
            .iter()
            .map(|(id, _, _)| format!("content/{id}.mxf")),
    );
    assert_eq!(
        drained, expected,
        "queue drains the front-insert first, then the segments in FIFO order",
    );
}
