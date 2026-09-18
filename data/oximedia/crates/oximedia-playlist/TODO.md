# oximedia-playlist TODO

## Current Status
- 39 modules covering broadcast playlist, scheduling, automation, transitions, EPG, and more
- Key features: PlayoutEngine, ClockSync, ScheduleEngine, crossfade, commercial breaks (SCTE-35)
- Modules include: automation, backup, clock, commercial, continuity, crossfade, epg, gap_filler, interstitial, live, multichannel, playlist_diff/export/filter/health/merge/priority/rules/segment/stats/sync/tempo/validator, queue_manager, recommendation_engine, repeat_policy, schedule, secondary, shuffle, smart_play, track_metadata, track_order, transition
- Dependencies: oximedia-core, oximedia-timecode, tokio, serde, chrono, tracing

## Enhancements
- [x] Add undo/redo support for playlist editing operations in `playlist` module (verified 2026-05-16; src/playlist/undo.rs, src/playlist_history.rs:23 PlaylistEditOperation undo/redo:336)
- [x] Implement playlist versioning with diff tracking in `playlist_diff` (verified 2026-05-16; src/playlist_diff.rs:75 PlaylistDiff, src/playlist_archive.rs:145 PlaylistArchive versioned snapshots)
- [x] Extend `crossfade` module with logarithmic and equal-power crossfade curves (verified 2026-05-16; src/crossfade.rs:70 Logarithmic/EqualPower crossfade variants, test:366)
- [x] Add playlist validation for total duration vs. scheduled time window in `playlist_validator` (verified 2026-05-16; src/playlist_validator.rs:127 max_total_duration, test_total_duration_violation:327)
- [x] Implement weighted shuffle algorithm in `shuffle` module to bias by genre/mood (verified 2026-05-16; src/shuffle.rs:63 WeightedShuffler, weighted_shuffle fn:89)
- [x] Enhance `gap_filler` with content-aware filler selection (match genre/mood of surrounding items) (implemented 2026-06-01: src/gap_filler.rs — `tags: Vec<String>` on `FillerItem`, `GapContext`, `fill_gap_with_context`, `jaccard_overlap`; tests: test_gap_filler_genre_preference, test_gap_filler_empty_context_uses_priority, test_jaccard_*)
  - **Goal:** Score filler candidates by genre/mood overlap with surrounding playlist items before the existing priority/duration sort.
  - **Design:** `src/gap_filler.rs:71` `FillerItem` has only `id/duration_ms/category/priority/recent_play_count/enabled` — no genre or mood fields. Add `tags: Vec<String>` (or separate `genre`/`mood` fields) to `FillerItem`. Extend `select_fillers`/`fill_gap` at :256/:240 to accept the surrounding items' tag context and compute a weighted Jaccard score (tag-overlap / tag-union), boosting candidates with higher overlap before applying the existing priority sort.
  - **Files:** `src/gap_filler.rs`, `TODO.md`.
  - **Tests:** filler selection prefers a same-genre/mood candidate over a higher-priority but genre-mismatched one (given appropriate weights); empty-tag fallback returns original priority order; Jaccard = 1.0 for identical tag sets, 0.0 for disjoint.
  - **Risk:** backward compatibility for callers that don't provide context tags — default to priority-only (current behavior) when context is empty.
- [x] Add conflict resolution strategies to `playlist_merge` (priority, interleave, overlap trim) (verified 2026-05-16; src/playlist_merge.rs:98 PlaylistMergeEngine, MergeStrategy::PriorityMerge/Interleave:285)
- [x] Extend `commercial` module with SCTE-35 splice_insert and time_signal command generation (verified 2026-05-16; src/commercial/scte35.rs:3 SpliceInsert:161, TimeSignal:179)

## New Features
- [x] Add M3U8/M3U playlist import/export support (parse and generate) (verified 2026-05-16; src/m3u.rs M3uPlaylist parser/writer, src/m3u8.rs M3U8 format)
- [x] Implement playlist archival with compression for long-term storage (verified 2026-05-16; src/playlist_archive.rs:145 PlaylistArchive versioned immutable snapshots)
- [x] Add real-time playlist notification system (webhook/callback on item transitions) (verified 2026-05-16; src/playlist_notify.rs:241 notify, PlaylistEventKind:408 lines)
- [x] Implement playlist rotation policies (daily, weekly, seasonal schedules) (verified 2026-05-16; src/playlist_rotation.rs:18 RotationStrategy, RotationPool:102, Daypart:111)
- [x] Add A/B testing support for playlist ordering strategies (implemented 2026-06-01: src/ab_test.rs — `AbExperiment`, `OrderingStrategy`, `MetricAccumulator`, FNV-1a deterministic assignment; tests: test_ab_assignment_is_deterministic, test_ab_traffic_split_approximately_uniform, test_ab_metric_accumulator)
  - **Goal:** Deterministic hash-bucket A/B experiment assignment for ordering strategy evaluation.
  - **Design:** Add new module `src/ab_test.rs`: `AbExperiment { id: String, variants: Vec<OrderingStrategy>, traffic_split: Vec<f64> }` with `assign_variant(user_id: &str) -> &OrderingStrategy` using a deterministic FNV/wyhash (or `DefaultHasher` seeded on `user_id + experiment_id`) modulo bucket count for reproducible assignment. Include per-variant `MetricAccumulator` for recording impressions/completions. Wire into `src/track_order.rs` and/or `src/recommendation_engine.rs:174`. Register in `src/lib.rs`. Keep file < 2000 lines.
  - **Files:** `src/ab_test.rs` (new), `src/lib.rs` (register), `TODO.md`.
  - **Tests:** assignment is deterministic across calls (same user_id → same variant); traffic split is approximately uniform over many user_ids; `MetricAccumulator` records correctly.
  - **Risk:** hash function must be stable across Rust versions — use a fixed algorithm, not `std::hash::DefaultHasher` (non-deterministic); keep module < 2000 lines.
- [x] Implement playlist analytics dashboard data export (JSON/CSV metrics) (implemented 2026-06-01: src/playlist_stats.rs — `to_json()`, `to_csv()`, `PlaylistStatsExport`; tests: test_to_json_roundtrip, test_to_csv_format, test_export_empty_stats)
  - **Goal:** Let callers export per-track and summary playlist statistics as JSON or CSV.
  - **Design:** Add `to_json(&self) -> Result<String, serde_json::Error>` and `to_csv(&self) -> String` methods to `PlaylistStats` near :319. `to_json` uses `serde_json::to_string_pretty` (already a dep) over the per-track `PlayStats` (plays, completion_rate, skips) + `PlaylistSummaryStats` aggregates. `to_csv` uses manual `writeln!` formatting: one header row + one data row per track + a summary footer row.
  - **Files:** `src/playlist_stats.rs`, `TODO.md`.
  - **Tests:** `to_json` output round-trips through `serde_json::from_str` and recovers all track stats; `to_csv` output has the correct column count per row; empty stats exports a valid header-only output.
  - **Risk:** serde_json is already a dep (`Cargo.toml`) — no new dep needed; CSV escaping for track names with commas.
- [x] Add multi-language EPG metadata support in `epg` module (implemented 2026-06-01: src/epg/generate.rs — `title: BTreeMap<String,String>`, `description: BTreeMap<String,String>`, `with_title`, `title_for` fallback; src/epg/xmltv.rs — per-language `<title lang="...">` / `<desc lang="...">` emission; tests: test_epg_multilang_xmltv_output, test_epg_title_for_fallback)
  - **Goal:** Allow EPG entries to carry title/description in multiple languages and emit XMLTV `lang` attributes.
  - **Design:** `src/epg/generate.rs:10` `ProgramEntry` has `title: String` / `description: Option<String>` — no language key. Change to `title: BTreeMap<LangCode, String>` and `description: BTreeMap<LangCode, String>` (where `LangCode = String` or a newtype). Update `EpgGenerator::generate_from_playlist` at :148 to populate per-language titles. In `src/epg/xmltv.rs:70` the `<title>` writer currently emits no `lang=` attribute — update to emit `<title lang="en">...</title>` (and `<desc lang="...">`) per XMLTV spec.
  - **Files:** `src/epg/generate.rs`, `src/epg/xmltv.rs`, `TODO.md`.
  - **Tests:** EPG entry with 2 language titles emits 2 `<title lang=...>` elements in XMLTV output; single-language entry is backward compatible; round-trip parse extracts the language tag.
  - **Risk:** struct change ripples to all `ProgramEntry { title: ... }` construction sites in `generate_from_playlist` and tests — update all call sites.

## Performance
- [x] Cache computed playlist statistics in `playlist_stats` to avoid recalculation (implemented 2026-06-01: src/playlist_stats.rs — `cached_summary: Option<PlaylistSummaryStats>`, `is_dirty: bool`, `summary()` method; invalidates on `record_play`/`reset`; tests: test_stats_cache_same_as_fresh, test_stats_cache_invalidates_on_record_play)
  - **Goal:** Amortize `PlaylistSummaryStats::from_tracks` recompute across many reads by caching and invalidating on mutation.
  - **Design:** `src/playlist_stats.rs:293` `PlaylistStats` has no cache field; :105 `from_tracks` recomputes from scratch each call. Add `cached_summary: Option<PlaylistSummaryStats>` + `dirty: bool` to `PlaylistStats`. Invalidate (`dirty = true`) on `record_play` at :307 (and any other mutation methods). On `summary()` (or equivalent getter), recompute only if `dirty`, store result, clear flag.
  - **Files:** `src/playlist_stats.rs`, `TODO.md`.
  - **Tests:** `summary()` returns the same result as a fresh `from_tracks` after mutation; calling `summary()` twice without mutation returns the same cached object (no recompute); `record_play` correctly invalidates; empty playlist handled.
  - **Risk:** cache invalidation must cover all mutation methods — enumerate them exhaustively; test the dirty-flag reset path.
- [x] Optimize `recommendation_engine` similarity computation with pre-computed feature vectors (implemented 2026-06-01: src/recommendation_engine.rs — `FeatureVector { genre_bits, popularity_norm, rating_norm }`, `FeatureVector::from_item`, `compute_with_vector`; tests: test_feature_vector_scoring_matches_string_scoring, test_feature_vector_popularity_in_range)
  - **Goal:** Replace per-call string-comparison genre scoring with a precomputed numeric vector dot product.
  - **Design:** `src/recommendation_engine.rs:113` `RecommendationScore::compute` and :131 `genre_boost` recompute `String::to_lowercase()+contains()` per item per call. Precompute a `FeatureVector { genre_bits: u64, popularity_norm: f32, rating_norm: f32 }` per `PlaylistItem` once at ingest (or lazily on first access via a cache field). Score via cheap bitset-overlap count + weighted dot product instead of string scanning.
  - **Files:** `src/recommendation_engine.rs`, `TODO.md`.
  - **Tests:** precomputed-vector scoring yields the same top-k ordering as the string-comparison path on a fixture set; feature vectors are stable across multiple calls; `popularity_norm` and `rating_norm` are in [0,1].
  - **Risk:** feature-vector cache must be invalidated if item metadata changes — scope to immutable items or add a version/dirty bit.
- [x] Use interval trees in `schedule` for O(log n) overlap detection instead of linear scan (implemented 2026-05-31: src/interval_tree.rs — augmented red-black BST (CLRS §14.3), IntervalTree<K,V> with insert/remove/query_overlapping/query_all_overlapping/iter_sorted; ScheduleIntervalIndex domain adapter for DateTime<Utc> via epoch-ms; 23 tests)
- [x] Batch database writes in `play_history` for high-throughput playout logging (implemented 2026-06-06: src/play_history.rs — `PlayHistory::batch_record(&[(u64,u64,bool)])` vectorized logging; seeds running per-track counts in one O(n) pass, appends N events with correct play_count, evicts overflow once at the end; PlayEvent gains PartialEq/Eq; documented equivalence boundary vs N record_play calls; tests: test_batch_record_equals_n_single_writes_unbounded, test_batch_record_eviction_window_matches, test_batch_record_empty_is_noop, test_batch_record_appends_to_existing_events)

## Testing
- [x] Add integration tests for full playlist lifecycle (create, schedule, play, log) (implemented 2026-06-22: tests/lifecycle_integration.rs — 4 deterministic end-to-end tests over a fixed simulated clock: full_lifecycle_create_schedule_play_log (Playlist/PlaylistItem CREATE -> ScheduleEngine timeline + active-window SCHEDULE -> PlayoutEngine state machine + Playlist cursor PLAY -> AsRunLog/MetadataTracker LOG), schedule_rejects_overlap_accepts_back_to_back (ScheduleEvent::ConflictDetected + back-to-back accept), live_insert_appears_in_asrun_at_correct_position (LiveInsertManager interrupt spliced into as-run at the right index), play_queue_drains_segments_in_order (PlayQueue FIFO + front-insert); no crate bug found, all assertions pass against real APIs)
- [x] Test `crossfade` module with edge cases (zero-duration items, single-item playlists) (implemented 2026-06-06: src/crossfade.rs — tests: test_schedule_gapless_single_item, test_schedule_gapless_single_item_with_crossfade_ignored, test_zero_duration_entry_no_nan, test_fade_curves_no_nan_at_boundaries — all 6 FadeCurve variants × t∈{0,0.5,1} finite + endpoint conventions)
- [x] Add stress tests for `multichannel` module with 50+ simultaneous channels (implemented 2026-06-06: tests/wave27_multichannel_stress.rs — test_50_channels_concurrent_start: Arc<ChannelManager>, 50 concurrent tokio tasks each add_channel+load_playlist+start_channel, asserts channel_count()==50, get_all_channels().len()==50, unique ids, no lock poisoning)
- [x] Test `clock` synchronization accuracy under simulated clock drift (implemented 2026-06-06: src/clock/offset.rs + src/clock/sync.rs — tests: test_drift_plus_100ms_resync, test_drift_minus_100ms_resync (literal fixed UTC instant, exact ±100ms), test_clocksync_offset_roundtrip (set_offset/get_offset round-trip + synchronize() populates last_sync_time))

## Documentation
- [ ] Document SCTE-35 integration workflow with real-world examples
- [ ] Add EPG generation tutorial with XMLTV output format specification
- [ ] Document playlist state machine transitions (idle -> scheduled -> playing -> completed)
