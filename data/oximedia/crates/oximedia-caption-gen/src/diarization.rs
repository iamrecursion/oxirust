//! Speaker diarization metadata: speaker turns, statistics, crosstalk detection,
//! and assigning speakers to caption blocks.
//!
//! ## Diarization Workflow
//!
//! A typical ASR pipeline produces a stream of text segments together with
//! speaker labels ("who spoke when").  This module consumes that output and
//! makes it easy to work with speaker identity throughout the caption
//! generation pipeline.
//!
//! 1. **Intern speaker labels** — Use [`SpeakerLabelPool`] to deduplicate
//!    speaker-name strings.  Interned labels are `Arc<str>` values; many
//!    `Speaker` records that share the same name incur only one heap
//!    allocation.
//!
//! 2. **Build speaker records** — Construct [`Speaker`] values for each unique
//!    speaker, setting an optional display name, gender, and BCP-47 language
//!    tag.
//!
//! 3. **Collect speaker turns** — Push [`SpeakerTurn`] values into a
//!    [`DiarizationResult`].  Each turn records the speaker id and a
//!    `[start_ms, end_ms)` interval.
//!
//! 4. **Merge short pauses** — Call [`merge_consecutive_turns`] (500 ms gap)
//!    or [`merge_consecutive_turns_with_gap`] with a custom threshold to
//!    collapse breath-pauses between turns by the same speaker.
//!
//! 5. **Analyse the result** — Use [`dominant_speaker`], [`speaker_stats`],
//!    and [`format_speaker_label`] to build display-ready metadata.
//!
//! 6. **Assign to caption blocks** — Call [`assign_speakers_to_blocks`] to
//!    set `CaptionBlock::speaker_id` based on the turn with the greatest time
//!    overlap for each block.  This step bridges diarization output with the
//!    [`crate::alignment`] module.
//!
//! ## Doctest: Building and querying a `DiarizationResult`
//!
//! ```rust
//! use oximedia_caption_gen::diarization::{
//!     SpeakerLabelPool, Speaker, SpeakerGender, SpeakerTurn,
//!     DiarizationResult, merge_consecutive_turns,
//!     dominant_speaker, speaker_stats, format_speaker_label,
//! };
//! use std::collections::HashMap;
//!
//! // 1. Intern display labels.
//! let pool = SpeakerLabelPool::new();
//! let _alice_arc = pool.intern("Alice");
//! let _bob_arc   = pool.intern("Bob");
//!
//! // 2. Create Speaker records.
//! let alice = Speaker {
//!     id: 0,
//!     name: Some("Alice".to_string()),
//!     gender: Some(SpeakerGender::Female),
//!     language: Some("en-US".to_string()),
//! };
//! let bob = Speaker {
//!     id: 1,
//!     name: Some("Bob".to_string()),
//!     gender: Some(SpeakerGender::Male),
//!     language: Some("en-GB".to_string()),
//! };
//!
//! // 3. Build a DiarizationResult with speaker turns.
//! let mut result = DiarizationResult::new();
//! result.speakers.insert(alice.id, alice.clone());
//! result.speakers.insert(bob.id, bob.clone());
//! result.turns.push(SpeakerTurn { speaker_id: 0, start_ms:     0, end_ms: 3_000 });
//! result.turns.push(SpeakerTurn { speaker_id: 0, start_ms: 3_200, end_ms: 5_000 });
//! result.turns.push(SpeakerTurn { speaker_id: 1, start_ms: 5_500, end_ms: 9_000 });
//!
//! // 4. Merge consecutive turns from the same speaker (500 ms gap threshold).
//! let merged = merge_consecutive_turns(&result);
//! // Alice's two turns are within 500 ms of each other and get merged.
//! assert_eq!(merged.len(), 2);
//!
//! // 5. Query dominant speaker and per-speaker statistics.
//! //    Alice: 0–3 000 ms + 3 200–5 000 ms = 4 800 ms total
//! //    Bob  : 5 500–9 000 ms               = 3 500 ms total
//! let stats = speaker_stats(&result);
//! let alice_time = stats[&0].total_time_ms;
//! let bob_time   = stats[&1].total_time_ms;
//! assert!(alice_time > bob_time, "Alice spoke more ({alice_time}ms vs {bob_time}ms)");
//! let dom = dominant_speaker(&result);
//! assert_eq!(dom, Some(0)); // Alice is the dominant speaker
//!
//! // 6. Format a display label.
//! let label = format_speaker_label(&result.speakers[&0]);
//! assert_eq!(label, "Alice");
//! let label_unknown = format_speaker_label(&Speaker { id: 99, name: None,
//!                                                     gender: None, language: None });
//! assert_eq!(label_unknown, "Speaker 99");
//! ```
//!
//! ## Crosstalk Detection
//!
//! [`CrosstalkDetector`] finds overlapping turns — sections where two speakers
//! are active at the same time.  By default any strict overlap is reported.
//! Use [`CrosstalkDetector::with_overlap_tolerance`] to ignore tiny boundary
//! artefacts (e.g., `0.05` means the overlap must cover at least 5 % of the
//! shorter turn before it is reported).
//!
//! ```rust
//! use oximedia_caption_gen::diarization::{CrosstalkDetector, DiarizationResult, SpeakerTurn};
//!
//! let mut result = DiarizationResult::new();
//! result.turns.push(SpeakerTurn { speaker_id: 0, start_ms: 0,   end_ms: 4_000 });
//! result.turns.push(SpeakerTurn { speaker_id: 1, start_ms: 3_000, end_ms: 7_000 });
//!
//! let overlaps = CrosstalkDetector::new().detect(&result);
//! assert_eq!(overlaps.len(), 1);
//! let (a, b) = &overlaps[0];
//! assert!(a.start_ms <= b.start_ms);
//! ```

use crate::alignment::CaptionBlock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ─── Speaker label interning ──────────────────────────────────────────────────

/// A simple string intern pool for `Speaker` display labels.
///
/// By interning speaker names, multiple `Speaker` structs that reference the
/// same name share a single allocation rather than duplicating the string.
///
/// The pool is thread-safe and can be shared across threads via an `Arc`.
#[derive(Debug, Default)]
pub struct SpeakerLabelPool {
    inner: Mutex<HashMap<String, Arc<str>>>,
}

impl SpeakerLabelPool {
    /// Create a new empty intern pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `label` and return an `Arc<str>` pointing to the unique copy.
    ///
    /// If `label` has been interned before, returns the existing `Arc`.
    /// Otherwise inserts a new entry.
    pub fn intern(&self, label: &str) -> Arc<str> {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = map.get(label) {
            return Arc::clone(existing);
        }
        let interned: Arc<str> = Arc::from(label);
        map.insert(label.to_string(), Arc::clone(&interned));
        interned
    }

    /// Return the number of unique labels currently interned.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Return `true` if no labels have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Gender of a speaker (best-effort; may not be known).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeakerGender {
    Male,
    Female,
    Other,
    Unknown,
}

/// Metadata for a single speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct Speaker {
    /// Unique numeric identifier (matches `SpeakerTurn::speaker_id`).
    pub id: u8,
    /// Optional display name (e.g., "Dr. Smith").
    pub name: Option<String>,
    pub gender: Option<SpeakerGender>,
    /// BCP-47 language tag, e.g. `"en-US"`.
    pub language: Option<String>,
}

/// A contiguous time span during which a single speaker is active.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    pub speaker_id: u8,
    pub start_ms: u64,
    pub end_ms: u64,
}

impl SpeakerTurn {
    /// Duration of this turn in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Whether this turn overlaps with another (strict overlap — shared boundary
    /// is not considered an overlap).
    pub fn overlaps_with(&self, other: &SpeakerTurn) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

/// Complete diarization result for a piece of content.
#[derive(Debug, Clone)]
pub struct DiarizationResult {
    /// Map from speaker id to speaker metadata.
    pub speakers: HashMap<u8, Speaker>,
    /// Speaker turns in temporal order.
    pub turns: Vec<SpeakerTurn>,
}

impl DiarizationResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self {
            speakers: HashMap::new(),
            turns: Vec::new(),
        }
    }

    /// Total speaking time across all turns.
    pub fn total_speech_ms(&self) -> u64 {
        self.turns.iter().map(|t| t.duration_ms()).sum()
    }
}

// ─── Turn merging ─────────────────────────────────────────────────────────────

/// Merge consecutive turns from the same speaker when the gap between them is
/// less than `max_gap_ms` (default 500 ms when called via the older API).
///
/// Turns are expected to be approximately time-ordered; this function sorts
/// them by `start_ms` before merging.
pub fn merge_consecutive_turns(result: &DiarizationResult) -> Vec<SpeakerTurn> {
    merge_consecutive_turns_with_gap(result, 500)
}

/// Merge consecutive turns from the same speaker when the gap between them is
/// less than the configurable `max_gap_ms` threshold.
///
/// Setting `max_gap_ms = 0` disables merging entirely (every turn is returned
/// as-is after sorting).  Very large values will merge almost everything.
///
/// Turns are sorted by `start_ms` before processing.
pub fn merge_consecutive_turns_with_gap(
    result: &DiarizationResult,
    max_gap_ms: u64,
) -> Vec<SpeakerTurn> {
    let mut sorted = result.turns.clone();
    sorted.sort_by_key(|t| t.start_ms);

    if sorted.is_empty() {
        return Vec::new();
    }

    let mut merged: Vec<SpeakerTurn> = Vec::new();

    for turn in sorted {
        if let Some(last) = merged.last_mut() {
            let gap = turn.start_ms.saturating_sub(last.end_ms);
            if last.speaker_id == turn.speaker_id && gap < max_gap_ms {
                // Extend the previous turn.
                last.end_ms = last.end_ms.max(turn.end_ms);
                continue;
            }
        }
        merged.push(turn);
    }

    merged
}

// ─── Speaker statistics ───────────────────────────────────────────────────────

/// Aggregate statistics for a single speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerStats {
    pub total_time_ms: u64,
    pub turn_count: u32,
    pub avg_turn_ms: u64,
}

/// Compute per-speaker statistics from a diarization result.
pub fn speaker_stats(result: &DiarizationResult) -> HashMap<u8, SpeakerStats> {
    let mut totals: HashMap<u8, (u64, u32)> = HashMap::new();

    for turn in &result.turns {
        let entry = totals.entry(turn.speaker_id).or_insert((0, 0));
        entry.0 += turn.duration_ms();
        entry.1 += 1;
    }

    totals
        .into_iter()
        .map(|(id, (total_ms, count))| {
            let avg = if count > 0 {
                total_ms / u64::from(count)
            } else {
                0
            };
            (
                id,
                SpeakerStats {
                    total_time_ms: total_ms,
                    turn_count: count,
                    avg_turn_ms: avg,
                },
            )
        })
        .collect()
}

/// Return the id of the speaker with the most total speaking time, or `None`
/// if there are no turns.
pub fn dominant_speaker(result: &DiarizationResult) -> Option<u8> {
    let stats = speaker_stats(result);
    stats
        .into_iter()
        .max_by_key(|(_, s)| s.total_time_ms)
        .map(|(id, _)| id)
}

// ─── Caption block assignment ─────────────────────────────────────────────────

/// Assign speaker ids to caption blocks based on which speaker turn overlaps
/// most with each block.
///
/// For each block, the speaker whose turn has the greatest overlap (in ms) with
/// the block's time range is assigned.  If no turn overlaps, the block's
/// `speaker_id` is left unchanged.
pub fn assign_speakers_to_blocks(blocks: &mut Vec<CaptionBlock>, diarization: &DiarizationResult) {
    for block in blocks.iter_mut() {
        let best = diarization
            .turns
            .iter()
            .filter_map(|turn| {
                let overlap_start = block.start_ms.max(turn.start_ms);
                let overlap_end = block.end_ms.min(turn.end_ms);
                if overlap_end > overlap_start {
                    Some((turn.speaker_id, overlap_end - overlap_start))
                } else {
                    None
                }
            })
            .max_by_key(|(_, overlap)| *overlap);

        if let Some((speaker_id, _)) = best {
            block.speaker_id = Some(speaker_id);
        }
    }
}

// ─── Speaker label formatting ─────────────────────────────────────────────────

/// Format a display label for a speaker.
///
/// Uses the speaker's name if available, otherwise falls back to
/// `"Speaker {id}"`.
pub fn format_speaker_label(speaker: &Speaker) -> String {
    match &speaker.name {
        Some(name) => name.clone(),
        None => format!("Speaker {}", speaker.id),
    }
}

// ─── Crosstalk detection ──────────────────────────────────────────────────────

/// Detects overlapping speaker turns in a diarization result.
///
/// Can be configured with a minimum overlap tolerance as a percentage of the
/// shorter turn's duration.  This avoids false positives from tiny boundary
/// overlaps that are artefacts of the diarization pipeline.
pub struct CrosstalkDetector {
    /// Minimum overlap as a fraction [0.0, 1.0] of the shorter turn's duration.
    /// Pairs whose overlap is less than this fraction are not reported.
    /// A value of `0.0` reports any strict overlap (the original behaviour).
    pub min_overlap_fraction: f32,
}

impl CrosstalkDetector {
    /// Create a detector with zero tolerance (reports any strict overlap).
    pub fn new() -> Self {
        Self {
            min_overlap_fraction: 0.0,
        }
    }

    /// Create a detector with the given minimum overlap tolerance.
    ///
    /// `min_overlap_fraction` is the minimum overlap as a fraction of the
    /// shorter turn's duration (range [0.0, 1.0]).  For example, `0.10` means
    /// that the overlap must be at least 10% of the shorter turn.
    pub fn with_overlap_tolerance(min_overlap_fraction: f32) -> Self {
        Self {
            min_overlap_fraction: min_overlap_fraction.clamp(0.0, 1.0),
        }
    }

    /// Find all pairs of turns that overlap in time (i.e., simultaneous speech).
    ///
    /// Returns pairs `(a, b)` where `a.start_ms <= b.start_ms`.
    ///
    /// Pairs whose overlap is less than `min_overlap_fraction` of the shorter
    /// turn's duration are excluded.
    pub fn find_overlapping_turns(result: &DiarizationResult) -> Vec<(SpeakerTurn, SpeakerTurn)> {
        Self::new().detect(result)
    }

    /// Find overlapping turns using the configured tolerance.
    pub fn detect(&self, result: &DiarizationResult) -> Vec<(SpeakerTurn, SpeakerTurn)> {
        let turns = &result.turns;
        let mut overlapping: Vec<(SpeakerTurn, SpeakerTurn)> = Vec::new();

        for i in 0..turns.len() {
            for j in (i + 1)..turns.len() {
                if !turns[i].overlaps_with(&turns[j]) {
                    continue;
                }

                // Compute actual overlap duration.
                let overlap_start = turns[i].start_ms.max(turns[j].start_ms);
                let overlap_end = turns[i].end_ms.min(turns[j].end_ms);
                if overlap_end <= overlap_start {
                    continue;
                }
                let overlap_ms = overlap_end - overlap_start;

                // Apply tolerance filter.
                if self.min_overlap_fraction > 0.0 {
                    let shorter_ms = turns[i].duration_ms().min(turns[j].duration_ms());
                    if shorter_ms == 0 {
                        continue;
                    }
                    let fraction = overlap_ms as f32 / shorter_ms as f32;
                    if fraction < self.min_overlap_fraction {
                        continue;
                    }
                }

                let (a, b) = if turns[i].start_ms <= turns[j].start_ms {
                    (turns[i].clone(), turns[j].clone())
                } else {
                    (turns[j].clone(), turns[i].clone())
                };
                overlapping.push((a, b));
            }
        }

        overlapping
    }
}

impl Default for CrosstalkDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Voice activity ratio ─────────────────────────────────────────────────────

/// Compute the fraction of `total_duration_ms` during which at least one
/// speaker is active.
///
/// Overlapping turns are counted only once (union of intervals).
/// Returns 0.0 if `total_duration_ms` is 0.
pub fn voice_activity_ratio(result: &DiarizationResult, total_duration_ms: u64) -> f32 {
    if total_duration_ms == 0 {
        return 0.0;
    }

    // Compute the union length of all turn intervals.
    let mut intervals: Vec<(u64, u64)> = result
        .turns
        .iter()
        .map(|t| (t.start_ms, t.end_ms))
        .collect();
    intervals.sort_by_key(|&(s, _)| s);

    let mut union_ms: u64 = 0;
    let mut cursor: u64 = 0;

    for (start, end) in intervals {
        let effective_start = start.max(cursor);
        if end > effective_start {
            union_ms += end - effective_start;
            cursor = end;
        }
    }

    (union_ms as f32 / total_duration_ms as f32).min(1.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_speaker(id: u8, name: Option<&str>) -> Speaker {
        Speaker {
            id,
            name: name.map(|s| s.to_string()),
            gender: None,
            language: None,
        }
    }

    fn make_turn(speaker_id: u8, start_ms: u64, end_ms: u64) -> SpeakerTurn {
        SpeakerTurn {
            speaker_id,
            start_ms,
            end_ms,
        }
    }

    fn make_block(id: u32, start_ms: u64, end_ms: u64) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec!["text".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    fn simple_result() -> DiarizationResult {
        let mut r = DiarizationResult::new();
        r.speakers.insert(1, make_speaker(1, Some("Alice")));
        r.speakers.insert(2, make_speaker(2, None));
        r.turns = vec![
            make_turn(1, 0, 3000),
            make_turn(2, 3000, 6000),
            make_turn(1, 6500, 9000),
        ];
        r
    }

    // --- SpeakerTurn ---

    #[test]
    fn speaker_turn_duration() {
        let t = make_turn(1, 1000, 4000);
        assert_eq!(t.duration_ms(), 3000);
    }

    #[test]
    fn speaker_turn_overlap_true() {
        let a = make_turn(1, 0, 2000);
        let b = make_turn(2, 1000, 3000);
        assert!(a.overlaps_with(&b));
    }

    #[test]
    fn speaker_turn_overlap_false_adjacent() {
        let a = make_turn(1, 0, 1000);
        let b = make_turn(2, 1000, 2000);
        // Shared boundary only — not a strict overlap.
        assert!(!a.overlaps_with(&b));
    }

    #[test]
    fn speaker_turn_overlap_false_separate() {
        let a = make_turn(1, 0, 1000);
        let b = make_turn(2, 2000, 3000);
        assert!(!a.overlaps_with(&b));
    }

    // --- merge_consecutive_turns ---

    #[test]
    fn merge_consecutive_empty() {
        let r = DiarizationResult::new();
        assert!(merge_consecutive_turns(&r).is_empty());
    }

    #[test]
    fn merge_consecutive_same_speaker_small_gap() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 1000), make_turn(1, 1200, 2000)];
        // Gap = 200ms < 500ms → merge.
        let result = merge_consecutive_turns(&r);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 2000);
    }

    #[test]
    fn merge_consecutive_different_speakers_not_merged() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 1000), make_turn(2, 1200, 2000)];
        let result = merge_consecutive_turns(&r);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_consecutive_large_gap_not_merged() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 1000), make_turn(1, 2000, 3000)];
        // Gap = 1000ms > 500ms → do not merge.
        let result = merge_consecutive_turns(&r);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_consecutive_sorts_before_merge() {
        let mut r = DiarizationResult::new();
        // Inserted in reverse order.
        r.turns = vec![make_turn(1, 1200, 2000), make_turn(1, 0, 1000)];
        let result = merge_consecutive_turns(&r);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ms, 0);
    }

    // --- speaker_stats ---

    #[test]
    fn speaker_stats_basic() {
        let r = simple_result();
        let stats = speaker_stats(&r);
        let s1 = stats.get(&1).expect("get should succeed");
        assert_eq!(s1.turn_count, 2);
        assert_eq!(s1.total_time_ms, 3000 + 2500); // 0-3000 + 6500-9000
        let s2 = stats.get(&2).expect("get should succeed");
        assert_eq!(s2.turn_count, 1);
        assert_eq!(s2.total_time_ms, 3000);
    }

    #[test]
    fn speaker_stats_avg_turn() {
        let r = simple_result();
        let stats = speaker_stats(&r);
        let s1 = stats.get(&1).expect("get should succeed");
        // (3000 + 2500) / 2 = 2750
        assert_eq!(s1.avg_turn_ms, 2750);
    }

    // --- dominant_speaker ---

    #[test]
    fn dominant_speaker_basic() {
        let r = simple_result();
        // Speaker 1 has 5500ms vs speaker 2's 3000ms.
        assert_eq!(dominant_speaker(&r), Some(1));
    }

    #[test]
    fn dominant_speaker_empty() {
        let r = DiarizationResult::new();
        assert_eq!(dominant_speaker(&r), None);
    }

    // --- assign_speakers_to_blocks ---

    #[test]
    fn assign_speakers_assigns_overlapping_speaker() {
        let r = simple_result();
        let mut blocks = vec![make_block(1, 500, 2000), make_block(2, 3500, 5000)];
        assign_speakers_to_blocks(&mut blocks, &r);
        assert_eq!(blocks[0].speaker_id, Some(1));
        assert_eq!(blocks[1].speaker_id, Some(2));
    }

    #[test]
    fn assign_speakers_no_overlap_unchanged() {
        let r = simple_result();
        let mut blocks = vec![make_block(1, 100_000, 101_000)];
        assign_speakers_to_blocks(&mut blocks, &r);
        // No turn covers this time range → speaker_id stays None.
        assert_eq!(blocks[0].speaker_id, None);
    }

    // --- format_speaker_label ---

    #[test]
    fn format_speaker_label_with_name() {
        let s = make_speaker(1, Some("Dr. Smith"));
        assert_eq!(format_speaker_label(&s), "Dr. Smith");
    }

    #[test]
    fn format_speaker_label_without_name() {
        let s = make_speaker(5, None);
        assert_eq!(format_speaker_label(&s), "Speaker 5");
    }

    // --- CrosstalkDetector ---

    #[test]
    fn find_overlapping_turns_none() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 1000), make_turn(2, 1000, 2000)];
        let overlaps = CrosstalkDetector::find_overlapping_turns(&r);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn find_overlapping_turns_detects_overlap() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 2000), make_turn(2, 1000, 3000)];
        let overlaps = CrosstalkDetector::find_overlapping_turns(&r);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].0.speaker_id, 1);
        assert_eq!(overlaps[0].1.speaker_id, 2);
    }

    #[test]
    fn find_overlapping_turns_multiple_overlaps() {
        let mut r = DiarizationResult::new();
        r.turns = vec![
            make_turn(1, 0, 3000),
            make_turn(2, 1000, 4000),
            make_turn(3, 2000, 5000),
        ];
        let overlaps = CrosstalkDetector::find_overlapping_turns(&r);
        // 1&2, 1&3, 2&3 → 3 pairs.
        assert_eq!(overlaps.len(), 3);
    }

    // --- voice_activity_ratio ---

    #[test]
    fn voice_activity_ratio_zero_duration() {
        let r = simple_result();
        assert_eq!(voice_activity_ratio(&r, 0), 0.0);
    }

    #[test]
    fn voice_activity_ratio_full_coverage() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 10000)];
        let ratio = voice_activity_ratio(&r, 10000);
        assert!((ratio - 1.0).abs() < 1e-5);
    }

    #[test]
    fn voice_activity_ratio_half_coverage() {
        let mut r = DiarizationResult::new();
        r.turns = vec![make_turn(1, 0, 5000)];
        let ratio = voice_activity_ratio(&r, 10000);
        assert!((ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn voice_activity_ratio_overlapping_turns_not_double_counted() {
        let mut r = DiarizationResult::new();
        // Two speakers both active for the first 5 seconds.
        r.turns = vec![make_turn(1, 0, 5000), make_turn(2, 0, 5000)];
        let ratio = voice_activity_ratio(&r, 10000);
        // Union = 5000ms, not 10000ms.
        assert!((ratio - 0.5).abs() < 1e-5);
    }

    #[test]
    fn diarization_result_total_speech_ms() {
        let r = simple_result();
        // 3000 + 3000 + 2500 = 8500
        assert_eq!(r.total_speech_ms(), 8500);
    }

    #[test]
    fn speaker_gender_variants_accessible() {
        let g = SpeakerGender::Female;
        assert_eq!(g, SpeakerGender::Female);
        let g2 = SpeakerGender::Unknown;
        assert_ne!(g, g2);
    }

    // --- merge_consecutive_turns_with_gap ---

    #[test]
    fn merge_with_gap_zero_does_not_merge() {
        let mut r = DiarizationResult::new();
        // Gap = 200ms; with max_gap_ms=0, nothing merges.
        r.turns = vec![make_turn(1, 0, 1000), make_turn(1, 1200, 2000)];
        let result = merge_consecutive_turns_with_gap(&r, 0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_with_large_gap_merges_far_turns() {
        let mut r = DiarizationResult::new();
        // 2000ms gap; default 500ms would not merge, but 3000ms threshold does.
        r.turns = vec![make_turn(1, 0, 1000), make_turn(1, 3000, 5000)];
        let result = merge_consecutive_turns_with_gap(&r, 3000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 5000);
    }

    // --- CrosstalkDetector with tolerance ---

    #[test]
    fn crosstalk_detector_with_tolerance_filters_small_overlap() {
        let mut r = DiarizationResult::new();
        // Turn 1: 0–2000ms, Turn 2: 1900–3000ms → overlap = 100ms
        // Turn 1 duration = 2000ms; overlap fraction = 100/2000 = 0.05
        // With tolerance=0.10, this should be filtered out.
        r.turns = vec![make_turn(1, 0, 2000), make_turn(2, 1900, 3000)];
        let detector = CrosstalkDetector::with_overlap_tolerance(0.10);
        let overlaps = detector.detect(&r);
        assert!(
            overlaps.is_empty(),
            "small overlap should be filtered by tolerance"
        );
    }

    #[test]
    fn crosstalk_detector_with_tolerance_keeps_large_overlap() {
        let mut r = DiarizationResult::new();
        // Turn 1: 0–2000ms, Turn 2: 500–3000ms → overlap = 1500ms
        // Shorter turn is 2000ms (Turn 1); fraction = 1500/2000 = 0.75 > 0.10
        r.turns = vec![make_turn(1, 0, 2000), make_turn(2, 500, 3000)];
        let detector = CrosstalkDetector::with_overlap_tolerance(0.10);
        let overlaps = detector.detect(&r);
        assert_eq!(overlaps.len(), 1);
    }

    // --- assign_speakers_to_blocks with 5+ simultaneous speakers ---

    #[test]
    fn assign_speakers_with_five_simultaneous_speakers() {
        let mut r = DiarizationResult::new();
        // All 5 speakers active for the same duration, speaker 3 has the most overlap.
        r.turns = vec![
            make_turn(1, 0, 100),
            make_turn(2, 0, 200),
            make_turn(3, 0, 1000), // longest overlap with block
            make_turn(4, 0, 150),
            make_turn(5, 0, 50),
        ];
        let mut blocks = vec![make_block(1, 0, 1500)];
        assign_speakers_to_blocks(&mut blocks, &r);
        // Speaker 3 has the most overlap (1000ms vs others).
        assert_eq!(blocks[0].speaker_id, Some(3));
    }
}
