#![allow(dead_code)]
//! EDL merging utilities for combining multiple EDLs into one.
//!
//! This module supports several merge strategies:
//! - **Append**: place the second EDL after the first on the timeline.
//! - **Interleave**: interleave events by their record-in timecodes.
//! - **Replace**: replace matching events in a base EDL with events from an overlay.
//! - **Union**: combine all events de-duplicating by event number.

use crate::event::EdlEvent;
use crate::timecode::EdlFrameRate;
use crate::{Edl, EdlFormat};
use std::collections::HashSet;

/// Strategy for merging two EDLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Append the second EDL after the first; renumber and offset record timecodes.
    Append,
    /// Interleave events from both EDLs by record-in timecode.
    Interleave,
    /// Replace events in the base EDL with events from the overlay that share the same number.
    Replace,
    /// Union of events from both EDLs; first wins on duplicate numbers.
    Union,
}

/// Options controlling the merge operation.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Merge strategy.
    pub strategy: MergeStrategy,
    /// Whether to renumber events after merging.
    pub renumber: bool,
    /// Whether to sort events by record-in after merging.
    pub sort_by_record_in: bool,
    /// Frame rate for the merged output (uses the first EDL's rate if `None`).
    pub frame_rate: Option<EdlFrameRate>,
    /// Title for the merged EDL (auto-generated if `None`).
    pub title: Option<String>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Append,
            renumber: true,
            sort_by_record_in: true,
            frame_rate: None,
            title: None,
        }
    }
}

impl MergeOptions {
    /// Create merge options with the given strategy.
    #[must_use]
    pub fn with_strategy(strategy: MergeStrategy) -> Self {
        Self {
            strategy,
            ..Self::default()
        }
    }

    /// Set whether to renumber events.
    #[must_use]
    pub fn renumber(mut self, value: bool) -> Self {
        self.renumber = value;
        self
    }

    /// Set whether to sort events by record-in.
    #[must_use]
    pub fn sort_by_record_in(mut self, value: bool) -> Self {
        self.sort_by_record_in = value;
        self
    }

    /// Set the output frame rate.
    #[must_use]
    pub fn frame_rate(mut self, rate: EdlFrameRate) -> Self {
        self.frame_rate = Some(rate);
        self
    }

    /// Set the output title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// Merge result with statistics.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged EDL.
    pub edl: Edl,
    /// Number of events contributed by the first EDL.
    pub from_first: usize,
    /// Number of events contributed by the second EDL.
    pub from_second: usize,
    /// Number of events that were replaced (Replace strategy).
    pub replaced: usize,
    /// Number of duplicate events skipped (Union strategy).
    pub duplicates_skipped: usize,
}

impl MergeResult {
    /// Total number of events in the merged EDL.
    #[must_use]
    pub fn total_events(&self) -> usize {
        self.edl.events.len()
    }
}

/// Merge two EDLs according to the given options.
///
/// # Arguments
///
/// * `first` - The first (base) EDL.
/// * `second` - The second (overlay / appended) EDL.
/// * `options` - Merge configuration.
#[must_use]
pub fn merge_edls(first: &Edl, second: &Edl, options: &MergeOptions) -> MergeResult {
    let frame_rate = options.frame_rate.unwrap_or(first.frame_rate);
    let format = first.format;
    let title = options.title.clone().or_else(|| {
        let t1 = first.title.as_deref().unwrap_or("EDL1");
        let t2 = second.title.as_deref().unwrap_or("EDL2");
        Some(format!("{t1} + {t2}"))
    });

    let mut result = MergeResult {
        edl: Edl::new(format),
        from_first: 0,
        from_second: 0,
        replaced: 0,
        duplicates_skipped: 0,
    };

    result.edl.set_frame_rate(frame_rate);
    if let Some(t) = title {
        result.edl.set_title(t);
    }

    match options.strategy {
        MergeStrategy::Append => merge_append(first, second, &mut result),
        MergeStrategy::Interleave => merge_interleave(first, second, &mut result),
        MergeStrategy::Replace => merge_replace(first, second, &mut result),
        MergeStrategy::Union => merge_union(first, second, &mut result),
    }

    if options.sort_by_record_in {
        result.edl.events.sort_by_key(|e| e.record_in.to_frames());
    }

    if options.renumber {
        result.edl.renumber_events();
    }

    result
}

/// Append strategy: copy all events from first, then all from second.
fn merge_append(first: &Edl, second: &Edl, result: &mut MergeResult) {
    for e in &first.events {
        result.edl.events.push(e.clone());
        result.from_first += 1;
    }
    for e in &second.events {
        result.edl.events.push(e.clone());
        result.from_second += 1;
    }
}

/// Interleave strategy: merge events sorted by record-in timecode.
fn merge_interleave(first: &Edl, second: &Edl, result: &mut MergeResult) {
    let mut all: Vec<(usize, &EdlEvent)> = first
        .events
        .iter()
        .map(|e| (0_usize, e))
        .chain(second.events.iter().map(|e| (1_usize, e)))
        .collect();

    all.sort_by_key(|(_, e)| e.record_in.to_frames());

    for (source, e) in all {
        result.edl.events.push(e.clone());
        if source == 0 {
            result.from_first += 1;
        } else {
            result.from_second += 1;
        }
    }
}

/// Replace strategy: start with first, replace matching event numbers from second.
fn merge_replace(first: &Edl, second: &Edl, result: &mut MergeResult) {
    let overlay_numbers: HashSet<u32> = second.events.iter().map(|e| e.number).collect();

    for e in &first.events {
        if overlay_numbers.contains(&e.number) {
            // Will be replaced by second's event
            if let Some(replacement) = second.events.iter().find(|s| s.number == e.number) {
                result.edl.events.push(replacement.clone());
                result.replaced += 1;
                result.from_second += 1;
            }
        } else {
            result.edl.events.push(e.clone());
            result.from_first += 1;
        }
    }

    // Add any events from second that are not in first
    let first_numbers: HashSet<u32> = first.events.iter().map(|e| e.number).collect();
    for e in &second.events {
        if !first_numbers.contains(&e.number) {
            result.edl.events.push(e.clone());
            result.from_second += 1;
        }
    }
}

/// Union strategy: include all unique event numbers, first wins on duplicates.
fn merge_union(first: &Edl, second: &Edl, result: &mut MergeResult) {
    let mut seen: HashSet<u32> = HashSet::new();

    for e in &first.events {
        seen.insert(e.number);
        result.edl.events.push(e.clone());
        result.from_first += 1;
    }

    for e in &second.events {
        if seen.contains(&e.number) {
            result.duplicates_skipped += 1;
        } else {
            seen.insert(e.number);
            result.edl.events.push(e.clone());
            result.from_second += 1;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Conflict resolution
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for resolving conflicts when events from two EDLs overlap
/// on the record timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Always prefer the event from source A (first EDL).
    PreferA,
    /// Always prefer the event from source B (second EDL).
    PreferB,
    /// Keep both events (may produce overlaps).
    KeepBoth,
    /// Prefer the longer event (greater duration).
    PreferLonger,
    /// Prefer the shorter event (smaller duration).
    PreferShorter,
}

/// A detected conflict between two events from different EDLs.
#[derive(Debug, Clone)]
pub struct MergeConflict {
    /// Event from source A.
    pub event_a: EdlEvent,
    /// Event from source B.
    pub event_b: EdlEvent,
    /// Which resolution was applied.
    pub resolution: ConflictResolution,
    /// The winning event number (or both if `KeepBoth`).
    pub resolved_to: ConflictOutcome,
}

/// Outcome of resolving a single conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictOutcome {
    /// The event from source A was chosen.
    KeptA,
    /// The event from source B was chosen.
    KeptB,
    /// Both events were kept.
    KeptBoth,
}

/// Result of a conflict-aware merge.
#[derive(Debug, Clone)]
pub struct ConflictMergeResult {
    /// The merged EDL.
    pub edl: Edl,
    /// Number of events contributed by source A.
    pub from_a: usize,
    /// Number of events contributed by source B.
    pub from_b: usize,
    /// Conflicts detected and how they were resolved.
    pub conflicts: Vec<MergeConflict>,
}

impl ConflictMergeResult {
    /// Total number of events in the merged EDL.
    #[must_use]
    pub fn total_events(&self) -> usize {
        self.edl.events.len()
    }

    /// Number of conflicts that were detected.
    #[must_use]
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Generate a human-readable conflict report.
    #[must_use]
    pub fn conflict_report(&self) -> String {
        if self.conflicts.is_empty() {
            return "No conflicts detected.".to_string();
        }
        let mut lines = Vec::new();
        lines.push(format!(
            "Merge Conflict Report: {} conflict(s)",
            self.conflicts.len()
        ));
        lines.push(String::new());
        for (i, conflict) in self.conflicts.iter().enumerate() {
            lines.push(format!(
                "Conflict #{}: Event {} (A, reel={}) vs Event {} (B, reel={})",
                i + 1,
                conflict.event_a.number,
                conflict.event_a.reel,
                conflict.event_b.number,
                conflict.event_b.reel,
            ));
            lines.push(format!(
                "  A: {} - {} | B: {} - {}",
                conflict.event_a.record_in,
                conflict.event_a.record_out,
                conflict.event_b.record_in,
                conflict.event_b.record_out,
            ));
            let outcome = match &conflict.resolved_to {
                ConflictOutcome::KeptA => "Kept event from source A",
                ConflictOutcome::KeptB => "Kept event from source B",
                ConflictOutcome::KeptBoth => "Kept both events",
            };
            lines.push(format!("  Resolution: {outcome}"));
        }
        lines.join("\n")
    }
}

/// Merge two EDLs with conflict detection and resolution.
///
/// Unlike `merge_edls`, this function detects when events from two EDLs
/// overlap on the record timeline and applies the specified conflict
/// resolution strategy.
///
/// # Arguments
///
/// * `source_a` - First (primary) EDL.
/// * `source_b` - Second (overlay) EDL.
/// * `resolution` - How to resolve timeline conflicts.
/// * `options` - Standard merge options (renumber, sort, etc.).
#[must_use]
pub fn merge_with_conflict_resolution(
    source_a: &Edl,
    source_b: &Edl,
    resolution: ConflictResolution,
    options: &MergeOptions,
) -> ConflictMergeResult {
    let frame_rate = options.frame_rate.unwrap_or(source_a.frame_rate);
    let format = source_a.format;
    let title = options.title.clone().or_else(|| {
        let t1 = source_a.title.as_deref().unwrap_or("EDL_A");
        let t2 = source_b.title.as_deref().unwrap_or("EDL_B");
        Some(format!("{t1} + {t2}"))
    });

    let mut result_edl = Edl::new(format);
    result_edl.set_frame_rate(frame_rate);
    if let Some(t) = title {
        result_edl.set_title(t);
    }

    let mut from_a = 0_usize;
    let mut from_b = 0_usize;
    let mut conflicts = Vec::new();

    // Find overlapping pairs between A and B events
    let mut b_handled: HashSet<usize> = HashSet::new();

    for event_a in &source_a.events {
        let mut conflicting_b_indices: Vec<usize> = Vec::new();

        for (bi, event_b) in source_b.events.iter().enumerate() {
            if event_a.overlaps_with(event_b) {
                conflicting_b_indices.push(bi);
            }
        }

        if conflicting_b_indices.is_empty() {
            // No conflict, add event_a directly
            result_edl.events.push(event_a.clone());
            from_a += 1;
        } else {
            // Resolve each conflict
            for &bi in &conflicting_b_indices {
                b_handled.insert(bi);
                let event_b = &source_b.events[bi];

                let (outcome, events_to_add) = resolve_conflict(event_a, event_b, resolution);

                let conflict = MergeConflict {
                    event_a: event_a.clone(),
                    event_b: event_b.clone(),
                    resolution,
                    resolved_to: outcome.clone(),
                };
                conflicts.push(conflict);

                for (src, ev) in events_to_add {
                    result_edl.events.push(ev);
                    match src {
                        ConflictSource::A => from_a += 1,
                        ConflictSource::B => from_b += 1,
                    }
                }
            }
            // If we handled conflicts but still need to add event_a
            // (it was already added in resolve_conflict if needed)
        }
    }

    // Add non-conflicting events from B
    for (bi, event_b) in source_b.events.iter().enumerate() {
        if !b_handled.contains(&bi) {
            result_edl.events.push(event_b.clone());
            from_b += 1;
        }
    }

    // Post-processing
    if options.sort_by_record_in {
        result_edl.events.sort_by_key(|e| e.record_in.to_frames());
    }
    if options.renumber {
        result_edl.renumber_events();
    }

    ConflictMergeResult {
        edl: result_edl,
        from_a,
        from_b,
        conflicts,
    }
}

/// Internal marker for which source an event came from.
#[derive(Debug, Clone, Copy)]
enum ConflictSource {
    A,
    B,
}

/// Resolve a single conflict between two overlapping events.
fn resolve_conflict(
    event_a: &EdlEvent,
    event_b: &EdlEvent,
    resolution: ConflictResolution,
) -> (ConflictOutcome, Vec<(ConflictSource, EdlEvent)>) {
    match resolution {
        ConflictResolution::PreferA => (
            ConflictOutcome::KeptA,
            vec![(ConflictSource::A, event_a.clone())],
        ),
        ConflictResolution::PreferB => (
            ConflictOutcome::KeptB,
            vec![(ConflictSource::B, event_b.clone())],
        ),
        ConflictResolution::KeepBoth => (
            ConflictOutcome::KeptBoth,
            vec![
                (ConflictSource::A, event_a.clone()),
                (ConflictSource::B, event_b.clone()),
            ],
        ),
        ConflictResolution::PreferLonger => {
            let dur_a = event_a.duration_frames();
            let dur_b = event_b.duration_frames();
            if dur_a >= dur_b {
                (
                    ConflictOutcome::KeptA,
                    vec![(ConflictSource::A, event_a.clone())],
                )
            } else {
                (
                    ConflictOutcome::KeptB,
                    vec![(ConflictSource::B, event_b.clone())],
                )
            }
        }
        ConflictResolution::PreferShorter => {
            let dur_a = event_a.duration_frames();
            let dur_b = event_b.duration_frames();
            if dur_a <= dur_b {
                (
                    ConflictOutcome::KeptA,
                    vec![(ConflictSource::A, event_a.clone())],
                )
            } else {
                (
                    ConflictOutcome::KeptB,
                    vec![(ConflictSource::B, event_b.clone())],
                )
            }
        }
    }
}

/// Convenience function: merge multiple EDLs sequentially using the Append strategy.
#[must_use]
pub fn merge_many(edls: &[&Edl], options: &MergeOptions) -> Edl {
    if edls.is_empty() {
        return Edl::new(EdlFormat::Cmx3600);
    }
    if edls.len() == 1 {
        return edls[0].clone();
    }

    let mut merged = edls[0].clone();
    for edl in &edls[1..] {
        let res = merge_edls(&merged, edl, options);
        merged = res.edl;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::{Edl, EdlFormat};

    fn make_event(num: u32, reel: &str, sec_in: u8, sec_out: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
        )
    }

    fn make_edl(title: &str, events: Vec<EdlEvent>) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title(title.to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);
        for e in events {
            edl.events.push(e);
        }
        edl
    }

    #[test]
    fn test_append_basic() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
        assert_eq!(result.from_first, 1);
        assert_eq!(result.from_second, 1);
    }

    #[test]
    fn test_append_renumber() {
        let a = make_edl("A", vec![make_event(10, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(20, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append).renumber(true);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.edl.events[0].number, 1);
        assert_eq!(result.edl.events[1].number, 2);
    }

    #[test]
    fn test_interleave() {
        let a = make_edl("A", vec![make_event(1, "R1", 10, 15)]);
        let b = make_edl("B", vec![make_event(2, "R2", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Interleave);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
        // After sort, event from b (sec 0-5) should come first
        assert_eq!(result.edl.events[0].reel, "R2");
    }

    #[test]
    fn test_replace_matching() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(1, "R1_REPLACED", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Replace);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 1);
        assert_eq!(result.replaced, 1);
        assert_eq!(result.edl.events[0].reel, "R1_REPLACED");
    }

    #[test]
    fn test_replace_adds_new() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Replace);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
    }

    #[test]
    fn test_union_first_wins() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(1, "R1_DUP", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Union);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 1);
        assert_eq!(result.duplicates_skipped, 1);
        assert_eq!(result.edl.events[0].reel, "R1");
    }

    #[test]
    fn test_union_adds_unique() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Union);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
    }

    #[test]
    fn test_merge_title_auto() {
        let a = make_edl("Reel_A", vec![]);
        let b = make_edl("Reel_B", vec![]);
        let opts = MergeOptions::default();
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.edl.title.as_deref(), Some("Reel_A + Reel_B"));
    }

    #[test]
    fn test_merge_title_custom() {
        let a = make_edl("A", vec![]);
        let b = make_edl("B", vec![]);
        let opts = MergeOptions::default().title("Custom Title");
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.edl.title.as_deref(), Some("Custom Title"));
    }

    #[test]
    fn test_merge_many_empty() {
        let edls: Vec<&Edl> = vec![];
        let opts = MergeOptions::default();
        let merged = merge_many(&edls, &opts);
        assert!(merged.events.is_empty());
    }

    #[test]
    fn test_merge_many_single() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let opts = MergeOptions::default();
        let merged = merge_many(&[&a], &opts);
        assert_eq!(merged.events.len(), 1);
    }

    #[test]
    fn test_merge_many_multiple() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let c = make_edl("C", vec![make_event(3, "R3", 10, 15)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append);
        let merged = merge_many(&[&a, &b, &c], &opts);
        assert_eq!(merged.events.len(), 3);
    }

    #[test]
    fn test_no_sort_option() {
        let a = make_edl("A", vec![make_event(1, "R1", 10, 15)]);
        let b = make_edl("B", vec![make_event(2, "R2", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append)
            .sort_by_record_in(false)
            .renumber(false);
        let result = merge_edls(&a, &b, &opts);
        // Without sorting, first EDL's event stays first
        assert_eq!(result.edl.events[0].number, 1);
    }

    // ── Conflict resolution tests ──

    fn make_overlapping_event(num: u32, reel: &str, sec_in: u8, sec_out: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
        )
    }

    #[test]
    fn test_conflict_prefer_a() {
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 5, 15)]);
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferA, &opts);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.conflicts[0].resolved_to, ConflictOutcome::KeptA);
        // Only event from A should be in result (plus no non-conflicting B events)
        assert_eq!(result.total_events(), 1);
        assert_eq!(result.edl.events[0].reel, "R1");
    }

    #[test]
    fn test_conflict_prefer_b() {
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 5, 15)]);
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferB, &opts);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.conflicts[0].resolved_to, ConflictOutcome::KeptB);
        assert_eq!(result.total_events(), 1);
        assert_eq!(result.edl.events[0].reel, "R2");
    }

    #[test]
    fn test_conflict_keep_both() {
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 5, 15)]);
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::KeepBoth, &opts);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.conflicts[0].resolved_to, ConflictOutcome::KeptBoth);
        assert_eq!(result.total_events(), 2);
    }

    #[test]
    fn test_conflict_prefer_longer() {
        // A is 10 sec, B is 5 sec
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 5, 10)]);
        let opts = MergeOptions::default();
        let result =
            merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferLonger, &opts);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.conflicts[0].resolved_to, ConflictOutcome::KeptA);
        assert_eq!(result.edl.events[0].reel, "R1");
    }

    #[test]
    fn test_conflict_prefer_shorter() {
        // A is 10 sec, B is 5 sec
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 5, 10)]);
        let opts = MergeOptions::default();
        let result =
            merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferShorter, &opts);
        assert_eq!(result.conflict_count(), 1);
        assert_eq!(result.conflicts[0].resolved_to, ConflictOutcome::KeptB);
        assert_eq!(result.edl.events[0].reel, "R2");
    }

    #[test]
    fn test_conflict_no_conflicts() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 10, 15)]);
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferA, &opts);
        assert_eq!(result.conflict_count(), 0);
        assert_eq!(result.total_events(), 2);
    }

    #[test]
    fn test_conflict_report_no_conflicts() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 10, 15)]);
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferA, &opts);
        let report = result.conflict_report();
        assert!(report.contains("No conflicts"));
    }

    #[test]
    fn test_conflict_report_with_conflicts() {
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 5, 15)]);
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferA, &opts);
        let report = result.conflict_report();
        assert!(report.contains("Conflict #1"));
        assert!(report.contains("Kept event from source A"));
    }

    #[test]
    fn test_conflict_mixed_overlap_and_non_overlap() {
        let a = make_edl(
            "A",
            vec![
                make_overlapping_event(1, "R1", 0, 10),
                make_event(2, "R3", 20, 25),
            ],
        );
        let b = make_edl(
            "B",
            vec![
                make_overlapping_event(1, "R2", 5, 15),
                make_event(3, "R4", 30, 35),
            ],
        );
        let opts = MergeOptions::default();
        let result = merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferA, &opts);
        assert_eq!(result.conflict_count(), 1);
        // R1 kept (A wins), R3 from A, R4 from B = 3 total
        assert_eq!(result.total_events(), 3);
    }

    #[test]
    fn test_conflict_prefer_longer_tie() {
        // Equal duration: prefer A (tie-break)
        let a = make_edl("A", vec![make_overlapping_event(1, "R1", 0, 10)]);
        let b = make_edl("B", vec![make_overlapping_event(1, "R2", 0, 10)]);
        let opts = MergeOptions::default();
        let result =
            merge_with_conflict_resolution(&a, &b, ConflictResolution::PreferLonger, &opts);
        assert_eq!(result.conflicts[0].resolved_to, ConflictOutcome::KeptA);
    }
}
