#![allow(dead_code)]
//! EDL comparison utilities for detecting differences between two EDLs.
//!
//! This module provides structural comparison of EDL files, identifying
//! added, removed, and modified events between two versions.

use crate::event::EdlEvent;
use crate::Edl;
use std::collections::HashMap;
use std::fmt;

/// The type of change detected between two EDL versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    /// An event was added in the new EDL.
    Added,
    /// An event was removed from the old EDL.
    Removed,
    /// An event was modified between versions.
    Modified,
    /// Source timecode was changed.
    SourceTimecodeChanged,
    /// Record timecode was changed.
    RecordTimecodeChanged,
    /// Reel assignment was changed.
    ReelChanged,
    /// Edit type was changed (e.g. Cut -> Dissolve).
    EditTypeChanged,
    /// Track type was changed.
    TrackChanged,
    /// Clip name was changed.
    ClipNameChanged,
}

impl fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Added => "ADDED",
            Self::Removed => "REMOVED",
            Self::Modified => "MODIFIED",
            Self::SourceTimecodeChanged => "SRC_TC_CHANGED",
            Self::RecordTimecodeChanged => "REC_TC_CHANGED",
            Self::ReelChanged => "REEL_CHANGED",
            Self::EditTypeChanged => "EDIT_TYPE_CHANGED",
            Self::TrackChanged => "TRACK_CHANGED",
            Self::ClipNameChanged => "CLIP_NAME_CHANGED",
        };
        write!(f, "{label}")
    }
}

/// A single difference detected between two EDLs.
#[derive(Debug, Clone)]
pub struct EdlDiff {
    /// Event number in the old EDL (if applicable).
    pub old_event_number: Option<u32>,
    /// Event number in the new EDL (if applicable).
    pub new_event_number: Option<u32>,
    /// The kind of change.
    pub kind: ChangeKind,
    /// Human-readable description of the change.
    pub description: String,
}

impl fmt::Display for EdlDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let old = self
            .old_event_number
            .map_or_else(|| "-".to_string(), |n| n.to_string());
        let new = self
            .new_event_number
            .map_or_else(|| "-".to_string(), |n| n.to_string());
        write!(
            f,
            "[{kind}] old={old} new={new}: {desc}",
            kind = self.kind,
            desc = self.description
        )
    }
}

/// Result of comparing two EDLs.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// List of differences found.
    pub diffs: Vec<EdlDiff>,
    /// Total events in the old EDL.
    pub old_event_count: usize,
    /// Total events in the new EDL.
    pub new_event_count: usize,
}

impl CompareResult {
    /// Returns `true` if the two EDLs are identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.diffs.is_empty()
    }

    /// Count of added events.
    #[must_use]
    pub fn added_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| d.kind == ChangeKind::Added)
            .count()
    }

    /// Count of removed events.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| d.kind == ChangeKind::Removed)
            .count()
    }

    /// Count of modified events.
    #[must_use]
    pub fn modified_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| !matches!(d.kind, ChangeKind::Added | ChangeKind::Removed))
            .count()
    }

    /// Produce a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "added={}, removed={}, modified={}, old_events={}, new_events={}",
            self.added_count(),
            self.removed_count(),
            self.modified_count(),
            self.old_event_count,
            self.new_event_count,
        )
    }
}

/// Strategy used to match events between two EDLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStrategy {
    /// Match events by their event number.
    ByEventNumber,
    /// Match events by record-in timecode.
    ByRecordIn,
    /// Match events by reel name and source-in timecode.
    ByReelAndSource,
}

/// Compare two EDLs and produce a list of differences.
///
/// # Arguments
///
/// * `old` - The baseline EDL.
/// * `new` - The updated EDL.
/// * `strategy` - How to match events between the two.
#[must_use]
pub fn compare_edls(old: &Edl, new: &Edl, strategy: MatchStrategy) -> CompareResult {
    let mut diffs = Vec::new();

    match strategy {
        MatchStrategy::ByEventNumber => {
            compare_by_event_number(old, new, &mut diffs);
        }
        MatchStrategy::ByRecordIn => {
            compare_by_record_in(old, new, &mut diffs);
        }
        MatchStrategy::ByReelAndSource => {
            compare_by_reel_and_source(old, new, &mut diffs);
        }
    }

    CompareResult {
        diffs,
        old_event_count: old.events.len(),
        new_event_count: new.events.len(),
    }
}

/// Compare events matched by event number.
fn compare_by_event_number(old: &Edl, new: &Edl, diffs: &mut Vec<EdlDiff>) {
    let old_map: HashMap<u32, &EdlEvent> = old.events.iter().map(|e| (e.number, e)).collect();
    let new_map: HashMap<u32, &EdlEvent> = new.events.iter().map(|e| (e.number, e)).collect();

    // Check for removed / modified
    for (&num, &old_evt) in &old_map {
        if let Some(&new_evt) = new_map.get(&num) {
            diff_events(old_evt, new_evt, diffs);
        } else {
            diffs.push(EdlDiff {
                old_event_number: Some(num),
                new_event_number: None,
                kind: ChangeKind::Removed,
                description: format!("Event {num} removed"),
            });
        }
    }

    // Check for added
    for &num in new_map.keys() {
        if !old_map.contains_key(&num) {
            diffs.push(EdlDiff {
                old_event_number: None,
                new_event_number: Some(num),
                kind: ChangeKind::Added,
                description: format!("Event {num} added"),
            });
        }
    }
}

/// Compare events matched by record-in timecode.
fn compare_by_record_in(old: &Edl, new: &Edl, diffs: &mut Vec<EdlDiff>) {
    let old_map: HashMap<u64, &EdlEvent> = old
        .events
        .iter()
        .map(|e| (e.record_in.to_frames(), e))
        .collect();
    let new_map: HashMap<u64, &EdlEvent> = new
        .events
        .iter()
        .map(|e| (e.record_in.to_frames(), e))
        .collect();

    for (&tc, &old_evt) in &old_map {
        if let Some(&new_evt) = new_map.get(&tc) {
            diff_events(old_evt, new_evt, diffs);
        } else {
            diffs.push(EdlDiff {
                old_event_number: Some(old_evt.number),
                new_event_number: None,
                kind: ChangeKind::Removed,
                description: format!("Event {} at record_in frame {} removed", old_evt.number, tc),
            });
        }
    }

    for (&tc, &new_evt) in &new_map {
        if !old_map.contains_key(&tc) {
            diffs.push(EdlDiff {
                old_event_number: None,
                new_event_number: Some(new_evt.number),
                kind: ChangeKind::Added,
                description: format!("Event {} at record_in frame {} added", new_evt.number, tc),
            });
        }
    }
}

/// Compare events matched by reel name + source-in.
fn compare_by_reel_and_source(old: &Edl, new: &Edl, diffs: &mut Vec<EdlDiff>) {
    let key_fn = |e: &EdlEvent| -> (String, u64) { (e.reel.clone(), e.source_in.to_frames()) };

    let old_map: HashMap<(String, u64), &EdlEvent> =
        old.events.iter().map(|e| (key_fn(e), e)).collect();
    let new_map: HashMap<(String, u64), &EdlEvent> =
        new.events.iter().map(|e| (key_fn(e), e)).collect();

    for (key, &old_evt) in &old_map {
        if let Some(&new_evt) = new_map.get(key) {
            diff_events(old_evt, new_evt, diffs);
        } else {
            diffs.push(EdlDiff {
                old_event_number: Some(old_evt.number),
                new_event_number: None,
                kind: ChangeKind::Removed,
                description: format!(
                    "Event {} (reel={}, src_in frame={}) removed",
                    old_evt.number, key.0, key.1
                ),
            });
        }
    }

    for (key, &new_evt) in &new_map {
        if !old_map.contains_key(key) {
            diffs.push(EdlDiff {
                old_event_number: None,
                new_event_number: Some(new_evt.number),
                kind: ChangeKind::Added,
                description: format!(
                    "Event {} (reel={}, src_in frame={}) added",
                    new_evt.number, key.0, key.1
                ),
            });
        }
    }
}

/// Produce field-level diffs between two matched events.
fn diff_events(old: &EdlEvent, new: &EdlEvent, diffs: &mut Vec<EdlDiff>) {
    if old.source_in != new.source_in || old.source_out != new.source_out {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::SourceTimecodeChanged,
            description: format!(
                "Source TC changed: {}-{} -> {}-{}",
                old.source_in, old.source_out, new.source_in, new.source_out
            ),
        });
    }

    if old.record_in != new.record_in || old.record_out != new.record_out {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::RecordTimecodeChanged,
            description: format!(
                "Record TC changed: {}-{} -> {}-{}",
                old.record_in, old.record_out, new.record_in, new.record_out
            ),
        });
    }

    if old.reel != new.reel {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::ReelChanged,
            description: format!("Reel changed: {} -> {}", old.reel, new.reel),
        });
    }

    if old.edit_type != new.edit_type {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::EditTypeChanged,
            description: format!("Edit type changed: {} -> {}", old.edit_type, new.edit_type),
        });
    }

    if old.track != new.track {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::TrackChanged,
            description: format!("Track changed: {} -> {}", old.track, new.track),
        });
    }

    if old.clip_name != new.clip_name {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::ClipNameChanged,
            description: format!(
                "Clip name changed: {:?} -> {:?}",
                old.clip_name, new.clip_name
            ),
        });
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Diff visualization
// ────────────────────────────────────────────────────────────────────────────

/// Style used for rendering the diff visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStyle {
    /// Compact style: one line per diff.
    Compact,
    /// Detailed style: full event context around each diff.
    Detailed,
    /// Side-by-side style: old and new events in parallel columns.
    SideBySide,
}

/// Options controlling diff visualization output.
#[derive(Debug, Clone)]
pub struct DiffVisualizationOptions {
    /// Style of the visualization.
    pub style: DiffStyle,
    /// Width of each column for side-by-side (default 40).
    pub column_width: usize,
    /// Whether to show unchanged events as context.
    pub show_context: bool,
    /// Number of unchanged events to show around each change (context lines).
    pub context_lines: usize,
}

impl Default for DiffVisualizationOptions {
    fn default() -> Self {
        Self {
            style: DiffStyle::Detailed,
            column_width: 40,
            show_context: true,
            context_lines: 1,
        }
    }
}

impl DiffVisualizationOptions {
    /// Create compact visualization options.
    #[must_use]
    pub fn compact() -> Self {
        Self {
            style: DiffStyle::Compact,
            ..Self::default()
        }
    }

    /// Create detailed visualization options.
    #[must_use]
    pub fn detailed() -> Self {
        Self {
            style: DiffStyle::Detailed,
            ..Self::default()
        }
    }

    /// Create side-by-side visualization options.
    #[must_use]
    pub fn side_by_side() -> Self {
        Self {
            style: DiffStyle::SideBySide,
            ..Self::default()
        }
    }
}

/// Produce a human-readable diff visualization between two EDLs.
///
/// This generates a string showing added, removed, and modified events
/// with clear visual markers (`+` for added, `-` for removed, `~` for modified).
///
/// # Arguments
///
/// * `old` - The baseline EDL.
/// * `new` - The updated EDL.
/// * `strategy` - Event matching strategy.
/// * `options` - Visualization options.
#[must_use]
pub fn visualize_diff(
    old: &Edl,
    new: &Edl,
    strategy: MatchStrategy,
    options: &DiffVisualizationOptions,
) -> String {
    let result = compare_edls(old, new, strategy);

    match options.style {
        DiffStyle::Compact => render_compact(&result),
        DiffStyle::Detailed => render_detailed(old, new, &result, options),
        DiffStyle::SideBySide => render_side_by_side(old, new, &result, options),
    }
}

/// Render compact diff output (one line per diff).
fn render_compact(result: &CompareResult) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "--- EDL Diff: {} change(s) ---",
        result.diffs.len()
    ));
    lines.push(format!(
        "Old: {} events | New: {} events",
        result.old_event_count, result.new_event_count
    ));
    lines.push(format!(
        "Added: {} | Removed: {} | Modified: {}",
        result.added_count(),
        result.removed_count(),
        result.modified_count()
    ));
    lines.push(String::new());

    for diff in &result.diffs {
        let marker = match diff.kind {
            ChangeKind::Added => "+",
            ChangeKind::Removed => "-",
            _ => "~",
        };
        lines.push(format!("{marker} {diff}"));
    }

    lines.join("\n")
}

/// Format a single event as a concise one-line summary.
fn format_event_summary(event: &EdlEvent) -> String {
    format!(
        "{:03}  {:<8} {} {} {} {} {} {}",
        event.number,
        event.reel,
        event.track,
        event.edit_type,
        event.source_in,
        event.source_out,
        event.record_in,
        event.record_out,
    )
}

/// Render detailed diff output with event context.
fn render_detailed(
    old: &Edl,
    new: &Edl,
    result: &CompareResult,
    options: &DiffVisualizationOptions,
) -> String {
    let mut lines = Vec::new();

    // Header
    let old_title = old.title.as_deref().unwrap_or("<untitled>");
    let new_title = new.title.as_deref().unwrap_or("<untitled>");
    lines.push(format!("--- {old_title} ({} events)", old.events.len()));
    lines.push(format!("+++ {new_title} ({} events)", new.events.len()));
    lines.push(format!(
        "Summary: +{} -{} ~{}",
        result.added_count(),
        result.removed_count(),
        result.modified_count()
    ));
    lines.push(String::new());

    if result.is_identical() {
        lines.push("(no differences)".to_string());
        return lines.join("\n");
    }

    // Collect changed event numbers for context tracking
    let changed_old: std::collections::HashSet<u32> = result
        .diffs
        .iter()
        .filter_map(|d| d.old_event_number)
        .collect();
    let changed_new: std::collections::HashSet<u32> = result
        .diffs
        .iter()
        .filter_map(|d| d.new_event_number)
        .collect();

    // Show removed events with context
    for diff in &result.diffs {
        match diff.kind {
            ChangeKind::Added => {
                if let Some(num) = diff.new_event_number {
                    if let Some(event) = new.events.iter().find(|e| e.number == num) {
                        lines.push(format!("+ {}", format_event_summary(event)));
                        if let Some(clip) = &event.clip_name {
                            lines.push(format!("+   CLIP: {clip}"));
                        }
                    }
                }
            }
            ChangeKind::Removed => {
                if let Some(num) = diff.old_event_number {
                    if let Some(event) = old.events.iter().find(|e| e.number == num) {
                        lines.push(format!("- {}", format_event_summary(event)));
                        if let Some(clip) = &event.clip_name {
                            lines.push(format!("-   CLIP: {clip}"));
                        }
                    }
                }
            }
            _ => {
                // Modified: show old and new
                let old_num = diff.old_event_number;
                let new_num = diff.new_event_number;
                if let Some(onum) = old_num {
                    if let Some(old_ev) = old.events.iter().find(|e| e.number == onum) {
                        lines.push(format!("- {}", format_event_summary(old_ev)));
                    }
                }
                if let Some(nnum) = new_num {
                    if let Some(new_ev) = new.events.iter().find(|e| e.number == nnum) {
                        lines.push(format!("+ {}", format_event_summary(new_ev)));
                    }
                }
                lines.push(format!("  >> {}", diff.description));
            }
        }
    }

    // Show context events if requested
    if options.show_context && options.context_lines > 0 {
        let mut context_lines_list = Vec::new();
        for event in &new.events {
            if !changed_new.contains(&event.number) && !changed_old.contains(&event.number) {
                // Check if this event is near a changed event
                let near_change = result.diffs.iter().any(|d| {
                    let is_near_old = d.old_event_number.map_or(false, |n| {
                        event.number.abs_diff(n) <= options.context_lines as u32
                    });
                    let is_near_new = d.new_event_number.map_or(false, |n| {
                        event.number.abs_diff(n) <= options.context_lines as u32
                    });
                    is_near_old || is_near_new
                });
                if near_change {
                    context_lines_list.push(format!("  {}", format_event_summary(event)));
                }
            }
        }
        if !context_lines_list.is_empty() {
            lines.push(String::new());
            lines.push("Context:".to_string());
            lines.extend(context_lines_list);
        }
    }

    lines.join("\n")
}

/// Render side-by-side diff output.
fn render_side_by_side(
    old: &Edl,
    new: &Edl,
    result: &CompareResult,
    options: &DiffVisualizationOptions,
) -> String {
    let w = options.column_width;
    let mut lines = Vec::new();

    // Header
    let separator = format!("{:-<width$}+{:-<width$}", "", "", width = w + 2);
    let old_title = old.title.as_deref().unwrap_or("<old>");
    let new_title = new.title.as_deref().unwrap_or("<new>");

    lines.push(format!(
        " {:<width$} | {:<width$}",
        truncate_str(old_title, w),
        truncate_str(new_title, w),
        width = w,
    ));
    lines.push(separator.clone());

    // Build maps for quick event lookup
    let old_by_num: HashMap<u32, &EdlEvent> = old.events.iter().map(|e| (e.number, e)).collect();
    let new_by_num: HashMap<u32, &EdlEvent> = new.events.iter().map(|e| (e.number, e)).collect();

    // Collect all event numbers, sorted
    let mut all_nums: Vec<u32> = old
        .events
        .iter()
        .map(|e| e.number)
        .chain(new.events.iter().map(|e| e.number))
        .collect();
    all_nums.sort_unstable();
    all_nums.dedup();

    let changed_old: std::collections::HashSet<u32> = result
        .diffs
        .iter()
        .filter_map(|d| d.old_event_number)
        .collect();
    let changed_new: std::collections::HashSet<u32> = result
        .diffs
        .iter()
        .filter_map(|d| d.new_event_number)
        .collect();

    for num in &all_nums {
        let old_ev = old_by_num.get(num);
        let new_ev = new_by_num.get(num);

        let is_changed = changed_old.contains(num) || changed_new.contains(num);
        let marker = if is_changed {
            match (old_ev, new_ev) {
                (None, Some(_)) => "+",
                (Some(_), None) => "-",
                _ => "~",
            }
        } else {
            " "
        };

        let left = old_ev.map_or_else(
            || String::new(),
            |e| format!("{:03} {:<6} {} {}", e.number, e.reel, e.track, e.edit_type),
        );
        let right = new_ev.map_or_else(
            || String::new(),
            |e| format!("{:03} {:<6} {} {}", e.number, e.reel, e.track, e.edit_type),
        );

        lines.push(format!(
            "{marker}{:<width$} | {:<width$}",
            truncate_str(&left, w),
            truncate_str(&right, w),
            width = w,
        ));
    }

    lines.push(separator);
    lines.push(format!(
        "Changes: +{} -{} ~{}",
        result.added_count(),
        result.removed_count(),
        result.modified_count()
    ));

    lines.join("\n")
}

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s[..max_len].to_string()
    }
}

/// Compute a numeric similarity score (0.0 .. 1.0) between two EDLs.
///
/// The score considers event count, matching events, and timecode proximity.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn similarity_score(old: &Edl, new: &Edl) -> f64 {
    if old.events.is_empty() && new.events.is_empty() {
        return 1.0;
    }
    let total = (old.events.len() + new.events.len()) as f64;
    if total == 0.0 {
        return 1.0;
    }

    let result = compare_edls(old, new, MatchStrategy::ByEventNumber);
    let diff_count = result.diffs.len() as f64;

    // Clamp to [0, 1]
    (1.0 - diff_count / total).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::{Edl, EdlFormat};

    fn make_event(num: u32, reel: &str, src_in_sec: u8, src_out_sec: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, src_in_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, src_out_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, src_in_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, src_out_sec, 0, fr).expect("failed to create"),
        )
    }

    fn make_edl(events: Vec<EdlEvent>) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        for e in events {
            edl.events.push(e);
        }
        edl
    }

    #[test]
    fn test_identical_edls() {
        let e1 = make_event(1, "A001", 0, 5);
        let edl_a = make_edl(vec![e1.clone()]);
        let edl_b = make_edl(vec![e1]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result.is_identical());
    }

    #[test]
    fn test_added_event() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert_eq!(result.added_count(), 1);
        assert_eq!(result.removed_count(), 0);
    }

    #[test]
    fn test_removed_event() {
        let edl_a = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert_eq!(result.removed_count(), 1);
    }

    #[test]
    fn test_reel_changed() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "B001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::ReelChanged));
    }

    #[test]
    fn test_edit_type_changed() {
        let mut evt = make_event(1, "A001", 0, 5);
        let edl_a = make_edl(vec![evt.clone()]);
        evt.edit_type = EditType::Dissolve;
        evt.transition_duration = Some(30);
        let edl_b = make_edl(vec![evt]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::EditTypeChanged));
    }

    #[test]
    fn test_source_timecode_changed() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 1, 6)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::SourceTimecodeChanged));
    }

    #[test]
    fn test_compare_by_record_in() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByRecordIn);
        assert!(result.is_identical());
    }

    #[test]
    fn test_compare_by_reel_and_source() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByReelAndSource);
        assert!(result.is_identical());
    }

    #[test]
    fn test_compare_both_empty() {
        let edl_a = make_edl(vec![]);
        let edl_b = make_edl(vec![]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result.is_identical());
    }

    #[test]
    fn test_summary_string() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        let s = result.summary();
        assert!(s.contains("added=1"));
    }

    #[test]
    fn test_similarity_identical() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let score = similarity_score(&edl_a, &edl_b);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_empty() {
        let edl_a = make_edl(vec![]);
        let edl_b = make_edl(vec![]);
        let score = similarity_score(&edl_a, &edl_b);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::Added.to_string(), "ADDED");
        assert_eq!(ChangeKind::Removed.to_string(), "REMOVED");
        assert_eq!(ChangeKind::Modified.to_string(), "MODIFIED");
    }

    #[test]
    fn test_diff_display() {
        let diff = EdlDiff {
            old_event_number: Some(1),
            new_event_number: None,
            kind: ChangeKind::Removed,
            description: "Event 1 removed".to_string(),
        };
        let s = diff.to_string();
        assert!(s.contains("REMOVED"));
        assert!(s.contains("old=1"));
    }

    #[test]
    fn test_clip_name_changed() {
        let mut evt_a = make_event(1, "A001", 0, 5);
        evt_a.clip_name = Some("clip_v1.mov".to_string());
        let mut evt_b = make_event(1, "A001", 0, 5);
        evt_b.clip_name = Some("clip_v2.mov".to_string());
        let edl_a = make_edl(vec![evt_a]);
        let edl_b = make_edl(vec![evt_b]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::ClipNameChanged));
    }

    // ── Diff visualization tests ──

    #[test]
    fn test_visualize_diff_compact_no_changes() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::compact(),
        );
        assert!(output.contains("0 change(s)"));
    }

    #[test]
    fn test_visualize_diff_compact_with_add() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::compact(),
        );
        assert!(output.contains("Added: 1"));
        assert!(output.contains("+"));
    }

    #[test]
    fn test_visualize_diff_compact_with_remove() {
        let edl_a = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::compact(),
        );
        assert!(output.contains("Removed: 1"));
        assert!(output.contains("-"));
    }

    #[test]
    fn test_visualize_diff_detailed_identical() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::detailed(),
        );
        assert!(output.contains("no differences"));
    }

    #[test]
    fn test_visualize_diff_detailed_modified() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "B001", 0, 5)]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::detailed(),
        );
        assert!(output.contains("~1"));
        assert!(output.contains(">>"));
    }

    #[test]
    fn test_visualize_diff_detailed_with_clip_name() {
        let mut evt_a = make_event(1, "A001", 0, 5);
        evt_a.clip_name = Some("clip_old.mov".to_string());
        let edl_a = make_edl(vec![evt_a]);
        let edl_b = make_edl(vec![]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::detailed(),
        );
        assert!(output.contains("clip_old.mov"));
    }

    #[test]
    fn test_visualize_diff_side_by_side() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::side_by_side(),
        );
        assert!(output.contains("|"));
        assert!(output.contains("+1"));
    }

    #[test]
    fn test_visualize_diff_side_by_side_removal() {
        let edl_a = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::side_by_side(),
        );
        assert!(output.contains("-1"));
    }

    #[test]
    fn test_visualize_diff_compact_mixed() {
        let edl_a = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let edl_b = make_edl(vec![
            make_event(1, "B001", 0, 5),   // modified
            make_event(3, "A003", 10, 15), // added, 2 removed
        ]);
        let output = visualize_diff(
            &edl_a,
            &edl_b,
            MatchStrategy::ByEventNumber,
            &DiffVisualizationOptions::compact(),
        );
        assert!(output.contains("Added: 1"));
        assert!(output.contains("Removed: 1"));
        assert!(output.contains("Modified: 1"));
    }

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        assert_eq!(truncate_str("hello world", 8), "hello...");
    }

    #[test]
    fn test_diff_visualization_options_default() {
        let opts = DiffVisualizationOptions::default();
        assert_eq!(opts.style, DiffStyle::Detailed);
        assert_eq!(opts.column_width, 40);
        assert!(opts.show_context);
    }
}
