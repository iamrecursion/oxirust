#![allow(dead_code)]
//! ASCII timeline visualization for EDL debugging.
//!
//! Renders an EDL as a horizontal ASCII art timeline, showing each event
//! as a labeled block proportional to its duration. Multiple tracks (video,
//! audio) are displayed on separate rows.
//!
//! This is intended as a quick visual debugging aid, not a production UI.

use crate::event::{EdlEvent, TrackType};
use crate::Edl;
use std::collections::BTreeMap;
use std::fmt;

/// Options controlling the ASCII timeline rendering.
#[derive(Debug, Clone)]
pub struct TimelineOptions {
    /// Total width of the timeline in characters (default 80).
    pub width: usize,
    /// Character used for filled event blocks.
    pub fill_char: char,
    /// Character used for empty timeline gaps.
    pub gap_char: char,
    /// Whether to show event numbers inside blocks.
    pub show_event_numbers: bool,
    /// Whether to show reel names inside blocks.
    pub show_reel_names: bool,
    /// Whether to show a timecode ruler at the bottom.
    pub show_ruler: bool,
    /// Whether to group events by track type.
    pub group_by_track: bool,
}

impl Default for TimelineOptions {
    fn default() -> Self {
        Self {
            width: 80,
            fill_char: '#',
            gap_char: '.',
            show_event_numbers: true,
            show_reel_names: true,
            show_ruler: true,
            group_by_track: true,
        }
    }
}

impl TimelineOptions {
    /// Create compact options (narrow, minimal info).
    #[must_use]
    pub fn compact() -> Self {
        Self {
            width: 60,
            fill_char: '#',
            gap_char: ' ',
            show_event_numbers: true,
            show_reel_names: false,
            show_ruler: false,
            group_by_track: false,
        }
    }

    /// Create wide options for detailed display.
    #[must_use]
    pub fn wide() -> Self {
        Self {
            width: 120,
            ..Self::default()
        }
    }
}

/// Represents a rendered track row in the ASCII timeline.
#[derive(Debug, Clone)]
struct TrackRow {
    /// Track label (e.g., "V", "A1", "A2").
    label: String,
    /// The rendered character row.
    cells: Vec<char>,
    /// Labels to overlay on the cells.
    labels: Vec<(usize, usize, String)>, // (start_col, end_col, label)
}

/// Result of rendering an ASCII timeline.
#[derive(Debug, Clone)]
pub struct AsciiTimeline {
    /// The rendered lines of the timeline.
    lines: Vec<String>,
    /// Width used for rendering.
    width: usize,
    /// Total record duration in frames.
    total_frames: u64,
    /// Number of events rendered.
    event_count: usize,
}

impl AsciiTimeline {
    /// Get the rendered timeline as a single string.
    #[must_use]
    pub fn to_string_rendered(&self) -> String {
        self.lines.join("\n")
    }

    /// Get the number of lines in the timeline.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Get the total frames covered by the timeline.
    #[must_use]
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Get the number of events rendered.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.event_count
    }
}

impl fmt::Display for AsciiTimeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_rendered())
    }
}

/// Render an EDL as an ASCII timeline.
///
/// Events are placed proportionally on a horizontal axis based on their
/// `record_in` and `record_out` timecodes. Each track type gets its own row.
///
/// # Arguments
///
/// * `edl` - The EDL to visualize.
/// * `options` - Rendering options.
#[must_use]
pub fn render_timeline(edl: &Edl, options: &TimelineOptions) -> AsciiTimeline {
    if edl.events.is_empty() {
        return AsciiTimeline {
            lines: vec!["(empty timeline)".to_string()],
            width: options.width,
            total_frames: 0,
            event_count: 0,
        };
    }

    // Calculate the global time range
    let min_frame = edl
        .events
        .iter()
        .map(|e| e.record_in.to_frames())
        .min()
        .unwrap_or(0);
    let max_frame = edl
        .events
        .iter()
        .map(|e| e.record_out.to_frames())
        .max()
        .unwrap_or(0);

    let total_frames = max_frame.saturating_sub(min_frame);
    if total_frames == 0 {
        return AsciiTimeline {
            lines: vec!["(zero-length timeline)".to_string()],
            width: options.width,
            total_frames: 0,
            event_count: edl.events.len(),
        };
    }

    // Usable width (minus track label column)
    let label_width = 4;
    let usable_width = options.width.saturating_sub(label_width + 3); // "V  |" prefix

    let mut lines: Vec<String> = Vec::new();

    // Title
    if let Some(title) = &edl.title {
        lines.push(format!("Timeline: {title}"));
    } else {
        lines.push("Timeline:".to_string());
    }
    lines.push(format!(
        "Events: {} | Duration: {} frames | {} fps",
        edl.events.len(),
        total_frames,
        edl.frame_rate.fps()
    ));
    lines.push(format!("{}", "=".repeat(options.width)));

    if options.group_by_track {
        // Group events by track type
        let mut track_groups: BTreeMap<String, Vec<&EdlEvent>> = BTreeMap::new();
        for event in &edl.events {
            let key = track_label(&event.track);
            track_groups.entry(key).or_default().push(event);
        }

        for (track_name, events) in &track_groups {
            let row = render_track_row(
                track_name,
                events,
                min_frame,
                total_frames,
                usable_width,
                options,
            );
            lines.push(row);
        }
    } else {
        // All events on a single row
        let all_events: Vec<&EdlEvent> = edl.events.iter().collect();
        let row = render_track_row(
            "ALL",
            &all_events,
            min_frame,
            total_frames,
            usable_width,
            options,
        );
        lines.push(row);
    }

    // Ruler
    if options.show_ruler {
        lines.push(render_ruler(
            min_frame,
            max_frame,
            label_width,
            usable_width,
            edl.frame_rate.fps(),
        ));
    }

    AsciiTimeline {
        lines,
        width: options.width,
        total_frames,
        event_count: edl.events.len(),
    }
}

/// Get a short label for a track type.
fn track_label(track: &TrackType) -> String {
    match track {
        TrackType::Video => "V".to_string(),
        TrackType::Audio(ch) => format!("A{}", ch.number()),
        TrackType::AudioPair => "AA".to_string(),
        TrackType::AudioWithVideo => "A/V".to_string(),
        TrackType::AudioPairWithVideo => "AAV".to_string(),
        TrackType::AudioMulti(_) => "AM".to_string(),
        TrackType::VideoWithAudioMulti(_) => "VAM".to_string(),
    }
}

/// Render a single track row.
#[allow(clippy::cast_precision_loss)]
fn render_track_row(
    label: &str,
    events: &[&EdlEvent],
    min_frame: u64,
    total_frames: u64,
    usable_width: usize,
    options: &TimelineOptions,
) -> String {
    // Initialize cells with gap character
    let mut cells: Vec<char> = vec![options.gap_char; usable_width];

    // Place events
    for event in events {
        let event_start = event.record_in.to_frames().saturating_sub(min_frame);
        let event_end = event.record_out.to_frames().saturating_sub(min_frame);

        let col_start = frame_to_col(event_start, total_frames, usable_width);
        let col_end = frame_to_col(event_end, total_frames, usable_width);
        let col_end = col_end.max(col_start + 1).min(usable_width);

        // Fill the cells
        for col in col_start..col_end {
            cells[col] = options.fill_char;
        }

        // Overlay label if space permits
        let block_width = col_end.saturating_sub(col_start);
        if block_width >= 2 {
            let inner_label = if options.show_event_numbers && options.show_reel_names {
                format!(
                    "{}:{}",
                    event.number,
                    truncate_label(&event.reel, block_width.saturating_sub(4))
                )
            } else if options.show_event_numbers {
                format!("{}", event.number)
            } else if options.show_reel_names {
                truncate_label(&event.reel, block_width.saturating_sub(1))
            } else {
                String::new()
            };

            if !inner_label.is_empty() && inner_label.len() < block_width {
                let start = col_start;
                for (i, ch) in inner_label.chars().enumerate() {
                    let col = start + i;
                    if col < col_end {
                        cells[col] = ch;
                    }
                }
            }
        }
    }

    let cell_str: String = cells.into_iter().collect();
    format!("{:<3} |{cell_str}|", label)
}

/// Convert a frame position to a column index.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn frame_to_col(frame: u64, total_frames: u64, usable_width: usize) -> usize {
    if total_frames == 0 {
        return 0;
    }
    let ratio = frame as f64 / total_frames as f64;
    let col = (ratio * usable_width as f64) as usize;
    col.min(usable_width)
}

/// Truncate a label to fit within `max_len` characters.
fn truncate_label(s: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 1 {
        format!("{}~", &s[..max_len - 1])
    } else {
        s[..1].to_string()
    }
}

/// Render a timecode ruler.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn render_ruler(
    min_frame: u64,
    max_frame: u64,
    label_width: usize,
    usable_width: usize,
    fps: u32,
) -> String {
    let total_frames = max_frame.saturating_sub(min_frame);
    if total_frames == 0 || usable_width == 0 {
        return String::new();
    }

    // Place tick marks at regular intervals
    let tick_count = (usable_width / 10).max(2).min(20);
    let mut ruler_chars = vec![' '; usable_width];
    let mut label_line = vec![' '; usable_width];

    for i in 0..=tick_count {
        let col = (i * usable_width) / tick_count;
        let col = col.min(usable_width.saturating_sub(1));

        ruler_chars[col] = '|';

        // Calculate timecode at this position
        let frame = min_frame + ((i as u64 * total_frames) / tick_count as u64);
        let total_seconds = if fps > 0 { frame / fps as u64 } else { 0 };
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;
        let tc_label = format!("{hours:02}:{minutes:02}:{seconds:02}");

        // Place label centered on the tick
        let label_start = col.saturating_sub(tc_label.len() / 2);
        for (j, ch) in tc_label.chars().enumerate() {
            let pos = label_start + j;
            if pos < usable_width {
                label_line[pos] = ch;
            }
        }
    }

    let ruler_str: String = ruler_chars.into_iter().collect();
    let label_str: String = label_line.into_iter().collect();
    let prefix = " ".repeat(label_width + 1);
    format!("{prefix}{ruler_str}\n{prefix}{label_str}")
}

/// Quick helper: render an EDL timeline with default options and return the string.
#[must_use]
pub fn render_timeline_string(edl: &Edl) -> String {
    let options = TimelineOptions::default();
    render_timeline(edl, &options).to_string_rendered()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, EdlEvent, TrackType};
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

    fn make_audio_event(num: u32, reel: &str, sec_in: u8, sec_out: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Audio(crate::audio::AudioChannel::A1),
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
    fn test_render_empty_edl() {
        let edl = make_edl("Empty", vec![]);
        let timeline = render_timeline(&edl, &TimelineOptions::default());
        assert!(timeline.to_string_rendered().contains("empty timeline"));
        assert_eq!(timeline.event_count(), 0);
    }

    #[test]
    fn test_render_single_event() {
        let edl = make_edl("Single", vec![make_event(1, "A001", 0, 10)]);
        let timeline = render_timeline(&edl, &TimelineOptions::default());
        let output = timeline.to_string_rendered();
        assert!(output.contains("Single"));
        assert!(output.contains("V"));
        assert_eq!(timeline.event_count(), 1);
    }

    #[test]
    fn test_render_multiple_events() {
        let edl = make_edl(
            "Multi",
            vec![
                make_event(1, "A001", 0, 5),
                make_event(2, "A002", 5, 10),
                make_event(3, "A003", 10, 20),
            ],
        );
        let timeline = render_timeline(&edl, &TimelineOptions::default());
        let output = timeline.to_string_rendered();
        assert!(output.contains("Multi"));
        assert_eq!(timeline.event_count(), 3);
        assert!(output.contains('|'));
    }

    #[test]
    fn test_render_multi_track() {
        let edl = make_edl(
            "MultiTrack",
            vec![
                make_event(1, "V001", 0, 10),
                make_audio_event(2, "A001", 0, 10),
            ],
        );
        let options = TimelineOptions::default();
        let timeline = render_timeline(&edl, &options);
        let output = timeline.to_string_rendered();
        // Should have both video and audio rows
        assert!(output.contains("V"));
        assert!(output.contains("A1"));
    }

    #[test]
    fn test_render_compact() {
        let edl = make_edl("Compact", vec![make_event(1, "A001", 0, 10)]);
        let timeline = render_timeline(&edl, &TimelineOptions::compact());
        let output = timeline.to_string_rendered();
        assert!(!output.is_empty());
        assert_eq!(timeline.event_count(), 1);
    }

    #[test]
    fn test_render_wide() {
        let edl = make_edl("Wide", vec![make_event(1, "A001", 0, 10)]);
        let timeline = render_timeline(&edl, &TimelineOptions::wide());
        let output = timeline.to_string_rendered();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_render_with_ruler() {
        let edl = make_edl(
            "Ruler",
            vec![make_event(1, "A001", 0, 10), make_event(2, "A002", 10, 20)],
        );
        let mut opts = TimelineOptions::default();
        opts.show_ruler = true;
        let timeline = render_timeline(&edl, &opts);
        let output = timeline.to_string_rendered();
        // Ruler should have timecodes
        assert!(output.contains(':'));
    }

    #[test]
    fn test_render_without_ruler() {
        let edl = make_edl("NoRuler", vec![make_event(1, "A001", 0, 10)]);
        let mut opts = TimelineOptions::default();
        opts.show_ruler = false;
        let timeline = render_timeline(&edl, &opts);
        assert!(timeline.line_count() >= 3); // title + stats + separator + row
    }

    #[test]
    fn test_render_no_grouping() {
        let edl = make_edl(
            "NoGroup",
            vec![
                make_event(1, "V001", 0, 10),
                make_audio_event(2, "A001", 0, 10),
            ],
        );
        let mut opts = TimelineOptions::default();
        opts.group_by_track = false;
        let timeline = render_timeline(&edl, &opts);
        let output = timeline.to_string_rendered();
        assert!(output.contains("ALL"));
    }

    #[test]
    fn test_render_timeline_string_helper() {
        let edl = make_edl("Helper", vec![make_event(1, "A001", 0, 10)]);
        let output = render_timeline_string(&edl);
        assert!(output.contains("Helper"));
    }

    #[test]
    fn test_frame_to_col_basic() {
        assert_eq!(frame_to_col(0, 100, 50), 0);
        assert_eq!(frame_to_col(50, 100, 50), 25);
        assert_eq!(frame_to_col(100, 100, 50), 50);
    }

    #[test]
    fn test_frame_to_col_zero_total() {
        assert_eq!(frame_to_col(0, 0, 50), 0);
    }

    #[test]
    fn test_truncate_label_short() {
        assert_eq!(truncate_label("AB", 5), "AB");
    }

    #[test]
    fn test_truncate_label_long() {
        assert_eq!(truncate_label("ABCDEFG", 4), "ABC~");
    }

    #[test]
    fn test_truncate_label_zero() {
        assert_eq!(truncate_label("ABC", 0), "");
    }

    #[test]
    fn test_track_label_video() {
        assert_eq!(track_label(&TrackType::Video), "V");
    }

    #[test]
    fn test_track_label_audio_pair() {
        assert_eq!(track_label(&TrackType::AudioPair), "AA");
    }

    #[test]
    fn test_timeline_display_impl() {
        let edl = make_edl("Display", vec![make_event(1, "A001", 0, 10)]);
        let timeline = render_timeline(&edl, &TimelineOptions::default());
        let displayed = format!("{timeline}");
        assert!(displayed.contains("Display"));
    }

    #[test]
    fn test_render_gap_between_events() {
        let edl = make_edl(
            "Gap",
            vec![make_event(1, "A001", 0, 5), make_event(2, "A002", 10, 15)],
        );
        let mut opts = TimelineOptions::default();
        opts.gap_char = '.';
        let timeline = render_timeline(&edl, &opts);
        let output = timeline.to_string_rendered();
        // Should have gap characters between the two events
        assert!(output.contains('.'));
    }

    #[test]
    fn test_timeline_total_frames() {
        let edl = make_edl(
            "Frames",
            vec![make_event(1, "A001", 0, 10), make_event(2, "A002", 10, 20)],
        );
        let timeline = render_timeline(&edl, &TimelineOptions::default());
        assert_eq!(timeline.total_frames(), 500); // 20 seconds * 25 fps
    }

    #[test]
    fn test_render_options_default() {
        let opts = TimelineOptions::default();
        assert_eq!(opts.width, 80);
        assert_eq!(opts.fill_char, '#');
        assert!(opts.show_event_numbers);
        assert!(opts.show_reel_names);
        assert!(opts.show_ruler);
        assert!(opts.group_by_track);
    }
}
