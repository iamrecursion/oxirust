//! Timeline exporter: export timeline to EDL (Edit Decision List) and other formats.
//!
//! Generates CMX 3600-compatible EDL output from a timeline, suitable for interchange
//! with professional NLEs and finishing systems.

use std::fmt::Write as FmtWrite;
use std::path::Path;

use crate::clip::MediaSource;
use crate::error::{TimelineError, TimelineResult};
use crate::timeline::Timeline;
use crate::transition::{TransitionType, WipeDirection};
use crate::types::Duration;

/// Options for EDL export.
#[derive(Debug, Clone)]
pub struct EdlExportOptions {
    /// Title written at the top of the EDL.
    pub title: String,
    /// Frame rate for timecode display (e.g. 24, 25, 30).
    pub frame_rate: u32,
    /// Whether to use drop-frame timecode notation.
    pub drop_frame: bool,
    /// Whether to include audio events.
    pub include_audio: bool,
    /// Whether to include video events.
    pub include_video: bool,
    /// Starting event number.
    pub start_event: u32,
}

impl Default for EdlExportOptions {
    fn default() -> Self {
        Self {
            title: "Untitled Edit".to_string(),
            frame_rate: 24,
            drop_frame: false,
            include_audio: true,
            include_video: true,
            start_event: 1,
        }
    }
}

/// A single event in an EDL.
#[derive(Debug, Clone)]
pub struct EdlEvent {
    /// Sequential event number.
    pub event_number: u32,
    /// Reel/source name (up to 8 chars).
    pub reel: String,
    /// Track designator (e.g. "V", "A", "AA", "V A").
    pub track: String,
    /// Edit type ("C" = cut, "D" = dissolve, "W" = wipe).
    pub edit_type: String,
    /// Wipe number if applicable.
    pub wipe_number: Option<u32>,
    /// Transition duration in frames (for dissolves/wipes).
    pub transition_duration: Option<u32>,
    /// Source in timecode.
    pub source_in: String,
    /// Source out timecode.
    pub source_out: String,
    /// Record in timecode.
    pub record_in: String,
    /// Record out timecode.
    pub record_out: String,
    /// Optional comment / clip name.
    pub comment: Option<String>,
}

impl EdlEvent {
    /// Format this event as a CMX 3600 line.
    #[must_use]
    pub fn to_cmx3600(&self) -> String {
        let mut s = String::new();
        // Event line
        let edit_str = if let (Some(wipe), Some(dur)) = (self.wipe_number, self.transition_duration)
        {
            format!("{} {:03} {:04}", self.edit_type, wipe, dur)
        } else if let Some(dur) = self.transition_duration {
            format!("{} {:04}", self.edit_type, dur)
        } else {
            self.edit_type.clone()
        };

        let _ = writeln!(
            s,
            "{:03}  {:<8} {} {}  {}  {}  {}  {}",
            self.event_number,
            self.reel,
            self.track,
            edit_str,
            self.source_in,
            self.source_out,
            self.record_in,
            self.record_out
        );

        if let Some(comment) = &self.comment {
            let _ = writeln!(s, "* FROM CLIP NAME: {comment}");
        }
        s
    }
}

/// Exports timeline to EDL (CMX 3600) format.
pub struct TimelineExporter;

impl TimelineExporter {
    /// Create a new exporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Export a timeline to EDL string.
    ///
    /// # Errors
    ///
    /// Returns an error if the timeline cannot be exported.
    pub fn to_edl(
        &self,
        timeline: &Timeline,
        options: &EdlExportOptions,
    ) -> TimelineResult<String> {
        let mut output = String::new();

        // EDL header
        let _ = writeln!(output, "TITLE: {}", options.title);
        let fcm = if options.drop_frame {
            "DROP FRAME"
        } else {
            "NON-DROP FRAME"
        };
        let _ = writeln!(output, "FCM: {fcm}");
        let _ = writeln!(output);

        let mut event_num = options.start_event;

        // Export video tracks
        if options.include_video {
            for track in &timeline.video_tracks {
                if track.hidden {
                    continue;
                }
                for clip in &track.clips {
                    if !clip.enabled {
                        continue;
                    }

                    let reel = self.reel_name(&clip.source);
                    let src_in = frames_to_tc(
                        clip.source_in.value(),
                        options.frame_rate,
                        options.drop_frame,
                    );
                    let src_out = frames_to_tc(
                        clip.source_out.value(),
                        options.frame_rate,
                        options.drop_frame,
                    );
                    let rec_in = frames_to_tc(
                        clip.timeline_in.value(),
                        options.frame_rate,
                        options.drop_frame,
                    );
                    let dur = clip.source_out.value() - clip.source_in.value();
                    let rec_out = frames_to_tc(
                        clip.timeline_in.value() + dur,
                        options.frame_rate,
                        options.drop_frame,
                    );

                    // Check for transition at this clip
                    let (edit_type, trans_dur) = if let Some(t) = timeline.transitions.get(&clip.id)
                    {
                        let et = transition_edit_type(t.transition_type);
                        let td = t.duration.0 as u32;
                        (et, Some(td))
                    } else {
                        ("C".to_string(), None)
                    };

                    let clip_name = clip.name.clone();

                    let event = EdlEvent {
                        event_number: event_num,
                        reel,
                        track: "V".to_string(),
                        edit_type,
                        wipe_number: None,
                        transition_duration: trans_dur,
                        source_in: src_in,
                        source_out: src_out,
                        record_in: rec_in,
                        record_out: rec_out,
                        comment: Some(clip_name),
                    };

                    output.push_str(&event.to_cmx3600());
                    event_num += 1;
                }
            }
        }

        // Export audio tracks
        if options.include_audio {
            let mut audio_num = 1usize;
            for track in &timeline.audio_tracks {
                if track.muted {
                    continue;
                }
                for clip in &track.clips {
                    if !clip.enabled {
                        continue;
                    }

                    let reel = self.reel_name(&clip.source);
                    let src_in = frames_to_tc(
                        clip.source_in.value(),
                        options.frame_rate,
                        options.drop_frame,
                    );
                    let src_out = frames_to_tc(
                        clip.source_out.value(),
                        options.frame_rate,
                        options.drop_frame,
                    );
                    let rec_in = frames_to_tc(
                        clip.timeline_in.value(),
                        options.frame_rate,
                        options.drop_frame,
                    );
                    let dur = clip.source_out.value() - clip.source_in.value();
                    let rec_out = frames_to_tc(
                        clip.timeline_in.value() + dur,
                        options.frame_rate,
                        options.drop_frame,
                    );

                    let track_str = if audio_num == 1 {
                        "A".to_string()
                    } else {
                        format!("A{audio_num}")
                    };

                    let event = EdlEvent {
                        event_number: event_num,
                        reel,
                        track: track_str,
                        edit_type: "C".to_string(),
                        wipe_number: None,
                        transition_duration: None,
                        source_in: src_in,
                        source_out: src_out,
                        record_in: rec_in,
                        record_out: rec_out,
                        comment: Some(clip.name.clone()),
                    };

                    output.push_str(&event.to_cmx3600());
                    event_num += 1;
                }
                audio_num += 1;
            }
        }

        Ok(output)
    }

    /// Export timeline to EDL and write to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_edl(
        &self,
        timeline: &Timeline,
        options: &EdlExportOptions,
        path: &Path,
    ) -> TimelineResult<()> {
        let content = self.to_edl(timeline, options)?;
        std::fs::write(path, &content).map_err(TimelineError::IoError)
    }

    fn reel_name(&self, source: &MediaSource) -> String {
        match source {
            MediaSource::File { path, .. } => {
                // Use stem of filename, truncated to 8 chars
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("AX");
                stem.chars().take(8).collect::<String>().to_uppercase()
            }
            MediaSource::Color { .. } => "COLOR   ".to_string(),
            MediaSource::BarsAndTone => "BARS    ".to_string(),
            _ => "AX      ".to_string(),
        }
    }
}

impl Default for TimelineExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert frame count to SMPTE timecode string.
#[must_use]
pub fn frames_to_tc(frames: i64, frame_rate: u32, drop_frame: bool) -> String {
    let fr = i64::from(frame_rate.max(1));
    let total = frames.max(0);
    let f = total % fr;
    let total_sec = total / fr;
    let s = total_sec % 60;
    let total_min = total_sec / 60;
    let m = total_min % 60;
    let h = total_min / 60;
    let sep = if drop_frame { ';' } else { ':' };
    format!("{h:02}:{m:02}:{s:02}{sep}{f:02}")
}

/// Convert SMPTE timecode string to frame count.
///
/// # Errors
///
/// Returns error if string is malformed.
pub fn tc_to_frames(tc: &str, frame_rate: u32) -> TimelineResult<i64> {
    let fr = i64::from(frame_rate);
    // Allow both : and ; as separator for last field
    let tc_clean = tc.replace(';', ":");
    let parts: Vec<&str> = tc_clean.split(':').collect();
    if parts.len() != 4 {
        return Err(TimelineError::InvalidTimecode(format!(
            "Expected HH:MM:SS:FF format, got: {tc}"
        )));
    }
    let h: i64 = parts[0]
        .parse()
        .map_err(|_| TimelineError::InvalidTimecode(format!("Invalid hours: {}", parts[0])))?;
    let m: i64 = parts[1]
        .parse()
        .map_err(|_| TimelineError::InvalidTimecode(format!("Invalid minutes: {}", parts[1])))?;
    let s: i64 = parts[2]
        .parse()
        .map_err(|_| TimelineError::InvalidTimecode(format!("Invalid seconds: {}", parts[2])))?;
    let f: i64 = parts[3]
        .parse()
        .map_err(|_| TimelineError::InvalidTimecode(format!("Invalid frames: {}", parts[3])))?;

    Ok(((h * 3600 + m * 60 + s) * fr) + f)
}

fn transition_edit_type(t: TransitionType) -> String {
    match t {
        TransitionType::Dissolve => "D".to_string(),
        TransitionType::DipToBlack | TransitionType::DipToWhite | TransitionType::DipToColor => {
            "D".to_string()
        }
        TransitionType::Wipe => "W".to_string(),
        TransitionType::Push | TransitionType::Slide => "W".to_string(),
        TransitionType::AudioCrossfade => "C".to_string(),
    }
}

/// Compute wipe direction number for CMX 3600.
#[allow(dead_code)]
fn wipe_number(dir: WipeDirection) -> u32 {
    match dir {
        WipeDirection::LeftToRight => 1,
        WipeDirection::RightToLeft => 1,
        WipeDirection::TopToBottom => 2,
        WipeDirection::BottomToTop => 2,
    }
}

/// Estimate total duration of timeline from all clips.
#[must_use]
pub fn timeline_duration(timeline: &Timeline) -> Duration {
    let max_frame = timeline
        .video_tracks
        .iter()
        .chain(timeline.audio_tracks.iter())
        .flat_map(|t| t.clips.iter())
        .map(|c| c.timeline_in.value() + (c.source_out.value() - c.source_in.value()))
        .max()
        .unwrap_or(0);
    Duration(max_frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frames_to_tc_zero() {
        assert_eq!(frames_to_tc(0, 24, false), "00:00:00:00");
    }

    #[test]
    fn test_frames_to_tc_one_frame() {
        assert_eq!(frames_to_tc(1, 24, false), "00:00:00:01");
    }

    #[test]
    fn test_frames_to_tc_one_second() {
        assert_eq!(frames_to_tc(24, 24, false), "00:00:01:00");
    }

    #[test]
    fn test_frames_to_tc_one_minute() {
        assert_eq!(frames_to_tc(24 * 60, 24, false), "00:01:00:00");
    }

    #[test]
    fn test_frames_to_tc_one_hour() {
        assert_eq!(frames_to_tc(24 * 3600, 24, false), "01:00:00:00");
    }

    #[test]
    fn test_frames_to_tc_drop_frame_sep() {
        let tc = frames_to_tc(100, 30, true);
        assert!(
            tc.contains(';'),
            "Drop frame should use ';' separator: {tc}"
        );
    }

    #[test]
    fn test_tc_to_frames_roundtrip() {
        let frames = 24 * 3600 + 24 * 600 + 24 * 35 + 7; // 1h 10m 35s 7f @ 24fps
        let tc = frames_to_tc(frames, 24, false);
        let recovered = tc_to_frames(&tc, 24).expect("should succeed in test");
        assert_eq!(recovered, frames);
    }

    #[test]
    fn test_tc_to_frames_invalid() {
        assert!(tc_to_frames("not_a_tc", 24).is_err());
        assert!(tc_to_frames("01:00:00", 24).is_err());
        assert!(tc_to_frames("ab:00:00:00", 24).is_err());
    }

    #[test]
    fn test_edl_event_cmx3600_format() {
        let event = EdlEvent {
            event_number: 1,
            reel: "AX".to_string(),
            track: "V".to_string(),
            edit_type: "C".to_string(),
            wipe_number: None,
            transition_duration: None,
            source_in: "00:00:00:00".to_string(),
            source_out: "00:00:10:00".to_string(),
            record_in: "00:00:00:00".to_string(),
            record_out: "00:00:10:00".to_string(),
            comment: Some("clip1".to_string()),
        };
        let line = event.to_cmx3600();
        assert!(line.contains("001"), "Should contain event number");
        assert!(line.contains("AX"), "Should contain reel name");
        assert!(line.contains("* FROM CLIP NAME: clip1"));
    }

    #[test]
    fn test_export_empty_timeline() {
        use oximedia_core::Rational;
        let exporter = TimelineExporter::new();
        let timeline =
            Timeline::new("My Edit", Rational::new(24, 1), 48000).expect("should succeed in test");
        let opts = EdlExportOptions {
            title: "Test Edit".to_string(),
            frame_rate: 24,
            ..Default::default()
        };
        let edl = exporter
            .to_edl(&timeline, &opts)
            .expect("should succeed in test");
        assert!(edl.starts_with("TITLE: Test Edit"));
        assert!(edl.contains("FCM: NON-DROP FRAME"));
    }

    #[test]
    fn test_export_with_drop_frame() {
        use oximedia_core::Rational;
        let exporter = TimelineExporter::new();
        let timeline = Timeline::new("test", Rational::new(30000, 1001), 48000)
            .expect("should succeed in test");
        let opts = EdlExportOptions {
            drop_frame: true,
            frame_rate: 30,
            ..Default::default()
        };
        let edl = exporter
            .to_edl(&timeline, &opts)
            .expect("should succeed in test");
        assert!(edl.contains("DROP FRAME"));
    }

    #[test]
    fn test_timeline_duration_empty() {
        use oximedia_core::Rational;
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        assert_eq!(timeline_duration(&timeline).0, 0);
    }

    #[test]
    fn test_default_exporter() {
        let _exp = TimelineExporter::default();
    }

    #[test]
    fn test_tc_to_frames_drop_frame_sep() {
        // Drop-frame uses ';' but we accept both
        let frames = tc_to_frames("01:00:00;00", 30).expect("should succeed in test");
        assert_eq!(frames, 30 * 3600);
    }
}
