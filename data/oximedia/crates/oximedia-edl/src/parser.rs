//! EDL parser implementation.
//!
//! This module provides a parser for CMX 3600 EDL files and related formats,
//! using the nom parser combinator library.

use crate::audio::AudioChannel;
use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent, TrackType};
use crate::motion::MotionEffect;
use crate::reel::ReelId;
use crate::timecode::{EdlFrameRate, EdlTimecode};
use crate::{Edl, EdlFormat};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{space0, space1},
    combinator::{map_res, opt, value},
    sequence::terminated,
    IResult, Parser,
};

/// Parse a complete EDL from a string.
///
/// # Errors
///
/// Returns an error if the EDL cannot be parsed.
pub fn parse_edl(input: &str) -> EdlResult<Edl> {
    let mut parser = EdlParser::new();
    parser.parse(input)
}

/// EDL parser with state management.
#[derive(Debug)]
pub struct EdlParser {
    /// Parsing mode (strict or lenient).
    pub strict_mode: bool,
    /// Current line number (for error reporting).
    current_line: usize,
}

impl EdlParser {
    /// Create a new EDL parser.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            strict_mode: false,
            current_line: 0,
        }
    }

    /// Create a new EDL parser in strict mode.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            strict_mode: true,
            current_line: 0,
        }
    }

    /// Enable or disable strict mode.
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Parse an EDL from a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL cannot be parsed.
    #[allow(clippy::too_many_lines)]
    pub fn parse(&mut self, input: &str) -> EdlResult<Edl> {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        let mut current_event: Option<EdlEvent> = None;

        for (line_num, line) in input.lines().enumerate() {
            self.current_line = line_num + 1;
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            // Parse comment lines
            if trimmed.starts_with('*') {
                if let Some(comment) = Self::parse_comment_line(trimmed) {
                    // Check for special comments
                    if comment.starts_with("FROM CLIP NAME:") {
                        if let Some(event) = &mut current_event {
                            let name = comment.trim_start_matches("FROM CLIP NAME:").trim();
                            event.set_clip_name(name.to_string());
                        }
                    } else if comment.starts_with("TO CLIP NAME:") {
                        if let Some(event) = &mut current_event {
                            let name = comment.trim_start_matches("TO CLIP NAME:").trim();
                            event.add_comment(format!("TO CLIP NAME: {name}"));
                        }
                    } else if comment.starts_with("M2") {
                        // Motion effect comment
                        if let Some(event) = &mut current_event {
                            if let Ok(effect) = MotionEffect::from_m2_comment(&comment) {
                                event.set_motion_effect(effect);
                            }
                        }
                    } else if let Some(event) = &mut current_event {
                        event.add_comment(comment);
                    }
                }
                continue;
            }

            // Parse header lines
            if trimmed.starts_with("TITLE:") {
                let title = trimmed.trim_start_matches("TITLE:").trim();
                edl.set_title(title.to_string());
                continue;
            }

            if trimmed.starts_with("FCM:") {
                let fcm = trimmed.trim_start_matches("FCM:").trim();
                let fcm_upper = fcm.to_uppercase();
                let frame_rate = if fcm_upper.contains("NON") {
                    EdlFrameRate::Fps2997NDF
                } else if fcm_upper.contains("DROP") {
                    EdlFrameRate::Fps2997DF
                } else {
                    EdlFrameRate::Fps2997NDF
                };
                edl.set_frame_rate(frame_rate);
                continue;
            }

            // Parse event lines
            if let Ok(event) = self.parse_event_line(trimmed, edl.frame_rate) {
                // Save previous event if any
                if let Some(prev_event) = current_event.take() {
                    edl.add_event(prev_event)
                        .map_err(|e| EdlError::parse(self.current_line, format!("{e}")))?;
                }
                current_event = Some(event);
            }
        }

        // Add the last event
        if let Some(event) = current_event {
            edl.add_event(event)
                .map_err(|e| EdlError::parse(self.current_line, format!("{e}")))?;
        }

        Ok(edl)
    }

    /// Parse a comment line (starts with *).
    fn parse_comment_line(line: &str) -> Option<String> {
        line.strip_prefix('*').map(|s| s.trim().to_string())
    }

    /// Parse an event line.
    ///
    /// Format: EVENT_NUM REEL TRACK EDIT_TYPE [DURATION] SRC_IN SRC_OUT REC_IN REC_OUT
    ///
    /// # Errors
    ///
    /// Returns an error if the event line is malformed.
    #[allow(clippy::too_many_lines)]
    fn parse_event_line(&self, line: &str, frame_rate: EdlFrameRate) -> EdlResult<EdlEvent> {
        let result = Self::event_line_parser(line, frame_rate);

        match result {
            Ok((_, event)) => Ok(event),
            Err(e) => Err(EdlError::parse(
                self.current_line,
                format!("Failed to parse event line: {e}"),
            )),
        }
    }

    /// Nom parser for event lines.
    fn event_line_parser(input: &str, frame_rate: EdlFrameRate) -> IResult<&str, EdlEvent> {
        let (input, _) = space0.parse(input)?;

        // Parse event number (3 digits, zero-padded)
        let mut parse_num = map_res(take_while1(|c: char| c.is_ascii_digit()), |s: &str| {
            s.parse::<u32>()
        });
        let (input, event_num) = parse_num.parse(input)?;

        let (input, _) = space1.parse(input)?;

        // Parse reel name
        let (input, reel) = take_while1(|c: char| !c.is_whitespace()).parse(input)?;
        let reel_id = ReelId::new(reel).map_err(|_| {
            nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
        })?;

        let (input, _) = space1.parse(input)?;

        // Parse track type
        let (input, track) = Self::track_type_parser(input)?;

        let (input, _) = space1.parse(input)?;

        // Parse edit type
        let (input, edit_type) = Self::edit_type_parser(input)?;

        let (input, _) = space0.parse(input)?;

        // Parse optional transition duration
        let parse_duration = map_res(take_while1(|c: char| c.is_ascii_digit()), |s: &str| {
            s.parse::<u32>()
        });
        let mut opt_duration = opt(terminated(parse_duration, space1));
        let (input, transition_duration) = opt_duration.parse(input)?;

        // Consume any remaining spaces before timecodes
        let (input, _) = space0.parse(input)?;

        // Parse timecodes
        let (input, source_in) = Self::timecode_parser(input, frame_rate)?;
        let (input, _) = space1.parse(input)?;
        let (input, source_out) = Self::timecode_parser(input, frame_rate)?;
        let (input, _) = space1.parse(input)?;
        let (input, record_in) = Self::timecode_parser(input, frame_rate)?;
        let (input, _) = space1.parse(input)?;
        let (input, record_out) = Self::timecode_parser(input, frame_rate)?;

        let mut event = EdlEvent::new(
            event_num,
            reel_id.to_string(),
            track,
            edit_type,
            source_in,
            source_out,
            record_in,
            record_out,
        );

        if let Some(duration) = transition_duration {
            event.set_transition_duration(duration);
        }

        Ok((input, event))
    }

    /// Parse track type.
    fn track_type_parser(input: &str) -> IResult<&str, TrackType> {
        alt((
            value(TrackType::AudioPairWithVideo, tag("AA/V")),
            value(TrackType::AudioWithVideo, tag("A/V")),
            value(TrackType::AudioPair, tag("AA")),
            value(TrackType::Audio(AudioChannel::A4), tag("A4")),
            value(TrackType::Audio(AudioChannel::A3), tag("A3")),
            value(TrackType::Audio(AudioChannel::A2), tag("A2")),
            value(TrackType::Audio(AudioChannel::A1), tag("A")),
            value(TrackType::Video, tag("V")),
        ))
        .parse(input)
    }

    /// Parse edit type.
    fn edit_type_parser(input: &str) -> IResult<&str, EditType> {
        alt((
            value(EditType::Cut, tag("C")),
            value(EditType::Dissolve, tag("D")),
            value(EditType::Wipe, tag("W")),
            value(EditType::Key, tag("K")),
        ))
        .parse(input)
    }

    /// Parse timecode (HH:MM:SS:FF or HH:MM:SS;FF).
    fn timecode_parser(input: &str, frame_rate: EdlFrameRate) -> IResult<&str, EdlTimecode> {
        let (input, tc_str) =
            take_while1(|c: char| c.is_ascii_digit() || c == ':' || c == ';').parse(input)?;

        let tc = EdlTimecode::parse(tc_str, frame_rate).map_err(|_| {
            nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
        })?;

        Ok((input, tc))
    }
}

impl Default for EdlParser {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Lazy parsing types ────────────────────────────────────────────────────

/// The eagerly-parsed header fields of an EDL event line.
///
/// These fields are cheap to compute — they only require scanning the first
/// whitespace-delimited token group on the event line.
#[derive(Debug, Clone)]
pub struct EventHeader {
    /// Event number.
    pub number: u32,
    /// Reel name.
    pub reel: String,
    /// Raw track-type string (e.g. `"V"`, `"A"`, `"AA/V"`).
    pub track_type_raw: String,
    /// Raw edit-type string (e.g. `"C"`, `"D"`).
    pub edit_type_raw: String,
}

/// The lazily-parsed detail fields of an EDL event block.
///
/// These fields are only parsed when first accessed via
/// [`LazyEvent::detail`].
#[derive(Debug, Clone)]
pub struct EventDetail {
    /// Optional transition duration in frames.
    pub transition_duration: Option<u32>,
    /// Source in timecode string.
    pub source_in_raw: String,
    /// Source out timecode string.
    pub source_out_raw: String,
    /// Record in timecode string.
    pub record_in_raw: String,
    /// Record out timecode string.
    pub record_out_raw: String,
    /// Comment lines associated with this event (raw text after `*`).
    pub comments: Vec<String>,
}

/// An EDL event whose detail fields are resolved lazily on first access.
///
/// The header (event number, reel, track/edit type) is parsed eagerly.
/// The detail (timecodes, transition duration, comments) is stored as a raw
/// string and only parsed on the first call to [`LazyEvent::detail`].
/// Subsequent calls return the cached value — the detail parser is never
/// invoked more than once per event.
pub struct LazyEvent {
    /// Eagerly parsed header fields.
    pub header: EventHeader,
    /// Raw unparsed rest of the event block (event line + following comment lines).
    raw_detail: String,
    /// Lazily resolved detail; `None` until first access.
    detail: std::cell::RefCell<Option<EventDetail>>,
    /// Frame rate needed by the detail parser.
    frame_rate: EdlFrameRate,
}

impl std::fmt::Debug for LazyEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyEvent")
            .field("header", &self.header)
            .field("raw_detail", &self.raw_detail)
            .field(
                "detail",
                if self.detail.borrow().is_some() {
                    &"Some(<resolved>)"
                } else {
                    &"None"
                },
            )
            .finish()
    }
}

impl LazyEvent {
    /// Access the event detail, parsing it on the first call and caching the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the raw detail cannot be parsed.
    pub fn detail(&self) -> EdlResult<std::cell::Ref<'_, EventDetail>> {
        // If not yet resolved, parse now and store.
        if self.detail.borrow().is_none() {
            let parsed = Self::parse_detail(&self.raw_detail, self.frame_rate)?;
            *self.detail.borrow_mut() = Some(parsed);
        }
        // SAFETY: we just guaranteed the RefCell contains Some(…).
        Ok(std::cell::Ref::map(self.detail.borrow(), |opt| {
            opt.as_ref().expect("detail was just populated")
        }))
    }

    /// Parse the raw detail string into an [`EventDetail`].
    fn parse_detail(raw: &str, frame_rate: EdlFrameRate) -> EdlResult<EventDetail> {
        let mut comments = Vec::new();
        let mut event_line: Option<&str> = None;

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('*') {
                if let Some(c) = trimmed.strip_prefix('*') {
                    comments.push(c.trim().to_string());
                }
            } else if event_line.is_none() {
                event_line = Some(trimmed);
            }
        }

        let ev_line = event_line.ok_or_else(|| EdlError::parse(0, "no event line in detail"))?;

        // We only need the timecode portion of the line.  The format after the
        // edit-type token is:  [duration] SRC_IN SRC_OUT REC_IN REC_OUT
        // Skip the first 3 tokens (number, reel, track) and the edit-type token.
        let tokens: Vec<&str> = ev_line.split_whitespace().collect();
        // tokens[0] = number, [1] = reel, [2] = track, [3] = edit_type
        // remaining tokens start at index 4
        if tokens.len() < 8 {
            return Err(EdlError::parse(0, "insufficient tokens on event line"));
        }

        let mut idx = 4usize;

        // Optional transition duration: present when the token at `idx` is all-digits
        // and the next tokens are timecodes.
        let transition_duration = if tokens.get(idx).map_or(false, |t| {
            t.chars().all(|c| c.is_ascii_digit())
                && t.len() <= 5
                && !t.contains(':')
                && !t.contains(';')
        }) && tokens.len() >= 9
        {
            let dur = tokens[idx]
                .parse::<u32>()
                .map_err(|_| EdlError::parse(0, "invalid transition duration"))?;
            idx += 1;
            Some(dur)
        } else {
            None
        };

        // We need exactly 4 timecode tokens.
        if tokens.len() < idx + 4 {
            return Err(EdlError::parse(0, "missing timecode tokens"));
        }

        // Validate they look like timecodes (contain ':' or ';').
        for tc_tok in &tokens[idx..idx + 4] {
            if !tc_tok.contains(':') && !tc_tok.contains(';') {
                return Err(EdlError::parse(
                    0,
                    format!("token does not look like a timecode: {tc_tok}"),
                ));
            }
        }

        // Verify the frame rate is understood (parse one timecode as a check).
        EdlTimecode::parse(tokens[idx], frame_rate)
            .map_err(|e| EdlError::parse(0, format!("invalid source_in timecode: {e}")))?;

        Ok(EventDetail {
            transition_duration,
            source_in_raw: tokens[idx].to_string(),
            source_out_raw: tokens[idx + 1].to_string(),
            record_in_raw: tokens[idx + 2].to_string(),
            record_out_raw: tokens[idx + 3].to_string(),
            comments,
        })
    }
}

/// Parse an EDL in lazy mode, returning a list of [`LazyEvent`]s.
///
/// Only the event header (number, reel, track, edit type) is parsed during
/// this call.  Detail fields (timecodes, transition duration, comments) are
/// deferred until [`LazyEvent::detail`] is called.
///
/// # Errors
///
/// Returns an error if the input cannot be scanned for headers.
pub fn parse_lazy(input: &str, frame_rate: EdlFrameRate) -> EdlResult<Vec<LazyEvent>> {
    let mut events: Vec<LazyEvent> = Vec::new();
    let mut current_header: Option<EventHeader> = None;
    let mut current_raw_lines: Vec<&str> = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // TITLE / FCM header lines — skip silently
        if trimmed.starts_with("TITLE:") || trimmed.starts_with("FCM:") {
            continue;
        }

        // Comment line: belongs to the current event block
        if trimmed.starts_with('*') {
            current_raw_lines.push(line);
            continue;
        }

        // Check if this line starts with an event number (digit-first)
        if trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            // Flush the previous event (if any)
            if let Some(header) = current_header.take() {
                let raw_detail = current_raw_lines.join("\n");
                current_raw_lines.clear();
                events.push(LazyEvent {
                    header,
                    raw_detail,
                    detail: std::cell::RefCell::new(None),
                    frame_rate,
                });
            }

            // Parse the header eagerly (number + reel + track + edit_type only)
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() >= 4 {
                let number = tokens[0]
                    .parse::<u32>()
                    .map_err(|_| EdlError::parse(0, "invalid event number"))?;
                current_header = Some(EventHeader {
                    number,
                    reel: tokens[1].to_string(),
                    track_type_raw: tokens[2].to_string(),
                    edit_type_raw: tokens[3].to_string(),
                });
                current_raw_lines.push(line);
            }
        }
    }

    // Flush the last event
    if let Some(header) = current_header {
        let raw_detail = current_raw_lines.join("\n");
        events.push(LazyEvent {
            header,
            raw_detail,
            detail: std::cell::RefCell::new(None),
            frame_rate,
        });
    }

    Ok(events)
}

// ─── Frame rate parsing helper ─────────────────────────────────────────────

/// Parse frame rate from FCM line.
#[allow(dead_code)]
fn parse_fcm(input: &str) -> EdlResult<EdlFrameRate> {
    let upper = input.to_uppercase();
    // Check for "NON" before checking for "DROP" since "NON DROP FRAME" contains "DROP"
    if upper.contains("NON") {
        Ok(EdlFrameRate::Fps2997NDF)
    } else if upper.contains("DROP") {
        Ok(EdlFrameRate::Fps2997DF)
    } else {
        Ok(EdlFrameRate::Fps2997NDF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_edl() {
        let edl_text = r#"TITLE: Test EDL
FCM: DROP FRAME

001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
* FROM CLIP NAME: SHOT_001.MOV

002  AX       V     D    030 01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00
* FROM CLIP NAME: SHOT_002.MOV
"#;

        let mut parser = EdlParser::new();
        let edl = parser.parse(edl_text).expect("failed to parse");

        assert_eq!(edl.title, Some("Test EDL".to_string()));
        assert_eq!(edl.events.len(), 2);
        assert_eq!(edl.events[0].number, 1);
        assert_eq!(edl.events[0].edit_type, EditType::Cut);
        assert_eq!(edl.events[1].number, 2);
        assert_eq!(edl.events[1].edit_type, EditType::Dissolve);
        assert_eq!(edl.events[1].transition_duration, Some(30));
    }

    #[test]
    fn test_parse_comment_line() {
        let comment = EdlParser::parse_comment_line("* This is a comment");
        assert_eq!(comment, Some("This is a comment".to_string()));
    }

    #[test]
    fn test_timecode_parser() {
        let (_, tc) = EdlParser::timecode_parser("01:02:03:04", EdlFrameRate::Fps25)
            .expect("operation should succeed");
        assert_eq!(tc.hours(), 1);
        assert_eq!(tc.minutes(), 2);
        assert_eq!(tc.seconds(), 3);
        assert_eq!(tc.frames(), 4);
    }

    #[test]
    fn test_track_type_parser() {
        let (_, track) = EdlParser::track_type_parser("V").expect("operation should succeed");
        assert_eq!(track, TrackType::Video);

        let (_, track) = EdlParser::track_type_parser("A").expect("operation should succeed");
        assert_eq!(track, TrackType::Audio(AudioChannel::A1));

        let (_, track) = EdlParser::track_type_parser("AA/V").expect("operation should succeed");
        assert_eq!(track, TrackType::AudioPairWithVideo);
    }

    #[test]
    fn test_edit_type_parser() {
        let (_, edit) = EdlParser::edit_type_parser("C").expect("operation should succeed");
        assert_eq!(edit, EditType::Cut);

        let (_, edit) = EdlParser::edit_type_parser("D").expect("operation should succeed");
        assert_eq!(edit, EditType::Dissolve);
    }

    #[test]
    fn test_event_line_parser() {
        let line = "001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00";
        let (_, event) = EdlParser::event_line_parser(line, EdlFrameRate::Fps2997DF)
            .expect("operation should succeed");

        assert_eq!(event.number, 1);
        assert_eq!(event.reel, "AX");
        assert_eq!(event.track, TrackType::Video);
        assert_eq!(event.edit_type, EditType::Cut);
    }

    #[test]
    fn test_event_with_transition() {
        let line = "002  AX       V     D    030 01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00";
        let (_, event) = EdlParser::event_line_parser(line, EdlFrameRate::Fps2997DF)
            .expect("operation should succeed");

        assert_eq!(event.number, 2);
        assert_eq!(event.edit_type, EditType::Dissolve);
        assert_eq!(event.transition_duration, Some(30));
    }

    #[test]
    fn test_parse_fcm() {
        assert_eq!(
            parse_fcm("DROP FRAME").expect("operation should succeed"),
            EdlFrameRate::Fps2997DF
        );
        assert_eq!(
            parse_fcm("NON-DROP FRAME").expect("operation should succeed"),
            EdlFrameRate::Fps2997NDF
        );
        assert_eq!(
            parse_fcm("NON DROP FRAME").expect("operation should succeed"),
            EdlFrameRate::Fps2997NDF
        );
    }

    #[test]
    fn test_parse_clip_name_comment() {
        let edl_text = r#"001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
* FROM CLIP NAME: test_clip.mov"#;

        let mut parser = EdlParser::new();
        let edl = parser.parse(edl_text).expect("failed to parse");

        assert_eq!(edl.events.len(), 1);
        assert_eq!(edl.events[0].clip_name, Some("test_clip.mov".to_string()));
    }

    const LAZY_SAMPLE_EDL: &str = "\
TITLE: Lazy Sample\n\
FCM: DROP FRAME\n\
\n\
001  A001     V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00\n\
* FROM CLIP NAME: shot001.mov\n\
\n\
002  A002     V     D    030 01:00:05;00 01:00:10;00 01:00:05;00 01:00:10;00\n\
* FROM CLIP NAME: shot002.mov\n\
* Generic comment\n\
\n\
003  B001     V     C        01:00:10;00 01:00:15;00 01:00:10;00 01:00:15;00\n";

    /// Accessing only the `.header` fields of lazy events must NOT invoke the
    /// detail parser.  We verify this by tracking the detail borrow count.
    #[test]
    fn test_lazy_parse_headers_only() {
        let events = parse_lazy(LAZY_SAMPLE_EDL, EdlFrameRate::Fps2997DF)
            .expect("parse_lazy should succeed");

        assert_eq!(events.len(), 3);

        // Access only header fields — detail must remain unparsed
        for ev in &events {
            let _ = ev.header.number;
            let _ = &ev.header.reel;
            let _ = &ev.header.track_type_raw;
            let _ = &ev.header.edit_type_raw;
        }

        // Confirm that no detail has been resolved yet
        for ev in &events {
            assert!(
                ev.detail.borrow().is_none(),
                "detail should not have been parsed when accessing only header fields"
            );
        }

        // Check header values are correct
        assert_eq!(events[0].header.number, 1);
        assert_eq!(events[0].header.reel, "A001");
        assert_eq!(events[1].header.number, 2);
        assert_eq!(events[1].header.reel, "A002");
        assert_eq!(events[2].header.number, 3);
        assert_eq!(events[2].header.reel, "B001");
    }

    /// Accessing `.detail()` should parse the raw block and cache it;
    /// a second call must return the same data without re-invoking the parser.
    #[test]
    fn test_lazy_detail_resolves() {
        let events = parse_lazy(LAZY_SAMPLE_EDL, EdlFrameRate::Fps2997DF)
            .expect("parse_lazy should succeed");

        assert_eq!(events.len(), 3);

        // Before any detail access, nothing is cached
        assert!(events[0].detail.borrow().is_none());

        // First access — triggers parsing
        {
            let detail = events[0].detail().expect("detail should resolve");
            assert_eq!(detail.source_in_raw, "01:00:00;00");
            assert_eq!(detail.source_out_raw, "01:00:05;00");
            assert_eq!(detail.record_in_raw, "01:00:00;00");
            assert_eq!(detail.record_out_raw, "01:00:05;00");
            assert!(detail.transition_duration.is_none());
            assert_eq!(detail.comments.len(), 1);
            assert!(detail.comments[0].contains("shot001"));
        }

        // After first access, detail is cached
        assert!(events[0].detail.borrow().is_some());

        // Second access — must return the same data (caching verified by
        // the fact that the RefCell still holds a single Some value)
        {
            let detail2 = events[0]
                .detail()
                .expect("detail should resolve on second call");
            assert_eq!(detail2.source_in_raw, "01:00:00;00");
        }

        // Event 2 has a transition duration
        {
            let detail2 = events[1].detail().expect("event 2 detail should resolve");
            assert_eq!(detail2.transition_duration, Some(30));
            assert_eq!(detail2.comments.len(), 2);
        }
    }
}
