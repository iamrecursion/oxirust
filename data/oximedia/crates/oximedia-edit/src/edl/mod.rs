//! Edit Decision List (EDL) parsing and generation.
//!
//! This module provides comprehensive support for various EDL formats used in
//! professional video editing:
//!
//! - **CMX 3600**: Industry-standard EDL format for linear editing systems
//! - **ALE (Avid Log Exchange)**: Metadata-rich format for Avid systems
//! - **FCPXML**: Final Cut Pro X XML project format
//! - **AAF-EDL**: Advanced Authoring Format style EDL representation
//! - **OpenTimelineIO**: Modern interchange format for editorial data
//!
//! # Format Conversion
//!
//! The module supports bidirectional conversion between formats, with automatic
//! handling of feature mapping and lossy conversion warnings:
//!
//! ```no_run
//! use oximedia_edit::edl::{Edl, EdlFormat};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse CMX 3600 EDL
//! let cmx_content = std::fs::read_to_string("timeline.edl")?;
//! let edl = Edl::parse(&cmx_content, EdlFormat::Cmx3600)?;
//!
//! // Convert to FCPXML
//! let fcpxml = edl.to_format(EdlFormat::FcpXml)?;
//! std::fs::write("timeline.fcpxml", fcpxml)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Validation
//!
//! Built-in validation ensures EDL integrity:
//!
//! ```no_run
//! use oximedia_edit::edl::{Edl, validator::EdlValidator};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let edl_content = "";
//! let edl = Edl::parse(&edl_content, oximedia_edit::edl::EdlFormat::Cmx3600)?;
//!
//! let validator = EdlValidator::new();
//! let report = validator.validate(&edl)?;
//!
//! if !report.is_valid() {
//!     for error in &report.errors {
//!         eprintln!("Error: {}", error);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

pub mod aaf_edl;
pub mod ale;
pub mod cmx3600;
pub mod converter;
pub mod fcpxml;
pub mod otio;
pub mod validator;

use oximedia_core::Rational;
use std::collections::HashMap;
use thiserror::Error;

/// EDL error types.
#[derive(Debug, Error)]
pub enum EdlError {
    /// Parse error with context.
    #[error("Parse error at line {line}: {message}")]
    ParseError {
        /// Line number where error occurred.
        line: usize,
        /// Error message.
        message: String,
    },

    /// Invalid timecode format.
    #[error("Invalid timecode: {0}")]
    InvalidTimecode(String),

    /// Invalid edit number.
    #[error("Invalid edit number: {0}")]
    InvalidEditNumber(u32),

    /// Unsupported feature.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// Conversion error.
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Validation error.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// XML error.
    #[error("XML error: {0}")]
    XmlError(String),

    /// JSON error.
    #[error("JSON error: {0}")]
    JsonError(String),

    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

/// Result type for EDL operations.
pub type EdlResult<T> = Result<T, EdlError>;

/// Supported EDL formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdlFormat {
    /// CMX 3600 format.
    Cmx3600,
    /// ALE (Avid Log Exchange) format.
    Ale,
    /// FCPXML (Final Cut Pro XML) format.
    FcpXml,
    /// AAF-style EDL format.
    AafEdl,
    /// OpenTimelineIO JSON format.
    Otio,
}

/// Timecode representation.
#[derive(Debug, Clone, PartialEq)]
pub struct Timecode {
    /// Hours (0-23).
    pub hours: u8,
    /// Minutes (0-59).
    pub minutes: u8,
    /// Seconds (0-59).
    pub seconds: u8,
    /// Frames (0-fps).
    pub frames: u8,
    /// Drop-frame flag.
    pub drop_frame: bool,
    /// Frame rate.
    pub frame_rate: Rational,
}

impl Timecode {
    /// Create a new timecode.
    #[must_use]
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        drop_frame: bool,
        frame_rate: Rational,
    ) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame,
            frame_rate,
        }
    }

    /// Create timecode from frame count.
    #[must_use]
    pub fn from_frames(frames: i64, frame_rate: Rational, drop_frame: bool) -> Self {
        let fps = frame_rate.to_f64() as i64;
        let frames_per_minute = fps * 60;
        let frames_per_hour = frames_per_minute * 60;

        let hours = (frames / frames_per_hour) as u8;
        let remaining = frames % frames_per_hour;
        let minutes = (remaining / frames_per_minute) as u8;
        let remaining = remaining % frames_per_minute;
        let seconds = (remaining / fps) as u8;
        let frames = (remaining % fps) as u8;

        Self::new(hours, minutes, seconds, frames, drop_frame, frame_rate)
    }

    /// Convert timecode to frame count.
    #[must_use]
    pub fn to_frames(&self) -> i64 {
        let fps = self.frame_rate.to_f64() as i64;
        let total_seconds =
            i64::from(self.hours) * 3600 + i64::from(self.minutes) * 60 + i64::from(self.seconds);
        total_seconds * fps + i64::from(self.frames)
    }

    /// Parse timecode from string (HH:MM:SS:FF or HH:MM:SS;FF for drop-frame).
    pub fn parse(s: &str, frame_rate: Rational) -> EdlResult<Self> {
        let parts: Vec<&str> = s.split(&[':', ';'][..]).collect();
        if parts.len() != 4 {
            return Err(EdlError::InvalidTimecode(s.to_string()));
        }

        let hours = parts[0]
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(s.to_string()))?;
        let minutes = parts[1]
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(s.to_string()))?;
        let seconds = parts[2]
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(s.to_string()))?;
        let frames = parts[3]
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(s.to_string()))?;
        let drop_frame = s.contains(';');

        Ok(Self::new(
            hours, minutes, seconds, frames, drop_frame, frame_rate,
        ))
    }

    /// Format timecode as string.
    #[must_use]
    pub fn format(&self) -> String {
        let separator = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

/// Edit type (transition type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditType {
    /// Cut (instant transition).
    Cut,
    /// Dissolve (cross-fade).
    Dissolve,
    /// Wipe transition.
    Wipe,
    /// Key (overlay/composite).
    Key,
}

/// Edit event in an EDL.
#[derive(Debug, Clone)]
pub struct EdlEvent {
    /// Event number.
    pub number: u32,
    /// Source reel/tape name.
    pub reel: String,
    /// Track type (V for video, A for audio, etc.).
    pub track: String,
    /// Edit type.
    pub edit_type: EditType,
    /// Source in timecode.
    pub source_in: Timecode,
    /// Source out timecode.
    pub source_out: Timecode,
    /// Record in timecode.
    pub record_in: Timecode,
    /// Record out timecode.
    pub record_out: Timecode,
    /// Transition duration (in frames, for dissolves/wipes).
    pub transition_duration: Option<u32>,
    /// Motion effects (speed changes, freeze frames, etc.).
    pub motion_effect: Option<MotionEffect>,
    /// Comments associated with this event.
    pub comments: Vec<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Motion effect specification.
#[derive(Debug, Clone)]
pub struct MotionEffect {
    /// Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed).
    pub speed: f64,
    /// Freeze frame flag.
    pub freeze: bool,
    /// Reverse motion flag.
    pub reverse: bool,
    /// Entry point (timecode).
    pub entry: Option<Timecode>,
}

/// Complete EDL structure.
#[derive(Debug, Clone)]
pub struct Edl {
    /// Title of the EDL.
    pub title: String,
    /// Frame rate.
    pub frame_rate: Rational,
    /// Drop-frame flag.
    pub drop_frame: bool,
    /// List of events.
    pub events: Vec<EdlEvent>,
    /// Global comments.
    pub comments: Vec<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl Edl {
    /// Create a new empty EDL.
    #[must_use]
    pub fn new(title: String, frame_rate: Rational, drop_frame: bool) -> Self {
        Self {
            title,
            frame_rate,
            drop_frame,
            events: Vec::new(),
            comments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Parse EDL from string with specified format.
    pub fn parse(content: &str, format: EdlFormat) -> EdlResult<Self> {
        match format {
            EdlFormat::Cmx3600 => cmx3600::parse(content),
            EdlFormat::Ale => ale::parse(content),
            EdlFormat::FcpXml => fcpxml::parse(content),
            EdlFormat::AafEdl => aaf_edl::parse(content),
            EdlFormat::Otio => otio::parse(content),
        }
    }

    /// Convert EDL to specified format.
    pub fn to_format(&self, format: EdlFormat) -> EdlResult<String> {
        match format {
            EdlFormat::Cmx3600 => cmx3600::write(self),
            EdlFormat::Ale => ale::write(self),
            EdlFormat::FcpXml => fcpxml::write(self),
            EdlFormat::AafEdl => aaf_edl::write(self),
            EdlFormat::Otio => otio::write(self),
        }
    }

    /// Add an event to the EDL.
    pub fn add_event(&mut self, event: EdlEvent) {
        self.events.push(event);
    }

    /// Get event by number.
    #[must_use]
    pub fn get_event(&self, number: u32) -> Option<&EdlEvent> {
        self.events.iter().find(|e| e.number == number)
    }

    /// Sort events by record in timecode.
    pub fn sort_events(&mut self) {
        self.events.sort_by_key(|e| e.record_in.to_frames());
    }

    /// Validate the EDL.
    pub fn validate(&self) -> EdlResult<validator::ValidationReport> {
        validator::EdlValidator::new().validate(self)
    }
}
