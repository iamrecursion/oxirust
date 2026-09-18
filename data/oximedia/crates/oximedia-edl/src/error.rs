//! Error types for EDL operations.

/// Result type for EDL operations.
pub type EdlResult<T> = Result<T, EdlError>;

/// Errors that can occur during EDL parsing, generation, and validation.
#[derive(Debug, thiserror::Error)]
pub enum EdlError {
    /// I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse error at a specific line.
    #[error("Parse error at line {line}: {message}")]
    Parse {
        /// Line number where the error occurred.
        line: usize,
        /// Description of the parse error.
        message: String,
    },

    /// Invalid timecode encountered.
    #[error("Invalid timecode at line {line}: {message}")]
    InvalidTimecode {
        /// Line number where the error occurred.
        line: usize,
        /// Description of the timecode error.
        message: String,
    },

    /// Invalid event number.
    #[error("Invalid event number: {0}")]
    InvalidEventNumber(u32),

    /// Invalid edit type.
    #[error("Invalid edit type: {0}")]
    InvalidEditType(String),

    /// Invalid track type.
    #[error("Invalid track type: {0}")]
    InvalidTrackType(String),

    /// Invalid transition duration.
    #[error("Invalid transition duration: {0}")]
    InvalidTransitionDuration(String),

    /// Invalid reel name.
    #[error("Invalid reel name: {0}")]
    InvalidReelName(String),

    /// Invalid motion effect.
    #[error("Invalid motion effect: {0}")]
    InvalidMotionEffect(String),

    /// Invalid audio channel.
    #[error("Invalid audio channel: {0}")]
    InvalidAudioChannel(String),

    /// Unsupported EDL format.
    #[error("Unsupported EDL format: {0}")]
    UnsupportedFormat(String),

    /// EDL validation error.
    #[error("Validation error: {0}")]
    ValidationError(String),

    /// Event not found.
    #[error("Event {0} not found")]
    EventNotFound(u32),

    /// Timecode out of range.
    #[error("Timecode out of range: {0}")]
    TimecodeOutOfRange(String),

    /// Invalid frame rate.
    #[error("Invalid frame rate: {0}")]
    InvalidFrameRate(String),

    /// Invalid drop frame mode.
    #[error("Invalid drop frame mode: {0}")]
    InvalidDropFrameMode(String),

    /// Comment parsing error.
    #[error("Comment parsing error at line {line}: {message}")]
    CommentError {
        /// Line number where the error occurred.
        line: usize,
        /// Description of the comment error.
        message: String,
    },

    /// Event overlap detected.
    #[error("Event overlap detected: event {event1} overlaps with event {event2}")]
    EventOverlap {
        /// First overlapping event.
        event1: u32,
        /// Second overlapping event.
        event2: u32,
    },

    /// Gap in timeline detected.
    #[error("Gap in timeline detected between event {event1} and event {event2}")]
    TimelineGap {
        /// Event before the gap.
        event1: u32,
        /// Event after the gap.
        event2: u32,
    },

    /// Invalid wipe pattern.
    #[error("Invalid wipe pattern: {0}")]
    InvalidWipePattern(String),

    /// Invalid key type.
    #[error("Invalid key type: {0}")]
    InvalidKeyType(String),

    /// Conversion error between EDL formats.
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid source timecode range.
    #[error("Invalid source timecode range: source out must be greater than source in")]
    InvalidSourceRange,

    /// Invalid record timecode range.
    #[error("Invalid record timecode range: record out must be greater than record in")]
    InvalidRecordRange,
}

impl EdlError {
    /// Creates a new parse error at the given line.
    #[must_use]
    pub fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            line,
            message: message.into(),
        }
    }

    /// Creates a new invalid timecode error.
    #[must_use]
    pub fn invalid_timecode(line: usize, message: impl Into<String>) -> Self {
        Self::InvalidTimecode {
            line,
            message: message.into(),
        }
    }

    /// Creates a new comment error.
    #[must_use]
    pub fn comment_error(line: usize, message: impl Into<String>) -> Self {
        Self::CommentError {
            line,
            message: message.into(),
        }
    }

    /// Creates a new validation error.
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::ValidationError(message.into())
    }

    /// Creates a new event overlap error.
    #[must_use]
    pub const fn event_overlap(event1: u32, event2: u32) -> Self {
        Self::EventOverlap { event1, event2 }
    }

    /// Creates a new timeline gap error.
    #[must_use]
    pub const fn timeline_gap(event1: u32, event2: u32) -> Self {
        Self::TimelineGap { event1, event2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error() {
        let err = EdlError::parse(42, "Invalid syntax");
        assert!(matches!(err, EdlError::Parse { line: 42, .. }));
        let msg = format!("{err}");
        assert!(msg.contains("42"));
        assert!(msg.contains("Invalid syntax"));
    }

    #[test]
    fn test_invalid_timecode() {
        let err = EdlError::invalid_timecode(10, "Out of range");
        assert!(matches!(err, EdlError::InvalidTimecode { line: 10, .. }));
    }

    #[test]
    fn test_validation_error() {
        let err = EdlError::validation("Timeline gap detected");
        assert!(matches!(err, EdlError::ValidationError(_)));
    }

    #[test]
    fn test_event_overlap() {
        let err = EdlError::event_overlap(1, 2);
        assert!(matches!(
            err,
            EdlError::EventOverlap {
                event1: 1,
                event2: 2
            }
        ));
    }
}
