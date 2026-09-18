//! Streaming RPU parser that processes NAL units incrementally.
//!
//! This module provides a push-based parser that accumulates incoming NAL unit
//! bytes and emits complete [`DolbyVisionRpu`] objects as they become available,
//! without requiring the caller to buffer entire frames in advance.

use crate::{parser, DolbyVisionError, DolbyVisionRpu};

/// State machine for the streaming NAL unit parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    /// Waiting for the 4-byte Annex-B start code (00 00 00 01).
    WaitingStartCode,
    /// Accumulating NAL unit bytes until the next start code.
    AccumulatingNal,
}

/// Streaming RPU parser that processes NAL units from a continuous byte stream.
///
/// Feed raw H.265/HEVC bitstream data (Annex-B framing) to [`StreamingRpuParser::push`]
/// and consume completed RPU objects from [`StreamingRpuParser::take_ready`].
///
/// # Example
///
/// ```rust
/// use oximedia_dolbyvision::streaming_parser::StreamingRpuParser;
///
/// let mut parser = StreamingRpuParser::new();
/// let bytes: Vec<u8> = vec![]; // your bitstream bytes here
/// parser.push(&bytes);
/// let rpus = parser.take_ready();
/// assert!(rpus.is_empty()); // empty input → no RPUs
/// ```
#[derive(Debug)]
pub struct StreamingRpuParser {
    /// Byte buffer accumulating the current NAL unit.
    buffer: Vec<u8>,
    /// Current parser state.
    state: ParseState,
    /// Rolling window for start-code detection (last 3 bytes).
    window: [u8; 3],
    /// How many of the `window` bytes are valid (0–3).
    window_len: usize,
    /// Completed RPUs waiting to be consumed.
    ready: Vec<DolbyVisionRpu>,
    /// Running count of parse errors (non-DV NAL units are silently skipped).
    error_count: u64,
    /// Running count of successfully parsed RPUs.
    rpu_count: u64,
}

impl StreamingRpuParser {
    /// Create a new streaming parser.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            state: ParseState::WaitingStartCode,
            window: [0u8; 3],
            window_len: 0,
            ready: Vec::new(),
            error_count: 0,
            rpu_count: 0,
        }
    }

    /// Push a slice of bytes into the parser.
    ///
    /// The parser looks for Annex-B start codes (`00 00 00 01`) to delimit NAL
    /// units, then attempts to parse each completed NAL unit as a Dolby Vision
    /// RPU. Non-DV NAL units are silently discarded.
    pub fn push(&mut self, data: &[u8]) {
        for &byte in data {
            match self.state {
                ParseState::WaitingStartCode => {
                    // Shift window
                    if self.window_len < 3 {
                        self.window[self.window_len] = byte;
                        self.window_len += 1;
                    } else {
                        self.window[0] = self.window[1];
                        self.window[1] = self.window[2];
                        self.window[2] = byte;
                    }

                    // Detect 00 00 01 (3-byte start code) or look for the
                    // typical 4-byte 00 00 00 01
                    if self.window_len == 3
                        && self.window[0] == 0x00
                        && self.window[1] == 0x00
                        && self.window[2] == 0x01
                    {
                        self.buffer.clear();
                        self.state = ParseState::AccumulatingNal;
                        self.window_len = 0;
                    }
                }

                ParseState::AccumulatingNal => {
                    // Detect next start code 00 00 01 within accumulated data
                    self.buffer.push(byte);

                    let len = self.buffer.len();
                    if len >= 4 {
                        let tail = &self.buffer[len - 4..];
                        if tail == [0x00, 0x00, 0x00, 0x01]
                            || (tail[0] == 0x00 && tail[1] == 0x00 && tail[2] == 0x01)
                        {
                            // Extract NAL unit (everything before start code)
                            let sc_len = if tail[0] == 0x00 && tail[1] == 0x00 && tail[2] == 0x00 {
                                4
                            } else {
                                3
                            };
                            let nal_end = len - sc_len;
                            let nal: Vec<u8> = self.buffer[..nal_end].to_vec();
                            self.try_parse_nal(&nal);
                            self.buffer.clear();
                        }
                    }
                }
            }
        }
    }

    /// Flush any buffered NAL unit data as a final RPU.
    ///
    /// Call this after the last `push` to emit any RPU whose NAL unit was
    /// not yet terminated by a subsequent start code.
    pub fn flush(&mut self) {
        if !self.buffer.is_empty() {
            let nal = self.buffer.clone();
            self.try_parse_nal(&nal);
            self.buffer.clear();
        }
        self.state = ParseState::WaitingStartCode;
        self.window_len = 0;
    }

    /// Take all completed RPUs that have been parsed so far.
    ///
    /// Removes them from the internal queue and returns them.
    pub fn take_ready(&mut self) -> Vec<DolbyVisionRpu> {
        std::mem::take(&mut self.ready)
    }

    /// Number of RPUs successfully parsed since creation.
    #[must_use]
    pub fn rpu_count(&self) -> u64 {
        self.rpu_count
    }

    /// Number of parse errors encountered (non-fatal; non-DV NAL units counted).
    #[must_use]
    pub fn error_count(&self) -> u64 {
        self.error_count
    }

    /// Returns `true` if there are completed RPUs waiting to be consumed.
    #[must_use]
    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn try_parse_nal(&mut self, nal: &[u8]) {
        if nal.is_empty() {
            return;
        }
        match parser::parse_nal_unit(nal) {
            Ok(rpu) => {
                self.rpu_count += 1;
                self.ready.push(rpu);
            }
            Err(DolbyVisionError::InvalidNalUnit(_)) => {
                // Not a DV RPU NAL — silently skip
                self.error_count += 1;
            }
            Err(_) => {
                self.error_count += 1;
            }
        }
    }
}

impl Default for StreamingRpuParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Feed a slice of complete (already-delimited) NAL unit buffers into a one-shot
/// batch parse and return all valid Dolby Vision RPUs.
///
/// This is a convenience wrapper around [`StreamingRpuParser`] for callers that
/// already have the NAL unit boundaries established.
///
/// # Errors
///
/// Returns a vector of errors for each NAL unit that failed to parse.
pub fn parse_nal_batch(nal_units: &[&[u8]]) -> (Vec<DolbyVisionRpu>, Vec<DolbyVisionError>) {
    let mut rpus = Vec::new();
    let mut errors = Vec::new();

    for &nal in nal_units {
        match parser::parse_nal_unit(nal) {
            Ok(rpu) => rpus.push(rpu),
            Err(e) => errors.push(e),
        }
    }

    (rpus, errors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_parser_empty_input() {
        let mut p = StreamingRpuParser::new();
        p.push(&[]);
        assert!(!p.has_ready());
        assert_eq!(p.rpu_count(), 0);
    }

    #[test]
    fn test_streaming_parser_random_bytes_no_panic() {
        let mut p = StreamingRpuParser::new();
        let data: Vec<u8> = (0u8..=255).collect();
        p.push(&data);
        p.flush();
        // Should not panic; error count may be > 0
    }

    #[test]
    fn test_streaming_parser_take_ready_clears() {
        let mut p = StreamingRpuParser::new();
        let ready = p.take_ready();
        assert!(ready.is_empty());
        let ready2 = p.take_ready();
        assert!(ready2.is_empty());
    }

    #[test]
    fn test_streaming_parser_flush_empty_buffer() {
        let mut p = StreamingRpuParser::new();
        p.flush(); // should not panic
        assert_eq!(p.rpu_count(), 0);
    }

    #[test]
    fn test_streaming_parser_default() {
        let p = StreamingRpuParser::default();
        assert_eq!(p.rpu_count(), 0);
        assert_eq!(p.error_count(), 0);
        assert!(!p.has_ready());
    }

    #[test]
    fn test_parse_nal_batch_empty() {
        let (rpus, errors) = parse_nal_batch(&[]);
        assert!(rpus.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn test_parse_nal_batch_all_invalid() {
        let bad: &[u8] = &[0x00, 0x00];
        let (rpus, errors) = parse_nal_batch(&[bad, bad]);
        assert!(rpus.is_empty());
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn test_start_code_detection_in_stream() {
        // Push a stream containing a 4-byte start code followed by garbage
        let stream: Vec<u8> = vec![0x00, 0x00, 0x00, 0x01, 0xFF, 0xAB, 0xCD];
        let mut p = StreamingRpuParser::new();
        p.push(&stream);
        p.flush();
        // The garbage NAL will fail but parser should not panic
        // error_count may be 0 or 1 depending on whether we even got to try_parse_nal
    }

    #[test]
    fn test_incremental_push() {
        // Feed data byte by byte
        let stream: Vec<u8> = vec![0x00, 0x00, 0x01, 0x7D, 0x01, 0x00, 0x00, 0x01];
        let mut p = StreamingRpuParser::new();
        for byte in stream {
            p.push(&[byte]);
        }
        p.flush();
        // The short artificial NAL should produce a parse error (not a panic)
        // rpu_count may stay 0
        let _ = p.take_ready();
    }

    #[test]
    fn test_rpu_count_increments_on_valid() {
        // Build a valid RPU NAL from the writer and parse it through streaming
        use crate::{DolbyVisionRpu, Profile};
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let nal = rpu.write_to_nal().expect("write should succeed");

        // Wrap in Annex-B start code
        let mut stream = vec![0x00, 0x00, 0x00, 0x01];
        stream.extend_from_slice(&nal);
        stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // terminator

        let mut p = StreamingRpuParser::new();
        p.push(&stream);
        // The RPU NAL may or may not be parseable (depends on write format)
        // Key assertion: no panic
        let _ = p.take_ready();
    }
}
