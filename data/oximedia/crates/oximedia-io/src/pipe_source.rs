//! Pipe and stdin media source.
//!
//! Provides [`PipeSource`] for reading media data from Unix pipes and stdin.
//! Pipes are non-seekable, forward-only byte streams.

#![allow(dead_code)]

/// State of a pipe source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipeState {
    /// The pipe is open and data can be read.
    Open,
    /// The pipe has reached end-of-file.
    Eof,
    /// The pipe was explicitly closed.
    Closed,
    /// An error occurred on the pipe.
    Error,
}

/// Configuration for a [`PipeSource`].
#[derive(Debug, Clone)]
pub struct PipeSourceConfig {
    /// Internal buffer capacity in bytes.
    pub buffer_capacity: usize,
    /// Descriptive label for the pipe (e.g. "stdin", "pipe:0").
    pub label: String,
}

impl Default for PipeSourceConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 64 * 1024,
            label: "pipe".to_string(),
        }
    }
}

impl PipeSourceConfig {
    /// Create a config for stdin.
    #[must_use]
    pub fn stdin() -> Self {
        Self {
            buffer_capacity: 64 * 1024,
            label: "stdin".to_string(),
        }
    }

    /// Set the buffer capacity.
    #[must_use]
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set the descriptive label.
    #[must_use]
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }
}

/// A media source that reads from a pipe or stdin-like stream.
///
/// `PipeSource` is explicitly non-seekable: it only supports forward reads.
/// Data is consumed in order and cannot be re-read once consumed.
///
/// # Example
///
/// ```
/// use oximedia_io::pipe_source::{PipeSource, PipeSourceConfig};
///
/// let mut pipe = PipeSource::new(PipeSourceConfig::stdin());
/// assert!(!pipe.is_seekable());
/// assert_eq!(pipe.position(), 0);
/// ```
#[derive(Debug)]
pub struct PipeSource {
    config: PipeSourceConfig,
    state: PipeState,
    buffer: Vec<u8>,
    /// How many valid bytes are in `buffer` (starting from index 0).
    buffer_len: usize,
    /// How many bytes have been consumed from the buffer.
    buffer_pos: usize,
    /// Total bytes read over the lifetime of this source.
    total_read: u64,
}

impl PipeSource {
    /// Create a new `PipeSource` with the given configuration.
    #[must_use]
    pub fn new(config: PipeSourceConfig) -> Self {
        let cap = config.buffer_capacity;
        Self {
            config,
            state: PipeState::Open,
            buffer: vec![0u8; cap],
            buffer_len: 0,
            buffer_pos: 0,
            total_read: 0,
        }
    }

    /// Create a `PipeSource` configured for stdin.
    #[must_use]
    pub fn from_stdin() -> Self {
        Self::new(PipeSourceConfig::stdin())
    }

    /// Return the pipe's label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.config.label
    }

    /// Return the current state of the pipe.
    #[must_use]
    pub fn state(&self) -> PipeState {
        self.state
    }

    /// Return the total number of bytes consumed from this pipe.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.total_read
    }

    /// Pipes are never seekable.
    #[must_use]
    pub fn is_seekable(&self) -> bool {
        false
    }

    /// Pipes have no known length (streaming).
    #[must_use]
    pub fn len(&self) -> Option<u64> {
        None
    }

    /// Always returns `false` since pipes have no known length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Return the amount of buffered-but-unread data.
    #[must_use]
    pub fn buffered_bytes(&self) -> usize {
        self.buffer_len - self.buffer_pos
    }

    /// Feed raw data into the internal buffer (simulates pipe input).
    ///
    /// Returns the number of bytes actually copied into the buffer.
    pub fn feed(&mut self, data: &[u8]) -> usize {
        if self.state != PipeState::Open {
            return 0;
        }
        // Compact the buffer first
        if self.buffer_pos > 0 {
            let remaining = self.buffer_len - self.buffer_pos;
            self.buffer.copy_within(self.buffer_pos..self.buffer_len, 0);
            self.buffer_len = remaining;
            self.buffer_pos = 0;
        }
        let space = self.buffer.len() - self.buffer_len;
        let to_copy = data.len().min(space);
        self.buffer[self.buffer_len..self.buffer_len + to_copy].copy_from_slice(&data[..to_copy]);
        self.buffer_len += to_copy;
        to_copy
    }

    /// Read from the internal buffer into `buf`.
    ///
    /// Returns the number of bytes read. Returns 0 when there is no
    /// buffered data (the caller should feed more or check for EOF).
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        if self.state == PipeState::Closed || self.state == PipeState::Error {
            return 0;
        }
        let available = self.buffer_len - self.buffer_pos;
        if available == 0 {
            if self.state == PipeState::Eof {
                return 0;
            }
            return 0;
        }
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + to_read]);
        self.buffer_pos += to_read;
        self.total_read += to_read as u64;
        to_read
    }

    /// Signal that the pipe has reached end-of-file.
    pub fn signal_eof(&mut self) {
        self.state = PipeState::Eof;
    }

    /// Close the pipe, discarding any remaining buffer.
    pub fn close(&mut self) {
        self.state = PipeState::Closed;
        self.buffer_len = 0;
        self.buffer_pos = 0;
    }

    /// Signal an error on the pipe.
    pub fn signal_error(&mut self) {
        self.state = PipeState::Error;
    }

    /// Return whether the pipe is still open for reading.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.state == PipeState::Open
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe_source_defaults() {
        let pipe = PipeSource::from_stdin();
        assert_eq!(pipe.label(), "stdin");
        assert_eq!(pipe.state(), PipeState::Open);
        assert_eq!(pipe.position(), 0);
        assert!(!pipe.is_seekable());
        assert_eq!(pipe.len(), None);
        assert!(!pipe.is_empty());
        assert!(pipe.is_open());
    }

    #[test]
    fn test_pipe_source_config() {
        let cfg = PipeSourceConfig::default()
            .with_buffer_capacity(1024)
            .with_label("my-pipe");
        let pipe = PipeSource::new(cfg);
        assert_eq!(pipe.label(), "my-pipe");
    }

    #[test]
    fn test_pipe_feed_and_read() {
        let cfg = PipeSourceConfig::default().with_buffer_capacity(32);
        let mut pipe = PipeSource::new(cfg);

        let fed = pipe.feed(&[1, 2, 3, 4, 5]);
        assert_eq!(fed, 5);
        assert_eq!(pipe.buffered_bytes(), 5);

        let mut buf = [0u8; 3];
        let n = pipe.read(&mut buf);
        assert_eq!(n, 3);
        assert_eq!(&buf, &[1, 2, 3]);
        assert_eq!(pipe.position(), 3);

        let n = pipe.read(&mut buf);
        assert_eq!(n, 2);
        assert_eq!(&buf[..2], &[4, 5]);
        assert_eq!(pipe.position(), 5);
    }

    #[test]
    fn test_pipe_read_empty() {
        let mut pipe = PipeSource::from_stdin();
        let mut buf = [0u8; 10];
        let n = pipe.read(&mut buf);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_pipe_eof() {
        let mut pipe = PipeSource::from_stdin();
        pipe.feed(&[1, 2, 3]);
        pipe.signal_eof();
        assert_eq!(pipe.state(), PipeState::Eof);
        assert!(!pipe.is_open());

        // Can still read buffered data
        let mut buf = [0u8; 10];
        let n = pipe.read(&mut buf);
        assert_eq!(n, 3);

        // After buffer exhausted, reads return 0
        let n = pipe.read(&mut buf);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_pipe_close_discards_buffer() {
        let mut pipe = PipeSource::from_stdin();
        pipe.feed(&[1, 2, 3]);
        pipe.close();
        assert_eq!(pipe.state(), PipeState::Closed);
        assert!(!pipe.is_open());

        let mut buf = [0u8; 10];
        let n = pipe.read(&mut buf);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_pipe_error_state() {
        let mut pipe = PipeSource::from_stdin();
        pipe.feed(&[1, 2]);
        pipe.signal_error();
        assert_eq!(pipe.state(), PipeState::Error);

        let mut buf = [0u8; 10];
        let n = pipe.read(&mut buf);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_pipe_feed_when_closed() {
        let mut pipe = PipeSource::from_stdin();
        pipe.close();
        let fed = pipe.feed(&[1, 2, 3]);
        assert_eq!(fed, 0);
    }

    #[test]
    fn test_pipe_buffer_compaction() {
        let cfg = PipeSourceConfig::default().with_buffer_capacity(8);
        let mut pipe = PipeSource::new(cfg);

        pipe.feed(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut buf = [0u8; 6];
        pipe.read(&mut buf); // consume 6

        // Feed more — triggers compaction of remaining 2 bytes
        let fed = pipe.feed(&[9, 10, 11, 12, 13, 14]);
        assert_eq!(fed, 6); // 2 remaining + 6 new = 8 (full)

        let mut big = [0u8; 8];
        let n = pipe.read(&mut big);
        assert_eq!(n, 8);
        assert_eq!(&big, &[7, 8, 9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_pipe_multiple_reads() {
        let mut pipe = PipeSource::from_stdin();
        pipe.feed(&[10, 20, 30, 40, 50]);

        let mut buf = [0u8; 2];
        assert_eq!(pipe.read(&mut buf), 2);
        assert_eq!(&buf, &[10, 20]);

        assert_eq!(pipe.read(&mut buf), 2);
        assert_eq!(&buf, &[30, 40]);

        assert_eq!(pipe.read(&mut buf), 1);
        assert_eq!(buf[0], 50);

        assert_eq!(pipe.position(), 5);
    }

    #[test]
    fn test_pipe_large_feed_capped_by_capacity() {
        let cfg = PipeSourceConfig::default().with_buffer_capacity(4);
        let mut pipe = PipeSource::new(cfg);
        let fed = pipe.feed(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(fed, 4); // only 4 fit
        assert_eq!(pipe.buffered_bytes(), 4);
    }
}
