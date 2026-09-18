//! HTTP/HTTPS media source for streaming from remote URLs.
//!
//! Provides [`HttpSource`] which implements [`MediaSource`](crate::source::MediaSource)
//! for reading media data over HTTP with byte-range request support.

#![allow(dead_code)]

use std::time::Duration;

/// Configuration for [`HttpSource`].
#[derive(Debug, Clone)]
pub struct HttpSourceConfig {
    /// Connection timeout.
    pub timeout: Duration,
    /// Number of retry attempts on transient errors.
    pub max_retries: u32,
    /// Delay between retries.
    pub retry_delay: Duration,
    /// User-Agent header value.
    pub user_agent: String,
    /// Whether to use byte-range requests for seeking.
    pub enable_range_requests: bool,
    /// Buffer size for chunked reading.
    pub buffer_size: usize,
}

impl Default for HttpSourceConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            user_agent: "OxiMedia/0.1".to_string(),
            enable_range_requests: true,
            buffer_size: 64 * 1024,
        }
    }
}

impl HttpSourceConfig {
    /// Create a new config with the given timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the number of retry attempts.
    #[must_use]
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set the buffer size for chunked reading.
    #[must_use]
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }
}

/// Represents a byte range for HTTP range requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// Start offset (inclusive).
    pub start: u64,
    /// End offset (inclusive), or `None` for open-ended.
    pub end: Option<u64>,
}

impl ByteRange {
    /// Create a range from `start` to `end` (inclusive).
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        Self {
            start,
            end: Some(end),
        }
    }

    /// Create an open-ended range starting at `start`.
    #[must_use]
    pub fn from_offset(start: u64) -> Self {
        Self { start, end: None }
    }

    /// Format as an HTTP Range header value.
    #[must_use]
    pub fn to_header_value(&self) -> String {
        match self.end {
            Some(end) => format!("bytes={}-{}", self.start, end),
            None => format!("bytes={}-", self.start),
        }
    }

    /// Compute the length of the range, if both bounds are known.
    #[must_use]
    pub fn length(&self) -> Option<u64> {
        self.end.map(|e| e - self.start + 1)
    }
}

impl std::fmt::Display for ByteRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_header_value())
    }
}

/// HTTP response metadata after a request.
#[derive(Debug, Clone)]
pub struct HttpResponseInfo {
    /// HTTP status code.
    pub status: u16,
    /// Content-Length, if provided.
    pub content_length: Option<u64>,
    /// Whether the server supports byte-range requests.
    pub accepts_ranges: bool,
    /// Content-Type header value.
    pub content_type: Option<String>,
    /// ETag header value (for caching / conditional requests).
    pub etag: Option<String>,
}

impl HttpResponseInfo {
    /// Returns `true` if the status code indicates success (2xx).
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Returns `true` if the response is a partial content response (206).
    #[must_use]
    pub fn is_partial_content(&self) -> bool {
        self.status == 206
    }
}

/// State of an HTTP media source connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpSourceState {
    /// Not yet connected.
    Disconnected,
    /// Connected and ready for reading.
    Connected,
    /// Encountered an error.
    Error,
    /// Reached end of content.
    EndOfStream,
}

/// An HTTP/HTTPS media source for streaming remote media.
///
/// Supports byte-range requests for pseudo-seeking, configurable
/// timeouts and retries, and tracks connection state.
///
/// # Example
///
/// ```
/// use oximedia_io::http_source::{HttpSource, HttpSourceConfig};
///
/// let config = HttpSourceConfig::default()
///     .with_timeout(std::time::Duration::from_secs(10))
///     .with_retries(5);
/// let source = HttpSource::new("https://example.com/video.mp4", config);
/// assert_eq!(source.url(), "https://example.com/video.mp4");
/// ```
#[derive(Debug)]
pub struct HttpSource {
    url: String,
    config: HttpSourceConfig,
    state: HttpSourceState,
    position: u64,
    content_length: Option<u64>,
    accepts_ranges: bool,
    buffer: Vec<u8>,
    buffer_pos: usize,
    buffer_len: usize,
    retry_count: u32,
}

impl HttpSource {
    /// Create a new `HttpSource` pointing to the given URL.
    #[must_use]
    pub fn new(url: &str, config: HttpSourceConfig) -> Self {
        let buf_size = config.buffer_size;
        Self {
            url: url.to_string(),
            config,
            state: HttpSourceState::Disconnected,
            position: 0,
            content_length: None,
            accepts_ranges: false,
            buffer: vec![0u8; buf_size],
            buffer_pos: 0,
            buffer_len: 0,
            retry_count: 0,
        }
    }

    /// Create an `HttpSource` with default configuration.
    #[must_use]
    pub fn with_defaults(url: &str) -> Self {
        Self::new(url, HttpSourceConfig::default())
    }

    /// Return the URL this source points to.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Return the current connection state.
    #[must_use]
    pub fn state(&self) -> HttpSourceState {
        self.state
    }

    /// Return the current read position.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Return the content length, if known.
    #[must_use]
    pub fn content_length(&self) -> Option<u64> {
        self.content_length
    }

    /// Return whether the server supports byte-range requests.
    #[must_use]
    pub fn accepts_ranges(&self) -> bool {
        self.accepts_ranges
    }

    /// Return the configuration.
    #[must_use]
    pub fn config(&self) -> &HttpSourceConfig {
        &self.config
    }

    /// Return whether the source is seekable (requires range request support).
    #[must_use]
    pub fn is_seekable(&self) -> bool {
        self.accepts_ranges && self.config.enable_range_requests
    }

    /// Simulate connecting to the URL and receiving response metadata.
    ///
    /// In a real implementation this would issue an HTTP HEAD request.
    pub fn set_response_info(&mut self, info: HttpResponseInfo) {
        self.content_length = info.content_length;
        self.accepts_ranges = info.accepts_ranges;
        if info.is_success() {
            self.state = HttpSourceState::Connected;
        } else {
            self.state = HttpSourceState::Error;
        }
    }

    /// Build a `ByteRange` header for reading `len` bytes from the current position.
    #[must_use]
    pub fn range_for_read(&self, len: u64) -> ByteRange {
        ByteRange::new(self.position, self.position + len - 1)
    }

    /// Seek to an absolute position (only if range requests are supported).
    ///
    /// # Errors
    ///
    /// Returns an error string if range requests are not available.
    pub fn seek_to(&mut self, pos: u64) -> Result<u64, String> {
        if !self.is_seekable() {
            return Err("Server does not support byte-range requests".to_string());
        }
        self.position = pos;
        // Invalidate internal buffer on seek
        self.buffer_pos = 0;
        self.buffer_len = 0;
        Ok(self.position)
    }

    /// Feed data into the internal buffer (simulates receiving HTTP response body).
    pub fn feed_data(&mut self, data: &[u8]) {
        let space = self.buffer.len() - self.buffer_len;
        let to_copy = data.len().min(space);
        self.buffer[self.buffer_len..self.buffer_len + to_copy].copy_from_slice(&data[..to_copy]);
        self.buffer_len += to_copy;
    }

    /// Read from the internal buffer.
    ///
    /// Returns the number of bytes read (0 means the buffer is exhausted).
    pub fn read_buffered(&mut self, buf: &mut [u8]) -> usize {
        let available = self.buffer_len - self.buffer_pos;
        if available == 0 {
            return 0;
        }
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&self.buffer[self.buffer_pos..self.buffer_pos + to_read]);
        self.buffer_pos += to_read;
        self.position += to_read as u64;

        // Reset buffer when fully consumed
        if self.buffer_pos == self.buffer_len {
            self.buffer_pos = 0;
            self.buffer_len = 0;
        }
        to_read
    }

    /// Check if a retry is appropriate and increment the counter.
    ///
    /// Returns `true` if a retry should be attempted.
    pub fn should_retry(&mut self) -> bool {
        if self.retry_count < self.config.max_retries {
            self.retry_count += 1;
            true
        } else {
            false
        }
    }

    /// Reset the retry counter (e.g. after a successful request).
    pub fn reset_retries(&mut self) {
        self.retry_count = 0;
    }

    /// Return the number of retries consumed so far.
    #[must_use]
    pub fn retry_count(&self) -> u32 {
        self.retry_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_source_config_defaults() {
        let cfg = HttpSourceConfig::default();
        assert_eq!(cfg.timeout, Duration::from_secs(30));
        assert_eq!(cfg.max_retries, 3);
        assert!(cfg.enable_range_requests);
        assert_eq!(cfg.buffer_size, 64 * 1024);
    }

    #[test]
    fn test_http_source_config_builder() {
        let cfg = HttpSourceConfig::default()
            .with_timeout(Duration::from_secs(60))
            .with_retries(5)
            .with_buffer_size(128 * 1024);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
        assert_eq!(cfg.max_retries, 5);
        assert_eq!(cfg.buffer_size, 128 * 1024);
    }

    #[test]
    fn test_byte_range_closed() {
        let r = ByteRange::new(100, 199);
        assert_eq!(r.to_header_value(), "bytes=100-199");
        assert_eq!(r.length(), Some(100));
        assert_eq!(r.to_string(), "bytes=100-199");
    }

    #[test]
    fn test_byte_range_open() {
        let r = ByteRange::from_offset(500);
        assert_eq!(r.to_header_value(), "bytes=500-");
        assert_eq!(r.length(), None);
    }

    #[test]
    fn test_http_response_info_success() {
        let info = HttpResponseInfo {
            status: 200,
            content_length: Some(1024),
            accepts_ranges: true,
            content_type: Some("video/mp4".to_string()),
            etag: None,
        };
        assert!(info.is_success());
        assert!(!info.is_partial_content());
    }

    #[test]
    fn test_http_response_info_partial() {
        let info = HttpResponseInfo {
            status: 206,
            content_length: Some(512),
            accepts_ranges: true,
            content_type: None,
            etag: None,
        };
        assert!(info.is_success());
        assert!(info.is_partial_content());
    }

    #[test]
    fn test_http_source_creation() {
        let src = HttpSource::with_defaults("https://example.com/video.mp4");
        assert_eq!(src.url(), "https://example.com/video.mp4");
        assert_eq!(src.state(), HttpSourceState::Disconnected);
        assert_eq!(src.position(), 0);
        assert_eq!(src.content_length(), None);
        assert!(!src.accepts_ranges());
    }

    #[test]
    fn test_http_source_set_response_info() {
        let mut src = HttpSource::with_defaults("https://example.com/media.webm");
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: Some(4096),
            accepts_ranges: true,
            content_type: Some("video/webm".to_string()),
            etag: Some("abc123".to_string()),
        });
        assert_eq!(src.state(), HttpSourceState::Connected);
        assert_eq!(src.content_length(), Some(4096));
        assert!(src.accepts_ranges());
        assert!(src.is_seekable());
    }

    #[test]
    fn test_http_source_error_response() {
        let mut src = HttpSource::with_defaults("https://example.com/404");
        src.set_response_info(HttpResponseInfo {
            status: 404,
            content_length: None,
            accepts_ranges: false,
            content_type: None,
            etag: None,
        });
        assert_eq!(src.state(), HttpSourceState::Error);
    }

    #[test]
    fn test_http_source_seek_with_range_support() {
        let mut src = HttpSource::with_defaults("https://example.com/video.mp4");
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: Some(10000),
            accepts_ranges: true,
            content_type: None,
            etag: None,
        });
        let pos = src.seek_to(500);
        assert_eq!(pos, Ok(500));
        assert_eq!(src.position(), 500);
    }

    #[test]
    fn test_http_source_seek_without_range_support() {
        let mut src = HttpSource::with_defaults("https://example.com/live");
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: None,
            accepts_ranges: false,
            content_type: None,
            etag: None,
        });
        assert!(src.seek_to(100).is_err());
    }

    #[test]
    fn test_http_source_range_for_read() {
        let mut src = HttpSource::with_defaults("https://example.com/v.mp4");
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: Some(10000),
            accepts_ranges: true,
            content_type: None,
            etag: None,
        });
        let _ = src.seek_to(100);
        let range = src.range_for_read(50);
        assert_eq!(range.start, 100);
        assert_eq!(range.end, Some(149));
    }

    #[test]
    fn test_http_source_feed_and_read() {
        let mut src = HttpSource::with_defaults("https://example.com/v.mp4");
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: Some(100),
            accepts_ranges: false,
            content_type: None,
            etag: None,
        });

        src.feed_data(&[1, 2, 3, 4, 5]);
        let mut buf = [0u8; 3];
        let n = src.read_buffered(&mut buf);
        assert_eq!(n, 3);
        assert_eq!(&buf, &[1, 2, 3]);
        assert_eq!(src.position(), 3);

        let n = src.read_buffered(&mut buf);
        assert_eq!(n, 2);
        assert_eq!(&buf[..2], &[4, 5]);
        assert_eq!(src.position(), 5);

        // Buffer exhausted
        let n = src.read_buffered(&mut buf);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_http_source_retry_logic() {
        let cfg = HttpSourceConfig::default().with_retries(2);
        let mut src = HttpSource::new("https://example.com/v.mp4", cfg);
        assert_eq!(src.retry_count(), 0);
        assert!(src.should_retry());
        assert_eq!(src.retry_count(), 1);
        assert!(src.should_retry());
        assert_eq!(src.retry_count(), 2);
        assert!(!src.should_retry()); // exhausted
        src.reset_retries();
        assert_eq!(src.retry_count(), 0);
        assert!(src.should_retry());
    }

    #[test]
    fn test_http_source_not_seekable_without_range_config() {
        let mut cfg = HttpSourceConfig::default();
        cfg.enable_range_requests = false;
        let mut src = HttpSource::new("https://example.com/v.mp4", cfg);
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: Some(1000),
            accepts_ranges: true,
            content_type: None,
            etag: None,
        });
        assert!(!src.is_seekable());
    }

    #[test]
    fn test_http_source_seek_invalidates_buffer() {
        let mut src = HttpSource::with_defaults("https://example.com/v.mp4");
        src.set_response_info(HttpResponseInfo {
            status: 200,
            content_length: Some(10000),
            accepts_ranges: true,
            content_type: None,
            etag: None,
        });
        src.feed_data(&[10, 20, 30]);
        let _ = src.seek_to(500);
        let mut buf = [0u8; 3];
        let n = src.read_buffered(&mut buf);
        assert_eq!(n, 0); // buffer was invalidated
    }
}
