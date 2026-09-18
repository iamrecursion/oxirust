//! Retrying I/O source wrapper with exponential backoff.
//!
//! [`RetryingSource`] wraps any reader and retries failed reads using
//! configurable exponential backoff with jitter. This is useful when
//! reading from unreliable sources (network streams, flaky storage).

#![allow(dead_code)]

use std::io::{self, Read};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the retry policy.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, just the initial attempt).
    pub max_retries: u32,
    /// Base delay in milliseconds before the first retry.
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds (cap for exponential growth).
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff (in thousandths, e.g. 2000 = 2.0x).
    pub backoff_multiplier_per_mille: u32,
    /// Whether to add jitter to the delay (±25% of computed delay).
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            backoff_multiplier_per_mille: 2000, // 2.0x
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a config with no retries (fail immediately).
    #[must_use]
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Create a config with a fixed number of retries and default backoff.
    #[must_use]
    pub fn with_max_retries(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Compute the delay for the given attempt number (0-based).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let multiplier = self.backoff_multiplier_per_mille as f64 / 1000.0;
        let delay = self.base_delay_ms as f64 * multiplier.powi(attempt as i32);
        let capped = delay.min(self.max_delay_ms as f64);

        if self.jitter {
            // Simple deterministic "jitter" based on attempt number.
            // In a real system you'd use a PRNG; here we use a hash-like
            // modulation to keep the code pure and deterministic for testing.
            let jitter_factor = match attempt % 4 {
                0 => 0.85,
                1 => 1.15,
                2 => 0.95,
                _ => 1.05,
            };
            (capped * jitter_factor) as u64
        } else {
            capped as u64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retry statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics collected by the retrying source.
#[derive(Debug, Clone, Default)]
pub struct RetryStats {
    /// Total number of read attempts (including initial attempts).
    pub total_attempts: u64,
    /// Number of reads that succeeded on the first try.
    pub first_try_successes: u64,
    /// Number of reads that succeeded after retrying.
    pub retry_successes: u64,
    /// Number of reads that failed after exhausting all retries.
    pub final_failures: u64,
    /// Total number of retries performed across all reads.
    pub total_retries: u64,
    /// Total bytes successfully read.
    pub total_bytes_read: u64,
    /// Total simulated delay in milliseconds (sum of all backoff waits).
    pub total_delay_ms: u64,
}

impl RetryStats {
    /// Success rate as a fraction in `0.0..=1.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let total_ops = self.first_try_successes + self.retry_successes + self.final_failures;
        if total_ops == 0 {
            1.0
        } else {
            (self.first_try_successes + self.retry_successes) as f64 / total_ops as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RetryingSource
// ─────────────────────────────────────────────────────────────────────────────

/// A function that is called with the computed delay in milliseconds.
/// In production this would sleep; in tests it can be a no-op.
pub type SleepFn = Box<dyn FnMut(u64)>;

/// A wrapper around any `Read` implementation that retries failed reads
/// with exponential backoff.
///
/// Non-retryable errors (e.g. `ErrorKind::InvalidInput`) are returned
/// immediately. Transient errors (`WouldBlock`, `Interrupted`, `TimedOut`,
/// `ConnectionReset`, `BrokenPipe`, `UnexpectedEof`) are retried.
pub struct RetryingSource<R> {
    inner: R,
    config: RetryConfig,
    stats: RetryStats,
    sleep_fn: SleepFn,
}

impl<R: Read> RetryingSource<R> {
    /// Create a new retrying source with default configuration.
    ///
    /// The sleep function defaults to a no-op (suitable for sync test usage;
    /// in production, pass a real sleep function).
    pub fn new(inner: R) -> Self {
        Self::with_config(inner, RetryConfig::default())
    }

    /// Create a new retrying source with custom configuration.
    pub fn with_config(inner: R, config: RetryConfig) -> Self {
        Self {
            inner,
            config,
            stats: RetryStats::default(),
            sleep_fn: Box::new(|_| {}),
        }
    }

    /// Set a custom sleep function (called with delay in ms before each retry).
    pub fn with_sleep_fn(mut self, f: SleepFn) -> Self {
        self.sleep_fn = f;
        self
    }

    /// Return a reference to the accumulated statistics.
    #[must_use]
    pub fn stats(&self) -> &RetryStats {
        &self.stats
    }

    /// Return a reference to the inner reader.
    #[must_use]
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Consume this wrapper and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Determine whether an I/O error is transient and worth retrying.
    fn is_retryable(err: &io::Error) -> bool {
        matches!(
            err.kind(),
            io::ErrorKind::WouldBlock
                | io::ErrorKind::Interrupted
                | io::ErrorKind::TimedOut
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::BrokenPipe
                | io::ErrorKind::UnexpectedEof
        )
    }
}

impl<R: Read> Read for RetryingSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut last_err = None;

        for attempt in 0..=self.config.max_retries {
            self.stats.total_attempts += 1;

            match self.inner.read(buf) {
                Ok(n) => {
                    self.stats.total_bytes_read += n as u64;
                    if attempt == 0 {
                        self.stats.first_try_successes += 1;
                    } else {
                        self.stats.retry_successes += 1;
                    }
                    return Ok(n);
                }
                Err(e) => {
                    if !Self::is_retryable(&e) || attempt == self.config.max_retries {
                        last_err = Some(e);
                        break;
                    }
                    // Transient error — apply backoff
                    let delay = self.config.delay_for_attempt(attempt);
                    self.stats.total_delay_ms += delay;
                    self.stats.total_retries += 1;
                    (self.sleep_fn)(delay);
                    last_err = Some(e);
                }
            }
        }

        self.stats.final_failures += 1;
        Err(last_err.unwrap_or_else(|| io::Error::new(io::ErrorKind::Other, "retry exhausted")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FailingReader — test helper
// ─────────────────────────────────────────────────────────────────────────────

/// A reader that fails a configurable number of times before succeeding.
/// Useful for testing the retry logic.
pub struct FailingReader {
    /// Data to return on success.
    data: Vec<u8>,
    /// Read position.
    pos: usize,
    /// Number of failures remaining before reads succeed.
    failures_remaining: u32,
    /// The error kind to produce on failure.
    error_kind: io::ErrorKind,
}

impl FailingReader {
    /// Create a reader that fails `failures` times with the given error kind,
    /// then succeeds with `data`.
    #[must_use]
    pub fn new(data: Vec<u8>, failures: u32, error_kind: io::ErrorKind) -> Self {
        Self {
            data,
            pos: 0,
            failures_remaining: failures,
            error_kind,
        }
    }
}

impl Read for FailingReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.failures_remaining > 0 {
            self.failures_remaining -= 1;
            return Err(io::Error::new(self.error_kind, "simulated failure"));
        }
        let available = self.data.len().saturating_sub(self.pos);
        let to_read = buf.len().min(available);
        if to_read == 0 {
            return Ok(0);
        }
        buf[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
        self.pos += to_read;
        Ok(to_read)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.base_delay_ms, 100);
        assert_eq!(cfg.max_delay_ms, 10_000);
    }

    #[test]
    fn test_retry_config_no_retry() {
        let cfg = RetryConfig::no_retry();
        assert_eq!(cfg.max_retries, 0);
    }

    #[test]
    fn test_delay_exponential_growth() {
        let cfg = RetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 100_000,
            backoff_multiplier_per_mille: 2000,
            jitter: false,
            ..Default::default()
        };
        assert_eq!(cfg.delay_for_attempt(0), 100); // 100 * 2^0
        assert_eq!(cfg.delay_for_attempt(1), 200); // 100 * 2^1
        assert_eq!(cfg.delay_for_attempt(2), 400); // 100 * 2^2
        assert_eq!(cfg.delay_for_attempt(3), 800); // 100 * 2^3
    }

    #[test]
    fn test_delay_capped_at_max() {
        let cfg = RetryConfig {
            base_delay_ms: 1000,
            max_delay_ms: 5000,
            backoff_multiplier_per_mille: 3000, // 3x
            jitter: false,
            ..Default::default()
        };
        // attempt 0: 1000, attempt 1: 3000, attempt 2: 9000 capped to 5000
        assert_eq!(cfg.delay_for_attempt(2), 5000);
    }

    #[test]
    fn test_delay_with_jitter_varies() {
        let cfg = RetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 100_000,
            backoff_multiplier_per_mille: 2000,
            jitter: true,
            ..Default::default()
        };
        let d0 = cfg.delay_for_attempt(0);
        let d1 = cfg.delay_for_attempt(1);
        // With jitter the values should differ from the pure exponential
        assert_ne!(d0, 100);
        assert_ne!(d1, 200);
        // But should be in the ballpark (within ±25%)
        assert!(d0 >= 75 && d0 <= 125);
        assert!(d1 >= 150 && d1 <= 250);
    }

    #[test]
    fn test_success_on_first_try() {
        let reader = FailingReader::new(b"hello".to_vec(), 0, io::ErrorKind::TimedOut);
        let mut source = RetryingSource::new(reader);
        let mut buf = [0u8; 10];
        let n = source.read(&mut buf).expect("should succeed");
        assert_eq!(n, 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(source.stats().first_try_successes, 1);
        assert_eq!(source.stats().total_retries, 0);
    }

    #[test]
    fn test_success_after_retries() {
        let reader = FailingReader::new(b"world".to_vec(), 2, io::ErrorKind::TimedOut);
        let mut source = RetryingSource::with_config(reader, RetryConfig::with_max_retries(5));
        let mut buf = [0u8; 10];
        let n = source.read(&mut buf).expect("should succeed after retries");
        assert_eq!(n, 5);
        assert_eq!(&buf[..5], b"world");
        assert_eq!(source.stats().retry_successes, 1);
        assert_eq!(source.stats().total_retries, 2);
    }

    #[test]
    fn test_failure_after_max_retries() {
        let reader = FailingReader::new(b"data".to_vec(), 10, io::ErrorKind::TimedOut);
        let mut source = RetryingSource::with_config(reader, RetryConfig::with_max_retries(3));
        let mut buf = [0u8; 10];
        let result = source.read(&mut buf);
        assert!(result.is_err());
        assert_eq!(source.stats().final_failures, 1);
        assert_eq!(source.stats().total_retries, 3);
    }

    #[test]
    fn test_non_retryable_error_fails_immediately() {
        let reader = FailingReader::new(b"data".to_vec(), 5, io::ErrorKind::InvalidInput);
        let mut source = RetryingSource::with_config(reader, RetryConfig::with_max_retries(10));
        let mut buf = [0u8; 10];
        let result = source.read(&mut buf);
        assert!(result.is_err());
        assert_eq!(source.stats().total_retries, 0); // no retries for non-retryable
        assert_eq!(source.stats().final_failures, 1);
    }

    #[test]
    fn test_zero_retries_config() {
        let reader = FailingReader::new(b"data".to_vec(), 1, io::ErrorKind::TimedOut);
        let mut source = RetryingSource::with_config(reader, RetryConfig::no_retry());
        let mut buf = [0u8; 10];
        let result = source.read(&mut buf);
        assert!(result.is_err());
        assert_eq!(source.stats().total_retries, 0);
    }

    #[test]
    fn test_stats_success_rate() {
        let mut stats = RetryStats::default();
        stats.first_try_successes = 8;
        stats.retry_successes = 1;
        stats.final_failures = 1;
        let rate = stats.success_rate();
        assert!((rate - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_success_rate_empty() {
        let stats = RetryStats::default();
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[test]
    fn test_sleep_fn_called() {
        let reader = FailingReader::new(b"ok".to_vec(), 2, io::ErrorKind::TimedOut);
        let sleep_call_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let counter = sleep_call_count.clone();
        let mut source = RetryingSource::with_config(
            reader,
            RetryConfig {
                max_retries: 5,
                jitter: false,
                ..Default::default()
            },
        )
        .with_sleep_fn(Box::new(move |_ms| {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }));
        let mut buf = [0u8; 10];
        let _ = source.read(&mut buf).expect("should succeed");
        assert!(source.stats().total_delay_ms > 0);
        assert_eq!(
            sleep_call_count.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn test_multiple_reads() {
        let reader = FailingReader::new(b"hello world".to_vec(), 0, io::ErrorKind::TimedOut);
        let mut source = RetryingSource::new(reader);

        let mut buf = [0u8; 5];
        let n1 = source.read(&mut buf).expect("first read");
        assert_eq!(n1, 5);
        assert_eq!(&buf[..5], b"hello");

        let n2 = source.read(&mut buf).expect("second read");
        assert_eq!(n2, 5);
        assert_eq!(&buf[..5], b" worl");

        assert_eq!(source.stats().total_bytes_read, 10);
    }

    #[test]
    fn test_into_inner() {
        let reader = FailingReader::new(b"test".to_vec(), 0, io::ErrorKind::TimedOut);
        let source = RetryingSource::new(reader);
        let inner = source.into_inner();
        assert_eq!(inner.data, b"test");
    }

    #[test]
    fn test_inner_ref() {
        let reader = FailingReader::new(b"ref".to_vec(), 0, io::ErrorKind::TimedOut);
        let source = RetryingSource::new(reader);
        assert_eq!(source.inner().data, b"ref");
    }

    #[test]
    fn test_interrupted_is_retryable() {
        let reader = FailingReader::new(b"x".to_vec(), 1, io::ErrorKind::Interrupted);
        let mut source = RetryingSource::with_config(reader, RetryConfig::with_max_retries(2));
        let mut buf = [0u8; 1];
        let n = source.read(&mut buf).expect("should succeed");
        assert_eq!(n, 1);
        assert_eq!(source.stats().retry_successes, 1);
    }

    #[test]
    fn test_connection_reset_is_retryable() {
        let reader = FailingReader::new(b"y".to_vec(), 1, io::ErrorKind::ConnectionReset);
        let mut source = RetryingSource::with_config(reader, RetryConfig::with_max_retries(2));
        let mut buf = [0u8; 1];
        let n = source.read(&mut buf).expect("should succeed");
        assert_eq!(n, 1);
    }

    #[test]
    fn test_broken_pipe_is_retryable() {
        let reader = FailingReader::new(b"z".to_vec(), 1, io::ErrorKind::BrokenPipe);
        let mut source = RetryingSource::with_config(reader, RetryConfig::with_max_retries(3));
        let mut buf = [0u8; 1];
        let n = source.read(&mut buf).expect("should succeed");
        assert_eq!(n, 1);
    }

    #[test]
    fn test_eof_at_end() {
        let reader = FailingReader::new(Vec::new(), 0, io::ErrorKind::TimedOut);
        let mut source = RetryingSource::new(reader);
        let mut buf = [0u8; 10];
        let n = source.read(&mut buf).expect("should return 0 at EOF");
        assert_eq!(n, 0);
    }
}
