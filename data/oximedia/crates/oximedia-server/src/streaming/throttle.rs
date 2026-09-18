//! Bandwidth throttling for streaming.

use bytes::Bytes;
use futures::Stream;
use pin_project::pin_project;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::time::sleep;

/// Bandwidth throttler for rate limiting streams.
#[pin_project]
pub struct BandwidthThrottler<S> {
    #[pin]
    inner: S,
    max_bytes_per_sec: usize,
    bytes_sent: usize,
    window_start: Instant,
    window_duration: Duration,
}

impl<S> BandwidthThrottler<S> {
    /// Creates a new bandwidth throttler.
    #[must_use]
    pub fn new(inner: S, max_bytes_per_sec: usize) -> Self {
        Self {
            inner,
            max_bytes_per_sec,
            bytes_sent: 0,
            window_start: Instant::now(),
            window_duration: Duration::from_secs(1),
        }
    }

    /// Checks if throttling is needed and sleeps if necessary.
    #[allow(dead_code)]
    async fn throttle(&mut self, bytes: usize) {
        self.bytes_sent += bytes;

        let elapsed = self.window_start.elapsed();
        if elapsed >= self.window_duration {
            // Reset window
            self.bytes_sent = bytes;
            self.window_start = Instant::now();
        } else if self.bytes_sent > self.max_bytes_per_sec {
            // Throttle
            let sleep_duration = self.window_duration.saturating_sub(elapsed);
            sleep(sleep_duration).await;
            self.bytes_sent = 0;
            self.window_start = Instant::now();
        }
    }
}

impl<S> Stream for BandwidthThrottler<S>
where
    S: Stream<Item = Result<Bytes, std::io::Error>>,
{
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let poll = this.inner.poll_next(cx);

        match poll {
            Poll::Ready(Some(Ok(bytes))) => {
                let len = bytes.len();
                // Note: In a real implementation, we would need to handle async throttling properly
                // For now, we just track the bytes
                *this.bytes_sent += len;
                Poll::Ready(Some(Ok(bytes)))
            }
            other => other,
        }
    }
}
