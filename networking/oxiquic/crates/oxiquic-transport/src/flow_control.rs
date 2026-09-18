//! Connection- and stream-level flow control (RFC 9000 Section 4).
//!
//! Flow control bounds how much stream data may be in flight before the
//! receiver has granted credit. This module tracks both directions:
//!
//! * **Send side** ([`SendFlowControl`], [`StreamSendFlow`]): the maximum byte
//!   offset the peer has authorised via `MAX_DATA` / `MAX_STREAM_DATA`, and how
//!   much we have already sent, so the connection can cap each `STREAM` frame
//!   and emit `DATA_BLOCKED` / `STREAM_DATA_BLOCKED` when it runs out of credit.
//! * **Receive side** ([`RecvFlowControl`], [`StreamRecvFlow`]): the limit we
//!   have advertised and how much the peer has consumed, so the connection can
//!   advance the limit (sending fresh `MAX_DATA` / `MAX_STREAM_DATA`) as the
//!   application reads.

/// Connection-level send-side flow control (RFC 9000 Section 4.1).
///
/// Flow control accounts for the *aggregate highest offset* of stream data
/// sent, not cumulative bytes; retransmitting already-sent data therefore does
/// not consume fresh credit (RFC 9000 Section 4.1).
#[derive(Debug, Clone, Default)]
pub struct SendFlowControl {
    /// Maximum aggregate byte offset the peer has authorised (`MAX_DATA`).
    max_data: u64,
    /// Aggregate highest offset we have sent across all streams.
    data_sent: u64,
    /// The limit at which we last reported being blocked (avoids re-sending the
    /// same `DATA_BLOCKED`).
    blocked_at: Option<u64>,
}

impl SendFlowControl {
    /// Create a send-side controller authorised up to `initial_max_data`.
    #[must_use]
    pub fn new(initial_max_data: u64) -> Self {
        Self {
            max_data: initial_max_data,
            data_sent: 0,
            blocked_at: None,
        }
    }

    /// Raise the connection limit on receiving a `MAX_DATA` frame; a smaller
    /// value is ignored (limits are monotonic, RFC 9000 Section 4.1).
    pub fn on_max_data(&mut self, max: u64) {
        if max > self.max_data {
            self.max_data = max;
            self.blocked_at = None;
        }
    }

    /// The remaining connection-level credit in bytes.
    #[must_use]
    pub fn available(&self) -> u64 {
        self.max_data.saturating_sub(self.data_sent)
    }

    /// Account for `n` *new* stream bytes being sent (raising the aggregate
    /// highest offset). Retransmissions pass `0` since they do not advance it.
    pub fn on_data_sent(&mut self, n: u64) {
        self.data_sent += n;
    }

    /// Whether the connection limit currently blocks sending more data.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.data_sent >= self.max_data
    }

    /// If newly blocked at the current limit, return that limit for a
    /// `DATA_BLOCKED` frame (once per limit value).
    pub fn take_blocked(&mut self) -> Option<u64> {
        if self.is_blocked() && self.blocked_at != Some(self.max_data) {
            self.blocked_at = Some(self.max_data);
            Some(self.max_data)
        } else {
            None
        }
    }

    /// The current connection limit (diagnostics/tests).
    #[cfg(test)]
    #[must_use]
    pub fn max_data(&self) -> u64 {
        self.max_data
    }
}

/// Per-stream send-side flow control (RFC 9000 Section 4.1).
#[derive(Debug, Clone, Default)]
pub struct StreamSendFlow {
    /// Maximum byte offset the peer has authorised on this stream
    /// (`MAX_STREAM_DATA`).
    max_stream_data: u64,
    /// Bytes already sent on this stream (its current offset).
    data_sent: u64,
    /// The limit at which we last reported being blocked.
    blocked_at: Option<u64>,
}

impl StreamSendFlow {
    /// Create a stream send controller authorised up to `initial`.
    #[must_use]
    pub fn new(initial: u64) -> Self {
        Self {
            max_stream_data: initial,
            data_sent: 0,
            blocked_at: None,
        }
    }

    /// Raise the stream limit on receiving a `MAX_STREAM_DATA` frame.
    pub fn on_max_stream_data(&mut self, max: u64) {
        if max > self.max_stream_data {
            self.max_stream_data = max;
            self.blocked_at = None;
        }
    }

    /// Remaining stream-level credit in bytes.
    #[must_use]
    pub fn available(&self) -> u64 {
        self.max_stream_data.saturating_sub(self.data_sent)
    }

    /// Account for `n` *new* bytes sent on the stream (advancing its offset);
    /// production code uses [`StreamSendFlow::record_high_offset`].
    #[cfg(test)]
    pub fn on_data_sent(&mut self, n: u64) {
        self.data_sent += n;
    }

    /// Record that data up to `high_offset` has now been sent on this stream,
    /// returning the number of *newly*-consumed bytes (zero for a pure
    /// retransmission of already-sent offsets). This is the value to charge
    /// against the connection-level limit (RFC 9000 Section 4.1).
    pub fn record_high_offset(&mut self, high_offset: u64) -> u64 {
        if high_offset > self.data_sent {
            let delta = high_offset - self.data_sent;
            self.data_sent = high_offset;
            delta
        } else {
            0
        }
    }

    /// Whether the stream limit currently blocks sending.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.data_sent >= self.max_stream_data
    }

    /// If newly blocked at the current limit, return it for a
    /// `STREAM_DATA_BLOCKED` frame (once per limit value).
    pub fn take_blocked(&mut self) -> Option<u64> {
        if self.is_blocked() && self.blocked_at != Some(self.max_stream_data) {
            self.blocked_at = Some(self.max_stream_data);
            Some(self.max_stream_data)
        } else {
            None
        }
    }
}

/// Connection-level receive-side flow control (RFC 9000 Section 4.1): tracks how
/// much credit we have advertised and how much the application has consumed so
/// the connection can advance the limit.
#[derive(Debug, Clone, Default)]
pub struct RecvFlowControl {
    /// The maximum aggregate offset we have advertised to the peer.
    max_data: u64,
    /// The window size we re-advertise above the consumed offset.
    window: u64,
    /// Total bytes the application has consumed across all streams.
    consumed: u64,
    /// Aggregate highest stream offset the peer has actually sent (the sum of
    /// every stream's highest received offset), used to enforce the advertised
    /// connection-level limit (RFC 9000 Section 4.1).
    received: u64,
}

impl RecvFlowControl {
    /// Create a receive-side controller advertising `window` initially.
    #[must_use]
    pub fn new(window: u64) -> Self {
        Self {
            max_data: window,
            window,
            consumed: 0,
            received: 0,
        }
    }

    /// Account for `n` bytes consumed by the application.
    pub fn on_data_consumed(&mut self, n: u64) {
        self.consumed += n;
    }

    /// Charge `delta` newly-received stream bytes (the increase in a stream's
    /// highest received offset) against the advertised connection-level limit.
    ///
    /// Returns `false` when the peer has exceeded `MAX_DATA`, in which case the
    /// caller MUST close the connection with `FLOW_CONTROL_ERROR` (RFC 9000
    /// Section 4.1). A `delta` of `0` (pure retransmission) never violates.
    #[must_use]
    pub fn on_data_received(&mut self, delta: u64) -> bool {
        self.received = self.received.saturating_add(delta);
        self.received <= self.max_data
    }

    /// The current advertised limit (diagnostics/tests).
    #[cfg(test)]
    #[must_use]
    pub fn max_data(&self) -> u64 {
        self.max_data
    }

    /// If the consumed offset has advanced far enough that the advertised limit
    /// should grow (more than half the window has been used since the last
    /// advertisement), return the new `MAX_DATA` limit to send. RFC 9000
    /// Section 4.1 recommends advancing the limit as data is consumed.
    pub fn maybe_update(&mut self) -> Option<u64> {
        let desired = self.consumed + self.window;
        // Advertise eagerly once half the window is consumed beyond the floor.
        if desired > self.max_data && self.max_data.saturating_sub(self.consumed) < self.window / 2
        {
            self.max_data = desired;
            Some(self.max_data)
        } else {
            None
        }
    }
}

/// Per-stream receive-side flow control (RFC 9000 Section 4.1).
#[derive(Debug, Clone, Default)]
pub struct StreamRecvFlow {
    max_stream_data: u64,
    window: u64,
    consumed: u64,
    /// Highest byte offset (`offset + len`) received on this stream, used to
    /// enforce the advertised stream-level limit and to charge the
    /// connection-level limit for newly-received bytes (RFC 9000 Section 4.1).
    received: u64,
}

impl StreamRecvFlow {
    /// Create a stream receive controller advertising `window` initially.
    #[must_use]
    pub fn new(window: u64) -> Self {
        Self {
            max_stream_data: window,
            window,
            consumed: 0,
            received: 0,
        }
    }

    /// Account for `n` bytes consumed on the stream by the application.
    pub fn on_data_consumed(&mut self, n: u64) {
        self.consumed += n;
    }

    /// Record a `STREAM` frame whose data ends at `end_offset` and enforce the
    /// advertised stream-level limit (RFC 9000 Section 4.1).
    ///
    /// Returns `Some(delta)` with the number of *newly*-received bytes (the rise
    /// in this stream's highest offset, `0` for a pure retransmission of already
    /// received offsets) when within the limit; that `delta` is what the caller
    /// charges against the connection-level limit. Returns `None` when the peer
    /// has sent past `MAX_STREAM_DATA`, in which case the caller MUST close the
    /// connection with `FLOW_CONTROL_ERROR`.
    #[must_use]
    pub fn on_stream_data(&mut self, end_offset: u64) -> Option<u64> {
        if end_offset > self.max_stream_data {
            return None;
        }
        let delta = end_offset.saturating_sub(self.received);
        if end_offset > self.received {
            self.received = end_offset;
        }
        Some(delta)
    }

    /// The highest byte offset received on this stream so far (RFC 9000
    /// Section 4.5). Used to reject a RESET_STREAM whose `final_size` is below
    /// data already received (`FINAL_SIZE_ERROR`).
    #[must_use]
    pub fn highest_received(&self) -> u64 {
        self.received
    }

    /// If the stream limit should grow, return the new `MAX_STREAM_DATA` value.
    pub fn maybe_update(&mut self) -> Option<u64> {
        let desired = self.consumed + self.window;
        if desired > self.max_stream_data
            && self.max_stream_data.saturating_sub(self.consumed) < self.window / 2
        {
            self.max_stream_data = desired;
            Some(self.max_stream_data)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_blocks_at_limit_then_unblocks() {
        let mut fc = SendFlowControl::new(100);
        assert_eq!(fc.available(), 100);
        fc.on_data_sent(100);
        assert!(fc.is_blocked());
        assert_eq!(fc.take_blocked(), Some(100));
        // Same limit does not re-emit.
        assert_eq!(fc.take_blocked(), None);
        // MAX_DATA raises the ceiling and unblocks.
        fc.on_max_data(250);
        assert!(!fc.is_blocked());
        assert_eq!(fc.available(), 150);
    }

    #[test]
    fn stream_send_flow_credit() {
        let mut fc = StreamSendFlow::new(50);
        fc.on_data_sent(30);
        assert_eq!(fc.available(), 20);
        fc.on_data_sent(20);
        assert!(fc.is_blocked());
        assert_eq!(fc.take_blocked(), Some(50));
        fc.on_max_stream_data(50); // no change
        assert!(fc.is_blocked());
        fc.on_max_stream_data(120);
        assert_eq!(fc.available(), 70);
    }

    #[test]
    fn recv_advertises_more_credit_as_consumed() {
        let mut fc = RecvFlowControl::new(100);
        assert_eq!(fc.max_data(), 100);
        // Consume 60 (> half the 100 window) -> advance to 60 + 100 = 160.
        fc.on_data_consumed(60);
        assert_eq!(fc.maybe_update(), Some(160));
        // No further update until more is consumed past the new half-window.
        assert_eq!(fc.maybe_update(), None);
    }

    #[test]
    fn stream_recv_enforces_limit() {
        let mut fc = StreamRecvFlow::new(100);
        // Within the limit: reports the newly-received delta.
        assert_eq!(fc.on_stream_data(40), Some(40));
        // A higher offset within the limit charges only the increment.
        assert_eq!(fc.on_stream_data(100), Some(60));
        // A retransmission of already-received offsets charges nothing.
        assert_eq!(fc.on_stream_data(80), Some(0));
        // One byte past the advertised limit is a violation.
        assert_eq!(fc.on_stream_data(101), None);
    }

    #[test]
    fn conn_recv_enforces_limit() {
        let mut fc = RecvFlowControl::new(100);
        assert!(fc.on_data_received(60));
        assert!(fc.on_data_received(40)); // exactly at the limit is fine
        assert!(!fc.on_data_received(1)); // one byte over -> violation
    }

    #[test]
    fn monotonic_max_data() {
        let mut fc = SendFlowControl::new(100);
        fc.on_max_data(50); // lower ignored
        assert_eq!(fc.max_data(), 100);
        fc.on_max_data(200);
        assert_eq!(fc.max_data(), 200);
    }
}
