//! Process-local zero-copy frame bypass registry.
//!
//! When a sender and receiver run in the **same OS process**, routing frames
//! through the full TCP encode → send → recv → decode pipeline wastes CPU and
//! memory.  `LocalBypassRegistry` short-circuits this by maintaining a
//! per-source [`tokio::sync::broadcast`] channel whose messages are
//! [`NdiFrame`] values.  Because [`bytes::Bytes`] is Arc-backed, cloning a
//! frame is O(1) — no pixel-data copy occurs.
//!
//! # Life-cycle
//!
//! 1. `NdiSender::new()` calls [`LocalBypassRegistry::register`] to obtain a
//!    `broadcast::Sender<NdiFrame>`.  A clone of that sender is kept inside
//!    `NdiSender`.
//! 2. When the sender broadcasts a frame, it first sends it via the bypass
//!    channel (if any subscribers exist), then continues with the normal TCP
//!    path for remote receivers.
//! 3. When `NdiReceiver` connects to a loopback/local address, it calls
//!    [`LocalBypassRegistry::subscribe`].  If a sender is registered under
//!    that source name, the receiver gets a `broadcast::Receiver<NdiFrame>`
//!    and feeds frames directly into its queues, skipping TCP.
//! 4. `Drop for NdiSender` calls [`LocalBypassRegistry::unregister`], which
//!    removes the entry and causes all subscriber receivers to see
//!    `RecvError::Closed` (handled gracefully).
//!
//! # Safety
//!
//! The registry uses only safe Rust: a `std::sync::Mutex<HashMap<…>>` guarded
//! by a `OnceLock` for single-process initialization.  No `unsafe` code.

#![allow(dead_code)]

use crate::protocol::NdiFrame;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tokio::sync::broadcast;

/// Capacity of the per-source broadcast channel (frames).
///
/// 16 frames gives ~0.5 s of headroom at 30 fps before a lagged receiver
/// starts dropping messages.
pub const BYPASS_CHANNEL_CAPACITY: usize = 16;

/// The global, process-local bypass registry.
static REGISTRY: OnceLock<LocalBypassRegistry> = OnceLock::new();

/// Process-local registry mapping source names to frame broadcast channels.
pub struct LocalBypassRegistry {
    senders: Mutex<HashMap<String, broadcast::Sender<NdiFrame>>>,
}

impl LocalBypassRegistry {
    /// Return the process-global registry, initialising it on first call.
    pub fn global() -> &'static Self {
        REGISTRY.get_or_init(|| Self {
            senders: Mutex::new(HashMap::new()),
        })
    }

    /// Register a sender for `source_name`.
    ///
    /// Returns the [`broadcast::Sender`] that the `NdiSender` should hold and
    /// use to distribute frames.  If a previous registration exists for the
    /// same name it is replaced (the old channel is closed, causing any
    /// existing subscribers to receive `RecvError::Closed`).
    pub fn register(&self, source_name: &str) -> broadcast::Sender<NdiFrame> {
        let (tx, _rx) = broadcast::channel(BYPASS_CHANNEL_CAPACITY);
        // Drop the initial receiver — real receivers subscribe via `subscribe()`.
        let mut guard = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(source_name.to_string(), tx.clone());
        tx
    }

    /// Subscribe to frames from `source_name`.
    ///
    /// Returns `Some(receiver)` when a sender with that name is registered,
    /// `None` otherwise.
    pub fn subscribe(&self, source_name: &str) -> Option<broadcast::Receiver<NdiFrame>> {
        let guard = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        guard.get(source_name).map(|tx| tx.subscribe())
    }

    /// Unregister the sender for `source_name`.
    ///
    /// After this call the channel is dropped, and any outstanding receivers
    /// will observe `RecvError::Closed` on their next `recv()`.
    pub fn unregister(&self, source_name: &str) {
        let mut guard = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        guard.remove(source_name);
    }

    /// Return `true` when an active sender is registered for `source_name`.
    pub fn is_registered(&self, source_name: &str) -> bool {
        let guard = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        guard.contains_key(source_name)
    }

    /// Return the number of registered sources.
    #[cfg(test)]
    pub fn registered_count(&self) -> usize {
        let guard = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        guard.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{NdiAudioFrame, NdiFrame, NdiVideoFrame};
    use crate::{AudioFormat, VideoFormat};
    use bytes::Bytes;

    fn make_video_frame(seq: u32) -> NdiFrame {
        let fmt = VideoFormat::new(320, 240, 30, 1);
        let data = Bytes::from(vec![0u8; 320 * 240 * 2]);
        NdiFrame::Video(NdiVideoFrame::new(seq, 1_000_000, fmt, data, 320 * 2))
    }

    fn make_audio_frame(seq: u32) -> NdiFrame {
        let fmt = AudioFormat::new(48_000, 2, 16);
        let data = Bytes::from(vec![0u8; 480 * 2 * 2]);
        NdiFrame::Audio(NdiAudioFrame::new(seq, 1_000_000, fmt, data, 480))
    }

    // Each test uses a unique source name to avoid cross-test interference in
    // the global registry.

    #[tokio::test]
    async fn test_register_and_subscribe() {
        let reg = LocalBypassRegistry::global();
        let name = "test_register_and_subscribe";

        let tx = reg.register(name);
        let mut rx = reg
            .subscribe(name)
            .expect("subscribe should succeed after register");

        let frame = make_video_frame(1);
        tx.send(frame.clone()).ok();

        let received = rx.recv().await.expect("should receive frame");
        assert!(matches!(received, NdiFrame::Video(_)));

        reg.unregister(name);
    }

    #[tokio::test]
    async fn test_subscribe_before_register_returns_none() {
        let reg = LocalBypassRegistry::global();
        let name = "test_subscribe_before_register_returns_none";

        assert!(reg.subscribe(name).is_none());
    }

    #[tokio::test]
    async fn test_unregister_closes_channel() {
        let reg = LocalBypassRegistry::global();
        let name = "test_unregister_closes_channel";

        let _tx = reg.register(name);
        let mut rx = reg.subscribe(name).expect("subscribe should succeed");

        // Unregister removes the sender from the map; when the sender clone
        // held by `_tx` also drops, the channel closes.
        reg.unregister(name);
        drop(_tx);

        let result = rx.recv().await;
        assert!(
            result.is_err(),
            "channel should be closed after unregister+drop"
        );
    }

    #[tokio::test]
    async fn test_multiple_subscribers_receive_same_frame() {
        let reg = LocalBypassRegistry::global();
        let name = "test_multiple_subscribers_receive_same_frame";

        let tx = reg.register(name);
        let mut rx1 = reg.subscribe(name).expect("sub 1");
        let mut rx2 = reg.subscribe(name).expect("sub 2");

        let frame = make_video_frame(99);
        tx.send(frame).ok();

        let f1 = rx1.recv().await.expect("rx1");
        let f2 = rx2.recv().await.expect("rx2");

        if let (NdiFrame::Video(v1), NdiFrame::Video(v2)) = (f1, f2) {
            assert_eq!(v1.header.sequence, 99);
            assert_eq!(v2.header.sequence, 99);
            // Both receivers see the same underlying Bytes (Arc-backed, no copy).
            assert_eq!(v1.data, v2.data);
        } else {
            panic!("expected Video frames");
        }

        reg.unregister(name);
    }

    #[tokio::test]
    async fn test_audio_frame_bypass() {
        let reg = LocalBypassRegistry::global();
        let name = "test_audio_frame_bypass";

        let tx = reg.register(name);
        let mut rx = reg.subscribe(name).expect("subscribe");

        let frame = make_audio_frame(7);
        tx.send(frame).ok();

        let received = rx.recv().await.expect("receive");
        assert!(matches!(received, NdiFrame::Audio(_)));

        reg.unregister(name);
    }

    #[tokio::test]
    async fn test_is_registered() {
        let reg = LocalBypassRegistry::global();
        let name = "test_is_registered";

        assert!(!reg.is_registered(name));
        let _tx = reg.register(name);
        assert!(reg.is_registered(name));
        reg.unregister(name);
        assert!(!reg.is_registered(name));
    }
}
