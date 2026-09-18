//! Mount-point registry — stream sources that RTSP clients can subscribe to.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Maximum number of queued RTP packets in a broadcast channel before
/// slow subscribers start receiving `Lagged` errors.
const BROADCAST_CAPACITY: usize = 512;

/// A single registered mount point — a stream source that clients DESCRIBE/SETUP/PLAY.
pub struct MountPoint {
    /// The URL path this mount point is registered at (e.g. `/stream`).
    pub path: String,
    /// Pre-computed SDP text returned in DESCRIBE responses.
    pub sdp: String,
    /// Broadcast sender for RTP packet bytes.
    sender: broadcast::Sender<Arc<Vec<u8>>>,
}

impl MountPoint {
    /// Create a new mount point.
    ///
    /// Returns the `MountPoint` and an initial `Receiver` you can use
    /// to monitor the stream (e.g. for testing).
    #[must_use]
    pub fn new(path: String, sdp: String) -> (Self, broadcast::Receiver<Arc<Vec<u8>>>) {
        let (sender, rx) = broadcast::channel(BROADCAST_CAPACITY);
        let mp = Self { path, sdp, sender };
        (mp, rx)
    }

    /// Subscribe to the RTP stream.
    ///
    /// Each subscriber gets an independent copy of every packet buffer.
    /// Slow subscribers receive a `Lagged` error when they fall more than
    /// `BROADCAST_CAPACITY` packets behind; the connection handler should
    /// `continue` on `Lagged` rather than disconnect.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<Vec<u8>>> {
        self.sender.subscribe()
    }

    /// Publish an RTP packet to all subscribers.
    ///
    /// Returns the number of active subscribers that received the packet.
    /// A return value of `0` simply means no one is playing yet.
    pub fn publish(&self, rtp_bytes: Arc<Vec<u8>>) -> usize {
        self.sender.send(rtp_bytes).unwrap_or(0)
    }
}

/// Registry of all active mount points, shared across connection handlers.
///
/// Cloning the registry is cheap — the inner `HashMap` is behind an `Arc<Mutex>`.
#[derive(Default, Clone)]
pub struct MountPointRegistry {
    inner: Arc<Mutex<HashMap<String, Arc<MountPoint>>>>,
}

impl MountPointRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a mount point and return a shared reference to it.
    ///
    /// If a mount point with the same path already exists it is replaced.
    pub fn register(&self, point: MountPoint) -> Arc<MountPoint> {
        let path = point.path.clone();
        let shared = Arc::new(point);
        let clone = Arc::clone(&shared);
        self.inner
            .lock()
            .expect("registry mutex poisoned")
            .insert(path, shared);
        clone
    }

    /// Look up a mount point by path.
    #[must_use]
    pub fn lookup(&self, path: &str) -> Option<Arc<MountPoint>> {
        self.inner
            .lock()
            .expect("registry mutex poisoned")
            .get(path)
            .cloned()
    }

    /// Remove a mount point.
    ///
    /// Returns `true` if the path was registered and has been removed.
    pub fn unregister(&self, path: &str) -> bool {
        self.inner
            .lock()
            .expect("registry mutex poisoned")
            .remove(path)
            .is_some()
    }

    /// Return a sorted list of all registered paths.
    #[must_use]
    pub fn list_paths(&self) -> Vec<String> {
        let mut paths: Vec<String> = self
            .inner
            .lock()
            .expect("registry mutex poisoned")
            .keys()
            .cloned()
            .collect();
        paths.sort();
        paths
    }
}
