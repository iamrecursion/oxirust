use super::*;
use crate::rtmp::RtmpClient;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::task::JoinHandle;

/// Bounded per-target queue between `forward()` and the background forwarder.
///
/// A full queue is honest back-pressure: packets are counted as dropped, never
/// silently reported as forwarded.
const RELAY_QUEUE_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
pub struct RelayTarget {
    /// RTMP URL to forward to (rtmp://host:port/app/stream).
    pub url: String,
    /// Whether this relay is currently active.
    pub active: bool,
    /// Bytes forwarded so far.
    pub bytes_forwarded: u64,
    /// Packets dropped due to back-pressure.
    pub packets_dropped: u64,
}

impl RelayTarget {
    /// Creates a new relay target.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            active: true,
            bytes_forwarded: 0,
            packets_dropped: 0,
        }
    }
}

/// A live outbound forwarder for a single `(stream_key, url)` pair.
///
/// The background task owns a real [`RtmpClient`], performs the RTMP
/// connect + publish handshake, and forwards media it receives over `tx`. The
/// atomics reflect the *real* state of that connection: `bytes_forwarded` only
/// advances after bytes are written to the socket, and `healthy` is set only
/// once publishing actually succeeds — so `forward()`/`stats()` can never
/// report "healthy + forwarded" while nothing is really happening.
struct RelayWorker {
    /// Bounded queue to the forwarder task.
    tx: mpsc::Sender<MediaPacket>,
    /// Bytes actually written to the outbound socket.
    bytes_forwarded: Arc<AtomicU64>,
    /// Packets dropped because the queue was full (back-pressure).
    packets_dropped: Arc<AtomicU64>,
    /// True once the connection is established and publishing.
    healthy: Arc<AtomicBool>,
    /// True while the initial connect/publish is still in progress.
    connecting: Arc<AtomicBool>,
    /// Background task handle (aborted on drop).
    handle: JoinHandle<()>,
}

impl RelayWorker {
    /// Spawns a forwarder that connects to `url` and drains its queue.
    fn spawn(url: String) -> Self {
        let (tx, rx) = mpsc::channel::<MediaPacket>(RELAY_QUEUE_CAPACITY);
        let bytes_forwarded = Arc::new(AtomicU64::new(0));
        let packets_dropped = Arc::new(AtomicU64::new(0));
        let healthy = Arc::new(AtomicBool::new(false));
        let connecting = Arc::new(AtomicBool::new(true));

        let handle = tokio::spawn(run_forwarder(
            url,
            rx,
            Arc::clone(&bytes_forwarded),
            Arc::clone(&healthy),
            Arc::clone(&connecting),
        ));

        Self {
            tx,
            bytes_forwarded,
            packets_dropped,
            healthy,
            connecting,
            handle,
        }
    }

    /// Whether this target should still be reported active: healthy, or the
    /// initial connect is still in flight. A finished-and-failed worker is not.
    fn is_up(&self) -> bool {
        self.healthy.load(Ordering::Relaxed) || self.connecting.load(Ordering::Relaxed)
    }
}

impl Drop for RelayWorker {
    fn drop(&mut self) {
        // Stop the outbound connection when the manager forgets the target.
        self.handle.abort();
    }
}

/// Background task: establish a real outbound RTMP session and forward media.
async fn run_forwarder(
    url: String,
    mut rx: mpsc::Receiver<MediaPacket>,
    bytes_forwarded: Arc<AtomicU64>,
    healthy: Arc<AtomicBool>,
    connecting: Arc<AtomicBool>,
) {
    let mut client = RtmpClient::new();

    // Real TCP connect + RTMP handshake. On failure the target is left
    // UNHEALTHY (connecting=false, healthy=false) and nothing is forwarded.
    if let Err(e) = client.connect(&url).await {
        tracing::warn!(target: "rtmp::relay", url = %url, error = %e, "relay connect failed");
        connecting.store(false, Ordering::Relaxed);
        healthy.store(false, Ordering::Relaxed);
        return;
    }

    // Publish under the URL's final path segment (rtmp://host/app/<stream>).
    let stream_name = url
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or("stream")
        .to_string();
    if let Err(e) = client.publish(&stream_name, "live").await {
        tracing::warn!(target: "rtmp::relay", url = %url, error = %e, "relay publish failed");
        connecting.store(false, Ordering::Relaxed);
        healthy.store(false, Ordering::Relaxed);
        let _ = client.close().await;
        return;
    }

    connecting.store(false, Ordering::Relaxed);
    healthy.store(true, Ordering::Relaxed);

    while let Some(packet) = rx.recv().await {
        let len = packet.data.len() as u64;
        let ptype = packet.packet_type;
        let send_result = match ptype {
            MediaPacketType::Audio => client.send_audio(packet.data).await,
            MediaPacketType::Video => client.send_video(packet.data).await,
            // Metadata packets would need AMF re-encoding to forward faithfully;
            // rather than mislabel them as A/V we skip them (honest omission).
            MediaPacketType::Data => Ok(()),
        };
        match send_result {
            Ok(()) => {
                if !matches!(ptype, MediaPacketType::Data) {
                    // Count only bytes that really reached the socket.
                    bytes_forwarded.fetch_add(len, Ordering::Relaxed);
                }
            }
            Err(e) => {
                tracing::warn!(target: "rtmp::relay", url = %url, error = %e, "relay send failed");
                break;
            }
        }
    }

    healthy.store(false, Ordering::Relaxed);
    let _ = client.close().await;
}

/// Relay manager for forwarding streams to other RTMP endpoints.
///
/// When a publisher pushes media the relay manager fans it out to all
/// configured target URLs via independent broadcast channels.  Each target
/// gets its own `broadcast::Receiver` so slow targets cannot stall the
/// publisher or other targets.
pub struct RelayManager {
    /// Relay targets keyed by stream key.
    targets: Arc<RwLock<HashMap<String, Vec<RelayTarget>>>>,
    /// Media channels for each active stream.
    channels: Arc<RwLock<HashMap<String, broadcast::Sender<MediaPacket>>>>,
    /// Live outbound forwarders keyed by `(stream_key, url)`.
    workers: Arc<RwLock<HashMap<(String, String), RelayWorker>>>,
}

impl RelayManager {
    /// Creates a new relay manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            targets: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(HashMap::new())),
            workers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Adds a relay target for the given stream key.
    pub async fn add_target(&self, stream_key: impl Into<String>, url: impl Into<String>) {
        let key = stream_key.into();
        let mut targets = self.targets.write().await;
        targets
            .entry(key)
            .or_insert_with(Vec::new)
            .push(RelayTarget::new(url));
    }

    /// Registers a media channel for a stream that is starting to publish.
    ///
    /// Returns a `broadcast::Receiver` the relay loop should subscribe to.
    pub async fn register_stream(
        &self,
        stream_key: impl Into<String>,
        tx: broadcast::Sender<MediaPacket>,
    ) -> broadcast::Receiver<MediaPacket> {
        let key = stream_key.into();
        let rx = tx.subscribe();
        let mut channels = self.channels.write().await;
        channels.insert(key, tx);
        rx
    }

    /// Forwards a media packet to all active targets for a given stream.
    ///
    /// Each target is serviced by a background task that opens a real outbound
    /// RTMP connection (lazily on the first packet) and forwards media to it.
    /// This method only hands the packet to that task's bounded queue; the
    /// task counts bytes after they hit the socket. A full queue is recorded
    /// as a drop, and a target whose connection failed is left unhealthy — no
    /// packet is ever reported forwarded unless it truly was.
    pub async fn forward(&self, stream_key: &str, packet: &MediaPacket) {
        // Snapshot the active target URLs under a short read lock.
        let urls: Vec<String> = {
            let targets = self.targets.read().await;
            match targets.get(stream_key) {
                Some(list) => list
                    .iter()
                    .filter(|t| t.active)
                    .map(|t| t.url.clone())
                    .collect(),
                None => return,
            }
        };
        if urls.is_empty() {
            return;
        }

        let mut workers = self.workers.write().await;
        for url in urls {
            let key = (stream_key.to_string(), url.clone());
            let worker = workers
                .entry(key)
                .or_insert_with(|| RelayWorker::spawn(url.clone()));

            match worker.tx.try_send(packet.clone()) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Honest back-pressure: the target cannot keep up.
                    worker.packets_dropped.fetch_add(1, Ordering::Relaxed);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    // The forwarder task exited (connect/send failed).
                    worker.healthy.store(false, Ordering::Relaxed);
                    worker.connecting.store(false, Ordering::Relaxed);
                }
            }
        }
    }

    /// Removes the channel for a stream that has stopped publishing and tears
    /// down any outbound forwarders for it (their `Drop` aborts the tasks).
    pub async fn unregister_stream(&self, stream_key: &str) {
        {
            let mut channels = self.channels.write().await;
            channels.remove(stream_key);
        }
        let mut workers = self.workers.write().await;
        workers.retain(|(sk, _), _| sk != stream_key);
    }

    /// Returns statistics for a stream, reflecting the live forwarder state.
    ///
    /// `bytes_forwarded` / `packets_dropped` / `active` are read from the
    /// running worker (if any) so they report reality, not intent: a target
    /// whose outbound connection failed is reported inactive with zero bytes
    /// forwarded.
    pub async fn stats(&self, stream_key: &str) -> Vec<RelayTarget> {
        let targets = self.targets.read().await;
        let workers = self.workers.read().await;
        let Some(list) = targets.get(stream_key) else {
            return Vec::new();
        };
        list.iter()
            .map(|t| {
                let key = (stream_key.to_string(), t.url.clone());
                if let Some(w) = workers.get(&key) {
                    RelayTarget {
                        url: t.url.clone(),
                        active: t.active && w.is_up(),
                        bytes_forwarded: w.bytes_forwarded.load(Ordering::Relaxed),
                        packets_dropped: w.packets_dropped.load(Ordering::Relaxed),
                    }
                } else {
                    t.clone()
                }
            })
            .collect()
    }

    /// Marks a target as inactive (e.g. after a failed connection).
    pub async fn mark_inactive(&self, stream_key: &str, url: &str) {
        let mut targets = self.targets.write().await;
        if let Some(list) = targets.get_mut(stream_key) {
            for target in list.iter_mut() {
                if target.url == url {
                    target.active = false;
                }
            }
        }
    }
}

impl Default for RelayManager {
    fn default() -> Self {
        Self::new()
    }
}
