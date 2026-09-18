//! Per-connection RTSP server handler.
//!
//! Each accepted TCP connection is split into an [`OwnedReadHalf`] and an
//! [`OwnedWriteHalf`] (see [`TcpStream::into_split`]). The connection actor:
//!
//! 1. Keeps the **read half** in the request-processing loop: it reads bytes
//!    into a buffer and calls `try_parse_request` until a complete RTSP request
//!    arrives, then dispatches to the appropriate method handler.
//! 2. Shares the **write half** behind an `Arc<Mutex<OwnedWriteHalf>>`. Both the
//!    RTSP response path and a dedicated RTP writer task acquire that mutex, so
//!    control responses and interleaved RTP frames serialize cleanly and never
//!    interleave mid-message on the wire (RFC 2326 §10.12 framing integrity).
//!
//! After PLAY, the RTP writer task runs as a sibling `tokio::spawn` and forwards
//! interleaved frames to the client **concurrently** with request processing —
//! it `recv().await`s frames from the mount point's broadcast channel and writes
//! them through the shared write half as they arrive (no polling, no busy-wait).
//! It is stopped cooperatively via an `Arc<AtomicBool>` flag plus an awaitable
//! `Notify` wakeup on PAUSE, TEARDOWN, re-PLAY, or connection drop.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{broadcast, Mutex, Notify};
use tokio::task::JoinHandle;

use super::registry::MountPointRegistry;
use super::state::{RtspServerConfig, RtspSession, RtspSessionState};
use crate::rtsp::message::{try_parse_request, Headers, RequestParseStatus, Response};
use crate::rtsp::transport::encode_interleaved;
use crate::rtsp::Method;

/// Read buffer capacity — 64 KiB should accommodate even large SDP bodies.
const READ_BUF_CAPACITY: usize = 65536;
/// I/O read chunk size.
const CHUNK_SIZE: usize = 4096;
/// Server identification string.
const SERVER_NAME: &str = concat!("oximedia-net-rtsp/", env!("CARGO_PKG_VERSION"));

/// Reason [`ServerConnection::run_inner`]'s request loop terminated.
///
/// Carries enough context for [`ServerConnection::run`] to log a meaningful
/// message instead of silently discarding the failure. Normal client-initiated
/// disconnects (EOF) are distinguished from actual parse/I-O errors so the
/// former can be logged at `debug` and the latter at `warn`.
enum ConnectionEnd {
    /// The client closed the TCP connection (read returned 0 bytes).
    ClosedByClient,
    /// A request could not be parsed as valid RTSP.
    Parse(crate::rtsp::message::ParseError),
    /// An I/O error occurred while reading from or writing to the socket.
    Io(std::io::Error),
}

/// Per-connection RTSP server actor.
pub struct ServerConnection {
    /// Read half of the client TCP connection — owned by the request loop.
    reader: OwnedReadHalf,
    /// Write half, shared between the response path and the RTP writer task so
    /// that responses and interleaved RTP frames never overlap on the wire.
    writer: Arc<Mutex<OwnedWriteHalf>>,
    config: Arc<RtspServerConfig>,
    registry: MountPointRegistry,
    session: Option<RtspSession>,
    read_buf: Vec<u8>,
    /// Cooperative stop flag for the currently-active RTP writer task.
    rtp_stop: Option<Arc<AtomicBool>>,
    /// Wakeup handle so the RTP writer task observes the stop flag promptly even
    /// while parked on `broadcast::Receiver::recv`.
    rtp_wake: Option<Arc<Notify>>,
    /// Join handle for the currently-active RTP writer task.
    rtp_task: Option<JoinHandle<()>>,
}

impl ServerConnection {
    /// Create a new connection handler.
    #[must_use]
    pub fn new(
        stream: TcpStream,
        config: Arc<RtspServerConfig>,
        registry: MountPointRegistry,
    ) -> Self {
        let (reader, writer) = stream.into_split();
        Self {
            reader,
            writer: Arc::new(Mutex::new(writer)),
            config,
            registry,
            session: None,
            read_buf: Vec::with_capacity(READ_BUF_CAPACITY),
            rtp_stop: None,
            rtp_wake: None,
            rtp_task: None,
        }
    }

    /// Drive the connection until the client disconnects or an error occurs.
    pub async fn run(mut self) {
        match self.run_inner().await {
            Ok(()) => {}
            Err(ConnectionEnd::ClosedByClient) => {
                tracing::debug!("RTSP client closed connection");
            }
            Err(ConnectionEnd::Parse(e)) => {
                tracing::warn!(error = %e, "RTSP connection closed: request parse error");
            }
            Err(ConnectionEnd::Io(e)) => {
                tracing::warn!(error = %e, "RTSP connection closed: I/O error");
            }
        }
        // Connection is closing — tear down the RTP writer task. We signal the
        // cooperative stop flag *and* hard-`abort()` the handle as a backstop:
        // the socket is going away, so stream integrity no longer matters and we
        // must guarantee the spawned task (which holds a clone of the shared
        // write half) is released rather than leaked.
        self.stop_rtp_task(true);
    }

    async fn run_inner(&mut self) -> Result<(), ConnectionEnd> {
        loop {
            // Try to parse a complete request from the read buffer.
            match try_parse_request(&self.read_buf).map_err(ConnectionEnd::Parse)? {
                RequestParseStatus::Complete { consumed, request } => {
                    self.read_buf.drain(..consumed);
                    let cseq = request
                        .headers
                        .get("CSeq")
                        .and_then(|v| v.trim().parse::<u32>().ok())
                        .unwrap_or(0);
                    let response = self.dispatch(&request, cseq).await;
                    let wire = response.encode();
                    // Serialize against the RTP writer task via the shared mutex.
                    // The guard is held only for this one response, so the writer
                    // task can interleave whole frames between responses.
                    let mut guard = self.writer.lock().await;
                    guard.write_all(&wire).await.map_err(ConnectionEnd::Io)?;
                    guard.flush().await.map_err(ConnectionEnd::Io)?;
                }
                RequestParseStatus::NeedMore => {
                    // Need more bytes — read from the socket's read half.
                    let mut chunk = [0u8; CHUNK_SIZE];
                    let n = self
                        .reader
                        .read(&mut chunk)
                        .await
                        .map_err(ConnectionEnd::Io)?;
                    if n == 0 {
                        return Err(ConnectionEnd::ClosedByClient);
                    }
                    self.read_buf.extend_from_slice(&chunk[..n]);
                }
            }
        }
    }

    /// Stop the active RTP writer task, if any.
    ///
    /// Always sets the cooperative stop flag and wakes the task so it exits
    /// between frames — because the writer task and the response path share the
    /// same write-half mutex, any in-flight interleaved frame is written in full
    /// before the next response, so the control stream is never corrupted.
    ///
    /// When `hard_abort` is `true` the join handle is additionally `abort()`ed
    /// (used on connection drop, where prompt release outranks graceful exit).
    /// Otherwise the handle is detached; the task is guaranteed to observe the
    /// stop flag and terminate on its own, so the request loop never blocks
    /// joining it (avoiding any write-backpressure deadlock).
    fn stop_rtp_task(&mut self, hard_abort: bool) {
        if let Some(flag) = self.rtp_stop.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(wake) = self.rtp_wake.take() {
            wake.notify_one();
        }
        if let Some(task) = self.rtp_task.take() {
            if hard_abort {
                task.abort();
            }
        }
    }

    /// Dispatch an incoming request to the correct handler.
    async fn dispatch(&mut self, req: &crate::rtsp::message::Request, cseq: u32) -> Response {
        match req.method {
            Method::Options => self.handle_options(cseq),
            Method::Describe => self.handle_describe(&req.uri, cseq),
            Method::Setup => self.handle_setup(&req.uri, req.headers.get("Transport"), cseq),
            Method::Play => self.handle_play(cseq).await,
            Method::Pause => self.handle_pause(cseq),
            Method::Teardown => self.handle_teardown(cseq),
            Method::GetParameter => self.handle_get_parameter(cseq),
            _ => {
                let mut r = Response::build(501, cseq);
                add_server_header(&mut r.headers);
                r
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Method handlers
    // ────────────────────────────────────────────────────────────────────────

    fn handle_options(&self, cseq: u32) -> Response {
        let mut r = Response::build(200, cseq);
        r.headers.insert(
            "Public",
            "OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, GET_PARAMETER",
        );
        add_server_header(&mut r.headers);
        r
    }

    fn handle_describe(&self, uri: &str, cseq: u32) -> Response {
        let path = extract_path(uri);
        match self.registry.lookup(&path) {
            Some(mp) => {
                let sdp_bytes = mp.sdp.as_bytes().to_vec();
                let sdp_len = sdp_bytes.len();
                let mut r = Response::build(200, cseq);
                r.headers.insert("Content-Type", "application/sdp");
                r.headers.insert("Content-Base", uri);
                r.headers.insert("Content-Length", sdp_len.to_string());
                add_server_header(&mut r.headers);
                r.body = sdp_bytes;
                r
            }
            None => {
                let mut r = Response::build(404, cseq);
                add_server_header(&mut r.headers);
                r
            }
        }
    }

    fn handle_setup(&mut self, uri: &str, transport_header: Option<&str>, cseq: u32) -> Response {
        // Extract interleaved channel IDs from Transport header.
        // Accept "RTP/AVP/TCP;...; interleaved=N-M" or fall back to 0-1.
        let (rtp_ch, rtcp_ch) = transport_header
            .and_then(parse_interleaved_channels)
            .unwrap_or((0, 1));

        let path = extract_path(uri);
        if self.registry.lookup(&path).is_none() {
            // Try the parent path (SETUP URI often includes /trackID=1 suffix).
            let parent = parent_path(&path);
            if self.registry.lookup(&parent).is_none() {
                let mut r = Response::build(404, cseq);
                add_server_header(&mut r.headers);
                return r;
            }
            // Use parent mount path.
            self.create_session(parent, rtp_ch);
        } else {
            self.create_session(path, rtp_ch);
        }

        let session_id = self
            .session
            .as_ref()
            .map(|s| s.id.clone())
            .unwrap_or_default();
        let timeout_secs = self.config.session_timeout.as_secs();

        let mut r = Response::build(200, cseq);
        r.headers.insert(
            "Transport",
            format!("RTP/AVP/TCP;unicast;interleaved={rtp_ch}-{rtcp_ch}"),
        );
        r.headers
            .insert("Session", format!("{session_id};timeout={timeout_secs}"));
        add_server_header(&mut r.headers);
        r
    }

    async fn handle_play(&mut self, cseq: u32) -> Response {
        let (session_id, mount_path, channel_id) = match self.session.as_mut() {
            Some(s)
                if s.state == RtspSessionState::Ready || s.state == RtspSessionState::Paused =>
            {
                let id = s.id.clone();
                let path = s.mount_path.clone();
                let ch = s.channel_id;
                s.state = RtspSessionState::Playing;
                s.refresh();
                (id, path, ch)
            }
            Some(s) if s.state == RtspSessionState::Playing => {
                // Already playing — just refresh and confirm.
                let id = s.id.clone();
                let timeout_secs = self.config.session_timeout.as_secs();
                s.refresh();
                let mut r = Response::build(200, cseq);
                r.headers
                    .insert("Session", format!("{id};timeout={timeout_secs}"));
                add_server_header(&mut r.headers);
                return r;
            }
            _ => {
                let mut r = Response::build(455, cseq); // Method Not Valid in This State
                add_server_header(&mut r.headers);
                return r;
            }
        };

        // Stop any existing RTP writer before starting a new one (e.g. re-PLAY
        // after PAUSE). Cooperative stop keeps the shared write half intact.
        self.stop_rtp_task(false);

        // Start a new RTP writer task that forwards interleaved frames to the
        // client concurrently with request processing. It subscribes to the
        // mount point's broadcast channel and writes each frame through the
        // shared write half as it arrives — no polling, no per-request drain.
        if let Some(mp) = self.registry.lookup(&mount_path) {
            let stop = Arc::new(AtomicBool::new(false));
            let wake = Arc::new(Notify::new());
            let broadcast_rx = mp.subscribe();
            let writer = Arc::clone(&self.writer);
            let task_stop = Arc::clone(&stop);
            let task_wake = Arc::clone(&wake);

            let handle = tokio::spawn(async move {
                rtp_writer_loop(broadcast_rx, writer, task_stop, task_wake, channel_id).await;
            });

            self.rtp_stop = Some(stop);
            self.rtp_wake = Some(wake);
            self.rtp_task = Some(handle);
        }

        let timeout_secs = self.config.session_timeout.as_secs();
        let mut r = Response::build(200, cseq);
        r.headers
            .insert("Session", format!("{session_id};timeout={timeout_secs}"));
        r.headers.insert("Range", "npt=0.000-");
        add_server_header(&mut r.headers);
        r
    }

    fn handle_pause(&mut self, cseq: u32) -> Response {
        // Stop the RTP writer task. The shared write-half mutex guarantees the
        // PAUSE response that follows is never spliced into a half-written frame.
        self.stop_rtp_task(false);

        match self.session.as_mut() {
            Some(s) if s.state == RtspSessionState::Playing => {
                let id = s.id.clone();
                let timeout_secs = self.config.session_timeout.as_secs();
                s.state = RtspSessionState::Paused;
                s.refresh();
                let mut r = Response::build(200, cseq);
                r.headers
                    .insert("Session", format!("{id};timeout={timeout_secs}"));
                add_server_header(&mut r.headers);
                r
            }
            _ => {
                let mut r = Response::build(455, cseq);
                add_server_header(&mut r.headers);
                r
            }
        }
    }

    fn handle_teardown(&mut self, cseq: u32) -> Response {
        // Stop the RTP writer task; the shared write-half mutex serializes the
        // TEARDOWN response after any in-flight frame.
        self.stop_rtp_task(false);
        self.session = None;
        let mut r = Response::build(200, cseq);
        add_server_header(&mut r.headers);
        r
    }

    fn handle_get_parameter(&mut self, cseq: u32) -> Response {
        // Keepalive — just refresh the session timeout and respond 200.
        let session_line = match self.session.as_mut() {
            Some(s) => {
                s.refresh();
                let timeout_secs = self.config.session_timeout.as_secs();
                format!("{};timeout={timeout_secs}", s.id)
            }
            None => String::new(),
        };

        let mut r = Response::build(200, cseq);
        if !session_line.is_empty() {
            r.headers.insert("Session", session_line);
        }
        add_server_header(&mut r.headers);
        r
    }

    // ────────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────────

    fn create_session(&mut self, mount_path: String, channel_id: u8) {
        let id = generate_session_id();
        let timeout = self.config.session_timeout;
        let mut session = RtspSession::new(id, mount_path, channel_id, timeout);
        session.state = RtspSessionState::Ready;
        self.session = Some(session);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Free-standing helpers
// ────────────────────────────────────────────────────────────────────────────

/// Add the standard `Server:` header to a response.
fn add_server_header(headers: &mut Headers) {
    headers.insert("Server", SERVER_NAME);
}

/// Extract the path component from an `rtsp://host[:port]/path` URI.
fn extract_path(uri: &str) -> String {
    if let Some(rest) = uri
        .strip_prefix("rtsp://")
        .or_else(|| uri.strip_prefix("rtsps://"))
    {
        if let Some(idx) = rest.find('/') {
            return rest[idx..].to_string();
        }
        return "/".to_string();
    }
    // Relative URI — use as-is.
    if uri.starts_with('/') {
        uri.to_string()
    } else {
        format!("/{uri}")
    }
}

/// Strip the last path component (e.g. `/stream/trackID=1` → `/stream`).
fn parent_path(path: &str) -> String {
    match path.rfind('/') {
        Some(0) | None => "/".to_string(),
        Some(idx) => path[..idx].to_string(),
    }
}

/// Parse `interleaved=N-M` out of a Transport header value.
fn parse_interleaved_channels(transport: &str) -> Option<(u8, u8)> {
    for part in transport.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("interleaved=") {
            let mut nums = rest.splitn(2, '-');
            let a = nums.next()?.trim().parse::<u8>().ok()?;
            let b = nums.next()?.trim().parse::<u8>().ok()?;
            return Some((a, b));
        }
    }
    None
}

/// Generate a unique session ID from system time and an atomic counter.
fn generate_session_id() -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{:012x}{:04x}", ts & 0xFFFF_FFFF_FFFF, seq & 0xFFFF)
}

/// RTP egress loop: forward interleaved RTP frames from the mount point's
/// broadcast channel to the shared TCP write half until cooperatively stopped.
///
/// Runs as a sibling task spawned by `handle_play`. Writes go through the same
/// `Arc<Mutex<OwnedWriteHalf>>` as RTSP responses, so an interleaved RTP frame
/// and a control response can never overlap mid-message on the wire
/// (RFC 2326 §10.12 framing integrity).
///
/// Cancellation is cooperative: the loop only re-checks `stop` at the top and
/// in the `wake` branch — never while a frame is being written — guaranteeing
/// each interleaved frame is emitted in full before the task exits. `wake`
/// (a [`Notify`]) is used so the task observes the stop flag promptly even while
/// parked on [`broadcast::Receiver::recv`], without busy-waiting.
async fn rtp_writer_loop(
    mut broadcast_rx: broadcast::Receiver<Arc<Vec<u8>>>,
    writer: Arc<Mutex<OwnedWriteHalf>>,
    stop: Arc<AtomicBool>,
    wake: Arc<Notify>,
    channel_id: u8,
) {
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        tokio::select! {
            // Bias toward the stop signal so a pending wakeup wins over a queued
            // frame and shutdown stays prompt.
            biased;
            // Cooperative cancellation point. `recv` is cancel-safe, so dropping
            // its future here loses no frame; we loop and the top-of-loop check
            // observes the stop flag. We never reach this branch mid-write.
            _ = wake.notified() => {
                continue;
            }
            frame = broadcast_rx.recv() => {
                match frame {
                    Ok(rtp_bytes) => {
                        // Cheap early-out before contending for the write lock.
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let framed = encode_interleaved(channel_id, &rtp_bytes);
                        let mut guard = writer.lock().await;
                        // Authoritative re-check under the lock: PAUSE/TEARDOWN set
                        // the stop flag during dispatch, before their response
                        // acquires this same lock. Re-checking here guarantees no
                        // RTP frame is ever written *after* such a response.
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        if guard.write_all(&framed).await.is_err() {
                            break;
                        }
                        if guard.flush().await.is_err() {
                            break;
                        }
                    }
                    // Sender (mount point) gone — nothing more will ever arrive.
                    Err(broadcast::error::RecvError::Closed) => break,
                    // Slow consumer fell behind; skip the gap and keep streaming.
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_path_from_rtsp_uri() {
        assert_eq!(
            extract_path("rtsp://cam.local:554/live/stream"),
            "/live/stream"
        );
        assert_eq!(extract_path("rtsp://cam/"), "/");
        assert_eq!(extract_path("rtsp://cam"), "/");
        assert_eq!(extract_path("/stream"), "/stream");
    }

    #[test]
    fn parent_path_strips_last_segment() {
        assert_eq!(parent_path("/stream/trackID=1"), "/stream");
        assert_eq!(parent_path("/stream"), "/");
        assert_eq!(parent_path("/"), "/");
    }

    #[test]
    fn parse_interleaved_channels_from_transport() {
        assert_eq!(
            parse_interleaved_channels("RTP/AVP/TCP;unicast;interleaved=0-1"),
            Some((0, 1))
        );
        assert_eq!(
            parse_interleaved_channels("RTP/AVP/TCP;interleaved=2-3;unicast"),
            Some((2, 3))
        );
        assert_eq!(parse_interleaved_channels("RTP/AVP/UDP;unicast"), None);
    }

    #[test]
    fn generate_session_id_is_unique() {
        let id1 = generate_session_id();
        let id2 = generate_session_id();
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
    }

    #[test]
    fn options_response_contains_public_header() {
        let _registry = MountPointRegistry::new();
        let _config = Arc::new(RtspServerConfig::default());
        // We can't create ServerConnection without a real TcpStream in a unit test,
        // so test the helper directly.
        let mut headers = Headers::new();
        add_server_header(&mut headers);
        assert_eq!(headers.get("server"), Some(SERVER_NAME));
    }
}
