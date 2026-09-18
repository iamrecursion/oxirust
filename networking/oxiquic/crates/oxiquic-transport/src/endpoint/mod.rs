//! Asynchronous UDP shell driving the [`Connection`] state machine over
//! `tokio`'s [`tokio::net::UdpSocket`].
//!
//! The protocol logic lives in the connection state machine; this module is the
//! thin I/O layer that:
//!
//! * binds a UDP socket,
//! * for a client, constructs a [`Connection`] and runs the handshake to
//!   completion by shuttling datagrams,
//! * for a server, demultiplexes many concurrent client connections on a single
//!   UDP socket using DCID-based routing, completing each handshake
//!   independently before delivering a [`QuicConnection`] via [`ServerEndpoint::accept`],
//! * exposes a [`QuicConnection`] handle for opening streams and reading data.
//!
//! # Background-driven connections
//!
//! [`QuicConnection::into_driven`] consumes a [`QuicConnection`] and returns a
//! [`DrivenConnection`] that runs the socket I/O loop in a background
//! [`tokio::task`]. Streams on a [`DrivenConnection`] expose the standard
//! [`tokio::io::AsyncWrite`] / [`tokio::io::AsyncRead`] traits through
//! [`SendStreamHandle`] / [`RecvStreamHandle`].
//!
//! # Multi-connection server demux
//!
//! When `ServerEndpoint::accept` is first called a background
//! `run_server_demux` task is spawned. It reads every datagram from the shared
//! UDP socket and routes it to the correct per-connection channel keyed by the
//! destination connection ID:
//!
//! * **Initial packets**: keyed by the client's chosen `initial_dcid` in
//!   `initial_map`. On the first packet for an unknown DCID a new handshake task
//!   is spawned.
//! * **Short-header (1-RTT) packets**: keyed by the server's issued 8-byte
//!   `local_cid` in `local_cid_map`.
//!
//! Each handshake task notifies the demux once it has derived its `local_cid`
//! so the demux can promote the entry from `initial_map` to `local_cid_map`.
//!
//! That loop, the RFC 9000 §8.1 Retry branch and the routing-table garbage
//! collection live in the private `demux` submodule; this file keeps the
//! endpoint/connection handle types.

mod demux;
pub mod driven;
pub mod ecn_recv;
mod socket_opts;
pub mod zero_rtt;

pub use driven::DrivenConnection;
pub use zero_rtt::ZeroRttAccepted;

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::sleep_until;

use oxiquic_core::{OxiQuicError, StreamId};
use rustls::pki_types::ServerName;
use rustls::{ClientConfig, ServerConfig};

use crate::connection::cid::CidEvent;
use crate::connection::{Connection, DatagramMeta, MtuConfig, RetryTranscript, Role};
use crate::handle::{RecvStreamHandle, SendStreamHandle, WriteCmd};
use crate::packet::{encode_retry_packet, parse_initial_token};
use crate::TransportConfig;

use demux::{run_server_demux, CidRouteUpdate};
use driven::{run_driven_connection, DrivenConnectionChannels};

/// Maximum UDP datagram OxiQUIC will read in one recv.
pub(super) const RECV_BUF: usize = 2048;

/// Length (bytes) of connection IDs OxiQUIC issues locally.  Must match
/// `LOCAL_CID_LEN` in `connection.rs` (both are 8).
const LOCAL_CID_LEN: usize = 8;

/// The only QUIC version this endpoint speaks (QUIC v1, RFC 9000).
const QUIC_V1: u32 = 0x0000_0001;

/// Type alias for the open-stream request channel element.
pub(super) type OpenStreamSender = oneshot::Sender<(StreamId, mpsc::Receiver<Vec<u8>>)>;
/// Type alias for an optional open-stream receiver used in the driven loop.
pub(super) type OptOpenRx = Option<mpsc::Receiver<OpenStreamSender>>;

/// Supported versions list for Version Negotiation responses.
const SUPPORTED_VERSIONS: &[u32] = &[QUIC_V1];

// ─────────────────────────────────────────────────────────────────────────────
// InboundSource
// ─────────────────────────────────────────────────────────────────────────────

/// Where inbound datagrams come from for a given connection.
///
/// * `Socket` — the connection reads directly from its own UDP socket clone.
///   Used for client connections and the legacy single-connection server path.
/// * `Channel` — the demux task has already read the datagram from the shared
///   socket and forwarded it here. Used for server connections in multi-demux
///   mode.
pub(super) enum InboundSource {
    /// Direct UDP socket receive.
    Socket {
        /// The socket to read from.
        socket: Arc<UdpSocket>,
        /// Whether the kernel accepted `IP_RECVTOS` / `IPV6_RECVTCLASS` on this
        /// socket, i.e. whether the inbound ECN codepoint can be read at all
        /// (RFC 9000 §13.4.1). When `false` the read path is a plain
        /// `recv_from` and no codepoint is ever reported.
        ecn: bool,
    },
    /// Pre-read datagrams forwarded from the demux task.
    Channel(mpsc::Receiver<InboundDatagram>),
}

/// One datagram as handed to the connection state machine, together with the
/// per-datagram metadata the I/O layer observed.
///
/// The ECN codepoint travels with the payload because only the socket read that
/// produced the datagram can report it; `None` means "not observed", never
/// Not-ECT (see [`ecn_recv`]).
#[derive(Debug)]
pub(super) struct InboundDatagram {
    /// The datagram payload.
    pub(super) data: Vec<u8>,
    /// Source address the datagram arrived from.
    pub(super) from: SocketAddr,
    /// IP ECN codepoint of the datagram, if the platform reported one.
    pub(super) ecn: Option<crate::ecn::EcnCodepoint>,
}

impl InboundDatagram {
    /// The [`DatagramMeta`] this datagram should be processed with.
    pub(super) fn meta(&self) -> DatagramMeta {
        match self.ecn {
            Some(cp) => DatagramMeta::with_ecn(self.from, cp),
            None => DatagramMeta::from_peer(self.from),
        }
    }
}

/// Receive one datagram from `inbound`, writing the payload into `buf` and
/// returning it with its source address and observed ECN codepoint.
///
/// Abstracting this out of `pump_once` avoids borrow-checker fights where the
/// async generator would require holding mutable references to both `inbound`
/// and other fields of `ConnectionDriver` simultaneously.
pub(super) async fn recv_inbound(
    inbound: &mut InboundSource,
    buf: &mut [u8],
) -> io::Result<InboundDatagram> {
    match inbound {
        InboundSource::Socket { socket, ecn } => {
            let sock = Arc::clone(socket);
            if *ecn {
                let (len, from, cp) = ecn_recv::recv_ecn_from(&sock, buf).await?;
                Ok(InboundDatagram {
                    data: buf[..len].to_vec(),
                    from,
                    ecn: cp,
                })
            } else {
                let (len, from) = sock.recv_from(buf).await?;
                Ok(InboundDatagram {
                    data: buf[..len].to_vec(),
                    from,
                    ecn: None,
                })
            }
        }
        InboundSource::Channel(rx) => rx
            .recv()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "channel closed")),
    }
}

/// Non-blocking attempt to receive one datagram from `inbound`.  Returns
/// `None` immediately when no datagram is waiting (rather than awaiting one).
/// Used to drain a burst of already-queued datagrams without yielding back to
/// the tokio scheduler between each one — which would round-trip through the
/// timer/wakeup infrastructure and add milliseconds of latency per datagram on
/// lightly-loaded systems.
fn try_recv_inbound(
    inbound: &mut InboundSource,
    buf: &mut [u8],
) -> Option<io::Result<InboundDatagram>> {
    match inbound {
        InboundSource::Socket { socket, ecn } => {
            let read = if *ecn {
                ecn_recv::try_recv_ecn_from(socket, buf)
            } else {
                socket
                    .try_recv_from(buf)
                    .map(|(len, from)| (len, from, None))
            };
            match read {
                Ok((len, from, cp)) => Some(Ok(InboundDatagram {
                    data: buf[..len].to_vec(),
                    from,
                    ecn: cp,
                })),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => None,
                Err(e) => Some(Err(e)),
            }
        }
        InboundSource::Channel(rx) => match rx.try_recv() {
            Ok(item) => Some(Ok(item)),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => None,
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => Some(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "channel closed",
            ))),
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClientEndpoint
// ─────────────────────────────────────────────────────────────────────────────

/// A bound QUIC client endpoint that can establish outgoing connections.
pub struct ClientEndpoint {
    socket: Arc<UdpSocket>,
    config: Arc<ClientConfig>,
    transport: TransportConfig,
    /// Whether this socket reports the inbound ECN codepoint (RFC 9000
    /// §13.4.1). `false` on platforms/kernels that refuse `IP_RECVTOS` /
    /// `IPV6_RECVTCLASS`; the connection then counts nothing and sends plain
    /// ACKs rather than inventing codepoints.
    ecn_recv: bool,
}

impl ClientEndpoint {
    /// Bind a client endpoint to `bind_addr` with the given rustls client
    /// configuration (which must be built from `oxiquic_crypto::quic_crypto_provider`).
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Io`] if the UDP socket cannot be bound.
    pub async fn bind(
        bind_addr: SocketAddr,
        config: Arc<ClientConfig>,
        transport: TransportConfig,
    ) -> Result<Self, OxiQuicError> {
        let socket = UdpSocket::bind(bind_addr).await?;
        // RFC 9000 §13.4.1: ask the kernel to report the ECN codepoint of every
        // inbound datagram. A refusal is not fatal — ECN feedback is simply not
        // produced (and never fabricated).
        let ecn_recv = ecn_recv::enable_ecn_recv(&socket).is_ok();
        Ok(Self {
            socket: Arc::new(socket),
            config,
            transport,
            ecn_recv,
        })
    }

    /// Whether this endpoint's socket reports inbound ECN codepoints
    /// (RFC 9000 §13.4.1).
    ///
    /// `false` means the platform or kernel refused `IP_RECVTOS` /
    /// `IPV6_RECVTCLASS`; connections from this endpoint then send plain ACK
    /// frames instead of ACK-ECN, and the peer's ECN validation correctly
    /// concludes that ECN is not usable on the path.
    #[must_use]
    pub fn reports_inbound_ecn(&self) -> bool {
        self.ecn_recv
    }

    /// The local address the endpoint is bound to.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Io`] if the address cannot be read.
    pub fn local_addr(&self) -> Result<SocketAddr, OxiQuicError> {
        Ok(self.socket.local_addr()?)
    }

    /// Connect to `server_addr` with a configurable handshake timeout.
    ///
    /// Wraps [`Self::connect`] with [`tokio::time::timeout`].  If the
    /// handshake does not complete within `timeout`, the future is dropped and
    /// [`OxiQuicError::Timeout`] is returned.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Timeout`] when the deadline elapses, or any
    /// error that [`Self::connect`] would return (bind failure, TLS failure, …).
    pub async fn connect_timeout(
        &self,
        addr: SocketAddr,
        server_name: &str,
        timeout: std::time::Duration,
    ) -> Result<QuicConnection, OxiQuicError> {
        tokio::time::timeout(timeout, self.connect(addr, server_name))
            .await
            .map_err(|_| OxiQuicError::Timeout)?
    }

    /// Connect to `server_addr`, validating its certificate against the
    /// configured roots and the supplied `server_name`. Drives the handshake to
    /// completion before returning the established connection.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] if binding, the handshake or the TLS
    /// negotiation fails, or the handshake times out.
    pub async fn connect(
        &self,
        server_addr: SocketAddr,
        server_name: &str,
    ) -> Result<QuicConnection, OxiQuicError> {
        let name = ServerName::try_from(server_name.to_string())
            .map_err(|_| OxiQuicError::Tls(format!("invalid server name {server_name}")))?;
        let params = self.transport.to_transport_params();
        let mtu_config = MtuConfig {
            max_mtu: self.transport.get_max_mtu(),
            discovery_enabled: true,
        };
        let conn = Connection::new_client_with_datagram_buf(
            self.config.clone(),
            name,
            server_addr,
            params,
            mtu_config,
            self.transport.get_congestion_controller(),
            self.transport.get_datagram_receive_buffer_size(),
        )?;
        let inbound = InboundSource::Socket {
            socket: Arc::clone(&self.socket),
            ecn: self.ecn_recv,
        };
        let mut driver =
            ConnectionDriver::new(Arc::clone(&self.socket), inbound, conn, Some(server_addr));
        driver.run_handshake().await?;
        driver
            .conn
            .set_keep_alive_interval(self.transport.get_keep_alive_interval());
        Ok(QuicConnection::new(driver))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ServerEndpoint
// ─────────────────────────────────────────────────────────────────────────────

/// A bound QUIC server endpoint that accepts many concurrent incoming connections
/// on a single UDP socket via DCID-based demultiplexing.
pub struct ServerEndpoint {
    socket: Arc<UdpSocket>,
    config: Arc<ServerConfig>,
    transport: TransportConfig,
    /// Lazily-initialised demux state. Populated on the first call to `accept`.
    demux: Mutex<Option<ServerDemuxState>>,
    /// Whether this socket reports the inbound ECN codepoint (RFC 9000
    /// §13.4.1); see [`ClientEndpoint::reports_inbound_ecn`].
    ecn_recv: bool,
}

impl ServerEndpoint {
    /// Bind a server endpoint to `bind_addr` with the given rustls server
    /// configuration (built from `oxiquic_crypto::quic_crypto_provider`).
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Io`] if the UDP socket cannot be bound.
    pub async fn bind(
        bind_addr: SocketAddr,
        config: Arc<ServerConfig>,
        transport: TransportConfig,
    ) -> Result<Self, OxiQuicError> {
        let socket = UdpSocket::bind(bind_addr).await?;
        // RFC 9000 §13.4.1 receive-side ECN; see `ClientEndpoint::bind`.
        let ecn_recv = ecn_recv::enable_ecn_recv(&socket).is_ok();
        Ok(Self {
            socket: Arc::new(socket),
            config,
            transport,
            demux: Mutex::new(None),
            ecn_recv,
        })
    }

    /// Whether this endpoint's socket reports inbound ECN codepoints
    /// (RFC 9000 §13.4.1); see [`ClientEndpoint::reports_inbound_ecn`].
    #[must_use]
    pub fn reports_inbound_ecn(&self) -> bool {
        self.ecn_recv
    }

    /// The local address the endpoint is bound to.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Io`] if the address cannot be read.
    pub fn local_addr(&self) -> Result<SocketAddr, OxiQuicError> {
        Ok(self.socket.local_addr()?)
    }

    /// Accept the next incoming connection.
    ///
    /// On the first call a background demux task is spawned that reads all
    /// datagrams from the shared UDP socket and routes them to per-connection
    /// channel-based inbound sources.  Subsequent calls simply await the next
    /// fully-established connection from the accept channel.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] if the demux accept channel is closed or a
    /// connection-level error is forwarded from the handshake task.
    pub async fn accept(&self) -> Result<QuicConnection, OxiQuicError> {
        let mut guard = self.demux.lock().await;
        if guard.is_none() {
            let state = Self::start_demux(
                Arc::clone(&self.socket),
                Arc::clone(&self.config),
                self.transport.clone(),
                self.ecn_recv,
            );
            *guard = Some(state);
        }
        let accept_rx = &mut guard
            .as_mut()
            .ok_or_else(|| OxiQuicError::Connection("demux not initialised".into()))?
            .accept_rx;
        accept_rx
            .recv()
            .await
            .ok_or_else(|| OxiQuicError::Connection("server accept channel closed".into()))?
    }

    /// Return an [`Incoming`] iterator that wraps repeated calls to
    /// [`Self::accept`].
    ///
    /// Use `.next().await` in a loop to accept connections without needing to
    /// hold an `async` block across the loop:
    ///
    /// ```rust,ignore
    /// let mut incoming = server.incoming();
    /// while let Some(conn) = incoming.next().await {
    ///     tokio::spawn(async move { /* handle conn */ });
    /// }
    /// ```
    pub fn incoming(&self) -> Incoming<'_> {
        Incoming { endpoint: self }
    }

    /// Spawn the background demux task and return the state record that holds
    /// its accept channel and join handle.
    fn start_demux(
        socket: Arc<UdpSocket>,
        config: Arc<ServerConfig>,
        transport: TransportConfig,
        ecn_recv: bool,
    ) -> ServerDemuxState {
        let (accept_tx, accept_rx) = mpsc::channel::<Result<QuicConnection, OxiQuicError>>(16);
        let task_handle = tokio::spawn(run_server_demux(
            socket, config, transport, accept_tx, ecn_recv,
        ));
        ServerDemuxState {
            accept_rx,
            _task_handle: task_handle,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ServerEndpointBuilder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`ServerEndpoint`] with optional session ticketer configuration.
///
/// Use this instead of [`ServerEndpoint::bind`] when you need to plug in a
/// custom [`rustls::server::ProducesTickets`] implementation (such as
/// `oxitls::OxiTicketer`) for QUIC 0-RTT session resumption.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use oxiquic_transport::{ServerEndpointBuilder, TransportConfig};
/// use oxitls::OxiTicketer;
///
/// let server = ServerEndpointBuilder::new("127.0.0.1:4433".parse()?, Arc::new(server_tls), TransportConfig::default())
///     .with_ticketer(Arc::new(OxiTicketer::new().expect("ticketer")))
///     .build()
///     .await?;
/// ```
pub struct ServerEndpointBuilder {
    bind_addr: SocketAddr,
    config: Arc<ServerConfig>,
    transport: TransportConfig,
    /// Optional custom session ticket provider for TLS session resumption and 0-RTT.
    ticketer: Option<Arc<dyn rustls::server::ProducesTickets>>,
}

impl ServerEndpointBuilder {
    /// Create a new builder bound to `bind_addr` with the given TLS and transport
    /// configuration.
    ///
    /// Call [`with_ticketer`][Self::with_ticketer] to override the session
    /// ticket provider before calling [`build`][Self::build].
    #[must_use]
    pub fn new(
        bind_addr: SocketAddr,
        config: Arc<ServerConfig>,
        transport: TransportConfig,
    ) -> Self {
        Self {
            bind_addr,
            config,
            transport,
            ticketer: None,
        }
    }

    /// Set a custom session ticket provider for TLS session resumption and 0-RTT.
    ///
    /// The ticketer is applied to the [`rustls::ServerConfig`] before binding the
    /// endpoint. Use `oxitls::OxiTicketer` for a pure-Rust AES-GCM-backed ticketer:
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use oxitls::OxiTicketer;
    /// builder.with_ticketer(Arc::new(OxiTicketer::new().expect("ticketer")));
    /// ```
    #[must_use]
    pub fn with_ticketer(mut self, ticketer: Arc<dyn rustls::server::ProducesTickets>) -> Self {
        self.ticketer = Some(ticketer);
        self
    }

    /// Set the ALPN protocol identifiers advertised in the TLS handshake.
    ///
    /// Replaces `alpn_protocols` on the underlying [`rustls::ServerConfig`]
    /// immediately. Call before [`build`][Self::build] to negotiate custom
    /// protocols on raw QUIC server endpoints.
    ///
    /// For HTTP/3 servers, prefer `H3ServerBuilder::with_tls_config` which
    /// automatically injects `b"h3"`. See [`oxiquic_core::alpn`] for well-known
    /// protocol constants.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use oxiquic_transport::{ServerEndpointBuilder, TransportConfig};
    ///
    /// let server = ServerEndpointBuilder::new(addr, config, TransportConfig::default())
    ///     .with_alpn_protocols(&[b"my-proto/1.0"])
    ///     .build()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn with_alpn_protocols(mut self, protocols: &[&[u8]]) -> Self {
        let mut cfg = (*self.config).clone();
        cfg.alpn_protocols = protocols.iter().map(|p| p.to_vec()).collect();
        self.config = Arc::new(cfg);
        self
    }

    /// Bind the [`ServerEndpoint`], applying the ticketer (if set) to the TLS
    /// configuration before binding the UDP socket.
    ///
    /// # Errors
    ///
    /// Returns [`OxiQuicError::Io`] if the UDP socket cannot be bound.
    pub async fn build(self) -> Result<ServerEndpoint, OxiQuicError> {
        let config = if let Some(ticketer) = self.ticketer {
            let mut cfg = (*self.config).clone();
            cfg.ticketer = ticketer;
            Arc::new(cfg)
        } else {
            self.config
        };
        ServerEndpoint::bind(self.bind_addr, config, self.transport).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ServerDemuxState
// ─────────────────────────────────────────────────────────────────────────────

/// Holds the background demux task handle and the accept channel receiver.
struct ServerDemuxState {
    accept_rx: mpsc::Receiver<Result<QuicConnection, OxiQuicError>>,
    /// Keeps the background task alive. Dropped when `ServerDemuxState` is
    /// dropped, which aborts the task if it has not already finished.
    _task_handle: tokio::task::JoinHandle<()>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Incoming — async iterator over accepted QuicConnections
// ─────────────────────────────────────────────────────────────────────────────

/// An async iterator over incoming QUIC connections.
///
/// Created by [`ServerEndpoint::incoming`].  Call `.next().await` to accept
/// connections one at a time without spawning a background task or requiring
/// the `Stream` trait.
///
/// # Lifetime
/// `Incoming` borrows the [`ServerEndpoint`] for its lifetime, so the endpoint
/// must outlive all `next()` calls.
pub struct Incoming<'a> {
    endpoint: &'a ServerEndpoint,
}

impl<'a> Incoming<'a> {
    /// Wait for the next established incoming connection.
    ///
    /// Returns `None` only when the server's accept channel has been
    /// permanently closed (i.e. the background demux task has exited), which
    /// typically means the [`ServerEndpoint`] is being torn down.
    pub async fn next(&self) -> Option<QuicConnection> {
        self.endpoint.accept().await.ok()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionDriver
// ─────────────────────────────────────────────────────────────────────────────

/// Owns the socket (for sends) + an inbound source + connection state and
/// shuttles datagrams between them.
struct ConnectionDriver {
    /// Used exclusively for sending outgoing datagrams.
    socket: Arc<UdpSocket>,
    /// The source of inbound datagrams (direct socket read or channel).
    inbound: InboundSource,
    conn: Connection,
    peer: Option<SocketAddr>,
    recv: Vec<u8>,
    /// Optional channel to the demux for CID routing table updates.
    /// `None` for client connections (no demux), `Some` for server connections.
    cid_route_tx: Option<mpsc::Sender<CidRouteUpdate>>,
    /// The inbound sender for this connection (needed when forwarding CID
    /// routing updates to the demux so it can register/unregister routes).
    self_tx: Option<mpsc::Sender<InboundDatagram>>,
    /// The ECN codepoint most recently applied to the socket, to avoid a
    /// redundant `setsockopt` on every flush (RFC 9000 §13.4).
    last_ecn: Option<crate::ecn::EcnCodepoint>,
}

impl ConnectionDriver {
    fn new(
        socket: Arc<UdpSocket>,
        inbound: InboundSource,
        conn: Connection,
        peer: Option<SocketAddr>,
    ) -> Self {
        Self {
            socket,
            inbound,
            conn,
            peer,
            recv: vec![0u8; RECV_BUF],
            cid_route_tx: None,
            self_tx: None,
            last_ecn: None,
        }
    }

    /// Decompose this driver into the parts needed by [`run_driven_connection`].
    ///
    /// The remaining fields (`recv`, `cid_route_tx`, `self_tx`) are dropped.
    /// This is the safe alternative to struct destructuring when a `Drop` impl
    /// exists on a containing type ([`QuicConnection`]).
    fn into_parts(
        self,
    ) -> (
        Arc<UdpSocket>,
        InboundSource,
        Connection,
        Option<SocketAddr>,
    ) {
        (self.socket, self.inbound, self.conn, self.peer)
    }

    /// Attach a CID routing channel so this driver can notify the demux when
    /// new connection IDs are issued or retired.
    fn with_cid_routing(
        mut self,
        cid_route_tx: mpsc::Sender<CidRouteUpdate>,
        self_tx: mpsc::Sender<InboundDatagram>,
    ) -> Self {
        self.cid_route_tx = Some(cid_route_tx);
        self.self_tx = Some(self_tx);
        self
    }

    /// Drain CID events from the connection and forward them to the demux.
    fn drain_cid_events(&mut self) {
        if let (Some(tx), Some(self_tx)) = (&self.cid_route_tx, &self.self_tx) {
            while let Some(event) = self.conn.pop_cid_event() {
                let update = CidRouteUpdate {
                    event,
                    conn_tx: self_tx.clone(),
                };
                // best-effort: if channel is full / closed, drop the event.
                let _ = tx.try_send(update);
            }
        } else {
            // No demux routing: drain events to prevent accumulation.
            while self.conn.pop_cid_event().is_some() {}
        }
    }

    /// Flush every datagram the connection currently wants to send.
    async fn flush(&mut self) -> Result<(), OxiQuicError> {
        loop {
            let mut out = Vec::new();
            let now = Instant::now();
            match self.conn.poll_transmit(now, &mut out) {
                Some(addr) if !out.is_empty() => {
                    // RFC 9000 §13.4: stamp the connection's desired ECN
                    // codepoint on the socket before sending.
                    socket_opts::sync_ecn(&self.socket, &mut self.conn, &mut self.last_ecn);
                    self.socket.send_to(&out, addr).await?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    /// Run the handshake to completion (or close/timeout).
    async fn run_handshake(&mut self) -> Result<(), OxiQuicError> {
        // Bound the handshake so a lost peer cannot hang the test forever.
        let deadline = Instant::now() + Duration::from_secs(10);
        self.flush().await?;
        while self.conn.is_handshaking() {
            if self.conn.is_closed() {
                return Err(self.close_error());
            }
            self.pump_once(deadline).await?;
        }
        // One more flush to emit the client's final handshake / HANDSHAKE_DONE.
        self.flush().await?;
        Ok(())
    }

    /// Receive one datagram (or time out), feed it to the connection, and flush
    /// any resulting output.
    /// Receive one datagram (or time out), feed it to the connection, and flush
    /// any resulting output.
    ///
    /// After the first blocking receive, any additional datagrams that are
    /// immediately available in the socket or channel buffer are drained in a
    /// tight non-blocking loop (up to [`BURST_DRAIN_LIMIT`] extra datagrams)
    /// before flushing. This coalesces ACK processing for a burst of packets —
    /// which is critical for throughput during slow-start: without batching, each
    /// `pump_once` call grows the congestion window by only one packet worth of
    /// ACKs and then yields back to the tokio scheduler (incurring scheduler
    /// wakeup latency, ~1–15 ms on macOS). With batching, a burst of N ACKs is
    /// processed in one call, growing the window by N × max_datagram at once.
    async fn pump_once(&mut self, deadline: Instant) -> Result<(), OxiQuicError> {
        /// Maximum number of additional (non-blocking) datagrams drained per
        /// `pump_once` call after the first blocking receive.  Caps the
        /// single-call work to a bounded amount, ensuring other async tasks still
        /// get scheduled.  128 was chosen to keep large-payload throughput high
        /// (each call processes up to 129 ACKs, growing the CUBIC window fast)
        /// while still yielding to the tokio scheduler regularly.
        const BURST_DRAIN_LIMIT: usize = 128;

        let timeout = self.conn.next_timeout().unwrap_or(deadline).min(deadline);
        tokio::select! {
            res = recv_inbound(&mut self.inbound, &mut self.recv) => {
                let datagram = res?;
                self.handle_inbound_datagram(datagram)?;
                // Drain additional immediately-available datagrams.  This
                // prevents per-datagram scheduler round-trips during ACK bursts.
                for _ in 0..BURST_DRAIN_LIMIT {
                    match try_recv_inbound(&mut self.inbound, &mut self.recv) {
                        Some(Ok(extra)) => self.handle_inbound_datagram(extra)?,
                        Some(Err(e)) => return Err(OxiQuicError::Io(e)),
                        None => break,
                    }
                }
                self.flush().await?;
            }
            () = sleep_until(timeout.into()) => {
                let now = Instant::now();
                self.conn.handle_timeout(now);
                if now >= deadline && self.conn.is_handshaking() {
                    return Err(OxiQuicError::Timeout);
                }
                self.flush().await?;
            }
        }
        Ok(())
    }

    /// Process a single inbound datagram: update peer-address state, feed the
    /// datagram to the connection state machine, and adopt any newly-validated
    /// path address.  Extracted from `pump_once` so the burst-drain loop can
    /// reuse it without duplicating the address-migration logic.
    fn handle_inbound_datagram(&mut self, datagram: InboundDatagram) -> Result<(), OxiQuicError> {
        let from = datagram.from;
        let meta = datagram.meta();
        if self.peer.is_none() {
            self.peer = Some(from);
        } else if self.peer != Some(from)
            && self.conn.is_established()
            && self.conn.candidate_peer_addr() != Some(from)
        {
            // Address change on an established connection: RFC 9000 §9.3
            // — register the candidate and kick off path validation.
            // NOTE: We trigger on any new source address, including
            // probing packets. Full RFC compliance would require
            // restricting to non-probing frames only (§9.3.1).
            //
            // Only the *first* datagram from the address starts a validation:
            // every later one keeps arriving from the same unfamiliar source
            // until the path is validated, and restarting the probe on each
            // would reset the RFC 9000 §8.2 retransmission backoff and abandon
            // deadline on every packet, so neither could ever fire.
            self.conn.set_candidate_peer_addr(from);
            let _ = self.conn.initiate_path_challenge();
        }
        let mut data = datagram.data;
        let now = Instant::now();
        // Report the datagram's true source address so the RFC 9000 §9.3
        // per-path anti-amplification allowance is credited to the path it
        // arrived on, and its observed ECN codepoint (RFC 9000 §13.4.1) so the
        // packets decrypted out of it are counted into their space's ECT/CE
        // tallies and echoed in ACK-ECN frames.
        self.conn.handle_datagram_with_meta(now, &mut data, meta)?;
        // After processing: if the path was just validated, adopt the
        // new peer address at the driver level too.
        if self.conn.path_validated() {
            self.peer = Some(self.conn.peer_addr());
        }
        // Drain CID routing events and forward them to the demux.
        self.drain_cid_events();
        Ok(())
    }

    fn close_error(&self) -> OxiQuicError {
        // `OxiQuicError` is not `Clone` (it wraps `io::Error`), so reconstruct
        // the error with full fidelity (preserving transport error codes and
        // application close codes) rather than flattening to a plain string.
        match self.conn.peer_close_reason() {
            Some(OxiQuicError::TransportError {
                code,
                frame_type,
                reason,
            }) => OxiQuicError::TransportError {
                code: *code,
                frame_type: *frame_type,
                reason: reason.clone(),
            },
            Some(OxiQuicError::ApplicationClose { code, reason }) => {
                OxiQuicError::ApplicationClose {
                    code: *code,
                    reason: reason.clone(),
                }
            }
            Some(other) => OxiQuicError::Connection(other.to_string()),
            None => OxiQuicError::Connection("connection closed".into()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuicConnection
// ─────────────────────────────────────────────────────────────────────────────

/// An established QUIC connection handle.
///
/// Exposes the stream API and a receive pump. Streams are opened with
/// [`QuicConnection::open_bidi`], data is queued with
/// [`QuicConnection::send`], and inbound data is read with
/// [`QuicConnection::read`] / [`QuicConnection::accept_uni_or_bidi_data`].
pub struct QuicConnection {
    /// Wrapped in `Option` so the `Drop` impl and `into_driven` can take
    /// ownership without needing struct destructuring (which is blocked on
    /// types that implement `Drop`).
    driver: Option<ConnectionDriver>,
}

impl QuicConnection {
    fn new(driver: ConnectionDriver) -> Self {
        Self {
            driver: Some(driver),
        }
    }

    /// Return a reference to the inner driver.
    ///
    /// Panics only if called after `into_driven` (not possible through the
    /// public API since `into_driven` consumes `self`).
    fn drv(&self) -> &ConnectionDriver {
        self.driver
            .as_ref()
            .expect("QuicConnection driver already consumed by into_driven")
    }

    /// Return a mutable reference to the inner driver.
    fn drv_mut(&mut self) -> &mut ConnectionDriver {
        self.driver
            .as_mut()
            .expect("QuicConnection driver already consumed by into_driven")
    }

    /// The role (client or server) of this endpoint.
    #[must_use]
    pub fn role(&self) -> Role {
        self.drv().conn.role()
    }

    /// The remote peer address for this connection.
    ///
    /// Returns `None` only if the handshake completed before the driver had a
    /// chance to record the peer address, which does not occur in normal
    /// operation. After a successful handshake this is guaranteed to be `Some`.
    ///
    /// For a server-side [`QuicConnection`] obtained from
    /// [`ServerEndpoint::accept`], this is the client's UDP source address. For
    /// a client-side connection this is the server address passed to
    /// [`ClientEndpoint::connect`].
    #[must_use]
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.drv().peer
    }

    /// Whether the connection has fully closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.drv().conn.is_closed()
    }

    /// The peer's transport parameters, available post-handshake.
    #[must_use]
    pub fn peer_transport_params(&self) -> Option<&oxiquic_core::TransportParams> {
        self.drv().conn.peer_transport_params()
    }

    /// Number of Retry packets this client accepted (always 0 for server-side
    /// connections). Exposed primarily for test observability — a value of 1
    /// confirms the Retry round-trip completed successfully.
    #[must_use]
    pub fn retry_count(&self) -> u64 {
        self.drv().conn.retry_count()
    }

    /// Initiate a QUIC key update on the next outgoing 1-RTT packet
    /// (RFC 9001 §6).
    ///
    /// Returns `true` if the key update was accepted (will take effect on the
    /// next send), or `false` if:
    /// * The handshake has not yet completed (no 1-RTT keys).
    /// * A key update was performed recently and the 3-PTO cooldown has not
    ///   elapsed (RFC 9001 §6.5).
    pub fn initiate_key_update(&mut self) -> bool {
        self.drv_mut().conn.initiate_key_update(Instant::now())
    }

    /// Number of completed key updates (locally- and peer-initiated) so far.
    /// Useful for test observability.
    #[must_use]
    pub fn key_update_count(&self) -> u64 {
        self.drv().conn.key_update_count()
    }

    /// Begin a path challenge toward the current peer (RFC 9000 §9).
    ///
    /// Queues an 8-byte `PATH_CHALLENGE` frame for the next outgoing 1-RTT
    /// packet.  Call [`Self::path_validated`] after the next send/receive
    /// cycle to check whether the peer echoed it back.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Connection`] if 1-RTT keys are not yet available.
    pub fn initiate_path_challenge(&mut self) -> Result<(), OxiQuicError> {
        self.drv_mut().conn.initiate_path_challenge()
    }

    /// Whether the most recent locally-initiated path challenge was answered
    /// by the peer (RFC 9000 §9.3).
    #[must_use]
    pub fn path_validated(&self) -> bool {
        self.drv().conn.path_validated()
    }

    /// The current confirmed path MTU in bytes (starts at 1200; advances as
    /// DPLPMTUD probes succeed).
    #[must_use]
    pub fn current_mtu(&self) -> u16 {
        self.drv().conn.current_mtu()
    }

    /// ECN validation state of the Application packet-number space
    /// (RFC 9000 §13.4.2).
    ///
    /// Reaches [`crate::ecn::EcnValidationState::Capable`] only once the peer
    /// has echoed ECN counts covering this endpoint's ECT(0)-marked packets, so
    /// it is also the observable proof that both halves of ECN — marking and
    /// [receiving](ecn_recv) — are working on this path.
    #[must_use]
    pub fn ecn_state(&self) -> crate::ecn::EcnValidationState {
        self.drv().conn.ecn_state()
    }

    /// The ECN codepoint counters this endpoint has *received* in the packet-
    /// number space `packet_type` belongs to (RFC 9000 §13.4.1) — exactly the
    /// values echoed to the peer in ACK-ECN frames.
    ///
    /// All-zero when the platform cannot report inbound codepoints (see
    /// [`ClientEndpoint::reports_inbound_ecn`]); a codepoint is never assumed.
    #[must_use]
    pub fn ecn_recv_counts(&self, packet_type: oxiquic_core::PacketType) -> crate::ecn::EcnCounts {
        self.drv().conn.ecn_recv_counts(packet_type)
    }

    /// Whether any send stream still has buffered data or an unsent FIN.
    ///
    /// Use together with [`bytes_in_flight`] to know when it is safe to drop
    /// the connection: when `has_pending_stream_data()` is `false` **and**
    /// `bytes_in_flight()` is `0`, every byte and the FIN have been
    /// transmitted and the peer's ACK has been processed.
    ///
    /// [`bytes_in_flight`]: Self::bytes_in_flight
    #[must_use]
    pub fn has_pending_stream_data(&self) -> bool {
        self.drv().conn.has_pending_stream_data()
    }

    /// The bytes currently in flight (ack-eliciting, unacknowledged).
    #[must_use]
    pub fn bytes_in_flight(&self) -> u64 {
        self.drv().conn.bytes_in_flight()
    }

    /// The MTU size of any in-flight probe, or `None` when no probe is pending.
    #[must_use]
    pub fn probe_mtu(&self) -> Option<u16> {
        self.drv().conn.probe_mtu()
    }

    /// Returns the ALPN protocol negotiated during the TLS handshake, if any.
    ///
    /// For HTTP/3, this should be `Some(b"h3".to_vec())` after a successful
    /// connection to an HTTP/3 server.
    #[must_use]
    pub fn negotiated_alpn(&self) -> Option<Vec<u8>> {
        self.drv().conn.negotiated_alpn()
    }

    /// Open a new bidirectional stream, returning its id.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Stream`] if the peer's stream limit has been
    /// reached (RFC 9000 §4.6). A `STREAMS_BLOCKED` frame is queued automatically.
    pub fn open_bidi(&mut self) -> Result<StreamId, OxiQuicError> {
        self.drv_mut().conn.open_bidi()
    }

    /// Open a bidirectional stream with an associated priority hint.
    ///
    /// The `priority` value is recorded in the returned stream ID (as a hint for
    /// future scheduler support) but does **not** currently affect packet ordering
    /// or scheduling — all streams share the same transmit queue.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Stream`] if the peer's stream limit has been
    /// reached.
    pub fn open_bidi_with_priority(&mut self, _priority: i32) -> Result<StreamId, OxiQuicError> {
        // Priority is stored as a hint; the current scheduler does not reorder
        // streams by priority. Future work: prioritised stream scheduling.
        self.drv_mut().conn.open_bidi()
    }

    /// Open a bidirectional stream with automatic retry on transient
    /// stream-limit errors (RFC 9000 §4.6 `STREAMS_BLOCKED` back-pressure).
    ///
    /// If the peer's concurrent-stream limit has been reached, this method
    /// retries up to `max_attempts` times, waiting `retry_delay` between each
    /// attempt.  A `STREAMS_BLOCKED` frame is emitted by the transport on each
    /// failed attempt, signalling the peer to raise its `MAX_STREAMS` limit.
    ///
    /// # Errors
    /// Returns [`OxiQuicError::Stream`] if all attempts are exhausted without
    /// success (the peer never raised its limit), or any non-transient error
    /// from the underlying connection.
    pub async fn open_bi_reliable(
        &mut self,
        max_attempts: u32,
        retry_delay: Duration,
    ) -> Result<StreamId, OxiQuicError> {
        let attempts = max_attempts.max(1);
        for attempt in 0..attempts {
            match self.drv_mut().conn.open_bidi() {
                Ok(sid) => return Ok(sid),
                Err(OxiQuicError::Stream(_)) if attempt + 1 < attempts => {
                    // Transient stream-limit: flush any STREAMS_BLOCKED frame
                    // we just queued and wait for the peer to raise its limit.
                    let _ = self.drv_mut().flush().await;
                    tokio::time::sleep(retry_delay).await;
                }
                Err(e) => return Err(e),
            }
        }
        // Exhausted all attempts — return the last error.
        self.drv_mut().conn.open_bidi()
    }

    /// The number of bidirectional streams opened on this connection since
    /// establishment, including streams opened by the local endpoint.
    ///
    /// Derived from the connection-level `streams_opened` counter maintained by
    /// the protocol state machine.
    #[must_use]
    pub fn streams_opened(&self) -> u64 {
        self.drv().conn.stats().streams_opened
    }

    /// The number of streams that have been fully closed.
    ///
    /// Currently returns `0` — the underlying counter is tracked but close
    /// events are not yet plumbed from the stream state machine to the stats
    /// snapshot.  This will be non-zero in a future release.
    #[must_use]
    pub fn streams_closed(&self) -> u64 {
        self.drv().conn.stats().streams_closed
    }

    /// The current smoothed round-trip time estimate for this connection.
    ///
    /// Sourced from the congestion controller's RTT estimator
    /// (RFC 9002 Section 5.3).  The value is updated after each ACK round trip.
    /// Before the first ACK is processed the returned duration is
    /// [`Duration::ZERO`].
    ///
    /// This is the *smoothed* RTT (`srtt`), not the latest single sample.
    /// For the latest sample use [`Self::stats`]`.rtt`.
    #[must_use]
    pub fn ping(&self) -> Duration {
        self.drv().conn.stats().smoothed_rtt
    }

    /// Queue `data` on `stream`, optionally finishing it, then flush.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] if the stream is unknown or sending fails.
    pub async fn send(
        &mut self,
        stream: StreamId,
        data: &[u8],
        fin: bool,
    ) -> Result<(), OxiQuicError> {
        self.drv_mut().conn.send_stream(stream, data, fin)?;
        self.drv_mut().flush().await
    }

    /// Wait until `stream` has at least one byte of in-order data (or is
    /// finished), returning `(bytes, fin)`. Pumps the socket until data
    /// arrives or the idle timeout fires.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] on I/O failure, connection close or timeout.
    pub async fn read(&mut self, stream: StreamId) -> Result<(Vec<u8>, bool), OxiQuicError> {
        let deadline = Instant::now() + Duration::from_secs(10);
        self.read_with_deadline(stream, deadline).await
    }

    /// Like [`QuicConnection::read`] but with a caller-supplied absolute
    /// deadline. Pumps the socket until data arrives on `stream` or `deadline`
    /// is reached.
    ///
    /// ACKs are flushed before returning data to the caller so that the sender
    /// is not starved of acknowledgements while the application processes a
    /// burst of already-buffered packets. Without this flush, the ACK for a
    /// received batch would be deferred until the read buffer empties and the
    /// next `pump_once` call fires — creating a bursty ACK pattern that keeps
    /// the sender's congestion window artificially small.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] on I/O failure, connection close or timeout.
    pub async fn read_with_deadline(
        &mut self,
        stream: StreamId,
        deadline: Instant,
    ) -> Result<(Vec<u8>, bool), OxiQuicError> {
        loop {
            let (bytes, fin) = self.drv_mut().conn.read_stream(stream)?;
            if !bytes.is_empty() || fin {
                // Flush any pending ACKs before returning so the sender is not
                // starved while the application drains a burst of buffered data.
                self.drv_mut().flush().await?;
                return Ok((bytes, fin));
            }
            if self.drv().conn.is_closed() {
                return Err(self.drv().close_error());
            }
            self.drv_mut().pump_once(deadline).await?;
            if Instant::now() >= deadline {
                return Err(OxiQuicError::Timeout);
            }
        }
    }

    /// Wait for the peer to open a stream and deliver data, returning the
    /// stream id and the first chunk of `(bytes, fin)`. Useful on the server
    /// side of the echo test.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] on I/O failure, connection close or timeout.
    pub async fn accept_uni_or_bidi_data(
        &mut self,
    ) -> Result<(StreamId, Vec<u8>, bool), OxiQuicError> {
        let deadline = Instant::now() + Duration::from_secs(10);
        self.accept_uni_or_bidi_data_with_deadline(deadline).await
    }

    /// Like [`QuicConnection::accept_uni_or_bidi_data`] but with a
    /// caller-supplied absolute deadline.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] on I/O failure, connection close or timeout.
    pub async fn accept_uni_or_bidi_data_with_deadline(
        &mut self,
        deadline: Instant,
    ) -> Result<(StreamId, Vec<u8>, bool), OxiQuicError> {
        loop {
            if let Some(id) = self.drv_mut().conn.poll_readable() {
                let (bytes, fin) = self.drv_mut().conn.read_stream(id)?;
                return Ok((id, bytes, fin));
            }
            if self.drv().conn.is_closed() {
                return Err(self.drv().close_error());
            }
            self.drv_mut().pump_once(deadline).await?;
            if Instant::now() >= deadline {
                return Err(OxiQuicError::Timeout);
            }
        }
    }

    /// Send an unreliable datagram to the peer (RFC 9221).
    ///
    /// # Errors
    /// Returns [`OxiQuicError`] if the peer does not support datagrams or if
    /// the payload exceeds the peer's advertised `max_datagram_frame_size`.
    pub async fn send_datagram(&mut self, data: Vec<u8>) -> Result<(), OxiQuicError> {
        self.drv_mut().conn.send_datagram(data)?;
        self.drv_mut().flush().await
    }

    /// Receive an unreliable datagram from the peer (RFC 9221).
    ///
    /// Pumps the connection until a datagram arrives, the idle timeout fires,
    /// or the connection closes.
    ///
    /// # Errors
    /// Returns [`OxiQuicError`] on I/O failure, connection close or timeout.
    pub async fn recv_datagram(&mut self) -> Result<Vec<u8>, OxiQuicError> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(dgram) = self.drv_mut().conn.recv_datagram() {
                return Ok(dgram);
            }
            if self.drv().conn.is_closed() {
                return Err(self.drv().close_error());
            }
            self.drv_mut().pump_once(deadline).await?;
            if Instant::now() >= deadline {
                return Err(OxiQuicError::Timeout);
            }
        }
    }

    /// Returns the maximum DATAGRAM payload the peer will accept, or `None` if
    /// the peer does not support unreliable datagrams (RFC 9221).
    #[must_use]
    pub fn max_datagram_size(&self) -> Option<usize> {
        self.drv().conn.max_datagram_size()
    }

    /// Takes the address-validation token received from the server via NEW_TOKEN,
    /// if any (RFC 9000 §8.1.3). Available on the client after the handshake
    /// completes.
    pub fn take_received_token(&mut self) -> Option<Vec<u8>> {
        self.drv_mut().conn.take_received_token()
    }

    /// Whether the server accepted 0-RTT early data (RFC 9001 §4.6).
    ///
    /// - `None`: no 0-RTT was attempted (first connection, no cached ticket) or
    ///   the handshake has not yet completed.
    /// - `Some(true)`: server accepted the early data.
    /// - `Some(false)`: server rejected early data; the data was re-sent in 1-RTT.
    #[must_use]
    pub fn zero_rtt_accepted(&self) -> Option<bool> {
        self.drv().conn.zero_rtt_accepted()
    }

    /// Gracefully close the connection with an application error code/reason.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] if flushing the close frame fails.
    pub async fn close(&mut self, error_code: u64, reason: &[u8]) -> Result<(), OxiQuicError> {
        self.drv_mut().conn.close(error_code, reason);
        self.drv_mut().flush().await
    }

    /// Pump the socket once to process any pending inbound datagrams (e.g. a
    /// peer's CONNECTION_CLOSE). Returns when one datagram is handled or the
    /// short timeout elapses.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] on I/O failure.
    pub async fn drive(&mut self) -> Result<(), OxiQuicError> {
        let deadline = Instant::now() + Duration::from_millis(200);
        let _ = self.drv_mut().pump_once(deadline).await;
        Ok(())
    }

    /// Return a snapshot of connection statistics (RTT estimates, byte and
    /// packet counters, loss count and current congestion window).
    ///
    /// See [`oxiquic_core::ConnectionStats`] for the full set of fields.
    #[must_use]
    pub fn stats(&self) -> oxiquic_core::ConnectionStats {
        self.drv().conn.stats()
    }

    /// Consume this [`QuicConnection`] and return a [`DrivenConnection`] that
    /// services the socket in a background [`tokio::task`].
    ///
    /// After calling `into_driven`, the connection loop runs autonomously.
    /// Streams are opened via [`DrivenConnection::open_bidi_stream`], which
    /// returns [`SendStreamHandle`] / [`RecvStreamHandle`] pairs implementing
    /// [`tokio::io::AsyncWrite`] / [`tokio::io::AsyncRead`].
    #[must_use]
    pub fn into_driven(mut self) -> DrivenConnection {
        // Channel capacities:
        //  - write_tx: 256 — enough to buffer a burst of small writes without
        //    stalling the application.
        //  - open_tx / open_uni_tx: 16 — opening many streams simultaneously is uncommon.
        //  - close_tx: 1 — at most one close is ever sent.
        //  - accept_bidi_tx / accept_uni_tx: 64 — enough to buffer incoming streams
        //    without stalling the protocol loop.
        let (write_tx, write_rx) = mpsc::channel::<(StreamId, WriteCmd)>(256);
        let (open_tx, open_rx) =
            mpsc::channel::<oneshot::Sender<(StreamId, mpsc::Receiver<Vec<u8>>)>>(16);
        let (open_uni_tx, open_uni_rx) =
            mpsc::channel::<oneshot::Sender<(StreamId, mpsc::Receiver<Vec<u8>>)>>(16);
        let (close_tx, close_rx) = mpsc::channel::<(u64, Vec<u8>)>(1);
        let (accept_bidi_tx, accept_bidi_rx) =
            mpsc::channel::<(SendStreamHandle, RecvStreamHandle)>(64);
        let (accept_uni_tx, accept_uni_rx) = mpsc::channel::<RecvStreamHandle>(64);

        let (socket, inbound, conn, peer) = self
            .driver
            .take()
            .expect("QuicConnection driver already consumed by into_driven")
            .into_parts();

        // Capture the negotiated ALPN and peer address before `conn` is moved
        // into the background task (handshake is already complete at this point).
        let negotiated_alpn = conn.negotiated_alpn();
        // `peer` is the driver-level peer address, set during the handshake.
        let peer_addr = peer;

        // Shared closed flag: the driver sets this to `true` with Release
        // ordering immediately before exiting. `DrivenConnection::is_closed`
        // reads it with Acquire ordering.
        let closed = Arc::new(AtomicBool::new(false));

        let task = tokio::spawn(run_driven_connection(
            socket,
            inbound,
            conn,
            peer,
            DrivenConnectionChannels {
                write_tx: write_tx.clone(),
                write_rx,
                open_rx,
                open_uni_rx,
                close_rx,
                accept_bidi_tx,
                accept_uni_tx,
                closed: Arc::clone(&closed),
            },
        ));

        DrivenConnection {
            write_tx,
            open_tx,
            open_uni_tx,
            close_tx,
            accept_bidi_rx: Arc::new(tokio::sync::Mutex::new(accept_bidi_rx)),
            accept_uni_rx: Arc::new(tokio::sync::Mutex::new(accept_uni_rx)),
            _task: Arc::new(task),
            negotiated_alpn,
            peer_addr,
            closed,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Drop for QuicConnection — best-effort graceful close
// ─────────────────────────────────────────────────────────────────────────────

impl Drop for QuicConnection {
    /// Queue a CONNECTION_CLOSE frame in the connection state machine when the
    /// handle is dropped without an explicit [`QuicConnection::close`] call.
    ///
    /// Because `Drop` cannot be `async`, this is best-effort: the
    /// `CONNECTION_CLOSE` frame is written into the connection's output buffer
    /// but **not** flushed to the socket.  If the caller later drives the
    /// connection (or calls [`QuicConnection::close`] before dropping), the
    /// frame will be sent.  If it is not driven again, the peer will rely on
    /// its idle timeout for cleanup.
    ///
    /// Already-closed connections are unaffected.
    fn drop(&mut self) {
        if let Some(d) = self.driver.as_mut() {
            if !d.conn.is_closed() {
                // Application error code 0, empty reason: we are not returning an
                // application-level error, just signalling a clean shutdown.
                d.conn.close(0, b"");
                // Flushing to the socket here would require blocking or spawning
                // (both are wrong in Drop); accept the best-effort limitation.
            }
        }
    }
}
