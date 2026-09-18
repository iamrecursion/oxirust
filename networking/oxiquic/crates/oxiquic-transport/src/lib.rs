//! QUIC transport layer for the OxiQUIC stack.
//!
//! `oxiquic-transport` is a Pure-Rust QUIC (RFC 9000 / 9001) implementation
//! built directly on the `rustls::quic` TLS 1.3 handshake API, driven by the
//! `oxiquic-crypto` `CryptoProvider` (no `ring`, no `aws-lc-rs`). It runs over
//! `tokio`'s asynchronous UDP sockets.
//!
//! The crate is split into a synchronous, I/O-free protocol core
//! ([`Connection`]) and a thin asynchronous shell ([`endpoint`]) that pumps
//! datagrams between the core and a UDP socket. A caller drives a client with
//! [`ClientEndpoint::bind`] then [`ClientEndpoint::connect`], a server with
//! [`ServerEndpoint::bind`] then [`ServerEndpoint::accept`]; both yield a
//! [`QuicConnection`] for opening bidirectional streams and reading/writing data.
//!
//! # Implementation status
//!
//! Implemented and proven over real UDP loopback:
//!
//! * **Initial handshake** — long-header packet coding, header protection,
//!   packet protection, coalesced-packet parsing, CRYPTO-frame reassembly
//!   driving the rustls TLS 1.3 handshake, ACKs and per-space packet numbers.
//! * **1-RTT + close** — 1-RTT keys on `KeyChange::OneRtt`, short-header
//!   packets, `HANDSHAKE_DONE`, `CONNECTION_CLOSE` and idle handling.
//! * **Stream data** — bidirectional stream state machines with ordered
//!   reassembly and a send/receive API.
//!
//! * **Loss detection & recovery** (RFC 9002 Sections 5-6) — sent-packet
//!   tracking per space, RTT estimation (latest, min, smoothed, rttvar),
//!   packet-number threshold and time-threshold loss detection, PTO (probe
//!   timeout) with exponential backoff, retransmission of lost CRYPTO/STREAM
//!   frames.
//! * **Congestion control** — CUBIC (RFC 9438, default), NewReno (RFC 9002
//!   Appendix B) and BBR v2 (model-based), selected via
//!   [`TransportConfig::congestion_controller`]. All three share the
//!   [`CongestionController`] dispatch enum; CUBIC tracks W_max, K and the
//!   cubic epoch for RFC 9438 window growth after loss.
//! * **Flow control** (RFC 9000 Section 4) — connection- and stream-level
//!   send/receive limits, MAX_DATA / MAX_STREAM_DATA generation and processing,
//!   DATA_BLOCKED / STREAM_DATA_BLOCKED signalling.
//!
//! * **Version Negotiation** (RFC 9000 §17.2.1) — server sends a VN packet
//!   when an Initial arrives with an unsupported version; client fails the
//!   connection with [`OxiQuicError::VersionNegotiation`] on receiving VN
//!   during the early handshake.
//!
//! * **Retry** (RFC 9000 §17.2.5, §7.3, RFC 9001 §5.8) — server optionally
//!   sends a Retry packet to force clients to prove their source address.
//!   Enable via [`TransportConfig::retry`] (off by default; the rationale is on
//!   that method). The client processes the Retry integrity tag, re-keys the
//!   Initial space, and retransmits with the echoed token. The connection-ID
//!   transcript is authenticated inside the TLS handshake: the server sends
//!   `original_destination_connection_id`, `retry_source_connection_id` and
//!   `initial_source_connection_id`, and the client rejects the connection with
//!   `TRANSPORT_PARAMETER_ERROR` if any of them disagrees with what it saw on
//!   the wire — which is what makes a forged Retry detectable, since the Retry
//!   integrity tag alone is keyed by a value that travels in the clear.
//!
//! * **Connection path migration** (RFC 9000 §9) — PATH_CHALLENGE / PATH_RESPONSE
//!   wire frames (frame types 0x1a / 0x1b), peer-address-change detection in the
//!   endpoint, and `initiate_path_challenge()` / `path_validated()` API on
//!   [`QuicConnection`].  A queued PATH_CHALLENGE is sent to the *candidate*
//!   address (§9.3.3) under that address's own anti-amplification allowance.
//!
//!   *Path validation is repaired on a timer* (§8.2.1, §8.2.4): a challenge
//!   that goes unanswered is replaced — with fresh unpredictable data, as the
//!   RFC requires of every PATH_CHALLENGE — once the validation PTO expires,
//!   doubling the interval on each attempt, and a PATH_RESPONSE to *any* still
//!   outstanding challenge validates the path (§8.2.3). Validation is abandoned
//!   after three times the larger of the current PTO and a new path's PTO, at
//!   which point the candidate address is dropped, its path is marked failed
//!   and the connection stays on the address it already validated. The timer is
//!   deliberately independent of the loss-detection timer, which RFC 9002
//!   §6.2.2.1 disarms exactly when the connection is amplification-blocked.
//!
//!   *Connection IDs are rotated on migration* (§9.5, issuing side): completing
//!   a migration raises the `retire_prior_to` this endpoint advertises past
//!   every CID the peer used on the old path and issues a full fresh batch in
//!   the same step, so the peer is forced off the connection IDs it was sending
//!   to before the move. Adopting a candidate address also tops the pool back
//!   up, so a migrating peer always has a spare to switch to. Issuance is
//!   bounded by the **peer's** `active_connection_id_limit` (§5.1.1), with the
//!   retired-but-not-yet-acknowledged CIDs excluded from that accounting and
//!   kept routable until the peer's RETIRE_CONNECTION_ID arrives.
//!
//!   Still deferred: the *consuming* half of §9.5 — this endpoint keeps sending
//!   to the migrated address with the same peer-issued destination connection ID
//!   it used before, where the RFC requires switching to an unused one, so an
//!   observer of both paths can still correlate them by that CID. Also deferred:
//!   the multipath draft's wire extensions (per-path packet-number spaces,
//!   PATH_ACK, multipath stream mapping), pending stabilisation of the draft.
//!
//! * **Anti-amplification limit** (RFC 9000 §8.1) — before the client's source
//!   address is validated a server sends at most three times the number of
//!   bytes it has received on the connection; the datagram builder is capped by
//!   the remaining allowance and, per RFC 9002 §6.2.2.1, the loss-detection
//!   timer stays disarmed while blocked. The address is validated by a
//!   successfully processed Handshake packet, by handshake completion, or by a
//!   valid Retry token. Per RFC 9000 §9.3, an address adopted *during* the
//!   connection (migration / NAT rebinding) does not inherit that validation:
//!   it is registered as its own path with its own three-times-received
//!   allowance, charged per datagram sent to it and lifted only when a matching
//!   PATH_RESPONSE validates it.
//!
//! * **Key update** (RFC 9001 §6) — key phase bit, per-epoch key derivation,
//!   3-PTO cooldown, `initiate_key_update()` + `key_update_count()` on
//!   [`Connection`] (4 tests in `tests/key_update.rs`).
//!
//! * **DPLPMTUD / path MTU discovery** (RFC 8899) — binary-search probe
//!   scheduling post-handshake, PING frames padded to candidate sizes, ACK and
//!   loss callbacks updating `current_mtu()`; enabled by default, ceiling
//!   configurable via [`TransportConfig::mtu_discovery`] (3 tests in
//!   `tests/mtu_discovery.rs`).
//!
//! Also implemented: **0-RTT** early data ([`ClientEndpoint::connect_0rtt`],
//! client send and server accept paths), **MAX_STREAMS / STREAMS_BLOCKED**
//! (frame codec, transport-parameter limits, and end-to-end peer-limit
//! processing), **RESET_STREAM / STOP_SENDING** (frame codec plus the
//! `reset()` / `stop_sending()` API on the stream handles), and **stateless
//! reset** token derivation (RFC 9000 §10.3) with incoming-reset detection.
//!
//! * **ECN** (RFC 9000 §13.4, RFC 9002 §7.4) — both directions are
//!   implemented. Outgoing packets are marked ECT(0) at the socket layer
//!   (`IP_TOS` / `IPV6_TCLASS` via `socket2`, with a graceful Not-ECT fallback
//!   where the platform refuses the option); the ECN counts in received ACK-ECN
//!   (0x03) frames are parsed and run through the per-space validation state
//!   machine ([`ecn::EcnController`]); a rise in the peer's CE counter triggers
//!   a congestion-window reduction. On the receive side, the codepoint reported
//!   by the I/O layer through [`connection::DatagramMeta`] is counted into the
//!   packet-number space of every packet decrypted from that datagram, and
//!   echoed back in ACK-ECN frames.
//!
//!   The bundled `tokio` [`endpoint`] reads the codepoint off the socket
//!   itself: [`endpoint::ecn_recv`] enables `IP_RECVTOS` / `IPV6_RECVTCLASS` at
//!   bind time and pulls the value out of the `recvmsg` ancillary data of every
//!   datagram, on Linux, Android, macOS, iOS and FreeBSD. The `unsafe` belongs
//!   to `nix`, so this crate remains `#![forbid(unsafe_code)]`, and readiness
//!   still goes through the `tokio` reactor.
//!
//!   Where the platform or the kernel refuses that, the failure is typed
//!   ([`endpoint::ecn_recv::EcnRecvUnsupported`], observable via
//!   [`ClientEndpoint::reports_inbound_ecn`]) and nothing is fabricated — no
//!   counter moves, plain (0x02) ACKs are sent, and (per RFC 9000 §13.4.2.1)
//!   the peer correctly disables its own ECN marking. `None` is never counted
//!   as Not-ECT.
//!
//! * **Per-path congestion control and RTT** — each multipath path owns a
//!   congestion controller and an RFC 9002 RTT estimator
//!   ([`connection::multipath::PathRecovery`]). Every sent packet records the
//!   path it went out on, so acknowledgements, losses and CE marks are
//!   attributed to that path only; the non-`Active` [`PathScheduler`] policies
//!   rank paths by their own measurements rather than a shared average.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bbr;
mod cc_dispatch;
pub mod coding;
mod config;
mod congestion;
pub mod connection;
mod crypto_stream;
mod cubic;
pub mod ecn;
pub mod endpoint;
mod flow_control;
pub mod frame;
#[cfg(feature = "h3-compat")]
pub mod h3_compat;
pub mod handle;
pub mod packet;
mod params_codec;
mod recovery;
mod sent_packet;
mod space;
mod stream;

pub use bbr::{Bbr, BbrState, DeliveryRateEstimator, RateSample};
pub use cc_dispatch::CongestionController;
pub use config::{CongestionAlgorithm, TransportConfig};
pub use connection::multipath::{
    MultipathState, PathAmplification, PathRecovery, PathScheduler, PathState, PathValidation,
};
pub use connection::{Connection, ConnectionState, DatagramMeta, MtuConfig, RetryTranscript, Role};
pub use ecn::{EcnAckOutcome, EcnCodepoint, EcnController, EcnCounts, EcnValidationState};
pub use endpoint::{
    ClientEndpoint, DrivenConnection, Incoming, QuicConnection, ServerEndpoint,
    ServerEndpointBuilder, ZeroRttAccepted,
};
#[cfg(feature = "h3-compat")]
pub use h3_compat::{
    H3BidiStream, H3RecvStream, H3SendStream, OxiQuicH3Connection, OxiQuicOpenStreams,
};
pub use handle::{BiStream, RecvStreamHandle, SendStreamHandle, UniRecvStream, UniSendStream};
pub use oxiquic_core::{ConnectionStats, OxiQuicError, StreamId, TransportParams};
pub use packet::{
    compute_retry_integrity_tag, decode_version_negotiation, encode_retry_packet,
    encode_version_negotiation, parse_retry_packet, verify_retry_integrity_tag,
};
