# Changelog

All notable changes to OxiQUIC are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.2.2] - Unreleased

## [0.2.1] - 2026-08-06

### Security

#### oxiquic-transport
- Anti-amplification limit (RFC 9000 §8.1): a server now sends at most three
  times the bytes received from an unvalidated client address, closing a
  spoofed-source reflection-amplification vector.
- STREAM_LIMIT_ERROR enforcement (RFC 9000 §4.6): STREAM, RESET_STREAM and
  MAX_STREAM_DATA frames referencing a stream ID beyond the advertised
  concurrency limit are now a connection error instead of unbounded per-stream
  map growth.
- CRYPTO reassembly buffering (RFC 9000 §7.5) is now bounded and raises
  CRYPTO_BUFFER_EXCEEDED, preventing unauthenticated pre-handshake memory growth.
- Per-path anti-amplification for addresses adopted during migration
  (RFC 9000 §9.3): an address the connection moves to mid-flight no longer
  inherits the handshake's address validation. It is registered as its own path
  with a fresh three-times-received allowance, every datagram sent to it is
  charged against that allowance, and only a matching PATH_RESPONSE lifts it.
  Previously a single injected packet from a spoofed source turned an
  established connection back into a reflector.
- Retry connection-ID transcript (RFC 9000 §7.3): the server now sends
  `original_destination_connection_id`, `retry_source_connection_id` and
  `initial_source_connection_id`, and both endpoints verify the peer's values
  against the connection IDs actually observed, closing the connection with
  TRANSPORT_PARAMETER_ERROR on any mismatch. Without this a forged Retry — which
  any observer can build, since the Retry integrity tag is keyed by a value sent
  in the clear — was undetectable, so Retry was not an address-validation
  mechanism at all.

### Fixed

#### oxiquic-transport
- Stream and connection flow control (RFC 9000 §4.1): received STREAM data is
  now checked against MAX_STREAM_DATA and MAX_DATA, raising FLOW_CONTROL_ERROR.
- RESET_STREAM `final_size` (RFC 9000 §4.5) is now counted toward stream- and
  connection-level flow control and validated: a final size below the highest
  received offset, contradicting an established final size, or past
  MAX_STREAM_DATA is now a FINAL_SIZE_ERROR / FLOW_CONTROL_ERROR (previously the
  field was parsed and discarded). STREAM data extending past a RESET_STREAM
  final size is likewise rejected.
- BBR min-RTT (rt_prop) now derives from the RFC 9002 `latest_rtt` sample for
  the largest newly-acked packet rather than approximating it from the batch's
  newest send time, which previously understated it and could trigger spurious
  ProbeRTT.
- Packet-number decode (RFC 9000 A.3) uses the correct `>=` half-window boundary.
- BBR round counter compares the acked packet's send-time `delivered` snapshot.
- Connection IDs issued to the peer are now bounded by the **peer's**
  `active_connection_id_limit` (RFC 9000 §5.1.1) rather than our own advertised
  value, which the pool was seeded with and never replaced. A peer advertising
  the RFC minimum of 2 against OxiQUIC's default of 7 was handed five CIDs too
  many and could legitimately answer with CONNECTION_ID_LIMIT_ERROR. Issuance is
  correspondingly deferred until the peer's transport parameters are applied.
- An MTU probe (RFC 8899) is no longer built into a datagram addressed to an
  unvalidated candidate address: the padded probe spent that address's
  RFC 9000 §9.3 three-times allowance on an MTU measurement and, because the
  probe branch returns early, displaced the PATH_CHALLENGE the datagram existed
  to carry. The probe now waits for the next datagram on the established path.
- The bundled endpoint no longer restarts path validation for every datagram
  from an unfamiliar source address — only the first one does. Repeating it reset
  the RFC 9000 §8.2 retransmission backoff and abandon deadline on each packet,
  so neither could ever fire; `set_candidate_peer_addr` is now idempotent for the
  address already being probed.

### Added

#### oxiquic-transport
- ECN (RFC 9000 §13.4 / RFC 9002 §7.4), both directions. Egress: ECT(0) marking,
  the validation state machine and the CE→congestion response. Ingress:
  `Connection::handle_datagram_with_meta` takes a `DatagramMeta` carrying the
  datagram's source address and IP ECN codepoint; the codepoint is counted into
  the packet-number space of every packet decrypted from that datagram
  (RFC 9000 §13.4.1) and echoed to the peer as an ACK-ECN (0x03) frame, which is
  what lets a conformant peer's ECN validation succeed. `Connection::ecn_recv_counts`
  exposes the counters.
- Receive-side ECN off the socket (`endpoint::ecn_recv`): the bundled `tokio`
  endpoint now enables `IP_RECVTOS` / `IPV6_RECVTCLASS` when it binds and reads
  the codepoint out of the `recvmsg` ancillary data of every datagram, on Linux,
  Android, macOS, iOS and FreeBSD. The `unsafe` belongs to `nix` (a new
  unix-only dependency), so `oxiquic-transport` keeps `#![forbid(unsafe_code)]`,
  and readiness still runs through the `tokio` reactor via `async_io`/`try_io`.
  Both endpoint types expose `reports_inbound_ecn()`, and `QuicConnection` gains
  `ecn_state()` / `ecn_recv_counts()`.
  On any other target — or if the kernel refuses the socket option — the reason
  is typed (`EcnRecvUnsupported::Platform` / `::SockOpt`) and nothing is
  synthesised: no counter moves, plain (0x02) ACKs are sent, and a missing
  codepoint is never counted as Not-ECT. `tests/ecn_socket.rs` proves every
  codepoint round-trips over IPv4 and IPv6 loopback, that DSCP bits never leak
  into the codepoint, and that a real endpoint pair drives RFC 9000 §13.4.2
  validation all the way to `Capable`.
- Per-path congestion control and RTT estimation
  (`multipath::PathRecovery`): every path owns a congestion controller and an
  RFC 9002 RTT estimator, every sent packet records the path it went out on, and
  acknowledgements, losses and CE marks are attributed to that path alone. The
  `LowestRtt` / `HighestBandwidth` / `RoundRobin` schedulers now rank paths by
  their own measurements instead of one shared average, and a path is gated by
  its own congestion window as well as the connection-level one.
- Multipath `PathScheduler` (Active / LowestRtt / HighestBandwidth / RoundRobin)
  wired into the send path.
- `Connection::new_server_after_retry` and `RetryTranscript` for building a
  post-Retry server connection with the authenticated §7.3 connection-ID
  transcript; the bundled server endpoint uses it automatically.
- A queued PATH_CHALLENGE is now sent to the *candidate* address
  (RFC 9000 §9.3.3) rather than the old one, bounded by that address's own
  anti-amplification allowance, and the challenge frame is only written into the
  datagram actually addressed to that candidate. If the candidate's allowance
  cannot yet cover a datagram the probe is deferred and traffic keeps flowing on
  the validated path — the per-path limit never stalls the connection, and
  `amplification_blocked()` (which disarms the loss-detection timer per
  RFC 9002 §6.2.2.1) is evaluated on the established peer address only. A
  validated address is promoted to the active path so the scheduler stays in
  step with the migration.
- Path-validation timer (RFC 9000 §8.2.1, §8.2.3, §8.2.4). A PATH_CHALLENGE that
  goes unanswered used to stall path validation forever, because the frame is
  not repaired by loss recovery — §8.2.1 requires unpredictable data in *every*
  challenge, so a lost one must be replaced rather than retransmitted verbatim.
  The connection now schedules a replacement challenge, with fresh nonce, once
  the validation PTO expires (the larger of the current PTO and a no-sample
  path's PTO), doubling the interval per attempt; a PATH_RESPONSE matching *any*
  still-outstanding challenge validates the path (§8.2.3), so a merely delayed
  probe is not wasted. After three times that PTO validation is abandoned
  (§8.2.4): the candidate address is dropped, its path is marked
  `PathValidation::Failed` and the connection stays on the address it already
  validated. The timer is independent of `loss_timer`, which RFC 9002 §6.2.2.1
  disarms exactly when a probe most needs to survive. New API:
  `initiate_path_challenge_at`, `path_validation_failed`,
  `path_challenge_attempts`, `outstanding_path_challenges`,
  `candidate_peer_addr`.
- Connection-ID issuance and rotation for migration (RFC 9000 §5.1.1, §9.5).
  Completing a migration now raises the `retire_prior_to` advertised in outgoing
  NEW_CONNECTION_ID frames past every CID the peer used on the old path and
  issues a full fresh batch in the same step, forcing the peer off the connection
  IDs it was sending to before the move; the rotation is rolled back if no
  replacement CID could be issued, so the peer is never told to retire everything
  with nothing to move to. (This is the issuing half of §9.5. The consuming half
  — switching *our own* outgoing destination CID to an unused peer-issued one —
  is not implemented, so an observer of both paths can still correlate them by
  that CID; it is recorded in the deferred list in `lib.rs`.) Adopting a candidate address also tops the
  pool back up, giving a migrating peer a spare CID to move to. Retired-but-not-
  yet-acknowledged CIDs are excluded from the limit accounting yet stay routable
  until the peer's RETIRE_CONNECTION_ID arrives, since packets carrying them may
  still be in flight. New API: `local_cid_seqs`, `local_cid_retire_prior_to`,
  `peer_cid_seqs`, `peer_cid_retire_threshold`.
- `cargo-fuzz` targets under `fuzz/` (a detached workspace, excluded from the
  main build) for the three attacker-facing, pre- and post-decrypt parsers:
  `peek_dcid` (unauthenticated packet-type/DCID classification), `packet_decode`
  (header-protection removal + AEAD decrypt of a long-header packet, using
  Initial keys derived from a fuzzer-chosen DCID) and `frame_decode`
  (`decode_frame` over an already-decrypted payload). Run with
  `cargo +nightly fuzz run <target>` from `crates/oxiquic-transport/fuzz`.

#### oxiquic
- Runnable examples: `quic_echo_server` / `quic_echo_client` (a self-signed
  loopback QUIC echo pair, `--features dangerous`) and `h3_get` (a
  self-contained HTTP/3 GET round trip against an in-process server,
  `--features h3`), mirroring the README quick-start snippets and compiled by
  `cargo build --examples`.

### Changed
- **Wire-compatibility break (pre-release).** RFC 9000 §7.3 requires both
  endpoints to send `initial_source_connection_id`, and 0.2.1 now rejects a peer
  that omits it with TRANSPORT_PARAMETER_ERROR. OxiQUIC ≤ 0.2.0 never sent the
  connection-ID transport parameters, so 0.2.1 cannot complete a handshake with
  an OxiQUIC ≤ 0.2.0 peer. Both sides must be upgraded together. Interop with
  other RFC 9000 implementations is unaffected — they have always sent them.
- `TransportConfig::retry` documents why the default stays `false`: the
  always-on RFC 9000 §8.1 three-times limit (plus the §9.3 per-path allowance)
  is the primary amplification defence, whereas Retry costs every connection a
  round trip and requires fleet-wide token-secret management.
- `endpoint/mod.rs` (1961 lines and growing) is split: the server demux loop,
  the Retry branch, the routing-table GC helper and their unit tests moved to
  `endpoint/demux.rs`, leaving `endpoint/mod.rs` at ~1290 lines. Visibility is
  unchanged (`pub mod driven` / `pub mod zero_rtt` still re-exported as before);
  the split is purely file organisation to stay under the 2000-line cap.
- Added `rustfmt.toml` (pins `edition = "2021"`; the existing formatting
  already matched rustfmt's stable defaults otherwise) and `clippy.toml`
  (pins `msrv = "1.85"` to match `rust-version`).
- Dev-dependencies on in-workspace crates (`oxiquic-crypto` in `oxiquic` and
  `oxiquic-h3`; a redundant `oxiquic-core` entry in `oxiquic-transport`, which
  duplicated its own normal dependency) now go through `workspace = true`
  instead of a bare `path = "../..."`, consistent with the rest of the
  workspace's dependency management.
- Bump `oxicrypto` dependency from `0.2.0` to `0.3.0` across the workspace.
- Bump `oxitls` / `oxitls-core` / `oxitls-rcgen` dependencies from `0.2.0` to
  `0.3.0` across the workspace.
- MSRV raised from `1.80` to `1.85` (`rust-version` in `[workspace.package]`;
  `clippy.toml`'s `msrv` already tracked this).
- `aead` bumped from `0.5.2` to `0.6.1`, `aes-gcm` from `0.10.3` to `0.11.0`
  and `chacha20poly1305` from `0.10.1` to `0.11.0` for the packet-protection
  AEAD chain used directly by `oxiquic-crypto`/`oxiquic-transport`. This
  resolves as a separate copy from the `aead` 0.5 chain still pulled in
  transitively by `oxitls-rustcrypto-provider`/`rustls-rustcrypto` — the two
  never need to share concrete AEAD types, since everything crossing that
  boundary goes through rustls's own `MessageEncrypter`/`MessageDecrypter`/
  `PacketKey` trait objects, not raw `aead::*` types.
- Version bump to 0.2.1.

## [0.2.0] - 2026-06-22

### Changed
- Bump `oxitls` dependency to `^0.2.0` across the workspace (oxiquic-crypto, oxiquic).

### Security
- Clears PENDING-REPUBLISH status: oxiquic 0.1.x depended on oxitls 0.1.x which had a
  native-cert leak (webpki-roots native store bleed-through). The 0.2.0 line ships with
  oxitls 0.2.0 which resolves that issue; all users of the `oxitls-provider` feature
  should upgrade.

## [0.1.4] - 2026-06-19

### Added

#### oxiquic-transport
- `QuicConnection::peer_addr() -> Option<SocketAddr>`: returns the remote peer
  address after a successful handshake. For server-side connections this is the
  client UDP source address; for client-side connections this is the server address
  passed to `ClientEndpoint::connect`. Guaranteed to be `Some` after the handshake
  in normal operation.
- `DrivenConnection::peer_addr() -> Option<SocketAddr>`: peer address preserved
  across `QuicConnection::into_driven()` so callers retain remote address information
  after moving the connection into background I/O mode.
- `DrivenConnection::is_closed() -> bool`: liveness hint backed by an `Arc<AtomicBool>`
  written with `Release` ordering by the driver task immediately before it exits.
  A `true` result is definitive (driver has stopped); `false` means the driver has
  not yet set the flag and may still be running. Reads with `Acquire` ordering so
  callers that observe `true` also observe all connection-state mutations that
  preceded the driver exit.

### Changed
- `oxiquic-transport` crate-level doc comment updated: clarifies that 0-RTT,
  MAX_STREAMS/STREAMS_BLOCKED, RESET_STREAM/STOP_SENDING, and stateless reset are
  all implemented; narrows the "not yet implemented" note to ECN only (RFC 9000 §13.4).
- End-to-end test `lossless_echo_round_trip_demo` comment corrected: the demo
  cannot exercise congestion control or loss detection on lossless loopback, but
  those subsystems are implemented and validated by their own unit tests.

### Fixed
- `into_driven` no longer silently drops the peer address: it now captures the
  driver's `peer` field before moving `conn` into the background task and stores it
  in `DrivenConnection::peer_addr`.

## [0.1.3] - 2026-06-15

### Changed
- Version bump to 0.1.3 across all workspace crates (no functional code changes).

---

## [0.1.2] - 2026-06-10

### Added

#### oxiquic-core
- `alpn` module: well-known ALPN protocol byte-string constants (`H3 = b"h3"`,
  `HTTP_0_9 = b"hq-interop"`) and the `alpn::protocols(&[&[u8]]) -> Vec<Vec<u8>>`
  builder helper for constructing owned ALPN lists from byte-string slices.
  Re-exported from the `oxiquic` facade as `oxiquic::alpn::{H3, HTTP_0_9, protocols}`.

#### oxiquic-transport
- `ServerEndpointBuilder::with_alpn_protocols(&[&[u8]]) -> Self` builder method:
  replaces `alpn_protocols` on the underlying `rustls::ServerConfig`, enabling
  ALPN negotiation on raw QUIC server endpoints without rebuilding the TLS config.
  Supersedes the `config_pair_with_alpn` test-only workaround used previously.

#### oxiquic (facade)
- `connect_with_alpn(addr, server_name, protocols)` convenience function: like
  `connect()` but sets `alpn_protocols` on the client TLS config before performing
  the handshake. After a successful connection, `QuicConnection::negotiated_alpn()`
  returns the protocol selected by the server.
- `listen_with_alpn(addr, cert_chain, private_key, protocols)` convenience function:
  like `listen()` but injects custom `alpn_protocols` into the server TLS config
  before binding the endpoint.
- `alpn` re-export module exposed at the crate root.

### Testing
- 3 new integration tests in `oxiquic-transport/tests/alpn.rs`:
  - `custom_alpn_roundtrip`: both sides advertise the same custom ALPN identifier;
    after the handshake `negotiated_alpn()` returns the identifier on both endpoints.
  - `alpn_not_set_does_not_panic`: no ALPN configured → handshake succeeds, both
    sides return `None` from `negotiated_alpn()`.
  - `server_endpoint_builder_with_alpn_protocols`: `ServerEndpointBuilder::with_alpn_protocols`
    correctly overrides ALPN configured at construction time; client sees the negotiated protocol.
- Total tests: 329 (unit + integration), all passing.

### Fixed
- `oxiquic-h3` `h3_response_status_codes` test: server task now calls
  `h3_server.shutdown(0)` after sending the response, so the driver task exits
  cleanly and the client receives `CONNECTION_CLOSE` before the server drops.
  Matches the pattern applied to `h3_get_roundtrip` in v0.1.1.

## [0.1.1] - 2026-06-04

### Added
- `bench_memory_usage` benchmark in `oxiquic-transport`: measures RSS delta per QUIC connection
  (1 and 10 connections) on Linux (`/proc/self/status`) and macOS (`mach_task_self()` task_info);
  prints a one-time per-connection kilobyte estimate alongside criterion timing data.
- `bench_h3_memory_profile` benchmark in `oxiquic-h3`: same RSS methodology applied to HTTP/3
  connections — reports per-H3-connection heap overhead and measures connection setup rate via
  criterion (`h3_memory/establish_n_h3_connections/{1,5}`).
- `bench_h3_push_overhead` benchmark in `oxiquic-h3`: documents server push stub latency vs an
  equivalent client-initiated GET; confirms the push stub path (always `NotImplemented` in h3
  0.0.8) adds no measurable network overhead (`h3_push_overhead/{client_get_1kb,push_stub_noop}`).
- `bench_h3_vs_h2_throughput` benchmark in `oxiquic-h3`: sustained throughput comparison of H3
  vs H2 at 256 KiB and 1 MiB payload sizes using `criterion::Throughput::Bytes` to report bytes/s
  (`h3_vs_h2_throughput/{h3,h2}_{256kb,1mb}`); exercises flow-control and congestion-window paths.

### Fixed
- `oxiquic-transport` driven connection loop: `io::ErrorKind::ConnectionRefused` (ICMP
  port-unreachable, sent when the peer's socket closes before a `CONNECTION_CLOSE` frame) is now
  treated as non-fatal — the loop continues instead of breaking, letting the QUIC loss-detection
  timer handle recovery per RFC 9000.
- `oxiquic-h3` `h3_get_roundtrip` integration test: server now calls `h3_conn.shutdown(0)` after
  serving the response, so the client receives `CONNECTION_CLOSE` and the driver task exits cleanly
  instead of racing against the QUIC idle-timeout.

## [0.1.0] — 2026-06-01

### Added

#### oxiquic-core
- `OxiQuicError` enum (thiserror): `NotImplemented`, `Connect`, `Stream`, `Timeout`, `Tls`,
  `Protocol`, `FrameEncoding`, `FlowControl`, `Io`, `Other` plus `is_timeout()`, `is_closed()`,
  `is_reset()` predicates.
- `StreamId(u64)` newtype: `initiator()`, `direction()`, `index()` per RFC 9000 §2.1.
- `ConnectionId` newtype with variable-length DCID/SCID support.
- `ConnectionStats` struct: RTT (min/smoothed/variance), bytes/packets sent/recv/lost,
  congestion window; `Display` impl with human-readable formatting.
- `TransportParams` struct: all RFC 9000 transport parameters with codec support.
- `FrameType` enum: full RFC 9000 frame type set (PADDING through HANDSHAKE_DONE).
- `QuicVersion` enum: V1 (RFC 9000), V2 (RFC 9369), VersionNegotiation.
- Optional `serde` feature gate for `StreamId`, `ConnectionId`, `ConnectionStats`.
- Optional `oxitls` feature gate for `From<TlsError>` conversion bridge.

#### oxiquic-crypto
- Pure-Rust QUIC crypto provider for `rustls` — no ring, no aws-lc-rs.
- AEAD packet protection: AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305 via RustCrypto.
- Header protection key derivation and mask application (RFC 9001).
- Initial key derivation: HKDF-SHA256 label expansion per RFC 9001 §5.2.
- HMAC-SHA256/SHA384/SHA512 implementations.
- HKDF-SHA256/SHA384 implementations.
- Three QUIC cipher suites: TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384,
  TLS_CHACHA20_POLY1305_SHA256 — all with `quic: Some(..)` for packet-key derivation.
- `quic_crypto_provider()` function returning the assembled `rustls::CryptoProvider`.
- Optional `oxitls-provider` feature: `oxitls_quic_provider()` sourced from oxitls.

#### oxiquic-transport
- `ClientEndpoint` / `ServerEndpoint`: tokio UDP-based QUIC endpoints.
- `QuicConnection` / `DrivenConnection`: full connection lifecycle management.
- QUIC 1-RTT TLS 1.3 handshake via `rustls::quic` module.
- QUIC 0-RTT early data: `ClientEndpoint::connect_0rtt()`, `ServerEndpointBuilder::with_max_early_data_size()`.
- `TransportConfig` builder: idle timeout, keep-alive interval, MTU discovery, stream/connection windows, congestion algorithm selection.
- Loss detection: RFC 9002 PTO, ACK-based loss detection, loss recovery state machine.
- Congestion control — Cubic (RFC 9438): slow start, congestion avoidance, fast recovery.
- Congestion control — BBR v2: bandwidth estimation, pacing, ProbeRTT, ProbeBW phases.
- `CongestionAlgorithm` selector: `Cubic | Bbr`.
- Stream multiplexing: bidirectional and unidirectional streams with independent flow control.
- Connection-level and stream-level flow control with MAX_DATA/MAX_STREAM_DATA/STREAMS_BLOCKED.
- `SendStreamHandle` (implements `AsyncWrite`) and `RecvStreamHandle` (implements `AsyncRead`).
- `BiStream`, `UniSendStream`, `UniRecvStream` type-safe direction wrappers.
- `QuicConnection::open_bi_reliable(max_attempts, retry_delay)` with back-pressure retry.
- `QuicConnection::ping() -> Duration` RTT measurement.
- Version negotiation: server sends VN packet for unknown versions; client handles gracefully.
- Stateless retry: HMAC-SHA256 token generation and validation (RFC 9000 §8.1).
- Key update: RFC 9001 §6 key phase bit, per-epoch key derivation, cooldown period.
- Connection migration: PATH_CHALLENGE/PATH_RESPONSE validation, candidate address promotion.
- Multi-connection server demux: DCID-based routing for concurrent clients.
- MTU discovery: DPLPMTUD (RFC 8899) binary-search probe with ACK/loss callbacks.
- Idle timeout enforcement and keep-alive PING frames.
- `ServerEndpoint::local_addr()` accessor.
- `h3-compat` feature: `h3::quic` trait implementations over oxiquic-transport streams.
- `dangerous` feature: `connect_insecure()` for dev/testing.

#### oxiquic-h3
- `H3Client` over in-house oxiquic-transport QUIC streams.
- `H3ClientBuilder`: `with_server_name()`, `with_tls_config()`, ALPN enforcement (`h3`).
- `H3Client::get()`, `post()`, `request()`, `close()` methods.
- `H3Response`: `status()`, `headers()`, `body_bytes()`, `body_text()`, `content_length()`,
  `content_type()`, `is_success()`.
- `RequestStream` for streaming request/response bodies over HTTP/3 DATA frames.
- `H3Server` accepting HTTP/3 connections over oxiquic-transport.
- `H3ServerBuilder`: `with_tls_config()`, `with_ticketer()`, `bind()`, ALPN enforcement.
- `H3Server::new(driven)` performing HTTP/3 SETTINGS exchange.
- `H3Server::accept()` returning `H3RequestContext`.
- `H3RequestContext::body()`, `respond()` for request handling.
- `H3Responder::push_promise()` stub (upstream-limited: h3 0.0.8 has no push API).
- Graceful shutdown: GOAWAY frame via `h3::server::Connection::shutdown()`.
- QPACK configuration fields (stored for forward-compat; h3 0.0.8 is stateless-only).
- `H3Error` enum: `Protocol`, `Qpack`, `Stream`, `Connection`, `Io`, `Tls`,
  `FrameUnexpected`, `SettingsError`, `MissingSettings`, `IdError`.
- `H3Settings`, `H3Request`, `H3Response` message types.
- Optional `serde` and `tracing` feature gates.

#### oxiquic (facade)
- Feature-gated unified re-exports: `transport` (default), `h3`, `dangerous`.
- `prelude` module and `h3_prelude` module.
- `connect(addr, server_name)` convenience function.
- `listen(addr, certs, key)` convenience function.
- `connect_insecure(addr, server_name)` under `dangerous` feature.
- `version()` and `quic_version()` accessors.

### Security
- Zero C/C++/Fortran dependencies in default features (`cargo tree --edges normal`
  contains no ring, aws-lc-rs, aws-lc-sys, openssl, or openssl-sys).
- FFI audit gate: `deny.toml` + `scripts/ffi-audit.sh` enforce ban list.
- Stateless reset token generation per RFC 9000 §10.3.1 (HMAC-SHA256).

### Notes
- Deferred: 0-RTT user-facing API (framing plumbed, round-trip handshake works);
  RESET_STREAM/STOP_SENDING end-to-end user API (framing complete); H3 server push
  (upstream-limited: h3 0.0.8); ALPN enforcement at H3 facade layer; multipath QUIC.
- 321 tests pass (unit + integration); zero clippy warnings; zero `unwrap()`/`panic!`
  in production code.
- ~22 000 SLOC across 5 crates.

[0.1.4]: https://github.com/cool-japan/oxiquic/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/cool-japan/oxiquic/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/cool-japan/oxiquic/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiquic/releases/tag/v0.1.1
