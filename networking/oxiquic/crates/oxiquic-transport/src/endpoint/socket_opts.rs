//! Socket-layer ECN marking (RFC 9000 §13.4).
//!
//! QUIC marks its egress with the ECT(0) codepoint by setting the ECN bits of
//! the IP Type-of-Service byte (`IP_TOS` for IPv4, `IPV6_TCLASS` for IPv6) on
//! the UDP socket. `tokio`'s [`UdpSocket`] does not expose these options, so we
//! reach through to the raw socket via [`socket2::SockRef`] — a thin, pure-Rust
//! wrapper over the OS `setsockopt` call (no C is compiled).
//!
//! Marking is per-socket rather than per-datagram: every datagram sent on the
//! socket carries the codepoint most recently applied. Callers therefore track
//! the last-applied codepoint and only re-apply when the connection's desired
//! codepoint changes. Where the platform refuses the option (e.g. Windows
//! without qWAVE, or an OS socket2 does not support it on), the caller marks ECN
//! unsupported so the validation state machine never leaves the testing state
//! and never expects ECN feedback.

use std::io;

use socket2::SockRef;
use tokio::net::UdpSocket;

use crate::connection::Connection;
use crate::ecn::EcnCodepoint;

/// Set the IP ECN codepoint stamped on datagrams sent from `socket`.
///
/// The socket's local address family selects the option: `IP_TOS` for IPv4 and
/// `IPV6_TCLASS` for IPv6. On a dual-stack (IPv6) socket both are set so that
/// IPv4-mapped destinations are also marked; a failure to set the non-primary
/// option for the family is ignored.
///
/// # Errors
/// Returns the underlying [`io::Error`] if the primary option for the socket's
/// address family cannot be set (the caller treats this as "ECN unsupported").
pub(crate) fn apply_ecn_tos(socket: &UdpSocket, cp: EcnCodepoint) -> io::Result<()> {
    let tos = u32::from(cp.to_tos_bits());
    let sock = SockRef::from(socket);
    match socket.local_addr()? {
        std::net::SocketAddr::V4(_) => sock.set_tos_v4(tos),
        std::net::SocketAddr::V6(_) => {
            // Primary for an IPv6 socket: the IPv6 traffic class.
            let primary = sock.set_tclass_v6(tos);
            // Best-effort for IPv4-mapped destinations on a dual-stack socket.
            let _ = sock.set_tos_v4(tos);
            primary
        }
    }
}

/// Ensure `socket` is marking datagrams with `conn`'s currently-desired ECN
/// codepoint, re-applying only when it changes from `last`.
///
/// On success `last` is updated to the applied codepoint. If the platform
/// refuses the socket option, ECN is permanently disabled on the connection
/// (all spaces marked unsupported) and `last` is pinned to Not-ECT so the
/// syscall is not retried on every flush.
pub(crate) fn sync_ecn(socket: &UdpSocket, conn: &mut Connection, last: &mut Option<EcnCodepoint>) {
    let desired = conn.ecn_desired_codepoint();
    if *last == Some(desired) {
        return;
    }
    match apply_ecn_tos(socket, desired) {
        Ok(()) => *last = Some(desired),
        Err(_) => {
            conn.mark_ecn_unsupported();
            *last = Some(EcnCodepoint::NotEct);
        }
    }
}
