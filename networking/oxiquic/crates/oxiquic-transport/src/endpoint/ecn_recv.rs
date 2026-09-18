//! Reading the inbound IP ECN codepoint off a UDP socket (RFC 9000 §13.4.1).
//!
//! QUIC's ECN feedback loop is symmetric: an endpoint marks its egress with
//! ECT(0) (see `super::socket_opts`) *and* counts the codepoint of every
//! datagram it receives, echoing the per-packet-number-space totals in ACK-ECN
//! (0x03) frames. Without the receive half a peer's RFC 9000 §13.4.2 validation
//! can never succeed, so the peer stops marking and the network's congestion
//! signal towards this endpoint is lost.
//!
//! The codepoint lives in the IP header, not the UDP payload, so it can only be
//! obtained as `recvmsg` ancillary data (`IP_RECVTOS` on IPv4, `IPV6_RECVTCLASS`
//! on IPv6). `tokio`'s [`tokio::net::UdpSocket`] exposes neither the socket option nor
//! `recvmsg`, so this module reaches through to the OS:
//!
//! * the socket options are set with [`socket2`], which is already used for the
//!   egress marking;
//! * the `recvmsg` call and the control-message walk are done with [`nix`],
//!   whose `recvmsg` is a safe wrapper — which is what lets this crate keep
//!   `#![forbid(unsafe_code)]` while still reading ancillary data. (That
//!   requirement is why the receive half was deferred in earlier releases.)
//! * readiness is still driven by `tokio` via
//!   [`tokio::net::UdpSocket::async_io`] / [`tokio::net::UdpSocket::try_io`],
//!   so the socket stays in the reactor and no thread is blocked.
//!
//! # Platform gating
//!
//! Only platforms whose `IP_RECVTOS` / `IPV6_RECVTCLASS` semantics are known
//! and testable are enabled (Linux, Android, macOS, iOS, FreeBSD). Everywhere
//! else [`enable_ecn_recv`] returns [`EcnRecvUnsupported::Platform`] and the
//! receive path degrades to a plain `recv_from` that reports **no** codepoint.
//!
//! Nothing is ever fabricated. Three states are kept distinct:
//!
//! | situation | reported |
//! |---|---|
//! | socket option refused / platform unsupported | `Err(EcnRecvUnsupported)`, then `None` forever |
//! | option set, but this datagram carried no ancillary data | `None` (nothing counted) |
//! | option set and ancillary data present | `Some(codepoint)` (counted, echoed) |
//!
//! A `None` is *not* Not-ECT: `crate::space::PacketSpace` counts nothing and
//! the ACK stays a plain 0x02 ACK, which is exactly what RFC 9000 §13.4.2.1
//! expects from an endpoint that cannot report codepoints.

use std::fmt;
use std::io;

use crate::ecn::EcnCodepoint;

/// Why inbound ECN codepoints cannot be reported for a socket.
///
/// Returned by [`enable_ecn_recv`]. The endpoint keeps running: it simply never
/// reports a codepoint, and (per RFC 9000 §13.4.2.1) the peer disables its own
/// ECN marking after failing validation.
#[derive(Debug)]
#[non_exhaustive]
pub enum EcnRecvUnsupported {
    /// The build target has no supported `recvmsg`/`cmsg` ECN path.
    Platform {
        /// [`std::env::consts::OS`] of the build target.
        os: &'static str,
    },
    /// The kernel refused `IP_RECVTOS` / `IPV6_RECVTCLASS` on this socket.
    SockOpt(io::Error),
    /// The socket's local address (needed to pick the address family) could not
    /// be read.
    LocalAddr(io::Error),
}

impl fmt::Display for EcnRecvUnsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Platform { os } => {
                write!(f, "inbound ECN codepoints are not readable on {os}")
            }
            Self::SockOpt(e) => write!(f, "IP_RECVTOS / IPV6_RECVTCLASS refused: {e}"),
            Self::LocalAddr(e) => write!(f, "socket local address unavailable: {e}"),
        }
    }
}

impl std::error::Error for EcnRecvUnsupported {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Platform { .. } => None,
            Self::SockOpt(e) | Self::LocalAddr(e) => Some(e),
        }
    }
}

/// The ECN codepoint carried in an IPv4 TOS / IPv6 traffic-class byte.
///
/// The upper six bits are the DSCP field and must not leak into the codepoint,
/// so only the low two bits are used (RFC 3168 §5).
#[must_use]
pub fn ecn_from_tos_byte(tos: u8) -> EcnCodepoint {
    EcnCodepoint::from_tos_bits(tos & 0b11)
}

#[cfg(all(
    unix,
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd"
    )
))]
mod imp {
    use std::io;
    use std::io::IoSliceMut;
    use std::net::{IpAddr, SocketAddr, SocketAddrV6};
    use std::os::fd::AsRawFd;

    use nix::sys::socket::{recvmsg, ControlMessageOwned, MsgFlags, SockaddrStorage};
    use socket2::SockRef;
    use tokio::io::Interest;
    use tokio::net::UdpSocket;

    use super::{ecn_from_tos_byte, EcnRecvUnsupported};
    use crate::ecn::EcnCodepoint;

    /// Ancillary-data scratch space. One TOS/traffic-class control message is
    /// at most `CMSG_SPACE(4)` bytes; 128 bytes leaves room for anything else
    /// the kernel decides to attach without ever truncating ours.
    ///
    /// `repr(align(8))` gives the buffer the alignment a `cmsghdr` wants, which
    /// keeps the kernel's `CMSG_*` pointer arithmetic well-formed.
    #[repr(align(8))]
    struct CmsgBuf([u8; 128]);

    impl CmsgBuf {
        const fn new() -> Self {
            Self([0u8; 128])
        }
    }

    /// Ask the kernel to attach the IP TOS / traffic-class byte of every
    /// inbound datagram as ancillary data.
    ///
    /// For an IPv6 socket the traffic-class option is the primary one and
    /// `IP_RECVTOS` is additionally requested on a best-effort basis so that
    /// IPv4-mapped peers on a dual-stack socket are covered too.
    ///
    /// # Errors
    /// [`EcnRecvUnsupported::LocalAddr`] if the socket's address family cannot
    /// be determined, [`EcnRecvUnsupported::SockOpt`] if the kernel refuses the
    /// primary option for that family.
    pub fn enable_ecn_recv(socket: &UdpSocket) -> Result<(), EcnRecvUnsupported> {
        let local = socket.local_addr().map_err(EcnRecvUnsupported::LocalAddr)?;
        let sock = SockRef::from(socket);
        match local {
            SocketAddr::V4(_) => sock
                .set_recv_tos_v4(true)
                .map_err(EcnRecvUnsupported::SockOpt),
            SocketAddr::V6(_) => {
                let primary = sock
                    .set_recv_tclass_v6(true)
                    .map_err(EcnRecvUnsupported::SockOpt);
                // Dual-stack sockets also receive IPv4-mapped traffic, whose
                // codepoint arrives as an IPv4 TOS control message.
                let _ = sock.set_recv_tos_v4(true);
                primary
            }
        }
    }

    /// Convert a `recvmsg` source address into the `std` representation.
    fn sockaddr_to_std(sa: &SockaddrStorage) -> Option<SocketAddr> {
        if let Some(v4) = sa.as_sockaddr_in() {
            return Some(SocketAddr::new(IpAddr::V4(v4.ip()), v4.port()));
        }
        let v6 = sa.as_sockaddr_in6()?;
        Some(SocketAddr::V6(SocketAddrV6::new(
            v6.ip(),
            v6.port(),
            v6.flowinfo(),
            v6.scope_id(),
        )))
    }

    /// Decode one control message into an ECN codepoint.
    ///
    /// The `(level, type)` pair and the payload width differ between platforms:
    /// Linux reports `(IPPROTO_IP, IP_TOS)` with one byte, the BSDs and macOS
    /// report `(IPPROTO_IP, IP_RECVTOS)` with one byte, and both report
    /// `(IPPROTO_IPV6, IPV6_TCLASS)` with a native-endian `int`. All of these
    /// are accepted; anything else yields `None`.
    fn decode_unknown_cmsg(level: i32, ty: i32, data: &[u8]) -> Option<EcnCodepoint> {
        let tos = match (level, ty) {
            (nix::libc::IPPROTO_IP, t) if t == nix::libc::IP_TOS || t == nix::libc::IP_RECVTOS => {
                *data.first()?
            }
            (nix::libc::IPPROTO_IPV6, t)
                if t == nix::libc::IPV6_TCLASS || t == nix::libc::IPV6_RECVTCLASS =>
            {
                match data.len() {
                    1 => data[0],
                    4 => {
                        let bytes: [u8; 4] = data.get(..4)?.try_into().ok()?;
                        // The kernel writes a native-endian `int`; only the low
                        // byte carries the traffic class.
                        (i32::from_ne_bytes(bytes) & 0xff) as u8
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(ecn_from_tos_byte(tos))
    }

    /// Extract the ECN codepoint from a decoded control message.
    ///
    /// `nix` decodes the TOS / traffic-class messages into typed variants on
    /// Linux and FreeBSD but hands them back as `Unknown` on macOS, so both
    /// shapes are handled.
    fn codepoint_of(cmsg: &ControlMessageOwned) -> Option<EcnCodepoint> {
        match cmsg {
            #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
            ControlMessageOwned::Ipv4Tos(tos) => Some(ecn_from_tos_byte(*tos)),
            #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
            ControlMessageOwned::Ipv6TClass(tc) => Some(ecn_from_tos_byte((*tc & 0xff) as u8)),
            ControlMessageOwned::Unknown(unknown) => decode_unknown_cmsg(
                unknown.cmsg_header.cmsg_level,
                unknown.cmsg_header.cmsg_type,
                &unknown.data_bytes,
            ),
            _ => None,
        }
    }

    /// One non-blocking `recvmsg`, returning the payload length, the source
    /// address and the ECN codepoint the kernel reported (if any).
    ///
    /// Returns [`std::io::ErrorKind::WouldBlock`] when no datagram is queued,
    /// which is what `tokio`'s readiness helpers use to re-arm the reactor.
    fn recvmsg_once(
        socket: &UdpSocket,
        buf: &mut [u8],
    ) -> io::Result<(usize, SocketAddr, Option<EcnCodepoint>)> {
        let mut cmsg = CmsgBuf::new();
        let mut iov = [IoSliceMut::new(buf)];
        let msg = recvmsg::<SockaddrStorage>(
            socket.as_raw_fd(),
            &mut iov,
            Some(&mut cmsg.0),
            MsgFlags::empty(),
        )
        .map_err(|e| io::Error::from_raw_os_error(e as i32))?;

        let from = msg
            .address
            .as_ref()
            .and_then(sockaddr_to_std)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "recvmsg returned no source address",
                )
            })?;

        let mut ecn = None;
        if let Ok(cmsgs) = msg.cmsgs() {
            for cmsg in cmsgs {
                if let Some(cp) = codepoint_of(&cmsg) {
                    ecn = Some(cp);
                }
            }
        }
        Ok((msg.bytes, from, ecn))
    }

    /// Await one datagram, reporting its ECN codepoint when the kernel
    /// attached one.
    ///
    /// # Errors
    /// Any error from the underlying `recvmsg`, or
    /// [`std::io::ErrorKind::InvalidData`] if the kernel returns a datagram
    /// without a source address.
    pub async fn recv_ecn_from(
        socket: &UdpSocket,
        buf: &mut [u8],
    ) -> io::Result<(usize, SocketAddr, Option<EcnCodepoint>)> {
        socket
            .async_io(Interest::READABLE, || recvmsg_once(socket, buf))
            .await
    }

    /// Non-blocking variant of [`recv_ecn_from`]: returns
    /// [`std::io::ErrorKind::WouldBlock`] when no datagram is queued.
    ///
    /// # Errors
    /// As [`recv_ecn_from`], plus `WouldBlock` when the socket is empty.
    pub fn try_recv_ecn_from(
        socket: &UdpSocket,
        buf: &mut [u8],
    ) -> io::Result<(usize, SocketAddr, Option<EcnCodepoint>)> {
        socket.try_io(Interest::READABLE, || recvmsg_once(socket, buf))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ipv4_tos_cmsg_decodes_every_codepoint() {
            // RFC 3168 §5: the ECN field is the low two bits of the TOS byte.
            for (tos, want) in [
                (0x00u8, EcnCodepoint::NotEct),
                (0x01, EcnCodepoint::Ect1),
                (0x02, EcnCodepoint::Ect0),
                (0x03, EcnCodepoint::Ce),
            ] {
                assert_eq!(
                    decode_unknown_cmsg(nix::libc::IPPROTO_IP, nix::libc::IP_TOS, &[tos]),
                    Some(want),
                    "IP_TOS {tos:#04x}"
                );
                assert_eq!(
                    decode_unknown_cmsg(nix::libc::IPPROTO_IP, nix::libc::IP_RECVTOS, &[tos]),
                    Some(want),
                    "IP_RECVTOS {tos:#04x}"
                );
            }
        }

        #[test]
        fn dscp_bits_never_leak_into_the_codepoint() {
            // 0xA2 = DSCP AF41 (0b101000) with ECT(0) in the low two bits.
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IP, nix::libc::IP_TOS, &[0xA2]),
                Some(EcnCodepoint::Ect0)
            );
            // 0xB8 = DSCP EF with Not-ECT.
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IP, nix::libc::IP_TOS, &[0xB8]),
                Some(EcnCodepoint::NotEct)
            );
        }

        #[test]
        fn ipv6_tclass_cmsg_accepts_both_widths() {
            // Native-endian `int`, as Linux/macOS deliver IPV6_TCLASS.
            let ce = 0x03i32.to_ne_bytes();
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IPV6, nix::libc::IPV6_TCLASS, &ce),
                Some(EcnCodepoint::Ce)
            );
            let ect1 = 0xA1i32.to_ne_bytes();
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IPV6, nix::libc::IPV6_TCLASS, &ect1),
                Some(EcnCodepoint::Ect1)
            );
            // Single-byte form.
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IPV6, nix::libc::IPV6_TCLASS, &[0x02]),
                Some(EcnCodepoint::Ect0)
            );
        }

        #[test]
        fn unrelated_or_truncated_cmsgs_report_nothing() {
            // A control message we do not understand must never be guessed at.
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IP, nix::libc::IP_TTL, &[64]),
                None
            );
            assert_eq!(
                decode_unknown_cmsg(nix::libc::IPPROTO_IP, nix::libc::IP_TOS, &[]),
                None
            );
            assert_eq!(
                decode_unknown_cmsg(
                    nix::libc::IPPROTO_IPV6,
                    nix::libc::IPV6_TCLASS,
                    &[0x00, 0x00]
                ),
                None
            );
        }
    }
}

#[cfg(not(all(
    unix,
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd"
    )
)))]
mod imp {
    use std::io;
    use std::net::SocketAddr;

    use tokio::net::UdpSocket;

    use super::EcnRecvUnsupported;
    use crate::ecn::EcnCodepoint;

    /// Inbound ECN codepoints cannot be read on this target.
    ///
    /// # Errors
    /// Always [`EcnRecvUnsupported::Platform`].
    pub fn enable_ecn_recv(socket: &UdpSocket) -> Result<(), EcnRecvUnsupported> {
        let _ = socket;
        Err(EcnRecvUnsupported::Platform {
            os: std::env::consts::OS,
        })
    }

    /// Await one datagram. The codepoint is always `None` on this target — it
    /// is never guessed at.
    ///
    /// # Errors
    /// Any error from the underlying `recv_from`.
    pub async fn recv_ecn_from(
        socket: &UdpSocket,
        buf: &mut [u8],
    ) -> io::Result<(usize, SocketAddr, Option<EcnCodepoint>)> {
        let (len, from) = socket.recv_from(buf).await?;
        Ok((len, from, None))
    }

    /// Non-blocking variant of [`recv_ecn_from`].
    ///
    /// # Errors
    /// Any error from the underlying `try_recv_from`, including `WouldBlock`.
    pub fn try_recv_ecn_from(
        socket: &UdpSocket,
        buf: &mut [u8],
    ) -> io::Result<(usize, SocketAddr, Option<EcnCodepoint>)> {
        let (len, from) = socket.try_recv_from(buf)?;
        Ok((len, from, None))
    }
}

pub use imp::{enable_ecn_recv, recv_ecn_from, try_recv_ecn_from};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tos_byte_masks_off_the_dscp_field() {
        assert_eq!(ecn_from_tos_byte(0x00), EcnCodepoint::NotEct);
        assert_eq!(ecn_from_tos_byte(0x01), EcnCodepoint::Ect1);
        assert_eq!(ecn_from_tos_byte(0x02), EcnCodepoint::Ect0);
        assert_eq!(ecn_from_tos_byte(0x03), EcnCodepoint::Ce);
        // DSCP CS3 (0b011000 << 2 = 0x60) with CE.
        assert_eq!(ecn_from_tos_byte(0x63), EcnCodepoint::Ce);
        assert_eq!(ecn_from_tos_byte(0xFC), EcnCodepoint::NotEct);
    }

    #[test]
    fn unsupported_reason_is_typed_and_displays() {
        let err = EcnRecvUnsupported::Platform { os: "plan9" };
        assert!(err.to_string().contains("plan9"));
        let err = EcnRecvUnsupported::SockOpt(io::Error::from(io::ErrorKind::PermissionDenied));
        assert!(err.to_string().contains("IP_RECVTOS"));
        assert!(std::error::Error::source(&err).is_some());
    }
}
