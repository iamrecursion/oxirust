//! Scatter/gather UDP I/O using `sendmmsg`/`recvmmsg` on Linux.
//!
//! `UdpScatterGather` wraps a standard UDP socket and exposes a `send_many`
//! method.  On Linux the implementation calls the `sendmmsg(2)` syscall,
//! which sends multiple datagrams in a single kernel entry; on all other
//! platforms it falls back to an ordinary `send_to` loop.
//!
//! # Platform notes
//!
//! - `sendmmsg` is Linux ≥ 3.0 only.  The feature is entirely gated behind
//!   `#[cfg(target_os = "linux")]`.
//! - No `libc` symbols appear in non-Linux builds; the crate remains 100 %
//!   portable.

use crate::error::{VideoIpError, VideoIpResult};
use std::net::{SocketAddr, UdpSocket};

// ============================================================================
// Linux-specific imports
// ============================================================================

#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;

// ============================================================================
// UdpScatterGather
// ============================================================================

/// UDP socket wrapper with batch send support.
///
/// On Linux, `send_many` uses a single `sendmmsg(2)` call.  On other
/// platforms it falls back to a sequential loop of `send_to`.
pub struct UdpScatterGather {
    socket: UdpSocket,
}

impl UdpScatterGather {
    /// Binds a new UDP socket to `addr`.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket cannot be created or bound.
    pub fn bind(addr: SocketAddr) -> VideoIpResult<Self> {
        let socket = UdpSocket::bind(addr).map_err(|e| VideoIpError::Transport(e.to_string()))?;
        Ok(Self { socket })
    }

    /// Creates a `UdpScatterGather` from an already-bound `UdpSocket`.
    #[must_use]
    pub fn from_socket(socket: UdpSocket) -> Self {
        Self { socket }
    }

    /// Returns the local address.
    ///
    /// # Errors
    ///
    /// Returns an error if `getsockname` fails.
    pub fn local_addr(&self) -> VideoIpResult<SocketAddr> {
        self.socket
            .local_addr()
            .map_err(|e| VideoIpError::Transport(e.to_string()))
    }

    /// Sends multiple datagrams to `addr` in a single operation where possible.
    ///
    /// On Linux this issues one `sendmmsg(2)` syscall.  On other platforms
    /// it issues one `send_to` per message.
    ///
    /// Returns the number of messages successfully sent.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying syscall / I/O fails.
    pub fn send_many(&self, msgs: &[&[u8]], addr: SocketAddr) -> VideoIpResult<usize> {
        if msgs.is_empty() {
            return Ok(0);
        }

        #[cfg(target_os = "linux")]
        return self.send_many_linux(msgs, addr);

        #[cfg(not(target_os = "linux"))]
        return self.send_many_fallback(msgs, addr);
    }

    // ------------------------------------------------------------------
    // Linux path: sendmmsg(2)
    // ------------------------------------------------------------------

    #[cfg(target_os = "linux")]
    #[allow(unsafe_code)]
    fn send_many_linux(&self, msgs: &[&[u8]], addr: SocketAddr) -> VideoIpResult<usize> {
        use libc::{
            iovec, mmsghdr, msghdr, sendmmsg, sockaddr_in, sockaddr_in6, AF_INET, AF_INET6,
            MSG_DONTWAIT,
        };
        use std::mem;

        let fd = self.socket.as_raw_fd();

        // Build the sockaddr from the Rust SocketAddr.
        // We use two buffers so the address lives long enough.
        let mut sa_in: sockaddr_in = unsafe { mem::zeroed() };
        let mut sa_in6: sockaddr_in6 = unsafe { mem::zeroed() };
        let (sa_ptr, sa_len): (*const libc::sockaddr, libc::socklen_t) = match addr {
            SocketAddr::V4(v4) => {
                sa_in.sin_family = AF_INET as libc::sa_family_t;
                sa_in.sin_port = v4.port().to_be();
                sa_in.sin_addr.s_addr = u32::from_ne_bytes(v4.ip().octets());
                (
                    (&sa_in as *const sockaddr_in).cast::<libc::sockaddr>(),
                    mem::size_of::<sockaddr_in>() as libc::socklen_t,
                )
            }
            SocketAddr::V6(v6) => {
                sa_in6.sin6_family = AF_INET6 as libc::sa_family_t;
                sa_in6.sin6_port = v6.port().to_be();
                sa_in6.sin6_addr.s6_addr = v6.ip().octets();
                (
                    (&sa_in6 as *const sockaddr_in6).cast::<libc::sockaddr>(),
                    mem::size_of::<sockaddr_in6>() as libc::socklen_t,
                )
            }
        };

        // Build parallel iovec and mmsghdr arrays.
        let mut iovecs: Vec<iovec> = msgs
            .iter()
            .map(|m| iovec {
                iov_base: m.as_ptr() as *mut libc::c_void,
                iov_len: m.len(),
            })
            .collect();

        let mut mhdr_vec: Vec<mmsghdr> = iovecs
            .iter_mut()
            .map(|iov| mmsghdr {
                msg_hdr: msghdr {
                    msg_name: sa_ptr as *mut libc::c_void,
                    msg_namelen: sa_len,
                    msg_iov: iov as *mut iovec,
                    msg_iovlen: 1,
                    msg_control: std::ptr::null_mut(),
                    msg_controllen: 0,
                    msg_flags: 0,
                },
                msg_len: 0,
            })
            .collect();

        // SAFETY: fd is valid, iovecs/mhdr_vec outlive the call, and the
        // pointers remain stable (Vec does not reallocate here).
        let sent = unsafe {
            sendmmsg(
                fd,
                mhdr_vec.as_mut_ptr(),
                mhdr_vec.len() as libc::c_uint,
                MSG_DONTWAIT,
            )
        };

        if sent < 0 {
            let err = std::io::Error::last_os_error();
            // EAGAIN / EWOULDBLOCK: nothing sent yet — return 0 rather than error.
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(0);
            }
            return Err(VideoIpError::Transport(format!("sendmmsg failed: {err}")));
        }

        Ok(sent as usize)
    }

    // ------------------------------------------------------------------
    // Fallback path: sequential send_to loop
    // ------------------------------------------------------------------

    #[cfg(not(target_os = "linux"))]
    fn send_many_fallback(&self, msgs: &[&[u8]], addr: SocketAddr) -> VideoIpResult<usize> {
        let mut count = 0usize;
        for msg in msgs {
            self.socket
                .send_to(msg, addr)
                .map_err(|e| VideoIpError::Transport(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    /// Receives a single datagram into `buf`.
    ///
    /// Returns `(bytes_received, sender_addr)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `recv_from` fails.
    pub fn recv(&self, buf: &mut [u8]) -> VideoIpResult<(usize, SocketAddr)> {
        self.socket
            .recv_from(buf)
            .map_err(|e| VideoIpError::Transport(e.to_string()))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddrV4};

    fn localhost_v4(port: u16) -> SocketAddr {
        SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port))
    }

    // ── Linux-specific: sendmmsg actually sends multiple messages ─────────────

    #[cfg(target_os = "linux")]
    #[test]
    fn test_sendmmsg_sends_multiple() {
        let sender = UdpScatterGather::bind(localhost_v4(0)).expect("bind sender");
        let receiver_sock = UdpSocket::bind(localhost_v4(0)).expect("bind receiver");
        let recv_addr = receiver_sock.local_addr().expect("recv local_addr");

        // Non-blocking on receiver so we can drain without blocking.
        receiver_sock
            .set_nonblocking(true)
            .expect("set nonblocking");

        let messages: Vec<Vec<u8>> = vec![b"alpha".to_vec(), b"beta".to_vec(), b"gamma".to_vec()];
        let refs: Vec<&[u8]> = messages.iter().map(|v| v.as_slice()).collect();

        let sent = sender.send_many(&refs, recv_addr).expect("send_many");
        assert_eq!(sent, 3, "sendmmsg should report 3 messages sent");

        // Receive and verify.
        let mut buf = [0u8; 256];
        let mut received_data: Vec<Vec<u8>> = Vec::new();
        for _ in 0..3 {
            match receiver_sock.recv_from(&mut buf) {
                Ok((n, _)) => received_data.push(buf[..n].to_vec()),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // might not arrive instantly — give it a brief spin
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    if let Ok((n, _)) = receiver_sock.recv_from(&mut buf) {
                        received_data.push(buf[..n].to_vec());
                    }
                }
                Err(e) => panic!("recv_from failed: {e}"),
            }
        }
        // All three messages should have been received.
        assert_eq!(received_data.len(), 3);
    }

    // ── Cross-platform fallback (always runs) ─────────────────────────────────

    #[test]
    fn test_udp_scatter_gather_fallback_non_linux() {
        let sender = UdpScatterGather::bind(localhost_v4(0)).expect("bind sender");
        let receiver_sock = UdpSocket::bind(localhost_v4(0)).expect("bind receiver");
        let recv_addr = receiver_sock.local_addr().expect("recv local_addr");

        let messages: Vec<Vec<u8>> = vec![b"hello".to_vec(), b"world".to_vec()];
        let refs: Vec<&[u8]> = messages.iter().map(|v| v.as_slice()).collect();

        // Call send_many — on non-Linux this always uses the fallback loop.
        let sent = sender.send_many(&refs, recv_addr).expect("send_many");
        assert_eq!(sent, 2);

        // Drain.
        let mut buf = [0u8; 64];
        let (n1, _) = receiver_sock.recv_from(&mut buf).expect("recv 1");
        let data1 = buf[..n1].to_vec();
        let (n2, _) = receiver_sock.recv_from(&mut buf).expect("recv 2");
        let data2 = buf[..n2].to_vec();

        assert_eq!(data1, b"hello");
        assert_eq!(data2, b"world");
    }

    #[test]
    fn test_udp_scatter_gather_send_empty() {
        let sender = UdpScatterGather::bind(localhost_v4(0)).expect("bind sender");
        let dummy_addr = localhost_v4(12345);
        let sent = sender.send_many(&[], dummy_addr).expect("send_many empty");
        assert_eq!(sent, 0);
    }

    #[test]
    fn test_udp_scatter_gather_local_addr() {
        let sg = UdpScatterGather::bind(localhost_v4(0)).expect("bind");
        let addr = sg.local_addr().expect("local_addr");
        assert_eq!(addr.ip(), Ipv4Addr::LOCALHOST);
        assert!(addr.port() > 0);
    }
}
