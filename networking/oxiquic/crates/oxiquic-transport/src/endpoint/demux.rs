//! Server-side demultiplexing: the accept loop that fans inbound datagrams out
//! to per-connection tasks.
//!
//! `run_server_demux` owns the listening UDP socket and routes every datagram to
//! the task that owns the connection it belongs to, keyed first by the client's
//! original Destination Connection ID and then by the connection ID this
//! endpoint issued. It also implements the RFC 9000 §8.1 Retry branch (token
//! issue, token validation and the §7.3 connection-ID transcript handed to
//! [`Connection::new_server_after_retry`]) and the routing-table garbage
//! collection driven by `CidRouteUpdate` messages from the connection tasks.
//!
//! Split out of `endpoint/mod.rs` so both files stay well under the 2000-line
//! policy cap; everything here keeps its original crate-internal visibility, so
//! the module boundary is purely file organisation.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Demux routing-table update helper
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias for the per-connection inbound datagram sender.
type ConnTx = mpsc::Sender<InboundDatagram>;

/// Apply one `CidRouteUpdate` to the two demux routing tables.
///
/// Extracted from the hot `cid_route_rx` drain loop so the same logic can be
/// exercised in unit tests without a live UDP socket or tokio runtime.
///
/// Returns `true` if the update was applied successfully, `false` if the CID
/// bytes had the wrong length for a `Register`/`Unregister` event (impossible
/// in normal operation, but guards against future refactors).
fn apply_cid_route_update(
    initial_map: &mut HashMap<Vec<u8>, ConnTx>,
    local_cid_map: &mut HashMap<[u8; LOCAL_CID_LEN], ConnTx>,
    update: CidRouteUpdate,
) -> bool {
    match update.event {
        CidEvent::Register(cid) => {
            let key_bytes = cid.as_bytes();
            if key_bytes.len() == LOCAL_CID_LEN {
                let key: [u8; LOCAL_CID_LEN] = match key_bytes.try_into() {
                    Ok(k) => k,
                    Err(_) => return false,
                };
                local_cid_map.insert(key, update.conn_tx);
                true
            } else {
                false
            }
        }
        CidEvent::Unregister(cid) => {
            let key_bytes = cid.as_bytes();
            if key_bytes.len() == LOCAL_CID_LEN {
                let key: [u8; LOCAL_CID_LEN] = match key_bytes.try_into() {
                    Ok(k) => k,
                    Err(_) => return false,
                };
                local_cid_map.remove(&key);
                true
            } else {
                false
            }
        }
        // Handshake completed; evict the initial DCID from the routing
        // table. This is the primary GC path — the lazy Closed(_) removal
        // on the Initial routing entry acts only as a defence-in-depth
        // backstop for cases where this event is never delivered.
        CidEvent::InitialRetired(dcid) => {
            initial_map.remove(&dcid);
            true
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// run_server_demux
// ─────────────────────────────────────────────────────────────────────────────

/// Background demux task: reads every datagram from the shared UDP socket and
/// routes it to the appropriate per-connection channel.
///
/// For each new connection the demux **synchronously** constructs the
/// `Connection` state machine (no I/O), extracts the server-issued `local_cid`,
/// creates a single `(hs_tx, hs_rx)` channel pair and registers it in **both**
/// maps in the same task tick — eliminating any race where Handshake packets
/// arrive before the notify is processed:
///
/// * `initial_map[dcid] = hs_tx.clone()` — routes client Initial retransmits.
/// * `local_cid_map[local_cid_bytes] = hs_tx` — routes Handshake + 1-RTT
///   packets (client switches to server's issued CID as DCID immediately after
///   seeing it in the server's first Initial).
pub(super) async fn run_server_demux(
    socket: Arc<UdpSocket>,
    config: Arc<ServerConfig>,
    mut transport: TransportConfig,
    accept_tx: mpsc::Sender<Result<QuicConnection, OxiQuicError>>,
    ecn_recv: bool,
) {
    // Map from client's initial DCID to the per-connection inbound channel.
    let mut initial_map: HashMap<Vec<u8>, ConnTx> = HashMap::new();
    // Map from server-issued 8-byte local_cid to the per-connection inbound channel.
    // This covers both long-header Handshake packets and short-header 1-RTT packets.
    let mut local_cid_map: HashMap<[u8; LOCAL_CID_LEN], ConnTx> = HashMap::new();

    // Channel for CID routing updates from connection tasks.
    // Capacity: 256 pending updates is ample for any realistic connection count.
    let (cid_route_tx, mut cid_route_rx) = mpsc::channel::<CidRouteUpdate>(256);

    let mut buf = vec![0u8; RECV_BUF];

    loop {
        // If accept_tx is closed no consumer is waiting; shut down the demux.
        if accept_tx.is_closed() {
            break;
        }

        // Drain any pending CID routing updates before blocking on the socket.
        // This keeps the routing table up-to-date for the next datagram.
        loop {
            match cid_route_rx.try_recv() {
                Ok(update) => {
                    apply_cid_route_update(&mut initial_map, &mut local_cid_map, update);
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        // RFC 9000 §13.4.1: when the kernel accepted IP_RECVTOS /
        // IPV6_RECVTCLASS the datagram's ECN codepoint is read alongside its
        // payload and travels with it to the owning connection task, which
        // counts it into the packet-number space of every packet it decrypts
        // out of this datagram. Otherwise `ecn` is `None` — never a fabricated
        // Not-ECT.
        let result = if ecn_recv {
            ecn_recv::recv_ecn_from(&socket, &mut buf).await
        } else {
            socket
                .recv_from(&mut buf)
                .await
                .map(|(len, from)| (len, from, None))
        };
        let (len, peer_addr, ecn) = match result {
            Ok(v) => v,
            Err(_) => {
                // Socket receive error; keep the demux running.
                continue;
            }
        };
        let datagram = buf[..len].to_vec();

        // Classify the packet and extract its DCID without decrypting.
        let (pkt_type, dcid) = match crate::packet::peek_dcid(&datagram, LOCAL_CID_LEN) {
            Ok(v) => v,
            Err(_) => {
                // Datagram too short or malformed; skip it.
                continue;
            }
        };

        // ── 1-RTT short header ──────────────────────────────────────────────
        if pkt_type == oxiquic_core::PacketType::Short {
            let key: [u8; LOCAL_CID_LEN] = match dcid.as_slice().try_into() {
                Ok(k) => k,
                Err(_) => continue,
            };
            if let Some(tx) = local_cid_map.get(&key) {
                match tx.try_send(InboundDatagram {
                    data: datagram,
                    from: peer_addr,
                    ecn,
                }) {
                    Ok(()) => {}
                    // Channel full: QUIC retransmits will re-deliver; drop this copy.
                    Err(mpsc::error::TrySendError::Full(_)) => {}
                    // Receiver gone: connection task has exited; clean up the map.
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        local_cid_map.remove(&key);
                    }
                }
            }
            // Unknown short-header CID: stale/misrouted datagram, silently drop.
            continue;
        }

        // ── Long-header packets (Initial / Handshake) ───────────────────────
        // Check local_cid_map first: the client switches to the server's issued
        // CID as DCID after receiving the server's first Initial. All subsequent
        // long-header packets (Handshake) carry the server local_cid as DCID.
        if dcid.len() == LOCAL_CID_LEN {
            let key: [u8; LOCAL_CID_LEN] = match dcid.as_slice().try_into() {
                Ok(k) => k,
                Err(_) => continue,
            };
            if let Some(tx) = local_cid_map.get(&key) {
                match tx.try_send(InboundDatagram {
                    data: datagram,
                    from: peer_addr,
                    ecn,
                }) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {}
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        local_cid_map.remove(&key);
                    }
                }
                continue;
            }
        }

        // Route Initial to an existing or new handshake task.
        if pkt_type != oxiquic_core::PacketType::Initial {
            // Non-Initial long-header for unknown DCID; silently drop.
            continue;
        }

        // Existing handshake in progress for this DCID?
        if let Some(tx) = initial_map.get(&dcid) {
            match tx.try_send(InboundDatagram {
                data: datagram,
                from: peer_addr,
                ecn,
            }) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {}
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    initial_map.remove(&dcid);
                }
            }
            continue;
        }

        // ── New connection: build Connection, register both maps, spawn task ─
        let (initial_version, initial_dcid_parsed, scid_parsed) =
            match parse_client_initial_ids(&datagram) {
                Some(ids) => ids,
                None => continue,
            };

        // RFC 9000 §17.2.1: if the client's Initial carries an unsupported
        // version, respond with a Version Negotiation packet and do NOT create
        // a connection.
        if initial_version != QUIC_V1 {
            let vn = crate::packet::encode_version_negotiation(
                &scid_parsed,         // client's SCID → VN's DCID
                &initial_dcid_parsed, // client's DCID → VN's SCID
                SUPPORTED_VERSIONS,
            );
            let _ = socket.send_to(&vn, peer_addr).await;
            continue;
        }

        // ── RFC 9000 §8.1 Retry address validation ───────────────────────────
        if transport.get_retry_enabled() {
            let token = parse_initial_token(&datagram).unwrap_or_default();

            if token.is_empty() {
                // No token: send Retry, do not create a connection yet.
                let retry_scid = random_scid_for_retry(config.crypto_provider().secure_random);
                let retry_token = transport.generate_retry_token(&initial_dcid_parsed, peer_addr);
                if let Some(retry_pkt) = encode_retry_packet(
                    &retry_scid,
                    &scid_parsed,         // echo client's SCID back as DCID
                    &initial_dcid_parsed, // odcid for integrity tag
                    &retry_token,
                ) {
                    let _ = socket.send_to(&retry_pkt, peer_addr).await;
                }
                // Do NOT enter initial_map: client must retry with token.
                continue;
            }

            // Token present: validate it.
            match transport.validate_retry_token(&token, peer_addr) {
                Some(odcid) => {
                    // Token valid.
                    //
                    // RFC 9001 §5.2: after a Retry, both the client and the server
                    // derive Initial keys from the Retry's SCID — which is the
                    // DCID of the client's second Initial (`initial_dcid_parsed`).
                    // The ODCID embedded in the token is the DCID of the
                    // client's *first* Initial and is carried into the TLS
                    // handshake as `original_destination_connection_id`
                    // (RFC 9000 §7.3) together with the Retry SCID, so the
                    // client can prove the Retry was not forged.
                    let transcript = RetryTranscript {
                        original_dcid: oxiquic_core::ConnectionId::new(odcid),
                        retry_scid: oxiquic_core::ConnectionId::new(initial_dcid_parsed.clone()),
                    };
                    let params = transport.to_transport_params();
                    let mtu_config = MtuConfig {
                        max_mtu: transport.get_max_mtu(),
                        discovery_enabled: true,
                    };
                    let server_cfg_early = apply_early_data_config(Arc::clone(&config), &transport);
                    let conn = match Connection::new_server_with_datagram_buf(
                        server_cfg_early,
                        oxiquic_core::ConnectionId::new(initial_dcid_parsed.clone()),
                        oxiquic_core::ConnectionId::new(scid_parsed),
                        peer_addr,
                        params,
                        mtu_config,
                        transport.get_congestion_controller(),
                        transport.get_datagram_receive_buffer_size(),
                        Some(transcript),
                    ) {
                        Ok(mut c) => {
                            // RFC 9000 §8.1: a valid Retry token already proves
                            // the client owns this address, so the three-times
                            // amplification limit does not apply.
                            c.mark_address_validated();
                            c
                        }
                        Err(e) => {
                            let _ = accept_tx.send(Err(e)).await;
                            continue;
                        }
                    };
                    let local_cid_bytes: [u8; LOCAL_CID_LEN] =
                        match conn.local_cid().as_bytes().try_into() {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                    let (hs_tx, hs_rx) = mpsc::channel::<InboundDatagram>(16384);
                    // Route future datagrams: after Retry the client uses the
                    // Retry SCID as DCID (`initial_dcid_parsed`).
                    // Clone before insert to preserve the key for `initial_dcid` arg.
                    let initial_dcid_key = initial_dcid_parsed.clone();
                    initial_map.insert(initial_dcid_parsed, hs_tx.clone());
                    local_cid_map.insert(local_cid_bytes, hs_tx.clone());
                    let _ = hs_tx.try_send(InboundDatagram {
                        data: datagram,
                        from: peer_addr,
                        ecn,
                    });
                    tokio::spawn(run_server_handshake(
                        conn,
                        peer_addr,
                        Arc::clone(&socket),
                        ServerHandshakeArgs {
                            hs_tx,
                            hs_rx,
                            accept_tx: accept_tx.clone(),
                            keep_alive_interval: transport.get_keep_alive_interval(),
                            cid_route_tx: cid_route_tx.clone(),
                            initial_dcid: initial_dcid_key,
                        },
                    ));
                    continue;
                }
                None => {
                    // Invalid token: send a fresh Retry.
                    let retry_scid = random_scid_for_retry(config.crypto_provider().secure_random);
                    let retry_token =
                        transport.generate_retry_token(&initial_dcid_parsed, peer_addr);
                    if let Some(retry_pkt) = encode_retry_packet(
                        &retry_scid,
                        &scid_parsed,
                        &initial_dcid_parsed,
                        &retry_token,
                    ) {
                        let _ = socket.send_to(&retry_pkt, peer_addr).await;
                    }
                    continue;
                }
            }
        }

        // ─── No Retry required: create connection immediately ─────────────────
        let params = transport.to_transport_params();
        let mtu_config = MtuConfig {
            max_mtu: transport.get_max_mtu(),
            discovery_enabled: true,
        };
        let server_cfg_early = apply_early_data_config(Arc::clone(&config), &transport);
        let conn = match Connection::new_server_with_datagram_buf(
            server_cfg_early,
            oxiquic_core::ConnectionId::new(initial_dcid_parsed.clone()),
            oxiquic_core::ConnectionId::new(scid_parsed),
            peer_addr,
            params,
            mtu_config,
            transport.get_congestion_controller(),
            transport.get_datagram_receive_buffer_size(),
            // No Retry was issued for this connection.
            None,
        ) {
            Ok(c) => c,
            Err(e) => {
                let _ = accept_tx.send(Err(e)).await;
                continue;
            }
        };

        let local_cid_bytes: [u8; LOCAL_CID_LEN] = match conn.local_cid().as_bytes().try_into() {
            Ok(b) => b,
            Err(_) => continue,
        };

        // Single channel for this connection's entire lifetime (handshake + 1-RTT).
        // Capacity of 16 384 datagrams (each up to 2 KiB) provides ~32 MiB of
        // buffering, avoiding spurious drops during bulk transfers where the
        // connection task may momentarily lag behind the demux.
        let (hs_tx, hs_rx) = mpsc::channel::<InboundDatagram>(16384);

        // Register in BOTH maps atomically (same demux-task tick): this prevents
        // the race where Handshake packets arrive before the local_cid is known.
        // GC is proactive: `run_server_handshake` sends `CidEvent::InitialRetired`
        // after the handshake completes so the demux removes the entry immediately.
        // The lazy closed-channel removal below acts as a defence-in-depth backstop.
        let initial_dcid_key = initial_dcid_parsed.clone();
        initial_map.insert(initial_dcid_parsed, hs_tx.clone());
        local_cid_map.insert(local_cid_bytes, hs_tx.clone());

        // Forward the first datagram into the channel.
        let _ = hs_tx.try_send(InboundDatagram {
            data: datagram,
            from: peer_addr,
            ecn,
        });

        tokio::spawn(run_server_handshake(
            conn,
            peer_addr,
            Arc::clone(&socket),
            ServerHandshakeArgs {
                hs_tx,
                hs_rx,
                accept_tx: accept_tx.clone(),
                keep_alive_interval: transport.get_keep_alive_interval(),
                cid_route_tx: cid_route_tx.clone(),
                initial_dcid: initial_dcid_key,
            },
        ));
    }
}

/// Apply `max_early_data_size` from transport config to the server TLS config,
/// returning a (possibly new) Arc. Clones the config only when needed.
fn apply_early_data_config(
    config: Arc<ServerConfig>,
    transport: &TransportConfig,
) -> Arc<ServerConfig> {
    let size = transport.get_max_early_data_size();
    if size == 0 {
        return config;
    }
    // Only clone if the current setting differs.
    if config.max_early_data_size == size {
        return config;
    }
    let mut cfg = (*config).clone();
    cfg.max_early_data_size = size;
    Arc::new(cfg)
}

/// Generate a random 8-byte SCID for a Retry packet using the server's CSPRNG.
fn random_scid_for_retry(rng: &dyn rustls::crypto::SecureRandom) -> Vec<u8> {
    let mut bytes = [0u8; 8];
    if rng.fill(&mut bytes).is_ok() {
        bytes.to_vec()
    } else {
        // Fallback: derive from time (does not panic but is not cryptographically strong).
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        (0u32..8)
            .map(|i| t.wrapping_add(i.wrapping_mul(0x9e37_79b9)) as u8)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// run_server_handshake
// ─────────────────────────────────────────────────────────────────────────────

/// Channels and routing metadata bundled for [`run_server_handshake`] to keep
/// the argument count within clippy's `too_many_arguments` limit.
struct ServerHandshakeArgs {
    /// Sender side of the per-connection inbound channel (registered in both
    /// `initial_map` and `local_cid_map` by the demux before spawning).
    hs_tx: ConnTx,
    /// Receiver side of the per-connection inbound channel.
    hs_rx: mpsc::Receiver<InboundDatagram>,
    /// Channel on which a successfully-established [`QuicConnection`] (or an
    /// error) is delivered to the accept loop.
    accept_tx: mpsc::Sender<Result<QuicConnection, OxiQuicError>>,
    /// Optional keep-alive ping interval forwarded from [`TransportConfig`].
    keep_alive_interval: Option<Duration>,
    /// Channel for notifying the demux of CID routing-table updates.
    cid_route_tx: mpsc::Sender<CidRouteUpdate>,
    /// The client's initial DCID bytes — the key used in `initial_map`.
    /// Sent as `CidEvent::InitialRetired` after handshake completes (success or
    /// failure) so the demux can reclaim the `initial_map` entry immediately
    /// rather than waiting for the channel to close.
    initial_dcid: Vec<u8>,
}

/// Per-connection handshake task spawned by the demux for each new Initial.
///
/// The demux has already:
///
/// * constructed the [`Connection`] state machine synchronously,
/// * registered `hs_tx` in both `initial_map` and `local_cid_map`, and
/// * forwarded the first datagram into `hs_rx`.
///
/// This task drives the handshake to completion and sends the fully-established
/// [`QuicConnection`] on `accept_tx`.
async fn run_server_handshake(
    conn: Connection,
    peer_addr: SocketAddr,
    socket: Arc<UdpSocket>,
    args: ServerHandshakeArgs,
) {
    let ServerHandshakeArgs {
        hs_tx,
        hs_rx,
        accept_tx,
        keep_alive_interval,
        cid_route_tx,
        initial_dcid,
    } = args;
    // The single `hs_rx` channel carries all datagrams for this connection —
    // Initial retransmits, Handshake packets, and post-handshake 1-RTT packets.
    // The demux registered this channel in both maps before spawning us, so no
    // routing race can occur.

    // Clone routing handles BEFORE they are consumed by `with_cid_routing`.
    // We need them after the handshake to send the `InitialRetired` event.
    let retire_tx = cid_route_tx.clone();
    let retire_conn_tx = hs_tx.clone();

    let inbound = InboundSource::Channel(hs_rx);
    let driver = ConnectionDriver::new(Arc::clone(&socket), inbound, conn, Some(peer_addr))
        .with_cid_routing(cid_route_tx, hs_tx);
    let mut driver = driver;

    // Drive the handshake.  The first datagram is already in the channel (the
    // demux forwarded it via `hs_tx.try_send` before spawning this task).
    let handshake_result = driver.run_handshake().await;

    // Proactively remove the initial DCID from `initial_map` now that the
    // handshake has completed (success or failure). Without this, long-lived
    // servers accumulate stale entries because the lazy closed-channel removal
    // at the demux never fires for successful connections (the channel remains
    // open through the post-handshake 1-RTT lifetime).
    let retire_update = CidRouteUpdate {
        event: CidEvent::InitialRetired(initial_dcid),
        conn_tx: retire_conn_tx,
    };
    // best-effort: if the demux is gone, we still complete cleanly.
    let _ = retire_tx.try_send(retire_update);

    if let Err(e) = handshake_result {
        let _ = accept_tx.send(Err(e)).await;
        return;
    }

    driver.conn.set_keep_alive_interval(keep_alive_interval);
    let _ = accept_tx.send(Ok(QuicConnection::new(driver))).await;
}

/// Extract `(version, dcid, scid)` from a client's first long-header Initial
/// packet without decrypting it.  Returns `None` for short-header or truncated
/// datagrams.
fn parse_client_initial_ids(datagram: &[u8]) -> Option<(u32, Vec<u8>, Vec<u8>)> {
    use crate::coding::Buf;
    let first = *datagram.first()?;
    if first & 0x80 == 0 {
        return None; // short header: not a first-flight Initial
    }
    let mut buf = Buf::new(datagram);
    let _ = buf.get_u8().ok()?;
    let version = buf.get_u32().ok()?;
    let dcid_len = buf.get_u8().ok()? as usize;
    let dcid = buf.get_bytes(dcid_len).ok()?.to_vec();
    let scid_len = buf.get_u8().ok()? as usize;
    let scid = buf.get_bytes(scid_len).ok()?.to_vec();
    Some((version, dcid, scid))
}

/// A routing update from a connection task to the demux: a CID has been
/// issued or retired and the demux's `local_cid_map` must be updated.
pub(super) struct CidRouteUpdate {
    /// The CID routing event.
    pub(super) event: CidEvent,
    /// The inbound channel sender for this connection, so the demux can
    /// register (or deregister) the CID in `local_cid_map`.
    pub(super) conn_tx: ConnTx,
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxiquic_core::ConnectionId;

    /// Build a no-op `ConnTx` backed by a channel whose receiver is
    /// immediately dropped — we never send through it in routing-table tests.
    fn dummy_conn_tx() -> ConnTx {
        let (tx, _rx) = mpsc::channel(1);
        tx
    }

    fn make_cid(bytes: &[u8]) -> ConnectionId {
        ConnectionId::from(bytes)
    }

    // ── InitialRetired removes the initial DCID from initial_map ─────────────

    /// After `apply_cid_route_update` processes an `InitialRetired` event the
    /// corresponding entry must be absent from `initial_map`.  This is the
    /// primary GC path resolved by the TODO at ~line 622.
    #[test]
    fn initial_retired_removes_entry_from_initial_map() {
        let mut initial_map: HashMap<Vec<u8>, ConnTx> = HashMap::new();
        let mut local_cid_map: HashMap<[u8; LOCAL_CID_LEN], ConnTx> = HashMap::new();

        let dcid = vec![0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        initial_map.insert(dcid.clone(), dummy_conn_tx());
        assert!(
            initial_map.contains_key(&dcid),
            "entry must be present before GC"
        );

        let update = CidRouteUpdate {
            event: CidEvent::InitialRetired(dcid.clone()),
            conn_tx: dummy_conn_tx(),
        };
        let applied = apply_cid_route_update(&mut initial_map, &mut local_cid_map, update);

        assert!(
            applied,
            "apply_cid_route_update must return true for InitialRetired"
        );
        assert!(
            !initial_map.contains_key(&dcid),
            "initial_map must not retain entry after InitialRetired"
        );
        assert!(local_cid_map.is_empty(), "local_cid_map must be unaffected");
    }

    /// Retiring a DCID that was never in `initial_map` is a no-op (idempotent).
    #[test]
    fn initial_retired_unknown_dcid_is_noop() {
        let mut initial_map: HashMap<Vec<u8>, ConnTx> = HashMap::new();
        let mut local_cid_map: HashMap<[u8; LOCAL_CID_LEN], ConnTx> = HashMap::new();

        let update = CidRouteUpdate {
            event: CidEvent::InitialRetired(vec![0xffu8; 8]),
            conn_tx: dummy_conn_tx(),
        };
        // Must not panic.
        let applied = apply_cid_route_update(&mut initial_map, &mut local_cid_map, update);
        assert!(applied);
        assert!(initial_map.is_empty());
    }

    // ── Register / Unregister still work after the refactor ──────────────────

    #[test]
    fn register_inserts_into_local_cid_map() {
        let mut initial_map: HashMap<Vec<u8>, ConnTx> = HashMap::new();
        let mut local_cid_map: HashMap<[u8; LOCAL_CID_LEN], ConnTx> = HashMap::new();

        let cid_bytes = [0x11u8; LOCAL_CID_LEN];
        let cid = make_cid(&cid_bytes);
        let update = CidRouteUpdate {
            event: CidEvent::Register(cid),
            conn_tx: dummy_conn_tx(),
        };
        let applied = apply_cid_route_update(&mut initial_map, &mut local_cid_map, update);
        assert!(applied);
        assert!(local_cid_map.contains_key(&cid_bytes));
        assert!(initial_map.is_empty());
    }

    #[test]
    fn unregister_removes_from_local_cid_map() {
        let mut initial_map: HashMap<Vec<u8>, ConnTx> = HashMap::new();
        let mut local_cid_map: HashMap<[u8; LOCAL_CID_LEN], ConnTx> = HashMap::new();

        let cid_bytes = [0x22u8; LOCAL_CID_LEN];
        local_cid_map.insert(cid_bytes, dummy_conn_tx());

        let cid = make_cid(&cid_bytes);
        let update = CidRouteUpdate {
            event: CidEvent::Unregister(cid),
            conn_tx: dummy_conn_tx(),
        };
        let applied = apply_cid_route_update(&mut initial_map, &mut local_cid_map, update);
        assert!(applied);
        assert!(!local_cid_map.contains_key(&cid_bytes));
    }

    /// Only the targeted initial DCID is removed; other entries survive.
    #[test]
    fn initial_retired_does_not_remove_other_entries() {
        let mut initial_map: HashMap<Vec<u8>, ConnTx> = HashMap::new();
        let mut local_cid_map: HashMap<[u8; LOCAL_CID_LEN], ConnTx> = HashMap::new();

        let dcid_a = vec![0xaau8; 8];
        let dcid_b = vec![0xbbu8; 8];
        initial_map.insert(dcid_a.clone(), dummy_conn_tx());
        initial_map.insert(dcid_b.clone(), dummy_conn_tx());

        let update = CidRouteUpdate {
            event: CidEvent::InitialRetired(dcid_a.clone()),
            conn_tx: dummy_conn_tx(),
        };
        apply_cid_route_update(&mut initial_map, &mut local_cid_map, update);

        assert!(!initial_map.contains_key(&dcid_a), "dcid_a must be removed");
        assert!(initial_map.contains_key(&dcid_b), "dcid_b must survive");
    }
}
