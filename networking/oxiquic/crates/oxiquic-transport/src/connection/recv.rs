//! Receive path: datagram ingestion, packet decryption, frame dispatch.
//!
//! Contains [`Connection::handle_datagram`] and all the helpers it calls:
//! long-header parsing, Retry handling, short-header / key-update decryption,
//! frame dispatch (`process_frames` / `handle_frame`), CRYPTO delivery, and
//! stream data ingestion.

use std::time::Instant;

use crate::crypto_stream::CryptoStreamError;
use crate::ecn::EcnCodepoint;
use crate::flow_control::StreamSendFlow;
use crate::frame::{decode_frame, Frame};
use crate::packet::{decrypt_short_packet_body, parse_long_packet, strip_short_header_protection};
use crate::stream::StreamError;
use oxiquic_core::{
    ConnectionId, Direction, FrameType, Initiator, OxiQuicError, PacketType, StreamId,
    TransportErrorCode,
};

use super::{Connection, ConnectionState, DatagramMeta, LOCAL_CID_LEN, QUIC_V1};

impl Connection {
    /// Feed a received UDP datagram into the connection. The datagram is
    /// decrypted in place (so it must be mutable and owned by the caller).
    ///
    /// Equivalent to [`Connection::handle_datagram_with_meta`] with metadata
    /// that names the connection's current peer address and reports no ECN
    /// codepoint. I/O layers that can report the source address and/or the IP
    /// ECN codepoint of the datagram should call `handle_datagram_with_meta`
    /// instead: the source address drives the RFC 9000 §9.3 per-path
    /// anti-amplification allowance and the codepoint drives ECN feedback.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] only for fatal protocol violations; packets
    /// that cannot yet be decrypted (missing keys) are silently skipped per
    /// RFC 9000 Section 5.2.
    pub fn handle_datagram(
        &mut self,
        now: Instant,
        datagram: &mut [u8],
    ) -> Result<(), OxiQuicError> {
        let meta = DatagramMeta::from_peer(self.peer_addr);
        self.handle_datagram_with_meta(now, datagram, meta)
    }

    /// Feed a received UDP datagram, together with the per-datagram metadata the
    /// I/O layer observed, into the connection.
    ///
    /// * [`DatagramMeta::src`] is credited against the anti-amplification
    ///   allowance of the path that address belongs to (RFC 9000 §8.1, §9.3).
    ///   An address that matches no known path only feeds the connection-level
    ///   allowance, so a spoofed source can never mint path state.
    /// * [`DatagramMeta::ecn`] is counted into the ECN codepoint tally of every
    ///   packet-number space that yields a decryptable packet from this
    ///   datagram, and echoed back to the peer in the ECN section of the next
    ///   ACK (RFC 9000 §13.4.1). `None` means the platform could not report a
    ///   codepoint; nothing is counted and plain (0x02) ACKs are sent.
    ///
    /// # Errors
    /// Returns an [`OxiQuicError`] only for fatal protocol violations; packets
    /// that cannot yet be decrypted (missing keys) are silently skipped per
    /// RFC 9000 Section 5.2.
    pub fn handle_datagram_with_meta(
        &mut self,
        now: Instant,
        datagram: &mut [u8],
        meta: DatagramMeta,
    ) -> Result<(), OxiQuicError> {
        self.arm_idle_timer(now);
        self.packets_recv += 1;
        let datagram_len = datagram.len() as u64;
        self.bytes_recv += datagram_len;
        // RFC 9000 §9.3: an address adopted during migration carries its own
        // 3x-received allowance until it is validated, so credit the path the
        // datagram actually arrived on rather than the connection as a whole.
        self.credit_path_bytes_received(meta.src, datagram_len);
        let ecn = meta.ecn;
        let mut offset = 0;
        while offset < datagram.len() {
            let first = datagram[offset];
            if first == 0 {
                break; // trailing padding between/after packets
            }
            let consumed = if first & 0x80 != 0 {
                self.recv_long_packet(now, datagram, offset, ecn)?
            } else {
                self.recv_short_packet(now, datagram, offset, ecn)?
            };
            match consumed {
                Some(end) => offset = end,
                None => break,
            }
            // Drive the handshake forward immediately so that keys installed by
            // a just-processed packet (e.g. Handshake keys after the Initial
            // carrying ServerHello) are available to decrypt the next coalesced
            // packet in this same datagram.
            self.pump_write_hs();
            // Attempt to install 0-RTT keys after each handshake pump so the
            // server can decrypt a coalesced 0-RTT packet in the same datagram.
            // Skip once established (keys already set or not applicable).
            if !self.handshake_complete && self.zero_rtt_keys.is_none() {
                self.try_install_zero_rtt_keys();
            }
        }
        Ok(())
    }

    /// Parse, decrypt and process one long-header packet at `offset`. Returns
    /// the offset of the next coalesced packet, or `None` to stop.
    fn recv_long_packet(
        &mut self,
        now: Instant,
        datagram: &mut [u8],
        offset: usize,
        ecn: Option<EcnCodepoint>,
    ) -> Result<Option<usize>, OxiQuicError> {
        let packet_type = match crate::packet::peek_dcid(&datagram[offset..], LOCAL_CID_LEN) {
            Ok((ty, _)) => ty,
            Err(_) => return Ok(None),
        };

        // RFC 9000 §17.2.1 / §6.2: A Version Negotiation packet arriving
        // during the handshake (and before any successful packet has been
        // processed) means the server does not support QUIC v1.  Fail the
        // connection.  If we have already received packets from this server we
        // MUST ignore VN (to prevent spoofed/replayed VN from killing live
        // connections).
        if packet_type == PacketType::VersionNegotiation {
            if self.role == super::Role::Client
                && self.state == ConnectionState::Handshaking
                && self.initial.largest_received().is_none()
            {
                let supported = crate::packet::decode_version_negotiation(&datagram[offset..])
                    .unwrap_or_default();
                return Err(OxiQuicError::VersionNegotiation { supported });
            }
            // Outside the early-handshake window: silently ignore.
            return Ok(None);
        }

        // ── RFC 9000 §17.2.5 / RFC 9001 §5.8: Retry ─────────────────────────
        if packet_type == PacketType::Retry {
            return self.recv_retry_packet(datagram, offset);
        }

        // Handle 0-RTT packets on the server side using the early keys.
        if packet_type == PacketType::ZeroRtt {
            return self.recv_zero_rtt_packet(now, datagram, offset, ecn);
        }

        let space = match packet_type {
            PacketType::Initial => &self.initial,
            PacketType::Handshake => &self.handshake,
            _ => return Ok(None),
        };
        let (packet_key, header_key) = match (space.remote_keys(), packet_type) {
            (Some(keys), _) => (keys.packet.as_ref(), keys.header.as_ref()),
            // No keys yet for this space: cannot decrypt; skip remaining.
            (None, _) => return Ok(None),
        };
        let largest = space.largest_received();
        let parsed =
            match parse_long_packet(datagram, offset, QUIC_V1, largest, packet_key, header_key) {
                Ok(p) => p,
                // A packet we cannot decrypt (e.g. keys not yet installed, or a
                // spurious/duplicate) is skipped per RFC 9000 Section 5.2.
                Err(_) => return Ok(None),
            };
        let next = parsed.consumed;

        // The destination connection ID must address us. A peer addresses its
        // early packets to the connection ID that seeds the Initial keys
        // (`initial_dcid`) and switches to our issued `local_cid` once it learns
        // it; accept either. Anything else is not ours (single-connection
        // server), so skip it.
        let dcid = parsed.dcid.as_slice();
        if !dcid.is_empty()
            && dcid != self.local_cid.as_bytes()
            && dcid != self.initial_dcid.as_bytes()
        {
            return Ok(Some(next));
        }

        // On the client, the server's SCID becomes our peer CID and (until the
        // first server packet) we adopt it as the destination for Handshake/1-RTT.
        if self.role == super::Role::Client
            && parsed.packet_type == PacketType::Initial
            && !parsed.scid.is_empty()
        {
            self.peer_cid = ConnectionId::new(parsed.scid.clone());
        }

        // RFC 9000 §8.1: successfully processing a Handshake-protected packet
        // proves the peer received our Initial flight, so its source address is
        // validated and the three-times amplification limit no longer applies.
        if parsed.packet_type == PacketType::Handshake && !self.address_validated {
            self.mark_address_validated();
        }

        let ack_eliciting = self.process_frames(now, parsed.packet_type, &parsed.payload)?;
        let space = match parsed.packet_type {
            PacketType::Initial => &mut self.initial,
            PacketType::Handshake => &mut self.handshake,
            _ => return Ok(Some(next)),
        };
        space.on_packet_received(parsed.packet_number, ack_eliciting, ecn);
        Ok(Some(next))
    }

    /// Process a Retry packet (RFC 9000 §17.2.5, RFC 9001 §5.8).
    ///
    /// A valid Retry causes the client to:
    /// 1. Verify the integrity tag.
    /// 2. Store the token.
    /// 3. Re-derive Initial keys from the Retry's SCID.
    /// 4. Reset the Initial PN space and re-queue the ClientHello.
    ///
    /// Returns `Ok(None)` to stop processing the datagram (no further coalesced
    /// packets can follow a Retry since it has no Length field).
    fn recv_retry_packet(
        &mut self,
        datagram: &[u8],
        offset: usize,
    ) -> Result<Option<usize>, OxiQuicError> {
        use crate::packet::{parse_retry_packet, verify_retry_integrity_tag};

        // Only the client processes Retry; only during the handshake; only once.
        if self.role != super::Role::Client
            || self.state != ConnectionState::Handshaking
            || self.retry_done
        {
            return Ok(None);
        }
        // Ignore if we have already received a successful packet from the server
        // (RFC 9000 §17.2.5.2).
        if self.initial.largest_received().is_some() {
            return Ok(None);
        }

        let packet = &datagram[offset..];

        // The ODCID used to verify the tag is the *current* initial_dcid
        // (before re-keying).
        let odcid = self.initial_dcid.as_bytes().to_vec();

        if !verify_retry_integrity_tag(&odcid, packet) {
            // Invalid integrity tag: silently ignore per RFC 9000 §17.2.5.2.
            return Ok(None);
        }

        let (scid, _dcid, token) = match parse_retry_packet(packet) {
            Some(t) => t,
            None => return Ok(None),
        };

        // Update peer CID to Retry's SCID (this becomes the DCID for our
        // retransmitted Initial and all subsequent packets).
        self.peer_cid = ConnectionId::new(scid.clone());
        // RFC 9000 §7.3: remember the Retry SCID so the server's
        // `retry_source_connection_id` transport parameter can be checked
        // against it once the handshake exposes the parameters. `original_dcid`
        // is deliberately *not* touched — it stays the DCID of our first
        // Initial, which is what `original_destination_connection_id` echoes.
        self.retry_scid = Some(ConnectionId::new(scid.clone()));
        self.retry_done = true;

        // Re-key the Initial space using the Retry's SCID as the new ODCID.
        self.rekey_initial_for_retry(scid, token);

        // Return None: no further packets are coalesced with a Retry.
        Ok(None)
    }

    /// Decrypt and process a 0-RTT long-header packet (RFC 9001 §4.6).
    ///
    /// Server-only: uses `zero_rtt_keys` (the early opening key). Shares the
    /// Application packet-number space (RFC 9001 §4.1.1). Returns `Ok(None)` if
    /// keys are not yet available (safe to drop — client will retransmit in 1-RTT
    /// per RFC 9000 §5.2).
    fn recv_zero_rtt_packet(
        &mut self,
        now: Instant,
        datagram: &mut [u8],
        offset: usize,
        ecn: Option<EcnCodepoint>,
    ) -> Result<Option<usize>, OxiQuicError> {
        // 0-RTT is only processed on the server side.
        if self.role != super::Role::Server {
            return Ok(None);
        }

        // We need to take the keys to avoid borrowing self twice.
        let keys = match self.zero_rtt_keys.take() {
            Some(k) => k,
            // Keys not yet derived: drop safely (RFC 9000 §5.2).
            None => return Ok(None),
        };

        let largest = self.application.largest_received();
        let parsed = match parse_long_packet(
            datagram,
            offset,
            QUIC_V1,
            largest,
            keys.packet.as_ref(),
            keys.header.as_ref(),
        ) {
            Ok(p) => {
                // Put keys back on success.
                self.zero_rtt_keys = Some(keys);
                p
            }
            Err(_) => {
                // Decryption failure: keys back, skip packet (RFC 9000 §5.2).
                self.zero_rtt_keys = Some(keys);
                return Ok(None);
            }
        };
        let next = parsed.consumed;

        // RFC 9001 §4.6: 0-RTT packets must NOT carry ACK, CRYPTO,
        // HANDSHAKE_DONE, NEW_TOKEN, PATH_CHALLENGE, or PATH_RESPONSE.
        // We route them through process_frames using PacketType::ZeroRtt;
        // forbidden frames will be handled gracefully (ACK processing works,
        // CRYPTO would be ignored since ZeroRtt matches no crypto stream).
        let ack_eliciting = self.process_frames(now, PacketType::ZeroRtt, &parsed.payload)?;
        self.application
            .on_packet_received(parsed.packet_number, ack_eliciting, ecn);
        Ok(Some(next))
    }

    /// Parse, decrypt and process one short-header (1-RTT) packet at `offset`.
    ///
    /// Implements the key-update receive path (RFC 9001 §6):
    /// 1. Strip header protection with the (unchanged) current header key.
    /// 2. Read the Key Phase bit from the deprotected first byte.
    /// 3. If the key phase matches the current sending phase, decrypt with
    ///    the current packet key.  On failure, try prev keys (reordered pre-update
    ///    packet).
    /// 4. If the key phase does NOT match, this is a peer-initiated key update:
    ///    decrypt with `next_1rtt_keys`.  On success, promote next→current and
    ///    compute a new next epoch.
    ///
    /// If all decryption attempts are exhausted, the raw bytes are examined for
    /// a stateless reset per RFC 9000 §10.3 before returning `Ok(None)`.
    pub(super) fn recv_short_packet(
        &mut self,
        now: Instant,
        datagram: &mut [u8],
        offset: usize,
        ecn: Option<EcnCodepoint>,
    ) -> Result<Option<usize>, OxiQuicError> {
        // Capture the length of this packet within the datagram for
        // stateless-reset detection.  We record this before any in-place
        // header-protection stripping so that `check_stateless_reset` can
        // always address the *original* trailing 16 bytes (RFC 9000 §10.3
        // checks the last 16 bytes of the received packet; header protection
        // only modifies the first byte and the packet-number bytes, never the
        // tail, so the token is intact regardless of stripping order).
        let packet_len = datagram[offset..].len();

        // Step 1: strip header protection in-place.  The borrow of
        // `self.application` is limited to this block; NLL drops it before
        // the AEAD step that might need `self.next_1rtt_keys`.
        let incoming_key_phase = {
            let hk = match self.application.remote_keys() {
                Some(keys) => keys.header.as_ref(),
                None => {
                    return self.check_stateless_reset(datagram, offset, packet_len);
                }
            };
            match strip_short_header_protection(datagram, offset, LOCAL_CID_LEN, hk) {
                Ok((kp, _)) => kp,
                Err(_) => {
                    return self.check_stateless_reset(datagram, offset, packet_len);
                }
            }
        }; // <-- borrow of self.application ends here

        let largest = self.application.largest_received();
        let current_key_phase = self.key_phase;

        if incoming_key_phase == current_key_phase {
            // --- Common path: same key phase → decrypt with current keys. ---
            let main_result = {
                let pk = match self.application.remote_keys() {
                    Some(keys) => keys.packet.as_ref(),
                    None => {
                        return self.check_stateless_reset(datagram, offset, packet_len);
                    }
                };
                decrypt_short_packet_body(datagram, offset, LOCAL_CID_LEN, largest, pk)
            }; // <-- borrow ends here

            let parsed = match main_result {
                Ok(p) => p,
                Err(_) => {
                    // Decryption failure with matching key phase: try prev keys
                    // for reordered pre-update packets (RFC 9001 §6.6).
                    let prev_result = match self.prev_1rtt_keys.as_ref() {
                        Some((prev_pk, retire_after)) if now <= *retire_after => {
                            let pk: &dyn rustls::quic::PacketKey = prev_pk.as_ref();
                            Some(decrypt_short_packet_body(
                                datagram,
                                offset,
                                LOCAL_CID_LEN,
                                largest,
                                pk,
                            ))
                        }
                        _ => None,
                    };
                    match prev_result {
                        Some(Ok(p)) => p,
                        _ => {
                            return self.check_stateless_reset(datagram, offset, packet_len);
                        }
                    }
                }
            };
            let consumed = parsed.consumed;
            let ack_eliciting = self.process_frames(now, PacketType::Short, &parsed.payload)?;
            self.application
                .on_packet_received(parsed.packet_number, ack_eliciting, ecn);
            Ok(Some(consumed))
        } else {
            // --- Key update path: key phase flipped → peer-initiated update. ---
            // Try to decrypt with the pre-derived next-epoch keys.
            let next_result = match self.next_1rtt_keys.as_ref() {
                Some(next_keys) => {
                    let pk: &dyn rustls::quic::PacketKey = next_keys.remote.as_ref();
                    decrypt_short_packet_body(datagram, offset, LOCAL_CID_LEN, largest, pk)
                }
                None => {
                    return self.check_stateless_reset(datagram, offset, packet_len);
                }
            }; // <-- borrow of self.next_1rtt_keys ends here

            let parsed = match next_result {
                Ok(p) => p,
                Err(_) => {
                    return self.check_stateless_reset(datagram, offset, packet_len);
                }
            };

            // Decryption succeeded → the peer has initiated a key update.
            // Rotate keys: next → current, derive new next, retire old current.
            self.perform_key_update(now);

            let consumed = parsed.consumed;
            let ack_eliciting = self.process_frames(now, PacketType::Short, &parsed.payload)?;
            self.application
                .on_packet_received(parsed.packet_number, ack_eliciting, ecn);
            Ok(Some(consumed))
        }
    }

    /// RFC 9000 §10.3: check whether the raw bytes at `datagram[offset..]` with
    /// length `packet_len` constitute a stateless reset.
    ///
    /// A stateless reset is recognised by:
    /// - The packet is at least 21 bytes long (RFC 9000 §10.3.1: the minimum
    ///   stateless reset size is 1 header byte + 4 pseudo-random bytes + 16-byte
    ///   token).
    /// - The last 16 bytes match a stateless reset token stored in our peer CID
    ///   pool (i.e., a token the peer sent us in a NEW_CONNECTION_ID frame).
    ///
    /// Note: `datagram` may have been modified in-place by header-protection
    /// stripping, but the last 16 bytes are never touched by that operation
    /// (header protection only alters the first byte and the packet-number
    /// bytes immediately after the connection ID), so reading from the original
    /// slice positions is always safe.
    fn check_stateless_reset(
        &self,
        datagram: &[u8],
        offset: usize,
        packet_len: usize,
    ) -> Result<Option<usize>, OxiQuicError> {
        // RFC 9000 §10.3: minimum stateless reset packet length is 21 bytes.
        if packet_len >= 21 {
            let end = offset + packet_len;
            // Safety: end is at most datagram.len() because packet_len was
            // computed as datagram[offset..].len() at the start of recv_short_packet.
            debug_assert!(end <= datagram.len());
            if end <= datagram.len() {
                let mut token = [0u8; 16];
                token.copy_from_slice(&datagram[end - 16..end]);
                if self.peer_cid_pool.matches_stateless_reset(&token) {
                    return Err(OxiQuicError::StatelessReset);
                }
            }
        }
        Ok(None)
    }

    /// Decode and act on the frames in a decrypted packet payload. Returns
    /// whether the packet was ack-eliciting.
    pub(super) fn process_frames(
        &mut self,
        now: Instant,
        packet_type: PacketType,
        payload: &[u8],
    ) -> Result<bool, OxiQuicError> {
        // Clear the per-packet scratch set used for RFC 9000 §19.16 same-packet
        // RETIRE_CONNECTION_ID validation.
        self.cids_issued_this_packet.clear();
        let mut buf = crate::coding::Buf::new(payload);
        let mut ack_eliciting = false;
        while !buf.is_empty() {
            let frame = decode_frame(&mut buf).map_err(|e| OxiQuicError::TransportError {
                code: e.code,
                frame_type: None,
                reason: e.detail.to_string(),
            })?;
            if frame.is_ack_eliciting() {
                ack_eliciting = true;
            }
            self.handle_frame(now, packet_type, frame)?;
        }
        Ok(ack_eliciting)
    }

    pub(super) fn handle_frame(
        &mut self,
        now: Instant,
        packet_type: PacketType,
        frame: Frame<'_>,
    ) -> Result<(), OxiQuicError> {
        match frame {
            Frame::Padding(_) | Frame::Ping => {}
            Frame::Ack {
                largest,
                delay,
                first_range,
                ranges,
                ecn,
            } => {
                self.process_ack(now, packet_type, largest, delay, first_range, &ranges, ecn);
            }
            Frame::ResetStream {
                stream_id,
                error_code,
                final_size,
            } => {
                self.recv_reset_stream(stream_id, error_code, final_size)?;
            }
            Frame::StopSending {
                stream_id,
                error_code,
            } => {
                self.recv_stop_sending(stream_id, error_code);
            }
            Frame::Crypto { offset, data } => {
                self.recv_crypto(packet_type, offset, data)?;
            }
            Frame::Stream {
                id,
                offset,
                fin,
                data,
            } => self.recv_stream(id, offset, fin, data)?,
            Frame::MaxData(max) => {
                // RFC 9000 Section 4.1: raise our connection-level send limit.
                self.send_flow.on_max_data(max);
            }
            Frame::MaxStreamData { id, max } => {
                self.recv_max_stream_data(id, max)?;
            }
            Frame::DataBlocked(_) | Frame::StreamDataBlocked { .. } => {
                // The peer reports it is blocked sending to us; our receive-side
                // limit advances as the application reads (see `advance_flow`),
                // so no immediate action beyond acknowledgement is required.
            }
            Frame::PathChallenge(data) => {
                // RFC 9000 §8.2.2: echo PATH_CHALLENGE data in a PATH_RESPONSE.
                // Always respond regardless of source address.
                self.pending_path_response = Some(data);
            }
            Frame::PathResponse(data) => {
                // Validate against every challenge still outstanding for this
                // attempt (RFC 9000 §8.2.3 — a retransmitted challenge carries
                // fresh bytes, and the peer may answer any of them); promote
                // the candidate address on success (§9.3). A mismatch is
                // silently ignored per §19.18.
                if self.accept_path_response(data) {
                    if let Some(addr) = self.candidate_peer_addr.take() {
                        self.peer_addr = addr;
                        // RFC 9000 §9.3: the address is now validated, so its
                        // per-path three-times allowance no longer applies.
                        self.mark_path_addr_validated(addr);
                        // RFC 9000 §9.5: the connection has moved, so retire the
                        // connection IDs used on the old path and hand the peer
                        // a fresh batch — reusing them would let an observer of
                        // both paths link them.
                        self.rotate_cids_after_migration();
                    }
                }
            }
            Frame::HandshakeDone => {
                if self.role == super::Role::Client {
                    self.handshake_done = true;
                }
            }
            Frame::NewConnectionId {
                seq,
                retire_prior_to,
                cid,
                stateless_reset_token,
            } => {
                // NEW_CONNECTION_ID is only valid in 1-RTT (Application) space
                // (RFC 9000 §19.15 forbids it in Initial/Handshake).
                if packet_type != PacketType::Short {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: Some(FrameType::NewConnectionId),
                        reason: "NEW_CONNECTION_ID received in non-1-RTT packet".to_string(),
                    });
                }
                self.peer_cid_pool.receive_new_cid(
                    seq,
                    retire_prior_to,
                    cid,
                    stateless_reset_token,
                )?;
                // Track this seq for same-packet RETIRE_CONNECTION_ID validation.
                self.cids_issued_this_packet.insert(seq);
            }
            Frame::RetireConnectionId { seq } => {
                // RETIRE_CONNECTION_ID is only valid in 1-RTT space.
                if packet_type != PacketType::Short {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: Some(FrameType::RetireConnectionId),
                        reason: "RETIRE_CONNECTION_ID received in non-1-RTT packet".to_string(),
                    });
                }
                // RFC 9000 §19.16: the peer MUST NOT retire a CID that was
                // issued in the same packet as this RETIRE_CONNECTION_ID.
                if self.cids_issued_this_packet.contains(&seq) {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: Some(FrameType::RetireConnectionId),
                        reason: "RETIRE_CONNECTION_ID for CID issued in same packet".to_string(),
                    });
                }
                if let Some(retired_cid) = self.local_cid_pool.handle_peer_retirement(seq) {
                    self.pending_cid_events
                        .push_back(crate::connection::cid::CidEvent::Unregister(retired_cid));
                    // Issue a replacement CID to keep the peer's pool replenished
                    // (RFC 9000 §5.1.1: maintain active_connection_id_limit supply).
                    let _ = self.maybe_issue_new_cid();
                }
            }
            Frame::ConnectionClose {
                error_code,
                application,
                reason,
                ..
            } => {
                let reason = String::from_utf8_lossy(&reason).into_owned();
                self.peer_closed = Some(if application {
                    OxiQuicError::ApplicationClose {
                        code: error_code,
                        reason,
                    }
                } else {
                    OxiQuicError::TransportError {
                        code: TransportErrorCode::from_u64(error_code),
                        frame_type: None,
                        reason,
                    }
                });
                self.state = ConnectionState::Closed;
            }
            Frame::MaxStreams { dir, max } => {
                // RFC 9000 §19.11: raise our stream-opening limit for the peer.
                match dir {
                    Direction::Bidirectional => {
                        if max > self.peer_max_streams_bidi {
                            self.peer_max_streams_bidi = max;
                        }
                    }
                    Direction::Unidirectional => {
                        if max > self.peer_max_streams_uni {
                            self.peer_max_streams_uni = max;
                        }
                    }
                }
            }
            Frame::StreamsBlocked { dir, .. } => {
                // Peer wants more streams — proactively raise our advertised limit
                // if we have headroom (RFC 9000 §19.14).
                self.maybe_raise_max_streams(dir);
            }
            Frame::NewToken(token) => {
                // RFC 9000 §19.7: server receiving NEW_TOKEN is a PROTOCOL_VIOLATION.
                if self.role == super::Role::Server {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: Some(oxiquic_core::FrameType::NewToken),
                        reason: "server received NEW_TOKEN".to_string(),
                    });
                }
                // Only valid in 1-RTT (Short header) space.
                if packet_type != PacketType::Short {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: Some(oxiquic_core::FrameType::NewToken),
                        reason: "NEW_TOKEN in non-1-RTT packet".to_string(),
                    });
                }
                self.received_token = Some(token.to_vec());
            }
            Frame::Datagram(data) => {
                // RFC 9221 §3: receiving a DATAGRAM when we advertised size 0 is
                // a PROTOCOL_VIOLATION.
                if self.local_max_datagram_frame_size == 0 {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: None,
                        reason: "received DATAGRAM but local max_datagram_frame_size is 0"
                            .to_string(),
                    });
                }
                if data.len() as u64 > self.local_max_datagram_frame_size {
                    return Err(OxiQuicError::TransportError {
                        code: TransportErrorCode::ProtocolViolation,
                        frame_type: None,
                        reason: format!(
                            "DATAGRAM {} bytes exceeds local max {}",
                            data.len(),
                            self.local_max_datagram_frame_size
                        ),
                    });
                }
                // Enqueue, evicting the oldest datagram if the buffer is full.
                let total: usize = self
                    .datagram_recv_queue
                    .iter()
                    .map(|d| d.len())
                    .sum::<usize>()
                    + data.len();
                if total > self.datagram_recv_buffer_limit {
                    self.datagram_recv_queue.pop_front();
                }
                self.datagram_recv_queue.push_back(data.to_vec());
            }
            Frame::Unsupported(_) => {}
        }
        Ok(())
    }

    /// Proactively raise the MAX_STREAMS limit for `dir` when a STREAMS_BLOCKED
    /// is received from the peer (RFC 9000 §4.6). Uses a simple policy: bump by
    /// the number of peer-initiated streams that have been closed.
    fn maybe_raise_max_streams(&mut self, dir: Direction) {
        match dir {
            Direction::Bidirectional => {
                let new_max = self.local_max_streams_bidi + self.closed_peer_bidi;
                if new_max > self.sent_max_streams_bidi {
                    self.pending_max_streams_bidi = Some(new_max);
                }
            }
            Direction::Unidirectional => {
                let new_max = self.local_max_streams_uni + self.closed_peer_uni;
                if new_max > self.sent_max_streams_uni {
                    self.pending_max_streams_uni = Some(new_max);
                }
            }
        }
    }

    pub(super) fn recv_crypto(
        &mut self,
        packet_type: PacketType,
        offset: u64,
        data: &[u8],
    ) -> Result<(), OxiQuicError> {
        let delivered = match packet_type {
            PacketType::Initial => self.initial_crypto.recv(offset, data),
            PacketType::Handshake => self.handshake_crypto.recv(offset, data),
            _ => Ok(None),
        };
        // RFC 9000 §7.5 / §19.6: an over-large CRYPTO offset is a frame-encoding
        // error, and more out-of-order CRYPTO data than we will buffer is
        // CRYPTO_BUFFER_EXCEEDED. Both are connection errors, which keeps the
        // pre-handshake reassembly state bounded for unauthenticated peers.
        let delivered = delivered.map_err(|e| match e {
            CryptoStreamError::BufferExceeded => OxiQuicError::TransportError {
                code: TransportErrorCode::CryptoBufferExceeded,
                frame_type: Some(FrameType::Crypto),
                reason: "buffered CRYPTO data exceeds the reassembly limit".into(),
            },
            CryptoStreamError::OffsetTooLarge => OxiQuicError::TransportError {
                code: TransportErrorCode::FrameEncodingError,
                frame_type: Some(FrameType::Crypto),
                reason: "CRYPTO frame end offset exceeds 2^62-1".into(),
            },
        })?;
        if let Some(bytes) = delivered {
            self.tls
                .read_hs(&bytes)
                .map_err(|e| OxiQuicError::Tls(e.to_string()))?;
        }
        Ok(())
    }

    /// Whether `stream_id` was opened by the remote endpoint (RFC 9000 §2.1):
    /// true when the initiator bit does *not* match our own role.
    pub(super) fn is_peer_initiated(&self, stream_id: StreamId) -> bool {
        match self.role {
            super::Role::Client => stream_id.initiator() == Initiator::Server,
            super::Role::Server => stream_id.initiator() == Initiator::Client,
        }
    }

    /// RFC 9000 §4.6: a frame that opens a peer-initiated stream whose index is
    /// at or beyond the concurrency limit we advertised is a connection error of
    /// type `STREAM_LIMIT_ERROR`.
    ///
    /// The comparison uses the largest `MAX_STREAMS` value we have actually
    /// transmitted (`sent_max_streams_*`, seeded from the initial transport
    /// parameter), so a stream the peer was legitimately granted is never
    /// rejected. Enforcing this also bounds `recv_streams`, `send_streams`,
    /// `stream_recv_flow`, `stream_send_flow` and `new_peer_streams`: without
    /// it a peer can grow all five without limit using one-byte STREAM frames
    /// or bare RESET_STREAM frames.
    pub(super) fn check_peer_stream_limit(&self, stream_id: StreamId) -> Result<(), OxiQuicError> {
        let limit = match stream_id.direction() {
            Direction::Bidirectional => self.sent_max_streams_bidi,
            Direction::Unidirectional => self.sent_max_streams_uni,
        };
        if stream_id.index() >= limit {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::StreamLimitError,
                frame_type: None,
                reason: format!(
                    "peer-initiated {} stream index {} exceeds advertised limit {}",
                    stream_id.direction(),
                    stream_id.index(),
                    limit
                ),
            });
        }
        Ok(())
    }

    /// RFC 9000 §19.8/§19.10: a frame that references a *locally*-initiated
    /// stream we have not opened yet is a `STREAM_STATE_ERROR`. This bounds the
    /// same maps for the half of the stream-ID space the peer cannot open.
    pub(super) fn check_local_stream_open(&self, stream_id: StreamId) -> Result<(), OxiQuicError> {
        let opened = match stream_id.direction() {
            Direction::Bidirectional => self.next_bidi_index,
            Direction::Unidirectional => self.next_uni_index,
        };
        if stream_id.index() >= opened {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::StreamStateError,
                frame_type: None,
                reason: format!(
                    "frame references locally-initiated {} stream {} which is not open",
                    stream_id.direction(),
                    stream_id.index()
                ),
            });
        }
        Ok(())
    }

    /// Validate the stream ID carried by a receive-side frame (`STREAM`,
    /// `RESET_STREAM`) before any per-stream state is created for it.
    fn check_recv_stream_id(&self, stream_id: StreamId) -> Result<(), OxiQuicError> {
        if self.is_peer_initiated(stream_id) {
            return self.check_peer_stream_limit(stream_id);
        }
        // A locally-initiated unidirectional stream is send-only: the peer must
        // never deliver data on it, nor reset it (RFC 9000 §3.2, §19.4).
        if stream_id.direction() == Direction::Unidirectional {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::StreamStateError,
                frame_type: None,
                reason: "receive-side frame on a send-only stream".into(),
            });
        }
        self.check_local_stream_open(stream_id)
    }

    /// Handle a received `MAX_STREAM_DATA` frame (RFC 9000 §19.10).
    ///
    /// The frame is validated before `stream_send_flow` is touched: a
    /// MAX_STREAM_DATA frame is only a few bytes on the wire, so an unchecked
    /// `entry(id).or_insert_with(..)` lets one datagram create hundreds of map
    /// entries with attacker-chosen keys.
    fn recv_max_stream_data(&mut self, id: u64, max: u64) -> Result<(), OxiQuicError> {
        let stream_id = StreamId::from(id);
        if self.is_peer_initiated(stream_id) {
            // A peer-initiated unidirectional stream is receive-only for us, so
            // the peer cannot grant us send credit on it (RFC 9000 §19.10).
            if stream_id.direction() == Direction::Unidirectional {
                return Err(OxiQuicError::TransportError {
                    code: TransportErrorCode::StreamStateError,
                    frame_type: Some(FrameType::MaxStreamData),
                    reason: "MAX_STREAM_DATA for a receive-only stream".into(),
                });
            }
            self.check_peer_stream_limit(stream_id)?;
        } else {
            self.check_local_stream_open(stream_id)?;
        }
        let initial = self.peer_initial_stream_limit();
        self.stream_send_flow
            .entry(id)
            .or_insert_with(|| StreamSendFlow::new(initial))
            .on_max_stream_data(max);
        Ok(())
    }

    pub(super) fn recv_stream(
        &mut self,
        id: u64,
        offset: u64,
        fin: bool,
        data: &[u8],
    ) -> Result<(), OxiQuicError> {
        // Auto-create the recv (and, for a peer-opened bidi stream, send) state.
        let stream_id = StreamId::from(id);
        // RFC 9000 §4.6 / §3.2: reject out-of-limit or not-yet-open stream IDs
        // *before* any per-stream map entry is created for them.
        self.check_recv_stream_id(stream_id)?;
        // RFC 9000 §4.5: STREAM data that extends beyond a final size already
        // established for this stream (by a FIN or a RESET_STREAM) is a
        // FINAL_SIZE_ERROR. This must be checked before the reset early-return
        // below (a reset stream returns `Ok(())` for further STREAM frames),
        // otherwise a peer could contradict a RESET_STREAM final size silently.
        let end_offset = offset.saturating_add(data.len() as u64);
        if let Some(final_size) = self
            .recv_streams
            .get(&id)
            .and_then(|s| s.known_final_size())
        {
            if end_offset > final_size {
                return Err(OxiQuicError::TransportError {
                    code: TransportErrorCode::FinalSizeError,
                    frame_type: None,
                    reason: "stream data exceeds the established final size".into(),
                });
            }
        }
        // Determine whether this stream ID is peer-initiated (i.e., opened by
        // the *remote* endpoint). If the initiator bit matches our role then the
        // stream was opened by us; otherwise it is peer-initiated.
        let peer_initiated = self.is_peer_initiated(stream_id);
        // Track peer-initiated streams the first time we see them so the
        // driven-connection layer can surface them via `poll_new_peer_stream`.
        let is_new = !self.recv_streams.contains_key(&id);
        if is_new && peer_initiated {
            self.new_peer_streams.push_back(stream_id);
        }
        if stream_id.direction() == Direction::Bidirectional {
            self.send_streams.entry(id).or_default();
            let initial = self.peer_initial_stream_limit();
            self.stream_send_flow
                .entry(id)
                .or_insert_with(|| StreamSendFlow::new(initial));
        }
        let local_max_stream_data = self.local_initial_max_stream_data;
        let stream_flow = self
            .stream_recv_flow
            .entry(id)
            .or_insert_with(|| crate::flow_control::StreamRecvFlow::new(local_max_stream_data));

        // RFC 9000 Section 4.1: enforce the advertised stream- and
        // connection-level receive limits. A peer that sends stream data beyond
        // either limit commits a FLOW_CONTROL_ERROR; rejecting it here also
        // bounds the reassembly buffers (a far-ahead offset is refused rather
        // than buffered).
        let new_bytes = match stream_flow.on_stream_data(end_offset) {
            Some(delta) => delta,
            None => {
                return Err(OxiQuicError::TransportError {
                    code: TransportErrorCode::FlowControlError,
                    frame_type: None,
                    reason: "stream data exceeds MAX_STREAM_DATA".into(),
                })
            }
        };
        if !self.recv_flow.on_data_received(new_bytes) {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::FlowControlError,
                frame_type: None,
                reason: "stream data exceeds MAX_DATA".into(),
            });
        }

        let s = self.recv_streams.entry(id).or_default();
        // If the stream has already been reset by the peer, ignore further
        // STREAM frames (RFC 9000 §3.2: "A receiver MUST ignore STREAM frames
        // after receiving RESET_STREAM for the same stream").
        if s.is_reset() {
            return Ok(());
        }
        match s.recv(offset, data, fin) {
            Ok(true) => self.readable.push_back(stream_id),
            Ok(false) => {}
            Err(StreamError::FinalSize) => {
                return Err(OxiQuicError::TransportError {
                    code: TransportErrorCode::FinalSizeError,
                    frame_type: None,
                    reason: "stream final size violation".into(),
                })
            }
            Err(StreamError::Reset(_)) => {
                // `recv()` never returns this variant today; included for
                // exhaustiveness in case StreamError gains new variants.
            }
        }
        Ok(())
    }

    /// Handle a received `RESET_STREAM` frame (RFC 9000 §19.4): mark the
    /// receive stream as reset so the application can observe the error code.
    ///
    /// The frame's `final_size` is not decorative: RFC 9000 §4.5 requires it to
    /// be counted toward both the stream- and connection-level flow-control
    /// windows and to be consistent with data already received. A `final_size`
    /// below the highest offset received (or contradicting a previously
    /// established final size, e.g. from a FIN) is a `FINAL_SIZE_ERROR`; a
    /// `final_size` past the advertised limit is a `FLOW_CONTROL_ERROR`.
    fn recv_reset_stream(
        &mut self,
        stream_id: u64,
        error_code: u64,
        final_size: u64,
    ) -> Result<(), OxiQuicError> {
        // RFC 9000 §4.6: RESET_STREAM carries no data, so without this check a
        // peer can create an unbounded number of `recv_streams` entries with
        // three-byte frames.
        self.check_recv_stream_id(StreamId::from(stream_id))?;

        // RFC 9000 §4.5 flow-control accounting: treat `final_size` exactly like
        // a STREAM frame ending at that offset. This enforces MAX_STREAM_DATA,
        // charges the newly-revealed bytes against MAX_DATA, and — crucially —
        // rejects a final size below data already received on the stream.
        let local_max_stream_data = self.local_initial_max_stream_data;
        let stream_flow = self
            .stream_recv_flow
            .entry(stream_id)
            .or_insert_with(|| crate::flow_control::StreamRecvFlow::new(local_max_stream_data));
        if final_size < stream_flow.highest_received() {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::FinalSizeError,
                frame_type: Some(FrameType::ResetStream),
                reason: "RESET_STREAM final size below the highest received offset".into(),
            });
        }
        let new_bytes = match stream_flow.on_stream_data(final_size) {
            Some(delta) => delta,
            None => {
                return Err(OxiQuicError::TransportError {
                    code: TransportErrorCode::FlowControlError,
                    frame_type: Some(FrameType::ResetStream),
                    reason: "RESET_STREAM final size exceeds MAX_STREAM_DATA".into(),
                })
            }
        };
        if !self.recv_flow.on_data_received(new_bytes) {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::FlowControlError,
                frame_type: Some(FrameType::ResetStream),
                reason: "RESET_STREAM final size exceeds MAX_DATA".into(),
            });
        }

        let s = match self.recv_streams.get_mut(&stream_id) {
            Some(s) => s,
            // Unknown stream: ignore per RFC 9000 §3.2 (may be a stream we
            // have never seen; create a stub entry to surface the reset).
            None => {
                // Only create a stub for peer-initiated streams; locally-initiated
                // receive streams have no data until we open them so a reset is
                // protocol-level odd but not fatal.
                self.recv_streams.entry(stream_id).or_default()
            }
        };
        // RFC 9000 §4.5: a RESET_STREAM final size that contradicts a final size
        // already established (e.g. by a FIN) is a FINAL_SIZE_ERROR.
        if s.set_final_size(final_size).is_err() {
            return Err(OxiQuicError::TransportError {
                code: TransportErrorCode::FinalSizeError,
                frame_type: Some(FrameType::ResetStream),
                reason: "RESET_STREAM final size contradicts the established final size".into(),
            });
        }
        s.apply_reset(error_code);
        // Surface to the application so it can observe the reset via read_stream.
        let sid = StreamId::from(stream_id);
        self.readable.push_back(sid);
        Ok(())
    }

    /// Handle a received `STOP_SENDING` frame (RFC 9000 §19.5): per RFC 9000
    /// §3.5, we SHOULD respond with a `RESET_STREAM` on the send side using
    /// the provided error code.
    fn recv_stop_sending(&mut self, stream_id: u64, error_code: u64) {
        if let Some(s) = self.send_streams.get_mut(&stream_id) {
            // If the stream has already been reset (e.g. a concurrent local
            // reset raced with this STOP_SENDING), re-use the existing reset
            // code so the peer receives a consistent RESET_STREAM (RFC 9000 §3.5).
            let effective_code = s.reset_code().unwrap_or(error_code);
            // Capture the final size before (potentially) resetting.
            let final_size = s.final_size();
            if !s.is_reset() {
                s.reset(effective_code);
            }
            // Queue a RESET_STREAM to inform the peer (RFC 9000 §3.5).
            self.pending_reset_streams
                .insert(stream_id, (effective_code, final_size));
        }
        // If the stream is unknown (we never opened it), ignore silently.
    }
}

#[cfg(test)]
mod flow_control_tests {
    use std::sync::Arc;

    use oxiquic_core::{OxiQuicError, TransportErrorCode, TransportParams};
    use rustls::pki_types::{CertificateDer, ServerName};
    use rustls::version::TLS13;
    use rustls::{ClientConfig, RootCertStore};

    use crate::Connection;

    /// Build a bare client `Connection` (handshake not run — `recv_stream`
    /// operates on the stream/flow-control state directly) whose advertised
    /// receive limits are `max_data` (connection) and `max_stream_data` (per
    /// bidi stream the peer sends on).
    fn client_conn(max_data: u64, max_stream_data: u64) -> Connection {
        let ck = oxitls_rcgen::generate_self_signed_ed25519(&["localhost"])
            .expect("generate self-signed cert");
        let cert_der = CertificateDer::from(ck.cert_der.clone());
        let provider = Arc::new(oxiquic_crypto::quic_crypto_provider());
        let mut roots = RootCertStore::empty();
        roots.add(cert_der).expect("trust self-signed cert");
        let client_cfg = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&TLS13])
            .expect("client TLS1.3")
            .with_root_certificates(roots)
            .with_no_client_auth();

        let params = TransportParams {
            initial_max_data: max_data,
            initial_max_stream_data_bidi_remote: max_stream_data,
            // The peer must be allowed to open the stream these tests use, or
            // the RFC 9000 §4.6 stream-limit check fires before flow control.
            initial_max_streams_bidi: 10,
            ..TransportParams::default()
        };

        Connection::new_client(
            Arc::new(client_cfg),
            ServerName::try_from("localhost").expect("server name"),
            std::net::SocketAddr::from(([127, 0, 0, 1], 4433)),
            params,
            Default::default(),
            Default::default(),
        )
        .expect("client conn")
    }

    /// The first *server*-initiated bidirectional stream — i.e. peer-initiated
    /// from the client connection under test (RFC 9000 Table 1).
    const PEER_BIDI_0: u64 = 1;

    #[test]
    fn stream_data_over_stream_limit_is_flow_control_error() {
        let mut conn = client_conn(1_000_000, 50);
        // 60 bytes at offset 0 ends at offset 60 > the 50-byte stream limit.
        let err = conn
            .recv_stream(PEER_BIDI_0, 0, false, &[0u8; 60])
            .expect_err("over-limit stream data must be rejected");
        match err {
            OxiQuicError::TransportError { code, .. } => {
                assert_eq!(code, TransportErrorCode::FlowControlError);
            }
            other => panic!("expected FLOW_CONTROL_ERROR, got {other:?}"),
        }
    }

    #[test]
    fn stream_data_over_conn_limit_is_flow_control_error() {
        // Stream limit is generous; the aggregate connection limit is 50.
        let mut conn = client_conn(50, 1_000_000);
        let err = conn
            .recv_stream(PEER_BIDI_0, 0, false, &[0u8; 60])
            .expect_err("over-limit stream data must be rejected");
        match err {
            OxiQuicError::TransportError { code, .. } => {
                assert_eq!(code, TransportErrorCode::FlowControlError);
            }
            other => panic!("expected FLOW_CONTROL_ERROR, got {other:?}"),
        }
    }

    #[test]
    fn stream_data_within_limits_is_accepted() {
        let mut conn = client_conn(1_000_000, 1_000_000);
        conn.recv_stream(PEER_BIDI_0, 0, false, &[0u8; 60])
            .expect("in-limit stream data must be accepted");
        // A retransmission of already-received offsets must not be double-charged
        // against the connection limit (it charges 0 new bytes).
        conn.recv_stream(PEER_BIDI_0, 0, false, &[0u8; 60])
            .expect("retransmitted stream data must be accepted");
    }
}

/// RFC 9000 §4.6 stream-concurrency enforcement: a peer that references stream
/// IDs beyond the limit we advertised (or locally-initiated IDs we never
/// opened) must be met with a connection error, not unbounded map growth.
#[cfg(test)]
mod stream_limit_tests {
    use std::sync::Arc;

    use oxiquic_core::{
        Direction, Initiator, OxiQuicError, StreamId, TransportErrorCode, TransportParams,
    };
    use rustls::pki_types::{CertificateDer, ServerName};
    use rustls::version::TLS13;
    use rustls::{ClientConfig, RootCertStore};

    use crate::frame::Frame;
    use crate::Connection;

    const STREAM_LIMIT: u64 = 4;

    /// A client connection advertising `STREAM_LIMIT` peer-initiated streams in
    /// each direction and generous flow-control windows.
    fn client_conn() -> Connection {
        let ck = oxitls_rcgen::generate_self_signed_ed25519(&["localhost"])
            .expect("generate self-signed cert");
        let cert_der = CertificateDer::from(ck.cert_der.clone());
        let provider = Arc::new(oxiquic_crypto::quic_crypto_provider());
        let mut roots = RootCertStore::empty();
        roots.add(cert_der).expect("trust self-signed cert");
        let client_cfg = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&TLS13])
            .expect("client TLS1.3")
            .with_root_certificates(roots)
            .with_no_client_auth();

        let params = TransportParams {
            initial_max_data: 1 << 30,
            initial_max_stream_data_bidi_remote: 1 << 20,
            initial_max_stream_data_bidi_local: 1 << 20,
            initial_max_stream_data_uni: 1 << 20,
            initial_max_streams_bidi: STREAM_LIMIT,
            initial_max_streams_uni: STREAM_LIMIT,
            ..TransportParams::default()
        };

        Connection::new_client(
            Arc::new(client_cfg),
            ServerName::try_from("localhost").expect("server name"),
            std::net::SocketAddr::from(([127, 0, 0, 1], 4433)),
            params,
            Default::default(),
            Default::default(),
        )
        .expect("client conn")
    }

    fn code_of(err: &OxiQuicError) -> TransportErrorCode {
        match err {
            OxiQuicError::TransportError { code, .. } => *code,
            other => panic!("expected a transport error, got {other:?}"),
        }
    }

    /// Peer-initiated (server) stream `index` in `dir`.
    fn peer_stream(dir: Direction, index: u64) -> u64 {
        StreamId::new(Initiator::Server, dir, index).as_u64()
    }

    #[test]
    fn stream_frame_beyond_advertised_limit_is_stream_limit_error() {
        let mut conn = client_conn();
        // The last legal index is STREAM_LIMIT - 1.
        conn.recv_stream(
            peer_stream(Direction::Bidirectional, STREAM_LIMIT - 1),
            0,
            false,
            b"x",
        )
        .expect("last in-limit stream is accepted");
        let err = conn
            .recv_stream(
                peer_stream(Direction::Bidirectional, STREAM_LIMIT),
                0,
                false,
                b"x",
            )
            .expect_err("stream index at the limit must be rejected");
        assert_eq!(code_of(&err), TransportErrorCode::StreamLimitError);
    }

    /// Regression: one-byte STREAM frames on ascending peer stream IDs used to
    /// grow `recv_streams` / `send_streams` / the flow-control maps without any
    /// bound. The connection error now fires and the maps stay capped.
    #[test]
    fn stream_id_flood_cannot_grow_the_stream_maps() {
        let mut conn = client_conn();
        let mut rejected = false;
        for index in 0..10_000u64 {
            if conn
                .recv_stream(peer_stream(Direction::Bidirectional, index), 0, false, b"x")
                .is_err()
            {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "the flood must be refused");
        assert!(conn.recv_streams.len() <= STREAM_LIMIT as usize);
        assert!(conn.new_peer_streams.len() <= STREAM_LIMIT as usize);
    }

    /// Regression: bare RESET_STREAM frames carry no data at all, so they were
    /// the cheapest way to grow `recv_streams`.
    #[test]
    fn reset_stream_flood_cannot_grow_the_stream_maps() {
        let mut conn = client_conn();
        let mut rejected = false;
        for index in 0..10_000u64 {
            let frame = Frame::ResetStream {
                stream_id: peer_stream(Direction::Unidirectional, index),
                error_code: 0,
                final_size: 0,
            };
            if conn
                .handle_frame(
                    std::time::Instant::now(),
                    oxiquic_core::PacketType::Short,
                    frame,
                )
                .is_err()
            {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "the RESET_STREAM flood must be refused");
        assert!(conn.recv_streams.len() <= STREAM_LIMIT as usize);
    }

    /// Regression: MAX_STREAM_DATA is only a few bytes on the wire and used to
    /// insert an attacker-keyed `stream_send_flow` entry unconditionally.
    #[test]
    fn max_stream_data_flood_cannot_grow_the_send_flow_map() {
        let mut conn = client_conn();
        let mut rejected = false;
        for index in 0..10_000u64 {
            let frame = Frame::MaxStreamData {
                id: peer_stream(Direction::Bidirectional, index),
                max: 1 << 20,
            };
            if conn
                .handle_frame(
                    std::time::Instant::now(),
                    oxiquic_core::PacketType::Short,
                    frame,
                )
                .is_err()
            {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "the MAX_STREAM_DATA flood must be refused");
        assert!(conn.stream_send_flow.len() <= STREAM_LIMIT as usize);
    }

    /// A MAX_STREAM_DATA naming a locally-initiated stream we never opened is a
    /// STREAM_STATE_ERROR (RFC 9000 §19.10) rather than a new map entry.
    #[test]
    fn max_stream_data_for_unopened_local_stream_is_stream_state_error() {
        let mut conn = client_conn();
        let id = StreamId::new(Initiator::Client, Direction::Bidirectional, 7).as_u64();
        let err = conn
            .handle_frame(
                std::time::Instant::now(),
                oxiquic_core::PacketType::Short,
                Frame::MaxStreamData { id, max: 1 << 20 },
            )
            .expect_err("unopened local stream must be rejected");
        assert_eq!(code_of(&err), TransportErrorCode::StreamStateError);
        assert_eq!(conn.stream_send_flow.len(), 0);
    }

    /// STREAM data on a locally-initiated *unidirectional* (send-only) stream is
    /// a STREAM_STATE_ERROR (RFC 9000 §3.2).
    #[test]
    fn stream_data_on_send_only_stream_is_stream_state_error() {
        let mut conn = client_conn();
        let id = StreamId::new(Initiator::Client, Direction::Unidirectional, 0).as_u64();
        let err = conn
            .recv_stream(id, 0, false, b"x")
            .expect_err("send-only stream must reject data");
        assert_eq!(code_of(&err), TransportErrorCode::StreamStateError);
    }

    /// A peer-initiated bidirectional stream index 0 (server stream ID 1).
    const PEER_BIDI_0: u64 = 1;

    fn reset_stream(conn: &mut Connection, id: u64, final_size: u64) -> Result<(), OxiQuicError> {
        conn.handle_frame(
            std::time::Instant::now(),
            oxiquic_core::PacketType::Short,
            Frame::ResetStream {
                stream_id: id,
                error_code: 0,
                final_size,
            },
        )
    }

    /// RFC 9000 §4.5: a RESET_STREAM whose final size is below the highest
    /// offset already received on the stream is a FINAL_SIZE_ERROR.
    #[test]
    fn reset_stream_final_size_below_received_is_final_size_error() {
        let mut conn = client_conn();
        conn.recv_stream(PEER_BIDI_0, 0, false, &[0u8; 100])
            .expect("in-limit stream data accepted");
        let err = reset_stream(&mut conn, PEER_BIDI_0, 0)
            .expect_err("final size below received offset must be rejected");
        assert_eq!(code_of(&err), TransportErrorCode::FinalSizeError);
    }

    /// RFC 9000 §4.5 (mirror direction): STREAM data extending past a final size
    /// already established by a RESET_STREAM is a FINAL_SIZE_ERROR — even though
    /// the stream is now reset and further STREAM frames are otherwise ignored.
    #[test]
    fn stream_data_beyond_reset_final_size_is_final_size_error() {
        let mut conn = client_conn();
        reset_stream(&mut conn, PEER_BIDI_0, 10).expect("reset with final size 10");
        let err = conn
            .recv_stream(PEER_BIDI_0, 0, false, &[0u8; 20])
            .expect_err("data past the reset final size must be rejected");
        assert_eq!(code_of(&err), TransportErrorCode::FinalSizeError);
    }

    /// RFC 9000 §4.5: a RESET_STREAM final size that contradicts a final size
    /// already established by a FIN is a FINAL_SIZE_ERROR.
    #[test]
    fn reset_stream_contradicting_fin_final_size_is_final_size_error() {
        let mut conn = client_conn();
        // A FIN at offset 5 establishes final size 5.
        conn.recv_stream(PEER_BIDI_0, 0, true, &[0u8; 5])
            .expect("fin at offset 5");
        let err = reset_stream(&mut conn, PEER_BIDI_0, 10)
            .expect_err("reset final size contradicting the FIN must be rejected");
        assert_eq!(code_of(&err), TransportErrorCode::FinalSizeError);
    }

    /// RFC 9000 §4.5: a RESET_STREAM final size beyond the advertised
    /// MAX_STREAM_DATA is a FLOW_CONTROL_ERROR (the final size consumes credit).
    #[test]
    fn reset_stream_final_size_over_stream_limit_is_flow_control_error() {
        let mut conn = client_conn();
        // client_conn advertises initial_max_stream_data_bidi_remote = 1 << 20.
        let err = reset_stream(&mut conn, PEER_BIDI_0, (1 << 20) + 1)
            .expect_err("final size past MAX_STREAM_DATA must be rejected");
        assert_eq!(code_of(&err), TransportErrorCode::FlowControlError);
    }

    /// A retransmitted RESET_STREAM carrying the same final size is idempotent:
    /// it charges zero new flow-control bytes and is accepted (RFC 9000 §4.5).
    #[test]
    fn duplicate_reset_stream_is_idempotent() {
        let mut conn = client_conn();
        reset_stream(&mut conn, PEER_BIDI_0, 50).expect("first reset accepted");
        reset_stream(&mut conn, PEER_BIDI_0, 50).expect("duplicate reset accepted");
    }
}

/// Connection-level ECN behaviour (RFC 9000 §13.4, RFC 9002 §7.4): feeding a
/// synthesized ACK-ECN through `process_ack` drives validation and the CE
/// congestion response on the in-process path (no real socket needed).
#[cfg(test)]
mod ecn_tests {
    use std::sync::Arc;
    use std::time::Instant;

    use oxiquic_core::{PacketType, TransportParams};
    use rustls::pki_types::{CertificateDer, ServerName};
    use rustls::version::TLS13;
    use rustls::{ClientConfig, RootCertStore};

    use crate::ecn::{EcnCodepoint, EcnCounts, EcnValidationState};
    use crate::recovery::SpaceIndex;
    use crate::sent_packet::SentPacket;
    use crate::Connection;

    fn client_conn() -> Connection {
        let ck = oxitls_rcgen::generate_self_signed_ed25519(&["localhost"])
            .expect("generate self-signed cert");
        let cert_der = CertificateDer::from(ck.cert_der.clone());
        let provider = Arc::new(oxiquic_crypto::quic_crypto_provider());
        let mut roots = RootCertStore::empty();
        roots.add(cert_der).expect("trust self-signed cert");
        let client_cfg = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&TLS13])
            .expect("client TLS1.3")
            .with_root_certificates(roots)
            .with_no_client_auth();
        Connection::new_client(
            Arc::new(client_cfg),
            ServerName::try_from("localhost").expect("server name"),
            std::net::SocketAddr::from(([127, 0, 0, 1], 4433)),
            TransportParams::default(),
            Default::default(),
            Default::default(),
        )
        .expect("client conn")
    }

    /// Register an in-flight, ECT(0)-marked Application-space packet.
    fn register_ect0(conn: &mut Connection, pn: u64, now: Instant) {
        let idx = SpaceIndex::Application as usize;
        conn.sent_packets[idx].on_packet_sent(SentPacket {
            packet_number: pn,
            time_sent: now,
            ack_eliciting: true,
            in_flight: true,
            sent_bytes: 1200,
            frames: vec![crate::sent_packet::SentFrame::Ping],
            rate_sample: None,
            ecn: EcnCodepoint::Ect0,
            path: 0,
        });
    }

    #[test]
    fn fresh_connection_marks_ect0() {
        let conn = client_conn();
        assert_eq!(conn.ecn_desired_codepoint(), EcnCodepoint::Ect0);
        assert_eq!(conn.ecn_state(), EcnValidationState::Testing);
    }

    #[test]
    fn ce_increase_shrinks_congestion_window() {
        let now = Instant::now();

        // Baseline connection: ACK-ECN with no CE increase.
        let mut base = client_conn();
        register_ect0(&mut base, 0, now);
        register_ect0(&mut base, 1, now);
        register_ect0(&mut base, 2, now);
        base.process_ack(
            now,
            PacketType::Short,
            2,
            0,
            2, // first_range=2 → acks [0,2]
            &[],
            Some(EcnCounts::new(3, 0, 0)),
        );
        let base_cwnd = base.congestion_window();
        assert_eq!(base.ecn_state(), EcnValidationState::Capable);

        // CE connection: identical ACK but with CE=2 (a congestion signal).
        let mut ce = client_conn();
        register_ect0(&mut ce, 0, now);
        register_ect0(&mut ce, 1, now);
        register_ect0(&mut ce, 2, now);
        ce.process_ack(
            now,
            PacketType::Short,
            2,
            0,
            2,
            &[],
            Some(EcnCounts::new(1, 0, 2)),
        );
        let ce_cwnd = ce.congestion_window();
        assert_eq!(ce.ecn_state(), EcnValidationState::Capable);

        // The CE report must have reduced the window relative to the no-CE ACK.
        assert!(
            ce_cwnd < base_cwnd,
            "CE-marked ACK must shrink cwnd: ce={ce_cwnd} base={base_cwnd}"
        );
    }

    #[test]
    fn decreasing_counts_fail_validation_and_stop_marking() {
        let now = Instant::now();
        let mut conn = client_conn();

        // First ACK establishes ECN as Capable with counts (2,0,0).
        register_ect0(&mut conn, 0, now);
        register_ect0(&mut conn, 1, now);
        conn.process_ack(
            now,
            PacketType::Short,
            1,
            0,
            1, // acks [0,1]
            &[],
            Some(EcnCounts::new(2, 0, 0)),
        );
        assert_eq!(conn.ecn_state(), EcnValidationState::Capable);
        assert_eq!(conn.ecn_desired_codepoint(), EcnCodepoint::Ect0);

        // Second ACK reports a *lower* ECT(0) count (1 < 2): a rollback that
        // must fail validation and disable ECN marking.
        register_ect0(&mut conn, 2, now);
        register_ect0(&mut conn, 3, now);
        conn.process_ack(
            now,
            PacketType::Short,
            3,
            0,
            1, // acks [2,3]
            &[],
            Some(EcnCounts::new(1, 0, 0)),
        );
        assert_eq!(conn.ecn_state(), EcnValidationState::Failed);
        assert_eq!(conn.ecn_desired_codepoint(), EcnCodepoint::NotEct);
    }
}
