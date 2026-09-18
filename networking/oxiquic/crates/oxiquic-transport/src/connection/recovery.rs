//! ACK processing, loss detection, PTO timers (RFC 9002).
//!
//! Contains [`Connection::process_ack`], loss detection helpers,
//! `set_loss_detection_timer`, `on_loss_timeout`, and retransmission queuing.

use std::time::{Duration, Instant};

use oxiquic_core::{Direction, PacketType};

use crate::frame::AckRange;
use crate::recovery::SpaceIndex;
use crate::sent_packet::SentFrame;

use super::Connection;

impl Connection {
    /// Map a received packet's type to its packet-number-space array index.
    pub(super) fn space_index(packet_type: PacketType) -> usize {
        match packet_type {
            PacketType::Initial => SpaceIndex::Initial as usize,
            PacketType::Handshake => SpaceIndex::Handshake as usize,
            _ => SpaceIndex::Application as usize,
        }
    }

    /// Process an `ACK` frame for `packet_type`'s space: acknowledge packets,
    /// sample the RTT, advance congestion control and run loss detection
    /// (RFC 9002 Section 5, 6, Appendix A.7).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn process_ack(
        &mut self,
        now: Instant,
        packet_type: PacketType,
        largest: u64,
        delay: u64,
        first_range: u64,
        ranges: &[AckRange],
        ecn: Option<crate::ecn::EcnCounts>,
    ) {
        let idx = Self::space_index(packet_type);
        // Record the largest acknowledged for packet-number truncation length.
        match packet_type {
            PacketType::Initial => self.initial.on_ack_received(largest),
            PacketType::Handshake => self.handshake.on_ack_received(largest),
            _ => self.application.on_ack_received(largest),
        }
        // Capture each newly-acked packet's send time before we knew it was
        // largest, to sample the RTT correctly.
        let range_pairs: Vec<(u64, u64)> = ranges.iter().map(|r| (r.gap, r.range)).collect();
        let outcome = self.sent_packets[idx].on_ack(largest, first_range, &range_pairs);

        if outcome.newly_acked.is_empty() {
            return;
        }

        // RTT sample: only when the largest acked is newly acked (RFC 9002 5.1).
        // Capture the *raw* latest_rtt so the congestion controller (BBR) can
        // feed it into its min-RTT filter instead of re-deriving it from send
        // times; `None` on any ACK that carries no fresh sample.
        let mut cc_rtt_sample: Option<Duration> = None;
        // The path the RTT sample belongs to, and the RFC 9002 §5.3 correction
        // terms, so the same sample can also feed that path's own estimator.
        let mut rtt_sample_path: Option<usize> = None;
        let mut rtt_ack_delay = Duration::ZERO;
        let mut rtt_max_ack_delay = Duration::ZERO;
        if outcome.largest_newly_acked == Some(largest) {
            if let Some(sent) = outcome
                .newly_acked
                .iter()
                .find(|p| p.packet_number == largest)
            {
                let rtt_sample = now.saturating_duration_since(sent.time_sent);
                cc_rtt_sample = Some(rtt_sample);
                rtt_sample_path = Some(sent.path);
                // Decode the peer's ack_delay (scaled by its ack_delay_exponent).
                let ack_delay = self.decode_ack_delay(delay);
                // max_ack_delay applies only once the handshake is confirmed and
                // only in the Application space (RFC 9002 5.3).
                let max_ack_delay =
                    if idx == SpaceIndex::Application as usize && self.handshake_done {
                        self.peer_max_ack_delay
                    } else {
                        Duration::ZERO
                    };
                rtt_ack_delay = ack_delay;
                rtt_max_ack_delay = max_ack_delay;
                self.rtt.update(rtt_sample, ack_delay, max_ack_delay);
            }
        }

        // Congestion: release acked bytes; grow the window for non-recovery acks.
        let acked_cc: Vec<(usize, Instant, Option<crate::bbr::RateSample>)> = outcome
            .newly_acked
            .iter()
            .filter(|p| p.in_flight)
            .map(|p| (p.sent_bytes, p.time_sent, p.rate_sample))
            .collect();
        if !acked_cc.is_empty() {
            self.congestion
                .on_packets_acked(&acked_cc, cc_rtt_sample, now);
        }

        // Per-path attribution: route each newly-acked packet's bytes to the
        // congestion controller of the path that carried it, and give the RTT
        // sample to that path's estimator only. Without this, ACKs arriving over
        // a 10 ms path and a 200 ms path would collapse into one smoothed RTT
        // and one window sized for the aggregate.
        let mut acked_paths: Vec<usize> = outcome
            .newly_acked
            .iter()
            .filter(|p| p.in_flight)
            .map(|p| p.path)
            .collect();
        acked_paths.sort_unstable();
        acked_paths.dedup();
        for path_idx in acked_paths {
            let path_acked: Vec<(usize, Instant, Option<crate::bbr::RateSample>)> = outcome
                .newly_acked
                .iter()
                .filter(|p| p.in_flight && p.path == path_idx)
                .map(|p| (p.sent_bytes, p.time_sent, p.rate_sample))
                .collect();
            let path_rtt = if rtt_sample_path == Some(path_idx) {
                cc_rtt_sample
            } else {
                None
            };
            self.multipath.on_path_acked(
                path_idx,
                &path_acked,
                path_rtt,
                rtt_ack_delay,
                rtt_max_ack_delay,
                now,
            );
        }

        // ECN validation (RFC 9000 §13.4.2, RFC 9002 §7.4). Count the
        // newly-acknowledged packets we sent with ECT(0), and find the send time
        // of the largest newly-acked packet to attribute a CE congestion event.
        let newly_acked_ect0 = outcome
            .newly_acked
            .iter()
            .filter(|p| p.ecn == crate::ecn::EcnCodepoint::Ect0)
            .count() as u64;
        let largest_newly_acked_sent_time = outcome
            .largest_newly_acked
            .and_then(|largest_pn| {
                outcome
                    .newly_acked
                    .iter()
                    .find(|p| p.packet_number == largest_pn)
            })
            .map(|p| p.time_sent);
        let ecn_outcome = self.ecn[idx].validate_ack(ecn, newly_acked_ect0);
        // On a CE-count increase, react like a loss (window reduction) without
        // removing bytes from flight, using the largest-acked send time so
        // repeated CE reports collapse into one recovery period.
        if ecn_outcome.ce_increase {
            if let Some(sent_time) = largest_newly_acked_sent_time {
                self.congestion.on_ecn_ce(sent_time, now);
                if let Some(path_idx) = rtt_sample_path {
                    self.multipath.on_path_ecn_ce(path_idx, sent_time, now);
                }
            }
        }

        // A valid ack-eliciting acknowledgement resets the PTO backoff.
        if outcome.acked_ack_eliciting {
            self.loss.reset_pto_count();
            self.probes_owed = 0;
        }

        // Check for MTU probe ACKs among newly-acked packets.
        for pkt in &outcome.newly_acked {
            for frame in &pkt.frames {
                if let crate::sent_packet::SentFrame::MtuProbe(size) = frame {
                    self.on_mtu_probe_acked(*size, now);
                }
            }
        }

        // Run loss detection for this space.
        self.detect_and_handle_loss(idx, now);
        self.set_loss_detection_timer(now);
    }

    /// Decode the peer's wire `ack_delay` value, scaled by its
    /// `ack_delay_exponent` (RFC 9000 Section 18.2 / 19.3).
    fn decode_ack_delay(&self, delay: u64) -> Duration {
        let exponent = self
            .peer_params
            .as_ref()
            .map(|p| p.ack_delay_exponent)
            .unwrap_or(oxiquic_core::DEFAULT_ACK_DELAY_EXPONENT);
        let micros = delay.saturating_mul(1u64 << exponent.min(20));
        Duration::from_micros(micros)
    }

    /// Detect lost packets in space `idx` and re-queue their retransmittable
    /// frames; feed the loss to congestion control (RFC 9002 Section 6.1, 6.2).
    pub(super) fn detect_and_handle_loss(&mut self, idx: usize, now: Instant) {
        let loss_delay = self.rtt.loss_delay();
        let (lost, next_loss_time) = self.sent_packets[idx].detect_lost(now, loss_delay);
        let space = match idx {
            0 => SpaceIndex::Initial,
            1 => SpaceIndex::Handshake,
            _ => SpaceIndex::Application,
        };
        self.loss.set_loss_time(space, next_loss_time);

        if lost.is_empty() {
            return;
        }

        // Congestion event from the newest lost in-flight packet.
        let mut lost_bytes = 0u64;
        let mut newest_lost: Option<Instant> = None;
        let mut oldest_lost: Option<Instant> = None;
        for p in &lost {
            if p.in_flight {
                lost_bytes += p.sent_bytes as u64;
                newest_lost = Some(match newest_lost {
                    Some(t) => t.max(p.time_sent),
                    None => p.time_sent,
                });
                oldest_lost = Some(match oldest_lost {
                    Some(t) => t.min(p.time_sent),
                    None => p.time_sent,
                });
            }
        }
        self.packets_lost += lost.len() as u64;

        // Re-queue retransmittable frames onto their originating streams. The
        // space index routes CRYPTO retransmission to the correct CRYPTO stream.
        // Also detect lost MTU probes and handle back-off.
        for packet in &lost {
            for frame in &packet.frames {
                if let SentFrame::MtuProbe(size) = frame {
                    self.on_mtu_probe_lost(*size);
                } else {
                    self.requeue_frame(idx, frame);
                }
            }
        }

        // Per-path loss attribution: only the path that carried the lost packets
        // reduces its window (RFC 9002 §7 applied per path).
        let mut lost_paths: Vec<usize> = lost
            .iter()
            .filter(|p| p.in_flight)
            .map(|p| p.path)
            .collect();
        lost_paths.sort_unstable();
        lost_paths.dedup();
        for path_idx in lost_paths {
            let mut path_bytes = 0u64;
            let mut path_newest: Option<Instant> = None;
            for p in lost.iter().filter(|p| p.in_flight && p.path == path_idx) {
                path_bytes += p.sent_bytes as u64;
                path_newest = Some(match path_newest {
                    Some(t) => t.max(p.time_sent),
                    None => p.time_sent,
                });
            }
            if let Some(newest) = path_newest {
                self.multipath
                    .on_path_lost(path_idx, path_bytes, newest, now);
            }
        }

        if let Some(newest) = newest_lost {
            self.congestion.on_packets_lost(lost_bytes, newest, now);
            // Persistent congestion: if the span of lost packets exceeds the
            // persistent-congestion duration, collapse the window (RFC 9002 7.6).
            if let Some(oldest) = oldest_lost {
                let pto_base = self.rtt.pto_base(self.peer_max_ack_delay);
                if crate::congestion::is_persistent_congestion(oldest, newest, pto_base) {
                    self.congestion.on_persistent_congestion();
                }
            }
        }
    }

    /// Re-queue a single lost frame's data onto the stream it came from. `idx`
    /// is the packet-number-space index the lost packet belonged to, used to
    /// route CRYPTO retransmission to the matching CRYPTO stream.
    pub(super) fn requeue_frame(&mut self, idx: usize, frame: &SentFrame) {
        match frame {
            SentFrame::Crypto { offset, data } => {
                if idx == SpaceIndex::Handshake as usize {
                    self.handshake_crypto.requeue(*offset, data.clone());
                } else {
                    self.initial_crypto.requeue(*offset, data.clone());
                }
            }
            SentFrame::Stream {
                id,
                offset,
                fin,
                data,
            } => {
                if let Some(stream) = self.send_streams.get_mut(id) {
                    stream.requeue(*offset, data.clone(), *fin);
                }
            }
            SentFrame::Ping => {
                // PING carries no data; a fresh probe is sent on PTO instead.
            }
            SentFrame::MtuProbe(_) => {
                // MTU probes are handled in detect_and_handle_loss via
                // on_mtu_probe_lost; they are never re-queued as retransmissions.
            }
            SentFrame::NewConnectionId {
                seq,
                retire_prior_to,
                cid,
                stateless_reset_token,
            } => {
                // On loss, re-queue the NEW_CONNECTION_ID for retransmission —
                // unless a migration has since asked the peer to retire this
                // sequence number, in which case the pool drops it (see
                // `LocalCidPool::requeue_issued`).
                use crate::connection::cid::IssuedCid;
                self.local_cid_pool.requeue_issued(IssuedCid {
                    seq: *seq,
                    cid: cid.clone(),
                    stateless_reset_token: *stateless_reset_token,
                });
                // The frame's own retire_prior_to is not replayed: the pool's
                // current threshold is authoritative and is re-read when the
                // replacement frame is built.
                let _ = retire_prior_to;
            }
            SentFrame::RetireConnectionId { seq } => {
                // On loss, re-queue the RETIRE_CONNECTION_ID for retransmission.
                self.peer_cid_pool.pending_retire.push_back(*seq);
            }
            SentFrame::MaxStreams { dir, max } => {
                // On loss, re-queue the MAX_STREAMS update if the value is still
                // current (only keep the highest advertised limit).
                match dir {
                    Direction::Bidirectional => {
                        if self
                            .pending_max_streams_bidi
                            .is_none_or(|existing| *max > existing)
                        {
                            self.pending_max_streams_bidi = Some(*max);
                        }
                    }
                    Direction::Unidirectional => {
                        if self
                            .pending_max_streams_uni
                            .is_none_or(|existing| *max > existing)
                        {
                            self.pending_max_streams_uni = Some(*max);
                        }
                    }
                }
            }
            SentFrame::StreamsBlocked { dir, limit } => match dir {
                Direction::Bidirectional => {
                    self.pending_streams_blocked_bidi = Some(*limit);
                }
                Direction::Unidirectional => {
                    self.pending_streams_blocked_uni = Some(*limit);
                }
            },
            SentFrame::NewToken { token } => {
                // On loss, re-queue the NEW_TOKEN only if none is already pending.
                if self.pending_new_token.is_none() {
                    self.pending_new_token = Some(token.clone());
                }
            }
        }
    }

    /// Recompute the loss-detection timer (RFC 9002 Appendix A.6/A.8): the
    /// earliest pending time-threshold loss deadline if any, otherwise the PTO
    /// armed from the earliest outstanding ack-eliciting packet across spaces.
    pub(super) fn set_loss_detection_timer(&mut self, _now: Instant) {
        // 0. RFC 9002 §6.2.2.1 / RFC 9000 §8.1: a server that has exhausted its
        // anti-amplification allowance cannot send anything, so the timer stays
        // disarmed — a PTO probe would count against the very budget that is
        // already spent. The client is responsible for unblocking the server by
        // sending more datagrams.
        if self.amplification_blocked() {
            self.loss_timer = None;
            return;
        }

        // 1. Time-threshold loss timer takes precedence.
        if let Some((loss_time, _space)) = self.loss.earliest_loss_time() {
            self.loss_timer = Some(loss_time);
            return;
        }

        // 2. Otherwise, the PTO from the earliest ack-eliciting packet.
        let earliest = self.earliest_ack_eliciting_across_spaces();
        match earliest {
            Some(sent_time) => {
                // max_ack_delay only applies in the Application space; using the
                // peer value here is a safe upper bound for the combined timer.
                let pto_base = self.rtt.pto_base(self.peer_max_ack_delay);
                self.loss_timer = Some(self.loss.pto_deadline(sent_time, pto_base));
            }
            None => {
                // No ack-eliciting data outstanding: disarm.
                self.loss_timer = None;
            }
        }
    }

    /// The earliest send time of any outstanding ack-eliciting packet across all
    /// packet-number spaces.
    pub(super) fn earliest_ack_eliciting_across_spaces(&self) -> Option<Instant> {
        let mut earliest: Option<Instant> = None;
        for store in &self.sent_packets {
            if let Some(t) = store.earliest_ack_eliciting_time() {
                earliest = Some(match earliest {
                    Some(e) => e.min(t),
                    None => t,
                });
            }
        }
        earliest
    }

    /// Fire the loss-detection timer at `now` (RFC 9002 Appendix A.9
    /// `OnLossDetectionTimeout`): either declare time-threshold losses, or, if
    /// the PTO fired, schedule probe packets to elicit fresh acknowledgements.
    pub(super) fn on_loss_timeout(&mut self, now: Instant) {
        // If a time-threshold loss is due, process it.
        if let Some((loss_time, space)) = self.loss.earliest_loss_time() {
            if now >= loss_time {
                let idx = space as usize;
                self.detect_and_handle_loss(idx, now);
                self.set_loss_detection_timer(now);
                return;
            }
        }

        // Otherwise this is a PTO expiry: back off and owe probe packets so the
        // next poll_transmit retransmits / sends a PING to elicit an ACK.
        if self.earliest_ack_eliciting_across_spaces().is_some() {
            self.loss.increase_pto_count();
            // Two probes per PTO (RFC 9002 Section 6.2.4), capped.
            self.probes_owed = self.probes_owed.saturating_add(2).min(4);
            self.requeue_for_probe(now);
        }
        self.set_loss_detection_timer(now);
    }

    /// On PTO, re-queue the oldest outstanding data so a probe carries useful
    /// retransmission (RFC 9002 Section 6.2.4). For the Application space this
    /// re-queues stream data; for Initial/Handshake it re-queues CRYPTO. If no
    /// data is outstanding a bare PING is sent by the probe path.
    fn requeue_for_probe(&mut self, _now: Instant) {
        // Re-queue the lowest-numbered outstanding ack-eliciting packet's frames
        // in each space that has outstanding ack-eliciting data. We do not
        // remove them from sent_packets (they remain tracked until acked/lost);
        // the resend queues simply ensure the data is re-emitted. To avoid
        // unbounded duplication we re-queue only the single oldest packet.
        for idx in 0..3 {
            if let Some(frames) = self.sent_packets[idx].oldest_ack_eliciting_frames() {
                for frame in frames {
                    self.requeue_frame(idx, &frame);
                }
            }
        }
    }
}
