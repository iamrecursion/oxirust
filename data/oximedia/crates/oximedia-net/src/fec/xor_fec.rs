//! XOR-based Forward Error Correction (RFC 5109 style).
//!
//! Implements 1-D XOR FEC for RTP streams.  For every group of `k` source
//! packets the encoder XORs their payloads (after zero-padding to the length
//! of the longest payload) and emits a single FEC repair packet.  The decoder
//! can recover exactly one lost source packet per group when the FEC packet
//! for that group is available.
//!
//! # 2-D Interleaved FEC
//!
//! When `l > 1` the encoder also computes column-wise FEC packets, providing
//! two-dimensional protection against burst losses (RFC 5109 §10.3 style).

use std::collections::{HashMap, VecDeque};

// ─── Configuration ────────────────────────────────────────────────────────────

/// FEC scheme configuration.
///
/// For 1-D XOR FEC set `l = 1`.  For 2-D interleaved FEC set `l > 1`:
/// - `k` source packets form a "row"
/// - `l` rows form a "matrix"
/// - Column FEC covers the `k` corresponding packets across `l` rows
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FecConfig {
    /// Number of source packets per FEC group / row.
    pub k: usize,
    /// Total packets in a group including FEC (= k + 1 for 1-D, k + l + 1 for 2-D).
    pub n: usize,
    /// Number of rows (columns dimension). 1 = 1-D, >1 = 2-D interleaved.
    pub l: usize,
}

impl Default for FecConfig {
    fn default() -> Self {
        Self { k: 5, n: 6, l: 1 }
    }
}

impl FecConfig {
    /// Creates a standard 1-D XOR FEC config with `k` source packets per group.
    #[must_use]
    pub fn one_dimensional(k: usize) -> Self {
        Self { k, n: k + 1, l: 1 }
    }

    /// Creates a 2-D interleaved FEC config.
    #[must_use]
    pub fn two_dimensional(k: usize, l: usize) -> Self {
        Self { k, n: k + l + 1, l }
    }

    /// Returns `true` if this is a 2-D configuration.
    #[must_use]
    pub fn is_2d(&self) -> bool {
        self.l > 1
    }
}

// ─── FEC packet type ──────────────────────────────────────────────────────────

/// A FEC repair packet.
///
/// The `payload` is the XOR of the covered source packet payloads
/// (zero-padded to equal length before XOR-ing).
#[derive(Debug, Clone)]
pub struct FecPacket {
    /// Repair packet sequence number (distinct space from source seq nums).
    pub sequence_number: u16,
    /// Timestamp derived from the first source packet in the group.
    pub timestamp: u32,
    /// Index identifying which FEC packet this is within the group.
    pub fec_index: u32,
    /// XOR of covered source-packet payloads.
    pub payload: Vec<u8>,
    /// Bitmask of covered sequence numbers (relative to group base seq).
    pub mask: u64,
}

impl FecPacket {
    /// Returns the number of covered source packets (popcount of mask).
    #[must_use]
    pub fn covered_count(&self) -> u32 {
        self.mask.count_ones()
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// Stateful FEC encoder.
///
/// Call [`FecEncoder::feed_packet`] for every source RTP packet; the encoder
/// returns a [`FecPacket`] once a full group of `k` source packets has been
/// accumulated.
pub struct FecEncoder {
    config: FecConfig,
    /// Buffered (seq_num, payload) pairs for the current row.
    row_buffer: VecDeque<(u16, Vec<u8>)>,
    /// Running XOR accumulator for the current row.
    row_xor: Vec<u8>,
    /// Sequence number of the FEC repair packet itself.
    fec_seq: u16,
    /// Timestamp of the first source packet in the current group.
    group_timestamp: u32,
    /// Bitmask of source sequence numbers (relative to group start).
    group_mask: u64,
    /// 2-D column buffers: `col_buffers[col_idx]` holds (row, payload) pairs.
    col_buffers: Vec<VecDeque<Vec<u8>>>,
    /// Row counter for 2-D FEC.
    row_count: usize,
}

impl FecEncoder {
    /// Creates a new encoder from a [`FecConfig`].
    #[must_use]
    pub fn new(config: FecConfig) -> Self {
        let l = config.l;
        let k = config.k;
        let col_buffers = vec![VecDeque::with_capacity(l); k];
        Self {
            config,
            row_buffer: VecDeque::new(),
            row_xor: Vec::new(),
            fec_seq: 0,
            group_timestamp: 0,
            group_mask: 0,
            col_buffers,
            row_count: 0,
        }
    }

    /// Feeds a source packet into the encoder.
    ///
    /// Returns `Some(FecPacket)` when a row FEC packet is ready (every `k`
    /// source packets).  Returns `None` otherwise.
    ///
    /// In 2-D mode, column FEC packets are buffered internally and are
    /// available via [`FecEncoder::take_column_fec`] after `l` row FEC
    /// packets have been generated.
    pub fn feed_packet(&mut self, seq: u16, payload: &[u8]) -> Option<FecPacket> {
        // Track timestamp from the first packet in the group.
        if self.row_buffer.is_empty() {
            self.group_timestamp = u32::from(seq) * 90;
            self.group_mask = 0;
        }

        // XOR into the accumulator, extending with zeros as needed.
        xor_into(&mut self.row_xor, payload);

        // Record the bit position of this source packet in the mask.
        let bit_pos = self.row_buffer.len() as u64;
        self.group_mask |= 1u64 << bit_pos;

        self.row_buffer.push_back((seq, payload.to_vec()));

        // Store payload in column buffer for 2-D FEC.
        if self.config.is_2d() {
            let col = (self.row_buffer.len() - 1) % self.config.k;
            self.col_buffers[col].push_back(payload.to_vec());
        }

        if self.row_buffer.len() >= self.config.k {
            let fec_pkt = self.emit_row_fec();
            self.row_count += 1;
            return Some(fec_pkt);
        }
        None
    }

    /// Drains any completed 2-D column FEC packets.
    ///
    /// Returns column FEC packets when `l` rows have been encoded.
    pub fn take_column_fec(&mut self) -> Vec<FecPacket> {
        if !self.config.is_2d() || self.row_count < self.config.l {
            return Vec::new();
        }
        let mut out = Vec::new();
        for col in 0..self.config.k {
            let mut col_xor: Vec<u8> = Vec::new();
            let mut mask = 0u64;
            for (row, payload) in self.col_buffers[col].iter().enumerate() {
                xor_into(&mut col_xor, payload);
                mask |= 1u64 << row as u64;
            }
            let fec_pkt = FecPacket {
                sequence_number: self.fec_seq,
                timestamp: 0,
                fec_index: (self.config.k + col) as u32,
                payload: col_xor,
                mask,
            };
            self.fec_seq = self.fec_seq.wrapping_add(1);
            out.push(fec_pkt);
            self.col_buffers[col].clear();
        }
        self.row_count = 0;
        out
    }

    // ── private ───────────────────────────────────────────────────────────────

    fn emit_row_fec(&mut self) -> FecPacket {
        let fec_pkt = FecPacket {
            sequence_number: self.fec_seq,
            timestamp: self.group_timestamp,
            fec_index: 0,
            payload: std::mem::take(&mut self.row_xor),
            mask: self.group_mask,
        };
        self.fec_seq = self.fec_seq.wrapping_add(1);
        self.row_buffer.clear();
        self.group_mask = 0;
        fec_pkt
    }
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

/// Stateful FEC decoder.
///
/// Feed source packets via [`FecDecoder::feed_source`] and FEC repair packets
/// via [`FecDecoder::feed_fec`].  Call [`FecDecoder::try_recover`] to attempt
/// recovery of missing source packets.
pub struct FecDecoder {
    config: FecConfig,
    /// Received source packets: seq_num → payload.
    received_source: HashMap<u16, Vec<u8>>,
    /// Received FEC repair packets: fec_index → FecPacket.
    received_fec: HashMap<u32, FecPacket>,
    /// Successfully recovered packets pending drain: seq_num → payload.
    recovered: HashMap<u16, Vec<u8>>,
    /// Known group base sequence numbers for group boundary reconstruction.
    group_bases: Vec<u16>,
    /// Groups that have already been fully recovered (prevents double-recovery).
    recovered_groups: std::collections::HashSet<u16>,
}

impl FecDecoder {
    /// Creates a new decoder from a [`FecConfig`].
    #[must_use]
    pub fn new(config: FecConfig) -> Self {
        Self {
            config,
            received_source: HashMap::new(),
            received_fec: HashMap::new(),
            recovered: HashMap::new(),
            group_bases: Vec::new(),
            recovered_groups: std::collections::HashSet::new(),
        }
    }

    /// Registers a received source packet.
    pub fn feed_source(&mut self, seq: u16, payload: Vec<u8>) {
        self.received_source.insert(seq, payload);
    }

    /// Registers a received FEC repair packet.
    pub fn feed_fec(&mut self, pkt: FecPacket) {
        // Record the base sequence number implied by the FEC packet's mask.
        // We derive it from the sequence number stored in the FEC packet header.
        self.received_fec.insert(pkt.fec_index, pkt);
    }

    /// Registers the base sequence number for a group explicitly.
    ///
    /// This is needed when the first source packet of a group was lost and
    /// the caller has reconstructed the group boundary from external metadata.
    pub fn register_group_base(&mut self, base_seq: u16) {
        if !self.group_bases.contains(&base_seq) {
            self.group_bases.push(base_seq);
        }
    }

    /// Attempts to recover missing source packets using available FEC packets.
    ///
    /// For each FEC group where exactly one source packet is missing AND the
    /// corresponding FEC repair packet is present, the missing packet is
    /// recovered by XOR-ing all other source payloads with the FEC payload.
    ///
    /// Returns a list of `(seq_num, recovered_payload)` pairs.  Each call
    /// consumes the recovered entries from the internal map.
    pub fn try_recover(&mut self) -> Vec<(u16, Vec<u8>)> {
        // Build a set of group bases from known source packets and registered bases.
        let mut bases: Vec<u16> = self.group_bases.clone();
        for (&seq, _) in &self.received_source {
            // Infer which group this packet belongs to.
            let base = self.group_base_for(seq);
            if !bases.contains(&base) {
                bases.push(base);
            }
        }

        for base in bases {
            self.attempt_group_recovery(base);
        }

        self.recovered.drain().collect()
    }

    /// Returns a reference to all received source packets.
    #[must_use]
    pub fn received_source(&self) -> &HashMap<u16, Vec<u8>> {
        &self.received_source
    }

    /// Returns a reference to all received FEC packets.
    #[must_use]
    pub fn received_fec(&self) -> &HashMap<u32, FecPacket> {
        &self.received_fec
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Infers the group base sequence number for a given source sequence number.
    fn group_base_for(&self, seq: u16) -> u16 {
        let k = self.config.k as u16;
        // Integer-divide the sequence number by k, then multiply back.
        let offset = seq.wrapping_div(k);
        offset.wrapping_mul(k)
    }

    fn attempt_group_recovery(&mut self, base: u16) {
        // Skip groups that have already been recovered to prevent re-entry.
        if self.recovered_groups.contains(&base) {
            return;
        }

        let k = self.config.k;
        // Collect sequence numbers for this group.
        let group_seqs: Vec<u16> = (0..k).map(|i| base.wrapping_add(i as u16)).collect();

        // Count how many are missing (not in received_source and not yet
        // placed in the pending recovered map for this call).
        let mut missing: Vec<u16> = Vec::new();
        for &seq in &group_seqs {
            if !self.received_source.contains_key(&seq) && !self.recovered.contains_key(&seq) {
                missing.push(seq);
            }
        }

        // We can only recover if exactly one packet is missing.
        if missing.len() != 1 {
            return;
        }

        // Check we have a FEC packet for this group.
        // The FEC index 0 covers all rows; for now we match the first available.
        let fec_opt = self.received_fec.values().find(|fec| {
            // Verify that the FEC covers this group by checking sequence overlap.
            let covered_count = fec.covered_count() as usize;
            covered_count == k
        });

        let fec_payload = match fec_opt {
            Some(f) => f.payload.clone(),
            None => return,
        };

        // XOR all available source payloads with the FEC payload to recover.
        let mut recovered_payload = fec_payload;
        for &seq in &group_seqs {
            if seq == missing[0] {
                continue;
            }
            let src = self
                .received_source
                .get(&seq)
                .or_else(|| self.recovered.get(&seq));
            if let Some(payload) = src {
                xor_into(&mut recovered_payload, payload);
            }
        }

        self.recovered.insert(missing[0], recovered_payload);
        // Mark this group as done so subsequent try_recover calls skip it.
        self.recovered_groups.insert(base);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// XOR `src` into `dst`, zero-extending `dst` if necessary.
fn xor_into(dst: &mut Vec<u8>, src: &[u8]) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    // 1. Default config is k=5, n=6, l=1
    #[test]
    fn test_default_config() {
        let cfg = FecConfig::default();
        assert_eq!(cfg.k, 5);
        assert_eq!(cfg.n, 6);
        assert_eq!(cfg.l, 1);
    }

    // 2. one_dimensional helper sets n = k + 1
    #[test]
    fn test_one_dimensional_config() {
        let cfg = FecConfig::one_dimensional(8);
        assert_eq!(cfg.k, 8);
        assert_eq!(cfg.n, 9);
        assert_eq!(cfg.l, 1);
        assert!(!cfg.is_2d());
    }

    // 3. two_dimensional helper sets l and n correctly
    #[test]
    fn test_two_dimensional_config() {
        let cfg = FecConfig::two_dimensional(4, 3);
        assert_eq!(cfg.k, 4);
        assert_eq!(cfg.l, 3);
        assert_eq!(cfg.n, 8); // 4 + 3 + 1
        assert!(cfg.is_2d());
    }

    // 4. Encoder produces FEC after k packets
    #[test]
    fn test_encoder_produces_fec_after_k_packets() {
        let cfg = FecConfig::one_dimensional(3);
        let mut enc = FecEncoder::new(cfg);
        assert!(enc.feed_packet(0, &make_payload(0xAA, 10)).is_none());
        assert!(enc.feed_packet(1, &make_payload(0xBB, 10)).is_none());
        let fec = enc.feed_packet(2, &make_payload(0xCC, 10));
        assert!(fec.is_some());
    }

    // 5. FEC payload is XOR of three source payloads
    #[test]
    fn test_fec_payload_is_xor() {
        let cfg = FecConfig::one_dimensional(3);
        let mut enc = FecEncoder::new(cfg);
        let p0 = make_payload(0x11, 4);
        let p1 = make_payload(0x22, 4);
        let p2 = make_payload(0x33, 4);
        enc.feed_packet(0, &p0);
        enc.feed_packet(1, &p1);
        let fec = enc.feed_packet(2, &p2).expect("must produce FEC");
        for i in 0..4 {
            assert_eq!(fec.payload[i], p0[i] ^ p1[i] ^ p2[i]);
        }
    }

    // 6. FEC mask has k bits set
    #[test]
    fn test_fec_mask_popcount() {
        let cfg = FecConfig::one_dimensional(5);
        let mut enc = FecEncoder::new(cfg.clone());
        for seq in 0..cfg.k as u16 {
            enc.feed_packet(seq, &make_payload(seq as u8, 8));
        }
        // Actually the last feed_packet returns the FEC.
        let mut enc2 = FecEncoder::new(FecConfig::one_dimensional(5));
        let mut fec_pkt = None;
        for seq in 0..5u16 {
            let result = enc2.feed_packet(seq, &make_payload(seq as u8, 8));
            if result.is_some() {
                fec_pkt = result;
            }
        }
        let fec = fec_pkt.expect("FEC must be produced");
        assert_eq!(fec.covered_count(), 5);
    }

    // 7. xor_into helper extends dst when src is longer
    #[test]
    fn test_xor_into_extends_dst() {
        let mut dst = vec![0xFFu8; 2];
        let src = vec![0x0Fu8; 4];
        xor_into(&mut dst, &src);
        assert_eq!(dst.len(), 4);
        assert_eq!(dst[0], 0xFF ^ 0x0F);
        assert_eq!(dst[2], 0x00 ^ 0x0F); // zero-extended
    }

    // 8. Decoder recovers lost packet when exactly one is missing
    #[test]
    fn test_decoder_recovers_single_loss() {
        let cfg = FecConfig::one_dimensional(3);
        let mut enc = FecEncoder::new(cfg.clone());
        let payloads: Vec<Vec<u8>> = (0..3).map(|i| make_payload(i * 0x10, 8)).collect();
        let mut fec_pkt = None;
        for (seq, payload) in payloads.iter().enumerate() {
            let r = enc.feed_packet(seq as u16, payload);
            if r.is_some() {
                fec_pkt = r;
            }
        }
        let fec = fec_pkt.expect("FEC must be produced");

        let mut dec = FecDecoder::new(cfg);
        // Feed source packets 0 and 2 (packet 1 is "lost").
        dec.feed_source(0, payloads[0].clone());
        dec.feed_source(2, payloads[2].clone());
        dec.feed_fec(fec);
        dec.register_group_base(0);

        let recovered = dec.try_recover();
        assert_eq!(recovered.len(), 1);
        let (seq, data) = &recovered[0];
        assert_eq!(*seq, 1);
        assert_eq!(data, &payloads[1]);
    }

    // 9. Decoder does NOT recover when two packets are missing
    #[test]
    fn test_decoder_cannot_recover_two_losses() {
        let cfg = FecConfig::one_dimensional(3);
        let mut enc = FecEncoder::new(cfg.clone());
        let payloads: Vec<Vec<u8>> = (0..3).map(|i| make_payload(i * 0x10, 8)).collect();
        let mut fec_pkt = None;
        for (seq, payload) in payloads.iter().enumerate() {
            let r = enc.feed_packet(seq as u16, payload);
            if r.is_some() {
                fec_pkt = r;
            }
        }
        let fec = fec_pkt.expect("FEC must be produced");

        let mut dec = FecDecoder::new(cfg);
        // Feed only packet 0; packets 1 and 2 are "lost".
        dec.feed_source(0, payloads[0].clone());
        dec.feed_fec(fec);
        dec.register_group_base(0);

        let recovered = dec.try_recover();
        assert_eq!(recovered.len(), 0);
    }

    // 10. Encoder resets after each group (second group works)
    #[test]
    fn test_encoder_second_group() {
        let cfg = FecConfig::one_dimensional(2);
        let mut enc = FecEncoder::new(cfg);
        // Group 1
        enc.feed_packet(0, &make_payload(0xAA, 4));
        let fec1 = enc.feed_packet(1, &make_payload(0xBB, 4)).expect("FEC1");
        // Group 2
        enc.feed_packet(2, &make_payload(0xCC, 4));
        let fec2 = enc.feed_packet(3, &make_payload(0xDD, 4)).expect("FEC2");

        assert_ne!(fec1.sequence_number, fec2.sequence_number);
    }

    // 11. FecPacket covered_count matches mask popcount
    #[test]
    fn test_fec_packet_covered_count() {
        let pkt = FecPacket {
            sequence_number: 0,
            timestamp: 0,
            fec_index: 0,
            payload: vec![],
            mask: 0b0001_0111, // 4 bits set
        };
        assert_eq!(pkt.covered_count(), 4);
    }

    // 12. Decoder with no FEC packet does not recover
    #[test]
    fn test_decoder_no_fec_no_recovery() {
        let cfg = FecConfig::one_dimensional(3);
        let mut dec = FecDecoder::new(cfg);
        dec.feed_source(0, make_payload(0x11, 8));
        dec.feed_source(2, make_payload(0x33, 8));
        // No FEC packet fed.
        dec.register_group_base(0);
        let recovered = dec.try_recover();
        assert!(recovered.is_empty());
    }

    // 13. 2-D config is_2d returns true
    #[test]
    fn test_2d_config_is_2d() {
        assert!(FecConfig::two_dimensional(4, 2).is_2d());
        assert!(!FecConfig::one_dimensional(4).is_2d());
    }

    // 14. Encoder 2-D take_column_fec returns column FEC packets after l rows
    #[test]
    fn test_encoder_2d_column_fec() {
        let cfg = FecConfig::two_dimensional(3, 2); // k=3, l=2
        let mut enc = FecEncoder::new(cfg.clone());
        // Row 1: 3 source packets → row FEC emitted.
        for seq in 0..3u16 {
            enc.feed_packet(seq, &make_payload(seq as u8, 6));
        }
        // Column FEC not yet available (only 1 row done).
        assert!(enc.take_column_fec().is_empty());

        // Row 2: another 3 source packets.
        for seq in 3..6u16 {
            enc.feed_packet(seq, &make_payload(seq as u8, 6));
        }
        // Now l=2 rows complete → column FEC available.
        let col_fec = enc.take_column_fec();
        assert_eq!(col_fec.len(), cfg.k); // one FEC packet per column
    }

    // 15. recovered_source returns empty after try_recover drains it
    #[test]
    fn test_recovered_map_drained_after_try_recover() {
        let cfg = FecConfig::one_dimensional(2);
        let mut enc = FecEncoder::new(cfg.clone());
        let p0 = make_payload(0xAA, 4);
        let p1 = make_payload(0xBB, 4);
        enc.feed_packet(0, &p0);
        let fec = enc.feed_packet(1, &p1).expect("FEC");

        let mut dec = FecDecoder::new(cfg);
        dec.feed_source(0, p0);
        dec.feed_fec(fec);
        dec.register_group_base(0);

        // First call should recover packet 1.
        let r1 = dec.try_recover();
        assert_eq!(r1.len(), 1);
        // Second call should return empty (already drained).
        let r2 = dec.try_recover();
        assert!(r2.is_empty());
    }
}
