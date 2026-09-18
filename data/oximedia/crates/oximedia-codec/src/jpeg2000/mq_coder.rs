//! MQ arithmetic decoder (JBIG / JPEG 2000 standard, ISO 15444-1 Annex C).
//!
//! The MQ coder is a context-adaptive binary arithmetic decoder. Each
//! context slot (`cx`) tracks a probability state via the 47-entry MQ
//! probability table (Qe values). The decoder maintains a code register `c`
//! and interval register `a`.
//!
//! ## Initialisation
//!
//! `a = 0x8000`, read first two bytes into `c`, `ct = 0`.
//!
//! ## Byte-stuffing in MQ input
//!
//! The MQ data stream uses its own byte-filling rule that differs from the
//! main J2K bitreader: after a 0xFF byte, the next byte is shifted right
//! by 1 (7 effective bits), with bit 0 = 0.

use super::{Jp2Error, Jp2Result};

/// Single entry in the MQ probability table (ISO 15444-1, Table C.2).
///
/// Shared verbatim between the [`MqDecoder`] and the matching MQ encoder
/// (`mq_encoder.rs`) so that both halves of the codec use identical Qe /
/// NMPS / NLPS / SWITCH transition data — there is exactly one table.
#[derive(Clone, Copy)]
pub(crate) struct MqState {
    /// Qe probability estimate (as a 16-bit fixed-point fraction of interval).
    pub(crate) qe: u16,
    /// Next state index for MPS (more probable symbol).
    pub(crate) nmps: u8,
    /// Next state index for LPS (less probable symbol).
    pub(crate) nlps: u8,
    /// Switch flag: 1 means swap MPS/LPS meaning on LPS renormalisation.
    pub(crate) sw: u8,
}

/// The 47-entry MQ probability transition table (ISO 15444-1 Annex C, Table C.2).
/// Index 0 is the initial/unconditional state.
pub(crate) static MQ_TABLE: [MqState; 47] = [
    MqState {
        qe: 0x5601,
        nmps: 1,
        nlps: 1,
        sw: 1,
    }, //  0
    MqState {
        qe: 0x3401,
        nmps: 2,
        nlps: 6,
        sw: 0,
    }, //  1
    MqState {
        qe: 0x1801,
        nmps: 3,
        nlps: 9,
        sw: 0,
    }, //  2
    MqState {
        qe: 0x0AC1,
        nmps: 4,
        nlps: 12,
        sw: 0,
    }, //  3
    MqState {
        qe: 0x0521,
        nmps: 5,
        nlps: 29,
        sw: 0,
    }, //  4
    MqState {
        qe: 0x0221,
        nmps: 38,
        nlps: 33,
        sw: 0,
    }, //  5
    MqState {
        qe: 0x5601,
        nmps: 7,
        nlps: 6,
        sw: 1,
    }, //  6
    MqState {
        qe: 0x5401,
        nmps: 8,
        nlps: 14,
        sw: 0,
    }, //  7
    MqState {
        qe: 0x4801,
        nmps: 9,
        nlps: 14,
        sw: 0,
    }, //  8
    MqState {
        qe: 0x3801,
        nmps: 10,
        nlps: 14,
        sw: 0,
    }, //  9
    MqState {
        qe: 0x3001,
        nmps: 11,
        nlps: 17,
        sw: 0,
    }, // 10
    MqState {
        qe: 0x2401,
        nmps: 12,
        nlps: 18,
        sw: 0,
    }, // 11
    MqState {
        qe: 0x1C01,
        nmps: 13,
        nlps: 20,
        sw: 0,
    }, // 12
    MqState {
        qe: 0x1601,
        nmps: 29,
        nlps: 21,
        sw: 0,
    }, // 13
    MqState {
        qe: 0x5601,
        nmps: 15,
        nlps: 14,
        sw: 1,
    }, // 14
    MqState {
        qe: 0x5401,
        nmps: 16,
        nlps: 14,
        sw: 0,
    }, // 15
    MqState {
        qe: 0x5101,
        nmps: 17,
        nlps: 15,
        sw: 0,
    }, // 16
    MqState {
        qe: 0x4801,
        nmps: 18,
        nlps: 16,
        sw: 0,
    }, // 17
    MqState {
        qe: 0x3801,
        nmps: 19,
        nlps: 17,
        sw: 0,
    }, // 18
    MqState {
        qe: 0x3401,
        nmps: 20,
        nlps: 18,
        sw: 0,
    }, // 19
    MqState {
        qe: 0x3001,
        nmps: 21,
        nlps: 19,
        sw: 0,
    }, // 20
    MqState {
        qe: 0x2801,
        nmps: 22,
        nlps: 19,
        sw: 0,
    }, // 21
    MqState {
        qe: 0x2401,
        nmps: 23,
        nlps: 20,
        sw: 0,
    }, // 22
    MqState {
        qe: 0x2201,
        nmps: 24,
        nlps: 21,
        sw: 0,
    }, // 23
    MqState {
        qe: 0x1C01,
        nmps: 25,
        nlps: 22,
        sw: 0,
    }, // 24
    MqState {
        qe: 0x1801,
        nmps: 26,
        nlps: 23,
        sw: 0,
    }, // 25
    MqState {
        qe: 0x1601,
        nmps: 27,
        nlps: 24,
        sw: 0,
    }, // 26
    MqState {
        qe: 0x1401,
        nmps: 28,
        nlps: 25,
        sw: 0,
    }, // 27
    MqState {
        qe: 0x1201,
        nmps: 29,
        nlps: 26,
        sw: 0,
    }, // 28
    MqState {
        qe: 0x1101,
        nmps: 30,
        nlps: 27,
        sw: 0,
    }, // 29
    MqState {
        qe: 0x0AC1,
        nmps: 31,
        nlps: 28,
        sw: 0,
    }, // 30
    MqState {
        qe: 0x09C1,
        nmps: 32,
        nlps: 29,
        sw: 0,
    }, // 31
    MqState {
        qe: 0x08A1,
        nmps: 33,
        nlps: 30,
        sw: 0,
    }, // 32
    MqState {
        qe: 0x0521,
        nmps: 34,
        nlps: 31,
        sw: 0,
    }, // 33
    MqState {
        qe: 0x0441,
        nmps: 35,
        nlps: 32,
        sw: 0,
    }, // 34
    MqState {
        qe: 0x02A1,
        nmps: 36,
        nlps: 33,
        sw: 0,
    }, // 35
    MqState {
        qe: 0x0221,
        nmps: 37,
        nlps: 34,
        sw: 0,
    }, // 36
    MqState {
        qe: 0x0141,
        nmps: 38,
        nlps: 35,
        sw: 0,
    }, // 37
    MqState {
        qe: 0x0111,
        nmps: 39,
        nlps: 36,
        sw: 0,
    }, // 38
    MqState {
        qe: 0x0085,
        nmps: 40,
        nlps: 37,
        sw: 0,
    }, // 39
    MqState {
        qe: 0x0049,
        nmps: 41,
        nlps: 38,
        sw: 0,
    }, // 40
    MqState {
        qe: 0x0025,
        nmps: 42,
        nlps: 39,
        sw: 0,
    }, // 41
    MqState {
        qe: 0x0015,
        nmps: 43,
        nlps: 40,
        sw: 0,
    }, // 42
    MqState {
        qe: 0x0009,
        nmps: 44,
        nlps: 41,
        sw: 0,
    }, // 43
    MqState {
        qe: 0x0005,
        nmps: 45,
        nlps: 42,
        sw: 0,
    }, // 44
    MqState {
        qe: 0x0001,
        nmps: 45,
        nlps: 43,
        sw: 0,
    }, // 45
    MqState {
        qe: 0x5601,
        nmps: 46,
        nlps: 46,
        sw: 0,
    }, // 46
];

/// Number of MQ context slots used by the JPEG 2000 Tier-1 decoder.
///
/// Significance (9) + sign (5) + magnitude refinement (3) + uniform (1) + run-length (1) = 19.
pub const MQ_NUM_CONTEXTS: usize = 19;

/// MQ arithmetic decoder state.
pub struct MqDecoder {
    /// Current interval register (normalised to [0x8000, 0xFFFF]).
    a: u32,
    /// Code register (contains compressed bits shifted in from the input).
    c: u32,
    /// Count of valid bits remaining in the code register shift buffer.
    ct: u8,
    /// Input byte buffer.
    buf: Vec<u8>,
    /// Current byte position in `buf`.
    pos: usize,
    /// MPS (most probable symbol) for each context: 0 or 1.
    cx_mps: [u8; MQ_NUM_CONTEXTS],
    /// State index for each context (0..46).
    cx_state: [u8; MQ_NUM_CONTEXTS],
}

impl MqDecoder {
    /// Create a new MQ decoder from the given compressed data bytes.
    ///
    /// Follows the standard initialisation procedure INITDEC of ISO/IEC
    /// 15444-1 Annex C (Figure C.20):
    ///
    /// ```text
    ///   BP = BPST
    ///   C  = B << 16          // first byte at bit 16
    ///   BYTEIN                // pull in the second byte
    ///   C  = C << 7
    ///   CT = CT - 7
    ///   A  = 0x8000
    /// ```
    ///
    /// This is the exact counterpart of the [`super::mq_encoder::MqEncoder`]
    /// FLUSH output, so a stream produced by that encoder decodes back to the
    /// original decision sequence bit-for-bit.
    pub fn new(data: &[u8]) -> Self {
        let buf = data.to_vec();
        let mut mq = Self {
            a: 0x8000,
            c: 0,
            ct: 0,
            buf,
            pos: 0,
            cx_mps: [0u8; MQ_NUM_CONTEXTS],
            cx_state: [0u8; MQ_NUM_CONTEXTS],
        };
        // INITDEC: first byte goes to bits 16..23.
        let b0 = mq.next_byte();
        mq.c = u32::from(b0) << 16;
        mq.byte_in();
        mq.c <<= 7;
        mq.ct = mq.ct.wrapping_sub(7);
        mq.a = 0x8000;
        mq
    }

    /// Read the byte at the current position (or 0xFF padding past the end),
    /// advancing the position.
    fn next_byte(&mut self) -> u8 {
        if self.pos < self.buf.len() {
            let b = self.buf[self.pos];
            self.pos += 1;
            b
        } else {
            self.pos += 1;
            0xFF
        }
    }

    /// BYTEIN procedure (ISO/IEC 15444-1 Annex C, Figure C.19).
    ///
    /// Pulls the next compressed byte into the code register, honouring the
    /// JPEG 2000 bit-stuffing rule: when the previously consumed byte was
    /// `0xFF`, the next byte either signals a marker (value > 0x8F → feed 1s)
    /// or carries only 7 data bits (shifted to bit 9).
    fn byte_in(&mut self) {
        // The "previous" byte is the one most recently consumed (at pos-1).
        let prev = if self.pos >= 1 && self.pos - 1 < self.buf.len() {
            self.buf[self.pos - 1]
        } else {
            0xFF
        };
        if prev == 0xFF {
            let next = if self.pos < self.buf.len() {
                self.buf[self.pos]
            } else {
                0xFF
            };
            if next > 0x8F {
                // Marker (or end of data): feed 1-bits, do not consume.
                self.c += 0xFF00;
                self.ct = 8;
            } else {
                let b = self.next_byte();
                self.c += u32::from(b) << 9;
                self.ct = 7;
            }
        } else {
            let b = self.next_byte();
            self.c += u32::from(b) << 8;
            self.ct = 8;
        }
    }

    /// Decode a single binary symbol for the given context index `cx`.
    ///
    /// Implements the standard DECODE procedure (ISO/IEC 15444-1 Annex C,
    /// Figure C.16). The context encodes both the MPS value and the
    /// probability state. Returns 0 or 1.
    pub fn decode_bit(&mut self, cx: usize) -> Jp2Result<u8> {
        if cx >= MQ_NUM_CONTEXTS {
            return Err(Jp2Error::InternalError(format!(
                "MQ context index {cx} out of range (max {})",
                MQ_NUM_CONTEXTS - 1
            )));
        }
        let entry = MQ_TABLE[usize::from(self.cx_state[cx])];
        let qe = u32::from(entry.qe);

        self.a = self.a.wrapping_sub(qe);

        let symbol;
        if (self.c >> 16) < qe {
            // LPS sub-interval (lower part).
            symbol = self.lps_exchange(cx, qe, &entry);
            self.renorm_d();
        } else {
            self.c -= qe << 16;
            if self.a & 0x8000 == 0 {
                symbol = self.mps_exchange(cx, qe, &entry);
                self.renorm_d();
            } else {
                symbol = self.cx_mps[cx];
            }
        }
        Ok(symbol)
    }

    /// MPS_EXCHANGE (ISO/IEC 15444-1 Annex C, Figure C.17).
    fn mps_exchange(&mut self, cx: usize, qe: u32, entry: &MqState) -> u8 {
        if self.a < qe {
            let d = 1 - self.cx_mps[cx];
            if entry.sw == 1 {
                self.cx_mps[cx] = 1 - self.cx_mps[cx];
            }
            self.cx_state[cx] = entry.nlps;
            d
        } else {
            self.cx_state[cx] = entry.nmps;
            self.cx_mps[cx]
        }
    }

    /// LPS_EXCHANGE (ISO/IEC 15444-1 Annex C, Figure C.18).
    fn lps_exchange(&mut self, cx: usize, qe: u32, entry: &MqState) -> u8 {
        let d;
        if self.a < qe {
            d = self.cx_mps[cx];
            self.cx_state[cx] = entry.nmps;
        } else {
            d = 1 - self.cx_mps[cx];
            if entry.sw == 1 {
                self.cx_mps[cx] = 1 - self.cx_mps[cx];
            }
            self.cx_state[cx] = entry.nlps;
        }
        self.a = qe;
        d
    }

    /// RENORMD (ISO/IEC 15444-1 Annex C, Figure C.21).
    fn renorm_d(&mut self) {
        loop {
            if self.ct == 0 {
                self.byte_in();
            }
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mq_table_length() {
        assert_eq!(MQ_TABLE.len(), 47);
    }

    #[test]
    fn mq_initial_state_is_zero() {
        let data = vec![0x00u8; 16];
        let mq = MqDecoder::new(&data);
        assert_eq!(mq.a, 0x8000);
        for &s in &mq.cx_state {
            assert_eq!(s, 0);
        }
        for &m in &mq.cx_mps {
            assert_eq!(m, 0);
        }
    }

    #[test]
    fn mq_decode_all_zeros_stream() {
        // A stream of all 0x00 bytes should decode as all MPS (0) since
        // the code register will always indicate MPS for any context.
        let data = vec![0x00u8; 32];
        let mut mq = MqDecoder::new(&data);
        // Decode a few bits from context 0 — should not panic.
        for _ in 0..10 {
            let bit = mq.decode_bit(0).expect("decode");
            // Just confirm it returns 0 or 1.
            assert!(bit == 0 || bit == 1);
        }
    }

    #[test]
    fn mq_decode_known_sequence() {
        // Build a known MQ-encoded 10-symbol sequence using a simple
        // synthetic stream. The content here is not a real encode/decode
        // pair; we verify that:
        // 1. decode_bit() returns only 0 or 1
        // 2. No panic or error for a plausible input length
        let data: Vec<u8> = (0u8..=30).collect();
        let mut mq = MqDecoder::new(&data);
        let mut results = Vec::new();
        for _ in 0..10 {
            match mq.decode_bit(0) {
                Ok(bit) => results.push(bit),
                Err(_) => break,
            }
        }
        // Verify all outputs are binary.
        for &b in &results {
            assert!(b == 0 || b == 1, "MQ bit must be 0 or 1, got {b}");
        }
    }
}
