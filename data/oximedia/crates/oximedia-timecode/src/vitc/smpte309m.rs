//! SMPTE 309M HD-VITC ANC packet encoding and decoding.
//!
//! SMPTE 309M defines HD-SDI timecode carried in ANC (ancillary data) packets.
//! The standard specifies DID=0x60, SDID=0x60, and 16 10-bit user data words.
//!
//! # Packet Structure
//! Each word is 10 bits:
//! - `Bits[7:0]`: data payload (BCD timecode or binary group byte)
//! - `Bits[9:8]`: odd parity — exactly one of these two bits is set so that the
//!   total number of 1-bits across all 10 positions is odd.
//!
//! ## Word layout (words 0–3: timecode; words 4–15: binary groups)
//! | Word | Content                                        |
//! |------|------------------------------------------------|
//! | 0    | `frames_units[3:0]` | `frames_tens[7:4]`; bit 8 = drop-frame flag |
//! | 1    | `seconds_units[3:0]` | `seconds_tens[7:4]`                        |
//! | 2    | `minutes_units[3:0]` | `minutes_tens[7:4]`; bit 8 = color-frame (0)|
//! | 3    | `hours_units[3:0]` | `hours_tens[7:4]`                            |
//! | 4–7  | binary group bytes 0–3 (from `binary_groups`)                  |
//! | 8–15 | binary group bytes 4–11 (zero-padded)                           |

use crate::{FrameRate, Timecode};

/// SMPTE 309M ancillary timecode packet for HD-SDI.
///
/// `DID=0x60`, `SDID=0x60`, 16 10-bit user data words.
/// `Bits[7:0]` of each word carry data; `bits[9:8]` carry odd parity.
#[derive(Debug, Clone, PartialEq)]
pub struct Smpte309mPacket {
    /// DID byte — always 0x60 for SMPTE 309M timecode.
    pub did: u8,
    /// SDID byte — always 0x60 for SMPTE 309M timecode.
    pub sdid: u8,
    /// 16 10-bit words (`bits[7:0]` = data; `bits[9:8]` = parity).
    pub payload: [u16; 16],
}

/// Compute odd parity for a byte per SMPTE 291M ANC encoding.
///
/// SMPTE 291M uses 10-bit words where bits[7:0] carry data and bits[9:8]
/// carry parity such that the total number of 1-bits across all 10 positions
/// is odd.  The parity encoding follows the convention:
/// - bit 9 (`b9`, "P"): 1 iff `data` has an **even** number of 1-bits
///   (so that `bits[8:0]` evaluated with b9 in position 8 has odd parity).
/// - bit 8 (`b8`, "P̄"): always 0 (not used; b9 alone carries the correction).
///
/// This ensures `popcount(bits[9:0]) % 2 == 1` (odd) for every byte value,
/// which the receiver validates via [`parity_ok`].
fn with_parity(data: u8) -> u16 {
    // Count 1-bits in the 8-bit data payload.
    let ones = data.count_ones();
    // If already odd → no parity bit needed → bits[9:8] = 0b00.
    // If even        → set bit 9 to make total (data_ones + 1) odd → bits[9:8] = 0b10.
    if ones % 2 == 0 {
        // Even data → set bit 9 to flip parity to odd.
        (1u16 << 9) | u16::from(data)
    } else {
        // Odd data → total already odd → no parity bit needed.
        u16::from(data)
    }
}

/// Verify that the parity of a 10-bit word is correct (odd parity over bits[9:0]).
fn parity_ok(word: u16) -> bool {
    (word & 0x3ff).count_ones() % 2 == 1
}

/// Encode a [`Timecode`] into a SMPTE 309M ANC packet.
///
/// `binary_groups` — 4 bytes of user binary group data (packed into words 4–7;
/// words 8–15 are zero-padded binary group bytes).  Pass `[0u8; 4]` if unused.
///
/// The returned packet has `DID=0x60`, `SDID=0x60`, and 16 parity-protected
/// 10-bit words encoding the timecode and binary group data.
pub fn encode_anc_timecode(tc: &Timecode, binary_groups: [u8; 4]) -> Smpte309mPacket {
    let frames_units = tc.frames % 10;
    let frames_tens = tc.frames / 10;
    let seconds_units = tc.seconds % 10;
    let seconds_tens = tc.seconds / 10;
    let minutes_units = tc.minutes % 10;
    let minutes_tens = tc.minutes / 10;
    let hours_units = tc.hours % 10;
    let hours_tens = tc.hours / 10;

    // Word 0: frames BCD, drop-frame flag in bit 8 of the data byte.
    // Per SMPTE 309M the drop-frame flag is carried in bit 7 of word-0 data
    // (the MSB of the data byte), but the specification also places it as
    // part of the parity calculation. We encode it as bit 7 of the 8-bit
    // data byte so that the parity helper sees it correctly.
    let word0_data: u8 = frames_units | (frames_tens << 4);
    // Drop-frame flag: embed in bit 7 of the data byte (after BCD digits
    // occupy bits[6:0] for frames ≤ 29).  For frames up to 29: tens ≤ 2
    // (bits 6:4), units ≤ 9 (bits 3:0), so bit 7 is free.
    let drop_flag: u8 = if tc.frame_rate.drop_frame { 0x80 } else { 0x00 };
    let word0_data = word0_data | drop_flag;

    let word1_data: u8 = seconds_units | (seconds_tens << 4);
    let word2_data: u8 = minutes_units | (minutes_tens << 4);
    // color-frame flag would occupy bit 7 of word 2; always 0 here.
    let word3_data: u8 = hours_units | (hours_tens << 4);

    // Build all 16 words with parity.
    let mut payload = [0u16; 16];
    payload[0] = with_parity(word0_data);
    payload[1] = with_parity(word1_data);
    payload[2] = with_parity(word2_data);
    payload[3] = with_parity(word3_data);

    // Words 4–7: caller-supplied binary group bytes 0–3.
    for (i, &bg) in binary_groups.iter().enumerate() {
        payload[4 + i] = with_parity(bg);
    }
    // Words 8–15: zero-padded binary group bytes 4–11.
    for i in 8..16usize {
        payload[i] = with_parity(0x00);
    }

    Smpte309mPacket {
        did: 0x60,
        sdid: 0x60,
        payload,
    }
}

/// Decode a SMPTE 309M ANC packet into a [`Timecode`] and binary group bytes.
///
/// Returns `None` if:
/// - `DID` or `SDID` do not equal `0x60`, or
/// - any of the 16 payload words fails its odd-parity check.
pub fn decode_anc_timecode(pkt: &Smpte309mPacket) -> Option<(Timecode, [u8; 4])> {
    if pkt.did != 0x60 || pkt.sdid != 0x60 {
        return None;
    }

    // Verify parity on all 16 words.
    for &word in &pkt.payload {
        if !parity_ok(word) {
            return None;
        }
    }

    // Extract data bytes (bits[7:0]).
    let w0 = (pkt.payload[0] & 0xff) as u8;
    let w1 = (pkt.payload[1] & 0xff) as u8;
    let w2 = (pkt.payload[2] & 0xff) as u8;
    let w3 = (pkt.payload[3] & 0xff) as u8;

    let drop_frame = (w0 & 0x80) != 0;

    let frames_units = w0 & 0x0f;
    let frames_tens = (w0 & 0x70) >> 4; // bits[6:4]
    let frames = frames_tens * 10 + frames_units;

    let seconds_units = w1 & 0x0f;
    let seconds_tens = (w1 & 0x70) >> 4;
    let seconds = seconds_tens * 10 + seconds_units;

    let minutes_units = w2 & 0x0f;
    let minutes_tens = (w2 & 0x70) >> 4;
    let minutes = minutes_tens * 10 + minutes_units;

    let hours_units = w3 & 0x0f;
    let hours_tens = (w3 & 0x70) >> 4;
    let hours = hours_tens * 10 + hours_units;

    // Binary groups: words 4–7.
    let mut binary_groups = [0u8; 4];
    for (i, slot) in binary_groups.iter_mut().enumerate() {
        *slot = (pkt.payload[4 + i] & 0xff) as u8;
    }

    // Reconstruct FrameRate: use drop-frame flag + 30fps as the default;
    // HD-SDI SMPTE 309M carries 29.97DF or 30NDF most commonly.
    let frame_rate = if drop_frame {
        FrameRate::Fps2997DF
    } else {
        FrameRate::Fps30
    };

    let tc = Timecode::from_raw_fields(hours, minutes, seconds, frames, 30, drop_frame, 0);

    // Validate the reconstructed timecode fields before returning.
    // Re-encode to check: use the public constructor for validation.
    let _ = Timecode::new(hours, minutes, seconds, frames, frame_rate).ok()?;

    Some((tc, binary_groups))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn make_tc(h: u8, m: u8, s: u8, f: u8, df: bool) -> Timecode {
        let rate = if df {
            FrameRate::Fps2997DF
        } else {
            FrameRate::Fps30
        };
        Timecode::new(h, m, s, f, rate).expect("valid timecode")
    }

    #[test]
    fn test_smpte309m_round_trip_zero() {
        let tc = make_tc(0, 0, 0, 0, false);
        let pkt = encode_anc_timecode(&tc, [0u8; 4]);
        let (decoded, _bg) = decode_anc_timecode(&pkt).expect("decode must succeed");
        assert_eq!(decoded.hours, tc.hours);
        assert_eq!(decoded.minutes, tc.minutes);
        assert_eq!(decoded.seconds, tc.seconds);
        assert_eq!(decoded.frames, tc.frames);
    }

    #[test]
    fn test_smpte309m_round_trip_max() {
        // 23:59:59:29 is valid for 30fps NDF.
        let tc = make_tc(23, 59, 59, 29, false);
        let pkt = encode_anc_timecode(&tc, [0u8; 4]);
        let (decoded, _bg) = decode_anc_timecode(&pkt).expect("decode must succeed");
        assert_eq!(decoded.hours, 23);
        assert_eq!(decoded.minutes, 59);
        assert_eq!(decoded.seconds, 59);
        assert_eq!(decoded.frames, 29);
    }

    #[test]
    fn test_smpte309m_parity_correct() {
        let tc = make_tc(1, 23, 45, 12, false);
        let pkt = encode_anc_timecode(&tc, [0xAB, 0xCD, 0xEF, 0x01]);
        // Every word in the payload must have odd parity across bits[9:0].
        for (i, &word) in pkt.payload.iter().enumerate() {
            assert!(
                parity_ok(word),
                "word {i} failed parity check: 0x{word:03x}"
            );
        }
    }

    #[test]
    fn test_smpte309m_binary_groups_passthrough() {
        let tc = make_tc(0, 0, 0, 0, false);
        let bg_in = [0x12u8, 0x34, 0x56, 0x78];
        let pkt = encode_anc_timecode(&tc, bg_in);
        let (_decoded, bg_out) = decode_anc_timecode(&pkt).expect("decode must succeed");
        assert_eq!(bg_out, bg_in);
    }

    #[test]
    fn test_smpte309m_drop_frame_flag() {
        // 29.97 DF timecode: 00:10:00:02 is a valid DF position.
        let tc = make_tc(0, 10, 0, 2, true);
        let pkt = encode_anc_timecode(&tc, [0u8; 4]);
        // Word 0 data byte should have bit 7 set for drop-frame.
        let w0_data = (pkt.payload[0] & 0xff) as u8;
        assert!(
            (w0_data & 0x80) != 0,
            "drop-frame flag not set in word 0: 0x{w0_data:02x}"
        );
        // And decode should produce a drop-frame timecode.
        let (decoded, _) = decode_anc_timecode(&pkt).expect("decode must succeed");
        assert!(decoded.frame_rate.drop_frame);
    }

    #[test]
    fn test_smpte309m_did_sdid_mismatch_returns_none() {
        let tc = make_tc(0, 0, 0, 0, false);
        let mut pkt = encode_anc_timecode(&tc, [0u8; 4]);
        pkt.did = 0x61; // wrong DID
        assert!(decode_anc_timecode(&pkt).is_none());
    }

    #[test]
    fn test_smpte309m_parity_error_returns_none() {
        let tc = make_tc(0, 0, 0, 0, false);
        let mut pkt = encode_anc_timecode(&tc, [0u8; 4]);
        // Flip a parity bit in word 0 to corrupt parity.
        pkt.payload[0] ^= 0x100;
        assert!(decode_anc_timecode(&pkt).is_none());
    }
}
