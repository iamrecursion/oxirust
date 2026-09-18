//! CEA-608 closed caption format (Line 21, NTSC)
//!
//! Implements encode and decode of the binary CEA-608 byte-pair stream.
//! The format uses odd parity on the high bit of each byte, channel 1 commands
//! are prefixed with 0x14, channel 2 with 0x1C.  Control code second bytes
//! match the low byte of the constants in [`control_codes`].

#[cfg(not(feature = "cea"))]
use crate::error::CaptionError;
use crate::error::Result;
use crate::formats::{FormatParser, FormatWriter};
use crate::types::CaptionTrack;
#[cfg(feature = "cea")]
use crate::types::{Caption, Language, Timestamp};

/// CEA-608 format parser.
pub struct Cea608Parser;

impl FormatParser for Cea608Parser {
    fn parse(&self, data: &[u8]) -> Result<CaptionTrack> {
        #[cfg(feature = "cea")]
        {
            parse_cea608(data)
        }
        #[cfg(not(feature = "cea"))]
        {
            let _ = data;
            Err(CaptionError::FeatureNotEnabled(
                "CEA-608 parsing requires 'cea' feature".to_string(),
            ))
        }
    }
}

/// CEA-608 format writer.
pub struct Cea608Writer;

impl FormatWriter for Cea608Writer {
    fn write(&self, track: &CaptionTrack) -> Result<Vec<u8>> {
        #[cfg(feature = "cea")]
        {
            encode_cea608(track)
        }
        #[cfg(not(feature = "cea"))]
        {
            let _ = track;
            Err(CaptionError::FeatureNotEnabled(
                "CEA-608 writing requires 'cea' feature".to_string(),
            ))
        }
    }
}

/// CEA-608 control codes (raw u16 value as transmitted on-wire before parity).
///
/// Each constant stores `(channel_byte << 8) | command_byte`.  For channel 1
/// the channel byte is `0x14`; for channel 2 it is `0x1C`.  The command byte
/// is the second byte of the pair.  Both bytes have odd parity added during
/// encoding via [`add_parity`].
pub mod control_codes {
    /// Resume caption loading (pop-on)
    pub const RCL: u16 = 0x9420;
    /// Resume direct captioning (paint-on)
    pub const RDC: u16 = 0x9429;
    /// Erase displayed memory
    pub const EDM: u16 = 0x942C;
    /// Carriage return (roll-up)
    pub const CR: u16 = 0x942D;
    /// Erase non-displayed memory
    pub const ENM: u16 = 0x942E;
    /// End of caption (flip memories — pop-on display trigger)
    pub const EOC: u16 = 0x942F;
}

// ─── Parity helpers ───────────────────────────────────────────────────────────

/// Set bit 7 of `byte` so the total number of 1-bits (over bits 0–7) is odd
/// (CEA-608 odd parity).  Only the low 7 bits carry data; bit 7 is the parity
/// bit.
#[must_use]
pub fn add_parity(byte: u8) -> u8 {
    let ones = (byte & 0x7F).count_ones();
    // Odd parity: total 1-bits including the parity bit must be odd.
    if ones % 2 == 0 {
        // Currently even — set the parity bit to make it odd.
        byte | 0x80
    } else {
        // Already odd — clear the parity bit.
        byte & 0x7F
    }
}

/// Strip the parity bit from a CEA-608 byte, returning the low 7 data bits.
#[must_use]
pub fn strip_parity(byte: u8) -> u8 {
    byte & 0x7F
}

// ─── CEA-608 channel-1 command byte constants (after stripping channel prefix) ─

/// Channel-1 command byte for RCL (Resume Caption Loading).
const CMD_RCL: u8 = 0x20;
/// Channel-1 command byte for RDC (Resume Direct Captioning).
const CMD_RDC: u8 = 0x29;
/// Channel-1 command byte for EDM (Erase Displayed Memory).
const CMD_EDM: u8 = 0x2C;
/// Channel-1 command byte for CR (Carriage Return).
const CMD_CR: u8 = 0x2D;
/// Channel-1 command byte for ENM (Erase Non-Displayed Memory).
const CMD_ENM: u8 = 0x2E;
/// Channel-1 command byte for EOC (End of Caption / Flip Memories).
const CMD_EOC: u8 = 0x2F;

/// Channel byte for CEA-608 channel 1 (after parity strip).
const CH1: u8 = 0x14;
/// Channel byte for CEA-608 channel 2 (after parity strip).
const CH2: u8 = 0x1C;

// ─── Encoder ─────────────────────────────────────────────────────────────────

#[cfg(feature = "cea")]
fn encode_cea608(track: &CaptionTrack) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();

    for caption in &track.captions {
        // ENM — erase non-displayed memory before loading.
        emit_control(&mut out, CH1, CMD_ENM);

        // Encode text characters as parity-protected byte pairs.
        // CEA-608 row capacity is 32 printable columns; each visible char
        // is sent as a two-byte pair (char, null-filler with parity).
        for ch in caption.text.chars().take(32) {
            let b = ch as u32;
            if (0x20..=0x7E).contains(&b) {
                out.push(add_parity(b as u8));
                // Null filler with correct odd parity: 0x80 has one 1-bit → odd.
                out.push(0x80);
            }
        }

        // EOC — flip non-displayed to displayed memory (pop-on display).
        emit_control(&mut out, CH1, CMD_EOC);
    }

    Ok(out)
}

/// Emit a two-byte CEA-608 control code pair with odd parity on each byte.
#[cfg(feature = "cea")]
fn emit_control(out: &mut Vec<u8>, channel: u8, command: u8) {
    out.push(add_parity(channel));
    out.push(add_parity(command));
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

#[cfg(feature = "cea")]
fn parse_cea608(data: &[u8]) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(Language::english());
    let mut current_text = String::new();
    // Synthetic timecode: each 33.37 ms frame at 29.97 fps.
    let frame_micros: i64 = 33_367; // ≈ 1_000_000 / 29.97
    let mut frame: i64 = 0;

    let mut i = 0usize;
    while i + 1 < data.len() {
        let b1 = strip_parity(data[i]);
        let b2 = strip_parity(data[i + 1]);
        i += 2;

        // Control code: b1 is CH1 (0x14) or CH2 (0x1C).
        if b1 == CH1 || b1 == CH2 {
            match b2 {
                CMD_RCL => {} // Resume Caption Loading — begin pop-on session.
                CMD_RDC => {} // Resume Direct Captioning — paint-on mode.
                CMD_EDM => {} // Erase Displayed Memory — clear screen.
                CMD_CR => {
                    // Carriage Return — roll-up scroll.
                    if !current_text.trim().is_empty() {
                        flush_caption(&mut track, &mut current_text, frame, frame_micros)?;
                    }
                }
                CMD_ENM => {
                    // Erase Non-Displayed Memory — discard any pending text.
                    current_text.clear();
                }
                CMD_EOC => {
                    // End of Caption — move non-displayed to displayed memory.
                    if !current_text.trim().is_empty() {
                        flush_caption(&mut track, &mut current_text, frame, frame_micros)?;
                    }
                }
                _ => {} // Other control codes (tab offsets, preamble, etc.) ignored.
            }
        } else if b1 >= 0x20 && b1 <= 0x7E {
            // Printable ASCII pair.
            current_text.push(b1 as char);
            if b2 >= 0x20 && b2 <= 0x7E {
                current_text.push(b2 as char);
            }
        }
        // Advance virtual frame counter for timing reconstruction.
        frame += 1;
    }

    // Flush any remaining text.
    if !current_text.trim().is_empty() {
        flush_caption(&mut track, &mut current_text, frame, frame_micros)?;
    }

    Ok(track)
}

/// Emit a [`Caption`] with a synthetic timestamp derived from the frame counter
/// and append it to `track`.  Resets `current_text` on success.
#[cfg(feature = "cea")]
fn flush_caption(
    track: &mut CaptionTrack,
    current_text: &mut String,
    frame: i64,
    frame_micros: i64,
) -> Result<()> {
    let text = current_text.trim().to_string();
    current_text.clear();

    let start = Timestamp::from_micros(frame * frame_micros);
    // Heuristic end time: give each caption a 2-second display window.
    let end = Timestamp::from_micros(frame * frame_micros + 2_000_000);

    let caption = Caption::new(start, end, text);
    track.add_caption(caption)
}

// ─── SCC integration tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        add_parity, strip_parity, Cea608Parser, Cea608Writer, CH1, CMD_ENM, CMD_EOC, CMD_RCL,
    };
    use crate::formats::{FormatParser, FormatWriter};
    use crate::types::CaptionTrack;
    #[cfg(feature = "cea")]
    use crate::types::{Caption, Language, Timestamp};
    #[cfg(not(feature = "cea"))]
    use crate::{error::CaptionError, types::Language};

    // ── Parity unit tests ──────────────────────────────────────────────────────

    #[test]
    fn test_add_parity_even_input() {
        // 'A' = 0x41 = 0b0100_0001 → 2 ones (even) → parity bit set → 0xC1
        let result = add_parity(b'A');
        let ones = result.count_ones();
        assert_eq!(
            ones % 2,
            1,
            "add_parity result must have odd number of 1-bits"
        );
    }

    #[test]
    fn test_add_parity_odd_input() {
        // 'B' = 0x42 = 0b0100_0010 → 2 ones (even) → parity bit set
        // 'C' = 0x43 = 0b0100_0011 → 3 ones (odd) → parity bit clear
        for byte in 0x20u8..=0x7Eu8 {
            let p = add_parity(byte);
            assert_eq!(
                p.count_ones() % 2,
                1,
                "byte 0x{byte:02X} → parity 0x{p:02X}: expected odd 1-count"
            );
        }
    }

    #[test]
    fn test_strip_parity_roundtrip() {
        for byte in 0x20u8..=0x7Eu8 {
            let with_parity = add_parity(byte);
            let stripped = strip_parity(with_parity);
            assert_eq!(stripped, byte & 0x7F);
        }
    }

    // ── Encode / decode round-trip tests (feature = "cea") ───────────────────

    #[cfg(feature = "cea")]
    #[test]
    fn test_encode_single_caption() {
        let mut track = CaptionTrack::new(Language::english());
        let caption = Caption::new(
            Timestamp::zero(),
            Timestamp::from_secs(3),
            "Hello world".to_string(),
        );
        track
            .add_caption(caption)
            .expect("adding caption to track should succeed");

        let writer = Cea608Writer;
        let bytes = writer
            .write(&track)
            .expect("CEA-608 encoding should succeed");
        assert!(!bytes.is_empty(), "encoded output must not be empty");
        // Every byte must satisfy odd parity.
        for &b in &bytes {
            assert_eq!(
                b.count_ones() % 2,
                1,
                "byte 0x{b:02X} does not satisfy odd parity"
            );
        }
    }

    #[cfg(feature = "cea")]
    #[test]
    fn test_encode_decode_roundtrip() {
        let mut track = CaptionTrack::new(Language::english());
        let texts = ["Hello world", "Testing CEA-608"];
        for (i, &text) in texts.iter().enumerate() {
            let start = Timestamp::from_secs(i as i64 * 4);
            let end = Timestamp::from_secs(i as i64 * 4 + 3);
            track
                .add_caption(Caption::new(start, end, text.to_string()))
                .expect("adding caption to track should succeed");
        }

        let writer = Cea608Writer;
        let encoded = writer
            .write(&track)
            .expect("CEA-608 encoding should succeed");

        let parser = Cea608Parser;
        let decoded = parser
            .parse(&encoded)
            .expect("CEA-608 decoding should succeed");

        assert_eq!(
            decoded.captions.len(),
            texts.len(),
            "decoded caption count should match encoded count"
        );
        for (original, decoded_cap) in texts.iter().zip(decoded.captions.iter()) {
            assert_eq!(
                decoded_cap.text.trim(),
                *original,
                "decoded text should match original"
            );
        }
    }

    #[cfg(feature = "cea")]
    #[test]
    fn test_decode_control_codes_only_produces_no_captions() {
        // ENM followed by EOC with no text should produce no captions.
        let mut data = Vec::new();
        data.push(add_parity(CH1));
        data.push(add_parity(CMD_ENM));
        data.push(add_parity(CH1));
        data.push(add_parity(CMD_EOC));

        let parser = Cea608Parser;
        let track = parser
            .parse(&data)
            .expect("parsing control-only stream should succeed");
        assert!(
            track.captions.is_empty(),
            "control-only stream should produce no captions"
        );
    }

    #[cfg(feature = "cea")]
    #[test]
    fn test_decode_eoc_triggers_caption_flush() {
        // Encode "Hi" manually: ENM, 'H', 'i', EOC.
        let mut data = Vec::new();
        data.push(add_parity(CH1));
        data.push(add_parity(CMD_ENM));
        // 'H' = 0x48, 'i' = 0x69 — emit as printable pair.
        data.push(add_parity(b'H'));
        data.push(add_parity(b'i'));
        data.push(add_parity(CH1));
        data.push(add_parity(CMD_EOC));

        let parser = Cea608Parser;
        let track = parser
            .parse(&data)
            .expect("parsing 'Hi' stream should succeed");
        assert_eq!(track.captions.len(), 1);
        assert_eq!(track.captions[0].text, "Hi");
    }

    #[cfg(feature = "cea")]
    #[test]
    fn test_scc_style_stream() {
        // Simulate a minimal SCC-like CEA-608 stream: RCL → text → EOC.
        let mut data = Vec::new();
        // Channel 1 RCL (Resume Caption Loading)
        data.push(add_parity(CH1));
        data.push(add_parity(CMD_RCL));
        // Text: "Test"
        for &b in b"Test" {
            data.push(add_parity(b));
            data.push(0x80); // null filler with odd parity (already 1 bit)
        }
        // EOC
        data.push(add_parity(CH1));
        data.push(add_parity(CMD_EOC));

        let parser = Cea608Parser;
        let track = parser
            .parse(&data)
            .expect("parsing SCC-style stream should succeed");
        assert!(
            !track.captions.is_empty(),
            "SCC-style stream must produce captions"
        );
        assert!(
            track.captions[0].text.contains("Test"),
            "caption text should contain 'Test', got '{}'",
            track.captions[0].text
        );
    }

    // ── Feature-not-enabled tests (no "cea" feature) ─────────────────────────

    #[cfg(not(feature = "cea"))]
    #[test]
    fn test_parser_without_feature_returns_error() {
        let parser = Cea608Parser;
        let result = parser.parse(b"\x94\x20");
        assert!(result.is_err());
        if let Err(CaptionError::FeatureNotEnabled(_)) = result {
            // expected
        } else {
            panic!("expected FeatureNotEnabled error");
        }
    }

    #[cfg(not(feature = "cea"))]
    #[test]
    fn test_writer_without_feature_returns_error() {
        let track = CaptionTrack::new(Language::english());
        let writer = Cea608Writer;
        let result = writer.write(&track);
        assert!(result.is_err());
        if let Err(CaptionError::FeatureNotEnabled(_)) = result {
            // expected
        } else {
            panic!("expected FeatureNotEnabled error");
        }
    }
}
