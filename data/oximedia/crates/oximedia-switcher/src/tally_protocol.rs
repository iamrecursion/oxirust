//! Tally protocol encoding and decoding for broadcast switchers.
//!
//! Implements serialization and parsing of tally status messages used to drive
//! tally lights on cameras and other production equipment.

#![allow(dead_code)]

/// Color state of a tally light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallyColor {
    /// Camera is on program (on-air).
    Red,
    /// Camera is on preview.
    Green,
    /// Camera is queued or standby.
    Amber,
    /// No tally active.
    Off,
}

impl TallyColor {
    /// Returns `true` if this color indicates the source is on-air.
    pub fn is_on_air(&self) -> bool {
        matches!(self, TallyColor::Red)
    }

    /// Returns `true` if this color indicates the source is on preview.
    pub fn is_preview(&self) -> bool {
        matches!(self, TallyColor::Green)
    }

    /// Returns `true` if the tally is active (not Off).
    pub fn is_active(&self) -> bool {
        !matches!(self, TallyColor::Off)
    }

    /// Convert to a byte suitable for wire transmission.
    pub fn to_byte(&self) -> u8 {
        match self {
            TallyColor::Red => 0x01,
            TallyColor::Green => 0x02,
            TallyColor::Amber => 0x03,
            TallyColor::Off => 0x00,
        }
    }

    /// Parse a byte from the wire.
    pub fn from_byte(b: u8) -> Self {
        match b {
            0x01 => TallyColor::Red,
            0x02 => TallyColor::Green,
            0x03 => TallyColor::Amber,
            _ => TallyColor::Off,
        }
    }
}

/// A tally message carrying per-camera color state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyMessage {
    /// Version byte for the protocol.
    pub version: u8,
    /// Per-camera tally colors indexed from 1.
    pub cameras: Vec<TallyColor>,
}

impl TallyMessage {
    /// Create a new tally message with the given camera states.
    pub fn new(cameras: Vec<TallyColor>) -> Self {
        Self {
            version: 1,
            cameras,
        }
    }

    /// Return the number of cameras in this message.
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Return the color for a specific camera (1-based index).
    /// Returns `TallyColor::Off` if the camera index is out of range.
    pub fn color_for(&self, camera: usize) -> TallyColor {
        if camera == 0 || camera > self.cameras.len() {
            TallyColor::Off
        } else {
            self.cameras[camera - 1]
        }
    }

    /// Return the number of on-air cameras.
    pub fn on_air_count(&self) -> usize {
        self.cameras.iter().filter(|c| c.is_on_air()).count()
    }
}

/// Protocol encoder/decoder for tally messages.
///
/// Wire format (v1):
/// ```text
/// [0x54][0x41][version:u8][count:u16 BE][color:u8 * count][0x0D][0x0A]
/// ```
#[derive(Debug, Default)]
pub struct TallyProtocol;

impl TallyProtocol {
    /// Create a new protocol instance.
    pub fn new() -> Self {
        Self
    }

    /// Encode a `TallyMessage` into a byte vector.
    pub fn encode_message(&self, msg: &TallyMessage) -> Vec<u8> {
        let count = msg.cameras.len() as u16;
        let mut buf = Vec::with_capacity(7 + msg.cameras.len());
        // Magic header
        buf.push(0x54); // 'T'
        buf.push(0x41); // 'A'
        buf.push(msg.version);
        buf.push((count >> 8) as u8);
        buf.push((count & 0xFF) as u8);
        for color in &msg.cameras {
            buf.push(color.to_byte());
        }
        // Terminator
        buf.push(0x0D);
        buf.push(0x0A);
        buf
    }

    /// Parse a response byte slice into a `TallyMessage`.
    ///
    /// Returns `None` if the bytes are malformed.
    pub fn parse_response(&self, data: &[u8]) -> Option<TallyMessage> {
        if data.len() < 7 {
            return None;
        }
        if data[0] != 0x54 || data[1] != 0x41 {
            return None;
        }
        let version = data[2];
        let count = (u16::from(data[3]) << 8 | u16::from(data[4])) as usize;
        if data.len() < 5 + count + 2 {
            return None;
        }
        let cameras: Vec<TallyColor> = data[5..5 + count]
            .iter()
            .map(|&b| TallyColor::from_byte(b))
            .collect();
        Some(TallyMessage { version, cameras })
    }

    /// Validate that encoded bytes round-trip correctly.
    pub fn validate_roundtrip(&self, msg: &TallyMessage) -> bool {
        let encoded = self.encode_message(msg);
        match self.parse_response(&encoded) {
            Some(decoded) => decoded == *msg,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_color_is_on_air() {
        assert!(TallyColor::Red.is_on_air());
        assert!(!TallyColor::Green.is_on_air());
        assert!(!TallyColor::Amber.is_on_air());
        assert!(!TallyColor::Off.is_on_air());
    }

    #[test]
    fn test_tally_color_is_preview() {
        assert!(TallyColor::Green.is_preview());
        assert!(!TallyColor::Red.is_preview());
    }

    #[test]
    fn test_tally_color_is_active() {
        assert!(TallyColor::Red.is_active());
        assert!(TallyColor::Green.is_active());
        assert!(TallyColor::Amber.is_active());
        assert!(!TallyColor::Off.is_active());
    }

    #[test]
    fn test_tally_color_to_byte() {
        assert_eq!(TallyColor::Red.to_byte(), 0x01);
        assert_eq!(TallyColor::Green.to_byte(), 0x02);
        assert_eq!(TallyColor::Amber.to_byte(), 0x03);
        assert_eq!(TallyColor::Off.to_byte(), 0x00);
    }

    #[test]
    fn test_tally_color_from_byte_roundtrip() {
        for b in [0x00u8, 0x01, 0x02, 0x03, 0xFF] {
            let color = TallyColor::from_byte(b);
            if b <= 0x03 {
                assert_eq!(color.to_byte(), b);
            } else {
                assert_eq!(color, TallyColor::Off);
            }
        }
    }

    #[test]
    fn test_tally_message_camera_count() {
        let msg = TallyMessage::new(vec![TallyColor::Red, TallyColor::Green, TallyColor::Off]);
        assert_eq!(msg.camera_count(), 3);
    }

    #[test]
    fn test_tally_message_color_for_valid() {
        let msg = TallyMessage::new(vec![TallyColor::Red, TallyColor::Green]);
        assert_eq!(msg.color_for(1), TallyColor::Red);
        assert_eq!(msg.color_for(2), TallyColor::Green);
    }

    #[test]
    fn test_tally_message_color_for_out_of_range() {
        let msg = TallyMessage::new(vec![TallyColor::Red]);
        assert_eq!(msg.color_for(0), TallyColor::Off);
        assert_eq!(msg.color_for(99), TallyColor::Off);
    }

    #[test]
    fn test_tally_message_on_air_count() {
        let msg = TallyMessage::new(vec![
            TallyColor::Red,
            TallyColor::Red,
            TallyColor::Green,
            TallyColor::Off,
        ]);
        assert_eq!(msg.on_air_count(), 2);
    }

    #[test]
    fn test_encode_message_header() {
        let proto = TallyProtocol::new();
        let msg = TallyMessage::new(vec![TallyColor::Red]);
        let encoded = proto.encode_message(&msg);
        assert_eq!(encoded[0], 0x54);
        assert_eq!(encoded[1], 0x41);
        assert_eq!(encoded[2], 1); // version
    }

    #[test]
    fn test_encode_message_count_field() {
        let proto = TallyProtocol::new();
        let msg = TallyMessage::new(vec![TallyColor::Red, TallyColor::Green, TallyColor::Off]);
        let encoded = proto.encode_message(&msg);
        let count = (u16::from(encoded[3]) << 8) | u16::from(encoded[4]);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_parse_response_roundtrip() {
        let proto = TallyProtocol::new();
        let msg = TallyMessage::new(vec![
            TallyColor::Red,
            TallyColor::Green,
            TallyColor::Amber,
            TallyColor::Off,
        ]);
        assert!(proto.validate_roundtrip(&msg));
    }

    #[test]
    fn test_parse_response_too_short() {
        let proto = TallyProtocol::new();
        assert!(proto.parse_response(&[0x54, 0x41]).is_none());
    }

    #[test]
    fn test_parse_response_bad_magic() {
        let proto = TallyProtocol::new();
        let data = vec![0xFF, 0xFF, 1, 0, 1, 0x01, 0x0D, 0x0A];
        assert!(proto.parse_response(&data).is_none());
    }

    #[test]
    fn test_validate_roundtrip_empty() {
        let proto = TallyProtocol::new();
        let msg = TallyMessage::new(vec![]);
        assert!(proto.validate_roundtrip(&msg));
    }
}
