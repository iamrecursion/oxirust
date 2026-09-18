//! MIDI Time Code (MTC) support.

use crate::error::{TimeSyncError, TimeSyncResult};
use oximedia_timecode::{FrameRate, Timecode};

/// MTC quarter-frame message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MtcQuarterFrame {
    /// Frame count LS nibble
    FrameLsb = 0,
    /// Frame count MS nibble
    FrameMsb = 1,
    /// Seconds count LS nibble
    SecondsLsb = 2,
    /// Seconds count MS nibble
    SecondsMsb = 3,
    /// Minutes count LS nibble
    MinutesLsb = 4,
    /// Minutes count MS nibble
    MinutesMsb = 5,
    /// Hours count LS nibble
    HoursLsb = 6,
    /// Hours count MS nibble + frame rate
    HoursMsbRate = 7,
}

impl MtcQuarterFrame {
    /// Convert from u8.
    #[must_use]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::FrameLsb),
            1 => Some(Self::FrameMsb),
            2 => Some(Self::SecondsLsb),
            3 => Some(Self::SecondsMsb),
            4 => Some(Self::MinutesLsb),
            5 => Some(Self::MinutesMsb),
            6 => Some(Self::HoursLsb),
            7 => Some(Self::HoursMsbRate),
            _ => None,
        }
    }
}

/// MTC frame rate encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MtcFrameRate {
    /// 24 fps
    Fps24 = 0,
    /// 25 fps
    Fps25 = 1,
    /// 29.97 fps (drop frame)
    Fps2997 = 2,
    /// 30 fps
    Fps30 = 3,
}

impl MtcFrameRate {
    /// Convert to standard `FrameRate`.
    #[must_use]
    pub fn to_frame_rate(self) -> FrameRate {
        match self {
            Self::Fps24 => FrameRate::Fps24,
            Self::Fps25 => FrameRate::Fps25,
            Self::Fps2997 => FrameRate::Fps2997DF,
            Self::Fps30 => FrameRate::Fps30,
        }
    }

    /// Convert from standard `FrameRate`.
    #[must_use]
    pub fn from_frame_rate(rate: FrameRate) -> Self {
        match rate {
            FrameRate::Fps24 => Self::Fps24,
            FrameRate::Fps25 => Self::Fps25,
            FrameRate::Fps2997DF | FrameRate::Fps2997NDF => Self::Fps2997,
            FrameRate::Fps30 => Self::Fps30,
            _ => Self::Fps25, // Default
        }
    }

    /// Convert from `FrameRateInfo`.
    #[must_use]
    pub fn from_frame_rate_info(fps: u8, drop_frame: bool) -> Self {
        match (fps, drop_frame) {
            (24, _) => Self::Fps24,
            (25, _) => Self::Fps25,
            (30, true) => Self::Fps2997,
            (30, false) => Self::Fps30,
            _ => Self::Fps25, // Default
        }
    }
}

/// MTC encoder.
pub struct MtcEncoder {
    /// Current quarter-frame position
    quarter_frame: u8,
}

impl MtcEncoder {
    /// Create a new MTC encoder.
    #[must_use]
    pub fn new() -> Self {
        Self { quarter_frame: 0 }
    }

    /// Encode next quarter-frame message.
    pub fn encode_quarter_frame(&mut self, timecode: &Timecode) -> TimeSyncResult<u8> {
        let qf_type = self.quarter_frame & 0x07;
        let mut data = (qf_type << 4) & 0x70;

        match MtcQuarterFrame::from_u8(qf_type) {
            Some(MtcQuarterFrame::FrameLsb) => {
                data |= timecode.frames & 0x0F;
            }
            Some(MtcQuarterFrame::FrameMsb) => {
                data |= (timecode.frames >> 4) & 0x0F;
            }
            Some(MtcQuarterFrame::SecondsLsb) => {
                data |= timecode.seconds & 0x0F;
            }
            Some(MtcQuarterFrame::SecondsMsb) => {
                data |= (timecode.seconds >> 4) & 0x0F;
            }
            Some(MtcQuarterFrame::MinutesLsb) => {
                data |= timecode.minutes & 0x0F;
            }
            Some(MtcQuarterFrame::MinutesMsb) => {
                data |= (timecode.minutes >> 4) & 0x0F;
            }
            Some(MtcQuarterFrame::HoursLsb) => {
                data |= timecode.hours & 0x0F;
            }
            Some(MtcQuarterFrame::HoursMsbRate) => {
                let rate = MtcFrameRate::from_frame_rate_info(
                    timecode.frame_rate.fps,
                    timecode.frame_rate.drop_frame,
                );
                data |= ((timecode.hours >> 4) & 0x01) | ((rate as u8) << 1);
            }
            None => {
                return Err(TimeSyncError::Timecode(
                    "Invalid quarter-frame type".to_string(),
                ));
            }
        }

        self.quarter_frame = (self.quarter_frame + 1) % 8;
        Ok(data)
    }

    /// Encode full-frame message (10 bytes).
    pub fn encode_full_frame(&self, timecode: &Timecode) -> TimeSyncResult<[u8; 10]> {
        let rate = MtcFrameRate::from_frame_rate_info(
            timecode.frame_rate.fps,
            timecode.frame_rate.drop_frame,
        );
        let hours_byte = timecode.hours | ((rate as u8) << 5);

        Ok([
            0xF0, // SysEx start
            0x7F, // Universal real-time
            0x7F, // Device ID (all)
            0x01, // MTC
            0x01, // Full message
            hours_byte,
            timecode.minutes,
            timecode.seconds,
            timecode.frames,
            0xF7, // SysEx end
        ])
    }
}

impl Default for MtcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// MTC decoder.
pub struct MtcDecoder {
    /// Partial timecode being assembled
    partial: [u8; 8],
    /// Quarter-frame counter
    quarter_count: u8,
}

impl MtcDecoder {
    /// Create a new MTC decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            partial: [0; 8],
            quarter_count: 0,
        }
    }

    /// Decode quarter-frame message.
    pub fn decode_quarter_frame(&mut self, data: u8) -> TimeSyncResult<Option<Timecode>> {
        let qf_type = (data >> 4) & 0x07;
        let value = data & 0x0F;

        if qf_type as usize >= 8 {
            return Err(TimeSyncError::Timecode(
                "Invalid quarter-frame type".to_string(),
            ));
        }

        self.partial[qf_type as usize] = value;
        self.quarter_count += 1;

        // Full timecode available after 8 quarter-frames
        if self.quarter_count >= 8 {
            self.quarter_count = 0;
            let timecode = self.assemble_timecode()?;
            return Ok(Some(timecode));
        }

        Ok(None)
    }

    /// Assemble timecode from partial data.
    fn assemble_timecode(&self) -> TimeSyncResult<Timecode> {
        let frames = self.partial[0] | (self.partial[1] << 4);
        let seconds = self.partial[2] | (self.partial[3] << 4);
        let minutes = self.partial[4] | (self.partial[5] << 4);
        let hours = self.partial[6] | ((self.partial[7] & 0x01) << 4);
        let rate_code = (self.partial[7] >> 1) & 0x03;

        let frame_rate = match rate_code {
            0 => FrameRate::Fps24,
            1 => FrameRate::Fps25,
            2 => FrameRate::Fps2997DF,
            3 => FrameRate::Fps30,
            _ => FrameRate::Fps25,
        };

        Timecode::new(hours, minutes, seconds, frames, frame_rate)
            .map_err(|e| TimeSyncError::Timecode(e.to_string()))
    }
}

impl Default for MtcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtc_encoder() {
        let mut encoder = MtcEncoder::new();
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("should succeed in test");

        // Encode 8 quarter-frames for complete timecode
        for _ in 0..8 {
            let _qf = encoder
                .encode_quarter_frame(&tc)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_mtc_full_frame() {
        let encoder = MtcEncoder::new();
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("should succeed in test");

        let full_frame = encoder
            .encode_full_frame(&tc)
            .expect("should succeed in test");
        assert_eq!(full_frame[0], 0xF0);
        assert_eq!(full_frame[9], 0xF7);
    }

    #[test]
    fn test_mtc_decoder() {
        let mut decoder = MtcDecoder::new();

        // Simulate receiving 8 quarter-frames
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("should succeed in test");
        let mut encoder = MtcEncoder::new();

        for i in 0..7 {
            let qf = encoder
                .encode_quarter_frame(&tc)
                .expect("should succeed in test");
            let result = decoder
                .decode_quarter_frame(qf)
                .expect("should succeed in test");
            assert!(
                result.is_none(),
                "Should not complete until 8th frame (at {})",
                i
            );
        }

        // 8th quarter-frame should complete timecode
        let qf = encoder
            .encode_quarter_frame(&tc)
            .expect("should succeed in test");
        let result = decoder
            .decode_quarter_frame(qf)
            .expect("should succeed in test");
        assert!(result.is_some());
    }
}
