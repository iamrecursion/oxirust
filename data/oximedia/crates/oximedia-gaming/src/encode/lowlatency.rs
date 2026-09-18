//! Low-latency encoding pipeline.
//!
//! Implements ultra-low latency encoding for game streaming.

use crate::{GamingError, GamingResult};
use std::time::{Duration, Instant};

/// Low-latency encoder.
pub struct LowLatencyEncoder {
    /// Encoder configuration.
    pub config: EncoderConfig,
    frames_encoded: u64,
    /// Total bytes output so far (for bitrate tracking).
    total_bytes_output: u64,
    /// Accumulated encoding time for average calculation.
    total_encoding_time: Duration,
    /// Pending B-frames that are buffered for reordering.
    pending_frames: Vec<EncodedFrame>,
}

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Input resolution
    pub resolution: (u32, u32),
    /// Target framerate
    pub framerate: u32,
    /// Bitrate in kbps
    pub bitrate: u32,
    /// Latency mode
    pub latency_mode: LatencyMode,
    /// Keyframe interval in frames
    pub keyframe_interval: u32,
    /// Use B-frames
    pub use_b_frames: bool,
    /// Rate control mode
    pub rate_control: RateControlMode,
}

/// Latency mode for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyMode {
    /// Ultra-low latency (<50ms) - no B-frames, minimal buffering
    UltraLow,
    /// Low latency (<100ms) - minimal B-frames
    Low,
    /// Normal latency - standard encoding
    Normal,
}

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant bitrate
    Cbr,
    /// Variable bitrate
    Vbr,
    /// Constant quality
    Cq,
}

/// Encoded frame data.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Encoded data
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: Duration,
    /// Decode timestamp
    pub dts: Duration,
    /// Is keyframe
    pub is_keyframe: bool,
}

impl LowLatencyEncoder {
    /// Create a new low-latency encoder.
    ///
    /// # Errors
    ///
    /// Returns error if encoder initialization fails.
    pub fn new(config: EncoderConfig) -> GamingResult<Self> {
        if config.resolution.0 == 0 || config.resolution.1 == 0 {
            return Err(GamingError::InvalidConfig(
                "Resolution must be non-zero".to_string(),
            ));
        }

        if config.framerate == 0 {
            return Err(GamingError::InvalidConfig(
                "Framerate must be non-zero".to_string(),
            ));
        }

        if config.bitrate < 500 {
            return Err(GamingError::InvalidConfig(
                "Bitrate must be at least 500 kbps".to_string(),
            ));
        }

        Ok(Self {
            config,
            frames_encoded: 0,
            total_bytes_output: 0,
            total_encoding_time: Duration::ZERO,
            pending_frames: Vec::new(),
        })
    }

    /// Encode a raw RGBA frame into a simulated encoded bitstream.
    ///
    /// The encoder produces a realistic encoded output by:
    /// 1. Computing a lightweight hash/summary of the input frame data.
    /// 2. Building a header with frame metadata (resolution, pts, keyframe flag).
    /// 3. Applying simulated "compression": the output size is derived from the
    ///    configured bitrate and framerate rather than being empty or a 1:1 copy.
    /// 4. Respecting latency mode settings (ultra-low disables B-frames entirely).
    ///
    /// # Errors
    ///
    /// Returns error if the input frame data size does not match the expected
    /// resolution.
    pub fn encode_frame(&mut self, frame_data: &[u8]) -> GamingResult<EncodedFrame> {
        let encode_start = Instant::now();

        let expected_size =
            (self.config.resolution.0 as usize) * (self.config.resolution.1 as usize) * 4;
        if frame_data.len() != expected_size {
            return Err(GamingError::EncodingError(format!(
                "Frame data size mismatch: expected {} bytes ({}x{}x4), got {}",
                expected_size,
                self.config.resolution.0,
                self.config.resolution.1,
                frame_data.len()
            )));
        }

        self.frames_encoded += 1;
        let seq = self.frames_encoded;

        let is_keyframe = seq % u64::from(self.config.keyframe_interval) == 1 || seq == 1;

        // Frame interval in milliseconds
        let frame_interval_ms = if self.config.framerate > 0 {
            1000 / u64::from(self.config.framerate)
        } else {
            16 // fallback ~60fps
        };

        let pts = Duration::from_millis(seq.saturating_sub(1) * frame_interval_ms);

        // DTS may differ from PTS when B-frames are used; for I/P frames they are equal
        let use_b_frames = self.config.use_b_frames
            && self.config.latency_mode == LatencyMode::Normal
            && !is_keyframe;

        let dts = if use_b_frames {
            // B-frames have DTS one interval before PTS
            Duration::from_millis(seq.saturating_sub(2) * frame_interval_ms)
        } else {
            pts
        };

        // Compute a lightweight hash of the frame for reproducible "encoded" output.
        // We sample bytes at regular intervals to keep this fast.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        let step = (frame_data.len() / 256).max(1);
        for i in (0..frame_data.len()).step_by(step) {
            hash ^= frame_data[i] as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
        }

        // Compute target encoded size from bitrate and framerate.
        // bitrate is in kbps, so bytes_per_frame = (bitrate * 1000) / (8 * fps)
        let bytes_per_frame = if self.config.framerate > 0 {
            ((u64::from(self.config.bitrate) * 1000) / (8 * u64::from(self.config.framerate)))
                .max(64) as usize
        } else {
            1024
        };

        // Keyframes are typically larger
        let target_size = if is_keyframe {
            bytes_per_frame * 3
        } else {
            bytes_per_frame
        };

        // Build the encoded bitstream:
        // [4 bytes: magic "OxiE"] [4 bytes: frame size BE] [8 bytes: pts_ms BE]
        // [1 byte: flags (keyframe, b-frame)] [remaining: pseudo-compressed data]
        let header_size = 17;
        let data_size = target_size.max(header_size);
        let mut data = Vec::with_capacity(data_size);

        // Magic
        data.extend_from_slice(b"OxiE");
        // Frame payload size (excluding header)
        let payload_size = (data_size - header_size) as u32;
        data.extend_from_slice(&payload_size.to_be_bytes());
        // PTS in milliseconds
        data.extend_from_slice(&pts.as_millis().to_be_bytes().as_slice()[8..16]);
        // Flags: bit 0 = keyframe, bit 1 = b-frame
        let flags: u8 =
            if is_keyframe { 0x01 } else { 0x00 } | if use_b_frames { 0x02 } else { 0x00 };
        data.push(flags);

        // Fill remaining bytes with deterministic pseudo-random data seeded by hash
        let mut rng_state = hash;
        while data.len() < data_size {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            data.push(rng_state as u8);
        }

        let encode_elapsed = encode_start.elapsed();
        self.total_encoding_time += encode_elapsed;
        self.total_bytes_output += data.len() as u64;

        let frame = EncodedFrame {
            data,
            pts,
            dts,
            is_keyframe,
        };

        Ok(frame)
    }

    /// Get encoder statistics.
    #[must_use]
    pub fn get_stats(&self) -> EncoderStats {
        let average_encoding_time = if self.frames_encoded > 0 {
            self.total_encoding_time / self.frames_encoded as u32
        } else {
            Duration::ZERO
        };

        // Current effective bitrate in kbps
        let current_bitrate = if self.frames_encoded > 0 {
            let duration_secs =
                (self.frames_encoded as f64) / (self.config.framerate as f64).max(1.0);
            if duration_secs > 0.0 {
                ((self.total_bytes_output as f64 * 8.0) / (duration_secs * 1000.0)) as u32
            } else {
                self.config.bitrate
            }
        } else {
            0
        };

        EncoderStats {
            frames_encoded: self.frames_encoded,
            average_encoding_time,
            current_bitrate,
        }
    }

    /// Flush encoder and get remaining buffered frames.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    pub fn flush(&mut self) -> GamingResult<Vec<EncodedFrame>> {
        let pending = std::mem::take(&mut self.pending_frames);
        Ok(pending)
    }

    /// Reset encoder state for a new encoding session.
    pub fn reset(&mut self) {
        self.frames_encoded = 0;
        self.total_bytes_output = 0;
        self.total_encoding_time = Duration::ZERO;
        self.pending_frames.clear();
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            resolution: (1920, 1080),
            framerate: 60,
            bitrate: 6000,
            latency_mode: LatencyMode::Low,
            keyframe_interval: 120,
            use_b_frames: false,
            rate_control: RateControlMode::Cbr,
        }
    }
}

/// Encoder statistics.
#[derive(Debug, Clone)]
pub struct EncoderStats {
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Average encoding time per frame
    pub average_encoding_time: Duration,
    /// Current bitrate in kbps
    pub current_bitrate: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = EncoderConfig::default();
        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.frames_encoded, 0);
    }

    #[test]
    fn test_invalid_resolution() {
        let mut config = EncoderConfig::default();
        config.resolution = (0, 0);
        let result = LowLatencyEncoder::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_bitrate() {
        let mut config = EncoderConfig::default();
        config.bitrate = 100;
        let result = LowLatencyEncoder::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_frame() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

        let frame_data = vec![0u8; 1920 * 1080 * 4];
        let encoded = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");

        assert!(encoded.is_keyframe); // first frame is always keyframe
        assert!(!encoded.data.is_empty());
        // Check magic header
        assert_eq!(&encoded.data[..4], b"OxiE");
    }

    #[test]
    fn test_encode_frame_wrong_size_rejected() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

        let frame_data = vec![0u8; 100]; // wrong size
        let result = encoder.encode_frame(&frame_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_keyframe_interval() {
        let mut config = EncoderConfig::default();
        config.keyframe_interval = 30;

        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        let frame_data = vec![0u8; 1920 * 1080 * 4];

        // First frame should be keyframe
        let frame1 = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");
        assert!(frame1.is_keyframe);

        // Next 29 frames should not be keyframes
        for _ in 0..29 {
            let frame = encoder
                .encode_frame(&frame_data)
                .expect("encode should succeed");
            assert!(!frame.is_keyframe);
        }

        // 31st frame should be keyframe
        let frame31 = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");
        assert!(frame31.is_keyframe);
    }

    #[test]
    fn test_ultra_low_latency_mode() {
        let mut config = EncoderConfig::default();
        config.latency_mode = LatencyMode::UltraLow;
        config.use_b_frames = false;

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.latency_mode, LatencyMode::UltraLow);
        assert!(!encoder.config.use_b_frames);
    }

    #[test]
    fn test_encoder_stats() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

        let frame_data = vec![0u8; 1920 * 1080 * 4];
        encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");

        let stats = encoder.get_stats();
        assert_eq!(stats.frames_encoded, 1);
        assert!(stats.current_bitrate > 0);
    }

    #[test]
    fn test_encoded_data_differs_for_different_input() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

        let frame_a = vec![0u8; 1920 * 1080 * 4];
        let mut frame_b = vec![128u8; 1920 * 1080 * 4];
        // Make it valid RGBA
        for i in (3..frame_b.len()).step_by(4) {
            frame_b[i] = 255;
        }

        let enc_a = encoder.encode_frame(&frame_a).expect("encode a");
        encoder.reset();
        let enc_b = encoder.encode_frame(&frame_b).expect("encode b");

        // Encoded data should differ because input differs
        assert_ne!(enc_a.data, enc_b.data);
    }

    #[test]
    fn test_keyframe_larger_than_p_frame() {
        let config = EncoderConfig {
            keyframe_interval: 10,
            ..EncoderConfig::default()
        };
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        let frame_data = vec![0u8; 1920 * 1080 * 4];

        let keyframe = encoder.encode_frame(&frame_data).expect("keyframe");
        let p_frame = encoder.encode_frame(&frame_data).expect("p-frame");

        assert!(keyframe.data.len() > p_frame.data.len());
    }

    #[test]
    fn test_encoder_reset() {
        let config = EncoderConfig::default();
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        let frame_data = vec![0u8; 1920 * 1080 * 4];

        encoder.encode_frame(&frame_data).expect("encode");
        assert_eq!(encoder.get_stats().frames_encoded, 1);

        encoder.reset();
        assert_eq!(encoder.get_stats().frames_encoded, 0);
    }

    #[test]
    fn test_pts_increases_per_frame() {
        let config = EncoderConfig {
            framerate: 30,
            ..EncoderConfig::default()
        };
        let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        let frame_data = vec![0u8; 1920 * 1080 * 4];

        let f1 = encoder.encode_frame(&frame_data).expect("f1");
        let f2 = encoder.encode_frame(&frame_data).expect("f2");
        let f3 = encoder.encode_frame(&frame_data).expect("f3");

        assert!(f2.pts > f1.pts);
        assert!(f3.pts > f2.pts);
    }
}
