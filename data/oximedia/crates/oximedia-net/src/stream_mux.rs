#![allow(dead_code)]
//! Stream multiplexer/demultiplexer for combining multiple logical channels
//! over a single transport.
//!
//! Frames are prefixed with a 1-byte channel ID and a 4-byte big-endian
//! payload length, giving a 5-byte header.

/// Configuration for a single logical channel inside the mux.
#[derive(Debug, Clone)]
pub struct StreamChannel {
    /// Unique channel identifier `[0, 255]`.
    pub id: u8,
    /// Human-readable label for logging.
    pub label: String,
    /// Maximum frame payload size in bytes.
    pub max_frame_size: usize,
    /// Whether this channel is currently accepting data.
    active: bool,
}

impl StreamChannel {
    /// Creates a new channel in the active state.
    #[must_use]
    pub fn new(id: u8, label: impl Into<String>, max_frame_size: usize) -> Self {
        Self {
            id,
            label: label.into(),
            max_frame_size,
            active: true,
        }
    }

    /// Returns `true` when the channel is accepting frames.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Activates the channel.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivates the channel; frames for this channel will be dropped.
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Configuration for the stream multiplexer.
#[derive(Debug, Clone)]
pub struct MuxConfig {
    /// Maximum number of channels allowed.
    pub max_channels: usize,
    /// Default maximum frame payload size for new channels.
    pub default_max_frame: usize,
}

impl Default for MuxConfig {
    fn default() -> Self {
        Self {
            max_channels: 16,
            default_max_frame: 65_535,
        }
    }
}

impl MuxConfig {
    /// Creates a new config.
    #[must_use]
    pub fn new(max_channels: usize, default_max_frame: usize) -> Self {
        Self {
            max_channels,
            default_max_frame,
        }
    }

    /// Returns the configured channel count limit.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.max_channels
    }
}

/// Error type for mux operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxError {
    /// Channel ID not found.
    UnknownChannel(u8),
    /// Channel is inactive.
    InactiveChannel(u8),
    /// Payload exceeds the channel's `max_frame_size`.
    PayloadTooLarge,
    /// Too many channels registered.
    CapacityExceeded,
    /// The provided byte slice is too short to contain a valid frame.
    FrameTooShort,
    /// Frame payload length field is inconsistent with the slice length.
    FrameLengthMismatch,
}

/// The wire header size: 1 byte channel ID + 4 bytes payload length.
const HEADER_LEN: usize = 5;

/// Stream multiplexer that combines multiple logical channels.
#[derive(Debug)]
pub struct StreamMux {
    channels: Vec<StreamChannel>,
    config: MuxConfig,
    frames_muxed: u64,
    frames_demuxed: u64,
}

impl StreamMux {
    /// Creates a new mux with the given configuration.
    #[must_use]
    pub fn new(config: MuxConfig) -> Self {
        Self {
            channels: Vec::new(),
            config,
            frames_muxed: 0,
            frames_demuxed: 0,
        }
    }

    /// Returns the number of registered channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Returns total frames muxed.
    #[must_use]
    pub fn frames_muxed(&self) -> u64 {
        self.frames_muxed
    }

    /// Returns total frames demuxed.
    #[must_use]
    pub fn frames_demuxed(&self) -> u64 {
        self.frames_demuxed
    }

    /// Adds a logical channel to the mux.
    ///
    /// Returns an error if the capacity limit is reached or the channel ID
    /// is already registered.
    pub fn add_channel(&mut self, channel: StreamChannel) -> Result<(), MuxError> {
        if self.channels.len() >= self.config.max_channels {
            return Err(MuxError::CapacityExceeded);
        }
        if self.channels.iter().any(|c| c.id == channel.id) {
            return Err(MuxError::UnknownChannel(channel.id));
        }
        self.channels.push(channel);
        Ok(())
    }

    /// Removes a channel by ID.
    pub fn remove_channel(&mut self, id: u8) {
        self.channels.retain(|c| c.id != id);
    }

    /// Multiplexes `payload` over the channel identified by `channel_id`.
    ///
    /// Frame format: `[channel_id: u8][length: u32 BE][payload: ...]`
    ///
    /// Returns the framed bytes.
    pub fn mux_frame(&mut self, channel_id: u8, payload: &[u8]) -> Result<Vec<u8>, MuxError> {
        let ch = self
            .channels
            .iter()
            .find(|c| c.id == channel_id)
            .ok_or(MuxError::UnknownChannel(channel_id))?;

        if !ch.is_active() {
            return Err(MuxError::InactiveChannel(channel_id));
        }
        if payload.len() > ch.max_frame_size {
            return Err(MuxError::PayloadTooLarge);
        }

        let len = payload.len() as u32;
        let mut frame = Vec::with_capacity(HEADER_LEN + payload.len());
        frame.push(channel_id);
        frame.extend_from_slice(&len.to_be_bytes());
        frame.extend_from_slice(payload);
        self.frames_muxed += 1;
        Ok(frame)
    }

    /// Demultiplexes a frame produced by [`Self::mux_frame`].
    ///
    /// Returns `(channel_id, payload_bytes)`.
    pub fn demux_frame<'a>(&mut self, frame: &'a [u8]) -> Result<(u8, &'a [u8]), MuxError> {
        if frame.len() < HEADER_LEN {
            return Err(MuxError::FrameTooShort);
        }
        let channel_id = frame[0];
        let len = u32::from_be_bytes([frame[1], frame[2], frame[3], frame[4]]) as usize;
        if frame.len() != HEADER_LEN + len {
            return Err(MuxError::FrameLengthMismatch);
        }
        let payload = &frame[HEADER_LEN..];
        self.frames_demuxed += 1;
        Ok((channel_id, payload))
    }
}

impl Default for StreamMux {
    fn default() -> Self {
        Self::new(MuxConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_mux() -> StreamMux {
        let mut mux = StreamMux::default();
        mux.add_channel(StreamChannel::new(1, "video", 1_000_000))
            .expect("should succeed in test");
        mux.add_channel(StreamChannel::new(2, "audio", 65_535))
            .expect("should succeed in test");
        mux
    }

    #[test]
    fn test_channel_is_active_default() {
        let ch = StreamChannel::new(0, "test", 1024);
        assert!(ch.is_active());
    }

    #[test]
    fn test_channel_deactivate() {
        let mut ch = StreamChannel::new(0, "test", 1024);
        ch.deactivate();
        assert!(!ch.is_active());
    }

    #[test]
    fn test_channel_activate() {
        let mut ch = StreamChannel::new(0, "test", 1024);
        ch.deactivate();
        ch.activate();
        assert!(ch.is_active());
    }

    #[test]
    fn test_mux_config_channel_count() {
        let cfg = MuxConfig::new(8, 4096);
        assert_eq!(cfg.channel_count(), 8);
    }

    #[test]
    fn test_mux_add_channel() {
        let mux = basic_mux();
        assert_eq!(mux.channel_count(), 2);
    }

    #[test]
    fn test_mux_add_duplicate_id_error() {
        let mut mux = basic_mux();
        let result = mux.add_channel(StreamChannel::new(1, "dup", 512));
        assert!(result.is_err());
    }

    #[test]
    fn test_mux_frame_roundtrip() {
        let mut mux = basic_mux();
        let payload = b"hello world";
        let frame = mux.mux_frame(1, payload).expect("should succeed in test");
        let (ch_id, recovered) = mux.demux_frame(&frame).expect("should succeed in test");
        assert_eq!(ch_id, 1);
        assert_eq!(recovered, payload);
    }

    #[test]
    fn test_mux_unknown_channel() {
        let mut mux = basic_mux();
        assert_eq!(
            mux.mux_frame(99, b"data").unwrap_err(),
            MuxError::UnknownChannel(99)
        );
    }

    #[test]
    fn test_mux_inactive_channel() {
        let mut mux = basic_mux();
        mux.channels[0].deactivate();
        assert_eq!(
            mux.mux_frame(1, b"data").unwrap_err(),
            MuxError::InactiveChannel(1)
        );
    }

    #[test]
    fn test_mux_payload_too_large() {
        let mut mux = StreamMux::default();
        mux.add_channel(StreamChannel::new(5, "small", 4))
            .expect("should succeed in test");
        assert_eq!(
            mux.mux_frame(5, b"toolarge").unwrap_err(),
            MuxError::PayloadTooLarge
        );
    }

    #[test]
    fn test_demux_frame_too_short() {
        let mut mux = basic_mux();
        assert_eq!(
            mux.demux_frame(&[1u8; 3]).unwrap_err(),
            MuxError::FrameTooShort
        );
    }

    #[test]
    fn test_demux_frame_length_mismatch() {
        let mut mux = basic_mux();
        // Header says payload is 100 bytes but we only provide 2
        let mut bad = vec![1u8, 0, 0, 0, 100];
        bad.extend_from_slice(&[0u8; 2]);
        assert_eq!(
            mux.demux_frame(&bad).unwrap_err(),
            MuxError::FrameLengthMismatch
        );
    }

    #[test]
    fn test_mux_frame_counter() {
        let mut mux = basic_mux();
        mux.mux_frame(1, b"a").expect("should succeed in test");
        mux.mux_frame(2, b"b").expect("should succeed in test");
        assert_eq!(mux.frames_muxed(), 2);
    }

    #[test]
    fn test_demux_frame_counter() {
        let mut mux = basic_mux();
        let frame = mux.mux_frame(1, b"test").expect("should succeed in test");
        mux.demux_frame(&frame).expect("should succeed in test");
        assert_eq!(mux.frames_demuxed(), 1);
    }
}
