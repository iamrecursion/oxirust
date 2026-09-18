//! SMPTE ST 2110-30 PCM audio over RTP.
//!
//! This module implements SMPTE ST 2110-30 which defines the transport of
//! PCM audio over RTP for professional broadcast applications. It supports
//! linear PCM audio with multiple channels, various sample rates, and bit depths.

use crate::error::{NetError, NetResult};
use crate::smpte2110::rtp::{RtpHeader, RtpPacket, MAX_RTP_PAYLOAD};
use bytes::Bytes;
use std::collections::HashMap;

/// RTP payload type for ST 2110-30 audio (dynamic range).
pub const RTP_PAYLOAD_TYPE_AUDIO: u8 = 97;

/// Audio sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSampleRate {
    /// 48 kHz (standard broadcast).
    Rate48kHz = 48000,
    /// 96 kHz (high quality).
    Rate96kHz = 96000,
}

impl AudioSampleRate {
    /// Gets the sample rate as u32.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Gets the sample rate as f64.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self as u32)
    }

    /// Creates from u32 value.
    pub fn from_u32(value: u32) -> NetResult<Self> {
        match value {
            48000 => Ok(Self::Rate48kHz),
            96000 => Ok(Self::Rate96kHz),
            _ => Err(NetError::protocol(format!(
                "Unsupported sample rate: {value}"
            ))),
        }
    }
}

/// Audio format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// Linear PCM (L16, L20, L24).
    LinearPCM,
    /// AES3 audio format.
    AES3,
}

impl AudioFormat {
    /// Gets the format string for SDP.
    #[must_use]
    pub const fn sdp_format(&self, bit_depth: u8) -> &'static str {
        match (self, bit_depth) {
            (Self::LinearPCM, 16) => "L16",
            (Self::LinearPCM, 20) => "L20",
            (Self::LinearPCM, 24) => "L24",
            (Self::AES3, _) => "AM824",
            _ => "L24", // Default to L24
        }
    }
}

/// Audio configuration for ST 2110-30.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate.
    pub sample_rate: AudioSampleRate,
    /// Bit depth (16, 20, or 24).
    pub bit_depth: u8,
    /// Number of audio channels.
    pub channels: u16,
    /// Audio format.
    pub format: AudioFormat,
    /// Packet time in microseconds (125, 250, 333, 1000, etc.).
    pub packet_time_us: u32,
    /// Maximum payload size.
    pub max_payload_size: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: AudioSampleRate::Rate48kHz,
            bit_depth: 24,
            channels: 2,
            format: AudioFormat::LinearPCM,
            packet_time_us: 1000, // 1ms
            max_payload_size: MAX_RTP_PAYLOAD,
        }
    }
}

impl AudioConfig {
    /// Calculates the number of samples per packet.
    #[must_use]
    pub fn samples_per_packet(&self) -> usize {
        let sample_rate = self.sample_rate.as_u32() as usize;
        (sample_rate * self.packet_time_us as usize) / 1_000_000
    }

    /// Calculates the bytes per sample (all channels).
    #[must_use]
    pub const fn bytes_per_sample(&self) -> usize {
        let bytes_per_channel = match self.bit_depth {
            16 => 2,
            20 => 3,
            24 => 3,
            _ => 3,
        };
        bytes_per_channel * self.channels as usize
    }

    /// Calculates the packet payload size in bytes.
    #[must_use]
    pub fn packet_payload_size(&self) -> usize {
        self.samples_per_packet() * self.bytes_per_sample()
    }

    /// Calculates the bitrate in bits per second.
    #[must_use]
    pub fn bitrate(&self) -> u64 {
        let sample_rate = self.sample_rate.as_u32() as u64;
        let bits_per_sample = u64::from(self.bit_depth) * u64::from(self.channels);
        sample_rate * bits_per_sample
    }

    /// Validates the configuration.
    pub fn validate(&self) -> NetResult<()> {
        if self.channels == 0 || self.channels > 64 {
            return Err(NetError::protocol(format!(
                "Invalid channel count: {} (must be 1-64)",
                self.channels
            )));
        }

        if self.bit_depth != 16 && self.bit_depth != 20 && self.bit_depth != 24 {
            return Err(NetError::protocol(format!(
                "Invalid bit depth: {} (must be 16, 20, or 24)",
                self.bit_depth
            )));
        }

        let payload_size = self.packet_payload_size();
        if payload_size > self.max_payload_size {
            return Err(NetError::protocol(format!(
                "Packet payload {} exceeds max size {}",
                payload_size, self.max_payload_size
            )));
        }

        Ok(())
    }
}

/// Audio packet containing PCM samples.
#[derive(Debug, Clone)]
pub struct AudioPacket {
    /// RTP header.
    pub header: RtpHeader,
    /// Audio samples (interleaved).
    pub samples: Bytes,
    /// Number of samples in this packet.
    pub num_samples: usize,
}

impl AudioPacket {
    /// Creates a new audio packet.
    #[must_use]
    pub fn new(header: RtpHeader, samples: Bytes, num_samples: usize) -> Self {
        Self {
            header,
            samples,
            num_samples,
        }
    }

    /// Parses an audio packet from RTP packet.
    pub fn from_rtp(rtp_packet: &RtpPacket, config: &AudioConfig) -> NetResult<Self> {
        let num_samples = rtp_packet.payload.len() / config.bytes_per_sample();

        Ok(Self {
            header: rtp_packet.header.clone(),
            samples: rtp_packet.payload.clone(),
            num_samples,
        })
    }

    /// Converts to RTP packet.
    #[must_use]
    pub fn to_rtp(&self) -> RtpPacket {
        RtpPacket {
            header: self.header.clone(),
            payload: self.samples.clone(),
        }
    }
}

/// Audio encoder for ST 2110-30.
pub struct AudioEncoder {
    /// Configuration.
    config: AudioConfig,
    /// Current RTP timestamp.
    current_timestamp: u32,
    /// Current sequence number.
    sequence_number: u16,
    /// SSRC.
    ssrc: u32,
    /// Sample buffer for partial packets.
    sample_buffer: Vec<u8>,
}

impl AudioEncoder {
    /// Creates a new audio encoder.
    pub fn new(config: AudioConfig, ssrc: u32) -> NetResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            current_timestamp: rand::random(),
            sequence_number: rand::random(),
            ssrc,
            sample_buffer: Vec::new(),
        })
    }

    /// Encodes audio samples into RTP packets.
    ///
    /// The samples should be interleaved PCM data (L R L R for stereo).
    pub fn encode_samples(&mut self, samples: &[u8]) -> NetResult<Vec<AudioPacket>> {
        // Add to buffer
        self.sample_buffer.extend_from_slice(samples);

        let mut packets = Vec::new();
        let samples_per_packet = self.config.samples_per_packet();
        let bytes_per_sample = self.config.bytes_per_sample();
        let packet_size = samples_per_packet * bytes_per_sample;

        // Create packets from buffer
        while self.sample_buffer.len() >= packet_size {
            let packet_data = self.sample_buffer.drain(..packet_size).collect::<Vec<_>>();

            let header = RtpHeader {
                padding: false,
                extension: false,
                csrc_count: 0,
                marker: false,
                payload_type: RTP_PAYLOAD_TYPE_AUDIO,
                sequence_number: self.sequence_number,
                timestamp: self.current_timestamp,
                ssrc: self.ssrc,
                csrcs: Vec::new(),
                extension_data: None,
            };

            packets.push(AudioPacket::new(
                header,
                Bytes::from(packet_data),
                samples_per_packet,
            ));

            self.sequence_number = self.sequence_number.wrapping_add(1);
            self.current_timestamp = self
                .current_timestamp
                .wrapping_add(samples_per_packet as u32);
        }

        Ok(packets)
    }

    /// Flushes any remaining samples (creates a partial packet).
    pub fn flush(&mut self) -> NetResult<Option<AudioPacket>> {
        if self.sample_buffer.is_empty() {
            return Ok(None);
        }

        let bytes_per_sample = self.config.bytes_per_sample();
        let num_samples = self.sample_buffer.len() / bytes_per_sample;

        if num_samples == 0 {
            return Ok(None);
        }

        let packet_data = self.sample_buffer.drain(..).collect::<Vec<_>>();

        let header = RtpHeader {
            padding: false,
            extension: false,
            csrc_count: 0,
            marker: true, // Mark partial packet
            payload_type: RTP_PAYLOAD_TYPE_AUDIO,
            sequence_number: self.sequence_number,
            timestamp: self.current_timestamp,
            ssrc: self.ssrc,
            csrcs: Vec::new(),
            extension_data: None,
        };

        self.sequence_number = self.sequence_number.wrapping_add(1);
        self.current_timestamp = self.current_timestamp.wrapping_add(num_samples as u32);

        Ok(Some(AudioPacket::new(
            header,
            Bytes::from(packet_data),
            num_samples,
        )))
    }

    /// Gets the current RTP timestamp.
    #[must_use]
    pub const fn current_timestamp(&self) -> u32 {
        self.current_timestamp
    }

    /// Gets the configuration.
    #[must_use]
    pub const fn config(&self) -> &AudioConfig {
        &self.config
    }
}

/// Audio decoder for ST 2110-30.
#[derive(Debug)]
pub struct AudioDecoder {
    /// Configuration.
    config: AudioConfig,
    /// Packet buffer for jitter handling.
    packet_buffer: HashMap<u32, AudioPacket>,
    /// Next expected timestamp.
    next_timestamp: Option<u32>,
    /// Maximum packets to buffer.
    max_buffered_packets: usize,
}

impl AudioDecoder {
    /// Creates a new audio decoder.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            packet_buffer: HashMap::new(),
            next_timestamp: None,
            max_buffered_packets: 50,
        }
    }

    /// Processes an RTP packet.
    pub fn process_rtp_packet(&mut self, rtp_packet: &RtpPacket) -> NetResult<()> {
        let audio_packet = AudioPacket::from_rtp(rtp_packet, &self.config)?;
        let timestamp = audio_packet.header.timestamp;

        // Initialize next timestamp if needed
        if self.next_timestamp.is_none() {
            self.next_timestamp = Some(timestamp);
        }

        // Buffer the packet
        self.packet_buffer.insert(timestamp, audio_packet);

        // Limit buffer size
        if self.packet_buffer.len() > self.max_buffered_packets {
            // Remove oldest packet
            if let Some(oldest_ts) = self.packet_buffer.keys().min().copied() {
                self.packet_buffer.remove(&oldest_ts);
            }
        }

        Ok(())
    }

    /// Gets the next audio packet in sequence.
    pub fn get_next_packet(&mut self) -> Option<AudioPacket> {
        if let Some(ts) = self.next_timestamp {
            if let Some(packet) = self.packet_buffer.remove(&ts) {
                let samples_per_packet = self.config.samples_per_packet() as u32;
                self.next_timestamp = Some(ts.wrapping_add(samples_per_packet));
                return Some(packet);
            }
        }
        None
    }

    /// Gets all available sequential packets.
    pub fn get_available_packets(&mut self) -> Vec<AudioPacket> {
        let mut packets = Vec::new();

        while let Some(packet) = self.get_next_packet() {
            packets.push(packet);
        }

        packets
    }

    /// Decodes packets to raw PCM samples.
    pub fn decode_packets(&mut self) -> Vec<u8> {
        let packets = self.get_available_packets();
        let mut samples = Vec::new();

        for packet in packets {
            samples.extend_from_slice(&packet.samples);
        }

        samples
    }

    /// Gets the configuration.
    #[must_use]
    pub const fn config(&self) -> &AudioConfig {
        &self.config
    }
}

/// PCM sample packing/unpacking utilities.
pub mod packing {
    /// Packs 24-bit PCM sample to 3 bytes (big-endian).
    pub fn pack_24bit(sample: i32) -> [u8; 3] {
        [
            ((sample >> 16) & 0xFF) as u8,
            ((sample >> 8) & 0xFF) as u8,
            (sample & 0xFF) as u8,
        ]
    }

    /// Unpacks 24-bit PCM sample from 3 bytes (big-endian).
    #[must_use]
    pub fn unpack_24bit(bytes: &[u8; 3]) -> i32 {
        let value = (i32::from(bytes[0]) << 16) | (i32::from(bytes[1]) << 8) | i32::from(bytes[2]);

        // Sign extend from 24 to 32 bits
        if (value & 0x800000) != 0 {
            value | 0xFF000000u32 as i32
        } else {
            value
        }
    }

    /// Packs 20-bit PCM sample to 3 bytes (aligned to MSB).
    pub fn pack_20bit(sample: i32) -> [u8; 3] {
        let shifted = sample << 4; // Align to 24-bit MSB
        pack_24bit(shifted)
    }

    /// Unpacks 20-bit PCM sample from 3 bytes.
    #[must_use]
    pub fn unpack_20bit(bytes: &[u8; 3]) -> i32 {
        let value_24 = unpack_24bit(bytes);
        value_24 >> 4 // Shift back to 20-bit
    }

    /// Packs 16-bit PCM sample to 2 bytes (big-endian).
    pub fn pack_16bit(sample: i16) -> [u8; 2] {
        [((sample >> 8) & 0xFF) as u8, (sample & 0xFF) as u8]
    }

    /// Unpacks 16-bit PCM sample from 2 bytes (big-endian).
    #[must_use]
    pub fn unpack_16bit(bytes: &[u8; 2]) -> i16 {
        (i16::from(bytes[0]) << 8) | i16::from(bytes[1])
    }

    /// Interleaves mono samples into stereo.
    pub fn interleave_stereo(left: &[i32], right: &[i32]) -> Vec<i32> {
        let mut interleaved = Vec::with_capacity(left.len() + right.len());

        for (l, r) in left.iter().zip(right.iter()) {
            interleaved.push(*l);
            interleaved.push(*r);
        }

        interleaved
    }

    /// De-interleaves stereo samples into mono channels.
    pub fn deinterleave_stereo(interleaved: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut left = Vec::with_capacity(interleaved.len() / 2);
        let mut right = Vec::with_capacity(interleaved.len() / 2);

        for chunk in interleaved.chunks_exact(2) {
            left.push(chunk[0]);
            right.push(chunk[1]);
        }

        (left, right)
    }

    /// Packs multi-channel samples.
    pub fn pack_multichannel(samples: &[i32], _channels: usize, bit_depth: u8) -> Vec<u8> {
        let mut packed = Vec::new();

        for sample in samples {
            let bytes = match bit_depth {
                16 => {
                    let s16 = (*sample as i16).to_be_bytes();
                    vec![s16[0], s16[1]]
                }
                20 => pack_20bit(*sample).to_vec(),
                24 => pack_24bit(*sample).to_vec(),
                _ => pack_24bit(*sample).to_vec(),
            };

            packed.extend_from_slice(&bytes);
        }

        packed
    }

    /// Unpacks multi-channel samples.
    pub fn unpack_multichannel(data: &[u8], _channels: usize, bit_depth: u8) -> Vec<i32> {
        let bytes_per_sample = match bit_depth {
            16 => 2,
            20 | 24 => 3,
            _ => 3,
        };

        let mut samples = Vec::new();

        for chunk in data.chunks_exact(bytes_per_sample) {
            let sample = match bit_depth {
                16 => {
                    if chunk.len() >= 2 {
                        i32::from(unpack_16bit(&[chunk[0], chunk[1]]))
                    } else {
                        0
                    }
                }
                20 => {
                    if chunk.len() >= 3 {
                        unpack_20bit(&[chunk[0], chunk[1], chunk[2]])
                    } else {
                        0
                    }
                }
                24 => {
                    if chunk.len() >= 3 {
                        unpack_24bit(&[chunk[0], chunk[1], chunk[2]])
                    } else {
                        0
                    }
                }
                _ => 0,
            };

            samples.push(sample);
        }

        samples
    }
}

/// AES3 audio format support.
pub mod aes3 {
    /// AES3 frame structure (2 subframes per frame).
    #[derive(Debug, Clone)]
    pub struct AES3Frame {
        /// Subframe A (left channel).
        pub subframe_a: AES3Subframe,
        /// Subframe B (right channel).
        pub subframe_b: AES3Subframe,
    }

    /// AES3 subframe (32 bits).
    #[derive(Debug, Clone, Copy)]
    pub struct AES3Subframe {
        /// Preamble (4 bits).
        pub preamble: u8,
        /// Audio sample (20 or 24 bits).
        pub audio_sample: i32,
        /// Validity bit.
        pub validity: bool,
        /// User data bit.
        pub user_data: bool,
        /// Channel status bit.
        pub channel_status: bool,
        /// Parity bit.
        pub parity: bool,
    }

    impl AES3Subframe {
        /// Creates a new AES3 subframe.
        #[must_use]
        pub const fn new(audio_sample: i32) -> Self {
            Self {
                preamble: 0,
                audio_sample,
                validity: false,
                user_data: false,
                channel_status: false,
                parity: false,
            }
        }

        /// Packs to 32-bit word.
        #[must_use]
        pub fn pack(&self) -> u32 {
            let mut word = 0u32;

            // Preamble (bits 0-3)
            word |= u32::from(self.preamble & 0x0F);

            // Audio sample (bits 4-27, 24 bits)
            word |= ((self.audio_sample as u32) & 0xFFFFFF) << 4;

            // V bit (bit 28)
            if self.validity {
                word |= 1 << 28;
            }

            // U bit (bit 29)
            if self.user_data {
                word |= 1 << 29;
            }

            // C bit (bit 30)
            if self.channel_status {
                word |= 1 << 30;
            }

            // P bit (bit 31)
            if self.parity {
                word |= 1 << 31;
            }

            word
        }

        /// Unpacks from 32-bit word.
        #[must_use]
        pub fn unpack(word: u32) -> Self {
            let preamble = (word & 0x0F) as u8;
            let audio_sample = ((word >> 4) & 0xFFFFFF) as i32;
            let validity = (word & (1 << 28)) != 0;
            let user_data = (word & (1 << 29)) != 0;
            let channel_status = (word & (1 << 30)) != 0;
            let parity = (word & (1 << 31)) != 0;

            Self {
                preamble,
                audio_sample,
                validity,
                user_data,
                channel_status,
                parity,
            }
        }
    }

    impl AES3Frame {
        /// Creates a new AES3 frame from stereo samples.
        #[must_use]
        pub fn new(left: i32, right: i32) -> Self {
            Self {
                subframe_a: AES3Subframe::new(left),
                subframe_b: AES3Subframe::new(right),
            }
        }

        /// Packs to 8 bytes.
        pub fn pack(&self) -> [u8; 8] {
            let word_a = self.subframe_a.pack();
            let word_b = self.subframe_b.pack();

            let mut bytes = [0u8; 8];
            bytes[0..4].copy_from_slice(&word_a.to_be_bytes());
            bytes[4..8].copy_from_slice(&word_b.to_be_bytes());

            bytes
        }

        /// Unpacks from 8 bytes.
        #[must_use]
        pub fn unpack(bytes: &[u8; 8]) -> Self {
            let mut word_a_bytes = [0u8; 4];
            let mut word_b_bytes = [0u8; 4];
            word_a_bytes.copy_from_slice(&bytes[0..4]);
            word_b_bytes.copy_from_slice(&bytes[4..8]);

            let word_a = u32::from_be_bytes(word_a_bytes);
            let word_b = u32::from_be_bytes(word_b_bytes);

            Self {
                subframe_a: AES3Subframe::unpack(word_a),
                subframe_b: AES3Subframe::unpack(word_b),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_sample_rate() {
        assert_eq!(AudioSampleRate::Rate48kHz.as_u32(), 48000);
        assert_eq!(AudioSampleRate::Rate96kHz.as_u32(), 96000);

        let rate = AudioSampleRate::from_u32(48000).expect("should succeed in test");
        assert_eq!(rate, AudioSampleRate::Rate48kHz);
    }

    #[test]
    fn test_audio_config() {
        let config = AudioConfig {
            sample_rate: AudioSampleRate::Rate48kHz,
            bit_depth: 24,
            channels: 2,
            format: AudioFormat::LinearPCM,
            packet_time_us: 1000,
            max_payload_size: MAX_RTP_PAYLOAD,
        };

        assert_eq!(config.samples_per_packet(), 48);
        assert_eq!(config.bytes_per_sample(), 6); // 3 bytes * 2 channels
        assert!(config.bitrate() > 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_24bit_packing() {
        let sample = 0x123456;
        let packed = packing::pack_24bit(sample);
        let unpacked = packing::unpack_24bit(&packed);
        assert_eq!(sample, unpacked);
    }

    #[test]
    fn test_16bit_packing() {
        let sample = 0x1234i16;
        let packed = packing::pack_16bit(sample);
        let unpacked = packing::unpack_16bit(&packed);
        assert_eq!(sample, unpacked);
    }

    #[test]
    fn test_stereo_interleave() {
        let left = vec![100, 200, 300];
        let right = vec![400, 500, 600];

        let interleaved = packing::interleave_stereo(&left, &right);
        assert_eq!(interleaved, vec![100, 400, 200, 500, 300, 600]);

        let (left_out, right_out) = packing::deinterleave_stereo(&interleaved);
        assert_eq!(left, left_out);
        assert_eq!(right, right_out);
    }

    #[test]
    fn test_audio_encoder() {
        let config = AudioConfig {
            sample_rate: AudioSampleRate::Rate48kHz,
            bit_depth: 24,
            channels: 2,
            format: AudioFormat::LinearPCM,
            packet_time_us: 1000,
            max_payload_size: MAX_RTP_PAYLOAD,
        };

        let mut encoder = AudioEncoder::new(config.clone(), 12345).expect("should succeed in test");

        // Generate one packet worth of samples
        let num_samples = config.samples_per_packet();
        let bytes_per_sample = config.bytes_per_sample();
        let sample_data = vec![0u8; num_samples * bytes_per_sample];

        let packets = encoder
            .encode_samples(&sample_data)
            .expect("should succeed in test");
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].num_samples, num_samples);
    }

    #[test]
    fn test_audio_decoder() {
        let config = AudioConfig {
            sample_rate: AudioSampleRate::Rate48kHz,
            bit_depth: 24,
            channels: 2,
            format: AudioFormat::LinearPCM,
            packet_time_us: 1000,
            max_payload_size: MAX_RTP_PAYLOAD,
        };

        let mut decoder = AudioDecoder::new(config.clone());

        // Create test packet
        let num_samples = config.samples_per_packet();
        let bytes_per_sample = config.bytes_per_sample();
        let sample_data = vec![0u8; num_samples * bytes_per_sample];

        let header = RtpHeader {
            padding: false,
            extension: false,
            csrc_count: 0,
            marker: false,
            payload_type: RTP_PAYLOAD_TYPE_AUDIO,
            sequence_number: 1,
            timestamp: 0,
            ssrc: 12345,
            csrcs: Vec::new(),
            extension_data: None,
        };

        let rtp_packet = RtpPacket {
            header,
            payload: Bytes::from(sample_data),
        };

        decoder
            .process_rtp_packet(&rtp_packet)
            .expect("should succeed in test");

        let packets = decoder.get_available_packets();
        assert_eq!(packets.len(), 1);
    }

    #[test]
    fn test_aes3_subframe() {
        let sample = 0x123456;
        let subframe = aes3::AES3Subframe::new(sample);
        let word = subframe.pack();
        let unpacked = aes3::AES3Subframe::unpack(word);

        assert_eq!(unpacked.audio_sample, sample);
    }

    #[test]
    fn test_aes3_frame() {
        let left = 0x111111;
        let right = 0x222222;

        let frame = aes3::AES3Frame::new(left, right);
        let packed = frame.pack();
        let unpacked = aes3::AES3Frame::unpack(&packed);

        assert_eq!(unpacked.subframe_a.audio_sample, left);
        assert_eq!(unpacked.subframe_b.audio_sample, right);
    }
}
