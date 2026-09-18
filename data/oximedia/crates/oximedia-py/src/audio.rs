//! Audio codec bindings.

use crate::error::PyOxiResult;
use crate::types::AudioFrame;
use oximedia_audio::{AudioBuffer, ChannelLayout as AudioChannelLayout};
use oximedia_audio::{AudioDecoder, AudioDecoderConfig, AudioEncoder, AudioEncoderConfig};
use oximedia_codec::opus::OpusDecoder as RustOpusDecoder;
use oximedia_core::CodecId;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

// ─────────────────────────────── Helpers ─────────────────────────────────────

/// Convert an `oximedia_audio::AudioFrame` to the Python-facing `AudioFrame`.
///
/// The Python `AudioFrame` wraps `oximedia_codec::AudioFrame` which uses
/// `Vec<u8>` for samples. We extract the raw bytes from the audio frame and
/// build the simpler codec frame.
fn audio_frame_to_py(frame: oximedia_audio::AudioFrame) -> AudioFrame {
    // Extract raw bytes from the AudioBuffer.
    let raw_bytes: Vec<u8> = match frame.samples {
        AudioBuffer::Interleaved(data) => data.to_vec(),
        AudioBuffer::Planar(planes) => {
            // Interleave all planes.
            let ch = planes.len();
            if ch == 0 {
                return AudioFrame::from_rust(oximedia_codec::AudioFrame::new(
                    vec![],
                    0,
                    frame.sample_rate,
                    0,
                    oximedia_codec::SampleFormat::F32,
                ));
            }
            let bytes_per_sample = frame.format.bytes_per_sample();
            let spc = planes[0].len() / bytes_per_sample.max(1);
            let mut out = Vec::with_capacity(spc * ch * bytes_per_sample);
            for i in 0..spc {
                for plane in &planes {
                    let start = i * bytes_per_sample;
                    let end = start + bytes_per_sample;
                    if end <= plane.len() {
                        out.extend_from_slice(&plane[start..end]);
                    }
                }
            }
            out
        }
    };

    // Map oximedia_core::SampleFormat → oximedia_codec::SampleFormat.
    let codec_format = match frame.format {
        oximedia_core::SampleFormat::F32 => oximedia_codec::SampleFormat::F32,
        oximedia_core::SampleFormat::S16 => oximedia_codec::SampleFormat::I16,
        _ => oximedia_codec::SampleFormat::F32,
    };

    let channel_count = match &frame.channels {
        AudioChannelLayout::Mono => 1,
        AudioChannelLayout::Stereo => 2,
        other => other.count(),
    };

    let bytes_per_sample = codec_format.sample_size();
    let sample_count = if bytes_per_sample == 0 || channel_count == 0 {
        0
    } else {
        raw_bytes.len() / (bytes_per_sample * channel_count)
    };

    let mut inner = oximedia_codec::AudioFrame::new(
        raw_bytes,
        sample_count,
        frame.sample_rate,
        channel_count,
        codec_format,
    );
    inner.pts = Some(frame.timestamp.pts);

    AudioFrame::from_rust(inner)
}

// ─────────────────────────────── Opus Decoder ────────────────────────────────

/// Opus audio decoder.
///
/// Decodes Opus compressed audio packets to raw PCM samples.
///
/// # Example
///
/// ```python
/// decoder = OpusDecoder(sample_rate=48000, channels=2)
/// frame = decoder.decode_packet(packet_data)
/// if frame:
///     print(f"Decoded audio: {frame.sample_count} samples, {frame.channels} channels")
///     samples = frame.to_f32()  # Get as float32 array
/// ```
#[pyclass]
pub struct OpusDecoder {
    inner: RustOpusDecoder,
}

#[pymethods]
impl OpusDecoder {
    /// Create a new Opus decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (typically 48000)
    /// * `channels` - Number of channels (1 = mono, 2 = stereo)
    #[new]
    fn new(sample_rate: u32, channels: usize) -> PyOxiResult<Self> {
        let inner =
            RustOpusDecoder::new(sample_rate, channels).map_err(crate::error::from_codec_error)?;
        Ok(Self { inner })
    }

    /// Decode a compressed Opus packet.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed Opus packet data
    ///
    /// Returns an `AudioFrame` containing decoded PCM samples.
    fn decode_packet(&mut self, py: Python<'_>, data: &[u8]) -> PyOxiResult<AudioFrame> {
        let buf: Vec<u8> = data.to_vec();
        let inner = &mut self.inner;
        let frame = py
            .detach(move || inner.decode_packet(&buf))
            .map_err(crate::error::from_codec_error)?;
        Ok(AudioFrame::from_rust(frame))
    }

    /// Get decoder sample rate.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }

    /// Get number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.inner.config().channels
    }

    fn __str__(&self) -> String {
        format!(
            "OpusDecoder({}Hz, {} channels)",
            self.inner.config().sample_rate,
            self.inner.config().channels
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "OpusDecoder(sample_rate={}, channels={})",
            self.inner.config().sample_rate,
            self.inner.config().channels
        )
    }
}

// ─────────────────────────────── Vorbis Decoder ──────────────────────────────

/// Vorbis audio decoder.
///
/// Decodes Ogg Vorbis audio packets. Supports the send_packet / receive_frame
/// pull model.
///
/// # Example
///
/// ```python
/// decoder = VorbisDecoder(sample_rate=44100, channels=2)
/// decoder.send_packet(packet_data, pts=0)
/// frame = decoder.receive_frame()
/// if frame:
///     print(f"Decoded: {frame.sample_count} samples")
/// ```
#[pyclass]
pub struct VorbisDecoder {
    inner: oximedia_audio::vorbis::VorbisDecoder,
    stored_sample_rate: u32,
    stored_channels: u8,
}

#[pymethods]
impl VorbisDecoder {
    /// Create a new Vorbis decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (e.g. 44100)
    /// * `channels` - Number of channels (1 = mono, 2 = stereo)
    #[new]
    #[pyo3(signature = (sample_rate=44100, channels=2))]
    fn new(sample_rate: u32, channels: u8) -> PyOxiResult<Self> {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            sample_rate,
            channels,
            extradata: None,
        };
        let inner = oximedia_audio::vorbis::VorbisDecoder::new(&config)
            .map_err(crate::error::from_audio_error)?;
        Ok(Self {
            inner,
            stored_sample_rate: sample_rate,
            stored_channels: channels,
        })
    }

    /// Send a compressed Vorbis packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed Vorbis packet bytes
    /// * `pts` - Presentation timestamp
    fn send_packet(&mut self, py: Python<'_>, data: &[u8], pts: i64) -> PyOxiResult<()> {
        let buf: Vec<u8> = data.to_vec();
        let inner = &mut self.inner;
        py.detach(move || inner.send_packet(&buf, pts))
            .map_err(crate::error::from_audio_error)
    }

    /// Receive a decoded audio frame.
    ///
    /// Returns `None` if no frame is available yet.
    fn receive_frame(&mut self, py: Python<'_>) -> PyOxiResult<Option<AudioFrame>> {
        let inner = &mut self.inner;
        let opt = py
            .detach(move || inner.receive_frame())
            .map_err(crate::error::from_audio_error)?;
        Ok(opt.map(audio_frame_to_py))
    }

    /// Flush the decoder, signalling end of stream.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_audio_error)
    }

    /// Get the configured sample rate.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.stored_sample_rate
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        usize::from(self.stored_channels)
    }

    fn __str__(&self) -> String {
        format!(
            "VorbisDecoder({}Hz, {} channels)",
            self.stored_sample_rate, self.stored_channels
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "VorbisDecoder(sample_rate={}, channels={})",
            self.stored_sample_rate, self.stored_channels
        )
    }
}

// ─────────────────────────────── FLAC Decoder ────────────────────────────────

/// FLAC lossless audio decoder.
///
/// Decodes FLAC compressed audio packets using the send_packet / receive_frame
/// pull model.
///
/// # Example
///
/// ```python
/// decoder = FlacDecoder(sample_rate=44100, channels=2)
/// decoder.send_packet(packet_data, pts=0)
/// frame = decoder.receive_frame()
/// if frame:
///     print(f"Decoded: {frame.sample_count} samples at {frame.sample_rate}Hz")
/// ```
#[pyclass]
pub struct FlacDecoder {
    inner: oximedia_audio::flac::FlacDecoder,
    stored_sample_rate: u32,
    stored_channels: u8,
    stored_bits_per_sample: u8,
}

#[pymethods]
impl FlacDecoder {
    /// Create a new FLAC decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (e.g. 44100)
    /// * `channels` - Number of channels (1 = mono, 2 = stereo)
    /// * `bits_per_sample` - Bit depth (e.g. 16 or 24)
    #[new]
    #[pyo3(signature = (sample_rate=44100, channels=2, bits_per_sample=16))]
    fn new(sample_rate: u32, channels: u8, bits_per_sample: u8) -> PyOxiResult<Self> {
        let config = AudioDecoderConfig {
            codec: CodecId::Flac,
            sample_rate,
            channels,
            extradata: None,
        };
        let inner = oximedia_audio::flac::FlacDecoder::new(&config)
            .map_err(crate::error::from_audio_error)?;
        Ok(Self {
            inner,
            stored_sample_rate: sample_rate,
            stored_channels: channels,
            stored_bits_per_sample: bits_per_sample,
        })
    }

    /// Send a compressed FLAC packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed FLAC frame bytes
    /// * `pts` - Presentation timestamp
    fn send_packet(&mut self, py: Python<'_>, data: &[u8], pts: i64) -> PyOxiResult<()> {
        let buf: Vec<u8> = data.to_vec();
        let inner = &mut self.inner;
        py.detach(move || inner.send_packet(&buf, pts))
            .map_err(crate::error::from_audio_error)
    }

    /// Receive a decoded audio frame.
    ///
    /// Returns `None` if no frame is ready yet.
    fn receive_frame(&mut self, py: Python<'_>) -> PyOxiResult<Option<AudioFrame>> {
        let inner = &mut self.inner;
        let opt = py
            .detach(move || inner.receive_frame())
            .map_err(crate::error::from_audio_error)?;
        Ok(opt.map(audio_frame_to_py))
    }

    /// Flush the decoder, signalling end of stream.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_audio_error)
    }

    /// Get the configured sample rate.
    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        if self.stored_sample_rate == 0 {
            None
        } else {
            Some(self.stored_sample_rate)
        }
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> Option<u8> {
        if self.stored_channels == 0 {
            None
        } else {
            Some(self.stored_channels)
        }
    }

    /// Get bits per sample (bit depth).
    #[getter]
    fn bits_per_sample(&self) -> Option<u8> {
        if self.stored_bits_per_sample == 0 {
            None
        } else {
            Some(self.stored_bits_per_sample)
        }
    }

    fn __str__(&self) -> String {
        format!(
            "FlacDecoder({}Hz, {} channels, {}bps)",
            self.stored_sample_rate, self.stored_channels, self.stored_bits_per_sample
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "FlacDecoder(sample_rate={}, channels={}, bits_per_sample={})",
            self.stored_sample_rate, self.stored_channels, self.stored_bits_per_sample
        )
    }
}

// ─────────────────────────────── Opus Encoder ────────────────────────────────

/// Configuration for the Opus encoder.
///
/// # Example
///
/// ```python
/// config = OpusEncoderConfig(sample_rate=48000, channels=2, bitrate=128000)
/// ```
#[pyclass]
#[derive(Clone)]
pub struct OpusEncoderConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Target bitrate in bits/sec.
    pub bitrate: u32,
    /// Frame size in samples.
    pub frame_size: u32,
}

#[pymethods]
impl OpusEncoderConfig {
    /// Create a new Opus encoder configuration.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (default 48000)
    /// * `channels` - Number of channels 1 or 2 (default 2)
    /// * `bitrate` - Target bitrate in bits/sec (default 128000)
    /// * `frame_size` - Frame size in samples (default 960 = 20ms at 48kHz)
    #[new]
    #[pyo3(signature = (sample_rate=48000, channels=2, bitrate=128000, frame_size=960))]
    fn new(sample_rate: u32, channels: u8, bitrate: u32, frame_size: u32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate,
            frame_size,
        }
    }

    /// Get sample rate.
    #[getter]
    fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get channel count.
    #[getter]
    fn get_channels(&self) -> u8 {
        self.channels
    }

    /// Get bitrate.
    #[getter]
    fn get_bitrate(&self) -> u32 {
        self.bitrate
    }

    /// Get frame size.
    #[getter]
    fn get_frame_size(&self) -> u32 {
        self.frame_size
    }

    fn __str__(&self) -> String {
        format!(
            "OpusEncoderConfig({}Hz, {} ch, {}bps, {} samples/frame)",
            self.sample_rate, self.channels, self.bitrate, self.frame_size
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "OpusEncoderConfig(sample_rate={}, channels={}, bitrate={}, frame_size={})",
            self.sample_rate, self.channels, self.bitrate, self.frame_size
        )
    }
}

/// Opus audio encoder.
///
/// Encodes raw PCM audio frames to compressed Opus packets.
///
/// # Example
///
/// ```python
/// config = OpusEncoderConfig(sample_rate=48000, channels=2, bitrate=128000)
/// encoder = OpusEncoder(config)
/// encoder.send_frame(audio_frame)
/// packet = encoder.receive_packet()
/// if packet:
///     print(f"Encoded {len(packet['data'])} bytes, pts={packet['pts']}")
/// ```
#[pyclass]
pub struct OpusEncoder {
    inner: oximedia_audio::opus::OpusEncoder,
    stored_bitrate: u32,
}

#[pymethods]
impl OpusEncoder {
    /// Create a new Opus encoder.
    ///
    /// # Arguments
    ///
    /// * `config` - Encoder configuration
    #[new]
    fn new(config: &OpusEncoderConfig) -> PyOxiResult<Self> {
        let enc_config = AudioEncoderConfig {
            codec: CodecId::Opus,
            sample_rate: config.sample_rate,
            channels: config.channels,
            bitrate: config.bitrate,
            frame_size: config.frame_size,
        };
        let inner = oximedia_audio::opus::OpusEncoder::new(enc_config)
            .map_err(crate::error::from_audio_error)?;
        Ok(Self {
            inner,
            stored_bitrate: config.bitrate,
        })
    }

    /// Send an audio frame to the encoder.
    ///
    /// The frame's sample data must be in F32 interleaved or planar format.
    ///
    /// # Arguments
    ///
    /// * `frame` - Audio frame to encode
    fn send_frame(&mut self, py: Python<'_>, frame: &AudioFrame) -> PyOxiResult<()> {
        // Convert Python AudioFrame (oximedia-codec) to oximedia-audio AudioFrame.
        let audio_frame = codec_frame_to_audio(frame)?;
        let inner = &mut self.inner;
        py.detach(move || inner.send_frame(&audio_frame))
            .map_err(crate::error::from_audio_error)
    }

    /// Receive an encoded Opus packet.
    ///
    /// Returns a dict with keys `data` (bytes), `pts` (int), `duration` (int),
    /// or `None` if no packet is ready (need more frames).
    fn receive_packet<'py>(&mut self, py: Python<'py>) -> PyOxiResult<Option<Bound<'py, PyDict>>> {
        let inner = &mut self.inner;
        let opt = py
            .detach(move || inner.receive_packet())
            .map_err(crate::error::from_audio_error)?;
        match opt {
            None => Ok(None),
            Some(pkt) => {
                let dict = PyDict::new(py);
                dict.set_item("data", PyBytes::new(py, &pkt.data))?;
                dict.set_item("pts", pkt.pts)?;
                dict.set_item("duration", pkt.duration)?;
                Ok(Some(dict))
            }
        }
    }

    /// Flush the encoder, signalling end of stream.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_audio_error)
    }

    /// Get the configured bitrate.
    #[getter]
    fn bitrate(&self) -> u32 {
        self.stored_bitrate
    }

    fn __str__(&self) -> String {
        format!(
            "OpusEncoder({}bps, {} ch, {}Hz)",
            self.inner.config().bitrate,
            self.inner.config().channels,
            self.inner.config().sample_rate,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "OpusEncoder(bitrate={}, channels={}, sample_rate={})",
            self.inner.config().bitrate,
            self.inner.config().channels,
            self.inner.config().sample_rate,
        )
    }
}

/// Convert a Python-facing `AudioFrame` (wrapping `oximedia_codec::AudioFrame`)
/// to an `oximedia_audio::AudioFrame` suitable for encoding.
fn codec_frame_to_audio(frame: &AudioFrame) -> PyOxiResult<oximedia_audio::AudioFrame> {
    use bytes::Bytes;
    use oximedia_audio::{AudioBuffer, AudioFrame as AudioAudioFrame, ChannelLayout};
    use oximedia_core::{Rational, SampleFormat as CoreSampleFormat, Timestamp};

    let inner = frame.inner();
    let core_format = match inner.format {
        oximedia_codec::SampleFormat::F32 => CoreSampleFormat::F32,
        oximedia_codec::SampleFormat::I16 => CoreSampleFormat::S16,
        _ => CoreSampleFormat::F32,
    };
    let channel_layout = ChannelLayout::from_count(inner.channels);
    let pts = inner.pts.unwrap_or(0);
    let timebase = Rational::new(1, i64::from(inner.sample_rate).max(1));

    Ok(AudioAudioFrame {
        format: core_format,
        sample_rate: inner.sample_rate,
        channels: channel_layout,
        samples: AudioBuffer::Interleaved(Bytes::from(inner.samples.clone())),
        timestamp: Timestamp::new(pts, timebase),
    })
}
