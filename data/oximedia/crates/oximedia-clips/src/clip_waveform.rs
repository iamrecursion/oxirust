//! Audio waveform generation for timeline thumbnails.
//!
//! `ClipWaveformGenerator` consumes raw PCM audio samples and produces a
//! compact `WaveformData` structure that drives waveform thumbnail rendering
//! in a timeline view.  Each pixel bucket stores the minimum and maximum
//! sample value (normalised to `[-1.0, 1.0]`) observed in the corresponding
//! audio segment.
//!
//! # File-based waveform generation
//!
//! On non-WASM targets, `ClipWaveformGenerator::generate` decodes an audio
//! file from disk and computes the waveform directly from the decoded PCM
//! samples.  Supported formats:
//!
//! - **WAV** (RIFF/WAVE): 8/16/24/32-bit integer PCM and 32/64-bit IEEE float,
//!   mono and multi-channel.
//! - **MP3**: MPEG-1/2 Layer I/II/III (patents expired 2017).  Requires the
//!   `audio-decode` feature (default on).
//! - **Ogg/Opus** and **Ogg/Vorbis**: Decoded via `OggDemuxer` + the matching
//!   `OpusDecoder`/`VorbisDecoder`.  Requires the `audio-decode` feature.
//! - **FLAC**: Decoded via `FlacDemuxer` + `FlacDecoder`.  Requires the
//!   `audio-decode` feature.
//! - **Matroska/MKA** (`.mka`, `.mkv`): Decoded via `MatroskaDemuxer`.
//!   Requires the `audio-decode` feature.
//!
//! Multi-channel audio is downmixed to mono by per-frame averaging.
//!
//! Sample-rate conversion is *not* performed: the per-pixel bucketing
//! is driven by the file's intrinsic sample rate, not by
//! `ClipWaveformGenerator::sample_rate` (which is used only by the
//! `generate_from_samples` API for caller-supplied buffers).  Skipping
//! resampling keeps the waveform faithful to the source recording and
//! avoids unnecessary CPU cost — `pixels_per_second` is a *display*
//! resolution and does not depend on the audio rate.
//!
//! Unsupported formats, missing files, or decoder errors yield an empty
//! `WaveformData` (`is_empty() == true`) rather than panicking, so callers
//! can fall back to a placeholder render without exception handling.
//!
//! On WASM targets the file API returns an empty waveform because direct
//! filesystem access is not available; callers should use
//! `generate_from_samples` after fetching PCM out-of-band.

#![allow(dead_code)]

use std::path::Path;

/// The result of a waveform computation: one `(min, max)` pair per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformData {
    /// Per-pixel amplitude peaks.  Each element is `(min_sample, max_sample)`
    /// normalised to the range `[-1.0, 1.0]`.
    pub peaks: Vec<(f32, f32)>,
    /// Total audio duration in seconds.
    pub duration_secs: f64,
    /// The pixels-per-second resolution at which the waveform was sampled.
    pub pixels_per_second: f32,
}

impl WaveformData {
    /// Construct an empty `WaveformData` for the given pixel resolution.
    ///
    /// Used as the fallback when file decoding fails (unsupported format,
    /// missing file, decoder error).  Callers can detect this case via
    /// [`Self::is_empty`].
    #[must_use]
    pub fn empty(pixels_per_second: f32) -> Self {
        Self {
            peaks: Vec::new(),
            duration_secs: 0.0,
            pixels_per_second,
        }
    }

    /// Returns the number of pixel columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.peaks.len()
    }

    /// Returns `true` if no peaks were computed (empty audio or zero-length
    /// source clip).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }

    /// Returns the overall peak amplitude (maximum absolute value across all
    /// pixel buckets).
    #[must_use]
    pub fn peak_amplitude(&self) -> f32 {
        self.peaks
            .iter()
            .map(|(lo, hi)| lo.abs().max(hi.abs()))
            .fold(0.0f32, f32::max)
    }
}

/// A simplified waveform thumbnail that stores RMS amplitude per pixel bucket.
///
/// Unlike `ClipWaveformGenerator` (which tracks min/max peaks), `WaveformThumbnail`
/// stores the root-mean-square (RMS) amplitude for each pixel column, giving a
/// perceptually accurate energy representation suitable for compact thumbnail display.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformThumbnail {
    /// RMS amplitude values in `[0.0, 1.0]`, one per pixel column.
    pub rms_values: Vec<f32>,
    /// Width in pixels (same as `rms_values.len()`).
    pub width: u32,
}

impl WaveformThumbnail {
    /// Generates a waveform thumbnail from raw PCM samples.
    ///
    /// `samples` should be normalised to `[-1.0, 1.0]`.  The returned `Vec<f32>`
    /// contains exactly `width` RMS values (or fewer if `samples` is too short).
    /// An empty slice returns a zero-filled vector of length `width`.
    ///
    /// This is a convenience free function wrapping the struct constructor.
    #[must_use]
    pub fn generate(samples: &[f32], width: u32) -> Vec<f32> {
        let w = width as usize;
        if w == 0 {
            return Vec::new();
        }
        if samples.is_empty() {
            return vec![0.0f32; w];
        }

        let samples_per_bucket = ((samples.len() as f64) / (w as f64)).max(1.0);
        let mut result = Vec::with_capacity(w);

        for bucket in 0..w {
            let start = (bucket as f64 * samples_per_bucket) as usize;
            let end = (((bucket + 1) as f64) * samples_per_bucket) as usize;
            let end = end.min(samples.len());

            if start >= samples.len() {
                result.push(0.0f32);
                continue;
            }

            let slice = &samples[start..end];
            let rms = if slice.is_empty() {
                0.0f32
            } else {
                let sum_sq: f32 = slice.iter().map(|&s| s * s).sum();
                (sum_sq / slice.len() as f32).sqrt()
            };
            result.push(rms);
        }

        result
    }

    /// Constructs a `WaveformThumbnail` from raw PCM samples.
    ///
    /// Equivalent to calling `generate` and wrapping the result in the struct.
    #[must_use]
    pub fn from_samples(samples: &[f32], width: u32) -> Self {
        let rms_values = Self::generate(samples, width);
        Self { rms_values, width }
    }

    /// Returns `true` if all RMS values are zero (silence or empty input).
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.rms_values.iter().all(|&v| v < f32::EPSILON)
    }

    /// Returns the peak RMS value across all pixel buckets.
    #[must_use]
    pub fn peak_rms(&self) -> f32 {
        self.rms_values.iter().cloned().fold(0.0f32, f32::max)
    }
}

/// Generator that computes per-pixel waveform data from PCM samples.
#[derive(Debug, Clone)]
pub struct ClipWaveformGenerator {
    /// Audio sample rate in Hz.
    sample_rate: f64,
}

impl ClipWaveformGenerator {
    /// Creates a new generator for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Generates waveform data from raw `f32` PCM samples.
    ///
    /// Samples must be normalised to `[-1.0, 1.0]`.  For multi-channel audio
    /// the caller should mix down or pass only one channel.
    ///
    /// `pixels_per_second` controls the horizontal resolution of the output:
    /// more pixels yield finer detail at the cost of a larger `peaks` vector.
    ///
    /// If `samples` is empty, an empty `WaveformData` with `duration_secs = 0`
    /// is returned.
    #[must_use]
    pub fn generate_from_samples(&self, samples: &[f32], pixels_per_second: f32) -> WaveformData {
        if samples.is_empty() || pixels_per_second <= 0.0 || self.sample_rate <= 0.0 {
            return WaveformData {
                peaks: Vec::new(),
                duration_secs: 0.0,
                pixels_per_second,
            };
        }

        let total_samples = samples.len();
        let duration_secs = total_samples as f64 / self.sample_rate;

        // How many samples map to one pixel column?
        let samples_per_pixel = self.sample_rate / f64::from(pixels_per_second);
        if samples_per_pixel < 1.0 {
            // Pixel rate exceeds sample rate — one sample per pixel max.
            let peaks: Vec<(f32, f32)> = samples.iter().map(|&s| (s, s)).collect();
            return WaveformData {
                peaks,
                duration_secs,
                pixels_per_second,
            };
        }

        let total_pixels = (duration_secs * f64::from(pixels_per_second)).ceil() as usize;
        let mut peaks = Vec::with_capacity(total_pixels);

        for pixel_idx in 0..total_pixels {
            let start = (pixel_idx as f64 * samples_per_pixel) as usize;
            let end = (((pixel_idx + 1) as f64) * samples_per_pixel) as usize;
            let end = end.min(total_samples);

            if start >= total_samples {
                break;
            }

            let slice = &samples[start..end];
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;

            for &s in slice {
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }

            // Guard against zero-length slices (shouldn't happen given checks above).
            if lo == f32::MAX {
                lo = 0.0;
                hi = 0.0;
            }

            peaks.push((lo, hi));
        }

        WaveformData {
            peaks,
            duration_secs,
            pixels_per_second,
        }
    }

    /// Decode an audio file and compute its waveform.
    ///
    /// Detects the file format by extension, decodes the PCM payload to `f32`
    /// samples (normalised to `[-1.0, 1.0]`), downmixes multi-channel audio to
    /// mono by per-frame averaging, then delegates to `generate_from_samples`
    /// using the file's intrinsic sample rate.
    ///
    /// Supported formats (on non-WASM targets):
    /// - WAV/WAVE (always available)
    /// - MP3 (requires `audio-decode` feature, default on)
    /// - Ogg/Opus, Ogg/Vorbis (requires `audio-decode` feature)
    /// - FLAC (requires `audio-decode` feature)
    /// - Matroska/MKA (requires `audio-decode` feature)
    ///
    /// Returns an empty `WaveformData` (`is_empty() == true`) on any of:
    ///
    /// * File-not-found or other I/O error.
    /// * Malformed container header or unsupported sample format.
    /// * `pixels_per_second <= 0.0`.
    ///
    /// The function never panics regardless of file contents.  Sample rate
    /// conversion is intentionally skipped — see the module-level docs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate(&self, audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        if pixels_per_second <= 0.0 {
            return WaveformData::empty(pixels_per_second);
        }

        // Detect format from file extension.
        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase());

        match ext.as_deref() {
            Some("mp3") => {
                #[cfg(feature = "audio-decode")]
                {
                    return self.generate_mp3(audio_path, pixels_per_second);
                }
                #[cfg(not(feature = "audio-decode"))]
                {
                    return WaveformData::empty(pixels_per_second);
                }
            }
            Some("ogg" | "opus" | "oga") => {
                #[cfg(feature = "audio-decode")]
                {
                    return self.generate_ogg(audio_path, pixels_per_second);
                }
                #[cfg(not(feature = "audio-decode"))]
                {
                    return WaveformData::empty(pixels_per_second);
                }
            }
            Some("flac") => {
                #[cfg(feature = "audio-decode")]
                {
                    return self.generate_flac(audio_path, pixels_per_second);
                }
                #[cfg(not(feature = "audio-decode"))]
                {
                    return WaveformData::empty(pixels_per_second);
                }
            }
            Some("mka" | "mkv") => {
                #[cfg(feature = "audio-decode")]
                {
                    return self.generate_matroska(audio_path, pixels_per_second);
                }
                #[cfg(not(feature = "audio-decode"))]
                {
                    return WaveformData::empty(pixels_per_second);
                }
            }
            _ => {}
        }

        // Fall through to WAV decode for `.wav`, unknown extensions, or magic-based detection.
        self.generate_wav(audio_path, pixels_per_second)
    }

    /// Decode a WAV file and return mono f32 samples via `generate_from_samples`.
    #[cfg(not(target_arch = "wasm32"))]
    fn generate_wav(&self, audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        let file = match std::fs::File::open(audio_path) {
            Ok(f) => f,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };
        let buf_reader = std::io::BufReader::new(file);

        let mut wav_reader = match oximedia_audio::wav::WavReader::new(buf_reader) {
            Ok(r) => r,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };

        let spec = wav_reader.spec();
        if spec.channels == 0 || spec.sample_rate == 0 {
            return WaveformData::empty(pixels_per_second);
        }

        let interleaved = match wav_reader.read_samples_f32() {
            Ok(s) => s,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };

        if interleaved.is_empty() {
            return WaveformData::empty(pixels_per_second);
        }

        // Downmix to mono. WAV samples are interleaved frame-by-frame:
        // [c0_f0, c1_f0, …, cN-1_f0, c0_f1, c1_f1, …].
        let channels = usize::from(spec.channels);
        let mono = downmix_interleaved_to_mono(&interleaved, channels);

        // Use the file's intrinsic sample rate for accurate duration and
        // bucketing.  Self's configured rate is irrelevant here.
        let local_generator = Self::new(f64::from(spec.sample_rate));
        local_generator.generate_from_samples(&mono, pixels_per_second)
    }

    /// Decode an MP3 file using `Mp3Decoder` and compute its waveform.
    ///
    /// # Note on tokio runtime
    ///
    /// This method creates a single-threaded tokio `Runtime` per call to drive
    /// the async demuxer.  The overhead is acceptable for waveform generation
    /// (a one-shot operation) but callers that already live inside an async
    /// context should ensure they are not blocking an async executor thread.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    fn generate_mp3(&self, audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        use oximedia_audio::{AudioDecoder, Mp3Decoder};
        use oximedia_io::FileSource;

        let path = audio_path.to_path_buf();

        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };

        let decode_result: Result<(Vec<f32>, u32), Box<dyn std::error::Error + Send + Sync>> = rt
            .block_on(async move {
                // MP3 is a self-synchronising frame-based bitstream. Feed the
                // entire file as one packet to the decoder.
                let source = FileSource::open(&path).await?;
                let data = read_source_to_vec(source).await?;

                let mut decoder = Mp3Decoder::default();

                // Feed all data in one packet.
                decoder
                    .send_packet(&data, 0)
                    .map_err(|e| format!("mp3 send_packet: {e}"))?;

                let mut mono_samples: Vec<f32> = Vec::new();
                let mut sample_rate: u32 = 44100;

                loop {
                    match decoder.receive_frame() {
                        Ok(Some(frame)) => {
                            if frame.sample_rate > 0 {
                                sample_rate = frame.sample_rate;
                            }
                            let channels = frame.channels.count();
                            let frame_mono = audio_frame_to_mono_f32(&frame, channels)?;
                            mono_samples.extend_from_slice(&frame_mono);
                        }
                        Ok(None) => break,
                        Err(oximedia_audio::AudioError::NeedMoreData) => break,
                        Err(oximedia_audio::AudioError::Eof) => break,
                        Err(e) => return Err(format!("mp3 receive_frame: {e}").into()),
                    }
                }

                Ok((mono_samples, sample_rate))
            });

        match decode_result {
            Ok((mono, sr)) if !mono.is_empty() => {
                let local_gen = Self::new(f64::from(sr));
                local_gen.generate_from_samples(&mono, pixels_per_second)
            }
            _ => WaveformData::empty(pixels_per_second),
        }
    }

    /// Decode an Ogg file (Opus or Vorbis) and compute its waveform.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    fn generate_ogg(&self, audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        use oximedia_audio::{AudioDecoder, AudioDecoderConfig};
        use oximedia_container::demux::{Demuxer, OggDemuxer};
        use oximedia_core::CodecId;
        use oximedia_io::FileSource;

        let path = audio_path.to_path_buf();

        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };

        let decode_result: Result<(Vec<f32>, u32), Box<dyn std::error::Error + Send + Sync>> = rt
            .block_on(async move {
                let source = FileSource::open(&path).await?;
                let mut demuxer = OggDemuxer::new(source);
                demuxer.probe().await?;

                let streams = demuxer.streams().to_vec();
                let audio_stream = streams
                    .iter()
                    .find(|s| s.is_audio())
                    .ok_or("no audio stream in ogg")?;

                let codec = audio_stream.codec;
                let sample_rate = audio_stream.codec_params.sample_rate.unwrap_or(48000);
                let channels = audio_stream.codec_params.channels.unwrap_or(2);
                let extradata = audio_stream.codec_params.extradata.clone();

                let config = AudioDecoderConfig {
                    codec,
                    sample_rate,
                    channels,
                    extradata: extradata.map(|b| b.to_vec()),
                };

                let mut decoder: Box<dyn AudioDecoder> = match codec {
                    CodecId::Opus => {
                        let dec = oximedia_audio::OpusDecoder::new(&config)
                            .map_err(|e| format!("opus decoder: {e}"))?;
                        Box::new(dec)
                    }
                    CodecId::Vorbis => {
                        let dec = oximedia_audio::VorbisDecoder::new(&config)
                            .map_err(|e| format!("vorbis decoder: {e}"))?;
                        Box::new(dec)
                    }
                    _ => return Err(format!("unsupported ogg codec: {codec:?}").into()),
                };

                let audio_stream_idx = audio_stream.index;
                let mut mono_samples: Vec<f32> = Vec::new();
                let mut detected_rate = sample_rate;

                loop {
                    match demuxer.read_packet().await {
                        Ok(pkt) if pkt.stream_index == audio_stream_idx => {
                            decoder
                                .send_packet(&pkt.data, pkt.pts())
                                .map_err(|e| format!("ogg send_packet: {e}"))?;

                            loop {
                                match decoder.receive_frame() {
                                    Ok(Some(frame)) => {
                                        if frame.sample_rate > 0 {
                                            detected_rate = frame.sample_rate;
                                        }
                                        let ch = frame.channels.count();
                                        let frame_mono = audio_frame_to_mono_f32(&frame, ch)?;
                                        mono_samples.extend_from_slice(&frame_mono);
                                    }
                                    Ok(None) => break,
                                    Err(oximedia_audio::AudioError::NeedMoreData) => break,
                                    Err(e) => return Err(format!("ogg receive_frame: {e}").into()),
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) if e.is_eof() => break,
                        Err(e) => return Err(format!("ogg read_packet: {e}").into()),
                    }
                }

                Ok((mono_samples, detected_rate))
            });

        match decode_result {
            Ok((mono, sr)) if !mono.is_empty() => {
                let local_gen = Self::new(f64::from(sr));
                local_gen.generate_from_samples(&mono, pixels_per_second)
            }
            _ => WaveformData::empty(pixels_per_second),
        }
    }

    /// Decode a FLAC file and compute its waveform.
    ///
    /// # Note on decoder stub
    ///
    /// The `FlacDecoder` audio codec is currently a stub implementation that
    /// returns `Ok(None)` from `receive_frame()`.  As a result, this path
    /// will return an empty `WaveformData` until the decoder is fully
    /// implemented.  The demuxer and codec wiring are correct and ready for
    /// when the decoder is completed.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    fn generate_flac(&self, audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        use oximedia_audio::{AudioDecoder, AudioDecoderConfig, FlacDecoder};
        use oximedia_container::demux::{Demuxer, FlacDemuxer};
        use oximedia_core::CodecId;
        use oximedia_io::FileSource;

        let path = audio_path.to_path_buf();

        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };

        let decode_result: Result<(Vec<f32>, u32), Box<dyn std::error::Error + Send + Sync>> = rt
            .block_on(async move {
                let source = FileSource::open(&path).await?;
                let mut demuxer = FlacDemuxer::new(source);
                demuxer.probe().await?;

                let streams = demuxer.streams().to_vec();
                let audio_stream = streams
                    .iter()
                    .find(|s| s.is_audio())
                    .ok_or("no audio stream in flac")?;

                let sample_rate = audio_stream.codec_params.sample_rate.unwrap_or(44100);
                let channels = audio_stream.codec_params.channels.unwrap_or(1);
                let extradata = audio_stream.codec_params.extradata.clone();

                let config = AudioDecoderConfig {
                    codec: CodecId::Flac,
                    sample_rate,
                    channels,
                    extradata: extradata.map(|b| b.to_vec()),
                };
                let mut decoder =
                    FlacDecoder::new(&config).map_err(|e| format!("flac decoder init: {e}"))?;

                let audio_stream_idx = audio_stream.index;
                let mut mono_samples: Vec<f32> = Vec::new();
                let mut detected_rate = sample_rate;

                loop {
                    match demuxer.read_packet().await {
                        Ok(pkt) if pkt.stream_index == audio_stream_idx => {
                            decoder
                                .send_packet(&pkt.data, pkt.pts())
                                .map_err(|e| format!("flac send_packet: {e}"))?;

                            loop {
                                match decoder.receive_frame() {
                                    Ok(Some(frame)) => {
                                        if frame.sample_rate > 0 {
                                            detected_rate = frame.sample_rate;
                                        }
                                        let ch = frame.channels.count();
                                        let frame_mono = audio_frame_to_mono_f32(&frame, ch)?;
                                        mono_samples.extend_from_slice(&frame_mono);
                                    }
                                    Ok(None) => break,
                                    Err(oximedia_audio::AudioError::NeedMoreData) => break,
                                    Err(e) => return Err(format!("flac receive_frame: {e}").into()),
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) if e.is_eof() => break,
                        Err(e) => return Err(format!("flac read_packet: {e}").into()),
                    }
                }

                Ok((mono_samples, detected_rate))
            });

        match decode_result {
            Ok((mono, sr)) if !mono.is_empty() => {
                let local_gen = Self::new(f64::from(sr));
                local_gen.generate_from_samples(&mono, pixels_per_second)
            }
            _ => WaveformData::empty(pixels_per_second),
        }
    }

    /// Decode a Matroska (MKV/MKA) file and compute its waveform.
    ///
    /// Dispatches to the appropriate audio decoder based on the codec ID
    /// of the first audio stream found.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    fn generate_matroska(&self, audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        use oximedia_audio::{AudioDecoder, AudioDecoderConfig};
        use oximedia_container::demux::{Demuxer, MatroskaDemuxer};
        use oximedia_core::CodecId;
        use oximedia_io::FileSource;

        let path = audio_path.to_path_buf();

        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(_) => return WaveformData::empty(pixels_per_second),
        };

        let decode_result: Result<(Vec<f32>, u32), Box<dyn std::error::Error + Send + Sync>> = rt
            .block_on(async move {
                let source = FileSource::open(&path).await?;
                let mut demuxer = MatroskaDemuxer::new(source);
                demuxer.probe().await?;

                let streams = demuxer.streams().to_vec();
                let audio_stream = streams
                    .iter()
                    .find(|s| s.is_audio())
                    .ok_or("no audio stream in matroska")?;

                let codec = audio_stream.codec;
                let sample_rate = audio_stream.codec_params.sample_rate.unwrap_or(48000);
                let channels = audio_stream.codec_params.channels.unwrap_or(2);
                let extradata = audio_stream.codec_params.extradata.clone();

                let config = AudioDecoderConfig {
                    codec,
                    sample_rate,
                    channels,
                    extradata: extradata.map(|b| b.to_vec()),
                };

                let mut decoder: Box<dyn AudioDecoder> = match codec {
                    CodecId::Opus => {
                        let dec = oximedia_audio::OpusDecoder::new(&config)
                            .map_err(|e| format!("opus decoder: {e}"))?;
                        Box::new(dec)
                    }
                    CodecId::Vorbis => {
                        let dec = oximedia_audio::VorbisDecoder::new(&config)
                            .map_err(|e| format!("vorbis decoder: {e}"))?;
                        Box::new(dec)
                    }
                    CodecId::Flac => {
                        let dec = oximedia_audio::FlacDecoder::new(&config)
                            .map_err(|e| format!("flac decoder: {e}"))?;
                        Box::new(dec)
                    }
                    CodecId::Mp3 => Box::new(oximedia_audio::Mp3Decoder::default()),
                    _ => return Err(format!("unsupported matroska codec: {codec:?}").into()),
                };

                let audio_stream_idx = audio_stream.index;
                let mut mono_samples: Vec<f32> = Vec::new();
                let mut detected_rate = sample_rate;

                loop {
                    match demuxer.read_packet().await {
                        Ok(pkt) if pkt.stream_index == audio_stream_idx => {
                            decoder
                                .send_packet(&pkt.data, pkt.pts())
                                .map_err(|e| format!("matroska send_packet: {e}"))?;

                            loop {
                                match decoder.receive_frame() {
                                    Ok(Some(frame)) => {
                                        if frame.sample_rate > 0 {
                                            detected_rate = frame.sample_rate;
                                        }
                                        let ch = frame.channels.count();
                                        let frame_mono = audio_frame_to_mono_f32(&frame, ch)?;
                                        mono_samples.extend_from_slice(&frame_mono);
                                    }
                                    Ok(None) => break,
                                    Err(oximedia_audio::AudioError::NeedMoreData) => break,
                                    Err(e) => {
                                        return Err(format!("matroska receive_frame: {e}").into())
                                    }
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) if e.is_eof() => break,
                        Err(e) => return Err(format!("matroska read_packet: {e}").into()),
                    }
                }

                Ok((mono_samples, detected_rate))
            });

        match decode_result {
            Ok((mono, sr)) if !mono.is_empty() => {
                let local_gen = Self::new(f64::from(sr));
                local_gen.generate_from_samples(&mono, pixels_per_second)
            }
            _ => WaveformData::empty(pixels_per_second),
        }
    }

    /// Decode an audio file and compute its waveform (WASM stub).
    ///
    /// Filesystem decoding is not available on WebAssembly; callers should
    /// fetch and decode audio out-of-band, then use `generate_from_samples`.
    /// This entry point exists so that the same API signature is exposed on
    /// every target — it always returns an empty `WaveformData`.
    #[cfg(target_arch = "wasm32")]
    pub fn generate(&self, _audio_path: &Path, pixels_per_second: f32) -> WaveformData {
        WaveformData::empty(pixels_per_second)
    }
}

/// Downmix interleaved multi-channel PCM to mono by averaging across channels.
///
/// If `channels == 1` or `channels == 0`, the input is returned as-is.
#[cfg(not(target_arch = "wasm32"))]
fn downmix_interleaved_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let frame_count = interleaved.len() / channels;
    let mut out = Vec::with_capacity(frame_count);
    let inv = 1.0_f32 / channels as f32;
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        out.push(sum * inv);
    }
    out
}

/// Convert an `AudioFrame` to interleaved mono `f32` samples.
///
/// Handles the `Interleaved` and `Planar` buffer layouts and the common
/// integer sample formats (U8, S16, S32, S24, F32, F64).
///
/// Returns an error string (for propagation via `Box<dyn Error>`) if the
/// format is not recognised.
#[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
fn audio_frame_to_mono_f32(
    frame: &oximedia_audio::AudioFrame,
    channels: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    use oximedia_audio::AudioBuffer;

    let ch = channels.max(1);

    let interleaved: Vec<f32> = match &frame.samples {
        AudioBuffer::Interleaved(data) => bytes_to_f32_interleaved(data, frame.format)?,
        AudioBuffer::Planar(planes) => {
            // Collect all planes, convert each to f32, then interleave.
            let plane_count = planes.len().max(1);
            let samples_per_plane = if planes.is_empty() {
                0
            } else {
                let bps = frame.format.bytes_per_sample();
                planes[0].len().checked_div(bps).unwrap_or(0)
            };
            let mut out = vec![0.0_f32; samples_per_plane * plane_count];
            for (p_idx, plane) in planes.iter().enumerate() {
                let plane_f32 = bytes_to_f32_interleaved(plane, frame.format)?;
                for (s_idx, &s) in plane_f32.iter().enumerate() {
                    // Interleave: out[s_idx * plane_count + p_idx]
                    let dst = s_idx * plane_count + p_idx;
                    if dst < out.len() {
                        out[dst] = s;
                    }
                }
            }
            out
        }
    };

    // Downmix to mono.
    Ok(downmix_interleaved_to_mono(&interleaved, ch))
}

/// Convert a byte slice in the given `SampleFormat` to normalised `f32` samples.
#[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
fn bytes_to_f32_interleaved(
    data: &[u8],
    format: oximedia_core::SampleFormat,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    use oximedia_core::SampleFormat;

    let bps = format.bytes_per_sample();
    if bps == 0 || data.is_empty() {
        return Ok(Vec::new());
    }
    let sample_count = data.len() / bps;
    let mut out = Vec::with_capacity(sample_count);

    match format {
        SampleFormat::U8 => {
            for &b in data.iter().take(sample_count) {
                out.push((f32::from(b) - 128.0) / 128.0);
            }
        }
        SampleFormat::S16 | SampleFormat::S16p => {
            for chunk in data.chunks_exact(2) {
                let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f32::from(v) / 32768.0);
            }
        }
        SampleFormat::S24 | SampleFormat::S24p => {
            for chunk in data.chunks_exact(3) {
                let raw = (i32::from(chunk[0]))
                    | (i32::from(chunk[1]) << 8)
                    | (i32::from(chunk[2]) << 16);
                let signed = if raw & 0x80_0000 != 0 {
                    raw | !0xFF_FFFF
                } else {
                    raw
                };
                #[allow(clippy::cast_precision_loss)]
                out.push(signed as f32 / 8_388_607.0);
            }
        }
        SampleFormat::S32 | SampleFormat::S32p => {
            for chunk in data.chunks_exact(4) {
                let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                #[allow(clippy::cast_precision_loss)]
                out.push(v as f32 / 2_147_483_647.0);
            }
        }
        SampleFormat::F32 | SampleFormat::F32p => {
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        SampleFormat::F64 | SampleFormat::F64p => {
            for chunk in data.chunks_exact(8) {
                let v = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                #[allow(clippy::cast_possible_truncation)]
                out.push(v as f32);
            }
        }
        // Guard against future variants added to the non-exhaustive enum.
        _ => {
            return Err(format!("unsupported sample format: {format:?}").into());
        }
    }

    Ok(out)
}

/// Read an entire `MediaSource` into a `Vec<u8>`.
///
/// Used for feeding raw bitstream data (e.g. MP3) to a packet-based decoder.
#[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
async fn read_source_to_vec(
    mut source: oximedia_io::FileSource,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use oximedia_io::MediaSource;

    let mut buf = Vec::new();
    let mut tmp = vec![0u8; 65536];
    loop {
        let n = source.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_samples(hz: f32, sample_rate: f64, duration_secs: f64) -> Vec<f32> {
        let n = (sample_rate * duration_secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (t * hz * std::f32::consts::TAU).sin()
            })
            .collect()
    }

    #[test]
    fn test_empty_samples_returns_empty_waveform() {
        let gen = ClipWaveformGenerator::new(48000.0);
        let wd = gen.generate_from_samples(&[], 100.0);
        assert!(wd.is_empty());
        assert_eq!(wd.duration_secs, 0.0);
    }

    #[test]
    fn test_duration_secs_correct() {
        let gen = ClipWaveformGenerator::new(48000.0);
        let samples = vec![0.0f32; 48000]; // 1 second
        let wd = gen.generate_from_samples(&samples, 100.0);
        assert!((wd.duration_secs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pixels_per_second_stored() {
        let gen = ClipWaveformGenerator::new(44100.0);
        let samples = vec![0.5f32; 44100];
        let wd = gen.generate_from_samples(&samples, 50.0);
        assert!((wd.pixels_per_second - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_waveform_width_proportional_to_duration() {
        let gen = ClipWaveformGenerator::new(48000.0);
        let samples_1s = vec![0.0f32; 48000];
        let wd_1s = gen.generate_from_samples(&samples_1s, 100.0);

        let samples_2s = vec![0.0f32; 96000];
        let wd_2s = gen.generate_from_samples(&samples_2s, 100.0);

        assert_eq!(wd_1s.width(), 100);
        assert_eq!(wd_2s.width(), 200);
    }

    #[test]
    fn test_positive_only_signal_peaks_non_negative_min() {
        let gen = ClipWaveformGenerator::new(44100.0);
        // DC offset at +0.5
        let samples = vec![0.5f32; 44100];
        let wd = gen.generate_from_samples(&samples, 10.0);
        for (lo, hi) in &wd.peaks {
            assert!(*lo >= 0.0, "min should be >= 0 for positive DC");
            assert!(*hi <= 1.0 + f32::EPSILON);
        }
    }

    #[test]
    fn test_min_max_ordering() {
        let gen = ClipWaveformGenerator::new(48000.0);
        let samples = sine_samples(440.0, 48000.0, 1.0);
        let wd = gen.generate_from_samples(&samples, 100.0);
        for (lo, hi) in &wd.peaks {
            assert!(lo <= hi, "min should always be <= max");
        }
    }

    #[test]
    fn test_peak_amplitude_for_sine() {
        let gen = ClipWaveformGenerator::new(48000.0);
        let samples = sine_samples(440.0, 48000.0, 1.0);
        let wd = gen.generate_from_samples(&samples, 100.0);
        let peak = wd.peak_amplitude();
        // A unit sine's amplitude ≈ 1.0 (within floating-point noise).
        assert!(peak > 0.9 && peak <= 1.0 + 1e-4, "peak={peak}");
    }

    #[test]
    fn test_zero_pixels_per_second_returns_empty() {
        let gen = ClipWaveformGenerator::new(48000.0);
        let samples = vec![0.0f32; 100];
        let wd = gen.generate_from_samples(&samples, 0.0);
        assert!(wd.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_generate_from_missing_file_returns_empty() {
        let gen = ClipWaveformGenerator::new(48000.0);
        // A path that almost certainly does not exist on any host filesystem.
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_definitely_missing_waveform_input.wav");
        // Ensure it really is absent.
        let _ = std::fs::remove_file(&path);
        let wd = gen.generate(&path, 100.0);
        assert!(wd.is_empty());
        assert_eq!(wd.duration_secs, 0.0);
        assert!((wd.pixels_per_second - 100.0).abs() < f32::EPSILON);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_generate_from_non_wav_file_returns_empty() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_not_a_wav.bin");
        // Random non-RIFF bytes.
        std::fs::write(&path, b"this is not a wav file at all").expect("write tmp file");

        let gen = ClipWaveformGenerator::new(48000.0);
        let wd = gen.generate(&path, 100.0);

        let _ = std::fs::remove_file(&path);
        assert!(wd.is_empty());
        assert_eq!(wd.duration_secs, 0.0);
    }

    /// Writes a one-second 440 Hz sine WAV (mono, 16-bit, 16 kHz) to `path`.
    #[cfg(not(target_arch = "wasm32"))]
    fn write_test_wav_mono_16k(
        path: &std::path::Path,
        freq_hz: f32,
        duration_secs: f32,
    ) -> std::io::Result<()> {
        use oximedia_audio::wav::{WavSpec, WavWriter};
        use std::fs::File;
        use std::io::BufWriter;

        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            float: false,
        };
        let sample_count = (16_000.0_f32 * duration_secs) as usize;

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let mut wav = WavWriter::new(writer, spec);

        for i in 0..sample_count {
            let t = i as f32 / 16_000.0_f32;
            let s = (t * freq_hz * std::f32::consts::TAU).sin() * 0.8;
            wav.write_sample_f32(s)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        wav.finalize()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_generate_from_real_wav_file() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_test_sine_440_mono.wav");
        let _ = std::fs::remove_file(&path);
        write_test_wav_mono_16k(&path, 440.0, 1.0).expect("write wav");

        // The configured sample rate is irrelevant on this code path — the
        // file's intrinsic 16 kHz rate drives bucketing.  Use a different
        // value to prove that.
        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 100.0);

        // Cleanup before any assertion can fail.
        let _ = std::fs::remove_file(&path);

        assert!(!wd.is_empty(), "decoded waveform should not be empty");
        // One second of audio at 100 px/sec → ~100 pixel columns.
        assert!(
            wd.width() >= 95 && wd.width() <= 105,
            "expected ~100 pixels, got {}",
            wd.width()
        );
        assert!(
            (wd.duration_secs - 1.0).abs() < 0.05,
            "expected ~1.0 s, got {}",
            wd.duration_secs
        );
        assert!((wd.pixels_per_second - 100.0).abs() < f32::EPSILON);

        // Peak amplitude of a 0.8-scaled sine should be near 0.8.
        let peak = wd.peak_amplitude();
        assert!(
            peak > 0.6 && peak <= 1.0 + 1e-4,
            "peak_amplitude={peak} outside expected band"
        );

        // The waveform must oscillate above and below zero (not DC).
        let mut has_pos = false;
        let mut has_neg = false;
        for (lo, hi) in &wd.peaks {
            if *hi > 0.1 {
                has_pos = true;
            }
            if *lo < -0.1 {
                has_neg = true;
            }
            assert!(lo <= hi);
            if has_pos && has_neg {
                break;
            }
        }
        assert!(has_pos, "sine should produce positive peaks");
        assert!(has_neg, "sine should produce negative peaks");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_generate_from_real_wav_stereo_downmix() {
        use oximedia_audio::wav::{WavSpec, WavWriter};
        use std::fs::File;
        use std::io::BufWriter;

        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_test_sine_stereo.wav");
        let _ = std::fs::remove_file(&path);

        // Stereo 16-bit/22.05 kHz, 0.5 s.  Left channel = sine, right
        // channel = inverted sine → after averaging, downmix is silence,
        // which is a strong correctness signal for the downmixer.
        let spec = WavSpec {
            channels: 2,
            sample_rate: 22_050,
            bits_per_sample: 16,
            float: false,
        };
        let file = File::create(&path).expect("create wav");
        let writer = BufWriter::new(file);
        let mut wav = WavWriter::new(writer, spec);

        let sample_count = (22_050.0_f32 * 0.5) as usize;
        for i in 0..sample_count {
            let t = i as f32 / 22_050.0_f32;
            let left = (t * 440.0 * std::f32::consts::TAU).sin() * 0.5;
            wav.write_sample_f32(left).expect("write left");
            wav.write_sample_f32(-left).expect("write right"); // anti-phase
        }
        wav.finalize().expect("finalize");

        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 50.0);

        let _ = std::fs::remove_file(&path);

        assert!(!wd.is_empty(), "stereo decode should yield non-empty data");
        // 0.5 s × 50 px/s → ~25 pixels.
        assert!(
            wd.width() >= 20 && wd.width() <= 30,
            "expected ~25 pixels, got {}",
            wd.width()
        );
        assert!((wd.duration_secs - 0.5).abs() < 0.05);
        // Anti-phase L/R averages to ~0 → peak amplitude should be tiny.
        let peak = wd.peak_amplitude();
        assert!(
            peak < 0.05,
            "downmix of anti-phase stereo should be near silence, got {peak}"
        );
    }

    // ---- audio-decode dispatch tests ----

    /// Helper: write a minimal WAV file and verify `generate()` dispatches to
    /// the WAV path correctly via extension-based routing.
    ///
    /// This is also the primary "plumbing compiles" smoke test for the new
    /// format-dispatch code that runs on every target.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_wav_dispatch_still_works_after_format_routing() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_dispatch_smoke_test.wav");
        let _ = std::fs::remove_file(&path);
        write_test_wav_mono_16k(&path, 220.0, 0.5).expect("write wav");

        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 100.0);
        let _ = std::fs::remove_file(&path);

        assert!(
            !wd.is_empty(),
            "WAV dispatch must still produce waveform data"
        );
        assert!(
            wd.width() >= 40 && wd.width() <= 60,
            "0.5 s @ 100 px/s → ~50 px"
        );
    }

    /// `generate()` with a missing MP3 path must return empty without panic.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    #[test]
    fn test_mp3_missing_file_returns_empty() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_definitely_missing.mp3");
        let _ = std::fs::remove_file(&path);

        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 100.0);
        assert!(wd.is_empty(), "missing mp3 should return empty waveform");
    }

    /// `generate()` with a corrupted/empty .ogg file must return empty without panic.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    #[test]
    fn test_ogg_corrupted_returns_empty() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_corrupted.ogg");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"not an ogg file").expect("write tmp");

        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 100.0);
        let _ = std::fs::remove_file(&path);

        assert!(wd.is_empty(), "corrupted ogg should return empty waveform");
    }

    /// `generate()` with a corrupted/empty .flac file must return empty without panic.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    #[test]
    fn test_flac_corrupted_returns_empty() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_corrupted.flac");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"not a flac file").expect("write tmp");

        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 100.0);
        let _ = std::fs::remove_file(&path);

        assert!(wd.is_empty(), "corrupted flac should return empty waveform");
    }

    /// `generate()` with a corrupted/empty .mka file must return empty without panic.
    #[cfg(all(not(target_arch = "wasm32"), feature = "audio-decode"))]
    #[test]
    fn test_mka_corrupted_returns_empty() {
        let mut path = std::env::temp_dir();
        path.push("oximedia_clips_corrupted.mka");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"not a matroska file").expect("write tmp");

        let gen = ClipWaveformGenerator::new(48_000.0);
        let wd = gen.generate(&path, 100.0);
        let _ = std::fs::remove_file(&path);

        assert!(wd.is_empty(), "corrupted mka should return empty waveform");
    }

    /// Verify `downmix_interleaved_to_mono` correctly averages stereo to mono.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_downmix_interleaved_stereo_averages_channels() {
        // Stereo frames: [L=1.0, R=-1.0] → mono should be 0.0.
        let stereo: Vec<f32> = (0..10).flat_map(|_| [1.0f32, -1.0f32]).collect();
        let mono = downmix_interleaved_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 10);
        for s in &mono {
            assert!(
                s.abs() < f32::EPSILON,
                "anti-phase stereo should cancel: {s}"
            );
        }
    }

    /// Verify mono pass-through in `downmix_interleaved_to_mono`.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_downmix_mono_passthrough() {
        let samples = vec![0.5f32; 20];
        let out = downmix_interleaved_to_mono(&samples, 1);
        assert_eq!(out, samples);
    }

    // NOTE: Real FLAC/Opus/Vorbis synthetic-file tests are not included because:
    //
    // (a) `FlacDecoder` and `VorbisDecoder` are stub implementations that return
    //     `Ok(None)` from `receive_frame()` — they will yield empty waveforms until
    //     the codec decoders are fully implemented.
    //
    // (b) `OpusDecoder` is functional but constructing a valid Ogg/Opus bitstream
    //     (with identification, comment, and audio pages in correct Ogg page framing)
    //     without a muxer is involved; the `OggMuxer` exists but requires an
    //     async context and full header negotiation.
    //
    // (c) MP3 encoding is not exposed in `oximedia-audio` so synthesising a test
    //     MP3 file requires external tooling.
    //
    // The error-path tests above (corrupted/missing files) confirm the new dispatch
    // code compiles and runs without panic for all four formats.  The WAV fallback
    // test confirms that the format-routing logic does not regress the working path.

    // ---- WaveformThumbnail tests ----

    #[test]
    fn test_waveform_thumbnail_empty_samples_returns_zeros() {
        let rms = WaveformThumbnail::generate(&[], 10);
        assert_eq!(rms.len(), 10);
        for v in &rms {
            assert!(*v < f32::EPSILON);
        }
    }

    #[test]
    fn test_waveform_thumbnail_zero_width_returns_empty() {
        let rms = WaveformThumbnail::generate(&[1.0f32; 100], 0);
        assert!(rms.is_empty());
    }

    #[test]
    fn test_waveform_thumbnail_correct_length() {
        let samples = vec![0.5f32; 48000];
        let rms = WaveformThumbnail::generate(&samples, 200);
        assert_eq!(rms.len(), 200);
    }

    #[test]
    fn test_waveform_thumbnail_dc_rms_value() {
        // DC offset at +0.7 → RMS should be ≈ 0.7
        let samples = vec![0.7f32; 48000];
        let rms = WaveformThumbnail::generate(&samples, 100);
        for v in &rms {
            assert!((*v - 0.7).abs() < 1e-3, "expected ≈0.7, got {v}");
        }
    }

    #[test]
    fn test_waveform_thumbnail_silence_is_silent() {
        let thumb = WaveformThumbnail::from_samples(&[0.0f32; 1000], 50);
        assert!(thumb.is_silent());
        assert!(thumb.peak_rms() < f32::EPSILON);
    }

    #[test]
    fn test_waveform_thumbnail_sine_nonzero_rms() {
        let samples = sine_samples(440.0, 48000.0, 1.0);
        let thumb = WaveformThumbnail::from_samples(&samples, 100);
        assert!(!thumb.is_silent());
        // RMS of unit sine ≈ 1/√2 ≈ 0.707
        let peak = thumb.peak_rms();
        assert!(peak > 0.5 && peak <= 1.0 + 1e-4, "peak_rms={peak}");
    }

    #[test]
    fn test_waveform_thumbnail_width_field_matches_rms_len() {
        let samples = vec![0.3f32; 4800];
        let thumb = WaveformThumbnail::from_samples(&samples, 48);
        assert_eq!(thumb.width as usize, thumb.rms_values.len());
    }
}
