//! Audio resampling utilities.
//!
//! This module provides high-quality, 100% Pure-Rust audio resampling built
//! on a band-limited windowed-sinc polyphase interpolator. No C, C++, or
//! Fortran code is compiled, and no third-party FFT backend is required.
//!
//! # Algorithm
//!
//! The resampler precomputes an oversampled prototype low-pass filter: a
//! sinc kernel shaped by a 4-term Blackman-Harris window (≈ −92 dB stopband),
//! sampled at `oversampling + 1` fractional phases of `taps` coefficients
//! each. Every phase row is normalized to unit DC gain, which removes
//! passband amplitude ripple at 0 Hz and keeps round-trips level-accurate.
//!
//! For each output sample the continuous input position is tracked with an
//! *exact rational accumulator* (no floating-point drift over arbitrarily
//! long streams). The two phase rows bracketing the fractional position are
//! linearly interpolated to synthesize the exact fractional-delay filter,
//! which is then applied to all channels.
//!
//! When downsampling, the kernel cutoff is scaled by the rate ratio so the
//! same filter performs anti-alias filtering and interpolation in one pass.
//!
//! Output sample `n` corresponds to input time `n / target_rate` — the
//! filter's group delay is compensated internally, so the output is
//! time-aligned with the input (streaming latency of `taps / 2` input
//! samples, but no leading silence and no phase shift).
//!
//! # Features
//!
//! - Multiple quality presets (Low, Medium, High, Best + Draft/Good aliases)
//! - Support for all sample formats (U8, S16, S32, F32, F64, and planar variants)
//! - Multi-channel audio support
//! - Streaming resampling with state management and `flush()` for stream tails
//! - Chunk-size invariant: splitting the input into arbitrary chunks produces
//!   bit-identical output to one-shot processing
//!
//! # Examples
//!
//! ```rust,ignore
//! use oximedia_audio::{Resampler, ResamplerQuality};
//!
//! let mut resampler = Resampler::new(44100, 48000, 2, ResamplerQuality::High)?;
//! // Process audio frames...
//! let out = resampler.resample(&frame)?;
//! // ... at end of stream:
//! let tail = resampler.flush()?;
//! ```

use crate::{AudioBuffer, AudioError, AudioFrame, AudioResult, ChannelLayout};
use bytes::Bytes;
use oximedia_core::SampleFormat;
use std::f64::consts::PI;

/// Resampling quality preset.
///
/// Higher quality settings use more CPU and memory but provide better
/// frequency response and lower aliasing.
///
/// # Preset aliases
///
/// Three named aliases follow the common `draft / good / best` convention:
///
/// | Alias | Maps to | Description |
/// |-------|---------|-------------|
/// | [`Draft`](ResamplerQuality::Draft) | `Low` | Fastest; real-time non-critical use |
/// | [`Good`](ResamplerQuality::Good)  | `High` | Long windowed-sinc filter |
/// | [`Best`](ResamplerQuality::Best)  | `Best` | Longest filter (highest quality) |
///
/// `Low`, `Medium`, `High`, and `Best` remain available for backward
/// compatibility and for when you need the intermediate `Medium` level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResamplerQuality {
    /// Low quality — fastest, suitable for real-time where quality is not critical.
    /// Uses a short (64-tap) windowed-sinc filter.
    ///
    /// Equivalent to [`Draft`](ResamplerQuality::Draft).
    Low,

    /// Medium quality — balanced speed and quality.
    /// Uses a 128-tap windowed-sinc filter.
    #[default]
    Medium,

    /// High quality — good quality for most applications.
    /// Uses a 192-tap windowed-sinc filter with fine phase resolution.
    ///
    /// Equivalent to [`Good`](ResamplerQuality::Good).
    High,

    /// Best quality — highest quality, most CPU intensive.
    /// Uses a 256-tap windowed-sinc filter with fine phase resolution.
    Best,

    // ── Named aliases ────────────────────────────────────────────────────────
    /// Draft quality — alias for [`Low`](ResamplerQuality::Low).
    ///
    /// Fastest setting, suitable for scrubbing or non-critical real-time use.
    Draft,

    /// Good quality — alias for [`High`](ResamplerQuality::High).
    ///
    /// Long windowed-sinc filter; a solid all-round choice.
    Good,
}

impl ResamplerQuality {
    /// Resolve an alias to its canonical variant.
    ///
    /// `Draft` → `Low`, `Good` → `High`; all other variants resolve to
    /// themselves.
    #[must_use]
    pub const fn canonical(self) -> Self {
        match self {
            Self::Draft => Self::Low,
            Self::Good => Self::High,
            other => other,
        }
    }

    /// Get windowed-sinc interpolation parameters for this quality level.
    #[must_use]
    fn sinc_params(&self) -> SincParams {
        match self.canonical() {
            Self::Low => SincParams {
                taps: 64,
                oversampling: 128,
                f_cutoff: 0.90,
            },
            Self::Medium => SincParams {
                taps: 128,
                oversampling: 192,
                f_cutoff: 0.925,
            },
            Self::High => SincParams {
                taps: 192,
                oversampling: 256,
                f_cutoff: 0.945,
            },
            // `Best` and (already-canonicalized) aliases.
            _ => SincParams {
                taps: 256,
                oversampling: 256,
                f_cutoff: 0.95,
            },
        }
    }
}

/// Windowed-sinc filter design parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SincParams {
    /// Filter length in input samples (even).
    taps: usize,
    /// Number of precomputed fractional phases (table has `oversampling + 1` rows).
    oversampling: usize,
    /// Relative cutoff frequency as a fraction of the lower Nyquist frequency.
    f_cutoff: f64,
}

/// Resampling strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResamplerStrategy {
    /// Passthrough (no resampling needed).
    Passthrough,
    /// Streaming windowed-sinc polyphase resampling.
    SincPolyphase,
}

// ─────────────────────────────────────────────────────────────────────────────
// Sinc polyphase engine
// ─────────────────────────────────────────────────────────────────────────────

/// Normalized sinc function: `sin(πx) / (πx)`.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// 4-term Blackman-Harris window on `u ∈ [-1, 1]` (0 outside).
///
/// Provides ≈ −92 dB peak sidelobe level.
fn blackman_harris(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        return 0.0;
    }
    const A0: f64 = 0.35875;
    const A1: f64 = 0.48829;
    const A2: f64 = 0.14128;
    const A3: f64 = 0.01168;
    A0 + A1 * (PI * u).cos() + A2 * (2.0 * PI * u).cos() + A3 * (3.0 * PI * u).cos()
}

/// Streaming windowed-sinc polyphase resampler engine (single instance
/// handles all channels so the interpolated coefficient row is shared).
struct SincEngine {
    /// Filter length in input samples (even).
    taps: usize,
    /// Half the filter length.
    half: i64,
    /// Number of fractional phases (table rows = `oversampling + 1`).
    oversampling: usize,
    /// Flattened coefficient table: `(oversampling + 1) × taps`, phase-major.
    table: Vec<f64>,
    /// Rational position step numerator: `source_rate / gcd`.
    step_num: u64,
    /// Rational position denominator: `target_rate / gcd`.
    denom: u64,
    /// Current fractional position numerator (`0..denom`).
    frac_num: u64,
    /// Integer input index of the next output sample's position.
    next_idx: i64,
    /// Per-channel input buffer (history + pending samples).
    buf: Vec<Vec<f32>>,
    /// Absolute input index of `buf[ch][0]`.
    buf_start: i64,
    /// Total number of real input samples pushed so far.
    pushed: i64,
    /// Channel count.
    channels: usize,
    /// Scratch row for the interpolated fractional-delay filter.
    scratch: Vec<f64>,
    /// Set once `flush()` has been called; further input is rejected.
    finished: bool,
}

impl SincEngine {
    /// Create a new engine for the given rates and filter parameters.
    fn new(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        params: SincParams,
    ) -> AudioResult<Self> {
        if params.taps < 4 || params.taps % 2 != 0 {
            return Err(AudioError::InvalidParameter(
                "Sinc filter length must be an even number >= 4".into(),
            ));
        }
        if params.oversampling < 2 {
            return Err(AudioError::InvalidParameter(
                "Sinc oversampling factor must be >= 2".into(),
            ));
        }

        let g = gcd(source_rate, target_rate);
        let step_num = u64::from(source_rate / g);
        let denom = u64::from(target_rate / g);

        // Anti-alias cutoff: relative to the input Nyquist frequency, limited
        // to the output Nyquist when downsampling.
        let ratio = f64::from(target_rate) / f64::from(source_rate);
        let cutoff = params.f_cutoff * ratio.min(1.0);

        let table = Self::build_table(params.taps, params.oversampling, cutoff);
        let half = (params.taps / 2) as i64;

        // Prime the history with `taps` zeros so the first output (centered
        // at input position 0) has a full window of valid history.
        let buf = vec![vec![0.0f32; params.taps]; channels];

        Ok(Self {
            taps: params.taps,
            half,
            oversampling: params.oversampling,
            table,
            step_num,
            denom,
            frac_num: 0,
            next_idx: 0,
            buf,
            buf_start: -(params.taps as i64),
            pushed: 0,
            channels,
            scratch: vec![0.0f64; params.taps],
            finished: false,
        })
    }

    /// Build the oversampled, per-phase DC-normalized windowed-sinc table.
    fn build_table(taps: usize, oversampling: usize, cutoff: f64) -> Vec<f64> {
        let half = (taps / 2) as i64;
        let half_f = half as f64;
        let mut table = vec![0.0f64; (oversampling + 1) * taps];

        for p in 0..=oversampling {
            let frac = p as f64 / oversampling as f64;
            let row = &mut table[p * taps..(p + 1) * taps];
            let mut sum = 0.0f64;
            for (k, slot) in row.iter_mut().enumerate() {
                // Tap `k` reads input index `idx - half + 1 + k`; the kernel
                // argument is (tap position) − (continuous position `idx + frac`).
                let x = (k as i64 - half + 1) as f64 - frac;
                let v = cutoff * sinc(cutoff * x) * blackman_harris(x / half_f);
                *slot = v;
                sum += v;
            }
            // Normalize each phase row to exactly unit DC gain.
            if sum.abs() > f64::EPSILON {
                for slot in row.iter_mut() {
                    *slot /= sum;
                }
            }
        }

        table
    }

    /// Synthesize the fractional-delay filter for the current position into
    /// `self.scratch` by linear interpolation between adjacent phase rows.
    fn fill_scratch(&mut self) {
        let frac = self.frac_num as f64 / self.denom as f64;
        let pf = frac * self.oversampling as f64;
        let p0 = (pf as usize).min(self.oversampling - 1);
        let w = pf - p0 as f64;

        let row0 = &self.table[p0 * self.taps..(p0 + 1) * self.taps];
        let row1 = &self.table[(p0 + 1) * self.taps..(p0 + 2) * self.taps];
        for ((s, &c0), &c1) in self.scratch.iter_mut().zip(row0).zip(row1) {
            *s = c0 + w * (c1 - c0);
        }
    }

    /// Append input samples to the internal buffer.
    fn push_input(&mut self, input: &[Vec<f32>]) {
        let mut added = 0usize;
        for (buf_ch, in_ch) in self.buf.iter_mut().zip(input.iter()) {
            buf_ch.extend_from_slice(in_ch);
            added = in_ch.len();
        }
        self.pushed += added as i64;
    }

    /// Produce all output samples whose filter window lies fully inside the
    /// currently buffered input, stopping (exclusively) at input position
    /// `limit` when given.
    fn produce(&mut self, limit: Option<i64>) -> Vec<Vec<f32>> {
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); self.channels];
        let available_end = self.buf_start + self.buf.first().map_or(0, Vec::len) as i64;

        loop {
            let idx = self.next_idx;
            if let Some(end) = limit {
                if idx >= end {
                    break;
                }
            }
            // The filter reads absolute input indices [idx-half+1, idx+half].
            if idx + self.half >= available_end {
                break;
            }

            self.fill_scratch();
            let rel0 = (idx - self.half + 1 - self.buf_start) as usize;
            for (ch, out_ch) in out.iter_mut().enumerate() {
                let window = &self.buf[ch][rel0..rel0 + self.taps];
                let mut acc = 0.0f64;
                for (&s, &c) in window.iter().zip(self.scratch.iter()) {
                    acc += f64::from(s) * c;
                }
                #[allow(clippy::cast_possible_truncation)]
                out_ch.push(acc as f32);
            }

            // Advance the exact rational position.
            self.frac_num += self.step_num;
            self.next_idx += (self.frac_num / self.denom) as i64;
            self.frac_num %= self.denom;
        }

        self.trim_front();
        out
    }

    /// Discard buffered samples no longer reachable by future outputs.
    fn trim_front(&mut self) {
        let keep_from = self.next_idx - self.half + 1;
        let len = self.buf.first().map_or(0, Vec::len) as i64;
        let drain = (keep_from - self.buf_start).clamp(0, len);
        if drain > 0 {
            let drain = drain as usize;
            for ch in &mut self.buf {
                ch.drain(..drain);
            }
            self.buf_start += drain as i64;
        }
    }

    /// Process a block of planar input, returning all producible output.
    fn process(&mut self, input: &[Vec<f32>]) -> AudioResult<Vec<Vec<f32>>> {
        if self.finished {
            return Err(AudioError::InvalidParameter(
                "Resampler already flushed; call reset() before feeding more input".into(),
            ));
        }
        self.push_input(input);
        Ok(self.produce(None))
    }

    /// Flush the stream tail: emits every remaining output sample whose
    /// position lies before the end of the pushed input, padding the filter
    /// window with zeros. After flushing, `process` returns an error until
    /// `reset` is called.
    fn flush(&mut self) -> Vec<Vec<f32>> {
        if self.finished {
            return vec![Vec::new(); self.channels];
        }
        self.finished = true;
        let end = self.pushed;
        // Zero-pad so any window ending before `end + taps` is complete.
        for ch in &mut self.buf {
            let padded_len = ch.len() + self.taps;
            ch.resize(padded_len, 0.0);
        }
        self.produce(Some(end))
    }

    /// Reset to the initial (stream start) state.
    fn reset(&mut self) {
        for ch in &mut self.buf {
            ch.clear();
            ch.resize(self.taps, 0.0);
        }
        self.buf_start = -(self.taps as i64);
        self.frac_num = 0;
        self.next_idx = 0;
        self.pushed = 0;
        self.finished = false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public resampler
// ─────────────────────────────────────────────────────────────────────────────

/// Audio resampler.
///
/// Provides high-quality audio resampling using a Pure-Rust band-limited
/// windowed-sinc polyphase interpolator (see the module docs for details).
pub struct Resampler {
    /// Source sample rate.
    source_rate: u32,
    /// Target sample rate.
    target_rate: u32,
    /// Channel count.
    channels: usize,
    /// Quality setting.
    quality: ResamplerQuality,
    /// Resampling strategy.
    strategy: ResamplerStrategy,
    /// Resampling ratio.
    ratio: f64,
    /// Internal streaming engine (`None` for passthrough).
    engine: Option<SincEngine>,
    /// Format of the most recently resampled frame (used by `flush`).
    last_format: Option<(SampleFormat, ChannelLayout)>,
}

impl Resampler {
    /// Create a new resampler with specified quality.
    ///
    /// # Arguments
    ///
    /// * `source_rate` - Input sample rate in Hz
    /// * `target_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `quality` - Quality preset
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid or resampler construction fails.
    pub fn new(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        quality: ResamplerQuality,
    ) -> AudioResult<Self> {
        Self::with_max_buffering(source_rate, target_rate, channels, quality, 8192)
    }

    /// Create a new resampler with specified maximum input buffer size.
    ///
    /// # Arguments
    ///
    /// * `source_rate` - Input sample rate in Hz
    /// * `target_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `quality` - Quality preset
    /// * `max_input_frames` - Retained for API compatibility. The streaming
    ///   engine consumes input eagerly, so internal buffering never exceeds
    ///   the filter length plus one resampling step; this parameter is
    ///   currently advisory.
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid or resampler construction fails.
    pub fn with_max_buffering(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        quality: ResamplerQuality,
        _max_input_frames: usize,
    ) -> AudioResult<Self> {
        if source_rate == 0 || target_rate == 0 {
            return Err(AudioError::InvalidParameter(
                "Sample rate must be non-zero".into(),
            ));
        }
        if channels == 0 || channels > 32 {
            return Err(AudioError::InvalidParameter(
                "Channel count must be between 1 and 32".into(),
            ));
        }

        let ratio = f64::from(target_rate) / f64::from(source_rate);

        let (strategy, engine) = if source_rate == target_rate {
            (ResamplerStrategy::Passthrough, None)
        } else {
            let engine =
                SincEngine::new(source_rate, target_rate, channels, quality.sinc_params())?;
            (ResamplerStrategy::SincPolyphase, Some(engine))
        };

        Ok(Self {
            source_rate,
            target_rate,
            channels,
            quality,
            strategy,
            ratio,
            engine,
            last_format: None,
        })
    }

    /// Resample an audio frame.
    ///
    /// This method handles streaming resampling with internal buffering.
    /// It may return fewer or more samples than the input depending on
    /// the resampling ratio; call [`flush`](Self::flush) at end of stream to
    /// drain the final `taps / 2` input samples of latency.
    ///
    /// # Errors
    ///
    /// Returns error if resampling fails or format conversion fails.
    pub fn resample(&mut self, input: &AudioFrame) -> AudioResult<AudioFrame> {
        // Fast path for passthrough
        if self.strategy == ResamplerStrategy::Passthrough {
            return Ok(input.clone());
        }

        // Verify channel count matches
        if input.channels.count() != self.channels {
            return Err(AudioError::InvalidParameter(format!(
                "Channel count mismatch: expected {}, got {}",
                self.channels,
                input.channels.count()
            )));
        }

        self.last_format = Some((input.format, input.channels.clone()));

        // Convert input to f32 planar format for the sinc engine
        let input_planar = self.convert_to_f32_planar(input)?;

        // Process through resampler
        let output_planar = self.process_samples(&input_planar)?;

        // Convert back to original format
        self.convert_from_f32_planar(&output_planar, input.format, &input.channels)
    }

    /// Flush the stream tail.
    ///
    /// Emits the remaining output samples that could not be produced during
    /// streaming because their filter window extended past the last pushed
    /// input sample (the missing samples are treated as silence). After a
    /// flush, [`reset`](Self::reset) must be called before resampling again.
    ///
    /// The returned frame uses the format of the most recently resampled
    /// frame (F32 interleaved if no frame was ever resampled).
    ///
    /// # Errors
    ///
    /// Returns error if format conversion fails.
    pub fn flush(&mut self) -> AudioResult<AudioFrame> {
        let (format, layout) = self
            .last_format
            .clone()
            .unwrap_or((SampleFormat::F32, ChannelLayout::from_count(self.channels)));

        match &mut self.engine {
            None => {
                let mut frame = AudioFrame::new(format, self.target_rate, layout);
                frame.samples = AudioBuffer::Interleaved(Bytes::new());
                Ok(frame)
            }
            Some(engine) => {
                let output_planar = engine.flush();
                self.convert_from_f32_planar(&output_planar, format, &layout)
            }
        }
    }

    /// Process samples through the resampler engine.
    fn process_samples(&mut self, input: &[Vec<f32>]) -> AudioResult<Vec<Vec<f32>>> {
        match &mut self.engine {
            None => Ok(input.to_vec()),
            Some(engine) => engine.process(input),
        }
    }

    /// Convert audio frame to f32 planar format.
    fn convert_to_f32_planar(&self, frame: &AudioFrame) -> AudioResult<Vec<Vec<f32>>> {
        let sample_count = frame.sample_count();
        let mut planar = vec![vec![0.0f32; sample_count]; self.channels];

        match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                self.deinterleave_to_f32(data, frame.format, &mut planar)?;
            }
            AudioBuffer::Planar(planes) => {
                self.planes_to_f32(planes, frame.format, &mut planar)?;
            }
        }

        Ok(planar)
    }

    /// Deinterleave and convert interleaved samples to f32.
    #[allow(clippy::cast_precision_loss)]
    fn deinterleave_to_f32(
        &self,
        data: &[u8],
        format: SampleFormat,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        let sample_count = output[0].len();
        let bytes_per_sample = format.bytes_per_sample();

        for sample_idx in 0..sample_count {
            for ch in 0..self.channels {
                let offset = (sample_idx * self.channels + ch) * bytes_per_sample;
                let value = self.read_sample(data, offset, format)?;
                output[ch][sample_idx] = value;
            }
        }

        Ok(())
    }

    /// Convert planar samples to f32.
    fn planes_to_f32(
        &self,
        planes: &[Bytes],
        format: SampleFormat,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        let sample_count = output[0].len();
        let bytes_per_sample = format.bytes_per_sample();

        for ch in 0..self.channels {
            if ch >= planes.len() {
                return Err(AudioError::InvalidData(
                    "Insufficient planes for channel count".into(),
                ));
            }
            for sample_idx in 0..sample_count {
                let offset = sample_idx * bytes_per_sample;
                let value = self.read_sample(&planes[ch], offset, format)?;
                output[ch][sample_idx] = value;
            }
        }

        Ok(())
    }

    /// Read a single sample and convert to f32.
    #[allow(clippy::cast_precision_loss)]
    fn read_sample(&self, data: &[u8], offset: usize, format: SampleFormat) -> AudioResult<f32> {
        let bytes_per_sample = format.bytes_per_sample();
        if offset + bytes_per_sample > data.len() {
            return Err(AudioError::InvalidData(
                "Sample offset out of bounds".into(),
            ));
        }

        let value = match format {
            SampleFormat::U8 => (f32::from(data[offset]) - 128.0) / 128.0,
            SampleFormat::S16 | SampleFormat::S16p => {
                let bytes = [data[offset], data[offset + 1]];
                let sample = i16::from_le_bytes(bytes);
                sample as f32 / 32768.0
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let sample = i32::from_le_bytes(bytes);
                sample as f32 / 2_147_483_648.0
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                f32::from_le_bytes(bytes)
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ];
                #[allow(clippy::cast_possible_truncation)]
                let result = f64::from_le_bytes(bytes) as f32;
                result
            }
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "Unsupported sample format: {format}"
                )))
            }
        };

        Ok(value)
    }

    /// Convert f32 planar samples back to audio frame format.
    fn convert_from_f32_planar(
        &self,
        planar: &[Vec<f32>],
        format: SampleFormat,
        channels: &ChannelLayout,
    ) -> AudioResult<AudioFrame> {
        if planar.is_empty() || planar[0].is_empty() {
            let mut frame = AudioFrame::new(format, self.target_rate, channels.clone());
            frame.samples = AudioBuffer::Interleaved(Bytes::new());
            return Ok(frame);
        }

        let mut frame = AudioFrame::new(format, self.target_rate, channels.clone());

        if format.is_planar() {
            frame.samples = self.f32_to_planar_bytes(planar, format)?;
        } else {
            frame.samples = self.f32_to_interleaved_bytes(planar, format)?;
        }

        Ok(frame)
    }

    /// Convert f32 planar to interleaved bytes.
    fn f32_to_interleaved_bytes(
        &self,
        planar: &[Vec<f32>],
        format: SampleFormat,
    ) -> AudioResult<AudioBuffer> {
        let sample_count = planar[0].len();
        let bytes_per_sample = format.bytes_per_sample();
        let total_bytes = sample_count * self.channels * bytes_per_sample;
        let mut data = vec![0u8; total_bytes];

        for sample_idx in 0..sample_count {
            for ch in 0..self.channels {
                let offset = (sample_idx * self.channels + ch) * bytes_per_sample;
                let value = planar[ch][sample_idx];
                self.write_sample(&mut data, offset, value, format)?;
            }
        }

        Ok(AudioBuffer::Interleaved(Bytes::from(data)))
    }

    /// Convert f32 planar to planar bytes.
    fn f32_to_planar_bytes(
        &self,
        planar: &[Vec<f32>],
        format: SampleFormat,
    ) -> AudioResult<AudioBuffer> {
        let sample_count = planar[0].len();
        let bytes_per_sample = format.bytes_per_sample();
        let plane_size = sample_count * bytes_per_sample;
        let mut planes = Vec::with_capacity(self.channels);

        for ch in 0..self.channels {
            let mut plane_data = vec![0u8; plane_size];
            for sample_idx in 0..sample_count {
                let offset = sample_idx * bytes_per_sample;
                let value = planar[ch][sample_idx];
                self.write_sample(&mut plane_data, offset, value, format)?;
            }
            planes.push(Bytes::from(plane_data));
        }

        Ok(AudioBuffer::Planar(planes))
    }

    /// Write a single f32 sample in the target format.
    #[allow(clippy::cast_possible_truncation)]
    fn write_sample(
        &self,
        data: &mut [u8],
        offset: usize,
        value: f32,
        format: SampleFormat,
    ) -> AudioResult<()> {
        let bytes_per_sample = format.bytes_per_sample();
        if offset + bytes_per_sample > data.len() {
            return Err(AudioError::InvalidData(
                "Sample offset out of bounds".into(),
            ));
        }

        match format {
            SampleFormat::U8 => {
                let sample = ((value.clamp(-1.0, 1.0) * 128.0) + 128.0) as u8;
                data[offset] = sample;
            }
            SampleFormat::S16 | SampleFormat::S16p => {
                let sample = (value.clamp(-1.0, 1.0) * 32767.0) as i16;
                let bytes = sample.to_le_bytes();
                data[offset..offset + 2].copy_from_slice(&bytes);
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                let sample = (value.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32;
                let bytes = sample.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                let bytes = value.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                let bytes = f64::from(value).to_le_bytes();
                data[offset..offset + 8].copy_from_slice(&bytes);
            }
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "Unsupported sample format: {format}"
                )))
            }
        }

        Ok(())
    }

    /// Check if resampling is needed.
    #[must_use]
    pub fn is_passthrough(&self) -> bool {
        self.source_rate == self.target_rate
    }

    /// Get the resampling ratio.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Get output sample count for given input sample count.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn output_sample_count(&self, input_samples: usize) -> usize {
        ((input_samples as f64) * self.ratio).ceil() as usize
    }

    /// Reset the resampler state.
    ///
    /// Clears internal buffers and resets the resampler to its initial state.
    pub fn reset(&mut self) {
        if let Some(engine) = &mut self.engine {
            engine.reset();
        }
    }

    /// Get the source sample rate.
    #[must_use]
    pub const fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Get the target sample rate.
    #[must_use]
    pub const fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Get the number of channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Get the quality setting.
    #[must_use]
    pub const fn quality(&self) -> ResamplerQuality {
        self.quality
    }
}

/// Calculate greatest common divisor.
#[must_use]
const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioBuffer, ChannelLayout};
    use bytes::Bytes;
    use oximedia_core::SampleFormat;

    fn mono_f32_frame(n: usize, value: f32, sample_rate: u32) -> AudioFrame {
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::F32, sample_rate, ChannelLayout::Mono);
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
        frame
    }

    /// Build a mono F32 interleaved frame from a sample slice.
    fn mono_frame_from(samples: &[f32], sample_rate: u32) -> AudioFrame {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::F32, sample_rate, ChannelLayout::Mono);
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));
        frame
    }

    /// Extract mono F32 samples from an interleaved frame.
    fn mono_samples(frame: &AudioFrame) -> Vec<f32> {
        match &frame.samples {
            AudioBuffer::Interleaved(data) => data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            AudioBuffer::Planar(planes) => planes[0]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }

    /// Generate `n` samples of a sine at `freq` Hz sampled at `rate` Hz.
    fn sine(freq: f64, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin()) as f32
            })
            .collect()
    }

    /// Least-squares fit of `a·sin + b·cos` at a known frequency; returns
    /// the SNR in dB of the fit over `y[skip..len-skip]`.
    fn sine_fit_snr_db(y: &[f32], freq: f64, rate: u32, skip: usize) -> f64 {
        assert!(y.len() > 2 * skip + 16, "signal too short for SNR fit");
        let seg = &y[skip..y.len() - skip];
        let w = 2.0 * std::f64::consts::PI * freq / f64::from(rate);

        let (mut s_ss, mut s_cc, mut s_sc) = (0.0f64, 0.0f64, 0.0f64);
        let (mut r_s, mut r_c) = (0.0f64, 0.0f64);
        for (i, &yv) in seg.iter().enumerate() {
            let t = (skip + i) as f64;
            let s = (w * t).sin();
            let c = (w * t).cos();
            let yf = f64::from(yv);
            s_ss += s * s;
            s_cc += c * c;
            s_sc += s * c;
            r_s += yf * s;
            r_c += yf * c;
        }
        let det = s_ss * s_cc - s_sc * s_sc;
        assert!(det.abs() > 1e-9, "degenerate sine fit");
        let a = (r_s * s_cc - r_c * s_sc) / det;
        let b = (r_c * s_ss - r_s * s_sc) / det;

        let (mut sig_pow, mut noise_pow) = (0.0f64, 0.0f64);
        for (i, &yv) in seg.iter().enumerate() {
            let t = (skip + i) as f64;
            let fit = a * (w * t).sin() + b * (w * t).cos();
            sig_pow += fit * fit;
            let e = f64::from(yv) - fit;
            noise_pow += e * e;
        }
        assert!(noise_pow > 0.0, "unexpected exact fit");
        10.0 * (sig_pow / noise_pow).log10()
    }

    /// Estimate frequency from positive-going zero crossings (sub-sample
    /// accurate via linear interpolation).
    fn zero_crossing_freq(y: &[f32], rate: u32, skip: usize) -> f64 {
        let seg = &y[skip..y.len() - skip];
        let mut crossings: Vec<f64> = Vec::new();
        for (i, pair) in seg.windows(2).enumerate() {
            if pair[0] < 0.0 && pair[1] >= 0.0 {
                let denom = f64::from(pair[1]) - f64::from(pair[0]);
                let frac = if denom.abs() > 1e-12 {
                    -f64::from(pair[0]) / denom
                } else {
                    0.5
                };
                crossings.push(i as f64 + frac);
            }
        }
        assert!(crossings.len() >= 2, "not enough zero crossings");
        let first = crossings[0];
        let last = crossings[crossings.len() - 1];
        (crossings.len() - 1) as f64 * f64::from(rate) / (last - first)
    }

    // ── Quality preset tests ─────────────────────────────────────────────────

    #[test]
    fn test_draft_is_alias_for_low() {
        assert_eq!(ResamplerQuality::Draft.canonical(), ResamplerQuality::Low);
    }

    #[test]
    fn test_good_is_alias_for_high() {
        assert_eq!(ResamplerQuality::Good.canonical(), ResamplerQuality::High);
    }

    #[test]
    fn test_best_canonical_is_best() {
        assert_eq!(ResamplerQuality::Best.canonical(), ResamplerQuality::Best);
    }

    #[test]
    fn test_medium_canonical_is_medium() {
        assert_eq!(
            ResamplerQuality::Medium.canonical(),
            ResamplerQuality::Medium
        );
    }

    #[test]
    fn test_quality_taps_monotonic() {
        let low = ResamplerQuality::Low.sinc_params().taps;
        let medium = ResamplerQuality::Medium.sinc_params().taps;
        let high = ResamplerQuality::High.sinc_params().taps;
        let best = ResamplerQuality::Best.sinc_params().taps;
        assert!(low <= medium, "Low ({low}) should be <= Medium ({medium})");
        assert!(
            medium <= high,
            "Medium ({medium}) should be <= High ({high})"
        );
        assert!(high <= best, "High ({high}) should be <= Best ({best})");
    }

    #[test]
    fn test_draft_params_same_as_low() {
        assert_eq!(
            ResamplerQuality::Draft.sinc_params(),
            ResamplerQuality::Low.sinc_params(),
        );
    }

    #[test]
    fn test_good_params_same_as_high() {
        assert_eq!(
            ResamplerQuality::Good.sinc_params(),
            ResamplerQuality::High.sinc_params(),
        );
    }

    // ── Construction / accessor tests ────────────────────────────────────────

    #[test]
    fn test_resampler_passthrough_same_rate() {
        let r = Resampler::new(48_000, 48_000, 1, ResamplerQuality::Medium)
            .expect("passthrough resampler");
        assert!(r.is_passthrough());
        assert_eq!(r.ratio(), 1.0);
    }

    #[test]
    fn test_resampler_upsample_ratio() {
        let r =
            Resampler::new(44_100, 48_000, 1, ResamplerQuality::Low).expect("upsample resampler");
        assert!(!r.is_passthrough());
        let expected = 48_000.0 / 44_100.0;
        assert!((r.ratio() - expected).abs() < 1e-6, "ratio mismatch");
    }

    #[test]
    fn test_resampler_construction_all_qualities() {
        for q in [
            ResamplerQuality::Low,
            ResamplerQuality::Medium,
            ResamplerQuality::High,
            ResamplerQuality::Best,
            ResamplerQuality::Draft,
            ResamplerQuality::Good,
        ] {
            let r = Resampler::new(44_100, 48_000, 2, q);
            assert!(
                r.is_ok(),
                "{q:?} resampler construction failed: {:?}",
                r.err()
            );
        }
    }

    #[test]
    fn test_resampler_passthrough_returns_correct_sample_rate() {
        let mut r =
            Resampler::new(48_000, 48_000, 1, ResamplerQuality::Medium).expect("passthrough");
        let frame = mono_f32_frame(512, 0.5, 48_000);
        let out = r.resample(&frame).expect("passthrough resample");
        assert_eq!(out.sample_rate, 48_000);
        assert_eq!(out.format, SampleFormat::F32);
    }

    #[test]
    fn test_resampler_source_target_rate_accessors() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Low).expect("resampler");
        assert_eq!(r.source_rate(), 44_100);
        assert_eq!(r.target_rate(), 48_000);
        assert_eq!(r.channels(), 1);
    }

    #[test]
    fn test_resampler_output_sample_count_upsample() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Low).expect("resampler");
        let out_count = r.output_sample_count(441);
        assert!(out_count >= 480, "expected >= 480, got {out_count}");
        assert!(out_count <= 483, "expected <= 483, got {out_count}");
    }

    #[test]
    fn test_resampler_error_on_zero_sample_rate() {
        let r = Resampler::new(0, 48_000, 1, ResamplerQuality::Low);
        assert!(r.is_err(), "zero source rate should fail");
    }

    #[test]
    fn test_resampler_error_on_zero_channels() {
        let r = Resampler::new(44_100, 48_000, 0, ResamplerQuality::Low);
        assert!(r.is_err(), "zero channels should fail");
    }

    #[test]
    fn test_resampler_quality_accessor() {
        let r = Resampler::new(44_100, 48_000, 1, ResamplerQuality::Best).expect("resampler");
        assert_eq!(r.quality(), ResamplerQuality::Best);
    }

    #[test]
    fn test_channel_count_mismatch_rejected() {
        let mut r = Resampler::new(48_000, 44_100, 2, ResamplerQuality::Low).expect("resampler");
        let frame = mono_f32_frame(256, 0.1, 48_000);
        assert!(r.resample(&frame).is_err(), "channel mismatch should fail");
    }

    // ── Kernel sanity tests ──────────────────────────────────────────────────

    #[test]
    fn test_kernel_rows_are_dc_normalized() {
        let taps = 64;
        let p = 32;
        let table = SincEngine::build_table(taps, p, 0.9);
        for row_idx in 0..=p {
            let row = &table[row_idx * taps..(row_idx + 1) * taps];
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "phase {row_idx} DC gain {sum} != 1"
            );
        }
    }

    #[test]
    fn test_kernel_phase_zero_is_near_identity_at_full_cutoff() {
        // With cutoff near 1 and frac = 0 the kernel approaches a discrete
        // delta: the center tap dominates.
        let taps = 64;
        let table = SincEngine::build_table(taps, 8, 0.999);
        let row0 = &table[..taps];
        let center = taps / 2 - 1; // tap index where x = 0 (k - half + 1 = 0)
        assert!(
            row0[center] > 0.9,
            "center coefficient {} should dominate",
            row0[center]
        );
    }

    // ── DSP correctness tests ────────────────────────────────────────────────

    /// Helper: run a full stream (process + flush) through a fresh resampler.
    fn run_stream(
        input: &[f32],
        source_rate: u32,
        target_rate: u32,
        quality: ResamplerQuality,
    ) -> Vec<f32> {
        let mut r =
            Resampler::new(source_rate, target_rate, 1, quality).expect("resampler construction");
        let frame = mono_frame_from(input, source_rate);
        let out = r.resample(&frame).expect("resample");
        let mut samples = mono_samples(&out);
        let tail = r.flush().expect("flush");
        samples.extend(mono_samples(&tail));
        samples
    }

    #[test]
    fn test_sine_1khz_48k_to_44k1_frequency_preserved() {
        let input = sine(1000.0, 48_000, 48_000);
        let out = run_stream(&input, 48_000, 44_100, ResamplerQuality::High);

        // Output length must match the rational ratio (±1 sample).
        let expected_len = 48_000.0 * 44_100.0 / 48_000.0;
        assert!(
            (out.len() as f64 - expected_len).abs() <= 1.0,
            "output length {} != expected {expected_len}",
            out.len()
        );

        let freq = zero_crossing_freq(&out, 44_100, 2000);
        assert!(
            (freq - 1000.0).abs() < 1.0,
            "frequency {freq} Hz deviates from 1000 Hz"
        );
    }

    #[test]
    fn test_sine_1khz_48k_to_44k1_snr_high_quality() {
        let input = sine(1000.0, 48_000, 48_000);
        let out = run_stream(&input, 48_000, 44_100, ResamplerQuality::High);
        let snr = sine_fit_snr_db(&out, 1000.0, 44_100, 2400);
        assert!(snr > 70.0, "High quality SNR {snr:.1} dB <= 70 dB");
    }

    #[test]
    fn test_sine_1khz_48k_to_44k1_snr_best_quality() {
        let input = sine(1000.0, 48_000, 48_000);
        let out = run_stream(&input, 48_000, 44_100, ResamplerQuality::Best);
        let snr = sine_fit_snr_db(&out, 1000.0, 44_100, 2400);
        assert!(snr > 75.0, "Best quality SNR {snr:.1} dB <= 75 dB");
    }

    #[test]
    fn test_sine_upsample_44k1_to_48k_snr() {
        let input = sine(1000.0, 44_100, 44_100);
        let out = run_stream(&input, 44_100, 48_000, ResamplerQuality::High);
        let snr = sine_fit_snr_db(&out, 1000.0, 48_000, 2400);
        assert!(snr > 70.0, "upsample SNR {snr:.1} dB <= 70 dB");
    }

    #[test]
    fn test_sine_low_quality_still_reasonable() {
        let input = sine(1000.0, 48_000, 48_000);
        let out = run_stream(&input, 48_000, 44_100, ResamplerQuality::Low);
        let snr = sine_fit_snr_db(&out, 1000.0, 44_100, 2400);
        assert!(snr > 50.0, "Low quality SNR {snr:.1} dB <= 50 dB");
    }

    #[test]
    fn test_round_trip_48k_44k1_48k_bounded() {
        let n = 48_000;
        let input = sine(1000.0, 48_000, n);
        let mid = run_stream(&input, 48_000, 44_100, ResamplerQuality::High);
        let back = run_stream(&mid, 44_100, 48_000, ResamplerQuality::High);

        // The round trip is time-aligned by construction: back[i] ≈ input[i].
        let usable = back.len().min(input.len());
        assert!(
            usable as f64 >= n as f64 * 0.99,
            "round trip lost too many samples: {usable} of {n}"
        );

        let skip = 2400;
        let (mut sig_pow, mut err_pow) = (0.0f64, 0.0f64);
        let mut max_err = 0.0f64;
        for i in skip..usable - skip {
            let s = f64::from(input[i]);
            let e = f64::from(back[i]) - s;
            sig_pow += s * s;
            err_pow += e * e;
            max_err = max_err.max(e.abs());
        }
        assert!(err_pow > 0.0, "unexpected exact round trip");
        let snr = 10.0 * (sig_pow / err_pow).log10();
        assert!(snr > 55.0, "round-trip SNR {snr:.1} dB <= 55 dB");
        assert!(max_err < 0.05, "round-trip max error {max_err} >= 0.05");
    }

    #[test]
    fn test_chunked_processing_bit_identical_to_one_shot() {
        let input = sine(440.0, 48_000, 20_000);
        let whole = run_stream(&input, 48_000, 44_100, ResamplerQuality::Medium);

        let mut r = Resampler::new(48_000, 44_100, 1, ResamplerQuality::Medium).expect("resampler");
        let mut chunked: Vec<f32> = Vec::new();
        let chunk_sizes = [480usize, 313, 1024, 7, 4096, 999];
        let mut pos = 0usize;
        let mut cs_idx = 0usize;
        while pos < input.len() {
            let len = chunk_sizes[cs_idx % chunk_sizes.len()].min(input.len() - pos);
            let frame = mono_frame_from(&input[pos..pos + len], 48_000);
            let out = r.resample(&frame).expect("chunked resample");
            chunked.extend(mono_samples(&out));
            pos += len;
            cs_idx += 1;
        }
        let tail = r.flush().expect("flush");
        chunked.extend(mono_samples(&tail));

        assert_eq!(
            whole.len(),
            chunked.len(),
            "chunked output length differs from one-shot"
        );
        for (i, (a, b)) in whole.iter().zip(chunked.iter()).enumerate() {
            assert!(a.to_bits() == b.to_bits(), "sample {i} differs: {a} vs {b}");
        }
    }

    #[test]
    fn test_stereo_channels_resampled_independently() {
        let n = 24_000;
        let left = sine(997.0, 48_000, n);
        let right = sine(1499.0, 48_000, n);

        let mut bytes = Vec::with_capacity(n * 8);
        for i in 0..n {
            bytes.extend_from_slice(&left[i].to_le_bytes());
            bytes.extend_from_slice(&right[i].to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::F32, 48_000, ChannelLayout::Stereo);
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));

        let mut r = Resampler::new(48_000, 44_100, 2, ResamplerQuality::High).expect("resampler");
        let out = r.resample(&frame).expect("stereo resample");
        let tail = r.flush().expect("flush");

        let extract = |f: &AudioFrame, ch: usize| -> Vec<f32> {
            match &f.samples {
                AudioBuffer::Interleaved(data) => data
                    .chunks_exact(8)
                    .map(|c| {
                        let off = ch * 4;
                        f32::from_le_bytes([c[off], c[off + 1], c[off + 2], c[off + 3]])
                    })
                    .collect(),
                AudioBuffer::Planar(_) => Vec::new(),
            }
        };
        let mut out_l = extract(&out, 0);
        out_l.extend(extract(&tail, 0));
        let mut out_r = extract(&out, 1);
        out_r.extend(extract(&tail, 1));

        let snr_l = sine_fit_snr_db(&out_l, 997.0, 44_100, 2400);
        let snr_r = sine_fit_snr_db(&out_r, 1499.0, 44_100, 2400);
        assert!(snr_l > 65.0, "left channel SNR {snr_l:.1} dB <= 65 dB");
        assert!(snr_r > 65.0, "right channel SNR {snr_r:.1} dB <= 65 dB");
    }

    #[test]
    fn test_dc_signal_preserved() {
        // Per-phase DC normalization must carry a constant through exactly.
        let input = vec![0.5f32; 20_000];
        let out = run_stream(&input, 48_000, 44_100, ResamplerQuality::Medium);
        let skip = 1000;
        for (i, &s) in out[skip..out.len() - skip].iter().enumerate() {
            assert!((s - 0.5).abs() < 1e-4, "DC sample {i} drifted: {s} != 0.5");
        }
    }

    #[test]
    fn test_extreme_downsample_produces_output() {
        let input = sine(500.0, 192_000, 192_000 / 4);
        let out = run_stream(&input, 192_000, 8_000, ResamplerQuality::Best);
        let expected = (192_000.0 / 4.0) * 8_000.0 / 192_000.0;
        assert!(
            (out.len() as f64 - expected).abs() <= 1.0,
            "extreme downsample length {} != {expected}",
            out.len()
        );
        let snr = sine_fit_snr_db(&out, 500.0, 8_000, 200);
        assert!(snr > 40.0, "extreme downsample SNR {snr:.1} dB <= 40 dB");
    }

    #[test]
    fn test_process_after_flush_errors_and_reset_recovers() {
        let mut r = Resampler::new(48_000, 44_100, 1, ResamplerQuality::Low).expect("resampler");
        let frame = mono_frame_from(&sine(1000.0, 48_000, 4800), 48_000);
        r.resample(&frame).expect("resample");
        r.flush().expect("flush");
        assert!(
            r.resample(&frame).is_err(),
            "resample after flush should error"
        );
        r.reset();
        assert!(
            r.resample(&frame).is_ok(),
            "resample after reset should succeed"
        );
    }

    #[test]
    fn test_flush_without_input_is_empty() {
        let mut r = Resampler::new(48_000, 44_100, 1, ResamplerQuality::Low).expect("resampler");
        let out = r.flush().expect("flush");
        assert_eq!(out.sample_count(), 0, "flush without input must be empty");
    }

    #[test]
    fn test_s16_format_round_trip_smoke() {
        let n = 9600;
        let mut bytes = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = (i as f64 * 2.0 * std::f64::consts::PI * 1000.0 / 48_000.0).sin();
            #[allow(clippy::cast_possible_truncation)]
            let s = (v * 30_000.0) as i16;
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut frame = AudioFrame::new(SampleFormat::S16, 48_000, ChannelLayout::Mono);
        frame.samples = AudioBuffer::Interleaved(Bytes::from(bytes));

        let mut r = Resampler::new(48_000, 44_100, 1, ResamplerQuality::Medium).expect("resampler");
        let out = r.resample(&frame).expect("S16 resample");
        assert_eq!(out.format, SampleFormat::S16);
        assert_eq!(out.sample_rate, 44_100);
        assert!(out.sample_count() > 0, "S16 resample produced no output");
    }
}

/// Common sample rate constants.
pub mod sample_rates {
    /// 8 kHz - Telephone quality.
    pub const RATE_8000: u32 = 8000;
    /// 11.025 kHz - Low quality audio.
    pub const RATE_11025: u32 = 11025;
    /// 16 kHz - Wideband speech.
    pub const RATE_16000: u32 = 16000;
    /// 22.05 kHz - Quarter of CD quality.
    pub const RATE_22050: u32 = 22050;
    /// 32 kHz - Digital radio.
    pub const RATE_32000: u32 = 32000;
    /// 44.1 kHz - CD quality.
    pub const RATE_44100: u32 = 44100;
    /// 48 kHz - Professional audio, DVD.
    pub const RATE_48000: u32 = 48000;
    /// 88.2 kHz - High-resolution audio (2x CD).
    pub const RATE_88200: u32 = 88200;
    /// 96 kHz - High-resolution audio, Blu-ray.
    pub const RATE_96000: u32 = 96000;
    /// 176.4 kHz - Ultra high-resolution (4x CD).
    pub const RATE_176400: u32 = 176400;
    /// 192 kHz - Ultra high-resolution.
    pub const RATE_192000: u32 = 192000;
}
