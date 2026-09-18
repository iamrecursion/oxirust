//! Offline bounce/render engine for the mixer.
//!
//! The [`OfflineBouncer`](crate::bounce::OfflineBouncer) processes the entire mixer graph faster-than-realtime,
//! writing the output to a buffer.  A progress callback is invoked periodically
//! so the caller can update a UI or cancel the operation.

use crate::{AudioMixer, MixerResult};
use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
use oximedia_core::SampleFormat;

// ---------------------------------------------------------------------------
// Progress callback
// ---------------------------------------------------------------------------

/// Progress information passed to the bounce callback.
#[derive(Debug, Clone, Copy)]
pub struct BounceProgress {
    /// Number of samples rendered so far.
    pub rendered_samples: u64,
    /// Total number of samples to render.
    pub total_samples: u64,
    /// Progress as a fraction (0.0..1.0).
    pub fraction: f64,
}

// ---------------------------------------------------------------------------
// OfflineBouncer
// ---------------------------------------------------------------------------

/// Offline bounce/render engine that processes the mixer graph without realtime
/// constraints.
///
/// # Usage
///
/// ```ignore
/// let mut bouncer = OfflineBouncer::new(sample_rate, buffer_size);
/// let result = bouncer.bounce(
///     &mut mixer,
///     &input_frames,
///     |progress| {
///         println!("Progress: {:.1}%", progress.fraction * 100.0);
///         true // return false to cancel
///     },
/// )?;
/// ```
pub struct OfflineBouncer {
    /// Sample rate.
    sample_rate: u32,
    /// Processing buffer size.
    buffer_size: usize,
    /// How often to invoke the progress callback (in blocks).
    progress_interval_blocks: usize,
}

impl OfflineBouncer {
    /// Create a new offline bouncer.
    #[must_use]
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        Self {
            sample_rate,
            buffer_size: buffer_size.max(1),
            progress_interval_blocks: 10,
        }
    }

    /// Set the progress callback interval (in blocks).
    pub fn set_progress_interval(&mut self, blocks: usize) {
        self.progress_interval_blocks = blocks.max(1);
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Bounce (render) the mixer output for a given sequence of input frames.
    ///
    /// Each input frame is processed through the mixer and the output is
    /// accumulated into the result buffer.
    ///
    /// The `progress_cb` is called periodically with a [`BounceProgress`]
    /// struct. Return `true` to continue, or `false` to cancel the bounce.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ProcessingError` if the mixer fails to process
    /// a frame or if the bounce is cancelled.
    pub fn bounce(
        &self,
        mixer: &mut AudioMixer,
        input_frames: &[AudioFrame],
        mut progress_cb: impl FnMut(BounceProgress) -> bool,
    ) -> MixerResult<BounceResult> {
        let total_frames = input_frames.len();
        #[allow(clippy::cast_precision_loss)]
        let total_samples = (total_frames as u64) * (self.buffer_size as u64);

        let mut output_left = Vec::with_capacity(total_samples as usize);
        let mut output_right = Vec::with_capacity(total_samples as usize);

        for (block_idx, frame) in input_frames.iter().enumerate() {
            // Process through mixer
            let output_frame = mixer.process(frame)?;

            // Extract output samples
            let raw = match &output_frame.samples {
                AudioBuffer::Interleaved(data) => data.as_ref().to_vec(),
                AudioBuffer::Planar(planes) => {
                    if let Some(first) = planes.first() {
                        first.as_ref().to_vec()
                    } else {
                        vec![0u8; self.buffer_size * 2 * 4]
                    }
                }
            };

            // Decode interleaved stereo f32
            let num_stereo_samples = raw.len() / 8; // 2 channels * 4 bytes
            for i in 0..num_stereo_samples {
                let l_offset = i * 8;
                let r_offset = i * 8 + 4;
                if r_offset + 4 <= raw.len() {
                    let l = f32::from_le_bytes([
                        raw[l_offset],
                        raw[l_offset + 1],
                        raw[l_offset + 2],
                        raw[l_offset + 3],
                    ]);
                    let r = f32::from_le_bytes([
                        raw[r_offset],
                        raw[r_offset + 1],
                        raw[r_offset + 2],
                        raw[r_offset + 3],
                    ]);
                    output_left.push(l);
                    output_right.push(r);
                }
            }

            // Progress callback
            if self.progress_interval_blocks > 0
                && (block_idx + 1) % self.progress_interval_blocks == 0
            {
                #[allow(clippy::cast_precision_loss)]
                let rendered = ((block_idx + 1) as u64) * (self.buffer_size as u64);
                #[allow(clippy::cast_precision_loss)]
                let fraction = if total_samples > 0 {
                    rendered as f64 / total_samples as f64
                } else {
                    1.0
                };

                let progress = BounceProgress {
                    rendered_samples: rendered,
                    total_samples,
                    fraction,
                };

                if !progress_cb(progress) {
                    return Err(crate::MixerError::ProcessingError(
                        "Bounce cancelled by user".into(),
                    ));
                }
            }
        }

        // Final progress callback at 100%
        let final_progress = BounceProgress {
            rendered_samples: total_samples,
            total_samples,
            fraction: 1.0,
        };
        let _ = progress_cb(final_progress);

        Ok(BounceResult {
            left: output_left,
            right: output_right,
            sample_rate: self.sample_rate,
            total_samples,
        })
    }

    /// Bounce from a mono signal buffer, splitting it into blocks and feeding
    /// the mixer.
    ///
    /// This is a convenience method when you have raw samples instead of
    /// pre-framed data.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ProcessingError` if the mixer fails.
    pub fn bounce_from_samples(
        &self,
        mixer: &mut AudioMixer,
        samples: &[f32],
        progress_cb: impl FnMut(BounceProgress) -> bool,
    ) -> MixerResult<BounceResult> {
        // Create input frames from raw samples
        let mut frames = Vec::new();
        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + self.buffer_size).min(samples.len());
            let block = &samples[offset..end];

            let mut frame =
                AudioFrame::new(SampleFormat::F32, self.sample_rate, ChannelLayout::Mono);
            let mut raw_bytes = Vec::with_capacity(block.len() * 4);
            for &s in block {
                raw_bytes.extend_from_slice(&s.to_le_bytes());
            }
            // Pad to full buffer size if needed
            let remaining = self.buffer_size - block.len();
            for _ in 0..remaining {
                raw_bytes.extend_from_slice(&0.0_f32.to_le_bytes());
            }
            frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(raw_bytes));
            frames.push(frame);
            offset = end;
        }

        self.bounce(mixer, &frames, progress_cb)
    }
}

impl std::fmt::Debug for OfflineBouncer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineBouncer")
            .field("sample_rate", &self.sample_rate)
            .field("buffer_size", &self.buffer_size)
            .field("progress_interval_blocks", &self.progress_interval_blocks)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// BounceResult
// ---------------------------------------------------------------------------

/// Result of an offline bounce operation.
#[derive(Debug, Clone)]
pub struct BounceResult {
    /// Left channel output samples.
    pub left: Vec<f32>,
    /// Right channel output samples.
    pub right: Vec<f32>,
    /// Sample rate.
    pub sample_rate: u32,
    /// Total samples rendered.
    pub total_samples: u64,
}

impl BounceResult {
    /// Get the duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.left.len() as f64 / f64::from(self.sample_rate)
        }
    }

    /// Get the number of output samples (per channel).
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.left.len()
    }

    /// Convert to an interleaved stereo `AudioFrame`.
    #[must_use]
    pub fn to_audio_frame(&self) -> AudioFrame {
        let mut frame = AudioFrame::new(SampleFormat::F32, self.sample_rate, ChannelLayout::Stereo);

        let n = self.left.len().min(self.right.len());
        let mut raw_bytes = Vec::with_capacity(n * 2 * 4);
        for i in 0..n {
            raw_bytes.extend_from_slice(&self.left[i].to_le_bytes());
            raw_bytes.extend_from_slice(&self.right[i].to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(raw_bytes));
        frame
    }

    /// Compute the peak amplitude across both channels.
    #[must_use]
    pub fn peak_amplitude(&self) -> f32 {
        let l_peak = self.left.iter().fold(0.0_f32, |acc, &s| acc.max(s.abs()));
        let r_peak = self.right.iter().fold(0.0_f32, |acc, &s| acc.max(s.abs()));
        l_peak.max(r_peak)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioMixer, ChannelType, MixerConfig};

    fn make_input_frame(sample_rate: u32, buffer_size: usize, value: f32) -> AudioFrame {
        let mut frame = AudioFrame::new(SampleFormat::F32, sample_rate, ChannelLayout::Mono);
        let mut raw = Vec::with_capacity(buffer_size * 4);
        for _ in 0..buffer_size {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(raw));
        frame
    }

    #[test]
    fn test_bouncer_creation() {
        let bouncer = OfflineBouncer::new(48000, 512);
        assert_eq!(bouncer.sample_rate(), 48000);
        assert_eq!(bouncer.buffer_size(), 512);
    }

    #[test]
    fn test_bounce_empty_input() {
        let bouncer = OfflineBouncer::new(48000, 512);
        let mut mixer = AudioMixer::new(MixerConfig::default());
        let result = bouncer
            .bounce(&mut mixer, &[], |_| true)
            .expect("bounce should succeed");
        assert_eq!(result.sample_count(), 0);
    }

    #[test]
    fn test_bounce_with_channel() {
        let config = MixerConfig {
            sample_rate: 48000,
            buffer_size: 64,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);
        let _ch = mixer
            .add_channel("Test".to_string(), ChannelType::Mono, ChannelLayout::Mono)
            .expect("add channel should succeed");

        let frames: Vec<AudioFrame> = (0..10).map(|_| make_input_frame(48000, 64, 0.5)).collect();

        let bouncer = OfflineBouncer::new(48000, 64);
        let result = bouncer
            .bounce(&mut mixer, &frames, |_| true)
            .expect("bounce should succeed");

        assert!(result.sample_count() > 0);
        assert_eq!(result.sample_rate, 48000);
    }

    #[test]
    fn test_bounce_cancelled() {
        let config = MixerConfig {
            sample_rate: 48000,
            buffer_size: 64,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);

        let frames: Vec<AudioFrame> = (0..100).map(|_| make_input_frame(48000, 64, 0.5)).collect();

        let mut bouncer = OfflineBouncer::new(48000, 64);
        bouncer.set_progress_interval(1);

        let mut calls = 0;
        let result = bouncer.bounce(&mut mixer, &frames, |_progress| {
            calls += 1;
            calls < 3 // Cancel after 3 callbacks
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_bounce_progress_callback() {
        let config = MixerConfig {
            sample_rate: 48000,
            buffer_size: 64,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);

        let frames: Vec<AudioFrame> = (0..20).map(|_| make_input_frame(48000, 64, 0.1)).collect();

        let mut bouncer = OfflineBouncer::new(48000, 64);
        bouncer.set_progress_interval(5);

        let mut progress_values = Vec::new();
        let result = bouncer
            .bounce(&mut mixer, &frames, |p| {
                progress_values.push(p.fraction);
                true
            })
            .expect("bounce should succeed");

        assert!(!progress_values.is_empty());
        // Last progress should be 1.0
        let last = progress_values.last().copied().unwrap_or(0.0);
        assert!(
            (last - 1.0).abs() < 0.01,
            "last progress should be 1.0, got {last}"
        );
        assert!(result.sample_count() > 0);
    }

    #[test]
    fn test_bounce_from_samples() {
        let config = MixerConfig {
            sample_rate: 48000,
            buffer_size: 64,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);

        let samples = vec![0.3_f32; 256];
        let bouncer = OfflineBouncer::new(48000, 64);
        let result = bouncer
            .bounce_from_samples(&mut mixer, &samples, |_| true)
            .expect("bounce_from_samples should succeed");

        assert!(result.sample_count() > 0);
    }

    #[test]
    fn test_bounce_result_duration() {
        let result = BounceResult {
            left: vec![0.0; 48000],
            right: vec![0.0; 48000],
            sample_rate: 48000,
            total_samples: 48000,
        };
        assert!((result.duration_seconds() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bounce_result_peak() {
        let result = BounceResult {
            left: vec![0.5, -0.8, 0.3],
            right: vec![0.1, 0.9, -0.2],
            sample_rate: 48000,
            total_samples: 3,
        };
        assert!((result.peak_amplitude() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounce_result_to_frame() {
        let result = BounceResult {
            left: vec![0.5, 0.3],
            right: vec![0.2, 0.4],
            sample_rate: 48000,
            total_samples: 2,
        };
        let frame = result.to_audio_frame();
        assert_eq!(frame.sample_rate, 48000);
    }

    #[test]
    fn test_bounce_result_zero_sample_rate() {
        let result = BounceResult {
            left: vec![],
            right: vec![],
            sample_rate: 0,
            total_samples: 0,
        };
        assert!((result.duration_seconds() - 0.0).abs() < f32::EPSILON as f64);
    }
}
