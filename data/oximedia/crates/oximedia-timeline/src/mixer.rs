//! Multi-track audio/video mixer.
//!
//! The `TrackMixer` combines multiple audio and video tracks at a given
//! timeline position, applying per-track volume, pan, and opacity settings.

use crate::error::{TimelineError, TimelineResult};
use crate::renderer::PixelBuffer;
use crate::timeline::Timeline;
use crate::types::Position;
use serde::{Deserialize, Serialize};

/// A single frame worth of mixed audio samples.
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Mixed samples per channel (float, -1.0 to 1.0).
    pub samples: Vec<Vec<f32>>,
    /// Number of audio channels.
    pub channels: usize,
    /// Samples per channel.
    pub samples_per_frame: usize,
}

impl AudioFrame {
    /// Create a silent audio frame.
    #[must_use]
    pub fn silent(channels: usize, samples_per_frame: usize) -> Self {
        Self {
            samples: vec![vec![0.0f32; samples_per_frame]; channels],
            channels,
            samples_per_frame,
        }
    }

    /// Get peak amplitude across all channels.
    #[must_use]
    pub fn peak(&self) -> f32 {
        self.samples
            .iter()
            .flat_map(|ch| ch.iter())
            .map(|s| s.abs())
            .fold(0.0f32, f32::max)
    }

    /// Get RMS (root mean square) across all channels.
    #[must_use]
    pub fn rms(&self) -> f32 {
        let total_samples: usize = self.channels * self.samples_per_frame;
        if total_samples == 0 {
            return 0.0;
        }
        let sum_sq: f32 = self
            .samples
            .iter()
            .flat_map(|ch| ch.iter())
            .map(|s| s * s)
            .sum();
        (sum_sq / total_samples as f32).sqrt()
    }
}

/// Per-track mix parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMixParams {
    /// Volume multiplier (0.0 = silence, 1.0 = unity, >1.0 = boost).
    pub volume: f32,
    /// Pan position (-1.0 = full left, 0.0 = center, 1.0 = full right).
    pub pan: f32,
    /// Video opacity (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,
    /// Whether this track contributes to the mix.
    pub active: bool,
}

impl Default for TrackMixParams {
    fn default() -> Self {
        Self {
            volume: 1.0,
            pan: 0.0,
            opacity: 1.0,
            active: true,
        }
    }
}

impl TrackMixParams {
    /// Create mix params for a given volume and pan.
    #[must_use]
    pub fn audio(volume: f32, pan: f32) -> Self {
        Self {
            volume: volume.max(0.0),
            pan: pan.clamp(-1.0, 1.0),
            ..Default::default()
        }
    }

    /// Create mix params for a video track with given opacity.
    #[must_use]
    pub fn video(opacity: f32) -> Self {
        Self {
            opacity: opacity.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Compute left/right channel gain from pan using constant-power panning.
    #[must_use]
    pub fn pan_gains(&self) -> (f32, f32) {
        let angle = (self.pan + 1.0) / 2.0 * std::f32::consts::FRAC_PI_2;
        let left = self.volume * angle.cos();
        let right = self.volume * angle.sin();
        (left, right)
    }
}

/// Result of a mix operation.
#[derive(Debug, Clone)]
pub struct MixResult {
    /// Mixed video frame (if video tracks present).
    pub video: Option<PixelBuffer>,
    /// Mixed audio frame.
    pub audio: AudioFrame,
    /// Number of video layers mixed.
    pub video_layers: usize,
    /// Number of audio tracks mixed.
    pub audio_tracks: usize,
    /// Whether any track was clipped (audio peak > 1.0 before limiting).
    pub clipped: bool,
}

/// Mixes multiple audio/video tracks at a timeline position.
pub struct TrackMixer {
    /// Output video width.
    pub video_width: u32,
    /// Output video height.
    pub video_height: u32,
    /// Audio channels in output.
    pub audio_channels: usize,
    /// Samples per frame.
    pub samples_per_frame: usize,
    /// Master volume multiplier.
    pub master_volume: f32,
}

impl TrackMixer {
    /// Create a new track mixer.
    #[must_use]
    pub fn new(
        video_width: u32,
        video_height: u32,
        audio_channels: usize,
        sample_rate: u32,
        frame_rate_num: u32,
        frame_rate_den: u32,
    ) -> Self {
        let samples_per_frame = if frame_rate_num > 0 && frame_rate_den > 0 {
            (sample_rate * frame_rate_den / frame_rate_num) as usize
        } else {
            1920 // default for 48kHz @ 25fps
        };
        Self {
            video_width,
            video_height,
            audio_channels,
            samples_per_frame,
            master_volume: 1.0,
        }
    }

    /// Mix all tracks at a given timeline position.
    ///
    /// # Errors
    ///
    /// Returns an error if position is negative.
    pub fn mix(
        &self,
        timeline: &Timeline,
        position: Position,
        track_params: &[TrackMixParams],
    ) -> TimelineResult<MixResult> {
        if position.value() < 0 {
            return Err(TimelineError::InvalidPosition(
                "Position cannot be negative".to_string(),
            ));
        }

        let mut video_buf: Option<PixelBuffer> = None;
        let mut audio_out = AudioFrame::silent(self.audio_channels, self.samples_per_frame);
        let mut video_layers = 0usize;
        let mut audio_tracks = 0usize;
        let mut clipped = false;

        // Default params used when track_params doesn't cover a track
        let default_params = TrackMixParams::default();

        // Video mixing (bottom-to-top composition)
        for (track_idx, track) in timeline.video_tracks.iter().enumerate() {
            if track.hidden || track.muted {
                continue;
            }
            let params = track_params.get(track_idx).unwrap_or(&default_params);
            if !params.active {
                continue;
            }

            // Determine if any clip is active at this position
            let has_content = track.clips.iter().any(|clip| {
                if !clip.enabled {
                    return false;
                }
                let start = clip.timeline_in.value();
                let dur = clip.source_out.value() - clip.source_in.value();
                position.value() >= start && position.value() < start + dur
            });

            if has_content {
                let color = self.track_to_color(track_idx);
                let layer = PixelBuffer::solid(self.video_width, self.video_height, color);
                match &mut video_buf {
                    None => {
                        video_buf = Some(layer);
                    }
                    Some(base) => {
                        base.composite_over(&layer, 0, 0, params.opacity);
                    }
                }
                video_layers += 1;
            }
        }

        // Audio mixing
        let audio_track_offset = timeline.video_tracks.len();
        for (i, track) in timeline.audio_tracks.iter().enumerate() {
            if track.muted {
                continue;
            }
            let track_idx = audio_track_offset + i;
            let params = track_params.get(track_idx).unwrap_or(&default_params);
            if !params.active {
                continue;
            }

            // Check if any clip is active
            let has_content = track.clips.iter().any(|clip| {
                if !clip.enabled {
                    return false;
                }
                let start = clip.timeline_in.value();
                let dur = clip.source_out.value() - clip.source_in.value();
                position.value() >= start && position.value() < start + dur
            });

            if !has_content {
                continue;
            }

            // Generate synthetic sine tone for this track (in lieu of real decoding)
            let (gain_l, gain_r) = params.pan_gains();
            let effective_vol = params.volume * track.volume * self.master_volume;
            let freq = 220.0 * (i as f32 + 1.0);
            let sr = timeline.sample_rate as f32;

            for s in 0..self.samples_per_frame {
                let t = (position.value() * self.samples_per_frame as i64 + s as i64) as f32 / sr;
                let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * effective_vol;
                if sample.abs() > 1.0 {
                    clipped = true;
                }
                if self.audio_channels >= 1 {
                    audio_out.samples[0][s] =
                        (audio_out.samples[0][s] + sample * gain_l).clamp(-1.0, 1.0);
                }
                if self.audio_channels >= 2 {
                    audio_out.samples[1][s] =
                        (audio_out.samples[1][s] + sample * gain_r).clamp(-1.0, 1.0);
                }
                for c in 2..self.audio_channels {
                    audio_out.samples[c][s] =
                        (audio_out.samples[c][s] + sample * effective_vol).clamp(-1.0, 1.0);
                }
            }
            audio_tracks += 1;
        }

        Ok(MixResult {
            video: video_buf,
            audio: audio_out,
            video_layers,
            audio_tracks,
            clipped,
        })
    }

    fn track_to_color(&self, track_idx: usize) -> [u8; 4] {
        // Deterministic color per track
        let colors: &[[u8; 4]] = &[
            [80, 120, 200, 255],
            [200, 80, 80, 255],
            [80, 200, 120, 255],
            [200, 180, 80, 255],
            [160, 80, 200, 255],
        ];
        colors[track_idx % colors.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mixer() -> TrackMixer {
        TrackMixer::new(64, 36, 2, 48000, 24, 1)
    }

    #[test]
    fn test_audio_frame_silent() {
        let af = AudioFrame::silent(2, 100);
        assert_eq!(af.channels, 2);
        assert_eq!(af.samples_per_frame, 100);
        assert_eq!(af.peak(), 0.0);
        assert_eq!(af.rms(), 0.0);
    }

    #[test]
    fn test_audio_frame_peak() {
        let mut af = AudioFrame::silent(1, 4);
        af.samples[0] = vec![0.5, -0.8, 0.3, 0.1];
        assert!((af.peak() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_audio_frame_rms() {
        let mut af = AudioFrame::silent(1, 4);
        af.samples[0] = vec![1.0, 1.0, 1.0, 1.0];
        assert!((af.rms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_track_mix_params_default() {
        let p = TrackMixParams::default();
        assert!((p.volume - 1.0).abs() < f32::EPSILON);
        assert!(p.active);
        assert_eq!(p.pan, 0.0);
    }

    #[test]
    fn test_pan_gains_center() {
        let p = TrackMixParams::audio(1.0, 0.0);
        let (l, r) = p.pan_gains();
        // Center: l = r = cos(pi/4) = sin(pi/4) = 1/sqrt(2)
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((l - expected).abs() < 0.01, "L={l}");
        assert!((r - expected).abs() < 0.01, "R={r}");
    }

    #[test]
    fn test_pan_gains_full_left() {
        let p = TrackMixParams::audio(1.0, -1.0);
        let (l, r) = p.pan_gains();
        assert!((l - 1.0).abs() < 0.01, "L should be ~1.0");
        assert!(r < 0.01, "R should be ~0.0");
    }

    #[test]
    fn test_pan_gains_full_right() {
        let p = TrackMixParams::audio(1.0, 1.0);
        let (l, r) = p.pan_gains();
        assert!(l < 0.01, "L should be ~0.0");
        assert!((r - 1.0).abs() < 0.01, "R should be ~1.0");
    }

    #[test]
    fn test_mix_empty_timeline() {
        use oximedia_core::Rational;
        let mixer = make_mixer();
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let result = mixer
            .mix(&timeline, Position::new(0), &[])
            .expect("should succeed in test");
        assert!(result.video.is_none());
        assert_eq!(result.video_layers, 0);
        assert_eq!(result.audio_tracks, 0);
    }

    #[test]
    fn test_mix_negative_position_error() {
        use oximedia_core::Rational;
        let mixer = make_mixer();
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let result = mixer.mix(&timeline, Position::new(-1), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_samples_per_frame_calc() {
        // 48000 / 24 = 2000 samples/frame
        let mixer = TrackMixer::new(1920, 1080, 2, 48000, 24, 1);
        assert_eq!(mixer.samples_per_frame, 2000);
    }

    #[test]
    fn test_mixer_master_volume() {
        let mut mixer = make_mixer();
        mixer.master_volume = 0.5;
        assert!((mixer.master_volume - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_track_mix_params_video() {
        let p = TrackMixParams::video(0.7);
        assert!((p.opacity - 0.7).abs() < f32::EPSILON);
        assert!((p.volume - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mix_result_audio_silent_when_no_clips() {
        use oximedia_core::Rational;
        let mixer = make_mixer();
        let timeline =
            Timeline::new("test", Rational::new(24, 1), 48000).expect("should succeed in test");
        let result = mixer
            .mix(&timeline, Position::new(5), &[])
            .expect("should succeed in test");
        assert_eq!(result.audio.peak(), 0.0);
    }
}
