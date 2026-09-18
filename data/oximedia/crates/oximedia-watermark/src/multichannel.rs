//! Multi-channel watermarking support.
//!
//! Extends the watermark embedding/detection pipeline to support stereo,
//! 5.1 surround, and arbitrary channel layouts with per-channel or joint
//! embedding strategies.

use crate::error::{WatermarkError, WatermarkResult};
use crate::{WatermarkConfig, WatermarkDetector, WatermarkEmbedder};

/// Channel layout for multi-channel audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Single channel (mono).
    Mono,
    /// Stereo (left, right).
    Stereo,
    /// 5.1 surround (FL, FR, C, LFE, SL, SR).
    Surround51,
    /// 7.1 surround (FL, FR, C, LFE, SL, SR, BL, BR).
    Surround71,
    /// Arbitrary number of channels.
    Custom(usize),
}

impl ChannelLayout {
    /// Number of channels in this layout.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Custom(n) => *n,
        }
    }
}

/// Strategy for embedding across multiple channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedStrategy {
    /// Embed the same watermark independently in each channel.
    /// Provides redundancy: detection works even if some channels are lost.
    Independent,
    /// Embed different payload segments across channels for higher capacity.
    /// Each channel carries a portion of the payload.
    Distributed,
    /// Embed in the mid (L+R) signal only, preserving stereo image.
    /// Only valid for stereo or layouts with L/R pairs.
    MidOnly,
    /// Embed complementary watermarks in L and R (anti-phase embedding).
    /// Watermark cancels in mono sum, survives in stereo difference.
    Complementary,
    /// Embed only in selected channels, leaving others untouched.
    Selective(u8),
}

/// Multi-channel watermark embedder.
pub struct MultiChannelEmbedder {
    config: WatermarkConfig,
    sample_rate: u32,
    layout: ChannelLayout,
    strategy: EmbedStrategy,
}

impl MultiChannelEmbedder {
    /// Create a new multi-channel embedder.
    #[must_use]
    pub fn new(
        config: WatermarkConfig,
        sample_rate: u32,
        layout: ChannelLayout,
        strategy: EmbedStrategy,
    ) -> Self {
        Self {
            config,
            sample_rate,
            layout,
            strategy,
        }
    }

    /// Embed watermark in interleaved multi-channel audio.
    ///
    /// Input: interleaved samples [L0, R0, L1, R1, ...] for stereo, etc.
    /// Output: watermarked interleaved samples.
    ///
    /// # Errors
    ///
    /// Returns error if audio length is not a multiple of channel count,
    /// or if embedding fails on any channel.
    pub fn embed_interleaved(
        &self,
        interleaved: &[f32],
        payload: &[u8],
    ) -> WatermarkResult<Vec<f32>> {
        let num_channels = self.layout.num_channels();

        if interleaved.len() % num_channels != 0 {
            return Err(WatermarkError::InvalidParameter(format!(
                "Interleaved length {} not divisible by {} channels",
                interleaved.len(),
                num_channels,
            )));
        }

        // Deinterleave
        let channels = deinterleave(interleaved, num_channels);

        // Embed
        let watermarked_channels = self.embed_channels(&channels, payload)?;

        // Reinterleave
        Ok(interleave(&watermarked_channels))
    }

    /// Embed watermark in separate channel buffers.
    ///
    /// # Errors
    ///
    /// Returns error if embedding fails.
    pub fn embed_channels(
        &self,
        channels: &[Vec<f32>],
        payload: &[u8],
    ) -> WatermarkResult<Vec<Vec<f32>>> {
        let num_channels = self.layout.num_channels();
        if channels.len() != num_channels {
            return Err(WatermarkError::InvalidParameter(format!(
                "Expected {} channels, got {}",
                num_channels,
                channels.len(),
            )));
        }

        match self.strategy {
            EmbedStrategy::Independent => self.embed_independent(channels, payload),
            EmbedStrategy::Distributed => self.embed_distributed(channels, payload),
            EmbedStrategy::MidOnly => self.embed_mid_only(channels, payload),
            EmbedStrategy::Complementary => self.embed_complementary(channels, payload),
            EmbedStrategy::Selective(mask) => self.embed_selective(channels, payload, mask),
        }
    }

    /// Independent: embed same watermark in every channel.
    fn embed_independent(
        &self,
        channels: &[Vec<f32>],
        payload: &[u8],
    ) -> WatermarkResult<Vec<Vec<f32>>> {
        let embedder = WatermarkEmbedder::new(self.config.clone(), self.sample_rate);
        let mut result = Vec::with_capacity(channels.len());

        for ch in channels {
            result.push(embedder.embed(ch, payload)?);
        }

        Ok(result)
    }

    /// Distributed: split payload across channels for higher capacity.
    fn embed_distributed(
        &self,
        channels: &[Vec<f32>],
        payload: &[u8],
    ) -> WatermarkResult<Vec<Vec<f32>>> {
        let num_channels = channels.len();
        let embedder = WatermarkEmbedder::new(self.config.clone(), self.sample_rate);
        let mut result = Vec::with_capacity(num_channels);

        // Split payload into chunks
        let chunk_size = payload.len().div_ceil(num_channels);

        for (i, ch) in channels.iter().enumerate() {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(payload.len());

            if start < payload.len() {
                let chunk = &payload[start..end];
                result.push(embedder.embed(ch, chunk)?);
            } else {
                // No more payload for this channel; leave untouched
                result.push(ch.clone());
            }
        }

        Ok(result)
    }

    /// Mid-only: embed in mid (L+R)/2, then redistribute.
    fn embed_mid_only(
        &self,
        channels: &[Vec<f32>],
        payload: &[u8],
    ) -> WatermarkResult<Vec<Vec<f32>>> {
        if channels.len() < 2 {
            return Err(WatermarkError::InvalidParameter(
                "MidOnly requires at least 2 channels".to_string(),
            ));
        }

        let left = &channels[0];
        let right = &channels[1];
        let n = left.len().min(right.len());

        // Compute mid/side
        let mid: Vec<f32> = (0..n).map(|i| (left[i] + right[i]) * 0.5).collect();
        let side: Vec<f32> = (0..n).map(|i| (left[i] - right[i]) * 0.5).collect();

        // Embed in mid only
        let embedder = WatermarkEmbedder::new(self.config.clone(), self.sample_rate);
        let watermarked_mid = embedder.embed(&mid, payload)?;

        // Reconstruct L/R from watermarked mid + original side
        let new_left: Vec<f32> = (0..n).map(|i| watermarked_mid[i] + side[i]).collect();
        let new_right: Vec<f32> = (0..n).map(|i| watermarked_mid[i] - side[i]).collect();

        let mut result = vec![new_left, new_right];

        // Copy remaining channels unchanged
        for ch in channels.iter().skip(2) {
            result.push(ch.clone());
        }

        Ok(result)
    }

    /// Complementary: embed in L, embed inverted in R.
    fn embed_complementary(
        &self,
        channels: &[Vec<f32>],
        payload: &[u8],
    ) -> WatermarkResult<Vec<Vec<f32>>> {
        if channels.len() < 2 {
            return Err(WatermarkError::InvalidParameter(
                "Complementary requires at least 2 channels".to_string(),
            ));
        }

        let embedder = WatermarkEmbedder::new(self.config.clone(), self.sample_rate);

        // Embed in left channel
        let watermarked_left = embedder.embed(&channels[0], payload)?;

        // Compute the watermark signal (difference)
        let n = channels[0].len().min(watermarked_left.len());
        let watermark_signal: Vec<f32> = (0..n)
            .map(|i| watermarked_left[i] - channels[0][i])
            .collect();

        // Subtract watermark from right channel (anti-phase)
        let n_right = channels[1].len().min(watermark_signal.len());
        let watermarked_right: Vec<f32> = (0..n_right)
            .map(|i| channels[1][i] - watermark_signal[i])
            .collect();

        let mut result = vec![watermarked_left, watermarked_right];

        // Copy remaining channels unchanged
        for ch in channels.iter().skip(2) {
            result.push(ch.clone());
        }

        Ok(result)
    }

    /// Selective: embed only in channels specified by bitmask.
    fn embed_selective(
        &self,
        channels: &[Vec<f32>],
        payload: &[u8],
        mask: u8,
    ) -> WatermarkResult<Vec<Vec<f32>>> {
        let embedder = WatermarkEmbedder::new(self.config.clone(), self.sample_rate);
        let mut result = Vec::with_capacity(channels.len());

        for (i, ch) in channels.iter().enumerate() {
            if i < 8 && (mask >> i) & 1 == 1 {
                result.push(embedder.embed(ch, payload)?);
            } else {
                result.push(ch.clone());
            }
        }

        Ok(result)
    }
}

/// Multi-channel watermark detector.
pub struct MultiChannelDetector {
    config: WatermarkConfig,
    layout: ChannelLayout,
    strategy: EmbedStrategy,
}

impl MultiChannelDetector {
    /// Create a new multi-channel detector.
    #[must_use]
    pub fn new(config: WatermarkConfig, layout: ChannelLayout, strategy: EmbedStrategy) -> Self {
        Self {
            config,
            layout,
            strategy,
        }
    }

    /// Detect watermark from interleaved multi-channel audio.
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect_interleaved(
        &self,
        interleaved: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        let num_channels = self.layout.num_channels();

        if interleaved.len() % num_channels != 0 {
            return Err(WatermarkError::InvalidParameter(format!(
                "Interleaved length {} not divisible by {} channels",
                interleaved.len(),
                num_channels,
            )));
        }

        let channels = deinterleave(interleaved, num_channels);
        self.detect_channels(&channels, expected_bits)
    }

    /// Detect watermark from separate channel buffers.
    ///
    /// # Errors
    ///
    /// Returns error if detection fails.
    pub fn detect_channels(
        &self,
        channels: &[Vec<f32>],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        match self.strategy {
            EmbedStrategy::Independent => self.detect_independent(channels, expected_bits),
            EmbedStrategy::Distributed => self.detect_distributed(channels, expected_bits),
            EmbedStrategy::MidOnly => self.detect_mid_only(channels, expected_bits),
            EmbedStrategy::Complementary => self.detect_complementary(channels, expected_bits),
            EmbedStrategy::Selective(mask) => self.detect_selective(channels, expected_bits, mask),
        }
    }

    /// Detect from any channel (independent mode) — try each channel.
    fn detect_independent(
        &self,
        channels: &[Vec<f32>],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        let detector = WatermarkDetector::new(self.config.clone());

        // Try each channel; return the first successful detection
        let mut last_err = WatermarkError::NotDetected;
        for ch in channels {
            match detector.detect(ch, expected_bits) {
                Ok(payload) => return Ok(payload),
                Err(e) => last_err = e,
            }
        }

        Err(last_err)
    }

    /// Detect distributed payload by reassembling chunks from each channel.
    fn detect_distributed(
        &self,
        channels: &[Vec<f32>],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        let detector = WatermarkDetector::new(self.config.clone());

        // Each channel has expected_bits / num_channels bits (approximately)
        let bits_per_channel = expected_bits.div_ceil(channels.len());

        let mut payload = Vec::new();
        for ch in channels {
            match detector.detect(ch, bits_per_channel) {
                Ok(chunk) => payload.extend_from_slice(&chunk),
                Err(e) => return Err(e),
            }
        }

        Ok(payload)
    }

    /// Detect from mid channel.
    fn detect_mid_only(
        &self,
        channels: &[Vec<f32>],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        if channels.len() < 2 {
            return Err(WatermarkError::InvalidParameter(
                "MidOnly requires at least 2 channels".to_string(),
            ));
        }

        let n = channels[0].len().min(channels[1].len());
        let mid: Vec<f32> = (0..n)
            .map(|i| (channels[0][i] + channels[1][i]) * 0.5)
            .collect();

        let detector = WatermarkDetector::new(self.config.clone());
        detector.detect(&mid, expected_bits)
    }

    /// Detect complementary watermark from L-R difference.
    fn detect_complementary(
        &self,
        channels: &[Vec<f32>],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        if channels.len() < 2 {
            return Err(WatermarkError::InvalidParameter(
                "Complementary requires at least 2 channels".to_string(),
            ));
        }

        // The watermark is in (L - R) / 2: doubled watermark, no original signal
        // Actually: L has +wm, R has -wm, so (L - R) / 2 = original_side + wm
        // Try detecting directly from left channel (which has the watermark)
        let detector = WatermarkDetector::new(self.config.clone());
        detector.detect(&channels[0], expected_bits)
    }

    /// Detect from selected channels.
    fn detect_selective(
        &self,
        channels: &[Vec<f32>],
        expected_bits: usize,
        mask: u8,
    ) -> WatermarkResult<Vec<u8>> {
        let detector = WatermarkDetector::new(self.config.clone());

        for (i, ch) in channels.iter().enumerate() {
            if i < 8 && (mask >> i) & 1 == 1 {
                match detector.detect(ch, expected_bits) {
                    Ok(payload) => return Ok(payload),
                    Err(_) => continue,
                }
            }
        }

        Err(WatermarkError::NotDetected)
    }
}

/// Deinterleave multi-channel audio.
#[must_use]
pub fn deinterleave(interleaved: &[f32], num_channels: usize) -> Vec<Vec<f32>> {
    let samples_per_channel = interleaved.len() / num_channels;
    let mut channels = vec![Vec::with_capacity(samples_per_channel); num_channels];

    for (i, &sample) in interleaved.iter().enumerate() {
        channels[i % num_channels].push(sample);
    }

    channels
}

/// Interleave separate channels into single buffer.
#[must_use]
pub fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    if channels.is_empty() {
        return Vec::new();
    }

    let max_len = channels.iter().map(|c| c.len()).max().unwrap_or(0);
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(max_len * num_channels);

    for i in 0..max_len {
        for ch in channels {
            interleaved.push(ch.get(i).copied().unwrap_or(0.0));
        }
    }

    interleaved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::PayloadCodec;
    use crate::Algorithm;

    fn make_config() -> WatermarkConfig {
        WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.1)
            .with_key(99999)
            .with_psychoacoustic(false)
    }

    fn expected_bits_for(payload: &[u8]) -> WatermarkResult<usize> {
        let codec = PayloadCodec::new(16, 8)?;
        let encoded = codec.encode(payload)?;
        Ok(encoded.len() * 8)
    }

    #[test]
    fn test_deinterleave_interleave_roundtrip() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![5.0, 6.0, 7.0, 8.0];
        let interleaved = interleave(&[left.clone(), right.clone()]);
        assert_eq!(interleaved, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);

        let channels = deinterleave(&interleaved, 2);
        assert_eq!(channels[0], left);
        assert_eq!(channels[1], right);
    }

    #[test]
    fn test_independent_stereo_embed_detect() {
        let config = make_config();
        let sample_count = 73728;
        let left = vec![0.0f32; sample_count];
        let right = vec![0.0f32; sample_count];
        let payload = b"WM";

        let embedder = MultiChannelEmbedder::new(
            config.clone(),
            44100,
            ChannelLayout::Stereo,
            EmbedStrategy::Independent,
        );

        let result = embedder
            .embed_channels(&[left, right], payload)
            .expect("embed should succeed");
        assert_eq!(result.len(), 2);

        let detector =
            MultiChannelDetector::new(config, ChannelLayout::Stereo, EmbedStrategy::Independent);

        let bits = expected_bits_for(payload).expect("codec should succeed");
        let detected = detector
            .detect_channels(&result, bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_mid_only_stereo() {
        let config = make_config();
        let sample_count = 73728;
        let left = vec![0.1f32; sample_count];
        let right = vec![0.1f32; sample_count];
        let payload = b"WM";

        let embedder = MultiChannelEmbedder::new(
            config.clone(),
            44100,
            ChannelLayout::Stereo,
            EmbedStrategy::MidOnly,
        );

        let result = embedder
            .embed_channels(&[left, right], payload)
            .expect("embed should succeed");
        assert_eq!(result.len(), 2);

        let detector =
            MultiChannelDetector::new(config, ChannelLayout::Stereo, EmbedStrategy::MidOnly);

        let bits = expected_bits_for(payload).expect("codec should succeed");
        let detected = detector
            .detect_channels(&result, bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_interleaved_embed_detect() {
        let config = make_config();
        let sample_count = 73728;
        let mut interleaved = Vec::with_capacity(sample_count * 2);
        for _ in 0..sample_count {
            interleaved.push(0.0f32); // L
            interleaved.push(0.0f32); // R
        }
        let payload = b"WM";

        let embedder = MultiChannelEmbedder::new(
            config.clone(),
            44100,
            ChannelLayout::Stereo,
            EmbedStrategy::Independent,
        );

        let watermarked = embedder
            .embed_interleaved(&interleaved, payload)
            .expect("embed should succeed");
        assert_eq!(watermarked.len(), interleaved.len());

        let detector =
            MultiChannelDetector::new(config, ChannelLayout::Stereo, EmbedStrategy::Independent);

        let bits = expected_bits_for(payload).expect("codec should succeed");
        let detected = detector
            .detect_interleaved(&watermarked, bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_selective_embed() {
        let config = make_config();
        let sample_count = 73728;
        let channels: Vec<Vec<f32>> = vec![vec![0.0f32; sample_count]; 6];
        let payload = b"WM";

        // Only embed in channels 0 and 2 (bitmask 0b00000101 = 5)
        let embedder = MultiChannelEmbedder::new(
            config.clone(),
            44100,
            ChannelLayout::Surround51,
            EmbedStrategy::Selective(5),
        );

        let result = embedder
            .embed_channels(&channels, payload)
            .expect("embed should succeed");
        assert_eq!(result.len(), 6);

        // Channels 1,3,4,5 should be unchanged
        assert_eq!(result[1], channels[1]);
        assert_eq!(result[3], channels[3]);
        assert_eq!(result[4], channels[4]);
        assert_eq!(result[5], channels[5]);

        // Channel 0 should be modified
        assert_ne!(result[0], channels[0]);

        let detector = MultiChannelDetector::new(
            config,
            ChannelLayout::Surround51,
            EmbedStrategy::Selective(5),
        );

        let bits = expected_bits_for(payload).expect("codec should succeed");
        let detected = detector
            .detect_channels(&result, bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_complementary_embed() {
        let config = make_config();
        let sample_count = 73728;
        let left = vec![0.0f32; sample_count];
        let right = vec![0.0f32; sample_count];
        let payload = b"WM";

        let embedder = MultiChannelEmbedder::new(
            config.clone(),
            44100,
            ChannelLayout::Stereo,
            EmbedStrategy::Complementary,
        );

        let result = embedder
            .embed_channels(&[left, right], payload)
            .expect("embed should succeed");

        // Watermark should be anti-phase: L[i] - orig_L[i] ≈ -(R[i] - orig_R[i])
        let n = result[0].len().min(result[1].len());
        let mut anti_phase_count = 0;
        for i in 0..n {
            let wm_l = result[0][i]; // orig was 0.0
            let wm_r = result[1][i];
            if (wm_l + wm_r).abs() < (wm_l.abs() + wm_r.abs()) * 0.5 + 1e-10 {
                anti_phase_count += 1;
            }
        }
        // Most samples should show anti-phase behavior
        assert!(
            anti_phase_count > n / 2,
            "expected anti-phase: {anti_phase_count}/{n}"
        );

        let detector =
            MultiChannelDetector::new(config, ChannelLayout::Stereo, EmbedStrategy::Complementary);

        let bits = expected_bits_for(payload).expect("codec should succeed");
        let detected = detector
            .detect_channels(&result, bits)
            .expect("detect should succeed");
        assert_eq!(detected.as_slice(), payload);
    }

    #[test]
    fn test_invalid_channel_count() {
        let config = make_config();
        let embedder = MultiChannelEmbedder::new(
            config,
            44100,
            ChannelLayout::Stereo,
            EmbedStrategy::Independent,
        );

        // Only 1 channel for stereo layout → error
        let result = embedder.embed_channels(&[vec![0.0; 1000]], b"test");
        assert!(result.is_err());
    }

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.num_channels(), 1);
        assert_eq!(ChannelLayout::Stereo.num_channels(), 2);
        assert_eq!(ChannelLayout::Surround51.num_channels(), 6);
        assert_eq!(ChannelLayout::Surround71.num_channels(), 8);
        assert_eq!(ChannelLayout::Custom(4).num_channels(), 4);
    }

    #[test]
    fn test_interleaved_invalid_length() {
        let config = make_config();
        let embedder = MultiChannelEmbedder::new(
            config,
            44100,
            ChannelLayout::Stereo,
            EmbedStrategy::Independent,
        );

        // Odd number of samples for stereo → error
        let result = embedder.embed_interleaved(&[0.0; 5], b"test");
        assert!(result.is_err());
    }

    #[test]
    fn test_mid_only_mono_fails() {
        let config = make_config();
        let embedder =
            MultiChannelEmbedder::new(config, 44100, ChannelLayout::Mono, EmbedStrategy::MidOnly);

        let result = embedder.embed_channels(&[vec![0.0; 1000]], b"test");
        assert!(result.is_err());
    }
}
