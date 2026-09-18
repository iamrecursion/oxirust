//! Stereo processing for MP3.
//!
//! This module implements stereo decoding including:
//! - Joint stereo (intensity stereo and MS stereo)
//! - Dual channel
//! - Regular stereo

use super::frame::{ChannelMode, JointStereoMode};

/// Stereo processor.
pub struct StereoProcessor {
    /// Intensity stereo state.
    intensity_scale: [f32; 576],
    /// MS stereo state.
    ms_state: bool,
}

impl Default for StereoProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoProcessor {
    /// Create new stereo processor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            intensity_scale: [1.0; 576],
            ms_state: false,
        }
    }

    /// Process stereo samples based on channel mode.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32], mode: ChannelMode) {
        match mode {
            ChannelMode::Stereo => {
                // No processing needed for regular stereo
            }
            ChannelMode::JointStereo(joint_mode) => {
                self.process_joint_stereo(left, right, joint_mode);
            }
            ChannelMode::DualChannel => {
                // Dual channel - treat as independent mono channels
            }
            ChannelMode::Mono => {
                // Copy mono to both channels
                right.copy_from_slice(left);
            }
        }
    }

    /// Process joint stereo (intensity and/or MS stereo).
    fn process_joint_stereo(&mut self, left: &mut [f32], right: &mut [f32], mode: JointStereoMode) {
        let len = left.len().min(right.len());

        if mode.ms_stereo {
            // MS stereo decoding: M/S -> L/R
            self.decode_ms_stereo(left, right, len);
        }

        if mode.intensity {
            // Intensity stereo decoding
            self.decode_intensity_stereo(left, right, len, mode.bound);
        }
    }

    /// Decode MS (Mid/Side) stereo to L/R.
    fn decode_ms_stereo(&mut self, left: &mut [f32], right: &mut [f32], len: usize) {
        const SQRT2_2: f32 = 0.707_106_78;

        for i in 0..len {
            let mid = left[i];
            let side = right[i];

            // L = (M + S) / sqrt(2)
            // R = (M - S) / sqrt(2)
            left[i] = (mid + side) * SQRT2_2;
            right[i] = (mid - side) * SQRT2_2;
        }

        self.ms_state = true;
    }

    /// Decode intensity stereo.
    fn decode_intensity_stereo(
        &mut self,
        left: &mut [f32],
        right: &mut [f32],
        len: usize,
        bound: u8,
    ) {
        // Intensity stereo uses left channel data and scales it for right
        let bound = bound as usize;

        if bound < len {
            for i in bound..len {
                let scale = self.intensity_scale[i];
                right[i] = left[i] * scale;
            }
        }
    }

    /// Set intensity stereo scale factors.
    pub fn set_intensity_scale(&mut self, scales: &[f32]) {
        let len = scales.len().min(self.intensity_scale.len());
        self.intensity_scale[..len].copy_from_slice(&scales[..len]);
    }

    /// Reset stereo state.
    pub fn reset(&mut self) {
        self.intensity_scale = [1.0; 576];
        self.ms_state = false;
    }

    /// Check if MS stereo is active.
    #[must_use]
    pub const fn is_ms_active(&self) -> bool {
        self.ms_state
    }
}

/// Calculate intensity stereo scale from position.
#[must_use]
pub fn intensity_scale_from_position(pos: u8, is_right: bool) -> f32 {
    // ISO/IEC 11172-3 intensity stereo table
    const INTENSITY_TABLE: [f32; 14] = [
        0.0,
        0.211_324_87,
        0.366_025_4,
        0.5,
        0.633_974_6,
        0.788_675_13,
        1.0,
        1.0,
        0.788_675_13,
        0.633_974_6,
        0.5,
        0.366_025_4,
        0.211_324_87,
        0.0,
    ];

    if pos >= 7 {
        // Right channel
        let idx = (pos as usize).min(13);
        if is_right {
            INTENSITY_TABLE[idx]
        } else {
            INTENSITY_TABLE[13 - idx]
        }
    } else {
        // Left channel
        let idx = pos as usize;
        if is_right {
            INTENSITY_TABLE[13 - idx]
        } else {
            INTENSITY_TABLE[idx]
        }
    }
}

/// Apply stereo width control.
pub fn apply_stereo_width(left: &mut [f32], right: &mut [f32], width: f32) {
    debug_assert!(left.len() == right.len());

    // width = 0.0: mono
    // width = 1.0: normal stereo
    // width > 1.0: widened stereo

    let width = width.max(0.0);

    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        let mid = (*l + *r) * 0.5;
        let side = (*l - *r) * 0.5;

        *l = mid + side * width;
        *r = mid - side * width;
    }
}

/// Convert stereo to mono (downmix).
pub fn downmix_to_mono(left: &[f32], right: &[f32], output: &mut [f32]) {
    debug_assert!(left.len() == right.len());
    debug_assert!(output.len() >= left.len());

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        output[i] = (l + r) * 0.5;
    }
}

/// Upmix mono to stereo (copy).
pub fn upmix_to_stereo(input: &[f32], left: &mut [f32], right: &mut [f32]) {
    debug_assert!(left.len() == right.len());
    debug_assert!(input.len() <= left.len());

    let len = input.len();
    left[..len].copy_from_slice(&input[..len]);
    right[..len].copy_from_slice(&input[..len]);
}

/// Apply stereo balance.
pub fn apply_balance(left: &mut [f32], right: &mut [f32], balance: f32) {
    debug_assert!(left.len() == right.len());

    // balance: -1.0 = left only, 0.0 = center, 1.0 = right only
    let balance = balance.clamp(-1.0, 1.0);

    let left_scale = if balance > 0.0 { 1.0 - balance } else { 1.0 };
    let right_scale = if balance < 0.0 { 1.0 + balance } else { 1.0 };

    for l in left.iter_mut() {
        *l *= left_scale;
    }

    for r in right.iter_mut() {
        *r *= right_scale;
    }
}

/// Swap left and right channels.
pub fn swap_channels(left: &mut [f32], right: &mut [f32]) {
    debug_assert!(left.len() == right.len());

    for (l, r) in left.iter_mut().zip(right.iter_mut()) {
        std::mem::swap(l, r);
    }
}

/// Check if stereo signal is mostly mono.
#[must_use]
pub fn is_mostly_mono(left: &[f32], right: &[f32], threshold: f32) -> bool {
    debug_assert!(left.len() == right.len());

    let mut diff_sum = 0.0f32;
    let mut total_sum = 0.0f32;

    for (&l, &r) in left.iter().zip(right.iter()) {
        diff_sum += (l - r).abs();
        total_sum += (l.abs() + r.abs()) * 0.5;
    }

    if total_sum == 0.0 {
        return true;
    }

    (diff_sum / total_sum) < threshold
}

/// Calculate stereo separation (correlation).
#[must_use]
pub fn calculate_separation(left: &[f32], right: &[f32]) -> f32 {
    debug_assert!(left.len() == right.len());

    let mut correlation = 0.0f32;
    let mut left_energy = 0.0f32;
    let mut right_energy = 0.0f32;

    for (&l, &r) in left.iter().zip(right.iter()) {
        correlation += l * r;
        left_energy += l * l;
        right_energy += r * r;
    }

    let denominator = (left_energy * right_energy).sqrt();
    if denominator == 0.0 {
        return 0.0;
    }

    (correlation / denominator).clamp(-1.0, 1.0)
}
