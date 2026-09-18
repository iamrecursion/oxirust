//! Surround sound processing for audio post-production.
//!
//! Provides surround format descriptions, channel mapping, VBAP-style panning,
//! and simple LFE management.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Surround sound format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundLayout {
    /// Single channel (mono).
    Mono,
    /// Two channels (L, R).
    Stereo,
    /// Three channels (L, R, C).
    Lrc,
    /// Four channels (L, R, Ls, Rs).
    Quad,
    /// Six channels (L, R, C, LFE, Ls, Rs).
    FiveOne,
    /// Eight channels (L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    SevenOne,
}

impl SurroundLayout {
    /// Number of audio channels in this layout.
    #[must_use]
    pub const fn channel_count(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Lrc => 3,
            Self::Quad => 4,
            Self::FiveOne => 6,
            Self::SevenOne => 8,
        }
    }

    /// Whether this layout contains a dedicated LFE channel.
    #[must_use]
    pub const fn has_lfe(self) -> bool {
        matches!(self, Self::FiveOne | Self::SevenOne)
    }
}

/// Assignment of named channels to physical track indices.
#[derive(Debug, Clone, Default)]
pub struct ChannelMap {
    /// List of (channel_name, track_index) pairs.
    pub assignments: Vec<(String, u8)>,
}

impl ChannelMap {
    /// Create an empty channel map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create the standard 5.1 channel map (L, R, C, LFE, Ls, Rs).
    #[must_use]
    pub fn new_51() -> Self {
        Self {
            assignments: vec![
                ("L".to_string(), 0),
                ("R".to_string(), 1),
                ("C".to_string(), 2),
                ("LFE".to_string(), 3),
                ("Ls".to_string(), 4),
                ("Rs".to_string(), 5),
            ],
        }
    }

    /// Create the standard 7.1 channel map (L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    #[must_use]
    pub fn new_71() -> Self {
        Self {
            assignments: vec![
                ("L".to_string(), 0),
                ("R".to_string(), 1),
                ("C".to_string(), 2),
                ("LFE".to_string(), 3),
                ("Lss".to_string(), 4),
                ("Rss".to_string(), 5),
                ("Lrs".to_string(), 6),
                ("Rrs".to_string(), 7),
            ],
        }
    }

    /// Find the track index assigned to a named channel.
    ///
    /// Returns `None` if the channel is not in the map.
    #[must_use]
    pub fn find_channel(&self, name: &str) -> Option<u8> {
        self.assignments
            .iter()
            .find(|(ch, _)| ch == name)
            .map(|(_, idx)| *idx)
    }

    /// Add or update a channel assignment.
    pub fn assign(&mut self, name: impl Into<String>, track_index: u8) {
        let name = name.into();
        if let Some(entry) = self.assignments.iter_mut().find(|(ch, _)| *ch == name) {
            entry.1 = track_index;
        } else {
            self.assignments.push((name, track_index));
        }
    }
}

/// VBAP-style surround panner for a given layout.
///
/// Maps a 2D position (x in –1..=1 left-right, y in –1..=1 back-front) to
/// per-channel gains. The gains are normalized so that the sum of squares equals 1.
#[derive(Debug, Clone)]
pub struct SurroundPannerNew {
    /// The surround layout used for panning.
    pub format: SurroundLayout,
}

impl SurroundPannerNew {
    /// Create a new panner for the given layout.
    #[must_use]
    pub fn new(format: SurroundLayout) -> Self {
        Self { format }
    }

    /// Compute per-channel gains for a position (x, y).
    ///
    /// - `x`: –1.0 = hard left, 0.0 = center, +1.0 = hard right.
    /// - `y`: –1.0 = rear, 0.0 = side, +1.0 = front.
    ///
    /// Returns a `Vec` of linear gains, one per channel, normalized so
    /// the sum of squares equals 1.0 (or all zeros for mono).
    #[must_use]
    pub fn pan(&self, x: f32, y: f32) -> Vec<f32> {
        let ch = self.format.channel_count() as usize;
        let mut gains = vec![0.0f32; ch];

        // Clamp inputs
        let x = x.clamp(-1.0, 1.0);
        let y = y.clamp(-1.0, 1.0);

        match self.format {
            SurroundLayout::Mono => {
                gains[0] = 1.0;
                return gains;
            }
            SurroundLayout::Stereo => {
                // Constant-power pan
                let angle = (x * 0.5 + 0.5) * std::f32::consts::FRAC_PI_2;
                gains[0] = angle.cos(); // L
                gains[1] = angle.sin(); // R
            }
            SurroundLayout::Lrc => {
                let right_frac = (x + 1.0) / 2.0;
                let left_frac = 1.0 - right_frac;
                let center_frac = 1.0 - right_frac.abs() - (left_frac - 0.5).abs();
                gains[0] = (left_frac * (1.0 - center_frac.max(0.0))).sqrt();
                gains[1] = (right_frac * (1.0 - center_frac.max(0.0))).sqrt();
                gains[2] = center_frac.max(0.0).sqrt();
            }
            SurroundLayout::Quad => {
                // L, R, Ls, Rs
                let front = ((y + 1.0) / 2.0).clamp(0.0, 1.0);
                let rear = 1.0 - front;
                let right = ((x + 1.0) / 2.0).clamp(0.0, 1.0);
                let left = 1.0 - right;
                gains[0] = (front * left).sqrt();
                gains[1] = (front * right).sqrt();
                gains[2] = (rear * left).sqrt();
                gains[3] = (rear * right).sqrt();
            }
            SurroundLayout::FiveOne => {
                // L, R, C, LFE, Ls, Rs
                let front = ((y + 1.0) / 2.0).clamp(0.0, 1.0);
                let rear = 1.0 - front;
                let right = ((x + 1.0) / 2.0).clamp(0.0, 1.0);
                let left = 1.0 - right;
                // Front: split between C and L/R
                let center_amt = 1.0 - (x.abs());
                let lr_front = 1.0 - center_amt * 0.5;
                gains[0] = (front * left * lr_front).sqrt();
                gains[1] = (front * right * lr_front).sqrt();
                gains[2] = (front * center_amt * 0.5).sqrt();
                gains[3] = 0.0; // LFE not driven by panning
                gains[4] = (rear * left).sqrt();
                gains[5] = (rear * right).sqrt();
            }
            SurroundLayout::SevenOne => {
                // L, R, C, LFE, Lss, Rss, Lrs, Rrs
                let front = ((y + 1.0) / 2.0).clamp(0.0, 1.0);
                let rear_total = 1.0 - front;
                let side = (rear_total * 0.5).sqrt();
                let rear = (rear_total * 0.5).sqrt();
                let right = ((x + 1.0) / 2.0).clamp(0.0, 1.0);
                let left = 1.0 - right;
                let center_amt = 1.0 - x.abs();
                let lr_front = 1.0 - center_amt * 0.5;
                gains[0] = (front * left * lr_front).sqrt();
                gains[1] = (front * right * lr_front).sqrt();
                gains[2] = (front * center_amt * 0.5).sqrt();
                gains[3] = 0.0;
                gains[4] = side * left; // Lss
                gains[5] = side * right; // Rss
                gains[6] = rear * left; // Lrs
                gains[7] = rear * right; // Rrs
            }
        }

        // Normalize so sum of squares = 1 (skip for mono which is already 1)
        if self.format != SurroundLayout::Mono {
            let sum_sq: f32 = gains.iter().map(|&g| g * g).sum();
            if sum_sq > 1e-9 {
                let norm = sum_sq.sqrt();
                for g in &mut gains {
                    *g /= norm;
                }
            }
        }

        gains
    }
}

/// Low-frequency effects manager with simple low-pass filtering.
#[derive(Debug, Clone)]
pub struct LfeManager {
    /// Low-pass crossover frequency in Hz.
    pub crossover_hz: f32,
    /// LFE channel gain in dB (applied after filtering).
    pub gain_db: f32,
}

impl LfeManager {
    /// Create a new LFE manager.
    #[must_use]
    pub fn new(crossover_hz: f32, gain_db: f32) -> Self {
        Self {
            crossover_hz,
            gain_db,
        }
    }

    /// Extract LFE content from a mono sample slice using a running-average low-pass filter.
    ///
    /// The window size is derived from the crossover frequency and sample rate.
    /// A linear gain is applied based on `gain_db`.
    ///
    /// Returns a `Vec<f32>` of the same length as `samples`.
    #[must_use]
    pub fn extract_lfe(&self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        if samples.is_empty() || sample_rate == 0 {
            return vec![];
        }

        // Window size = samples per half-cycle at crossover frequency
        // Clamp to at least 1 to avoid division by zero
        let window = ((sample_rate as f32 / (2.0 * self.crossover_hz.max(1.0))) as usize).max(1);
        let linear_gain = 10_f32.powf(self.gain_db / 20.0);

        let n = samples.len();
        let mut output = vec![0.0f32; n];

        // Running-average low-pass
        let mut running_sum = 0.0f32;
        // Pre-fill with zeros (causal filter)
        for (i, &s) in samples.iter().enumerate() {
            running_sum += s;
            if i >= window {
                running_sum -= samples[i - window];
            }
            let effective_window = (i + 1).min(window);
            output[i] = (running_sum / effective_window as f32) * linear_gain;
        }

        output
    }
}

// ── Upmixing Algorithms ───────────────────────────────────────────────────────

/// Error type for surround upmixing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum UpmixError {
    /// The input channel count does not match the expected source layout.
    InputChannelMismatch {
        /// Expected number of channels.
        expected: usize,
        /// Actual number of channels received.
        got: usize,
    },
    /// The input buffer is empty.
    EmptyInput,
}

impl std::fmt::Display for UpmixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputChannelMismatch { expected, got } => {
                write!(f, "Input channel mismatch: expected {expected}, got {got}")
            }
            Self::EmptyInput => write!(f, "Input buffer is empty"),
        }
    }
}

impl std::error::Error for UpmixError {}

/// Stereo-to-5.1 upmixer using a Lexicon-style passive matrix decode.
///
/// Channel assignment in the output vector:
/// `[L, R, C, LFE, Ls, Rs]`
///
/// Algorithm:
/// - L and R pass through directly.
/// - Centre (C) = 0.707 × (L + R).
/// - LFE = low-shelved mid signal (first-order IIR at 80 Hz). Set to 0 if
///   `lfe_enabled` is `false`.
/// - Ls and Rs are derived from the difference signal (L − R) with independent
///   decorrelation delays, scaled by `surround_level`.
pub struct StereoTo51Upmixer {
    /// Whether to populate the LFE channel with a bass-managed signal.
    pub lfe_enabled: bool,
    /// Low-pass crossover frequency for LFE extraction (Hz).
    pub lfe_crossover_hz: f32,
    /// Level for the surround channels (linear, default 0.707).
    pub surround_level: f32,
    /// Decorrelation delay for Ls in samples (default 11).
    pub ls_delay_samples: usize,
    /// Decorrelation delay for Rs in samples (default 13).
    pub rs_delay_samples: usize,
    // Internal state for LFE low-pass filter.
    lfe_lp_state: f32,
    // Decorrelation delay line buffers.
    ls_delay_buf: Vec<f32>,
    rs_delay_buf: Vec<f32>,
    ls_delay_pos: usize,
    rs_delay_pos: usize,
}

impl std::fmt::Debug for StereoTo51Upmixer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StereoTo51Upmixer")
            .field("lfe_enabled", &self.lfe_enabled)
            .field("lfe_crossover_hz", &self.lfe_crossover_hz)
            .field("surround_level", &self.surround_level)
            .field("ls_delay_samples", &self.ls_delay_samples)
            .field("rs_delay_samples", &self.rs_delay_samples)
            .finish()
    }
}

impl StereoTo51Upmixer {
    /// Create a new stereo-to-5.1 upmixer.
    #[must_use]
    pub fn new() -> Self {
        let ls_delay_samples = 11;
        let rs_delay_samples = 13;
        Self {
            lfe_enabled: true,
            lfe_crossover_hz: 80.0,
            surround_level: std::f32::consts::FRAC_1_SQRT_2, // 0.707
            ls_delay_samples,
            rs_delay_samples,
            lfe_lp_state: 0.0,
            ls_delay_buf: vec![0.0; ls_delay_samples.max(1)],
            rs_delay_buf: vec![0.0; rs_delay_samples.max(1)],
            ls_delay_pos: 0,
            rs_delay_pos: 0,
        }
    }

    /// Update decorrelation delay buffer sizes after changing `ls_delay_samples`
    /// or `rs_delay_samples`.
    pub fn rebuild_delay_buffers(&mut self) {
        self.ls_delay_buf = vec![0.0; self.ls_delay_samples.max(1)];
        self.rs_delay_buf = vec![0.0; self.rs_delay_samples.max(1)];
        self.ls_delay_pos = 0;
        self.rs_delay_pos = 0;
    }

    /// Upmix a block of interleaved stereo samples (L R L R …) to 6 separate
    /// channel buffers `[L, R, C, LFE, Ls, Rs]`.
    ///
    /// `input` must contain an even number of samples (stereo interleaved).
    ///
    /// # Errors
    ///
    /// Returns [`UpmixError::InputChannelMismatch`] if the input length is odd
    /// (cannot represent interleaved stereo), or [`UpmixError::EmptyInput`] for
    /// empty input.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_interleaved(
        &mut self,
        input: &[f32],
        sample_rate: u32,
    ) -> Result<[Vec<f32>; 6], UpmixError> {
        if input.is_empty() {
            return Err(UpmixError::EmptyInput);
        }
        if input.len() % 2 != 0 {
            return Err(UpmixError::InputChannelMismatch {
                expected: 2,
                got: 1,
            });
        }

        let frames = input.len() / 2;

        // Pre-compute LFE low-pass coefficient.
        let dt = 1.0 / sample_rate as f32;
        let rc_lfe = 1.0 / (2.0 * std::f32::consts::PI * self.lfe_crossover_hz.max(1.0));
        let lfe_alpha = dt / (rc_lfe + dt); // first-order LP coefficient

        let mut out_l = Vec::with_capacity(frames);
        let mut out_r = Vec::with_capacity(frames);
        let mut out_c = Vec::with_capacity(frames);
        let mut out_lfe = Vec::with_capacity(frames);
        let mut out_ls = Vec::with_capacity(frames);
        let mut out_rs = Vec::with_capacity(frames);

        let ls_buf_len = self.ls_delay_buf.len();
        let rs_buf_len = self.rs_delay_buf.len();

        for frame in 0..frames {
            let l = input[frame * 2];
            let r = input[frame * 2 + 1];

            // L and R direct.
            out_l.push(l);
            out_r.push(r);

            // Centre: mono sum attenuated.
            out_c.push((l + r) * std::f32::consts::FRAC_1_SQRT_2);

            // LFE: low-pass filtered mid.
            let mid = (l + r) * 0.5;
            self.lfe_lp_state = self.lfe_lp_state + lfe_alpha * (mid - self.lfe_lp_state);
            out_lfe.push(if self.lfe_enabled {
                self.lfe_lp_state
            } else {
                0.0
            });

            // Surround: difference signal with decorrelation delays.
            let diff = (l - r) * self.surround_level;

            // Write new sample into Ls delay buffer and read delayed sample.
            self.ls_delay_buf[self.ls_delay_pos % ls_buf_len] = diff;
            let ls_read_pos = (self.ls_delay_pos + 1) % ls_buf_len;
            let ls_out = self.ls_delay_buf[ls_read_pos];
            self.ls_delay_pos = (self.ls_delay_pos + 1) % ls_buf_len;

            // Rs delay (negative polarity for decorrelation).
            self.rs_delay_buf[self.rs_delay_pos % rs_buf_len] = -diff;
            let rs_read_pos = (self.rs_delay_pos + 1) % rs_buf_len;
            let rs_out = self.rs_delay_buf[rs_read_pos];
            self.rs_delay_pos = (self.rs_delay_pos + 1) % rs_buf_len;

            out_ls.push(ls_out);
            out_rs.push(rs_out);
        }

        Ok([out_l, out_r, out_c, out_lfe, out_ls, out_rs])
    }

    /// Reset internal filter/delay state.
    pub fn reset(&mut self) {
        self.lfe_lp_state = 0.0;
        for s in &mut self.ls_delay_buf {
            *s = 0.0;
        }
        for s in &mut self.rs_delay_buf {
            *s = 0.0;
        }
        self.ls_delay_pos = 0;
        self.rs_delay_pos = 0;
    }
}

impl Default for StereoTo51Upmixer {
    fn default() -> Self {
        Self::new()
    }
}

/// 5.1-to-7.1 upmixer that synthesises two additional side-surround channels.
///
/// Input channel layout: `[L, R, C, LFE, Ls, Rs]` (6 channels).
/// Output channel layout: `[L, R, C, LFE, Lss, Rss, Lrs, Rrs]` (8 channels).
///
/// - Front, centre, and LFE pass through directly.
/// - The original `Ls`/`Rs` become the new rear-surround pair (`Lrs`/`Rrs`).
/// - The new side-surround channels (`Lss`/`Rss`) are derived from a blend of
///   the front and original surround channels, decorrelated via small delays.
pub struct FiveOneTo71Upmixer {
    /// Blend factor between front and surround for the side-surround channel
    /// (0.0 = pure surround, 1.0 = pure front blend; default 0.4).
    pub side_blend: f32,
    /// Linear gain applied to both new side-surround channels (default 0.707).
    pub side_level: f32,
    /// Decorrelation delay for Lss in samples (default 7).
    pub lss_delay_samples: usize,
    /// Decorrelation delay for Rss in samples (default 9).
    pub rss_delay_samples: usize,
    lss_delay_buf: Vec<f32>,
    rss_delay_buf: Vec<f32>,
    lss_delay_pos: usize,
    rss_delay_pos: usize,
}

impl std::fmt::Debug for FiveOneTo71Upmixer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FiveOneTo71Upmixer")
            .field("side_blend", &self.side_blend)
            .field("side_level", &self.side_level)
            .field("lss_delay_samples", &self.lss_delay_samples)
            .field("rss_delay_samples", &self.rss_delay_samples)
            .finish()
    }
}

impl FiveOneTo71Upmixer {
    /// Create a new 5.1-to-7.1 upmixer.
    #[must_use]
    pub fn new() -> Self {
        let lss_delay_samples = 7;
        let rss_delay_samples = 9;
        Self {
            side_blend: 0.4,
            side_level: std::f32::consts::FRAC_1_SQRT_2,
            lss_delay_samples,
            rss_delay_samples,
            lss_delay_buf: vec![0.0; lss_delay_samples.max(1)],
            rss_delay_buf: vec![0.0; rss_delay_samples.max(1)],
            lss_delay_pos: 0,
            rss_delay_pos: 0,
        }
    }

    /// Upmix six separate channel buffers (5.1) to eight (7.1).
    ///
    /// `channels` must contain exactly 6 buffers of equal length.
    ///
    /// Returns `[L, R, C, LFE, Lss, Rss, Lrs, Rrs]`.
    ///
    /// # Errors
    ///
    /// Returns [`UpmixError::InputChannelMismatch`] if fewer or more than 6 channels
    /// are provided, or [`UpmixError::EmptyInput`] if any channel is empty.
    pub fn process(&mut self, channels: &[Vec<f32>]) -> Result<[Vec<f32>; 8], UpmixError> {
        if channels.len() != 6 {
            return Err(UpmixError::InputChannelMismatch {
                expected: 6,
                got: channels.len(),
            });
        }
        let frames = channels[0].len();
        if frames == 0 {
            return Err(UpmixError::EmptyInput);
        }

        let lss_buf_len = self.lss_delay_buf.len();
        let rss_buf_len = self.rss_delay_buf.len();

        let mut out_l = Vec::with_capacity(frames);
        let mut out_r = Vec::with_capacity(frames);
        let mut out_c = Vec::with_capacity(frames);
        let mut out_lfe = Vec::with_capacity(frames);
        let mut out_lss = Vec::with_capacity(frames);
        let mut out_rss = Vec::with_capacity(frames);
        let mut out_lrs = Vec::with_capacity(frames);
        let mut out_rrs = Vec::with_capacity(frames);

        for frame in 0..frames {
            let l = channels[0].get(frame).copied().unwrap_or(0.0);
            let r = channels[1].get(frame).copied().unwrap_or(0.0);
            let c = channels[2].get(frame).copied().unwrap_or(0.0);
            let lfe = channels[3].get(frame).copied().unwrap_or(0.0);
            let ls = channels[4].get(frame).copied().unwrap_or(0.0);
            let rs = channels[5].get(frame).copied().unwrap_or(0.0);

            // Front, C, and LFE pass through.
            out_l.push(l);
            out_r.push(r);
            out_c.push(c);
            out_lfe.push(lfe);

            // Original surround → rear surround.
            out_lrs.push(ls);
            out_rrs.push(rs);

            // New side surround: blend front and surround.
            let lss_signal = (l * self.side_blend + ls * (1.0 - self.side_blend)) * self.side_level;
            let rss_signal = (r * self.side_blend + rs * (1.0 - self.side_blend)) * self.side_level;

            // Decorrelation delays.
            self.lss_delay_buf[self.lss_delay_pos % lss_buf_len] = lss_signal;
            let lss_read = (self.lss_delay_pos + 1) % lss_buf_len;
            let lss_out = self.lss_delay_buf[lss_read];
            self.lss_delay_pos = (self.lss_delay_pos + 1) % lss_buf_len;

            self.rss_delay_buf[self.rss_delay_pos % rss_buf_len] = rss_signal;
            let rss_read = (self.rss_delay_pos + 1) % rss_buf_len;
            let rss_out = self.rss_delay_buf[rss_read];
            self.rss_delay_pos = (self.rss_delay_pos + 1) % rss_buf_len;

            out_lss.push(lss_out);
            out_rss.push(rss_out);
        }

        Ok([
            out_l, out_r, out_c, out_lfe, out_lss, out_rss, out_lrs, out_rrs,
        ])
    }

    /// Reset internal delay state.
    pub fn reset(&mut self) {
        for s in &mut self.lss_delay_buf {
            *s = 0.0;
        }
        for s in &mut self.rss_delay_buf {
            *s = 0.0;
        }
        self.lss_delay_pos = 0;
        self.rss_delay_pos = 0;
    }
}

impl Default for FiveOneTo71Upmixer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Dolby Atmos Object-Based Audio ───────────────────────────────────────────

/// A single Dolby Atmos audio object with 3D positional metadata.
///
/// Each object represents an independent audio element that can be placed
/// and moved in 3D space (azimuth, elevation, distance). The renderer maps
/// objects to physical loudspeakers or headphone binaural output at decode time.
#[derive(Debug, Clone)]
pub struct AtmosObject {
    /// Object identifier (unique within a session).
    pub id: u32,
    /// Human-readable label (e.g. "Character A Dialogue").
    pub label: String,
    /// Azimuth angle in degrees: 0 = front centre, +90 = right, −90 = left.
    pub azimuth_deg: f32,
    /// Elevation angle in degrees: 0 = ear level, +90 = overhead.
    pub elevation_deg: f32,
    /// Distance from the listener (1.0 = reference distance).
    pub distance: f32,
    /// Linear gain applied to the object's audio (0.0 = silent).
    pub gain: f32,
    /// Whether this object is currently active in the mix.
    pub active: bool,
    /// Diverge factor for spreading a point source across multiple speakers
    /// (0.0 = pure point source, 1.0 = fully diverged to bed).
    pub diverge: f32,
}

impl AtmosObject {
    /// Create a new Atmos object positioned at the front centre.
    #[must_use]
    pub fn new(id: u32, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            azimuth_deg: 0.0,
            elevation_deg: 0.0,
            distance: 1.0,
            gain: 1.0,
            active: true,
            diverge: 0.0,
        }
    }

    /// Set 3D position.
    #[must_use]
    pub fn with_position(mut self, azimuth_deg: f32, elevation_deg: f32, distance: f32) -> Self {
        self.azimuth_deg = azimuth_deg.clamp(-180.0, 180.0);
        self.elevation_deg = elevation_deg.clamp(-90.0, 90.0);
        self.distance = distance.max(0.01);
        self
    }

    /// Set linear gain.
    #[must_use]
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain = gain.max(0.0);
        self
    }

    /// Set diverge factor (0.0 – 1.0).
    #[must_use]
    pub fn with_diverge(mut self, diverge: f32) -> Self {
        self.diverge = diverge.clamp(0.0, 1.0);
        self
    }

    /// Compute the distance attenuation factor (inverse-square law, clamped).
    #[must_use]
    pub fn distance_attenuation(&self) -> f32 {
        (1.0 / self.distance.max(0.01).powi(2)).min(1.0)
    }

    /// Return azimuth/elevation as radians for renderer calculations.
    #[must_use]
    pub fn position_radians(&self) -> (f32, f32) {
        (
            self.azimuth_deg.to_radians(),
            self.elevation_deg.to_radians(),
        )
    }
}

/// Bed channel layout for an Atmos presentation.
///
/// A "bed" is the channel-based base layer (e.g. 7.1 or 5.1) that object audio
/// is mixed on top of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtmosBedLayout {
    /// 5.1 bed.
    FiveOne,
    /// 7.1 bed.
    SevenOne,
    /// 7.1.2 bed (adds 2 overhead channels).
    SevenOneTwoOh,
    /// 7.1.4 bed (adds 4 overhead channels).
    SevenOneFourOh,
}

impl AtmosBedLayout {
    /// Number of channels in this bed layout.
    #[must_use]
    pub const fn channel_count(self) -> u8 {
        match self {
            Self::FiveOne => 6,
            Self::SevenOne => 8,
            Self::SevenOneTwoOh => 10,
            Self::SevenOneFourOh => 12,
        }
    }

    /// Whether this layout includes overhead channels.
    #[must_use]
    pub const fn has_overhead(self) -> bool {
        matches!(self, Self::SevenOneTwoOh | Self::SevenOneFourOh)
    }
}

/// Dolby Atmos session layout managing a bed and a set of audio objects.
///
/// This struct represents the metadata layer of an Atmos mix. Actual audio
/// data for each object is managed externally (e.g. by the DAW track system).
#[derive(Debug)]
pub struct AtmosLayout {
    /// Bed (channel-based) layout.
    pub bed: AtmosBedLayout,
    /// Maximum number of simultaneous objects (Atmos supports up to 128 in cinema,
    /// 16 in home).
    pub max_objects: u32,
    /// Current objects in the presentation.
    objects: Vec<AtmosObject>,
    /// Next object ID counter.
    next_id: u32,
}

impl AtmosLayout {
    /// Create a new Atmos layout with the specified bed and object ceiling.
    #[must_use]
    pub fn new(bed: AtmosBedLayout, max_objects: u32) -> Self {
        Self {
            bed,
            max_objects: max_objects.clamp(1, 128),
            objects: Vec::new(),
            next_id: 1,
        }
    }

    /// Create a standard home-theatre Atmos layout (7.1.4 bed, 16 objects).
    #[must_use]
    pub fn home_theatre() -> Self {
        Self::new(AtmosBedLayout::SevenOneFourOh, 16)
    }

    /// Create a cinema Atmos layout (7.1 bed, 128 objects).
    #[must_use]
    pub fn cinema() -> Self {
        Self::new(AtmosBedLayout::SevenOne, 128)
    }

    /// Add an audio object to the presentation.
    ///
    /// Returns `None` if the maximum object count is already reached.
    pub fn add_object(&mut self, label: impl Into<String>) -> Option<u32> {
        if self.objects.len() as u32 >= self.max_objects {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.objects.push(AtmosObject::new(id, label));
        Some(id)
    }

    /// Get an immutable reference to an object by ID.
    #[must_use]
    pub fn get_object(&self, id: u32) -> Option<&AtmosObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Get a mutable reference to an object by ID.
    pub fn get_object_mut(&mut self, id: u32) -> Option<&mut AtmosObject> {
        self.objects.iter_mut().find(|o| o.id == id)
    }

    /// Remove an object by ID.  Returns the removed object if found.
    pub fn remove_object(&mut self, id: u32) -> Option<AtmosObject> {
        if let Some(pos) = self.objects.iter().position(|o| o.id == id) {
            Some(self.objects.remove(pos))
        } else {
            None
        }
    }

    /// Number of active objects.
    #[must_use]
    pub fn active_object_count(&self) -> usize {
        self.objects.iter().filter(|o| o.active).count()
    }

    /// Total object count (including inactive).
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Move an object to a new 3D position.
    ///
    /// Returns `false` if the object ID was not found.
    pub fn move_object(
        &mut self,
        id: u32,
        azimuth_deg: f32,
        elevation_deg: f32,
        distance: f32,
    ) -> bool {
        if let Some(obj) = self.get_object_mut(id) {
            obj.azimuth_deg = azimuth_deg.clamp(-180.0, 180.0);
            obj.elevation_deg = elevation_deg.clamp(-90.0, 90.0);
            obj.distance = distance.max(0.01);
            true
        } else {
            false
        }
    }

    /// Render object-to-bed downmix coefficients for a given object using
    /// vector-based amplitude panning (VBAP-lite) onto the bed channels.
    ///
    /// Returns one linear gain per bed channel, or `None` if the object is not found.
    #[must_use]
    pub fn object_bed_coefficients(&self, id: u32) -> Option<Vec<f32>> {
        let obj = self.get_object(id)?;
        let n = self.bed.channel_count() as usize;
        let mut gains = vec![0.0f32; n];

        let az = obj.azimuth_deg;
        let el = obj.elevation_deg;
        let dist_att = obj.distance_attenuation();
        let total_gain = obj.gain * dist_att;

        // Normalised azimuth: -1 (left) .. +1 (right)
        let az_norm = (az / 180.0).clamp(-1.0, 1.0);
        // Elevation weight: 0 = ear level, 1 = overhead
        let el_norm = (el / 90.0).clamp(0.0, 1.0);

        // Pan law.
        let pan_angle = (az_norm * 0.5 + 0.5) * std::f32::consts::FRAC_PI_2;
        let pan_r = pan_angle.sin();
        let pan_l = pan_angle.cos();

        let floor_weight = 1.0 - el_norm;

        // Distribute to floor channels based on bed layout (index scheme):
        // 7.1.4 → 0=L 1=R 2=C 3=LFE 4=Lss 5=Rss 6=Lrs 7=Rrs 8=Ltf 9=Rtf 10=Ltr 11=Rtr
        // 7.1.2 → same but only 8+9 for top.
        // 7.1   → 0=L 1=R 2=C 3=LFE 4=Lss 5=Rss 6=Lrs 7=Rrs
        // 5.1   → 0=L 1=R 2=C 3=LFE 4=Ls 5=Rs
        match self.bed {
            AtmosBedLayout::FiveOne => {
                gains[0] = pan_l * floor_weight * total_gain * (1.0 - obj.diverge);
                gains[1] = pan_r * floor_weight * total_gain * (1.0 - obj.diverge);
                gains[2] = (1.0 - az_norm.abs()) * floor_weight * total_gain * obj.diverge;
                // LFE: no direct object panning.
                gains[3] = 0.0;
                gains[4] = pan_l * (1.0 - floor_weight) * total_gain;
                gains[5] = pan_r * (1.0 - floor_weight) * total_gain;
            }
            AtmosBedLayout::SevenOne => {
                gains[0] = pan_l * floor_weight * total_gain;
                gains[1] = pan_r * floor_weight * total_gain;
                gains[2] = (1.0 - az_norm.abs()) * floor_weight * total_gain * 0.5;
                gains[3] = 0.0; // LFE
                gains[4] = pan_l * floor_weight * total_gain * 0.5;
                gains[5] = pan_r * floor_weight * total_gain * 0.5;
                gains[6] = pan_l * (1.0 - floor_weight) * total_gain * 0.5;
                gains[7] = pan_r * (1.0 - floor_weight) * total_gain * 0.5;
            }
            AtmosBedLayout::SevenOneTwoOh => {
                gains[0] = pan_l * floor_weight * total_gain;
                gains[1] = pan_r * floor_weight * total_gain;
                gains[2] = (1.0 - az_norm.abs()) * floor_weight * total_gain * 0.5;
                gains[3] = 0.0; // LFE
                gains[4] = pan_l * floor_weight * total_gain * 0.5;
                gains[5] = pan_r * floor_weight * total_gain * 0.5;
                gains[6] = pan_l * (1.0 - floor_weight) * total_gain * 0.3;
                gains[7] = pan_r * (1.0 - floor_weight) * total_gain * 0.3;
                gains[8] = pan_l * el_norm * total_gain; // Ltf
                gains[9] = pan_r * el_norm * total_gain; // Rtf
            }
            AtmosBedLayout::SevenOneFourOh => {
                gains[0] = pan_l * floor_weight * total_gain;
                gains[1] = pan_r * floor_weight * total_gain;
                gains[2] = (1.0 - az_norm.abs()) * floor_weight * total_gain * 0.5;
                gains[3] = 0.0; // LFE
                gains[4] = pan_l * floor_weight * total_gain * 0.5;
                gains[5] = pan_r * floor_weight * total_gain * 0.5;
                gains[6] = pan_l * (1.0 - floor_weight) * total_gain * 0.3;
                gains[7] = pan_r * (1.0 - floor_weight) * total_gain * 0.3;
                // Overhead: front pair
                gains[8] = pan_l * el_norm * total_gain * 0.7;
                gains[9] = pan_r * el_norm * total_gain * 0.7;
                // Overhead: rear pair
                gains[10] = pan_l * el_norm * total_gain * 0.3;
                gains[11] = pan_r * el_norm * total_gain * 0.3;
            }
        }

        Some(gains)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surround_layout_channel_count_mono() {
        assert_eq!(SurroundLayout::Mono.channel_count(), 1);
    }

    #[test]
    fn test_surround_layout_channel_count_stereo() {
        assert_eq!(SurroundLayout::Stereo.channel_count(), 2);
    }

    #[test]
    fn test_surround_layout_channel_count_51() {
        assert_eq!(SurroundLayout::FiveOne.channel_count(), 6);
    }

    #[test]
    fn test_surround_layout_channel_count_71() {
        assert_eq!(SurroundLayout::SevenOne.channel_count(), 8);
    }

    #[test]
    fn test_has_lfe_51() {
        assert!(SurroundLayout::FiveOne.has_lfe());
    }

    #[test]
    fn test_has_lfe_stereo() {
        assert!(!SurroundLayout::Stereo.has_lfe());
    }

    #[test]
    fn test_channel_map_51_find_lfe() {
        let map = ChannelMap::new_51();
        assert_eq!(map.find_channel("LFE"), Some(3));
    }

    #[test]
    fn test_channel_map_71_find_lrs() {
        let map = ChannelMap::new_71();
        assert_eq!(map.find_channel("Lrs"), Some(6));
    }

    #[test]
    fn test_channel_map_find_missing() {
        let map = ChannelMap::new_51();
        assert_eq!(map.find_channel("Unknown"), None);
    }

    #[test]
    fn test_channel_map_assign_new() {
        let mut map = ChannelMap::new();
        map.assign("X", 5);
        assert_eq!(map.find_channel("X"), Some(5));
    }

    #[test]
    fn test_channel_map_assign_update() {
        let mut map = ChannelMap::new_51();
        map.assign("L", 10);
        assert_eq!(map.find_channel("L"), Some(10));
    }

    #[test]
    fn test_panner_mono_returns_one() {
        let panner = SurroundPannerNew::new(SurroundLayout::Mono);
        let gains = panner.pan(0.5, 0.5);
        assert_eq!(gains.len(), 1);
        assert!((gains[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_panner_stereo_center_equal() {
        let panner = SurroundPannerNew::new(SurroundLayout::Stereo);
        let gains = panner.pan(0.0, 0.0);
        assert_eq!(gains.len(), 2);
        // At center (x=0), both channels should be approximately equal
        let diff = (gains[0] - gains[1]).abs();
        assert!(diff < 0.1, "L and R differ by {diff} at center");
    }

    #[test]
    fn test_panner_51_returns_six_gains() {
        let panner = SurroundPannerNew::new(SurroundLayout::FiveOne);
        let gains = panner.pan(0.0, 1.0);
        assert_eq!(gains.len(), 6);
    }

    #[test]
    fn test_panner_71_returns_eight_gains() {
        let panner = SurroundPannerNew::new(SurroundLayout::SevenOne);
        let gains = panner.pan(0.0, 0.0);
        assert_eq!(gains.len(), 8);
    }

    #[test]
    fn test_panner_gains_normalized() {
        let panner = SurroundPannerNew::new(SurroundLayout::Stereo);
        let gains = panner.pan(0.3, 0.0);
        let sum_sq: f32 = gains.iter().map(|&g| g * g).sum();
        assert!((sum_sq - 1.0).abs() < 1e-5, "sum_sq={sum_sq}");
    }

    #[test]
    fn test_lfe_manager_output_length() {
        let lfe = LfeManager::new(120.0, 0.0);
        let samples = vec![0.5f32; 100];
        let out = lfe.extract_lfe(&samples, 48000);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_lfe_manager_empty_input() {
        let lfe = LfeManager::new(120.0, 0.0);
        let out = lfe.extract_lfe(&[], 48000);
        assert!(out.is_empty());
    }

    #[test]
    fn test_lfe_manager_gain_applied() {
        let lfe_unity = LfeManager::new(120.0, 0.0);
        let lfe_boost = LfeManager::new(120.0, 6.0); // ~2x linear
        let samples = vec![1.0f32; 100];
        let out_unity = lfe_unity.extract_lfe(&samples, 48000);
        let out_boost = lfe_boost.extract_lfe(&samples, 48000);
        // Last sample should reflect the gain ratio
        assert!(
            out_boost[99] > out_unity[99],
            "Boost should increase output"
        );
    }

    // ── StereoTo51Upmixer tests ───────────────────────────────────────────────

    #[test]
    fn test_stereo_to_51_output_shape() {
        let mut upmixer = StereoTo51Upmixer::new();
        let input: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01).sin()).collect();
        let channels = upmixer
            .process_interleaved(&input, 48000)
            .expect("upmix should succeed");
        assert_eq!(channels.len(), 6);
        assert_eq!(channels[0].len(), 100); // frames = input.len() / 2
    }

    #[test]
    fn test_stereo_to_51_empty_input() {
        let mut upmixer = StereoTo51Upmixer::new();
        assert!(upmixer.process_interleaved(&[], 48000).is_err());
    }

    #[test]
    fn test_stereo_to_51_odd_input_error() {
        let mut upmixer = StereoTo51Upmixer::new();
        let input = vec![0.0f32; 5]; // odd = not stereo-interleaved
        assert!(upmixer.process_interleaved(&input, 48000).is_err());
    }

    #[test]
    fn test_stereo_to_51_lfe_disabled() {
        let mut upmixer = StereoTo51Upmixer::new();
        upmixer.lfe_enabled = false;
        let input = vec![0.5f32; 200];
        let channels = upmixer
            .process_interleaved(&input, 48000)
            .expect("upmix should succeed");
        // LFE channel (index 3) should be all zeros.
        assert!(
            channels[3].iter().all(|&s| s == 0.0),
            "LFE should be silent"
        );
    }

    #[test]
    fn test_stereo_to_51_reset_clears_state() {
        let mut upmixer = StereoTo51Upmixer::new();
        let input: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        upmixer
            .process_interleaved(&input, 48000)
            .expect("upmix should succeed");
        upmixer.reset();
        assert_eq!(upmixer.lfe_lp_state, 0.0);
    }

    // ── FiveOneTo71Upmixer tests ──────────────────────────────────────────────

    #[test]
    fn test_51_to_71_output_shape() {
        let mut upmixer = FiveOneTo71Upmixer::new();
        let channels: Vec<Vec<f32>> = (0..6).map(|_| vec![0.3f32; 100]).collect();
        let out = upmixer.process(&channels).expect("upmix should succeed");
        assert_eq!(out.len(), 8);
        assert_eq!(out[0].len(), 100);
    }

    #[test]
    fn test_51_to_71_wrong_channel_count() {
        let mut upmixer = FiveOneTo71Upmixer::new();
        let channels: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; 100]).collect();
        assert!(upmixer.process(&channels).is_err());
    }

    #[test]
    fn test_51_to_71_empty_channels() {
        let mut upmixer = FiveOneTo71Upmixer::new();
        let channels: Vec<Vec<f32>> = (0..6).map(|_| vec![]).collect();
        assert!(upmixer.process(&channels).is_err());
    }

    #[test]
    fn test_51_to_71_rear_surround_pass_through() {
        let mut upmixer = FiveOneTo71Upmixer::new();
        let ls = vec![0.8f32; 50];
        let rs = vec![0.6f32; 50];
        let channels: Vec<Vec<f32>> = vec![
            vec![0.0; 50], // L
            vec![0.0; 50], // R
            vec![0.0; 50], // C
            vec![0.0; 50], // LFE
            ls.clone(),    // Ls
            rs.clone(),    // Rs
        ];
        let out = upmixer.process(&channels).expect("upmix should succeed");
        // Lrs (index 6) should equal original Ls.
        assert_eq!(out[6], ls, "Lrs should be original Ls");
        assert_eq!(out[7], rs, "Rrs should be original Rs");
    }

    // ── AtmosLayout tests ─────────────────────────────────────────────────────

    #[test]
    fn test_atmos_layout_home_theatre() {
        let layout = AtmosLayout::home_theatre();
        assert_eq!(layout.bed, AtmosBedLayout::SevenOneFourOh);
        assert_eq!(layout.max_objects, 16);
    }

    #[test]
    fn test_atmos_layout_cinema() {
        let layout = AtmosLayout::cinema();
        assert_eq!(layout.bed, AtmosBedLayout::SevenOne);
        assert_eq!(layout.max_objects, 128);
    }

    #[test]
    fn test_atmos_add_and_get_object() {
        let mut layout = AtmosLayout::home_theatre();
        let id = layout
            .add_object("Dialogue")
            .expect("add_object should succeed");
        let obj = layout.get_object(id).expect("get_object should succeed");
        assert_eq!(obj.label, "Dialogue");
        assert_eq!(layout.object_count(), 1);
    }

    #[test]
    fn test_atmos_max_objects_enforced() {
        let mut layout = AtmosLayout::new(AtmosBedLayout::FiveOne, 2);
        assert!(layout.add_object("A").is_some());
        assert!(layout.add_object("B").is_some());
        // Third object should be rejected.
        assert!(layout.add_object("C").is_none());
    }

    #[test]
    fn test_atmos_remove_object() {
        let mut layout = AtmosLayout::home_theatre();
        let id = layout.add_object("FX").expect("add_object should succeed");
        assert_eq!(layout.object_count(), 1);
        assert!(layout.remove_object(id).is_some());
        assert_eq!(layout.object_count(), 0);
    }

    #[test]
    fn test_atmos_move_object() {
        let mut layout = AtmosLayout::home_theatre();
        let id = layout
            .add_object("Voice")
            .expect("add_object should succeed");
        assert!(layout.move_object(id, 45.0, 30.0, 2.0));
        let obj = layout.get_object(id).expect("get_object should succeed");
        assert!((obj.azimuth_deg - 45.0).abs() < 1e-5);
        assert!((obj.elevation_deg - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_atmos_object_bed_coefficients_51() {
        let mut layout = AtmosLayout::new(AtmosBedLayout::FiveOne, 16);
        let id = layout
            .add_object("Center")
            .expect("add_object should succeed");
        let coeffs = layout
            .object_bed_coefficients(id)
            .expect("coefficients should exist");
        assert_eq!(coeffs.len(), 6);
    }

    #[test]
    fn test_atmos_object_bed_coefficients_714() {
        let mut layout = AtmosLayout::home_theatre();
        let id = layout
            .add_object("Overhead")
            .expect("add_object should succeed");
        // Place overhead.
        layout.move_object(id, 0.0, 90.0, 1.0);
        let coeffs = layout
            .object_bed_coefficients(id)
            .expect("coefficients should exist");
        assert_eq!(coeffs.len(), 12);
        // Overhead channels (8–11) should have non-zero gains.
        let overhead_sum: f32 = coeffs[8..].iter().sum();
        assert!(overhead_sum > 0.0, "Overhead channels should have gain");
    }

    #[test]
    fn test_atmos_bed_layout_channel_count() {
        assert_eq!(AtmosBedLayout::FiveOne.channel_count(), 6);
        assert_eq!(AtmosBedLayout::SevenOne.channel_count(), 8);
        assert_eq!(AtmosBedLayout::SevenOneTwoOh.channel_count(), 10);
        assert_eq!(AtmosBedLayout::SevenOneFourOh.channel_count(), 12);
    }

    #[test]
    fn test_atmos_bed_layout_has_overhead() {
        assert!(!AtmosBedLayout::FiveOne.has_overhead());
        assert!(!AtmosBedLayout::SevenOne.has_overhead());
        assert!(AtmosBedLayout::SevenOneTwoOh.has_overhead());
        assert!(AtmosBedLayout::SevenOneFourOh.has_overhead());
    }

    #[test]
    fn test_atmos_object_distance_attenuation() {
        let obj = AtmosObject::new(1, "test").with_position(0.0, 0.0, 2.0);
        // At distance 2, inv-sq gives 0.25.
        assert!((obj.distance_attenuation() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_atmos_active_object_count() {
        let mut layout = AtmosLayout::cinema();
        let id1 = layout.add_object("A").expect("should succeed");
        layout.add_object("B").expect("should succeed");
        if let Some(obj) = layout.get_object_mut(id1) {
            obj.active = false;
        }
        assert_eq!(layout.active_object_count(), 1);
    }
}
