//! Pixel mapping for LED walls in virtual production.
//!
//! Provides tools for mapping rendered pixels to physical LED panel coordinates,
//! including HDR remapping, gamma correction, and per-panel calibration offsets.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// An RGB pixel with floating-point components in linear light [0.0, ∞).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearPixel {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

impl LinearPixel {
    /// Create a new linear pixel.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Clamp all channels to [0.0, `max_nits`].
    #[must_use]
    pub fn clamp_nits(self, max_nits: f32) -> Self {
        Self {
            r: self.r.clamp(0.0, max_nits),
            g: self.g.clamp(0.0, max_nits),
            b: self.b.clamp(0.0, max_nits),
        }
    }

    /// Scale all channels by a factor.
    #[must_use]
    pub fn scale(self, factor: f32) -> Self {
        Self {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
        }
    }

    /// Convert to 8-bit sRGB (gamma-corrected) output.
    #[must_use]
    pub fn to_srgb8(self) -> [u8; 3] {
        let apply_gamma = |c: f32| -> u8 {
            let clamped = c.clamp(0.0, 1.0);
            let gamma = if clamped <= 0.003_130_8 {
                clamped * 12.92
            } else {
                1.055 * clamped.powf(1.0 / 2.4) - 0.055
            };
            (gamma * 255.0 + 0.5) as u8
        };
        [
            apply_gamma(self.r),
            apply_gamma(self.g),
            apply_gamma(self.b),
        ]
    }

    /// Convert to 10-bit output (values 0–1023).
    #[must_use]
    pub fn to_10bit(self) -> [u16; 3] {
        let conv = |c: f32| -> u16 { (c.clamp(0.0, 1.0) * 1023.0 + 0.5) as u16 };
        [conv(self.r), conv(self.g), conv(self.b)]
    }
}

/// A per-panel calibration offset that adjusts pixel values.
#[derive(Debug, Clone, Copy)]
pub struct PanelCalibrationOffset {
    /// Additive gain offset for red [−1.0, 1.0].
    pub r_gain: f32,
    /// Additive gain offset for green [−1.0, 1.0].
    pub g_gain: f32,
    /// Additive gain offset for blue [−1.0, 1.0].
    pub b_gain: f32,
    /// Brightness multiplier (1.0 = no change).
    pub brightness: f32,
}

impl PanelCalibrationOffset {
    /// Create a new calibration offset.
    #[must_use]
    pub fn new(r_gain: f32, g_gain: f32, b_gain: f32, brightness: f32) -> Self {
        Self {
            r_gain,
            g_gain,
            b_gain,
            brightness: brightness.max(0.0),
        }
    }

    /// Identity calibration (no change).
    #[must_use]
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Apply calibration to a pixel.
    #[must_use]
    pub fn apply(&self, pixel: LinearPixel) -> LinearPixel {
        LinearPixel {
            r: ((pixel.r + self.r_gain) * self.brightness).max(0.0),
            g: ((pixel.g + self.g_gain) * self.brightness).max(0.0),
            b: ((pixel.b + self.b_gain) * self.brightness).max(0.0),
        }
    }
}

impl Default for PanelCalibrationOffset {
    fn default() -> Self {
        Self::identity()
    }
}

/// HDR tone-mapping mode for mapping scene-linear HDR to LED output range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneMappingMode {
    /// No tone mapping — clip values above peak nits.
    Clip,
    /// Simple linear scale to fit peak nits.
    #[default]
    LinearScale,
    /// Reinhard global tone mapping.
    Reinhard,
    /// ACES filmic tone mapping approximation.
    AcesFilmic,
}

/// Applies HDR tone mapping from scene-linear input to LED output range.
#[allow(dead_code)]
pub struct HdrToneMapper {
    /// Peak nits of the LED wall.
    peak_nits: f32,
    /// Tone mapping mode.
    mode: ToneMappingMode,
}

impl HdrToneMapper {
    /// Create a new tone mapper.
    #[must_use]
    pub fn new(peak_nits: f32, mode: ToneMappingMode) -> Self {
        Self { peak_nits, mode }
    }

    /// Map a scene-linear pixel to LED output range [0.0, 1.0].
    #[must_use]
    pub fn map(&self, pixel: LinearPixel) -> LinearPixel {
        match self.mode {
            ToneMappingMode::Clip => pixel.clamp_nits(self.peak_nits).scale(1.0 / self.peak_nits),
            ToneMappingMode::LinearScale => pixel.scale(1.0 / self.peak_nits).clamp_nits(1.0),
            ToneMappingMode::Reinhard => self.reinhard(pixel),
            ToneMappingMode::AcesFilmic => self.aces(pixel),
        }
    }

    fn reinhard(&self, pixel: LinearPixel) -> LinearPixel {
        let scale = 1.0 / self.peak_nits;
        let map = |c: f32| -> f32 {
            let x = c * scale;
            x / (1.0 + x)
        };
        LinearPixel::new(map(pixel.r), map(pixel.g), map(pixel.b))
    }

    fn aces(&self, pixel: LinearPixel) -> LinearPixel {
        // Narkowicz 2015 ACES approximation
        let scale = 1.0 / self.peak_nits;
        let map = |c: f32| -> f32 {
            let x = c * scale;
            let a = 2.51_f32;
            let b = 0.03_f32;
            let c_const = 2.43_f32;
            let d = 0.59_f32;
            let e = 0.14_f32;
            ((x * (a * x + b)) / (x * (c_const * x + d) + e)).clamp(0.0, 1.0)
        };
        LinearPixel::new(map(pixel.r), map(pixel.g), map(pixel.b))
    }

    /// Peak nits configured for this mapper.
    #[must_use]
    pub fn peak_nits(&self) -> f32 {
        self.peak_nits
    }

    /// Tone mapping mode.
    #[must_use]
    pub fn mode(&self) -> ToneMappingMode {
        self.mode
    }
}

/// Precomputed lookup table mapping every global pixel coordinate to its
/// (panel_idx, panel_col, panel_row) triple.
///
/// When the total pixel count fits within `memory_cap_pixels`, the LUT is
/// built eagerly so that subsequent lookups are O(1) array access.  For
/// very large walls it returns `None` from `build`, and the caller should
/// fall back to arithmetic (`PixelMapper::global_to_panel`).
pub struct GlobalPixelLut {
    /// Flat Vec of (panel_idx, panel_col, panel_row) — row-major order.
    entries: Vec<(u32, u32, u32)>,
    frame_width: u32,
    frame_height: u32,
    /// Maximum pixels before refusing to allocate.
    memory_cap_pixels: usize,
}

/// Default memory cap: 32 megapixels (≈ 384 MB at 12 bytes/entry).
pub const DEFAULT_MEMORY_CAP: usize = 32 * 1024 * 1024;

/// Minimal description of a panel needed to build the LUT.
#[derive(Debug, Clone, Copy)]
pub struct PanelDesc {
    /// Horizontal pixel offset of this panel within the global frame.
    pub x_offset: u32,
    /// Vertical pixel offset of this panel within the global frame.
    pub y_offset: u32,
    /// Width of this panel in pixels.
    pub width: u32,
    /// Height of this panel in pixels.
    pub height: u32,
}

impl GlobalPixelLut {
    /// Build the LUT from a slice of panels and the overall frame dimensions.
    ///
    /// Returns `None` when total pixel count exceeds `DEFAULT_MEMORY_CAP`.
    #[must_use]
    pub fn build(panels: &[PanelDesc], frame_width: u32, frame_height: u32) -> Option<Self> {
        Self::build_with_cap(panels, frame_width, frame_height, DEFAULT_MEMORY_CAP)
    }

    /// Build with an explicit memory cap.
    ///
    /// Returns `None` when total pixel count exceeds `memory_cap_pixels`.
    #[must_use]
    pub fn build_with_cap(
        panels: &[PanelDesc],
        frame_width: u32,
        frame_height: u32,
        memory_cap_pixels: usize,
    ) -> Option<Self> {
        let total = frame_width as usize * frame_height as usize;
        if total > memory_cap_pixels {
            return None;
        }

        // Fill entries with a sentinel meaning "no panel" (u32::MAX).
        let sentinel = (u32::MAX, u32::MAX, u32::MAX);
        let mut entries = vec![sentinel; total];

        for (idx, panel) in panels.iter().enumerate() {
            let panel_idx = idx as u32;
            for row in 0..panel.height {
                for col in 0..panel.width {
                    let gx = panel.x_offset + col;
                    let gy = panel.y_offset + row;
                    if gx < frame_width && gy < frame_height {
                        let gi = gy as usize * frame_width as usize + gx as usize;
                        entries[gi] = (panel_idx, col, row);
                    }
                }
            }
        }

        Some(Self {
            entries,
            frame_width,
            frame_height,
            memory_cap_pixels,
        })
    }

    /// Look up a global pixel coordinate.
    ///
    /// Returns `None` when the coordinate is out of range or lies in an
    /// uncovered region between panels.
    #[must_use]
    pub fn lookup(&self, global_x: u32, global_y: u32) -> Option<(u32, u32, u32)> {
        if global_x >= self.frame_width || global_y >= self.frame_height {
            return None;
        }
        let idx = global_y as usize * self.frame_width as usize + global_x as usize;
        let entry = self.entries[idx];
        if entry.0 == u32::MAX {
            None
        } else {
            Some(entry)
        }
    }

    /// Frame width this LUT was built for.
    #[must_use]
    pub fn frame_width(&self) -> u32 {
        self.frame_width
    }

    /// Frame height this LUT was built for.
    #[must_use]
    pub fn frame_height(&self) -> u32 {
        self.frame_height
    }

    /// Memory cap (pixels) used to gate construction.
    #[must_use]
    pub fn memory_cap_pixels(&self) -> usize {
        self.memory_cap_pixels
    }

    /// Number of entries in the LUT (frame_width × frame_height).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Maps a 2D buffer of pixels to panel-local pixel buffers.
///
/// When the wall fits within `DEFAULT_MEMORY_CAP` pixels, a `GlobalPixelLut`
/// is built at construction time and used for O(1) lookups.  For larger walls
/// the arithmetic path is used instead.
pub struct PixelMapper {
    /// Total wall width in pixels.
    wall_width: u32,
    /// Total wall height in pixels.
    wall_height: u32,
    /// Width of each panel tile.
    tile_width: u32,
    /// Height of each panel tile.
    tile_height: u32,
    /// Optional precomputed LUT (available when wall fits in memory cap).
    lut: Option<GlobalPixelLut>,
}

impl PixelMapper {
    /// Create a new pixel mapper.
    ///
    /// A `GlobalPixelLut` is built automatically when the total pixel count
    /// is within `DEFAULT_MEMORY_CAP`.  For larger walls the LUT is omitted
    /// and arithmetic is used instead.
    #[must_use]
    pub fn new(wall_width: u32, wall_height: u32, tile_width: u32, tile_height: u32) -> Self {
        let lut = if tile_width > 0 && tile_height > 0 {
            // Build a uniform-tile panel descriptor list.
            let cols = wall_width.div_ceil(tile_width);
            let rows = wall_height.div_ceil(tile_height);
            let mut panels: Vec<PanelDesc> = Vec::with_capacity((cols * rows) as usize);
            for row in 0..rows {
                for col in 0..cols {
                    let x_offset = col * tile_width;
                    let y_offset = row * tile_height;
                    let w = tile_width.min(wall_width.saturating_sub(x_offset));
                    let h = tile_height.min(wall_height.saturating_sub(y_offset));
                    panels.push(PanelDesc {
                        x_offset,
                        y_offset,
                        width: w,
                        height: h,
                    });
                }
            }
            GlobalPixelLut::build(&panels, wall_width, wall_height)
        } else {
            None
        };

        Self {
            wall_width,
            wall_height,
            tile_width,
            tile_height,
            lut,
        }
    }

    /// Number of panel columns.
    #[must_use]
    pub fn panel_cols(&self) -> u32 {
        if self.tile_width == 0 {
            return 0;
        }
        self.wall_width.div_ceil(self.tile_width)
    }

    /// Number of panel rows.
    #[must_use]
    pub fn panel_rows(&self) -> u32 {
        if self.tile_height == 0 {
            return 0;
        }
        self.wall_height.div_ceil(self.tile_height)
    }

    /// Convert a global pixel coordinate to (col, row, `local_x`, `local_y`).
    ///
    /// Uses the precomputed LUT when available, otherwise falls back to
    /// arithmetic division.
    #[must_use]
    pub fn global_to_panel(&self, gx: u32, gy: u32) -> Option<(u32, u32, u32, u32)> {
        if self.tile_width == 0 || self.tile_height == 0 {
            return None;
        }
        if gx >= self.wall_width || gy >= self.wall_height {
            return None;
        }

        // Fast path: LUT available.
        if let Some(ref lut) = self.lut {
            // LUT entry is (panel_idx_in_col_row_order, panel_col, panel_row).
            // panel_idx = row * num_cols + col  →  col = idx % num_cols
            if let Some((pidx, local_x, local_y)) = lut.lookup(gx, gy) {
                let num_cols = self.panel_cols();
                let col = pidx % num_cols;
                let row = pidx / num_cols;
                return Some((col, row, local_x, local_y));
            }
            return None;
        }

        // Arithmetic fallback.
        let col = gx / self.tile_width;
        let row = gy / self.tile_height;
        let local_x = gx % self.tile_width;
        let local_y = gy % self.tile_height;
        Some((col, row, local_x, local_y))
    }

    /// Convert (col, row, `local_x`, `local_y`) back to global coordinates.
    #[must_use]
    pub fn panel_to_global(&self, col: u32, row: u32, local_x: u32, local_y: u32) -> (u32, u32) {
        (
            col * self.tile_width + local_x,
            row * self.tile_height + local_y,
        )
    }

    /// Wall resolution.
    #[must_use]
    pub fn resolution(&self) -> (u32, u32) {
        (self.wall_width, self.wall_height)
    }

    /// Returns a reference to the precomputed LUT, if available.
    #[must_use]
    pub fn lut(&self) -> Option<&GlobalPixelLut> {
        self.lut.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_pixel_clamp_nits() {
        let px = LinearPixel::new(2000.0, 500.0, -10.0);
        let clamped = px.clamp_nits(1000.0);
        assert_eq!(clamped.r, 1000.0);
        assert_eq!(clamped.g, 500.0);
        assert_eq!(clamped.b, 0.0);
    }

    #[test]
    fn test_linear_pixel_scale() {
        let px = LinearPixel::new(1.0, 0.5, 0.25);
        let scaled = px.scale(2.0);
        assert!((scaled.r - 2.0).abs() < 1e-6);
        assert!((scaled.g - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_pixel_to_srgb8_black() {
        let px = LinearPixel::new(0.0, 0.0, 0.0);
        assert_eq!(px.to_srgb8(), [0, 0, 0]);
    }

    #[test]
    fn test_linear_pixel_to_srgb8_white() {
        let px = LinearPixel::new(1.0, 1.0, 1.0);
        assert_eq!(px.to_srgb8(), [255, 255, 255]);
    }

    #[test]
    fn test_linear_pixel_to_10bit_white() {
        let px = LinearPixel::new(1.0, 1.0, 1.0);
        assert_eq!(px.to_10bit(), [1023, 1023, 1023]);
    }

    #[test]
    fn test_linear_pixel_to_10bit_black() {
        let px = LinearPixel::new(0.0, 0.0, 0.0);
        assert_eq!(px.to_10bit(), [0, 0, 0]);
    }

    #[test]
    fn test_calibration_identity() {
        let cal = PanelCalibrationOffset::identity();
        let px = LinearPixel::new(0.5, 0.5, 0.5);
        let out = cal.apply(px);
        assert!((out.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_brightness() {
        let cal = PanelCalibrationOffset::new(0.0, 0.0, 0.0, 0.5);
        let px = LinearPixel::new(1.0, 1.0, 1.0);
        let out = cal.apply(px);
        assert!((out.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_gain_offset() {
        let cal = PanelCalibrationOffset::new(0.1, -0.1, 0.0, 1.0);
        let px = LinearPixel::new(0.5, 0.5, 0.5);
        let out = cal.apply(px);
        assert!((out.r - 0.6).abs() < 1e-5);
        assert!((out.g - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_calibration_no_negative_output() {
        let cal = PanelCalibrationOffset::new(-1.0, -1.0, -1.0, 1.0);
        let px = LinearPixel::new(0.0, 0.0, 0.0);
        let out = cal.apply(px);
        assert!(out.r >= 0.0);
        assert!(out.g >= 0.0);
        assert!(out.b >= 0.0);
    }

    #[test]
    fn test_tone_mapping_clip() {
        let mapper = HdrToneMapper::new(1000.0, ToneMappingMode::Clip);
        let px = LinearPixel::new(500.0, 1500.0, 0.0);
        let out = mapper.map(px);
        assert!((out.r - 0.5).abs() < 1e-5);
        assert!((out.g - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tone_mapping_linear_scale() {
        let mapper = HdrToneMapper::new(1000.0, ToneMappingMode::LinearScale);
        let px = LinearPixel::new(500.0, 1000.0, 0.0);
        let out = mapper.map(px);
        assert!((out.r - 0.5).abs() < 1e-5);
        assert!((out.g - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tone_mapping_reinhard_positive() {
        let mapper = HdrToneMapper::new(1000.0, ToneMappingMode::Reinhard);
        let px = LinearPixel::new(500.0, 500.0, 500.0);
        let out = mapper.map(px);
        // Reinhard: x/(1+x) where x = 500/1000 = 0.5 → 0.5/1.5 ≈ 0.333
        assert!(out.r > 0.0 && out.r < 1.0);
    }

    #[test]
    fn test_tone_mapping_aces_positive() {
        let mapper = HdrToneMapper::new(1000.0, ToneMappingMode::AcesFilmic);
        let px = LinearPixel::new(100.0, 500.0, 1000.0);
        let out = mapper.map(px);
        assert!(out.r >= 0.0 && out.r <= 1.0);
        assert!(out.g >= 0.0 && out.g <= 1.0);
        assert!(out.b >= 0.0 && out.b <= 1.0);
    }

    #[test]
    fn test_pixel_mapper_panel_cols_rows() {
        let mapper = PixelMapper::new(1920, 1080, 256, 128);
        // ceil(1920/256) = 8, ceil(1080/128) = 9 (since 1080 = 8*128 + 56)
        assert_eq!(mapper.panel_cols(), 8);
        assert_eq!(mapper.panel_rows(), 9);
    }

    #[test]
    fn test_pixel_mapper_global_to_panel() {
        let mapper = PixelMapper::new(512, 256, 256, 128);
        let result = mapper.global_to_panel(300, 150);
        assert!(result.is_some());
        let (col, row, lx, ly) = result.expect("should succeed in test");
        assert_eq!(col, 1);
        assert_eq!(row, 1);
        assert_eq!(lx, 300 - 256);
        assert_eq!(ly, 150 - 128);
    }

    #[test]
    fn test_pixel_mapper_global_to_panel_out_of_bounds() {
        let mapper = PixelMapper::new(512, 256, 256, 128);
        assert!(mapper.global_to_panel(512, 0).is_none());
        assert!(mapper.global_to_panel(0, 256).is_none());
    }

    #[test]
    fn test_pixel_mapper_panel_to_global_roundtrip() {
        let mapper = PixelMapper::new(1024, 512, 256, 128);
        let (gx, gy) = mapper.panel_to_global(2, 1, 50, 30);
        let result = mapper.global_to_panel(gx, gy);
        assert!(result.is_some());
        let (col, row, lx, ly) = result.expect("should succeed in test");
        assert_eq!((col, row, lx, ly), (2, 1, 50, 30));
    }
}
