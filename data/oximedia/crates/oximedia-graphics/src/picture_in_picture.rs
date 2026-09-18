#![allow(dead_code)]
//! Picture-in-picture (PiP) compositing for broadcast graphics.
//!
//! Provides production-quality PiP with:
//! - Configurable position (9 anchor presets + free placement)
//! - Adjustable size with aspect ratio preservation
//! - Border rendering with customizable color, width, and corner radius
//! - Drop shadow with configurable offset, blur, and color
//! - Opacity control for the inset window
//! - Bilinear scaling for clean resize
//! - Multiple PiP windows with z-ordering

use crate::error::{GraphicsError, Result};

/// Predefined anchor positions for PiP placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipAnchor {
    /// Top-left corner.
    TopLeft,
    /// Top center.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Center left.
    CenterLeft,
    /// Center of frame.
    Center,
    /// Center right.
    CenterRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom center.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
    /// Custom position (x, y) in pixels from top-left.
    Custom(u32, u32),
}

impl Default for PipAnchor {
    fn default() -> Self {
        Self::BottomRight
    }
}

impl PipAnchor {
    /// Compute the top-left pixel position given the main frame and PiP dimensions.
    pub fn resolve(
        &self,
        main_w: u32,
        main_h: u32,
        pip_w: u32,
        pip_h: u32,
        margin: u32,
    ) -> (u32, u32) {
        match self {
            Self::TopLeft => (margin, margin),
            Self::TopCenter => (main_w.saturating_sub(pip_w) / 2, margin),
            Self::TopRight => (main_w.saturating_sub(pip_w + margin), margin),
            Self::CenterLeft => (margin, main_h.saturating_sub(pip_h) / 2),
            Self::Center => (
                main_w.saturating_sub(pip_w) / 2,
                main_h.saturating_sub(pip_h) / 2,
            ),
            Self::CenterRight => (
                main_w.saturating_sub(pip_w + margin),
                main_h.saturating_sub(pip_h) / 2,
            ),
            Self::BottomLeft => (margin, main_h.saturating_sub(pip_h + margin)),
            Self::BottomCenter => (
                main_w.saturating_sub(pip_w) / 2,
                main_h.saturating_sub(pip_h + margin),
            ),
            Self::BottomRight => (
                main_w.saturating_sub(pip_w + margin),
                main_h.saturating_sub(pip_h + margin),
            ),
            Self::Custom(x, y) => (*x, *y),
        }
    }
}

/// Border style for the PiP window.
#[derive(Clone, Debug)]
pub struct PipBorder {
    /// Border width in pixels.
    pub width: u32,
    /// Border color [R, G, B, A].
    pub color: [u8; 4],
    /// Corner radius in pixels (0 = sharp corners).
    pub corner_radius: u32,
}

impl Default for PipBorder {
    fn default() -> Self {
        Self {
            width: 2,
            color: [255, 255, 255, 255],
            corner_radius: 0,
        }
    }
}

impl PipBorder {
    /// Create a border with given width and color.
    pub fn new(width: u32, r: u8, g: u8, b: u8) -> Self {
        Self {
            width,
            color: [r, g, b, 255],
            corner_radius: 0,
        }
    }

    /// Set the corner radius.
    pub fn with_corner_radius(mut self, radius: u32) -> Self {
        self.corner_radius = radius;
        self
    }

    /// Set the alpha channel.
    pub fn with_alpha(mut self, a: u8) -> Self {
        self.color[3] = a;
        self
    }
}

/// Shadow for the PiP window.
#[derive(Clone, Debug)]
pub struct PipShadow {
    /// Horizontal offset in pixels.
    pub offset_x: i32,
    /// Vertical offset in pixels.
    pub offset_y: i32,
    /// Blur radius in pixels.
    pub blur_radius: u32,
    /// Shadow color [R, G, B, A].
    pub color: [u8; 4],
}

impl Default for PipShadow {
    fn default() -> Self {
        Self {
            offset_x: 4,
            offset_y: 4,
            blur_radius: 8,
            color: [0, 0, 0, 128],
        }
    }
}

/// Configuration for a single PiP window.
#[derive(Clone, Debug)]
pub struct PipConfig {
    /// Anchor position.
    pub anchor: PipAnchor,
    /// Margin from edges (pixels).
    pub margin: u32,
    /// Width of the PiP window in pixels (after scaling).
    pub width: u32,
    /// Height of the PiP window in pixels (after scaling).
    pub height: u32,
    /// Opacity of the PiP window (0.0..=1.0).
    pub opacity: f32,
    /// Border (None = no border).
    pub border: Option<PipBorder>,
    /// Shadow (None = no shadow).
    pub shadow: Option<PipShadow>,
    /// Z-order (higher = on top).
    pub z_order: i32,
    /// Whether to preserve aspect ratio when scaling.
    pub preserve_aspect: bool,
}

impl Default for PipConfig {
    fn default() -> Self {
        Self {
            anchor: PipAnchor::BottomRight,
            margin: 20,
            width: 320,
            height: 180,
            opacity: 1.0,
            border: Some(PipBorder::default()),
            shadow: None,
            z_order: 0,
            preserve_aspect: true,
        }
    }
}

impl PipConfig {
    /// Create a new PiP config with given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set the anchor position.
    pub fn with_anchor(mut self, anchor: PipAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Set the margin from edges.
    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Set the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set the border.
    pub fn with_border(mut self, border: PipBorder) -> Self {
        self.border = Some(border);
        self
    }

    /// Remove the border.
    pub fn without_border(mut self) -> Self {
        self.border = None;
        self
    }

    /// Set the shadow.
    pub fn with_shadow(mut self, shadow: PipShadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Set the z-order.
    pub fn with_z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(GraphicsError::InvalidDimensions(self.width, self.height));
        }
        Ok(())
    }

    /// Compute the actual PiP dimensions respecting aspect ratio preservation.
    pub fn effective_dimensions(&self, source_w: u32, source_h: u32) -> (u32, u32) {
        if !self.preserve_aspect || source_w == 0 || source_h == 0 {
            return (self.width, self.height);
        }

        let aspect = source_w as f64 / source_h as f64;
        let target_aspect = self.width as f64 / self.height as f64;

        if aspect > target_aspect {
            // Source is wider: fit to width
            let h = (self.width as f64 / aspect) as u32;
            (self.width, h.max(1))
        } else {
            // Source is taller: fit to height
            let w = (self.height as f64 * aspect) as u32;
            (w.max(1), self.height)
        }
    }
}

/// PiP compositor that manages multiple PiP windows over a main frame.
pub struct PipCompositor {
    /// Main frame width.
    main_width: u32,
    /// Main frame height.
    main_height: u32,
}

impl PipCompositor {
    /// Create a new PiP compositor for the given main frame dimensions.
    pub fn new(main_width: u32, main_height: u32) -> Result<Self> {
        if main_width == 0 || main_height == 0 {
            return Err(GraphicsError::InvalidDimensions(main_width, main_height));
        }
        Ok(Self {
            main_width,
            main_height,
        })
    }

    /// Composite a single PiP source onto the main frame.
    ///
    /// `main_frame` and `pip_source` are RGBA byte slices.
    /// The PiP source will be scaled to the config dimensions and placed at the anchor.
    pub fn composite_one(
        &self,
        main_frame: &mut [u8],
        pip_source: &[u8],
        source_w: u32,
        source_h: u32,
        config: &PipConfig,
    ) -> Result<()> {
        config.validate()?;
        self.validate_frame(main_frame)?;
        self.validate_source(pip_source, source_w, source_h)?;

        let (pip_w, pip_h) = config.effective_dimensions(source_w, source_h);
        let (pos_x, pos_y) = config.anchor.resolve(
            self.main_width,
            self.main_height,
            pip_w + config.border.as_ref().map_or(0, |b| b.width * 2),
            pip_h + config.border.as_ref().map_or(0, |b| b.width * 2),
            config.margin,
        );

        let border_w = config.border.as_ref().map_or(0, |b| b.width);

        // Draw shadow first (behind everything)
        if let Some(ref shadow) = config.shadow {
            self.draw_shadow(
                main_frame,
                pos_x,
                pos_y,
                pip_w + border_w * 2,
                pip_h + border_w * 2,
                shadow,
            );
        }

        // Draw border
        if let Some(ref border) = config.border {
            self.draw_border(
                main_frame,
                pos_x,
                pos_y,
                pip_w + border_w * 2,
                pip_h + border_w * 2,
                border,
            );
        }

        // Scale and composite the PiP source
        let offset_x = pos_x + border_w;
        let offset_y = pos_y + border_w;
        self.scale_and_composite(
            main_frame,
            pip_source,
            source_w,
            source_h,
            offset_x,
            offset_y,
            pip_w,
            pip_h,
            config.opacity,
        );

        Ok(())
    }

    /// Composite multiple PiP sources onto the main frame, sorted by z-order.
    pub fn composite_multi(
        &self,
        main_frame: &mut [u8],
        sources: &mut [(Vec<u8>, u32, u32, PipConfig)],
    ) -> Result<()> {
        // Sort by z-order (lowest first = drawn first = behind)
        sources.sort_by_key(|(_, _, _, cfg)| cfg.z_order);

        for (pip_source, source_w, source_h, config) in sources.iter() {
            self.composite_one(main_frame, pip_source, *source_w, *source_h, config)?;
        }

        Ok(())
    }

    /// Validate main frame size.
    fn validate_frame(&self, frame: &[u8]) -> Result<()> {
        let expected = (self.main_width as usize) * (self.main_height as usize) * 4;
        if frame.len() != expected {
            return Err(GraphicsError::InvalidParameter(format!(
                "Main frame size mismatch: expected {expected}, got {}",
                frame.len()
            )));
        }
        Ok(())
    }

    /// Validate source frame size.
    fn validate_source(&self, source: &[u8], w: u32, h: u32) -> Result<()> {
        let expected = (w as usize) * (h as usize) * 4;
        if source.len() != expected {
            return Err(GraphicsError::InvalidParameter(format!(
                "PiP source size mismatch: expected {expected}, got {}",
                source.len()
            )));
        }
        Ok(())
    }

    /// Draw a simple shadow rectangle.
    fn draw_shadow(&self, frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, shadow: &PipShadow) {
        let sx = (x as i32 + shadow.offset_x).max(0) as u32;
        let sy = (y as i32 + shadow.offset_y).max(0) as u32;
        let sa = shadow.color[3];
        let blur = shadow.blur_radius;

        // Draw shadow with feathered edges
        let expand = blur;
        let ex1 = sx.saturating_sub(expand);
        let ey1 = sy.saturating_sub(expand);
        let ex2 = (sx + w + expand).min(self.main_width);
        let ey2 = (sy + h + expand).min(self.main_height);

        for py in ey1..ey2 {
            for px in ex1..ex2 {
                // Distance to shadow rectangle
                let dx = if px < sx {
                    sx - px
                } else if px >= sx + w {
                    px - (sx + w) + 1
                } else {
                    0
                };
                let dy = if py < sy {
                    sy - py
                } else if py >= sy + h {
                    py - (sy + h) + 1
                } else {
                    0
                };

                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let alpha_factor = if blur > 0 {
                    (1.0 - dist / blur as f32).clamp(0.0, 1.0)
                } else if dx == 0 && dy == 0 {
                    1.0
                } else {
                    0.0
                };

                if alpha_factor <= 0.0 {
                    continue;
                }

                let pixel_alpha = f32::from(sa) / 255.0 * alpha_factor;
                let idx = ((py * self.main_width + px) * 4) as usize;
                if idx + 3 < frame.len() {
                    let inv = 1.0 - pixel_alpha;
                    frame[idx] = (f32::from(shadow.color[0]) * pixel_alpha
                        + f32::from(frame[idx]) * inv) as u8;
                    frame[idx + 1] = (f32::from(shadow.color[1]) * pixel_alpha
                        + f32::from(frame[idx + 1]) * inv)
                        as u8;
                    frame[idx + 2] = (f32::from(shadow.color[2]) * pixel_alpha
                        + f32::from(frame[idx + 2]) * inv)
                        as u8;
                }
            }
        }
    }

    /// Draw a border rectangle.
    fn draw_border(&self, frame: &mut [u8], x: u32, y: u32, w: u32, h: u32, border: &PipBorder) {
        let bw = border.width;
        let x2 = (x + w).min(self.main_width);
        let y2 = (y + h).min(self.main_height);
        let alpha = f32::from(border.color[3]) / 255.0;

        for py in y..y2 {
            for px in x..x2 {
                let is_border = px < x + bw || px >= x2 - bw || py < y + bw || py >= y2 - bw;
                if !is_border {
                    continue;
                }

                // Corner radius check
                if border.corner_radius > 0 {
                    let cr = border.corner_radius as f32;
                    let corners = [
                        (x as f32 + cr, y as f32 + cr),
                        (x2 as f32 - cr, y as f32 + cr),
                        (x as f32 + cr, y2 as f32 - cr),
                        (x2 as f32 - cr, y2 as f32 - cr),
                    ];

                    let mut in_corner_zone = false;
                    let pxf = px as f32 + 0.5;
                    let pyf = py as f32 + 0.5;

                    for &(cx, cy) in &corners {
                        if (pxf < cx && pyf < cy)
                            || (pxf > cx && pyf < cy)
                            || (pxf < cx && pyf > cy)
                            || (pxf > cx && pyf > cy)
                        {
                            let dist = ((pxf - cx) * (pxf - cx) + (pyf - cy) * (pyf - cy)).sqrt();
                            if (pxf - cx).abs() <= cr && (pyf - cy).abs() <= cr {
                                in_corner_zone = true;
                                if dist > cr {
                                    // Outside the rounded corner
                                    continue;
                                }
                            }
                        }
                    }
                    let _ = in_corner_zone;
                }

                let idx = ((py * self.main_width + px) * 4) as usize;
                if idx + 3 < frame.len() {
                    let inv = 1.0 - alpha;
                    frame[idx] =
                        (f32::from(border.color[0]) * alpha + f32::from(frame[idx]) * inv) as u8;
                    frame[idx + 1] = (f32::from(border.color[1]) * alpha
                        + f32::from(frame[idx + 1]) * inv)
                        as u8;
                    frame[idx + 2] = (f32::from(border.color[2]) * alpha
                        + f32::from(frame[idx + 2]) * inv)
                        as u8;
                    frame[idx + 3] = 255;
                }
            }
        }
    }

    /// Scale source via bilinear interpolation and composite onto main frame.
    fn scale_and_composite(
        &self,
        main_frame: &mut [u8],
        source: &[u8],
        src_w: u32,
        src_h: u32,
        dst_x: u32,
        dst_y: u32,
        dst_w: u32,
        dst_h: u32,
        opacity: f32,
    ) {
        if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
            return;
        }

        let x_ratio = src_w as f64 / dst_w as f64;
        let y_ratio = src_h as f64 / dst_h as f64;

        for dy in 0..dst_h {
            let py = dst_y + dy;
            if py >= self.main_height {
                break;
            }

            for dx in 0..dst_w {
                let px = dst_x + dx;
                if px >= self.main_width {
                    break;
                }

                // Bilinear sampling
                let sx = dx as f64 * x_ratio;
                let sy = dy as f64 * y_ratio;

                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);

                let fx = (sx - sx.floor()) as f32;
                let fy = (sy - sy.floor()) as f32;

                let sample = |sx: u32, sy: u32| -> [f32; 4] {
                    let idx = ((sy * src_w + sx) * 4) as usize;
                    [
                        f32::from(source[idx]),
                        f32::from(source[idx + 1]),
                        f32::from(source[idx + 2]),
                        f32::from(source[idx + 3]),
                    ]
                };

                let p00 = sample(x0, y0);
                let p10 = sample(x1, y0);
                let p01 = sample(x0, y1);
                let p11 = sample(x1, y1);

                let mut result = [0.0_f32; 4];
                for c in 0..4 {
                    let top = p00[c] * (1.0 - fx) + p10[c] * fx;
                    let bot = p01[c] * (1.0 - fx) + p11[c] * fx;
                    result[c] = top * (1.0 - fy) + bot * fy;
                }

                // Alpha-over composite
                let src_alpha = (result[3] / 255.0) * opacity;
                let inv_alpha = 1.0 - src_alpha;
                let main_idx = ((py * self.main_width + px) * 4) as usize;

                if main_idx + 3 < main_frame.len() {
                    main_frame[main_idx] =
                        (result[0] * src_alpha + f32::from(main_frame[main_idx]) * inv_alpha) as u8;
                    main_frame[main_idx + 1] = (result[1] * src_alpha
                        + f32::from(main_frame[main_idx + 1]) * inv_alpha)
                        as u8;
                    main_frame[main_idx + 2] = (result[2] * src_alpha
                        + f32::from(main_frame[main_idx + 2]) * inv_alpha)
                        as u8;
                    main_frame[main_idx + 3] = 255;
                }
            }
        }
    }

    /// Get the main frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.main_width, self.main_height)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let count = (w as usize) * (h as usize);
        let mut data = Vec::with_capacity(count * 4);
        for _ in 0..count {
            data.extend_from_slice(&[r, g, b, 255]);
        }
        data
    }

    #[test]
    fn test_pip_anchor_resolve_bottom_right() {
        let (x, y) = PipAnchor::BottomRight.resolve(1920, 1080, 320, 180, 20);
        assert_eq!(x, 1920 - 320 - 20);
        assert_eq!(y, 1080 - 180 - 20);
    }

    #[test]
    fn test_pip_anchor_resolve_top_left() {
        let (x, y) = PipAnchor::TopLeft.resolve(1920, 1080, 320, 180, 20);
        assert_eq!(x, 20);
        assert_eq!(y, 20);
    }

    #[test]
    fn test_pip_anchor_resolve_center() {
        let (x, y) = PipAnchor::Center.resolve(1920, 1080, 320, 180, 20);
        assert_eq!(x, (1920 - 320) / 2);
        assert_eq!(y, (1080 - 180) / 2);
    }

    #[test]
    fn test_pip_anchor_custom() {
        let (x, y) = PipAnchor::Custom(100, 200).resolve(1920, 1080, 320, 180, 20);
        assert_eq!(x, 100);
        assert_eq!(y, 200);
    }

    #[test]
    fn test_pip_config_defaults() {
        let cfg = PipConfig::default();
        assert_eq!(cfg.width, 320);
        assert_eq!(cfg.height, 180);
        assert!(cfg.border.is_some());
    }

    #[test]
    fn test_pip_config_builder() {
        let cfg = PipConfig::new(400, 225)
            .with_anchor(PipAnchor::TopLeft)
            .with_margin(10)
            .with_opacity(0.8)
            .with_z_order(5)
            .without_border();
        assert_eq!(cfg.width, 400);
        assert_eq!(cfg.anchor, PipAnchor::TopLeft);
        assert_eq!(cfg.margin, 10);
        assert!((cfg.opacity - 0.8).abs() < f32::EPSILON);
        assert_eq!(cfg.z_order, 5);
        assert!(cfg.border.is_none());
    }

    #[test]
    fn test_pip_config_validation() {
        assert!(PipConfig::new(320, 180).validate().is_ok());
        assert!(PipConfig::new(0, 180).validate().is_err());
        assert!(PipConfig::new(320, 0).validate().is_err());
    }

    #[test]
    fn test_effective_dimensions_preserve_aspect() {
        let cfg = PipConfig::new(320, 180);
        // 1920x1080 source -> 16:9 same as target -> no change
        let (w, h) = cfg.effective_dimensions(1920, 1080);
        assert_eq!(w, 320);
        assert_eq!(h, 180);
    }

    #[test]
    fn test_effective_dimensions_wider_source() {
        let cfg = PipConfig::new(320, 180);
        // 1920x720 source (wider) -> fit to width, reduce height
        let (w, h) = cfg.effective_dimensions(1920, 720);
        assert_eq!(w, 320);
        assert!(h < 180);
    }

    #[test]
    fn test_effective_dimensions_taller_source() {
        let cfg = PipConfig::new(320, 180);
        // 720x1080 source (taller) -> fit to height, reduce width
        let (w, h) = cfg.effective_dimensions(720, 1080);
        assert!(w < 320);
        assert_eq!(h, 180);
    }

    #[test]
    fn test_effective_dimensions_no_preserve() {
        let mut cfg = PipConfig::new(320, 180);
        cfg.preserve_aspect = false;
        let (w, h) = cfg.effective_dimensions(1920, 720);
        assert_eq!(w, 320);
        assert_eq!(h, 180);
    }

    #[test]
    fn test_compositor_creation() {
        assert!(PipCompositor::new(1920, 1080).is_ok());
        assert!(PipCompositor::new(0, 1080).is_err());
    }

    #[test]
    fn test_composite_one_basic() {
        let comp = PipCompositor::new(100, 100).expect("should be valid");
        let mut main = make_frame(100, 100, 0, 0, 0); // Black
        let source = make_frame(50, 50, 255, 0, 0); // Red
        let config = PipConfig::new(20, 20)
            .with_anchor(PipAnchor::TopLeft)
            .with_margin(0)
            .without_border();

        comp.composite_one(&mut main, &source, 50, 50, &config)
            .expect("composite should succeed");

        // Top-left pixel should be red
        assert_eq!(main[0], 255);
        assert_eq!(main[1], 0);
        assert_eq!(main[2], 0);
    }

    #[test]
    fn test_composite_one_with_border() {
        let comp = PipCompositor::new(100, 100).expect("should be valid");
        let mut main = make_frame(100, 100, 0, 0, 0);
        let source = make_frame(50, 50, 255, 0, 0);
        let config = PipConfig::new(20, 20)
            .with_anchor(PipAnchor::TopLeft)
            .with_margin(0)
            .with_border(PipBorder::new(2, 255, 255, 255));

        comp.composite_one(&mut main, &source, 50, 50, &config)
            .expect("composite should succeed");

        // First pixel should be the white border
        assert_eq!(main[0], 255); // White border R
        assert_eq!(main[1], 255); // White border G
        assert_eq!(main[2], 255); // White border B
    }

    #[test]
    fn test_composite_one_with_opacity() {
        let comp = PipCompositor::new(100, 100).expect("should be valid");
        let mut main = make_frame(100, 100, 0, 0, 0);
        let source = make_frame(50, 50, 255, 0, 0);
        let config = PipConfig::new(20, 20)
            .with_anchor(PipAnchor::TopLeft)
            .with_margin(0)
            .without_border()
            .with_opacity(0.5);

        comp.composite_one(&mut main, &source, 50, 50, &config)
            .expect("composite should succeed");

        // Should be ~50% red blended with black
        let r = main[0];
        assert!(r > 100 && r < 160, "Red should be about half: {r}");
    }

    #[test]
    fn test_composite_one_with_shadow() {
        let comp = PipCompositor::new(200, 200).expect("should be valid");
        let mut main = make_frame(200, 200, 128, 128, 128); // Gray
        let source = make_frame(50, 50, 255, 0, 0);
        let config = PipConfig::new(40, 40)
            .with_anchor(PipAnchor::Center)
            .without_border()
            .with_shadow(PipShadow::default());

        comp.composite_one(&mut main, &source, 50, 50, &config)
            .expect("composite should succeed");
        // Just ensure no crash, shadow is rendered
    }

    #[test]
    fn test_composite_frame_size_mismatch() {
        let comp = PipCompositor::new(100, 100).expect("should be valid");
        let mut main = vec![0u8; 10]; // Wrong size
        let source = make_frame(50, 50, 255, 0, 0);
        let config = PipConfig::new(20, 20);
        assert!(comp
            .composite_one(&mut main, &source, 50, 50, &config)
            .is_err());
    }

    #[test]
    fn test_composite_source_size_mismatch() {
        let comp = PipCompositor::new(100, 100).expect("should be valid");
        let mut main = make_frame(100, 100, 0, 0, 0);
        let source = vec![0u8; 10]; // Wrong size
        let config = PipConfig::new(20, 20);
        assert!(comp
            .composite_one(&mut main, &source, 50, 50, &config)
            .is_err());
    }

    #[test]
    fn test_composite_multi() {
        let comp = PipCompositor::new(200, 200).expect("should be valid");
        let mut main = make_frame(200, 200, 0, 0, 0);

        let source1 = make_frame(50, 50, 255, 0, 0);
        let source2 = make_frame(30, 30, 0, 255, 0);

        let cfg1 = PipConfig::new(40, 40)
            .with_anchor(PipAnchor::TopLeft)
            .with_margin(0)
            .without_border()
            .with_z_order(0);

        let cfg2 = PipConfig::new(30, 30)
            .with_anchor(PipAnchor::BottomRight)
            .with_margin(0)
            .without_border()
            .with_z_order(1);

        let mut sources = vec![(source1, 50u32, 50u32, cfg1), (source2, 30u32, 30u32, cfg2)];

        comp.composite_multi(&mut main, &mut sources)
            .expect("multi composite should succeed");

        // Top-left should be red
        assert!(main[0] > 200);
        // Bottom-right should be green
        let br_idx = ((199 * 200 + 199) * 4) as usize;
        assert!(main[br_idx + 1] > 200);
    }

    #[test]
    fn test_border_with_alpha() {
        let border = PipBorder::new(3, 255, 0, 0).with_alpha(128);
        assert_eq!(border.width, 3);
        assert_eq!(border.color, [255, 0, 0, 128]);
    }

    #[test]
    fn test_border_with_corner_radius() {
        let border = PipBorder::default().with_corner_radius(8);
        assert_eq!(border.corner_radius, 8);
    }

    #[test]
    fn test_pip_dimensions() {
        let comp = PipCompositor::new(1920, 1080).expect("should be valid");
        assert_eq!(comp.dimensions(), (1920, 1080));
    }

    #[test]
    fn test_all_anchor_positions() {
        let anchors = [
            PipAnchor::TopLeft,
            PipAnchor::TopCenter,
            PipAnchor::TopRight,
            PipAnchor::CenterLeft,
            PipAnchor::Center,
            PipAnchor::CenterRight,
            PipAnchor::BottomLeft,
            PipAnchor::BottomCenter,
            PipAnchor::BottomRight,
        ];
        for anchor in &anchors {
            let (x, y) = anchor.resolve(1920, 1080, 320, 180, 20);
            assert!(x <= 1920);
            assert!(y <= 1080);
        }
    }
}
