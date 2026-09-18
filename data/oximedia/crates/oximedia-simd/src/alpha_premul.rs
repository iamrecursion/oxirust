//! Alpha pre-multiplication and un-pre-multiplication helpers.
//!
//! Pre-multiplied alpha is required by many compositing pipelines and GPU
//! texture formats. These helpers convert between straight and pre-multiplied
//! representations.

#![allow(dead_code)]

/// Pre-multiply an RGBA pixel.
///
/// Multiplies each colour channel by `a / 255`.  For `a == 0` the result is
/// `[0, 0, 0, 0]`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn premultiply(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    if a == 0 {
        return [0, 0, 0, 0];
    }
    let scale = f32::from(a) / 255.0;
    let pm = |c: u8| (f32::from(c) * scale + 0.5) as u8;
    [pm(r), pm(g), pm(b), a]
}

/// Un-pre-multiply an RGBA pixel (convert from pre-multiplied to straight).
///
/// Divides each colour channel by `a / 255`.  For `a == 0` the result is
/// `[0, 0, 0, 0]`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn unpremultiply(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    if a == 0 {
        return [0, 0, 0, 0];
    }
    let scale = 255.0 / f32::from(a);
    let upm = |c: u8| (f32::from(c) * scale).clamp(0.0, 255.0) as u8;
    [upm(r), upm(g), upm(b), a]
}

/// Pre-multiply every pixel in a packed RGBA buffer in-place.
///
/// `buf` must have a length that is a multiple of 4.
///
/// # Panics
///
/// Panics if `buf.len()` is not a multiple of 4.
pub fn premultiply_buffer(buf: &mut [u8]) {
    assert!(
        buf.len().is_multiple_of(4),
        "buffer length must be a multiple of 4"
    );
    for chunk in buf.chunks_exact_mut(4) {
        let [r, g, b, a] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let pm = premultiply(r, g, b, a);
        chunk.copy_from_slice(&pm);
    }
}

/// Un-pre-multiply every pixel in a packed RGBA buffer in-place.
///
/// `buf` must have a length that is a multiple of 4.
///
/// # Panics
///
/// Panics if `buf.len()` is not a multiple of 4.
pub fn unpremultiply_buffer(buf: &mut [u8]) {
    assert!(
        buf.len().is_multiple_of(4),
        "buffer length must be a multiple of 4"
    );
    for chunk in buf.chunks_exact_mut(4) {
        let [r, g, b, a] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let upm = unpremultiply(r, g, b, a);
        chunk.copy_from_slice(&upm);
    }
}

/// Statistics collected during a pre-multiply / un-pre-multiply round-trip.
#[derive(Debug, Clone, Default)]
pub struct AlphaPremulStats {
    max_channel_error: u8,
    pixel_count: usize,
}

impl AlphaPremulStats {
    /// Measure round-trip error on a straight-alpha buffer.
    ///
    /// Returns statistics including the maximum per-channel absolute error
    /// introduced by the premultiply → unpremultiply round-trip.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn measure_roundtrip(buf: &[u8]) -> Self {
        assert!(buf.len().is_multiple_of(4));
        let mut max_err = 0u8;
        let mut count = 0usize;

        for chunk in buf.chunks_exact(4) {
            let (r, g, b, a) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            let pm = premultiply(r, g, b, a);
            let [rr, rg, rb, _] = unpremultiply(pm[0], pm[1], pm[2], pm[3]);

            let err = |orig: u8, recovered: u8| {
                (i16::from(orig) - i16::from(recovered)).unsigned_abs() as u8
            };
            max_err = max_err.max(err(r, rr)).max(err(g, rg)).max(err(b, rb));
            count += 1;
        }

        Self {
            max_channel_error: max_err,
            pixel_count: count,
        }
    }

    /// Maximum per-channel absolute error observed during the round-trip.
    #[must_use]
    pub fn max_error(&self) -> u8 {
        self.max_channel_error
    }

    /// Number of pixels analysed.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.pixel_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_premultiply_opaque() {
        let pm = premultiply(200, 100, 50, 255);
        assert_eq!(pm, [200, 100, 50, 255]);
    }

    #[test]
    fn test_premultiply_transparent() {
        let pm = premultiply(200, 100, 50, 0);
        assert_eq!(pm, [0, 0, 0, 0]);
    }

    #[test]
    fn test_premultiply_half_alpha() {
        let pm = premultiply(200, 100, 0, 128);
        // 200 * 128/255 ≈ 100
        assert!((i16::from(pm[0]) - 100).abs() <= 2, "r={}", pm[0]);
    }

    #[test]
    fn test_unpremultiply_transparent() {
        let upm = unpremultiply(0, 0, 0, 0);
        assert_eq!(upm, [0, 0, 0, 0]);
    }

    #[test]
    fn test_unpremultiply_opaque() {
        let upm = unpremultiply(128, 64, 32, 255);
        assert_eq!(upm, [128, 64, 32, 255]);
    }

    #[test]
    fn test_roundtrip_opaque() {
        for v in [0u8, 64, 128, 192, 255] {
            let pm = premultiply(v, v, v, 255);
            let upm = unpremultiply(pm[0], pm[1], pm[2], pm[3]);
            assert_eq!(upm[0], v, "channel mismatch for v={v}");
        }
    }

    #[test]
    fn test_premultiply_buffer_basic() {
        let mut buf = [100u8, 50, 25, 128, 200, 100, 50, 255];
        premultiply_buffer(&mut buf);
        // Second pixel is fully opaque, unchanged
        assert_eq!(buf[4], 200);
        assert_eq!(buf[5], 100);
        assert_eq!(buf[6], 50);
        assert_eq!(buf[7], 255);
    }

    #[test]
    fn test_unpremultiply_buffer_roundtrip() {
        let original = [200u8, 100, 50, 255, 0, 0, 0, 0];
        let mut buf = original;
        premultiply_buffer(&mut buf);
        unpremultiply_buffer(&mut buf);
        // Opaque pixel should be unchanged
        assert_eq!(buf[0], original[0]);
        assert_eq!(buf[1], original[1]);
        assert_eq!(buf[2], original[2]);
    }

    #[test]
    fn test_stats_max_error() {
        let buf = [200u8, 100, 50, 200]; // one pixel
        let stats = AlphaPremulStats::measure_roundtrip(&buf);
        assert!(
            stats.max_error() <= 2,
            "error too large: {}",
            stats.max_error()
        );
        assert_eq!(stats.pixel_count(), 1);
    }

    #[test]
    fn test_stats_opaque_zero_error() {
        // Fully opaque should round-trip perfectly
        let buf = [128u8, 64, 32, 255];
        let stats = AlphaPremulStats::measure_roundtrip(&buf);
        assert_eq!(stats.max_error(), 0);
    }

    #[test]
    fn test_stats_transparent_zero_channels() {
        // For a fully transparent pixel (a=0), premultiply zeroes the channels,
        // so unpremultiply also returns zeros.  The round-trip error equals the
        // original colour value because the alpha encoding discards it.
        let buf = [0u8, 0, 0, 0]; // start with already-zero colours
        let stats = AlphaPremulStats::measure_roundtrip(&buf);
        assert_eq!(stats.max_error(), 0);
        assert_eq!(stats.pixel_count(), 1);
    }
}
