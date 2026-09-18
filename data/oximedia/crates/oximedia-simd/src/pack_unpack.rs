//! Pixel packing and unpacking utilities for SIMD pipelines.
//!
//! Provides efficient pack/unpack operations for common pixel formats
//! used in video processing, enabling fast conversion between packed
//! integer representations and component tuples.

#![allow(dead_code)]

/// Pixel packing format descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackFormat {
    /// 8-bit per channel RGB, packed into 24 bits (3 bytes).
    Rgb888,
    /// 8-bit per channel RGBA, packed into 32 bits (4 bytes).
    Rgba8888,
    /// 8-bit per channel BGR, packed into 24 bits (3 bytes).
    Bgr888,
    /// 8-bit per channel BGRA, packed into 32 bits (4 bytes).
    Bgra8888,
    /// 5-6-5 RGB packed into 16 bits.
    Rgb565,
    /// 4-4-4-4 RGBA packed into 16 bits.
    Rgba4444,
    /// 10-10-10-2 RGB+alpha packed into 32 bits.
    Rgb10A2,
}

impl PackFormat {
    /// Returns the number of bytes required per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            PackFormat::Rgb888 | PackFormat::Bgr888 => 3,
            PackFormat::Rgba8888 | PackFormat::Bgra8888 | PackFormat::Rgb10A2 => 4,
            PackFormat::Rgb565 | PackFormat::Rgba4444 => 2,
        }
    }

    /// Returns the number of bits per pixel.
    #[must_use]
    pub const fn bits_per_pixel(self) -> usize {
        self.bytes_per_pixel() * 8
    }

    /// Returns whether this format includes an alpha channel.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            PackFormat::Rgba8888
                | PackFormat::Bgra8888
                | PackFormat::Rgba4444
                | PackFormat::Rgb10A2
        )
    }
}

/// Pack an RGB triple (8-bit each) into a single `u32` (low 24 bits used).
///
/// Layout: `0x00RRGGBB`
#[must_use]
#[inline]
pub fn pack_rgb888(r: u8, g: u8, b: u8) -> u32 {
    (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b)
}

/// Unpack a `u32` (low 24 bits) into an RGB triple (8-bit each).
///
/// Expects layout: `0x00RRGGBB`
#[must_use]
#[inline]
pub fn unpack_rgb888(v: u32) -> (u8, u8, u8) {
    let r = ((v >> 16) & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let b = (v & 0xFF) as u8;
    (r, g, b)
}

/// Pack an RGBA quadruple (8-bit each) into a single `u32`.
///
/// Layout: `0xRRGGBBAA`
#[must_use]
#[inline]
pub fn pack_rgba8888(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (u32::from(r) << 24) | (u32::from(g) << 16) | (u32::from(b) << 8) | u32::from(a)
}

/// Unpack a `u32` into an RGBA quadruple (8-bit each).
///
/// Expects layout: `0xRRGGBBAA`
#[must_use]
#[inline]
pub fn unpack_rgba8888(v: u32) -> (u8, u8, u8, u8) {
    let r = ((v >> 24) & 0xFF) as u8;
    let g = ((v >> 16) & 0xFF) as u8;
    let b = ((v >> 8) & 0xFF) as u8;
    let a = (v & 0xFF) as u8;
    (r, g, b, a)
}

/// Pack a BGR triple (8-bit each) into a single `u32` (low 24 bits used).
///
/// Layout: `0x00BBGGRR`
#[must_use]
#[inline]
pub fn pack_bgr888(r: u8, g: u8, b: u8) -> u32 {
    (u32::from(b) << 16) | (u32::from(g) << 8) | u32::from(r)
}

/// Unpack a `u32` (low 24 bits, BGR layout) into an RGB triple.
///
/// Expects layout: `0x00BBGGRR`
#[must_use]
#[inline]
pub fn unpack_bgr888(v: u32) -> (u8, u8, u8) {
    let b = ((v >> 16) & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let r = (v & 0xFF) as u8;
    (r, g, b)
}

/// Pack an RGB triple into 16-bit RGB565 format.
///
/// R: 5 bits (bits 15–11), G: 6 bits (bits 10–5), B: 5 bits (bits 4–0).
#[must_use]
#[inline]
pub fn pack_rgb565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = u16::from(r >> 3);
    let g6 = u16::from(g >> 2);
    let b5 = u16::from(b >> 3);
    (r5 << 11) | (g6 << 5) | b5
}

/// Unpack a 16-bit RGB565 value into an 8-bit RGB triple.
///
/// Expands each component back to 8-bit range.
#[must_use]
#[inline]
pub fn unpack_rgb565(v: u16) -> (u8, u8, u8) {
    let r5 = (v >> 11) & 0x1F;
    let g6 = (v >> 5) & 0x3F;
    let b5 = v & 0x1F;
    // Expand to 8-bit by left-shifting and replicating upper bits
    let r = ((r5 << 3) | (r5 >> 2)) as u8;
    let g = ((g6 << 2) | (g6 >> 4)) as u8;
    let b = ((b5 << 3) | (b5 >> 2)) as u8;
    (r, g, b)
}

/// Pack a slice of `(r, g, b, a)` tuples into a `Vec<u32>` using RGBA8888 layout.
#[must_use]
pub fn pack_rgba8888_slice(pixels: &[(u8, u8, u8, u8)]) -> Vec<u32> {
    pixels
        .iter()
        .map(|&(r, g, b, a)| pack_rgba8888(r, g, b, a))
        .collect()
}

/// Unpack a slice of `u32` values (RGBA8888 layout) into a `Vec<(u8, u8, u8, u8)>`.
#[must_use]
pub fn unpack_rgba8888_slice(packed: &[u32]) -> Vec<(u8, u8, u8, u8)> {
    packed.iter().map(|&v| unpack_rgba8888(v)).collect()
}

/// Pack a slice of `(r, g, b)` tuples into a `Vec<u32>` using RGB888 layout.
#[must_use]
pub fn pack_rgb888_slice(pixels: &[(u8, u8, u8)]) -> Vec<u32> {
    pixels
        .iter()
        .map(|&(r, g, b)| pack_rgb888(r, g, b))
        .collect()
}

/// Unpack a slice of `u32` values (RGB888 layout) into a `Vec<(u8, u8, u8)>`.
#[must_use]
pub fn unpack_rgb888_slice(packed: &[u32]) -> Vec<(u8, u8, u8)> {
    packed.iter().map(|&v| unpack_rgb888(v)).collect()
}

/// Byte-level packer: writes pixels into a `&mut [u8]` buffer in RGB888 order.
///
/// Returns the number of bytes written, or `None` if the buffer is too small.
pub fn pack_rgb888_into_bytes(pixels: &[(u8, u8, u8)], dst: &mut [u8]) -> Option<usize> {
    let required = pixels.len() * 3;
    if dst.len() < required {
        return None;
    }
    for (i, &(r, g, b)) in pixels.iter().enumerate() {
        dst[i * 3] = r;
        dst[i * 3 + 1] = g;
        dst[i * 3 + 2] = b;
    }
    Some(required)
}

/// Byte-level unpacker: reads pixels from a `&[u8]` buffer in RGB888 order.
///
/// Returns a `Vec` of `(r, g, b)` tuples.  Any trailing incomplete pixel is ignored.
#[must_use]
pub fn unpack_rgb888_from_bytes(src: &[u8]) -> Vec<(u8, u8, u8)> {
    src.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_format_bytes_per_pixel() {
        assert_eq!(PackFormat::Rgb888.bytes_per_pixel(), 3);
        assert_eq!(PackFormat::Rgba8888.bytes_per_pixel(), 4);
        assert_eq!(PackFormat::Bgr888.bytes_per_pixel(), 3);
        assert_eq!(PackFormat::Bgra8888.bytes_per_pixel(), 4);
        assert_eq!(PackFormat::Rgb565.bytes_per_pixel(), 2);
        assert_eq!(PackFormat::Rgba4444.bytes_per_pixel(), 2);
        assert_eq!(PackFormat::Rgb10A2.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_pack_format_has_alpha() {
        assert!(!PackFormat::Rgb888.has_alpha());
        assert!(PackFormat::Rgba8888.has_alpha());
        assert!(!PackFormat::Bgr888.has_alpha());
        assert!(PackFormat::Bgra8888.has_alpha());
        assert!(PackFormat::Rgba4444.has_alpha());
        assert!(PackFormat::Rgb10A2.has_alpha());
    }

    #[test]
    fn test_pack_rgb888_roundtrip() {
        let (r, g, b) = (0xAA_u8, 0xBB_u8, 0xCC_u8);
        let packed = pack_rgb888(r, g, b);
        let (ur, ug, ub) = unpack_rgb888(packed);
        assert_eq!((ur, ug, ub), (r, g, b));
    }

    #[test]
    fn test_pack_rgb888_value() {
        assert_eq!(pack_rgb888(0xFF, 0x00, 0x00), 0x00FF0000);
        assert_eq!(pack_rgb888(0x00, 0xFF, 0x00), 0x0000FF00);
        assert_eq!(pack_rgb888(0x00, 0x00, 0xFF), 0x000000FF);
    }

    #[test]
    fn test_pack_rgba8888_roundtrip() {
        let (r, g, b, a) = (0x11_u8, 0x22_u8, 0x33_u8, 0xFF_u8);
        let packed = pack_rgba8888(r, g, b, a);
        let (ur, ug, ub, ua) = unpack_rgba8888(packed);
        assert_eq!((ur, ug, ub, ua), (r, g, b, a));
    }

    #[test]
    fn test_pack_bgr888_roundtrip() {
        let (r, g, b) = (0x10_u8, 0x20_u8, 0x30_u8);
        let packed = pack_bgr888(r, g, b);
        let (ur, ug, ub) = unpack_bgr888(packed);
        assert_eq!((ur, ug, ub), (r, g, b));
    }

    #[test]
    fn test_pack_rgb565_roundtrip_approximate() {
        // RGB565 is lossy (reduces precision), so check approximate round-trip
        let (r, g, b) = (0xF8_u8, 0xFC_u8, 0xF8_u8); // multiples of 8 / 4
        let packed = pack_rgb565(r, g, b);
        let (ur, ug, ub) = unpack_rgb565(packed);
        // Allow +-7 for R and B (5-bit), +-3 for G (6-bit)
        assert!((i32::from(ur) - i32::from(r)).abs() <= 7);
        assert!((i32::from(ug) - i32::from(g)).abs() <= 3);
        assert!((i32::from(ub) - i32::from(b)).abs() <= 7);
    }

    #[test]
    fn test_pack_rgba8888_slice_roundtrip() {
        let pixels = vec![(255, 0, 128, 64), (0, 255, 0, 255), (10, 20, 30, 40)];
        let packed = pack_rgba8888_slice(&pixels);
        let unpacked = unpack_rgba8888_slice(&packed);
        assert_eq!(unpacked, pixels);
    }

    #[test]
    fn test_pack_rgb888_slice_roundtrip() {
        let pixels = vec![(100, 150, 200), (0, 0, 0), (255, 255, 255)];
        let packed = pack_rgb888_slice(&pixels);
        let unpacked = unpack_rgb888_slice(&packed);
        assert_eq!(unpacked, pixels);
    }

    #[test]
    fn test_pack_rgb888_into_bytes() {
        let pixels = vec![(1, 2, 3), (4, 5, 6)];
        let mut buf = vec![0u8; 6];
        let written = pack_rgb888_into_bytes(&pixels, &mut buf);
        assert_eq!(written, Some(6));
        assert_eq!(buf, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_pack_rgb888_into_bytes_too_small() {
        let pixels = vec![(1, 2, 3), (4, 5, 6)];
        let mut buf = vec![0u8; 4]; // too small
        let written = pack_rgb888_into_bytes(&pixels, &mut buf);
        assert_eq!(written, None);
    }

    #[test]
    fn test_unpack_rgb888_from_bytes() {
        let bytes = [10u8, 20, 30, 40, 50, 60, 70]; // trailing byte ignored
        let pixels = unpack_rgb888_from_bytes(&bytes);
        assert_eq!(pixels, [(10, 20, 30), (40, 50, 60)]);
    }

    #[test]
    fn test_bits_per_pixel() {
        assert_eq!(PackFormat::Rgb565.bits_per_pixel(), 16);
        assert_eq!(PackFormat::Rgba8888.bits_per_pixel(), 32);
    }

    #[test]
    fn test_unpack_rgb888_black_white() {
        assert_eq!(unpack_rgb888(0x000000), (0, 0, 0));
        assert_eq!(unpack_rgb888(0xFFFFFF), (255, 255, 255));
    }
}
