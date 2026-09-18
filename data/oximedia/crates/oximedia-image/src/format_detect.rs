//! Magic-byte image format detection.
//!
//! Identifies image file formats by examining the leading bytes of file
//! data without relying on file-name extensions.

#![allow(dead_code)]

/// Known image file formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    /// JPEG / JFIF (SOI marker `FF D8`).
    Jpeg,
    /// PNG (`\x89PNG`).
    Png,
    /// TIFF – little-endian (`II*\0`) or big-endian (`MM\0*`).
    Tiff,
    /// OpenEXR (`v/1\x01`).
    Exr,
    /// DPX (`SDPX` or `XPDS`).
    Dpx,
    /// Cineon (`\x80\x2a\x5f\xd7`).
    Cineon,
    /// BMP (`BM`).
    Bmp,
    /// GIF (`GIF87a` or `GIF89a`).
    Gif,
    /// WebP (`RIFF….WEBP`).
    WebP,
    /// AVIF / HEIF / HEIC (ISO Base Media File Format).
    Heif,
    /// JPEG 2000 (JP2 signature box).
    Jpeg2000,
    /// Format could not be identified from the provided bytes.
    Unknown,
}

impl ImageFormat {
    /// Human-readable name of this format.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Png => "PNG",
            Self::Tiff => "TIFF",
            Self::Exr => "OpenEXR",
            Self::Dpx => "DPX",
            Self::Cineon => "Cineon",
            Self::Bmp => "BMP",
            Self::Gif => "GIF",
            Self::WebP => "WebP",
            Self::Heif => "HEIF/HEIC",
            Self::Jpeg2000 => "JPEG 2000",
            Self::Unknown => "Unknown",
        }
    }

    /// Canonical file extension (without leading dot).
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Tiff => "tiff",
            Self::Exr => "exr",
            Self::Dpx => "dpx",
            Self::Cineon => "cin",
            Self::Bmp => "bmp",
            Self::Gif => "gif",
            Self::WebP => "webp",
            Self::Heif => "heic",
            Self::Jpeg2000 => "jp2",
            Self::Unknown => "",
        }
    }

    /// Return `true` if the format supports HDR (high dynamic range) data.
    #[must_use]
    pub const fn is_hdr(&self) -> bool {
        matches!(self, Self::Exr | Self::Dpx | Self::Cineon)
    }

    /// Return `true` if the format supports lossless compression.
    #[must_use]
    pub const fn is_lossless(&self) -> bool {
        matches!(
            self,
            Self::Png | Self::Tiff | Self::Exr | Self::Dpx | Self::Cineon | Self::Bmp
        )
    }
}

/// Detects [`ImageFormat`] by examining raw byte slices (magic numbers).
pub struct FormatDetector;

impl FormatDetector {
    /// Detect the image format from the first bytes of a file.
    ///
    /// At least 12 bytes are needed for reliable detection; fewer bytes may
    /// return [`ImageFormat::Unknown`] for ambiguous formats.
    #[must_use]
    pub fn detect(header: &[u8]) -> ImageFormat {
        if header.len() < 4 {
            return ImageFormat::Unknown;
        }

        // JPEG: FF D8 FF
        if header.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return ImageFormat::Jpeg;
        }

        // PNG: 89 50 4E 47 0D 0A 1A 0A
        if header.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return ImageFormat::Png;
        }

        // OpenEXR: 76 2F 31 01
        if header.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
            return ImageFormat::Exr;
        }

        // DPX big-endian: "SDPX"
        if header.starts_with(b"SDPX") {
            return ImageFormat::Dpx;
        }

        // DPX little-endian: "XPDS"
        if header.starts_with(b"XPDS") {
            return ImageFormat::Dpx;
        }

        // Cineon: 80 2A 5F D7
        if header.starts_with(&[0x80, 0x2A, 0x5F, 0xD7]) {
            return ImageFormat::Cineon;
        }

        // TIFF little-endian: 49 49 2A 00
        if header.starts_with(&[0x49, 0x49, 0x2A, 0x00]) {
            return ImageFormat::Tiff;
        }

        // TIFF big-endian: 4D 4D 00 2A
        if header.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
            return ImageFormat::Tiff;
        }

        // BMP: "BM"
        if header.starts_with(b"BM") {
            return ImageFormat::Bmp;
        }

        // GIF: "GIF87a" or "GIF89a"
        if header.starts_with(b"GIF87a") || header.starts_with(b"GIF89a") {
            return ImageFormat::Gif;
        }

        // WebP: "RIFF" at 0, "WEBP" at offset 8
        if header.len() >= 12 && header.starts_with(b"RIFF") && &header[8..12] == b"WEBP" {
            return ImageFormat::WebP;
        }

        // JPEG 2000: 00 00 00 0C 6A 50 20 20
        if header.len() >= 8
            && header.starts_with(&[0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20])
        {
            return ImageFormat::Jpeg2000;
        }

        // HEIF / HEIC: ftyp box at offset 4 with "heic", "heix", "mif1", etc.
        if header.len() >= 12 && &header[4..8] == b"ftyp" {
            let brand = &header[8..12];
            if brand == b"heic" || brand == b"heix" || brand == b"mif1" || brand == b"msf1" {
                return ImageFormat::Heif;
            }
        }

        ImageFormat::Unknown
    }

    /// Attempt to detect format from a file path extension (case-insensitive
    /// fallback when byte data is unavailable).
    #[must_use]
    pub fn from_extension(path: &str) -> ImageFormat {
        let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" | "jfif" => ImageFormat::Jpeg,
            "png" => ImageFormat::Png,
            "tif" | "tiff" => ImageFormat::Tiff,
            "exr" => ImageFormat::Exr,
            "dpx" => ImageFormat::Dpx,
            "cin" | "cineon" => ImageFormat::Cineon,
            "bmp" | "dib" => ImageFormat::Bmp,
            "gif" => ImageFormat::Gif,
            "webp" => ImageFormat::WebP,
            "heic" | "heif" | "avif" => ImageFormat::Heif,
            "jp2" | "j2k" | "jpf" => ImageFormat::Jpeg2000,
            _ => ImageFormat::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_jpeg() {
        let header = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Jpeg);
    }

    #[test]
    fn test_detect_png() {
        let header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Png);
    }

    #[test]
    fn test_detect_exr() {
        let header = [0x76, 0x2F, 0x31, 0x01, 0x00, 0x00];
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Exr);
    }

    #[test]
    fn test_detect_dpx_be() {
        let header = *b"SDPX\x00\x00\x10\x00";
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Dpx);
    }

    #[test]
    fn test_detect_dpx_le() {
        let header = *b"XPDS\x00\x00\x10\x00";
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Dpx);
    }

    #[test]
    fn test_detect_tiff_le() {
        let header = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Tiff);
    }

    #[test]
    fn test_detect_tiff_be() {
        let header = [0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Tiff);
    }

    #[test]
    fn test_detect_bmp() {
        let header = *b"BM\x36\x00\x00\x00";
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Bmp);
    }

    #[test]
    fn test_detect_gif() {
        let header = *b"GIF89a";
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Gif);
    }

    #[test]
    fn test_detect_webp() {
        let mut header = [0u8; 12];
        header[..4].copy_from_slice(b"RIFF");
        header[8..12].copy_from_slice(b"WEBP");
        assert_eq!(FormatDetector::detect(&header), ImageFormat::WebP);
    }

    #[test]
    fn test_detect_unknown() {
        let header = [0x00, 0x01, 0x02, 0x03, 0x04];
        assert_eq!(FormatDetector::detect(&header), ImageFormat::Unknown);
    }

    #[test]
    fn test_detect_too_short() {
        assert_eq!(
            FormatDetector::detect(&[0xFF, 0xD8, 0xFF]),
            ImageFormat::Unknown
        );
    }

    #[test]
    fn test_from_extension_jpeg() {
        assert_eq!(
            FormatDetector::from_extension("frame.jpeg"),
            ImageFormat::Jpeg
        );
        assert_eq!(
            FormatDetector::from_extension("FRAME.JPG"),
            ImageFormat::Jpeg
        );
    }

    #[test]
    fn test_from_extension_exr() {
        assert_eq!(
            FormatDetector::from_extension("render.exr"),
            ImageFormat::Exr
        );
    }

    #[test]
    fn test_is_hdr_flag() {
        assert!(ImageFormat::Exr.is_hdr());
        assert!(ImageFormat::Dpx.is_hdr());
        assert!(!ImageFormat::Jpeg.is_hdr());
    }

    #[test]
    fn test_is_lossless_flag() {
        assert!(ImageFormat::Png.is_lossless());
        assert!(!ImageFormat::Jpeg.is_lossless());
        assert!(!ImageFormat::WebP.is_lossless()); // lossy by default
    }

    #[test]
    fn test_format_names_non_empty() {
        let formats = [
            ImageFormat::Jpeg,
            ImageFormat::Png,
            ImageFormat::Tiff,
            ImageFormat::Exr,
            ImageFormat::Dpx,
        ];
        for f in &formats {
            assert!(!f.name().is_empty(), "{f:?} name is empty");
        }
    }
}
