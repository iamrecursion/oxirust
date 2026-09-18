//! Format compatibility matching between proxy and original media formats.
#![allow(dead_code)]

/// Supported proxy format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxyFormatType {
    /// H.264 in an MP4 container — widely compatible edit-ready format.
    H264Mp4,
    /// H.264 in an MXF container — broadcast edit-ready format.
    H264Mxf,
    /// Apple ProRes 422 Proxy — edit-ready for Final Cut Pro workflows.
    ProResProxy,
    /// DNxHD/DNxHR — edit-ready for Avid workflows.
    DnxHd,
    /// VP9 in WebM — web-delivery proxy format.
    Vp9Webm,
    /// JPEG 2000 — high-quality proxy for DCP and broadcast.
    Jpeg2000,
}

impl ProxyFormatType {
    /// Return `true` if this format is considered edit-ready (fast random access,
    /// intra-only or low-GOP).
    pub fn is_edit_ready(self) -> bool {
        matches!(
            self,
            Self::H264Mxf | Self::ProResProxy | Self::DnxHd | Self::Jpeg2000
        )
    }

    /// Return a human-readable label for this format.
    pub fn label(self) -> &'static str {
        match self {
            Self::H264Mp4 => "H.264 / MP4",
            Self::H264Mxf => "H.264 / MXF",
            Self::ProResProxy => "Apple ProRes 422 Proxy",
            Self::DnxHd => "Avid DNxHD/DNxHR",
            Self::Vp9Webm => "VP9 / WebM",
            Self::Jpeg2000 => "JPEG 2000",
        }
    }

    /// Return a typical file extension for this format.
    pub fn extension(self) -> &'static str {
        match self {
            Self::H264Mp4 => "mp4",
            Self::H264Mxf | Self::DnxHd | Self::Jpeg2000 => "mxf",
            Self::ProResProxy => "mov",
            Self::Vp9Webm => "webm",
        }
    }
}

/// Describes format compatibility constraints for a proxy.
#[derive(Debug, Clone)]
pub struct FormatCompat {
    /// The proxy format type.
    pub format: ProxyFormatType,
    /// Maximum width this format/config can handle.
    pub max_width: u32,
    /// Maximum height this format/config can handle.
    pub max_height: u32,
    /// Whether this entry is preferred for new projects.
    pub preferred: bool,
}

impl FormatCompat {
    /// Create a new `FormatCompat` entry.
    pub fn new(format: ProxyFormatType, max_width: u32, max_height: u32) -> Self {
        Self {
            format,
            max_width,
            max_height,
            preferred: false,
        }
    }

    /// Mark this entry as preferred.
    pub fn as_preferred(mut self) -> Self {
        self.preferred = true;
        self
    }

    /// Return `true` if the given resolution fits within this format's limits.
    pub fn resolution_ok(&self, width: u32, height: u32) -> bool {
        width <= self.max_width && height <= self.max_height
    }
}

/// Matcher that selects the best compatible proxy format for a given resolution.
#[derive(Debug, Default)]
pub struct ProxyFormatMatcher {
    entries: Vec<FormatCompat>,
}

impl ProxyFormatMatcher {
    /// Create an empty matcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a matcher pre-populated with common broadcast/editing formats.
    pub fn standard() -> Self {
        let mut m = Self::new();
        m.add(FormatCompat::new(ProxyFormatType::ProResProxy, 3840, 2160).as_preferred());
        m.add(FormatCompat::new(ProxyFormatType::DnxHd, 3840, 2160));
        m.add(FormatCompat::new(ProxyFormatType::H264Mp4, 1920, 1080));
        m.add(FormatCompat::new(ProxyFormatType::Vp9Webm, 1920, 1080));
        m
    }

    /// Add a format compatibility entry.
    pub fn add(&mut self, compat: FormatCompat) {
        self.entries.push(compat);
    }

    /// Return all entries whose resolution limit accommodates the given dimensions.
    pub fn find_compatible(&self, width: u32, height: u32) -> Vec<&FormatCompat> {
        self.entries
            .iter()
            .filter(|e| e.resolution_ok(width, height))
            .collect()
    }

    /// Return the best match: preferred first, then highest-resolution limit.
    pub fn best_match(&self, width: u32, height: u32) -> Option<&FormatCompat> {
        let compatible = self.find_compatible(width, height);
        // Preferred formats come first.
        if let Some(pref) = compatible.iter().find(|e| e.preferred) {
            return Some(pref);
        }
        // Otherwise pick the one with the largest max area (most capable).
        compatible
            .into_iter()
            .max_by_key(|e| e.max_width * e.max_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_type_is_edit_ready_true() {
        assert!(ProxyFormatType::ProResProxy.is_edit_ready());
        assert!(ProxyFormatType::DnxHd.is_edit_ready());
        assert!(ProxyFormatType::H264Mxf.is_edit_ready());
        assert!(ProxyFormatType::Jpeg2000.is_edit_ready());
    }

    #[test]
    fn test_format_type_is_edit_ready_false() {
        assert!(!ProxyFormatType::H264Mp4.is_edit_ready());
        assert!(!ProxyFormatType::Vp9Webm.is_edit_ready());
    }

    #[test]
    fn test_format_type_label_not_empty() {
        for fmt in [
            ProxyFormatType::H264Mp4,
            ProxyFormatType::H264Mxf,
            ProxyFormatType::ProResProxy,
            ProxyFormatType::DnxHd,
            ProxyFormatType::Vp9Webm,
            ProxyFormatType::Jpeg2000,
        ] {
            assert!(!fmt.label().is_empty());
        }
    }

    #[test]
    fn test_format_type_extension() {
        assert_eq!(ProxyFormatType::H264Mp4.extension(), "mp4");
        assert_eq!(ProxyFormatType::ProResProxy.extension(), "mov");
        assert_eq!(ProxyFormatType::Vp9Webm.extension(), "webm");
    }

    #[test]
    fn test_format_compat_resolution_ok() {
        let fc = FormatCompat::new(ProxyFormatType::H264Mp4, 1920, 1080);
        assert!(fc.resolution_ok(1920, 1080));
        assert!(fc.resolution_ok(1280, 720));
        assert!(!fc.resolution_ok(3840, 2160));
    }

    #[test]
    fn test_format_compat_preferred_flag() {
        let fc = FormatCompat::new(ProxyFormatType::ProResProxy, 3840, 2160).as_preferred();
        assert!(fc.preferred);
    }

    #[test]
    fn test_matcher_find_compatible_all() {
        let matcher = ProxyFormatMatcher::standard();
        let results = matcher.find_compatible(1280, 720);
        // All standard entries support at least 1920x1080 or 3840x2160
        assert!(!results.is_empty());
    }

    #[test]
    fn test_matcher_find_compatible_4k_filters_hd_only() {
        let matcher = ProxyFormatMatcher::standard();
        let results = matcher.find_compatible(3840, 2160);
        // Only entries with max >= 3840x2160 should pass
        for entry in &results {
            assert!(entry.resolution_ok(3840, 2160));
        }
    }

    #[test]
    fn test_matcher_best_match_prefers_preferred() {
        let matcher = ProxyFormatMatcher::standard();
        let best = matcher
            .best_match(1280, 720)
            .expect("should succeed in test");
        assert!(best.preferred);
    }

    #[test]
    fn test_matcher_best_match_4k() {
        let matcher = ProxyFormatMatcher::standard();
        let best = matcher
            .best_match(3840, 2160)
            .expect("should succeed in test");
        assert!(best.resolution_ok(3840, 2160));
    }

    #[test]
    fn test_matcher_best_match_empty_returns_none() {
        let matcher = ProxyFormatMatcher::new();
        assert!(matcher.best_match(1920, 1080).is_none());
    }

    #[test]
    fn test_matcher_add_single_entry() {
        let mut matcher = ProxyFormatMatcher::new();
        matcher.add(FormatCompat::new(ProxyFormatType::DnxHd, 1920, 1080));
        let best = matcher
            .best_match(1920, 1080)
            .expect("should succeed in test");
        assert_eq!(best.format, ProxyFormatType::DnxHd);
    }

    #[test]
    fn test_matcher_no_compatible_for_oversized() {
        let mut matcher = ProxyFormatMatcher::new();
        matcher.add(FormatCompat::new(ProxyFormatType::H264Mp4, 640, 480));
        let results = matcher.find_compatible(1920, 1080);
        assert!(results.is_empty());
    }
}
