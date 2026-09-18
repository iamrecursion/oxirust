//! Intelligent proxy selection and recommendation.
//!
//! This module analyses the editing context (NLE software, resolution, network
//! conditions) and recommends the most appropriate proxy configuration for a
//! given set of source files.

#![allow(dead_code)]

use crate::transcode_queue::ProxySpec;
use serde::{Deserialize, Serialize};

/// The NLE (Non-Linear Editor) software being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditingSoftware {
    /// DaVinci Resolve (Blackmagic Design).
    Resolve,
    /// Adobe Premiere Pro.
    Premiere,
    /// Avid Media Composer.
    Avid,
    /// Apple Final Cut Pro.
    FinalCut,
    /// VEGAS Pro (Magix).
    Vegas,
    /// Kdenlive (KDE).
    Kdenlive,
}

impl EditingSoftware {
    /// The proxy codec preferred by this software.
    #[must_use]
    pub fn preferred_proxy_codec(self) -> &'static str {
        match self {
            Self::Resolve => "prores_proxy",
            Self::Premiere => "h264",
            Self::Avid => "dnxhd",
            Self::FinalCut => "prores_proxy",
            Self::Vegas => "h264",
            Self::Kdenlive => "h264",
        }
    }
}

/// Information about an editing session's context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditingContext {
    /// NLE software in use.
    pub software: EditingSoftware,
    /// Project/sequence resolution (width, height).
    pub resolution: (u32, u32),
    /// Codecs the editing system can decode in real-time.
    pub codec_support: Vec<String>,
    /// Available network bandwidth in Mbit/s (for shared storage workflows).
    pub network_speed_mbps: f32,
}

impl EditingContext {
    /// Create a new editing context.
    #[must_use]
    pub fn new(
        software: EditingSoftware,
        resolution: (u32, u32),
        codec_support: Vec<String>,
        network_speed_mbps: f32,
    ) -> Self {
        Self {
            software,
            resolution,
            codec_support,
            network_speed_mbps,
        }
    }
}

/// Source file specification used when choosing a proxy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpec {
    /// File path.
    pub path: String,
    /// Source resolution (width, height).
    pub resolution: (u32, u32),
    /// Source codec identifier.
    pub codec: String,
    /// Source bitrate in kbit/s.
    pub bitrate_kbps: u32,
    /// Source frame rate.
    pub fps: f32,
}

impl SourceSpec {
    /// Create a new source spec.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        resolution: (u32, u32),
        codec: impl Into<String>,
        bitrate_kbps: u32,
        fps: f32,
    ) -> Self {
        Self {
            path: path.into(),
            resolution,
            codec: codec.into(),
            bitrate_kbps,
            fps,
        }
    }
}

/// Recommends proxy specifications based on editing context.
pub struct ProxyRecommender;

impl ProxyRecommender {
    /// Recommend a list of proxy specs for the given sources and editing context.
    ///
    /// The recommendation strategy:
    /// 1. Use the codec preferred by the editing software.
    /// 2. Target the editing context resolution (downscaling if the source is larger).
    /// 3. Adjust bitrate based on network speed: slower networks → lower bitrate.
    #[must_use]
    pub fn recommend(context: &EditingContext, source_specs: &[SourceSpec]) -> Vec<ProxySpec> {
        let preferred_codec = context.software.preferred_proxy_codec().to_string();

        // Determine target resolution: use context resolution, but do not upscale
        let (ctx_w, ctx_h) = context.resolution;

        source_specs
            .iter()
            .map(|src| {
                let (src_w, src_h) = src.resolution;
                // Do not upscale
                let target_w = ctx_w.min(src_w);
                let target_h = ctx_h.min(src_h);

                // Bitrate: scale by pixel area ratio vs context resolution,
                // then clamp based on network speed
                let area_ratio =
                    (target_w * target_h) as f32 / (src_w.max(1) * src_h.max(1)) as f32;
                let base_bitrate_kbps = (src.bitrate_kbps as f32 * area_ratio) as u32;

                // Cap bitrate to what the network can sustain (rough heuristic:
                // leave 50% of bandwidth headroom, convert Mbit/s → kbit/s)
                let max_net_kbps = (context.network_speed_mbps * 1000.0 * 0.5) as u32;
                let bitrate_kbps = if max_net_kbps > 0 {
                    base_bitrate_kbps.min(max_net_kbps)
                } else {
                    base_bitrate_kbps
                };

                // Use the software's preferred codec, unless not supported
                let codec = if context
                    .codec_support
                    .iter()
                    .any(|c| c == preferred_codec.as_str())
                    || context.codec_support.is_empty()
                {
                    preferred_codec.clone()
                } else {
                    // Fall back to first supported codec
                    context
                        .codec_support
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "h264".to_string())
                };

                ProxySpec::new((target_w, target_h), codec, bitrate_kbps.max(500))
            })
            .collect()
    }
}

/// Result of a proxy–context compatibility check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    /// Whether the proxy is fully compatible with the editing context.
    pub compatible: bool,
    /// Non-fatal warnings (e.g. sub-optimal codec choice).
    pub warnings: Vec<String>,
    /// Compatibility score (0.0 = incompatible, 1.0 = perfect).
    pub score: f32,
}

impl CompatibilityResult {
    /// Create a fully compatible result with no warnings.
    #[must_use]
    pub fn ok() -> Self {
        Self {
            compatible: true,
            warnings: vec![],
            score: 1.0,
        }
    }

    /// Create an incompatible result.
    #[must_use]
    pub fn incompatible(reason: impl Into<String>) -> Self {
        Self {
            compatible: false,
            warnings: vec![reason.into()],
            score: 0.0,
        }
    }
}

/// Checks whether a proxy spec is compatible with an editing context.
pub struct ProxyCompatibilityChecker;

impl ProxyCompatibilityChecker {
    /// Check compatibility and return a detailed result.
    #[must_use]
    pub fn check(proxy: &ProxySpec, context: &EditingContext) -> CompatibilityResult {
        let mut warnings = Vec::new();
        let mut score = 1.0f32;

        // 1. Codec check
        let preferred = context.software.preferred_proxy_codec();
        if proxy.codec != preferred {
            warnings.push(format!(
                "Proxy codec '{}' is not the preferred codec '{}' for {:?}",
                proxy.codec, preferred, context.software
            ));
            score -= 0.2;
        }

        // 2. Resolution check: proxy should not exceed context resolution
        let (p_w, p_h) = proxy.resolution;
        let (c_w, c_h) = context.resolution;
        if p_w > c_w || p_h > c_h {
            warnings.push(format!(
                "Proxy resolution {}×{} exceeds editing resolution {}×{}",
                p_w, p_h, c_w, c_h
            ));
            score -= 0.2;
        }

        // 3. Bitrate vs network check
        let max_net_kbps = (context.network_speed_mbps * 1000.0 * 0.5) as u32;
        if max_net_kbps > 0 && proxy.bitrate_kbps > max_net_kbps {
            warnings.push(format!(
                "Proxy bitrate {} kbps exceeds safe network limit {} kbps",
                proxy.bitrate_kbps, max_net_kbps
            ));
            score -= 0.3;
        }

        // 4. Codec support list (if provided)
        if !context.codec_support.is_empty()
            && !context
                .codec_support
                .iter()
                .any(|c| c == proxy.codec.as_str())
        {
            warnings.push(format!(
                "Proxy codec '{}' is not in the supported codec list",
                proxy.codec
            ));
            score -= 0.3;
        }

        CompatibilityResult {
            compatible: score > 0.4,
            warnings,
            score: score.clamp(0.0, 1.0),
        }
    }
}

/// Estimates disk storage required for a batch of proxy files.
pub struct ProxyStorageEstimator;

impl ProxyStorageEstimator {
    /// Estimate the total storage in gigabytes.
    ///
    /// # Arguments
    /// * `source_count` – Number of source files.
    /// * `avg_duration_mins` – Average duration of each source file in minutes.
    /// * `spec` – Proxy specification (bitrate drives the estimate).
    #[must_use]
    pub fn estimate_gb(source_count: u32, avg_duration_mins: f32, spec: &ProxySpec) -> f64 {
        if source_count == 0 || avg_duration_mins <= 0.0 {
            return 0.0;
        }
        // bitrate_kbps * 1000 bits/kbit → bits/s
        // * 60 seconds/minute * avg_duration_mins → total bits
        // / 8 → bytes, / 1e9 → gigabytes
        let bits_per_file = spec.bitrate_kbps as f64 * 1_000.0 * 60.0 * avg_duration_mins as f64;
        let bytes_per_file = bits_per_file / 8.0;
        let total_bytes = bytes_per_file * source_count as f64;
        total_bytes / 1_000_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(software: EditingSoftware, res: (u32, u32), net: f32) -> EditingContext {
        EditingContext::new(software, res, vec![], net)
    }

    #[test]
    fn test_editing_software_preferred_codec() {
        assert_eq!(
            EditingSoftware::Resolve.preferred_proxy_codec(),
            "prores_proxy"
        );
        assert_eq!(EditingSoftware::Premiere.preferred_proxy_codec(), "h264");
        assert_eq!(EditingSoftware::Avid.preferred_proxy_codec(), "dnxhd");
        assert_eq!(
            EditingSoftware::FinalCut.preferred_proxy_codec(),
            "prores_proxy"
        );
        assert_eq!(EditingSoftware::Vegas.preferred_proxy_codec(), "h264");
        assert_eq!(EditingSoftware::Kdenlive.preferred_proxy_codec(), "h264");
    }

    #[test]
    fn test_recommender_codec_matches_software() {
        let ctx = make_context(EditingSoftware::Avid, (1920, 1080), 1000.0);
        let src = vec![SourceSpec::new(
            "/a.mov",
            (3840, 2160),
            "h264",
            100_000,
            25.0,
        )];
        let recs = ProxyRecommender::recommend(&ctx, &src);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].codec, "dnxhd");
    }

    #[test]
    fn test_recommender_does_not_upscale() {
        let ctx = make_context(EditingSoftware::Premiere, (3840, 2160), 1000.0);
        let src = vec![SourceSpec::new(
            "/b.mov",
            (1920, 1080),
            "h264",
            10_000,
            25.0,
        )];
        let recs = ProxyRecommender::recommend(&ctx, &src);
        assert_eq!(recs[0].resolution, (1920, 1080));
    }

    #[test]
    fn test_recommender_network_cap() {
        // Network: 1 Mbit/s → max 500 kbps usable
        let ctx = make_context(EditingSoftware::Premiere, (1920, 1080), 1.0);
        let src = vec![SourceSpec::new(
            "/c.mov",
            (1920, 1080),
            "h264",
            100_000,
            25.0,
        )];
        let recs = ProxyRecommender::recommend(&ctx, &src);
        assert!(recs[0].bitrate_kbps <= 500);
    }

    #[test]
    fn test_recommender_empty_sources() {
        let ctx = make_context(EditingSoftware::Resolve, (1920, 1080), 100.0);
        let recs = ProxyRecommender::recommend(&ctx, &[]);
        assert!(recs.is_empty());
    }

    #[test]
    fn test_compatibility_check_perfect() {
        let proxy = ProxySpec::new((1920, 1080), "prores_proxy", 10_000);
        let ctx = EditingContext::new(
            EditingSoftware::Resolve,
            (1920, 1080),
            vec!["prores_proxy".to_string()],
            1000.0,
        );
        let result = ProxyCompatibilityChecker::check(&proxy, &ctx);
        assert!(result.compatible);
        assert!(result.warnings.is_empty());
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_compatibility_check_wrong_codec() {
        let proxy = ProxySpec::new((1920, 1080), "h264", 10_000);
        let ctx = EditingContext::new(
            EditingSoftware::Avid,
            (1920, 1080),
            vec!["dnxhd".to_string()],
            1000.0,
        );
        let result = ProxyCompatibilityChecker::check(&proxy, &ctx);
        assert!(!result.warnings.is_empty());
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_compatibility_check_resolution_too_large() {
        let proxy = ProxySpec::new((3840, 2160), "h264", 5_000);
        let ctx = make_context(EditingSoftware::Premiere, (1920, 1080), 1000.0);
        let result = ProxyCompatibilityChecker::check(&proxy, &ctx);
        assert!(!result.warnings.is_empty());
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_compatibility_check_bitrate_too_high() {
        // Network 1 Mbit/s → max 500 kbps
        let proxy = ProxySpec::new((1280, 720), "h264", 50_000);
        let ctx = make_context(EditingSoftware::Premiere, (1920, 1080), 1.0);
        let result = ProxyCompatibilityChecker::check(&proxy, &ctx);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_storage_estimator_basic() {
        let spec = ProxySpec::new((1920, 1080), "h264", 8_000);
        let gb = ProxyStorageEstimator::estimate_gb(100, 5.0, &spec);
        // 8000 kbps * 1000 * 60 * 5 / 8 / 1e9 * 100 = 30 GB
        assert!((gb - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_storage_estimator_zero_files() {
        let spec = ProxySpec::new((1920, 1080), "h264", 8_000);
        let gb = ProxyStorageEstimator::estimate_gb(0, 5.0, &spec);
        assert!((gb - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_storage_estimator_zero_duration() {
        let spec = ProxySpec::new((1920, 1080), "h264", 8_000);
        let gb = ProxyStorageEstimator::estimate_gb(10, 0.0, &spec);
        assert!((gb - 0.0).abs() < f64::EPSILON);
    }
}

// ============================================================================
// Multi-Resolution Smart Proxy
// ============================================================================

/// A named proxy variant at a specific resolution tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionVariant {
    /// Human-readable tier label (e.g. "quarter", "half", "full").
    pub label: String,
    /// Fraction of source resolution (0.0 < scale ≤ 1.0).
    pub scale: f32,
    /// Explicit resolution (width, height), computed from source + scale.
    pub resolution: (u32, u32),
    /// Codec identifier.
    pub codec: String,
    /// Bitrate in kbit/s.
    pub bitrate_kbps: u32,
}

impl ResolutionVariant {
    /// Construct a variant from a source resolution and a scale factor.
    ///
    /// Width and height are rounded down to the nearest even number to keep
    /// them compatible with YUV 4:2:0 encoding.
    #[must_use]
    pub fn from_source(
        label: impl Into<String>,
        source: (u32, u32),
        scale: f32,
        codec: impl Into<String>,
        base_bitrate_kbps: u32,
    ) -> Self {
        let (sw, sh) = source;
        let w = ((sw as f32 * scale) as u32) & !1; // even
        let h = ((sh as f32 * scale) as u32) & !1;
        let actual_scale = (w * h) as f32 / ((sw * sh).max(1) as f32);
        let bitrate = (base_bitrate_kbps as f32 * actual_scale) as u32;
        Self {
            label: label.into(),
            scale,
            resolution: (w.max(2), h.max(2)),
            codec: codec.into(),
            bitrate_kbps: bitrate.max(200),
        }
    }
}

/// A multi-resolution proxy set containing quarter, half, and full variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiResolutionProxy {
    /// Source file path.
    pub source_path: String,
    /// Quarter-resolution variant (25% area).
    pub quarter: ResolutionVariant,
    /// Half-resolution variant (50% area).
    pub half: ResolutionVariant,
    /// Full-resolution variant.
    pub full: ResolutionVariant,
}

impl MultiResolutionProxy {
    /// Build a complete multi-resolution proxy set from a source spec.
    ///
    /// All three variants share the same codec as the preferred NLE codec for
    /// `software`. Bitrates are derived from `base_bitrate_kbps` scaled by
    /// pixel-area ratio.
    #[must_use]
    pub fn from_source(
        source_path: impl Into<String>,
        source_resolution: (u32, u32),
        codec: impl Into<String>,
        base_bitrate_kbps: u32,
    ) -> Self {
        let codec = codec.into();
        let path = source_path.into();
        Self {
            source_path: path,
            quarter: ResolutionVariant::from_source(
                "quarter",
                source_resolution,
                0.5, // 0.5 linear → 0.25 area
                &codec,
                base_bitrate_kbps,
            ),
            half: ResolutionVariant::from_source(
                "half",
                source_resolution,
                0.707, // √0.5 linear → 0.5 area
                &codec,
                base_bitrate_kbps,
            ),
            full: ResolutionVariant::from_source(
                "full",
                source_resolution,
                1.0,
                &codec,
                base_bitrate_kbps,
            ),
        }
    }
}

/// Selects the best `ResolutionVariant` for a given display window size.
///
/// The algorithm picks the smallest variant whose resolution is at least as
/// large as the display window in both dimensions, falling back to `full` if
/// no variant meets the threshold.
pub struct DisplayAwareSelector;

impl DisplayAwareSelector {
    /// Choose the appropriate variant for a display of `(display_w, display_h)` pixels.
    ///
    /// Selection rules (applied in order, first match wins):
    /// 1. If display area ≤ quarter area → use `quarter`
    /// 2. If display area ≤ half area    → use `half`
    /// 3. Otherwise                      → use `full`
    #[must_use]
    pub fn select<'a>(
        proxy: &'a MultiResolutionProxy,
        display: (u32, u32),
    ) -> &'a ResolutionVariant {
        let display_area = display.0 as u64 * display.1 as u64;
        let (qw, qh) = proxy.quarter.resolution;
        let quarter_area = qw as u64 * qh as u64;
        let (hw, hh) = proxy.half.resolution;
        let half_area = hw as u64 * hh as u64;

        if display_area <= quarter_area {
            &proxy.quarter
        } else if display_area <= half_area {
            &proxy.half
        } else {
            &proxy.full
        }
    }

    /// Select and return the label string for the chosen variant.
    #[must_use]
    pub fn select_label(proxy: &MultiResolutionProxy, display: (u32, u32)) -> &str {
        Self::select(proxy, display).label.as_str()
    }
}

#[cfg(test)]
mod multi_res_tests {
    use super::*;

    fn make_proxy() -> MultiResolutionProxy {
        MultiResolutionProxy::from_source("/src/4k.mov", (3840, 2160), "h264", 50_000)
    }

    #[test]
    fn test_multi_res_variants_created() {
        let p = make_proxy();
        // quarter: 3840*0.5=1920, 2160*0.5=1080
        assert_eq!(p.quarter.resolution, (1920, 1080));
        // half: 3840*0.707≈2714 (even), 2160*0.707≈1527 (even)
        let (hw, hh) = p.half.resolution;
        assert!(hw > 1920 && hw < 3840);
        assert!(hh > 1080 && hh < 2160);
        // full
        assert_eq!(p.full.resolution, (3840, 2160));
    }

    #[test]
    fn test_variant_label() {
        let p = make_proxy();
        assert_eq!(p.quarter.label, "quarter");
        assert_eq!(p.half.label, "half");
        assert_eq!(p.full.label, "full");
    }

    #[test]
    fn test_variant_scale() {
        let p = make_proxy();
        assert!((p.quarter.scale - 0.5).abs() < 1e-3);
        assert!((p.full.scale - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_display_aware_selects_quarter_for_small_display() {
        let p = make_proxy();
        // 960×540 is smaller than quarter (1920×1080)
        let label = DisplayAwareSelector::select_label(&p, (960, 540));
        assert_eq!(label, "quarter");
    }

    #[test]
    fn test_display_aware_selects_half_for_medium_display() {
        let p = make_proxy();
        // 1920×1080 fits quarter exactly — select quarter
        // 2000×1200 > quarter (1920×1080) but ≤ half → half
        let label = DisplayAwareSelector::select_label(&p, (2000, 1200));
        assert_eq!(label, "half");
    }

    #[test]
    fn test_display_aware_selects_full_for_large_display() {
        let p = make_proxy();
        // 3840×2160 is the full display
        let label = DisplayAwareSelector::select_label(&p, (3840, 2160));
        assert_eq!(label, "full");
    }

    #[test]
    fn test_display_aware_exact_quarter_area() {
        let p = make_proxy();
        let (qw, qh) = p.quarter.resolution;
        // Exactly the quarter area should resolve to quarter
        let label = DisplayAwareSelector::select_label(&p, (qw, qh));
        assert_eq!(label, "quarter");
    }

    #[test]
    fn test_bitrate_scales_with_area() {
        let p = make_proxy();
        // Quarter has smaller bitrate than half which has smaller than full
        assert!(p.quarter.bitrate_kbps < p.half.bitrate_kbps);
        assert!(p.half.bitrate_kbps < p.full.bitrate_kbps);
    }

    #[test]
    fn test_select_returns_reference() {
        let p = make_proxy();
        let variant = DisplayAwareSelector::select(&p, (640, 360));
        assert!(!variant.label.is_empty());
    }

    #[test]
    fn test_multi_res_codec_propagated() {
        let p = MultiResolutionProxy::from_source("/a.mov", (1920, 1080), "prores_proxy", 20_000);
        assert_eq!(p.quarter.codec, "prores_proxy");
        assert_eq!(p.half.codec, "prores_proxy");
        assert_eq!(p.full.codec, "prores_proxy");
    }

    #[test]
    fn test_resolution_variant_from_source_even_dimensions() {
        // Source 1001×999 → even rounding
        let v = ResolutionVariant::from_source("half", (1001, 999), 0.5, "h264", 10_000);
        assert_eq!(v.resolution.0 % 2, 0);
        assert_eq!(v.resolution.1 % 2, 0);
    }
}
