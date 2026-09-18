//! Proxy/offline-to-online conform workflow with resolution scaling.
//!
//! In professional post-production, editors frequently cut with *proxy* media —
//! lower-resolution (often 1/2, 1/4, or 1/8 of the online resolution) copies
//! that are faster to work with in a non-linear editor.  At delivery time the
//! proxy timeline must be *conformed* back to the full-resolution online media.
//!
//! This module provides:
//!
//! * **[`ProxyResolution`]** — canonical proxy scales (½, ¼, ⅛, 1:1).
//! * **[`ProxyConformConfig`]** — parameters controlling how proxy paths are
//!   mapped to online paths and how timecodes/frame counts are translated.
//! * **[`ProxyConformTranslator`]** — translates a list of proxy
//!   [`ClipReference`]s to online equivalents, adjusting frame counts and
//!   resolving file paths.
//! * **[`ProxyRelinkStrategy`]** — configurable path relink strategy
//!   (directory swap, suffix replacement, regex substitution).
//!
//! # Frame count scaling
//!
//! When the proxy frame rate differs from the online frame rate (e.g. 23.976
//! proxy → 23.976 online — same; or 29.97 proxy → 23.976 online pull-down),
//! each timecode's frame count is scaled by `online_fps / proxy_fps`.
//! For same-rate proxies the only change is the file path.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_conform::proxy_conform::{ProxyConformConfig, ProxyConformTranslator, ProxyRelinkStrategy, ProxyResolution};
//!
//! let config = ProxyConformConfig::new(ProxyResolution::Quarter)
//!     .with_relink(ProxyRelinkStrategy::DirectorySwap {
//!         proxy_dir: "/media/proxies".into(),
//!         online_dir: "/media/online".into(),
//!     });
//!
//! let translator = ProxyConformTranslator::new(config);
//! // translator.translate_clips(&proxy_clips) → Vec<ClipReference>
//! ```

use crate::error::ConformResult;
use crate::types::{ClipReference, FrameRate};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────────────────────
// ProxyResolution
// ─────────────────────────────────────────────────────────────────────────────

/// Proxy resolution relative to online media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProxyResolution {
    /// Full resolution — no scaling (1:1 match).
    Full,
    /// Half resolution on each axis (½ × ½ = ¼ total pixels).
    Half,
    /// Quarter resolution on each axis (¼ × ¼ = 1/16 total pixels).
    Quarter,
    /// Eighth resolution on each axis.
    Eighth,
    /// Custom scale: `(numerator, denominator)` applied to each axis.
    Custom(u32, u32),
}

impl ProxyResolution {
    /// Scale factor for one axis as a rational: `(num, den)`.
    ///
    /// Returns `(1, 1)` for [`Full`][ProxyResolution::Full].
    #[must_use]
    pub const fn axis_scale(self) -> (u32, u32) {
        match self {
            Self::Full => (1, 1),
            Self::Half => (1, 2),
            Self::Quarter => (1, 4),
            Self::Eighth => (1, 8),
            Self::Custom(n, d) => (n, d),
        }
    }

    /// Scale a dimension (width or height) from online to proxy space.
    #[must_use]
    pub fn scale_online_to_proxy(self, dim: u32) -> u32 {
        let (n, d) = self.axis_scale();
        if let Some(q) = (dim * n + d / 2).checked_div(d) {
            q
        } else {
            dim
        }
    }

    /// Scale a dimension from proxy space back to online.
    #[must_use]
    pub fn scale_proxy_to_online(self, dim: u32) -> u32 {
        let (n, d) = self.axis_scale();
        if let Some(q) = (dim * d + n / 2).checked_div(n) {
            q
        } else {
            dim
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Full => "1:1".to_string(),
            Self::Half => "1:2".to_string(),
            Self::Quarter => "1:4".to_string(),
            Self::Eighth => "1:8".to_string(),
            Self::Custom(n, d) => format!("{n}:{d}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProxyRelinkStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// How proxy file paths are translated to online file paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProxyRelinkStrategy {
    /// Swap a proxy directory prefix with an online directory prefix.
    ///
    /// Example: `/media/proxies/shot_001.mov` → `/media/online/shot_001.mov`
    DirectorySwap {
        /// Directory prefix that identifies proxy files.
        proxy_dir: PathBuf,
        /// Directory prefix to substitute for the online files.
        online_dir: PathBuf,
    },
    /// Replace a filename suffix (before extension) with another suffix.
    ///
    /// Example: `shot_001_proxy.mov` → `shot_001.mov` (suffix `"_proxy"` → `""`)
    SuffixReplace {
        /// Filename suffix to remove (without extension).
        proxy_suffix: String,
        /// Replacement suffix to add (empty string to just remove).
        online_suffix: String,
    },
    /// Replace a directory name component in the path.
    ///
    /// Example: `.../Proxy/...` → `.../Online/...`
    DirectoryComponent {
        /// Directory name to replace.
        proxy_component: String,
        /// Replacement directory name.
        online_component: String,
    },
    /// Keep the same path unchanged.  Used when proxy and online share the same
    /// storage location (e.g. scaled MXF with same filename).
    Identity,
}

impl ProxyRelinkStrategy {
    /// Translate a proxy file path to the corresponding online path.
    ///
    /// Returns `None` if the path does not match the strategy's pattern.
    #[must_use]
    pub fn translate(&self, proxy_path: &Path) -> Option<PathBuf> {
        match self {
            Self::Identity => Some(proxy_path.to_path_buf()),

            Self::DirectorySwap {
                proxy_dir,
                online_dir,
            } => proxy_path
                .strip_prefix(proxy_dir)
                .ok()
                .map(|rel| online_dir.join(rel)),

            Self::SuffixReplace {
                proxy_suffix,
                online_suffix,
            } => {
                let stem = proxy_path.file_stem()?.to_str()?;
                let ext = proxy_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let online_stem = if stem.ends_with(proxy_suffix.as_str()) {
                    let new_stem = &stem[..stem.len() - proxy_suffix.len()];
                    format!("{new_stem}{online_suffix}")
                } else {
                    return None; // pattern not matched
                };
                let online_filename = if ext.is_empty() {
                    online_stem
                } else {
                    format!("{online_stem}.{ext}")
                };
                let parent = proxy_path.parent().unwrap_or(Path::new(""));
                Some(parent.join(online_filename))
            }

            Self::DirectoryComponent {
                proxy_component,
                online_component,
            } => {
                let proxy_str = proxy_path.to_str()?;
                // Replace only the first occurrence of the directory component
                let needle = format!("/{proxy_component}/");
                let replacement = format!("/{online_component}/");
                if let Some(pos) = proxy_str.find(&needle) {
                    let mut result = proxy_str.to_string();
                    result.replace_range(pos..pos + needle.len(), &replacement);
                    Some(PathBuf::from(result))
                } else {
                    None
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProxyConformConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a proxy-to-online conform session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConformConfig {
    /// Proxy resolution relative to online.
    pub proxy_resolution: ProxyResolution,
    /// Path relink strategy.
    pub relink_strategy: ProxyRelinkStrategy,
    /// Frame rate of the proxy media.
    pub proxy_fps: Option<FrameRate>,
    /// Frame rate of the online media (if different from proxy).
    /// `None` means same as proxy.
    pub online_fps: Option<FrameRate>,
    /// Whether to fail hard if a path cannot be translated.
    pub strict: bool,
}

impl ProxyConformConfig {
    /// Create a config for the given proxy resolution with identity relinking.
    #[must_use]
    pub fn new(proxy_resolution: ProxyResolution) -> Self {
        Self {
            proxy_resolution,
            relink_strategy: ProxyRelinkStrategy::Identity,
            proxy_fps: None,
            online_fps: None,
            strict: true,
        }
    }

    /// Set the relink strategy.
    #[must_use]
    pub fn with_relink(mut self, strategy: ProxyRelinkStrategy) -> Self {
        self.relink_strategy = strategy;
        self
    }

    /// Set proxy frame rate.
    #[must_use]
    pub fn with_proxy_fps(mut self, fps: FrameRate) -> Self {
        self.proxy_fps = Some(fps);
        self
    }

    /// Set online frame rate.
    #[must_use]
    pub fn with_online_fps(mut self, fps: FrameRate) -> Self {
        self.online_fps = Some(fps);
        self
    }

    /// Enable/disable strict mode.
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Returns the effective online FPS (same as proxy if not overridden).
    #[must_use]
    pub fn effective_online_fps(&self, proxy_fps: FrameRate) -> FrameRate {
        self.online_fps.unwrap_or(proxy_fps)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProxyConformTranslator
// ─────────────────────────────────────────────────────────────────────────────

/// Translates proxy [`ClipReference`]s to online equivalents.
pub struct ProxyConformTranslator {
    config: ProxyConformConfig,
}

impl ProxyConformTranslator {
    /// Create a translator from a config.
    #[must_use]
    pub fn new(config: ProxyConformConfig) -> Self {
        Self { config }
    }

    /// Translate a single [`ClipReference`] from proxy to online.
    ///
    /// Path translation is applied to `source_file`.  If the path cannot be
    /// translated and `config.strict` is `true`, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if path translation fails in strict mode.
    pub fn translate_clip(&self, clip: &ClipReference) -> ConformResult<ClipReference> {
        let mut online_clip = clip.clone();

        // ── 1. Relink source file path ─────────────────────────────────────
        if let Some(ref proxy_path_str) = clip.source_file {
            let proxy_path = Path::new(proxy_path_str);
            match self.config.relink_strategy.translate(proxy_path) {
                Some(online_path) => {
                    online_clip.source_file =
                        Some(online_path.to_str().unwrap_or(proxy_path_str).to_string());
                }
                None => {
                    // Always return an error for path translation failure.
                    // In non-strict mode, translate_clips() catches this and
                    // records it as a failure rather than aborting.
                    return Err(crate::error::ConformError::Edl(format!(
                        "ProxyConformTranslator: cannot translate path '{proxy_path_str}' with current strategy"
                    )));
                }
            }
        }

        // ── 2. Frame-rate scaling ──────────────────────────────────────────
        let proxy_fps = self.config.proxy_fps.unwrap_or(clip.fps);
        let online_fps = self.config.effective_online_fps(proxy_fps);

        if (proxy_fps.as_f64() - online_fps.as_f64()).abs() > 1e-4 {
            let scale = online_fps.as_f64() / proxy_fps.as_f64();
            online_clip.source_in = scale_timecode(clip.source_in, proxy_fps, online_fps, scale);
            online_clip.source_out = scale_timecode(clip.source_out, proxy_fps, online_fps, scale);
            online_clip.record_in = scale_timecode(clip.record_in, clip.fps, online_fps, scale);
            online_clip.record_out = scale_timecode(clip.record_out, clip.fps, online_fps, scale);
        }

        online_clip.fps = online_fps;

        // ── 3. Annotate metadata ───────────────────────────────────────────
        online_clip.metadata.insert(
            "proxy_resolution".to_string(),
            self.config.proxy_resolution.label(),
        );
        online_clip
            .metadata
            .insert("conformed_from_proxy".to_string(), "true".to_string());

        Ok(online_clip)
    }

    /// Translate a batch of proxy clips to online equivalents.
    ///
    /// Errors accumulate into a [`ProxyConformReport`]; individual failures only
    /// abort the entire batch if `config.strict` is `true`.
    ///
    /// # Errors
    ///
    /// Returns an error in strict mode if any clip fails to translate.
    pub fn translate_clips(&self, clips: &[ClipReference]) -> ConformResult<ProxyConformReport> {
        let mut translated = Vec::with_capacity(clips.len());
        let mut failures: Vec<ProxyConformFailure> = Vec::new();

        for clip in clips {
            match self.translate_clip(clip) {
                Ok(online) => translated.push(online),
                Err(e) => {
                    if self.config.strict {
                        return Err(e);
                    }
                    failures.push(ProxyConformFailure {
                        clip_id: clip.id.clone(),
                        reason: e.to_string(),
                    });
                    translated.push(clip.clone()); // keep proxy as fallback
                }
            }
        }

        Ok(ProxyConformReport {
            translated_clips: translated,
            failures,
            proxy_resolution: self.config.proxy_resolution,
        })
    }
}

/// Scale a timecode's frame count by `scale = online_fps / proxy_fps`.
fn scale_timecode(
    tc: crate::types::Timecode,
    from_fps: FrameRate,
    to_fps: FrameRate,
    scale: f64,
) -> crate::types::Timecode {
    let proxy_frames = tc.to_frames(from_fps);
    let online_frames = (proxy_frames as f64 * scale).round() as u64;
    crate::types::Timecode::from_frames(online_frames, to_fps)
}

// ─────────────────────────────────────────────────────────────────────────────
// ProxyConformReport
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a proxy-to-online translation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConformReport {
    /// Translated (online) clip references.
    pub translated_clips: Vec<ClipReference>,
    /// Any clips that could not be translated (only present in non-strict mode).
    pub failures: Vec<ProxyConformFailure>,
    /// Proxy resolution that was applied.
    pub proxy_resolution: ProxyResolution,
}

impl ProxyConformReport {
    /// Returns `true` if all clips were successfully translated.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty()
    }

    /// Number of successfully translated clips.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.translated_clips.len() - self.failures.len()
    }
}

/// A single proxy conform failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConformFailure {
    /// ID of the clip that failed to translate.
    pub clip_id: String,
    /// Human-readable reason for the failure.
    pub reason: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameRate, Timecode, TrackType};
    use std::collections::HashMap;

    fn make_clip(id: &str, source_file: Option<&str>) -> ClipReference {
        let tc = Timecode::new(1, 0, 0, 0);
        let tc_out = Timecode::new(1, 0, 5, 0);
        let mut meta = HashMap::new();
        meta.insert("reel".to_string(), "R1".to_string());
        ClipReference {
            id: id.to_string(),
            source_file: source_file.map(str::to_string),
            source_in: tc,
            source_out: tc_out,
            record_in: tc,
            record_out: tc_out,
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: meta,
        }
    }

    // ── ProxyResolution ────────────────────────────────────────────────────

    #[test]
    fn test_resolution_axis_scale() {
        assert_eq!(ProxyResolution::Full.axis_scale(), (1, 1));
        assert_eq!(ProxyResolution::Half.axis_scale(), (1, 2));
        assert_eq!(ProxyResolution::Quarter.axis_scale(), (1, 4));
        assert_eq!(ProxyResolution::Eighth.axis_scale(), (1, 8));
    }

    #[test]
    fn test_resolution_scale_round_trip() {
        let res = ProxyResolution::Quarter;
        let online_dim = 1920u32;
        let proxy_dim = res.scale_online_to_proxy(online_dim);
        assert_eq!(proxy_dim, 480);
        let back = res.scale_proxy_to_online(proxy_dim);
        assert_eq!(back, online_dim);
    }

    #[test]
    fn test_resolution_label() {
        assert_eq!(ProxyResolution::Quarter.label(), "1:4");
        assert_eq!(ProxyResolution::Custom(1, 3).label(), "1:3");
    }

    // ── ProxyRelinkStrategy ────────────────────────────────────────────────

    #[test]
    fn test_directory_swap() {
        let strategy = ProxyRelinkStrategy::DirectorySwap {
            proxy_dir: PathBuf::from("/media/proxies"),
            online_dir: PathBuf::from("/media/online"),
        };
        let result = strategy
            .translate(Path::new("/media/proxies/scene_01/shot_001.mov"))
            .expect("should translate");
        assert_eq!(result, PathBuf::from("/media/online/scene_01/shot_001.mov"));
    }

    #[test]
    fn test_directory_swap_no_match() {
        let strategy = ProxyRelinkStrategy::DirectorySwap {
            proxy_dir: PathBuf::from("/media/proxies"),
            online_dir: PathBuf::from("/media/online"),
        };
        // Path not under proxy_dir
        let result = strategy.translate(Path::new("/media/other/shot.mov"));
        assert!(result.is_none());
    }

    #[test]
    fn test_suffix_replace() {
        let strategy = ProxyRelinkStrategy::SuffixReplace {
            proxy_suffix: "_proxy".to_string(),
            online_suffix: String::new(),
        };
        let result = strategy
            .translate(Path::new("/media/shot_001_proxy.mov"))
            .expect("should translate");
        assert_eq!(result, PathBuf::from("/media/shot_001.mov"));
    }

    #[test]
    fn test_suffix_replace_no_match() {
        let strategy = ProxyRelinkStrategy::SuffixReplace {
            proxy_suffix: "_proxy".to_string(),
            online_suffix: String::new(),
        };
        let result = strategy.translate(Path::new("/media/shot_001.mov"));
        assert!(result.is_none());
    }

    #[test]
    fn test_directory_component_swap() {
        let strategy = ProxyRelinkStrategy::DirectoryComponent {
            proxy_component: "Proxy".to_string(),
            online_component: "Online".to_string(),
        };
        let result = strategy
            .translate(Path::new("/media/Proxy/shot_001.mov"))
            .expect("should translate");
        assert_eq!(result, PathBuf::from("/media/Online/shot_001.mov"));
    }

    #[test]
    fn test_identity_strategy() {
        let strategy = ProxyRelinkStrategy::Identity;
        let path = Path::new("/media/shot.mov");
        let result = strategy.translate(path).expect("identity always succeeds");
        assert_eq!(result, path);
    }

    // ── ProxyConformTranslator ─────────────────────────────────────────────

    #[test]
    fn test_translate_clip_directory_swap() {
        let config = ProxyConformConfig::new(ProxyResolution::Quarter).with_relink(
            ProxyRelinkStrategy::DirectorySwap {
                proxy_dir: PathBuf::from("/proxy"),
                online_dir: PathBuf::from("/online"),
            },
        );
        let translator = ProxyConformTranslator::new(config);
        let clip = make_clip("c1", Some("/proxy/scene/shot.mov"));
        let online = translator.translate_clip(&clip).expect("should succeed");
        assert_eq!(
            online.source_file.as_deref(),
            Some("/online/scene/shot.mov")
        );
        assert_eq!(
            online.metadata.get("proxy_resolution").map(|s| s.as_str()),
            Some("1:4")
        );
        assert_eq!(
            online
                .metadata
                .get("conformed_from_proxy")
                .map(|s| s.as_str()),
            Some("true")
        );
    }

    #[test]
    fn test_translate_clip_identity_no_path_change() {
        let config = ProxyConformConfig::new(ProxyResolution::Half);
        let translator = ProxyConformTranslator::new(config);
        let clip = make_clip("c1", Some("/media/shot.mov"));
        let online = translator.translate_clip(&clip).expect("should succeed");
        assert_eq!(online.source_file.as_deref(), Some("/media/shot.mov"));
    }

    #[test]
    fn test_translate_clips_batch() {
        let config = ProxyConformConfig::new(ProxyResolution::Quarter).with_relink(
            ProxyRelinkStrategy::DirectorySwap {
                proxy_dir: PathBuf::from("/proxy"),
                online_dir: PathBuf::from("/online"),
            },
        );
        let translator = ProxyConformTranslator::new(config);
        let clips = vec![
            make_clip("c1", Some("/proxy/a.mov")),
            make_clip("c2", Some("/proxy/b.mov")),
        ];
        let report = translator.translate_clips(&clips).expect("should succeed");
        assert!(report.is_clean());
        assert_eq!(report.translated_clips.len(), 2);
        assert_eq!(
            report.translated_clips[0].source_file.as_deref(),
            Some("/online/a.mov")
        );
    }

    #[test]
    fn test_non_strict_partial_failure() {
        let config = ProxyConformConfig::new(ProxyResolution::Half)
            .with_relink(ProxyRelinkStrategy::SuffixReplace {
                proxy_suffix: "_proxy".to_string(),
                online_suffix: String::new(),
            })
            .with_strict(false);
        let translator = ProxyConformTranslator::new(config);
        let clips = vec![
            make_clip("c1", Some("/media/shot_001_proxy.mov")), // matches
            make_clip("c2", Some("/media/shot_002.mov")),       // no match → fallback
        ];
        let report = translator.translate_clips(&clips).expect("should succeed");
        assert_eq!(report.translated_clips.len(), 2);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].clip_id, "c2");
        assert!(!report.is_clean());
    }

    #[test]
    fn test_proxy_resolution_full_is_identity_scale() {
        let res = ProxyResolution::Full;
        assert_eq!(res.scale_online_to_proxy(1920), 1920);
        assert_eq!(res.scale_proxy_to_online(1080), 1080);
    }
}
