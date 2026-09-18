//! AI-assisted automatic content tagging (pure Rust, rule-based).
//!
//! Provides an extensible tagging pipeline where multiple [`TaggingBackend`]
//! implementations contribute [`AutoTag`] candidates.  The built-in
//! [`RuleBasedTagger`] uses deterministic signal analysis (dominant-colour
//! histograms, luminance, zero-crossing rate, aspect ratio, codec names …)
//! rather than any ML model or C/Fortran library.
//!
//! # Quick start
//!
//! ```text
//! let tagger = AutoTagger::new().with_rule_based();
//! let meta = ContentMetadata { filename: "demo.mp4".into(), … };
//! let tags = tagger.tag_content(&meta, None, None);
//! ```

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A single automatically generated tag produced by a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTag {
    /// The tag label string (lower-case, hyphenated).
    pub tag: String,
    /// Confidence score in the range `0.0..=1.0`.
    pub confidence: f32,
    /// Semantic category of the tag.
    pub category: TagCategory,
    /// Name of the backend that produced this tag.
    pub source: String,
}

/// Semantic category for an [`AutoTag`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TagCategory {
    /// Indoor / outdoor / landscape / urban scene classification.
    Scene,
    /// Detected objects (person, car, building, …).
    Object,
    /// Dominant or notable colours.
    Color,
    /// Mood / emotion signals (happy, energetic, calm, …).
    Emotion,
    /// Music genre (rock, classical, …).
    Genre,
    /// Perceptual quality (high-quality, noisy, blurry, …).
    Quality,
    /// Technical attributes (codec, resolution, format, …).
    Technical,
    /// User-defined category.
    Custom(String),
}

/// Descriptive metadata about a piece of content, used by rule-based backends.
#[derive(Debug, Clone)]
pub struct ContentMetadata {
    /// Original filename including extension.
    pub filename: String,
    /// Lowercase file extension without leading dot.
    pub extension: String,
    /// Duration in seconds, if known.
    pub duration_secs: Option<f64>,
    /// Frame width in pixels.
    pub width: Option<u32>,
    /// Frame height in pixels.
    pub height: Option<u32>,
    /// Video codec identifier (e.g. `"av1"`, `"h264"`).
    pub video_codec: Option<String>,
    /// Audio codec identifier (e.g. `"opus"`, `"aac"`).
    pub audio_codec: Option<String>,
    /// Nominal bitrate in kbps.
    pub bitrate_kbps: Option<u32>,
    /// File size in bytes.
    pub file_size_bytes: u64,
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// A pluggable tagging backend.
///
/// All methods are infallible — implementations must handle errors internally
/// and return an empty `Vec` on failure.
pub trait TaggingBackend: Send + Sync {
    /// Analyse raw pixel data (`RGBA` or `RGB` bytes, row-major) and return tags.
    fn tag_image(&self, pixels: &[u8], width: u32, height: u32) -> Vec<AutoTag>;
    /// Analyse mono PCM audio samples (normalised to `-1.0..=1.0`) and return tags.
    fn tag_audio(&self, samples: &[f32], sample_rate: u32) -> Vec<AutoTag>;
    /// Derive tags from file metadata alone.
    fn tag_metadata(&self, metadata: &ContentMetadata) -> Vec<AutoTag>;
    /// Human-readable name of this backend.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// RuleBasedTagger
// ---------------------------------------------------------------------------

/// Deterministic, rule-based tagging backend — no ML, no C FFI.
pub struct RuleBasedTagger;

/// Number of bytes per pixel for an RGB pixel row.
const BYTES_PER_PIXEL_RGB: usize = 3;

impl RuleBasedTagger {
    fn source() -> String {
        "rule-based".to_string()
    }

    // -----------------------------------------------------------------------
    // Image helpers
    // -----------------------------------------------------------------------

    /// Detect the bytes-per-pixel stride (4 for RGBA, 3 for RGB, or raw guess).
    fn detect_stride(pixels: &[u8], width: u32, height: u32) -> usize {
        let pixel_count = (width as usize).saturating_mul(height as usize);
        if pixel_count == 0 {
            return BYTES_PER_PIXEL_RGB;
        }
        if pixels.len() % pixel_count == 0 {
            pixels.len() / pixel_count
        } else {
            BYTES_PER_PIXEL_RGB
        }
    }

    /// Compute mean luminance (Y′ ≈ 0.299·R + 0.587·G + 0.114·B).
    fn mean_luminance(pixels: &[u8], stride: usize) -> f32 {
        if pixels.is_empty() || stride < 3 {
            return 0.0;
        }
        let chunks = pixels.chunks_exact(stride);
        let mut sum = 0u64;
        let mut count = 0u64;
        for chunk in chunks {
            let r = chunk[0] as u64;
            let g = chunk[1] as u64;
            let b = chunk[2] as u64;
            sum += 299 * r + 587 * g + 114 * b;
            count += 1000;
        }
        if count == 0 {
            return 0.0;
        }
        (sum / count) as f32 / 255.0
    }

    /// Dominant colour analysis: split image into 8 horizontal zones and
    /// accumulate per-channel averages.  Returns a list of (label, confidence).
    fn dominant_colors(pixels: &[u8], stride: usize) -> Vec<(String, f32)> {
        if pixels.is_empty() || stride < 3 {
            return Vec::new();
        }
        let total_pixels = pixels.len() / stride;
        if total_pixels == 0 {
            return Vec::new();
        }

        // Accumulate per 8 bins (by pixel index modulo 8).
        let mut bins: [(u64, u64, u64, u64); 8] = [(0, 0, 0, 0); 8];
        for (i, chunk) in pixels.chunks_exact(stride).enumerate() {
            let bin = i % 8;
            bins[bin].0 += chunk[0] as u64;
            bins[bin].1 += chunk[1] as u64;
            bins[bin].2 += chunk[2] as u64;
            bins[bin].3 += 1;
        }

        // Aggregate global averages.
        let mut total_r = 0u64;
        let mut total_g = 0u64;
        let mut total_b = 0u64;
        let mut total_n = 0u64;
        for (r, g, b, n) in &bins {
            total_r += r;
            total_g += g;
            total_b += b;
            total_n += n;
        }
        if total_n == 0 {
            return Vec::new();
        }
        let avg_r = (total_r / total_n) as f32;
        let avg_g = (total_g / total_n) as f32;
        let avg_b = (total_b / total_n) as f32;

        // Map dominant channel to a colour label.
        let max_chan = avg_r.max(avg_g).max(avg_b);
        let threshold = 80.0_f32;

        let mut tags = Vec::new();

        // Red dominant.
        if avg_r == max_chan && avg_r - avg_g > threshold && avg_r - avg_b > threshold {
            tags.push(("red".to_string(), avg_r / 255.0 * 0.8 + 0.2));
        }
        // Green dominant.
        if avg_g == max_chan && avg_g - avg_r > threshold && avg_g - avg_b > threshold {
            tags.push(("green".to_string(), avg_g / 255.0 * 0.8 + 0.2));
        }
        // Blue dominant.
        if avg_b == max_chan && avg_b - avg_r > threshold && avg_b - avg_g > threshold {
            tags.push(("blue".to_string(), avg_b / 255.0 * 0.8 + 0.2));
        }
        // Grey / achromatic.
        let spread = (avg_r - avg_g)
            .abs()
            .max((avg_g - avg_b).abs())
            .max((avg_r - avg_b).abs());
        if spread < 30.0 {
            let grey_confidence = 0.6 + (1.0 - spread / 30.0) * 0.3;
            tags.push(("grey".to_string(), grey_confidence));
        }

        tags
    }

    // -----------------------------------------------------------------------
    // Audio helpers
    // -----------------------------------------------------------------------

    /// Root-mean-square energy of an audio sample slice.
    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    /// Zero-crossing rate (normalised to `0..=1`).
    fn zero_crossing_rate(samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        let crossings = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();
        crossings as f32 / (samples.len() - 1) as f32
    }
}

impl TaggingBackend for RuleBasedTagger {
    fn name(&self) -> &str {
        "rule-based"
    }

    /// Analyse pixel data and return colour, brightness, and aspect-ratio tags.
    fn tag_image(&self, pixels: &[u8], width: u32, height: u32) -> Vec<AutoTag> {
        let mut tags = Vec::new();

        if pixels.is_empty() || width == 0 || height == 0 {
            return tags;
        }

        let stride = Self::detect_stride(pixels, width, height);

        // --- Brightness ---
        let lum = Self::mean_luminance(pixels, stride);
        let (brightness_label, brightness_conf) = if lum > 0.65 {
            ("bright", 0.5 + lum * 0.4)
        } else if lum < 0.25 {
            ("dark", 0.5 + (1.0 - lum) * 0.4)
        } else {
            ("night", 0.35)
        };
        tags.push(AutoTag {
            tag: brightness_label.to_string(),
            confidence: brightness_conf.min(1.0),
            category: TagCategory::Scene,
            source: Self::source(),
        });

        // --- Dominant colours ---
        for (label, conf) in Self::dominant_colors(pixels, stride) {
            tags.push(AutoTag {
                tag: label,
                confidence: conf.min(1.0),
                category: TagCategory::Color,
                source: Self::source(),
            });
        }

        // --- Aspect ratio ---
        let aspect = width as f32 / height as f32;
        let (ar_label, ar_conf) = if (aspect - 1.0).abs() < 0.05 {
            ("square", 0.95)
        } else if aspect < 0.75 {
            ("portrait", 0.9)
        } else if aspect > 2.5 {
            ("ultrawide", 0.9)
        } else {
            ("landscape", 0.85)
        };
        tags.push(AutoTag {
            tag: ar_label.to_string(),
            confidence: ar_conf,
            category: TagCategory::Scene,
            source: Self::source(),
        });

        // --- Resolution quality ---
        let mega_pixels = (width as f64 * height as f64) / 1_000_000.0;
        if mega_pixels >= 8.0 {
            tags.push(AutoTag {
                tag: "high-resolution".to_string(),
                confidence: 0.9,
                category: TagCategory::Quality,
                source: Self::source(),
            });
        } else if mega_pixels < 0.3 {
            tags.push(AutoTag {
                tag: "low-resolution".to_string(),
                confidence: 0.8,
                category: TagCategory::Quality,
                source: Self::source(),
            });
        }

        tags
    }

    /// Analyse PCM samples and return energy, speech/music, and duration tags.
    fn tag_audio(&self, samples: &[f32], sample_rate: u32) -> Vec<AutoTag> {
        let mut tags = Vec::new();

        if samples.is_empty() || sample_rate == 0 {
            return tags;
        }

        // --- RMS energy → loud/quiet ---
        let rms = Self::rms(samples);
        let (energy_label, energy_conf) = if rms > 0.4 {
            ("loud", 0.5 + rms * 0.5)
        } else if rms < 0.05 {
            ("quiet", 0.9)
        } else {
            ("moderate-volume", 0.6)
        };
        tags.push(AutoTag {
            tag: energy_label.to_string(),
            confidence: energy_conf.min(1.0),
            category: TagCategory::Emotion,
            source: Self::source(),
        });

        // --- ZCR → speech / music / silence ---
        let zcr = Self::zero_crossing_rate(samples);
        let (content_label, content_conf) = if rms < 0.01 {
            ("silence", 0.95)
        } else if zcr > 0.15 {
            // High zero-crossing rate typical of speech / noisy signals.
            ("speech", 0.55 + zcr * 0.3)
        } else {
            ("music", 0.6)
        };
        tags.push(AutoTag {
            tag: content_label.to_string(),
            confidence: content_conf.min(1.0),
            category: TagCategory::Genre,
            source: Self::source(),
        });

        // --- Duration ---
        let duration_secs = samples.len() as f64 / sample_rate as f64;
        let dur_label = if duration_secs < 30.0 {
            "short-clip"
        } else if duration_secs < 600.0 {
            "medium-length"
        } else {
            "long-form"
        };
        tags.push(AutoTag {
            tag: dur_label.to_string(),
            confidence: 0.95,
            category: TagCategory::Technical,
            source: Self::source(),
        });

        tags
    }

    /// Derive tags from file metadata (extension, resolution, codec, bitrate).
    fn tag_metadata(&self, meta: &ContentMetadata) -> Vec<AutoTag> {
        let mut tags = Vec::new();

        // --- Format from extension ---
        let ext = meta.extension.to_lowercase();
        if !ext.is_empty() {
            tags.push(AutoTag {
                tag: format!("format:{}", ext),
                confidence: 1.0,
                category: TagCategory::Technical,
                source: Self::source(),
            });
        }

        // --- Resolution classification ---
        if let (Some(w), Some(h)) = (meta.width, meta.height) {
            let short_side = w.min(h);
            let res_label = if short_side >= 4320 {
                "8K"
            } else if short_side >= 2160 {
                "4K"
            } else if short_side >= 720 {
                "HD"
            } else {
                "SD"
            };
            tags.push(AutoTag {
                tag: res_label.to_string(),
                confidence: 0.98,
                category: TagCategory::Technical,
                source: Self::source(),
            });
        }

        // --- Codec tags ---
        if let Some(ref vc) = meta.video_codec {
            if !vc.is_empty() {
                tags.push(AutoTag {
                    tag: format!("codec:{}", vc.to_lowercase()),
                    confidence: 0.95,
                    category: TagCategory::Technical,
                    source: Self::source(),
                });
            }
        }
        if let Some(ref ac) = meta.audio_codec {
            if !ac.is_empty() {
                tags.push(AutoTag {
                    tag: format!("audio-codec:{}", ac.to_lowercase()),
                    confidence: 0.95,
                    category: TagCategory::Technical,
                    source: Self::source(),
                });
            }
        }

        // --- Bitrate → quality estimate ---
        if let Some(kbps) = meta.bitrate_kbps {
            let quality_label = if kbps >= 8000 {
                "high-quality"
            } else if kbps >= 2000 {
                "standard-quality"
            } else if kbps >= 500 {
                "compressed"
            } else {
                "low-bitrate"
            };
            let conf = match quality_label {
                "high-quality" => 0.85,
                "standard-quality" => 0.75,
                "compressed" => 0.70,
                _ => 0.65,
            };
            tags.push(AutoTag {
                tag: quality_label.to_string(),
                confidence: conf,
                category: TagCategory::Quality,
                source: Self::source(),
            });
        }

        // --- Duration classification ---
        if let Some(secs) = meta.duration_secs {
            let dur_label = if secs < 30.0 {
                "short-clip"
            } else if secs < 600.0 {
                "medium-length"
            } else {
                "long-form"
            };
            tags.push(AutoTag {
                tag: dur_label.to_string(),
                confidence: 0.95,
                category: TagCategory::Technical,
                source: Self::source(),
            });
        }

        tags
    }
}

// ---------------------------------------------------------------------------
// AutoTagger
// ---------------------------------------------------------------------------

/// Multi-backend tagger that aggregates, filters, and deduplicates tags.
pub struct AutoTagger {
    backends: Vec<Box<dyn TaggingBackend>>,
    /// Minimum confidence threshold; tags below this are discarded.
    pub min_confidence: f32,
    /// Maximum tags returned per [`TagCategory`] variant.
    pub max_tags_per_category: usize,
    /// When `true`, tags whose stems are identical (after stripping common
    /// suffixes) are deduplicated, keeping the highest-confidence entry.
    pub dedup_similar: bool,
}

impl Default for AutoTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTagger {
    /// Create an `AutoTagger` with sensible defaults (no backends yet).
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            min_confidence: 0.3,
            max_tags_per_category: 5,
            dedup_similar: true,
        }
    }

    /// Convenience builder that adds the built-in [`RuleBasedTagger`].
    #[must_use]
    pub fn with_rule_based(mut self) -> Self {
        self.backends.push(Box::new(RuleBasedTagger));
        self
    }

    /// Add a custom backend.
    pub fn add_backend(&mut self, backend: Box<dyn TaggingBackend>) {
        self.backends.push(backend);
    }

    /// Run all backends, then apply confidence filtering, deduplication, and
    /// per-category limits.  Returns tags sorted by confidence descending.
    #[must_use]
    pub fn tag_content(
        &self,
        metadata: &ContentMetadata,
        image: Option<(&[u8], u32, u32)>,
        audio: Option<(&[f32], u32)>,
    ) -> Vec<AutoTag> {
        let mut all_tags: Vec<AutoTag> = Vec::new();

        for backend in &self.backends {
            // Metadata tags.
            all_tags.extend(backend.tag_metadata(metadata));

            // Image tags.
            if let Some((pixels, w, h)) = image {
                all_tags.extend(backend.tag_image(pixels, w, h));
            }

            // Audio tags.
            if let Some((samples, sr)) = audio {
                all_tags.extend(backend.tag_audio(samples, sr));
            }
        }

        // Filter by minimum confidence.
        all_tags.retain(|t| t.confidence >= self.min_confidence);

        // Dedup by tag label (keep highest confidence).
        if self.dedup_similar {
            all_tags = Self::dedup_by_label(all_tags);
        }

        // Sort by confidence descending before limiting per category.
        all_tags.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Per-category limit.
        all_tags = Self::limit_per_category(all_tags, self.max_tags_per_category);

        // Final sort.
        all_tags.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_tags
    }

    /// Deduplicate tags with the same label, keeping the highest-confidence entry.
    fn dedup_by_label(mut tags: Vec<AutoTag>) -> Vec<AutoTag> {
        // Sort by label, then by descending confidence.
        tags.sort_by(|a, b| {
            a.tag.cmp(&b.tag).then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        tags.dedup_by(|a, b| a.tag == b.tag);
        tags
    }

    /// Keep only the top-N tags per category (sorted by confidence desc).
    fn limit_per_category(tags: Vec<AutoTag>, max_per: usize) -> Vec<AutoTag> {
        use std::collections::HashMap;
        let mut buckets: HashMap<String, Vec<AutoTag>> = HashMap::new();
        for tag in tags {
            let key = format!("{:?}", tag.category);
            buckets.entry(key).or_default().push(tag);
        }
        let mut result = Vec::new();
        for (_, mut bucket) in buckets {
            // Already sorted desc by confidence from caller.
            bucket.truncate(max_per);
            result.extend(bucket);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_meta(ext: &str) -> ContentMetadata {
        ContentMetadata {
            filename: format!("test.{ext}"),
            extension: ext.to_string(),
            duration_secs: None,
            width: None,
            height: None,
            video_codec: None,
            audio_codec: None,
            bitrate_kbps: None,
            file_size_bytes: 1024,
        }
    }

    // -----------------------------------------------------------------------
    // RuleBasedTagger — metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_format_tag() {
        let tagger = RuleBasedTagger;
        let meta = sample_meta("mp4");
        let tags = tagger.tag_metadata(&meta);
        assert!(
            tags.iter().any(|t| t.tag == "format:mp4"),
            "expected format:mp4 tag, got: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_metadata_resolution_4k() {
        let tagger = RuleBasedTagger;
        let mut meta = sample_meta("mkv");
        meta.width = Some(3840);
        meta.height = Some(2160);
        let tags = tagger.tag_metadata(&meta);
        assert!(
            tags.iter().any(|t| t.tag == "4K"),
            "expected 4K tag, got: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_metadata_resolution_hd() {
        let tagger = RuleBasedTagger;
        let mut meta = sample_meta("mp4");
        meta.width = Some(1920);
        meta.height = Some(1080);
        let tags = tagger.tag_metadata(&meta);
        assert!(tags.iter().any(|t| t.tag == "HD"));
    }

    #[test]
    fn test_metadata_codec_tag() {
        let tagger = RuleBasedTagger;
        let mut meta = sample_meta("mkv");
        meta.video_codec = Some("AV1".to_string());
        meta.audio_codec = Some("Opus".to_string());
        let tags = tagger.tag_metadata(&meta);
        assert!(tags.iter().any(|t| t.tag == "codec:av1"));
        assert!(tags.iter().any(|t| t.tag == "audio-codec:opus"));
    }

    #[test]
    fn test_metadata_bitrate_quality() {
        let tagger = RuleBasedTagger;
        let mut meta = sample_meta("mp4");
        meta.bitrate_kbps = Some(10000);
        let tags = tagger.tag_metadata(&meta);
        assert!(tags.iter().any(|t| t.tag == "high-quality"));
    }

    #[test]
    fn test_metadata_duration_short_clip() {
        let tagger = RuleBasedTagger;
        let mut meta = sample_meta("mp4");
        meta.duration_secs = Some(15.0);
        let tags = tagger.tag_metadata(&meta);
        assert!(tags.iter().any(|t| t.tag == "short-clip"));
    }

    // -----------------------------------------------------------------------
    // RuleBasedTagger — image
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_brightness_bright() {
        let tagger = RuleBasedTagger;
        // All-white pixel.
        let pixels = vec![255u8, 255, 255, 255];
        let tags = tagger.tag_image(&pixels, 1, 1);
        assert!(
            tags.iter().any(|t| t.tag == "bright"),
            "tags: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image_brightness_dark() {
        let tagger = RuleBasedTagger;
        // All-black pixel.
        let pixels = vec![0u8, 0, 0, 255];
        let tags = tagger.tag_image(&pixels, 1, 1);
        assert!(
            tags.iter().any(|t| t.tag == "dark"),
            "tags: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image_aspect_portrait() {
        let tagger = RuleBasedTagger;
        // Tall portrait: 10×100.  Create 10*100*3 = 3000 bytes (RGB, mid-grey).
        let pixels = vec![128u8; 10 * 100 * 3];
        let tags = tagger.tag_image(&pixels, 10, 100);
        assert!(
            tags.iter().any(|t| t.tag == "portrait"),
            "tags: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image_aspect_landscape() {
        let tagger = RuleBasedTagger;
        let pixels = vec![128u8; 200 * 100 * 3];
        let tags = tagger.tag_image(&pixels, 200, 100);
        assert!(tags.iter().any(|t| t.tag == "landscape"));
    }

    #[test]
    fn test_image_dominant_color_blue() {
        let tagger = RuleBasedTagger;
        // Pure blue pixel (R=0, G=0, B=255).
        let pixels = vec![0u8, 0, 255, 255];
        let tags = tagger.tag_image(&pixels, 1, 1);
        assert!(
            tags.iter().any(|t| t.tag == "blue"),
            "tags: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // RuleBasedTagger — audio
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_silence() {
        let tagger = RuleBasedTagger;
        let samples = vec![0.0f32; 44100];
        let tags = tagger.tag_audio(&samples, 44100);
        assert!(
            tags.iter().any(|t| t.tag == "silence" || t.tag == "quiet"),
            "tags: {:?}",
            tags.iter().map(|t| &t.tag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_audio_loud() {
        let tagger = RuleBasedTagger;
        let samples = vec![0.9f32; 44100];
        let tags = tagger.tag_audio(&samples, 44100);
        assert!(tags.iter().any(|t| t.tag == "loud"));
    }

    #[test]
    fn test_audio_duration_short_clip() {
        let tagger = RuleBasedTagger;
        let samples = vec![0.1f32; 44100]; // 1 second
        let tags = tagger.tag_audio(&samples, 44100);
        assert!(tags.iter().any(|t| t.tag == "short-clip"));
    }

    // -----------------------------------------------------------------------
    // AutoTagger pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_tagger_confidence_filter() {
        let mut tagger = AutoTagger::new().with_rule_based();
        tagger.min_confidence = 0.99; // very high — should filter most tags
        let mut meta = sample_meta("mp4");
        meta.width = Some(1920);
        meta.height = Some(1080);
        let tags = tagger.tag_content(&meta, None, None);
        // All returned tags must meet threshold.
        for tag in &tags {
            assert!(
                tag.confidence >= 0.99,
                "tag '{}' confidence {} below threshold",
                tag.tag,
                tag.confidence
            );
        }
    }

    #[test]
    fn test_auto_tagger_dedup() {
        // Use two backends that both emit the same tag.
        struct DupBackend;
        impl TaggingBackend for DupBackend {
            fn name(&self) -> &str {
                "dup"
            }
            fn tag_image(&self, _: &[u8], _: u32, _: u32) -> Vec<AutoTag> {
                Vec::new()
            }
            fn tag_audio(&self, _: &[f32], _: u32) -> Vec<AutoTag> {
                Vec::new()
            }
            fn tag_metadata(&self, _: &ContentMetadata) -> Vec<AutoTag> {
                vec![
                    AutoTag {
                        tag: "duplicate".into(),
                        confidence: 0.7,
                        category: TagCategory::Technical,
                        source: "dup".into(),
                    },
                    AutoTag {
                        tag: "duplicate".into(),
                        confidence: 0.9,
                        category: TagCategory::Technical,
                        source: "dup".into(),
                    },
                ]
            }
        }

        let mut tagger = AutoTagger::new();
        tagger.add_backend(Box::new(DupBackend));
        let tags = tagger.tag_content(&sample_meta("mp4"), None, None);
        let dup_count = tags.iter().filter(|t| t.tag == "duplicate").count();
        assert_eq!(dup_count, 1, "expected dedup to remove duplicate tag");
        assert!(
            (tags
                .iter()
                .find(|t| t.tag == "duplicate")
                .expect("tag")
                .confidence
                - 0.9)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_auto_tagger_max_per_category() {
        struct SpamBackend;
        impl TaggingBackend for SpamBackend {
            fn name(&self) -> &str {
                "spam"
            }
            fn tag_image(&self, _: &[u8], _: u32, _: u32) -> Vec<AutoTag> {
                Vec::new()
            }
            fn tag_audio(&self, _: &[f32], _: u32) -> Vec<AutoTag> {
                Vec::new()
            }
            fn tag_metadata(&self, _: &ContentMetadata) -> Vec<AutoTag> {
                (0..10)
                    .map(|i| AutoTag {
                        tag: format!("tech-tag-{i}"),
                        confidence: 0.5,
                        category: TagCategory::Technical,
                        source: "spam".into(),
                    })
                    .collect()
            }
        }

        let mut tagger = AutoTagger::new();
        tagger.max_tags_per_category = 3;
        tagger.add_backend(Box::new(SpamBackend));
        let tags = tagger.tag_content(&sample_meta("mp4"), None, None);
        let technical_count = tags
            .iter()
            .filter(|t| t.category == TagCategory::Technical)
            .count();
        assert!(
            technical_count <= 3,
            "expected at most 3 technical tags, got {technical_count}"
        );
    }

    #[test]
    fn test_auto_tag_serialization() {
        let tag = AutoTag {
            tag: "HD".into(),
            confidence: 0.98,
            category: TagCategory::Technical,
            source: "rule-based".into(),
        };
        let json = serde_json::to_string(&tag).expect("serialize");
        let back: AutoTag = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.tag, "HD");
        assert!((back.confidence - 0.98).abs() < 0.001);
    }
}
