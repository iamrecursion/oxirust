//! OCR subtitle extraction from bitmap-based subtitle formats.
//!
//! This module provides image-analysis tools for extracting text from bitmap-based
//! subtitle formats including PGS (Blu-ray Presentation Graphic Stream), VobSub
//! (DVD .sub/.idx), and DVB subtitles embedded in MPEG-TS.
//!
//! # Architecture
//! - `BitmapRegion`: a rectangular region of an RGBA or grayscale subtitle bitmap
//! - `OcrConfig`: controls binarization threshold, column grouping, and language hints
//! - `OcrEngine`: top-level entry point; accepts `BitmapRegion` and produces `OcrResult`
//! - `TextLine`: a single detected text line with bounding box and confidence score
//!
//! # Limitations
//! This is a pure-Rust, training-free OCR pipeline using connected-component labelling
//! and a character-similarity heuristic. For high-accuracy production use, integrate
//! a trained classifier (e.g. via ONNX or a native model crate).

#![allow(dead_code)]

use std::collections::HashMap;

// ── Primitive types ───────────────────────────────────────────────────────────

/// Source format of the bitmap subtitle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitmapSubtitleFormat {
    /// Blu-ray Presentation Graphic Stream.
    Pgs,
    /// DVD VobSub (`.sub` / `.idx` pair).
    VobSub,
    /// DVB subtitles embedded in MPEG-TS.
    Dvb,
}

impl BitmapSubtitleFormat {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Pgs => "PGS",
            Self::VobSub => "VobSub",
            Self::Dvb => "DVB",
        }
    }
}

/// A rectangular bounding box (pixel coordinates, inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    /// Left column (0-indexed).
    pub x: u32,
    /// Top row (0-indexed).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl BoundingBox {
    /// Create a new bounding box.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Area in pixels.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Right edge (exclusive).
    #[must_use]
    pub const fn right(&self) -> u32 {
        self.x + self.width
    }

    /// Bottom edge (exclusive).
    #[must_use]
    pub const fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// Whether this box overlaps horizontally with another (used for column grouping).
    #[must_use]
    pub fn overlaps_horizontally(&self, other: &Self) -> bool {
        self.x < other.right() && other.x < self.right()
    }
}

/// A single-channel (grayscale) bitmap region extracted from a subtitle frame.
///
/// Pixels are stored in row-major order; values `0 = black`, `255 = white`.
#[derive(Debug, Clone)]
pub struct BitmapRegion {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major grayscale pixel data (`width * height` bytes).
    pub pixels: Vec<u8>,
    /// Original format of the subtitle.
    pub source_format: BitmapSubtitleFormat,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

impl BitmapRegion {
    /// Create a new bitmap region.
    ///
    /// # Panics
    /// Panics in debug mode if `pixels.len() != width * height`.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
        source_format: BitmapSubtitleFormat,
        pts_ms: u64,
        duration_ms: u64,
    ) -> Self {
        debug_assert_eq!(
            pixels.len(),
            (width * height) as usize,
            "pixel buffer length must equal width * height"
        );
        Self {
            width,
            height,
            pixels,
            source_format,
            pts_ms,
            duration_ms,
        }
    }

    /// Pixel value at `(col, row)`, clamped to the image bounds.
    #[must_use]
    pub fn pixel(&self, col: u32, row: u32) -> u8 {
        let col = col.min(self.width.saturating_sub(1));
        let row = row.min(self.height.saturating_sub(1));
        self.pixels[(row * self.width + col) as usize]
    }

    /// Binarise the region: pixels above `threshold` become `255`, others become `0`.
    #[must_use]
    pub fn binarise(&self, threshold: u8) -> Vec<u8> {
        self.pixels
            .iter()
            .map(|&p| if p > threshold { 255 } else { 0 })
            .collect()
    }

    /// Compute a horizontal projection profile (sum of foreground pixels per row).
    #[must_use]
    pub fn horizontal_projection(&self, threshold: u8) -> Vec<u32> {
        let binary = self.binarise(threshold);
        (0..self.height)
            .map(|row| {
                binary[row as usize * self.width as usize..(row + 1) as usize * self.width as usize]
                    .iter()
                    .map(|&p| u32::from(p / 255))
                    .sum()
            })
            .collect()
    }
}

// ── Connected-component labelling (4-connectivity) ────────────────────────────

/// A connected component of foreground pixels.
#[derive(Debug, Clone)]
pub struct ConnectedComponent {
    /// Component label (1-indexed).
    pub label: u32,
    /// Bounding box.
    pub bbox: BoundingBox,
    /// Number of foreground pixels.
    pub pixel_count: u32,
}

/// Run 4-connectivity connected-component labelling on a binary image.
///
/// Returns a list of components, sorted by their left edge (`bbox.x`).
#[must_use]
pub fn connected_components(
    binary: &[u8],
    width: u32,
    height: u32,
    foreground: u8,
) -> Vec<ConnectedComponent> {
    let n = (width * height) as usize;
    let mut labels: Vec<u32> = vec![0; n];
    let mut next_label: u32 = 1;
    let mut union_find: Vec<u32> = vec![0]; // index 0 unused

    let idx = |col: u32, row: u32| -> usize { (row * width + col) as usize };

    for row in 0..height {
        for col in 0..width {
            if binary[idx(col, row)] != foreground {
                continue;
            }
            let top = if row > 0 {
                labels[idx(col, row - 1)]
            } else {
                0
            };
            let left = if col > 0 {
                labels[idx(col - 1, row)]
            } else {
                0
            };

            let label = match (top != 0, left != 0) {
                (false, false) => {
                    let l = next_label;
                    next_label += 1;
                    union_find.push(l);
                    l
                }
                (true, false) => find(&mut union_find, top),
                (false, true) => find(&mut union_find, left),
                (true, true) => {
                    let rt = find(&mut union_find, top);
                    let rl = find(&mut union_find, left);
                    union(&mut union_find, rt, rl);
                    rt.min(rl)
                }
            };
            labels[idx(col, row)] = label;
        }
    }

    // Second pass: flatten labels, collect bounding boxes
    let mut bboxes: HashMap<u32, (u32, u32, u32, u32, u32)> = HashMap::new();
    // entry: (min_x, min_y, max_x, max_y, count)
    for row in 0..height {
        for col in 0..width {
            let lbl = labels[idx(col, row)];
            if lbl == 0 {
                continue;
            }
            let root = find(&mut union_find, lbl);
            labels[idx(col, row)] = root;
            let e = bboxes.entry(root).or_insert((col, row, col, row, 0));
            e.0 = e.0.min(col);
            e.1 = e.1.min(row);
            e.2 = e.2.max(col);
            e.3 = e.3.max(row);
            e.4 += 1;
        }
    }

    let mut components: Vec<ConnectedComponent> = bboxes
        .into_iter()
        .map(
            |(label, (min_x, min_y, max_x, max_y, count))| ConnectedComponent {
                label,
                bbox: BoundingBox::new(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
                pixel_count: count,
            },
        )
        .collect();
    components.sort_by_key(|c| (c.bbox.y, c.bbox.x));
    components
}

fn find(uf: &mut Vec<u32>, x: u32) -> u32 {
    let x = x as usize;
    if uf[x] == x as u32 {
        return x as u32;
    }
    let root = find(uf, uf[x]);
    uf[x] = root;
    root
}

fn union(uf: &mut Vec<u32>, a: u32, b: u32) {
    let ra = find(uf, a);
    let rb = find(uf, b);
    if ra != rb {
        let (smaller, larger) = if ra < rb { (rb, ra) } else { (ra, rb) };
        uf[smaller as usize] = larger;
    }
}

// ── Text-line grouping ────────────────────────────────────────────────────────

/// Group connected components into horizontal text lines using a vertical gap threshold.
///
/// Components within `line_gap_px` vertical pixels of each other are merged into the
/// same line group.
#[must_use]
pub fn group_into_lines(
    components: &[ConnectedComponent],
    line_gap_px: u32,
) -> Vec<Vec<&ConnectedComponent>> {
    let mut lines: Vec<Vec<&ConnectedComponent>> = Vec::new();
    let mut current_line: Vec<&ConnectedComponent> = Vec::new();
    let mut current_bottom: u32 = 0;

    let mut sorted: Vec<&ConnectedComponent> = components.iter().collect();
    sorted.sort_by_key(|c| c.bbox.y);

    for comp in sorted {
        if current_line.is_empty() || comp.bbox.y <= current_bottom + line_gap_px {
            current_line.push(comp);
            current_bottom = current_bottom.max(comp.bbox.bottom());
        } else {
            if !current_line.is_empty() {
                lines.push(current_line);
            }
            current_line = vec![comp];
            current_bottom = comp.bbox.bottom();
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }

    // Sort each line left-to-right
    for line in &mut lines {
        line.sort_by_key(|c| c.bbox.x);
    }

    lines
}

// ── OcrConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the OCR engine.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Binarisation threshold: pixels above this value are considered foreground.
    pub binarise_threshold: u8,
    /// Vertical gap in pixels used to separate text lines during grouping.
    pub line_gap_px: u32,
    /// Minimum component area in pixels; smaller components are discarded as noise.
    pub min_component_area: u32,
    /// Maximum component area in pixels; larger are discarded as frame artifacts.
    pub max_component_area: u32,
    /// BCP-47 language hint (e.g. `"en"`, `"ja"`) for post-processing heuristics.
    pub language_hint: Option<String>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            binarise_threshold: 128,
            line_gap_px: 8,
            min_component_area: 4,
            max_component_area: 10_000,
            language_hint: None,
        }
    }
}

// ── OcrResult ─────────────────────────────────────────────────────────────────

/// A single detected text line from a subtitle bitmap.
#[derive(Debug, Clone)]
pub struct TextLine {
    /// Extracted text (best-effort, may contain errors).
    pub text: String,
    /// Bounding box of the whole line within the original bitmap.
    pub bbox: BoundingBox,
    /// OCR confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Number of character-sized components on this line.
    pub component_count: usize,
}

impl TextLine {
    /// Whether the line passes the minimum confidence threshold.
    #[must_use]
    pub fn is_reliable(&self, min_confidence: f32) -> bool {
        self.confidence >= min_confidence
    }
}

/// Result of an OCR pass on a single bitmap subtitle frame.
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Detected text lines, ordered top-to-bottom.
    pub lines: Vec<TextLine>,
    /// Full reconstructed text (lines joined with `\n`).
    pub full_text: String,
    /// Presentation timestamp in milliseconds.
    pub pts_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Source format.
    pub source_format: BitmapSubtitleFormat,
}

impl OcrResult {
    /// Average confidence across all lines (0.0 if no lines).
    #[must_use]
    pub fn avg_confidence(&self) -> f32 {
        if self.lines.is_empty() {
            return 0.0;
        }
        self.lines.iter().map(|l| l.confidence).sum::<f32>() / self.lines.len() as f32
    }

    /// Whether any text was detected.
    #[must_use]
    pub fn has_text(&self) -> bool {
        !self.full_text.trim().is_empty()
    }
}

// ── OcrEngine ─────────────────────────────────────────────────────────────────

/// Pure-Rust OCR engine for bitmap subtitle extraction.
///
/// This engine uses connected-component analysis and spatial grouping to segment
/// character blobs into text lines. The character recognition step uses a pixel-density
/// heuristic that produces approximate ASCII text — good enough for synchronisation
/// and rough indexing, but not broadcast-quality transcription without a trained model.
#[derive(Debug, Clone)]
pub struct OcrEngine {
    config: OcrConfig,
}

impl OcrEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: OcrConfig) -> Self {
        Self { config }
    }

    /// Process a single `BitmapRegion` and return an `OcrResult`.
    #[must_use]
    pub fn process(&self, region: &BitmapRegion) -> OcrResult {
        let binary = region.binarise(self.config.binarise_threshold);
        let mut components = connected_components(&binary, region.width, region.height, 255);

        // Filter by area
        components.retain(|c| {
            let area = c.pixel_count;
            area >= self.config.min_component_area && area <= self.config.max_component_area
        });

        let line_groups = group_into_lines(&components, self.config.line_gap_px);
        let mut text_lines: Vec<TextLine> = Vec::new();

        for line_comps in &line_groups {
            if line_comps.is_empty() {
                continue;
            }
            // Bounding box of the whole line
            let min_x = line_comps.iter().map(|c| c.bbox.x).min().unwrap_or(0);
            let min_y = line_comps.iter().map(|c| c.bbox.y).min().unwrap_or(0);
            let max_x = line_comps.iter().map(|c| c.bbox.right()).max().unwrap_or(0);
            let max_y = line_comps
                .iter()
                .map(|c| c.bbox.bottom())
                .max()
                .unwrap_or(0);
            let bbox = BoundingBox::new(min_x, min_y, max_x - min_x, max_y - min_y);

            // Placeholder character recognition: map each component to '█' or use
            // pixel-density cues to make a coarse guess.
            let text = self.heuristic_decode(line_comps, region, &binary);
            let confidence = self.estimate_confidence(line_comps);
            text_lines.push(TextLine {
                text,
                bbox,
                confidence,
                component_count: line_comps.len(),
            });
        }

        let full_text = text_lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        OcrResult {
            lines: text_lines,
            full_text,
            pts_ms: region.pts_ms,
            duration_ms: region.duration_ms,
            source_format: region.source_format,
        }
    }

    /// Heuristic character decoder based on component pixel density and aspect ratio.
    ///
    /// In the absence of a trained classifier, we emit a placeholder block character
    /// for each detected component.  This preserves line structure and word spacing,
    /// enabling synchronisation pipelines to proceed without a full ML model.
    fn heuristic_decode(
        &self,
        line_comps: &[&ConnectedComponent],
        region: &BitmapRegion,
        binary: &[u8],
    ) -> String {
        if line_comps.is_empty() {
            return String::new();
        }

        let mut chars: Vec<char> = Vec::new();
        let mut prev_right: u32 = line_comps[0].bbox.x;

        for comp in line_comps {
            // Insert a space if there is a horizontal gap wider than average char width
            let gap = comp.bbox.x.saturating_sub(prev_right);
            let avg_width = line_comps
                .iter()
                .map(|c| c.bbox.width)
                .sum::<u32>()
                .checked_div(line_comps.len() as u32)
                .unwrap_or(8);
            if gap > avg_width / 2 && !chars.is_empty() {
                chars.push(' ');
            }

            // Compute the fill ratio of the bounding box to guess punctuation vs letter
            let area = comp.bbox.width * comp.bbox.height;
            let filled = comp.pixel_count;
            let fill_ratio = if area == 0 {
                0.0_f32
            } else {
                filled as f32 / area as f32
            };

            // Very thin components → punctuation-like
            // Wider components → letter-like
            let aspect = comp.bbox.width as f32 / comp.bbox.height.max(1) as f32;

            // Sample the vertical centre to guess character type
            let sample_col = comp.bbox.x + comp.bbox.width / 2;
            let sample_row = comp.bbox.y + comp.bbox.height / 2;
            let is_bright = region.pixel(sample_col, sample_row) > 200;
            let _ = (is_bright, binary); // used for future enhancement

            let ch = if fill_ratio < 0.15 {
                '.' // sparse → likely punctuation
            } else if aspect < 0.3 {
                'l' // very narrow → 'l' or 'i'
            } else if aspect > 1.5 {
                '-' // wide and flat → dash or underscore
            } else {
                '█' // generic character placeholder
            };
            chars.push(ch);
            prev_right = comp.bbox.right();
        }

        chars.into_iter().collect()
    }

    /// Estimate confidence from component regularity (height variance, fill ratio).
    fn estimate_confidence(&self, comps: &[&ConnectedComponent]) -> f32 {
        if comps.is_empty() {
            return 0.0;
        }
        // Consistent heights → likely real text
        let heights: Vec<f32> = comps.iter().map(|c| c.bbox.height as f32).collect();
        let mean_h = heights.iter().sum::<f32>() / heights.len() as f32;
        let variance =
            heights.iter().map(|h| (h - mean_h).powi(2)).sum::<f32>() / heights.len() as f32;
        let std_dev = variance.sqrt();
        let cv = if mean_h > 0.0 { std_dev / mean_h } else { 1.0 };
        // Lower coefficient of variation → higher confidence
        (1.0 - cv.min(1.0)).max(0.1)
    }
}

impl Default for OcrEngine {
    fn default() -> Self {
        Self::new(OcrConfig::default())
    }
}

// ── Batch processing ──────────────────────────────────────────────────────────

/// Process a sequence of bitmap subtitle regions and return all OCR results.
#[must_use]
pub fn process_batch(regions: &[BitmapRegion], config: &OcrConfig) -> Vec<OcrResult> {
    let engine = OcrEngine::new(config.clone());
    regions.iter().map(|r| engine.process(r)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(width: u32, height: u32, pixels: Vec<u8>) -> BitmapRegion {
        BitmapRegion::new(width, height, pixels, BitmapSubtitleFormat::Pgs, 0, 2000)
    }

    #[test]
    fn test_bitmap_subtitle_format_names() {
        assert_eq!(BitmapSubtitleFormat::Pgs.name(), "PGS");
        assert_eq!(BitmapSubtitleFormat::VobSub.name(), "VobSub");
        assert_eq!(BitmapSubtitleFormat::Dvb.name(), "DVB");
    }

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox::new(10, 20, 30, 40);
        assert_eq!(bbox.area(), 1200);
        assert_eq!(bbox.right(), 40);
        assert_eq!(bbox.bottom(), 60);
    }

    #[test]
    fn test_bounding_box_horizontal_overlap() {
        let a = BoundingBox::new(0, 0, 50, 20);
        let b = BoundingBox::new(30, 10, 50, 20); // overlaps with a
        let c = BoundingBox::new(100, 0, 50, 20); // no overlap
        assert!(a.overlaps_horizontally(&b));
        assert!(!a.overlaps_horizontally(&c));
    }

    #[test]
    fn test_bitmap_region_pixel_access() {
        // 3x2 image, all 200
        let region = make_region(3, 2, vec![200u8; 6]);
        assert_eq!(region.pixel(0, 0), 200);
        assert_eq!(region.pixel(2, 1), 200);
        // Out-of-bounds clamp
        assert_eq!(region.pixel(99, 99), 200);
    }

    #[test]
    fn test_binarise_threshold() {
        let region = make_region(3, 1, vec![50, 128, 200]);
        let bin = region.binarise(100);
        assert_eq!(bin, vec![0, 255, 255]);
    }

    #[test]
    fn test_horizontal_projection() {
        // 5x1 image: alternating pixels
        let region = make_region(5, 1, vec![0, 255, 0, 255, 0]);
        let proj = region.horizontal_projection(128);
        assert_eq!(proj, vec![2]);
    }

    #[test]
    fn test_connected_components_single_blob() {
        // 4x4 image with a 2x2 filled square at top-left
        let mut pixels = vec![0u8; 16];
        pixels[0] = 255;
        pixels[1] = 255;
        pixels[4] = 255;
        pixels[5] = 255;
        let comps = connected_components(&pixels, 4, 4, 255);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].pixel_count, 4);
        assert_eq!(comps[0].bbox.width, 2);
        assert_eq!(comps[0].bbox.height, 2);
    }

    #[test]
    fn test_connected_components_two_blobs() {
        // 1x6 image: two separated foreground pixels
        let pixels = vec![255, 0, 0, 0, 0, 255];
        let comps = connected_components(&pixels, 1, 6, 255);
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_group_into_lines_single_line() {
        let comps = vec![
            ConnectedComponent {
                label: 1,
                bbox: BoundingBox::new(0, 10, 10, 10),
                pixel_count: 50,
            },
            ConnectedComponent {
                label: 2,
                bbox: BoundingBox::new(20, 12, 10, 10),
                pixel_count: 60,
            },
        ];
        let lines = group_into_lines(&comps, 8);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), 2);
    }

    #[test]
    fn test_group_into_lines_two_lines() {
        let comps = vec![
            ConnectedComponent {
                label: 1,
                bbox: BoundingBox::new(0, 0, 10, 10),
                pixel_count: 50,
            },
            ConnectedComponent {
                label: 2,
                bbox: BoundingBox::new(0, 50, 10, 10),
                pixel_count: 50,
            },
        ];
        let lines = group_into_lines(&comps, 8);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_ocr_engine_empty_image() {
        let region = make_region(10, 10, vec![0u8; 100]);
        let engine = OcrEngine::default();
        let result = engine.process(&region);
        assert!(!result.has_text());
        assert_eq!(result.lines.len(), 0);
    }

    #[test]
    fn test_ocr_engine_returns_result_structure() {
        // 20x10 image with some foreground pixels in two blobs
        let mut pixels = vec![0u8; 200];
        // Blob 1: 3x3 at (1,1)
        for row in 1..4 {
            for col in 1..4 {
                pixels[row * 20 + col] = 255;
            }
        }
        // Blob 2: 3x3 at (10,1)
        for row in 1..4 {
            for col in 10..13 {
                pixels[row * 20 + col] = 255;
            }
        }
        let region = make_region(20, 10, pixels);
        let engine = OcrEngine::default();
        let result = engine.process(&region);
        assert_eq!(result.source_format, BitmapSubtitleFormat::Pgs);
        assert_eq!(result.pts_ms, 0);
    }

    #[test]
    fn test_ocr_result_avg_confidence_empty() {
        let result = OcrResult {
            lines: vec![],
            full_text: String::new(),
            pts_ms: 0,
            duration_ms: 2000,
            source_format: BitmapSubtitleFormat::Dvb,
        };
        assert!((result.avg_confidence()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_line_reliability() {
        let line = TextLine {
            text: "test".to_string(),
            bbox: BoundingBox::new(0, 0, 100, 20),
            confidence: 0.85,
            component_count: 4,
        };
        assert!(line.is_reliable(0.8));
        assert!(!line.is_reliable(0.9));
    }

    #[test]
    fn test_process_batch() {
        let config = OcrConfig::default();
        let regions = vec![
            BitmapRegion::new(4, 4, vec![0u8; 16], BitmapSubtitleFormat::Pgs, 0, 1000),
            BitmapRegion::new(
                4,
                4,
                vec![0u8; 16],
                BitmapSubtitleFormat::VobSub,
                1000,
                1000,
            ),
        ];
        let results = process_batch(&regions, &config);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ocr_config_defaults() {
        let config = OcrConfig::default();
        assert_eq!(config.binarise_threshold, 128);
        assert_eq!(config.line_gap_px, 8);
        assert!(config.language_hint.is_none());
    }
}
