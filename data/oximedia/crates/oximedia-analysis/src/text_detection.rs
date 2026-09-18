//! On-screen text detection and OCR metadata module.
//!
//! Provides data structures and utilities for working with text regions
//! detected in video frames, building timelines of text appearances,
//! and searching for specific text content.

use serde::{Deserialize, Serialize};

/// A rectangular region within a frame that contains detected text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    /// Horizontal position of the top-left corner (pixels or normalized)
    pub x: f32,
    /// Vertical position of the top-left corner (pixels or normalized)
    pub y: f32,
    /// Region width
    pub width: f32,
    /// Region height
    pub height: f32,
    /// OCR confidence score in [0.0, 1.0]
    pub confidence: f64,
    /// Detected text content
    pub text: String,
}

impl TextRegion {
    /// Create a new `TextRegion`.
    #[must_use]
    pub fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        confidence: f64,
        text: impl Into<String>,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            confidence,
            text: text.into(),
        }
    }

    /// Area of this region in square pixels (or normalized units squared).
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Returns `true` if this region's area exceeds the given threshold.
    #[must_use]
    pub fn is_large(&self, threshold: f32) -> bool {
        self.area() > threshold
    }

    /// Center point of this region as `(cx, cy)`.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

/// All text regions detected in a single video frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFrame {
    /// Frame index (0-based)
    pub frame_id: u64,
    /// Frame timestamp in milliseconds
    pub timestamp_ms: u64,
    /// All text regions found in this frame
    pub regions: Vec<TextRegion>,
}

impl TextFrame {
    /// Create a new `TextFrame`.
    #[must_use]
    pub fn new(frame_id: u64, timestamp_ms: u64) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            regions: Vec::new(),
        }
    }

    /// Create a `TextFrame` with pre-populated regions.
    #[must_use]
    pub fn with_regions(frame_id: u64, timestamp_ms: u64, regions: Vec<TextRegion>) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            regions,
        }
    }

    /// Returns `true` if at least one text region was detected in this frame.
    #[must_use]
    pub fn has_text(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Sum of the areas of all text regions in this frame.
    #[must_use]
    pub fn total_text_area(&self) -> f32 {
        self.regions.iter().map(TextRegion::area).sum()
    }

    /// Concatenate all detected text strings, separated by spaces.
    #[must_use]
    pub fn all_text(&self) -> String {
        self.regions
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A timeline of text frames spanning an entire video.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TextTimeline {
    /// Ordered list of frames (with or without text)
    pub frames: Vec<TextFrame>,
}

impl TextTimeline {
    /// Create an empty `TextTimeline`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a frame to the timeline.
    pub fn add_frame(&mut self, frame: TextFrame) {
        self.frames.push(frame);
    }

    /// Return all frames that contain at least one text region.
    #[must_use]
    pub fn frames_with_text(&self) -> Vec<&TextFrame> {
        self.frames.iter().filter(|f| f.has_text()).collect()
    }

    /// Percentage of frames (0.0–100.0) that contain on-screen text.
    #[must_use]
    pub fn text_coverage_pct(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let with_text = self.frames.iter().filter(|f| f.has_text()).count();
        (with_text as f64 / self.frames.len() as f64) * 100.0
    }

    /// Return all frames whose concatenated text contains `query` (case-insensitive).
    #[must_use]
    pub fn search_text(&self, query: &str) -> Vec<&TextFrame> {
        let q = query.to_lowercase();
        self.frames
            .iter()
            .filter(|f| f.all_text().to_lowercase().contains(&q))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(w: f32, h: f32, text: &str) -> TextRegion {
        TextRegion::new(10.0, 20.0, w, h, 0.95, text)
    }

    fn make_text_frame(frame_id: u64, ts: u64, regions: Vec<TextRegion>) -> TextFrame {
        TextFrame::with_regions(frame_id, ts, regions)
    }

    #[test]
    fn test_region_area() {
        let r = make_region(100.0, 50.0, "Hello");
        assert!((r.area() - 5000.0).abs() < 1e-3);
    }

    #[test]
    fn test_region_is_large_true() {
        let r = make_region(100.0, 100.0, "Big");
        assert!(r.is_large(5000.0));
    }

    #[test]
    fn test_region_is_large_false() {
        let r = make_region(10.0, 10.0, "Tiny");
        assert!(!r.is_large(5000.0));
    }

    #[test]
    fn test_region_center() {
        let r = TextRegion::new(10.0, 20.0, 100.0, 50.0, 0.9, "test");
        let (cx, cy) = r.center();
        assert!((cx - 60.0).abs() < 1e-3);
        assert!((cy - 45.0).abs() < 1e-3);
    }

    #[test]
    fn test_frame_has_text_true() {
        let f = make_text_frame(0, 0, vec![make_region(10.0, 10.0, "Hi")]);
        assert!(f.has_text());
    }

    #[test]
    fn test_frame_has_text_false() {
        let f = TextFrame::new(0, 0);
        assert!(!f.has_text());
    }

    #[test]
    fn test_frame_total_text_area() {
        let regions = vec![make_region(10.0, 10.0, "A"), make_region(20.0, 5.0, "B")];
        let f = make_text_frame(0, 0, regions);
        assert!((f.total_text_area() - 200.0).abs() < 1e-3); // 100 + 100
    }

    #[test]
    fn test_frame_all_text() {
        let regions = vec![
            make_region(10.0, 10.0, "Hello"),
            make_region(10.0, 10.0, "World"),
        ];
        let f = make_text_frame(0, 0, regions);
        assert_eq!(f.all_text(), "Hello World");
    }

    #[test]
    fn test_timeline_frames_with_text() {
        let mut tl = TextTimeline::new();
        tl.add_frame(make_text_frame(0, 0, vec![make_region(10.0, 10.0, "A")]));
        tl.add_frame(TextFrame::new(1, 40));
        tl.add_frame(make_text_frame(2, 80, vec![make_region(10.0, 10.0, "B")]));
        let with_text = tl.frames_with_text();
        assert_eq!(with_text.len(), 2);
    }

    #[test]
    fn test_timeline_text_coverage_pct() {
        let mut tl = TextTimeline::new();
        tl.add_frame(make_text_frame(0, 0, vec![make_region(10.0, 10.0, "X")]));
        tl.add_frame(TextFrame::new(1, 40));
        tl.add_frame(TextFrame::new(2, 80));
        tl.add_frame(make_text_frame(3, 120, vec![make_region(10.0, 10.0, "Y")]));
        let pct = tl.text_coverage_pct();
        assert!((pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_timeline_text_coverage_empty() {
        let tl = TextTimeline::new();
        assert_eq!(tl.text_coverage_pct(), 0.0);
    }

    #[test]
    fn test_timeline_search_text_found() {
        let mut tl = TextTimeline::new();
        tl.add_frame(make_text_frame(
            0,
            0,
            vec![make_region(10.0, 10.0, "Breaking News")],
        ));
        tl.add_frame(make_text_frame(
            1,
            40,
            vec![make_region(10.0, 10.0, "Sports Score")],
        ));
        let results = tl.search_text("breaking");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].frame_id, 0);
    }

    #[test]
    fn test_timeline_search_text_not_found() {
        let mut tl = TextTimeline::new();
        tl.add_frame(make_text_frame(
            0,
            0,
            vec![make_region(10.0, 10.0, "Hello")],
        ));
        let results = tl.search_text("xyz123");
        assert!(results.is_empty());
    }

    #[test]
    fn test_timeline_search_case_insensitive() {
        let mut tl = TextTimeline::new();
        tl.add_frame(make_text_frame(
            0,
            0,
            vec![make_region(10.0, 10.0, "WEATHER UPDATE")],
        ));
        let results = tl.search_text("weather");
        assert_eq!(results.len(), 1);
    }
}
