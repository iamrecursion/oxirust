//! Lazy EDL parsing — only parses event headers until details are accessed.
//!
//! For large EDLs, parsing every event's timecodes, comments, and metadata
//! upfront can be expensive. This module provides a `LazyEdl` that stores
//! raw line ranges per event and only fully parses them on demand.

#![allow(dead_code)]

use crate::error::{EdlError, EdlResult};
use crate::event::EdlEvent;
use crate::parser::EdlParser;
use crate::timecode::EdlFrameRate;
use std::cell::RefCell;

/// A lazy-parsed EDL event header (cheap to compute).
#[derive(Debug, Clone)]
pub struct LazyEventHeader {
    /// Event number parsed from the line.
    pub number: u32,
    /// Reel name.
    pub reel: String,
    /// Byte offset of this event's first line in the source text.
    pub line_start: usize,
    /// Byte offset of the last line belonging to this event (exclusive).
    pub line_end: usize,
}

/// A lazy EDL document — events are only fully parsed when accessed.
#[derive(Debug)]
pub struct LazyEdl {
    /// The raw source text.
    source: String,
    /// Title extracted from header.
    pub title: Option<String>,
    /// Frame rate extracted from FCM line.
    pub frame_rate: EdlFrameRate,
    /// Lazily-parsed event headers.
    headers: Vec<LazyEventHeader>,
    /// Cache of fully parsed events (indexed by header index).
    cache: RefCell<Vec<Option<EdlEvent>>>,
}

impl LazyEdl {
    /// Create a lazy EDL from source text by scanning headers only.
    ///
    /// # Errors
    ///
    /// Returns an error if the header section cannot be parsed.
    pub fn from_str(input: &str) -> EdlResult<Self> {
        let mut title: Option<String> = None;
        let mut frame_rate = EdlFrameRate::Fps2997NDF;
        let mut headers = Vec::new();
        let mut current_header: Option<LazyEventHeader> = None;

        let lines: Vec<&str> = input.lines().collect();

        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            // Parse header lines
            if trimmed.starts_with("TITLE:") {
                title = Some(trimmed.trim_start_matches("TITLE:").trim().to_string());
                continue;
            }

            if trimmed.starts_with("FCM:") {
                let fcm = trimmed.trim_start_matches("FCM:").trim().to_uppercase();
                frame_rate = if fcm.contains("NON") {
                    EdlFrameRate::Fps2997NDF
                } else if fcm.contains("DROP") {
                    EdlFrameRate::Fps2997DF
                } else {
                    EdlFrameRate::Fps2997NDF
                };
                continue;
            }

            // Comment lines belong to the current event
            if trimmed.starts_with('*') {
                if let Some(h) = &mut current_header {
                    h.line_end = idx + 1;
                }
                continue;
            }

            // Try to parse as event line (starts with digits)
            if trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                // Save previous header
                if let Some(h) = current_header.take() {
                    headers.push(h);
                }

                // Extract event number and reel from the line
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let number = parts[0].parse::<u32>().unwrap_or(0);
                    let reel = parts[1].to_string();
                    current_header = Some(LazyEventHeader {
                        number,
                        reel,
                        line_start: idx,
                        line_end: idx + 1,
                    });
                }
            }
        }

        // Save last header
        if let Some(h) = current_header {
            headers.push(h);
        }

        let cache_len = headers.len();
        Ok(Self {
            source: input.to_string(),
            title,
            frame_rate,
            headers,
            cache: RefCell::new(vec![None; cache_len]),
        })
    }

    /// Get the number of events (from headers).
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.headers.len()
    }

    /// Get all event headers (cheap — no full parsing).
    #[must_use]
    pub fn headers(&self) -> &[LazyEventHeader] {
        &self.headers
    }

    /// Get a specific event header by index.
    #[must_use]
    pub fn header(&self, index: usize) -> Option<&LazyEventHeader> {
        self.headers.get(index)
    }

    /// Get all event numbers (cheap).
    #[must_use]
    pub fn event_numbers(&self) -> Vec<u32> {
        self.headers.iter().map(|h| h.number).collect()
    }

    /// Get all reel names (cheap).
    #[must_use]
    pub fn reel_names(&self) -> Vec<&str> {
        self.headers.iter().map(|h| h.reel.as_str()).collect()
    }

    /// Fully parse and return a specific event by index.
    ///
    /// Results are cached, so subsequent calls for the same index are free.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be parsed.
    pub fn get_event(&self, index: usize) -> EdlResult<EdlEvent> {
        // Check cache
        {
            let cache = self.cache.borrow();
            if let Some(Some(event)) = cache.get(index) {
                return Ok(event.clone());
            }
        }

        let header = self
            .headers
            .get(index)
            .ok_or(EdlError::EventNotFound(index as u32))?;

        // Extract the lines for this event
        let lines: Vec<&str> = self.source.lines().collect();
        let event_text: String = lines
            .get(header.line_start..header.line_end)
            .unwrap_or(&[])
            .join("\n");

        // Parse using the standard parser
        let mut parser = EdlParser::new();
        let mini_edl = parser.parse(&event_text)?;

        let event = mini_edl
            .events
            .into_iter()
            .next()
            .ok_or_else(|| EdlError::parse(header.line_start, "Failed to parse event"))?;

        // Cache the result
        {
            let mut cache = self.cache.borrow_mut();
            if let Some(slot) = cache.get_mut(index) {
                *slot = Some(event.clone());
            }
        }

        Ok(event)
    }

    /// Find event indices matching a reel name.
    #[must_use]
    pub fn find_by_reel(&self, reel: &str) -> Vec<usize> {
        self.headers
            .iter()
            .enumerate()
            .filter(|(_, h)| h.reel == reel)
            .map(|(i, _)| i)
            .collect()
    }

    /// Force-parse all events and return them.
    ///
    /// # Errors
    ///
    /// Returns an error if any event cannot be parsed.
    pub fn parse_all(&self) -> EdlResult<Vec<EdlEvent>> {
        let mut events = Vec::with_capacity(self.headers.len());
        for i in 0..self.headers.len() {
            events.push(self.get_event(i)?);
        }
        Ok(events)
    }

    /// Return the count of cached (already parsed) events.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cache.borrow().iter().filter(|e| e.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_EDL: &str = "TITLE: Lazy Test\n\
        FCM: NON-DROP FRAME\n\
        \n\
        001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
        * FROM CLIP NAME: shot001.mov\n\
        \n\
        002  A002     V     C        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n\
        * FROM CLIP NAME: shot002.mov\n\
        \n\
        003  B001     V     C        01:00:10:00 01:00:15:00 01:00:10:00 01:00:15:00\n";

    #[test]
    fn test_lazy_parse_headers() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        assert_eq!(lazy.title, Some("Lazy Test".to_string()));
        assert_eq!(lazy.frame_rate, EdlFrameRate::Fps2997NDF);
        assert_eq!(lazy.event_count(), 3);
    }

    #[test]
    fn test_lazy_event_numbers() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        assert_eq!(lazy.event_numbers(), vec![1, 2, 3]);
    }

    #[test]
    fn test_lazy_reel_names() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        assert_eq!(lazy.reel_names(), vec!["A001", "A002", "B001"]);
    }

    #[test]
    fn test_lazy_headers() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        let h = lazy.header(0).expect("header should exist");
        assert_eq!(h.number, 1);
        assert_eq!(h.reel, "A001");
    }

    #[test]
    fn test_lazy_get_event() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        assert_eq!(lazy.cached_count(), 0);

        let event = lazy.get_event(0).expect("event should parse");
        assert_eq!(event.number, 1);
        assert_eq!(event.reel, "A001");
        assert_eq!(lazy.cached_count(), 1);

        // Second access should hit cache
        let event2 = lazy.get_event(0).expect("cached event should return");
        assert_eq!(event2.number, 1);
        assert_eq!(lazy.cached_count(), 1);
    }

    #[test]
    fn test_lazy_get_event_with_clip_name() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        let event = lazy.get_event(0).expect("event should parse");
        assert_eq!(event.clip_name, Some("shot001.mov".to_string()));
    }

    #[test]
    fn test_lazy_get_all_events() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        let events = lazy.parse_all().expect("parse_all should succeed");
        assert_eq!(events.len(), 3);
        assert_eq!(events[2].reel, "B001");
        assert_eq!(lazy.cached_count(), 3);
    }

    #[test]
    fn test_lazy_find_by_reel() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        let indices = lazy.find_by_reel("A001");
        assert_eq!(indices, vec![0]);

        let indices2 = lazy.find_by_reel("B001");
        assert_eq!(indices2, vec![2]);

        let indices3 = lazy.find_by_reel("MISSING");
        assert!(indices3.is_empty());
    }

    #[test]
    fn test_lazy_event_out_of_bounds() {
        let lazy = LazyEdl::from_str(SAMPLE_EDL).expect("parse should succeed");
        assert!(lazy.get_event(99).is_err());
    }

    #[test]
    fn test_lazy_empty_edl() {
        let lazy =
            LazyEdl::from_str("TITLE: Empty\nFCM: DROP FRAME\n").expect("parse should succeed");
        assert_eq!(lazy.event_count(), 0);
        assert_eq!(lazy.title, Some("Empty".to_string()));
    }

    #[test]
    fn test_lazy_drop_frame() {
        let edl =
            "FCM: DROP FRAME\n001  AX  V  C  01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00\n";
        let lazy = LazyEdl::from_str(edl).expect("parse should succeed");
        assert_eq!(lazy.frame_rate, EdlFrameRate::Fps2997DF);
    }
}
