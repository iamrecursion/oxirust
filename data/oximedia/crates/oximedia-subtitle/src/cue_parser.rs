//! Cue-based subtitle parser for WebVTT and SRT cue timestamp formats.
//!
//! Provides a low-level cue entry parser that works at the text line level,
//! useful for streaming or incremental subtitle parsing.

#![allow(dead_code)]

/// A parsed timestamp from a cue line (hours:minutes:seconds.millis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueTimestamp {
    /// Hours component.
    pub hours: u32,
    /// Minutes component.
    pub minutes: u32,
    /// Seconds component.
    pub seconds: u32,
    /// Milliseconds component.
    pub millis: u32,
}

impl CueTimestamp {
    /// Create a new `CueTimestamp`.
    pub fn new(hours: u32, minutes: u32, seconds: u32, millis: u32) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            millis,
        }
    }

    /// Return the total duration in milliseconds.
    pub fn total_ms(&self) -> u64 {
        let h = u64::from(self.hours);
        let m = u64::from(self.minutes);
        let s = u64::from(self.seconds);
        let ms = u64::from(self.millis);
        h * 3_600_000 + m * 60_000 + s * 1_000 + ms
    }

    /// Parse a timestamp string of the form `HH:MM:SS.mmm` or `MM:SS.mmm`.
    ///
    /// Returns `None` if the string cannot be parsed.
    pub fn parse(s: &str) -> Option<Self> {
        // Normalize the separator between seconds and millis (comma or dot)
        let s = s.replace(',', ".");
        let parts: Vec<&str> = s.split(':').collect();
        match parts.len() {
            3 => {
                let hours: u32 = parts[0].parse().ok()?;
                let minutes: u32 = parts[1].parse().ok()?;
                let sec_parts: Vec<&str> = parts[2].split('.').collect();
                let seconds: u32 = sec_parts.first()?.parse().ok()?;
                let millis: u32 = sec_parts.get(1).and_then(|m| parse_millis(m)).unwrap_or(0);
                Some(Self::new(hours, minutes, seconds, millis))
            }
            2 => {
                let minutes: u32 = parts[0].parse().ok()?;
                let sec_parts: Vec<&str> = parts[1].split('.').collect();
                let seconds: u32 = sec_parts.first()?.parse().ok()?;
                let millis: u32 = sec_parts.get(1).and_then(|m| parse_millis(m)).unwrap_or(0);
                Some(Self::new(0, minutes, seconds, millis))
            }
            _ => None,
        }
    }
}

/// Parse a milliseconds string, padding or truncating to 3 digits.
fn parse_millis(s: &str) -> Option<u32> {
    let padded = format!("{:0<3}", &s[..s.len().min(3)]);
    padded.parse().ok()
}

/// A single cue entry with a start/end timestamp and text body.
#[derive(Debug, Clone)]
pub struct CueEntry {
    /// Optional cue identifier.
    pub id: Option<String>,
    /// Start time of the cue.
    pub start: CueTimestamp,
    /// End time of the cue.
    pub end: CueTimestamp,
    /// Text body of the cue (may be multi-line).
    pub text: String,
}

impl CueEntry {
    /// Create a new `CueEntry`.
    pub fn new(id: Option<String>, start: CueTimestamp, end: CueTimestamp, text: String) -> Self {
        Self {
            id,
            start,
            end,
            text,
        }
    }

    /// Returns `true` if the cue has a non-empty text body and a valid time range.
    pub fn is_valid(&self) -> bool {
        !self.text.trim().is_empty() && self.end.total_ms() > self.start.total_ms()
    }

    /// Return the duration of the cue in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end.total_ms().saturating_sub(self.start.total_ms())
    }
}

/// Incremental cue parser that processes one line at a time.
#[derive(Debug, Default)]
pub struct CueParser {
    pending_id: Option<String>,
    pending_start: Option<CueTimestamp>,
    pending_end: Option<CueTimestamp>,
    pending_text: Vec<String>,
    entries: Vec<CueEntry>,
}

impl CueParser {
    /// Create a new `CueParser`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one line of input to the parser.
    ///
    /// Returns `true` if a complete cue was finalized during this call.
    pub fn parse_line(&mut self, line: &str) -> bool {
        let trimmed = line.trim();

        // Empty line signals end of a cue block
        if trimmed.is_empty() {
            return self.finalize_entry();
        }

        // Check for arrow separator (timing line)
        if trimmed.contains("-->") {
            let parts: Vec<&str> = trimmed.splitn(2, "-->").collect();
            if parts.len() == 2 {
                let start_str = parts[0].trim();
                // Strip optional cue settings after the end timestamp
                let end_str = parts[1].split_whitespace().next().unwrap_or("");
                self.pending_start = CueTimestamp::parse(start_str);
                self.pending_end = CueTimestamp::parse(end_str);
            }
            return false;
        }

        // If we have timing but no text yet — check if this is an ID line
        if self.pending_start.is_none() && self.pending_end.is_none() {
            self.pending_id = Some(trimmed.to_string());
        } else {
            self.pending_text.push(trimmed.to_string());
        }
        false
    }

    /// Finalize a pending cue entry.  Returns `true` if a cue was added.
    fn finalize_entry(&mut self) -> bool {
        if let (Some(start), Some(end)) = (self.pending_start.take(), self.pending_end.take()) {
            let text = self.pending_text.join("\n");
            let id = self.pending_id.take();
            self.pending_text.clear();
            let entry = CueEntry::new(id, start, end, text);
            self.entries.push(entry);
            return true;
        }
        // Reset incomplete state
        self.pending_id = None;
        self.pending_text.clear();
        false
    }

    /// Flush any remaining pending entry and return the parsed document.
    pub fn finish(mut self) -> CueDocument {
        self.finalize_entry();
        CueDocument {
            entries: self.entries,
        }
    }
}

/// A collection of parsed cue entries.
#[derive(Debug, Default)]
pub struct CueDocument {
    /// All parsed cue entries in order.
    pub entries: Vec<CueEntry>,
}

impl CueDocument {
    /// Return the number of cue entries.
    pub fn cue_count(&self) -> usize {
        self.entries.len()
    }

    /// Return the total span in milliseconds (from first start to last end).
    pub fn duration_ms(&self) -> u64 {
        let start = self
            .entries
            .first()
            .map(|e| e.start.total_ms())
            .unwrap_or(0);
        let end = self.entries.last().map(|e| e.end.total_ms()).unwrap_or(0);
        end.saturating_sub(start)
    }

    /// Return all valid entries.
    pub fn valid_entries(&self) -> Vec<&CueEntry> {
        self.entries.iter().filter(|e| e.is_valid()).collect()
    }

    /// Find entries active at a given timestamp (ms).
    pub fn active_at(&self, ts_ms: u64) -> Vec<&CueEntry> {
        self.entries
            .iter()
            .filter(|e| e.start.total_ms() <= ts_ms && e.end.total_ms() > ts_ms)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cue_timestamp_total_ms_simple() {
        let ts = CueTimestamp::new(0, 0, 1, 500);
        assert_eq!(ts.total_ms(), 1500);
    }

    #[test]
    fn test_cue_timestamp_total_ms_hours() {
        let ts = CueTimestamp::new(1, 0, 0, 0);
        assert_eq!(ts.total_ms(), 3_600_000);
    }

    #[test]
    fn test_cue_timestamp_parse_three_part() {
        let ts = CueTimestamp::parse("00:01:02.500").expect("should succeed in test");
        assert_eq!(ts.hours, 0);
        assert_eq!(ts.minutes, 1);
        assert_eq!(ts.seconds, 2);
        assert_eq!(ts.millis, 500);
    }

    #[test]
    fn test_cue_timestamp_parse_two_part() {
        let ts = CueTimestamp::parse("01:30.250").expect("should succeed in test");
        assert_eq!(ts.hours, 0);
        assert_eq!(ts.minutes, 1);
        assert_eq!(ts.seconds, 30);
        assert_eq!(ts.millis, 250);
    }

    #[test]
    fn test_cue_timestamp_parse_comma_separator() {
        let ts = CueTimestamp::parse("00:00:05,100").expect("should succeed in test");
        assert_eq!(ts.millis, 100);
    }

    #[test]
    fn test_cue_timestamp_parse_invalid() {
        assert!(CueTimestamp::parse("not-a-time").is_none());
    }

    #[test]
    fn test_cue_entry_is_valid_ok() {
        let start = CueTimestamp::new(0, 0, 0, 0);
        let end = CueTimestamp::new(0, 0, 2, 0);
        let entry = CueEntry::new(None, start, end, "Hello".to_string());
        assert!(entry.is_valid());
    }

    #[test]
    fn test_cue_entry_is_valid_empty_text() {
        let start = CueTimestamp::new(0, 0, 0, 0);
        let end = CueTimestamp::new(0, 0, 2, 0);
        let entry = CueEntry::new(None, start, end, "   ".to_string());
        assert!(!entry.is_valid());
    }

    #[test]
    fn test_cue_entry_is_valid_inverted_times() {
        let start = CueTimestamp::new(0, 0, 5, 0);
        let end = CueTimestamp::new(0, 0, 3, 0);
        let entry = CueEntry::new(None, start, end, "Hello".to_string());
        assert!(!entry.is_valid());
    }

    #[test]
    fn test_cue_entry_duration_ms() {
        let start = CueTimestamp::new(0, 0, 1, 0);
        let end = CueTimestamp::new(0, 0, 4, 500);
        let entry = CueEntry::new(None, start, end, "Hi".to_string());
        assert_eq!(entry.duration_ms(), 3500);
    }

    #[test]
    fn test_cue_parser_parse_webvtt_block() {
        let lines = vec!["00:00:01.000 --> 00:00:04.000", "Hello world", ""];
        let mut parser = CueParser::new();
        for line in lines {
            parser.parse_line(line);
        }
        let doc = parser.finish();
        assert_eq!(doc.cue_count(), 1);
        assert_eq!(doc.entries[0].text, "Hello world");
    }

    #[test]
    fn test_cue_parser_multiple_entries() {
        let input =
            "00:00:01.000 --> 00:00:03.000\nFirst\n\n00:00:05.000 --> 00:00:07.000\nSecond\n\n";
        let mut parser = CueParser::new();
        for line in input.lines() {
            parser.parse_line(line);
        }
        let doc = parser.finish();
        assert_eq!(doc.cue_count(), 2);
    }

    #[test]
    fn test_cue_document_duration_ms() {
        let mut parser = CueParser::new();
        let lines = [
            "00:00:00.000 --> 00:00:02.000",
            "A",
            "",
            "00:00:05.000 --> 00:00:10.000",
            "B",
            "",
        ];
        for l in &lines {
            parser.parse_line(l);
        }
        let doc = parser.finish();
        assert_eq!(doc.duration_ms(), 10_000);
    }

    #[test]
    fn test_cue_document_active_at() {
        let mut parser = CueParser::new();
        let lines = ["00:00:01.000 --> 00:00:03.000", "Active", ""];
        for l in &lines {
            parser.parse_line(l);
        }
        let doc = parser.finish();
        assert_eq!(doc.active_at(2_000).len(), 1);
        assert_eq!(doc.active_at(500).len(), 0);
    }

    #[test]
    fn test_cue_document_valid_entries() {
        let start = CueTimestamp::new(0, 0, 0, 0);
        let end = CueTimestamp::new(0, 0, 2, 0);
        let good = CueEntry::new(None, start, end, "OK".to_string());
        let bad = CueEntry::new(None, start, start, "".to_string());
        let doc = CueDocument {
            entries: vec![good, bad],
        };
        assert_eq!(doc.valid_entries().len(), 1);
    }
}
