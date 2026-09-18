#![allow(dead_code)]
//! Ordered switch list (cut sheet) for multi-camera editing.
//!
//! A [`SwitchList`] records the sequence of camera-angle changes that define
//! a multi-camera edit. Each [`SwitchEntry`] describes when and to which
//! camera the program output should switch, along with an optional transition
//! type. The list can be validated, merged, exported as text, and analyzed for
//! coverage statistics.

use std::fmt;

/// Transition type between camera angles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    /// Hard cut – instantaneous switch.
    Cut,
    /// Dissolve / cross-fade over the given number of frames.
    Dissolve {
        /// Duration of the dissolve in frames.
        frames: u32,
    },
    /// Wipe with the given pattern id.
    Wipe {
        /// Wipe pattern identifier.
        pattern: u32,
    },
    /// Dip to colour (usually black) then reveal new source.
    DipToBlack {
        /// Duration of the dip in frames.
        frames: u32,
    },
}

impl fmt::Display for TransitionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cut => write!(f, "CUT"),
            Self::Dissolve { frames } => write!(f, "DISSOLVE({frames}fr)"),
            Self::Wipe { pattern } => write!(f, "WIPE(#{pattern})"),
            Self::DipToBlack { frames } => write!(f, "DIP({frames}fr)"),
        }
    }
}

/// Single entry in a switch list.
#[derive(Debug, Clone)]
pub struct SwitchEntry {
    /// Frame number at which the switch occurs.
    pub frame: u64,
    /// Target camera angle index.
    pub angle: usize,
    /// Transition used for this switch.
    pub transition: TransitionKind,
    /// Optional human-readable note / reason.
    pub note: String,
}

impl SwitchEntry {
    /// Create a simple cut entry.
    #[must_use]
    pub fn cut(frame: u64, angle: usize) -> Self {
        Self {
            frame,
            angle,
            transition: TransitionKind::Cut,
            note: String::new(),
        }
    }

    /// Create a dissolve entry.
    #[must_use]
    pub fn dissolve(frame: u64, angle: usize, dur_frames: u32) -> Self {
        Self {
            frame,
            angle,
            transition: TransitionKind::Dissolve { frames: dur_frames },
            note: String::new(),
        }
    }

    /// Attach a note.
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        self.note = note.to_owned();
        self
    }
}

impl fmt::Display for SwitchEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "F{:06} -> CAM{} [{}]",
            self.frame, self.angle, self.transition
        )
    }
}

/// An ordered list of switch events that defines a multi-camera edit.
#[derive(Debug, Clone)]
pub struct SwitchList {
    /// The entries, kept sorted by frame.
    entries: Vec<SwitchEntry>,
    /// Total number of camera angles referenced.
    angle_count: usize,
}

impl SwitchList {
    /// Create a new empty switch list for the given number of angles.
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            entries: Vec::new(),
            angle_count,
        }
    }

    /// Add an entry. The list is re-sorted by frame.
    pub fn add(&mut self, entry: SwitchEntry) {
        self.entries.push(entry);
        self.entries.sort_by_key(|e| e.frame);
    }

    /// Number of switch events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read-only access to entries.
    #[must_use]
    pub fn entries(&self) -> &[SwitchEntry] {
        &self.entries
    }

    /// Remove the entry at a given position (sorted index).
    pub fn remove(&mut self, idx: usize) -> Option<SwitchEntry> {
        if idx < self.entries.len() {
            Some(self.entries.remove(idx))
        } else {
            None
        }
    }

    /// Find the active angle at a given frame (the most recent switch at or before `frame`).
    #[must_use]
    pub fn active_angle_at(&self, frame: u64) -> Option<usize> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.frame <= frame)
            .map(|e| e.angle)
    }

    /// Validate the list. Returns a list of warning messages.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for duplicate frames
        for i in 1..self.entries.len() {
            if self.entries[i].frame == self.entries[i - 1].frame {
                warnings.push(format!(
                    "Duplicate frame {} at entries {} and {}",
                    self.entries[i].frame,
                    i - 1,
                    i,
                ));
            }
        }

        // Check for out-of-range angles
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.angle >= self.angle_count {
                warnings.push(format!(
                    "Entry {} references angle {} but angle_count is {}",
                    i, entry.angle, self.angle_count,
                ));
            }
        }

        warnings
    }

    /// Return per-angle frame counts (approximate, based on sequential entries).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn coverage_stats(&self, total_frames: u64) -> Vec<(usize, u64, f64)> {
        let mut counts = vec![0u64; self.angle_count];

        for i in 0..self.entries.len() {
            let start = self.entries[i].frame;
            let end = if i + 1 < self.entries.len() {
                self.entries[i + 1].frame
            } else {
                total_frames
            };
            if end > start {
                let angle = self.entries[i].angle;
                if angle < self.angle_count {
                    counts[angle] += end - start;
                }
            }
        }

        counts
            .into_iter()
            .enumerate()
            .map(|(angle, c)| {
                let pct = if total_frames > 0 {
                    c as f64 / total_frames as f64 * 100.0
                } else {
                    0.0
                };
                (angle, c, pct)
            })
            .collect()
    }

    /// Export the list as plain text (one line per entry).
    #[must_use]
    pub fn to_text(&self) -> String {
        self.entries
            .iter()
            .map(|e| {
                if e.note.is_empty() {
                    format!("{e}")
                } else {
                    format!("{e}  // {}", e.note)
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Merge another switch list into this one.
    pub fn merge(&mut self, other: &SwitchList) {
        for entry in &other.entries {
            self.entries.push(entry.clone());
        }
        self.entries.sort_by_key(|e| e.frame);
        if other.angle_count > self.angle_count {
            self.angle_count = other.angle_count;
        }
    }

    /// Return the frame of the last switch, or 0 if empty.
    #[must_use]
    pub fn last_frame(&self) -> u64 {
        self.entries.last().map_or(0, |e| e.frame)
    }

    /// Count how many times each transition kind is used.
    #[must_use]
    pub fn transition_counts(&self) -> (usize, usize, usize, usize) {
        let mut cuts = 0usize;
        let mut dissolves = 0usize;
        let mut wipes = 0usize;
        let mut dips = 0usize;
        for e in &self.entries {
            match e.transition {
                TransitionKind::Cut => cuts += 1,
                TransitionKind::Dissolve { .. } => dissolves += 1,
                TransitionKind::Wipe { .. } => wipes += 1,
                TransitionKind::DipToBlack { .. } => dips += 1,
            }
        }
        (cuts, dissolves, wipes, dips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switch_entry_cut() {
        let e = SwitchEntry::cut(100, 0);
        assert_eq!(e.frame, 100);
        assert_eq!(e.angle, 0);
        assert_eq!(e.transition, TransitionKind::Cut);
    }

    #[test]
    fn test_switch_entry_dissolve() {
        let e = SwitchEntry::dissolve(200, 1, 15);
        assert_eq!(e.transition, TransitionKind::Dissolve { frames: 15 });
    }

    #[test]
    fn test_switch_entry_with_note() {
        let e = SwitchEntry::cut(50, 2).with_note("speaker change");
        assert_eq!(e.note, "speaker change");
    }

    #[test]
    fn test_switch_entry_display() {
        let e = SwitchEntry::cut(42, 3);
        let s = format!("{e}");
        assert!(s.contains("F000042"));
        assert!(s.contains("CAM3"));
        assert!(s.contains("CUT"));
    }

    #[test]
    fn test_switch_list_add_and_sort() {
        let mut list = SwitchList::new(4);
        list.add(SwitchEntry::cut(300, 2));
        list.add(SwitchEntry::cut(100, 0));
        list.add(SwitchEntry::cut(200, 1));
        assert_eq!(list.len(), 3);
        assert_eq!(list.entries()[0].frame, 100);
        assert_eq!(list.entries()[2].frame, 300);
    }

    #[test]
    fn test_switch_list_is_empty() {
        let list = SwitchList::new(2);
        assert!(list.is_empty());
    }

    #[test]
    fn test_active_angle_at() {
        let mut list = SwitchList::new(3);
        list.add(SwitchEntry::cut(0, 0));
        list.add(SwitchEntry::cut(100, 1));
        list.add(SwitchEntry::cut(200, 2));

        assert_eq!(list.active_angle_at(0), Some(0));
        assert_eq!(list.active_angle_at(50), Some(0));
        assert_eq!(list.active_angle_at(100), Some(1));
        assert_eq!(list.active_angle_at(199), Some(1));
        assert_eq!(list.active_angle_at(200), Some(2));
    }

    #[test]
    fn test_validate_no_warnings() {
        let mut list = SwitchList::new(3);
        list.add(SwitchEntry::cut(0, 0));
        list.add(SwitchEntry::cut(100, 1));
        assert!(list.validate().is_empty());
    }

    #[test]
    fn test_validate_out_of_range_angle() {
        let mut list = SwitchList::new(2);
        list.add(SwitchEntry::cut(0, 5));
        let warnings = list.validate();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("angle 5"));
    }

    #[test]
    fn test_validate_duplicate_frames() {
        let mut list = SwitchList::new(3);
        list.add(SwitchEntry::cut(100, 0));
        list.add(SwitchEntry::cut(100, 1));
        let warnings = list.validate();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Duplicate"));
    }

    #[test]
    fn test_coverage_stats() {
        let mut list = SwitchList::new(2);
        list.add(SwitchEntry::cut(0, 0));
        list.add(SwitchEntry::cut(50, 1));
        let stats = list.coverage_stats(100);
        assert_eq!(stats[0].1, 50); // angle 0 used for 50 frames
        assert_eq!(stats[1].1, 50); // angle 1 used for 50 frames
    }

    #[test]
    fn test_to_text() {
        let mut list = SwitchList::new(2);
        list.add(SwitchEntry::cut(0, 0).with_note("start"));
        list.add(SwitchEntry::cut(100, 1));
        let text = list.to_text();
        assert!(text.contains("start"));
        assert!(text.contains("CAM1"));
    }

    #[test]
    fn test_merge() {
        let mut a = SwitchList::new(2);
        a.add(SwitchEntry::cut(0, 0));
        let mut b = SwitchList::new(3);
        b.add(SwitchEntry::cut(50, 2));
        a.merge(&b);
        assert_eq!(a.len(), 2);
        assert_eq!(a.entries()[1].angle, 2);
    }

    #[test]
    fn test_transition_counts() {
        let mut list = SwitchList::new(3);
        list.add(SwitchEntry::cut(0, 0));
        list.add(SwitchEntry::dissolve(100, 1, 10));
        list.add(SwitchEntry::cut(200, 2));
        let (cuts, dissolves, wipes, dips) = list.transition_counts();
        assert_eq!(cuts, 2);
        assert_eq!(dissolves, 1);
        assert_eq!(wipes, 0);
        assert_eq!(dips, 0);
    }

    #[test]
    fn test_last_frame() {
        let mut list = SwitchList::new(2);
        assert_eq!(list.last_frame(), 0);
        list.add(SwitchEntry::cut(500, 1));
        assert_eq!(list.last_frame(), 500);
    }

    #[test]
    fn test_remove_entry() {
        let mut list = SwitchList::new(2);
        list.add(SwitchEntry::cut(0, 0));
        list.add(SwitchEntry::cut(100, 1));
        let removed = list.remove(0);
        assert!(removed.is_some());
        assert_eq!(list.len(), 1);
        assert_eq!(list.entries()[0].frame, 100);
    }

    #[test]
    fn test_transition_kind_display() {
        assert_eq!(format!("{}", TransitionKind::Cut), "CUT");
        assert_eq!(
            format!("{}", TransitionKind::Dissolve { frames: 10 }),
            "DISSOLVE(10fr)"
        );
        assert_eq!(
            format!("{}", TransitionKind::Wipe { pattern: 3 }),
            "WIPE(#3)"
        );
        assert_eq!(
            format!("{}", TransitionKind::DipToBlack { frames: 5 }),
            "DIP(5fr)"
        );
    }
}
