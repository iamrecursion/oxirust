#![allow(dead_code)]
//! Track colour assignment for timeline organisation.

/// Palette of available track colours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackColor {
    /// Red (high-energy / alert).
    Red,
    /// Orange (warm accent).
    Orange,
    /// Yellow (bright highlight).
    Yellow,
    /// Green (natural / OK).
    Green,
    /// Teal (cool accent).
    Teal,
    /// Blue (default / calm).
    Blue,
    /// Purple (creative).
    Purple,
    /// Pink (soft accent).
    Pink,
    /// White (neutral).
    White,
    /// Grey (muted / inactive).
    Grey,
}

impl TrackColor {
    /// Returns the `(r, g, b)` tuple (0–255) for this colour.
    #[must_use]
    pub fn rgb(&self) -> (u8, u8, u8) {
        match self {
            TrackColor::Red => (220, 53, 69),
            TrackColor::Orange => (253, 126, 20),
            TrackColor::Yellow => (255, 193, 7),
            TrackColor::Green => (25, 135, 84),
            TrackColor::Teal => (32, 201, 151),
            TrackColor::Blue => (13, 110, 253),
            TrackColor::Purple => (111, 66, 193),
            TrackColor::Pink => (214, 51, 132),
            TrackColor::White => (255, 255, 255),
            TrackColor::Grey => (108, 117, 125),
        }
    }

    /// Returns the hex string representation (e.g. `"#DC3545"`).
    #[must_use]
    pub fn hex(&self) -> String {
        let (r, g, b) = self.rgb();
        format!("#{r:02X}{g:02X}{b:02X}")
    }

    /// All available colours in palette order.
    #[must_use]
    pub fn all() -> &'static [TrackColor] {
        &[
            TrackColor::Red,
            TrackColor::Orange,
            TrackColor::Yellow,
            TrackColor::Green,
            TrackColor::Teal,
            TrackColor::Blue,
            TrackColor::Purple,
            TrackColor::Pink,
            TrackColor::White,
            TrackColor::Grey,
        ]
    }
}

/// Associates a colour with a specific track id.
#[derive(Debug, Clone)]
pub struct TrackColorAssignment {
    /// Track identifier.
    pub track_id: u64,
    /// Assigned colour.
    assigned: TrackColor,
}

impl TrackColorAssignment {
    /// Creates a new colour assignment.
    #[must_use]
    pub fn new(track_id: u64, color: TrackColor) -> Self {
        Self {
            track_id,
            assigned: color,
        }
    }

    /// Returns the assigned colour.
    #[must_use]
    pub fn color(&self) -> TrackColor {
        self.assigned
    }

    /// Changes the assigned colour.
    pub fn set_color(&mut self, color: TrackColor) {
        self.assigned = color;
    }
}

/// Manages colour assignments across many tracks.
#[derive(Debug, Default)]
pub struct TrackColorManager {
    assignments: Vec<TrackColorAssignment>,
    palette_cursor: usize,
}

impl TrackColorManager {
    /// Creates a new manager with no assignments.
    #[must_use]
    pub fn new() -> Self {
        Self {
            assignments: Vec::new(),
            palette_cursor: 0,
        }
    }

    /// Assigns `color` to `track_id`, replacing any existing assignment.
    pub fn assign(&mut self, track_id: u64, color: TrackColor) {
        if let Some(a) = self.assignments.iter_mut().find(|a| a.track_id == track_id) {
            a.set_color(color);
        } else {
            self.assignments
                .push(TrackColorAssignment::new(track_id, color));
        }
    }

    /// Auto-assigns the next palette colour to `track_id`.
    pub fn auto_assign(&mut self, track_id: u64) -> TrackColor {
        let all = TrackColor::all();
        let color = all[self.palette_cursor % all.len()];
        self.palette_cursor += 1;
        self.assign(track_id, color);
        color
    }

    /// Returns the colour assigned to `track_id`, or `None` if unassigned.
    #[must_use]
    pub fn color_for_track(&self, track_id: u64) -> Option<TrackColor> {
        self.assignments
            .iter()
            .find(|a| a.track_id == track_id)
            .map(TrackColorAssignment::color)
    }

    /// Number of assigned tracks.
    #[must_use]
    pub fn count(&self) -> usize {
        self.assignments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_red_rgb() {
        let (r, _g, _b) = TrackColor::Red.rgb();
        assert_eq!(r, 220);
    }

    #[test]
    fn test_hex_format() {
        let hex = TrackColor::Blue.hex();
        assert!(hex.starts_with('#'));
        assert_eq!(hex.len(), 7);
    }

    #[test]
    fn test_all_colors_count() {
        assert_eq!(TrackColor::all().len(), 10);
    }

    #[test]
    fn test_white_rgb() {
        assert_eq!(TrackColor::White.rgb(), (255, 255, 255));
    }

    #[test]
    fn test_assignment_color() {
        let a = TrackColorAssignment::new(1, TrackColor::Green);
        assert_eq!(a.color(), TrackColor::Green);
    }

    #[test]
    fn test_assignment_set_color() {
        let mut a = TrackColorAssignment::new(1, TrackColor::Red);
        a.set_color(TrackColor::Blue);
        assert_eq!(a.color(), TrackColor::Blue);
    }

    #[test]
    fn test_manager_assign_new() {
        let mut mgr = TrackColorManager::new();
        mgr.assign(10, TrackColor::Teal);
        assert_eq!(mgr.color_for_track(10), Some(TrackColor::Teal));
    }

    #[test]
    fn test_manager_assign_overwrite() {
        let mut mgr = TrackColorManager::new();
        mgr.assign(5, TrackColor::Purple);
        mgr.assign(5, TrackColor::Yellow);
        assert_eq!(mgr.color_for_track(5), Some(TrackColor::Yellow));
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_manager_missing_track_is_none() {
        let mgr = TrackColorManager::new();
        assert!(mgr.color_for_track(99).is_none());
    }

    #[test]
    fn test_auto_assign_cycles_palette() {
        let mut mgr = TrackColorManager::new();
        let all = TrackColor::all();
        for (i, &expected) in all.iter().enumerate() {
            let got = mgr.auto_assign(i as u64 + 1);
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_auto_assign_wraps_around() {
        let mut mgr = TrackColorManager::new();
        let all = TrackColor::all();
        // Use up whole palette
        for i in 0..all.len() {
            mgr.auto_assign(i as u64 + 1);
        }
        // Next should wrap to first colour
        let wrapped = mgr.auto_assign(99);
        assert_eq!(wrapped, all[0]);
    }

    #[test]
    fn test_manager_count() {
        let mut mgr = TrackColorManager::new();
        mgr.assign(1, TrackColor::Pink);
        mgr.assign(2, TrackColor::Orange);
        assert_eq!(mgr.count(), 2);
    }
}
