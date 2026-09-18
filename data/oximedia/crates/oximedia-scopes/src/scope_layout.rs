#![allow(dead_code)]
//! Multi-scope layout management for video monitoring.
//!
//! Allows multiple scopes (waveform, vectorscope, histogram, …) to be
//! arranged on screen simultaneously in configurable positions.

use crate::ScopeType;

/// Screen position for a scope widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopePosition {
    /// Upper-left quadrant.
    TopLeft,
    /// Upper-right quadrant.
    TopRight,
    /// Lower-left quadrant.
    BottomLeft,
    /// Lower-right quadrant.
    BottomRight,
    /// Full-screen, occupies the entire display.
    Fullscreen,
}

impl ScopePosition {
    /// Returns the normalised X offset `[0.0, 1.0]` from the left edge.
    #[must_use]
    pub fn x_offset(self) -> f32 {
        match self {
            Self::TopLeft | Self::BottomLeft => 0.0,
            Self::TopRight | Self::BottomRight => 0.5,
            Self::Fullscreen => 0.0,
        }
    }

    /// Returns the normalised Y offset `[0.0, 1.0]` from the top edge.
    #[must_use]
    pub fn y_offset(self) -> f32 {
        match self {
            Self::TopLeft | Self::TopRight => 0.0,
            Self::BottomLeft | Self::BottomRight => 0.5,
            Self::Fullscreen => 0.0,
        }
    }

    /// Returns the normalised width `(0.0, 1.0]` the scope occupies.
    #[must_use]
    pub fn norm_width(self) -> f32 {
        match self {
            Self::Fullscreen => 1.0,
            _ => 0.5,
        }
    }

    /// Returns the normalised height `(0.0, 1.0]` the scope occupies.
    #[must_use]
    pub fn norm_height(self) -> f32 {
        match self {
            Self::Fullscreen => 1.0,
            _ => 0.5,
        }
    }

    /// Returns all non-fullscreen positions.
    #[must_use]
    pub fn quadrants() -> [Self; 4] {
        [
            Self::TopLeft,
            Self::TopRight,
            Self::BottomLeft,
            Self::BottomRight,
        ]
    }
}

/// A single scope entry in a layout.
#[derive(Debug, Clone)]
pub struct ScopeEntry {
    /// The scope type to render at this position.
    pub scope_type: ScopeType,
    /// Screen position for the scope.
    pub position: ScopePosition,
    /// Opacity for the scope overlay `[0.0, 1.0]`.
    pub opacity: f32,
}

impl ScopeEntry {
    /// Creates a new entry with full opacity.
    #[must_use]
    pub fn new(scope_type: ScopeType, position: ScopePosition) -> Self {
        Self {
            scope_type,
            position,
            opacity: 1.0,
        }
    }

    /// Creates a new entry with a custom opacity.
    #[must_use]
    pub fn with_opacity(scope_type: ScopeType, position: ScopePosition, opacity: f32) -> Self {
        Self {
            scope_type,
            position,
            opacity: opacity.clamp(0.0, 1.0),
        }
    }
}

/// Layout manager for multiple simultaneous scopes.
#[derive(Debug, Clone, Default)]
pub struct ScopeLayout {
    entries: Vec<ScopeEntry>,
}

impl ScopeLayout {
    /// Creates an empty layout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a scope at the given position.
    ///
    /// If `position` is `Fullscreen`, all existing entries are removed
    /// because a full-screen scope occupies the whole display.
    pub fn add_scope(&mut self, scope_type: ScopeType, position: ScopePosition) {
        if position == ScopePosition::Fullscreen {
            self.entries.clear();
        }
        self.entries.push(ScopeEntry::new(scope_type, position));
    }

    /// Adds a scope with custom opacity.
    pub fn add_scope_with_opacity(
        &mut self,
        scope_type: ScopeType,
        position: ScopePosition,
        opacity: f32,
    ) {
        self.entries
            .push(ScopeEntry::with_opacity(scope_type, position, opacity));
    }

    /// Removes all scopes at the given position.
    pub fn remove_position(&mut self, position: ScopePosition) {
        self.entries.retain(|e| e.position != position);
    }

    /// Returns the set of positions currently occupied.
    #[must_use]
    pub fn positions_used(&self) -> Vec<ScopePosition> {
        let mut seen = Vec::new();
        for entry in &self.entries {
            if !seen.contains(&entry.position) {
                seen.push(entry.position);
            }
        }
        seen
    }

    /// Returns the number of scope entries in this layout.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when there are no scope entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Validates the layout.
    ///
    /// A layout is valid when:
    /// - It is non-empty.
    /// - A `Fullscreen` scope, if present, is the only entry.
    /// - No position is used more than once (except in edge cases with opacity).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let has_fullscreen = self
            .entries
            .iter()
            .any(|e| e.position == ScopePosition::Fullscreen);
        if has_fullscreen {
            return self.entries.len() == 1;
        }
        // Check for duplicate positions
        let positions = self.positions_used();
        positions.len() == self.entries.len()
    }

    /// Returns an iterator over all scope entries.
    pub fn entries(&self) -> &[ScopeEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_x_offset_top_left() {
        assert_eq!(ScopePosition::TopLeft.x_offset(), 0.0);
    }

    #[test]
    fn test_position_x_offset_top_right() {
        assert_eq!(ScopePosition::TopRight.x_offset(), 0.5);
    }

    #[test]
    fn test_position_y_offset_bottom() {
        assert_eq!(ScopePosition::BottomLeft.y_offset(), 0.5);
    }

    #[test]
    fn test_position_fullscreen_offsets() {
        assert_eq!(ScopePosition::Fullscreen.x_offset(), 0.0);
        assert_eq!(ScopePosition::Fullscreen.y_offset(), 0.0);
    }

    #[test]
    fn test_position_fullscreen_norm_size() {
        assert_eq!(ScopePosition::Fullscreen.norm_width(), 1.0);
        assert_eq!(ScopePosition::Fullscreen.norm_height(), 1.0);
    }

    #[test]
    fn test_position_quadrant_norm_size() {
        assert_eq!(ScopePosition::TopLeft.norm_width(), 0.5);
        assert_eq!(ScopePosition::BottomRight.norm_height(), 0.5);
    }

    #[test]
    fn test_scope_entry_opacity_clamped() {
        let e = ScopeEntry::with_opacity(ScopeType::Vectorscope, ScopePosition::TopLeft, 2.5);
        assert_eq!(e.opacity, 1.0);
    }

    #[test]
    fn test_layout_add_scope() {
        let mut layout = ScopeLayout::new();
        layout.add_scope(ScopeType::WaveformLuma, ScopePosition::TopLeft);
        assert_eq!(layout.len(), 1);
    }

    #[test]
    fn test_layout_positions_used() {
        let mut layout = ScopeLayout::new();
        layout.add_scope(ScopeType::WaveformLuma, ScopePosition::TopLeft);
        layout.add_scope(ScopeType::Vectorscope, ScopePosition::TopRight);
        let pos = layout.positions_used();
        assert_eq!(pos.len(), 2);
    }

    #[test]
    fn test_layout_is_valid_two_scopes() {
        let mut layout = ScopeLayout::new();
        layout.add_scope(ScopeType::WaveformLuma, ScopePosition::TopLeft);
        layout.add_scope(ScopeType::Vectorscope, ScopePosition::TopRight);
        assert!(layout.is_valid());
    }

    #[test]
    fn test_layout_is_valid_empty() {
        let layout = ScopeLayout::new();
        assert!(!layout.is_valid());
    }

    #[test]
    fn test_layout_fullscreen_clears_others() {
        let mut layout = ScopeLayout::new();
        layout.add_scope(ScopeType::WaveformLuma, ScopePosition::TopLeft);
        layout.add_scope(ScopeType::Vectorscope, ScopePosition::Fullscreen);
        assert_eq!(layout.len(), 1);
        assert!(layout.is_valid());
    }

    #[test]
    fn test_layout_remove_position() {
        let mut layout = ScopeLayout::new();
        layout.add_scope(ScopeType::WaveformLuma, ScopePosition::TopLeft);
        layout.add_scope(ScopeType::Vectorscope, ScopePosition::BottomRight);
        layout.remove_position(ScopePosition::TopLeft);
        assert_eq!(layout.len(), 1);
        assert_eq!(layout.entries()[0].position, ScopePosition::BottomRight);
    }

    #[test]
    fn test_quadrants_count() {
        assert_eq!(ScopePosition::quadrants().len(), 4);
    }
}
