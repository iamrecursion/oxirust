#![allow(dead_code)]
//! Transition events for EDL sequences.
//!
//! An EDL transition event describes the type and duration of a transition
//! between two edit events (dissolves, wipes, keys, etc.).  This module
//! provides [`TransitionType`], [`TransitionEvent`], and
//! [`TransitionEventList`] for building and querying transition sequences.

use std::fmt;

/// Supported transition types in an EDL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TransitionType {
    /// Hard cut (no transition).
    Cut,
    /// Cross-dissolve.
    Dissolve,
    /// SMPTE wipe identified by a numeric pattern code.
    Wipe(u16),
    /// Key transition (luminance or chroma key).
    Key,
    /// Dip-to-black.
    DipToBlack,
    /// Dip-to-white.
    DipToWhite,
}

impl TransitionType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Cut => "Cut".to_string(),
            Self::Dissolve => "Dissolve".to_string(),
            Self::Wipe(code) => format!("Wipe {code}"),
            Self::Key => "Key".to_string(),
            Self::DipToBlack => "Dip to Black".to_string(),
            Self::DipToWhite => "Dip to White".to_string(),
        }
    }

    /// CMX 3600 edit-type code character.
    #[must_use]
    pub fn cmx_code(&self) -> &'static str {
        match self {
            Self::Cut => "C",
            Self::Dissolve => "D",
            Self::Wipe(_) => "W",
            Self::Key => "K",
            Self::DipToBlack => "DB",
            Self::DipToWhite => "DW",
        }
    }

    /// Whether this transition has a duration (non-cut).
    #[must_use]
    pub const fn has_duration(&self) -> bool {
        !matches!(self, Self::Cut)
    }

    /// Whether this is a dissolve variant.
    #[must_use]
    pub const fn is_dissolve_variant(&self) -> bool {
        matches!(self, Self::Dissolve | Self::DipToBlack | Self::DipToWhite)
    }
}

impl fmt::Display for TransitionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

/// A single transition event between two edit points.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TransitionEvent {
    /// Event number in the EDL.
    event_number: u32,
    /// Transition type.
    transition_type: TransitionType,
    /// Duration of the transition in frames.
    duration_frames: u32,
    /// Frame rate (frames per second).
    frame_rate: f64,
    /// Optional reel name of the outgoing source.
    outgoing_reel: Option<String>,
    /// Optional reel name of the incoming source.
    incoming_reel: Option<String>,
    /// Optional comment.
    comment: Option<String>,
}

impl TransitionEvent {
    /// Create a new transition event.
    #[must_use]
    pub fn new(
        event_number: u32,
        transition_type: TransitionType,
        duration_frames: u32,
        frame_rate: f64,
    ) -> Self {
        Self {
            event_number,
            transition_type,
            duration_frames,
            frame_rate,
            outgoing_reel: None,
            incoming_reel: None,
            comment: None,
        }
    }

    /// Builder: set outgoing reel.
    #[must_use]
    pub fn with_outgoing_reel(mut self, reel: impl Into<String>) -> Self {
        self.outgoing_reel = Some(reel.into());
        self
    }

    /// Builder: set incoming reel.
    #[must_use]
    pub fn with_incoming_reel(mut self, reel: impl Into<String>) -> Self {
        self.incoming_reel = Some(reel.into());
        self
    }

    /// Builder: set comment.
    #[must_use]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Event number.
    #[must_use]
    pub fn event_number(&self) -> u32 {
        self.event_number
    }

    /// Transition type.
    #[must_use]
    pub fn transition_type(&self) -> TransitionType {
        self.transition_type
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u32 {
        self.duration_frames
    }

    /// Frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    /// Duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.frame_rate <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f64 / self.frame_rate
    }

    /// Outgoing reel name.
    #[must_use]
    pub fn outgoing_reel(&self) -> Option<&str> {
        self.outgoing_reel.as_deref()
    }

    /// Incoming reel name.
    #[must_use]
    pub fn incoming_reel(&self) -> Option<&str> {
        self.incoming_reel.as_deref()
    }

    /// Optional comment.
    #[must_use]
    pub fn comment(&self) -> Option<&str> {
        self.comment.as_deref()
    }

    /// Format as a CMX 3600 transition field string.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_cmx_string(&self) -> String {
        if self.transition_type == TransitionType::Cut {
            "C".to_string()
        } else {
            format!(
                "{}    {:03}",
                self.transition_type.cmx_code(),
                self.duration_frames
            )
        }
    }
}

/// An ordered list of [`TransitionEvent`]s.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TransitionEventList {
    events: Vec<TransitionEvent>,
}

impl TransitionEventList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Push a transition event.
    pub fn push(&mut self, event: TransitionEvent) {
        self.events.push(event);
    }

    /// Remove and return the last event.
    pub fn pop(&mut self) -> Option<TransitionEvent> {
        self.events.pop()
    }

    /// Number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get an event by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&TransitionEvent> {
        self.events.get(index)
    }

    /// Return all dissolve events.
    #[must_use]
    pub fn dissolves(&self) -> Vec<&TransitionEvent> {
        self.events
            .iter()
            .filter(|e| e.transition_type.is_dissolve_variant())
            .collect()
    }

    /// Return all wipe events.
    #[must_use]
    pub fn wipes(&self) -> Vec<&TransitionEvent> {
        self.events
            .iter()
            .filter(|e| matches!(e.transition_type, TransitionType::Wipe(_)))
            .collect()
    }

    /// Total transition duration in frames.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.events
            .iter()
            .map(|e| u64::from(e.duration_frames))
            .sum()
    }

    /// Iterator over events.
    pub fn iter(&self) -> impl Iterator<Item = &TransitionEvent> {
        self.events.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_type_label() {
        assert_eq!(TransitionType::Cut.label(), "Cut");
        assert_eq!(TransitionType::Dissolve.label(), "Dissolve");
        assert_eq!(TransitionType::Wipe(4).label(), "Wipe 4");
        assert_eq!(TransitionType::Key.label(), "Key");
        assert_eq!(TransitionType::DipToBlack.label(), "Dip to Black");
        assert_eq!(TransitionType::DipToWhite.label(), "Dip to White");
    }

    #[test]
    fn test_transition_type_cmx_code() {
        assert_eq!(TransitionType::Cut.cmx_code(), "C");
        assert_eq!(TransitionType::Dissolve.cmx_code(), "D");
        assert_eq!(TransitionType::Wipe(1).cmx_code(), "W");
        assert_eq!(TransitionType::Key.cmx_code(), "K");
    }

    #[test]
    fn test_transition_type_has_duration() {
        assert!(!TransitionType::Cut.has_duration());
        assert!(TransitionType::Dissolve.has_duration());
        assert!(TransitionType::Wipe(1).has_duration());
    }

    #[test]
    fn test_transition_type_is_dissolve_variant() {
        assert!(TransitionType::Dissolve.is_dissolve_variant());
        assert!(TransitionType::DipToBlack.is_dissolve_variant());
        assert!(TransitionType::DipToWhite.is_dissolve_variant());
        assert!(!TransitionType::Cut.is_dissolve_variant());
        assert!(!TransitionType::Wipe(1).is_dissolve_variant());
    }

    #[test]
    fn test_transition_type_display() {
        assert_eq!(format!("{}", TransitionType::Key), "Key");
    }

    #[test]
    fn test_transition_event_creation() {
        let ev = TransitionEvent::new(1, TransitionType::Dissolve, 30, 25.0);
        assert_eq!(ev.event_number(), 1);
        assert_eq!(ev.transition_type(), TransitionType::Dissolve);
        assert_eq!(ev.duration_frames(), 30);
        assert!((ev.frame_rate() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transition_event_builders() {
        let ev = TransitionEvent::new(2, TransitionType::Wipe(4), 15, 30.0)
            .with_outgoing_reel("REEL_A")
            .with_incoming_reel("REEL_B")
            .with_comment("smooth wipe");
        assert_eq!(ev.outgoing_reel(), Some("REEL_A"));
        assert_eq!(ev.incoming_reel(), Some("REEL_B"));
        assert_eq!(ev.comment(), Some("smooth wipe"));
    }

    #[test]
    fn test_transition_event_duration_seconds() {
        let ev = TransitionEvent::new(1, TransitionType::Dissolve, 50, 25.0);
        let dur = ev.duration_seconds();
        assert!((dur - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_transition_event_duration_zero_rate() {
        let ev = TransitionEvent::new(1, TransitionType::Dissolve, 50, 0.0);
        assert_eq!(ev.duration_seconds(), 0.0);
    }

    #[test]
    fn test_transition_event_to_cmx_string_cut() {
        let ev = TransitionEvent::new(1, TransitionType::Cut, 0, 25.0);
        assert_eq!(ev.to_cmx_string(), "C");
    }

    #[test]
    fn test_transition_event_to_cmx_string_dissolve() {
        let ev = TransitionEvent::new(1, TransitionType::Dissolve, 30, 25.0);
        assert_eq!(ev.to_cmx_string(), "D    030");
    }

    #[test]
    fn test_event_list_push_pop() {
        let mut list = TransitionEventList::new();
        assert!(list.is_empty());
        list.push(TransitionEvent::new(1, TransitionType::Cut, 0, 25.0));
        assert_eq!(list.len(), 1);
        let popped = list.pop();
        assert!(popped.is_some());
        assert!(list.is_empty());
    }

    #[test]
    fn test_event_list_dissolves_and_wipes() {
        let mut list = TransitionEventList::new();
        list.push(TransitionEvent::new(1, TransitionType::Cut, 0, 25.0));
        list.push(TransitionEvent::new(2, TransitionType::Dissolve, 30, 25.0));
        list.push(TransitionEvent::new(3, TransitionType::Wipe(1), 15, 25.0));
        list.push(TransitionEvent::new(
            4,
            TransitionType::DipToBlack,
            20,
            25.0,
        ));

        assert_eq!(list.dissolves().len(), 2);
        assert_eq!(list.wipes().len(), 1);
    }

    #[test]
    fn test_event_list_total_duration() {
        let mut list = TransitionEventList::new();
        list.push(TransitionEvent::new(1, TransitionType::Dissolve, 30, 25.0));
        list.push(TransitionEvent::new(2, TransitionType::Wipe(1), 20, 25.0));
        assert_eq!(list.total_duration_frames(), 50);
    }

    #[test]
    fn test_event_list_get() {
        let mut list = TransitionEventList::new();
        list.push(TransitionEvent::new(1, TransitionType::Cut, 0, 25.0));
        assert!(list.get(0).is_some());
        assert!(list.get(1).is_none());
    }

    #[test]
    fn test_event_list_iter() {
        let mut list = TransitionEventList::new();
        list.push(TransitionEvent::new(1, TransitionType::Cut, 0, 25.0));
        list.push(TransitionEvent::new(2, TransitionType::Dissolve, 30, 25.0));
        assert_eq!(list.iter().count(), 2);
    }
}
