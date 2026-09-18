#![allow(dead_code)]
//! Tally state machine for tracking input source status in live production.
//!
//! Provides a state machine that tracks tally light states across all
//! inputs, supporting program (red), preview (green), and ISO recording
//! tallies with priority resolution and transition tracking.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Tally color / priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TallyColor {
    /// No tally (off).
    Off,
    /// Green tally (preview).
    Green,
    /// Amber tally (ISO recording or secondary).
    Amber,
    /// Red tally (program / on-air).
    Red,
}

impl fmt::Display for TallyColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "OFF"),
            Self::Green => write!(f, "GREEN"),
            Self::Amber => write!(f, "AMBER"),
            Self::Red => write!(f, "RED"),
        }
    }
}

/// Source of a tally assignment (which M/E or bus originated it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallySource {
    /// Main program bus of an M/E row.
    MeProgram(usize),
    /// Preview bus of an M/E row.
    MePreview(usize),
    /// Aux bus output.
    AuxBus(usize),
    /// ISO recording assignment.
    IsoRecord(usize),
    /// Downstream keyer fill source.
    DskFill(usize),
    /// Multiviewer selection.
    Multiviewer(usize),
}

impl fmt::Display for TallySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeProgram(me) => write!(f, "M/E{} PGM", me + 1),
            Self::MePreview(me) => write!(f, "M/E{} PVW", me + 1),
            Self::AuxBus(aux) => write!(f, "AUX{}", aux + 1),
            Self::IsoRecord(iso) => write!(f, "ISO{}", iso + 1),
            Self::DskFill(dsk) => write!(f, "DSK{} Fill", dsk + 1),
            Self::Multiviewer(mv) => write!(f, "MV{}", mv + 1),
        }
    }
}

/// A single tally assignment linking source, input, and color.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyAssignment {
    /// Which source assigned this tally.
    pub source: TallySource,
    /// The input being tallied.
    pub input_id: usize,
    /// Tally color.
    pub color: TallyColor,
    /// Timestamp when this assignment was made.
    pub assigned_at: Instant,
}

impl TallyAssignment {
    /// Create a new tally assignment.
    pub fn new(source: TallySource, input_id: usize, color: TallyColor) -> Self {
        Self {
            source,
            input_id,
            color,
            assigned_at: Instant::now(),
        }
    }

    /// How long this assignment has been active.
    pub fn age(&self) -> Duration {
        self.assigned_at.elapsed()
    }
}

/// Resolved tally state for a single input (after priority resolution).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTally {
    /// The input ID.
    pub input_id: usize,
    /// Highest priority color.
    pub color: TallyColor,
    /// All contributing sources.
    pub sources: Vec<TallySource>,
    /// Whether this input is in transition.
    pub in_transition: bool,
}

impl ResolvedTally {
    /// Create a new resolved tally.
    pub fn new(input_id: usize) -> Self {
        Self {
            input_id,
            color: TallyColor::Off,
            sources: Vec::new(),
            in_transition: false,
        }
    }

    /// Whether this input is on-air (program).
    pub fn is_on_air(&self) -> bool {
        self.color == TallyColor::Red
    }

    /// Whether this input is in preview.
    pub fn is_preview(&self) -> bool {
        self.color == TallyColor::Green
    }

    /// Number of sources contributing to this tally.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

impl fmt::Display for ResolvedTally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Input {} [{}]", self.input_id, self.color)?;
        if self.in_transition {
            write!(f, " (TRANS)")?;
        }
        Ok(())
    }
}

/// Event emitted when tally state changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyChangeEvent {
    /// Input that changed.
    pub input_id: usize,
    /// Previous color.
    pub previous_color: TallyColor,
    /// New color.
    pub new_color: TallyColor,
    /// Source that triggered the change.
    pub trigger_source: TallySource,
    /// Timestamp of the change.
    pub timestamp: Instant,
}

impl fmt::Display for TallyChangeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Input {} : {} -> {} (via {})",
            self.input_id, self.previous_color, self.new_color, self.trigger_source
        )
    }
}

/// Tally state machine that tracks all assignments and resolves priorities.
#[derive(Debug, Clone)]
pub struct TallyStateMachine {
    /// All active assignments.
    assignments: Vec<TallyAssignment>,
    /// Cached resolved states per input.
    resolved: HashMap<usize, ResolvedTally>,
    /// Inputs currently in transition.
    in_transition: HashMap<usize, bool>,
    /// Maximum number of inputs tracked.
    max_inputs: usize,
    /// History of change events (ring buffer).
    history: Vec<TallyChangeEvent>,
    /// Maximum history size.
    max_history: usize,
}

impl TallyStateMachine {
    /// Create a new tally state machine.
    pub fn new(max_inputs: usize) -> Self {
        Self {
            assignments: Vec::new(),
            resolved: HashMap::new(),
            in_transition: HashMap::new(),
            max_inputs,
            history: Vec::new(),
            max_history: 256,
        }
    }

    /// Set maximum history buffer size.
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Add or update a tally assignment.
    pub fn assign(&mut self, source: TallySource, input_id: usize, color: TallyColor) {
        // Remove existing assignment from same source for same input
        self.assignments
            .retain(|a| !(a.source == source && a.input_id == input_id));

        if color != TallyColor::Off {
            self.assignments
                .push(TallyAssignment::new(source, input_id, color));
        }

        self.resolve(input_id, source);
    }

    /// Remove all assignments from a given source.
    pub fn clear_source(&mut self, source: TallySource) {
        let affected: Vec<usize> = self
            .assignments
            .iter()
            .filter(|a| a.source == source)
            .map(|a| a.input_id)
            .collect();

        self.assignments.retain(|a| a.source != source);

        for input_id in affected {
            self.resolve(input_id, source);
        }
    }

    /// Set transition state for an input.
    pub fn set_in_transition(&mut self, input_id: usize, in_transition: bool) {
        self.in_transition.insert(input_id, in_transition);
        if let Some(resolved) = self.resolved.get_mut(&input_id) {
            resolved.in_transition = in_transition;
        }
    }

    /// Get resolved tally for an input.
    pub fn get_resolved(&self, input_id: usize) -> ResolvedTally {
        self.resolved
            .get(&input_id)
            .cloned()
            .unwrap_or_else(|| ResolvedTally::new(input_id))
    }

    /// Get all resolved tallies.
    pub fn get_all_resolved(&self) -> Vec<ResolvedTally> {
        let mut result: Vec<_> = self.resolved.values().cloned().collect();
        result.sort_by_key(|r| r.input_id);
        result
    }

    /// Get inputs that are currently on-air (red tally).
    pub fn on_air_inputs(&self) -> Vec<usize> {
        self.resolved
            .values()
            .filter(|r| r.color == TallyColor::Red)
            .map(|r| r.input_id)
            .collect()
    }

    /// Get inputs that are currently in preview (green tally).
    pub fn preview_inputs(&self) -> Vec<usize> {
        self.resolved
            .values()
            .filter(|r| r.color == TallyColor::Green)
            .map(|r| r.input_id)
            .collect()
    }

    /// Get change history.
    pub fn history(&self) -> &[TallyChangeEvent] {
        &self.history
    }

    /// Clear all assignments and resolved state.
    pub fn clear_all(&mut self) {
        self.assignments.clear();
        self.resolved.clear();
        self.in_transition.clear();
    }

    /// Number of active assignments.
    pub fn assignment_count(&self) -> usize {
        self.assignments.len()
    }

    /// Number of inputs with non-off tally.
    pub fn active_input_count(&self) -> usize {
        self.resolved
            .values()
            .filter(|r| r.color != TallyColor::Off)
            .count()
    }

    /// Resolve tally state for a given input by finding highest priority color.
    fn resolve(&mut self, input_id: usize, trigger_source: TallySource) {
        let old_color = self
            .resolved
            .get(&input_id)
            .map_or(TallyColor::Off, |r| r.color);

        let mut resolved = ResolvedTally::new(input_id);
        resolved.in_transition = self.in_transition.get(&input_id).copied().unwrap_or(false);

        for assignment in &self.assignments {
            if assignment.input_id == input_id {
                resolved.sources.push(assignment.source);
                if assignment.color > resolved.color {
                    resolved.color = assignment.color;
                }
            }
        }

        let new_color = resolved.color;
        self.resolved.insert(input_id, resolved);

        if old_color != new_color {
            let event = TallyChangeEvent {
                input_id,
                previous_color: old_color,
                new_color,
                trigger_source,
                timestamp: Instant::now(),
            };
            self.history.push(event);
            if self.history.len() > self.max_history {
                self.history.remove(0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tally_color_ordering() {
        assert!(TallyColor::Off < TallyColor::Green);
        assert!(TallyColor::Green < TallyColor::Amber);
        assert!(TallyColor::Amber < TallyColor::Red);
    }

    #[test]
    fn test_tally_color_display() {
        assert_eq!(format!("{}", TallyColor::Red), "RED");
        assert_eq!(format!("{}", TallyColor::Green), "GREEN");
        assert_eq!(format!("{}", TallyColor::Off), "OFF");
    }

    #[test]
    fn test_tally_source_display() {
        assert_eq!(format!("{}", TallySource::MeProgram(0)), "M/E1 PGM");
        assert_eq!(format!("{}", TallySource::MePreview(1)), "M/E2 PVW");
        assert_eq!(format!("{}", TallySource::AuxBus(0)), "AUX1");
    }

    #[test]
    fn test_new_state_machine() {
        let sm = TallyStateMachine::new(16);
        assert_eq!(sm.assignment_count(), 0);
        assert_eq!(sm.active_input_count(), 0);
    }

    #[test]
    fn test_assign_program() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        let resolved = sm.get_resolved(1);
        assert_eq!(resolved.color, TallyColor::Red);
        assert!(resolved.is_on_air());
        assert_eq!(resolved.source_count(), 1);
    }

    #[test]
    fn test_assign_preview() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MePreview(0), 2, TallyColor::Green);
        let resolved = sm.get_resolved(2);
        assert_eq!(resolved.color, TallyColor::Green);
        assert!(resolved.is_preview());
    }

    #[test]
    fn test_priority_resolution() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MePreview(0), 1, TallyColor::Green);
        sm.assign(TallySource::MeProgram(1), 1, TallyColor::Red);
        let resolved = sm.get_resolved(1);
        assert_eq!(resolved.color, TallyColor::Red);
        assert_eq!(resolved.source_count(), 2);
    }

    #[test]
    fn test_clear_source() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        sm.assign(TallySource::MeProgram(0), 2, TallyColor::Red);
        sm.clear_source(TallySource::MeProgram(0));
        assert_eq!(sm.get_resolved(1).color, TallyColor::Off);
        assert_eq!(sm.get_resolved(2).color, TallyColor::Off);
    }

    #[test]
    fn test_on_air_inputs() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        sm.assign(TallySource::MePreview(0), 2, TallyColor::Green);
        sm.assign(TallySource::MeProgram(1), 3, TallyColor::Red);
        let on_air = sm.on_air_inputs();
        assert_eq!(on_air.len(), 2);
        assert!(on_air.contains(&1));
        assert!(on_air.contains(&3));
    }

    #[test]
    fn test_preview_inputs() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MePreview(0), 5, TallyColor::Green);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        let preview = sm.preview_inputs();
        assert_eq!(preview.len(), 1);
        assert!(preview.contains(&5));
    }

    #[test]
    fn test_transition_state() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        sm.set_in_transition(1, true);
        let resolved = sm.get_resolved(1);
        assert!(resolved.in_transition);
    }

    #[test]
    fn test_change_history() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Off);
        assert!(sm.history().len() >= 2);
        assert_eq!(sm.history()[0].new_color, TallyColor::Red);
    }

    #[test]
    fn test_clear_all() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 1, TallyColor::Red);
        sm.assign(TallySource::MePreview(0), 2, TallyColor::Green);
        sm.clear_all();
        assert_eq!(sm.assignment_count(), 0);
        assert_eq!(sm.active_input_count(), 0);
    }

    #[test]
    fn test_resolved_tally_display() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 3, TallyColor::Red);
        sm.set_in_transition(3, true);
        let resolved = sm.get_resolved(3);
        let s = format!("{resolved}");
        assert!(s.contains("Input 3"));
        assert!(s.contains("RED"));
        assert!(s.contains("TRANS"));
    }

    #[test]
    fn test_tally_change_event_display() {
        let event = TallyChangeEvent {
            input_id: 2,
            previous_color: TallyColor::Off,
            new_color: TallyColor::Red,
            trigger_source: TallySource::MeProgram(0),
            timestamp: Instant::now(),
        };
        let s = format!("{event}");
        assert!(s.contains("Input 2"));
        assert!(s.contains("OFF -> RED"));
    }

    #[test]
    fn test_get_all_resolved_sorted() {
        let mut sm = TallyStateMachine::new(16);
        sm.assign(TallySource::MeProgram(0), 5, TallyColor::Red);
        sm.assign(TallySource::MePreview(0), 1, TallyColor::Green);
        let all = sm.get_all_resolved();
        assert_eq!(all.len(), 2);
        assert!(all[0].input_id < all[1].input_id);
    }
}
