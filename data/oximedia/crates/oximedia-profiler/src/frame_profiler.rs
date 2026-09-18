//! Per-frame phase profiling with budget enforcement.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

/// A named phase within a single frame's processing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FramePhase {
    /// Input / capture phase.
    Input,
    /// Decode phase.
    Decode,
    /// Processing / filtering phase.
    Process,
    /// Encode phase.
    Encode,
    /// Output / display phase.
    Output,
    /// Custom / user-defined phase.
    Custom(u32),
}

impl FramePhase {
    /// Human-readable name for the phase.
    pub fn phase_name(&self) -> String {
        match self {
            FramePhase::Input => "input".to_string(),
            FramePhase::Decode => "decode".to_string(),
            FramePhase::Process => "process".to_string(),
            FramePhase::Encode => "encode".to_string(),
            FramePhase::Output => "output".to_string(),
            FramePhase::Custom(id) => format!("custom_{id}"),
        }
    }
}

/// Timing data for a single frame.
#[derive(Debug, Clone)]
pub struct FrameProfile {
    /// Frame sequence number.
    pub frame_id: u64,
    /// Duration in milliseconds for each recorded phase.
    pub phase_durations_ms: HashMap<FramePhase, f64>,
}

impl FrameProfile {
    /// Create an empty frame profile.
    pub fn new(frame_id: u64) -> Self {
        Self {
            frame_id,
            phase_durations_ms: HashMap::new(),
        }
    }

    /// Sum of all phase durations (total frame time in ms).
    pub fn total_ms(&self) -> f64 {
        self.phase_durations_ms.values().sum()
    }

    /// Phase with the highest duration, or `None` if empty.
    pub fn slowest_phase(&self) -> Option<(FramePhase, f64)> {
        self.phase_durations_ms
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&phase, &ms)| (phase, ms))
    }

    /// Duration of a specific phase, or `0.0` if not present.
    pub fn phase_ms(&self, phase: FramePhase) -> f64 {
        *self.phase_durations_ms.get(&phase).unwrap_or(&0.0)
    }

    /// Returns `true` if the frame's total time exceeds `budget_ms`.
    pub fn exceeds_budget(&self, budget_ms: f64) -> bool {
        self.total_ms() > budget_ms
    }
}

/// A report aggregating multiple frame profiles.
#[derive(Debug, Default)]
pub struct FrameReport {
    /// All collected frame profiles.
    pub frames: Vec<FrameProfile>,
    /// Target frame budget in ms (e.g., 16.67 for 60 fps).
    pub budget_ms: f64,
}

impl FrameReport {
    /// Create a report with a given frame budget.
    pub fn new(budget_ms: f64) -> Self {
        Self {
            frames: Vec::new(),
            budget_ms,
        }
    }

    /// Average frame time across all frames in ms.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_frame_ms(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let total: f64 = self.frames.iter().map(|f| f.total_ms()).sum();
        total / self.frames.len() as f64
    }

    /// Number of frames that exceeded the budget.
    pub fn budget_violations(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| f.exceeds_budget(self.budget_ms))
            .count()
    }

    /// The single slowest frame (highest total_ms), if any.
    pub fn slowest_frame(&self) -> Option<&FrameProfile> {
        self.frames.iter().max_by(|a, b| {
            a.total_ms()
                .partial_cmp(&b.total_ms())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Profiler that measures per-frame phase timings.
#[derive(Debug)]
pub struct FrameProfiler {
    frame_counter: u64,
    current_frame: Option<FrameProfile>,
    active_phases: HashMap<FramePhase, Instant>,
    completed_frames: Vec<FrameProfile>,
    budget_ms: f64,
}

impl FrameProfiler {
    /// Create a new frame profiler with the given frame budget.
    pub fn new(budget_ms: f64) -> Self {
        Self {
            frame_counter: 0,
            current_frame: None,
            active_phases: HashMap::new(),
            completed_frames: Vec::new(),
            budget_ms,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.current_frame = Some(FrameProfile::new(self.frame_counter));
        self.frame_counter += 1;
        self.active_phases.clear();
    }

    /// Mark the start of a phase within the current frame.
    pub fn start_phase(&mut self, phase: FramePhase) {
        self.active_phases.insert(phase, Instant::now());
    }

    /// Mark the end of a phase within the current frame, recording its duration.
    pub fn end_phase(&mut self, phase: FramePhase) {
        if let Some(start) = self.active_phases.remove(&phase) {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if let Some(ref mut frame) = self.current_frame {
                frame
                    .phase_durations_ms
                    .entry(phase)
                    .and_modify(|v| *v += elapsed_ms)
                    .or_insert(elapsed_ms);
            }
        }
    }

    /// Finish the current frame and store it.
    pub fn end_frame(&mut self) {
        if let Some(frame) = self.current_frame.take() {
            self.completed_frames.push(frame);
        }
    }

    /// Build and return a frame report from all completed frames.
    pub fn frame_report(&self) -> FrameReport {
        let mut report = FrameReport::new(self.budget_ms);
        report.frames.clone_from(&self.completed_frames);
        report
    }

    /// Number of completed frames.
    pub fn completed_count(&self) -> usize {
        self.completed_frames.len()
    }

    /// The most recently completed frame, if any.
    pub fn last_frame(&self) -> Option<&FrameProfile> {
        self.completed_frames.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_phase_names() {
        assert_eq!(FramePhase::Input.phase_name(), "input");
        assert_eq!(FramePhase::Decode.phase_name(), "decode");
        assert_eq!(FramePhase::Process.phase_name(), "process");
        assert_eq!(FramePhase::Encode.phase_name(), "encode");
        assert_eq!(FramePhase::Output.phase_name(), "output");
        assert_eq!(FramePhase::Custom(7).phase_name(), "custom_7");
    }

    #[test]
    fn test_frame_profile_total_ms_empty() {
        let fp = FrameProfile::new(0);
        assert_eq!(fp.total_ms(), 0.0);
    }

    #[test]
    fn test_frame_profile_total_ms() {
        let mut fp = FrameProfile::new(0);
        fp.phase_durations_ms.insert(FramePhase::Decode, 5.0);
        fp.phase_durations_ms.insert(FramePhase::Process, 10.0);
        assert!((fp.total_ms() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_profile_slowest_phase_empty() {
        let fp = FrameProfile::new(1);
        assert!(fp.slowest_phase().is_none());
    }

    #[test]
    fn test_frame_profile_slowest_phase() {
        let mut fp = FrameProfile::new(1);
        fp.phase_durations_ms.insert(FramePhase::Decode, 3.0);
        fp.phase_durations_ms.insert(FramePhase::Encode, 12.0);
        fp.phase_durations_ms.insert(FramePhase::Output, 1.0);
        let (phase, ms) = fp.slowest_phase().expect("should succeed in test");
        assert_eq!(phase, FramePhase::Encode);
        assert!((ms - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_profile_exceeds_budget() {
        let mut fp = FrameProfile::new(2);
        fp.phase_durations_ms.insert(FramePhase::Process, 20.0);
        assert!(fp.exceeds_budget(16.67));
        assert!(!fp.exceeds_budget(25.0));
    }

    #[test]
    fn test_frame_report_avg_empty() {
        let report = FrameReport::new(16.67);
        assert_eq!(report.avg_frame_ms(), 0.0);
    }

    #[test]
    fn test_frame_report_avg_frame_ms() {
        let mut report = FrameReport::new(16.67);
        let mut f1 = FrameProfile::new(0);
        f1.phase_durations_ms.insert(FramePhase::Process, 10.0);
        let mut f2 = FrameProfile::new(1);
        f2.phase_durations_ms.insert(FramePhase::Process, 20.0);
        report.frames.push(f1);
        report.frames.push(f2);
        assert!((report.avg_frame_ms() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_report_budget_violations() {
        let mut report = FrameReport::new(16.67);
        let mut f1 = FrameProfile::new(0);
        f1.phase_durations_ms.insert(FramePhase::Process, 10.0);
        let mut f2 = FrameProfile::new(1);
        f2.phase_durations_ms.insert(FramePhase::Process, 20.0);
        report.frames.push(f1);
        report.frames.push(f2);
        assert_eq!(report.budget_violations(), 1);
    }

    #[test]
    fn test_frame_profiler_basic_flow() {
        let mut p = FrameProfiler::new(16.67);
        p.begin_frame();
        p.start_phase(FramePhase::Decode);
        p.end_phase(FramePhase::Decode);
        p.end_frame();
        assert_eq!(p.completed_count(), 1);
        let frame = p.last_frame().expect("should succeed in test");
        assert!(frame.phase_ms(FramePhase::Decode) >= 0.0);
    }

    #[test]
    fn test_frame_profiler_multiple_frames() {
        let mut p = FrameProfiler::new(16.67);
        for _ in 0..3 {
            p.begin_frame();
            p.start_phase(FramePhase::Process);
            p.end_phase(FramePhase::Process);
            p.end_frame();
        }
        assert_eq!(p.completed_count(), 3);
        let report = p.frame_report();
        assert_eq!(report.frames.len(), 3);
    }

    #[test]
    fn test_frame_profiler_orphan_end_phase_no_panic() {
        // Calling end_phase without a matching start_phase should be a no-op.
        let mut p = FrameProfiler::new(16.67);
        p.begin_frame();
        p.end_phase(FramePhase::Encode); // no matching start
        p.end_frame();
        let frame = p.last_frame().expect("should succeed in test");
        assert_eq!(frame.phase_ms(FramePhase::Encode), 0.0);
    }

    #[test]
    fn test_frame_profiler_report_budget() {
        let p = FrameProfiler::new(33.33);
        let report = p.frame_report();
        assert!((report.budget_ms - 33.33).abs() < 1e-9);
    }
}
