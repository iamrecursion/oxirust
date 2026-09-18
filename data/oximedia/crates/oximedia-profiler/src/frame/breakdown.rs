//! Per-stage frame breakdown.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Frame processing stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameStage {
    /// Input processing.
    Input,

    /// Update logic.
    Update,

    /// Physics simulation.
    Physics,

    /// Rendering.
    Render,

    /// Post-processing.
    PostProcess,

    /// UI rendering.
    UI,

    /// Present/swap.
    Present,

    /// Other/custom stage.
    Other,
}

/// Frame breakdown showing time spent in each stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameBreakdown {
    /// Time spent in each stage.
    pub stage_times: HashMap<FrameStage, Duration>,

    /// Total frame time.
    pub total_time: Duration,

    /// Percentage of time per stage.
    pub percentages: HashMap<FrameStage, f64>,
}

impl FrameBreakdown {
    /// Create a new frame breakdown.
    pub fn new() -> Self {
        Self {
            stage_times: HashMap::new(),
            total_time: Duration::ZERO,
            percentages: HashMap::new(),
        }
    }

    /// Add time for a stage.
    pub fn add_stage(&mut self, stage: FrameStage, duration: Duration) {
        *self.stage_times.entry(stage).or_insert(Duration::ZERO) += duration;
        self.total_time += duration;
        self.update_percentages();
    }

    /// Update percentage calculations.
    fn update_percentages(&mut self) {
        self.percentages.clear();
        if self.total_time.as_secs_f64() > 0.0 {
            for (stage, &duration) in &self.stage_times {
                let percentage = (duration.as_secs_f64() / self.total_time.as_secs_f64()) * 100.0;
                self.percentages.insert(*stage, percentage);
            }
        }
    }

    /// Get the time for a stage.
    pub fn get_stage_time(&self, stage: FrameStage) -> Duration {
        self.stage_times
            .get(&stage)
            .copied()
            .unwrap_or(Duration::ZERO)
    }

    /// Get the percentage for a stage.
    pub fn get_stage_percentage(&self, stage: FrameStage) -> f64 {
        self.percentages.get(&stage).copied().unwrap_or(0.0)
    }

    /// Get the slowest stage.
    pub fn slowest_stage(&self) -> Option<(FrameStage, Duration)> {
        self.stage_times
            .iter()
            .max_by_key(|(_, &duration)| duration)
            .map(|(stage, &duration)| (*stage, duration))
    }

    /// Generate a report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Total Frame Time: {:?}\n\n", self.total_time));

        let mut stages: Vec<_> = self.stage_times.iter().collect();
        stages.sort_by(|a, b| b.1.cmp(a.1));

        for (stage, duration) in stages {
            let percentage = self.percentages.get(stage).copied().unwrap_or(0.0);
            report.push_str(&format!(
                "{:?}: {:?} ({:.2}%)\n",
                stage, duration, percentage
            ));
        }

        report
    }
}

impl Default for FrameBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame breakdown tracker.
#[derive(Debug)]
pub struct FrameBreakdownTracker {
    current_breakdown: FrameBreakdown,
    current_stage: Option<(FrameStage, Instant)>,
    breakdowns: Vec<FrameBreakdown>,
    max_samples: usize,
}

impl FrameBreakdownTracker {
    /// Create a new frame breakdown tracker.
    pub fn new(max_samples: usize) -> Self {
        Self {
            current_breakdown: FrameBreakdown::new(),
            current_stage: None,
            breakdowns: Vec::new(),
            max_samples,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.current_breakdown = FrameBreakdown::new();
        self.current_stage = None;
    }

    /// Begin a stage.
    pub fn begin_stage(&mut self, stage: FrameStage) {
        if let Some((prev_stage, start)) = self.current_stage.take() {
            let duration = start.elapsed();
            self.current_breakdown.add_stage(prev_stage, duration);
        }
        self.current_stage = Some((stage, Instant::now()));
    }

    /// End the current stage.
    pub fn end_stage(&mut self) {
        if let Some((stage, start)) = self.current_stage.take() {
            let duration = start.elapsed();
            self.current_breakdown.add_stage(stage, duration);
        }
    }

    /// End the current frame.
    pub fn end_frame(&mut self) {
        self.end_stage(); // End any active stage
        self.breakdowns.push(self.current_breakdown.clone());

        if self.breakdowns.len() > self.max_samples {
            self.breakdowns.remove(0);
        }
    }

    /// Get average breakdown across all frames.
    pub fn average_breakdown(&self) -> FrameBreakdown {
        if self.breakdowns.is_empty() {
            return FrameBreakdown::new();
        }

        let mut avg = FrameBreakdown::new();
        let count = self.breakdowns.len() as u32;

        for breakdown in &self.breakdowns {
            for (stage, &duration) in &breakdown.stage_times {
                let current = avg.stage_times.entry(*stage).or_insert(Duration::ZERO);
                *current += duration / count;
            }
            avg.total_time += breakdown.total_time / count;
        }

        avg.update_percentages();
        avg
    }

    /// Get the current breakdown.
    pub fn current_breakdown(&self) -> &FrameBreakdown {
        &self.current_breakdown
    }

    /// Get all breakdowns.
    pub fn breakdowns(&self) -> &[FrameBreakdown] {
        &self.breakdowns
    }
}

impl Default for FrameBreakdownTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_breakdown() {
        let mut breakdown = FrameBreakdown::new();
        breakdown.add_stage(FrameStage::Update, Duration::from_millis(10));
        breakdown.add_stage(FrameStage::Render, Duration::from_millis(15));

        assert_eq!(breakdown.total_time, Duration::from_millis(25));
        assert_eq!(
            breakdown.get_stage_time(FrameStage::Update),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn test_frame_breakdown_percentages() {
        let mut breakdown = FrameBreakdown::new();
        breakdown.add_stage(FrameStage::Update, Duration::from_millis(10));
        breakdown.add_stage(FrameStage::Render, Duration::from_millis(30));

        let update_pct = breakdown.get_stage_percentage(FrameStage::Update);
        let render_pct = breakdown.get_stage_percentage(FrameStage::Render);

        assert!((update_pct - 25.0).abs() < 0.1);
        assert!((render_pct - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_slowest_stage() {
        let mut breakdown = FrameBreakdown::new();
        breakdown.add_stage(FrameStage::Update, Duration::from_millis(10));
        breakdown.add_stage(FrameStage::Render, Duration::from_millis(30));

        let (slowest, duration) = breakdown.slowest_stage().expect("should succeed in test");
        assert_eq!(slowest, FrameStage::Render);
        assert_eq!(duration, Duration::from_millis(30));
    }

    #[test]
    fn test_frame_breakdown_tracker() {
        let mut tracker = FrameBreakdownTracker::new(10);

        tracker.begin_frame();
        tracker.begin_stage(FrameStage::Update);
        std::thread::sleep(Duration::from_millis(1));
        tracker.end_stage();
        tracker.end_frame();

        assert_eq!(tracker.breakdowns().len(), 1);
    }

    #[test]
    fn test_average_breakdown() {
        let mut tracker = FrameBreakdownTracker::new(10);

        for _ in 0..5 {
            tracker.begin_frame();
            tracker.begin_stage(FrameStage::Update);
            std::thread::sleep(Duration::from_millis(1));
            tracker.end_stage();
            tracker.end_frame();
        }

        let avg = tracker.average_breakdown();
        assert!(avg.get_stage_time(FrameStage::Update) > Duration::ZERO);
    }
}
