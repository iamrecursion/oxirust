//! GPU timeline analysis.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// GPU event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuEventType {
    /// Command buffer submission.
    Submit,

    /// Render pass begin.
    RenderPassBegin,

    /// Render pass end.
    RenderPassEnd,

    /// Compute pass begin.
    ComputePassBegin,

    /// Compute pass end.
    ComputePassEnd,

    /// Memory transfer.
    Transfer,

    /// Pipeline barrier.
    Barrier,

    /// Present.
    Present,
}

/// A GPU event in the timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEvent {
    /// Event type.
    pub event_type: GpuEventType,

    /// Event name/label.
    pub name: String,

    /// Start timestamp.
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,

    /// Duration.
    pub duration: Duration,

    /// Queue index.
    pub queue: u32,

    /// Command buffer ID.
    pub command_buffer: u64,
}

impl GpuEvent {
    /// Create a new GPU event.
    pub fn new(event_type: GpuEventType, name: String, queue: u32, command_buffer: u64) -> Self {
        Self {
            event_type,
            name,
            start_time: Instant::now(),
            duration: Duration::ZERO,
            queue,
            command_buffer,
        }
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Get the end time.
    pub fn end_time(&self) -> Instant {
        self.start_time + self.duration
    }
}

/// GPU timeline for tracking events.
#[derive(Debug)]
pub struct GpuTimeline {
    events: Vec<GpuEvent>,
    start_time: Instant,
}

impl GpuTimeline {
    /// Create a new GPU timeline.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Add an event to the timeline.
    pub fn add_event(&mut self, event: GpuEvent) {
        self.events.push(event);
    }

    /// Begin a new event.
    pub fn begin_event(
        &mut self,
        event_type: GpuEventType,
        name: String,
        queue: u32,
        command_buffer: u64,
    ) -> usize {
        let event = GpuEvent::new(event_type, name, queue, command_buffer);
        self.events.push(event);
        self.events.len() - 1
    }

    /// End an event.
    pub fn end_event(&mut self, index: usize) {
        if let Some(event) = self.events.get_mut(index) {
            event.duration = event.start_time.elapsed();
        }
    }

    /// Get all events.
    pub fn events(&self) -> &[GpuEvent] {
        &self.events
    }

    /// Get events by type.
    pub fn events_by_type(&self, event_type: GpuEventType) -> Vec<&GpuEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Get total duration.
    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get average event duration by type.
    pub fn avg_duration(&self, event_type: GpuEventType) -> Duration {
        let events: Vec<_> = self.events_by_type(event_type);
        if events.is_empty() {
            return Duration::ZERO;
        }

        let total: Duration = events.iter().map(|e| e.duration).sum();
        total / events.len() as u32
    }

    /// Clear the timeline.
    pub fn clear(&mut self) {
        self.events.clear();
        self.start_time = Instant::now();
    }

    /// Get event count.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Generate a summary report.
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Timeline Duration: {:?}\n", self.total_duration()));
        report.push_str(&format!("Total Events: {}\n\n", self.event_count()));

        let event_types = [
            GpuEventType::Submit,
            GpuEventType::RenderPassBegin,
            GpuEventType::ComputePassBegin,
            GpuEventType::Transfer,
            GpuEventType::Present,
        ];

        for event_type in &event_types {
            let count = self.events_by_type(*event_type).len();
            let avg_duration = self.avg_duration(*event_type);
            if count > 0 {
                report.push_str(&format!(
                    "{:?}: {} events, avg {:?}\n",
                    event_type, count, avg_duration
                ));
            }
        }

        report
    }
}

impl Default for GpuTimeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_event() {
        let event = GpuEvent::new(GpuEventType::RenderPassBegin, "main_pass".to_string(), 0, 1);
        assert_eq!(event.event_type, GpuEventType::RenderPassBegin);
        assert_eq!(event.name, "main_pass");
        assert_eq!(event.queue, 0);
    }

    #[test]
    fn test_gpu_timeline() {
        let mut timeline = GpuTimeline::new();
        assert_eq!(timeline.event_count(), 0);

        let event = GpuEvent::new(GpuEventType::Submit, "cmd1".to_string(), 0, 1)
            .with_duration(Duration::from_millis(1));

        timeline.add_event(event);
        assert_eq!(timeline.event_count(), 1);
    }

    #[test]
    fn test_begin_end_event() {
        let mut timeline = GpuTimeline::new();

        let idx = timeline.begin_event(GpuEventType::RenderPassBegin, "pass".to_string(), 0, 1);

        std::thread::sleep(Duration::from_millis(1));
        timeline.end_event(idx);

        assert_eq!(timeline.event_count(), 1);
        assert!(timeline.events()[0].duration > Duration::ZERO);
    }

    #[test]
    fn test_events_by_type() {
        let mut timeline = GpuTimeline::new();

        timeline.add_event(GpuEvent::new(
            GpuEventType::Submit,
            "cmd1".to_string(),
            0,
            1,
        ));
        timeline.add_event(GpuEvent::new(
            GpuEventType::Submit,
            "cmd2".to_string(),
            0,
            2,
        ));
        timeline.add_event(GpuEvent::new(
            GpuEventType::Transfer,
            "transfer".to_string(),
            1,
            3,
        ));

        let submits = timeline.events_by_type(GpuEventType::Submit);
        assert_eq!(submits.len(), 2);

        let transfers = timeline.events_by_type(GpuEventType::Transfer);
        assert_eq!(transfers.len(), 1);
    }

    #[test]
    fn test_avg_duration() {
        let mut timeline = GpuTimeline::new();

        timeline.add_event(
            GpuEvent::new(GpuEventType::Submit, "1".to_string(), 0, 1)
                .with_duration(Duration::from_millis(10)),
        );
        timeline.add_event(
            GpuEvent::new(GpuEventType::Submit, "2".to_string(), 0, 2)
                .with_duration(Duration::from_millis(20)),
        );

        let avg = timeline.avg_duration(GpuEventType::Submit);
        assert_eq!(avg, Duration::from_millis(15));
    }
}
