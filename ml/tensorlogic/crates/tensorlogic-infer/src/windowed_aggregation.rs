//! Windowed aggregation over tensor data streams.
//!
//! Supports tumbling windows (non-overlapping), sliding windows (overlapping),
//! session windows (gap-based), and count-based windows.

/// Type of window to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Non-overlapping windows of fixed duration in milliseconds.
    Tumbling { size_ms: u64 },
    /// Overlapping windows with a fixed size and advance step.
    Sliding { size_ms: u64, step_ms: u64 },
    /// Session windows separated by inactivity gaps.
    Session { gap_ms: u64 },
    /// Count-based window of `size` elements that advances by `step` elements.
    Count { size: usize, step: usize },
}

/// Aggregation function applied within each window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowAggregation {
    /// Sum of all values.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Maximum value.
    Max,
    /// Minimum value.
    Min,
    /// Number of elements (as f64).
    Count,
    /// Last (most recent) value in the window.
    LastValue,
    /// First (earliest) value in the window.
    FirstValue,
}

/// Configuration for windowed aggregation.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Window type (Tumbling, Sliding, Session, or Count).
    pub window_type: WindowType,
    /// Aggregation function to apply within each window.
    pub aggregation: WindowAggregation,
    /// Emit a partial result for windows that do not span a complete interval.
    pub emit_partial: bool,
    /// Minimum number of elements required before a window result is emitted.
    pub min_elements: usize,
}

impl WindowConfig {
    /// Create a tumbling window configuration.
    pub fn tumbling(size_ms: u64, aggregation: WindowAggregation) -> Self {
        WindowConfig {
            window_type: WindowType::Tumbling { size_ms },
            aggregation,
            emit_partial: false,
            min_elements: 1,
        }
    }

    /// Create a sliding window configuration.
    pub fn sliding(size_ms: u64, step_ms: u64, aggregation: WindowAggregation) -> Self {
        WindowConfig {
            window_type: WindowType::Sliding { size_ms, step_ms },
            aggregation,
            emit_partial: false,
            min_elements: 1,
        }
    }

    /// Create a count-based window configuration.
    pub fn count(size: usize, step: usize, aggregation: WindowAggregation) -> Self {
        WindowConfig {
            window_type: WindowType::Count { size, step },
            aggregation,
            emit_partial: false,
            min_elements: 1,
        }
    }

    /// Set whether partial (incomplete) windows should be emitted.
    pub fn with_emit_partial(mut self, emit: bool) -> Self {
        self.emit_partial = emit;
        self
    }

    /// Set the minimum number of elements required to emit a window.
    pub fn with_min_elements(mut self, min: usize) -> Self {
        self.min_elements = min;
        self
    }
}

/// The result of processing a single window.
#[derive(Debug, Clone)]
pub struct WindowResult {
    /// Window start time in milliseconds (0 for count-based windows, counts as index).
    pub start_ms: u64,
    /// Window end time in milliseconds (equal to end-index for count-based windows).
    pub end_ms: u64,
    /// Number of data elements contained in this window.
    pub element_count: usize,
    /// Aggregated value for this window.
    pub value: f64,
    /// Whether this window is complete (as opposed to a partial/trailing window).
    pub is_complete: bool,
}

/// Main windowed aggregation processor.
pub struct WindowedAggregation {
    config: WindowConfig,
}

impl WindowedAggregation {
    /// Create a new processor with the given configuration.
    pub fn new(config: WindowConfig) -> Self {
        WindowedAggregation { config }
    }

    /// Process a sequence of `(timestamp_ms, value)` pairs using tumbling windows.
    ///
    /// Each event is assigned to exactly one window `[start, start + size_ms)`.
    pub fn process_tumbling(&self, events: &[(u64, f64)]) -> Vec<WindowResult> {
        if events.is_empty() {
            return Vec::new();
        }

        let size_ms = match self.config.window_type {
            WindowType::Tumbling { size_ms } => size_ms,
            _ => return Vec::new(),
        };

        let first_ts = events[0].0;
        // Align window start to a multiple of size_ms relative to the first event.
        let window_start_base = (first_ts / size_ms) * size_ms;

        let last_ts = events.iter().map(|(t, _)| *t).max().unwrap_or(first_ts);
        // Number of complete windows needed to cover all events.
        let num_windows = ((last_ts.saturating_sub(window_start_base)) / size_ms + 1) as usize;

        let mut results = Vec::new();

        for i in 0..num_windows {
            let start = window_start_base + i as u64 * size_ms;
            let end = start + size_ms;

            let window_values: Vec<f64> = events
                .iter()
                .filter(|(t, _)| *t >= start && *t < end)
                .map(|(_, v)| *v)
                .collect();

            if window_values.len() < self.config.min_elements {
                continue;
            }

            // A tumbling window is "partial" only when emit_partial is enabled AND
            // the stream ends before the window's upper boundary. When emit_partial is
            // false we only emit windows that have at least min_elements events;
            // in that scenario every emitted window is considered complete.
            // When emit_partial is true, the last window may be incomplete if the stream
            // ended before filling the window boundary.
            let is_complete = if self.config.emit_partial {
                // Under emit_partial semantics, mark the window as partial when the
                // stream does not reach the window end.
                last_ts >= end
            } else {
                // Without emit_partial, we only emit "full-enough" windows and
                // consider all of them complete.
                true
            };

            if !is_complete && !self.config.emit_partial {
                // This branch is unreachable since is_complete=true when !emit_partial,
                // but kept for clarity.
                continue;
            }

            let value = WindowedAggregation::aggregate(&window_values, self.config.aggregation);
            results.push(WindowResult {
                start_ms: start,
                end_ms: end,
                element_count: window_values.len(),
                value,
                is_complete,
            });
        }

        results
    }

    /// Process a sequence of `(timestamp_ms, value)` pairs using sliding windows.
    ///
    /// Windows overlap: each window advances by `step_ms` and spans `size_ms`.
    pub fn process_sliding(&self, events: &[(u64, f64)]) -> Vec<WindowResult> {
        if events.is_empty() {
            return Vec::new();
        }

        let (size_ms, step_ms) = match self.config.window_type {
            WindowType::Sliding { size_ms, step_ms } => (size_ms, step_ms),
            _ => return Vec::new(),
        };

        let first_ts = events[0].0;
        let last_ts = events.iter().map(|(t, _)| *t).max().unwrap_or(first_ts);

        // Align the first window to a step_ms boundary.
        let start_base = (first_ts / step_ms) * step_ms;

        let mut results = Vec::new();
        let mut window_start = start_base;

        loop {
            if window_start > last_ts {
                break;
            }
            let window_end = window_start + size_ms;

            let window_values: Vec<f64> = events
                .iter()
                .filter(|(t, _)| *t >= window_start && *t < window_end)
                .map(|(_, v)| *v)
                .collect();

            if window_values.len() >= self.config.min_elements {
                // When emit_partial is false, every non-empty window is considered
                // complete (batch semantics). Under emit_partial mode, mark the
                // window as partial when the stream has not advanced past the window end.
                let is_complete = if self.config.emit_partial {
                    last_ts >= window_end
                } else {
                    true
                };
                if is_complete || self.config.emit_partial {
                    let value =
                        WindowedAggregation::aggregate(&window_values, self.config.aggregation);
                    results.push(WindowResult {
                        start_ms: window_start,
                        end_ms: window_end,
                        element_count: window_values.len(),
                        value,
                        is_complete,
                    });
                }
            }

            window_start += step_ms;
        }

        results
    }

    /// Process a flat slice of values using count-based windows.
    ///
    /// Window `i` covers `values[i*step .. i*step+size]`.
    pub fn process_count(&self, values: &[f64]) -> Vec<WindowResult> {
        if values.is_empty() {
            return Vec::new();
        }

        let (size, step) = match self.config.window_type {
            WindowType::Count { size, step } => (size, step),
            _ => return Vec::new(),
        };

        if step == 0 || size == 0 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let mut offset = 0usize;

        loop {
            if offset >= values.len() {
                break;
            }
            let end = (offset + size).min(values.len());
            let window_values = &values[offset..end];

            let is_complete = offset + size <= values.len();

            if window_values.len() < self.config.min_elements {
                break;
            }

            if !is_complete && !self.config.emit_partial {
                break;
            }

            let value = WindowedAggregation::aggregate(window_values, self.config.aggregation);
            results.push(WindowResult {
                start_ms: offset as u64,
                end_ms: (offset + size) as u64,
                element_count: window_values.len(),
                value,
                is_complete,
            });

            offset += step;
        }

        results
    }

    /// Process session windows: start a new session whenever the gap between
    /// consecutive events exceeds `gap_ms`.
    pub fn process_session(&self, events: &[(u64, f64)]) -> Vec<WindowResult> {
        if events.is_empty() {
            return Vec::new();
        }

        let gap_ms = match self.config.window_type {
            WindowType::Session { gap_ms } => gap_ms,
            _ => return Vec::new(),
        };

        let mut results = Vec::new();
        let mut session_start = events[0].0;
        let mut session_values: Vec<f64> = Vec::new();

        for (idx, (ts, val)) in events.iter().enumerate() {
            if idx > 0 {
                let prev_ts = events[idx - 1].0;
                let gap = ts.saturating_sub(prev_ts);
                if gap > gap_ms {
                    // Close the current session.
                    if session_values.len() >= self.config.min_elements {
                        let value = WindowedAggregation::aggregate(
                            &session_values,
                            self.config.aggregation,
                        );
                        results.push(WindowResult {
                            start_ms: session_start,
                            end_ms: events[idx - 1].0,
                            element_count: session_values.len(),
                            value,
                            is_complete: true,
                        });
                    }
                    // Begin new session.
                    session_start = *ts;
                    session_values.clear();
                }
            }
            session_values.push(*val);
        }

        // Emit the final session.
        if session_values.len() >= self.config.min_elements {
            let value = WindowedAggregation::aggregate(&session_values, self.config.aggregation);
            results.push(WindowResult {
                start_ms: session_start,
                end_ms: events.last().map(|(t, _)| *t).unwrap_or(session_start),
                element_count: session_values.len(),
                value,
                is_complete: true,
            });
        }

        results
    }

    /// Apply an aggregation function to a slice of values.
    pub fn aggregate(values: &[f64], agg: WindowAggregation) -> f64 {
        match agg {
            WindowAggregation::Sum => values.iter().copied().fold(0.0_f64, |acc, v| acc + v),
            WindowAggregation::Mean => {
                if values.is_empty() {
                    0.0
                } else {
                    let sum: f64 = values.iter().copied().fold(0.0_f64, |acc, v| acc + v);
                    sum / values.len() as f64
                }
            }
            WindowAggregation::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            WindowAggregation::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
            WindowAggregation::Count => values.len() as f64,
            WindowAggregation::LastValue => values.last().copied().unwrap_or(0.0),
            WindowAggregation::FirstValue => values.first().copied().unwrap_or(0.0),
        }
    }

    /// Process events using the window type specified in the configuration.
    pub fn process(&self, events: &[(u64, f64)]) -> Vec<WindowResult> {
        match self.config.window_type {
            WindowType::Tumbling { .. } => self.process_tumbling(events),
            WindowType::Sliding { .. } => self.process_sliding(events),
            WindowType::Session { .. } => self.process_session(events),
            WindowType::Count { .. } => {
                // For count windows, extract only the values (timestamps are ignored).
                let values: Vec<f64> = events.iter().map(|(_, v)| *v).collect();
                self.process_count(&values)
            }
        }
    }
}

/// Error type for windowed aggregation operations.
#[derive(Debug, thiserror::Error)]
pub enum WindowError {
    /// The sliding step size exceeds the window size.
    #[error("Step size {step} must be <= window size {size}")]
    StepExceedsSize { step: u64, size: u64 },
    /// The event stream was empty.
    #[error("Empty event stream")]
    EmptyStream,
    /// The window configuration is invalid.
    #[error("Invalid window configuration: {0}")]
    InvalidConfig(String),
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Tumbling window tests ----

    #[test]
    fn test_tumbling_sum_basic() {
        // 10 events at 0, 100, 200, ..., 900 ms, all value 1.0
        let events: Vec<(u64, f64)> = (0..10u64).map(|i| (i * 100, 1.0)).collect();
        let cfg = WindowConfig::tumbling(500, WindowAggregation::Sum);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_tumbling(&events);

        assert_eq!(results.len(), 2, "Expected 2 tumbling windows");
        assert_eq!(results[0].element_count, 5);
        assert!((results[0].value - 5.0).abs() < 1e-9);
        assert_eq!(results[1].element_count, 5);
        assert!((results[1].value - 5.0).abs() < 1e-9);
        assert!(results[0].is_complete);
        assert!(results[1].is_complete);
    }

    #[test]
    fn test_tumbling_non_overlapping() {
        // Each event should be counted in exactly one window.
        let events: Vec<(u64, f64)> = (0..10u64).map(|i| (i * 100, 1.0)).collect();
        let cfg = WindowConfig::tumbling(500, WindowAggregation::Count);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_tumbling(&events);

        let total_counted: usize = results.iter().map(|r| r.element_count).sum();
        assert_eq!(
            total_counted, 10,
            "Each event must appear in exactly one window"
        );
    }

    #[test]
    fn test_tumbling_empty_returns_empty() {
        let cfg = WindowConfig::tumbling(500, WindowAggregation::Sum);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_tumbling(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tumbling_with_partial_window() {
        // 7 events at 0..=600 ms (step 100), window size 500 ms, emit_partial=true
        let events: Vec<(u64, f64)> = (0..7u64).map(|i| (i * 100, 1.0)).collect();
        let cfg = WindowConfig::tumbling(500, WindowAggregation::Sum).with_emit_partial(true);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_tumbling(&events);

        assert_eq!(results.len(), 2, "Expected complete + partial window");
        // First window: events at 0,100,200,300,400 → sum=5
        assert_eq!(results[0].element_count, 5);
        assert!((results[0].value - 5.0).abs() < 1e-9);
        assert!(results[0].is_complete);
        // Second window: events at 500,600 → sum=2, incomplete
        assert_eq!(results[1].element_count, 2);
        assert!((results[1].value - 2.0).abs() < 1e-9);
        assert!(!results[1].is_complete);
    }

    // ---- Sliding window tests ----

    #[test]
    fn test_sliding_overlapping_windows() {
        // 5 events at 0,100,200,300,400; sliding(size=300, step=100, Sum)
        let events: Vec<(u64, f64)> = (0..5u64).map(|i| (i * 100, 1.0)).collect();
        let cfg = WindowConfig::sliding(300, 100, WindowAggregation::Sum);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_sliding(&events);

        // Should produce more than one window (windows overlap).
        assert!(!results.is_empty());
        // First complete window [0,300): events 0,100,200 → sum=3
        assert!(
            (results[0].value - 3.0).abs() < 1e-9,
            "First window sum should be 3.0, got {}",
            results[0].value
        );
    }

    #[test]
    fn test_sliding_step_equals_size_is_tumbling() {
        // Sliding with step == size behaves like tumbling.
        let events: Vec<(u64, f64)> = (0..10u64).map(|i| (i * 100, 1.0)).collect();
        let cfg_sliding = WindowConfig::sliding(500, 500, WindowAggregation::Sum);
        let cfg_tumbling = WindowConfig::tumbling(500, WindowAggregation::Sum);
        let wa_s = WindowedAggregation::new(cfg_sliding);
        let wa_t = WindowedAggregation::new(cfg_tumbling);
        let sliding_results = wa_s.process_sliding(&events);
        let tumbling_results = wa_t.process_tumbling(&events);
        assert_eq!(
            sliding_results.len(),
            tumbling_results.len(),
            "Sliding with step==size should match tumbling"
        );
        for (s, t) in sliding_results.iter().zip(tumbling_results.iter()) {
            assert!((s.value - t.value).abs() < 1e-9);
        }
    }

    #[test]
    fn test_sliding_mean() {
        // 3 events at 0,1,2 ms, values 1.0,2.0,3.0
        // Sliding(size=3ms, step=1ms, Mean), emit_partial=true
        let events = vec![(0u64, 1.0_f64), (1, 2.0), (2, 3.0)];
        let cfg = WindowConfig::sliding(3, 1, WindowAggregation::Mean).with_emit_partial(true);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_sliding(&events);
        // First window [0,3): mean of 1,2,3 = 2.0
        assert!(!results.is_empty());
        assert!(
            (results[0].value - 2.0).abs() < 1e-9,
            "Mean should be 2.0, got {}",
            results[0].value
        );
    }

    // ---- Count window tests ----

    #[test]
    fn test_count_window_basic() {
        // 9 values [1..=9], count(3, 3, Sum) → 3 non-overlapping windows
        let values: Vec<f64> = (1..=9u32).map(|v| v as f64).collect();
        let cfg = WindowConfig::count(3, 3, WindowAggregation::Sum);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_count(&values);

        assert_eq!(results.len(), 3);
        assert!((results[0].value - 6.0).abs() < 1e-9); // 1+2+3
        assert!((results[1].value - 15.0).abs() < 1e-9); // 4+5+6
        assert!((results[2].value - 24.0).abs() < 1e-9); // 7+8+9
    }

    #[test]
    fn test_count_window_sliding() {
        // 8 values [1..=8], count(4, 2, Sum)
        // Windows: [1,2,3,4]→10, [3,4,5,6]→18, [5,6,7,8]→26
        let values: Vec<f64> = (1..=8u32).map(|v| v as f64).collect();
        let cfg = WindowConfig::count(4, 2, WindowAggregation::Sum);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_count(&values);

        assert_eq!(results.len(), 3);
        assert!((results[0].value - 10.0).abs() < 1e-9);
        assert!((results[1].value - 18.0).abs() < 1e-9);
        assert!((results[2].value - 26.0).abs() < 1e-9);
    }

    #[test]
    fn test_count_window_min() {
        // 6 values [3,1,4,1,5,9], count(3,3,Min)
        let values = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0];
        let cfg = WindowConfig::count(3, 3, WindowAggregation::Min);
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_count(&values);

        assert_eq!(results.len(), 2);
        assert!((results[0].value - 1.0).abs() < 1e-9); // min(3,1,4)=1
        assert!((results[1].value - 1.0).abs() < 1e-9); // min(1,5,9)=1
    }

    // ---- Aggregate + session tests ----

    #[test]
    fn test_aggregate_all_strategies() {
        let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::Sum) - 15.0).abs() < 1e-9
        );
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::Mean) - 3.0).abs() < 1e-9
        );
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::Max) - 5.0).abs() < 1e-9
        );
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::Min) - 1.0).abs() < 1e-9
        );
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::Count) - 5.0).abs() < 1e-9
        );
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::FirstValue) - 1.0).abs()
                < 1e-9
        );
        assert!(
            (WindowedAggregation::aggregate(&values, WindowAggregation::LastValue) - 5.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_session_window_gap_detection() {
        // Events: (0,1),(100,2),(200,3) — gap < 500ms, then (1500,4),(1600,5) — gap > 500ms
        let events = vec![
            (0u64, 1.0_f64),
            (100, 2.0),
            (200, 3.0),
            (1500, 4.0),
            (1600, 5.0),
        ];
        let cfg = WindowConfig {
            window_type: WindowType::Session { gap_ms: 500 },
            aggregation: WindowAggregation::Sum,
            emit_partial: false,
            min_elements: 1,
        };
        let wa = WindowedAggregation::new(cfg);
        let results = wa.process_session(&events);

        assert_eq!(results.len(), 2, "Expected 2 sessions");
        assert_eq!(
            results[0].element_count, 3,
            "First session should have 3 elements"
        );
        assert_eq!(
            results[1].element_count, 2,
            "Second session should have 2 elements"
        );
        assert!(results[0].is_complete);
        assert!(results[1].is_complete);
    }
}
