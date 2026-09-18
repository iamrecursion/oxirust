//! # Instrumentation Framework for SciRS2 v0.2.0
//!
//! This module provides a comprehensive instrumentation framework for adding
//! observability to SciRS2 operations with minimal code changes.
//!
//! # Features
//!
//! - **Automatic Instrumentation**: Attribute macros for automatic function instrumentation
//! - **Custom Metrics**: Easy-to-use metrics collection
//! - **Performance Zones**: Mark critical sections for profiling
//! - **Event Recording**: Record significant events with context
//! - **Zero Overhead**: Compile-time elimination when disabled
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::{instrument, profiling::instrumentation::record_event};
//!
//! fn compute_fft(data: &[f64]) -> Vec<f64> {
//!     instrument!("compute_fft");
//!     record_event("fft_started", &[("size", &data.len())]);
//!     // ... computation ...
//!     record_event("fft_completed", &[]);
//!     vec![]
//! }
//! ```

#[cfg(feature = "instrumentation")]
use crate::CoreResult;
#[cfg(feature = "instrumentation")]
use std::collections::HashMap;
#[cfg(feature = "instrumentation")]
use std::sync::Arc;
#[cfg(feature = "instrumentation")]
use std::time::{Duration, Instant};

/// Event recorder for instrumentation
#[cfg(feature = "instrumentation")]
#[derive(Clone)]
pub struct InstrumentationEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: Instant,
    /// Event attributes
    pub attributes: HashMap<String, String>,
    /// Event duration (if applicable)
    pub duration: Option<Duration>,
}

#[cfg(feature = "instrumentation")]
impl InstrumentationEvent {
    /// Create a new event
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: Instant::now(),
            attributes: HashMap::new(),
            duration: None,
        }
    }

    /// Add an attribute to the event
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Set the duration of the event
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }
}

/// Global event recorder
#[cfg(feature = "instrumentation")]
static EVENT_RECORDER: once_cell::sync::Lazy<Arc<parking_lot::Mutex<Vec<InstrumentationEvent>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(parking_lot::Mutex::new(Vec::new())));

/// Record an instrumentation event
#[cfg(feature = "instrumentation")]
pub fn record_event(name: &str, attributes: &[(&str, &dyn std::fmt::Display)]) {
    let mut event = InstrumentationEvent::new(name);

    for (key, value) in attributes {
        event = event.with_attribute(*key, value.to_string());
    }

    EVENT_RECORDER.lock().push(event);
}

/// Record an event with duration
#[cfg(feature = "instrumentation")]
pub fn record_event_with_duration(
    name: &str,
    duration: Duration,
    attributes: &[(&str, &dyn std::fmt::Display)],
) {
    let mut event = InstrumentationEvent::new(name).with_duration(duration);

    for (key, value) in attributes {
        event = event.with_attribute(*key, value.to_string());
    }

    EVENT_RECORDER.lock().push(event);
}

/// Get all recorded events
#[cfg(feature = "instrumentation")]
pub fn get_events() -> Vec<InstrumentationEvent> {
    EVENT_RECORDER.lock().clone()
}

/// Clear all recorded events
#[cfg(feature = "instrumentation")]
pub fn clear_events() {
    EVENT_RECORDER.lock().clear();
}

/// Instrumentation scope for automatic timing
#[cfg(feature = "instrumentation")]
pub struct InstrumentationScope {
    name: String,
    start: Instant,
    attributes: HashMap<String, String>,
}

#[cfg(feature = "instrumentation")]
impl InstrumentationScope {
    /// Create a new instrumentation scope
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute to the scope
    pub fn add_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// End the scope manually and return duration
    pub fn end(self) -> Duration {
        let duration = self.start.elapsed();

        // Record the event
        let mut event = InstrumentationEvent::new(&self.name).with_duration(duration);
        for (key, value) in self.attributes.clone() {
            event = event.with_attribute(key, value);
        }

        EVENT_RECORDER.lock().push(event);

        duration
    }
}

#[cfg(feature = "instrumentation")]
impl Drop for InstrumentationScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();

        // Record the event
        let mut event = InstrumentationEvent::new(&self.name).with_duration(duration);
        for (key, value) in self.attributes.clone() {
            event = event.with_attribute(key, value);
        }

        EVENT_RECORDER.lock().push(event);
    }
}

/// Macro for instrumenting a function
#[macro_export]
#[cfg(feature = "instrumentation")]
macro_rules! instrument {
    ($name:expr) => {
        let _scope = $crate::profiling::instrumentation::InstrumentationScope::new($name);
    };
    ($name:expr, $($attr_key:expr => $attr_value:expr),* $(,)?) => {
        let mut _scope = $crate::profiling::instrumentation::InstrumentationScope::new($name);
        $(
            _scope.add_attribute($attr_key, $attr_value.to_string());
        )*
    };
}

/// Macro for instrumenting a code block
#[macro_export]
#[cfg(feature = "instrumentation")]
macro_rules! instrument_block {
    ($name:expr, $body:block) => {{
        let _scope = $crate::profiling::instrumentation::InstrumentationScope::new($name);
        $body
    }};
}

/// Performance counter for tracking metrics
#[cfg(feature = "instrumentation")]
pub struct PerformanceCounter {
    name: String,
    count: parking_lot::Mutex<u64>,
    total_duration: parking_lot::Mutex<Duration>,
}

#[cfg(feature = "instrumentation")]
impl PerformanceCounter {
    /// Create a new performance counter
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: parking_lot::Mutex::new(0),
            total_duration: parking_lot::Mutex::new(Duration::ZERO),
        }
    }

    /// Increment the counter
    pub fn increment(&self) {
        *self.count.lock() += 1;
    }

    /// Add duration to the counter
    pub fn add_duration(&self, duration: Duration) {
        *self.total_duration.lock() += duration;
        self.increment();
    }

    /// Get the current count
    pub fn count(&self) -> u64 {
        *self.count.lock()
    }

    /// Get the total duration
    pub fn total_duration(&self) -> Duration {
        *self.total_duration.lock()
    }

    /// Get the average duration
    pub fn average_duration(&self) -> Option<Duration> {
        let count = self.count();
        if count == 0 {
            None
        } else {
            Some(self.total_duration() / count as u32)
        }
    }

    /// Reset the counter
    pub fn reset(&self) {
        *self.count.lock() = 0;
        *self.total_duration.lock() = Duration::ZERO;
    }

    /// Get a summary of the counter
    pub fn summary(&self) -> String {
        let count = self.count();
        let total = self.total_duration();
        let avg = self.average_duration();

        format!(
            "{}: count={}, total={:?}, avg={:?}",
            self.name,
            count,
            total,
            avg.unwrap_or(Duration::ZERO)
        )
    }
}

/// Global performance counters registry
#[cfg(feature = "instrumentation")]
static COUNTERS: once_cell::sync::Lazy<
    Arc<parking_lot::Mutex<HashMap<String, Arc<PerformanceCounter>>>>,
> = once_cell::sync::Lazy::new(|| Arc::new(parking_lot::Mutex::new(HashMap::new())));

/// Get or create a performance counter
#[cfg(feature = "instrumentation")]
pub fn get_counter(name: &str) -> Arc<PerformanceCounter> {
    let mut counters = COUNTERS.lock();
    counters
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(PerformanceCounter::new(name)))
        .clone()
}

/// Get all performance counters
#[cfg(feature = "instrumentation")]
pub fn get_all_counters() -> HashMap<String, Arc<PerformanceCounter>> {
    COUNTERS.lock().clone()
}

/// Print summary of all counters
#[cfg(feature = "instrumentation")]
pub fn print_counter_summary() {
    let counters = get_all_counters();
    println!("Performance Counter Summary:");
    println!("===========================");

    for (name, counter) in counters {
        println!("{}", counter.summary());
    }
}

/// Stub implementations when instrumentation feature is disabled
#[cfg(not(feature = "instrumentation"))]
#[allow(unused_imports)]
use crate::CoreResult;

#[cfg(not(feature = "instrumentation"))]
pub fn record_event(_name: &str, _attributes: &[(&str, &dyn std::fmt::Display)]) {}

#[cfg(not(feature = "instrumentation"))]
pub fn clear_events() {}

#[cfg(test)]
#[cfg(feature = "instrumentation")]
mod tests {
    use super::*;

    #[test]
    fn test_event_recording() {
        clear_events();

        record_event("test_event", &[("key", &"value")]);

        let events = get_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_event");
        assert_eq!(events[0].attributes.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_instrumentation_scope() {
        clear_events();

        {
            let _scope = InstrumentationScope::new("test_scope");
            std::thread::sleep(Duration::from_millis(10));
        }

        let events = get_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_scope");
        assert!(events[0].duration.is_some());
    }

    #[test]
    fn test_performance_counter() {
        let counter = PerformanceCounter::new("test_counter");

        counter.increment();
        counter.add_duration(Duration::from_millis(100));

        assert_eq!(counter.count(), 2);
        assert!(counter.total_duration() >= Duration::from_millis(100));

        let avg = counter.average_duration();
        assert!(avg.is_some());
    }

    #[test]
    fn test_global_counter() {
        let counter1 = get_counter("global_test");
        let counter2 = get_counter("global_test");

        counter1.increment();
        assert_eq!(counter2.count(), 1);
    }

    #[test]
    fn test_counter_summary() {
        let counter = PerformanceCounter::new("summary_test");
        counter.add_duration(Duration::from_millis(50));
        counter.add_duration(Duration::from_millis(150));

        let summary = counter.summary();
        assert!(summary.contains("summary_test"));
        assert!(summary.contains("count=2"));
    }
}
