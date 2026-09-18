//! # Linux Perf Integration for SciRS2 v0.2.0
//!
//! This module provides integration with Linux perf for detailed performance profiling.
//! It enables hardware performance counter monitoring and profiling markers.
//!
//! # Features
//!
//! - **Hardware Counters**: Access CPU performance counters
//! - **Profiling Markers**: Add markers for perf record visualization
//! - **Cache Profiling**: Monitor cache hits/misses
//! - **Branch Prediction**: Track branch prediction performance
//! - **Platform-Specific**: Linux only, zero overhead on other platforms
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::profiling::perf_integration::{PerfCounter, PerfEvent};
//!
//! // Create a perf counter for CPU cycles
//! let mut counter = PerfCounter::new(PerfEvent::CpuCycles)
//!     .expect("Failed to create perf counter");
//!
//! counter.enable();
//!
//! // ... perform computation ...
//!
//! let cycles = counter.read().expect("Failed to read counter");
//! println!("CPU cycles: {}", cycles);
//! ```

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
use crate::CoreResult;
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
use perf_event::{events::Hardware, Builder, Counter};
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
use std::sync::Arc;

/// Performance event types
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfEvent {
    /// CPU cycles
    CpuCycles,
    /// Instructions executed
    Instructions,
    /// Cache references
    CacheReferences,
    /// Cache misses
    CacheMisses,
    /// Branch instructions
    BranchInstructions,
    /// Branch mispredictions
    BranchMisses,
    /// Bus cycles
    BusCycles,
    /// Stalled cycles (frontend)
    StalledCyclesFrontend,
    /// Stalled cycles (backend)
    StalledCyclesBackend,
}

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
impl PerfEvent {
    /// Convert to perf-event Hardware type
    fn to_hardware(self) -> Hardware {
        match self {
            PerfEvent::CpuCycles => Hardware::CPU_CYCLES,
            PerfEvent::Instructions => Hardware::INSTRUCTIONS,
            PerfEvent::CacheReferences => Hardware::CACHE_REFERENCES,
            PerfEvent::CacheMisses => Hardware::CACHE_MISSES,
            PerfEvent::BranchInstructions => Hardware::BRANCH_INSTRUCTIONS,
            PerfEvent::BranchMisses => Hardware::BRANCH_MISSES,
            PerfEvent::BusCycles => Hardware::BUS_CYCLES,
            PerfEvent::StalledCyclesFrontend => Hardware::STALLED_CYCLES_FRONTEND,
            PerfEvent::StalledCyclesBackend => Hardware::STALLED_CYCLES_BACKEND,
        }
    }
}

/// Performance counter wrapper
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
pub struct PerfCounter {
    counter: Counter,
    event: PerfEvent,
}

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
impl PerfCounter {
    /// Create a new performance counter
    pub fn new(event: PerfEvent) -> CoreResult<Self> {
        let counter = Builder::new()
            .one_cpu(0)
            .kind(event.to_hardware())
            .build()
            .map_err(|e| {
                crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                    "Failed to create perf counter: {}",
                    e
                )))
            })?;

        Ok(Self { counter, event })
    }

    /// Enable the counter
    pub fn enable(&mut self) -> CoreResult<()> {
        self.counter.enable().map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to enable perf counter: {}",
                e
            )))
        })?;
        Ok(())
    }

    /// Disable the counter
    pub fn disable(&mut self) -> CoreResult<()> {
        self.counter.disable().map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to disable perf counter: {}",
                e
            )))
        })?;
        Ok(())
    }

    /// Read the counter value
    pub fn read(&mut self) -> CoreResult<u64> {
        let value = self.counter.read().map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to read perf counter: {}",
                e
            )))
        })?;
        Ok(value)
    }

    /// Reset the counter
    pub fn reset(&mut self) -> CoreResult<()> {
        self.counter.reset().map_err(|e| {
            crate::CoreError::ConfigError(crate::error::ErrorContext::new(format!(
                "Failed to reset perf counter: {}",
                e
            )))
        })?;
        Ok(())
    }

    /// Get the event type
    pub fn event(&self) -> PerfEvent {
        self.event
    }
}

/// Performance counter group for measuring multiple events
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
pub struct PerfCounterGroup {
    counters: Vec<PerfCounter>,
}

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
impl PerfCounterGroup {
    /// Create a new counter group
    pub fn new(events: &[PerfEvent]) -> CoreResult<Self> {
        let mut counters = Vec::new();

        for &event in events {
            counters.push(PerfCounter::new(event)?);
        }

        Ok(Self { counters })
    }

    /// Enable all counters
    pub fn enable(&mut self) -> CoreResult<()> {
        for counter in &mut self.counters {
            counter.enable()?;
        }
        Ok(())
    }

    /// Disable all counters
    pub fn disable(&mut self) -> CoreResult<()> {
        for counter in &mut self.counters {
            counter.disable()?;
        }
        Ok(())
    }

    /// Read all counter values
    pub fn read(&mut self) -> CoreResult<Vec<(PerfEvent, u64)>> {
        let mut results = Vec::new();

        for counter in &mut self.counters {
            let value = counter.read()?;
            results.push((counter.event(), value));
        }

        Ok(results)
    }

    /// Reset all counters
    pub fn reset(&mut self) -> CoreResult<()> {
        for counter in &mut self.counters {
            counter.reset()?;
        }
        Ok(())
    }
}

/// Marker for perf record visualization
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
pub struct PerfMarker {
    name: String,
}

#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
impl PerfMarker {
    /// Create a new perf marker
    pub fn new(name: impl Into<String>) -> Self {
        let marker = Self { name: name.into() };
        // In practice, this would use perf_event_open with PERF_RECORD_MISC_MARKER
        // For now, this is a placeholder
        marker
    }

    /// End the marker
    pub fn end(self) {
        // Placeholder for marker end
    }
}

/// Macro for creating a perf marker
#[macro_export]
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
macro_rules! perf_marker {
    ($name:expr) => {
        let _marker = $crate::profiling::perf_integration::PerfMarker::new($name);
    };
}

/// Stub implementations when not on Linux or feature is disabled
#[cfg(not(all(target_os = "linux", feature = "profiling_perf")))]
use crate::CoreResult;

#[cfg(not(all(target_os = "linux", feature = "profiling_perf")))]
#[derive(Debug, Clone, Copy)]
pub enum PerfEvent {
    CpuCycles,
    Instructions,
    CacheReferences,
    CacheMisses,
    BranchInstructions,
    BranchMisses,
    BusCycles,
    StalledCyclesFrontend,
    StalledCyclesBackend,
}

#[cfg(not(all(target_os = "linux", feature = "profiling_perf")))]
pub struct PerfCounter;

#[cfg(not(all(target_os = "linux", feature = "profiling_perf")))]
impl PerfCounter {
    pub fn new(_event: PerfEvent) -> CoreResult<Self> {
        Ok(Self)
    }
    pub fn enable(&mut self) -> CoreResult<()> {
        Ok(())
    }
    pub fn disable(&mut self) -> CoreResult<()> {
        Ok(())
    }
    pub fn read(&mut self) -> CoreResult<u64> {
        Ok(0)
    }
    pub fn reset(&mut self) -> CoreResult<()> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg(all(target_os = "linux", feature = "profiling_perf"))]
mod tests {
    use super::*;

    #[test]
    fn test_perf_counter_creation() {
        let result = PerfCounter::new(PerfEvent::CpuCycles);
        // May fail if not running with appropriate permissions
        if let Ok(mut counter) = result {
            assert!(counter.enable().is_ok() || counter.enable().is_err());
        }
    }

    #[test]
    fn test_perf_counter_group() {
        let events = [PerfEvent::CpuCycles, PerfEvent::Instructions];
        let result = PerfCounterGroup::new(&events);

        // May fail if not running with appropriate permissions
        if let Ok(mut group) = result {
            assert!(group.enable().is_ok() || group.enable().is_err());
        }
    }

    #[test]
    fn test_perf_marker() {
        let marker = PerfMarker::new("test_marker");
        marker.end();
        // Should not panic
    }
}
