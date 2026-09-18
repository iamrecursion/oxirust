#![allow(dead_code)]
//! Genlock master clock management for multi-camera synchronization.
//!
//! Provides a virtual genlock master that distributes a reference clock to
//! multiple camera sources, tracks per-source phase offsets, and detects
//! lock / drift / free-run status. This is the "control-plane" complement
//! to the per-frame genlock simulation in `sync::genlock`.

use std::collections::HashMap;
use std::fmt;

/// Reference signal standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RefStandard {
    /// Black-burst analog reference (SD).
    BlackBurst,
    /// Tri-level sync (HD).
    TriLevel,
    /// PTP (IEEE 1588 / SMPTE ST 2059).
    Ptp,
    /// Internal free-run oscillator.
    Internal,
}

impl fmt::Display for RefStandard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlackBurst => write!(f, "BlackBurst"),
            Self::TriLevel => write!(f, "TriLevel"),
            Self::Ptp => write!(f, "PTP"),
            Self::Internal => write!(f, "Internal"),
        }
    }
}

/// Lock status of a source relative to the master reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockStatus {
    /// Source is phase-locked to the master.
    Locked,
    /// Source is acquiring lock.
    Locking,
    /// Source has drifted beyond tolerance.
    Drifted,
    /// Source is free-running (no reference).
    FreeRun,
    /// No signal detected from source.
    NoSignal,
}

impl fmt::Display for LockStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Locked => write!(f, "LOCKED"),
            Self::Locking => write!(f, "LOCKING"),
            Self::Drifted => write!(f, "DRIFTED"),
            Self::FreeRun => write!(f, "FREE-RUN"),
            Self::NoSignal => write!(f, "NO-SIGNAL"),
        }
    }
}

/// Phase offset measured in sub-frame units (nanoseconds).
#[derive(Debug, Clone, Copy)]
pub struct PhaseOffset {
    /// Offset in nanoseconds (positive = source leads master).
    pub nanoseconds: i64,
}

impl PhaseOffset {
    /// Create from nanoseconds.
    #[must_use]
    pub fn from_ns(ns: i64) -> Self {
        Self { nanoseconds: ns }
    }

    /// Create zero offset.
    #[must_use]
    pub fn zero() -> Self {
        Self { nanoseconds: 0 }
    }

    /// Absolute value of the offset.
    #[must_use]
    pub fn abs_ns(&self) -> u64 {
        self.nanoseconds.unsigned_abs()
    }

    /// Convert to microseconds (truncated).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn as_micros_f64(&self) -> f64 {
        self.nanoseconds as f64 / 1_000.0
    }
}

impl fmt::Display for PhaseOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nanoseconds >= 0 {
            write!(f, "+{}ns", self.nanoseconds)
        } else {
            write!(f, "{}ns", self.nanoseconds)
        }
    }
}

/// Per-source tracking state held by the genlock master.
#[derive(Debug, Clone)]
pub struct SourceState {
    /// Human-readable source name.
    pub name: String,
    /// Current lock status.
    pub status: LockStatus,
    /// Latest measured phase offset.
    pub offset: PhaseOffset,
    /// Number of consecutive locked samples.
    pub locked_count: u64,
    /// Number of consecutive drifted samples.
    pub drifted_count: u64,
}

impl SourceState {
    /// Create an initial state for a named source.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            status: LockStatus::NoSignal,
            offset: PhaseOffset::zero(),
            locked_count: 0,
            drifted_count: 0,
        }
    }
}

/// Configuration for the genlock master.
#[derive(Debug, Clone)]
pub struct GenlockConfig {
    /// Reference standard to distribute.
    pub standard: RefStandard,
    /// Lock tolerance in nanoseconds. Offsets below this are "locked".
    pub lock_tolerance_ns: u64,
    /// Drift threshold in nanoseconds. Offsets above this trigger "drifted".
    pub drift_threshold_ns: u64,
    /// Number of consecutive locked samples to transition from Locking -> Locked.
    pub lock_acquire_samples: u64,
    /// Frame rate numerator (e.g., 30000 for 29.97).
    pub frame_rate_num: u32,
    /// Frame rate denominator (e.g., 1001 for 29.97).
    pub frame_rate_den: u32,
}

impl Default for GenlockConfig {
    fn default() -> Self {
        Self {
            standard: RefStandard::TriLevel,
            lock_tolerance_ns: 500,
            drift_threshold_ns: 5_000,
            lock_acquire_samples: 10,
            frame_rate_num: 25,
            frame_rate_den: 1,
        }
    }
}

impl GenlockConfig {
    /// Frame interval in nanoseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frame_interval_ns(&self) -> u64 {
        let interval =
            f64::from(self.frame_rate_den) / f64::from(self.frame_rate_num) * 1_000_000_000.0;
        interval as u64
    }
}

/// The genlock master that tracks multiple sources.
#[derive(Debug)]
pub struct GenlockMaster {
    /// Configuration.
    config: GenlockConfig,
    /// Per-source states keyed by source id.
    sources: HashMap<u32, SourceState>,
    /// Next source id to assign.
    next_id: u32,
    /// Master clock tick counter.
    tick: u64,
}

impl GenlockMaster {
    /// Create a new genlock master with the given configuration.
    #[must_use]
    pub fn new(config: GenlockConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            next_id: 0,
            tick: 0,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(GenlockConfig::default())
    }

    /// Register a new source. Returns the assigned source id.
    pub fn add_source(&mut self, name: &str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.sources.insert(id, SourceState::new(name));
        id
    }

    /// Remove a source by id.
    pub fn remove_source(&mut self, id: u32) -> bool {
        self.sources.remove(&id).is_some()
    }

    /// Number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get the lock status of a source.
    #[must_use]
    pub fn status(&self, id: u32) -> Option<LockStatus> {
        self.sources.get(&id).map(|s| s.status)
    }

    /// Get the current phase offset of a source.
    #[must_use]
    pub fn offset(&self, id: u32) -> Option<PhaseOffset> {
        self.sources.get(&id).map(|s| s.offset)
    }

    /// Report a phase measurement from a source.
    pub fn report_phase(&mut self, id: u32, offset: PhaseOffset) {
        if let Some(state) = self.sources.get_mut(&id) {
            state.offset = offset;
            let abs = offset.abs_ns();

            if abs <= self.config.lock_tolerance_ns {
                state.drifted_count = 0;
                state.locked_count += 1;
                if state.locked_count >= self.config.lock_acquire_samples {
                    state.status = LockStatus::Locked;
                } else {
                    state.status = LockStatus::Locking;
                }
            } else if abs > self.config.drift_threshold_ns {
                state.locked_count = 0;
                state.drifted_count += 1;
                state.status = LockStatus::Drifted;
            } else {
                // Between tolerance and drift threshold
                state.locked_count = 0;
                state.drifted_count = 0;
                state.status = LockStatus::Locking;
            }
        }
    }

    /// Mark a source as having no signal.
    pub fn report_no_signal(&mut self, id: u32) {
        if let Some(state) = self.sources.get_mut(&id) {
            state.status = LockStatus::NoSignal;
            state.locked_count = 0;
            state.drifted_count = 0;
        }
    }

    /// Advance the master tick counter by one.
    pub fn tick(&mut self) {
        self.tick += 1;
    }

    /// Current tick.
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Check whether all sources are locked.
    #[must_use]
    pub fn all_locked(&self) -> bool {
        !self.sources.is_empty()
            && self
                .sources
                .values()
                .all(|s| s.status == LockStatus::Locked)
    }

    /// Return sources that are not locked.
    #[must_use]
    pub fn unlocked_sources(&self) -> Vec<(u32, &SourceState)> {
        self.sources
            .iter()
            .filter(|(_, s)| s.status != LockStatus::Locked)
            .map(|(id, s)| (*id, s))
            .collect()
    }

    /// Reference standard in use.
    #[must_use]
    pub fn standard(&self) -> RefStandard {
        self.config.standard
    }

    /// Generate a summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "GenlockMaster [{}] tick={} sources={}",
            self.config.standard,
            self.tick,
            self.sources.len()
        )];
        let mut ids: Vec<u32> = self.sources.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let s = &self.sources[&id];
            lines.push(format!(
                "  src{}: {} status={} offset={}",
                id, s.name, s.status, s.offset
            ));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_offset_zero() {
        let p = PhaseOffset::zero();
        assert_eq!(p.nanoseconds, 0);
        assert_eq!(p.abs_ns(), 0);
    }

    #[test]
    fn test_phase_offset_positive() {
        let p = PhaseOffset::from_ns(1500);
        assert_eq!(p.abs_ns(), 1500);
        assert!((p.as_micros_f64() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_phase_offset_negative() {
        let p = PhaseOffset::from_ns(-800);
        assert_eq!(p.abs_ns(), 800);
        assert_eq!(format!("{p}"), "-800ns");
    }

    #[test]
    fn test_phase_offset_display_positive() {
        let p = PhaseOffset::from_ns(100);
        assert_eq!(format!("{p}"), "+100ns");
    }

    #[test]
    fn test_genlock_config_defaults() {
        let cfg = GenlockConfig::default();
        assert_eq!(cfg.standard, RefStandard::TriLevel);
        assert_eq!(cfg.lock_tolerance_ns, 500);
    }

    #[test]
    fn test_frame_interval_25fps() {
        let cfg = GenlockConfig::default(); // 25/1
        let interval = cfg.frame_interval_ns();
        assert_eq!(interval, 40_000_000); // 40ms
    }

    #[test]
    fn test_add_source() {
        let mut master = GenlockMaster::with_defaults();
        let id = master.add_source("Cam1");
        assert_eq!(id, 0);
        assert_eq!(master.source_count(), 1);
        assert_eq!(master.status(id), Some(LockStatus::NoSignal));
    }

    #[test]
    fn test_remove_source() {
        let mut master = GenlockMaster::with_defaults();
        let id = master.add_source("Cam1");
        assert!(master.remove_source(id));
        assert_eq!(master.source_count(), 0);
        assert!(!master.remove_source(id)); // already removed
    }

    #[test]
    fn test_lock_acquisition() {
        let mut cfg = GenlockConfig::default();
        cfg.lock_acquire_samples = 3;
        let mut master = GenlockMaster::new(cfg);
        let id = master.add_source("Cam1");

        // Report within tolerance 3 times
        master.report_phase(id, PhaseOffset::from_ns(100));
        assert_eq!(master.status(id), Some(LockStatus::Locking));
        master.report_phase(id, PhaseOffset::from_ns(-50));
        assert_eq!(master.status(id), Some(LockStatus::Locking));
        master.report_phase(id, PhaseOffset::from_ns(200));
        assert_eq!(master.status(id), Some(LockStatus::Locked));
    }

    #[test]
    fn test_drift_detection() {
        let mut master = GenlockMaster::with_defaults();
        let id = master.add_source("Cam1");
        master.report_phase(id, PhaseOffset::from_ns(10_000)); // > 5000 threshold
        assert_eq!(master.status(id), Some(LockStatus::Drifted));
    }

    #[test]
    fn test_all_locked() {
        let mut cfg = GenlockConfig::default();
        cfg.lock_acquire_samples = 1;
        let mut master = GenlockMaster::new(cfg);
        let a = master.add_source("A");
        let b = master.add_source("B");
        assert!(!master.all_locked());

        master.report_phase(a, PhaseOffset::from_ns(10));
        master.report_phase(b, PhaseOffset::from_ns(-20));
        assert!(master.all_locked());
    }

    #[test]
    fn test_unlocked_sources() {
        let mut master = GenlockMaster::with_defaults();
        master.add_source("A");
        master.add_source("B");
        let unlocked = master.unlocked_sources();
        assert_eq!(unlocked.len(), 2); // both NoSignal initially
    }

    #[test]
    fn test_no_signal_report() {
        let mut cfg = GenlockConfig::default();
        cfg.lock_acquire_samples = 1;
        let mut master = GenlockMaster::new(cfg);
        let id = master.add_source("Cam1");
        master.report_phase(id, PhaseOffset::from_ns(0));
        assert_eq!(master.status(id), Some(LockStatus::Locked));

        master.report_no_signal(id);
        assert_eq!(master.status(id), Some(LockStatus::NoSignal));
    }

    #[test]
    fn test_tick_counter() {
        let mut master = GenlockMaster::with_defaults();
        assert_eq!(master.current_tick(), 0);
        master.tick();
        master.tick();
        assert_eq!(master.current_tick(), 2);
    }

    #[test]
    fn test_summary() {
        let mut master = GenlockMaster::with_defaults();
        master.add_source("Cam1");
        let s = master.summary();
        assert!(s.contains("GenlockMaster"));
        assert!(s.contains("Cam1"));
    }

    #[test]
    fn test_ref_standard_display() {
        assert_eq!(format!("{}", RefStandard::Ptp), "PTP");
        assert_eq!(format!("{}", RefStandard::BlackBurst), "BlackBurst");
    }

    #[test]
    fn test_lock_status_display() {
        assert_eq!(format!("{}", LockStatus::Locked), "LOCKED");
        assert_eq!(format!("{}", LockStatus::FreeRun), "FREE-RUN");
    }
}
