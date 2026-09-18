// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Render blade compute management.
//!
//! Manages a pool of render blades (individual compute nodes in a render farm),
//! tracking hardware specs, task affinity assignments, and pool lifecycle.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Hardware specification
// ─────────────────────────────────────────────────────────────────────────────

/// CPU architecture of a render blade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuArch {
    /// x86-64 (AMD64)
    X86_64,
    /// ARM64 / `AArch64`
    Arm64,
    /// RISC-V 64-bit
    RiscV64,
}

/// GPU vendor identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuVendor {
    /// NVIDIA
    Nvidia,
    /// AMD / ATI
    Amd,
    /// Intel Xe / Arc
    Intel,
    /// No GPU
    None,
}

/// Hardware specification for a single render blade.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeSpec {
    /// Blade identifier string (hostname or UUID).
    pub blade_id: String,
    /// CPU architecture.
    pub cpu_arch: CpuArch,
    /// Number of logical CPU cores.
    pub cpu_cores: u32,
    /// CPU clock speed in MHz.
    pub cpu_mhz: u32,
    /// Total RAM in megabytes.
    pub ram_mb: u64,
    /// GPU vendor present on this blade.
    pub gpu_vendor: GpuVendor,
    /// GPU VRAM in megabytes (0 if no GPU).
    pub gpu_vram_mb: u64,
    /// Number of GPU devices.
    pub gpu_count: u32,
    /// Local scratch disk space in megabytes.
    pub scratch_mb: u64,
    /// Network bandwidth in Mbit/s.
    pub net_mbps: u32,
}

impl BladeSpec {
    /// Create a minimal CPU-only blade specification.
    #[must_use]
    pub fn cpu_only(blade_id: impl Into<String>, cores: u32, ram_mb: u64) -> Self {
        Self {
            blade_id: blade_id.into(),
            cpu_arch: CpuArch::X86_64,
            cpu_cores: cores,
            cpu_mhz: 3000,
            ram_mb,
            gpu_vendor: GpuVendor::None,
            gpu_vram_mb: 0,
            gpu_count: 0,
            scratch_mb: 100_000,
            net_mbps: 10_000,
        }
    }

    /// Create a GPU-accelerated blade specification.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn gpu_blade(
        blade_id: impl Into<String>,
        cores: u32,
        ram_mb: u64,
        vendor: GpuVendor,
        gpu_count: u32,
        gpu_vram_mb: u64,
    ) -> Self {
        Self {
            blade_id: blade_id.into(),
            cpu_arch: CpuArch::X86_64,
            cpu_cores: cores,
            cpu_mhz: 3600,
            ram_mb,
            gpu_vendor: vendor,
            gpu_vram_mb,
            gpu_count,
            scratch_mb: 200_000,
            net_mbps: 25_000,
        }
    }

    /// Compute a relative performance score (higher = faster).
    ///
    /// Simple heuristic: cores × clock + gpu bonus.
    #[must_use]
    pub fn performance_score(&self) -> f64 {
        let cpu_score = f64::from(self.cpu_cores) * (f64::from(self.cpu_mhz) / 1000.0);
        let gpu_score = if self.gpu_vendor == GpuVendor::None {
            0.0
        } else {
            f64::from(self.gpu_count) * (self.gpu_vram_mb as f64 / 1024.0) * 2.0
        };
        cpu_score + gpu_score
    }

    /// Returns `true` if this blade has at least one GPU.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.gpu_vendor != GpuVendor::None && self.gpu_count > 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Task affinity
// ─────────────────────────────────────────────────────────────────────────────

/// Affinity rule that links a job type tag to a set of preferred blades.
#[derive(Debug, Clone)]
pub struct AffinityRule {
    /// Tag identifying the job category (e.g. "`gpu_render`", "sim", "compositing").
    pub tag: String,
    /// Blade IDs preferred for this tag (ordered by preference).
    pub preferred_blades: Vec<String>,
    /// Whether to exclusively route tagged tasks to preferred blades.
    pub exclusive: bool,
}

impl AffinityRule {
    /// Create a non-exclusive affinity rule.
    #[must_use]
    pub fn new(tag: impl Into<String>, preferred_blades: Vec<String>) -> Self {
        Self {
            tag: tag.into(),
            preferred_blades,
            exclusive: false,
        }
    }

    /// Create an exclusive affinity rule (tasks go only to preferred blades).
    #[must_use]
    pub fn exclusive(tag: impl Into<String>, preferred_blades: Vec<String>) -> Self {
        Self {
            tag: tag.into(),
            preferred_blades,
            exclusive: true,
        }
    }

    /// Returns `true` if the given blade is preferred for this rule.
    #[must_use]
    pub fn is_preferred(&self, blade_id: &str) -> bool {
        self.preferred_blades.iter().any(|b| b == blade_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blade state
// ─────────────────────────────────────────────────────────────────────────────

/// Operational state of a blade in the pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BladeState {
    /// Blade is idle and ready to accept tasks.
    Idle,
    /// Blade is actively rendering.
    Busy,
    /// Blade is draining (no new tasks will be assigned, waits for current to finish).
    Draining,
    /// Blade has been taken offline for maintenance.
    Offline,
    /// Blade has reported an error condition.
    Error(String),
}

/// Runtime information for a registered blade.
#[derive(Debug, Clone, PartialEq)]
pub struct BladeEntry {
    /// Static hardware specification.
    pub spec: BladeSpec,
    /// Current operational state.
    pub state: BladeState,
    /// Number of tasks currently assigned.
    pub active_task_count: u32,
    /// Total tasks completed since registration.
    pub completed_tasks: u64,
    /// Total tasks failed since registration.
    pub failed_tasks: u64,
    /// Current CPU utilization percentage (0–100).
    pub cpu_utilization: f32,
    /// Current RAM utilization percentage (0–100).
    pub ram_utilization: f32,
}

impl BladeEntry {
    /// Create a new idle blade entry from a spec.
    #[must_use]
    pub fn new(spec: BladeSpec) -> Self {
        Self {
            spec,
            state: BladeState::Idle,
            active_task_count: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            cpu_utilization: 0.0,
            ram_utilization: 0.0,
        }
    }

    /// Returns `true` if the blade can accept another task.
    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self.state, BladeState::Idle | BladeState::Busy)
            && self.active_task_count < self.spec.cpu_cores
    }

    /// Update utilization metrics.
    pub fn update_utilization(&mut self, cpu: f32, ram: f32) {
        self.cpu_utilization = cpu.clamp(0.0, 100.0);
        self.ram_utilization = ram.clamp(0.0, 100.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blade pool
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for blade pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BladePoolError {
    /// Blade with the given ID is not registered.
    BladeNotFound(String),
    /// Blade with the given ID is already registered.
    BladeAlreadyExists(String),
    /// No blades are available for task assignment.
    NoAvailableBlades,
    /// Affinity rule tag conflicts with an existing rule.
    AffinityConflict(String),
}

impl std::fmt::Display for BladePoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BladeNotFound(id) => write!(f, "blade not found: {id}"),
            Self::BladeAlreadyExists(id) => write!(f, "blade already exists: {id}"),
            Self::NoAvailableBlades => write!(f, "no available blades"),
            Self::AffinityConflict(tag) => write!(f, "affinity conflict for tag: {tag}"),
        }
    }
}

/// A managed pool of render blades with affinity-aware assignment.
#[derive(Debug, Default)]
pub struct BladePool {
    blades: HashMap<String, BladeEntry>,
    affinity_rules: Vec<AffinityRule>,
}

impl BladePool {
    /// Create a new empty blade pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a blade in the pool.
    ///
    /// Returns `Err` if a blade with the same ID is already registered.
    pub fn register(&mut self, spec: BladeSpec) -> Result<(), BladePoolError> {
        if self.blades.contains_key(&spec.blade_id) {
            return Err(BladePoolError::BladeAlreadyExists(spec.blade_id.clone()));
        }
        let id = spec.blade_id.clone();
        self.blades.insert(id, BladeEntry::new(spec));
        Ok(())
    }

    /// Deregister a blade, removing it from the pool.
    pub fn deregister(&mut self, blade_id: &str) -> Result<BladeEntry, BladePoolError> {
        self.blades
            .remove(blade_id)
            .ok_or_else(|| BladePoolError::BladeNotFound(blade_id.to_string()))
    }

    /// Add an affinity rule to the pool.
    pub fn add_affinity_rule(&mut self, rule: AffinityRule) {
        self.affinity_rules.retain(|r| r.tag != rule.tag);
        self.affinity_rules.push(rule);
    }

    /// Select the best available blade for a job with the given tags.
    ///
    /// Preference order:
    /// 1. Blades matching an exclusive affinity rule for any of `tags`
    /// 2. Blades matching a non-exclusive affinity rule for any of `tags`
    /// 3. Idle blades with the highest performance score
    pub fn select_blade(&self, tags: &[&str]) -> Result<&BladeEntry, BladePoolError> {
        // Exclusive affinity first
        for tag in tags {
            if let Some(rule) = self
                .affinity_rules
                .iter()
                .find(|r| r.exclusive && r.tag == *tag)
            {
                for blade_id in &rule.preferred_blades {
                    if let Some(entry) = self.blades.get(blade_id) {
                        if entry.is_available() {
                            return Ok(entry);
                        }
                    }
                }
                // Exclusive rule but no blade available → hard fail
                return Err(BladePoolError::NoAvailableBlades);
            }
        }

        // Non-exclusive affinity preferred
        for tag in tags {
            if let Some(rule) = self
                .affinity_rules
                .iter()
                .find(|r| !r.exclusive && r.tag == *tag)
            {
                for blade_id in &rule.preferred_blades {
                    if let Some(entry) = self.blades.get(blade_id) {
                        if entry.is_available() {
                            return Ok(entry);
                        }
                    }
                }
            }
        }

        // Fallback: highest performance score among available blades
        self.blades
            .values()
            .filter(|e| e.is_available())
            .max_by(|a, b| {
                a.spec
                    .performance_score()
                    .partial_cmp(&b.spec.performance_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(BladePoolError::NoAvailableBlades)
    }

    /// Return the total number of registered blades.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blades.len()
    }

    /// Returns `true` when the pool has no registered blades.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blades.is_empty()
    }

    /// Count blades in a specific state.
    #[must_use]
    pub fn count_in_state(&self, state: &BladeState) -> usize {
        self.blades.values().filter(|e| &e.state == state).count()
    }

    /// Retrieve a mutable reference to a blade entry.
    pub fn get_mut(&mut self, blade_id: &str) -> Option<&mut BladeEntry> {
        self.blades.get_mut(blade_id)
    }

    /// Retrieve an immutable reference to a blade entry.
    #[must_use]
    pub fn get(&self, blade_id: &str) -> Option<&BladeEntry> {
        self.blades.get(blade_id)
    }

    /// Summarise pool capacity: `(total_cores, available_cores)`.
    #[must_use]
    pub fn capacity(&self) -> (u32, u32) {
        let total: u32 = self.blades.values().map(|e| e.spec.cpu_cores).sum();
        let available: u32 = self
            .blades
            .values()
            .filter(|e| e.is_available())
            .map(|e| e.spec.cpu_cores.saturating_sub(e.active_task_count))
            .sum();
        (total, available)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(id: &str, cores: u32) -> BladeSpec {
        BladeSpec::cpu_only(id, cores, 16_384)
    }

    #[test]
    fn test_blade_spec_cpu_only() {
        let s = BladeSpec::cpu_only("blade-1", 32, 65536);
        assert_eq!(s.blade_id, "blade-1");
        assert_eq!(s.cpu_cores, 32);
        assert!(!s.has_gpu());
    }

    #[test]
    fn test_blade_spec_gpu_blade() {
        let s = BladeSpec::gpu_blade("blade-g1", 16, 131_072, GpuVendor::Nvidia, 2, 24_576);
        assert_eq!(s.gpu_count, 2);
        assert!(s.has_gpu());
    }

    #[test]
    fn test_performance_score_cpu_only() {
        let s = BladeSpec::cpu_only("b", 32, 64_000);
        let score = s.performance_score();
        // 32 * 3.0 = 96.0
        assert!((score - 96.0).abs() < 0.001, "score={score}");
    }

    #[test]
    fn test_performance_score_gpu_blade() {
        let s = BladeSpec::gpu_blade("b", 16, 64_000, GpuVendor::Nvidia, 2, 24_576);
        let cpu = 16.0 * 3.6;
        let gpu = 2.0 * (24_576.0 / 1024.0) * 2.0;
        let expected = cpu + gpu;
        assert!((s.performance_score() - expected).abs() < 0.01);
    }

    #[test]
    fn test_blade_entry_availability() {
        let spec = make_spec("blade-1", 8);
        let mut entry = BladeEntry::new(spec);
        assert!(entry.is_available());
        entry.active_task_count = 8; // saturated
        assert!(!entry.is_available());
    }

    #[test]
    fn test_blade_entry_offline_not_available() {
        let spec = make_spec("blade-2", 8);
        let mut entry = BladeEntry::new(spec);
        entry.state = BladeState::Offline;
        assert!(!entry.is_available());
    }

    #[test]
    fn test_pool_register_and_deregister() {
        let mut pool = BladePool::new();
        pool.register(make_spec("b1", 16))
            .expect("should succeed in test");
        assert_eq!(pool.len(), 1);
        pool.deregister("b1").expect("should succeed in test");
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_duplicate_register() {
        let mut pool = BladePool::new();
        pool.register(make_spec("b1", 16))
            .expect("should succeed in test");
        let res = pool.register(make_spec("b1", 8));
        assert_eq!(res, Err(BladePoolError::BladeAlreadyExists("b1".into())));
    }

    #[test]
    fn test_pool_deregister_missing() {
        let mut pool = BladePool::new();
        let res = pool.deregister("ghost");
        assert_eq!(res, Err(BladePoolError::BladeNotFound("ghost".into())));
    }

    #[test]
    fn test_pool_select_best_blade() {
        let mut pool = BladePool::new();
        pool.register(make_spec("small", 4))
            .expect("should succeed in test");
        pool.register(make_spec("large", 64))
            .expect("should succeed in test");
        let blade = pool.select_blade(&[]).expect("should succeed in test");
        assert_eq!(blade.spec.blade_id, "large");
    }

    #[test]
    fn test_pool_affinity_preferred() {
        let mut pool = BladePool::new();
        pool.register(make_spec("fast", 8))
            .expect("should succeed in test");
        pool.register(make_spec("slow", 4))
            .expect("should succeed in test");
        pool.add_affinity_rule(AffinityRule::new("sim", vec!["slow".into()]));
        let blade = pool.select_blade(&["sim"]).expect("should succeed in test");
        assert_eq!(blade.spec.blade_id, "slow");
    }

    #[test]
    fn test_pool_exclusive_affinity() {
        let mut pool = BladePool::new();
        pool.register(make_spec("gpu-1", 8))
            .expect("should succeed in test");
        pool.register(make_spec("cpu-1", 32))
            .expect("should succeed in test");
        pool.add_affinity_rule(AffinityRule::exclusive("gpu_render", vec!["gpu-1".into()]));
        let blade = pool
            .select_blade(&["gpu_render"])
            .expect("should succeed in test");
        assert_eq!(blade.spec.blade_id, "gpu-1");
    }

    #[test]
    fn test_pool_exclusive_no_available_fails() {
        let mut pool = BladePool::new();
        let spec = make_spec("gpu-1", 8);
        pool.register(spec.clone()).expect("should succeed in test");
        // Take the blade offline
        pool.get_mut("gpu-1").expect("should succeed in test").state = BladeState::Offline;
        pool.add_affinity_rule(AffinityRule::exclusive("gpu_render", vec!["gpu-1".into()]));
        let res = pool.select_blade(&["gpu_render"]);
        assert_eq!(res, Err(BladePoolError::NoAvailableBlades));
        // Suppress the unused variable warning
        let _ = spec.blade_id.len();
    }

    #[test]
    fn test_pool_capacity() {
        let mut pool = BladePool::new();
        pool.register(make_spec("b1", 16))
            .expect("should succeed in test");
        pool.register(make_spec("b2", 8))
            .expect("should succeed in test");
        let (total, available) = pool.capacity();
        assert_eq!(total, 24);
        assert_eq!(available, 24);
    }

    #[test]
    fn test_pool_count_in_state() {
        let mut pool = BladePool::new();
        pool.register(make_spec("b1", 8))
            .expect("should succeed in test");
        pool.register(make_spec("b2", 8))
            .expect("should succeed in test");
        pool.get_mut("b2").expect("should succeed in test").state = BladeState::Draining;
        assert_eq!(pool.count_in_state(&BladeState::Idle), 1);
        assert_eq!(pool.count_in_state(&BladeState::Draining), 1);
    }

    #[test]
    fn test_update_utilization_clamp() {
        let mut entry = BladeEntry::new(make_spec("b", 8));
        entry.update_utilization(150.0, -10.0);
        assert_eq!(entry.cpu_utilization, 100.0);
        assert_eq!(entry.ram_utilization, 0.0);
    }

    #[test]
    fn test_affinity_rule_is_preferred() {
        let rule = AffinityRule::new("tag", vec!["b1".into(), "b2".into()]);
        assert!(rule.is_preferred("b1"));
        assert!(!rule.is_preferred("b3"));
    }
}
