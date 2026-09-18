//! Resource reservation: pre-allocate GPU/CPU cores for high-priority jobs.
//!
//! In a shared batch processing environment, high-priority jobs can be starved
//! of resources by a flood of lower-priority work.  This module provides a
//! reservation system that guarantees resources are available when needed.
//!
//! ## Concepts
//!
//! - **ResourcePool**: tracks total, allocated, and reserved resources.
//! - **Reservation**: a named allocation that holds resources until released.
//! - **ReservationManager**: coordinates reservations across pools.

#![allow(dead_code)]

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};
use crate::types::JobId;

// ---------------------------------------------------------------------------
// Resource specification
// ---------------------------------------------------------------------------

/// Describes a set of resources to reserve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    /// Number of CPU cores requested.
    pub cpu_cores: f64,
    /// Memory in MiB.
    pub memory_mib: u64,
    /// Number of GPU devices requested.
    pub gpu_count: u32,
    /// Disk I/O bandwidth in MiB/s (0 = no reservation).
    pub disk_io_mib_per_sec: u64,
}

impl ResourceSpec {
    /// Create a CPU-only resource specification.
    #[must_use]
    pub fn cpu_only(cores: f64, memory_mib: u64) -> Self {
        Self {
            cpu_cores: cores,
            memory_mib,
            gpu_count: 0,
            disk_io_mib_per_sec: 0,
        }
    }

    /// Create a GPU resource specification.
    #[must_use]
    pub fn with_gpu(cores: f64, memory_mib: u64, gpu_count: u32) -> Self {
        Self {
            cpu_cores: cores,
            memory_mib,
            gpu_count,
            disk_io_mib_per_sec: 0,
        }
    }

    /// Builder: set disk I/O.
    #[must_use]
    pub fn with_disk_io(mut self, mib_per_sec: u64) -> Self {
        self.disk_io_mib_per_sec = mib_per_sec;
        self
    }

    /// Whether this specification requests any resources at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cpu_cores <= 0.0
            && self.memory_mib == 0
            && self.gpu_count == 0
            && self.disk_io_mib_per_sec == 0
    }
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self::cpu_only(1.0, 256)
    }
}

// ---------------------------------------------------------------------------
// Resource pool
// ---------------------------------------------------------------------------

/// A pool of a single resource type with allocation tracking.
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Human-readable name (e.g. "cpu_cores", "memory_mib").
    pub name: String,
    /// Total available capacity.
    pub total: f64,
    /// Currently allocated (in-use by running jobs).
    pub allocated: f64,
    /// Currently reserved (held for future use).
    pub reserved: f64,
}

impl ResourcePool {
    /// Create a new pool.
    #[must_use]
    pub fn new(name: impl Into<String>, total: f64) -> Self {
        Self {
            name: name.into(),
            total,
            allocated: 0.0,
            reserved: 0.0,
        }
    }

    /// Available capacity (total - allocated - reserved).
    #[must_use]
    pub fn available(&self) -> f64 {
        (self.total - self.allocated - self.reserved).max(0.0)
    }

    /// Utilization as a fraction (0.0..=1.0).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.total <= 0.0 {
            return 0.0;
        }
        ((self.allocated + self.reserved) / self.total).min(1.0)
    }

    /// Whether `amount` can be reserved.
    #[must_use]
    pub fn can_reserve(&self, amount: f64) -> bool {
        amount <= self.available()
    }

    /// Reserve `amount`. Returns `Err` if insufficient.
    fn reserve(&mut self, amount: f64) -> Result<()> {
        if amount > self.available() {
            return Err(BatchError::ResourceError(format!(
                "Cannot reserve {amount} from pool '{}': only {} available",
                self.name,
                self.available()
            )));
        }
        self.reserved += amount;
        Ok(())
    }

    /// Release `amount` from reservation.
    fn release_reservation(&mut self, amount: f64) {
        self.reserved = (self.reserved - amount).max(0.0);
    }

    /// Allocate `amount` (move from reserved to allocated).
    fn allocate_from_reservation(&mut self, amount: f64) {
        self.reserved = (self.reserved - amount).max(0.0);
        self.allocated += amount;
    }

    /// Release `amount` from allocation.
    fn release_allocation(&mut self, amount: f64) {
        self.allocated = (self.allocated - amount).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Reservation
// ---------------------------------------------------------------------------

/// Priority level for reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReservationPriority {
    /// Normal priority — may be preempted.
    Normal = 0,
    /// High priority — preempts normal.
    High = 1,
    /// Critical priority — never preempted.
    Critical = 2,
}

impl std::fmt::Display for ReservationPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// State of a reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationState {
    /// Resources are reserved but not yet allocated.
    Pending,
    /// Resources are allocated and in use.
    Active,
    /// Reservation has been released.
    Released,
    /// Reservation expired without being used.
    Expired,
}

impl std::fmt::Display for ReservationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Active => write!(f, "Active"),
            Self::Released => write!(f, "Released"),
            Self::Expired => write!(f, "Expired"),
        }
    }
}

/// A named reservation holding resources for a specific job.
#[derive(Debug, Clone)]
pub struct Reservation {
    /// Unique reservation ID.
    pub reservation_id: String,
    /// Job that this reservation is for.
    pub job_id: JobId,
    /// Resources reserved.
    pub spec: ResourceSpec,
    /// Priority of this reservation.
    pub priority: ReservationPriority,
    /// Current state.
    pub state: ReservationState,
    /// Unix timestamp when the reservation was created.
    pub created_at_secs: u64,
    /// Maximum time in seconds to hold the reservation before expiration.
    pub ttl_secs: u64,
    /// Optional label for this reservation.
    pub label: Option<String>,
}

impl Reservation {
    /// Whether this reservation has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if self.ttl_secs == 0 {
            return false; // no expiration
        }
        let now = current_timestamp();
        now.saturating_sub(self.created_at_secs) > self.ttl_secs
    }

    /// Time remaining before expiration (0 if already expired or no TTL).
    #[must_use]
    pub fn remaining_secs(&self) -> u64 {
        if self.ttl_secs == 0 {
            return u64::MAX; // effectively infinite
        }
        let elapsed = current_timestamp().saturating_sub(self.created_at_secs);
        self.ttl_secs.saturating_sub(elapsed)
    }

    /// Age of this reservation in seconds.
    #[must_use]
    pub fn age_secs(&self) -> u64 {
        current_timestamp().saturating_sub(self.created_at_secs)
    }
}

// ---------------------------------------------------------------------------
// Reservation manager
// ---------------------------------------------------------------------------

/// Manages resource reservations across multiple resource pools.
#[derive(Debug)]
pub struct ReservationManager {
    /// Resource pools keyed by name.
    pools: RwLock<HashMap<String, ResourcePool>>,
    /// Active reservations keyed by reservation ID.
    reservations: RwLock<HashMap<String, Reservation>>,
    /// Next reservation ID counter.
    next_id: std::sync::atomic::AtomicU64,
    /// Default TTL for new reservations (seconds).
    default_ttl_secs: u64,
}

impl ReservationManager {
    /// Create a new manager with the given resource capacity.
    #[must_use]
    pub fn new(total_cpu: f64, total_memory_mib: u64, total_gpus: u32) -> Self {
        let mut pools = HashMap::new();
        pools.insert("cpu".to_string(), ResourcePool::new("cpu", total_cpu));
        pools.insert(
            "memory".to_string(),
            ResourcePool::new("memory", total_memory_mib as f64),
        );
        pools.insert(
            "gpu".to_string(),
            ResourcePool::new("gpu", total_gpus as f64),
        );

        Self {
            pools: RwLock::new(pools),
            reservations: RwLock::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicU64::new(1),
            default_ttl_secs: 300, // 5 minutes
        }
    }

    /// Set the default TTL for new reservations.
    pub fn set_default_ttl(&mut self, secs: u64) {
        self.default_ttl_secs = secs;
    }

    /// Add a custom resource pool.
    pub fn add_pool(&self, pool: ResourcePool) {
        self.pools.write().insert(pool.name.clone(), pool);
    }

    /// Reserve resources for a job.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] if insufficient resources are available.
    pub fn reserve(
        &self,
        job_id: JobId,
        spec: ResourceSpec,
        priority: ReservationPriority,
    ) -> Result<String> {
        self.reserve_with_ttl(job_id, spec, priority, self.default_ttl_secs)
    }

    /// Reserve resources with a custom TTL.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] if insufficient resources are available.
    pub fn reserve_with_ttl(
        &self,
        job_id: JobId,
        spec: ResourceSpec,
        priority: ReservationPriority,
        ttl_secs: u64,
    ) -> Result<String> {
        // First, expire any stale reservations.
        self.expire_stale();

        // Check availability.
        {
            let pools = self.pools.read();
            self.check_availability(&pools, &spec)?;
        }

        // Perform the reservation.
        let reservation_id = format!(
            "rsv-{}",
            self.next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );

        {
            let mut pools = self.pools.write();
            self.do_reserve(&mut pools, &spec)?;
        }

        let reservation = Reservation {
            reservation_id: reservation_id.clone(),
            job_id,
            spec,
            priority,
            state: ReservationState::Pending,
            created_at_secs: current_timestamp(),
            ttl_secs,
            label: None,
        };

        self.reservations
            .write()
            .insert(reservation_id.clone(), reservation);

        Ok(reservation_id)
    }

    /// Activate a reservation (move from reserved to allocated).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] if the reservation is not found or not pending.
    pub fn activate(&self, reservation_id: &str) -> Result<()> {
        let mut reservations = self.reservations.write();
        let reservation = reservations.get_mut(reservation_id).ok_or_else(|| {
            BatchError::ResourceError(format!("Reservation not found: {reservation_id}"))
        })?;

        if reservation.state != ReservationState::Pending {
            return Err(BatchError::ResourceError(format!(
                "Reservation '{reservation_id}' is not pending (state: {})",
                reservation.state
            )));
        }

        let mut pools = self.pools.write();
        if let Some(pool) = pools.get_mut("cpu") {
            pool.allocate_from_reservation(reservation.spec.cpu_cores);
        }
        if let Some(pool) = pools.get_mut("memory") {
            pool.allocate_from_reservation(reservation.spec.memory_mib as f64);
        }
        if let Some(pool) = pools.get_mut("gpu") {
            pool.allocate_from_reservation(reservation.spec.gpu_count as f64);
        }

        reservation.state = ReservationState::Active;
        Ok(())
    }

    /// Release a reservation (return resources to the pool).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::ResourceError`] if the reservation is not found.
    pub fn release(&self, reservation_id: &str) -> Result<()> {
        let mut reservations = self.reservations.write();
        let reservation = reservations.get_mut(reservation_id).ok_or_else(|| {
            BatchError::ResourceError(format!("Reservation not found: {reservation_id}"))
        })?;

        let mut pools = self.pools.write();

        match reservation.state {
            ReservationState::Pending => {
                // Release from reserved.
                if let Some(pool) = pools.get_mut("cpu") {
                    pool.release_reservation(reservation.spec.cpu_cores);
                }
                if let Some(pool) = pools.get_mut("memory") {
                    pool.release_reservation(reservation.spec.memory_mib as f64);
                }
                if let Some(pool) = pools.get_mut("gpu") {
                    pool.release_reservation(reservation.spec.gpu_count as f64);
                }
            }
            ReservationState::Active => {
                // Release from allocated.
                if let Some(pool) = pools.get_mut("cpu") {
                    pool.release_allocation(reservation.spec.cpu_cores);
                }
                if let Some(pool) = pools.get_mut("memory") {
                    pool.release_allocation(reservation.spec.memory_mib as f64);
                }
                if let Some(pool) = pools.get_mut("gpu") {
                    pool.release_allocation(reservation.spec.gpu_count as f64);
                }
            }
            ReservationState::Released | ReservationState::Expired => {
                // Already released — no-op.
            }
        }

        reservation.state = ReservationState::Released;
        Ok(())
    }

    /// Get a snapshot of all resource pools.
    #[must_use]
    pub fn pool_status(&self) -> Vec<ResourcePool> {
        self.pools.read().values().cloned().collect()
    }

    /// Get a specific pool's status.
    #[must_use]
    pub fn get_pool(&self, name: &str) -> Option<ResourcePool> {
        self.pools.read().get(name).cloned()
    }

    /// List all active reservations.
    #[must_use]
    pub fn active_reservations(&self) -> Vec<Reservation> {
        self.reservations
            .read()
            .values()
            .filter(|r| {
                matches!(
                    r.state,
                    ReservationState::Pending | ReservationState::Active
                )
            })
            .cloned()
            .collect()
    }

    /// Get a specific reservation.
    #[must_use]
    pub fn get_reservation(&self, reservation_id: &str) -> Option<Reservation> {
        self.reservations.read().get(reservation_id).cloned()
    }

    /// Number of active (non-released, non-expired) reservations.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.reservations
            .read()
            .values()
            .filter(|r| {
                matches!(
                    r.state,
                    ReservationState::Pending | ReservationState::Active
                )
            })
            .count()
    }

    /// Whether resources are available for the given spec.
    #[must_use]
    pub fn can_reserve(&self, spec: &ResourceSpec) -> bool {
        let pools = self.pools.read();
        self.check_availability(&pools, spec).is_ok()
    }

    /// Expire stale reservations and return their resources.
    ///
    /// Returns the number of reservations expired.
    pub fn expire_stale(&self) -> usize {
        let mut expired_ids = Vec::new();

        {
            let reservations = self.reservations.read();
            for (id, r) in reservations.iter() {
                if r.state == ReservationState::Pending && r.is_expired() {
                    expired_ids.push(id.clone());
                }
            }
        }

        let mut count = 0;
        for id in &expired_ids {
            let mut reservations = self.reservations.write();
            if let Some(r) = reservations.get_mut(id) {
                if r.state == ReservationState::Pending {
                    let mut pools = self.pools.write();
                    if let Some(pool) = pools.get_mut("cpu") {
                        pool.release_reservation(r.spec.cpu_cores);
                    }
                    if let Some(pool) = pools.get_mut("memory") {
                        pool.release_reservation(r.spec.memory_mib as f64);
                    }
                    if let Some(pool) = pools.get_mut("gpu") {
                        pool.release_reservation(r.spec.gpu_count as f64);
                    }
                    r.state = ReservationState::Expired;
                    count += 1;
                }
            }
        }

        count
    }

    /// Summary statistics.
    #[must_use]
    pub fn stats(&self) -> ReservationStats {
        let reservations = self.reservations.read();
        let mut pending = 0usize;
        let mut active = 0usize;
        let mut released = 0usize;
        let mut expired = 0usize;

        for r in reservations.values() {
            match r.state {
                ReservationState::Pending => pending += 1,
                ReservationState::Active => active += 1,
                ReservationState::Released => released += 1,
                ReservationState::Expired => expired += 1,
            }
        }

        let pools = self.pools.read();
        let cpu_util = pools.get("cpu").map(|p| p.utilization()).unwrap_or(0.0);
        let mem_util = pools.get("memory").map(|p| p.utilization()).unwrap_or(0.0);
        let gpu_util = pools.get("gpu").map(|p| p.utilization()).unwrap_or(0.0);

        ReservationStats {
            pending_reservations: pending,
            active_reservations: active,
            released_reservations: released,
            expired_reservations: expired,
            cpu_utilization: cpu_util,
            memory_utilization: mem_util,
            gpu_utilization: gpu_util,
        }
    }

    // ── Private helpers ─────────────────────────────────────────────────

    fn check_availability(
        &self,
        pools: &HashMap<String, ResourcePool>,
        spec: &ResourceSpec,
    ) -> Result<()> {
        if let Some(pool) = pools.get("cpu") {
            if !pool.can_reserve(spec.cpu_cores) {
                return Err(BatchError::ResourceError(format!(
                    "Insufficient CPU: need {} cores, {} available",
                    spec.cpu_cores,
                    pool.available()
                )));
            }
        }
        if let Some(pool) = pools.get("memory") {
            if !pool.can_reserve(spec.memory_mib as f64) {
                return Err(BatchError::ResourceError(format!(
                    "Insufficient memory: need {} MiB, {} MiB available",
                    spec.memory_mib,
                    pool.available()
                )));
            }
        }
        if let Some(pool) = pools.get("gpu") {
            if !pool.can_reserve(spec.gpu_count as f64) {
                return Err(BatchError::ResourceError(format!(
                    "Insufficient GPU: need {}, {} available",
                    spec.gpu_count,
                    pool.available()
                )));
            }
        }
        Ok(())
    }

    fn do_reserve(
        &self,
        pools: &mut HashMap<String, ResourcePool>,
        spec: &ResourceSpec,
    ) -> Result<()> {
        if let Some(pool) = pools.get_mut("cpu") {
            pool.reserve(spec.cpu_cores)?;
        }
        if let Some(pool) = pools.get_mut("memory") {
            pool.reserve(spec.memory_mib as f64)?;
        }
        if let Some(pool) = pools.get_mut("gpu") {
            pool.reserve(spec.gpu_count as f64)?;
        }
        Ok(())
    }
}

/// Summary statistics for the reservation system.
#[derive(Debug, Clone)]
pub struct ReservationStats {
    /// Number of pending reservations.
    pub pending_reservations: usize,
    /// Number of active (allocated) reservations.
    pub active_reservations: usize,
    /// Number of released reservations.
    pub released_reservations: usize,
    /// Number of expired reservations.
    pub expired_reservations: usize,
    /// CPU utilization fraction (0.0..=1.0).
    pub cpu_utilization: f64,
    /// Memory utilization fraction (0.0..=1.0).
    pub memory_utilization: f64,
    /// GPU utilization fraction (0.0..=1.0).
    pub gpu_utilization: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ResourceSpec ────────────────────────────────────────────────────
    #[test]
    fn test_resource_spec_cpu_only() {
        let spec = ResourceSpec::cpu_only(4.0, 1024);
        assert_eq!(spec.cpu_cores, 4.0);
        assert_eq!(spec.memory_mib, 1024);
        assert_eq!(spec.gpu_count, 0);
        assert!(!spec.is_empty());
    }

    #[test]
    fn test_resource_spec_with_gpu() {
        let spec = ResourceSpec::with_gpu(8.0, 2048, 2);
        assert_eq!(spec.gpu_count, 2);
    }

    #[test]
    fn test_resource_spec_with_disk_io() {
        let spec = ResourceSpec::cpu_only(1.0, 256).with_disk_io(100);
        assert_eq!(spec.disk_io_mib_per_sec, 100);
    }

    #[test]
    fn test_resource_spec_empty() {
        let spec = ResourceSpec {
            cpu_cores: 0.0,
            memory_mib: 0,
            gpu_count: 0,
            disk_io_mib_per_sec: 0,
        };
        assert!(spec.is_empty());
    }

    #[test]
    fn test_resource_spec_default() {
        let spec = ResourceSpec::default();
        assert_eq!(spec.cpu_cores, 1.0);
        assert_eq!(spec.memory_mib, 256);
    }

    // ── ResourcePool ────────────────────────────────────────────────────
    #[test]
    fn test_resource_pool_available() {
        let mut pool = ResourcePool::new("cpu", 16.0);
        assert!((pool.available() - 16.0).abs() < f64::EPSILON);

        pool.allocated = 4.0;
        pool.reserved = 2.0;
        assert!((pool.available() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_pool_utilization() {
        let mut pool = ResourcePool::new("mem", 1000.0);
        assert!((pool.utilization()).abs() < f64::EPSILON);

        pool.allocated = 300.0;
        pool.reserved = 200.0;
        assert!((pool.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_pool_can_reserve() {
        let pool = ResourcePool::new("gpu", 4.0);
        assert!(pool.can_reserve(3.0));
        assert!(pool.can_reserve(4.0));
        assert!(!pool.can_reserve(5.0));
    }

    // ── ReservationManager basic ────────────────────────────────────────
    #[test]
    fn test_manager_creation() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let cpu = mgr.get_pool("cpu").expect("should have cpu pool");
        assert!((cpu.total - 16.0).abs() < f64::EPSILON);
        let gpu = mgr.get_pool("gpu").expect("should have gpu pool");
        assert!((gpu.total - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reserve_and_release() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let spec = ResourceSpec::cpu_only(4.0, 1024);

        let rid = mgr
            .reserve(JobId::from("j1"), spec, ReservationPriority::Normal)
            .expect("should reserve");

        // Check pool status.
        let cpu = mgr.get_pool("cpu").expect("should have cpu pool");
        assert!((cpu.reserved - 4.0).abs() < f64::EPSILON);
        assert!((cpu.available() - 12.0).abs() < f64::EPSILON);

        // Release.
        mgr.release(&rid).expect("should release");
        let cpu = mgr.get_pool("cpu").expect("should have cpu pool");
        assert!((cpu.reserved).abs() < f64::EPSILON);
        assert!((cpu.available() - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reserve_activate_release() {
        let mgr = ReservationManager::new(8.0, 16384, 2);
        let spec = ResourceSpec::with_gpu(2.0, 512, 1);

        let rid = mgr
            .reserve(JobId::from("j2"), spec, ReservationPriority::High)
            .expect("should reserve");

        // Activate.
        mgr.activate(&rid).expect("should activate");
        let reservation = mgr.get_reservation(&rid).expect("should exist");
        assert_eq!(reservation.state, ReservationState::Active);

        // Resources should move from reserved to allocated.
        let cpu = mgr.get_pool("cpu").expect("should have cpu pool");
        assert!((cpu.reserved).abs() < f64::EPSILON);
        assert!((cpu.allocated - 2.0).abs() < f64::EPSILON);

        // Release.
        mgr.release(&rid).expect("should release");
        let cpu = mgr.get_pool("cpu").expect("should have cpu pool");
        assert!((cpu.allocated).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reserve_insufficient_cpu() {
        let mgr = ReservationManager::new(4.0, 32768, 0);
        let spec = ResourceSpec::cpu_only(8.0, 256);
        let result = mgr.reserve(JobId::from("j3"), spec, ReservationPriority::Normal);
        assert!(result.is_err());
    }

    #[test]
    fn test_reserve_insufficient_memory() {
        let mgr = ReservationManager::new(16.0, 1024, 0);
        let spec = ResourceSpec::cpu_only(1.0, 2048);
        let result = mgr.reserve(JobId::from("j4"), spec, ReservationPriority::Normal);
        assert!(result.is_err());
    }

    #[test]
    fn test_reserve_insufficient_gpu() {
        let mgr = ReservationManager::new(16.0, 32768, 1);
        let spec = ResourceSpec::with_gpu(1.0, 256, 2);
        let result = mgr.reserve(JobId::from("j5"), spec, ReservationPriority::Normal);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_reservations() {
        let mgr = ReservationManager::new(16.0, 32768, 4);

        let r1 = mgr
            .reserve(
                JobId::from("j1"),
                ResourceSpec::cpu_only(4.0, 4096),
                ReservationPriority::Normal,
            )
            .expect("should reserve");
        let r2 = mgr
            .reserve(
                JobId::from("j2"),
                ResourceSpec::cpu_only(4.0, 4096),
                ReservationPriority::High,
            )
            .expect("should reserve");

        assert_eq!(mgr.active_count(), 2);

        mgr.release(&r1).expect("should release");
        mgr.release(&r2).expect("should release");
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_can_reserve() {
        let mgr = ReservationManager::new(8.0, 16384, 0);
        assert!(mgr.can_reserve(&ResourceSpec::cpu_only(4.0, 1024)));
        assert!(!mgr.can_reserve(&ResourceSpec::cpu_only(20.0, 1024)));
    }

    #[test]
    fn test_activate_non_pending_returns_error() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let rid = mgr
            .reserve(
                JobId::from("j1"),
                ResourceSpec::cpu_only(1.0, 256),
                ReservationPriority::Normal,
            )
            .expect("should reserve");
        mgr.release(&rid).expect("should release");
        let result = mgr.activate(&rid);
        assert!(result.is_err());
    }

    #[test]
    fn test_release_nonexistent_returns_error() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let result = mgr.release("no-such-reservation");
        assert!(result.is_err());
    }

    #[test]
    fn test_active_reservations_list() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let _r1 = mgr
            .reserve(
                JobId::from("j1"),
                ResourceSpec::cpu_only(1.0, 256),
                ReservationPriority::Normal,
            )
            .expect("should reserve");
        let _r2 = mgr
            .reserve(
                JobId::from("j2"),
                ResourceSpec::cpu_only(1.0, 256),
                ReservationPriority::High,
            )
            .expect("should reserve");

        let active = mgr.active_reservations();
        assert_eq!(active.len(), 2);
    }

    // ── Stats ───────────────────────────────────────────────────────────
    #[test]
    fn test_reservation_stats() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let rid = mgr
            .reserve(
                JobId::from("j1"),
                ResourceSpec::cpu_only(4.0, 8192),
                ReservationPriority::Normal,
            )
            .expect("should reserve");
        mgr.activate(&rid).expect("should activate");

        let stats = mgr.stats();
        assert_eq!(stats.active_reservations, 1);
        assert!(stats.cpu_utilization > 0.0);
        assert!(stats.memory_utilization > 0.0);
    }

    // ── Pool status ─────────────────────────────────────────────────────
    #[test]
    fn test_pool_status() {
        let mgr = ReservationManager::new(8.0, 4096, 2);
        let pools = mgr.pool_status();
        assert!(pools.len() >= 3); // cpu, memory, gpu
    }

    // ── Custom pool ─────────────────────────────────────────────────────
    #[test]
    fn test_add_custom_pool() {
        let mgr = ReservationManager::new(8.0, 4096, 0);
        mgr.add_pool(ResourcePool::new("disk_io", 1000.0));
        let pool = mgr.get_pool("disk_io").expect("should have custom pool");
        assert!((pool.total - 1000.0).abs() < f64::EPSILON);
    }

    // ── Reservation priority display ────────────────────────────────────
    #[test]
    fn test_reservation_priority_display() {
        assert_eq!(ReservationPriority::Normal.to_string(), "Normal");
        assert_eq!(ReservationPriority::High.to_string(), "High");
        assert_eq!(ReservationPriority::Critical.to_string(), "Critical");
    }

    // ── Reservation state display ───────────────────────────────────────
    #[test]
    fn test_reservation_state_display() {
        assert_eq!(ReservationState::Pending.to_string(), "Pending");
        assert_eq!(ReservationState::Active.to_string(), "Active");
        assert_eq!(ReservationState::Released.to_string(), "Released");
        assert_eq!(ReservationState::Expired.to_string(), "Expired");
    }

    // ── Reservation age / remaining ─────────────────────────────────────
    #[test]
    fn test_reservation_age() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let rid = mgr
            .reserve(
                JobId::from("j1"),
                ResourceSpec::cpu_only(1.0, 256),
                ReservationPriority::Normal,
            )
            .expect("should reserve");
        let r = mgr.get_reservation(&rid).expect("should exist");
        assert!(r.age_secs() < 5);
        assert!(r.remaining_secs() > 0);
    }

    // ── Double release is idempotent ────────────────────────────────────
    #[test]
    fn test_double_release_is_ok() {
        let mgr = ReservationManager::new(16.0, 32768, 4);
        let rid = mgr
            .reserve(
                JobId::from("j1"),
                ResourceSpec::cpu_only(1.0, 256),
                ReservationPriority::Normal,
            )
            .expect("should reserve");
        mgr.release(&rid).expect("first release");
        mgr.release(&rid).expect("second release should be ok");
    }

    // ── Priority ordering ───────────────────────────────────────────────
    #[test]
    fn test_reservation_priority_ordering() {
        assert!(ReservationPriority::Critical > ReservationPriority::High);
        assert!(ReservationPriority::High > ReservationPriority::Normal);
    }
}
