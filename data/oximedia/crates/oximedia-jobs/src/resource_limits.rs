//! Job resource limit enforcement.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Category of a compute or I/O resource.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// CPU processing capacity (measured in logical cores).
    Cpu,
    /// System memory (measured in GiB by convention).
    Memory,
    /// Disk I/O or storage (measured in GiB by convention).
    Disk,
    /// Network bandwidth (measured in Mbps by convention).
    Network,
    /// GPU compute or memory (measured in GiB by convention).
    Gpu,
}

impl ResourceType {
    /// Returns `true` if this resource type is a compute resource (CPU or GPU).
    #[must_use]
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::Cpu | Self::Gpu)
    }
}

/// A limit on a single resource.
#[derive(Clone, Debug)]
pub struct ResourceLimit {
    /// The resource being limited.
    pub resource: ResourceType,
    /// The maximum allowed value.
    pub max_value: f64,
    /// Human-readable unit string (e.g. "cores", "GiB", "Mbps").
    pub unit: String,
}

impl ResourceLimit {
    /// Create a new `ResourceLimit`.
    #[must_use]
    pub fn new(resource: ResourceType, max_value: f64, unit: impl Into<String>) -> Self {
        Self {
            resource,
            max_value,
            unit: unit.into(),
        }
    }

    /// Convenience constructor: `n` CPU cores.
    #[must_use]
    pub fn cpu_cores(n: f64) -> Self {
        Self::new(ResourceType::Cpu, n, "cores")
    }

    /// Convenience constructor: `gb` GiB of memory.
    #[must_use]
    pub fn memory_gb(gb: f64) -> Self {
        Self::new(ResourceType::Memory, gb, "GiB")
    }

    /// Returns `true` if `used` exceeds `self.max_value`.
    #[must_use]
    pub fn is_exceeded(&self, used: f64) -> bool {
        used > self.max_value
    }

    /// Returns the usage as a percentage of the limit (clamped to [0, 100]).
    #[must_use]
    pub fn usage_pct(&self, used: f64) -> f32 {
        if self.max_value <= 0.0 {
            return 100.0;
        }
        ((used / self.max_value * 100.0).clamp(0.0, 100.0)) as f32
    }
}

/// A collection of resource limits that together define a job's resource profile.
#[derive(Clone, Debug, Default)]
pub struct ResourceProfile {
    /// Individual limits, one per resource type (by convention).
    pub limits: Vec<ResourceLimit>,
}

impl ResourceProfile {
    /// Create an empty `ResourceProfile`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a `ResourceLimit` to this profile.
    pub fn add(&mut self, limit: ResourceLimit) {
        self.limits.push(limit);
    }

    /// Returns `true` if `used` is within the limit for `resource`.
    ///
    /// If no limit is defined for the resource, the check passes (`true`).
    #[must_use]
    pub fn check(&self, resource: &ResourceType, used: f64) -> bool {
        match self.limit_for(resource) {
            Some(limit) => !limit.is_exceeded(used),
            None => true,
        }
    }

    /// Return a reference to the limit for the given resource, if one exists.
    #[must_use]
    pub fn limit_for(&self, resource: &ResourceType) -> Option<&ResourceLimit> {
        self.limits.iter().find(|l| &l.resource == resource)
    }
}

/// Associates `ResourceProfile`s with job ids.
#[derive(Clone, Debug, Default)]
pub struct ResourceBudget {
    /// Pairs of `(job_id, profile)`.
    pub profiles: Vec<(u64, ResourceProfile)>,
}

impl ResourceBudget {
    /// Create an empty `ResourceBudget`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a profile to a job, replacing any existing profile for that job.
    pub fn assign_profile(&mut self, job_id: u64, profile: ResourceProfile) {
        if let Some(existing) = self.profiles.iter_mut().find(|(id, _)| *id == job_id) {
            existing.1 = profile;
        } else {
            self.profiles.push((job_id, profile));
        }
    }

    /// Retrieve the profile for a job by id.
    #[must_use]
    pub fn get_profile(&self, job_id: u64) -> Option<&ResourceProfile> {
        self.profiles
            .iter()
            .find(|(id, _)| *id == job_id)
            .map(|(_, p)| p)
    }

    /// Number of jobs tracked in this budget.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.profiles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- ResourceType tests ----------

    #[test]
    fn test_is_compute_cpu() {
        assert!(ResourceType::Cpu.is_compute());
    }

    #[test]
    fn test_is_compute_gpu() {
        assert!(ResourceType::Gpu.is_compute());
    }

    #[test]
    fn test_is_compute_memory_false() {
        assert!(!ResourceType::Memory.is_compute());
    }

    #[test]
    fn test_is_compute_disk_false() {
        assert!(!ResourceType::Disk.is_compute());
    }

    #[test]
    fn test_is_compute_network_false() {
        assert!(!ResourceType::Network.is_compute());
    }

    // ---------- ResourceLimit tests ----------

    #[test]
    fn test_cpu_cores_constructor() {
        let lim = ResourceLimit::cpu_cores(8.0);
        assert_eq!(lim.resource, ResourceType::Cpu);
        assert!((lim.max_value - 8.0).abs() < 1e-9);
        assert_eq!(lim.unit, "cores");
    }

    #[test]
    fn test_memory_gb_constructor() {
        let lim = ResourceLimit::memory_gb(16.0);
        assert_eq!(lim.resource, ResourceType::Memory);
        assert!((lim.max_value - 16.0).abs() < 1e-9);
        assert_eq!(lim.unit, "GiB");
    }

    #[test]
    fn test_is_exceeded_true() {
        let lim = ResourceLimit::cpu_cores(4.0);
        assert!(lim.is_exceeded(5.0));
    }

    #[test]
    fn test_is_exceeded_false() {
        let lim = ResourceLimit::cpu_cores(4.0);
        assert!(!lim.is_exceeded(3.9));
    }

    #[test]
    fn test_usage_pct_half() {
        let lim = ResourceLimit::memory_gb(8.0);
        let pct = lim.usage_pct(4.0);
        assert!((pct - 50.0).abs() < 0.1, "pct = {pct}");
    }

    #[test]
    fn test_usage_pct_clamped() {
        let lim = ResourceLimit::cpu_cores(4.0);
        let pct = lim.usage_pct(100.0);
        assert!((pct - 100.0).abs() < 0.1);
    }

    // ---------- ResourceProfile tests ----------

    #[test]
    fn test_profile_check_within_limit() {
        let mut profile = ResourceProfile::new();
        profile.add(ResourceLimit::cpu_cores(8.0));
        assert!(profile.check(&ResourceType::Cpu, 7.0));
    }

    #[test]
    fn test_profile_check_exceeded() {
        let mut profile = ResourceProfile::new();
        profile.add(ResourceLimit::memory_gb(4.0));
        assert!(!profile.check(&ResourceType::Memory, 5.0));
    }

    #[test]
    fn test_profile_check_missing_resource_passes() {
        let profile = ResourceProfile::new();
        assert!(profile.check(&ResourceType::Gpu, 999.0));
    }

    #[test]
    fn test_profile_limit_for_found() {
        let mut profile = ResourceProfile::new();
        profile.add(ResourceLimit::cpu_cores(4.0));
        assert!(profile.limit_for(&ResourceType::Cpu).is_some());
    }

    #[test]
    fn test_profile_limit_for_not_found() {
        let profile = ResourceProfile::new();
        assert!(profile.limit_for(&ResourceType::Network).is_none());
    }

    // ---------- ResourceBudget tests ----------

    #[test]
    fn test_budget_assign_and_get() {
        let mut budget = ResourceBudget::new();
        let mut profile = ResourceProfile::new();
        profile.add(ResourceLimit::cpu_cores(2.0));
        budget.assign_profile(101, profile);
        assert!(budget.get_profile(101).is_some());
    }

    #[test]
    fn test_budget_job_count() {
        let mut budget = ResourceBudget::new();
        budget.assign_profile(1, ResourceProfile::new());
        budget.assign_profile(2, ResourceProfile::new());
        assert_eq!(budget.job_count(), 2);
    }

    #[test]
    fn test_budget_get_missing() {
        let budget = ResourceBudget::new();
        assert!(budget.get_profile(999).is_none());
    }
}
