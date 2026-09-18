//! Render farm license management — license types, usage tracking, and server.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The type of render-farm license.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseType {
    /// One license per CPU core.
    CpuCore,
    /// One license per GPU unit.
    GpuUnit,
    /// Named-user (seat) license.
    NamedUser,
    /// Floating (concurrent) user license.
    FloatingUser,
}

impl LicenseType {
    /// Returns `true` if this license type is billed per CPU core.
    #[must_use]
    pub fn is_per_core(&self) -> bool {
        matches!(self, Self::CpuCore)
    }
}

/// Tracks allocation state for a single license type on a server.
#[derive(Debug, Clone)]
pub struct LicenseUsage {
    /// The license type this usage record tracks.
    pub license_type: LicenseType,
    /// Total number of licenses purchased.
    pub total_count: u32,
    /// Licenses currently in use.
    pub used_count: u32,
    /// Licenses pre-reserved (not yet checked out but unavailable).
    pub reserved_count: u32,
}

impl LicenseUsage {
    /// Creates a new `LicenseUsage` record.
    #[must_use]
    pub fn new(license_type: LicenseType, total_count: u32) -> Self {
        Self {
            license_type,
            total_count,
            used_count: 0,
            reserved_count: 0,
        }
    }

    /// Number of licenses still available for checkout.
    #[must_use]
    pub fn available_count(&self) -> u32 {
        self.total_count
            .saturating_sub(self.used_count)
            .saturating_sub(self.reserved_count)
    }

    /// Utilisation as a percentage (0.0 – 100.0).
    #[must_use]
    pub fn utilization_pct(&self) -> f32 {
        if self.total_count == 0 {
            return 0.0;
        }
        (self.used_count as f32 / self.total_count as f32) * 100.0
    }

    /// Returns `true` if at least `n` licenses are available.
    #[must_use]
    pub fn can_checkout(&self, n: u32) -> bool {
        self.available_count() >= n
    }
}

/// A license server holding multiple [`LicenseUsage`] records.
#[derive(Debug)]
pub struct LicenseServer {
    usages: Vec<LicenseUsage>,
    /// Human-readable server name.
    pub server_name: String,
}

impl LicenseServer {
    /// Creates a new `LicenseServer` with no license pools.
    #[must_use]
    pub fn new(server_name: impl Into<String>) -> Self {
        Self {
            usages: Vec::new(),
            server_name: server_name.into(),
        }
    }

    /// Adds a license usage pool to this server.
    pub fn add_usage(&mut self, usage: LicenseUsage) {
        self.usages.push(usage);
    }

    /// Attempts to check out `count` licenses of type `lt`.
    ///
    /// Returns `true` and increments `used_count` on success.
    /// Returns `false` if insufficient licenses are available.
    pub fn checkout(&mut self, lt: &LicenseType, count: u32) -> bool {
        if let Some(u) = self.usages.iter_mut().find(|u| &u.license_type == lt) {
            if u.can_checkout(count) {
                u.used_count += count;
                return true;
            }
        }
        false
    }

    /// Returns `count` licenses of type `lt` back to the pool.
    ///
    /// Saturates at zero to avoid underflow.
    pub fn return_license(&mut self, lt: &LicenseType, count: u32) {
        if let Some(u) = self.usages.iter_mut().find(|u| &u.license_type == lt) {
            u.used_count = u.used_count.saturating_sub(count);
        }
    }

    /// Returns a reference to the usage record for `lt`, if present.
    #[must_use]
    pub fn find_usage(&self, lt: &LicenseType) -> Option<&LicenseUsage> {
        self.usages.iter().find(|u| &u.license_type == lt)
    }

    /// Average utilisation across all license pools (0.0 – 100.0).
    #[must_use]
    pub fn total_utilization(&self) -> f32 {
        if self.usages.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.usages.iter().map(LicenseUsage::utilization_pct).sum();
        sum / self.usages.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_core_is_per_core() {
        assert!(LicenseType::CpuCore.is_per_core());
    }

    #[test]
    fn test_gpu_unit_not_per_core() {
        assert!(!LicenseType::GpuUnit.is_per_core());
    }

    #[test]
    fn test_named_user_not_per_core() {
        assert!(!LicenseType::NamedUser.is_per_core());
    }

    #[test]
    fn test_floating_user_not_per_core() {
        assert!(!LicenseType::FloatingUser.is_per_core());
    }

    #[test]
    fn test_available_count_full() {
        let u = LicenseUsage::new(LicenseType::GpuUnit, 10);
        assert_eq!(u.available_count(), 10);
    }

    #[test]
    fn test_available_count_after_use() {
        let mut u = LicenseUsage::new(LicenseType::CpuCore, 8);
        u.used_count = 3;
        assert_eq!(u.available_count(), 5);
    }

    #[test]
    fn test_available_count_with_reservation() {
        let mut u = LicenseUsage::new(LicenseType::FloatingUser, 10);
        u.used_count = 4;
        u.reserved_count = 2;
        assert_eq!(u.available_count(), 4);
    }

    #[test]
    fn test_utilization_pct_zero() {
        let u = LicenseUsage::new(LicenseType::NamedUser, 0);
        assert!((u.utilization_pct() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_utilization_pct_half() {
        let mut u = LicenseUsage::new(LicenseType::CpuCore, 10);
        u.used_count = 5;
        assert!((u.utilization_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_can_checkout_true() {
        let u = LicenseUsage::new(LicenseType::GpuUnit, 5);
        assert!(u.can_checkout(5));
    }

    #[test]
    fn test_can_checkout_false() {
        let u = LicenseUsage::new(LicenseType::GpuUnit, 3);
        assert!(!u.can_checkout(4));
    }

    #[test]
    fn test_server_checkout_success() {
        let mut srv = LicenseServer::new("farm-lic-01");
        srv.add_usage(LicenseUsage::new(LicenseType::CpuCore, 16));
        assert!(srv.checkout(&LicenseType::CpuCore, 8));
        let u = srv
            .find_usage(&LicenseType::CpuCore)
            .expect("should succeed in test");
        assert_eq!(u.used_count, 8);
    }

    #[test]
    fn test_server_checkout_fail_insufficient() {
        let mut srv = LicenseServer::new("farm-lic-01");
        srv.add_usage(LicenseUsage::new(LicenseType::GpuUnit, 2));
        assert!(!srv.checkout(&LicenseType::GpuUnit, 5));
    }

    #[test]
    fn test_server_return_license() {
        let mut srv = LicenseServer::new("srv");
        let mut usage = LicenseUsage::new(LicenseType::FloatingUser, 10);
        usage.used_count = 6;
        srv.add_usage(usage);
        srv.return_license(&LicenseType::FloatingUser, 3);
        let u = srv
            .find_usage(&LicenseType::FloatingUser)
            .expect("should succeed in test");
        assert_eq!(u.used_count, 3);
    }

    #[test]
    fn test_server_find_usage_missing() {
        let srv = LicenseServer::new("srv");
        assert!(srv.find_usage(&LicenseType::NamedUser).is_none());
    }

    #[test]
    fn test_server_total_utilization_empty() {
        let srv = LicenseServer::new("srv");
        assert!((srv.total_utilization() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_server_total_utilization_nonzero() {
        let mut srv = LicenseServer::new("srv");
        let mut u = LicenseUsage::new(LicenseType::CpuCore, 10);
        u.used_count = 5;
        srv.add_usage(u);
        assert!((srv.total_utilization() - 50.0).abs() < 0.01);
    }
}
