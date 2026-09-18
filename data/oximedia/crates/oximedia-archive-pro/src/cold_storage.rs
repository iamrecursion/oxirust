//! Cold and deep-freeze storage tier management for large media archives.
//!
//! Implements a policy-driven tiering model that classifies assets into hot,
//! warm, cold, and frozen storage tiers based on last-access time, file size,
//! and explicit policy rules.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::path::PathBuf;

/// Storage tier classification.
///
/// Tiers are ordered by access speed (fastest first) and cost (cheapest last).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StorageTier {
    /// Online SSD/NVMe — instant access, highest cost.
    Hot,
    /// Nearline spinning disk — seconds to access, moderate cost.
    Warm,
    /// Tape / cloud archive — minutes to hours to access, low cost.
    Cold,
    /// Deep-freeze / glacier — hours to days to access, very low cost.
    Frozen,
}

impl std::fmt::Display for StorageTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Hot => "Hot",
            Self::Warm => "Warm",
            Self::Cold => "Cold",
            Self::Frozen => "Frozen",
        };
        write!(f, "{s}")
    }
}

impl StorageTier {
    /// Relative cost factor (Hot = 1.0 baseline).
    #[must_use]
    pub fn relative_cost(self) -> f32 {
        match self {
            Self::Hot => 1.0,
            Self::Warm => 0.4,
            Self::Cold => 0.08,
            Self::Frozen => 0.02,
        }
    }

    /// Typical maximum retrieval latency as a human-readable string.
    #[must_use]
    pub fn retrieval_latency(self) -> &'static str {
        match self {
            Self::Hot => "< 1 ms",
            Self::Warm => "< 10 s",
            Self::Cold => "< 4 h",
            Self::Frozen => "< 48 h",
        }
    }
}

// ---------------------------------------------------------------------------
// ColdStoragePolicy
// ---------------------------------------------------------------------------

/// Rules that determine when an asset should be moved to a cooler tier.
#[derive(Debug, Clone)]
pub struct ColdStoragePolicy {
    /// Move to Warm after this many days without access.
    pub warm_after_days: u32,
    /// Move to Cold after this many days without access.
    pub cold_after_days: u32,
    /// Move to Frozen after this many days without access.
    pub frozen_after_days: u32,
    /// Minimum file size in bytes to be considered for tiering.
    pub min_size_bytes: u64,
    /// If `true`, assets marked as "always-hot" are never downgraded.
    pub respect_pinned: bool,
}

impl Default for ColdStoragePolicy {
    fn default() -> Self {
        Self {
            warm_after_days: 30,
            cold_after_days: 180,
            frozen_after_days: 365,
            min_size_bytes: 1024 * 1024, // 1 MiB
            respect_pinned: true,
        }
    }
}

impl ColdStoragePolicy {
    /// Validate policy constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.warm_after_days >= self.cold_after_days {
            return Err("warm_after_days must be less than cold_after_days".to_string());
        }
        if self.cold_after_days >= self.frozen_after_days {
            return Err("cold_after_days must be less than frozen_after_days".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Asset descriptor
// ---------------------------------------------------------------------------

/// Minimal descriptor for an asset under cold-storage management.
#[derive(Debug, Clone)]
pub struct AssetDescriptor {
    /// Filesystem path of the asset.
    pub path: PathBuf,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Days since last access (0 = accessed today).
    pub days_since_access: u32,
    /// Whether the asset is pinned to hot storage.
    pub pinned: bool,
    /// Current assigned tier.
    pub tier: StorageTier,
}

impl AssetDescriptor {
    /// Create a new asset descriptor defaulting to Hot tier.
    #[must_use]
    pub fn new(path: PathBuf, size_bytes: u64, days_since_access: u32) -> Self {
        Self {
            path,
            size_bytes,
            days_since_access,
            pinned: false,
            tier: StorageTier::Hot,
        }
    }
}

// ---------------------------------------------------------------------------
// ColdStorageManager
// ---------------------------------------------------------------------------

/// Manages a catalogue of assets and applies tiering policies to them.
///
/// This is a pure planning engine — it produces tier-change recommendations
/// without performing actual data movement.
pub struct ColdStorageManager {
    policy: ColdStoragePolicy,
    assets: Vec<AssetDescriptor>,
}

impl ColdStorageManager {
    /// Create a manager with the given policy.
    #[must_use]
    pub fn new(policy: ColdStoragePolicy) -> Self {
        Self {
            policy,
            assets: Vec::new(),
        }
    }

    /// Register an asset with the manager.
    pub fn register(&mut self, asset: AssetDescriptor) {
        self.assets.push(asset);
    }

    /// Compute the recommended tier for a single asset under the current policy.
    #[must_use]
    pub fn recommended_tier(&self, asset: &AssetDescriptor) -> StorageTier {
        if self.policy.respect_pinned && asset.pinned {
            return StorageTier::Hot;
        }
        if asset.size_bytes < self.policy.min_size_bytes {
            return asset.tier; // too small to bother tiering
        }
        let days = asset.days_since_access;
        if days >= self.policy.frozen_after_days {
            StorageTier::Frozen
        } else if days >= self.policy.cold_after_days {
            StorageTier::Cold
        } else if days >= self.policy.warm_after_days {
            StorageTier::Warm
        } else {
            StorageTier::Hot
        }
    }

    /// Apply the policy to all registered assets and return a list of
    /// `(path, old_tier, new_tier)` for assets that need to move.
    #[must_use]
    pub fn apply_policy(&self) -> Vec<(PathBuf, StorageTier, StorageTier)> {
        self.assets
            .iter()
            .filter_map(|a| {
                let new_tier = self.recommended_tier(a);
                if new_tier != a.tier {
                    Some((a.path.clone(), a.tier, new_tier))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Estimate monthly storage cost for all registered assets (USD, rough).
    ///
    /// Uses $0.023 / GiB-month as the Hot baseline.
    #[must_use]
    pub fn estimated_monthly_cost_usd(&self) -> f64 {
        const HOT_COST_PER_GIB: f64 = 0.023;
        self.assets
            .iter()
            .map(|a| {
                let gib = a.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                let factor = self.recommended_tier(a).relative_cost() as f64;
                gib * HOT_COST_PER_GIB * factor
            })
            .sum()
    }

    /// Return the total number of registered assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Return assets currently assigned to a given tier.
    #[must_use]
    pub fn assets_in_tier(&self, tier: StorageTier) -> Vec<&AssetDescriptor> {
        self.assets.iter().filter(|a| a.tier == tier).collect()
    }

    /// Return the policy in use.
    #[must_use]
    pub fn policy(&self) -> &ColdStoragePolicy {
        &self.policy
    }

    /// Return a mutable reference to a registered asset by path.
    pub fn find_asset_mut(&mut self, path: &std::path::Path) -> Option<&mut AssetDescriptor> {
        self.assets.iter_mut().find(|a| a.path == path)
    }
}

// ---------------------------------------------------------------------------
// TierTransition
// ---------------------------------------------------------------------------

/// Describes a tier transition event: an asset moving between tiers.
#[derive(Debug, Clone)]
pub struct TierTransition {
    /// Path of the asset.
    pub path: PathBuf,
    /// Original tier before the transition.
    pub from_tier: StorageTier,
    /// Target tier after the transition.
    pub to_tier: StorageTier,
    /// Direction of the transition.
    pub direction: TransitionDirection,
    /// Estimated time for the transition to complete (seconds).
    pub estimated_duration_secs: u64,
    /// Priority of the transition (lower = higher priority).
    pub priority: u32,
}

/// Direction of a tier transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionDirection {
    /// Moving to a colder (cheaper, slower) tier.
    CoolDown,
    /// Moving to a warmer (more expensive, faster) tier.
    WarmUp,
}

impl std::fmt::Display for TransitionDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoolDown => write!(f, "Cool-Down"),
            Self::WarmUp => write!(f, "Warm-Up"),
        }
    }
}

// ---------------------------------------------------------------------------
// WarmUpRequest
// ---------------------------------------------------------------------------

/// A request to warm up (retrieve) an asset from cold or frozen storage.
#[derive(Debug, Clone)]
pub struct WarmUpRequest {
    /// Path of the asset to warm up.
    pub path: PathBuf,
    /// Target tier to warm up to.
    pub target_tier: StorageTier,
    /// Priority (0 = highest).
    pub priority: u32,
    /// Reason for the warm-up request.
    pub reason: String,
}

// ---------------------------------------------------------------------------
// TransitionResult
// ---------------------------------------------------------------------------

/// Result of executing a tier transition.
#[derive(Debug, Clone)]
pub struct TransitionResult {
    /// The transition that was executed.
    pub transition: TierTransition,
    /// Whether the transition was successful.
    pub success: bool,
    /// Error message if the transition failed.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// TierTransitionManager
// ---------------------------------------------------------------------------

/// Manages automated warm-up and cool-down transitions between storage tiers.
///
/// This is a planning and tracking engine. It computes which assets need to
/// transition between tiers and manages the queue of pending transitions.
/// Actual data movement is delegated to the caller.
pub struct TierTransitionManager {
    storage_manager: ColdStorageManager,
    pending_cool_downs: Vec<TierTransition>,
    pending_warm_ups: Vec<TierTransition>,
    completed: Vec<TransitionResult>,
}

impl TierTransitionManager {
    /// Create a new transition manager wrapping an existing storage manager.
    #[must_use]
    pub fn new(storage_manager: ColdStorageManager) -> Self {
        Self {
            storage_manager,
            pending_cool_downs: Vec::new(),
            pending_warm_ups: Vec::new(),
            completed: Vec::new(),
        }
    }

    /// Register an asset with the underlying storage manager.
    pub fn register_asset(&mut self, asset: AssetDescriptor) {
        self.storage_manager.register(asset);
    }

    /// Compute all necessary cool-down transitions based on the current policy.
    ///
    /// This scans all registered assets and identifies those that should be
    /// moved to a colder tier. The transitions are added to the pending queue.
    pub fn plan_cool_downs(&mut self) {
        let moves = self.storage_manager.apply_policy();
        self.pending_cool_downs.clear();

        for (path, old_tier, new_tier) in moves {
            // Only process cool-down moves (to a colder tier)
            if new_tier > old_tier {
                let estimated_duration = Self::estimate_transition_duration(old_tier, new_tier);
                let priority = Self::compute_priority(old_tier, new_tier);
                self.pending_cool_downs.push(TierTransition {
                    path,
                    from_tier: old_tier,
                    to_tier: new_tier,
                    direction: TransitionDirection::CoolDown,
                    estimated_duration_secs: estimated_duration,
                    priority,
                });
            }
        }

        // Sort by priority (lower = higher priority)
        self.pending_cool_downs.sort_by_key(|t| t.priority);
    }

    /// Submit a warm-up request for an asset.
    ///
    /// This creates a pending warm-up transition to bring an asset from a
    /// cold/frozen tier back to a warmer tier for access.
    ///
    /// # Errors
    ///
    /// Returns an error if the asset is not found or the target tier is not
    /// warmer than the current tier.
    pub fn request_warm_up(&mut self, request: WarmUpRequest) -> std::result::Result<(), String> {
        let asset = self
            .storage_manager
            .find_asset_mut(&request.path)
            .ok_or_else(|| format!("Asset not found: {}", request.path.display()))?;

        if request.target_tier >= asset.tier {
            return Err(format!(
                "Target tier {} is not warmer than current tier {}",
                request.target_tier, asset.tier
            ));
        }

        let estimated_duration =
            Self::estimate_transition_duration(asset.tier, request.target_tier);

        self.pending_warm_ups.push(TierTransition {
            path: request.path,
            from_tier: asset.tier,
            to_tier: request.target_tier,
            direction: TransitionDirection::WarmUp,
            estimated_duration_secs: estimated_duration,
            priority: request.priority,
        });

        // Sort warm-ups by priority
        self.pending_warm_ups.sort_by_key(|t| t.priority);

        Ok(())
    }

    /// Execute the next pending cool-down transition.
    ///
    /// In a real system this would initiate data movement; here it updates the
    /// asset's tier and records the result.
    pub fn execute_next_cool_down(&mut self) -> Option<TransitionResult> {
        let transition = self.pending_cool_downs.first()?.clone();
        self.pending_cool_downs.remove(0);

        // Update the asset tier in the storage manager
        let success = if let Some(asset) = self.storage_manager.find_asset_mut(&transition.path) {
            asset.tier = transition.to_tier;
            true
        } else {
            false
        };

        let result = TransitionResult {
            transition,
            success,
            error: if success {
                None
            } else {
                Some("Asset not found during execution".to_string())
            },
        };
        self.completed.push(result.clone());
        Some(result)
    }

    /// Execute the next pending warm-up transition.
    pub fn execute_next_warm_up(&mut self) -> Option<TransitionResult> {
        let transition = self.pending_warm_ups.first()?.clone();
        self.pending_warm_ups.remove(0);

        let success = if let Some(asset) = self.storage_manager.find_asset_mut(&transition.path) {
            asset.tier = transition.to_tier;
            // Reset days_since_access since this is an access event
            asset.days_since_access = 0;
            true
        } else {
            false
        };

        let result = TransitionResult {
            transition,
            success,
            error: if success {
                None
            } else {
                Some("Asset not found during execution".to_string())
            },
        };
        self.completed.push(result.clone());
        Some(result)
    }

    /// Execute all pending transitions (cool-downs first, then warm-ups).
    pub fn execute_all(&mut self) -> Vec<TransitionResult> {
        let mut results = Vec::new();
        while let Some(r) = self.execute_next_cool_down() {
            results.push(r);
        }
        while let Some(r) = self.execute_next_warm_up() {
            results.push(r);
        }
        results
    }

    /// Returns the number of pending cool-down transitions.
    #[must_use]
    pub fn pending_cool_down_count(&self) -> usize {
        self.pending_cool_downs.len()
    }

    /// Returns the number of pending warm-up transitions.
    #[must_use]
    pub fn pending_warm_up_count(&self) -> usize {
        self.pending_warm_ups.len()
    }

    /// Returns a reference to completed transitions.
    #[must_use]
    pub fn completed_transitions(&self) -> &[TransitionResult] {
        &self.completed
    }

    /// Returns a reference to the underlying storage manager.
    #[must_use]
    pub fn storage_manager(&self) -> &ColdStorageManager {
        &self.storage_manager
    }

    /// Estimate transition duration in seconds based on tier distance.
    fn estimate_transition_duration(from: StorageTier, to: StorageTier) -> u64 {
        let tier_value = |t: StorageTier| -> u64 {
            match t {
                StorageTier::Hot => 0,
                StorageTier::Warm => 1,
                StorageTier::Cold => 2,
                StorageTier::Frozen => 3,
            }
        };
        let from_v = tier_value(from);
        let to_v = tier_value(to);
        let distance = if from_v > to_v {
            from_v - to_v
        } else {
            to_v - from_v
        };
        // Base estimate: 60 seconds per tier level crossed
        distance * 60
    }

    /// Compute transition priority. Larger jumps get higher priority (lower number).
    fn compute_priority(from: StorageTier, to: StorageTier) -> u32 {
        let tier_value = |t: StorageTier| -> u32 {
            match t {
                StorageTier::Hot => 0,
                StorageTier::Warm => 1,
                StorageTier::Cold => 2,
                StorageTier::Frozen => 3,
            }
        };
        let from_v = tier_value(from);
        let to_v = tier_value(to);
        // Larger distance = lower priority number = higher priority
        let distance = if from_v > to_v {
            from_v - to_v
        } else {
            to_v - from_v
        };
        10u32.saturating_sub(distance)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> ColdStorageManager {
        ColdStorageManager::new(ColdStoragePolicy::default())
    }

    fn asset(days: u32, mb: u64) -> AssetDescriptor {
        AssetDescriptor::new(
            PathBuf::from(format!("/archive/{days}d_{mb}mb.mxf")),
            mb * 1024 * 1024,
            days,
        )
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(StorageTier::Hot.to_string(), "Hot");
        assert_eq!(StorageTier::Frozen.to_string(), "Frozen");
    }

    #[test]
    fn test_tier_ordering() {
        assert!(StorageTier::Hot < StorageTier::Warm);
        assert!(StorageTier::Warm < StorageTier::Cold);
        assert!(StorageTier::Cold < StorageTier::Frozen);
    }

    #[test]
    fn test_tier_relative_cost() {
        assert!((StorageTier::Hot.relative_cost() - 1.0).abs() < 1e-5);
        assert!(StorageTier::Frozen.relative_cost() < StorageTier::Cold.relative_cost());
    }

    #[test]
    fn test_tier_retrieval_latency_nonempty() {
        for tier in [
            StorageTier::Hot,
            StorageTier::Warm,
            StorageTier::Cold,
            StorageTier::Frozen,
        ] {
            assert!(!tier.retrieval_latency().is_empty());
        }
    }

    #[test]
    fn test_policy_default_valid() {
        assert!(ColdStoragePolicy::default().validate().is_ok());
    }

    #[test]
    fn test_policy_bad_warm_cold_order() {
        let p = ColdStoragePolicy {
            warm_after_days: 200,
            cold_after_days: 100,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_policy_bad_cold_frozen_order() {
        let p = ColdStoragePolicy {
            cold_after_days: 400,
            frozen_after_days: 300,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_recommended_tier_hot_recent() {
        let m = default_manager();
        let a = asset(5, 100); // accessed 5 days ago, 100 MiB
        assert_eq!(m.recommended_tier(&a), StorageTier::Hot);
    }

    #[test]
    fn test_recommended_tier_warm() {
        let m = default_manager();
        let a = asset(60, 100); // 60 days, well past warm threshold of 30
        assert_eq!(m.recommended_tier(&a), StorageTier::Warm);
    }

    #[test]
    fn test_recommended_tier_cold() {
        let m = default_manager();
        let a = asset(200, 100); // > 180 days
        assert_eq!(m.recommended_tier(&a), StorageTier::Cold);
    }

    #[test]
    fn test_recommended_tier_frozen() {
        let m = default_manager();
        let a = asset(400, 100); // > 365 days
        assert_eq!(m.recommended_tier(&a), StorageTier::Frozen);
    }

    #[test]
    fn test_recommended_tier_pinned_stays_hot() {
        let m = default_manager();
        let mut a = asset(500, 100);
        a.pinned = true;
        assert_eq!(m.recommended_tier(&a), StorageTier::Hot);
    }

    #[test]
    fn test_recommended_tier_small_file_unchanged() {
        let m = default_manager();
        let small = AssetDescriptor::new(PathBuf::from("/tiny.txt"), 512, 500);
        // Size < min_size_bytes → keep current tier (Hot)
        assert_eq!(m.recommended_tier(&small), StorageTier::Hot);
    }

    #[test]
    fn test_apply_policy_detects_moves() {
        let mut m = default_manager();
        let mut a = asset(400, 100);
        a.tier = StorageTier::Hot; // currently Hot but should be Frozen
        m.register(a);
        let moves = m.apply_policy();
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].2, StorageTier::Frozen);
    }

    #[test]
    fn test_apply_policy_no_move_needed() {
        let mut m = default_manager();
        let mut a = asset(400, 100);
        a.tier = StorageTier::Frozen; // already correct
        m.register(a);
        let moves = m.apply_policy();
        assert!(moves.is_empty());
    }

    #[test]
    fn test_estimated_monthly_cost_zero_assets() {
        let m = default_manager();
        assert!((m.estimated_monthly_cost_usd() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimated_monthly_cost_positive() {
        let mut m = default_manager();
        m.register(asset(5, 1024)); // 1 GiB, Hot
        assert!(m.estimated_monthly_cost_usd() > 0.0);
    }

    #[test]
    fn test_asset_count() {
        let mut m = default_manager();
        assert_eq!(m.asset_count(), 0);
        m.register(asset(10, 50));
        m.register(asset(200, 100));
        assert_eq!(m.asset_count(), 2);
    }

    #[test]
    fn test_assets_in_tier() {
        let mut m = default_manager();
        let mut a = asset(10, 50);
        a.tier = StorageTier::Hot;
        let mut b = asset(200, 100);
        b.tier = StorageTier::Cold;
        m.register(a);
        m.register(b);
        assert_eq!(m.assets_in_tier(StorageTier::Hot).len(), 1);
        assert_eq!(m.assets_in_tier(StorageTier::Cold).len(), 1);
        assert!(m.assets_in_tier(StorageTier::Frozen).is_empty());
    }

    // ── TierTransitionManager tests ─────────────────────────────

    #[test]
    fn test_transition_direction_display() {
        assert_eq!(TransitionDirection::CoolDown.to_string(), "Cool-Down");
        assert_eq!(TransitionDirection::WarmUp.to_string(), "Warm-Up");
    }

    #[test]
    fn test_plan_cool_downs_identifies_assets() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(400, 100);
        a.tier = StorageTier::Hot; // Should be Frozen
        mgr.register_asset(a);

        mgr.plan_cool_downs();
        assert_eq!(mgr.pending_cool_down_count(), 1);
    }

    #[test]
    fn test_plan_cool_downs_no_moves_needed() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(400, 100);
        a.tier = StorageTier::Frozen; // Already correct
        mgr.register_asset(a);

        mgr.plan_cool_downs();
        assert_eq!(mgr.pending_cool_down_count(), 0);
    }

    #[test]
    fn test_execute_cool_down() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(400, 100);
        a.tier = StorageTier::Hot;
        mgr.register_asset(a);

        mgr.plan_cool_downs();
        let result = mgr.execute_next_cool_down();
        assert!(result.is_some());
        let result = result.expect("operation should succeed");
        assert!(result.success);
        assert_eq!(result.transition.direction, TransitionDirection::CoolDown);
        assert_eq!(result.transition.to_tier, StorageTier::Frozen);
    }

    #[test]
    fn test_request_warm_up() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(400, 100);
        a.tier = StorageTier::Frozen;
        mgr.register_asset(a);

        let request = WarmUpRequest {
            path: PathBuf::from("/archive/400d_100mb.mxf"),
            target_tier: StorageTier::Hot,
            priority: 0,
            reason: "Urgent access request".to_string(),
        };
        assert!(mgr.request_warm_up(request).is_ok());
        assert_eq!(mgr.pending_warm_up_count(), 1);
    }

    #[test]
    fn test_request_warm_up_invalid_tier() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(10, 100);
        a.tier = StorageTier::Hot;
        mgr.register_asset(a);

        let request = WarmUpRequest {
            path: PathBuf::from("/archive/10d_100mb.mxf"),
            target_tier: StorageTier::Cold, // Colder, not warmer
            priority: 0,
            reason: "Should fail".to_string(),
        };
        assert!(mgr.request_warm_up(request).is_err());
    }

    #[test]
    fn test_request_warm_up_asset_not_found() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let request = WarmUpRequest {
            path: PathBuf::from("/nonexistent.mxf"),
            target_tier: StorageTier::Hot,
            priority: 0,
            reason: "Does not exist".to_string(),
        };
        assert!(mgr.request_warm_up(request).is_err());
    }

    #[test]
    fn test_execute_warm_up_resets_access() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(500, 100);
        a.tier = StorageTier::Frozen;
        mgr.register_asset(a);

        let request = WarmUpRequest {
            path: PathBuf::from("/archive/500d_100mb.mxf"),
            target_tier: StorageTier::Hot,
            priority: 0,
            reason: "Research access".to_string(),
        };
        mgr.request_warm_up(request)
            .expect("operation should succeed");

        let result = mgr
            .execute_next_warm_up()
            .expect("operation should succeed");
        assert!(result.success);
        assert_eq!(result.transition.to_tier, StorageTier::Hot);

        // After warm-up, the asset should be in Hot tier
        let asset = mgr.storage_manager().assets_in_tier(StorageTier::Hot);
        assert!(!asset.is_empty());
    }

    #[test]
    fn test_execute_all_transitions() {
        let mut mgr = TierTransitionManager::new(default_manager());

        // Asset that needs to cool down
        let mut a = asset(400, 100);
        a.tier = StorageTier::Hot;
        mgr.register_asset(a);

        // Asset that is already cold, will be warmed up
        let mut b = asset(200, 200);
        b.tier = StorageTier::Cold;
        mgr.register_asset(b);

        // Plan cool-downs
        mgr.plan_cool_downs();

        // Request warm-up
        let request = WarmUpRequest {
            path: PathBuf::from("/archive/200d_200mb.mxf"),
            target_tier: StorageTier::Hot,
            priority: 0,
            reason: "Needed for editing".to_string(),
        };
        mgr.request_warm_up(request)
            .expect("operation should succeed");

        let results = mgr.execute_all();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_completed_transitions_tracking() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(400, 100);
        a.tier = StorageTier::Hot;
        mgr.register_asset(a);

        mgr.plan_cool_downs();
        mgr.execute_next_cool_down();

        assert_eq!(mgr.completed_transitions().len(), 1);
        assert!(mgr.completed_transitions()[0].success);
    }

    #[test]
    fn test_transition_estimated_duration() {
        let mut mgr = TierTransitionManager::new(default_manager());
        let mut a = asset(400, 100);
        a.tier = StorageTier::Hot; // Will need to go to Frozen (3 tiers)
        mgr.register_asset(a);

        mgr.plan_cool_downs();
        assert_eq!(mgr.pending_cool_down_count(), 1);
        // Hot -> Frozen = 3 tiers * 60s = 180s
        // Accessing through pending transitions
        let transitions = &mgr.pending_cool_downs;
        assert_eq!(transitions[0].estimated_duration_secs, 180);
    }

    #[test]
    fn test_multiple_cool_downs_sorted_by_priority() {
        let mut mgr = TierTransitionManager::new(default_manager());

        // Asset going Hot -> Warm (1 tier, lower priority)
        let mut a = asset(60, 100);
        a.tier = StorageTier::Hot;
        mgr.register_asset(a);

        // Asset going Hot -> Frozen (3 tiers, higher priority)
        let mut b = asset(400, 200);
        b.tier = StorageTier::Hot;
        mgr.register_asset(b);

        mgr.plan_cool_downs();
        assert_eq!(mgr.pending_cool_down_count(), 2);

        // The Frozen transition should come first (lower priority number)
        let first = mgr
            .execute_next_cool_down()
            .expect("operation should succeed");
        assert_eq!(first.transition.to_tier, StorageTier::Frozen);
    }
}
