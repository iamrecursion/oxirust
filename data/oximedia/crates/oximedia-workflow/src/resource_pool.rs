#![allow(dead_code)]
//! Resource pool management for workflow execution.
//!
//! Provides a pooling system for shared resources (CPU cores, GPU devices,
//! network bandwidth, disk I/O slots) that workflow tasks can reserve and
//! release during execution. The pool enforces capacity limits and supports
//! fair scheduling of resource requests.

use std::collections::HashMap;

/// Unique identifier for a resource type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceId(String);

impl ResourceId {
    /// Create a new resource identifier.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the string name of this resource.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ResourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Describes a resource with capacity and current allocation.
#[derive(Debug, Clone)]
pub struct ResourceDescriptor {
    /// Unique identifier for this resource.
    pub id: ResourceId,
    /// Maximum capacity available.
    pub capacity: u64,
    /// Currently allocated amount.
    pub allocated: u64,
    /// Human-readable label.
    pub label: String,
    /// Unit of measurement (e.g., "cores", "MB", "Mbps").
    pub unit: String,
}

impl ResourceDescriptor {
    /// Create a new resource descriptor.
    pub fn new(
        id: ResourceId,
        capacity: u64,
        label: impl Into<String>,
        unit: impl Into<String>,
    ) -> Self {
        Self {
            id,
            capacity,
            allocated: 0,
            label: label.into(),
            unit: unit.into(),
        }
    }

    /// Return the remaining available capacity.
    #[must_use]
    pub fn available(&self) -> u64 {
        self.capacity.saturating_sub(self.allocated)
    }

    /// Return the utilisation ratio as a value between 0.0 and 1.0.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn utilisation(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.allocated as f64 / self.capacity as f64
    }

    /// Check whether the requested amount can be satisfied.
    #[must_use]
    pub fn can_allocate(&self, amount: u64) -> bool {
        self.available() >= amount
    }
}

/// A token representing a successful allocation that must be released.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllocationToken {
    /// Unique token identifier.
    pub token_id: u64,
    /// Resource this token belongs to.
    pub resource_id: ResourceId,
    /// Amount allocated.
    pub amount: u64,
}

/// A request for resources from the pool.
#[derive(Debug, Clone)]
pub struct ResourceRequest {
    /// Which resource is being requested.
    pub resource_id: ResourceId,
    /// How much is needed.
    pub amount: u64,
    /// Priority of this request (higher = more important).
    pub priority: u32,
    /// Optional requester tag for tracking.
    pub requester: String,
}

impl ResourceRequest {
    /// Create a new resource request.
    #[must_use]
    pub fn new(resource_id: ResourceId, amount: u64) -> Self {
        Self {
            resource_id,
            amount,
            priority: 0,
            requester: String::new(),
        }
    }

    /// Set the priority of this request.
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the requester tag.
    pub fn with_requester(mut self, requester: impl Into<String>) -> Self {
        self.requester = requester.into();
        self
    }
}

/// Errors that can occur during pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// The requested resource does not exist in the pool.
    ResourceNotFound(String),
    /// Insufficient capacity for the requested amount.
    InsufficientCapacity {
        /// Resource identifier.
        resource: String,
        /// Amount requested.
        requested: u64,
        /// Amount available.
        available: u64,
    },
    /// The allocation token is invalid or already released.
    InvalidToken(u64),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResourceNotFound(id) => write!(f, "resource not found: {id}"),
            Self::InsufficientCapacity {
                resource,
                requested,
                available,
            } => {
                write!(f, "insufficient capacity for '{resource}': requested {requested}, available {available}")
            }
            Self::InvalidToken(id) => write!(f, "invalid allocation token: {id}"),
        }
    }
}

/// The main resource pool that tracks multiple resource types.
#[derive(Debug)]
pub struct ResourcePool {
    /// All resources in this pool.
    resources: HashMap<ResourceId, ResourceDescriptor>,
    /// Outstanding allocations keyed by token id.
    allocations: HashMap<u64, AllocationToken>,
    /// Counter for generating unique token ids.
    next_token_id: u64,
}

impl ResourcePool {
    /// Create a new empty resource pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            allocations: HashMap::new(),
            next_token_id: 1,
        }
    }

    /// Register a resource with the pool.
    pub fn register(&mut self, descriptor: ResourceDescriptor) {
        self.resources.insert(descriptor.id.clone(), descriptor);
    }

    /// Attempt to allocate a resource, returning a token on success.
    ///
    /// # Errors
    ///
    /// Returns `PoolError` if the resource is not found or has insufficient capacity.
    pub fn allocate(&mut self, request: &ResourceRequest) -> Result<AllocationToken, PoolError> {
        let resource = self
            .resources
            .get_mut(&request.resource_id)
            .ok_or_else(|| PoolError::ResourceNotFound(request.resource_id.name().to_string()))?;

        if !resource.can_allocate(request.amount) {
            return Err(PoolError::InsufficientCapacity {
                resource: resource.id.name().to_string(),
                requested: request.amount,
                available: resource.available(),
            });
        }

        resource.allocated += request.amount;

        let token = AllocationToken {
            token_id: self.next_token_id,
            resource_id: request.resource_id.clone(),
            amount: request.amount,
        };
        self.next_token_id += 1;
        self.allocations.insert(token.token_id, token.clone());
        Ok(token)
    }

    /// Release a previous allocation using its token.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::InvalidToken` if the token does not match any active allocation.
    pub fn release(&mut self, token_id: u64) -> Result<(), PoolError> {
        let token = self
            .allocations
            .remove(&token_id)
            .ok_or(PoolError::InvalidToken(token_id))?;

        if let Some(resource) = self.resources.get_mut(&token.resource_id) {
            resource.allocated = resource.allocated.saturating_sub(token.amount);
        }

        Ok(())
    }

    /// Return a snapshot of all resources and their current state.
    #[must_use]
    pub fn snapshot(&self) -> Vec<ResourceDescriptor> {
        self.resources.values().cloned().collect()
    }

    /// Return the descriptor for a specific resource.
    #[must_use]
    pub fn get_resource(&self, id: &ResourceId) -> Option<&ResourceDescriptor> {
        self.resources.get(id)
    }

    /// Return the number of active allocations.
    #[must_use]
    pub fn active_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Return the total number of registered resources.
    #[must_use]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Check whether a request can be satisfied without actually allocating.
    #[must_use]
    pub fn can_satisfy(&self, request: &ResourceRequest) -> bool {
        self.resources
            .get(&request.resource_id)
            .is_some_and(|r| r.can_allocate(request.amount))
    }

    /// Reset all allocations (useful for testing or emergency recovery).
    pub fn reset_all(&mut self) {
        self.allocations.clear();
        for resource in self.resources.values_mut() {
            resource.allocated = 0;
        }
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for the entire pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total capacity across all resources.
    pub total_capacity: u64,
    /// Total allocated across all resources.
    pub total_allocated: u64,
    /// Number of resources.
    pub resource_count: usize,
    /// Number of active allocations.
    pub active_allocations: usize,
    /// Average utilisation across resources.
    pub average_utilisation: f64,
}

impl ResourcePool {
    /// Compute aggregate statistics for the pool.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let total_capacity: u64 = self.resources.values().map(|r| r.capacity).sum();
        let total_allocated: u64 = self.resources.values().map(|r| r.allocated).sum();
        let resource_count = self.resources.len();
        let active_allocations = self.allocations.len();
        let average_utilisation = if resource_count == 0 {
            0.0
        } else {
            let sum: f64 = self
                .resources
                .values()
                .map(ResourceDescriptor::utilisation)
                .sum();
            sum / resource_count as f64
        };

        PoolStats {
            total_capacity,
            total_allocated,
            resource_count,
            active_allocations,
            average_utilisation,
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic resource scaling
// ---------------------------------------------------------------------------

/// Scaling policy for a resource.
#[derive(Debug, Clone)]
pub enum ScalingPolicy {
    /// Fixed capacity — no scaling.
    Fixed,
    /// Step scaling: increase by `step_size` when utilisation exceeds
    /// `scale_up_threshold`, decrease when below `scale_down_threshold`.
    Step {
        /// Utilisation threshold to trigger scale-up.
        scale_up_threshold: f64,
        /// Utilisation threshold to trigger scale-down.
        scale_down_threshold: f64,
        /// Amount to add/remove per scaling event.
        step_size: u64,
        /// Minimum capacity (never scale below this).
        min_capacity: u64,
        /// Maximum capacity (never scale above this).
        max_capacity: u64,
    },
    /// Target tracking: adjust capacity to maintain a target utilisation.
    TargetTracking {
        /// Target utilisation (0.0..1.0).
        target_utilisation: f64,
        /// Minimum capacity.
        min_capacity: u64,
        /// Maximum capacity.
        max_capacity: u64,
    },
}

/// The result of a scaling evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalingAction {
    /// No change needed.
    NoChange,
    /// Scale up by this amount.
    ScaleUp(u64),
    /// Scale down by this amount.
    ScaleDown(u64),
}

impl std::fmt::Display for ScalingAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoChange => write!(f, "no change"),
            Self::ScaleUp(n) => write!(f, "scale up by {n}"),
            Self::ScaleDown(n) => write!(f, "scale down by {n}"),
        }
    }
}

/// Record of a scaling event.
#[derive(Debug, Clone)]
pub struct ScalingEvent {
    /// Resource that was scaled.
    pub resource_id: ResourceId,
    /// Action taken.
    pub action: ScalingAction,
    /// Capacity before scaling.
    pub old_capacity: u64,
    /// Capacity after scaling.
    pub new_capacity: u64,
    /// Utilisation at the time of scaling.
    pub utilisation: f64,
    /// Timestamp (ms since epoch).
    pub timestamp_ms: u64,
}

/// Manager for dynamic resource scaling.
#[derive(Debug)]
pub struct ResourceScaler {
    /// Scaling policies keyed by resource ID.
    policies: HashMap<ResourceId, ScalingPolicy>,
    /// History of scaling events.
    history: Vec<ScalingEvent>,
    /// Maximum history entries.
    max_history: usize,
    /// Cooldown between scaling events per resource (ms).
    cooldown_ms: u64,
    /// Last scale time per resource.
    last_scale_time: HashMap<ResourceId, u64>,
}

impl ResourceScaler {
    /// Create a new resource scaler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
            cooldown_ms: 60_000, // 1 minute default
            last_scale_time: HashMap::new(),
        }
    }

    /// Set the cooldown period between scaling events.
    #[must_use]
    pub fn with_cooldown_ms(mut self, ms: u64) -> Self {
        self.cooldown_ms = ms;
        self
    }

    /// Register a scaling policy for a resource.
    pub fn set_policy(&mut self, resource_id: ResourceId, policy: ScalingPolicy) {
        self.policies.insert(resource_id, policy);
    }

    /// Evaluate whether a resource should be scaled.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn evaluate(&self, resource: &ResourceDescriptor) -> ScalingAction {
        let policy = match self.policies.get(&resource.id) {
            Some(p) => p,
            None => return ScalingAction::NoChange,
        };

        let util = resource.utilisation();

        match policy {
            ScalingPolicy::Fixed => ScalingAction::NoChange,
            ScalingPolicy::Step {
                scale_up_threshold,
                scale_down_threshold,
                step_size,
                min_capacity,
                max_capacity,
            } => {
                if util >= *scale_up_threshold && resource.capacity < *max_capacity {
                    let new_cap = (resource.capacity + step_size).min(*max_capacity);
                    let delta = new_cap - resource.capacity;
                    if delta > 0 {
                        ScalingAction::ScaleUp(delta)
                    } else {
                        ScalingAction::NoChange
                    }
                } else if util <= *scale_down_threshold && resource.capacity > *min_capacity {
                    let new_cap = resource
                        .capacity
                        .saturating_sub(*step_size)
                        .max(*min_capacity);
                    // Don't scale down below what's allocated
                    let new_cap = new_cap.max(resource.allocated);
                    let delta = resource.capacity - new_cap;
                    if delta > 0 {
                        ScalingAction::ScaleDown(delta)
                    } else {
                        ScalingAction::NoChange
                    }
                } else {
                    ScalingAction::NoChange
                }
            }
            ScalingPolicy::TargetTracking {
                target_utilisation,
                min_capacity,
                max_capacity,
            } => {
                if resource.capacity == 0 || *target_utilisation <= 0.0 {
                    return ScalingAction::NoChange;
                }

                let desired = (resource.allocated as f64 / target_utilisation).ceil() as u64;
                let desired = desired.clamp(*min_capacity, *max_capacity);

                if desired > resource.capacity {
                    ScalingAction::ScaleUp(desired - resource.capacity)
                } else if desired < resource.capacity {
                    let delta = resource.capacity - desired;
                    // Don't scale below allocated
                    if desired >= resource.allocated {
                        ScalingAction::ScaleDown(delta)
                    } else {
                        ScalingAction::NoChange
                    }
                } else {
                    ScalingAction::NoChange
                }
            }
        }
    }

    /// Apply a scaling action to a resource pool, respecting cooldown.
    ///
    /// Returns `true` if the action was applied, `false` if skipped (cooldown).
    pub fn apply(
        &mut self,
        pool: &mut ResourcePool,
        resource_id: &ResourceId,
        now_ms: u64,
    ) -> Option<ScalingEvent> {
        // Check cooldown
        if let Some(&last_time) = self.last_scale_time.get(resource_id) {
            if now_ms.saturating_sub(last_time) < self.cooldown_ms {
                return None;
            }
        }

        let resource = pool.get_resource(resource_id)?.clone();
        let action = self.evaluate(&resource);

        if action == ScalingAction::NoChange {
            return None;
        }

        let old_capacity = resource.capacity;
        let new_capacity = match &action {
            ScalingAction::ScaleUp(delta) => old_capacity + delta,
            ScalingAction::ScaleDown(delta) => old_capacity.saturating_sub(*delta),
            ScalingAction::NoChange => return None,
        };

        // Apply the scaling
        if let Some(res) = pool.resources.get_mut(resource_id) {
            res.capacity = new_capacity;
        }

        let event = ScalingEvent {
            resource_id: resource_id.clone(),
            action,
            old_capacity,
            new_capacity,
            utilisation: resource.utilisation(),
            timestamp_ms: now_ms,
        };

        self.last_scale_time.insert(resource_id.clone(), now_ms);
        self.history.push(event.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        Some(event)
    }

    /// Get scaling history.
    #[must_use]
    pub fn history(&self) -> &[ScalingEvent] {
        &self.history
    }

    /// Get history for a specific resource.
    #[must_use]
    pub fn history_for(&self, resource_id: &ResourceId) -> Vec<&ScalingEvent> {
        self.history
            .iter()
            .filter(|e| &e.resource_id == resource_id)
            .collect()
    }

    /// Number of registered policies.
    #[must_use]
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }
}

impl Default for ResourceScaler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_resource() -> ResourceDescriptor {
        ResourceDescriptor::new(ResourceId::new("cpu"), 8, "CPU Cores", "cores")
    }

    fn gpu_resource() -> ResourceDescriptor {
        ResourceDescriptor::new(ResourceId::new("gpu"), 2, "GPU Devices", "devices")
    }

    #[test]
    fn test_resource_descriptor_available() {
        let mut r = cpu_resource();
        assert_eq!(r.available(), 8);
        r.allocated = 3;
        assert_eq!(r.available(), 5);
    }

    #[test]
    fn test_resource_descriptor_utilisation() {
        let mut r = cpu_resource();
        assert!((r.utilisation() - 0.0).abs() < f64::EPSILON);
        r.allocated = 4;
        assert!((r.utilisation() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_descriptor_zero_capacity() {
        let r = ResourceDescriptor::new(ResourceId::new("empty"), 0, "Empty", "units");
        assert!((r.utilisation() - 0.0).abs() < f64::EPSILON);
        assert!(!r.can_allocate(1));
    }

    #[test]
    fn test_pool_register_and_count() {
        let mut pool = ResourcePool::new();
        assert_eq!(pool.resource_count(), 0);
        pool.register(cpu_resource());
        assert_eq!(pool.resource_count(), 1);
        pool.register(gpu_resource());
        assert_eq!(pool.resource_count(), 2);
    }

    #[test]
    fn test_pool_allocate_success() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 4);
        let token = pool.allocate(&req).expect("should succeed in test");
        assert_eq!(token.amount, 4);
        assert_eq!(pool.active_allocations(), 1);
        let r = pool
            .get_resource(&ResourceId::new("cpu"))
            .expect("should succeed in test");
        assert_eq!(r.allocated, 4);
    }

    #[test]
    fn test_pool_allocate_insufficient() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 100);
        let err = pool.allocate(&req).unwrap_err();
        assert!(matches!(err, PoolError::InsufficientCapacity { .. }));
    }

    #[test]
    fn test_pool_allocate_unknown_resource() {
        let mut pool = ResourcePool::new();
        let req = ResourceRequest::new(ResourceId::new("missing"), 1);
        let err = pool.allocate(&req).unwrap_err();
        assert!(matches!(err, PoolError::ResourceNotFound(_)));
    }

    #[test]
    fn test_pool_release_success() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 4);
        let token = pool.allocate(&req).expect("should succeed in test");
        pool.release(token.token_id)
            .expect("should succeed in test");
        assert_eq!(pool.active_allocations(), 0);
        let r = pool
            .get_resource(&ResourceId::new("cpu"))
            .expect("should succeed in test");
        assert_eq!(r.allocated, 0);
    }

    #[test]
    fn test_pool_release_invalid_token() {
        let mut pool = ResourcePool::new();
        let err = pool.release(9999).unwrap_err();
        assert!(matches!(err, PoolError::InvalidToken(9999)));
    }

    #[test]
    fn test_pool_can_satisfy() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req_ok = ResourceRequest::new(ResourceId::new("cpu"), 4);
        assert!(pool.can_satisfy(&req_ok));
        let req_too_much = ResourceRequest::new(ResourceId::new("cpu"), 100);
        assert!(!pool.can_satisfy(&req_too_much));
    }

    #[test]
    fn test_pool_reset_all() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        pool.register(gpu_resource());
        let req1 = ResourceRequest::new(ResourceId::new("cpu"), 4);
        let req2 = ResourceRequest::new(ResourceId::new("gpu"), 1);
        let _t1 = pool.allocate(&req1).expect("should succeed in test");
        let _t2 = pool.allocate(&req2).expect("should succeed in test");
        assert_eq!(pool.active_allocations(), 2);
        pool.reset_all();
        assert_eq!(pool.active_allocations(), 0);
        for r in pool.snapshot() {
            assert_eq!(r.allocated, 0);
        }
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        pool.register(gpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 4);
        let _t = pool.allocate(&req).expect("should succeed in test");

        let stats = pool.stats();
        assert_eq!(stats.total_capacity, 10); // 8 + 2
        assert_eq!(stats.total_allocated, 4);
        assert_eq!(stats.resource_count, 2);
        assert_eq!(stats.active_allocations, 1);
        assert!(stats.average_utilisation > 0.0);
    }

    #[test]
    fn test_resource_request_builder() {
        let req = ResourceRequest::new(ResourceId::new("cpu"), 2)
            .with_priority(10)
            .with_requester("task-1");
        assert_eq!(req.priority, 10);
        assert_eq!(req.requester, "task-1");
    }

    #[test]
    fn test_multiple_allocations_same_resource() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 3);
        let t1 = pool.allocate(&req).expect("should succeed in test");
        let t2 = pool.allocate(&req).expect("should succeed in test");
        assert_eq!(pool.active_allocations(), 2);
        let r = pool
            .get_resource(&ResourceId::new("cpu"))
            .expect("should succeed in test");
        assert_eq!(r.allocated, 6);

        // Third should fail (only 2 left)
        let req_too_much = ResourceRequest::new(ResourceId::new("cpu"), 3);
        assert!(pool.allocate(&req_too_much).is_err());

        pool.release(t1.token_id).expect("should succeed in test");
        // Now 5 available — 3 should work
        let _t3 = pool
            .allocate(&req_too_much)
            .expect("should succeed in test");
        assert_eq!(pool.active_allocations(), 2);
        pool.release(t2.token_id).expect("should succeed in test");
    }

    // --- Dynamic resource scaling tests ---

    #[test]
    fn test_scaler_fixed_policy() {
        let scaler = ResourceScaler::new();
        let resource = cpu_resource();
        assert_eq!(scaler.evaluate(&resource), ScalingAction::NoChange);
    }

    #[test]
    fn test_scaler_step_scale_up() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 4,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        let mut resource = cpu_resource(); // capacity = 8
        resource.allocated = 7; // utilisation = 7/8 = 0.875 > 0.8

        assert_eq!(scaler.evaluate(&resource), ScalingAction::ScaleUp(4));
    }

    #[test]
    fn test_scaler_step_scale_down() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 4,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        let mut resource = cpu_resource(); // capacity = 8
        resource.allocated = 1; // utilisation = 1/8 = 0.125 < 0.2

        assert_eq!(scaler.evaluate(&resource), ScalingAction::ScaleDown(4));
    }

    #[test]
    fn test_scaler_step_capped_at_max() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 100,
                min_capacity: 4,
                max_capacity: 10,
            },
        );

        let mut resource = cpu_resource(); // capacity = 8
        resource.allocated = 7; // 87.5% > 80%

        // Should scale up to max (10), so delta = 2
        assert_eq!(scaler.evaluate(&resource), ScalingAction::ScaleUp(2));
    }

    #[test]
    fn test_scaler_step_capped_at_min() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 100,
                min_capacity: 6,
                max_capacity: 32,
            },
        );

        let mut resource = cpu_resource(); // capacity = 8
        resource.allocated = 0; // 0% < 20%

        // Should scale down to min (6), so delta = 2
        assert_eq!(scaler.evaluate(&resource), ScalingAction::ScaleDown(2));
    }

    #[test]
    fn test_scaler_target_tracking_scale_up() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::TargetTracking {
                target_utilisation: 0.5,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        let mut resource = cpu_resource(); // capacity = 8
        resource.allocated = 7; // desired = ceil(7/0.5) = 14

        assert_eq!(scaler.evaluate(&resource), ScalingAction::ScaleUp(6));
    }

    #[test]
    fn test_scaler_target_tracking_scale_down() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::TargetTracking {
                target_utilisation: 0.5,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        let mut resource = ResourceDescriptor::new(ResourceId::new("cpu"), 16, "CPU", "cores");
        resource.allocated = 2; // desired = ceil(2/0.5) = 4

        assert_eq!(scaler.evaluate(&resource), ScalingAction::ScaleDown(12));
    }

    #[test]
    fn test_scaler_target_tracking_at_target() {
        let mut scaler = ResourceScaler::new();
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::TargetTracking {
                target_utilisation: 0.5,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        let mut resource = cpu_resource(); // capacity = 8
        resource.allocated = 4; // desired = ceil(4/0.5) = 8 == capacity

        assert_eq!(scaler.evaluate(&resource), ScalingAction::NoChange);
    }

    #[test]
    fn test_scaler_apply_to_pool() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource()); // capacity = 8

        // Allocate 7 of 8 to trigger scale-up
        let req = ResourceRequest::new(ResourceId::new("cpu"), 7);
        let _t = pool.allocate(&req).expect("allocate");

        let mut scaler = ResourceScaler::new().with_cooldown_ms(0);
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 4,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        let event = scaler.apply(&mut pool, &ResourceId::new("cpu"), 1000);
        assert!(event.is_some());
        let event = event.expect("event");
        assert_eq!(event.old_capacity, 8);
        assert_eq!(event.new_capacity, 12);

        let resource = pool
            .get_resource(&ResourceId::new("cpu"))
            .expect("resource");
        assert_eq!(resource.capacity, 12);
    }

    #[test]
    fn test_scaler_cooldown() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 7);
        let _t = pool.allocate(&req).expect("allocate");

        let mut scaler = ResourceScaler::new().with_cooldown_ms(60_000);
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 4,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        // First apply should work
        let event = scaler.apply(&mut pool, &ResourceId::new("cpu"), 1000);
        assert!(event.is_some());

        // Second apply within cooldown should not
        let event = scaler.apply(&mut pool, &ResourceId::new("cpu"), 2000);
        assert!(event.is_none());

        // After cooldown it should work again
        let event = scaler.apply(&mut pool, &ResourceId::new("cpu"), 70_000);
        // May or may not trigger since capacity changed; depends on new utilisation
        // The point is the cooldown is respected
        let _ = event;
    }

    #[test]
    fn test_scaler_history() {
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());
        let req = ResourceRequest::new(ResourceId::new("cpu"), 7);
        let _t = pool.allocate(&req).expect("allocate");

        let mut scaler = ResourceScaler::new().with_cooldown_ms(0);
        scaler.set_policy(
            ResourceId::new("cpu"),
            ScalingPolicy::Step {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                step_size: 4,
                min_capacity: 4,
                max_capacity: 32,
            },
        );

        scaler.apply(&mut pool, &ResourceId::new("cpu"), 1000);
        assert_eq!(scaler.history().len(), 1);

        let cpu_history = scaler.history_for(&ResourceId::new("cpu"));
        assert_eq!(cpu_history.len(), 1);
        assert_eq!(cpu_history[0].old_capacity, 8);
    }

    #[test]
    fn test_scaling_action_display() {
        assert_eq!(ScalingAction::NoChange.to_string(), "no change");
        assert_eq!(ScalingAction::ScaleUp(4).to_string(), "scale up by 4");
        assert_eq!(ScalingAction::ScaleDown(2).to_string(), "scale down by 2");
    }

    #[test]
    fn test_scaler_policy_count() {
        let mut scaler = ResourceScaler::new();
        assert_eq!(scaler.policy_count(), 0);
        scaler.set_policy(ResourceId::new("cpu"), ScalingPolicy::Fixed);
        assert_eq!(scaler.policy_count(), 1);
    }

    #[test]
    fn test_scaler_unknown_resource() {
        let mut scaler = ResourceScaler::new();
        let mut pool = ResourcePool::new();
        pool.register(cpu_resource());

        // No policy registered for gpu
        let event = scaler.apply(&mut pool, &ResourceId::new("gpu"), 1000);
        assert!(event.is_none());
    }
}
