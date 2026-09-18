//! Advanced Scaling Module
//!
//! Provides advanced scaling mechanisms for distributed optimization systems.

use crate::error::{MetricsError, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Advanced scaling system
#[derive(Debug, Clone)]
pub struct AdvancedScalingManager {
    node_id: String,
    scaling_policies: HashMap<String, ScalingPolicy>,
    resource_monitors: HashMap<String, ResourceMonitor>,
    scaling_history: Vec<ScalingEvent>,
    auto_scaling_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ScalingPolicy {
    min_instances: u32,
    max_instances: u32,
    target_cpu_utilization: f64,
    target_memory_utilization: f64,
    scale_up_cooldown: Duration,
    scale_down_cooldown: Duration,
    scale_up_threshold: f64,
    scale_down_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    cpu_utilization: f64,
    memory_utilization: f64,
    network_utilization: f64,
    storage_utilization: f64,
    last_updated: Instant,
}

#[derive(Debug, Clone)]
pub struct ScalingEvent {
    timestamp: Instant,
    event_type: ScalingEventType,
    service_name: String,
    instances_before: u32,
    instances_after: u32,
    trigger_reason: String,
}

#[derive(Debug, Clone)]
pub enum ScalingEventType {
    ScaleUp,
    ScaleDown,
    AutoScale,
    ManualScale,
}

impl AdvancedScalingManager {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            scaling_policies: HashMap::new(),
            resource_monitors: HashMap::new(),
            scaling_history: Vec::new(),
            auto_scaling_enabled: false,
        }
    }

    pub fn add_scaling_policy(&mut self, service_name: String, policy: ScalingPolicy) {
        self.scaling_policies.insert(service_name, policy);
    }

    pub fn enable_auto_scaling(&mut self) {
        self.auto_scaling_enabled = true;
    }

    pub fn disable_auto_scaling(&mut self) {
        self.auto_scaling_enabled = false;
    }

    pub fn update_resource_metrics(&mut self, service_name: String, monitor: ResourceMonitor) {
        self.resource_monitors.insert(service_name, monitor);
    }

    pub fn evaluate_scaling_needs(&mut self) -> Result<Vec<ScalingDecision>> {
        if !self.auto_scaling_enabled {
            return Ok(Vec::new());
        }

        let mut decisions = Vec::new();

        for (service_name, policy) in &self.scaling_policies {
            if let Some(monitor) = self.resource_monitors.get(service_name) {
                if let Some(decision) = self.should_scale(service_name, policy, monitor)? {
                    decisions.push(decision);
                }
            }
        }

        Ok(decisions)
    }

    fn should_scale(
        &self,
        service_name: &str,
        policy: &ScalingPolicy,
        monitor: &ResourceMonitor,
    ) -> Result<Option<ScalingDecision>> {
        let cpu_pressure = monitor.cpu_utilization > policy.scale_up_threshold;
        let memory_pressure = monitor.memory_utilization > policy.scale_up_threshold;
        let under_utilized = monitor.cpu_utilization < policy.scale_down_threshold
            && monitor.memory_utilization < policy.scale_down_threshold;

        if cpu_pressure || memory_pressure {
            return Ok(Some(ScalingDecision {
                service_name: service_name.to_string(),
                action: ScalingAction::ScaleUp,
                target_instances: self.calculate_scale_up_target(policy, monitor),
                reason: format!(
                    "CPU: {:.2}%, Memory: {:.2}%",
                    monitor.cpu_utilization * 100.0,
                    monitor.memory_utilization * 100.0
                ),
            }));
        }

        if under_utilized {
            return Ok(Some(ScalingDecision {
                service_name: service_name.to_string(),
                action: ScalingAction::ScaleDown,
                target_instances: self.calculate_scale_down_target(policy, monitor),
                reason: format!(
                    "Under-utilized - CPU: {:.2}%, Memory: {:.2}%",
                    monitor.cpu_utilization * 100.0,
                    monitor.memory_utilization * 100.0
                ),
            }));
        }

        Ok(None)
    }

    fn calculate_scale_up_target(&self, policy: &ScalingPolicy, _monitor: &ResourceMonitor) -> u32 {
        // Simple scale up by 1 instance, could be more sophisticated
        policy.max_instances.min(policy.min_instances + 1)
    }

    fn calculate_scale_down_target(
        &self,
        policy: &ScalingPolicy,
        _monitor: &ResourceMonitor,
    ) -> u32 {
        // Simple scale down by 1 instance, could be more sophisticated
        policy
            .min_instances
            .max(policy.min_instances.saturating_sub(1))
    }

    pub fn execute_scaling(&mut self, decision: ScalingDecision) -> Result<()> {
        let event = ScalingEvent {
            timestamp: Instant::now(),
            event_type: match decision.action {
                ScalingAction::ScaleUp => ScalingEventType::AutoScale,
                ScalingAction::ScaleDown => ScalingEventType::AutoScale,
            },
            service_name: decision.service_name.clone(),
            instances_before: 1, // Would be actual current count
            instances_after: decision.target_instances,
            trigger_reason: decision.reason,
        };

        self.scaling_history.push(event);
        // Here would be actual scaling implementation
        Ok(())
    }

    pub fn get_scaling_history(&self) -> &[ScalingEvent] {
        &self.scaling_history
    }
}

#[derive(Debug, Clone)]
pub struct ScalingDecision {
    pub service_name: String,
    pub action: ScalingAction,
    pub target_instances: u32,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
}

impl Default for ScalingPolicy {
    fn default() -> Self {
        Self {
            min_instances: 1,
            max_instances: 10,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            scale_up_cooldown: Duration::from_secs(300),
            scale_down_cooldown: Duration::from_secs(600),
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
        }
    }
}
