// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Simulation cache support for distributed rendering.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Simulation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationType {
    /// Fluid simulation
    Fluid,
    /// Particle simulation
    Particle,
    /// Cloth simulation
    Cloth,
    /// Rigid body simulation
    RigidBody,
    /// Soft body simulation
    SoftBody,
}

/// Simulation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationCache {
    /// Simulation type
    pub sim_type: SimulationType,
    /// Cache path
    pub cache_path: PathBuf,
    /// Frame range
    pub start_frame: u32,
    /// End frame
    pub end_frame: u32,
    /// Cache file format
    pub format: String,
    /// Created at
    pub created_at: DateTime<Utc>,
}

/// Simulation manager
pub struct SimulationManager {
    caches: HashMap<String, SimulationCache>,
}

impl SimulationManager {
    /// Create a new simulation manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            caches: HashMap::new(),
        }
    }

    /// Register simulation cache
    pub fn register_cache(&mut self, id: String, cache: SimulationCache) {
        self.caches.insert(id, cache);
    }

    /// Get simulation cache
    #[must_use]
    pub fn get_cache(&self, id: &str) -> Option<&SimulationCache> {
        self.caches.get(id)
    }

    /// List all caches
    #[must_use]
    pub fn list_caches(&self) -> Vec<&SimulationCache> {
        self.caches.values().collect()
    }
}

impl Default for SimulationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_manager_creation() {
        let manager = SimulationManager::new();
        assert_eq!(manager.caches.len(), 0);
    }

    #[test]
    fn test_register_cache() {
        let mut manager = SimulationManager::new();

        let cache = SimulationCache {
            sim_type: SimulationType::Fluid,
            cache_path: PathBuf::from("/cache/fluid_001"),
            start_frame: 1,
            end_frame: 100,
            format: "abc".to_string(),
            created_at: Utc::now(),
        };

        manager.register_cache("fluid_001".to_string(), cache);
        assert!(manager.get_cache("fluid_001").is_some());
    }
}
