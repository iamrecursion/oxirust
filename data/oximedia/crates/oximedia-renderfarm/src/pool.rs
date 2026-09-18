// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Worker pool management.

use crate::error::{Error, Result};
use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Pool identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolId(Uuid);

impl PoolId {
    /// Create a new pool ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for PoolId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PoolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Pool priority access level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Public pool - anyone can use
    Public,
    /// Private pool - only specific projects
    Private,
    /// Reserved pool - highest priority jobs only
    Reserved,
}

/// Worker pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPool {
    /// Pool ID
    pub id: PoolId,
    /// Pool name
    pub name: String,
    /// Pool description
    pub description: String,
    /// Worker IDs in this pool
    pub workers: HashSet<WorkerId>,
    /// Access level
    pub access_level: AccessLevel,
    /// Allowed projects (for private pools)
    pub allowed_projects: HashSet<String>,
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
    /// Current job count
    pub current_jobs: usize,
    /// Pool tags
    pub tags: HashMap<String, String>,
    /// Created at
    pub created_at: DateTime<Utc>,
}

impl WorkerPool {
    /// Create a new pool
    #[must_use]
    pub fn new(name: String, description: String) -> Self {
        Self {
            id: PoolId::new(),
            name,
            description,
            workers: HashSet::new(),
            access_level: AccessLevel::Public,
            allowed_projects: HashSet::new(),
            max_concurrent_jobs: 100,
            current_jobs: 0,
            tags: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Add worker to pool
    pub fn add_worker(&mut self, worker_id: WorkerId) -> Result<()> {
        self.workers.insert(worker_id);
        Ok(())
    }

    /// Remove worker from pool
    pub fn remove_worker(&mut self, worker_id: WorkerId) -> Result<()> {
        self.workers.remove(&worker_id);
        Ok(())
    }

    /// Check if pool has capacity
    #[must_use]
    pub const fn has_capacity(&self) -> bool {
        self.current_jobs < self.max_concurrent_jobs
    }

    /// Check if project is allowed to use this pool
    #[must_use]
    pub fn is_project_allowed(&self, project_id: &str) -> bool {
        match self.access_level {
            AccessLevel::Public => true,
            AccessLevel::Private | AccessLevel::Reserved => {
                self.allowed_projects.contains(project_id)
            }
        }
    }

    /// Get worker count
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get available workers
    #[must_use]
    pub fn get_workers(&self) -> Vec<WorkerId> {
        self.workers.iter().copied().collect()
    }
}

/// Pool manager
#[derive(Default)]
pub struct PoolManager {
    pools: HashMap<PoolId, WorkerPool>,
    worker_to_pools: HashMap<WorkerId, HashSet<PoolId>>,
}

impl PoolManager {
    /// Create a new pool manager
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new pool
    pub fn create_pool(&mut self, name: String, description: String) -> PoolId {
        let pool = WorkerPool::new(name, description);
        let pool_id = pool.id;
        self.pools.insert(pool_id, pool);
        pool_id
    }

    /// Get pool by ID
    #[must_use]
    pub fn get_pool(&self, pool_id: PoolId) -> Option<&WorkerPool> {
        self.pools.get(&pool_id)
    }

    /// Get mutable pool by ID
    pub fn get_pool_mut(&mut self, pool_id: PoolId) -> Option<&mut WorkerPool> {
        self.pools.get_mut(&pool_id)
    }

    /// Delete pool
    pub fn delete_pool(&mut self, pool_id: PoolId) -> Result<()> {
        if let Some(pool) = self.pools.remove(&pool_id) {
            // Remove pool from worker mappings
            for worker_id in pool.workers {
                if let Some(pools) = self.worker_to_pools.get_mut(&worker_id) {
                    pools.remove(&pool_id);
                }
            }
        }
        Ok(())
    }

    /// Add worker to pool
    pub fn add_worker_to_pool(&mut self, pool_id: PoolId, worker_id: WorkerId) -> Result<()> {
        let pool = self
            .pools
            .get_mut(&pool_id)
            .ok_or_else(|| Error::PoolNotFound(pool_id.to_string()))?;

        pool.add_worker(worker_id)?;

        self.worker_to_pools
            .entry(worker_id)
            .or_default()
            .insert(pool_id);

        Ok(())
    }

    /// Remove worker from pool
    pub fn remove_worker_from_pool(&mut self, pool_id: PoolId, worker_id: WorkerId) -> Result<()> {
        let pool = self
            .pools
            .get_mut(&pool_id)
            .ok_or_else(|| Error::PoolNotFound(pool_id.to_string()))?;

        pool.remove_worker(worker_id)?;

        if let Some(pools) = self.worker_to_pools.get_mut(&worker_id) {
            pools.remove(&pool_id);
        }

        Ok(())
    }

    /// Get pools for worker
    #[must_use]
    pub fn get_worker_pools(&self, worker_id: WorkerId) -> Vec<PoolId> {
        self.worker_to_pools
            .get(&worker_id)
            .map_or_else(Vec::new, |pools| pools.iter().copied().collect())
    }

    /// Find pools with capacity
    #[must_use]
    pub fn find_pools_with_capacity(&self) -> Vec<PoolId> {
        self.pools
            .iter()
            .filter(|(_, pool)| pool.has_capacity())
            .map(|(id, _)| *id)
            .collect()
    }

    /// List all pools
    #[must_use]
    pub fn list_pools(&self) -> Vec<PoolId> {
        self.pools.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = WorkerPool::new("Test Pool".to_string(), "A test pool".to_string());
        assert_eq!(pool.name, "Test Pool");
        assert_eq!(pool.worker_count(), 0);
        assert!(pool.has_capacity());
    }

    #[test]
    fn test_pool_add_worker() -> Result<()> {
        let mut pool = WorkerPool::new("Test".to_string(), "Test".to_string());
        let worker_id = WorkerId::new();

        pool.add_worker(worker_id)?;
        assert_eq!(pool.worker_count(), 1);
        assert!(pool.workers.contains(&worker_id));

        Ok(())
    }

    #[test]
    fn test_pool_remove_worker() -> Result<()> {
        let mut pool = WorkerPool::new("Test".to_string(), "Test".to_string());
        let worker_id = WorkerId::new();

        pool.add_worker(worker_id)?;
        pool.remove_worker(worker_id)?;
        assert_eq!(pool.worker_count(), 0);

        Ok(())
    }

    #[test]
    fn test_pool_capacity() {
        let mut pool = WorkerPool::new("Test".to_string(), "Test".to_string());
        pool.max_concurrent_jobs = 10;
        pool.current_jobs = 5;

        assert!(pool.has_capacity());

        pool.current_jobs = 10;
        assert!(!pool.has_capacity());
    }

    #[test]
    fn test_pool_access() {
        let mut pool = WorkerPool::new("Test".to_string(), "Test".to_string());

        // Public pool - everyone allowed
        assert!(pool.is_project_allowed("project1"));

        // Private pool
        pool.access_level = AccessLevel::Private;
        pool.allowed_projects.insert("project1".to_string());

        assert!(pool.is_project_allowed("project1"));
        assert!(!pool.is_project_allowed("project2"));
    }

    #[test]
    fn test_pool_manager_create() {
        let mut manager = PoolManager::new();
        let pool_id = manager.create_pool("Test".to_string(), "Test".to_string());

        assert!(manager.get_pool(pool_id).is_some());
    }

    #[test]
    fn test_pool_manager_add_worker() -> Result<()> {
        let mut manager = PoolManager::new();
        let pool_id = manager.create_pool("Test".to_string(), "Test".to_string());
        let worker_id = WorkerId::new();

        manager.add_worker_to_pool(pool_id, worker_id)?;

        let pool = manager.get_pool(pool_id).expect("should succeed in test");
        assert_eq!(pool.worker_count(), 1);

        let pools = manager.get_worker_pools(worker_id);
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0], pool_id);

        Ok(())
    }

    #[test]
    fn test_pool_manager_remove_worker() -> Result<()> {
        let mut manager = PoolManager::new();
        let pool_id = manager.create_pool("Test".to_string(), "Test".to_string());
        let worker_id = WorkerId::new();

        manager.add_worker_to_pool(pool_id, worker_id)?;
        manager.remove_worker_from_pool(pool_id, worker_id)?;

        let pool = manager.get_pool(pool_id).expect("should succeed in test");
        assert_eq!(pool.worker_count(), 0);

        Ok(())
    }

    #[test]
    fn test_pool_manager_delete_pool() -> Result<()> {
        let mut manager = PoolManager::new();
        let pool_id = manager.create_pool("Test".to_string(), "Test".to_string());
        let worker_id = WorkerId::new();

        manager.add_worker_to_pool(pool_id, worker_id)?;
        manager.delete_pool(pool_id)?;

        assert!(manager.get_pool(pool_id).is_none());
        assert!(manager.get_worker_pools(worker_id).is_empty());

        Ok(())
    }

    #[test]
    fn test_pool_manager_find_capacity() {
        let mut manager = PoolManager::new();

        let pool1_id = manager.create_pool("Pool1".to_string(), "Test".to_string());
        let pool2_id = manager.create_pool("Pool2".to_string(), "Test".to_string());

        // Set pool1 to full capacity
        {
            let pool1 = manager
                .get_pool_mut(pool1_id)
                .expect("should succeed in test");
            pool1.max_concurrent_jobs = 10;
            pool1.current_jobs = 10;
        }

        let available = manager.find_pools_with_capacity();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0], pool2_id);
    }
}
