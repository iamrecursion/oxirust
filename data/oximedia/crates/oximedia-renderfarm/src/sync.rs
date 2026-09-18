// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Asset synchronization across workers.

use crate::error::{Error, Result};
use crate::worker::WorkerId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Sync status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncStatus {
    /// Not started
    Pending,
    /// In progress
    Syncing,
    /// Completed
    Completed,
    /// Failed
    Failed,
}

/// Sync job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncJob {
    /// Job ID
    pub id: String,
    /// Source path
    pub source: PathBuf,
    /// Target workers
    pub targets: Vec<WorkerId>,
    /// Status
    pub status: SyncStatus,
    /// Progress (0.0 to 1.0)
    pub progress: f64,
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Started at
    pub started_at: Option<DateTime<Utc>>,
    /// Completed at
    pub completed_at: Option<DateTime<Utc>>,
}

/// Sync manager
pub struct SyncManager {
    jobs: HashMap<String, SyncJob>,
    worker_assets: HashMap<WorkerId, HashSet<PathBuf>>,
}

impl SyncManager {
    /// Create a new sync manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            worker_assets: HashMap::new(),
        }
    }

    /// Create sync job
    pub fn create_sync_job(
        &mut self,
        source: PathBuf,
        targets: Vec<WorkerId>,
        total_bytes: u64,
    ) -> String {
        let id = uuid::Uuid::new_v4().to_string();

        let job = SyncJob {
            id: id.clone(),
            source,
            targets,
            status: SyncStatus::Pending,
            progress: 0.0,
            bytes_transferred: 0,
            total_bytes,
            started_at: None,
            completed_at: None,
        };

        self.jobs.insert(id.clone(), job);
        id
    }

    /// Start sync job
    pub fn start_sync(&mut self, job_id: &str) -> Result<()> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| Error::Other(format!("Sync job not found: {job_id}")))?;

        job.status = SyncStatus::Syncing;
        job.started_at = Some(Utc::now());

        Ok(())
    }

    /// Update sync progress
    pub fn update_progress(&mut self, job_id: &str, bytes_transferred: u64) -> Result<()> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| Error::Other(format!("Sync job not found: {job_id}")))?;

        job.bytes_transferred = bytes_transferred;
        job.progress = if job.total_bytes > 0 {
            bytes_transferred as f64 / job.total_bytes as f64
        } else {
            0.0
        };

        Ok(())
    }

    /// Complete sync job
    pub fn complete_sync(&mut self, job_id: &str) -> Result<()> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| Error::Other(format!("Sync job not found: {job_id}")))?;

        job.status = SyncStatus::Completed;
        job.progress = 1.0;
        job.completed_at = Some(Utc::now());

        // Mark assets as available on workers
        for worker_id in &job.targets {
            self.worker_assets
                .entry(*worker_id)
                .or_default()
                .insert(job.source.clone());
        }

        Ok(())
    }

    /// Check if asset is available on worker
    #[must_use]
    pub fn is_asset_available(&self, worker_id: WorkerId, path: &PathBuf) -> bool {
        self.worker_assets
            .get(&worker_id)
            .is_some_and(|assets| assets.contains(path))
    }

    /// Get sync job
    #[must_use]
    pub fn get_sync_job(&self, job_id: &str) -> Option<&SyncJob> {
        self.jobs.get(job_id)
    }

    /// List active sync jobs
    #[must_use]
    pub fn list_active_jobs(&self) -> Vec<&SyncJob> {
        self.jobs
            .values()
            .filter(|j| j.status == SyncStatus::Syncing)
            .collect()
    }
}

impl Default for SyncManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_manager_creation() {
        let manager = SyncManager::new();
        assert_eq!(manager.jobs.len(), 0);
    }

    #[test]
    fn test_create_sync_job() {
        let mut manager = SyncManager::new();
        let source = PathBuf::from("/assets/texture.png");
        let targets = vec![WorkerId::new()];

        let job_id = manager.create_sync_job(source, targets, 1000);
        assert!(manager.get_sync_job(&job_id).is_some());
    }

    #[test]
    fn test_sync_job_lifecycle() -> Result<()> {
        let mut manager = SyncManager::new();
        let source = PathBuf::from("/assets/texture.png");
        let targets = vec![WorkerId::new()];

        let job_id = manager.create_sync_job(source.clone(), targets.clone(), 1000);

        // Start sync
        manager.start_sync(&job_id)?;
        let job = manager
            .get_sync_job(&job_id)
            .expect("should succeed in test");
        assert_eq!(job.status, SyncStatus::Syncing);

        // Update progress
        manager.update_progress(&job_id, 500)?;
        let job = manager
            .get_sync_job(&job_id)
            .expect("should succeed in test");
        assert_eq!(job.progress, 0.5);

        // Complete sync
        manager.complete_sync(&job_id)?;
        let job = manager
            .get_sync_job(&job_id)
            .expect("should succeed in test");
        assert_eq!(job.status, SyncStatus::Completed);

        Ok(())
    }

    #[test]
    fn test_asset_availability() -> Result<()> {
        let mut manager = SyncManager::new();
        let source = PathBuf::from("/assets/texture.png");
        let worker_id = WorkerId::new();
        let targets = vec![worker_id];

        let job_id = manager.create_sync_job(source.clone(), targets, 1000);
        manager.start_sync(&job_id)?;
        manager.complete_sync(&job_id)?;

        assert!(manager.is_asset_available(worker_id, &source));

        Ok(())
    }
}
