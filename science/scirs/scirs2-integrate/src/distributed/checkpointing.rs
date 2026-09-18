//! Checkpointing and fault tolerance for distributed integration
//!
//! This module provides checkpointing capabilities for fault tolerance in
//! distributed ODE solving, allowing recovery from node failures.

use crate::common::IntegrateFloat;
use crate::distributed::types::{
    ChunkId, ChunkResult, ChunkResultStatus, DistributedError, DistributedResult,
    FaultToleranceMode, JobId, NodeId,
};
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Checkpoint manager for distributed computation
pub struct CheckpointManager<F: IntegrateFloat> {
    /// Checkpoint storage directory
    storage_path: PathBuf,
    /// Active checkpoints by job
    checkpoints: RwLock<HashMap<JobId, Vec<Checkpoint<F>>>>,
    /// Next checkpoint ID
    next_checkpoint_id: AtomicU64,
    /// Configuration
    config: CheckpointConfig,
    /// Checkpoint creation times
    checkpoint_times: Mutex<VecDeque<Instant>>,
}

/// Configuration for checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Maximum checkpoints to keep per job
    pub max_checkpoints_per_job: usize,
    /// Checkpoint interval (number of chunks)
    pub interval_chunks: usize,
    /// Checkpoint interval (duration)
    pub interval_duration: Duration,
    /// Enable disk persistence
    pub persist_to_disk: bool,
    /// Compress checkpoints
    pub compress: bool,
    /// Verify checkpoints after writing
    pub verify_writes: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints_per_job: 5,
            interval_chunks: 10,
            interval_duration: Duration::from_secs(60),
            persist_to_disk: true,
            compress: false,
            verify_writes: true,
        }
    }
}

/// A checkpoint containing computation state
#[derive(Debug, Clone)]
pub struct Checkpoint<F: IntegrateFloat> {
    /// Unique checkpoint ID
    pub id: u64,
    /// Job this checkpoint belongs to
    pub job_id: JobId,
    /// Timestamp when created
    pub timestamp: SystemTime,
    /// Completed chunks
    pub completed_chunks: Vec<ChunkCheckpoint<F>>,
    /// In-progress chunks (for recovery)
    pub in_progress_chunks: Vec<ChunkId>,
    /// Global state (e.g., iteration count)
    pub global_state: CheckpointGlobalState<F>,
    /// Validation hash
    pub validation_hash: u64,
}

/// Checkpoint data for a single chunk
#[derive(Debug, Clone)]
pub struct ChunkCheckpoint<F: IntegrateFloat> {
    /// Chunk ID
    pub chunk_id: ChunkId,
    /// Final time
    pub final_time: F,
    /// Final state
    pub final_state: Array1<F>,
    /// Final derivative (if available)
    pub final_derivative: Option<Array1<F>>,
    /// Node that processed this chunk
    pub node_id: NodeId,
    /// Processing time
    pub processing_time: Duration,
}

/// Global state for checkpoint
#[derive(Debug, Clone, Default)]
pub struct CheckpointGlobalState<F: IntegrateFloat> {
    /// Current iteration
    pub iteration: usize,
    /// Total chunks completed
    pub chunks_completed: usize,
    /// Total chunks remaining
    pub chunks_remaining: usize,
    /// Current time progress
    pub current_time: F,
    /// Accumulated error estimate
    pub error_estimate: F,
}

impl<F: IntegrateFloat> CheckpointManager<F> {
    /// Create a new checkpoint manager
    pub fn new(storage_path: PathBuf, config: CheckpointConfig) -> DistributedResult<Self> {
        // Create storage directory if it doesn't exist
        if config.persist_to_disk {
            fs::create_dir_all(&storage_path).map_err(|e| {
                DistributedError::CheckpointError(format!(
                    "Failed to create checkpoint directory: {}",
                    e
                ))
            })?;
        }

        Ok(Self {
            storage_path,
            checkpoints: RwLock::new(HashMap::new()),
            next_checkpoint_id: AtomicU64::new(1),
            config,
            checkpoint_times: Mutex::new(VecDeque::new()),
        })
    }

    /// Create a new checkpoint
    pub fn create_checkpoint(
        &self,
        job_id: JobId,
        completed_chunks: Vec<ChunkResult<F>>,
        in_progress_chunks: Vec<ChunkId>,
        global_state: CheckpointGlobalState<F>,
    ) -> DistributedResult<u64> {
        let checkpoint_id = self.next_checkpoint_id.fetch_add(1, Ordering::SeqCst);

        // Convert chunk results to checkpoint format
        let chunk_checkpoints: Vec<ChunkCheckpoint<F>> = completed_chunks
            .into_iter()
            .filter(|r| r.status == ChunkResultStatus::Success)
            .map(|r| ChunkCheckpoint {
                chunk_id: r.chunk_id,
                final_time: r.time_points.last().copied().unwrap_or(F::zero()),
                final_state: r.final_state.clone(),
                final_derivative: r.final_derivative.clone(),
                node_id: r.node_id,
                processing_time: r.processing_time,
            })
            .collect();

        // Calculate validation hash
        let validation_hash = self.calculate_hash(&chunk_checkpoints, &global_state);

        let checkpoint = Checkpoint {
            id: checkpoint_id,
            job_id,
            timestamp: SystemTime::now(),
            completed_chunks: chunk_checkpoints,
            in_progress_chunks,
            global_state,
            validation_hash,
        };

        // Store in memory
        {
            let mut checkpoints = self.checkpoints.write().map_err(|_| {
                DistributedError::CheckpointError("Failed to acquire checkpoint lock".to_string())
            })?;

            let job_checkpoints = checkpoints.entry(job_id).or_insert_with(Vec::new);
            job_checkpoints.push(checkpoint.clone());

            // Trim old checkpoints
            while job_checkpoints.len() > self.config.max_checkpoints_per_job {
                let removed = job_checkpoints.remove(0);
                if self.config.persist_to_disk {
                    let _ = self.delete_from_disk(job_id, removed.id);
                }
            }
        }

        // Persist to disk
        if self.config.persist_to_disk {
            self.save_to_disk(&checkpoint)?;
        }

        // Record checkpoint time
        if let Ok(mut times) = self.checkpoint_times.lock() {
            times.push_back(Instant::now());
            while times.len() > 100 {
                times.pop_front();
            }
        }

        Ok(checkpoint_id)
    }

    /// Get the latest checkpoint for a job
    pub fn get_latest_checkpoint(&self, job_id: JobId) -> Option<Checkpoint<F>> {
        match self.checkpoints.read() {
            Ok(checkpoints) => checkpoints.get(&job_id).and_then(|cps| cps.last().cloned()),
            Err(_) => None,
        }
    }

    /// Get a specific checkpoint
    pub fn get_checkpoint(&self, job_id: JobId, checkpoint_id: u64) -> Option<Checkpoint<F>> {
        match self.checkpoints.read() {
            Ok(checkpoints) => checkpoints
                .get(&job_id)
                .and_then(|cps| cps.iter().find(|cp| cp.id == checkpoint_id).cloned()),
            Err(_) => None,
        }
    }

    /// Restore from a checkpoint
    pub fn restore(
        &self,
        job_id: JobId,
        checkpoint_id: Option<u64>,
    ) -> DistributedResult<Checkpoint<F>> {
        let checkpoint = if let Some(id) = checkpoint_id {
            self.get_checkpoint(job_id, id)
        } else {
            self.get_latest_checkpoint(job_id)
        };

        let checkpoint = checkpoint.ok_or_else(|| {
            DistributedError::CheckpointError(format!("No checkpoint found for job {:?}", job_id))
        })?;

        // Validate checkpoint
        let expected_hash =
            self.calculate_hash(&checkpoint.completed_chunks, &checkpoint.global_state);
        if expected_hash != checkpoint.validation_hash {
            return Err(DistributedError::CheckpointError(
                "Checkpoint validation failed".to_string(),
            ));
        }

        Ok(checkpoint)
    }

    /// Delete all checkpoints for a job
    pub fn cleanup_job(&self, job_id: JobId) -> DistributedResult<()> {
        if let Ok(mut checkpoints) = self.checkpoints.write() {
            if let Some(job_cps) = checkpoints.remove(&job_id) {
                if self.config.persist_to_disk {
                    for cp in job_cps {
                        let _ = self.delete_from_disk(job_id, cp.id);
                    }
                }
            }
        }
        Ok(())
    }

    /// Calculate a hash for validation
    fn calculate_hash(
        &self,
        chunks: &[ChunkCheckpoint<F>],
        global_state: &CheckpointGlobalState<F>,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash chunk data
        for chunk in chunks {
            chunk.chunk_id.0.hash(&mut hasher);
            chunk.node_id.0.hash(&mut hasher);

            // Hash state values
            for val in chunk.final_state.iter() {
                let bits = val.to_f64().unwrap_or(0.0).to_bits();
                bits.hash(&mut hasher);
            }
        }

        // Hash global state
        global_state.iteration.hash(&mut hasher);
        global_state.chunks_completed.hash(&mut hasher);
        global_state.chunks_remaining.hash(&mut hasher);

        hasher.finish()
    }

    /// Save checkpoint to disk
    fn save_to_disk(&self, checkpoint: &Checkpoint<F>) -> DistributedResult<()> {
        let filename = format!(
            "checkpoint_{}_{}.bin",
            checkpoint.job_id.value(),
            checkpoint.id
        );
        let path = self.storage_path.join(&filename);

        // Serialize checkpoint (simplified - use proper serialization in production)
        let data = self.serialize_checkpoint(checkpoint)?;

        let mut file = File::create(&path).map_err(|e| {
            DistributedError::CheckpointError(format!("Failed to create checkpoint file: {}", e))
        })?;

        file.write_all(&data).map_err(|e| {
            DistributedError::CheckpointError(format!("Failed to write checkpoint: {}", e))
        })?;

        // Verify if configured
        if self.config.verify_writes {
            let mut verify_file = File::open(&path).map_err(|e| {
                DistributedError::CheckpointError(format!(
                    "Failed to verify checkpoint file: {}",
                    e
                ))
            })?;

            let mut verify_data = Vec::new();
            verify_file.read_to_end(&mut verify_data).map_err(|e| {
                DistributedError::CheckpointError(format!("Failed to read back checkpoint: {}", e))
            })?;

            if verify_data != data {
                return Err(DistributedError::CheckpointError(
                    "Checkpoint verification failed".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Delete checkpoint from disk
    fn delete_from_disk(&self, job_id: JobId, checkpoint_id: u64) -> DistributedResult<()> {
        let filename = format!("checkpoint_{}_{}.bin", job_id.value(), checkpoint_id);
        let path = self.storage_path.join(&filename);

        if path.exists() {
            fs::remove_file(&path).map_err(|e| {
                DistributedError::CheckpointError(format!(
                    "Failed to delete checkpoint file: {}",
                    e
                ))
            })?;
        }

        Ok(())
    }

    /// Serialize checkpoint to bytes
    fn serialize_checkpoint(&self, checkpoint: &Checkpoint<F>) -> DistributedResult<Vec<u8>> {
        let mut data = Vec::new();

        // Write header
        data.extend_from_slice(&checkpoint.id.to_le_bytes());
        data.extend_from_slice(&checkpoint.job_id.value().to_le_bytes());
        data.extend_from_slice(&checkpoint.validation_hash.to_le_bytes());

        // Write timestamp
        let timestamp_secs = checkpoint
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        data.extend_from_slice(&timestamp_secs.to_le_bytes());

        // Write global state
        data.extend_from_slice(&checkpoint.global_state.iteration.to_le_bytes());
        data.extend_from_slice(&checkpoint.global_state.chunks_completed.to_le_bytes());
        data.extend_from_slice(&checkpoint.global_state.chunks_remaining.to_le_bytes());

        // Write chunk count
        data.extend_from_slice(&(checkpoint.completed_chunks.len() as u64).to_le_bytes());

        // Write each chunk
        for chunk in &checkpoint.completed_chunks {
            data.extend_from_slice(&chunk.chunk_id.0.to_le_bytes());
            data.extend_from_slice(&chunk.node_id.0.to_le_bytes());

            let time_f64 = chunk.final_time.to_f64().unwrap_or(0.0);
            data.extend_from_slice(&time_f64.to_le_bytes());

            // Write state
            data.extend_from_slice(&(chunk.final_state.len() as u64).to_le_bytes());
            for val in chunk.final_state.iter() {
                let val_f64 = val.to_f64().unwrap_or(0.0);
                data.extend_from_slice(&val_f64.to_le_bytes());
            }
        }

        Ok(data)
    }

    /// Check if checkpoint is due
    pub fn should_checkpoint(&self, chunks_since_last: usize) -> bool {
        // Check chunk interval
        if chunks_since_last >= self.config.interval_chunks {
            return true;
        }

        // Check time interval
        if let Ok(times) = self.checkpoint_times.lock() {
            if let Some(last_time) = times.back() {
                if last_time.elapsed() >= self.config.interval_duration {
                    return true;
                }
            } else {
                // No checkpoints yet, time to create one
                return chunks_since_last > 0;
            }
        }

        false
    }

    /// Get checkpoint statistics
    pub fn get_statistics(&self) -> CheckpointStatistics {
        let mut total_checkpoints = 0;
        let mut total_chunks_saved = 0;

        if let Ok(checkpoints) = self.checkpoints.read() {
            for (_, job_cps) in checkpoints.iter() {
                total_checkpoints += job_cps.len();
                for cp in job_cps {
                    total_chunks_saved += cp.completed_chunks.len();
                }
            }
        }

        CheckpointStatistics {
            total_checkpoints,
            total_chunks_saved,
            storage_path: self.storage_path.clone(),
        }
    }
}

/// Statistics about checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointStatistics {
    /// Total number of checkpoints
    pub total_checkpoints: usize,
    /// Total chunks saved across all checkpoints
    pub total_chunks_saved: usize,
    /// Storage path
    pub storage_path: PathBuf,
}

/// Fault tolerance coordinator
pub struct FaultToleranceCoordinator<F: IntegrateFloat> {
    /// Checkpoint manager
    checkpoint_manager: Arc<CheckpointManager<F>>,
    /// Fault tolerance mode
    mode: FaultToleranceMode,
    /// Failed nodes
    failed_nodes: RwLock<HashSet<NodeId>>,
    /// Chunks pending retry
    pending_retry: Mutex<Vec<ChunkId>>,
    /// Recovery callbacks
    recovery_callbacks: RwLock<Vec<Arc<dyn Fn(JobId) + Send + Sync>>>,
}

impl<F: IntegrateFloat> FaultToleranceCoordinator<F> {
    /// Create a new fault tolerance coordinator
    pub fn new(checkpoint_manager: Arc<CheckpointManager<F>>, mode: FaultToleranceMode) -> Self {
        Self {
            checkpoint_manager,
            mode,
            failed_nodes: RwLock::new(HashSet::new()),
            pending_retry: Mutex::new(Vec::new()),
            recovery_callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Handle node failure
    pub fn handle_node_failure(
        &self,
        node_id: NodeId,
        affected_chunks: Vec<ChunkId>,
    ) -> DistributedResult<RecoveryAction> {
        // Record failed node
        if let Ok(mut failed) = self.failed_nodes.write() {
            failed.insert(node_id);
        }

        match self.mode {
            FaultToleranceMode::None => {
                // No recovery, just report failure
                Err(DistributedError::NodeFailure(
                    node_id,
                    "Node failed, no fault tolerance enabled".to_string(),
                ))
            }
            FaultToleranceMode::Standard => {
                // Queue chunks for retry
                if let Ok(mut pending) = self.pending_retry.lock() {
                    pending.extend(affected_chunks.iter().cloned());
                }
                Ok(RecoveryAction::RetryChunks(affected_chunks))
            }
            FaultToleranceMode::HighAvailability => {
                // Immediate failover with replicas (if available)
                if let Ok(mut pending) = self.pending_retry.lock() {
                    pending.extend(affected_chunks.iter().cloned());
                }
                Ok(RecoveryAction::FailoverAndRetry(affected_chunks))
            }
            FaultToleranceMode::CheckpointRecovery => {
                // Full recovery from checkpoint
                Ok(RecoveryAction::RestoreFromCheckpoint)
            }
        }
    }

    /// Handle chunk failure
    pub fn handle_chunk_failure(
        &self,
        chunk_id: ChunkId,
        node_id: NodeId,
        error: &str,
        can_retry: bool,
    ) -> DistributedResult<RecoveryAction> {
        if can_retry && self.mode != FaultToleranceMode::None {
            if let Ok(mut pending) = self.pending_retry.lock() {
                pending.push(chunk_id);
            }
            Ok(RecoveryAction::RetryChunks(vec![chunk_id]))
        } else if self.mode == FaultToleranceMode::CheckpointRecovery {
            Ok(RecoveryAction::RestoreFromCheckpoint)
        } else {
            Err(DistributedError::ChunkError(
                chunk_id,
                format!("Unrecoverable error on node {}: {}", node_id, error),
            ))
        }
    }

    /// Get chunks pending retry
    pub fn get_pending_retries(&self) -> Vec<ChunkId> {
        match self.pending_retry.lock() {
            Ok(pending) => pending.clone(),
            Err(_) => Vec::new(),
        }
    }

    /// Clear pending retries
    pub fn clear_pending_retries(&self) -> Vec<ChunkId> {
        match self.pending_retry.lock() {
            Ok(mut pending) => std::mem::take(&mut *pending),
            Err(_) => Vec::new(),
        }
    }

    /// Check if a node has failed
    pub fn is_node_failed(&self, node_id: NodeId) -> bool {
        match self.failed_nodes.read() {
            Ok(failed) => failed.contains(&node_id),
            Err(_) => false,
        }
    }

    /// Mark node as recovered
    pub fn mark_node_recovered(&self, node_id: NodeId) {
        if let Ok(mut failed) = self.failed_nodes.write() {
            failed.remove(&node_id);
        }
    }

    /// Recover a job from its latest checkpoint
    pub fn recover_job(&self, job_id: JobId) -> DistributedResult<Checkpoint<F>> {
        let checkpoint = self.checkpoint_manager.restore(job_id, None)?;

        // Invoke recovery callbacks
        if let Ok(callbacks) = self.recovery_callbacks.read() {
            for cb in callbacks.iter() {
                cb(job_id);
            }
        }

        Ok(checkpoint)
    }

    /// Register a recovery callback
    pub fn on_recovery<F2>(&self, callback: F2)
    where
        F2: Fn(JobId) + Send + Sync + 'static,
    {
        if let Ok(mut callbacks) = self.recovery_callbacks.write() {
            callbacks.push(Arc::new(callback));
        }
    }

    /// Get failed node count
    pub fn failed_node_count(&self) -> usize {
        match self.failed_nodes.read() {
            Ok(failed) => failed.len(),
            Err(_) => 0,
        }
    }
}

/// Action to take for recovery
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry specific chunks
    RetryChunks(Vec<ChunkId>),
    /// Failover to backup and retry
    FailoverAndRetry(Vec<ChunkId>),
    /// Restore from checkpoint
    RestoreFromCheckpoint,
    /// No action needed
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_storage_path() -> PathBuf {
        std::env::temp_dir().join(format!("scirs_checkpoint_test_{}", std::process::id()))
    }

    #[test]
    fn test_checkpoint_creation() {
        let path = temp_storage_path();
        let manager: CheckpointManager<f64> =
            CheckpointManager::new(path.clone(), CheckpointConfig::default())
                .expect("Failed to create manager");

        let job_id = JobId::new(1);
        let global_state = CheckpointGlobalState::default();

        let checkpoint_id = manager
            .create_checkpoint(job_id, Vec::new(), Vec::new(), global_state)
            .expect("Failed to create checkpoint");

        assert!(checkpoint_id > 0);

        let checkpoint = manager.get_latest_checkpoint(job_id);
        assert!(checkpoint.is_some());

        // Cleanup
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn test_checkpoint_restore() {
        let path = temp_storage_path();
        let mut config = CheckpointConfig::default();
        config.persist_to_disk = false;

        let manager: CheckpointManager<f64> =
            CheckpointManager::new(path.clone(), config).expect("Failed to create manager");

        let job_id = JobId::new(1);
        let global_state = CheckpointGlobalState {
            iteration: 5,
            chunks_completed: 10,
            ..Default::default()
        };

        let _ = manager.create_checkpoint(job_id, Vec::new(), Vec::new(), global_state.clone());

        let restored = manager.restore(job_id, None).expect("Failed to restore");
        assert_eq!(restored.global_state.iteration, 5);
        assert_eq!(restored.global_state.chunks_completed, 10);

        // Cleanup
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn test_fault_tolerance_coordinator() {
        let path = temp_storage_path();
        let mut config = CheckpointConfig::default();
        config.persist_to_disk = false;

        let manager = Arc::new(
            CheckpointManager::<f64>::new(path.clone(), config).expect("Failed to create manager"),
        );

        let coordinator = FaultToleranceCoordinator::new(manager, FaultToleranceMode::Standard);

        let action = coordinator
            .handle_node_failure(NodeId::new(1), vec![ChunkId::new(1), ChunkId::new(2)])
            .expect("Failed to handle failure");

        match action {
            RecoveryAction::RetryChunks(chunks) => {
                assert_eq!(chunks.len(), 2);
            }
            _ => panic!("Expected RetryChunks action"),
        }

        assert!(coordinator.is_node_failed(NodeId::new(1)));

        // Cleanup
        let _ = fs::remove_dir_all(&path);
    }
}
