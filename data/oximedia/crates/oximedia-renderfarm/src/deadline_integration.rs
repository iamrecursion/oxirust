// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Deadline render manager integration (simulation).
//!
//! This module simulates the Thinkbox Deadline render management system,
//! providing job submission, progress tracking, and farm reporting.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A render job submitted to Deadline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineJob {
    /// Unique job identifier.
    pub id: String,
    /// Human-readable job name.
    pub name: String,
    /// The Deadline plugin to use (e.g. "Nuke", "Houdini").
    pub plugin: String,
    /// List of frame numbers to render.
    pub frames: Vec<u32>,
    /// Job priority (0 = lowest, 100 = highest).
    pub priority: u32,
    /// Worker pool name.
    pub pool: String,
    /// Worker group name.
    pub group: String,
}

impl DeadlineJob {
    /// Create a new Deadline job.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        plugin: DeadlinePlugin,
        frames: Vec<u32>,
        priority: u32,
        pool: impl Into<String>,
        group: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            plugin: plugin.name().to_string(),
            frames,
            priority,
            pool: pool.into(),
            group: group.into(),
        }
    }
}

/// Supported Deadline render plugins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadlinePlugin {
    /// Foundry Nuke (compositing).
    Nuke,
    /// `SideFX` Houdini (VFX / simulation).
    Houdini,
    /// Autodesk Maya (3D animation).
    Maya,
    /// Blender (open-source 3D).
    Blender,
    /// Adobe After Effects (motion graphics).
    AfterEffects,
    /// `FFmpeg` (transcoding / conversion).
    FFmpeg,
    /// Custom plugin identified by name.
    Custom(String),
}

impl DeadlinePlugin {
    /// The plugin name string as used by Deadline.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Nuke => "Nuke",
            Self::Houdini => "Houdini",
            Self::Maya => "Maya",
            Self::Blender => "Blender",
            Self::AfterEffects => "AfterEffects",
            Self::FFmpeg => "FFmpeg",
            Self::Custom(n) => n.as_str(),
        }
    }

    /// Default chunk size (frames per task) for this plugin.
    #[must_use]
    pub fn default_chunk_size(&self) -> u32 {
        match self {
            Self::Nuke => 1,
            Self::Houdini => 1,
            Self::Maya => 5,
            Self::Blender => 1,
            Self::AfterEffects => 1,
            Self::FFmpeg => 100,
            Self::Custom(_) => 1,
        }
    }
}

/// A worker pool available in the Deadline farm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlinePool {
    /// Pool name.
    pub name: String,
    /// All machine names in the pool.
    pub machines: Vec<String>,
    /// Number of machines currently online.
    pub online_count: u32,
    /// Number of machines available (not rendering).
    pub available_count: u32,
}

impl DeadlinePool {
    /// Create a new pool definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        machines: Vec<String>,
        online_count: u32,
        available_count: u32,
    ) -> Self {
        Self {
            name: name.into(),
            machines,
            online_count,
            available_count,
        }
    }
}

/// Status of an individual render task (one frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Waiting for a worker.
    Pending,
    /// Currently rendering.
    Rendering,
    /// Render completed successfully.
    Completed,
    /// Render failed.
    Failed,
    /// Task has been suspended.
    Suspended,
}

/// A single-frame render task associated with a Deadline job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineTask {
    /// The parent job ID.
    pub job_id: String,
    /// Frame number.
    pub frame: u32,
    /// Name of the worker assigned to this task.
    pub worker: String,
    /// Current task status.
    pub status: TaskStatus,
    /// Render time in milliseconds (set when completed).
    pub render_time_ms: Option<u64>,
}

impl DeadlineTask {
    /// Create a new pending task.
    #[must_use]
    pub fn new(job_id: impl Into<String>, frame: u32, worker: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            frame,
            worker: worker.into(),
            status: TaskStatus::Pending,
            render_time_ms: None,
        }
    }
}

/// Simulated Deadline render manager.
///
/// All rendering is simulated: `tick()` advances time and moves tasks forward.
pub struct DeadlineSimulator {
    jobs: HashMap<String, DeadlineJob>,
    tasks: Vec<DeadlineTask>,
    /// Simulated time in milliseconds.
    current_time_ms: u64,
    /// Simulated render speed: ms per frame.
    ms_per_frame: u64,
    /// Tasks started at which simulated time.
    task_start_times: HashMap<(String, u32), u64>,
}

impl DeadlineSimulator {
    /// Create a new simulator.
    ///
    /// * `ms_per_frame` – how many simulated milliseconds it takes to render one frame.
    #[must_use]
    pub fn new(ms_per_frame: u64) -> Self {
        Self {
            jobs: HashMap::new(),
            tasks: Vec::new(),
            current_time_ms: 0,
            ms_per_frame,
            task_start_times: HashMap::new(),
        }
    }

    /// Create a simulator with default timing (100 ms/frame).
    #[must_use]
    pub fn default_sim() -> Self {
        Self::new(100)
    }

    /// Submit a job and return its ID.
    pub fn submit_job(&mut self, job: DeadlineJob) -> String {
        let id = job.id.clone();
        // Create a pending task for each frame
        for &frame in &job.frames {
            self.tasks
                .push(DeadlineTask::new(id.clone(), frame, "worker-0"));
        }
        self.jobs.insert(id.clone(), job);
        id
    }

    /// Advance simulation by `time_ms` milliseconds.
    ///
    /// During each tick, pending tasks start and running tasks progress towards completion.
    pub fn tick(&mut self, time_ms: u64) {
        let new_time = self.current_time_ms + time_ms;

        for task in &mut self.tasks {
            let key = (task.job_id.clone(), task.frame);
            match task.status {
                TaskStatus::Pending => {
                    // Start rendering immediately
                    task.status = TaskStatus::Rendering;
                    self.task_start_times
                        .insert(key.clone(), self.current_time_ms);
                    // Check if the render also completes within this tick
                    let elapsed = new_time.saturating_sub(self.current_time_ms);
                    if elapsed >= self.ms_per_frame {
                        task.status = TaskStatus::Completed;
                        task.render_time_ms = Some(self.ms_per_frame);
                    }
                }
                TaskStatus::Rendering => {
                    let started_at = self
                        .task_start_times
                        .get(&key)
                        .copied()
                        .unwrap_or(self.current_time_ms);
                    let elapsed = new_time.saturating_sub(started_at);
                    if elapsed >= self.ms_per_frame {
                        task.status = TaskStatus::Completed;
                        task.render_time_ms = Some(self.ms_per_frame);
                    }
                }
                _ => {}
            }
        }

        self.current_time_ms = new_time;
    }

    /// Return the completion progress (0.0–1.0) of a job.
    ///
    /// Returns 0.0 if the job does not exist or has no frames.
    #[must_use]
    pub fn job_progress(&self, job_id: &str) -> f32 {
        let total: usize = self.tasks.iter().filter(|t| t.job_id == job_id).count();
        if total == 0 {
            return 0.0;
        }
        let done: usize = self
            .tasks
            .iter()
            .filter(|t| t.job_id == job_id && t.status == TaskStatus::Completed)
            .count();
        done as f32 / total as f32
    }

    /// Return references to all fully completed jobs.
    #[must_use]
    pub fn completed_jobs(&self) -> Vec<&DeadlineJob> {
        self.jobs
            .values()
            .filter(|job| {
                let total = self.tasks.iter().filter(|t| t.job_id == job.id).count();
                let done = self
                    .tasks
                    .iter()
                    .filter(|t| t.job_id == job.id && t.status == TaskStatus::Completed)
                    .count();
                total > 0 && done == total
            })
            .collect()
    }

    /// Return all tasks for a given job.
    #[must_use]
    pub fn job_tasks(&self, job_id: &str) -> Vec<&DeadlineTask> {
        self.tasks.iter().filter(|t| t.job_id == job_id).collect()
    }
}

/// Summary report for the entire farm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarmReport {
    /// Total jobs submitted.
    pub total_jobs: u32,
    /// Jobs completed successfully.
    pub completed: u32,
    /// Jobs that failed.
    pub failed: u32,
    /// Average render time per task in milliseconds.
    pub avg_render_time_ms: f64,
    /// Farm efficiency as a percentage (0–100).
    pub efficiency_pct: f32,
}

impl FarmReport {
    /// Build a farm report from a simulator.
    #[must_use]
    pub fn from_simulator(sim: &DeadlineSimulator) -> Self {
        let total_jobs = sim.jobs.len() as u32;
        let completed = sim.completed_jobs().len() as u32;
        let failed = 0u32; // simulation does not produce failures

        let completed_tasks: Vec<_> = sim
            .tasks
            .iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .collect();
        let avg_render_time_ms = if completed_tasks.is_empty() {
            0.0
        } else {
            completed_tasks
                .iter()
                .filter_map(|t| t.render_time_ms)
                .sum::<u64>() as f64
                / completed_tasks.len() as f64
        };

        let total_tasks = sim.tasks.len() as f32;
        let efficiency_pct = if total_tasks > 0.0 {
            (completed_tasks.len() as f32 / total_tasks) * 100.0
        } else {
            0.0
        };

        Self {
            total_jobs,
            completed,
            failed,
            avg_render_time_ms,
            efficiency_pct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str, frames: Vec<u32>) -> DeadlineJob {
        DeadlineJob::new(
            id,
            id,
            DeadlinePlugin::Blender,
            frames,
            50,
            "general",
            "cpu",
        )
    }

    #[test]
    fn test_deadline_plugin_name() {
        assert_eq!(DeadlinePlugin::Nuke.name(), "Nuke");
        assert_eq!(DeadlinePlugin::Houdini.name(), "Houdini");
        assert_eq!(DeadlinePlugin::Maya.name(), "Maya");
        assert_eq!(DeadlinePlugin::Blender.name(), "Blender");
        assert_eq!(DeadlinePlugin::AfterEffects.name(), "AfterEffects");
        assert_eq!(DeadlinePlugin::FFmpeg.name(), "FFmpeg");
        assert_eq!(
            DeadlinePlugin::Custom("MyPlugin".to_string()).name(),
            "MyPlugin"
        );
    }

    #[test]
    fn test_deadline_plugin_chunk_size() {
        assert_eq!(DeadlinePlugin::Nuke.default_chunk_size(), 1);
        assert_eq!(DeadlinePlugin::Maya.default_chunk_size(), 5);
        assert_eq!(DeadlinePlugin::FFmpeg.default_chunk_size(), 100);
    }

    #[test]
    fn test_submit_job() {
        let mut sim = DeadlineSimulator::default_sim();
        let job = make_job("j1", vec![1, 2, 3]);
        let id = sim.submit_job(job);
        assert_eq!(id, "j1");
        assert_eq!(sim.job_tasks("j1").len(), 3);
    }

    #[test]
    fn test_job_progress_zero_before_tick() {
        let mut sim = DeadlineSimulator::new(100);
        sim.submit_job(make_job("j2", vec![1, 2]));
        // Before any tick, tasks are Pending
        assert!((sim.job_progress("j2") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_job_progress_after_full_tick() {
        let mut sim = DeadlineSimulator::new(100);
        sim.submit_job(make_job("j3", vec![1, 2, 3]));
        // Tick enough to complete all frames
        sim.tick(200); // start + complete
        assert!((sim.job_progress("j3") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_completed_jobs() {
        let mut sim = DeadlineSimulator::new(100);
        sim.submit_job(make_job("j4", vec![1]));
        assert_eq!(sim.completed_jobs().len(), 0);
        sim.tick(200);
        assert_eq!(sim.completed_jobs().len(), 1);
    }

    #[test]
    fn test_job_progress_nonexistent() {
        let sim = DeadlineSimulator::default_sim();
        assert!((sim.job_progress("ghost") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multiple_jobs() {
        let mut sim = DeadlineSimulator::new(50);
        sim.submit_job(make_job("a", vec![1, 2]));
        sim.submit_job(make_job("b", vec![10]));
        sim.tick(200);
        assert_eq!(sim.completed_jobs().len(), 2);
    }

    #[test]
    fn test_farm_report() {
        let mut sim = DeadlineSimulator::new(100);
        sim.submit_job(make_job("r1", vec![1, 2, 3]));
        sim.tick(300);
        let report = FarmReport::from_simulator(&sim);
        assert_eq!(report.total_jobs, 1);
        assert_eq!(report.completed, 1);
        assert!(report.avg_render_time_ms > 0.0);
        assert!((report.efficiency_pct - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_deadline_pool_creation() {
        let pool = DeadlinePool::new(
            "render_farm",
            vec!["node01".to_string(), "node02".to_string()],
            2,
            1,
        );
        assert_eq!(pool.name, "render_farm");
        assert_eq!(pool.machines.len(), 2);
        assert_eq!(pool.online_count, 2);
        assert_eq!(pool.available_count, 1);
    }

    #[test]
    fn test_deadline_task_initial_state() {
        let task = DeadlineTask::new("job_x", 42, "worker-1");
        assert_eq!(task.frame, 42);
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.render_time_ms.is_none());
    }
}
