//! Workflow planning and estimation.

use crate::{ProxyGenerationSettings, ProxyOptimizer, Result};
use std::path::Path;

/// Workflow planner for estimating time and resources.
pub struct WorkflowPlanner {
    optimizer: ProxyOptimizer,
}

impl WorkflowPlanner {
    /// Create a new workflow planner.
    #[must_use]
    pub fn new() -> Self {
        Self {
            optimizer: ProxyOptimizer::new(),
        }
    }

    /// Plan a proxy generation workflow.
    pub fn plan_generation(
        &self,
        inputs: &[MediaInfo],
        settings: ProxyGenerationSettings,
    ) -> Result<WorkflowPlan> {
        let total_duration: f64 = inputs.iter().map(|m| m.duration).sum();
        let total_input_size: u64 = inputs.iter().map(|m| m.file_size).sum();

        let mut estimated_output_size = 0u64;
        let mut estimated_encoding_time = 0.0;

        for media in inputs {
            let output_size = self
                .optimizer
                .estimate_output_size(&settings, media.duration);
            let encoding_time = self
                .optimizer
                .estimate_encoding_time(&settings, media.duration);

            estimated_output_size += output_size;
            estimated_encoding_time += encoding_time;
        }

        let space_savings = if total_input_size > estimated_output_size {
            total_input_size - estimated_output_size
        } else {
            0
        };

        let compression_ratio = if total_input_size > 0 {
            estimated_output_size as f64 / total_input_size as f64
        } else {
            0.0
        };

        Ok(WorkflowPlan {
            total_files: inputs.len(),
            total_duration,
            total_input_size,
            estimated_output_size,
            estimated_encoding_time,
            space_savings,
            compression_ratio,
            recommended_parallel_jobs: calculate_recommended_jobs(inputs.len()),
        })
    }

    /// Plan a complete offline-to-online workflow.
    pub fn plan_offline_workflow(
        &self,
        inputs: &[MediaInfo],
        settings: ProxyGenerationSettings,
        estimated_editing_time: f64,
    ) -> Result<OfflineWorkflowPlan> {
        let generation_plan = self.plan_generation(inputs, settings)?;
        let encoding_time = generation_plan.estimated_encoding_time;

        // Estimate conforming time (typically faster than generation)
        let estimated_conform_time = encoding_time * 0.1;

        // Total workflow time
        let total_time = encoding_time + estimated_editing_time + estimated_conform_time;

        Ok(OfflineWorkflowPlan {
            generation_plan,
            estimated_editing_time,
            estimated_conform_time,
            total_workflow_time: total_time,
            phases: vec![
                WorkflowPhase {
                    name: "Proxy Generation".to_string(),
                    estimated_time: encoding_time,
                },
                WorkflowPhase {
                    name: "Offline Editing".to_string(),
                    estimated_time: estimated_editing_time,
                },
                WorkflowPhase {
                    name: "Conforming".to_string(),
                    estimated_time: estimated_conform_time,
                },
            ],
        })
    }

    /// Estimate storage requirements for a workflow.
    #[must_use]
    pub fn estimate_storage(
        &self,
        inputs: &[MediaInfo],
        settings: &ProxyGenerationSettings,
        keep_originals: bool,
    ) -> StorageEstimate {
        let total_original_size: u64 = inputs.iter().map(|m| m.file_size).sum();

        let total_proxy_size: u64 = inputs
            .iter()
            .map(|m| self.optimizer.estimate_output_size(settings, m.duration))
            .sum();

        let working_storage = if keep_originals {
            total_original_size + total_proxy_size
        } else {
            total_proxy_size
        };

        // Add 20% buffer for temporary files and renders
        let recommended_storage = (working_storage as f64 * 1.2) as u64;

        StorageEstimate {
            original_size: total_original_size,
            proxy_size: total_proxy_size,
            working_storage,
            recommended_storage,
            space_saved: if total_original_size > total_proxy_size {
                total_original_size - total_proxy_size
            } else {
                0
            },
        }
    }
}

impl Default for WorkflowPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Media file information.
#[derive(Debug, Clone)]
pub struct MediaInfo {
    /// File path.
    pub path: std::path::PathBuf,

    /// File size in bytes.
    pub file_size: u64,

    /// Duration in seconds.
    pub duration: f64,

    /// Video resolution (width, height).
    pub resolution: (u32, u32),

    /// Frame rate.
    pub frame_rate: f64,
}

impl MediaInfo {
    /// Create media info from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len();

        // Placeholder: would use oximedia-core to extract actual media info
        Ok(Self {
            path: path.to_path_buf(),
            file_size,
            duration: 0.0,
            resolution: (0, 0),
            frame_rate: 0.0,
        })
    }
}

/// Workflow plan.
#[derive(Debug, Clone)]
pub struct WorkflowPlan {
    /// Total number of files.
    pub total_files: usize,

    /// Total duration in seconds.
    pub total_duration: f64,

    /// Total input size in bytes.
    pub total_input_size: u64,

    /// Estimated output size in bytes.
    pub estimated_output_size: u64,

    /// Estimated encoding time in seconds.
    pub estimated_encoding_time: f64,

    /// Space savings in bytes.
    pub space_savings: u64,

    /// Compression ratio.
    pub compression_ratio: f64,

    /// Recommended number of parallel jobs.
    pub recommended_parallel_jobs: usize,
}

impl WorkflowPlan {
    /// Get a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Workflow Plan:\n\
             Files: {}\n\
             Total Duration: {:.1} hours\n\
             Input Size: {}\n\
             Estimated Output: {}\n\
             Space Savings: {} ({:.1}%)\n\
             Estimated Time: {}\n\
             Recommended Parallel Jobs: {}",
            self.total_files,
            self.total_duration / 3600.0,
            format_bytes(self.total_input_size),
            format_bytes(self.estimated_output_size),
            format_bytes(self.space_savings),
            (1.0 - self.compression_ratio) * 100.0,
            format_duration(self.estimated_encoding_time),
            self.recommended_parallel_jobs
        )
    }
}

/// Offline workflow plan.
#[derive(Debug, Clone)]
pub struct OfflineWorkflowPlan {
    /// Generation plan.
    pub generation_plan: WorkflowPlan,

    /// Estimated editing time.
    pub estimated_editing_time: f64,

    /// Estimated conforming time.
    pub estimated_conform_time: f64,

    /// Total workflow time.
    pub total_workflow_time: f64,

    /// Workflow phases.
    pub phases: Vec<WorkflowPhase>,
}

/// A phase in the workflow.
#[derive(Debug, Clone)]
pub struct WorkflowPhase {
    /// Phase name.
    pub name: String,

    /// Estimated time in seconds.
    pub estimated_time: f64,
}

/// Storage estimate.
#[derive(Debug, Clone)]
pub struct StorageEstimate {
    /// Original files size.
    pub original_size: u64,

    /// Proxy files size.
    pub proxy_size: u64,

    /// Working storage needed.
    pub working_storage: u64,

    /// Recommended storage with buffer.
    pub recommended_storage: u64,

    /// Space saved compared to originals.
    pub space_saved: u64,
}

fn calculate_recommended_jobs(total_files: usize) -> usize {
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Use half of CPUs for encoding, or number of files, whichever is smaller
    (cpu_count / 2).max(1).min(total_files)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

fn format_duration(seconds: f64) -> String {
    let hours = (seconds / 3600.0).floor() as u64;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u64;
    let secs = (seconds % 60.0).floor() as u64;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_planner() {
        let planner = WorkflowPlanner::new();

        let inputs = vec![
            MediaInfo {
                path: "test1.mov".into(),
                file_size: 1_000_000_000,
                duration: 60.0,
                resolution: (1920, 1080),
                frame_rate: 25.0,
            },
            MediaInfo {
                path: "test2.mov".into(),
                file_size: 1_000_000_000,
                duration: 60.0,
                resolution: (1920, 1080),
                frame_rate: 25.0,
            },
        ];

        let settings = ProxyGenerationSettings::quarter_res_h264();
        let plan = planner
            .plan_generation(&inputs, settings)
            .expect("should succeed in test");

        assert_eq!(plan.total_files, 2);
        assert_eq!(plan.total_duration, 120.0);
        assert!(plan.estimated_output_size > 0);
        assert!(plan.estimated_encoding_time > 0.0);
    }

    #[test]
    fn test_offline_workflow_plan() {
        let planner = WorkflowPlanner::new();

        let inputs = vec![MediaInfo {
            path: "test.mov".into(),
            file_size: 1_000_000_000,
            duration: 600.0,
            resolution: (1920, 1080),
            frame_rate: 25.0,
        }];

        let settings = ProxyGenerationSettings::quarter_res_h264();
        let plan = planner
            .plan_offline_workflow(&inputs, settings, 7200.0)
            .expect("should succeed in test");

        assert_eq!(plan.phases.len(), 3);
        assert!(plan.total_workflow_time > 0.0);
    }

    #[test]
    fn test_storage_estimate() {
        let planner = WorkflowPlanner::new();

        let inputs = vec![MediaInfo {
            path: "test.mov".into(),
            file_size: 1_000_000_000,
            duration: 600.0,
            resolution: (1920, 1080),
            frame_rate: 25.0,
        }];

        let settings = ProxyGenerationSettings::quarter_res_h264();
        let estimate = planner.estimate_storage(&inputs, &settings, true);

        assert_eq!(estimate.original_size, 1_000_000_000);
        assert!(estimate.proxy_size > 0);
        assert!(estimate.working_storage > 0);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3665.0), "1h 1m 5s");
    }

    #[test]
    fn test_calculate_recommended_jobs() {
        let jobs = calculate_recommended_jobs(10);
        assert!(jobs > 0);
        assert!(jobs <= 10);
    }
}
