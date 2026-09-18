//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{IoError, Result};
use chrono::{DateTime, Datelike, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use super::types::{
    ResourceRequirements, ScheduleConfig, Task, TaskType, Workflow, WorkflowBuilder,
    WorkflowExecutor, WorkflowState, WorkflowStatus,
};

/// Task builders for common operations
pub mod tasks {
    use super::*;
    /// Create a data ingestion task
    pub fn data_ingestion(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::DataIngestion)
    }
    /// Create a transformation task
    pub fn transform(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Transform)
    }
    /// Create a validation task
    pub fn validation(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Validation)
    }
    /// Create an export task
    pub fn export(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Export)
    }
    /// Task builder
    pub struct TaskBuilder {
        task: Task,
    }
    impl TaskBuilder {
        pub fn new(id: impl Into<String>, name: impl Into<String>, task_type: TaskType) -> Self {
            Self {
                task: Task {
                    id: id.into(),
                    name: name.into(),
                    task_type,
                    config: json!({}),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    resources: ResourceRequirements::default(),
                },
            }
        }
        pub fn config(mut self, config: serde_json::Value) -> Self {
            self.task.config = config;
            self
        }
        pub fn input(mut self, input: impl Into<String>) -> Self {
            self.task.inputs.push(input.into());
            self
        }
        pub fn output(mut self, output: impl Into<String>) -> Self {
            self.task.outputs.push(output.into());
            self
        }
        pub fn resources(mut self, cpu: usize, memorygb: f64) -> Self {
            self.task.resources.cpu_cores = Some(cpu);
            self.task.resources.memorygb = Some(memorygb);
            self
        }
        pub fn build(self) -> Task {
            self.task
        }
    }
}
/// Workflow templates for common patterns
pub mod templates {
    use super::*;
    /// Create an ETL (Extract-Transform-Load) workflow
    pub fn etlworkflow(name: impl Into<String>) -> WorkflowBuilder {
        let _name = name.into();
        let id = format!("etl_{}", Utc::now().timestamp());
        WorkflowBuilder::new(&id, &_name)
            .description("Standard ETL workflow template")
            .add_task(
                tasks::data_ingestion("extract", "Extract Data")
                    .config(serde_json::json!(
                        { "source" : "database", "query" : "SELECT * FROM raw_data" }
                    ))
                    .output("raw_data")
                    .build(),
            )
            .add_task(
                tasks::transform("transform", "Transform Data")
                    .input("raw_data")
                    .output("transformed_data")
                    .config(serde_json::json!(
                        { "operations" : ["normalize", "aggregate", "filter"] }
                    ))
                    .build(),
            )
            .add_task(
                tasks::validation("validate", "Validate Data")
                    .input("transformed_data")
                    .output("validated_data")
                    .build(),
            )
            .add_task(
                tasks::export("load", "Load Data")
                    .input("validated_data")
                    .config(serde_json::json!(
                        { "destination" : "warehouse", "table" : "processed_data" }
                    ))
                    .build(),
            )
            .add_dependency("transform", "extract")
            .add_dependency("validate", "transform")
            .add_dependency("load", "validate")
    }
    /// Create a batch processing workflow
    pub fn batch_processing(name: impl Into<String>, _batch_size: usize) -> WorkflowBuilder {
        let name = name.into();
        let id = format!("batch_{}", Utc::now().timestamp());
        WorkflowBuilder::new(&id, &name)
            .description("Batch processing workflow template")
            .configure(|config| {
                config.max_parallel_tasks = 8;
                config.scheduling = Some(ScheduleConfig {
                    cron: Some("0 2 * * *".to_string()),
                    interval: None,
                    start_time: None,
                    end_time: None,
                });
            })
    }
}
/// Workflow monitoring and metrics
pub mod monitoring {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowMetrics {
        pub total_executions: usize,
        pub successful_executions: usize,
        pub failed_executions: usize,
        pub average_duration: Duration,
        pub task_metrics: HashMap<String, TaskMetrics>,
    }
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskMetrics {
        pub total_runs: usize,
        pub success_rate: f64,
        pub average_duration: Duration,
        pub retry_rate: f64,
    }
    /// Collect metrics for a workflow
    pub fn collect_metrics(states: &[WorkflowState]) -> WorkflowMetrics {
        let total = states.len();
        let successful = states
            .iter()
            .filter(|s| s.status == WorkflowStatus::Success)
            .count();
        let failed = states
            .iter()
            .filter(|s| s.status == WorkflowStatus::Failed)
            .count();
        let durations: Vec<Duration> = states
            .iter()
            .filter_map(|s| match (s.start_time, s.end_time) {
                (Some(start), Some(end)) => Some(end - start),
                _ => None,
            })
            .collect();
        let avg_duration = if durations.is_empty() {
            Duration::seconds(0)
        } else {
            let total_secs: i64 = durations.iter().map(|d| d.num_seconds()).sum();
            Duration::seconds(total_secs / durations.len() as i64)
        };
        WorkflowMetrics {
            total_executions: total,
            successful_executions: successful,
            failed_executions: failed,
            average_duration: avg_duration,
            task_metrics: HashMap::new(),
        }
    }
}
/// Advanced scheduling capabilities
pub mod scheduling {
    use super::*;
    use cron::Schedule as CronSchedule;
    use std::str::FromStr;
    /// Advanced scheduler with support for complex scheduling patterns
    pub struct WorkflowScheduler {
        schedules: HashMap<String, ScheduledWorkflow>,
        executor: Arc<WorkflowExecutor>,
        running: Arc<Mutex<bool>>,
    }
    #[derive(Debug)]
    struct ScheduledWorkflow {
        workflow: Workflow,
        schedule: WorkflowSchedule,
        last_run: Option<DateTime<Utc>>,
        next_run: Option<DateTime<Utc>>,
    }
    #[derive(Debug, Clone)]
    pub enum WorkflowSchedule {
        Cron(String),
        Interval { seconds: u64 },
        FixedDelay { seconds: u64 },
        OneTime(DateTime<Utc>),
        Complex(ComplexSchedule),
    }
    #[derive(Debug, Clone)]
    pub struct ComplexSchedule {
        pub business_days_only: bool,
        pub exclude_holidays: bool,
        pub timezone: String,
        pub blackout_periods: Vec<(DateTime<Utc>, DateTime<Utc>)>,
        pub dependencies: Vec<ScheduleDependency>,
    }
    #[derive(Debug, Clone)]
    pub enum ScheduleDependency {
        FileArrival { path: String, pattern: String },
        DataAvailability { source: String, threshold: f64 },
        ExternalTrigger { webhook: String },
        WorkflowCompletion { workflowid: String },
    }
    impl WorkflowScheduler {
        pub fn new(executor: Arc<WorkflowExecutor>) -> Self {
            Self {
                schedules: HashMap::new(),
                executor,
                running: Arc::new(Mutex::new(false)),
            }
        }
        /// Schedule a workflow
        pub fn schedule(&mut self, workflow: Workflow, schedule: WorkflowSchedule) -> Result<()> {
            let next_run = self.calculate_next_run(&schedule, None)?;
            self.schedules.insert(
                workflow.id.clone(),
                ScheduledWorkflow {
                    workflow,
                    schedule,
                    last_run: None,
                    next_run,
                },
            );
            Ok(())
        }
        /// Calculate next run time based on schedule
        fn calculate_next_run(
            &self,
            schedule: &WorkflowSchedule,
            last_run: Option<DateTime<Utc>>,
        ) -> Result<Option<DateTime<Utc>>> {
            match schedule {
                WorkflowSchedule::Cron(cron_expr) => {
                    let schedule = CronSchedule::from_str(cron_expr)
                        .map_err(|e| IoError::Other(format!("Invalid cron expression: {e}")))?;
                    let after = last_run.unwrap_or_else(Utc::now);
                    Ok(schedule.after(&after).next())
                }
                WorkflowSchedule::Interval { seconds } => {
                    let base = last_run.unwrap_or_else(Utc::now);
                    Ok(Some(base + Duration::seconds(*seconds as i64)))
                }
                WorkflowSchedule::FixedDelay { seconds } => {
                    Ok(Some(Utc::now() + Duration::seconds(*seconds as i64)))
                }
                WorkflowSchedule::OneTime(time) => {
                    if *time > Utc::now() {
                        Ok(Some(*time))
                    } else {
                        Ok(None)
                    }
                }
                WorkflowSchedule::Complex(complex) => {
                    self.calculate_complex_schedule(complex, last_run)
                }
            }
        }
        fn calculate_complex_schedule(
            &self,
            complex: &ComplexSchedule,
            last_run: Option<DateTime<Utc>>,
        ) -> Result<Option<DateTime<Utc>>> {
            let mut next_run = last_run.unwrap_or_else(Utc::now) + Duration::days(1);
            if complex.business_days_only {
                while next_run.weekday().num_days_from_monday() >= 5 {
                    next_run += Duration::days(1);
                }
            }
            for (start, end) in &complex.blackout_periods {
                if next_run >= *start && next_run <= *end {
                    next_run = *end + Duration::seconds(1);
                }
            }
            Ok(Some(next_run))
        }
        /// Start the scheduler
        pub fn start(&self) -> Result<()> {
            *self.running.lock().expect("Operation failed") = true;
            Ok(())
        }
        /// Stop the scheduler
        pub fn stop(&self) {
            *self.running.lock().expect("Operation failed") = false;
        }
    }
}
