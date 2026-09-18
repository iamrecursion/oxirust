// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Advanced job scheduling and pipeline management.

use crate::job::{
    Condition, Job, JobBuilder, JobPayload, JobStatus, Priority, ResourceQuota, RetryPolicy,
};
use crate::queue::JobQueue;
use chrono::{DateTime, Datelike, Duration, Utc};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// Scheduler errors
#[derive(Debug, Error)]
pub enum SchedulerError {
    /// Invalid pipeline
    #[error("Invalid pipeline: {0}")]
    InvalidPipeline(String),

    /// Circular dependency
    #[error("Circular dependency detected")]
    CircularDependency,

    /// Job not found
    #[error("Job not found: {0}")]
    JobNotFound(Uuid),

    /// Queue error
    #[error("Queue error: {0}")]
    QueueError(String),
}

/// Result type for scheduler operations
pub type Result<T> = std::result::Result<T, SchedulerError>;

/// Job schedule specification
#[derive(Debug, Clone)]
pub enum Schedule {
    /// Run immediately
    Immediate,
    /// Run at specific time
    At(DateTime<Utc>),
    /// Run after delay
    After(Duration),
    /// Run daily at specific time
    Daily(u32, u32),
    /// Run weekly on specific day and time
    Weekly(chrono::Weekday, u32, u32),
    /// Run on cron schedule
    Cron(String),
}

impl Schedule {
    /// Calculate next execution time
    #[must_use]
    pub fn next_time(&self) -> Option<DateTime<Utc>> {
        match self {
            Self::Immediate | Self::Cron(_) => None,
            Self::At(dt) => Some(*dt),
            Self::After(duration) => Some(Utc::now() + *duration),
            Self::Daily(hour, minute) => {
                let now = Utc::now();
                let mut next = now
                    .date_naive()
                    .and_hms_opt(*hour, *minute, 0)
                    .map(|dt| dt.and_utc())?;
                if next <= now {
                    next = (now + Duration::days(1))
                        .date_naive()
                        .and_hms_opt(*hour, *minute, 0)
                        .map(|dt| dt.and_utc())?;
                }
                Some(next)
            }
            Self::Weekly(weekday, hour, minute) => {
                let now = Utc::now();
                let current_weekday = now.weekday();
                let days_until = (i64::from(weekday.num_days_from_monday())
                    - i64::from(current_weekday.num_days_from_monday())
                    + 7)
                    % 7;
                let target_date = if days_until == 0 {
                    let today_time = now
                        .date_naive()
                        .and_hms_opt(*hour, *minute, 0)
                        .map(|dt| dt.and_utc())?;
                    if today_time > now {
                        now.date_naive()
                    } else {
                        (now + Duration::days(7)).date_naive()
                    }
                } else {
                    (now + Duration::days(days_until)).date_naive()
                };
                Some(
                    target_date
                        .and_hms_opt(*hour, *minute, 0)
                        .map(|dt| dt.and_utc())?,
                )
            }
        }
    }
}

/// Pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage name
    pub name: String,
    /// Job payload
    pub payload: JobPayload,
    /// Priority
    pub priority: Priority,
    /// Resource quota
    pub resource_quota: Option<ResourceQuota>,
    /// Retry policy
    pub retry_policy: Option<RetryPolicy>,
    /// Execution condition
    pub condition: Option<Condition>,
    /// Tags
    pub tags: Vec<String>,
}

impl PipelineStage {
    /// Create a new pipeline stage
    #[must_use]
    pub fn new(name: String, payload: JobPayload) -> Self {
        Self {
            name,
            payload,
            priority: Priority::Normal,
            resource_quota: None,
            retry_policy: None,
            condition: None,
            tags: Vec::new(),
        }
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set resource quota
    #[must_use]
    pub fn with_resource_quota(mut self, quota: ResourceQuota) -> Self {
        self.resource_quota = Some(quota);
        self
    }

    /// Set retry policy
    #[must_use]
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    /// Set execution condition
    #[must_use]
    pub fn with_condition(mut self, condition: Condition) -> Self {
        self.condition = Some(condition);
        self
    }

    /// Add tag
    #[must_use]
    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }
}

/// Job pipeline
pub struct Pipeline {
    /// Pipeline name
    name: String,
    /// Pipeline stages
    stages: Vec<PipelineStage>,
    /// Pipeline tags
    tags: Vec<String>,
}

impl Pipeline {
    /// Create a new pipeline
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            stages: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Add a stage to the pipeline
    #[must_use]
    pub fn add_stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Add a tag to the pipeline
    #[must_use]
    pub fn add_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// Build the pipeline into jobs
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline is invalid
    pub fn build(&self) -> Result<Vec<Job>> {
        if self.stages.is_empty() {
            return Err(SchedulerError::InvalidPipeline(
                "Pipeline has no stages".to_string(),
            ));
        }

        let mut jobs = Vec::new();
        let mut prev_id: Option<Uuid> = None;

        for (i, stage) in self.stages.iter().enumerate() {
            let mut builder = JobBuilder::new(
                format!("{}::{}", self.name, stage.name),
                stage.priority,
                stage.payload.clone(),
            );

            for tag in &self.tags {
                builder = builder.tag(tag.clone());
            }

            for tag in &stage.tags {
                builder = builder.tag(tag.clone());
            }

            if let Some(quota) = &stage.resource_quota {
                builder = builder.resource_quota(quota.clone());
            }

            if let Some(policy) = &stage.retry_policy {
                builder = builder.retry_policy(policy.clone());
            }

            if i > 0 {
                if let Some(prev) = prev_id {
                    if let Some(condition) = &stage.condition {
                        builder = builder.condition(condition.clone());
                    } else {
                        builder = builder.condition(Condition::OnSuccess(prev));
                    }
                    builder = builder.dependency(prev);
                }
            }

            let mut job = builder.build();
            prev_id = Some(job.id);

            if i > 0 {
                job.status = JobStatus::Waiting;
            }

            jobs.push(job);
        }

        let job_ids: Vec<Uuid> = jobs.iter().map(|j| j.id).collect();
        for i in 0..jobs.len() - 1 {
            jobs[i].next_jobs.push(job_ids[i + 1]);
        }

        Ok(jobs)
    }
}

/// Job scheduler
pub struct JobScheduler {
    /// Associated job queue
    queue: Option<JobQueue>,
}

impl Default for JobScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl JobScheduler {
    /// Create a new job scheduler
    #[must_use]
    pub fn new() -> Self {
        Self { queue: None }
    }

    /// Set the job queue
    #[must_use]
    pub fn with_queue(mut self, queue: JobQueue) -> Self {
        self.queue = Some(queue);
        self
    }

    /// Schedule a single job
    ///
    /// # Errors
    ///
    /// Returns an error if scheduling fails
    pub async fn schedule_job(&self, mut job: Job, schedule: Schedule) -> Result<Uuid> {
        if let Some(next_time) = schedule.next_time() {
            job = job.with_schedule(next_time);
        }

        if let Some(queue) = &self.queue {
            queue
                .submit(job.clone())
                .await
                .map_err(|e| SchedulerError::QueueError(e.to_string()))?;
        }

        Ok(job.id)
    }

    /// Schedule a pipeline
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline is invalid or scheduling fails
    pub async fn schedule_pipeline(
        &self,
        pipeline: Pipeline,
        schedule: Schedule,
    ) -> Result<Vec<Uuid>> {
        let mut jobs = pipeline.build()?;

        if let Some(next_time) = schedule.next_time() {
            jobs[0] = jobs[0].clone().with_schedule(next_time);
        }

        let mut job_ids = Vec::new();

        if let Some(queue) = &self.queue {
            for job in jobs {
                let id = queue
                    .submit(job)
                    .await
                    .map_err(|e| SchedulerError::QueueError(e.to_string()))?;
                job_ids.push(id);
            }
        } else {
            for job in &jobs {
                job_ids.push(job.id);
            }
        }

        Ok(job_ids)
    }

    /// Create a batch of jobs with dependencies
    ///
    /// # Errors
    ///
    /// Returns an error if dependencies are invalid
    pub fn create_batch(
        &self,
        jobs: Vec<Job>,
        dependencies: HashMap<Uuid, Vec<Uuid>>,
    ) -> Result<Vec<Job>> {
        if Self::has_circular_dependencies(&dependencies) {
            return Err(SchedulerError::CircularDependency);
        }

        let mut job_map: HashMap<Uuid, Job> = jobs.into_iter().map(|j| (j.id, j)).collect();

        for (job_id, deps) in dependencies {
            if let Some(job) = job_map.get_mut(&job_id) {
                job.dependencies = deps;
            } else {
                return Err(SchedulerError::JobNotFound(job_id));
            }
        }

        Ok(job_map.into_values().collect())
    }

    /// Check for circular dependencies
    fn has_circular_dependencies(dependencies: &HashMap<Uuid, Vec<Uuid>>) -> bool {
        fn visit(
            node: Uuid,
            dependencies: &HashMap<Uuid, Vec<Uuid>>,
            visited: &mut HashMap<Uuid, bool>,
            rec_stack: &mut HashMap<Uuid, bool>,
        ) -> bool {
            visited.insert(node, true);
            rec_stack.insert(node, true);

            if let Some(deps) = dependencies.get(&node) {
                for dep in deps {
                    if !visited.get(dep).unwrap_or(&false) {
                        if visit(*dep, dependencies, visited, rec_stack) {
                            return true;
                        }
                    } else if *rec_stack.get(dep).unwrap_or(&false) {
                        return true;
                    }
                }
            }

            rec_stack.insert(node, false);
            false
        }

        let mut visited = HashMap::new();
        let mut rec_stack = HashMap::new();

        for node in dependencies.keys() {
            if !visited.get(node).unwrap_or(&false)
                && visit(*node, dependencies, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    /// Create a conditional job chain
    #[must_use]
    pub fn create_conditional_chain(
        &self,
        main_job: &Job,
        on_success: Option<Job>,
        on_failure: Option<Job>,
    ) -> Vec<Job> {
        let mut jobs = vec![main_job.clone()];

        if let Some(mut success_job) = on_success {
            success_job = success_job.with_condition(Condition::OnSuccess(main_job.id));
            success_job = success_job.with_dependency(main_job.id);
            success_job.status = JobStatus::Waiting;
            jobs.push(success_job);
        }

        if let Some(mut failure_job) = on_failure {
            failure_job = failure_job.with_condition(Condition::OnFailure(main_job.id));
            failure_job = failure_job.with_dependency(main_job.id);
            failure_job.status = JobStatus::Waiting;
            jobs.push(failure_job);
        }

        jobs
    }

    /// Create a fan-out job (one job spawning multiple parallel jobs)
    #[must_use]
    pub fn create_fan_out(&self, source_job: &Job, target_jobs: Vec<Job>) -> Vec<Job> {
        let mut jobs = vec![source_job.clone()];

        for mut target in target_jobs {
            target = target.with_dependency(source_job.id);
            target = target.with_condition(Condition::OnSuccess(source_job.id));
            target.status = JobStatus::Waiting;
            jobs.push(target);
        }

        jobs
    }

    /// Create a fan-in job (multiple jobs merging into one)
    #[must_use]
    pub fn create_fan_in(&self, source_jobs: Vec<Job>, target_job: Job) -> Vec<Job> {
        let mut jobs = source_jobs;

        let source_ids: Vec<Uuid> = jobs.iter().map(|j| j.id).collect();

        let mut final_job = target_job;
        final_job.dependencies.clone_from(&source_ids);
        final_job = final_job.with_condition(Condition::AllSuccess(source_ids));
        final_job.status = JobStatus::Waiting;

        jobs.push(final_job);
        jobs
    }

    /// Create a recurring job schedule
    ///
    /// # Errors
    ///
    /// Returns an error if scheduling fails
    pub async fn schedule_recurring(
        &self,
        job_template: Job,
        schedule: Schedule,
        count: u32,
    ) -> Result<Vec<Uuid>> {
        let mut job_ids = Vec::new();
        let mut next_time = schedule.next_time();

        for i in 0..count {
            let mut job = job_template.clone();
            job.id = Uuid::new_v4();
            job.name = format!("{}-{}", job_template.name, i);

            if let Some(time) = next_time {
                job = job.with_schedule(time);

                if let Some(queue) = &self.queue {
                    let id = queue
                        .submit(job)
                        .await
                        .map_err(|e| SchedulerError::QueueError(e.to_string()))?;
                    job_ids.push(id);
                }

                next_time = match &schedule {
                    Schedule::Daily(hour, minute) => {
                        Some(time + Duration::days(1)).and_then(|dt| {
                            dt.date_naive()
                                .and_hms_opt(*hour, *minute, 0)
                                .map(|dt| dt.and_utc())
                        })
                    }
                    Schedule::Weekly(_, hour, minute) => {
                        Some(time + Duration::weeks(1)).and_then(|dt| {
                            dt.date_naive()
                                .and_hms_opt(*hour, *minute, 0)
                                .map(|dt| dt.and_utc())
                        })
                    }
                    _ => None,
                };
            }
        }

        Ok(job_ids)
    }
}

/// Pipeline builder for fluent API
pub struct PipelineBuilder {
    pipeline: Pipeline,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            pipeline: Pipeline::new(name),
        }
    }

    /// Add a stage
    #[must_use]
    pub fn stage(mut self, stage: PipelineStage) -> Self {
        self.pipeline = self.pipeline.add_stage(stage);
        self
    }

    /// Add a tag
    #[must_use]
    pub fn tag(mut self, tag: String) -> Self {
        self.pipeline = self.pipeline.add_tag(tag);
        self
    }

    /// Build the pipeline
    #[must_use]
    pub fn build(self) -> Pipeline {
        self.pipeline
    }
}
