#![allow(dead_code)]
//! Job templates for reusable, parameterised job definitions.
//!
//! Templates allow defining common job shapes once and instantiating them
//! with different parameters at runtime, reducing boilerplate and ensuring
//! consistency across similar jobs.

use std::collections::HashMap;
use std::fmt;

/// Priority level for template-instantiated jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplatePriority {
    /// Critical priority — executed first.
    Critical,
    /// High priority.
    High,
    /// Normal / default priority.
    Normal,
    /// Low priority — background work.
    Low,
}

impl fmt::Display for TemplatePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplatePriority::Critical => write!(f, "critical"),
            TemplatePriority::High => write!(f, "high"),
            TemplatePriority::Normal => write!(f, "normal"),
            TemplatePriority::Low => write!(f, "low"),
        }
    }
}

/// A parameter slot in a job template.
#[derive(Debug, Clone, PartialEq)]
pub struct TemplateParam {
    /// Parameter name (used as a placeholder key).
    pub name: String,
    /// Human-readable description of the parameter.
    pub description: String,
    /// Whether this parameter must be provided at instantiation.
    pub required: bool,
    /// Default value if not provided and not required.
    pub default_value: Option<String>,
    /// Optional validation regex pattern.
    pub validation_pattern: Option<String>,
}

impl TemplateParam {
    /// Create a new required parameter.
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
            default_value: None,
            validation_pattern: None,
        }
    }

    /// Create a new optional parameter with a default.
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
            default_value: Some(default.into()),
            validation_pattern: None,
        }
    }

    /// Add a validation pattern.
    pub fn with_validation(mut self, pattern: impl Into<String>) -> Self {
        self.validation_pattern = Some(pattern.into());
        self
    }

    /// Resolve the effective value for this parameter given user-supplied values.
    pub fn resolve(&self, supplied: Option<&String>) -> Result<String, TemplateError> {
        match (supplied, &self.default_value) {
            (Some(val), _) => Ok(val.clone()),
            (None, Some(def)) => Ok(def.clone()),
            (None, None) if self.required => {
                Err(TemplateError::MissingParameter(self.name.clone()))
            }
            (None, None) => Ok(String::new()),
        }
    }
}

/// Errors that can occur when working with job templates.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplateError {
    /// A required parameter was not supplied.
    MissingParameter(String),
    /// A parameter value failed validation.
    ValidationFailed {
        /// The parameter that failed.
        param: String,
        /// The value that was provided.
        value: String,
        /// The pattern it was validated against.
        pattern: String,
    },
    /// The template was not found.
    NotFound(String),
    /// Duplicate template name.
    Duplicate(String),
    /// Template body contains an undefined placeholder.
    UndefinedPlaceholder(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::MissingParameter(p) => write!(f, "Missing required parameter: {p}"),
            TemplateError::ValidationFailed {
                param,
                value,
                pattern,
            } => {
                write!(
                    f,
                    "Validation failed for {param}={value} (pattern: {pattern})"
                )
            }
            TemplateError::NotFound(name) => write!(f, "Template not found: {name}"),
            TemplateError::Duplicate(name) => write!(f, "Duplicate template: {name}"),
            TemplateError::UndefinedPlaceholder(ph) => {
                write!(f, "Undefined placeholder in template body: {ph}")
            }
        }
    }
}

/// A reusable job template.
#[derive(Debug, Clone)]
pub struct JobTemplate {
    /// Unique template identifier.
    pub name: String,
    /// Human-readable description of the template.
    pub description: String,
    /// Version string for the template.
    pub version: String,
    /// Default priority for jobs created from this template.
    pub default_priority: TemplatePriority,
    /// Template parameters.
    pub params: Vec<TemplateParam>,
    /// Body text with `{{param_name}}` placeholders.
    pub body: String,
    /// Tags to apply to instantiated jobs.
    pub tags: Vec<String>,
    /// Maximum retries for instantiated jobs.
    pub max_retries: u32,
    /// Timeout in seconds for instantiated jobs.
    pub timeout_secs: u64,
}

impl JobTemplate {
    /// Create a new job template.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version: "1.0.0".to_string(),
            default_priority: TemplatePriority::Normal,
            params: Vec::new(),
            body: body.into(),
            tags: Vec::new(),
            max_retries: 3,
            timeout_secs: 3600,
        }
    }

    /// Add a parameter to the template.
    pub fn with_param(mut self, param: TemplateParam) -> Self {
        self.params.push(param);
        self
    }

    /// Set the default priority.
    pub fn with_priority(mut self, priority: TemplatePriority) -> Self {
        self.default_priority = priority;
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set max retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Instantiate the template with the given parameter values, producing a resolved job spec.
    pub fn instantiate(
        &self,
        values: &HashMap<String, String>,
    ) -> Result<JobInstance, TemplateError> {
        // Resolve all parameters
        let mut resolved = HashMap::new();
        for param in &self.params {
            let value = param.resolve(values.get(&param.name))?;
            resolved.insert(param.name.clone(), value);
        }
        // Substitute placeholders in body
        let mut body = self.body.clone();
        for (key, val) in &resolved {
            let placeholder = format!("{{{{{key}}}}}");
            body = body.replace(&placeholder, val);
        }
        Ok(JobInstance {
            template_name: self.name.clone(),
            template_version: self.version.clone(),
            priority: self.default_priority,
            resolved_body: body,
            resolved_params: resolved,
            tags: self.tags.clone(),
            max_retries: self.max_retries,
            timeout_secs: self.timeout_secs,
        })
    }

    /// List the names of all required parameters.
    pub fn required_params(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// List the names of all optional parameters.
    pub fn optional_params(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter(|p| !p.required)
            .map(|p| p.name.as_str())
            .collect()
    }
}

/// A concrete job instance produced from a template.
#[derive(Debug, Clone)]
pub struct JobInstance {
    /// Name of the source template.
    pub template_name: String,
    /// Version of the source template.
    pub template_version: String,
    /// Priority of this job.
    pub priority: TemplatePriority,
    /// Resolved body with all placeholders substituted.
    pub resolved_body: String,
    /// Map of resolved parameter values.
    pub resolved_params: HashMap<String, String>,
    /// Tags inherited from the template.
    pub tags: Vec<String>,
    /// Max retries.
    pub max_retries: u32,
    /// Timeout in seconds.
    pub timeout_secs: u64,
}

/// Registry for managing multiple job templates.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    /// Storage for templates keyed by name.
    templates: HashMap<String, JobTemplate>,
}

impl TemplateRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template. Returns an error if a template with the same name already exists.
    pub fn register(&mut self, template: JobTemplate) -> Result<(), TemplateError> {
        if self.templates.contains_key(&template.name) {
            return Err(TemplateError::Duplicate(template.name.clone()));
        }
        self.templates.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a template by name.
    pub fn unregister(&mut self, name: &str) -> Result<JobTemplate, TemplateError> {
        self.templates
            .remove(name)
            .ok_or_else(|| TemplateError::NotFound(name.to_string()))
    }

    /// Look up a template by name.
    pub fn get(&self, name: &str) -> Result<&JobTemplate, TemplateError> {
        self.templates
            .get(name)
            .ok_or_else(|| TemplateError::NotFound(name.to_string()))
    }

    /// Instantiate a template by name with the given values.
    pub fn instantiate(
        &self,
        name: &str,
        values: &HashMap<String, String>,
    ) -> Result<JobInstance, TemplateError> {
        let template = self.get(name)?;
        template.instantiate(values)
    }

    /// List all registered template names.
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|k| k.as_str()).collect()
    }

    /// Return the number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Check whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

// ===========================================================================
// Conditional stage execution
// ===========================================================================

/// Arbitrary output produced by a pipeline stage.
///
/// Kept as a simple key-value map so stages can carry typed payloads without
/// imposing a rigid schema.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageOutput {
    /// Key-value fields emitted by the stage.
    pub fields: HashMap<String, String>,
}

impl StageOutput {
    /// Create an empty output.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key-value field.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Retrieve a field by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }
}

/// The outcome of executing a pipeline stage.
#[derive(Debug, Clone)]
pub enum StageOutcome {
    /// The stage ran successfully and produced `output`.
    Completed(StageOutput),
    /// The stage ran but failed with the given error message.
    Failed(String),
    /// The stage was skipped for the given reason (e.g. predecessor failed or
    /// a condition predicate returned `false`).
    Skipped(String),
}

impl StageOutcome {
    /// Returns `true` for the [`Completed`] variant.
    ///
    /// [`Completed`]: StageOutcome::Completed
    #[must_use]
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed(_))
    }

    /// Returns `true` for the [`Failed`] variant.
    ///
    /// [`Failed`]: StageOutcome::Failed
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Returns `true` for the [`Skipped`] variant.
    ///
    /// [`Skipped`]: StageOutcome::Skipped
    #[must_use]
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped(_))
    }
}

/// Gate condition for a pipeline stage.
///
/// Both `require_success` and `output_predicate` are evaluated when the
/// previous stage's [`StageOutcome`] is available.  If either test fails the
/// current stage is emitted as [`StageOutcome::Skipped`].
pub struct StageCondition {
    /// When `true`, the stage is skipped if the *previous* stage failed.
    pub require_success: bool,
    /// Optional additional predicate evaluated against the previous stage's
    /// [`StageOutput`].  A return value of `false` causes the stage to be
    /// skipped.
    pub output_predicate: Option<Box<dyn Fn(&StageOutput) -> bool + Send + Sync>>,
}

impl fmt::Debug for StageCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageCondition")
            .field("require_success", &self.require_success)
            .field(
                "output_predicate",
                &self.output_predicate.as_ref().map(|_| "<fn>"),
            )
            .finish()
    }
}

impl StageCondition {
    /// Create a condition that simply requires the predecessor to succeed.
    #[must_use]
    pub fn require_success() -> Self {
        Self {
            require_success: true,
            output_predicate: None,
        }
    }

    /// Create a condition with a custom output predicate.
    pub fn with_predicate(
        predicate: impl Fn(&StageOutput) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            require_success: false,
            output_predicate: Some(Box::new(predicate)),
        }
    }

    /// Evaluate the condition against `prev_outcome`.
    ///
    /// Returns `Ok(())` when the stage should run, or
    /// `Err(reason)` when it should be skipped.
    pub fn evaluate(&self, prev_outcome: &StageOutcome) -> Result<(), String> {
        if self.require_success && prev_outcome.is_failed() {
            return Err("predecessor failed".to_string());
        }
        if let Some(pred) = &self.output_predicate {
            match prev_outcome {
                StageOutcome::Completed(output) => {
                    if !pred(output) {
                        return Err("predicate not met".to_string());
                    }
                }
                StageOutcome::Skipped(_) => {
                    // A skipped predecessor never satisfies a predicate.
                    return Err("predecessor was skipped".to_string());
                }
                StageOutcome::Failed(_) => {
                    // Failed predecessors never satisfy a predicate either.
                    return Err("predecessor failed".to_string());
                }
            }
        }
        Ok(())
    }
}

/// A single named stage in a [`ConditionalPipeline`].
///
/// The `executor` closure is called when the stage should run.  It receives
/// the previous stage's [`StageOutput`] (or a default for the first stage)
/// and returns a new [`StageOutput`].
pub struct ConditionalStage {
    /// Human-readable name for this stage.
    pub name: String,
    /// Optional gate condition evaluated against the preceding stage.
    pub condition: Option<StageCondition>,
    /// Execution closure.  Receives the previous output, returns new output.
    executor: Box<dyn Fn(&StageOutput) -> Result<StageOutput, String> + Send + Sync>,
}

impl fmt::Debug for ConditionalStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalStage")
            .field("name", &self.name)
            .field("condition", &self.condition)
            .finish()
    }
}

impl ConditionalStage {
    /// Create a new stage without a condition.
    pub fn new(
        name: impl Into<String>,
        executor: impl Fn(&StageOutput) -> Result<StageOutput, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            condition: None,
            executor: Box::new(executor),
        }
    }

    /// Attach a condition to this stage (builder style).
    #[must_use]
    pub fn with_condition(mut self, cond: StageCondition) -> Self {
        self.condition = Some(cond);
        self
    }
}

/// A simple sequential pipeline of [`ConditionalStage`]s.
///
/// Stages are executed in order.  Each stage is given the opportunity to skip
/// itself based on the outcome of the preceding stage via [`StageCondition`].
#[derive(Debug, Default)]
pub struct ConditionalPipeline {
    stages: Vec<ConditionalStage>,
}

impl ConditionalPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a stage.
    pub fn add_stage(&mut self, stage: ConditionalStage) {
        self.stages.push(stage);
    }

    /// Execute all stages sequentially, returning a [`StageOutcome`] per stage.
    ///
    /// The first stage always runs (it has no predecessor to check).
    /// Subsequent stages are gated by their [`StageCondition`] (if any).
    pub fn execute(&self) -> Vec<(String, StageOutcome)> {
        let mut results: Vec<(String, StageOutcome)> = Vec::new();
        let mut prev_outcome: StageOutcome = StageOutcome::Completed(StageOutput::new());

        for (idx, stage) in self.stages.iter().enumerate() {
            // The very first stage runs unconditionally.
            let outcome = if idx == 0 {
                match (stage.executor)(&StageOutput::new()) {
                    Ok(output) => StageOutcome::Completed(output),
                    Err(msg) => StageOutcome::Failed(msg),
                }
            } else {
                // Evaluate gate condition (if any) against the previous stage.
                if let Some(cond) = &stage.condition {
                    match cond.evaluate(&prev_outcome) {
                        Err(reason) => StageOutcome::Skipped(reason),
                        Ok(()) => {
                            let empty = StageOutput::new();
                            let prev_output = match &prev_outcome {
                                StageOutcome::Completed(o) => o,
                                _ => &empty,
                            };
                            match (stage.executor)(prev_output) {
                                Ok(output) => StageOutcome::Completed(output),
                                Err(msg) => StageOutcome::Failed(msg),
                            }
                        }
                    }
                } else {
                    // No condition — run unconditionally.
                    let empty = StageOutput::new();
                    let prev_output = match &prev_outcome {
                        StageOutcome::Completed(o) => o,
                        _ => &empty,
                    };
                    match (stage.executor)(prev_output) {
                        Ok(output) => StageOutcome::Completed(output),
                        Err(msg) => StageOutcome::Failed(msg),
                    }
                }
            };

            prev_outcome = outcome.clone();
            results.push((stage.name.clone(), outcome));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_template() -> JobTemplate {
        JobTemplate::new(
            "transcode-hd",
            "Transcode video to HD",
            "transcode {{input}} -> {{output}} at {{bitrate}}",
        )
        .with_param(TemplateParam::required("input", "Input file path"))
        .with_param(TemplateParam::required("output", "Output file path"))
        .with_param(TemplateParam::optional(
            "bitrate",
            "Target bitrate",
            "5000000",
        ))
        .with_tag("video")
        .with_tag("transcode")
        .with_priority(TemplatePriority::High)
    }

    #[test]
    fn test_template_creation() {
        let t = sample_template();
        assert_eq!(t.name, "transcode-hd");
        assert_eq!(t.params.len(), 3);
        assert_eq!(t.default_priority, TemplatePriority::High);
    }

    #[test]
    fn test_instantiate_success() {
        let t = sample_template();
        let mut vals = HashMap::new();
        vals.insert("input".to_string(), "video.mp4".to_string());
        vals.insert("output".to_string(), "video_hd.mp4".to_string());
        let instance = t.instantiate(&vals).expect("instance should be valid");
        assert_eq!(
            instance.resolved_body,
            "transcode video.mp4 -> video_hd.mp4 at 5000000"
        );
        assert_eq!(instance.tags.len(), 2);
    }

    #[test]
    fn test_instantiate_with_override() {
        let t = sample_template();
        let mut vals = HashMap::new();
        vals.insert("input".to_string(), "a.mp4".to_string());
        vals.insert("output".to_string(), "b.mp4".to_string());
        vals.insert("bitrate".to_string(), "8000000".to_string());
        let instance = t.instantiate(&vals).expect("instance should be valid");
        assert!(instance.resolved_body.contains("8000000"));
    }

    #[test]
    fn test_instantiate_missing_required() {
        let t = sample_template();
        let vals = HashMap::new();
        let result = t.instantiate(&vals);
        assert!(result.is_err());
        if let Err(TemplateError::MissingParameter(p)) = result {
            assert_eq!(p, "input");
        }
    }

    #[test]
    fn test_required_params_list() {
        let t = sample_template();
        let req = t.required_params();
        assert_eq!(req.len(), 2);
        assert!(req.contains(&"input"));
        assert!(req.contains(&"output"));
    }

    #[test]
    fn test_optional_params_list() {
        let t = sample_template();
        let opt = t.optional_params();
        assert_eq!(opt.len(), 1);
        assert!(opt.contains(&"bitrate"));
    }

    #[test]
    fn test_template_priority_display() {
        assert_eq!(TemplatePriority::Critical.to_string(), "critical");
        assert_eq!(TemplatePriority::High.to_string(), "high");
        assert_eq!(TemplatePriority::Normal.to_string(), "normal");
        assert_eq!(TemplatePriority::Low.to_string(), "low");
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        assert_eq!(reg.len(), 1);
        assert!(reg.get("transcode-hd").is_ok());
    }

    #[test]
    fn test_registry_duplicate() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        let result = reg.register(sample_template());
        assert!(matches!(result, Err(TemplateError::Duplicate(_))));
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        let removed = reg
            .unregister("transcode-hd")
            .expect("removed should be valid");
        assert_eq!(removed.name, "transcode-hd");
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_not_found() {
        let reg = TemplateRegistry::new();
        assert!(matches!(
            reg.get("nonexistent"),
            Err(TemplateError::NotFound(_))
        ));
    }

    #[test]
    fn test_registry_instantiate() {
        let mut reg = TemplateRegistry::new();
        reg.register(sample_template())
            .expect("test expectation failed");
        let mut vals = HashMap::new();
        vals.insert("input".to_string(), "x.mp4".to_string());
        vals.insert("output".to_string(), "y.mp4".to_string());
        let instance = reg
            .instantiate("transcode-hd", &vals)
            .expect("instance should be valid");
        assert!(instance.resolved_body.contains("x.mp4"));
    }

    #[test]
    fn test_template_with_version_and_timeout() {
        let t = JobTemplate::new("test", "desc", "body")
            .with_version("2.0.0")
            .with_timeout(7200)
            .with_max_retries(5);
        assert_eq!(t.version, "2.0.0");
        assert_eq!(t.timeout_secs, 7200);
        assert_eq!(t.max_retries, 5);
    }

    #[test]
    fn test_template_error_display() {
        let e = TemplateError::MissingParameter("input".to_string());
        assert_eq!(e.to_string(), "Missing required parameter: input");
        let e2 = TemplateError::Duplicate("dup".to_string());
        assert_eq!(e2.to_string(), "Duplicate template: dup");
    }

    // -----------------------------------------------------------------------
    // ConditionalPipeline / StageCondition tests
    // -----------------------------------------------------------------------

    fn ok_stage(name: &str) -> ConditionalStage {
        let name = name.to_string();
        ConditionalStage::new(name, |_prev| {
            let mut out = StageOutput::new();
            out.insert("status", "ok");
            Ok(out)
        })
    }

    fn fail_stage(name: &str) -> ConditionalStage {
        let name = name.to_string();
        ConditionalStage::new(name, |_prev| Err("stage error".to_string()))
    }

    /// A stage with `require_success` must be skipped when its predecessor
    /// fails.
    #[test]
    fn test_conditional_stage_skips_on_failed_predecessor() {
        let mut pipeline = ConditionalPipeline::new();
        pipeline.add_stage(fail_stage("stage-1"));
        pipeline.add_stage(
            ConditionalStage::new("stage-2", |_| {
                let mut out = StageOutput::new();
                out.insert("ran", "yes");
                Ok(out)
            })
            .with_condition(StageCondition::require_success()),
        );

        let results = pipeline.execute();
        assert_eq!(results.len(), 2);
        assert!(results[0].1.is_failed(), "stage-1 should fail");
        assert!(
            results[1].1.is_skipped(),
            "stage-2 should be skipped due to failed predecessor"
        );
    }

    /// A stage with `require_success` must run when its predecessor succeeds.
    #[test]
    fn test_conditional_stage_runs_on_success_predecessor() {
        let mut pipeline = ConditionalPipeline::new();
        pipeline.add_stage(ok_stage("stage-1"));
        pipeline.add_stage(
            ConditionalStage::new("stage-2", |_| {
                let mut out = StageOutput::new();
                out.insert("ran", "yes");
                Ok(out)
            })
            .with_condition(StageCondition::require_success()),
        );

        let results = pipeline.execute();
        assert_eq!(results.len(), 2);
        assert!(results[0].1.is_completed(), "stage-1 should complete");
        assert!(
            results[1].1.is_completed(),
            "stage-2 should run after successful predecessor"
        );
    }

    /// A stage gated by an output predicate must be skipped when the predicate
    /// returns `false` and run when it returns `true`.
    #[test]
    fn test_conditional_stage_output_predicate_gating() {
        // Stage 1 sets a "quality" field to "high".
        let mut pipeline = ConditionalPipeline::new();
        pipeline.add_stage(ConditionalStage::new("ingest", |_| {
            let mut out = StageOutput::new();
            out.insert("quality", "low"); // predicate will reject this
            Ok(out)
        }));
        // Stage 2 only runs when quality == "high".
        pipeline.add_stage(
            ConditionalStage::new("premium-encode", |_| {
                let mut out = StageOutput::new();
                out.insert("encoded", "premium");
                Ok(out)
            })
            .with_condition(StageCondition::with_predicate(|prev_out| {
                prev_out.get("quality") == Some("high")
            })),
        );

        let results = pipeline.execute();
        assert_eq!(results.len(), 2);
        assert!(results[0].1.is_completed(), "ingest should complete");
        assert!(
            results[1].1.is_skipped(),
            "premium-encode should be skipped because quality != high"
        );

        // Now verify that "high" quality passes the predicate.
        let mut pipeline2 = ConditionalPipeline::new();
        pipeline2.add_stage(ConditionalStage::new("ingest", |_| {
            let mut out = StageOutput::new();
            out.insert("quality", "high");
            Ok(out)
        }));
        pipeline2.add_stage(
            ConditionalStage::new("premium-encode", |_| {
                let mut out = StageOutput::new();
                out.insert("encoded", "premium");
                Ok(out)
            })
            .with_condition(StageCondition::with_predicate(|prev_out| {
                prev_out.get("quality") == Some("high")
            })),
        );
        let results2 = pipeline2.execute();
        assert!(
            results2[1].1.is_completed(),
            "premium-encode should run when quality == high"
        );
    }
}
