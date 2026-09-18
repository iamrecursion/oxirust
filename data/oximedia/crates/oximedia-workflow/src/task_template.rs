//! Task template instantiation for `oximedia-workflow`.
//!
//! [`TaskTemplate`] stores a parameterised task blueprint; [`TaskInstantiator`]
//! resolves the template parameters against a provided key→value map and
//! returns a concrete [`InstantiatedTask`] ready for execution.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Template parameter ────────────────────────────────────────────────────────

/// A single named parameter that can be substituted into a task template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemplateParam {
    /// Parameter name used as the substitution key (`{{name}}`).
    pub name: String,
    /// Human-readable description shown in documentation / UIs.
    pub description: String,
    /// Optional default value used when the caller does not provide one.
    pub default: Option<String>,
    /// Whether this parameter must be supplied (no default).
    pub required: bool,
}

impl TemplateParam {
    /// Creates a required parameter with no default.
    #[must_use]
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default: None,
            required: true,
        }
    }

    /// Creates an optional parameter with a default value.
    #[must_use]
    pub fn optional(
        name: impl Into<String>,
        description: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            default: Some(default.into()),
            required: false,
        }
    }
}

// ── Task template ─────────────────────────────────────────────────────────────

/// A reusable task blueprint with substitutable `{{parameter}}` placeholders.
///
/// Parameter placeholders inside `command_template` use double-brace syntax:
/// `{{param_name}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskTemplate {
    /// Unique name for this template (e.g. `"transcode-h264"`).
    pub name: String,
    /// Human-readable description of what this template does.
    pub description: String,
    /// Command string with `{{param}}` placeholders.
    pub command_template: String,
    /// Declared parameters for this template.
    pub params: Vec<TemplateParam>,
}

impl TaskTemplate {
    /// Creates a new task template.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        command_template: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            command_template: command_template.into(),
            params: Vec::new(),
        }
    }

    /// Adds a parameter declaration to this template.
    #[must_use]
    pub fn with_param(mut self, param: TemplateParam) -> Self {
        self.params.push(param);
        self
    }

    /// Returns the names of all required (no default) parameters.
    #[must_use]
    pub fn required_params(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect()
    }
}

// ── Instantiated task ─────────────────────────────────────────────────────────

/// A fully resolved task produced by [`TaskInstantiator::instantiate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstantiatedTask {
    /// Name of the template that was used.
    pub template_name: String,
    /// Resolved command with all placeholders substituted.
    pub command: String,
    /// The parameter values actually used (including defaults).
    pub resolved_params: HashMap<String, String>,
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors that can occur during template instantiation.
#[derive(Debug, Clone, PartialEq)]
pub enum InstantiateError {
    /// A required parameter was not provided and has no default.
    MissingParam(String),
}

impl std::fmt::Display for InstantiateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingParam(name) => write!(f, "required parameter '{name}' was not provided"),
        }
    }
}

impl std::error::Error for InstantiateError {}

// ── Instantiator ──────────────────────────────────────────────────────────────

/// Resolves [`TaskTemplate`] placeholders against caller-supplied values.
#[derive(Debug, Default, Clone)]
pub struct TaskInstantiator;

impl TaskInstantiator {
    /// Creates a new instantiator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Instantiates `template` using the provided `args` map.
    ///
    /// Parameter resolution order:
    /// 1. Value in `args`.
    /// 2. Parameter `default` (if present).
    /// 3. Error – [`InstantiateError::MissingParam`].
    ///
    /// # Errors
    ///
    /// Returns [`InstantiateError::MissingParam`] if a required parameter has
    /// no value in `args` and no declared default.
    pub fn instantiate(
        &self,
        template: &TaskTemplate,
        args: &HashMap<String, String>,
    ) -> Result<InstantiatedTask, InstantiateError> {
        let mut resolved: HashMap<String, String> = HashMap::new();

        for param in &template.params {
            let value = if let Some(v) = args.get(&param.name) {
                v.clone()
            } else if let Some(ref default) = param.default {
                default.clone()
            } else {
                return Err(InstantiateError::MissingParam(param.name.clone()));
            };
            resolved.insert(param.name.clone(), value);
        }

        // Substitute placeholders in the command
        let mut command = template.command_template.clone();
        for (key, value) in &resolved {
            let placeholder = format!("{{{{{key}}}}}");
            command = command.replace(&placeholder, value);
        }

        // Also substitute any args that weren't declared as formal params
        for (key, value) in args {
            if !resolved.contains_key(key) {
                let placeholder = format!("{{{{{key}}}}}");
                command = command.replace(&placeholder, value);
            }
        }

        Ok(InstantiatedTask {
            template_name: template.name.clone(),
            command,
            resolved_params: resolved,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn transcode_template() -> TaskTemplate {
        TaskTemplate::new(
            "transcode-h264",
            "Transcode input to H.264",
            "ffmpeg -i {{input}} -c:v libx264 -crf {{crf}} {{output}}",
        )
        .with_param(TemplateParam::required("input", "Source file path"))
        .with_param(TemplateParam::required("output", "Destination file path"))
        .with_param(TemplateParam::optional("crf", "CRF quality value", "23"))
    }

    #[test]
    fn test_instantiate_all_provided() {
        let inst = TaskInstantiator::new();
        let template = transcode_template();
        let mut args = HashMap::new();
        args.insert("input".to_string(), "/src/video.mp4".to_string());
        args.insert("output".to_string(), "/dst/video.mp4".to_string());
        args.insert("crf".to_string(), "18".to_string());
        let result = inst
            .instantiate(&template, &args)
            .expect("should succeed in test");
        assert!(result.command.contains("/src/video.mp4"));
        assert!(result.command.contains("/dst/video.mp4"));
        assert!(result.command.contains("18"));
    }

    #[test]
    fn test_instantiate_uses_default() {
        let inst = TaskInstantiator::new();
        let template = transcode_template();
        let mut args = HashMap::new();
        args.insert("input".to_string(), "/src/video.mp4".to_string());
        args.insert("output".to_string(), "/dst/video.mp4".to_string());
        let result = inst
            .instantiate(&template, &args)
            .expect("should succeed in test");
        assert_eq!(result.resolved_params["crf"], "23");
        assert!(result.command.contains("23"));
    }

    #[test]
    fn test_instantiate_missing_required_errors() {
        let inst = TaskInstantiator::new();
        let template = transcode_template();
        let args = HashMap::new(); // missing 'input' and 'output'
        let err = inst.instantiate(&template, &args).unwrap_err();
        assert!(matches!(err, InstantiateError::MissingParam(_)));
    }

    #[test]
    fn test_instantiate_template_name_preserved() {
        let inst = TaskInstantiator::new();
        let template = transcode_template();
        let mut args = HashMap::new();
        args.insert("input".to_string(), "a".to_string());
        args.insert("output".to_string(), "b".to_string());
        let result = inst
            .instantiate(&template, &args)
            .expect("should succeed in test");
        assert_eq!(result.template_name, "transcode-h264");
    }

    #[test]
    fn test_required_params_list() {
        let template = transcode_template();
        let req = template.required_params();
        assert!(req.contains(&"input"));
        assert!(req.contains(&"output"));
        assert!(!req.contains(&"crf"));
    }

    #[test]
    fn test_template_param_required() {
        let p = TemplateParam::required("src", "Source path");
        assert!(p.required);
        assert!(p.default.is_none());
    }

    #[test]
    fn test_template_param_optional() {
        let p = TemplateParam::optional("quality", "Quality", "high");
        assert!(!p.required);
        assert_eq!(p.default.as_deref(), Some("high"));
    }

    #[test]
    fn test_instantiate_error_display() {
        let err = InstantiateError::MissingParam("input".to_string());
        assert!(err.to_string().contains("input"));
    }

    #[test]
    fn test_no_placeholders_unchanged() {
        let inst = TaskInstantiator::new();
        let template = TaskTemplate::new("noop", "No-op", "echo hello");
        let args = HashMap::new();
        let result = inst
            .instantiate(&template, &args)
            .expect("should succeed in test");
        assert_eq!(result.command, "echo hello");
    }

    #[test]
    fn test_multiple_same_param_uses() {
        let inst = TaskInstantiator::new();
        let template = TaskTemplate::new("copy", "Copy file", "cp {{src}} {{src}}.bak")
            .with_param(TemplateParam::required("src", "Source"));
        let mut args = HashMap::new();
        args.insert("src".to_string(), "/file.txt".to_string());
        let result = inst
            .instantiate(&template, &args)
            .expect("should succeed in test");
        assert_eq!(result.command, "cp /file.txt /file.txt.bak");
    }

    #[test]
    fn test_resolved_params_contains_all_declared() {
        let inst = TaskInstantiator::new();
        let template = transcode_template();
        let mut args = HashMap::new();
        args.insert("input".to_string(), "in.mp4".to_string());
        args.insert("output".to_string(), "out.mp4".to_string());
        let result = inst
            .instantiate(&template, &args)
            .expect("should succeed in test");
        assert!(result.resolved_params.contains_key("crf"));
        assert!(result.resolved_params.contains_key("input"));
        assert!(result.resolved_params.contains_key("output"));
    }

    #[test]
    fn test_template_with_no_params() {
        let t = TaskTemplate::new("ping", "Ping host", "ping -c 1 localhost");
        assert!(t.required_params().is_empty());
    }

    #[test]
    fn test_instantiate_second_required_missing() {
        let inst = TaskInstantiator::new();
        let template = transcode_template();
        let mut args = HashMap::new();
        args.insert("input".to_string(), "in.mp4".to_string());
        // 'output' missing
        let err = inst.instantiate(&template, &args).unwrap_err();
        assert!(matches!(err, InstantiateError::MissingParam(ref n) if n == "output"));
    }
}
