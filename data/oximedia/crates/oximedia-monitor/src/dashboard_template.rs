//! Dashboard templating with variable substitution.
//!
//! Enables reusable dashboard definitions where metric names, thresholds, and
//! labels can be parameterized using `{{variable}}` placeholders.
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::dashboard_template::{DashboardTemplate, TemplateVariables};
//!
//! let template = DashboardTemplate::new(
//!     "Host Dashboard",
//!     "cpu_usage{host=\"{{host}}\"} > {{threshold}}",
//! );
//!
//! let mut vars = TemplateVariables::new();
//! vars.set("host", "server-01");
//! vars.set("threshold", "90");
//!
//! let rendered = template.render(&vars).expect("render ok");
//! assert_eq!(rendered, "cpu_usage{host=\"server-01\"} > 90");
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// Template variable bag
// ---------------------------------------------------------------------------

/// A collection of named string variables for template substitution.
#[derive(Debug, Clone, Default)]
pub struct TemplateVariables {
    vars: HashMap<String, String>,
}

impl TemplateVariables {
    /// Create an empty variable set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or overwrite) a variable.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.vars.insert(key.into(), value.into());
    }

    /// Get the value of a variable.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(String::as_str)
    }

    /// Remove a variable.
    pub fn remove(&mut self, key: &str) {
        self.vars.remove(key);
    }

    /// Merge another variable set (values from `other` override existing).
    pub fn merge(&mut self, other: &TemplateVariables) {
        for (k, v) in &other.vars {
            self.vars.insert(k.clone(), v.clone());
        }
    }

    /// Returns true if there are no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// Number of variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vars.len()
    }

    /// All variable keys.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        self.vars.keys().map(String::as_str).collect()
    }
}

impl From<HashMap<String, String>> for TemplateVariables {
    fn from(map: HashMap<String, String>) -> Self {
        Self { vars: map }
    }
}

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// Behaviour when a referenced variable is not found during rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingVarPolicy {
    /// Return an error listing the unresolved placeholders.
    Error,
    /// Leave the placeholder unchanged (`{{var}}`).
    Preserve,
    /// Replace with an empty string.
    Empty,
}

impl Default for MissingVarPolicy {
    fn default() -> Self {
        Self::Error
    }
}

/// A dashboard template with `{{variable}}` placeholders.
#[derive(Debug, Clone)]
pub struct DashboardTemplate {
    /// Template name / identifier.
    pub name: String,
    /// Raw template body.
    pub body: String,
    /// Policy for handling unresolved variables.
    pub missing_policy: MissingVarPolicy,
}

impl DashboardTemplate {
    /// Create a new template with default policy (error on missing vars).
    #[must_use]
    pub fn new(name: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            body: body.into(),
            missing_policy: MissingVarPolicy::Error,
        }
    }

    /// Set the missing-variable policy.
    #[must_use]
    pub fn with_policy(mut self, policy: MissingVarPolicy) -> Self {
        self.missing_policy = policy;
        self
    }

    /// Render the template by substituting all `{{var}}` placeholders.
    ///
    /// # Errors
    ///
    /// Returns an error if `missing_policy == Error` and any placeholder has
    /// no corresponding variable.
    pub fn render(&self, vars: &TemplateVariables) -> MonitorResult<String> {
        let mut output = self.body.clone();
        let mut missing: Vec<String> = Vec::new();
        // `search_pos` tracks where in `output` to resume searching so that
        // preserved (unchanged) placeholders are not re-processed.
        let mut search_pos: usize = 0;

        // Iteratively replace all `{{...}}` occurrences.
        loop {
            let start = match output[search_pos..].find("{{") {
                Some(s) => search_pos + s,
                None => break,
            };
            let end = match output[start..].find("}}") {
                Some(e) => start + e + 2,
                None => break, // unclosed placeholder — leave as-is
            };

            let placeholder = output[start..end].to_string(); // e.g. "{{host}}"
            let var_name = placeholder[2..placeholder.len() - 2].trim();

            let replacement = match vars.get(var_name) {
                Some(v) => v.to_string(),
                None => match self.missing_policy {
                    MissingVarPolicy::Error => {
                        missing.push(var_name.to_string());
                        // Continue to find all missing, then report.
                        // Replace temporarily to avoid infinite loop.
                        let marker = format!("\x00MISSING:{var_name}\x00");
                        search_pos = start + marker.len();
                        output.replace_range(start..end, &marker);
                        continue;
                    }
                    MissingVarPolicy::Preserve => {
                        // Advance past this placeholder without replacing it.
                        search_pos = end;
                        continue;
                    }
                    MissingVarPolicy::Empty => String::new(),
                },
            };

            search_pos = start + replacement.len();
            output.replace_range(start..end, &replacement);
        }

        if !missing.is_empty() {
            // Restore the missing markers before returning an error.
            for name in &missing {
                let marker = format!("\x00MISSING:{name}\x00");
                output = output.replace(&marker, &format!("{{{{{name}}}}}"));
            }
            return Err(MonitorError::Other(format!(
                "Template '{}' has unresolved variables: {}",
                self.name,
                missing.join(", ")
            )));
        }

        Ok(output)
    }

    /// Extract all variable names referenced in this template.
    #[must_use]
    pub fn variable_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        let mut pos = 0;
        let body = &self.body;
        while pos < body.len() {
            if let Some(start) = body[pos..].find("{{") {
                let abs_start = pos + start + 2;
                if let Some(end) = body[abs_start..].find("}}") {
                    let var_name = body[abs_start..abs_start + end].trim().to_string();
                    if !var_name.is_empty() && !names.contains(&var_name) {
                        names.push(var_name);
                    }
                    pos = abs_start + end + 2;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        names
    }

    /// Returns true if the template body contains no placeholders.
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.variable_names().is_empty()
    }

    /// Validate that all referenced variables are present in `vars`.
    ///
    /// Returns the list of missing variable names (empty = all present).
    #[must_use]
    pub fn validate(&self, vars: &TemplateVariables) -> Vec<String> {
        self.variable_names()
            .into_iter()
            .filter(|name| vars.get(name).is_none())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Dashboard template registry
// ---------------------------------------------------------------------------

/// A named registry of dashboard templates.
#[derive(Debug, Default)]
pub struct TemplateRegistry {
    templates: HashMap<String, DashboardTemplate>,
}

impl TemplateRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a template under its name.
    pub fn register(&mut self, template: DashboardTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Get a template by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DashboardTemplate> {
        self.templates.get(name)
    }

    /// Render a named template with given variables.
    ///
    /// # Errors
    ///
    /// Returns an error if the template is not found or rendering fails.
    pub fn render(&self, name: &str, vars: &TemplateVariables) -> MonitorResult<String> {
        let template = self.templates.get(name).ok_or_else(|| {
            MonitorError::Other(format!("Template '{name}' not found in registry"))
        })?;
        template.render(vars)
    }

    /// Number of registered templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Returns true if no templates are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// All registered template names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.templates.keys().map(String::as_str).collect()
    }

    /// Remove a template by name.
    pub fn remove(&mut self, name: &str) -> Option<DashboardTemplate> {
        self.templates.remove(name)
    }
}

// ---------------------------------------------------------------------------
// Pre-built templates
// ---------------------------------------------------------------------------

/// Built-in dashboard templates for common media pipeline monitoring patterns.
pub struct BuiltInTemplates;

impl BuiltInTemplates {
    /// Host-level resource dashboard.
    ///
    /// Variables: `host`, `warning_cpu`, `critical_cpu`, `warning_mem`,
    ///            `critical_mem`.
    #[must_use]
    pub fn host_resources() -> DashboardTemplate {
        DashboardTemplate::new(
            "Host Resources",
            concat!(
                "CPU: cpu_usage{host=\"{{host}}\"}\n",
                "  Warning  > {{warning_cpu}}%\n",
                "  Critical > {{critical_cpu}}%\n",
                "Memory: memory_usage{host=\"{{host}}\"}\n",
                "  Warning  > {{warning_mem}}%\n",
                "  Critical > {{critical_mem}}%"
            ),
        )
    }

    /// Encoding pipeline dashboard.
    ///
    /// Variables: `pipeline`, `min_fps`, `max_latency_ms`.
    #[must_use]
    pub fn encoding_pipeline() -> DashboardTemplate {
        DashboardTemplate::new(
            "Encoding Pipeline",
            concat!(
                "Pipeline: {{pipeline}}\n",
                "  FPS      >= {{min_fps}}\n",
                "  Latency  <= {{max_latency_ms}} ms\n",
                "  Metric:  encoder.fps{pipeline=\"{{pipeline}}\"}\n",
                "  Metric:  encoder.latency{pipeline=\"{{pipeline}}\"}"
            ),
        )
    }

    /// SLO compliance dashboard.
    ///
    /// Variables: `service`, `slo_target`.
    #[must_use]
    pub fn slo_compliance() -> DashboardTemplate {
        DashboardTemplate::new(
            "SLO Compliance",
            concat!(
                "Service: {{service}}\n",
                "SLO Target: {{slo_target}}%\n",
                "Metric: slo_compliance{service=\"{{service}}\"}\n",
                "Alert when below {{slo_target}}%"
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TemplateVariables --

    #[test]
    fn test_variables_set_get() {
        let mut vars = TemplateVariables::new();
        vars.set("host", "server-01");
        assert_eq!(vars.get("host"), Some("server-01"));
    }

    #[test]
    fn test_variables_get_missing() {
        let vars = TemplateVariables::new();
        assert!(vars.get("missing").is_none());
    }

    #[test]
    fn test_variables_remove() {
        let mut vars = TemplateVariables::new();
        vars.set("k", "v");
        vars.remove("k");
        assert!(vars.get("k").is_none());
    }

    #[test]
    fn test_variables_merge() {
        let mut base = TemplateVariables::new();
        base.set("a", "1");
        base.set("b", "2");

        let mut extra = TemplateVariables::new();
        extra.set("b", "overridden");
        extra.set("c", "3");

        base.merge(&extra);
        assert_eq!(base.get("a"), Some("1"));
        assert_eq!(base.get("b"), Some("overridden"));
        assert_eq!(base.get("c"), Some("3"));
    }

    #[test]
    fn test_variables_is_empty() {
        let mut vars = TemplateVariables::new();
        assert!(vars.is_empty());
        vars.set("x", "y");
        assert!(!vars.is_empty());
    }

    #[test]
    fn test_variables_len() {
        let mut vars = TemplateVariables::new();
        vars.set("a", "1");
        vars.set("b", "2");
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_variables_from_hashmap() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), "val".to_string());
        let vars = TemplateVariables::from(map);
        assert_eq!(vars.get("key"), Some("val"));
    }

    // -- DashboardTemplate rendering --

    #[test]
    fn test_render_simple_substitution() {
        let tmpl = DashboardTemplate::new("t", "Hello, {{name}}!");
        let mut vars = TemplateVariables::new();
        vars.set("name", "World");
        let out = tmpl.render(&vars).expect("render ok");
        assert_eq!(out, "Hello, World!");
    }

    #[test]
    fn test_render_multiple_variables() {
        let tmpl = DashboardTemplate::new("t", "{{a}} + {{b}} = {{c}}");
        let mut vars = TemplateVariables::new();
        vars.set("a", "1");
        vars.set("b", "2");
        vars.set("c", "3");
        let out = tmpl.render(&vars).expect("render ok");
        assert_eq!(out, "1 + 2 = 3");
    }

    #[test]
    fn test_render_repeated_variable() {
        let tmpl = DashboardTemplate::new("t", "{{x}} and {{x}} again");
        let mut vars = TemplateVariables::new();
        vars.set("x", "foo");
        let out = tmpl.render(&vars).expect("render ok");
        assert_eq!(out, "foo and foo again");
    }

    #[test]
    fn test_render_no_placeholders() {
        let tmpl = DashboardTemplate::new("t", "static text");
        let vars = TemplateVariables::new();
        let out = tmpl.render(&vars).expect("render ok");
        assert_eq!(out, "static text");
    }

    #[test]
    fn test_render_missing_var_error_policy() {
        let tmpl = DashboardTemplate::new("t", "cpu > {{threshold}}");
        let vars = TemplateVariables::new(); // no threshold
        let result = tmpl.render(&vars);
        assert!(result.is_err());
        let msg = result.expect_err("must be error").to_string();
        assert!(
            msg.contains("threshold"),
            "error should mention 'threshold'"
        );
    }

    #[test]
    fn test_render_missing_var_preserve_policy() {
        let tmpl = DashboardTemplate::new("t", "cpu > {{threshold}}")
            .with_policy(MissingVarPolicy::Preserve);
        let vars = TemplateVariables::new();
        let out = tmpl.render(&vars).expect("render ok with preserve");
        assert_eq!(out, "cpu > {{threshold}}");
    }

    #[test]
    fn test_render_missing_var_empty_policy() {
        let tmpl = DashboardTemplate::new("t", "prefix {{missing}} suffix")
            .with_policy(MissingVarPolicy::Empty);
        let vars = TemplateVariables::new();
        let out = tmpl.render(&vars).expect("render ok with empty");
        assert_eq!(out, "prefix  suffix");
    }

    #[test]
    fn test_render_whitespace_in_placeholder() {
        let tmpl = DashboardTemplate::new("t", "{{ name }}");
        let mut vars = TemplateVariables::new();
        vars.set("name", "trimmed");
        let out = tmpl.render(&vars).expect("render ok");
        assert_eq!(out, "trimmed");
    }

    #[test]
    fn test_render_complex_prometheus_query() {
        let query = "cpu_usage{host=\"{{host}}\",env=\"{{env}}\"} > {{threshold}}";
        let tmpl = DashboardTemplate::new("prom_query", query);
        let mut vars = TemplateVariables::new();
        vars.set("host", "server-01");
        vars.set("env", "prod");
        vars.set("threshold", "90");
        let out = tmpl.render(&vars).expect("render ok");
        assert_eq!(out, "cpu_usage{host=\"server-01\",env=\"prod\"} > 90");
    }

    // -- DashboardTemplate introspection --

    #[test]
    fn test_variable_names_extraction() {
        let tmpl = DashboardTemplate::new("t", "{{a}} and {{b}} and {{a}} again, {{c}}");
        let mut names = tmpl.variable_names();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_variable_names_empty_for_static() {
        let tmpl = DashboardTemplate::new("t", "no variables here");
        assert!(tmpl.variable_names().is_empty());
        assert!(tmpl.is_static());
    }

    #[test]
    fn test_is_static_false_with_vars() {
        let tmpl = DashboardTemplate::new("t", "has {{var}}");
        assert!(!tmpl.is_static());
    }

    #[test]
    fn test_validate_all_present() {
        let tmpl = DashboardTemplate::new("t", "{{a}} {{b}}");
        let mut vars = TemplateVariables::new();
        vars.set("a", "1");
        vars.set("b", "2");
        assert!(tmpl.validate(&vars).is_empty());
    }

    #[test]
    fn test_validate_missing_reported() {
        let tmpl = DashboardTemplate::new("t", "{{a}} {{b}} {{c}}");
        let mut vars = TemplateVariables::new();
        vars.set("a", "1");
        let missing = tmpl.validate(&vars);
        let mut missing_sorted = missing.clone();
        missing_sorted.sort();
        assert_eq!(missing_sorted, vec!["b", "c"]);
    }

    // -- TemplateRegistry --

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TemplateRegistry::new();
        reg.register(DashboardTemplate::new("my-tmpl", "body {{x}}"));
        assert!(reg.get("my-tmpl").is_some());
        assert!(reg.get("other").is_none());
    }

    #[test]
    fn test_registry_render() {
        let mut reg = TemplateRegistry::new();
        reg.register(DashboardTemplate::new("t", "Hello {{name}}"));
        let mut vars = TemplateVariables::new();
        vars.set("name", "World");
        let out = reg.render("t", &vars).expect("render ok");
        assert_eq!(out, "Hello World");
    }

    #[test]
    fn test_registry_render_not_found_error() {
        let reg = TemplateRegistry::new();
        let vars = TemplateVariables::new();
        assert!(reg.render("nonexistent", &vars).is_err());
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let mut reg = TemplateRegistry::new();
        assert!(reg.is_empty());
        reg.register(DashboardTemplate::new("t1", ""));
        reg.register(DashboardTemplate::new("t2", ""));
        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = TemplateRegistry::new();
        reg.register(DashboardTemplate::new("t", "body"));
        let removed = reg.remove("t");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_names() {
        let mut reg = TemplateRegistry::new();
        reg.register(DashboardTemplate::new("alpha", ""));
        reg.register(DashboardTemplate::new("beta", ""));
        let mut names = reg.names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    // -- Built-in templates --

    #[test]
    fn test_builtin_host_resources_render() {
        let tmpl = BuiltInTemplates::host_resources();
        let mut vars = TemplateVariables::new();
        vars.set("host", "server-01");
        vars.set("warning_cpu", "70");
        vars.set("critical_cpu", "90");
        vars.set("warning_mem", "80");
        vars.set("critical_mem", "95");
        let out = tmpl.render(&vars).expect("render ok");
        assert!(out.contains("server-01"));
        assert!(out.contains("70%"));
        assert!(out.contains("90%"));
    }

    #[test]
    fn test_builtin_encoding_pipeline_render() {
        let tmpl = BuiltInTemplates::encoding_pipeline();
        let mut vars = TemplateVariables::new();
        vars.set("pipeline", "av1-4k");
        vars.set("min_fps", "23.976");
        vars.set("max_latency_ms", "50");
        let out = tmpl.render(&vars).expect("render ok");
        assert!(out.contains("av1-4k"));
        assert!(out.contains("23.976"));
        assert!(out.contains("50 ms"));
    }

    #[test]
    fn test_builtin_slo_compliance_render() {
        let tmpl = BuiltInTemplates::slo_compliance();
        let mut vars = TemplateVariables::new();
        vars.set("service", "media-api");
        vars.set("slo_target", "99.9");
        let out = tmpl.render(&vars).expect("render ok");
        assert!(out.contains("media-api"));
        assert!(out.contains("99.9%"));
    }

    #[test]
    fn test_builtin_host_resources_variable_names() {
        let tmpl = BuiltInTemplates::host_resources();
        let mut names = tmpl.variable_names();
        names.sort();
        assert!(names.contains(&"host".to_string()));
        assert!(names.contains(&"warning_cpu".to_string()));
        assert!(names.contains(&"critical_cpu".to_string()));
    }
}
