// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Template-based conversion system.

use crate::{ConversionError, ConversionOptions, Result};
use std::collections::HashMap;
use std::path::Path;

/// Template system for batch conversions.
#[derive(Debug, Clone)]
pub struct TemplateSystem {
    templates: HashMap<String, ConversionTemplate>,
}

impl TemplateSystem {
    /// Create a new template system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    /// Add a template.
    pub fn add_template(&mut self, name: String, template: ConversionTemplate) {
        self.templates.insert(name, template);
    }

    /// Get a template by name.
    #[must_use]
    pub fn get_template(&self, name: &str) -> Option<&ConversionTemplate> {
        self.templates.get(name)
    }

    /// Remove a template.
    pub fn remove_template(&mut self, name: &str) -> Option<ConversionTemplate> {
        self.templates.remove(name)
    }

    /// List all template names.
    #[must_use]
    pub fn list_templates(&self) -> Vec<&String> {
        self.templates.keys().collect()
    }

    /// Apply a template with variables.
    pub fn apply_template(
        &self,
        template_name: &str,
        variables: &super::TemplateVariables,
    ) -> Result<ConversionOptions> {
        let template = self.get_template(template_name).ok_or_else(|| {
            ConversionError::Template(format!("Template not found: {template_name}"))
        })?;

        template.apply(variables)
    }

    /// Load templates from a file.
    pub fn load_from_file<P: AsRef<Path>>(&mut self, _path: P) -> Result<()> {
        // Placeholder for loading templates from JSON/TOML
        Ok(())
    }

    /// Save templates to a file.
    pub fn save_to_file<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        // Placeholder for saving templates to JSON/TOML
        Ok(())
    }
}

impl Default for TemplateSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// A conversion template.
#[derive(Debug, Clone)]
pub struct ConversionTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Output filename pattern
    pub output_pattern: String,
    /// Base conversion options
    pub options: ConversionOptions,
    /// Required variables
    pub required_variables: Vec<String>,
}

impl ConversionTemplate {
    /// Create a new template.
    pub fn new<S: Into<String>>(name: S, output_pattern: S, options: ConversionOptions) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            output_pattern: output_pattern.into(),
            options,
            required_variables: Vec::new(),
        }
    }

    /// Set the template description.
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }

    /// Add a required variable.
    pub fn with_required_variable<S: Into<String>>(mut self, var: S) -> Self {
        self.required_variables.push(var.into());
        self
    }

    /// Apply the template with variables.
    pub fn apply(&self, variables: &super::TemplateVariables) -> Result<ConversionOptions> {
        // Check required variables
        for var in &self.required_variables {
            if !variables.has(var) {
                return Err(ConversionError::Template(format!(
                    "Required variable missing: {var}"
                )));
            }
        }

        // Return options (in a real implementation, this would substitute variables)
        Ok(self.options.clone())
    }

    /// Generate output filename from variables.
    #[must_use]
    pub fn generate_output_name(&self, variables: &super::TemplateVariables) -> String {
        variables.substitute(&self.output_pattern)
    }

    /// Validate the template.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(ConversionError::Template(
                "Template name cannot be empty".to_string(),
            ));
        }

        if self.output_pattern.is_empty() {
            return Err(ConversionError::Template(
                "Output pattern cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let system = TemplateSystem::new();
        assert!(system.templates.is_empty());
    }

    #[test]
    fn test_add_template() {
        let mut system = TemplateSystem::new();
        let template = ConversionTemplate::new("test", "{name}.mp4", ConversionOptions::default());

        system.add_template("test".to_string(), template);
        assert_eq!(system.templates.len(), 1);
        assert!(system.get_template("test").is_some());
    }

    #[test]
    fn test_remove_template() {
        let mut system = TemplateSystem::new();
        let template = ConversionTemplate::new("test", "{name}.mp4", ConversionOptions::default());

        system.add_template("test".to_string(), template);
        let removed = system.remove_template("test");

        assert!(removed.is_some());
        assert!(system.templates.is_empty());
    }

    #[test]
    fn test_list_templates() {
        let mut system = TemplateSystem::new();
        system.add_template(
            "test1".to_string(),
            ConversionTemplate::new("test1", "out.mp4", ConversionOptions::default()),
        );
        system.add_template(
            "test2".to_string(),
            ConversionTemplate::new("test2", "out.mp4", ConversionOptions::default()),
        );

        let list = system.list_templates();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_template_validation() {
        let template = ConversionTemplate::new("test", "output.mp4", ConversionOptions::default());

        assert!(template.validate().is_ok());

        let invalid = ConversionTemplate::new("", "output.mp4", ConversionOptions::default());

        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_template_builder() {
        let template = ConversionTemplate::new("test", "{name}.mp4", ConversionOptions::default())
            .with_description("Test template")
            .with_required_variable("name");

        assert_eq!(template.description, "Test template");
        assert_eq!(template.required_variables.len(), 1);
    }
}
