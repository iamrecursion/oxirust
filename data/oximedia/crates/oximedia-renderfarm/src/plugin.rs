// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Plugin system for custom job types and schedulers.

use crate::error::Result;
use std::collections::HashMap;

/// Plugin interface
pub trait Plugin: Send + Sync {
    /// Plugin name
    fn name(&self) -> &str;

    /// Plugin version
    fn version(&self) -> &str;

    /// Initialize plugin
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown plugin
    fn shutdown(&mut self) -> Result<()>;
}

/// Plugin registry
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn Plugin>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register plugin
    pub fn register(&mut self, plugin: Box<dyn Plugin>) -> Result<()> {
        let name = plugin.name().to_string();
        self.plugins.insert(name, plugin);
        Ok(())
    }

    /// Get plugin
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Box<dyn Plugin>> {
        self.plugins.get(name)
    }

    /// List all plugins
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.plugins.keys().map(String::as_str).collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn name(&self) -> &'static str {
            "test"
        }

        fn version(&self) -> &'static str {
            "1.0"
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_plugin_registry() -> Result<()> {
        let mut registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin);

        registry.register(plugin)?;
        assert!(registry.get("test").is_some());

        Ok(())
    }
}
