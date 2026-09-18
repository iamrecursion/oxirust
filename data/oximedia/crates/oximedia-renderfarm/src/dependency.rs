// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Dependency management and resolution.

use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Dependency type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Asset file
    Asset(PathBuf),
    /// Plugin
    Plugin {
        /// Plugin name
        name: String,
        /// Plugin version
        version: String,
    },
    /// Font
    Font(String),
    /// License
    License(String),
    /// Python package
    PythonPackage {
        /// Package name
        name: String,
        /// Package version
        version: String,
    },
}

/// Dependency status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyStatus {
    /// Available
    Available,
    /// Missing
    Missing,
    /// Downloading
    Downloading,
    /// Version mismatch
    VersionMismatch,
}

/// Dependency info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Dependency type
    pub dep_type: DependencyType,
    /// Status
    pub status: DependencyStatus,
    /// Size in bytes
    pub size: u64,
    /// Checksum
    pub checksum: Option<String>,
    /// Last checked
    pub last_checked: DateTime<Utc>,
}

/// Dependency graph
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Dependencies
    pub dependencies: HashMap<String, Dependency>,
    /// Dependency relationships
    pub relationships: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            relationships: HashMap::new(),
        }
    }

    /// Add dependency
    pub fn add_dependency(&mut self, id: String, dependency: Dependency) {
        self.dependencies.insert(id, dependency);
    }

    /// Add relationship
    pub fn add_relationship(&mut self, parent: String, child: String) {
        self.relationships.entry(parent).or_default().insert(child);
    }

    /// Get dependency
    #[must_use]
    pub fn get_dependency(&self, id: &str) -> Option<&Dependency> {
        self.dependencies.get(id)
    }

    /// Get all dependencies
    #[must_use]
    pub fn get_all_dependencies(&self) -> Vec<&Dependency> {
        self.dependencies.values().collect()
    }

    /// Check if all dependencies are available
    #[must_use]
    pub fn all_available(&self) -> bool {
        self.dependencies
            .values()
            .all(|d| d.status == DependencyStatus::Available)
    }

    /// Get missing dependencies
    #[must_use]
    pub fn get_missing(&self) -> Vec<&Dependency> {
        self.dependencies
            .values()
            .filter(|d| d.status == DependencyStatus::Missing)
            .collect()
    }

    /// Resolve dependencies (topological sort)
    pub fn resolve(&self) -> Result<Vec<String>> {
        let mut resolved = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        for id in self.dependencies.keys() {
            if !visited.contains(id) {
                self.visit(id, &mut visited, &mut temp_mark, &mut resolved)?;
            }
        }

        Ok(resolved)
    }

    fn visit(
        &self,
        id: &str,
        visited: &mut HashSet<String>,
        temp_mark: &mut HashSet<String>,
        resolved: &mut Vec<String>,
    ) -> Result<()> {
        if temp_mark.contains(id) {
            return Err(Error::Dependency(
                "Circular dependency detected".to_string(),
            ));
        }

        if !visited.contains(id) {
            temp_mark.insert(id.to_string());

            if let Some(children) = self.relationships.get(id) {
                for child in children {
                    self.visit(child, visited, temp_mark, resolved)?;
                }
            }

            temp_mark.remove(id);
            visited.insert(id.to_string());
            resolved.push(id.to_string());
        }

        Ok(())
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Dependency resolver
pub struct DependencyResolver {
    /// Asset paths
    asset_paths: Vec<PathBuf>,
    /// Plugin registry
    plugins: HashMap<String, HashSet<String>>,
    /// Font registry
    fonts: HashSet<String>,
}

impl DependencyResolver {
    /// Create a new dependency resolver
    #[must_use]
    pub fn new() -> Self {
        Self {
            asset_paths: Vec::new(),
            plugins: HashMap::new(),
            fonts: HashSet::new(),
        }
    }

    /// Add asset path
    pub fn add_asset_path<P: Into<PathBuf>>(&mut self, path: P) {
        self.asset_paths.push(path.into());
    }

    /// Register plugin
    pub fn register_plugin(&mut self, name: String, version: String) {
        self.plugins.entry(name).or_default().insert(version);
    }

    /// Register font
    pub fn register_font(&mut self, name: String) {
        self.fonts.insert(name);
    }

    /// Resolve dependency
    pub fn resolve(&self, dep: &DependencyType) -> Result<DependencyStatus> {
        match dep {
            DependencyType::Asset(path) => {
                if path.exists() {
                    Ok(DependencyStatus::Available)
                } else {
                    // Check in asset paths
                    for base_path in &self.asset_paths {
                        let full_path = base_path.join(path);
                        if full_path.exists() {
                            return Ok(DependencyStatus::Available);
                        }
                    }
                    Ok(DependencyStatus::Missing)
                }
            }
            DependencyType::Plugin { name, version } => {
                if let Some(versions) = self.plugins.get(name) {
                    if versions.contains(version) {
                        Ok(DependencyStatus::Available)
                    } else {
                        Ok(DependencyStatus::VersionMismatch)
                    }
                } else {
                    Ok(DependencyStatus::Missing)
                }
            }
            DependencyType::Font(name) => {
                if self.fonts.contains(name) {
                    Ok(DependencyStatus::Available)
                } else {
                    Ok(DependencyStatus::Missing)
                }
            }
            DependencyType::License(_) => {
                // Simplified: always available
                Ok(DependencyStatus::Available)
            }
            DependencyType::PythonPackage {
                name: _name,
                version: _version,
            } => {
                // Simplified: check if we have the package
                Ok(DependencyStatus::Missing)
            }
        }
    }

    /// Resolve all dependencies in a graph
    pub fn resolve_graph(&self, graph: &mut DependencyGraph) -> Result<()> {
        for dep in graph.dependencies.values_mut() {
            let status = self.resolve(&dep.dep_type)?;
            dep.status = status;
            dep.last_checked = Utc::now();
        }
        Ok(())
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();

        let dep = Dependency {
            dep_type: DependencyType::Asset(PathBuf::from("test.png")),
            status: DependencyStatus::Available,
            size: 1000,
            checksum: None,
            last_checked: Utc::now(),
        };

        graph.add_dependency("dep1".to_string(), dep);
        assert!(graph.get_dependency("dep1").is_some());
    }

    #[test]
    fn test_dependency_relationships() {
        let mut graph = DependencyGraph::new();

        graph.add_relationship("parent".to_string(), "child1".to_string());
        graph.add_relationship("parent".to_string(), "child2".to_string());

        assert_eq!(
            graph
                .relationships
                .get("parent")
                .expect("should succeed in test")
                .len(),
            2
        );
    }

    #[test]
    fn test_dependency_resolver() -> Result<()> {
        let mut resolver = DependencyResolver::new();
        resolver.register_plugin("blender".to_string(), "3.0".to_string());

        let dep = DependencyType::Plugin {
            name: "blender".to_string(),
            version: "3.0".to_string(),
        };

        let status = resolver.resolve(&dep)?;
        assert_eq!(status, DependencyStatus::Available);

        Ok(())
    }

    #[test]
    fn test_missing_dependency() -> Result<()> {
        let resolver = DependencyResolver::new();
        let dep = DependencyType::Plugin {
            name: "missing".to_string(),
            version: "1.0".to_string(),
        };

        let status = resolver.resolve(&dep)?;
        assert_eq!(status, DependencyStatus::Missing);

        Ok(())
    }

    #[test]
    fn test_resolve_graph() -> Result<()> {
        let mut graph = DependencyGraph::new();
        let mut resolver = DependencyResolver::new();

        resolver.register_plugin("plugin1".to_string(), "1.0".to_string());

        let dep = Dependency {
            dep_type: DependencyType::Plugin {
                name: "plugin1".to_string(),
                version: "1.0".to_string(),
            },
            status: DependencyStatus::Missing,
            size: 0,
            checksum: None,
            last_checked: Utc::now(),
        };

        graph.add_dependency("dep1".to_string(), dep);
        resolver.resolve_graph(&mut graph)?;

        assert_eq!(
            graph
                .get_dependency("dep1")
                .expect("should succeed in test")
                .status,
            DependencyStatus::Available
        );

        Ok(())
    }
}
