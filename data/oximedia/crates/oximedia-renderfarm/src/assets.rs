// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Asset management and distribution.

use crate::error::{Error, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Asset type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetType {
    /// Texture image
    Texture,
    /// 3D model
    Model,
    /// Audio file
    Audio,
    /// Video file
    Video,
    /// Cache file
    Cache,
    /// Plugin
    Plugin,
    /// Font
    Font,
    /// Other
    Other,
}

/// Asset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Asset ID
    pub id: String,
    /// File path
    pub path: PathBuf,
    /// Asset type
    pub asset_type: AssetType,
    /// Size in bytes
    pub size: u64,
    /// Checksum
    pub checksum: String,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Last accessed
    pub last_accessed: DateTime<Utc>,
    /// Reference count
    pub reference_count: u32,
}

/// Asset manager
pub struct AssetManager {
    assets: HashMap<String, Asset>,
    path_to_id: HashMap<PathBuf, String>,
}

impl AssetManager {
    /// Create a new asset manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
            path_to_id: HashMap::new(),
        }
    }

    /// Register asset
    pub fn register_asset(&mut self, asset: Asset) -> Result<()> {
        let id = asset.id.clone();
        let path = asset.path.clone();

        self.assets.insert(id.clone(), asset);
        self.path_to_id.insert(path, id);

        Ok(())
    }

    /// Get asset by ID
    #[must_use]
    pub fn get_asset(&self, id: &str) -> Option<&Asset> {
        self.assets.get(id)
    }

    /// Get asset by path
    #[must_use]
    pub fn get_asset_by_path(&self, path: &PathBuf) -> Option<&Asset> {
        let id = self.path_to_id.get(path)?;
        self.assets.get(id)
    }

    /// Increment reference count
    pub fn add_reference(&mut self, id: &str) -> Result<()> {
        let asset = self
            .assets
            .get_mut(id)
            .ok_or_else(|| Error::AssetNotFound(id.to_string()))?;

        asset.reference_count += 1;
        asset.last_accessed = Utc::now();

        Ok(())
    }

    /// Decrement reference count
    pub fn remove_reference(&mut self, id: &str) -> Result<()> {
        let asset = self
            .assets
            .get_mut(id)
            .ok_or_else(|| Error::AssetNotFound(id.to_string()))?;

        asset.reference_count = asset.reference_count.saturating_sub(1);

        Ok(())
    }

    /// Get unreferenced assets
    #[must_use]
    pub fn get_unreferenced(&self) -> Vec<&Asset> {
        self.assets
            .values()
            .filter(|a| a.reference_count == 0)
            .collect()
    }

    /// Get total asset size
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.assets.values().map(|a| a.size).sum()
    }

    /// List all assets
    #[must_use]
    pub fn list_assets(&self) -> Vec<&Asset> {
        self.assets.values().collect()
    }
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_asset() -> Asset {
        Asset {
            id: uuid::Uuid::new_v4().to_string(),
            path: PathBuf::from("/assets/test.png"),
            asset_type: AssetType::Texture,
            size: 1000,
            checksum: "abc123".to_string(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            reference_count: 0,
        }
    }

    #[test]
    fn test_asset_manager_creation() {
        let manager = AssetManager::new();
        assert_eq!(manager.assets.len(), 0);
    }

    #[test]
    fn test_register_asset() -> Result<()> {
        let mut manager = AssetManager::new();
        let asset = create_test_asset();
        let id = asset.id.clone();

        manager.register_asset(asset)?;
        assert!(manager.get_asset(&id).is_some());

        Ok(())
    }

    #[test]
    fn test_get_asset_by_path() -> Result<()> {
        let mut manager = AssetManager::new();
        let asset = create_test_asset();
        let path = asset.path.clone();

        manager.register_asset(asset)?;
        assert!(manager.get_asset_by_path(&path).is_some());

        Ok(())
    }

    #[test]
    fn test_reference_counting() -> Result<()> {
        let mut manager = AssetManager::new();
        let asset = create_test_asset();
        let id = asset.id.clone();

        manager.register_asset(asset)?;

        manager.add_reference(&id)?;
        assert_eq!(
            manager
                .get_asset(&id)
                .expect("should succeed in test")
                .reference_count,
            1
        );

        manager.remove_reference(&id)?;
        assert_eq!(
            manager
                .get_asset(&id)
                .expect("should succeed in test")
                .reference_count,
            0
        );

        Ok(())
    }

    #[test]
    fn test_unreferenced_assets() -> Result<()> {
        let mut manager = AssetManager::new();

        let asset1 = create_test_asset();
        let id1 = asset1.id.clone();
        manager.register_asset(asset1)?;

        let asset2 = create_test_asset();
        let _id2 = asset2.id.clone();
        manager.register_asset(asset2)?;

        manager.add_reference(&id1)?;

        let unreferenced = manager.get_unreferenced();
        assert_eq!(unreferenced.len(), 1);

        Ok(())
    }
}
