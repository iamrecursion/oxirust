//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::random::prelude::*;

use super::types::{RealWorldConfig, RealWorldDatasets};

/// Convenience functions for loading specific real-world datasets
#[allow(dead_code)]
pub fn load_adult() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_adult()
}
/// Load Titanic dataset
#[allow(dead_code)]
pub fn load_titanic() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_titanic()
}
/// Load California Housing dataset
#[allow(dead_code)]
pub fn load_california_housing() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_california_housing()
}
/// Load Heart Disease dataset
#[allow(dead_code)]
pub fn load_heart_disease() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_heart_disease()
}
/// Load Red Wine Quality dataset
#[allow(dead_code)]
pub fn load_red_wine_quality() -> Result<Dataset> {
    let config = RealWorldConfig::default();
    let mut loader = RealWorldDatasets::new(config)?;
    loader.load_red_wine_quality()
}
/// List all available real-world datasets
#[allow(dead_code)]
pub fn list_real_world_datasets() -> Vec<String> {
    let config = RealWorldConfig::default();
    let loader = RealWorldDatasets::new(config).expect("Operation failed");
    loader.list_datasets()
}
