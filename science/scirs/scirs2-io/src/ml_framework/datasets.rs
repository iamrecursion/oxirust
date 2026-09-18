//! Dataset utilities for ML frameworks
#![allow(dead_code)]

use crate::ml_framework::types::MLTensor;
use std::collections::HashMap;

/// ML dataset container
#[derive(Clone)]
pub struct MLDataset {
    pub features: Vec<MLTensor>,
    pub labels: Option<Vec<MLTensor>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MLDataset {
    /// Create new dataset
    pub fn new(features: Vec<MLTensor>) -> Self {
        Self {
            features,
            labels: None,
            metadata: HashMap::new(),
        }
    }

    /// Add labels
    pub fn with_labels(mut self, labels: Vec<MLTensor>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Split into train/test sets
    pub fn train_test_split(&self, testratio: f32) -> (MLDataset, MLDataset) {
        let n = self.len();
        let test_size = (n as f32 * testratio) as usize;
        let train_size = n - test_size;

        let train_features = self.features[..train_size].to_vec();
        let test_features = self.features[train_size..].to_vec();

        let (train_labels, test_labels) = if let Some(labels) = &self.labels {
            (
                Some(labels[..train_size].to_vec()),
                Some(labels[train_size..].to_vec()),
            )
        } else {
            (None, None)
        };

        let train_dataset = MLDataset {
            features: train_features,
            labels: train_labels,
            metadata: self.metadata.clone(),
        };

        let test_dataset = MLDataset {
            features: test_features,
            labels: test_labels,
            metadata: self.metadata.clone(),
        };

        (train_dataset, test_dataset)
    }
}
