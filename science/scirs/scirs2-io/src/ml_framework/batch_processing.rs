//! Batch processing support for ML models

use crate::error::Result;
use crate::ml_framework::{datasets, MLTensor};
use scirs2_core::parallel_ops::*;

/// Batch processor for ML models
pub struct BatchProcessor {
    batch_size: usize,
    #[allow(dead_code)]
    prefetch_factor: usize,
}

impl BatchProcessor {
    pub fn new(batchsize: usize) -> Self {
        Self {
            batch_size: batchsize,
            prefetch_factor: 2,
        }
    }

    /// Process data in batches
    pub fn process_batches<F>(&self, data: &[MLTensor], processfn: F) -> Result<Vec<MLTensor>>
    where
        F: Fn(&[MLTensor]) -> Result<Vec<MLTensor>> + Send + Sync,
    {
        let results: Result<Vec<Vec<MLTensor>>> =
            data.par_chunks(self.batch_size).map(processfn).collect();

        results.map(|chunks| chunks.into_iter().flatten().collect())
    }

    /// Create data loader
    pub fn create_dataloader(&self, dataset: &datasets::MLDataset) -> DataLoader {
        DataLoader {
            dataset: dataset.clone(),
            batch_size: self.batch_size,
            shuffle: false,
            current_idx: 0,
        }
    }
}

/// Data loader for batched iteration
#[derive(Clone)]
pub struct DataLoader {
    dataset: datasets::MLDataset,
    batch_size: usize,
    shuffle: bool,
    current_idx: usize,
}

impl Iterator for DataLoader {
    type Item = (Vec<MLTensor>, Option<Vec<MLTensor>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let features = self.dataset.features[self.current_idx..end_idx].to_vec();
        let labels = self
            .dataset
            .labels
            .as_ref()
            .map(|l| l[self.current_idx..end_idx].to_vec());

        self.current_idx = end_idx;
        Some((features, labels))
    }
}
