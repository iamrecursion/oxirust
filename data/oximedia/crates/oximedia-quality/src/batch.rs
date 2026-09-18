//! Batch quality assessment for multiple videos.
//!
//! Provides utilities for assessing quality of multiple video files
//! and generating comparison reports.

use crate::{
    metrics::{AggregatedMetrics, ComparisonReport, VideoInfo},
    Frame, MetricType, PoolingMethod, QualityAssessor, QualityScore,
};
use oximedia_core::OxiResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for batch assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Metrics to compute
    pub metrics: Vec<MetricType>,
    /// Pooling method for temporal aggregation
    pub pooling: PoolingMethod,
    /// Enable parallel processing
    pub parallel: bool,
    /// Report progress every N frames
    pub progress_interval: Option<usize>,
}

impl BatchConfig {
    /// Creates a new batch configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: vec![
                MetricType::Psnr,
                MetricType::Ssim,
                MetricType::MsSsim,
                MetricType::Vmaf,
            ],
            pooling: PoolingMethod::Mean,
            parallel: true,
            progress_interval: Some(100),
        }
    }

    /// Adds a metric to the configuration.
    pub fn with_metric(mut self, metric: MetricType) -> Self {
        self.metrics.push(metric);
        self
    }

    /// Sets the pooling method.
    pub fn with_pooling(mut self, pooling: PoolingMethod) -> Self {
        self.pooling = pooling;
        self
    }

    /// Enables or disables parallel processing.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of batch assessment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResult {
    /// Comparison report
    pub report: ComparisonReport,
    /// Total frames processed
    pub frames_processed: usize,
    /// Processing time in seconds
    pub processing_time: f64,
}

impl BatchResult {
    /// Creates a new batch result.
    #[must_use]
    pub fn new(report: ComparisonReport, frames_processed: usize, processing_time: f64) -> Self {
        Self {
            report,
            frames_processed,
            processing_time,
        }
    }

    /// Returns the overall quality score.
    #[must_use]
    pub fn overall_quality(&self) -> f64 {
        self.report.overall_quality
    }

    /// Exports result to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Batch quality assessment coordinator.
pub struct BatchAssessment {
    /// Configuration
    config: BatchConfig,
    /// Quality assessor
    assessor: QualityAssessor,
}

impl BatchAssessment {
    /// Creates a new batch assessment with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BatchConfig::new(),
            assessor: QualityAssessor::new(),
        }
    }

    /// Creates a batch assessment with custom configuration.
    #[must_use]
    pub fn with_config(config: BatchConfig) -> Self {
        Self {
            config,
            assessor: QualityAssessor::new(),
        }
    }

    /// Assesses quality between reference and distorted frame sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if frame counts don't match or assessment fails.
    pub fn assess_frames(
        &self,
        reference_frames: &[Frame],
        distorted_frames: &[Frame],
        reference_info: VideoInfo,
        distorted_info: VideoInfo,
    ) -> OxiResult<BatchResult> {
        if reference_frames.len() != distorted_frames.len() {
            return Err(oximedia_core::OxiError::InvalidData(
                "Reference and distorted frame counts must match".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        let mut report = ComparisonReport::new(reference_info, distorted_info);

        // Process each metric
        for &metric in &self.config.metrics {
            if !metric.requires_reference() {
                continue; // Skip no-reference metrics in batch mode
            }

            let scores = self.assess_metric(reference_frames, distorted_frames, metric)?;
            let aggregated = AggregatedMetrics::from_scores(&scores);
            report.add_metrics(aggregated);
        }

        report.compute_overall_quality();

        let processing_time = start_time.elapsed().as_secs_f64();

        Ok(BatchResult::new(
            report,
            reference_frames.len(),
            processing_time,
        ))
    }

    /// Assesses a single metric across all frames.
    fn assess_metric(
        &self,
        reference_frames: &[Frame],
        distorted_frames: &[Frame],
        metric: MetricType,
    ) -> OxiResult<Vec<QualityScore>> {
        if self.config.parallel {
            self.assess_metric_parallel(reference_frames, distorted_frames, metric)
        } else {
            self.assess_metric_sequential(reference_frames, distorted_frames, metric)
        }
    }

    /// Assesses metric sequentially.
    fn assess_metric_sequential(
        &self,
        reference_frames: &[Frame],
        distorted_frames: &[Frame],
        metric: MetricType,
    ) -> OxiResult<Vec<QualityScore>> {
        let mut scores = Vec::with_capacity(reference_frames.len());

        for (i, (ref_frame, dist_frame)) in reference_frames
            .iter()
            .zip(distorted_frames.iter())
            .enumerate()
        {
            let mut score = self.assessor.assess(ref_frame, dist_frame, metric)?;
            score.frame_num = Some(i);
            scores.push(score);

            if let Some(interval) = self.config.progress_interval {
                if (i + 1) % interval == 0 {
                    // Progress reporting would go here
                }
            }
        }

        Ok(scores)
    }

    /// Assesses metric in parallel.
    fn assess_metric_parallel(
        &self,
        reference_frames: &[Frame],
        distorted_frames: &[Frame],
        metric: MetricType,
    ) -> OxiResult<Vec<QualityScore>> {
        let scores: Result<Vec<QualityScore>, _> = reference_frames
            .par_iter()
            .zip(distorted_frames.par_iter())
            .enumerate()
            .map(|(i, (ref_frame, dist_frame))| {
                let mut score = self.assessor.assess(ref_frame, dist_frame, metric)?;
                score.frame_num = Some(i);
                Ok(score)
            })
            .collect();

        scores
    }

    /// Assesses no-reference quality for a single video.
    ///
    /// # Errors
    ///
    /// Returns an error if assessment fails.
    pub fn assess_no_reference(
        &self,
        frames: &[Frame],
        _video_info: VideoInfo,
        metrics: &[MetricType],
    ) -> OxiResult<Vec<AggregatedMetrics>> {
        let mut results = Vec::new();

        for &metric in metrics {
            if !metric.is_no_reference() {
                continue;
            }

            let scores: Vec<QualityScore> = frames
                .iter()
                .enumerate()
                .map(|(i, frame)| -> OxiResult<QualityScore> {
                    let mut score = self.assessor.assess_no_reference(frame, metric)?;
                    score.frame_num = Some(i);
                    Ok(score)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let aggregated = AggregatedMetrics::from_scores(&scores);
            results.push(aggregated);
        }

        Ok(results)
    }
}

impl Default for BatchAssessment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    fn create_test_frames(count: usize, width: usize, height: usize) -> Vec<Frame> {
        (0..count)
            .map(|i| create_test_frame(width, height, (100 + i) as u8))
            .collect()
    }

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::new()
            .with_metric(MetricType::Vif)
            .with_pooling(PoolingMethod::HarmonicMean);

        assert!(config.metrics.contains(&MetricType::Vif));
        assert_eq!(config.pooling, PoolingMethod::HarmonicMean);
    }

    #[test]
    fn test_batch_assessment() {
        let batch = BatchAssessment::new();

        let ref_frames = create_test_frames(5, 64, 64);
        let dist_frames = create_test_frames(5, 64, 64);

        let ref_info = VideoInfo::new(
            "reference.mp4".to_string(),
            64,
            64,
            5,
            30.0,
            "yuv420p".to_string(),
        );
        let dist_info = VideoInfo::new(
            "distorted.mp4".to_string(),
            64,
            64,
            5,
            30.0,
            "yuv420p".to_string(),
        );

        let result = batch
            .assess_frames(&ref_frames, &dist_frames, ref_info, dist_info)
            .expect("should succeed in test");

        assert_eq!(result.frames_processed, 5);
        assert!(result.processing_time > 0.0);
        assert!(!result.report.metrics.is_empty());
    }

    #[test]
    fn test_no_reference_assessment() {
        let batch = BatchAssessment::new();
        let frames = create_test_frames(3, 64, 64);

        let video_info = VideoInfo::new(
            "video.mp4".to_string(),
            64,
            64,
            3,
            30.0,
            "yuv420p".to_string(),
        );

        let metrics = vec![MetricType::Blur, MetricType::Noise];

        let results = batch
            .assess_no_reference(&frames, video_info, &metrics)
            .expect("should succeed in test");

        assert!(!results.is_empty());
    }

    #[test]
    fn test_mismatched_frame_counts() {
        let batch = BatchAssessment::new();

        let ref_frames = create_test_frames(5, 64, 64);
        let dist_frames = create_test_frames(3, 64, 64);

        let ref_info = VideoInfo::new(
            "ref.mp4".to_string(),
            64,
            64,
            5,
            30.0,
            "yuv420p".to_string(),
        );
        let dist_info = VideoInfo::new(
            "dist.mp4".to_string(),
            64,
            64,
            3,
            30.0,
            "yuv420p".to_string(),
        );

        let result = batch.assess_frames(&ref_frames, &dist_frames, ref_info, dist_info);
        assert!(result.is_err());
    }
}
