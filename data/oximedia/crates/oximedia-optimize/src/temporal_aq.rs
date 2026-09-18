#![allow(dead_code)]
//! Temporal adaptive quantization for video encoding.
//!
//! This module extends standard spatial AQ by incorporating temporal information
//! (motion activity, scene changes, fade detection) to dynamically adjust QP values
//! across frames. It helps allocate more bits to high-motion or visually complex
//! temporal segments while saving bits on static or slowly-changing regions.
//!
//! The [`TemporalAqEngine`] can be connected to the spatial [`AqEngine`](crate::aq::AqEngine)
//! via [`TemporalAqBridge`], which combines temporal and spatial QP offsets into
//! a unified per-block decision.

/// Number of frames kept in the sliding temporal window.
const DEFAULT_WINDOW_SIZE: usize = 16;

/// Per-frame temporal activity metrics.
#[derive(Debug, Clone)]
pub struct TemporalActivity {
    /// Frame index in decode order.
    pub frame_index: u64,
    /// Average absolute motion vector magnitude.
    pub avg_motion_magnitude: f64,
    /// Percentage of macroblocks with significant motion (0.0-1.0).
    pub motion_coverage: f64,
    /// Frame-level SAD (Sum of Absolute Differences) vs. previous frame.
    pub frame_sad: f64,
    /// Whether this frame is detected as a scene change.
    pub is_scene_change: bool,
    /// Fade strength (0.0 = no fade, 1.0 = full fade).
    pub fade_strength: f64,
}

impl TemporalActivity {
    /// Creates a new temporal activity measurement.
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            avg_motion_magnitude: 0.0,
            motion_coverage: 0.0,
            frame_sad: 0.0,
            is_scene_change: false,
            fade_strength: 0.0,
        }
    }

    /// Returns a composite temporal complexity score in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn complexity_score(&self) -> f64 {
        let motion_factor = (self.avg_motion_magnitude / 32.0).min(1.0);
        let coverage_factor = self.motion_coverage;
        let fade_factor = self.fade_strength * 0.5;
        ((motion_factor * 0.5 + coverage_factor * 0.3 + fade_factor * 0.2) as f64).min(1.0)
    }
}

/// Configuration for the temporal AQ engine.
#[derive(Debug, Clone)]
pub struct TemporalAqConfig {
    /// Number of frames in the sliding analysis window.
    pub window_size: usize,
    /// Maximum QP delta that can be applied temporally.
    pub max_qp_delta: i8,
    /// Sensitivity to motion changes (0.0-2.0, 1.0 = normal).
    pub motion_sensitivity: f64,
    /// SAD threshold for scene change detection.
    pub scene_change_threshold: f64,
    /// Fade detection sensitivity.
    pub fade_sensitivity: f64,
    /// Strength of temporal smoothing (0.0 = no smoothing, 1.0 = max).
    pub smoothing_factor: f64,
}

impl Default for TemporalAqConfig {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_WINDOW_SIZE,
            max_qp_delta: 6,
            motion_sensitivity: 1.0,
            scene_change_threshold: 50_000.0,
            fade_sensitivity: 0.5,
            smoothing_factor: 0.3,
        }
    }
}

/// Result of temporal AQ analysis for a single frame.
#[derive(Debug, Clone)]
pub struct TemporalAqResult {
    /// Recommended QP delta for this frame.
    pub qp_delta: i8,
    /// Temporal complexity score used for the decision.
    pub complexity: f64,
    /// Whether the frame is at a scene boundary.
    pub at_scene_boundary: bool,
    /// Smoothed complexity after temporal filtering.
    pub smoothed_complexity: f64,
}

/// Temporal adaptive quantization engine.
#[derive(Debug)]
pub struct TemporalAqEngine {
    /// Engine configuration.
    config: TemporalAqConfig,
    /// Sliding window of recent frame activities.
    window: Vec<TemporalActivity>,
    /// Running average of complexity scores.
    running_avg: f64,
    /// Number of frames processed.
    frames_processed: u64,
}

impl TemporalAqEngine {
    /// Creates a new temporal AQ engine with the given configuration.
    pub fn new(config: TemporalAqConfig) -> Self {
        Self {
            config,
            window: Vec::new(),
            running_avg: 0.5,
            frames_processed: 0,
        }
    }

    /// Creates a temporal AQ engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(TemporalAqConfig::default())
    }

    /// Returns the engine configuration.
    pub fn config(&self) -> &TemporalAqConfig {
        &self.config
    }

    /// Processes a new frame's temporal activity and returns AQ recommendations.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_frame(&mut self, activity: TemporalActivity) -> TemporalAqResult {
        let raw_complexity = activity.complexity_score() * self.config.motion_sensitivity;
        let is_scene_change =
            activity.is_scene_change || activity.frame_sad > self.config.scene_change_threshold;

        // Temporal smoothing
        let smoothed = if is_scene_change {
            raw_complexity
        } else {
            self.running_avg * self.config.smoothing_factor
                + raw_complexity * (1.0 - self.config.smoothing_factor)
        };

        // Update running average
        let alpha = 2.0 / (self.config.window_size as f64 + 1.0);
        self.running_avg = self.running_avg * (1.0 - alpha) + raw_complexity * alpha;

        // Compute QP delta: more complex = lower QP (more bits)
        let deviation = smoothed - 0.5;
        let delta_f = -deviation * f64::from(self.config.max_qp_delta) * 2.0;
        let clamped = delta_f
            .round()
            .max(f64::from(-self.config.max_qp_delta))
            .min(f64::from(self.config.max_qp_delta));

        #[allow(clippy::cast_possible_truncation)]
        let qp_delta = clamped as i8;

        // Maintain window
        self.window.push(activity);
        if self.window.len() > self.config.window_size {
            self.window.remove(0);
        }
        self.frames_processed += 1;

        TemporalAqResult {
            qp_delta,
            complexity: raw_complexity,
            at_scene_boundary: is_scene_change,
            smoothed_complexity: smoothed,
        }
    }

    /// Returns the number of frames processed so far.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Returns the current running average complexity.
    pub fn running_average(&self) -> f64 {
        self.running_avg
    }

    /// Returns the current window size (number of buffered frames).
    pub fn current_window_len(&self) -> usize {
        self.window.len()
    }

    /// Resets the engine state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.running_avg = 0.5;
        self.frames_processed = 0;
    }

    /// Computes the variance of complexity scores in the current window.
    #[allow(clippy::cast_precision_loss)]
    pub fn window_complexity_variance(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let scores: Vec<f64> = self.window.iter().map(|a| a.complexity_score()).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        var
    }

    /// Detects whether the current window represents a fade.
    pub fn detect_fade_in_window(&self) -> bool {
        if self.window.len() < 3 {
            return false;
        }
        let fade_count = self.window.iter().filter(|a| a.fade_strength > 0.1).count();
        #[allow(clippy::cast_precision_loss)]
        let ratio = fade_count as f64 / self.window.len() as f64;
        ratio > self.config.fade_sensitivity
    }

    /// Returns the window contents for analysis.
    pub fn window(&self) -> &[TemporalActivity] {
        &self.window
    }
}

/// Bridge between temporal AQ and spatial AQ engines for unified per-block QP adaptation.
///
/// The bridge combines temporal (frame-level) QP offsets from [`TemporalAqEngine`]
/// with spatial (block-level) QP offsets from [`AqEngine`](crate::aq::AqEngine) to produce
/// a final per-block QP decision.
#[derive(Debug)]
pub struct TemporalAqBridge {
    /// Temporal AQ engine for frame-level decisions.
    temporal_engine: TemporalAqEngine,
    /// Weight for temporal component (0.0-1.0).
    temporal_weight: f64,
    /// Weight for spatial component (0.0-1.0).
    spatial_weight: f64,
    /// Maximum combined QP delta.
    max_combined_delta: i8,
    /// Last temporal result for combining with spatial.
    last_temporal_result: Option<TemporalAqResult>,
}

impl TemporalAqBridge {
    /// Creates a new bridge with given weights.
    pub fn new(config: TemporalAqConfig, temporal_weight: f64, spatial_weight: f64) -> Self {
        Self {
            temporal_engine: TemporalAqEngine::new(config),
            temporal_weight: temporal_weight.clamp(0.0, 1.0),
            spatial_weight: spatial_weight.clamp(0.0, 1.0),
            max_combined_delta: 8,
            last_temporal_result: None,
        }
    }

    /// Creates a bridge with default balanced weights (0.4 temporal, 0.6 spatial).
    pub fn with_defaults() -> Self {
        Self::new(TemporalAqConfig::default(), 0.4, 0.6)
    }

    /// Processes a frame's temporal activity to update the temporal component.
    pub fn update_temporal(&mut self, activity: TemporalActivity) -> &TemporalAqResult {
        let result = self.temporal_engine.process_frame(activity);
        self.last_temporal_result = Some(result);
        // Safe: we just inserted it
        self.last_temporal_result.as_ref().expect("just inserted")
    }

    /// Combines temporal QP delta with a spatial AQ QP offset for a specific block.
    ///
    /// This is the main integration point: call `update_temporal` once per frame,
    /// then call `combine_with_spatial` for each block in that frame.
    #[allow(clippy::cast_possible_truncation)]
    pub fn combine_with_spatial(&self, spatial_qp_offset: i8) -> CombinedAqResult {
        let temporal_delta = self
            .last_temporal_result
            .as_ref()
            .map(|r| r.qp_delta)
            .unwrap_or(0);

        let temporal_f = f64::from(temporal_delta) * self.temporal_weight;
        let spatial_f = f64::from(spatial_qp_offset) * self.spatial_weight;
        let combined_f = temporal_f + spatial_f;

        let combined = combined_f
            .round()
            .max(f64::from(-self.max_combined_delta))
            .min(f64::from(self.max_combined_delta)) as i8;

        let at_scene_boundary = self
            .last_temporal_result
            .as_ref()
            .map(|r| r.at_scene_boundary)
            .unwrap_or(false);

        CombinedAqResult {
            final_qp_delta: combined,
            temporal_component: temporal_delta,
            spatial_component: spatial_qp_offset,
            at_scene_boundary,
        }
    }

    /// Returns the underlying temporal engine for direct access.
    pub fn temporal_engine(&self) -> &TemporalAqEngine {
        &self.temporal_engine
    }

    /// Returns the last temporal analysis result.
    pub fn last_temporal_result(&self) -> Option<&TemporalAqResult> {
        self.last_temporal_result.as_ref()
    }

    /// Sets the maximum combined delta.
    pub fn set_max_combined_delta(&mut self, max_delta: i8) {
        self.max_combined_delta = max_delta.max(1);
    }

    /// Resets the bridge state.
    pub fn reset(&mut self) {
        self.temporal_engine.reset();
        self.last_temporal_result = None;
    }
}

/// Combined spatial + temporal AQ result for a single block.
#[derive(Debug, Clone)]
pub struct CombinedAqResult {
    /// Final QP delta after combining temporal and spatial components.
    pub final_qp_delta: i8,
    /// Temporal component (frame-level).
    pub temporal_component: i8,
    /// Spatial component (block-level).
    pub spatial_component: i8,
    /// Whether this frame is at a scene boundary.
    pub at_scene_boundary: bool,
}

/// Estimates optimal QP adjustment for a given temporal complexity value.
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
pub fn estimate_qp_for_complexity(complexity: f64, base_qp: u8, max_delta: i8) -> u8 {
    let delta_f = -(complexity - 0.5) * f64::from(max_delta) * 2.0;
    let adjusted = f64::from(base_qp) + delta_f;
    adjusted.round().max(1.0).min(51.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_activity_new() {
        let ta = TemporalActivity::new(42);
        assert_eq!(ta.frame_index, 42);
        assert!((ta.avg_motion_magnitude - 0.0).abs() < f64::EPSILON);
        assert!(!ta.is_scene_change);
    }

    #[test]
    fn test_temporal_activity_complexity_score() {
        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 16.0; // 0.5 normalized
        ta.motion_coverage = 0.5;
        ta.fade_strength = 0.0;
        let score = ta.complexity_score();
        // 0.5*0.5 + 0.5*0.3 + 0.0 = 0.25 + 0.15 = 0.40
        assert!((score - 0.40).abs() < 0.01);
    }

    #[test]
    fn test_temporal_activity_complexity_clamped() {
        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 1000.0;
        ta.motion_coverage = 1.0;
        ta.fade_strength = 1.0;
        let score = ta.complexity_score();
        assert!(score <= 1.0);
    }

    #[test]
    fn test_temporal_aq_config_default() {
        let cfg = TemporalAqConfig::default();
        assert_eq!(cfg.window_size, DEFAULT_WINDOW_SIZE);
        assert_eq!(cfg.max_qp_delta, 6);
    }

    #[test]
    fn test_temporal_aq_engine_new() {
        let engine = TemporalAqEngine::with_defaults();
        assert_eq!(engine.frames_processed(), 0);
        assert_eq!(engine.current_window_len(), 0);
    }

    #[test]
    fn test_temporal_aq_process_static_frame() {
        let mut engine = TemporalAqEngine::with_defaults();
        let ta = TemporalActivity::new(0);
        let result = engine.process_frame(ta);
        // Low complexity -> positive QP delta (save bits)
        assert!(result.qp_delta >= 0);
        assert_eq!(engine.frames_processed(), 1);
    }

    #[test]
    fn test_temporal_aq_process_high_motion() {
        let mut engine = TemporalAqEngine::with_defaults();
        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 64.0;
        ta.motion_coverage = 0.9;
        let result = engine.process_frame(ta);
        // High complexity -> negative QP delta (more bits)
        assert!(result.qp_delta <= 0);
    }

    #[test]
    fn test_temporal_aq_scene_change() {
        let mut engine = TemporalAqEngine::with_defaults();
        let mut ta = TemporalActivity::new(0);
        ta.is_scene_change = true;
        ta.frame_sad = 100_000.0;
        let result = engine.process_frame(ta);
        assert!(result.at_scene_boundary);
    }

    #[test]
    fn test_temporal_aq_window_fill() {
        let config = TemporalAqConfig {
            window_size: 4,
            ..Default::default()
        };
        let mut engine = TemporalAqEngine::new(config);
        for i in 0..10 {
            let ta = TemporalActivity::new(i);
            engine.process_frame(ta);
        }
        assert_eq!(engine.current_window_len(), 4);
        assert_eq!(engine.frames_processed(), 10);
    }

    #[test]
    fn test_temporal_aq_reset() {
        let mut engine = TemporalAqEngine::with_defaults();
        engine.process_frame(TemporalActivity::new(0));
        engine.process_frame(TemporalActivity::new(1));
        engine.reset();
        assert_eq!(engine.frames_processed(), 0);
        assert_eq!(engine.current_window_len(), 0);
    }

    #[test]
    fn test_window_complexity_variance_empty() {
        let engine = TemporalAqEngine::with_defaults();
        assert!((engine.window_complexity_variance() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_window_complexity_variance_uniform() {
        let mut engine = TemporalAqEngine::with_defaults();
        for i in 0..5 {
            engine.process_frame(TemporalActivity::new(i));
        }
        // All frames have same zero complexity
        let var = engine.window_complexity_variance();
        assert!(var < 0.01);
    }

    #[test]
    fn test_detect_fade_empty_window() {
        let engine = TemporalAqEngine::with_defaults();
        assert!(!engine.detect_fade_in_window());
    }

    #[test]
    fn test_estimate_qp_for_complexity() {
        let qp = estimate_qp_for_complexity(0.5, 28, 6);
        assert_eq!(qp, 28); // neutral complexity = no change

        let qp_high = estimate_qp_for_complexity(1.0, 28, 6);
        assert!(qp_high < 28); // high complexity = lower QP

        let qp_low = estimate_qp_for_complexity(0.0, 28, 6);
        assert!(qp_low > 28); // low complexity = higher QP
    }

    #[test]
    fn test_estimate_qp_clamped() {
        let qp = estimate_qp_for_complexity(100.0, 1, 50);
        assert!(qp >= 1);
        assert!(qp <= 51);
    }

    // --- New tests for TemporalAqBridge ---

    #[test]
    fn test_bridge_creation() {
        let bridge = TemporalAqBridge::with_defaults();
        assert!(bridge.last_temporal_result().is_none());
        assert_eq!(bridge.temporal_engine().frames_processed(), 0);
    }

    #[test]
    fn test_bridge_update_temporal() {
        let mut bridge = TemporalAqBridge::with_defaults();
        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 20.0;
        ta.motion_coverage = 0.5;
        let _ = bridge.update_temporal(ta);
        assert!(bridge.last_temporal_result().is_some());
        let complexity = bridge
            .last_temporal_result()
            .map(|r| r.complexity)
            .unwrap_or(0.0);
        assert!(
            complexity > 0.0,
            "Complexity should be > 0 for motion: {}",
            complexity
        );
    }

    #[test]
    fn test_bridge_combine_no_temporal() {
        let bridge = TemporalAqBridge::with_defaults();
        let combined = bridge.combine_with_spatial(-3);
        // No temporal update yet, temporal component should be 0
        assert_eq!(combined.temporal_component, 0);
        assert_eq!(combined.spatial_component, -3);
        // Final delta should be weighted spatial only
        assert!(combined.final_qp_delta <= 0);
    }

    #[test]
    fn test_bridge_combine_with_temporal() {
        let mut bridge = TemporalAqBridge::new(TemporalAqConfig::default(), 0.5, 0.5);
        // High motion frame
        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 64.0;
        ta.motion_coverage = 0.9;
        bridge.update_temporal(ta);

        // Combine with spatial offset
        let combined = bridge.combine_with_spatial(-2);
        assert!(
            combined.final_qp_delta != 0,
            "Combined delta should be non-zero"
        );
    }

    #[test]
    fn test_bridge_scene_boundary_propagation() {
        let mut bridge = TemporalAqBridge::with_defaults();
        let mut ta = TemporalActivity::new(0);
        ta.is_scene_change = true;
        ta.frame_sad = 100_000.0;
        bridge.update_temporal(ta);
        let combined = bridge.combine_with_spatial(0);
        assert!(combined.at_scene_boundary);
    }

    #[test]
    fn test_bridge_combined_delta_clamping() {
        let mut bridge = TemporalAqBridge::new(
            TemporalAqConfig {
                max_qp_delta: 6,
                ..Default::default()
            },
            1.0,
            1.0,
        );
        bridge.set_max_combined_delta(4);

        // Static frame gives positive temporal delta
        let ta = TemporalActivity::new(0);
        bridge.update_temporal(ta);
        // Combine with large spatial offset
        let combined = bridge.combine_with_spatial(6);
        assert!(
            combined.final_qp_delta <= 4,
            "Combined delta should be clamped: {}",
            combined.final_qp_delta
        );
        assert!(
            combined.final_qp_delta >= -4,
            "Combined delta should be clamped: {}",
            combined.final_qp_delta
        );
    }

    #[test]
    fn test_bridge_reset() {
        let mut bridge = TemporalAqBridge::with_defaults();
        bridge.update_temporal(TemporalActivity::new(0));
        bridge.reset();
        assert!(bridge.last_temporal_result().is_none());
        assert_eq!(bridge.temporal_engine().frames_processed(), 0);
    }

    #[test]
    fn test_bridge_high_motion_gets_negative_temporal_delta() {
        let mut bridge = TemporalAqBridge::with_defaults();
        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 64.0;
        ta.motion_coverage = 0.9;
        bridge.update_temporal(ta);
        let temporal_delta = bridge
            .last_temporal_result()
            .map(|r| r.qp_delta)
            .unwrap_or(0);
        assert!(
            temporal_delta <= 0,
            "High motion should give negative temporal delta: {}",
            temporal_delta
        );
    }

    #[test]
    fn test_bridge_static_gets_positive_temporal_delta() {
        let mut bridge = TemporalAqBridge::with_defaults();
        let ta = TemporalActivity::new(0);
        bridge.update_temporal(ta);
        let temporal_delta = bridge
            .last_temporal_result()
            .map(|r| r.qp_delta)
            .unwrap_or(0);
        assert!(
            temporal_delta >= 0,
            "Static frame should give non-negative temporal delta: {}",
            temporal_delta
        );
    }

    #[test]
    fn test_bridge_weights_affect_result() {
        // All temporal weight
        let mut bridge_t = TemporalAqBridge::new(TemporalAqConfig::default(), 1.0, 0.0);
        // All spatial weight
        let mut bridge_s = TemporalAqBridge::new(TemporalAqConfig::default(), 0.0, 1.0);

        let mut ta = TemporalActivity::new(0);
        ta.avg_motion_magnitude = 40.0;
        ta.motion_coverage = 0.7;
        bridge_t.update_temporal(ta.clone());
        bridge_s.update_temporal(ta);

        let combined_t = bridge_t.combine_with_spatial(-3);
        let combined_s = bridge_s.combine_with_spatial(-3);

        // With all-temporal weight, spatial component should be zeroed
        // With all-spatial weight, temporal component should be zeroed
        // They should differ
        assert_ne!(
            combined_t.final_qp_delta, combined_s.final_qp_delta,
            "Different weights should produce different results"
        );
    }

    #[test]
    fn test_combined_aq_result_fields() {
        let result = CombinedAqResult {
            final_qp_delta: -2,
            temporal_component: -3,
            spatial_component: -1,
            at_scene_boundary: false,
        };
        assert_eq!(result.final_qp_delta, -2);
        assert_eq!(result.temporal_component, -3);
        assert_eq!(result.spatial_component, -1);
        assert!(!result.at_scene_boundary);
    }
}
