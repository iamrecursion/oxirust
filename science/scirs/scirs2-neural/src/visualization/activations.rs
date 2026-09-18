//! Layer activation and feature visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing layer activations,
//! feature maps, activation histograms, and attention patterns.

use super::config::VisualizationConfig;
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{ArrayD, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::NumAssign;
use serde::Serialize;
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
/// Layer activation visualizer
#[allow(dead_code)]
pub struct ActivationVisualizer<F: Float + Debug + ScalarOperand + NumAssign> {
    /// Model reference
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Cached activations
    activation_cache: HashMap<String, ArrayD<F>>,
}
/// Activation visualization options
#[derive(Debug, Clone, Serialize)]
pub struct ActivationVisualizationOptions {
    /// Layers to visualize
    pub target_layers: Vec<String>,
    /// Visualization type
    pub visualization_type: ActivationVisualizationType,
    /// Normalization method
    pub normalization: ActivationNormalization,
    /// Color mapping
    pub colormap: Colormap,
    /// Aggregation method for multi-channel data
    pub aggregation: ChannelAggregation,
}

/// Types of activation visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationVisualizationType {
    /// Feature maps as heatmaps
    FeatureMaps,
    /// Activation histograms
    Histograms,
    /// Statistics summary
    Statistics,
    /// Spatial attention maps
    AttentionMaps,
    /// Activation flow
    ActivationFlow,
}

/// Activation normalization methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ActivationNormalization {
    /// No normalization
    None,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Percentile-based normalization
    Percentile(f64, f64),
    /// Custom normalization function
    Custom(String),
}

/// Color mapping for visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Colormap {
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
    /// Inferno colormap
    Inferno,
    /// Jet colormap
    Jet,
    /// Grayscale
    Gray,
    /// Red-blue diverging
    RdBu,
    /// Custom colormap
    Custom(Vec<String>),
}

/// Channel aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ChannelAggregation {
    /// No aggregation (show all channels)
    None,
    /// Average across channels
    Mean,
    /// Maximum across channels
    Max,
    /// Minimum across channels
    Min,
    /// Standard deviation across channels
    Std,
    /// Select specific channels
    Select(Vec<usize>),
}

/// Activation statistics for a layer
#[derive(Debug, Clone, Serialize)]
pub struct ActivationStatistics<F: Float + Debug + serde::Serialize + NumAssign> {
    /// Layer name
    pub layer_name: String,
    /// Mean activation value
    pub mean: F,
    /// Standard deviation
    pub std: F,
    /// Minimum value
    pub min: F,
    /// Maximum value
    pub max: F,
    /// Percentiles (5%, 25%, 50%, 75%, 95%)
    pub percentiles: [F; 5],
    /// Sparsity (fraction of zero or near-zero activations)
    pub sparsity: f64,
    /// Dead neurons (always zero)
    pub dead_neurons: usize,
    /// Total neurons
    pub total_neurons: usize,
}

/// Feature map information
#[derive(Debug, Clone, Serialize)]
pub struct FeatureMapInfo {
    /// Feature map index
    pub feature_index: usize,
    /// Spatial dimensions (height, width)
    pub spatial_dims: (usize, usize),
    /// Channel dimension
    pub channels: usize,
    /// Activation range (min, max)
    pub activation_range: (f64, f64),
}

/// Activation histogram data
#[derive(Debug, Clone, Serialize)]
pub struct ActivationHistogram<F: Float + Debug + NumAssign> {
    /// Layer name
    pub layer_name: String,
    /// Histogram bins
    pub bins: Vec<F>,
    /// Bin counts
    pub counts: Vec<usize>,
    /// Bin edges
    pub edges: Vec<F>,
    /// Total sample count
    pub total_samples: usize,
}

// Implementation for ActivationVisualizer
impl<
        F: Float
            + Debug
            + std::fmt::Display
            + 'static
            + scirs2_core::numeric::FromPrimitive
            + ScalarOperand
            + Send
            + Sync
            + serde::Serialize
            + NumAssign,
    > ActivationVisualizer<F>
{
    /// Create a new activation visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            activation_cache: HashMap::new(),
        }
    }
    /// Visualize layer activations for given input
    pub fn visualize_activations(
        &mut self,
        input: &ArrayD<F>,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // Compute activations
        self.compute_activations(input, &options.target_layers)?;
        // Generate visualizations based on type
        match options.visualization_type {
            ActivationVisualizationType::FeatureMaps => self.generate_feature_maps(options),
            ActivationVisualizationType::Histograms => self.generate_histograms(options),
            ActivationVisualizationType::Statistics => self.generate_statistics(options),
            ActivationVisualizationType::AttentionMaps => self.generate_attention_maps(options),
            ActivationVisualizationType::ActivationFlow => self.generate_activation_flow(options),
        }
    }

    /// Get cached activations for a layer
    pub fn get_cached_activations(&self, layer_name: &str) -> Option<&ArrayD<F>> {
        self.activation_cache.get(layer_name)
    }

    /// Clear the activation cache
    pub fn clear_cache(&mut self) {
        self.activation_cache.clear();
    }

    /// Get activation statistics for all cached layers
    pub fn get_activation_statistics(&self) -> Result<Vec<ActivationStatistics<F>>> {
        let mut stats = Vec::new();
        for (layer_name, activations) in &self.activation_cache {
            let layer_stats = self.compute_layer_statistics(layer_name, activations)?;
            stats.push(layer_stats);
        }
        Ok(stats)
    }

    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    }
    fn compute_activations(&mut self, input: &ArrayD<F>, target_layers: &[String]) -> Result<()> {
        let mut current_output = input.clone();
        // Store input if requested
        if target_layers.is_empty() || target_layers.contains(&"input".to_string()) {
            self.activation_cache
                .insert("input".to_string(), input.clone());
        }
        // Run forward pass through each layer and capture activations
        for (layer_idx, layer) in self.model.layers().iter().enumerate() {
            current_output = layer.forward(&current_output)?;
            let layer_name = format!("layer_{}", layer_idx);
            // Store activation if this layer is requested or if no specific layers requested
            if target_layers.is_empty() || target_layers.contains(&layer_name) {
                self.activation_cache
                    .insert(layer_name, current_output.clone());
            }
        }
        Ok(())
    }
    fn generate_feature_maps(
        &self,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        let mut output_paths = Vec::new();
        for layer_name in &options.target_layers {
            if let Some(activations) = self.activation_cache.get(layer_name) {
                let feature_maps = self.process_activations_for_visualization(
                    activations,
                    &options.normalization,
                    &options.aggregation,
                )?;
                // Generate SVG visualization
                let svg_content =
                    self.create_feature_map_svg(&feature_maps, layer_name, &options.colormap)?;
                let output_path = self
                    .config
                    .output_dir
                    .join(format!("{}_feature_maps.svg", layer_name));
                std::fs::write(&output_path, svg_content).map_err(|e| {
                    NeuralError::IOError(format!("Failed to write feature map: {}", e))
                })?;
                output_paths.push(output_path);
            }
        }
        Ok(output_paths)
    }
    fn generate_histograms(
        &self,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        let mut output_paths = Vec::new();
        for layer_name in &options.target_layers {
            if let Some(activations) = self.activation_cache.get(layer_name) {
                let histogram = self.compute_activation_histogram(layer_name, activations, 50)?;
                // Generate SVG histogram
                let svg_content = self.create_histogram_svg(&histogram)?;
                let output_path = self
                    .config
                    .output_dir
                    .join(format!("{}_histogram.svg", layer_name));
                std::fs::write(&output_path, svg_content).map_err(|e| {
                    NeuralError::IOError(format!("Failed to write histogram: {}", e))
                })?;
                output_paths.push(output_path);
            }
        }
        Ok(output_paths)
    }
    fn generate_statistics(
        &self,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        let mut all_stats = Vec::new();
        for layer_name in &options.target_layers {
            if let Some(activations) = self.activation_cache.get(layer_name) {
                let stats = self.compute_layer_statistics(layer_name, activations)?;
                all_stats.push(stats);
            }
        }
        // Generate JSON statistics report
        let json_content = serde_json::to_string_pretty(&all_stats).map_err(|e| {
            NeuralError::SerializationError(format!("Failed to serialize statistics: {}", e))
        })?;
        let json_path = self.config.output_dir.join("activation_statistics.json");
        std::fs::write(&json_path, json_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write statistics: {}", e)))?;
        // Generate SVG statistics visualization
        let svg_content = self.create_statistics_svg(&all_stats)?;
        let svg_path = self.config.output_dir.join("activation_statistics.svg");
        std::fs::write(&svg_path, svg_content).map_err(|e| {
            NeuralError::IOError(format!("Failed to write statistics visualization: {}", e))
        })?;
        Ok(vec![json_path, svg_path])
    }
    fn generate_attention_maps(
        &self,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        let mut output_paths = Vec::new();
        for layer_name in &options.target_layers {
            if let Some(activations) = self.activation_cache.get(layer_name) {
                // Check if activations have spatial dimensions suitable for attention maps
                if activations.ndim() >= 3 {
                    let attention_map = self.compute_spatial_attention(activations)?;
                    let svg_content = self.create_attention_map_svg(&attention_map, layer_name)?;
                    let output_path = self
                        .config
                        .output_dir
                        .join(format!("{}_attention.svg", layer_name));
                    std::fs::write(&output_path, svg_content).map_err(|e| {
                        NeuralError::IOError(format!("Failed to write attention map: {}", e))
                    })?;
                    output_paths.push(output_path);
                }
            }
        }
        Ok(output_paths)
    }
    fn generate_activation_flow(
        &self,
        options: &ActivationVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // Compute activation flow between consecutive layers
        let mut flow_data = Vec::new();
        let sorted_layers: Vec<_> = options.target_layers.iter().collect();
        for i in 0..sorted_layers.len().saturating_sub(1) {
            let from_layer = sorted_layers[i];
            let to_layer = sorted_layers[i + 1];
            if let (Some(from_activations), Some(to_activations)) = (
                self.activation_cache.get(from_layer),
                self.activation_cache.get(to_layer),
            ) {
                let flow_intensity =
                    self.compute_activation_flow(from_activations, to_activations)?;
                flow_data.push((from_layer.clone(), to_layer.clone(), flow_intensity));
            }
        }
        if !flow_data.is_empty() {
            let svg_content = self.create_flow_diagram_svg(&flow_data)?;
            let output_path = self.config.output_dir.join("activation_flow.svg");
            std::fs::write(&output_path, svg_content).map_err(|e| {
                NeuralError::IOError(format!("Failed to write activation flow: {}", e))
            })?;
            Ok(vec![output_path])
        } else {
            Ok(Vec::new())
        }
    }
    fn compute_layer_statistics(
        &self,
        layer_name: &str,
        activations: &ArrayD<F>,
    ) -> Result<ActivationStatistics<F>> {
        let total_elements = activations.len();
        if total_elements == 0 {
            return Err(NeuralError::InvalidArgument(
                "Empty activation tensor".to_string(),
            ));
        }
        // Compute basic statistics
        let mut sum = F::zero();
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();
        let mut zero_count = 0;
        for &val in activations.iter() {
            sum += val;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
            if val.abs() < F::from(1e-6).unwrap_or(F::zero()) {
                zero_count += 1;
            }
        }
        let mean = sum / F::from(total_elements).unwrap_or(F::one());
        // Compute standard deviation
        let mut variance_sum = F::zero();
        for &val in activations.iter() {
            let diff = val - mean;
            variance_sum += diff * diff;
        }
        let variance = variance_sum / F::from(total_elements - 1).unwrap_or(F::one());
        let std = variance.sqrt();
        // Compute percentiles (simplified implementation)
        let mut sorted_values: Vec<F> = activations.iter().copied().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentiles = [
            sorted_values[total_elements * 5 / 100],  // 5%
            sorted_values[total_elements * 25 / 100], // 25%
            sorted_values[total_elements * 50 / 100], // 50%
            sorted_values[total_elements * 75 / 100], // 75%
            sorted_values[total_elements * 95 / 100], // 95%
        ];
        let sparsity = zero_count as f64 / total_elements as f64;
        Ok(ActivationStatistics {
            layer_name: layer_name.to_string(),
            mean,
            std,
            min: min_val,
            max: max_val,
            percentiles,
            sparsity,
            dead_neurons: zero_count,
            total_neurons: total_elements,
        })
    }
    fn process_activations_for_visualization(
        &self,
        activations: &ArrayD<F>,
        normalization: &ActivationNormalization,
        aggregation: &ChannelAggregation,
    ) -> Result<ArrayD<F>> {
        let mut processed = activations.clone();
        // Apply channel aggregation first
        processed = match aggregation {
            ChannelAggregation::None => processed,
            ChannelAggregation::Mean => {
                if processed.ndim() > 2 {
                    let mean_axis = processed.ndim() - 1; // Usually channel is last dimension
                    processed
                        .mean_axis(scirs2_core::ndarray::Axis(mean_axis))
                        .expect("Operation failed")
                        .insert_axis(scirs2_core::ndarray::Axis(mean_axis))
                } else {
                    processed
                }
            }
            ChannelAggregation::Max => {
                if processed.ndim() > 2 {
                    let max_axis = processed.ndim() - 1;
                    let max_values = processed.fold_axis(
                        scirs2_core::ndarray::Axis(max_axis),
                        F::neg_infinity(),
                        |&acc, &x| acc.max(x),
                    );
                    max_values.insert_axis(scirs2_core::ndarray::Axis(max_axis))
                } else {
                    processed
                }
            }
            ChannelAggregation::Min => {
                if processed.ndim() > 2 {
                    let min_axis = processed.ndim() - 1;
                    let min_values = processed.fold_axis(
                        scirs2_core::ndarray::Axis(min_axis),
                        F::infinity(),
                        |&acc, &x| acc.min(x),
                    );
                    min_values.insert_axis(scirs2_core::ndarray::Axis(min_axis))
                } else {
                    processed
                }
            }
            ChannelAggregation::Std => {
                if processed.ndim() > 2 {
                    let std_axis = processed.ndim() - 1;
                    let mean = processed
                        .mean_axis(scirs2_core::ndarray::Axis(std_axis))
                        .expect("Operation failed");
                    let variance =
                        processed.map_axis(scirs2_core::ndarray::Axis(std_axis), |channel| {
                            let mean_val = mean.iter().next().copied().unwrap_or(F::zero());
                            let variance_sum = channel
                                .iter()
                                .map(|&x| (x - mean_val) * (x - mean_val))
                                .fold(F::zero(), |acc, x| acc + x);
                            (variance_sum / F::from(channel.len()).unwrap_or(F::one())).sqrt()
                        });
                    variance.insert_axis(scirs2_core::ndarray::Axis(std_axis))
                } else {
                    processed
                }
            }
            ChannelAggregation::Select(channels) => {
                if processed.ndim() > 2 && !channels.is_empty() {
                    let channel_axis = processed.ndim() - 1;
                    let mut selected_slices = Vec::new();
                    for &channel_idx in channels {
                        if channel_idx < processed.shape()[channel_axis] {
                            let slice = processed
                                .index_axis(scirs2_core::ndarray::Axis(channel_axis), channel_idx);
                            selected_slices
                                .push(slice.insert_axis(scirs2_core::ndarray::Axis(channel_axis)));
                        }
                    }
                    if !selected_slices.is_empty() {
                        scirs2_core::ndarray::concatenate(
                            scirs2_core::ndarray::Axis(channel_axis),
                            &selected_slices.iter().map(|x| x.view()).collect::<Vec<_>>(),
                        )
                        .map_err(|_| {
                            NeuralError::DimensionMismatch(
                                "Failed to concatenate selected channels".to_string(),
                            )
                        })?
                    } else {
                        processed
                    }
                } else {
                    processed
                }
            }
        };
        // Apply normalization
        processed = match normalization {
            ActivationNormalization::None => processed,
            ActivationNormalization::MinMax => {
                let min_val = processed.iter().copied().fold(F::infinity(), F::min);
                let max_val = processed.iter().copied().fold(F::neg_infinity(), F::max);
                let range = max_val - min_val;
                if range > F::zero() {
                    processed.mapv(|x| (x - min_val) / range)
                } else {
                    processed.mapv(|_| F::zero())
                }
            }
            ActivationNormalization::ZScore => {
                let mean = processed.mean_or(F::zero());
                let variance = processed
                    .iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .fold(F::zero(), |acc, x| acc + x)
                    / F::from(processed.len()).unwrap_or(F::one());
                let std = variance.sqrt();
                if std > F::zero() {
                    processed.mapv(|x| (x - mean) / std)
                } else {
                    processed.mapv(|_| F::zero())
                }
            }
            ActivationNormalization::Percentile(low, high) => {
                let mut values: Vec<F> = processed.iter().copied().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = values.len();
                let low_idx = ((low / 100.0) * n as f64) as usize;
                let high_idx = ((high / 100.0) * n as f64) as usize;
                if low_idx < n && high_idx < n && low_idx < high_idx {
                    let low_val = values[low_idx];
                    let high_val = values[high_idx];
                    let range = high_val - low_val;
                    if range > F::zero() {
                        processed.mapv(|x| ((x - low_val) / range).max(F::zero()).min(F::one()))
                    } else {
                        processed.mapv(|_| F::zero())
                    }
                } else {
                    processed
                }
            }
            ActivationNormalization::Custom(_) => {
                // Custom normalization would require function pointer or callback
                // For now, fall back to no normalization
                processed
            }
        };
        Ok(processed)
    }
    fn create_feature_map_svg(
        &self,
        feature_maps: &ArrayD<F>,
        layer_name: &str,
        colormap: &Colormap,
    ) -> Result<String> {
        let width = self.config.style.layout.width;
        let height = self.config.style.layout.height;
        // Get color scheme
        let colors = self.get_colormap_colors(colormap);
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
"#,
            width, height, width, height
        );
        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"{}\" font-size=\"{}\" fill=\"#333\">{} Feature Maps</text>\n",
            width / 2, self.config.style.font.family,
            (self.config.style.font.size as f32 * self.config.style.font.title_scale) as u32,
            layer_name
        ));
        // Simple grid visualization of feature maps
        if feature_maps.ndim() >= 2 {
            let shape = feature_maps.shape();
            let map_height = shape[0].min(32); // Limit visualization size
            let map_width = shape[1].min(32);
            let cell_width = (width - 100) / map_width as u32;
            let cell_height = (height - 100) / map_height as u32;
            for i in 0..map_height {
                for j in 0..map_width {
                    let value = if let Some(&val) = feature_maps.get([i, j].as_slice()) {
                        val
                    } else {
                        F::zero()
                    };
                    let intensity = (value.to_f64().unwrap_or(0.0) * 255.0).clamp(0.0, 255.0) as u8;
                    let color = if colors.len() > 1 {
                        // Interpolate between colors
                        let color_idx =
                            (intensity as f64 / 255.0 * (colors.len() - 1) as f64) as usize;
                        colors[color_idx.min(colors.len() - 1)].clone()
                    } else {
                        format!("rgb({},{},{})", intensity, intensity, intensity)
                    };
                    svg.push_str(&format!(
                        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"#ccc\" stroke-width=\"0.5\"/>\n",
                        50 + j * cell_width as usize,
                        50 + i * cell_height as usize,
                        cell_width,
                        cell_height,
                        color
                    ));
                }
            }
        }
        svg.push_str("</svg>");
        Ok(svg)
    }
    fn compute_activation_histogram(
        &self,
        layer_name: &str,
        activations: &ArrayD<F>,
        num_bins: usize,
    ) -> Result<ActivationHistogram<F>> {
        let values: Vec<F> = activations.iter().copied().collect();
        if values.is_empty() {
            return Err(NeuralError::ValidationError(
                "Empty activations for histogram".to_string(),
            ));
        }
        let min_val = values.iter().copied().fold(F::infinity(), F::min);
        let max_val = values.iter().copied().fold(F::neg_infinity(), F::max);
        let range = max_val - min_val;
        if range <= F::zero() {
            // All values are the same
            return Ok(ActivationHistogram {
                layer_name: layer_name.to_string(),
                bins: vec![min_val],
                counts: vec![values.len()],
                edges: vec![min_val, max_val],
                total_samples: values.len(),
            });
        }
        let bin_width = range / F::from(num_bins).unwrap_or(F::one());
        let mut bins = Vec::with_capacity(num_bins);
        let mut counts = vec![0; num_bins];
        let mut edges = Vec::with_capacity(num_bins + 1);
        // Create bin edges and centers
        for i in 0..=num_bins {
            edges.push(min_val + F::from(i).unwrap_or(F::zero()) * bin_width);
        }
        for i in 0..num_bins {
            bins.push(
                min_val
                    + (F::from(i).unwrap_or(F::zero()) + F::from(0.5).unwrap_or(F::zero()))
                        * bin_width,
            );
        }
        // Count values in each bin
        for &value in &values {
            let bin_idx = ((value - min_val) / bin_width)
                .to_usize()
                .unwrap_or(0)
                .min(num_bins - 1);
            counts[bin_idx] += 1;
        }
        Ok(ActivationHistogram {
            layer_name: layer_name.to_string(),
            bins,
            counts,
            edges,
            total_samples: values.len(),
        })
    }

    fn create_histogram_svg(&self, histogram: &ActivationHistogram<F>) -> Result<String> {
        let width = self.config.style.layout.width;
        let height = self.config.style.layout.height;
        let mut svg = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
            <svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            width, height
        );
        svg.push_str("</svg>");
        Ok(svg)
    }

    fn create_statistics_svg(&self, stats: &[ActivationStatistics<F>]) -> Result<String> {
        let width = self.config.style.layout.width;
        let height = self.config.style.layout.height;
        let mut svg = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
            <svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            width, height
        );
        svg.push_str("</svg>");
        Ok(svg)
    }

    fn compute_spatial_attention(&self, activations: &ArrayD<F>) -> Result<ArrayD<F>> {
        // Stub implementation
        Ok(activations.clone())
    }

    fn create_attention_map_svg(
        &self,
        attention_map: &ArrayD<F>,
        layer_name: &str,
    ) -> Result<String> {
        let width = self.config.style.layout.width;
        let height = self.config.style.layout.height;
        let mut svg = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
            <svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            width, height
        );
        svg.push_str("</svg>");
        Ok(svg)
    }

    fn compute_activation_flow(
        &self,
        from_activations: &ArrayD<F>,
        to_activations: &ArrayD<F>,
    ) -> Result<f64> {
        Ok(0.0)
    }

    fn create_flow_diagram_svg(&self, flow_data: &[(String, String, f64)]) -> Result<String> {
        let width = self.config.style.layout.width;
        let height = self.config.style.layout.height;
        let mut svg = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
            <svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            width, height
        );
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"{}\" font-size=\"{}\" fill=\"#333\">Activation Flow Diagram</text>\n",
            width / 2, self.config.style.font.family, self.config.style.font.size
        ));
        svg.push_str("</svg>");
        Ok(svg)
    }

    fn get_colormap_colors(&self, colormap: &Colormap) -> Vec<String> {
        match colormap {
            Colormap::Viridis => vec![
                "#440154".to_string(),
                "#482677".to_string(),
                "#3f4a8a".to_string(),
                "#31678e".to_string(),
                "#26838f".to_string(),
                "#1f9d8a".to_string(),
                "#6cce5a".to_string(),
                "#b6de2b".to_string(),
                "#fee825".to_string(),
            ],
            Colormap::Plasma => vec![
                "#0c0786".to_string(),
                "#40039a".to_string(),
                "#6a0a83".to_string(),
                "#8b0aa5".to_string(),
                "#a83eaf".to_string(),
                "#c06fad".to_string(),
                "#d8a1a3".to_string(),
                "#f0d3a3".to_string(),
                "#fcffa4".to_string(),
            ],
            Colormap::Inferno => vec![
                "#000003".to_string(),
                "#1f0c48".to_string(),
                "#581845".to_string(),
                "#8b1538".to_string(),
                "#b71f2b".to_string(),
                "#db4c26".to_string(),
                "#ed7953".to_string(),
                "#fbad76".to_string(),
            ],
            Colormap::Jet => vec![
                "#00007f".to_string(),
                "#0000ff".to_string(),
                "#007fff".to_string(),
                "#00ffff".to_string(),
                "#7fff00".to_string(),
                "#ffff00".to_string(),
                "#ff7f00".to_string(),
                "#ff0000".to_string(),
                "#7f0000".to_string(),
            ],
            Colormap::Gray => vec![
                "#000000".to_string(),
                "#404040".to_string(),
                "#808080".to_string(),
                "#c0c0c0".to_string(),
                "#ffffff".to_string(),
            ],
            Colormap::RdBu => vec![
                "#053061".to_string(),
                "#2166ac".to_string(),
                "#4393c3".to_string(),
                "#92c5de".to_string(),
                "#d1e5f0".to_string(),
                "#f7f7f7".to_string(),
                "#fddbc7".to_string(),
                "#f4a582".to_string(),
                "#d6604d".to_string(),
                "#b2182b".to_string(),
                "#67001f".to_string(),
            ],
            Colormap::Custom(colors) => colors.clone(),
        }
    }
}

// Default implementations for configuration types
impl Default for ActivationVisualizationOptions {
    fn default() -> Self {
        Self {
            target_layers: Vec::new(),
            visualization_type: ActivationVisualizationType::FeatureMaps,
            normalization: ActivationNormalization::MinMax,
            colormap: Colormap::Viridis,
            aggregation: ChannelAggregation::Mean,
        }
    }
}

impl Default for FeatureMapInfo {
    fn default() -> Self {
        Self {
            feature_index: 0,
            spatial_dims: (1, 1),
            channels: 1,
            activation_range: (0.0, 1.0),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use scirs2_core::random::SeedableRng;
    #[test]
    fn test_activation_visualizer_creation() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).expect("Operation failed"));
        let config = VisualizationConfig::default();
        let visualizer = ActivationVisualizer::new(model, config);
        assert!(visualizer.activation_cache.is_empty());
    }

    #[test]
    fn test_activation_visualization_options_default() {
        let options = ActivationVisualizationOptions::default();
        assert_eq!(
            options.visualization_type,
            ActivationVisualizationType::FeatureMaps
        );
        assert_eq!(options.normalization, ActivationNormalization::MinMax);
        assert_eq!(options.colormap, Colormap::Viridis);
        assert_eq!(options.aggregation, ChannelAggregation::Mean);
    }

    #[test]
    fn test_activation_visualization_types() {
        let types = [
            ActivationVisualizationType::FeatureMaps,
            ActivationVisualizationType::Histograms,
            ActivationVisualizationType::Statistics,
            ActivationVisualizationType::AttentionMaps,
            ActivationVisualizationType::ActivationFlow,
        ];
        assert_eq!(types.len(), 5);
        assert_eq!(types[0], ActivationVisualizationType::FeatureMaps);
    }

    #[test]
    fn test_normalization_methods() {
        let none = ActivationNormalization::None;
        let minmax = ActivationNormalization::MinMax;
        let zscore = ActivationNormalization::ZScore;
        let percentile = ActivationNormalization::Percentile(5.0, 95.0);
        assert_eq!(none, ActivationNormalization::None);
        assert_eq!(minmax, ActivationNormalization::MinMax);
        assert_eq!(zscore, ActivationNormalization::ZScore);
        match percentile {
            ActivationNormalization::Percentile(low, high) => {
                assert_eq!(low, 5.0);
                assert_eq!(high, 95.0);
            }
            _ => unreachable!("Expected percentile normalization"),
        }
    }

    #[test]
    fn test_colormaps() {
        let colormaps = [
            Colormap::Viridis,
            Colormap::Plasma,
            Colormap::Inferno,
            Colormap::Jet,
            Colormap::Gray,
            Colormap::RdBu,
        ];
        assert_eq!(colormaps.len(), 6);
        assert_eq!(colormaps[0], Colormap::Viridis);
        let custom = Colormap::Custom(vec!["#ff0000".to_string(), "#00ff00".to_string()]);
        match custom {
            Colormap::Custom(colors) => assert_eq!(colors.len(), 2),
            _ => unreachable!("Expected custom colormap"),
        }
    }

    #[test]
    fn test_channel_aggregation() {
        let aggregations = [
            ChannelAggregation::None,
            ChannelAggregation::Mean,
            ChannelAggregation::Max,
            ChannelAggregation::Min,
            ChannelAggregation::Std,
            ChannelAggregation::Select(vec![0, 1, 2]),
        ];
        assert_eq!(aggregations.len(), 6);
        assert_eq!(aggregations[1], ChannelAggregation::Mean);
        match &aggregations[5] {
            ChannelAggregation::Select(channels) => assert_eq!(channels.len(), 3),
            _ => unreachable!("Expected select aggregation"),
        }
    }

    #[test]
    fn test_feature_map_info_default() {
        let info = FeatureMapInfo::default();
        assert_eq!(info.feature_index, 0);
        assert_eq!(info.spatial_dims, (1, 1));
        assert_eq!(info.channels, 1);
        assert_eq!(info.activation_range, (0.0, 1.0));
    }

    #[test]
    fn test_cache_operations() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).expect("Operation failed"));
        let config = VisualizationConfig::default();
        let mut visualizer = ActivationVisualizer::new(model, config);
        assert!(visualizer.get_cached_activations("test_layer").is_none());
        visualizer.clear_cache();
    }
}
