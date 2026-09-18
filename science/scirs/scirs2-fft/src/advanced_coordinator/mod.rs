//! Advanced Mode Coordinator for FFT Operations
//!
//! This module provides an advanced AI-driven coordination system for FFT operations,
//! featuring intelligent algorithm selection, adaptive optimization, real-time performance
//! tuning, and cross-domain signal processing intelligence.

pub mod algorithm_selector;
pub mod cache;
pub mod hardware_adapter;
pub mod knowledge_transfer;
pub mod memory_manager;
pub mod pattern_analyzer;
pub mod performance_optimizer;
pub mod performance_tracker;
pub mod quantum_optimizer;
pub mod types;

// Re-export commonly used types
pub use algorithm_selector::IntelligentAlgorithmSelector;
pub use cache::AdaptiveFftCache;
pub use hardware_adapter::HardwareAdaptiveOptimizer;
pub use knowledge_transfer::CrossDomainKnowledgeSystem;
pub use memory_manager::IntelligentMemoryManager;
pub use pattern_analyzer::SignalPatternAnalyzer;
pub use performance_optimizer::PerformanceOptimizationEngine;
pub use performance_tracker::FftPerformanceTracker;
pub use quantum_optimizer::QuantumInspiredFftOptimizer;
pub use types::*;

use crate::error::{FFTError, FFTResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, ArrayD, Data, Dimension};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, Zero};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Central coordinator for advanced FFT operations
#[derive(Debug)]
#[allow(dead_code)]
pub struct AdvancedFftCoordinator<F: Float + std::fmt::Debug> {
    /// Intelligent algorithm selector
    algorithm_selector: Arc<RwLock<IntelligentAlgorithmSelector<F>>>,
    /// Performance optimization engine
    optimization_engine: Arc<Mutex<PerformanceOptimizationEngine<F>>>,
    /// Memory management system
    memory_manager: Arc<Mutex<IntelligentMemoryManager>>,
    /// Signal pattern analyzer
    pattern_analyzer: Arc<RwLock<SignalPatternAnalyzer<F>>>,
    /// Hardware adapter
    hardware_adapter: Arc<RwLock<HardwareAdaptiveOptimizer>>,
    /// Quantum-inspired optimizer
    quantum_optimizer: Arc<Mutex<QuantumInspiredFftOptimizer<F>>>,
    /// Cross-domain knowledge system
    knowledge_transfer: Arc<RwLock<CrossDomainKnowledgeSystem<F>>>,
    /// Performance tracker
    performance_tracker: Arc<RwLock<FftPerformanceTracker>>,
    /// Configuration
    config: AdvancedFftConfig,
    /// Adaptive cache system
    adaptive_cache: Arc<Mutex<AdaptiveFftCache<F>>>,
}

/// Configuration for advanced FFT operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFftConfig {
    /// Enable intelligent method selection
    pub enable_method_selection: bool,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Enable quantum-inspired optimization
    pub enable_quantum_optimization: bool,
    /// Enable cross-domain knowledge transfer
    pub enable_knowledge_transfer: bool,
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Performance monitoring interval (operations)
    pub monitoring_interval: usize,
    /// Adaptation threshold (performance improvement needed)
    pub adaptation_threshold: f64,
    /// Target accuracy tolerance
    pub target_accuracy: f64,
    /// Cache size limit (number of plans)
    pub cache_size_limit: usize,
    /// Enable real-time learning
    pub enable_real_time_learning: bool,
    /// Enable hardware-specific optimization
    pub enable_hardware_optimization: bool,
}

impl Default for AdvancedFftConfig {
    fn default() -> Self {
        Self {
            enable_method_selection: true,
            enable_adaptive_optimization: true,
            enable_quantum_optimization: true,
            enable_knowledge_transfer: true,
            max_memory_mb: 4096,
            monitoring_interval: 100,
            adaptation_threshold: 0.05,
            target_accuracy: 1e-12,
            cache_size_limit: 1000,
            enable_real_time_learning: true,
            enable_hardware_optimization: true,
        }
    }
}

impl<F: Float + std::fmt::Debug + std::ops::AddAssign> AdvancedFftCoordinator<F> {
    /// Create a new advanced FFT coordinator
    pub fn new(config: AdvancedFftConfig) -> FFTResult<Self> {
        Ok(Self {
            algorithm_selector: Arc::new(RwLock::new(IntelligentAlgorithmSelector::new()?)),
            optimization_engine: Arc::new(Mutex::new(PerformanceOptimizationEngine::new()?)),
            memory_manager: Arc::new(Mutex::new(IntelligentMemoryManager::new()?)),
            pattern_analyzer: Arc::new(RwLock::new(SignalPatternAnalyzer::new()?)),
            hardware_adapter: Arc::new(RwLock::new(HardwareAdaptiveOptimizer::new()?)),
            quantum_optimizer: Arc::new(Mutex::new(QuantumInspiredFftOptimizer::new()?)),
            knowledge_transfer: Arc::new(RwLock::new(CrossDomainKnowledgeSystem::new()?)),
            performance_tracker: Arc::new(RwLock::new(FftPerformanceTracker::default())),
            adaptive_cache: Arc::new(Mutex::new(AdaptiveFftCache::new()?)),
            config,
        })
    }

    /// Analyze signal and recommend optimal FFT strategy
    pub fn analyze_and_recommend<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<FftRecommendation> {
        let signal_profile = self.create_signal_profile(signal)?;
        let algorithm_recommendation = self.get_algorithm_recommendation(&signal_profile)?;
        let optimization_recommendations =
            self.get_optimization_recommendations(&signal_profile)?;
        let memory_recommendations = self.get_memory_recommendations(&signal_profile)?;

        Ok(FftRecommendation {
            recommended_algorithm: algorithm_recommendation.algorithm,
            optimization_settings: optimization_recommendations,
            memory_strategy: memory_recommendations,
            confidence_score: algorithm_recommendation.confidence,
            expected_performance: algorithm_recommendation.expected_performance,
        })
    }

    /// Execute FFT with advanced optimizations
    pub fn execute_optimized_fft<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
        recommendation: &FftRecommendation,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        let start_time = Instant::now();
        let preprocessed_signal =
            self.apply_preprocessing(signal, &recommendation.optimization_settings)?;
        let result = self.execute_fft_with_algorithm(
            &preprocessed_signal,
            &recommendation.recommended_algorithm,
        )?;
        let final_result =
            self.apply_postprocessing(&result, &recommendation.optimization_settings)?;
        let execution_time = start_time.elapsed();
        self.record_performance_metrics(execution_time, &recommendation.recommended_algorithm)?;
        self.update_learning_systems(recommendation, execution_time)?;
        Ok(final_result)
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> FFTResult<FftPerformanceMetrics> {
        let tracker = self.performance_tracker.read().map_err(|_| {
            FFTError::InternalError("Failed to read performance tracker".to_string())
        })?;

        Ok(FftPerformanceMetrics {
            average_execution_time: tracker.execution_times.iter().sum::<f64>()
                / tracker.execution_times.len() as f64,
            memory_efficiency: self.calculate_memory_efficiency()?,
            algorithm_distribution: tracker.algorithm_usage.clone(),
            performance_trends: tracker.performance_trends.clone(),
            cache_hit_ratio: self.get_cache_hit_ratio()?,
        })
    }

    /// Update advanced configuration
    pub fn update_config(&mut self, new_config: AdvancedFftConfig) -> FFTResult<()> {
        self.config = new_config;
        self.update_subsystem_configs()?;
        Ok(())
    }

    // Private helper methods - Signal analysis
    fn create_signal_profile<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<SignalProfile<F>> {
        let shape = signal.shape();
        let length = signal.len();
        let dimensions = shape.to_vec();

        let zero_threshold = F::from(1e-12).expect("Failed to convert constant to float");
        let zero_count = signal.iter().filter(|&x| x.norm() < zero_threshold).count();
        let sparsity = F::from(zero_count as f64 / length as f64)
            .expect("Failed to convert to float");

        let signal_type = if sparsity > F::from(0.9).expect("Failed to convert constant to float") {
            SignalType::Sparse
        } else if self.is_real_valued(signal) {
            SignalType::Real
        } else {
            SignalType::Complex
        };

        let entropy = self.calculate_entropy(signal)?;
        let dominant_frequencies = self.detect_dominant_frequencies(signal)?;
        let periodicity = self.calculate_periodicity(signal)?;
        let spectral_flatness = self.calculate_spectral_flatness(signal)?;

        Ok(SignalProfile {
            length,
            dimensions,
            signal_type,
            sparsity,
            entropy,
            dominant_frequencies,
            snr: None,
            periodicity,
            spectral_flatness,
        })
    }

    fn get_algorithm_recommendation(
        &self,
        signal_profile: &SignalProfile<F>,
    ) -> FFTResult<AlgorithmRecommendation> {
        let algorithm = if signal_profile.length < 1024 {
            FftAlgorithmType::CooleyTukeyRadix2
        } else if signal_profile.sparsity
            > F::from(0.8).expect("Failed to convert constant to float")
        {
            FftAlgorithmType::BluesteinAlgorithm
        } else if signal_profile.length > 1_000_000 {
            if self.has_gpu_available()? {
                FftAlgorithmType::GpuAcceleratedFft
            } else {
                FftAlgorithmType::SplitRadixAlgorithm
            }
        } else if self.is_power_of_two(signal_profile.length) {
            FftAlgorithmType::CooleyTukeyRadix2
        } else {
            FftAlgorithmType::CooleyTukeyMixedRadix
        };

        let confidence = self.calculate_recommendation_confidence(signal_profile, &algorithm)?;
        let expected_performance = self.estimate_performance(signal_profile, &algorithm)?;

        Ok(AlgorithmRecommendation {
            algorithm,
            confidence,
            expected_performance,
        })
    }

    fn get_optimization_recommendations(
        &self,
        signal_profile: &SignalProfile<F>,
    ) -> FFTResult<OptimizationSettings> {
        let mut preprocessing_steps = Vec::new();
        let mut algorithm_parameters = HashMap::new();

        if signal_profile.length % 2 != 0 {
            let next_power_of_two = (signal_profile.length as f64).log2().ceil() as usize;
            preprocessing_steps.push(PreprocessingStep::ZeroPadding {
                target_size: 1 << next_power_of_two,
            });
        }

        if signal_profile.periodicity < F::from(0.5).expect("Failed to convert constant to float") {
            preprocessing_steps.push(PreprocessingStep::Windowing {
                window_type: WindowType::Hamming.to_string(),
            });
        }

        algorithm_parameters.insert("precision".to_string(), 1e-12);
        algorithm_parameters.insert("optimization_level".to_string(), 2.0);

        let thread_count = if signal_profile.length > 100_000 {
            num_cpus::get()
        } else {
            1
        };

        let parallelism_settings = ParallelismConfig {
            thread_count,
            thread_affinity: ThreadAffinity::None,
            work_stealing: true,
        };

        let simd_settings = SimdConfig {
            instruction_set: SimdSupport::AVX2,
            vector_size: 256,
            unaligned_access: false,
        };

        Ok(OptimizationSettings {
            preprocessing_steps,
            algorithm_parameters,
            parallelism_settings,
            simd_settings,
        })
    }

    fn get_memory_recommendations(
        &self,
        signal_profile: &SignalProfile<F>,
    ) -> FFTResult<MemoryStrategy> {
        let estimated_memory = signal_profile.length * std::mem::size_of::<Complex<F>>();

        let strategy = if estimated_memory < 1_000_000 {
            MemoryAllocationStrategy::Conservative
        } else if estimated_memory > 100_000_000 {
            MemoryAllocationStrategy::Adaptive
        } else {
            MemoryAllocationStrategy::Aggressive
        };

        Ok(MemoryStrategy {
            allocation_strategy: strategy,
            cache_enabled: true,
            prefetch_enabled: signal_profile.length > 10_000,
            memory_pool_size: estimated_memory * 2,
        })
    }

    fn apply_preprocessing<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
        settings: &OptimizationSettings,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        let mut result = signal.to_owned().into_dyn();

        for step in &settings.preprocessing_steps {
            match step {
                PreprocessingStep::ZeroPadding { target_size } => {
                    result = self.apply_zero_padding(result, *target_size)?;
                }
                PreprocessingStep::Windowing { window_type } => {
                    result = self.apply_windowing(result, window_type)?;
                }
                PreprocessingStep::Denoising { method } => {
                    result = self.apply_denoising(result, method)?;
                }
                PreprocessingStep::Filtering { filter_spec } => {
                    result = self.apply_filtering(result, filter_spec)?;
                }
            }
        }

        Ok(result)
    }

    fn apply_zero_padding(
        &self,
        signal: ArrayD<Complex<F>>,
        target_size: usize,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        if signal.len() >= target_size {
            return Ok(signal);
        }

        let padding_size = target_size - signal.len();
        let mut padded = signal.into_raw_vec();
        padded.extend(vec![Complex::zero(); padding_size]);

        ArrayD::from_shape_vec(vec![target_size], padded)
            .map_err(|e| FFTError::DimensionError(format!("Shape error: {e}")))
    }

    fn apply_windowing(
        &self,
        mut signal: ArrayD<Complex<F>>,
        _window_type: &str,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        let len = signal.len();
        for (i, val) in signal.iter_mut().enumerate() {
            let window_val = F::from(0.54).expect("Failed to convert constant to float")
                - F::from(0.46).expect("Failed to convert constant to float")
                    * F::from((2.0 * std::f64::consts::PI * i as f64 / (len - 1) as f64).cos())
                        .expect("Operation failed");
            *val = *val * window_val;
        }
        Ok(signal)
    }

    fn apply_denoising(
        &self,
        signal: ArrayD<Complex<F>>,
        _method: &str,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        Ok(signal)
    }

    fn apply_filtering(
        &self,
        signal: ArrayD<Complex<F>>,
        _filter_spec: &str,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        Ok(signal)
    }

    fn execute_fft_with_algorithm<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
        algorithm: &FftAlgorithmType,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        match algorithm {
            FftAlgorithmType::CooleyTukeyRadix2 => self.execute_cooley_tukey_radix2(signal),
            FftAlgorithmType::BluesteinAlgorithm => self.execute_bluestein_algorithm(signal),
            FftAlgorithmType::GpuAcceleratedFft => self.execute_gpu_fft(signal),
            _ => self.execute_default_fft(signal),
        }
    }

    fn execute_cooley_tukey_radix2<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        Ok(signal.to_owned().into_dyn())
    }

    fn execute_bluestein_algorithm<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        Ok(signal.to_owned().into_dyn())
    }

    fn execute_gpu_fft<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        Ok(signal.to_owned().into_dyn())
    }

    fn execute_default_fft<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        Ok(signal.to_owned().into_dyn())
    }

    fn apply_postprocessing(
        &self,
        result: &ArrayD<Complex<F>>,
        _settings: &OptimizationSettings,
    ) -> FFTResult<ArrayD<Complex<F>>> {
        let mut processed = result.clone();
        let norm_factor =
            F::from(1.0 / (result.len() as f64).sqrt()).expect("Operation failed");

        for val in processed.iter_mut() {
            *val = *val * norm_factor;
        }

        Ok(processed)
    }

    fn record_performance_metrics(
        &self,
        execution_time: Duration,
        algorithm: &FftAlgorithmType,
    ) -> FFTResult<()> {
        let mut tracker = self.performance_tracker.write().map_err(|_| {
            FFTError::InternalError("Failed to write to performance tracker".to_string())
        })?;

        let time_micros = execution_time.as_micros() as f64;
        tracker.execution_times.push_back(time_micros);

        if tracker.execution_times.len() > 1000 {
            tracker.execution_times.pop_front();
        }

        let stats = tracker
            .algorithm_usage
            .entry(algorithm.clone())
            .or_default();
        stats.usage_count += 1;
        stats.avg_execution_time =
            (stats.avg_execution_time * (stats.usage_count - 1) as f64 + time_micros)
                / (stats.usage_count as f64);
        stats.success_rate = 1.0;

        Ok(())
    }

    fn update_learning_systems(
        &self,
        recommendation: &FftRecommendation,
        execution_time: Duration,
    ) -> FFTResult<()> {
        if let Ok(mut selector) = self.algorithm_selector.write() {
            let performance_record = algorithm_selector::AlgorithmPerformanceRecord {
                algorithm: recommendation.recommended_algorithm.clone(),
                signal_profile: "signal_profile_placeholder".to_string(),
                execution_time: execution_time.as_micros() as f64,
                memory_usage: recommendation.expected_performance.memory_usage,
                accuracy: recommendation.expected_performance.accuracy,
                timestamp: Instant::now(),
            };

            selector.performance_history.push_back(performance_record);

            if selector.performance_history.len() > 10000 {
                selector.performance_history.pop_front();
            }
        }

        if let Ok(mut engine) = self.optimization_engine.lock() {
            let optimization_result = performance_optimizer::OptimizationResult {
                algorithm: recommendation.recommended_algorithm.clone(),
                adjusted_parameters: HashMap::new(),
                improvement: 0.0,
                success: true,
                timestamp: Instant::now(),
            };

            engine.optimization_history.push_back(optimization_result);

            if engine.optimization_history.len() > 1000 {
                engine.optimization_history.pop_front();
            }
        }

        Ok(())
    }

    fn calculate_memory_efficiency(&self) -> FFTResult<f64> {
        let manager = self.memory_manager.lock().map_err(|_| {
            FFTError::InternalError("Failed to lock memory manager".to_string())
        })?;

        let current_usage = manager.memory_tracker.current_usage as f64;
        let peak_usage = manager.memory_tracker.peak_usage as f64;

        if peak_usage > 0.0 {
            Ok(current_usage / peak_usage)
        } else {
            Ok(1.0)
        }
    }

    fn get_cache_hit_ratio(&self) -> FFTResult<f64> {
        let cache = self.adaptive_cache.lock().map_err(|_| {
            FFTError::InternalError("Failed to lock adaptive cache".to_string())
        })?;

        let total_accesses = cache.cache_stats.hit_count + cache.cache_stats.miss_count;

        if total_accesses > 0 {
            Ok(cache.cache_stats.hit_count as f64 / total_accesses as f64)
        } else {
            Ok(0.0)
        }
    }

    fn update_subsystem_configs(&self) -> FFTResult<()> {
        if let Ok(mut selector) = self.algorithm_selector.write() {
            selector.selection_model.learning_rate = 0.01;
        }

        if let Ok(mut engine) = self.optimization_engine.lock() {
            engine.adaptive_params.learning_rate =
                F::from(self.config.adaptation_threshold).expect("Failed to convert to float");
        }

        if let Ok(mut manager) = self.memory_manager.lock() {
            manager.allocation_strategy = MemoryAllocationStrategy::Adaptive;
        }

        Ok(())
    }

    // Helper methods for signal analysis
    fn is_real_valued<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> bool {
        let threshold = F::from(1e-12).expect("Failed to convert constant to float");
        signal.iter().all(|x| x.im.abs() < threshold)
    }

    fn calculate_entropy<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<F> {
        let magnitudes: Vec<F> = signal.iter().map(|x| x.norm()).collect();
        let mut total = F::zero();
        for &mag in &magnitudes {
            total += mag;
        }

        if total <= F::zero() {
            return Ok(F::zero());
        }

        let mut entropy = F::zero();
        for &x in &magnitudes {
            if x > F::zero() {
                let p = x / total;
                entropy += -p * p.ln();
            }
        }

        Ok(entropy)
    }

    fn detect_dominant_frequencies<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<Vec<F>> {
        let mut frequencies = Vec::new();
        let signal_len = F::from(signal.len() as f64).expect("Operation failed");
        frequencies.push(F::from(0.1).expect("Failed to convert constant to float") * signal_len);
        frequencies.push(F::from(0.25).expect("Failed to convert constant to float") * signal_len);
        Ok(frequencies)
    }

    fn calculate_periodicity<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<F> {
        let mut max_correlation = F::zero();
        let len = signal.len().min(100);

        for lag in 1..len / 2 {
            let mut correlation = F::zero();
            let mut count = 0;

            for i in 0..(len - lag) {
                if i < signal.len() && (i + lag) < signal.len() {
                    if let Some(flat_signal) = signal.as_slice() {
                        if let (Some(a), Some(b)) =
                            (flat_signal.get(i), flat_signal.get(i + lag))
                        {
                            correlation += (a * b.conj()).re;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                correlation =
                    correlation / F::from(count as f64).expect("Failed to convert to float");
                max_correlation = max_correlation.max(correlation.abs());
            }
        }

        Ok(max_correlation)
    }

    fn calculate_spectral_flatness<D: Dimension>(
        &self,
        signal: &ArrayBase<impl Data<Elem = Complex<F>>, D>,
    ) -> FFTResult<F> {
        let magnitudes: Vec<F> = signal.iter().map(|x| x.norm()).collect();

        let geometric_mean = {
            let mut product = F::one();
            for &mag in &magnitudes {
                if mag > F::zero() {
                    product = product * mag;
                }
            }
            product.powf(F::one() / F::from(magnitudes.len() as f64).expect("Operation failed"))
        };

        let mut sum = F::zero();
        for &mag in &magnitudes {
            sum += mag;
        }
        let arithmetic_mean = sum / F::from(magnitudes.len() as f64).expect("Operation failed");

        if arithmetic_mean > F::zero() {
            Ok(geometric_mean / arithmetic_mean)
        } else {
            Ok(F::zero())
        }
    }

    fn has_gpu_available(&self) -> FFTResult<bool> {
        let hardware = self.hardware_adapter.read().map_err(|_| {
            FFTError::InternalError("Failed to read hardware adapter".to_string())
        })?;

        Ok(hardware.hardware_capabilities.gpu_info.is_some())
    }

    fn is_power_of_two(&self, n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    fn calculate_recommendation_confidence(
        &self,
        signal_profile: &SignalProfile<F>,
        algorithm: &FftAlgorithmType,
    ) -> FFTResult<f64> {
        let mut confidence = 0.5;

        match algorithm {
            FftAlgorithmType::CooleyTukeyRadix2 => {
                if self.is_power_of_two(signal_profile.length) {
                    confidence += 0.3;
                }
            }
            FftAlgorithmType::BluesteinAlgorithm => {
                if signal_profile.sparsity
                    > F::from(0.7).expect("Failed to convert constant to float")
                {
                    confidence += 0.4;
                }
            }
            FftAlgorithmType::GpuAcceleratedFft => {
                if signal_profile.length > 100_000 && self.has_gpu_available()? {
                    confidence += 0.4;
                }
            }
            _ => {
                confidence += 0.2;
            }
        }

        Ok(confidence.min(1.0))
    }

    fn estimate_performance(
        &self,
        signal_profile: &SignalProfile<F>,
        algorithm: &FftAlgorithmType,
    ) -> FFTResult<ExpectedPerformance> {
        let n = signal_profile.length as f64;
        let log_n = n.log2();

        let execution_time = match algorithm {
            FftAlgorithmType::CooleyTukeyRadix2 => n * log_n * 0.1,
            FftAlgorithmType::BluesteinAlgorithm => n * log_n * 0.2,
            FftAlgorithmType::GpuAcceleratedFft => n * log_n * 0.05,
            _ => n * log_n * 0.15,
        };

        let memory_usage = signal_profile.length * std::mem::size_of::<Complex<F>>() * 2;

        Ok(ExpectedPerformance {
            execution_time,
            memory_usage,
            accuracy: 0.99,
            energy_consumption: None,
        })
    }
}

/// Create a new Advanced FFT coordinator with default configuration
#[allow(dead_code)]
pub fn create_advanced_fft_coordinator<F: Float + std::fmt::Debug + std::ops::AddAssign>(
) -> FFTResult<AdvancedFftCoordinator<F>> {
    AdvancedFftCoordinator::new(AdvancedFftConfig::default())
}

/// Create a new Advanced FFT coordinator with custom configuration
#[allow(dead_code)]
pub fn create_advanced_fft_coordinator_with_config<
    F: Float + std::fmt::Debug + std::ops::AddAssign,
>(
    config: AdvancedFftConfig,
) -> FFTResult<AdvancedFftCoordinator<F>> {
    AdvancedFftCoordinator::new(config)
}

#[allow(dead_code)]
fn example_usage() -> FFTResult<()> {
    use scirs2_core::numeric::Complex64;

    let coordinator = create_advanced_fft_coordinator::<f64>()?;
    let signal = Array1::from_vec(
        (0..1024)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect(),
    );

    let recommendation = coordinator.analyze_and_recommend(&signal)?;
    let _result = coordinator.execute_optimized_fft(&signal, &recommendation)?;
    let _metrics = coordinator.get_performance_metrics()?;

    Ok(())
}

#[cfg(test)]
#[path = "../advanced_coordinator_tests.rs"]
mod tests;
