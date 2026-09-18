//! Common types and enums for advanced FFT coordination

use scirs2_core::numeric::Complex;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::time::Instant;

/// FFT algorithm types
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum FftAlgorithmType {
    /// Cooley-Tukey radix-2
    CooleyTukeyRadix2,
    /// Cooley-Tukey mixed radix
    CooleyTukeyMixedRadix,
    /// Prime factor algorithm
    PrimeFactorAlgorithm,
    /// Chirp Z-transform
    ChirpZTransform,
    /// Bluestein's algorithm
    BluesteinAlgorithm,
    /// Split-radix algorithm
    SplitRadixAlgorithm,
    /// GPU-accelerated FFT
    GpuAcceleratedFft,
    /// SIMD-optimized FFT
    SimdOptimizedFft,
    /// Quantum-inspired FFT
    QuantumInspiredFft,
}

/// Signal size classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum SizeClass {
    /// Small signals (< 1K points)
    Small,
    /// Medium signals (1K - 1M points)
    Medium,
    /// Large signals (1M - 1B points)
    Large,
    /// Massive signals (> 1B points)
    Massive,
}

/// Signal type classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum SignalType {
    /// Real-valued signals
    Real,
    /// Complex-valued signals
    Complex,
    /// Sparse signals
    Sparse,
    /// Structured signals (e.g., images)
    Structured,
    /// Random/noise signals
    Random,
    /// Periodic signals
    Periodic,
}

/// Hardware profile for optimization
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum HardwareProfile {
    /// CPU-only processing
    CpuOnly,
    /// GPU-accelerated processing
    GpuAccelerated,
    /// Mixed CPU/GPU processing
    Hybrid,
    /// Distributed processing
    Distributed,
    /// Edge device processing
    Edge,
}

/// Signal profile for analysis
#[derive(Debug, Clone)]
pub struct SignalProfile<F: Float> {
    /// Signal length
    pub length: usize,
    /// Signal dimensionality
    pub dimensions: Vec<usize>,
    /// Signal type
    pub signal_type: SignalType,
    /// Sparsity ratio (0.0 - 1.0)
    pub sparsity: F,
    /// Signal entropy
    pub entropy: F,
    /// Dominant frequencies
    pub dominant_frequencies: Vec<F>,
    /// Signal-to-noise ratio
    pub snr: Option<F>,
    /// Periodicity score
    pub periodicity: F,
    /// Spectral flatness
    pub spectral_flatness: F,
}

/// Window type enumeration
pub enum WindowType {
    /// Hamming window
    Hamming,
    /// Hanning window
    Hanning,
    /// Blackman window
    Blackman,
}

impl ToString for WindowType {
    fn to_string(&self) -> String {
        match self {
            WindowType::Hamming => "hamming".to_string(),
            WindowType::Hanning => "hanning".to_string(),
            WindowType::Blackman => "blackman".to_string(),
        }
    }
}

/// Pattern type classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum PatternType {
    /// Sinusoidal patterns
    Sinusoidal,
    /// Chirp patterns
    Chirp,
    /// Impulse patterns
    Impulse,
    /// Step function patterns
    Step,
    /// Random/noise patterns
    Random,
    /// Fractal patterns
    Fractal,
    /// Periodic patterns
    Periodic,
    /// Chaotic patterns
    Chaotic,
}

/// Frequency profile classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum FrequencyProfile {
    /// Low frequency dominated
    LowFrequency,
    /// High frequency dominated
    HighFrequency,
    /// Broadband
    Broadband,
    /// Narrowband
    Narrowband,
    /// Multi-peak
    MultiPeak,
}

/// Preprocessing step recommendation
#[derive(Debug, Clone)]
pub enum PreprocessingStep {
    /// Apply windowing function
    Windowing { window_type: String },
    /// Zero-padding
    ZeroPadding { target_size: usize },
    /// Signal denoising
    Denoising { method: String },
    /// Frequency domain filtering
    Filtering { filter_spec: String },
}

/// SIMD support level
#[derive(Debug, Clone)]
pub enum SimdSupport {
    /// No SIMD support
    None,
    /// SSE support
    SSE,
    /// AVX support
    AVX,
    /// AVX2 support
    AVX2,
    /// AVX-512 support
    AVX512,
    /// ARM NEON support
    NEON,
}

/// Thread affinity settings
#[derive(Debug, Clone)]
pub enum ThreadAffinity {
    /// No specific affinity
    None,
    /// Pin to specific cores
    PinToCores { cores: Vec<usize> },
    /// NUMA-aware affinity
    NumaAware,
}

/// Memory allocation strategy
#[derive(Debug, Clone)]
pub enum MemoryAllocationStrategy {
    /// Conservative allocation
    Conservative,
    /// Aggressive pre-allocation
    Aggressive,
    /// Adaptive based on usage patterns
    Adaptive,
    /// Custom allocation strategy
    Custom { strategy: String },
}

/// Optimization strategy
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Minimize execution time
    MinimizeTime,
    /// Minimize memory usage
    MinimizeMemory,
    /// Maximize accuracy
    MaximizeAccuracy,
    /// Balance time and memory
    Balanced,
    /// Custom weighted optimization
    Custom { weights: OptimizationWeights },
}

/// Optimization weights for custom strategy
#[derive(Debug, Clone)]
pub struct OptimizationWeights {
    /// Weight for execution time (0.0 - 1.0)
    pub time_weight: f64,
    /// Weight for memory usage (0.0 - 1.0)
    pub memory_weight: f64,
    /// Weight for accuracy (0.0 - 1.0)
    pub accuracy_weight: f64,
    /// Weight for energy efficiency (0.0 - 1.0)
    pub energy_weight: f64,
}

/// Performance trends
#[derive(Debug, Default, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend (positive = getting slower)
    pub execution_time_trend: f64,
    /// Memory usage trend (positive = using more memory)
    pub memory_usage_trend: f64,
    /// Accuracy trend (positive = getting more accurate)
    pub accuracy_trend: f64,
    /// Overall performance score
    pub overall_performance_score: f64,
}

/// Algorithm usage statistics
#[derive(Debug, Clone, Default)]
pub struct AlgorithmStats {
    /// Usage count
    pub usage_count: usize,
    /// Average execution time
    pub avg_execution_time: f64,
    /// Average memory usage
    pub avg_memory_usage: usize,
    /// Success rate
    pub success_rate: f64,
}

/// Parallelism configuration
#[derive(Debug, Clone)]
pub struct ParallelismConfig {
    /// Number of threads to use
    pub thread_count: usize,
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinity,
    /// Work stealing enabled
    pub work_stealing: bool,
}

/// SIMD configuration
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// SIMD instruction set to use
    pub instruction_set: SimdSupport,
    /// Vector size preference
    pub vector_size: usize,
    /// Enable unaligned access
    pub unaligned_access: bool,
}

/// Optimization settings
#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    /// Preprocessing steps
    pub preprocessing_steps: Vec<PreprocessingStep>,
    /// Algorithm parameters
    pub algorithm_parameters: HashMap<String, f64>,
    /// Parallelism settings
    pub parallelism_settings: ParallelismConfig,
    /// SIMD settings
    pub simd_settings: SimdConfig,
}

/// Memory strategy recommendation
#[derive(Debug, Clone)]
pub struct MemoryStrategy {
    /// Memory allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
    /// Enable caching
    pub cache_enabled: bool,
    /// Enable prefetching
    pub prefetch_enabled: bool,
    /// Memory pool size (bytes)
    pub memory_pool_size: usize,
}

/// Expected performance characteristics
#[derive(Debug, Clone)]
pub struct ExpectedPerformance {
    /// Expected execution time (microseconds)
    pub execution_time: f64,
    /// Expected memory usage (bytes)
    pub memory_usage: usize,
    /// Expected accuracy
    pub accuracy: f64,
    /// Expected energy consumption
    pub energy_consumption: Option<f64>,
}

/// Algorithm recommendation result
#[derive(Debug, Clone)]
pub struct AlgorithmRecommendation {
    /// Recommended algorithm
    pub algorithm: FftAlgorithmType,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Expected performance
    pub expected_performance: ExpectedPerformance,
}

/// FFT recommendation result
#[derive(Debug, Clone)]
pub struct FftRecommendation {
    /// Recommended algorithm
    pub recommended_algorithm: FftAlgorithmType,
    /// Optimization settings
    pub optimization_settings: OptimizationSettings,
    /// Memory strategy
    pub memory_strategy: MemoryStrategy,
    /// Confidence score (0.0 - 1.0)
    pub confidence_score: f64,
    /// Expected performance characteristics
    pub expected_performance: ExpectedPerformance,
}

/// FFT performance metrics
#[derive(Debug, Clone)]
pub struct FftPerformanceMetrics {
    /// Average execution time (microseconds)
    pub average_execution_time: f64,
    /// Memory efficiency score (0.0 - 1.0)
    pub memory_efficiency: f64,
    /// Algorithm usage distribution
    pub algorithm_distribution: HashMap<FftAlgorithmType, AlgorithmStats>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Quantum gate for optimization
#[derive(Debug, Clone)]
pub enum QuantumGate<F: Float> {
    /// Hadamard gate
    Hadamard { qubit: usize },
    /// Pauli-X gate
    PauliX { qubit: usize },
    /// Pauli-Y gate
    PauliY { qubit: usize },
    /// Pauli-Z gate
    PauliZ { qubit: usize },
    /// CNOT gate
    CNOT { control: usize, target: usize },
    /// Rotation gate
    Rotation { qubit: usize, angle: F },
    /// Custom gate
    Custom { matrix: scirs2_core::ndarray::Array2<Complex<F>> },
}

/// Annealing schedule
#[derive(Debug, Clone)]
pub enum AnnealingSchedule<F: Float> {
    /// Linear schedule
    Linear,
    /// Exponential schedule
    Exponential { decay_rate: F },
    /// Custom schedule
    Custom { schedule: Vec<F> },
}
