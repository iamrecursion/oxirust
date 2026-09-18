//! Filter graph configuration bindings.
//!
//! Provides Python-accessible configuration types for building filter graphs.
//! These are lightweight configuration objects - use the Rust `oximedia-graph`
//! crate directly for the full filter graph runtime.

use pyo3::prelude::*;

// ─────────────────────────────── Scale ───────────────────────────────────────

/// Configuration for a scale (resize) filter.
///
/// # Example
///
/// ```python
/// config = ScaleConfig(width=1280, height=720, algorithm="bilinear")
/// print(config.filter_name())   # "scale"
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyScaleConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Scaling algorithm name (e.g. "bilinear", "bicubic", "nearest").
    pub algorithm: String,
}

#[pymethods]
impl PyScaleConfig {
    /// Create a new scale filter configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Output width in pixels
    /// * `height` - Output height in pixels
    /// * `algorithm` - Scaling algorithm (default "bilinear")
    #[new]
    #[pyo3(signature = (width, height, algorithm = "bilinear".to_string()))]
    fn new(width: u32, height: u32, algorithm: String) -> Self {
        Self {
            width,
            height,
            algorithm,
        }
    }

    /// Return the filter name identifier.
    fn filter_name(&self) -> &str {
        "scale"
    }

    /// Get the output width.
    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    /// Get the output height.
    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    /// Get the scaling algorithm.
    #[getter]
    fn algorithm(&self) -> &str {
        &self.algorithm
    }

    fn __str__(&self) -> String {
        format!(
            "ScaleConfig({}x{}, algorithm='{}')",
            self.width, self.height, self.algorithm
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "ScaleConfig(width={}, height={}, algorithm='{}')",
            self.width, self.height, self.algorithm
        )
    }
}

// ─────────────────────────────── Crop ────────────────────────────────────────

/// Configuration for a crop filter.
///
/// # Example
///
/// ```python
/// config = CropConfig(x=100, y=50, width=1280, height=720)
/// print(config.filter_name())   # "crop"
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyCropConfig {
    /// X offset in pixels from the left edge.
    pub x: u32,
    /// Y offset in pixels from the top edge.
    pub y: u32,
    /// Crop width in pixels.
    pub width: u32,
    /// Crop height in pixels.
    pub height: u32,
}

#[pymethods]
impl PyCropConfig {
    /// Create a new crop filter configuration.
    ///
    /// # Arguments
    ///
    /// * `x` - X offset in pixels
    /// * `y` - Y offset in pixels
    /// * `width` - Crop width in pixels
    /// * `height` - Crop height in pixels
    #[new]
    fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return the filter name identifier.
    fn filter_name(&self) -> &str {
        "crop"
    }

    /// Get the X offset.
    #[getter]
    fn x(&self) -> u32 {
        self.x
    }

    /// Get the Y offset.
    #[getter]
    fn y(&self) -> u32 {
        self.y
    }

    /// Get the crop width.
    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    /// Get the crop height.
    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    fn __str__(&self) -> String {
        format!(
            "CropConfig(x={}, y={}, {}x{})",
            self.x, self.y, self.width, self.height
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "CropConfig(x={}, y={}, width={}, height={})",
            self.x, self.y, self.width, self.height
        )
    }
}

// ─────────────────────────────── Volume ──────────────────────────────────────

/// Configuration for a volume (gain) filter.
///
/// # Example
///
/// ```python
/// config = VolumeConfig(gain=0.5)   # -6dB
/// print(config.filter_name())       # "volume"
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyVolumeConfig {
    /// Linear gain factor (1.0 = unity, 2.0 = +6dB, 0.5 = -6dB).
    pub gain: f32,
}

#[pymethods]
impl PyVolumeConfig {
    /// Create a new volume filter configuration.
    ///
    /// # Arguments
    ///
    /// * `gain` - Linear gain factor (default 1.0 = unity gain)
    #[new]
    #[pyo3(signature = (gain = 1.0))]
    fn new(gain: f32) -> Self {
        Self { gain }
    }

    /// Return the filter name identifier.
    fn filter_name(&self) -> &str {
        "volume"
    }

    /// Get the gain factor.
    #[getter]
    fn gain(&self) -> f32 {
        self.gain
    }

    /// Get the gain in decibels.
    fn gain_db(&self) -> f32 {
        20.0 * self.gain.max(f32::EPSILON).log10()
    }

    fn __str__(&self) -> String {
        format!("VolumeConfig(gain={:.3})", self.gain)
    }

    fn __repr__(&self) -> String {
        format!("VolumeConfig(gain={})", self.gain)
    }
}

// ─────────────────────────────── Normalize ───────────────────────────────────

/// Configuration for an audio normalize filter.
///
/// # Example
///
/// ```python
/// config = NormalizeConfig(mode="ebu")   # EBU R128 normalization
/// print(config.filter_name())            # "normalize"
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyNormalizeConfig {
    /// Normalization mode: "peak", "rms", or "ebu".
    pub mode: String,
    /// Target level (in dBFS for peak/rms, in LUFS for ebu).
    pub target_level: f32,
}

#[pymethods]
impl PyNormalizeConfig {
    /// Create a new normalize filter configuration.
    ///
    /// # Arguments
    ///
    /// * `mode` - Normalization mode: "peak", "rms", or "ebu" (default "ebu")
    /// * `target_level` - Target loudness level (default -23.0 LUFS for EBU R128)
    #[new]
    #[pyo3(signature = (mode = "ebu".to_string(), target_level = -23.0))]
    fn new(mode: String, target_level: f32) -> Self {
        Self { mode, target_level }
    }

    /// Return the filter name identifier.
    fn filter_name(&self) -> &str {
        "normalize"
    }

    /// Get the normalization mode.
    #[getter]
    fn mode(&self) -> &str {
        &self.mode
    }

    /// Get the target level.
    #[getter]
    fn target_level(&self) -> f32 {
        self.target_level
    }

    fn __str__(&self) -> String {
        format!(
            "NormalizeConfig(mode='{}', target={:.1})",
            self.mode, self.target_level
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "NormalizeConfig(mode='{}', target_level={})",
            self.mode, self.target_level
        )
    }
}
