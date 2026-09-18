//! Python bindings for quality metrics from `oximedia-quality`.
//!
//! Provides full-reference metrics (PSNR, SSIM) and no-reference metrics
//! (BRISQUE, NIQE, Blockiness, Blur, Noise) via `PyQualityAssessor` and
//! standalone functions.
//!
//! All CPU-intensive metric computations release the GIL via `py.detach()`
//! so Python threads are not serialised during image quality analysis.

use pyo3::prelude::*;

use oximedia_core::PixelFormat;
use oximedia_quality::{Frame, MetricType, QualityAssessor};
use std::collections::HashMap;

/// Quality score result accessible from Python.
#[pyclass]
#[derive(Clone)]
pub struct PyQualityScore {
    /// Name of the metric (e.g. "PSNR", "SSIM").
    #[pyo3(get)]
    pub metric: String,
    /// Overall score value.
    #[pyo3(get)]
    pub score: f64,
    /// Per-component breakdown (e.g. Y / Cb / Cr channels).
    #[pyo3(get)]
    pub components: HashMap<String, f64>,
}

#[pymethods]
impl PyQualityScore {
    fn __repr__(&self) -> String {
        format!(
            "PyQualityScore(metric='{}', score={:.4})",
            self.metric, self.score
        )
    }
}

/// Quality assessor wrapping `oximedia-quality`.
///
/// Each metric computation constructs an assessor inside `py.detach()` so the
/// GIL is released for the duration of the CPU-intensive calculation.
#[pyclass]
pub struct PyQualityAssessor {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn make_frame(data: &[u8], width: usize, height: usize) -> PyResult<Frame> {
    let mut frame = Frame::new(width, height, PixelFormat::Gray8)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    let expected = width * height;
    if data.len() < expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Data too small: need {expected}, got {}",
            data.len()
        )));
    }
    frame.planes[0] = data[..expected].to_vec();
    Ok(frame)
}

fn metric_name(mt: MetricType) -> String {
    match mt {
        MetricType::Psnr => "PSNR".to_string(),
        MetricType::Ssim => "SSIM".to_string(),
        MetricType::MsSsim => "MS-SSIM".to_string(),
        MetricType::Vmaf => "VMAF".to_string(),
        MetricType::Vif => "VIF".to_string(),
        MetricType::Fsim => "FSIM".to_string(),
        MetricType::Niqe => "NIQE".to_string(),
        MetricType::Brisque => "BRISQUE".to_string(),
        MetricType::Blockiness => "Blockiness".to_string(),
        MetricType::Blur => "Blur".to_string(),
        MetricType::Noise => "Noise".to_string(),
        _ => "Unknown".to_string(),
    }
}

fn to_py_score(qs: &oximedia_quality::QualityScore) -> PyQualityScore {
    PyQualityScore {
        metric: metric_name(qs.metric),
        score: qs.score,
        components: qs.components.clone(),
    }
}

fn assess_full_ref(
    py: Python<'_>,
    ref_data: &[u8],
    dist_data: &[u8],
    width: usize,
    height: usize,
    mt: MetricType,
) -> PyResult<PyQualityScore> {
    // Build frames while holding the GIL (they borrow &[u8] which is already Send-safe as Vec)
    let reference = make_frame(ref_data, width, height)?;
    let distorted = make_frame(dist_data, width, height)?;
    // Release GIL for the CPU-intensive metric computation
    let result = py.detach(move || {
        let assessor = QualityAssessor::new();
        assessor
            .assess(&reference, &distorted, mt)
            .map(|qs| to_py_score(&qs))
            .map_err(|e| e.to_string())
    });
    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

fn assess_no_ref(
    py: Python<'_>,
    data: &[u8],
    width: usize,
    height: usize,
    mt: MetricType,
) -> PyResult<PyQualityScore> {
    let frame = make_frame(data, width, height)?;
    // Release GIL for the CPU-intensive metric computation
    let result = py.detach(move || {
        let assessor = QualityAssessor::new();
        assessor
            .assess_no_reference(&frame, mt)
            .map(|qs| to_py_score(&qs))
            .map_err(|e| e.to_string())
    });
    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// ---------------------------------------------------------------------------
// PyMethods
// ---------------------------------------------------------------------------

#[pymethods]
impl PyQualityAssessor {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Compute PSNR between reference and distorted frames.
    fn compute_psnr(
        &self,
        py: Python<'_>,
        ref_data: &[u8],
        dist_data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_full_ref(py, ref_data, dist_data, width, height, MetricType::Psnr)
    }

    /// Compute SSIM between reference and distorted frames.
    fn compute_ssim(
        &self,
        py: Python<'_>,
        ref_data: &[u8],
        dist_data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_full_ref(py, ref_data, dist_data, width, height, MetricType::Ssim)
    }

    /// Compute BRISQUE no-reference quality score.
    fn compute_brisque(
        &self,
        py: Python<'_>,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_no_ref(py, data, width, height, MetricType::Brisque)
    }

    /// Compute NIQE no-reference quality score.
    fn compute_niqe(
        &self,
        py: Python<'_>,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<PyQualityScore> {
        assess_no_ref(py, data, width, height, MetricType::Niqe)
    }

    /// Generate a comprehensive quality report with all no-reference metrics.
    fn quality_report(
        &self,
        py: Python<'_>,
        data: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<Vec<PyQualityScore>> {
        let no_ref_metrics = [
            MetricType::Brisque,
            MetricType::Niqe,
            MetricType::Blockiness,
            MetricType::Blur,
            MetricType::Noise,
        ];
        let mut results = Vec::with_capacity(no_ref_metrics.len());
        for mt in &no_ref_metrics {
            results.push(assess_no_ref(py, data, width, height, *mt)?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Compute PSNR between two grayscale images. Returns the score as a float.
#[pyfunction]
pub fn compute_psnr(
    py: Python<'_>,
    ref_data: &[u8],
    dist_data: &[u8],
    width: usize,
    height: usize,
) -> PyResult<f64> {
    let qs = assess_full_ref(py, ref_data, dist_data, width, height, MetricType::Psnr)?;
    Ok(qs.score)
}

/// Compute SSIM between two grayscale images. Returns the score as a float.
#[pyfunction]
pub fn compute_ssim(
    py: Python<'_>,
    ref_data: &[u8],
    dist_data: &[u8],
    width: usize,
    height: usize,
) -> PyResult<f64> {
    let qs = assess_full_ref(py, ref_data, dist_data, width, height, MetricType::Ssim)?;
    Ok(qs.score)
}

/// Generate a quality report with all no-reference metrics for a grayscale image.
#[pyfunction]
pub fn quality_report(
    py: Python<'_>,
    data: &[u8],
    width: usize,
    height: usize,
) -> PyResult<Vec<PyQualityScore>> {
    let no_ref_metrics = [
        MetricType::Brisque,
        MetricType::Niqe,
        MetricType::Blockiness,
        MetricType::Blur,
        MetricType::Noise,
    ];
    let mut results = Vec::with_capacity(no_ref_metrics.len());
    for mt in &no_ref_metrics {
        results.push(assess_no_ref(py, data, width, height, *mt)?);
    }
    Ok(results)
}
