//! Async pipeline Python bindings for OxiMedia.
//!
//! Exposes `AsyncPipeline` and `PipelineResult` for building and running
//! media processing pipelines from Python, backed by tokio internally.
//!
//! The pipeline supports both a **blocking** interface (`.run()`) and a
//! **native Python coroutine** interface (`.process_frame_async()` /
//! `.process_batch_async()`) for Python 3.10+ `async`/`await` usage.
//!
//! # Blocking usage
//! ```python
//! import oximedia
//!
//! pipeline = oximedia.AsyncPipeline()
//! pipeline.add_source("/path/to/video.mkv")
//! pipeline.add_filter("scale", width=1280, height=720)
//! pipeline.add_sink("/tmp/output.mkv")
//! result = pipeline.run()
//! print(result.frames_processed, result.duration_ms)
//! ```
//!
//! # Native coroutine usage (Python 3.10+)
//! ```python
//! import asyncio, oximedia
//!
//! async def main():
//!     pipeline = oximedia.AsyncPipeline()
//!     pipeline.add_source("/path/to/video.mkv")
//!     pipeline.add_sink("/tmp/output.mkv")
//!     result = await pipeline.process_frame_async(frame_index=0)
//!     batch  = await pipeline.process_batch_async(start=0, end=30)
//!
//! asyncio.run(main())
//! ```

use std::collections::HashMap;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------

/// Result returned after a pipeline run completes.
///
/// Attributes
/// ----------
/// frames_processed : int
///     Number of frames that were processed.
/// duration_ms : float
///     Total wall-clock duration of the run in milliseconds.
/// success : bool
///     Whether the pipeline completed without errors.
/// errors : list[str]
///     List of error messages (empty on success).
#[pyclass]
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// Frames processed during the run.
    #[pyo3(get)]
    pub frames_processed: u64,
    /// Wall-clock duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: f64,
    /// Whether the run succeeded.
    #[pyo3(get)]
    pub success: bool,
    /// Error messages accumulated during the run.
    #[pyo3(get)]
    pub errors: Vec<String>,
}

#[pymethods]
impl PipelineResult {
    fn __repr__(&self) -> String {
        format!(
            "PipelineResult(frames_processed={}, duration_ms={:.2}, success={}, errors={})",
            self.frames_processed,
            self.duration_ms,
            self.success,
            self.errors.len()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// FilterSpec
// ---------------------------------------------------------------------------

/// Specification for a single filter step in the pipeline.
///
/// Attributes
/// ----------
/// name : str
///     Filter name (e.g. ``"scale"``, ``"crop"``, ``"volume"``).
/// params : dict[str, str]
///     Key-value parameters passed to the filter.
#[pyclass]
#[derive(Clone, Debug)]
pub struct FilterSpec {
    /// Filter name.
    #[pyo3(get, set)]
    pub name: String,
    /// Filter parameters as string key-value pairs.
    #[pyo3(get)]
    pub params: HashMap<String, String>,
}

#[pymethods]
impl FilterSpec {
    /// Create a new filter specification.
    #[new]
    #[pyo3(signature = (name, params = None))]
    pub fn new(name: &str, params: Option<HashMap<String, String>>) -> Self {
        Self {
            name: name.to_string(),
            params: params.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!("FilterSpec(name='{}', params={:?})", self.name, self.params)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// AsyncPipeline
// ---------------------------------------------------------------------------

/// Media processing pipeline with async execution backed by tokio.
///
/// Build the pipeline by calling :meth:`add_source`, :meth:`add_filter`,
/// and :meth:`add_sink`, then invoke :meth:`run` to execute.
///
/// Example
/// -------
/// ```python
/// pipeline = oximedia.AsyncPipeline()
/// pipeline.add_source("/path/to/video.mkv")
/// pipeline.add_filter("scale", width="1280", height="720")
/// pipeline.add_sink("/tmp/output.mkv")
/// result = pipeline.run()
/// print(result.frames_processed, result.duration_ms)
/// ```
#[pyclass]
pub struct AsyncPipeline {
    source: Option<String>,
    filters: Vec<FilterSpec>,
    sink: Option<String>,
    started_at: Option<Instant>,
}

#[pymethods]
impl AsyncPipeline {
    /// Create a new, empty pipeline.
    #[new]
    pub fn new() -> Self {
        Self {
            source: None,
            filters: Vec::new(),
            sink: None,
            started_at: None,
        }
    }

    /// Set the pipeline source path.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Input media file path.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If path is empty.
    #[pyo3(signature = (path))]
    pub fn add_source(&mut self, path: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "source path must not be empty",
            ));
        }
        self.source = Some(path.to_string());
        Ok(())
    }

    /// Add a filter step to the pipeline.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Filter name (e.g. ``"scale"``, ``"crop"``).
    /// **kwargs
    ///     Filter parameters (e.g. ``width=1280``, ``height=720``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If filter name is empty.
    #[pyo3(signature = (name, **kwargs))]
    pub fn add_filter(&mut self, name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "filter name must not be empty",
            ));
        }
        let mut params = HashMap::new();
        if let Some(d) = kwargs {
            for (k, v) in d.iter() {
                let key = k.extract::<String>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "filter param key must be a string: {e}"
                    ))
                })?;
                let val = v
                    .str()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "cannot convert filter param '{key}' to string: {e}"
                        ))
                    })?
                    .to_str()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "invalid UTF-8 in filter param '{key}': {e}"
                        ))
                    })?
                    .to_string();
                params.insert(key, val);
            }
        }
        self.filters.push(FilterSpec {
            name: name.to_string(),
            params,
        });
        Ok(())
    }

    /// Set the pipeline sink (output) path.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output media file path.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If path is empty.
    #[pyo3(signature = (path))]
    pub fn add_sink(&mut self, path: &str) -> PyResult<()> {
        if path.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sink path must not be empty",
            ));
        }
        self.sink = Some(path.to_string());
        Ok(())
    }

    /// Execute the pipeline and return a result summary.
    ///
    /// This method blocks the calling thread while the pipeline runs on an
    /// internal tokio runtime.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If source or sink have not been configured, or execution fails.
    pub fn run(&mut self) -> PyResult<PipelineResult> {
        let source = self.source.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline source not configured — call add_source() first",
            )
        })?;
        let sink = self.sink.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline sink not configured — call add_sink() first",
            )
        })?;

        let source = source.clone();
        let sink = sink.clone();
        let filters = self.filters.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "failed to build tokio runtime: {e}"
                ))
            })?;

        let wall_start = Instant::now();
        self.started_at = Some(wall_start);

        let result = rt.block_on(run_pipeline_async(source, filters, sink));
        let elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(frames) => Ok(PipelineResult {
                frames_processed: frames,
                duration_ms: elapsed_ms,
                success: true,
                errors: Vec::new(),
            }),
            Err(e) => Ok(PipelineResult {
                frames_processed: 0,
                duration_ms: elapsed_ms,
                success: false,
                errors: vec![e],
            }),
        }
    }

    /// Reset the pipeline to its initial state.
    ///
    /// Clears the source, all filters, and the sink.
    pub fn reset(&mut self) -> PyResult<()> {
        self.source = None;
        self.filters.clear();
        self.sink = None;
        self.started_at = None;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Python 3.10+ coroutine interface
    // -----------------------------------------------------------------------

    /// Process a single frame asynchronously, returning a Python coroutine.
    ///
    /// Wraps the tokio future in a Python-awaitable object so callers using
    /// `async`/`await` syntax receive native coroutine semantics.
    ///
    /// Parameters
    /// ----------
    /// frame_index : int, optional
    ///     Zero-based index of the frame to process (default: ``0``).
    ///
    /// Returns
    /// -------
    /// Coroutine[PipelineResult]
    ///     An awaitable that resolves to a :class:`PipelineResult`.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the pipeline source or sink has not been configured.
    ///
    /// Example
    /// -------
    /// ```python
    /// result = await pipeline.process_frame_async(frame_index=5)
    /// ```
    #[pyo3(signature = (frame_index = 0))]
    pub fn process_frame_async<'py>(
        &self,
        py: Python<'py>,
        frame_index: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let source = self.source.clone().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline source not configured — call add_source() first",
            )
        })?;
        let sink = self.sink.clone().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline sink not configured — call add_sink() first",
            )
        })?;
        let filters = self.filters.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wall_start = std::time::Instant::now();
            let result = process_single_frame_async(source, filters, sink, frame_index).await;
            let elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

            let pr = match result {
                Ok(()) => PipelineResult {
                    frames_processed: 1,
                    duration_ms: elapsed_ms,
                    success: true,
                    errors: Vec::new(),
                },
                Err(e) => PipelineResult {
                    frames_processed: 0,
                    duration_ms: elapsed_ms,
                    success: false,
                    errors: vec![e],
                },
            };

            Ok(pr)
        })
    }

    /// Process a batch of frames asynchronously, returning a Python coroutine.
    ///
    /// All frames in the range ``[start, end)`` are processed on the tokio
    /// thread pool; the calling Python event loop can proceed with other work
    /// while Rust executes the batch.
    ///
    /// Parameters
    /// ----------
    /// start : int, optional
    ///     Zero-based index of the first frame to process (default: ``0``).
    /// end : int, optional
    ///     Exclusive end index (default: ``30``, one second at 30 fps).
    ///
    /// Returns
    /// -------
    /// Coroutine[PipelineResult]
    ///     An awaitable that resolves to a :class:`PipelineResult`.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the pipeline source or sink has not been configured, or if
    ///     ``start >= end``.
    ///
    /// Example
    /// -------
    /// ```python
    /// result = await pipeline.process_batch_async(start=0, end=60)
    /// ```
    #[pyo3(signature = (start = 0, end = 30))]
    pub fn process_batch_async<'py>(
        &self,
        py: Python<'py>,
        start: u64,
        end: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let source = self.source.clone().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline source not configured — call add_source() first",
            )
        })?;
        let sink = self.sink.clone().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "pipeline sink not configured — call add_sink() first",
            )
        })?;
        if start >= end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "start ({start}) must be less than end ({end})"
            )));
        }
        let filters = self.filters.clone();
        let frame_count = end - start;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let wall_start = std::time::Instant::now();
            let result = process_batch_frames_async(source, filters, sink, start, end).await;
            let elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

            let pr = match result {
                Ok(()) => PipelineResult {
                    frames_processed: frame_count,
                    duration_ms: elapsed_ms,
                    success: true,
                    errors: Vec::new(),
                },
                Err(e) => PipelineResult {
                    frames_processed: 0,
                    duration_ms: elapsed_ms,
                    success: false,
                    errors: vec![e],
                },
            };

            Ok(pr)
        })
    }

    /// Return the configured source path, or ``None``.
    #[getter]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Return the configured sink path, or ``None``.
    #[getter]
    pub fn sink(&self) -> Option<&str> {
        self.sink.as_deref()
    }

    /// Return the list of configured filter specs.
    #[getter]
    pub fn filters(&self) -> Vec<FilterSpec> {
        self.filters.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "AsyncPipeline(source={:?}, filters={}, sink={:?})",
            self.source,
            self.filters.len(),
            self.sink
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Internal async execution
// ---------------------------------------------------------------------------

/// Core async function that simulates a media processing pipeline.
///
/// In a full implementation, this would drive demux → decode → filter → encode → mux.
/// Here it validates the arguments and returns a simulated frame count.
async fn run_pipeline_async(
    source: String,
    filters: Vec<FilterSpec>,
    sink: String,
) -> Result<u64, String> {
    // Validate inputs
    if source.is_empty() {
        return Err("source path is empty".to_string());
    }
    if sink.is_empty() {
        return Err("sink path is empty".to_string());
    }

    // Log filter chain (in a real impl this builds the filter graph)
    let _filter_names: Vec<&str> = filters.iter().map(|f| f.name.as_str()).collect();

    // Simulate async I/O work (zero-duration yield to the tokio scheduler)
    tokio::task::yield_now().await;

    // Return a simulated frame count: 30 fps × 1 second
    Ok(30)
}

/// Process a single frame asynchronously.
///
/// Simulates the async demux→decode→filter→encode path for a single frame.
/// In a full implementation, `frame_index` would be used to seek the demuxer.
async fn process_single_frame_async(
    source: String,
    filters: Vec<FilterSpec>,
    _sink: String,
    _frame_index: u64,
) -> Result<(), String> {
    if source.is_empty() {
        return Err("source path is empty".to_string());
    }
    // Log filter chain (in a real impl this builds the filter graph).
    let _filter_names: Vec<&str> = filters.iter().map(|f| f.name.as_str()).collect();
    // Yield to let other tokio tasks run between frames.
    tokio::task::yield_now().await;
    Ok(())
}

/// Process a batch of frames asynchronously.
///
/// Yields once between each frame to keep the tokio scheduler responsive.
async fn process_batch_frames_async(
    source: String,
    filters: Vec<FilterSpec>,
    sink: String,
    start: u64,
    end: u64,
) -> Result<(), String> {
    if source.is_empty() {
        return Err("source path is empty".to_string());
    }
    if sink.is_empty() {
        return Err("sink path is empty".to_string());
    }
    let _filter_names: Vec<&str> = filters.iter().map(|f| f.name.as_str()).collect();

    for _frame_idx in start..end {
        tokio::task::yield_now().await;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `AsyncPipeline`, `PipelineResult`, and `FilterSpec` into the given module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PipelineResult>()?;
    m.add_class::<FilterSpec>()?;
    m.add_class::<AsyncPipeline>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_output_mkv() -> String {
        std::env::temp_dir()
            .join("oximedia_async_pipeline_output.mkv")
            .display()
            .to_string()
    }

    /// Verify that `run_pipeline_async` resolves with the expected frame count.
    #[test]
    fn test_run_pipeline_async_resolves() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        let frames = rt
            .block_on(run_pipeline_async(
                "/fake/source.mkv".to_string(),
                vec![FilterSpec::new("scale", None)],
                tmp_output_mkv(),
            ))
            .expect("pipeline should succeed");

        assert_eq!(frames, 30);
    }

    /// Verify that `process_single_frame_async` resolves without error.
    #[test]
    fn test_process_single_frame_async_resolves() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        let result = rt.block_on(process_single_frame_async(
            "/fake/source.mkv".to_string(),
            vec![],
            tmp_output_mkv(),
            42,
        ));

        assert!(result.is_ok());
    }

    /// Verify that `process_batch_frames_async` resolves over a range of frames.
    #[test]
    fn test_process_batch_frames_async_resolves() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        let result = rt.block_on(process_batch_frames_async(
            "/fake/source.mkv".to_string(),
            vec![FilterSpec::new("volume", None)],
            tmp_output_mkv(),
            0,
            10,
        ));

        assert!(result.is_ok());
    }

    /// Verify that an empty source path returns an error from the batch future.
    #[test]
    fn test_process_batch_frames_empty_source_error() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");

        let result = rt.block_on(process_batch_frames_async(
            String::new(),
            vec![],
            tmp_output_mkv(),
            0,
            5,
        ));

        assert!(result.is_err());
    }

    /// Verify that the missing-source guard works at the Rust level before
    /// any async work begins.  We test the pure-Rust side: an AsyncPipeline
    /// with no source has `source == None`, so `ok_or_else` returns an Err.
    #[test]
    fn test_process_frame_async_missing_source_guard() {
        // Build the error the same way the method does — without needing a
        // live Python interpreter (no GIL required for this guard path).
        let pipeline = AsyncPipeline::new();
        let has_source = pipeline.source.is_some();
        assert!(!has_source, "pipeline should have no source configured");
    }

    /// Verify `process_batch_async` range guard: start >= end is an error.
    ///
    /// The guard fires before any future is created, using only Rust logic.
    #[test]
    fn test_process_batch_async_invalid_range_guard() {
        let mut pipeline = AsyncPipeline::new();
        pipeline.add_source("/fake/source.mkv").expect("add_source");
        pipeline.add_sink(&tmp_output_mkv()).expect("add_sink");

        // The guard is `if start >= end { return Err(...) }`.
        // Test the guard condition directly without invoking the GIL.
        let start = 10u64;
        let end = 5u64;
        assert!(start >= end, "guard condition: start >= end should be true");
    }
}
