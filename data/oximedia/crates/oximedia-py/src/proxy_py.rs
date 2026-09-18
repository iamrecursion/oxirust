//! Python bindings for proxy media generation and management.
//!
//! Provides `PyProxyGenerator`, `PyProxyConfig`, `PyProxyFile` and standalone
//! functions for generating and managing proxy media files.

use oximedia_transcode::TranscodePipeline;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn gen_id() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("proxy-{:016x}", dur.as_nanos())
}

// ---------------------------------------------------------------------------
// PyProxyConfig
// ---------------------------------------------------------------------------

/// Configuration for proxy generation.
#[pyclass]
#[derive(Clone)]
pub struct PyProxyConfig {
    /// Resolution preset: quarter, half, full.
    #[pyo3(get, set)]
    pub resolution: String,
    /// Quality preset: low, medium, high.
    #[pyo3(get, set)]
    pub quality: String,
    /// Codec: vp9, av1.
    #[pyo3(get, set)]
    pub codec: String,
    /// Target bitrate in bps (0 = auto).
    #[pyo3(get, set)]
    pub bitrate: u64,
}

#[pymethods]
impl PyProxyConfig {
    #[new]
    #[pyo3(signature = (resolution="quarter", quality="medium", codec="vp9", bitrate=0))]
    fn new(resolution: &str, quality: &str, codec: &str, bitrate: u64) -> Self {
        Self {
            resolution: resolution.to_string(),
            quality: quality.to_string(),
            codec: codec.to_string(),
            bitrate,
        }
    }

    /// Get the resolution scale factor (0.0-1.0).
    fn scale_factor(&self) -> f64 {
        match self.resolution.as_str() {
            "quarter" => 0.25,
            "half" => 0.5,
            "full" => 1.0,
            _ => 0.25,
        }
    }

    /// Estimate output bitrate for a given resolution.
    fn estimated_bitrate(&self, width: u32, height: u32) -> u64 {
        if self.bitrate > 0 {
            return self.bitrate;
        }
        let scale = self.scale_factor();
        let pixels = (width as f64 * scale) * (height as f64 * scale);
        let quality_mult = match self.quality.as_str() {
            "low" => 0.5,
            "medium" => 1.0,
            "high" => 2.0,
            _ => 1.0,
        };
        (pixels * 2.0 * quality_mult) as u64
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("resolution".to_string(), self.resolution.clone());
        m.insert("quality".to_string(), self.quality.clone());
        m.insert("codec".to_string(), self.codec.clone());
        m.insert("bitrate".to_string(), self.bitrate.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProxyConfig(resolution='{}', quality='{}', codec='{}')",
            self.resolution, self.quality, self.codec
        )
    }
}

// ---------------------------------------------------------------------------
// PyProxyFile
// ---------------------------------------------------------------------------

/// A proxy file entry linking proxy to original.
#[pyclass]
#[derive(Clone)]
pub struct PyProxyFile {
    /// Proxy identifier.
    #[pyo3(get)]
    pub id: String,
    /// Original file path.
    #[pyo3(get)]
    pub original_path: String,
    /// Proxy file path.
    #[pyo3(get)]
    pub proxy_path: String,
    /// Resolution preset used.
    #[pyo3(get)]
    pub resolution: String,
    /// Quality preset used.
    #[pyo3(get)]
    pub quality: String,
    /// Codec used.
    #[pyo3(get)]
    pub codec: String,
    /// Original file size in bytes.
    #[pyo3(get)]
    pub original_size: u64,
    /// Proxy file size in bytes.
    #[pyo3(get)]
    pub proxy_size: u64,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
}

#[pymethods]
impl PyProxyFile {
    /// Compression ratio (original / proxy).
    fn compression_ratio(&self) -> f64 {
        if self.proxy_size == 0 {
            return 0.0;
        }
        self.original_size as f64 / self.proxy_size as f64
    }

    /// Space savings as percentage.
    fn space_savings_pct(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        (1.0 - self.proxy_size as f64 / self.original_size as f64) * 100.0
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("original_path".to_string(), self.original_path.clone());
        m.insert("proxy_path".to_string(), self.proxy_path.clone());
        m.insert("resolution".to_string(), self.resolution.clone());
        m.insert("quality".to_string(), self.quality.clone());
        m.insert("codec".to_string(), self.codec.clone());
        m.insert("original_size".to_string(), self.original_size.to_string());
        m.insert("proxy_size".to_string(), self.proxy_size.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProxyFile(original='{}', proxy='{}', resolution='{}')",
            self.original_path, self.proxy_path, self.resolution
        )
    }
}

// ---------------------------------------------------------------------------
// PyProxyGenerator
// ---------------------------------------------------------------------------

/// Proxy media generator and manager.
#[pyclass]
pub struct PyProxyGenerator {
    config: PyProxyConfig,
    proxies: Vec<PyProxyFile>,
}

#[pymethods]
impl PyProxyGenerator {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyProxyConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(|| PyProxyConfig::new("quarter", "medium", "vp9", 0)),
            proxies: Vec::new(),
        }
    }

    /// Generate a proxy for a source file.
    ///
    /// Delegates to the real `oximedia_transcode::TranscodePipeline` to
    /// actually transcode `original_path` into `proxy_path`. If the output
    /// container is not one the pipeline can honestly produce, or the
    /// pipeline itself fails, this returns a real `PyErr` — it never writes
    /// a placeholder/marker file in place of a real proxy.
    ///
    /// # Errors
    ///
    /// Returns `PyValueError` if `original_path` does not exist or
    /// `proxy_path`'s extension is not one of the containers the pipeline
    /// can produce (`mkv`, `webm`, `ogg`, `oga`, `opus`). Returns
    /// `PyRuntimeError` if directory creation, the async runtime, or the
    /// transcode pipeline itself fails.
    fn generate(&mut self, original_path: &str, proxy_path: &str) -> PyResult<PyProxyFile> {
        let orig = std::path::Path::new(original_path);
        if !orig.exists() {
            return Err(PyValueError::new_err(format!(
                "Original file not found: {original_path}"
            )));
        }

        let orig_meta = std::fs::metadata(orig)
            .map_err(|e| PyRuntimeError::new_err(format!("Metadata error: {e}")))?;

        // Determine whether the output extension is supported by TranscodePipeline
        // *before* touching the filesystem, so an unsupported request has no
        // side effects (no directory created, no file written).
        let out_ext = std::path::Path::new(proxy_path)
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase)
            .unwrap_or_default();
        if !matches!(out_ext.as_str(), "mkv" | "webm" | "ogg" | "oga" | "opus") {
            return Err(PyValueError::new_err(format!(
                "Unsupported proxy output container '.{out_ext}' for '{proxy_path}'; the \
                 transcode pipeline currently supports: mkv, webm, ogg, oga, opus"
            )));
        }

        // Ensure the proxy output directory exists.
        if let Some(parent) = std::path::Path::new(proxy_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create proxy dir: {e}"))
                })?;
            }
        }

        // Run the real transcode pipeline in a current-thread async runtime.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create async runtime: {e}")))?;

        let input_pb = std::path::PathBuf::from(original_path);
        let output_pb = std::path::PathBuf::from(proxy_path);

        let run_result = rt.block_on(async {
            let mut pipeline = TranscodePipeline::builder()
                .input(input_pb)
                .output(output_pb)
                .track_progress(false)
                .build()
                .map_err(|e| format!("Pipeline build error: {e}"))?;
            pipeline
                .execute()
                .await
                .map_err(|e| format!("Pipeline exec error: {e}"))
        });

        let actual_size = match run_result {
            Ok(transcode_out) => transcode_out.file_size,
            Err(e) => {
                // Real pipeline failure — never paper over it with a fabricated
                // result. Best-effort clean up any partial output the failed
                // attempt may have left behind, then propagate a real error.
                let _ = std::fs::remove_file(proxy_path);
                return Err(PyRuntimeError::new_err(format!(
                    "Proxy generation failed for '{original_path}' -> '{proxy_path}': {e}"
                )));
            }
        };

        let proxy_file = PyProxyFile {
            id: gen_id(),
            original_path: original_path.to_string(),
            proxy_path: proxy_path.to_string(),
            resolution: self.config.resolution.clone(),
            quality: self.config.quality.clone(),
            codec: self.config.codec.clone(),
            original_size: orig_meta.len(),
            proxy_size: actual_size,
            created_at: now_timestamp(),
        };

        let result = proxy_file.clone();
        self.proxies.push(proxy_file);
        Ok(result)
    }

    /// List all generated proxies.
    fn list_proxies(&self) -> Vec<PyProxyFile> {
        self.proxies.clone()
    }

    /// Get proxy count.
    fn proxy_count(&self) -> usize {
        self.proxies.len()
    }

    /// Get current config.
    fn config(&self) -> PyProxyConfig {
        self.config.clone()
    }

    /// Total space saved across all proxies.
    fn total_space_saved(&self) -> u64 {
        self.proxies
            .iter()
            .map(|p| p.original_size.saturating_sub(p.proxy_size))
            .sum()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProxyGenerator(proxies={}, config={})",
            self.proxies.len(),
            self.config.__repr__()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Generate a proxy for a single file with default settings.
#[pyfunction]
#[pyo3(signature = (original_path, proxy_path, resolution="quarter", quality="medium"))]
pub fn generate_proxy(
    original_path: &str,
    proxy_path: &str,
    resolution: &str,
    quality: &str,
) -> PyResult<PyProxyFile> {
    let config = PyProxyConfig::new(resolution, quality, "vp9", 0);
    let mut gen = PyProxyGenerator::new(Some(config));
    gen.generate(original_path, proxy_path)
}

/// List supported proxy formats.
#[pyfunction]
pub fn list_proxy_formats() -> Vec<HashMap<String, String>> {
    let formats = vec![
        ("vp9", "WebM/VP9", "Good balance of quality and size"),
        ("av1", "WebM/AV1", "Best compression, slower encoding"),
    ];
    formats
        .into_iter()
        .map(|(codec, name, desc)| {
            let mut m = HashMap::new();
            m.insert("codec".to_string(), codec.to_string());
            m.insert("name".to_string(), name.to_string());
            m.insert("description".to_string(), desc.to_string());
            m
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all proxy bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProxyConfig>()?;
    m.add_class::<PyProxyFile>()?;
    m.add_class::<PyProxyGenerator>()?;
    m.add_function(wrap_pyfunction!(generate_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(list_proxy_formats, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-py-proxy-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_proxy_config_scale() {
        let cfg = PyProxyConfig::new("quarter", "medium", "vp9", 0);
        assert!((cfg.scale_factor() - 0.25).abs() < f64::EPSILON);

        let cfg = PyProxyConfig::new("half", "medium", "vp9", 0);
        assert!((cfg.scale_factor() - 0.5).abs() < f64::EPSILON);

        let cfg = PyProxyConfig::new("full", "medium", "vp9", 0);
        assert!((cfg.scale_factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proxy_config_estimated_bitrate() {
        let cfg = PyProxyConfig::new("quarter", "medium", "vp9", 5_000_000);
        assert_eq!(cfg.estimated_bitrate(1920, 1080), 5_000_000);

        let cfg = PyProxyConfig::new("quarter", "medium", "vp9", 0);
        let br = cfg.estimated_bitrate(1920, 1080);
        assert!(br > 0);
    }

    #[test]
    fn test_proxy_file_compression() {
        let pf = PyProxyFile {
            id: "proxy-001".to_string(),
            original_path: tmp_str("orig.mov"),
            proxy_path: tmp_str("proxy.webm"),
            resolution: "quarter".to_string(),
            quality: "medium".to_string(),
            codec: "vp9".to_string(),
            original_size: 1_000_000,
            proxy_size: 100_000,
            created_at: "0".to_string(),
        };
        assert!((pf.compression_ratio() - 10.0).abs() < f64::EPSILON);
        assert!((pf.space_savings_pct() - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proxy_file_zero_size() {
        let pf = PyProxyFile {
            id: "proxy-002".to_string(),
            original_path: tmp_str("orig.mov"),
            proxy_path: tmp_str("proxy.webm"),
            resolution: "quarter".to_string(),
            quality: "medium".to_string(),
            codec: "vp9".to_string(),
            original_size: 0,
            proxy_size: 0,
            created_at: "0".to_string(),
        };
        assert!((pf.compression_ratio() - 0.0).abs() < f64::EPSILON);
        assert!((pf.space_savings_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_list_proxy_formats() {
        let formats = list_proxy_formats();
        assert_eq!(formats.len(), 2);
        assert!(formats[0].contains_key("codec"));
    }

    /// Unique per-call temp path so parallel tests never collide. Preserves
    /// `name`'s extension (if any) at the very end of the filename, since
    /// `generate()` dispatches on `Path::extension()`.
    fn unique_tmp(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::path::Path::new(name);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
        let filename = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => format!("oximedia-py-proxy-gen-{stem}-{nanos}.{ext}"),
            None => format!("oximedia-py-proxy-gen-{stem}-{nanos}"),
        };
        std::env::temp_dir().join(filename)
    }

    /// Like `.expect()`, but never touches `PyErr`'s `Debug`/`Display` impl.
    /// Those internally require the Python GIL; in a bare `cargo test` /
    /// `nextest` process for this crate (no embedded Python interpreter),
    /// formatting a `PyErr` while already unwinding from a failed `.expect()`
    /// triggers a second panic ("interpreter not initialized") and aborts
    /// the process (SIGABRT) instead of printing a readable message. This
    /// panics with just `msg` on `Err`, which is always safe.
    trait PyResultTestExt<T> {
        fn expect_ok(self, msg: &str) -> T;
    }

    impl<T> PyResultTestExt<T> for PyResult<T> {
        // Deliberately discards the `PyErr` without formatting it (see the
        // trait doc comment above for why `.expect(msg)` is unsafe here).
        #[allow(clippy::match_wild_err_arm)]
        fn expect_ok(self, msg: &str) -> T {
            match self {
                Ok(v) => v,
                Err(_) => panic!("{msg}"),
            }
        }
    }

    /// Regression test for the fabrication bug: an unsupported output
    /// container must return a real `Err`, and must not create any
    /// placeholder/marker file at `proxy_path`.
    #[test]
    fn test_generate_unsupported_container_returns_err_without_side_effects() {
        let input = unique_tmp("unsupported-in.bin");
        std::fs::write(&input, b"does not need to be real media, only to exist")
            .expect("write test input");
        let output = unique_tmp("unsupported-out.mp4");
        let _ = std::fs::remove_file(&output);

        let mut gen = PyProxyGenerator::new(None);
        let result = gen.generate(
            input.to_str().expect("valid utf8 path"),
            output.to_str().expect("valid utf8 path"),
        );

        assert!(
            result.is_err(),
            "unsupported output container must return Err, not fabricate a proxy"
        );
        assert!(
            !output.exists(),
            "no marker/placeholder file should be written for an unsupported container"
        );

        let _ = std::fs::remove_file(&input);
    }

    /// Regression test for the fabrication bug: a real pipeline failure
    /// (unrecognizable input container) must return a real `Err`, and must
    /// not leave a marker file behind at `proxy_path`.
    #[test]
    fn test_generate_pipeline_failure_returns_err_without_marker_file() {
        let input = unique_tmp("garbage-in.mkv");
        // Non-empty bytes that do not match any known container magic, so
        // the real pipeline's format probe fails for real.
        std::fs::write(
            &input,
            b"this is definitely not a real matroska or ogg container",
        )
        .expect("write test input");
        let output = unique_tmp("garbage-out.mkv");
        let _ = std::fs::remove_file(&output);

        let mut gen = PyProxyGenerator::new(None);
        let result = gen.generate(
            input.to_str().expect("valid utf8 path"),
            output.to_str().expect("valid utf8 path"),
        );

        assert!(
            result.is_err(),
            "a real pipeline failure must return Err, not a fabricated proxy result"
        );
        assert!(
            !output.exists(),
            "no marker file should remain after a failed real pipeline attempt"
        );

        let _ = std::fs::remove_file(&input);
    }

    /// Positive control: a genuinely valid input through a supported
    /// container produces a real, non-empty output file via the real
    /// pipeline (proving the success path is real, not just the failure
    /// path).
    #[test]
    fn test_generate_real_success_produces_real_nonempty_output() {
        use oximedia_container::{
            mux::{MatroskaMuxer, MuxerConfig},
            Muxer, Packet, PacketFlags, StreamInfo,
        };
        use oximedia_core::{CodecId, Rational, Timestamp};
        use oximedia_io::MemorySource;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build test runtime");

        let input = unique_tmp("real-in.mkv");
        rt.block_on(async {
            let in_buf = MemorySource::new_writable(64 * 1024);
            let mut muxer = MatroskaMuxer::new(in_buf, MuxerConfig::new());
            let mut video = StreamInfo::new(0, CodecId::Vp9, Rational::new(1, 1000));
            video.codec_params.width = Some(320);
            video.codec_params.height = Some(240);
            muxer.add_stream(video).expect("add stream");
            muxer.write_header().await.expect("write header");
            for i in 0u64..10 {
                let data = vec![0x42u8, 0x00, (i & 0xFF) as u8, 0x01];
                let pkt = Packet::new(
                    0,
                    bytes::Bytes::from(data),
                    Timestamp::new(i as i64 * 33, Rational::new(1, 1000)),
                    PacketFlags::KEYFRAME,
                );
                muxer.write_packet(&pkt).await.expect("write packet");
            }
            muxer.write_trailer().await.expect("write trailer");
            let sink = muxer.into_sink();
            tokio::fs::write(&input, sink.written_data())
                .await
                .expect("write real input file");
        });

        let output = unique_tmp("real-out.webm");
        let _ = std::fs::remove_file(&output);

        let mut gen = PyProxyGenerator::new(None);
        let result = gen.generate(
            input.to_str().expect("valid utf8 path"),
            output.to_str().expect("valid utf8 path"),
        );

        let proxy_file = result.expect_ok("a valid input with a supported container must succeed");
        assert!(proxy_file.proxy_size > 0, "real output must be non-empty");
        let real_len = std::fs::metadata(&output)
            .expect("real output file must exist on disk")
            .len();
        assert!(real_len > 0);
        assert_eq!(gen.proxy_count(), 1);

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
