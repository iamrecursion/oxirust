//! Core types for Python bindings.
// Buffer protocol (`__getbuffer__`/`__releasebuffer__`) requires unsafe fn and
// unsafe blocks because it directly manipulates raw `Py_buffer` pointers via
// the C Python API.  This is a narrowly scoped, necessary exception to the
// workspace `unsafe_code = "deny"` policy.
#![allow(unsafe_code)]

use oximedia_codec::{
    AudioFrame as RustAudioFrame, ChannelLayout as RustChannelLayout,
    SampleFormat as RustSampleFormat, VideoFrame as RustVideoFrame,
};
use oximedia_core::PixelFormat as RustPixelFormat;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::ffi::{c_int, CString};

/// Pixel format for video frames.
#[pyclass]
#[derive(Clone, Copy)]
pub struct PixelFormat {
    inner: RustPixelFormat,
}

#[pymethods]
impl PixelFormat {
    /// YUV 4:2:0 planar format (most common).
    #[classattr]
    const YUV420P: &'static str = "yuv420p";

    /// YUV 4:2:2 planar format.
    #[classattr]
    const YUV422P: &'static str = "yuv422p";

    /// YUV 4:4:4 planar format.
    #[classattr]
    const YUV444P: &'static str = "yuv444p";

    /// Grayscale 8-bit.
    #[classattr]
    const GRAY8: &'static str = "gray8";

    /// Create a new pixel format from string.
    #[new]
    fn new(format: &str) -> PyResult<Self> {
        let inner = match format {
            "yuv420p" => RustPixelFormat::Yuv420p,
            "yuv422p" => RustPixelFormat::Yuv422p,
            "yuv444p" => RustPixelFormat::Yuv444p,
            "gray8" => RustPixelFormat::Gray8,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown pixel format: {format}"
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Check if format is planar.
    fn is_planar(&self) -> bool {
        self.inner.is_planar()
    }

    /// Get number of planes.
    fn plane_count(&self) -> usize {
        self.inner.plane_count() as usize
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PixelFormat('{:?}')", self.inner)
    }
}

impl PixelFormat {
    #[must_use]
    pub fn inner(&self) -> RustPixelFormat {
        self.inner
    }

    #[must_use]
    pub fn from_rust(inner: RustPixelFormat) -> Self {
        Self { inner }
    }

    /// Public Rust-facing constructor (mirrors the `#[new]` pymethods fn).
    pub fn new_rust(format: &str) -> PyResult<Self> {
        let inner = match format {
            "yuv420p" => RustPixelFormat::Yuv420p,
            "yuv422p" => RustPixelFormat::Yuv422p,
            "yuv444p" => RustPixelFormat::Yuv444p,
            "gray8" => RustPixelFormat::Gray8,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown pixel format: {format}"
                )))
            }
        };
        Ok(Self { inner })
    }
}

/// Audio sample format.
#[pyclass]
#[derive(Clone, Copy)]
pub struct SampleFormat {
    inner: RustSampleFormat,
}

#[pymethods]
impl SampleFormat {
    /// 32-bit floating point.
    #[classattr]
    const F32: &'static str = "f32";

    /// 16-bit signed integer.
    #[classattr]
    const I16: &'static str = "i16";

    /// Create a new sample format from string.
    #[new]
    fn new(format: &str) -> PyResult<Self> {
        let inner = match format {
            "f32" => RustSampleFormat::F32,
            "i16" => RustSampleFormat::I16,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sample format: {format}"
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Get sample size in bytes.
    fn sample_size(&self) -> usize {
        self.inner.sample_size()
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("SampleFormat('{:?}')", self.inner)
    }
}

impl SampleFormat {
    #[must_use]
    pub fn inner(&self) -> RustSampleFormat {
        self.inner
    }

    #[must_use]
    pub fn from_rust(inner: RustSampleFormat) -> Self {
        Self { inner }
    }

    /// Public Rust-facing constructor (mirrors the `#[new]` pymethods fn).
    pub fn new_rust(format: &str) -> PyResult<Self> {
        let inner = match format {
            "f32" => RustSampleFormat::F32,
            "i16" => RustSampleFormat::I16,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sample format: {format}"
                )))
            }
        };
        Ok(Self { inner })
    }
}

/// Audio channel layout.
#[pyclass]
#[derive(Clone, Copy)]
pub struct ChannelLayout {
    inner: RustChannelLayout,
}

#[pymethods]
impl ChannelLayout {
    /// Single channel (mono).
    #[classattr]
    const MONO: &'static str = "mono";

    /// Two channels (stereo).
    #[classattr]
    const STEREO: &'static str = "stereo";

    /// Create a new channel layout from string.
    #[new]
    fn new(layout: &str) -> PyResult<Self> {
        let inner = match layout {
            "mono" => RustChannelLayout::Mono,
            "stereo" => RustChannelLayout::Stereo,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown channel layout: {layout}"
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Get number of channels.
    fn channel_count(&self) -> usize {
        self.inner.channel_count()
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("ChannelLayout('{:?}')", self.inner)
    }
}

impl ChannelLayout {
    #[must_use]
    pub fn inner(&self) -> RustChannelLayout {
        self.inner
    }
}

/// Rational number for frame rates and timebases.
#[pyclass]
#[derive(Clone, Copy)]
pub struct Rational {
    num: i32,
    den: i32,
}

#[pymethods]
impl Rational {
    /// Create a new rational number.
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator
    /// * `den` - Denominator
    #[new]
    fn new(num: i32, den: i32) -> Self {
        Self { num, den }
    }

    /// Get numerator.
    #[getter]
    fn num(&self) -> i32 {
        self.num
    }

    /// Get denominator.
    #[getter]
    fn den(&self) -> i32 {
        self.den
    }

    /// Convert to float.
    fn to_float(&self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }

    /// Pickle support: return constructor arguments.
    fn __getnewargs__(&self) -> (i32, i32) {
        (self.num, self.den)
    }

    fn __str__(&self) -> String {
        format!("{}/{}", self.num, self.den)
    }

    fn __repr__(&self) -> String {
        format!("Rational({}, {})", self.num, self.den)
    }
}

impl Rational {
    #[must_use]
    pub fn to_rust(&self) -> oximedia_core::Rational {
        oximedia_core::Rational::new(i64::from(self.num), i64::from(self.den))
    }

    #[must_use]
    pub fn from_rust(r: oximedia_core::Rational) -> Self {
        Self {
            num: r.num as i32,
            den: r.den as i32,
        }
    }
}

/// Video frame containing decoded pixel data.
#[pyclass]
pub struct VideoFrame {
    pub(crate) inner: RustVideoFrame,
}

#[pymethods]
impl VideoFrame {
    /// Create a new video frame.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `format` - Pixel format
    #[new]
    fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        Self {
            inner: RustVideoFrame::new(format.inner(), width, height),
        }
    }

    /// Get frame width.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Get frame height.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Get pixel format.
    #[getter]
    fn format(&self) -> PixelFormat {
        PixelFormat::from_rust(self.inner.format)
    }

    /// Get presentation timestamp.
    #[getter]
    pub fn pts(&self) -> i64 {
        self.inner.timestamp.pts
    }

    /// Set presentation timestamp.
    #[setter]
    fn set_pts(&mut self, pts: i64) {
        self.inner.timestamp.pts = pts;
    }

    /// Get plane data by index.
    ///
    /// # Arguments
    ///
    /// * `index` - Plane index (0 = Y, 1 = U, 2 = V for YUV formats)
    fn plane_data<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyBytes>> {
        if index >= self.inner.planes.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Plane index {} out of range (frame has {} planes)",
                index,
                self.inner.planes.len()
            )));
        }
        Ok(PyBytes::new(py, &self.inner.planes[index].data))
    }

    /// Get plane stride by index.
    fn plane_stride(&self, index: usize) -> PyResult<usize> {
        if index >= self.inner.planes.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Plane index {} out of range (frame has {} planes)",
                index,
                self.inner.planes.len()
            )));
        }
        Ok(self.inner.planes[index].stride)
    }

    /// Get number of planes.
    fn plane_count(&self) -> usize {
        self.inner.planes.len()
    }

    /// Return a zero-copy buffer view for the given plane index.
    ///
    /// The returned [`PyVideoPlaneBuffer`] implements Python's buffer protocol,
    /// so `numpy.asarray(frame.plane(0))` will create a zero-copy 2D numpy array
    /// with shape `(height, width)` and dtype `uint8`.
    ///
    /// # Arguments
    ///
    /// * `index` - Plane index (0 = Y/luma, 1 = U/Cb, 2 = V/Cr for YUV formats)
    fn plane(slf: Py<Self>, py: Python<'_>, index: usize) -> PyResult<Py<PyVideoPlaneBuffer>> {
        // Validate index while holding a borrow, then release.
        let (height, width) = {
            let frame = slf.borrow(py);
            if index >= frame.inner.planes.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Plane index {} out of range (frame has {} planes)",
                    index,
                    frame.inner.planes.len()
                )));
            }
            let h = frame.inner.planes[index].height as usize;
            let w = frame.inner.planes[index].width as usize;
            // Fall back to frame dimensions for plane 0 when plane dims are 0
            let (fh, fw) = (frame.inner.height as usize, frame.inner.width as usize);
            let (ph, pw) = if h == 0 && w == 0 { (fh, fw) } else { (h, w) };
            (ph, pw)
        };
        Py::new(
            py,
            PyVideoPlaneBuffer {
                owner: slf,
                plane_index: index,
                height,
                width,
            },
        )
    }

    fn __str__(&self) -> String {
        format!(
            "VideoFrame({}x{}, {:?})",
            self.inner.width, self.inner.height, self.inner.format
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoFrame(width={}, height={}, format={:?}, pts={})",
            self.inner.width, self.inner.height, self.inner.format, self.inner.timestamp.pts
        )
    }
}

impl VideoFrame {
    #[must_use]
    pub fn from_rust(inner: RustVideoFrame) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn inner(&self) -> &RustVideoFrame {
        &self.inner
    }

    /// Public Rust-facing constructor (mirrors the `#[new]` pymethods fn).
    #[must_use]
    pub fn new_rust(width: u32, height: u32, format: PixelFormat) -> Self {
        Self {
            inner: RustVideoFrame::new(format.inner(), width, height),
        }
    }

    /// Public Rust-facing width accessor.
    #[must_use]
    pub fn width_rust(&self) -> u32 {
        self.inner.width
    }

    /// Public Rust-facing height accessor.
    #[must_use]
    pub fn height_rust(&self) -> u32 {
        self.inner.height
    }

    /// Public Rust-facing PTS setter.
    pub fn set_pts_rust(&mut self, pts: i64) {
        self.inner.timestamp.pts = pts;
    }
}

/// Audio frame containing decoded PCM samples.
///
/// Implements Python's buffer protocol so `numpy.asarray(frame)` returns a
/// zero-copy 2D view with shape `(sample_count, channels)` and the
/// appropriate numpy dtype (`float32`, `int16`, etc.).
///
/// Multi-view safety: each call to `__getbuffer__` independently owns its
/// heap allocations via `(*view).internal` and `(*view).format` raw pointers.
/// There are no per-struct fields that could be overwritten when a second
/// buffer view is acquired concurrently.
#[pyclass]
pub struct AudioFrame {
    inner: RustAudioFrame,
}

#[pymethods]
impl AudioFrame {
    /// Create a new audio frame.
    ///
    /// # Arguments
    ///
    /// * `samples` - Raw sample data (interleaved if multi-channel)
    /// * `sample_count` - Number of samples per channel
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `format` - Sample format
    #[new]
    #[pyo3(signature = (samples, sample_count, sample_rate, channels, format))]
    fn new(
        samples: Vec<u8>,
        sample_count: usize,
        sample_rate: u32,
        channels: usize,
        format: SampleFormat,
    ) -> Self {
        Self {
            inner: RustAudioFrame::new(
                samples,
                sample_count,
                sample_rate,
                channels,
                format.inner(),
            ),
        }
    }

    /// Get sample data as bytes.
    fn samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.samples)
    }

    /// Get number of samples per channel.
    #[getter]
    pub fn sample_count(&self) -> usize {
        self.inner.sample_count
    }

    /// Get sample rate in Hz.
    #[getter]
    pub fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// Get number of channels.
    #[getter]
    pub fn channels(&self) -> usize {
        self.inner.channels
    }

    /// Get sample format.
    #[getter]
    fn format(&self) -> SampleFormat {
        SampleFormat::from_rust(self.inner.format)
    }

    /// Get presentation timestamp.
    #[getter]
    fn pts(&self) -> Option<i64> {
        self.inner.pts
    }

    /// Get duration in seconds.
    fn duration_seconds(&self) -> f64 {
        self.inner.duration_seconds()
    }

    /// Convert samples to f32 array.
    pub fn to_f32(&self) -> PyResult<Vec<f32>> {
        self.inner.to_f32().map_err(crate::error::from_codec_error)
    }

    /// Convert samples to i16 array.
    fn to_i16(&self) -> PyResult<Vec<i16>> {
        self.inner.to_i16().map_err(crate::error::from_codec_error)
    }

    fn __str__(&self) -> String {
        format!(
            "AudioFrame({} samples, {}Hz, {} channels, {:?})",
            self.inner.sample_count, self.inner.sample_rate, self.inner.channels, self.inner.format
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioFrame(sample_count={}, sample_rate={}, channels={}, format={:?})",
            self.inner.sample_count, self.inner.sample_rate, self.inner.channels, self.inner.format
        )
    }

    /// Buffer protocol: expose sample data as a 2D numpy-compatible buffer.
    ///
    /// Shape: `(sample_count, channels)`, dtype derived from the sample format.
    ///
    /// Each call independently allocates shape/stride data stored in
    /// `(*view).internal` and the format string in `(*view).format`.
    /// This makes concurrent multi-view access safe — no struct-level fields
    /// are touched that could be overwritten by a second simultaneous view.
    ///
    /// # Safety
    ///
    /// The caller (Python's buffer protocol machinery) must guarantee that
    /// `view` is a valid, writable pointer to a `Py_buffer` struct.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "view pointer is null",
            ));
        }

        // Reject writable requests — our buffer is always read-only.
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "AudioFrame buffer is read-only",
            ));
        }

        let sample_count = slf.inner.sample_count as ffi::Py_ssize_t;
        let channels = slf.inner.channels as ffi::Py_ssize_t;
        let itemsize = slf.inner.format.sample_size() as ffi::Py_ssize_t;

        // Per-view heap allocation: [shape0, shape1, stride0, stride1].
        // Owned through (*view).internal — never stored on the struct, so
        // concurrent views are safe (each owns its own allocation).
        let ss = Box::new([
            sample_count,
            channels,
            channels * itemsize, // row stride: bytes between rows
            itemsize,            // col stride: bytes between columns
        ]);
        let ss_raw = Box::into_raw(ss);

        // Format string: 'B' = u8, 'h' = i16, 'i' = i32, 'f' = f32
        let fmt_str = match slf.inner.format {
            RustSampleFormat::U8 => "B",
            RustSampleFormat::I16 => "h",
            RustSampleFormat::I32 => "i",
            RustSampleFormat::F32 => "f",
        };
        // Only allocate format CString if the caller requested it.
        let fmt_ptr = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            CString::new(fmt_str)
                .map_err(|e| {
                    // Reclaim the shape/stride box before returning the error.
                    // SAFETY: ss_raw was just created by Box::into_raw above.
                    unsafe {
                        drop(Box::from_raw(ss_raw));
                    }
                    pyo3::exceptions::PyBufferError::new_err(format!(
                        "failed to build format string: {e}"
                    ))
                })?
                .into_raw()
        } else {
            std::ptr::null_mut()
        };

        // SAFETY: view is valid (checked above); data pointer is into slf.inner.samples
        // which remains alive as long as the Python object is alive (enforced by
        // (*view).obj holding an incremented refcount below).
        unsafe {
            // Increment refcount: keep the Python object alive for the buffer's lifetime.
            (*view).obj = pyo3::ffi::compat::Py_NewRef(slf.as_ptr().cast::<pyo3::ffi::PyObject>());

            (*view).buf = slf
                .inner
                .samples
                .as_ptr()
                .cast_mut()
                .cast::<std::ffi::c_void>();
            (*view).len = slf.inner.samples.len() as ffi::Py_ssize_t;
            (*view).readonly = 1;
            (*view).itemsize = itemsize;
            (*view).ndim = 2;

            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                ss_raw.cast::<ffi::Py_ssize_t>()
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                ss_raw.cast::<ffi::Py_ssize_t>().add(2)
            } else {
                std::ptr::null_mut()
            };
            (*view).format = fmt_ptr;
            (*view).suboffsets = std::ptr::null_mut();
            // Park the shape/strides box raw pointer in (*view).internal for
            // per-view ownership.  __releasebuffer__ will reconstruct and drop it.
            (*view).internal = ss_raw.cast::<std::ffi::c_void>();
        }

        Ok(())
    }

    /// Buffer protocol: release resources allocated during `__getbuffer__`.
    ///
    /// Reconstructs and drops the per-view `Box<[Py_ssize_t; 4]>` from
    /// `(*view).internal` and the `CString` from `(*view).format`.
    ///
    /// # Safety
    ///
    /// Called by Python's buffer protocol machinery; `view` must be the same
    /// pointer that was passed to `__getbuffer__`.
    unsafe fn __releasebuffer__(&mut self, view: *mut ffi::Py_buffer) {
        // SAFETY: (*view).internal was set to ss_raw from Box::into_raw in __getbuffer__.
        unsafe {
            if !(*view).internal.is_null() {
                drop(Box::from_raw(
                    (*view).internal.cast::<[ffi::Py_ssize_t; 4]>(),
                ));
                (*view).internal = std::ptr::null_mut();
            }
            if !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = std::ptr::null_mut();
            }
        }
    }
}

impl AudioFrame {
    #[must_use]
    pub fn from_rust(inner: RustAudioFrame) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn inner(&self) -> &RustAudioFrame {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// PyVideoPlaneBuffer
// ---------------------------------------------------------------------------

/// Zero-copy numpy-compatible buffer view for a single plane of a [`VideoFrame`].
///
/// Obtained via `VideoFrame.plane(index)`.  Implements Python's buffer protocol
/// so that `numpy.asarray(frame.plane(0))` returns a 2-D `uint8` array with
/// shape `(height, width)` backed directly by the frame's memory — no copy.
///
/// The `VideoFrame` owner is kept alive (via `Py<VideoFrame>`) for as long as
/// any active numpy view exists.
///
/// Multi-view safety: each `__getbuffer__` call independently allocates its
/// shape/stride data, owned via `(*view).internal`.  No per-struct fields are
/// overwritten, so concurrent numpy views on the same plane object are safe.
#[pyclass]
pub struct PyVideoPlaneBuffer {
    /// Keeps the source `VideoFrame` alive for the lifetime of this object.
    owner: Py<VideoFrame>,
    /// Which plane (0 = Y/luma, 1 = U/Cb, 2 = V/Cr …).
    plane_index: usize,
    /// Plane height in pixels (rows).
    height: usize,
    /// Plane width in pixels (columns).
    width: usize,
}

#[pymethods]
impl PyVideoPlaneBuffer {
    /// Plane index within the parent frame.
    #[getter]
    fn plane_index(&self) -> usize {
        self.plane_index
    }

    /// Plane height in pixels.
    #[getter]
    fn height(&self) -> usize {
        self.height
    }

    /// Plane width in pixels.
    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    fn __str__(&self) -> String {
        format!(
            "PyVideoPlaneBuffer(plane={}, {}x{})",
            self.plane_index, self.height, self.width
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVideoPlaneBuffer(plane_index={}, height={}, width={})",
            self.plane_index, self.height, self.width
        )
    }

    /// Buffer protocol: expose plane data as a 2D numpy-compatible buffer.
    ///
    /// Shape: `(height, width)`, dtype `uint8`.
    ///
    /// Each call independently allocates shape/stride data owned through
    /// `(*view).internal` and the format string owned through `(*view).format`.
    /// No per-struct fields are modified, making concurrent multi-view access safe.
    ///
    /// # Safety
    ///
    /// The caller (Python's buffer protocol machinery) must guarantee that
    /// `view` is a valid, writable pointer to a `Py_buffer` struct.
    unsafe fn __getbuffer__(
        slf: PyRef<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "view pointer is null",
            ));
        }

        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "PyVideoPlaneBuffer is read-only",
            ));
        }

        // Borrow the owning VideoFrame via the Python token embedded in slf.
        // `slf.py()` returns the Python token without any unsafe block — we hold
        // the GIL because we were called from Python through the buffer protocol.
        let py = slf.py();
        let frame = slf.owner.borrow(py);
        if slf.plane_index >= frame.inner.planes.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Plane index {} out of range",
                slf.plane_index
            )));
        }
        let plane = &frame.inner.planes[slf.plane_index];
        let data_ptr = plane.data.as_ptr();
        let data_len = plane.data.len();
        let height = slf.height as ffi::Py_ssize_t;
        let width = slf.width as ffi::Py_ssize_t;
        let stride = plane.stride as ffi::Py_ssize_t;
        // Release the borrow; data_ptr remains valid because the VideoFrame is
        // kept alive via `(*view).obj` below.
        drop(frame);

        // Per-view heap allocation: [height, width, row_stride, col_stride=1].
        // Owned through (*view).internal — never stored on the struct.
        let ss = Box::new([
            height, width, stride, // bytes between rows
            1isize, // bytes between columns (u8, so 1)
        ]);
        let ss_raw = Box::into_raw(ss);

        // Only allocate format CString if the caller requested it.
        let fmt_ptr = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
            CString::new("B")
                .map_err(|e| {
                    // Reclaim the shape/stride box before propagating the error.
                    // SAFETY: ss_raw was just created by Box::into_raw above.
                    unsafe {
                        drop(Box::from_raw(ss_raw));
                    }
                    pyo3::exceptions::PyBufferError::new_err(format!(
                        "failed to build format string: {e}"
                    ))
                })?
                .into_raw()
        } else {
            std::ptr::null_mut()
        };

        // SAFETY: view is valid (checked above); data_ptr points into plane.data
        // which is owned by the VideoFrame that `slf.owner` keeps alive.
        unsafe {
            // Increment refcount on this (PyVideoPlaneBuffer) object so Python
            // knows the buffer source is alive for the buffer's lifetime.
            (*view).obj = pyo3::ffi::compat::Py_NewRef(slf.as_ptr().cast::<pyo3::ffi::PyObject>());

            (*view).buf = data_ptr.cast_mut().cast::<std::ffi::c_void>();
            (*view).len = data_len as ffi::Py_ssize_t;
            (*view).readonly = 1;
            (*view).itemsize = 1; // u8
            (*view).ndim = 2;

            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                ss_raw.cast::<ffi::Py_ssize_t>()
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                ss_raw.cast::<ffi::Py_ssize_t>().add(2)
            } else {
                std::ptr::null_mut()
            };
            (*view).format = fmt_ptr;
            (*view).suboffsets = std::ptr::null_mut();
            // Park the shape/strides box raw pointer for per-view ownership.
            // __releasebuffer__ will reconstruct and drop it.
            (*view).internal = ss_raw.cast::<std::ffi::c_void>();
        }

        Ok(())
    }

    /// Buffer protocol: release resources allocated during `__getbuffer__`.
    ///
    /// Reconstructs and drops the per-view `Box<[Py_ssize_t; 4]>` from
    /// `(*view).internal` and the `CString` from `(*view).format`.
    ///
    /// # Safety
    ///
    /// Called by Python's buffer protocol machinery; `view` must be the same
    /// pointer that was passed to `__getbuffer__`.
    unsafe fn __releasebuffer__(&mut self, view: *mut ffi::Py_buffer) {
        // SAFETY: (*view).internal was set to ss_raw (Box::into_raw) in __getbuffer__.
        unsafe {
            if !(*view).internal.is_null() {
                drop(Box::from_raw(
                    (*view).internal.cast::<[ffi::Py_ssize_t; 4]>(),
                ));
                (*view).internal = std::ptr::null_mut();
            }
            if !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = std::ptr::null_mut();
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// Encoder preset (speed vs quality tradeoff).
#[pyclass]
#[derive(Clone, Copy)]
pub struct EncoderPreset {
    inner: oximedia_codec::EncoderPreset,
}

#[pymethods]
impl EncoderPreset {
    /// Fastest encoding, lowest quality.
    #[classattr]
    const ULTRAFAST: &'static str = "ultrafast";

    /// Fast encoding.
    #[classattr]
    const FAST: &'static str = "fast";

    /// Balanced speed and quality.
    #[classattr]
    const MEDIUM: &'static str = "medium";

    /// Slower, better quality.
    #[classattr]
    const SLOW: &'static str = "slow";

    /// Very slow, high quality.
    #[classattr]
    const VERYSLOW: &'static str = "veryslow";

    /// Create a new encoder preset from string.
    #[new]
    fn new(preset: &str) -> PyResult<Self> {
        let inner = match preset {
            "ultrafast" => oximedia_codec::EncoderPreset::Ultrafast,
            "superfast" => oximedia_codec::EncoderPreset::Superfast,
            "veryfast" => oximedia_codec::EncoderPreset::Veryfast,
            "faster" => oximedia_codec::EncoderPreset::Faster,
            "fast" => oximedia_codec::EncoderPreset::Fast,
            "medium" => oximedia_codec::EncoderPreset::Medium,
            "slow" => oximedia_codec::EncoderPreset::Slow,
            "slower" => oximedia_codec::EncoderPreset::Slower,
            "veryslow" => oximedia_codec::EncoderPreset::Veryslow,
            "placebo" => oximedia_codec::EncoderPreset::Placebo,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown encoder preset: {preset}"
                )))
            }
        };
        Ok(Self { inner })
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("EncoderPreset('{:?}')", self.inner)
    }
}

impl EncoderPreset {
    #[must_use]
    pub fn inner(&self) -> oximedia_codec::EncoderPreset {
        self.inner
    }
}

/// Encoder configuration.
#[pyclass]
#[derive(Clone)]
pub struct EncoderConfig {
    pub(crate) inner: oximedia_codec::EncoderConfig,
}

#[pymethods]
impl EncoderConfig {
    /// Create a new encoder configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `framerate` - Frame rate as (numerator, denominator) tuple
    /// * `crf` - Constant rate factor (quality, lower = better, typically 18-28)
    /// * `preset` - Encoder preset (default: "medium")
    /// * `keyint` - Keyframe interval in frames (default: 250)
    #[new]
    #[pyo3(signature = (width, height, framerate=(30, 1), crf=28.0, preset=None, keyint=250))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: u32,
        height: u32,
        framerate: (i32, i32),
        crf: f32,
        preset: Option<EncoderPreset>,
        keyint: u32,
    ) -> Self {
        let mut inner = oximedia_codec::EncoderConfig::default();
        inner.width = width;
        inner.height = height;
        inner.framerate =
            oximedia_core::Rational::new(i64::from(framerate.0), i64::from(framerate.1));
        inner.bitrate = oximedia_codec::BitrateMode::Crf(crf);
        inner.preset = preset.map_or(oximedia_codec::EncoderPreset::Medium, |p| p.inner());
        inner.keyint = keyint;
        Self { inner }
    }

    /// Get frame width.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Get frame height.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Get frame rate as (numerator, denominator).
    #[getter]
    fn framerate(&self) -> (i32, i32) {
        (
            self.inner.framerate.num as i32,
            self.inner.framerate.den as i32,
        )
    }

    /// Get keyframe interval.
    #[getter]
    fn keyint(&self) -> u32 {
        self.inner.keyint
    }

    /// Pickle support: reduce to constructor args.
    fn __getstate__(&self) -> (u32, u32, (i32, i32), u32) {
        (
            self.inner.width,
            self.inner.height,
            (
                self.inner.framerate.num as i32,
                self.inner.framerate.den as i32,
            ),
            self.inner.keyint,
        )
    }

    /// Pickle support: restore from constructor args.
    fn __setstate__(&mut self, state: (u32, u32, (i32, i32), u32)) -> PyResult<()> {
        let (width, height, (fps_num, fps_den), keyint) = state;
        self.inner.width = width;
        self.inner.height = height;
        self.inner.framerate = oximedia_core::Rational::new(i64::from(fps_num), i64::from(fps_den));
        self.inner.keyint = keyint;
        Ok(())
    }

    fn __str__(&self) -> String {
        format!(
            "EncoderConfig({}x{}, {}fps)",
            self.inner.width,
            self.inner.height,
            self.inner.framerate.num / self.inner.framerate.den
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "EncoderConfig(width={}, height={}, framerate=({}, {}), keyint={})",
            self.inner.width,
            self.inner.height,
            self.inner.framerate.num,
            self.inner.framerate.den,
            self.inner.keyint
        )
    }
}

impl EncoderConfig {
    #[must_use]
    pub fn inner(&self) -> &oximedia_codec::EncoderConfig {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::buffer::PyBuffer;

    /// Build a minimal allocated VideoFrame for use in buffer-protocol tests.
    fn make_yuv_frame(width: u32, height: u32) -> VideoFrame {
        let mut inner = RustVideoFrame::new(oximedia_core::PixelFormat::Yuv420p, width, height);
        inner.allocate();
        // Fill Y plane with a distinctive value to verify zero-copy.
        if !inner.planes.is_empty() {
            inner.planes[0].data.iter_mut().for_each(|b| *b = 0xAB);
        }
        VideoFrame { inner }
    }

    // ── PyVideoPlaneBuffer ────────────────────────────────────────────────────

    #[test]
    fn test_plane_buffer_shape_and_dtype() {
        Python::initialize();
        Python::attach(|py| {
            let frame = make_yuv_frame(16, 8);
            let vf = Py::new(py, frame).expect("should create Py<VideoFrame>");
            // plane(0) = Y, shape expected (8, 16)
            let plane_buf = VideoFrame::plane(vf, py, 0).expect("plane(0) should succeed");
            let pb = plane_buf.borrow(py);
            assert_eq!(pb.height(), 8);
            assert_eq!(pb.width(), 16);
            assert_eq!(pb.plane_index(), 0);
        });
    }

    #[test]
    fn test_plane_buffer_out_of_range() {
        Python::initialize();
        Python::attach(|py| {
            let frame = make_yuv_frame(4, 4);
            let vf = Py::new(py, frame).expect("should create Py<VideoFrame>");
            // YUV420 has 3 planes; index 3 is out of range.
            let result = VideoFrame::plane(vf, py, 10);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_plane_buffer_protocol_readable() {
        Python::initialize();
        Python::attach(|py| {
            let frame = make_yuv_frame(16, 8);
            let vf = Py::new(py, frame).expect("should create Py<VideoFrame>");
            let plane_buf = VideoFrame::plane(vf, py, 0).expect("plane(0) should succeed");
            // Access via PyO3's PyBuffer<u8> wrapper (exercises __getbuffer__).
            let bound = plane_buf.into_bound(py);
            let buf = PyBuffer::<u8>::get(&bound).expect("buffer protocol should work");
            assert_eq!(buf.dimensions(), 2);
            assert_eq!(buf.shape(), &[8, 16]);
            assert_eq!(buf.item_size(), 1);
            // The first byte of the Y plane was set to 0xAB.
            let data: Vec<u8> = buf.to_vec(py).expect("to_vec should succeed");
            assert_eq!(data.len(), 8 * 16);
            assert!(data.iter().all(|&b| b == 0xAB));
        });
    }

    #[test]
    fn test_plane_index_u_plane_shape() {
        Python::initialize();
        Python::attach(|py| {
            let frame = make_yuv_frame(16, 8);
            let vf = Py::new(py, frame).expect("Py::new");
            // U plane (index 1) should be half the luma dimensions.
            let plane_buf = VideoFrame::plane(vf, py, 1).expect("plane(1) should succeed");
            let pb = plane_buf.borrow(py);
            // allocate() calls plane_dimensions which computes (8, 4) for YUV420p chroma.
            assert_eq!(pb.height(), 4);
            assert_eq!(pb.width(), 8);
        });
    }

    // ── AudioFrame buffer protocol ────────────────────────────────────────────

    fn make_audio_frame_f32(sample_count: usize, channels: usize) -> AudioFrame {
        // Interleaved f32 samples: sample_count * channels * 4 bytes.
        let total_bytes = sample_count * channels * 4;
        let mut raw = vec![0u8; total_bytes];
        // Fill the first sample of channel 0 with a recognisable bit pattern.
        // f32: 1.0 = 0x3F80_0000 (little-endian: [0x00, 0x00, 0x80, 0x3F]).
        if total_bytes >= 4 {
            raw[0] = 0x00;
            raw[1] = 0x00;
            raw[2] = 0x80;
            raw[3] = 0x3F;
        }
        AudioFrame {
            inner: RustAudioFrame::new(raw, sample_count, 48000, channels, RustSampleFormat::F32),
        }
    }

    fn make_audio_frame_i16(sample_count: usize, channels: usize) -> AudioFrame {
        let total_bytes = sample_count * channels * 2;
        let raw = vec![0u8; total_bytes];
        AudioFrame {
            inner: RustAudioFrame::new(raw, sample_count, 44100, channels, RustSampleFormat::I16),
        }
    }

    #[test]
    fn test_audio_frame_buffer_shape_f32() {
        Python::initialize();
        Python::attach(|py| {
            let af = make_audio_frame_f32(1024, 2);
            let af_py = Py::new(py, af).expect("Py::new");
            let bound = af_py.into_bound(py);
            let buf = PyBuffer::<f32>::get(&bound).expect("buffer protocol for f32");
            assert_eq!(buf.dimensions(), 2);
            assert_eq!(buf.shape(), &[1024, 2]);
            assert_eq!(buf.item_size(), 4);
        });
    }

    #[test]
    fn test_audio_frame_buffer_shape_i16() {
        Python::initialize();
        Python::attach(|py| {
            let af = make_audio_frame_i16(512, 1);
            let af_py = Py::new(py, af).expect("Py::new");
            let bound = af_py.into_bound(py);
            let buf = PyBuffer::<i16>::get(&bound).expect("buffer protocol for i16");
            assert_eq!(buf.dimensions(), 2);
            assert_eq!(buf.shape(), &[512, 1]);
            assert_eq!(buf.item_size(), 2);
        });
    }

    #[test]
    fn test_audio_frame_buffer_data_f32() {
        Python::initialize();
        Python::attach(|py| {
            let af = make_audio_frame_f32(4, 1);
            let af_py = Py::new(py, af).expect("Py::new");
            let bound = af_py.into_bound(py);
            let buf = PyBuffer::<f32>::get(&bound).expect("buffer protocol");
            let data = buf.to_vec(py).expect("to_vec");
            // First element was encoded as f32 1.0.
            assert!((data[0] - 1.0f32).abs() < f32::EPSILON);
            // Remaining elements are 0.0.
            assert!(data[1..].iter().all(|&v| v == 0.0));
        });
    }

    #[test]
    fn test_audio_frame_buffer_writable_rejected() {
        Python::initialize();
        Python::attach(|py| {
            let af = make_audio_frame_f32(4, 1);
            let af_py = Py::new(py, af).expect("Py::new");
            let bound = af_py.into_bound(py);
            // PyBuffer requires writable access if the element type is f32;
            // in practice PyBuffer::get uses PyBUF_RECORDS (no writable flag), so
            // this should succeed.  We test that the buffer is not mutable by
            // checking the readonly flag directly.
            let buf = PyBuffer::<f32>::get(&bound).expect("buffer protocol");
            assert!(buf.readonly());
        });
    }

    /// Multi-view safety test: two simultaneous buffer views on the same AudioFrame
    /// must both report the correct shape/dtype.  With the old struct-field approach
    /// the second `PyBuffer::get` would overwrite the Box pointer used by the first
    /// view, causing a use-after-free on `buf1.shape()`.  With per-view ownership
    /// through `(*view).internal` both views are independent and correct.
    #[test]
    fn test_audio_frame_multi_view_safety() {
        Python::initialize();
        Python::attach(|py| {
            let af = make_audio_frame_f32(256, 2);
            let af_py = Py::new(py, af).expect("Py::new");
            let bound = af_py.into_bound(py);
            // Acquire two views simultaneously.
            let buf1 = PyBuffer::<f32>::get(&bound).expect("first buffer view");
            let buf2 = PyBuffer::<f32>::get(&bound).expect("second buffer view");
            // Both views must report the same shape.
            assert_eq!(buf1.shape(), &[256, 2]);
            assert_eq!(buf2.shape(), &[256, 2]);
            assert_eq!(buf1.item_size(), 4);
            assert_eq!(buf2.item_size(), 4);
            // Read data through both — no UAF if the implementation is correct.
            let data1 = buf1.to_vec(py).expect("to_vec buf1");
            let data2 = buf2.to_vec(py).expect("to_vec buf2");
            assert_eq!(data1.len(), 256 * 2);
            assert_eq!(data1, data2);
        });
    }

    /// Multi-view safety test for PyVideoPlaneBuffer: two simultaneous numpy views
    /// on the same plane object must both see the correct shape and data.
    #[test]
    fn test_video_plane_multi_view_safety() {
        Python::initialize();
        Python::attach(|py| {
            let frame = make_yuv_frame(8, 4);
            let vf = Py::new(py, frame).expect("Py::new VideoFrame");
            let plane_buf = VideoFrame::plane(vf, py, 0).expect("plane(0)");
            let bound = plane_buf.into_bound(py);
            // Acquire two views simultaneously.
            let buf1 = PyBuffer::<u8>::get(&bound).expect("first plane view");
            let buf2 = PyBuffer::<u8>::get(&bound).expect("second plane view");
            // Both must agree on shape.
            assert_eq!(buf1.shape(), &[4, 8]);
            assert_eq!(buf2.shape(), &[4, 8]);
            // Data must agree and match the fill value.
            let data1 = buf1.to_vec(py).expect("to_vec buf1");
            let data2 = buf2.to_vec(py).expect("to_vec buf2");
            assert!(data1.iter().all(|&b| b == 0xAB));
            assert_eq!(data1, data2);
        });
    }
}
