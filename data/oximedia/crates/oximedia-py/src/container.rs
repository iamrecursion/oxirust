//! Container demuxer and muxer bindings.

use crate::error::{from_container_error, PyOxiResult};
use oximedia_container::{
    demux::{
        Demuxer as RustDemuxer, MatroskaDemuxer as RustMatroskaDemuxer,
        OggDemuxer as RustOggDemuxer,
    },
    mux::{
        MatroskaMuxer as RustMatroskaMuxer, Muxer as RustMuxer, MuxerConfig as RustMuxerConfig,
        OggMuxer as RustOggMuxer,
    },
    Packet as RustPacket, PacketFlags, StreamInfo as RustStreamInfo,
};
use oximedia_core::{Rational, Timestamp};
use oximedia_io::FileSource;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Media packet containing compressed data.
#[pyclass]
#[derive(Clone)]
pub struct Packet {
    inner: RustPacket,
}

#[pymethods]
impl Packet {
    /// Create a new packet.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed packet data
    /// * `stream_index` - Stream index this packet belongs to
    /// * `pts` - Presentation timestamp
    /// * `dts` - Decode timestamp (optional)
    /// * `duration` - Duration in timebase units (optional)
    /// * `keyframe` - Is this a keyframe
    /// * `timebase_num` - Timebase numerator (default: 1)
    /// * `timebase_den` - Timebase denominator (default: 1000)
    #[new]
    #[pyo3(signature = (data, stream_index, pts, dts=None, duration=None, keyframe=false, timebase_num=1, timebase_den=1000))]
    fn new(
        data: Vec<u8>,
        stream_index: usize,
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
        keyframe: bool,
        timebase_num: i32,
        timebase_den: i32,
    ) -> Self {
        let mut timestamp = Timestamp::new(
            pts,
            Rational::new(i64::from(timebase_num), i64::from(timebase_den)),
        );
        timestamp.dts = dts;
        timestamp.duration = duration;

        let flags = if keyframe {
            PacketFlags::KEYFRAME
        } else {
            PacketFlags::empty()
        };

        let inner = RustPacket::new(stream_index, bytes::Bytes::from(data), timestamp, flags);
        Self { inner }
    }

    /// Get packet data.
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }

    /// Get stream index.
    #[getter]
    fn stream_index(&self) -> usize {
        self.inner.stream_index
    }

    /// Get presentation timestamp.
    #[getter]
    fn pts(&self) -> i64 {
        self.inner.pts()
    }

    /// Get decode timestamp.
    #[getter]
    fn dts(&self) -> Option<i64> {
        self.inner.dts()
    }

    /// Get duration.
    #[getter]
    fn duration(&self) -> Option<i64> {
        self.inner.duration()
    }

    /// Check if this is a keyframe.
    fn is_keyframe(&self) -> bool {
        self.inner.is_keyframe()
    }

    /// Get packet size in bytes.
    fn size(&self) -> usize {
        self.inner.size()
    }

    fn __str__(&self) -> String {
        format!(
            "Packet(stream={}, size={}, pts={}, keyframe={})",
            self.inner.stream_index,
            self.inner.size(),
            self.inner.pts(),
            self.inner.is_keyframe()
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Packet(stream_index={}, size={}, pts={}, dts={:?}, duration={:?}, keyframe={})",
            self.inner.stream_index,
            self.inner.size(),
            self.inner.pts(),
            self.inner.dts(),
            self.inner.duration(),
            self.inner.is_keyframe()
        )
    }
}

impl Packet {
    #[must_use]
    pub fn from_rust(inner: RustPacket) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn inner(&self) -> &RustPacket {
        &self.inner
    }
}

/// Stream information.
#[pyclass]
#[derive(Clone)]
pub struct StreamInfo {
    inner: RustStreamInfo,
}

#[pymethods]
impl StreamInfo {
    /// Get stream index.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    /// Get codec name.
    #[getter]
    fn codec(&self) -> String {
        format!("{:?}", self.inner.codec)
    }

    /// Get stream timebase as (numerator, denominator).
    #[getter]
    fn timebase(&self) -> (i32, i32) {
        (
            self.inner.timebase.num as i32,
            self.inner.timebase.den as i32,
        )
    }

    /// Get video width if this is a video stream.
    #[getter]
    fn width(&self) -> Option<u32> {
        self.inner.codec_params.width
    }

    /// Get video height if this is a video stream.
    #[getter]
    fn height(&self) -> Option<u32> {
        self.inner.codec_params.height
    }

    /// Get audio sample rate if this is an audio stream.
    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        self.inner.codec_params.sample_rate
    }

    /// Get audio channel count if this is an audio stream.
    #[getter]
    fn channels(&self) -> Option<u8> {
        self.inner.codec_params.channels
    }

    fn __str__(&self) -> String {
        format!(
            "StreamInfo(index={}, codec={:?})",
            self.inner.index, self.inner.codec
        )
    }

    fn __repr__(&self) -> String {
        let codec_info = if let (Some(width), Some(height)) = (
            self.inner.codec_params.width,
            self.inner.codec_params.height,
        ) {
            format!("video, {width}x{height}")
        } else if let (Some(sample_rate), Some(channels)) = (
            self.inner.codec_params.sample_rate,
            self.inner.codec_params.channels,
        ) {
            format!("audio, {sample_rate}Hz, {channels} channels")
        } else {
            "unknown".to_string()
        };
        format!(
            "StreamInfo(index={}, codec={:?}, {})",
            self.inner.index, self.inner.codec, codec_info
        )
    }
}

impl StreamInfo {
    #[must_use]
    pub fn from_rust(inner: RustStreamInfo) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn inner(&self) -> &RustStreamInfo {
        &self.inner
    }

    #[must_use]
    pub fn into_rust(self) -> RustStreamInfo {
        self.inner
    }
}

/// Matroska/WebM demuxer.
///
/// Demuxes Matroska (.mkv) and `WebM` (.webm) container formats.
///
/// # Example
///
/// ```python
/// demuxer = MatroskaDemuxer("video.mkv")
/// demuxer.probe()
///
/// for stream in demuxer.streams():
///     print(f"Stream {stream.index}: {stream.codec} {stream.width}x{stream.height}")
///
/// while True:
///     try:
///         packet = demuxer.read_packet()
///         print(f"Read packet: stream={packet.stream_index}, size={packet.size()}")
///     except StopIteration:
///         break
/// ```
#[pyclass]
pub struct MatroskaDemuxer {
    demuxer: Option<RustMatroskaDemuxer<FileSource>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl MatroskaDemuxer {
    /// Create a new Matroska demuxer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Matroska/WebM file
    #[new]
    fn new(path: &str) -> PyOxiResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| from_container_error(&format!("Failed to create runtime: {e}")))?,
        );

        let path_owned = path.to_string();
        let source = runtime
            .block_on(async move { FileSource::open(&path_owned).await })
            .map_err(crate::error::from_oxi_error)?;

        let demuxer = RustMatroskaDemuxer::new(source);

        Ok(Self {
            demuxer: Some(demuxer),
            runtime,
        })
    }

    /// Probe the file and parse headers.
    ///
    /// Must be called before reading packets.
    fn probe(&mut self) -> PyOxiResult<()> {
        let demuxer = self
            .demuxer
            .as_mut()
            .ok_or_else(|| from_container_error("Demuxer is closed"))?;

        self.runtime
            .block_on(async { demuxer.probe().await })
            .map_err(crate::error::from_oxi_error)?;

        Ok(())
    }

    /// Read the next packet.
    ///
    /// Raises `StopIteration` when there are no more packets.
    fn read_packet(&mut self) -> PyOxiResult<Packet> {
        let demuxer = self
            .demuxer
            .as_mut()
            .ok_or_else(|| from_container_error("Demuxer is closed"))?;

        let packet = self.runtime.block_on(async { demuxer.read_packet().await });

        match packet {
            Ok(pkt) => Ok(Packet::from_rust(pkt)),
            Err(oximedia_core::OxiError::Eof) => Err(PyErr::new::<
                pyo3::exceptions::PyStopIteration,
                _,
            >("End of file")),
            Err(e) => Err(crate::error::from_oxi_error(e)),
        }
    }

    /// Get information about all streams.
    fn streams(&self) -> PyOxiResult<Vec<StreamInfo>> {
        let demuxer = self
            .demuxer
            .as_ref()
            .ok_or_else(|| from_container_error("Demuxer is closed"))?;

        Ok(demuxer
            .streams()
            .iter()
            .cloned()
            .map(StreamInfo::from_rust)
            .collect())
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: drop the inner demuxer (close file handle).
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        self.demuxer = None;
        Ok(false)
    }

    /// Close the demuxer explicitly.
    fn close(&mut self) {
        self.demuxer = None;
    }

    /// Iterator protocol: return self.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iterator protocol: read next packet or raise StopIteration.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Packet> {
        slf.read_packet()
    }

    fn __str__(&self) -> String {
        "MatroskaDemuxer".to_string()
    }

    fn __repr__(&self) -> String {
        let stream_count = self.demuxer.as_ref().map_or(0, |d| d.streams().len());
        let closed = self.demuxer.is_none();
        format!("MatroskaDemuxer(streams={stream_count}, closed={closed})")
    }
}

/// Ogg demuxer.
///
/// Demuxes Ogg container format (.ogg, .opus, .oga).
///
/// # Example
///
/// ```python
/// demuxer = OggDemuxer("audio.opus")
/// demuxer.probe()
///
/// for stream in demuxer.streams():
///     print(f"Stream {stream.index}: {stream.codec}")
///
/// while True:
///     try:
///         packet = demuxer.read_packet()
///         print(f"Read packet: stream={packet.stream_index}, size={packet.size()}")
///     except StopIteration:
///         break
/// ```
#[pyclass]
pub struct OggDemuxer {
    demuxer: Option<RustOggDemuxer<FileSource>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl OggDemuxer {
    /// Create a new Ogg demuxer.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Ogg file
    #[new]
    fn new(path: &str) -> PyOxiResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| from_container_error(&format!("Failed to create runtime: {e}")))?,
        );

        let path_owned = path.to_string();
        let source = runtime
            .block_on(async move { FileSource::open(&path_owned).await })
            .map_err(crate::error::from_oxi_error)?;

        let demuxer = RustOggDemuxer::new(source);

        Ok(Self {
            demuxer: Some(demuxer),
            runtime,
        })
    }

    /// Probe the file and parse headers.
    ///
    /// Must be called before reading packets.
    fn probe(&mut self) -> PyOxiResult<()> {
        let demuxer = self
            .demuxer
            .as_mut()
            .ok_or_else(|| from_container_error("Demuxer is closed"))?;

        self.runtime
            .block_on(async { demuxer.probe().await })
            .map_err(crate::error::from_oxi_error)?;

        Ok(())
    }

    /// Read the next packet.
    ///
    /// Raises `StopIteration` when there are no more packets.
    fn read_packet(&mut self) -> PyOxiResult<Packet> {
        let demuxer = self
            .demuxer
            .as_mut()
            .ok_or_else(|| from_container_error("Demuxer is closed"))?;

        let packet = self.runtime.block_on(async { demuxer.read_packet().await });

        match packet {
            Ok(pkt) => Ok(Packet::from_rust(pkt)),
            Err(oximedia_core::OxiError::Eof) => Err(PyErr::new::<
                pyo3::exceptions::PyStopIteration,
                _,
            >("End of file")),
            Err(e) => Err(crate::error::from_oxi_error(e)),
        }
    }

    /// Get information about all streams.
    fn streams(&self) -> PyOxiResult<Vec<StreamInfo>> {
        let demuxer = self
            .demuxer
            .as_ref()
            .ok_or_else(|| from_container_error("Demuxer is closed"))?;

        Ok(demuxer
            .streams()
            .iter()
            .cloned()
            .map(StreamInfo::from_rust)
            .collect())
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: close file handle.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        self.demuxer = None;
        Ok(false)
    }

    /// Close the demuxer explicitly.
    fn close(&mut self) {
        self.demuxer = None;
    }

    /// Iterator protocol: return self.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iterator protocol: read next packet or raise StopIteration.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Packet> {
        slf.read_packet()
    }

    fn __str__(&self) -> String {
        "OggDemuxer".to_string()
    }

    fn __repr__(&self) -> String {
        let stream_count = self.demuxer.as_ref().map_or(0, |d| d.streams().len());
        let closed = self.demuxer.is_none();
        format!("OggDemuxer(streams={stream_count}, closed={closed})")
    }
}

/// Matroska/WebM muxer.
///
/// Muxes compressed packets into Matroska (.mkv) or `WebM` (.webm) container format.
///
/// # Example
///
/// ```python
/// config = MuxerConfig(title="My Video")
/// muxer = MatroskaMuxer("output.mkv", config)
///
/// # Add streams
/// video_stream = StreamInfo(...)
/// audio_stream = StreamInfo(...)
/// muxer.add_stream(video_stream)
/// muxer.add_stream(audio_stream)
///
/// # Write header
/// muxer.write_header()
///
/// # Write packets
/// for packet in packets:
///     muxer.write_packet(packet)
///
/// # Finalize
/// muxer.write_trailer()
/// ```
#[pyclass]
pub struct MatroskaMuxer {
    muxer: Option<RustMatroskaMuxer<FileSource>>,
    runtime: Arc<Runtime>,
    config: RustMuxerConfig,
}

#[pymethods]
impl MatroskaMuxer {
    /// Create a new Matroska muxer.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `title` - Optional title metadata
    #[new]
    #[pyo3(signature = (path, title=None))]
    fn new(path: &str, title: Option<String>) -> PyOxiResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| from_container_error(&format!("Failed to create runtime: {e}")))?,
        );

        let mut config = RustMuxerConfig::new();
        if let Some(t) = title {
            config = config.with_title(t);
        }

        let path_owned = path.to_string();
        let sink = runtime
            .block_on(async move { FileSource::create(&path_owned).await })
            .map_err(crate::error::from_oxi_error)?;

        let muxer = RustMatroskaMuxer::new(sink, config.clone());

        Ok(Self {
            muxer: Some(muxer),
            runtime,
            config,
        })
    }

    /// Add a stream to the muxer.
    ///
    /// # Arguments
    ///
    /// * `stream_info` - Stream information
    ///
    /// Returns the assigned stream index.
    fn add_stream(&mut self, stream_info: StreamInfo) -> PyOxiResult<usize> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        muxer
            .add_stream(stream_info.into_rust())
            .map_err(crate::error::from_oxi_error)
    }

    /// Write the container header.
    ///
    /// Must be called after all streams are added and before writing packets.
    fn write_header(&mut self) -> PyOxiResult<()> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        self.runtime
            .block_on(async { muxer.write_header().await })
            .map_err(crate::error::from_oxi_error)
    }

    /// Write a packet to the container.
    ///
    /// # Arguments
    ///
    /// * `packet` - Packet to write
    fn write_packet(&mut self, packet: &Packet) -> PyOxiResult<()> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        self.runtime
            .block_on(async { muxer.write_packet(packet.inner()).await })
            .map_err(crate::error::from_oxi_error)
    }

    /// Write the container trailer and finalize the file.
    fn write_trailer(&mut self) -> PyOxiResult<()> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        self.runtime
            .block_on(async { muxer.write_trailer().await })
            .map_err(crate::error::from_oxi_error)
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: write trailer and close.
    ///
    /// Automatically calls `write_trailer()` before releasing resources if the
    /// muxer was still open (no exception propagation from trailer errors to
    /// avoid masking the original exception).
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        // Attempt to write trailer; ignore errors on context manager exit.
        let _ = self.write_trailer();
        self.muxer = None;
        Ok(false)
    }

    /// Close the muxer explicitly (writes trailer first).
    fn close(&mut self) -> PyOxiResult<()> {
        let result = self.write_trailer();
        self.muxer = None;
        result
    }

    fn __str__(&self) -> String {
        "MatroskaMuxer".to_string()
    }

    fn __repr__(&self) -> String {
        let closed = self.muxer.is_none();
        format!(
            "MatroskaMuxer(title={:?}, closed={closed})",
            self.config.title
        )
    }
}

/// Ogg muxer.
///
/// Muxes compressed packets into Ogg container format (.ogg, .opus, .oga).
///
/// # Example
///
/// ```python
/// muxer = OggMuxer("output.opus")
///
/// # Add stream
/// audio_stream = StreamInfo(...)
/// muxer.add_stream(audio_stream)
///
/// # Write header
/// muxer.write_header()
///
/// # Write packets
/// for packet in packets:
///     muxer.write_packet(packet)
///
/// # Finalize
/// muxer.write_trailer()
/// ```
#[pyclass]
pub struct OggMuxer {
    muxer: Option<RustOggMuxer<FileSource>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl OggMuxer {
    /// Create a new Ogg muxer.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    #[new]
    fn new(path: &str) -> PyOxiResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Runtime::new()
                .map_err(|e| from_container_error(&format!("Failed to create runtime: {e}")))?,
        );

        let config = RustMuxerConfig::new();

        let path_owned = path.to_string();
        let sink = runtime
            .block_on(async move { FileSource::create(&path_owned).await })
            .map_err(crate::error::from_oxi_error)?;

        let muxer = RustOggMuxer::new(sink, config);

        Ok(Self {
            muxer: Some(muxer),
            runtime,
        })
    }

    /// Add a stream to the muxer.
    ///
    /// # Arguments
    ///
    /// * `stream_info` - Stream information
    ///
    /// Returns the assigned stream index.
    fn add_stream(&mut self, stream_info: StreamInfo) -> PyOxiResult<usize> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        muxer
            .add_stream(stream_info.into_rust())
            .map_err(crate::error::from_oxi_error)
    }

    /// Write the container header.
    ///
    /// Must be called after all streams are added and before writing packets.
    fn write_header(&mut self) -> PyOxiResult<()> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        self.runtime
            .block_on(async { muxer.write_header().await })
            .map_err(crate::error::from_oxi_error)
    }

    /// Write a packet to the container.
    ///
    /// # Arguments
    ///
    /// * `packet` - Packet to write
    fn write_packet(&mut self, packet: &Packet) -> PyOxiResult<()> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        self.runtime
            .block_on(async { muxer.write_packet(packet.inner()).await })
            .map_err(crate::error::from_oxi_error)
    }

    /// Write the container trailer and finalize the file.
    fn write_trailer(&mut self) -> PyOxiResult<()> {
        let muxer = self
            .muxer
            .as_mut()
            .ok_or_else(|| from_container_error("Muxer is closed"))?;

        self.runtime
            .block_on(async { muxer.write_trailer().await })
            .map_err(crate::error::from_oxi_error)
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: write trailer and close.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.write_trailer();
        self.muxer = None;
        Ok(false)
    }

    /// Close the muxer explicitly.
    fn close(&mut self) -> PyOxiResult<()> {
        let result = self.write_trailer();
        self.muxer = None;
        result
    }

    fn __str__(&self) -> String {
        "OggMuxer".to_string()
    }

    fn __repr__(&self) -> String {
        let closed = self.muxer.is_none();
        format!("OggMuxer(closed={closed})")
    }
}
