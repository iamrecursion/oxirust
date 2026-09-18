//! Video codec bindings.

use crate::error::PyOxiResult;
use crate::types::{EncoderConfig, VideoFrame};
use oximedia_codec::{Av1Decoder as RustAv1Decoder, Av1Encoder as RustAv1Encoder};
use oximedia_codec::{VideoDecoder as RustVideoDecoder, VideoEncoder as RustVideoEncoder};
use oximedia_codec::{Vp8Decoder as RustVp8Decoder, Vp9Decoder as RustVp9Decoder};
use pyo3::prelude::*;

/// AV1 video decoder.
///
/// Decodes AV1 compressed video packets to raw video frames.
///
/// # Example
///
/// ```python
/// decoder = Av1Decoder()
/// decoder.send_packet(packet_data, pts=0)
/// frame = decoder.receive_frame()
/// if frame:
///     print(f"Decoded frame: {frame.width}x{frame.height}")
/// ```
#[pyclass]
pub struct Av1Decoder {
    inner: RustAv1Decoder,
}

#[pymethods]
impl Av1Decoder {
    /// Create a new AV1 decoder.
    #[new]
    fn new() -> PyOxiResult<Self> {
        let config = oximedia_codec::DecoderConfig::default();
        let inner = RustAv1Decoder::new(config).map_err(crate::error::from_codec_error)?;
        Ok(Self { inner })
    }

    /// Send a compressed packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed packet data
    /// * `pts` - Presentation timestamp
    fn send_packet(&mut self, py: Python<'_>, data: &[u8], pts: i64) -> PyOxiResult<()> {
        // Copy packet bytes so the closure becomes Send + 'static-friendly.
        let buf: Vec<u8> = data.to_vec();
        let inner = &mut self.inner;
        py.detach(move || inner.send_packet(&buf, pts))
            .map_err(crate::error::from_codec_error)
    }

    /// Receive a decoded frame.
    ///
    /// Returns `None` if more data is needed.
    fn receive_frame(&mut self, py: Python<'_>) -> PyOxiResult<Option<VideoFrame>> {
        let inner = &mut self.inner;
        py.detach(move || inner.receive_frame())
            .map(|opt| opt.map(VideoFrame::from_rust))
            .map_err(crate::error::from_codec_error)
    }

    /// Flush the decoder.
    ///
    /// Call after all packets have been sent to retrieve remaining frames.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_codec_error)
    }

    /// Reset the decoder state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get output dimensions (width, height) if known.
    fn dimensions(&self) -> Option<(u32, u32)> {
        self.inner.dimensions()
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: flush and reset the decoder.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.flush(py);
        self.reset();
        Ok(false)
    }

    fn __str__(&self) -> String {
        "Av1Decoder".to_string()
    }

    fn __repr__(&self) -> String {
        format!("Av1Decoder(dimensions={:?})", self.inner.dimensions())
    }
}

/// AV1 video encoder.
///
/// Encodes raw video frames to AV1 compressed packets.
///
/// # Example
///
/// ```python
/// config = EncoderConfig(width=1920, height=1080, framerate=(30, 1), crf=28.0)
/// encoder = Av1Encoder(config)
/// encoder.send_frame(frame)
/// packet = encoder.receive_packet()
/// if packet:
///     print(f"Encoded packet: {len(packet['data'])} bytes, keyframe={packet['keyframe']}")
/// ```
#[pyclass]
pub struct Av1Encoder {
    inner: RustAv1Encoder,
}

#[pymethods]
impl Av1Encoder {
    /// Create a new AV1 encoder.
    ///
    /// # Arguments
    ///
    /// * `config` - Encoder configuration
    #[new]
    fn new(config: EncoderConfig) -> PyOxiResult<Self> {
        let inner =
            RustAv1Encoder::new(config.inner().clone()).map_err(crate::error::from_codec_error)?;
        Ok(Self { inner })
    }

    /// Send a raw frame to the encoder.
    ///
    /// # Arguments
    ///
    /// * `frame` - Video frame to encode
    fn send_frame(&mut self, py: Python<'_>, frame: &VideoFrame) -> PyOxiResult<()> {
        let frame_inner = frame.inner();
        let inner = &mut self.inner;
        py.detach(move || inner.send_frame(frame_inner))
            .map_err(crate::error::from_codec_error)
    }

    /// Receive an encoded packet.
    ///
    /// Returns `None` if more frames are needed.
    ///
    /// Returns a dictionary with keys:
    /// - `data`: bytes - Compressed packet data
    /// - `pts`: int - Presentation timestamp
    /// - `dts`: int - Decode timestamp
    /// - `keyframe`: bool - Is this a keyframe
    /// - `duration`: Optional[int] - Duration in timebase units
    fn receive_packet(&mut self, py: Python<'_>) -> PyOxiResult<Option<Py<PyAny>>> {
        // Run the encode pull on a worker thread without holding the GIL.
        let inner = &mut self.inner;
        let packet = py
            .detach(move || inner.receive_packet())
            .map_err(crate::error::from_codec_error)?;

        match packet {
            Some(pkt) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("data", pyo3::types::PyBytes::new(py, &pkt.data))?;
                dict.set_item("pts", pkt.pts)?;
                dict.set_item("dts", pkt.dts)?;
                dict.set_item("keyframe", pkt.keyframe)?;
                dict.set_item("duration", pkt.duration)?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    /// Flush the encoder.
    ///
    /// Call after all frames have been sent to retrieve remaining packets.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_codec_error)
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: flush the encoder.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.flush(py);
        Ok(false)
    }

    fn __str__(&self) -> String {
        "Av1Encoder".to_string()
    }

    fn __repr__(&self) -> String {
        format!("Av1Encoder(config={:?})", self.inner.config())
    }
}

/// VP9 video decoder.
///
/// Decodes VP9 compressed video packets to raw video frames.
///
/// # Example
///
/// ```python
/// decoder = Vp9Decoder()
/// decoder.send_packet(packet_data, pts=0)
/// frame = decoder.receive_frame()
/// if frame:
///     print(f"Decoded frame: {frame.width}x{frame.height}")
/// ```
#[pyclass]
pub struct Vp9Decoder {
    inner: RustVp9Decoder,
}

#[pymethods]
impl Vp9Decoder {
    /// Create a new VP9 decoder.
    #[new]
    fn new() -> PyOxiResult<Self> {
        let config = oximedia_codec::DecoderConfig::default();
        let inner = RustVp9Decoder::new(config).map_err(crate::error::from_codec_error)?;
        Ok(Self { inner })
    }

    /// Send a compressed packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed packet data
    /// * `pts` - Presentation timestamp
    fn send_packet(&mut self, py: Python<'_>, data: &[u8], pts: i64) -> PyOxiResult<()> {
        let buf: Vec<u8> = data.to_vec();
        let inner = &mut self.inner;
        py.detach(move || inner.send_packet(&buf, pts))
            .map_err(crate::error::from_codec_error)
    }

    /// Receive a decoded frame.
    ///
    /// Returns `None` if more data is needed.
    fn receive_frame(&mut self, py: Python<'_>) -> PyOxiResult<Option<VideoFrame>> {
        let inner = &mut self.inner;
        py.detach(move || inner.receive_frame())
            .map(|opt| opt.map(VideoFrame::from_rust))
            .map_err(crate::error::from_codec_error)
    }

    /// Flush the decoder.
    ///
    /// Call after all packets have been sent to retrieve remaining frames.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_codec_error)
    }

    /// Reset the decoder state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get output dimensions (width, height) if known.
    fn dimensions(&self) -> Option<(u32, u32)> {
        self.inner.dimensions()
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: flush and reset.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.flush(py);
        self.reset();
        Ok(false)
    }

    fn __str__(&self) -> String {
        "Vp9Decoder".to_string()
    }

    fn __repr__(&self) -> String {
        format!("Vp9Decoder(dimensions={:?})", self.inner.dimensions())
    }
}

/// VP8 video decoder.
///
/// Decodes VP8 compressed video packets to raw video frames.
///
/// # Example
///
/// ```python
/// decoder = Vp8Decoder()
/// decoder.send_packet(packet_data, pts=0)
/// frame = decoder.receive_frame()
/// if frame:
///     print(f"Decoded frame: {frame.width}x{frame.height}")
/// ```
#[pyclass]
pub struct Vp8Decoder {
    inner: RustVp8Decoder,
}

#[pymethods]
impl Vp8Decoder {
    /// Create a new VP8 decoder.
    #[new]
    fn new() -> PyOxiResult<Self> {
        let config = oximedia_codec::DecoderConfig::default();
        let inner = RustVp8Decoder::new(config).map_err(crate::error::from_codec_error)?;
        Ok(Self { inner })
    }

    /// Send a compressed packet to the decoder.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed packet data
    /// * `pts` - Presentation timestamp
    fn send_packet(&mut self, py: Python<'_>, data: &[u8], pts: i64) -> PyOxiResult<()> {
        let buf: Vec<u8> = data.to_vec();
        let inner = &mut self.inner;
        py.detach(move || inner.send_packet(&buf, pts))
            .map_err(crate::error::from_codec_error)
    }

    /// Receive a decoded frame.
    ///
    /// Returns `None` if more data is needed.
    fn receive_frame(&mut self, py: Python<'_>) -> PyOxiResult<Option<VideoFrame>> {
        let inner = &mut self.inner;
        py.detach(move || inner.receive_frame())
            .map(|opt| opt.map(VideoFrame::from_rust))
            .map_err(crate::error::from_codec_error)
    }

    /// Flush the decoder.
    ///
    /// Call after all packets have been sent to retrieve remaining frames.
    fn flush(&mut self, py: Python<'_>) -> PyOxiResult<()> {
        let inner = &mut self.inner;
        py.detach(move || inner.flush())
            .map_err(crate::error::from_codec_error)
    }

    /// Reset the decoder state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get output dimensions (width, height) if known.
    fn dimensions(&self) -> Option<(u32, u32)> {
        self.inner.dimensions()
    }

    /// Context manager __enter__: return self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager __exit__: flush and reset.
    #[pyo3(signature = (_exc_type, _exc_val, _exc_tb))]
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let _ = self.flush(py);
        self.reset();
        Ok(false)
    }

    fn __str__(&self) -> String {
        "Vp8Decoder".to_string()
    }

    fn __repr__(&self) -> String {
        format!("Vp8Decoder(dimensions={:?})", self.inner.dimensions())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// Task-2 disposition: the `receive_frame` consumer sites above already
// propagate a decode `Err` cleanly. Each `receive_frame` body is
// `py.detach(...).map(...).map_err(crate::error::from_codec_error)`: the
// `.map` (which would construct a `VideoFrame`) only runs on the `Ok`
// branch, so an `Err` from the underlying `oximedia_codec` decoder short-
// circuits straight to `from_codec_error` and becomes a real Python
// `OxiMediaError` exception (see `crate::error::from_codec_error`). There is
// no `.unwrap()`/`.expect()`/panic anywhere in this file, and no code path
// that silently turns a decode `Err` into a blank/placeholder `VideoFrame`.
//
// The tests below exercise this with a real, already-implemented error
// condition (`CodecError::Eof`, returned by `receive_frame` once the decoder
// has been flushed and its output queue is empty) rather than depending on
// the timing of the separate in-flight change that makes unsupported AV1/VP9
// bitstream decode itself return `Err`. Whatever `CodecError` variant the
// codec crate raises, it flows through the identical `map_err` path
// exercised here.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn av1_decoder_eof_after_flush_is_clear_pyerr_not_panic() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut decoder = Av1Decoder::new().expect("decoder should construct");
            decoder.flush(py).expect("flush should succeed");
            // No packets were ever sent, so the output queue is empty; after
            // flush the decoder must report a real end-of-stream error
            // rather than silently returning None (which means "need more
            // data") or fabricating a blank frame.
            // `VideoFrame` does not implement `Debug`, so match explicitly
            // instead of `.expect_err(..)` (which requires `T: Debug`).
            match decoder.receive_frame(py) {
                Err(err) => {
                    let msg = err.to_string();
                    assert!(
                        !msg.is_empty(),
                        "decode error must carry an actionable message, not an empty string"
                    );
                }
                Ok(frame) => panic!(
                    "receive_frame after flush with an empty queue must return Err, not \
                     Ok({frame:?}) — a blank/placeholder frame must never masquerade as success",
                    frame = frame.is_some()
                ),
            }
        });
    }

    #[test]
    fn vp9_decoder_eof_after_flush_is_clear_pyerr_not_panic() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut decoder = Vp9Decoder::new().expect("decoder should construct");
            decoder.flush(py).expect("flush should succeed");
            let result = decoder.receive_frame(py);
            assert!(
                result.is_err(),
                "receive_frame after flush with an empty queue must return Err, not Ok(blank frame)"
            );
        });
    }

    #[test]
    fn vp8_decoder_eof_after_flush_is_clear_pyerr_not_panic() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut decoder = Vp8Decoder::new().expect("decoder should construct");
            decoder.flush(py).expect("flush should succeed");
            let result = decoder.receive_frame(py);
            assert!(
                result.is_err(),
                "receive_frame after flush with an empty queue must return Err, not Ok(blank frame)"
            );
        });
    }

    #[test]
    fn av1_decoder_no_data_no_flush_returns_none_not_err() {
        // Sanity check on the *other* branch: before flush, an empty queue
        // means "need more data" (Ok(None)), which is distinct from the real
        // end-of-stream Err exercised above.
        pyo3::Python::initialize();
        Python::attach(|py| {
            let mut decoder = Av1Decoder::new().expect("decoder should construct");
            let result = decoder.receive_frame(py);
            assert!(
                result.is_ok(),
                "no data, no flush => Ok(None), not an error"
            );
            assert!(result.expect("checked above").is_none());
        });
    }
}
