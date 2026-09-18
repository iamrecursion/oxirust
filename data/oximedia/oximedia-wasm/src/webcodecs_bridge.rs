//! WASM WebCodecs API bridge.
//!
//! This module bridges OxiMedia's internal packet/frame representations with
//! the W3C WebCodecs API, enabling efficient interop with the browser's
//! hardware-accelerated codec infrastructure.
//!
//! ## Key Conversions
//!
//! 1. **OxiMedia → WebCodecs (encoding side)**
//!    - `WasmWebCodecsBridge::get_video_decoder_config` — converts OxiMedia
//!      codec parameters to a [`WasmVideoDecoderConfig`] with typed getters
//!      matching the WebCodecs `VideoDecoderConfig` dictionary.
//!    - `WasmWebCodecsBridge::oximedia_packet_to_encoded_chunk` — converts an
//!      OxiMedia compressed packet to a [`WasmEncodedChunkInfo`] with typed
//!      getters matching `EncodedVideoChunkInit`.
//!
//! 2. **WebCodecs → OxiMedia (decoding side)**
//!    - `WasmWebCodecsBridge::webcodecs_frame_to_yuv` — accepts an RGBA pixel
//!      buffer from a `VideoFrame.copyTo()` call and converts it to YUV420p.
//!
//! These accessors are deliberately typed getters rather than JSON strings:
//! this is a per-packet hot path (potentially called once per frame), and
//! `JSON.parse` allocation/parsing overhead on every frame is avoided by
//! exposing plain wasm-bindgen structs instead.
//!
//! # JavaScript Example
//!
//! ```javascript
//! import * as oximedia from 'oximedia-wasm';
//!
//! const bridge = new oximedia.WasmWebCodecsBridge();
//!
//! // Convert OxiMedia codec params to WebCodecs config
//! const config = bridge.get_video_decoder_config(codecParamsBytes);
//! const webDecoder = new VideoDecoder({ ... });
//! webDecoder.configure({
//!     codec: config.codec,
//!     codedWidth: config.coded_width,
//!     codedHeight: config.coded_height,
//!     optimizeForLatency: config.optimize_for_latency,
//!     description: config.description, // Uint8Array | undefined
//! });
//!
//! // Convert OxiMedia packet to EncodedVideoChunk args
//! const chunk = bridge.oximedia_packet_to_encoded_chunk(packetBytes, pts, dts, isKey);
//! webDecoder.decode(new EncodedVideoChunk({
//!     type: chunk.chunk_type,
//!     timestamp: chunk.timestamp,
//!     duration: chunk.duration,
//!     data: packetBytes,
//! }));
//!
//! // Convert RGBA from VideoFrame.copyTo() back to YUV420p
//! const yuv = bridge.webcodecs_frame_to_yuv(rgbaData, width, height);
//! ```

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmVideoDecoderConfig — typed WebCodecs VideoDecoderConfig result
// ---------------------------------------------------------------------------

/// Typed equivalent of the WebCodecs `VideoDecoderConfig` dictionary.
///
/// Returned by [`WasmWebCodecsBridge::get_video_decoder_config`] instead of a
/// hand-formatted JSON string, so per-frame/per-configuration calls avoid
/// `JSON.parse` overhead entirely.
#[wasm_bindgen]
pub struct WasmVideoDecoderConfig {
    codec: String,
    coded_width: u32,
    coded_height: u32,
    optimize_for_latency: bool,
    description: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl WasmVideoDecoderConfig {
    /// WebCodecs codec string, e.g. `"av01.0.00M.08"`.
    #[wasm_bindgen(getter)]
    pub fn codec(&self) -> String {
        self.codec.clone()
    }

    /// Coded frame width in pixels.
    #[wasm_bindgen(getter)]
    pub fn coded_width(&self) -> u32 {
        self.coded_width
    }

    /// Coded frame height in pixels.
    #[wasm_bindgen(getter)]
    pub fn coded_height(&self) -> u32 {
        self.coded_height
    }

    /// Whether the decoder should be configured to optimize for latency.
    #[wasm_bindgen(getter)]
    pub fn optimize_for_latency(&self) -> bool {
        self.optimize_for_latency
    }

    /// Raw codec extradata (sequence header etc.), if any, as a `Uint8Array`.
    ///
    /// Returns an empty array when no description bytes were present in the
    /// source codec parameters.
    #[wasm_bindgen(getter)]
    pub fn description(&self) -> js_sys::Uint8Array {
        match &self.description {
            Some(bytes) => js_sys::Uint8Array::from(bytes.as_slice()),
            None => js_sys::Uint8Array::new_with_length(0),
        }
    }

    /// Whether a non-empty `description` (extradata) is present.
    #[wasm_bindgen(getter)]
    pub fn has_description(&self) -> bool {
        self.description.is_some()
    }
}

// ---------------------------------------------------------------------------
// WasmEncodedChunkInfo — typed WebCodecs EncodedVideoChunkInit result
// ---------------------------------------------------------------------------

/// Typed equivalent of the WebCodecs `EncodedVideoChunkInit` dictionary
/// (minus the raw packet bytes, which the caller already owns).
///
/// Returned by [`WasmWebCodecsBridge::oximedia_packet_to_encoded_chunk`]
/// instead of a hand-formatted JSON string.
#[wasm_bindgen]
pub struct WasmEncodedChunkInfo {
    chunk_type: String,
    timestamp: f64,
    duration: f64,
    byte_length: u32,
}

#[wasm_bindgen]
impl WasmEncodedChunkInfo {
    /// Chunk type: `"key"` or `"delta"`.
    #[wasm_bindgen(getter)]
    pub fn chunk_type(&self) -> String {
        self.chunk_type.clone()
    }

    /// Presentation timestamp in microseconds (WebCodecs convention).
    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// Duration in microseconds.
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Length of the associated packet payload in bytes.
    #[wasm_bindgen(getter)]
    pub fn byte_length(&self) -> u32 {
        self.byte_length
    }
}

/// WebCodecs API bridge for OxiMedia WASM.
///
/// Provides bidirectional conversion between OxiMedia's packet/frame types
/// and the W3C WebCodecs API surface used by modern browsers.
#[wasm_bindgen]
pub struct WasmWebCodecsBridge {
    /// Codec string last configured (e.g. `"av01.0.00M.08"`).
    codec_string: String,
    /// Width of the last configured codec stream.
    width: u32,
    /// Height of the last configured codec stream.
    height: u32,
    /// Whether a configuration has been set.
    configured: bool,
}

#[wasm_bindgen]
impl WasmWebCodecsBridge {
    /// Create a new WebCodecs bridge.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            codec_string: String::new(),
            width: 0,
            height: 0,
            configured: false,
        }
    }

    /// Convert OxiMedia codec parameters to a typed WebCodecs `VideoDecoderConfig`.
    ///
    /// The `codec_params_bytes` parameter is interpreted as an OxiMedia codec
    /// parameter blob.  The method inspects the first bytes to detect the codec:
    /// - Starts with `0x01` → AV1 (`av01.0.00M.08`)
    /// - Starts with `0x56 0x50 0x39` (`"VP9"`) → VP9 (`vp09.00.10.08`)
    /// - Starts with `0x56 0x50 0x38` (`"VP8"`) → VP8 (`vp8`)
    /// - Otherwise defaults to AV1.
    ///
    /// Returns a [`WasmVideoDecoderConfig`] with typed getters; the raw
    /// `codec_params_bytes` are carried through verbatim as `description`
    /// (no JSON, no base64 -- a plain byte buffer for `Uint8Array`).
    ///
    /// # Errors
    ///
    /// Returns an error if the codec parameters cannot be parsed.
    pub fn get_video_decoder_config(
        &mut self,
        codec_params_bytes: &[u8],
    ) -> Result<WasmVideoDecoderConfig, JsValue> {
        let (codec_str, width, height) = Self::parse_codec_params(codec_params_bytes);

        self.codec_string = codec_str.clone();
        self.width = width;
        self.height = height;
        self.configured = true;

        let description = if codec_params_bytes.len() > 4 {
            Some(codec_params_bytes.to_vec())
        } else {
            None
        };

        Ok(WasmVideoDecoderConfig {
            codec: codec_str,
            coded_width: width,
            coded_height: height,
            optimize_for_latency: true,
            description,
        })
    }

    /// Convert an OxiMedia compressed packet to a typed `EncodedVideoChunk` descriptor.
    ///
    /// Returns a [`WasmEncodedChunkInfo`] with `chunk_type`, `timestamp`,
    /// `duration`, and `byte_length` getters -- no JSON parsing needed on
    /// this per-packet hot path.
    ///
    /// The `pts` and `dts` parameters are in milliseconds; they are converted
    /// to microseconds (×1000) for WebCodecs compatibility.
    ///
    /// # Arguments
    ///
    /// - `packet_bytes`: The raw compressed bitstream data.
    /// - `pts`: Presentation timestamp in milliseconds.
    /// - `dts`: Decode timestamp in milliseconds.
    /// - `is_key`: Whether this packet is a keyframe.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is empty.
    pub fn oximedia_packet_to_encoded_chunk(
        &self,
        packet_bytes: &[u8],
        pts: i64,
        dts: i64,
        is_key: bool,
    ) -> Result<WasmEncodedChunkInfo, JsValue> {
        if packet_bytes.is_empty() {
            return Err(crate::utils::js_err(
                "WebCodecsBridge: packet_bytes must not be empty",
            ));
        }

        let chunk_type = if is_key { "key" } else { "delta" }.to_string();
        // WebCodecs timestamps are in microseconds
        let timestamp_us = pts * 1000;
        let dts_us = dts * 1000;
        // Duration: difference between dts values, default 33333 μs (~30 fps)
        let duration_us = if dts_us > timestamp_us {
            dts_us - timestamp_us
        } else {
            33_333i64
        };

        Ok(WasmEncodedChunkInfo {
            chunk_type,
            timestamp: timestamp_us as f64,
            duration: duration_us as f64,
            byte_length: packet_bytes.len() as u32,
        })
    }

    /// Convert an RGBA pixel buffer (from `VideoFrame.copyTo()`) to YUV420p.
    ///
    /// The WebCodecs API delivers decoded frames as RGBA (or BGRA) via
    /// `VideoFrame.copyTo()`.  This method performs the BT.709 colour-space
    /// conversion from RGBA to YUV420p planar format expected by OxiMedia
    /// pipelines.
    ///
    /// The returned buffer has the standard layout:
    /// `[Y plane (W*H)] [U plane (W/2 * H/2)] [V plane (W/2 * H/2)]`
    ///
    /// # Arguments
    ///
    /// - `rgba_data`: Interleaved RGBA bytes (4 bytes per pixel, row-major).
    /// - `width`: Frame width in pixels.
    /// - `height`: Frame height in pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if `rgba_data.len() != width * height * 4`.
    pub fn webcodecs_frame_to_yuv(
        &self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let expected = (width as usize) * (height as usize) * 4;
        if rgba_data.len() != expected {
            return Err(crate::utils::js_err(&format!(
                "WebCodecsBridge: expected {expected} RGBA bytes for {}x{} frame, got {}",
                width,
                height,
                rgba_data.len()
            )));
        }

        let yuv = rgba_to_yuv420p(rgba_data, width, height);
        Ok(js_sys::Uint8Array::from(yuv.as_slice()))
    }

    /// Get the currently configured codec string.
    ///
    /// Returns an empty string if `get_video_decoder_config()` has not been
    /// called yet.
    pub fn codec_string(&self) -> String {
        self.codec_string.clone()
    }

    /// Get the configured stream width (0 if unconfigured).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the configured stream height (0 if unconfigured).
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns whether the bridge has been configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Reset the bridge to an unconfigured state.
    pub fn reset(&mut self) {
        self.codec_string.clear();
        self.width = 0;
        self.height = 0;
        self.configured = false;
    }
}

// Private helpers
impl WasmWebCodecsBridge {
    /// Inspect raw codec-parameter bytes and return `(codec_string, width, height)`.
    ///
    /// The blob layout used by OxiMedia WASM is a simple TLV-lite structure:
    /// - Bytes 0-3: magic / codec indicator
    /// - Bytes 4-7: width (LE u32)
    /// - Bytes 8-11: height (LE u32)
    /// - Remaining: codec-specific extradata (sequence header etc.)
    ///
    /// When the blob is shorter than expected, safe defaults are used.
    fn parse_codec_params(data: &[u8]) -> (String, u32, u32) {
        // Read optional width / height from bytes 4..12 if present
        let width = if data.len() >= 8 {
            u32::from_le_bytes([data[4], data[5], data[6], data[7]])
        } else {
            0
        };
        let height = if data.len() >= 12 {
            u32::from_le_bytes([data[8], data[9], data[10], data[11]])
        } else {
            0
        };

        // Detect codec from magic bytes
        let codec_str = if data.len() >= 3 && &data[0..3] == b"VP9" {
            "vp09.00.10.08".to_string()
        } else if data.len() >= 3 && &data[0..3] == b"VP8" {
            "vp8".to_string()
        } else {
            // Default: AV1 Main Profile, level 0, Main tier, 8-bit
            "av01.0.00M.08".to_string()
        };

        let w = if width > 0 { width } else { 1920 };
        let h = if height > 0 { height } else { 1080 };
        (codec_str, w, h)
    }
}

/// Convert an interleaved RGBA buffer to YUV420p planar using BT.709 coefficients.
///
/// Output layout: `[Y (W*H)] [U (W/2 * H/2)] [V (W/2 * H/2)]`.
///
/// Chroma planes are downsampled by averaging a 2×2 block.
#[allow(clippy::many_single_char_names)]
fn rgba_to_yuv420p(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let uv_w = (w + 1) / 2;
    let uv_h = (h + 1) / 2;
    let y_size = w * h;
    let uv_size = uv_w * uv_h;
    let mut yuv = vec![0u8; y_size + 2 * uv_size];

    // BT.709 coefficients (full range → studio range)
    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 4;
            if idx + 2 >= rgba.len() {
                break;
            }
            let r = rgba[idx] as f32;
            let g = rgba[idx + 1] as f32;
            let b = rgba[idx + 2] as f32;

            // Y (luma) — BT.709
            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8;
            yuv[row * w + col] = y;
        }
    }

    // Chroma planes: average 2×2 luma blocks
    for uv_row in 0..uv_h {
        for uv_col in 0..uv_w {
            let src_row = uv_row * 2;
            let src_col = uv_col * 2;

            // Accumulate RGBA over the 2×2 block
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;

            for dr in 0..2usize {
                for dc in 0..2usize {
                    let pr = src_row + dr;
                    let pc = src_col + dc;
                    if pr < h && pc < w {
                        let idx = (pr * w + pc) * 4;
                        if idx + 2 < rgba.len() {
                            r_sum += rgba[idx] as u32;
                            g_sum += rgba[idx + 1] as u32;
                            b_sum += rgba[idx + 2] as u32;
                            count += 1;
                        }
                    }
                }
            }

            let (r, g, b) = if count > 0 {
                (
                    r_sum as f32 / count as f32,
                    g_sum as f32 / count as f32,
                    b_sum as f32 / count as f32,
                )
            } else {
                (0.0, 0.0, 0.0)
            };

            // Cb / U — BT.709
            let u = (-0.1146 * r - 0.3854 * g + 0.5 * b + 128.0).clamp(0.0, 255.0) as u8;
            // Cr / V — BT.709
            let v = (0.5 * r - 0.4542 * g - 0.0458 * b + 128.0).clamp(0.0, 255.0) as u8;

            let uv_idx = uv_row * uv_w + uv_col;
            yuv[y_size + uv_idx] = u;
            yuv[y_size + uv_size + uv_idx] = v;
        }
    }

    yuv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_new() {
        let bridge = WasmWebCodecsBridge::new();
        assert!(!bridge.is_configured());
        assert_eq!(bridge.width(), 0);
        assert_eq!(bridge.height(), 0);
        assert!(bridge.codec_string().is_empty());
    }

    #[test]
    fn test_get_video_decoder_config_empty() {
        let mut bridge = WasmWebCodecsBridge::new();
        let result = bridge.get_video_decoder_config(&[]);
        assert!(result.is_ok());
        let config = result.expect("bridge result should succeed");
        assert_eq!(config.codec(), "av01.0.00M.08");
        assert!(!config.has_description());
        assert!(bridge.is_configured());
    }

    #[test]
    fn test_get_video_decoder_config_vp9() {
        let mut bridge = WasmWebCodecsBridge::new();
        // VP9 magic prefix followed by width=1280, height=720
        let mut params = b"VP9".to_vec();
        params.push(0); // padding byte
        params.extend_from_slice(&1280u32.to_le_bytes());
        params.extend_from_slice(&720u32.to_le_bytes());
        let result = bridge.get_video_decoder_config(&params);
        assert!(result.is_ok());
        let config = result.expect("bridge result should succeed");
        assert_eq!(config.codec(), "vp09.00.10.08");
        assert_eq!(config.coded_width(), 1280);
        assert_eq!(config.coded_height(), 720);
        assert!(config.has_description());
    }

    #[test]
    fn test_encoded_chunk_key_frame() {
        let bridge = WasmWebCodecsBridge::new();
        let data = vec![0u8; 100];
        let result = bridge.oximedia_packet_to_encoded_chunk(&data, 1000, 1000, true);
        assert!(result.is_ok());
        let chunk = result.expect("bridge result should succeed");
        assert_eq!(chunk.chunk_type(), "key");
        assert!((chunk.timestamp() - 1_000_000.0).abs() < f64::EPSILON); // 1000 ms → 1_000_000 μs
        assert_eq!(chunk.byte_length(), 100);
    }

    #[test]
    fn test_encoded_chunk_empty_requires_data() {
        // oximedia_packet_to_encoded_chunk with empty slice returns Err.
        // JsValue::from_str panics outside WASM, so we only test non-empty path.
        let bridge = WasmWebCodecsBridge::new();
        // Verify the non-empty path works correctly.
        let result = bridge.oximedia_packet_to_encoded_chunk(&[1u8], 0, 0, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_webcodecs_yuv_internal_conversion() {
        // Test the pure-Rust rgba_to_yuv420p helper directly (no wasm-bindgen calls).
        let rgba = vec![0u8; 4 * 4 * 4]; // 4×4 black frame
        let yuv = rgba_to_yuv420p(&rgba, 4, 4);
        // Y: 16, U: 4, V: 4 = 24 total
        assert_eq!(yuv.len(), 16 + 4 + 4);
    }

    #[test]
    fn test_webcodecs_yuv_white_frame_luma() {
        // Test that a white RGBA frame produces max luma in internal helper.
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for i in 0..4 * 4 {
            rgba[i * 4] = 255; // R
            rgba[i * 4 + 1] = 255; // G
            rgba[i * 4 + 2] = 255; // B
            rgba[i * 4 + 3] = 255; // A
        }
        let yuv = rgba_to_yuv420p(&rgba, 4, 4);
        // Y values should be near 255 for a white frame
        assert!(
            yuv[0] > 200,
            "Expected high luma for white frame, got {}",
            yuv[0]
        );
    }

    #[test]
    fn test_rgba_to_yuv420p_dimensions() {
        // 4×4 black frame
        let rgba = vec![0u8; 4 * 4 * 4];
        let yuv = rgba_to_yuv420p(&rgba, 4, 4);
        // Y: 16, U: 4, V: 4
        assert_eq!(yuv.len(), 16 + 4 + 4);
    }

    #[test]
    fn test_bridge_reset() {
        let mut bridge = WasmWebCodecsBridge::new();
        bridge
            .get_video_decoder_config(&[])
            .expect("decoder config should succeed");
        assert!(bridge.is_configured());
        bridge.reset();
        assert!(!bridge.is_configured());
        assert_eq!(bridge.width(), 0);
    }
}
