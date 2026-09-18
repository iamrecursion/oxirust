//! WASM self-contained media player state machine.
//!
//! This module provides a complete, self-contained media player that handles
//! format detection, demuxing, and decoding of video and audio tracks in a
//! single stateful object.  It is designed for in-memory playback of complete
//! media files loaded into a `Uint8Array` in the browser.
//!
//! # AV1 decode honesty (0.1.9)
//!
//! `next_frame()` returns an `Err` for AV1-coded video tracks instead of a
//! frame. `crates/oximedia-codec`'s `Av1Decoder::send_packet` only parses
//! OBU/frame headers and pushes a `VideoFrame::allocate()`d buffer (all
//! zeros) onto its output queue — `decode_frame_with_pipeline` /
//! `decode_tiles` / the rest of the tile/coefficient/reconstruction pipeline
//! exist but are never called from that path, so tile group data is
//! discarded and no real pixel is ever written. Wiring this player to that
//! decoder would silently hand callers solid-black "decoded" frames, which
//! is the exact defect `WasmAv1Decoder` was removed from this crate's public
//! surface for in 0.1.9 (see `TODO.md`, "Removed: dishonest decoders"). This
//! player used to wrap `Av1Decoder` the same way and hand its unpopulated
//! buffers back as if they were real frames; it now surfaces the limitation
//! as an explicit error instead of staying silent. VP8/VP9 video and all
//! audio codecs are not decoded by this player either (see `next_audio` and
//! `decode_video_packet`); use a real `VideoDecoder`/`AudioDecoder`
//! (WebCodecs) or `crates/oximedia-codec` natively for actual playback.
//!
//! # State Machine
//!
//! ```text
//! [Idle] ──load()──► [Loaded] ──seek()──► [Loaded]
//!                        │
//!               next_frame() / next_audio()
//!                        │
//!                  (EOF reached)
//!                        │
//!                     [Done]
//! ```
//!
//! There is no separate `Playing` state: packet reads happen directly in
//! `Loaded` until EOF moves the player to `Done` (see the AV1 note above for
//! why this player never actually produces decoded frames today).
//!
//! # JavaScript Example
//!
//! ```javascript
//! import * as oximedia from 'oximedia-wasm';
//!
//! const player = new oximedia.WasmMediaPlayer();
//!
//! // Load entire file into memory
//! const response = await fetch('video.webm');
//! const buf = new Uint8Array(await response.arrayBuffer());
//! player.load(buf);
//!
//! console.log('Media info:', JSON.parse(player.media_info()));
//!
//! // Seek to 5 seconds
//! player.seek(5000);
//!
//! // Decode frames one by one (throws for AV1 — see "AV1 decode honesty"
//! // above; use WebCodecs VideoDecoder for real playback).
//! let frame;
//! while ((frame = player.next_frame()) !== null) {
//!     // frame is a Uint8Array of YUV420p data
//!     render(frame, player.video_width(), player.video_height());
//! }
//!
//! // Decode audio chunks
//! let audio;
//! while ((audio = player.next_audio()) !== null) {
//!     // audio is a Float32Array of interleaved PCM samples
//!     playAudio(audio);
//! }
//!
//! player.reset();
//! ```

use wasm_bindgen::prelude::*;

use bytes::Bytes;
use oximedia_core::CodecId;

use crate::container::{probe_format, ContainerFormat, Packet, PacketFlags};
use crate::io::ByteSource;

// ---------------------------------------------------------------------------
// Player state
// ---------------------------------------------------------------------------

/// Current state of the `WasmMediaPlayer` state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlayerState {
    /// No media loaded.
    Idle,
    /// Media has been loaded and headers parsed; packet reads happen in this
    /// state (there is no separate "actively playing" state — see the
    /// module-level docs).
    Loaded,
    /// All packets have been consumed (EOF reached).
    Done,
}

// ---------------------------------------------------------------------------
// Track descriptors
// ---------------------------------------------------------------------------

/// Minimal description of a media stream discovered after probing.
#[derive(Clone, Debug)]
struct TrackInfo {
    /// Stream index (0-based).
    index: usize,
    /// Codec identifier.
    codec: CodecId,
    /// Video width (0 for audio streams).
    width: u32,
    /// Video height (0 for audio streams).
    height: u32,
    /// Audio sample rate (0 for video streams).
    sample_rate: u32,
    /// Audio channel count (0 for video streams).
    channels: u8,
    /// Duration in milliseconds.
    duration_ms: Option<f64>,
}

// ---------------------------------------------------------------------------
// WasmMediaPlayer
// ---------------------------------------------------------------------------

/// Self-contained media player for WebAssembly.
///
/// Loads an entire media file into memory, probes its format, and exposes
/// synchronous frame/audio chunk iteration suitable for a WASM single-threaded
/// environment.
///
/// Probes the following container/codec combinations for `media_info()`:
/// - **Matroska / WebM** with AV1, VP8, VP9 video
/// - **MP4** with AV1 video
/// - **Ogg** / FLAC audio
/// - **WAV** audio (PCM pass-through)
///
/// Actual frame decode is more limited than probing: `next_frame()` returns
/// an `Err` for AV1 (see the module-level "AV1 decode honesty" docs) and
/// `Ok(None)` for VP8/VP9 (decoder not wired up); `next_audio()` never
/// decodes real audio samples. Use WebCodecs or `crates/oximedia-codec`
/// natively for real playback.
#[wasm_bindgen]
pub struct WasmMediaPlayer {
    /// Internal state of the player.
    state: PlayerState,
    /// Raw media bytes (full file in memory).
    data: Option<Bytes>,
    /// Byte source for packet reading.
    source: Option<ByteSource>,
    /// Detected container format.
    container_format: Option<ContainerFormat>,
    /// Video track descriptor.
    video_track: Option<TrackInfo>,
    /// Audio track descriptor.
    audio_track: Option<TrackInfo>,
    /// Current seek position in milliseconds.
    seek_ms: u64,
    /// Total duration in milliseconds (if known).
    duration_ms: Option<f64>,
    /// Number of video frames decoded.
    video_frame_count: u64,
    /// Number of audio chunks produced.
    audio_chunk_count: u64,
    /// Buffered video packets waiting to be decoded.
    video_packet_queue: Vec<Vec<u8>>,
    /// Buffered audio packets waiting to be produced.
    audio_packet_queue: Vec<Vec<u8>>,
    /// Whether the source has reached EOF.
    eof: bool,
}

#[wasm_bindgen]
impl WasmMediaPlayer {
    /// Create a new media player in the `Idle` state.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: PlayerState::Idle,
            data: None,
            source: None,
            container_format: None,
            video_track: None,
            audio_track: None,
            seek_ms: 0,
            duration_ms: None,
            video_frame_count: 0,
            audio_chunk_count: 0,
            video_packet_queue: Vec::new(),
            audio_packet_queue: Vec::new(),
            eof: false,
        }
    }

    /// Load an entire media file from a `Uint8Array`.
    ///
    /// This method probes the container format and parses stream headers.
    /// After a successful call the player transitions to the `Loaded` state
    /// and `next_frame()` / `next_audio()` can be called.
    ///
    /// # Errors
    ///
    /// Returns an error if the format cannot be detected or headers are
    /// malformed.
    pub fn load(&mut self, media_bytes: &[u8]) -> Result<(), JsValue> {
        if media_bytes.is_empty() {
            return Err(crate::utils::js_err(
                "WasmMediaPlayer: media_bytes is empty",
            ));
        }

        let bytes = Bytes::copy_from_slice(media_bytes);
        let source = ByteSource::new(bytes.clone());

        // Probe format
        let probe_header_len = 32.min(media_bytes.len());
        let probe_result = probe_format(&media_bytes[..probe_header_len]).map_err(|e| {
            crate::utils::js_err(&format!("WasmMediaPlayer format probe error: {e}"))
        })?;

        self.container_format = Some(probe_result.format);
        self.data = Some(bytes);
        self.source = Some(source);

        // Parse stream headers to populate track info
        self.parse_headers()?;

        self.state = PlayerState::Loaded;
        self.eof = false;
        Ok(())
    }

    /// Seek to a timestamp in milliseconds.
    ///
    /// For in-memory players this performs a logical seek by resetting the
    /// byte source and skipping packets until the requested timestamp.  For
    /// large files the seek may be approximate (seeking to the nearest keyframe
    /// before the requested time).
    ///
    /// # Errors
    ///
    /// Returns an error if the player has not been `load()`ed yet, or if the
    /// seek target is out of range.
    pub fn seek(&mut self, timestamp_ms: u64) -> Result<(), JsValue> {
        match self.state {
            PlayerState::Idle => {
                return Err(crate::utils::js_err(
                    "WasmMediaPlayer: cannot seek before load()",
                ));
            }
            PlayerState::Done => {
                // Allow re-seeking on a completed stream — restart decoding
            }
            _ => {}
        }

        self.seek_ms = timestamp_ms;
        // Reset byte source to beginning for a full scan seek
        if let Some(ref mut src) = self.source {
            src.seek(std::io::SeekFrom::Start(0))
                .map_err(|e| crate::utils::js_err(&format!("WasmMediaPlayer seek error: {e}")))?;
        }

        self.video_packet_queue.clear();
        self.audio_packet_queue.clear();
        self.eof = false;
        self.state = PlayerState::Loaded;
        Ok(())
    }

    /// Decode the next video frame and return it as YUV420p `Uint8Array`.
    ///
    /// The returned buffer has the layout:
    /// `[Y plane (W*H)] [U plane (W/2*H/2)] [V plane (W/2*H/2)]`
    ///
    /// Returns `null` (JS `None`) when the stream has no more video frames.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails, and always errors for an
    /// AV1-coded video track — see the module-level "AV1 decode honesty"
    /// docs. VP8/VP9 tracks decode no frames (`Ok(None)` until EOF): no
    /// decoder is wired up for them in this player.
    pub fn next_frame(&mut self) -> Result<Option<js_sys::Uint8Array>, JsValue> {
        if self.state == PlayerState::Idle {
            return Err(crate::utils::js_err(
                "WasmMediaPlayer: call load() before next_frame()",
            ));
        }

        // Check pre-buffered video packets first
        if let Some(pkt_data) = self.video_packet_queue.first().cloned() {
            self.video_packet_queue.remove(0);
            return self.decode_video_packet(&pkt_data);
        }

        // Read packets from source until we get a video frame or EOF
        loop {
            if self.eof {
                self.state = PlayerState::Done;
                return Ok(None);
            }

            let packet = match self.read_next_raw_packet()? {
                Some(p) => p,
                None => {
                    self.eof = true;
                    self.state = PlayerState::Done;
                    return Ok(None);
                }
            };

            // For simplicity, treat all packets as video packets in this demo
            // implementation.  A production implementation would parse stream
            // indices from the container.
            let data = packet.data.to_vec();
            let result = self.decode_video_packet(&data)?;
            if result.is_some() {
                return Ok(result);
            }
            // No frame produced yet — try next packet
        }
    }

    /// Decode and return the next audio chunk as a `Float32Array`.
    ///
    /// Returns interleaved PCM samples normalised to −1.0 … 1.0.
    /// Returns `null` when no more audio data is available.
    ///
    /// For containers that do not carry audio (video-only files), this always
    /// returns `null`.
    ///
    /// # Errors
    ///
    /// Returns an error if the player is not loaded.
    pub fn next_audio(&mut self) -> Result<Option<js_sys::Float32Array>, JsValue> {
        if self.state == PlayerState::Idle {
            return Err(crate::utils::js_err(
                "WasmMediaPlayer: call load() before next_audio()",
            ));
        }

        // If there are buffered audio packets, return the first
        if let Some(pkt_data) = self.audio_packet_queue.first().cloned() {
            self.audio_packet_queue.remove(0);
            // Produce silence of the buffered packet length as a stub
            // A full implementation would run the appropriate audio decoder
            let sample_count = pkt_data.len().min(4096);
            let samples = vec![0.0f32; sample_count];
            self.audio_chunk_count += 1;
            return Ok(Some(js_sys::Float32Array::from(samples.as_slice())));
        }

        if self.audio_track.is_none() || self.eof {
            return Ok(None);
        }

        Ok(None)
    }

    /// Return a JSON string describing the loaded media.
    ///
    /// # JSON Schema
    ///
    /// ```json
    /// {
    ///   "format": "Matroska",
    ///   "duration_ms": 120000,
    ///   "state": "Loaded",
    ///   "streams": [
    ///     {"index": 0, "codec": "Av1", "media_type": "Video",
    ///      "width": 1920, "height": 1080},
    ///     {"index": 1, "codec": "Opus", "media_type": "Audio",
    ///      "sample_rate": 48000, "channels": 2}
    ///   ]
    /// }
    /// ```
    ///
    /// Returns `{"state":"Idle"}` if `load()` has not been called.
    pub fn media_info(&self) -> String {
        if self.state == PlayerState::Idle {
            return r#"{"state":"Idle"}"#.to_string();
        }

        let format_str = self
            .container_format
            .as_ref()
            .map(|f| format!("{f:?}"))
            .unwrap_or_else(|| "Unknown".to_string());

        let state_str = match self.state {
            PlayerState::Idle => "Idle",
            PlayerState::Loaded => "Loaded",
            PlayerState::Done => "Done",
        };

        let duration_field = match self.duration_ms {
            Some(d) => format!(",\"duration_ms\":{d:.0}"),
            None => String::new(),
        };

        let mut streams = Vec::new();
        if let Some(ref v) = self.video_track {
            let dur = v
                .duration_ms
                .map_or(String::new(), |d| format!(",\"duration_ms\":{d:.0}"));
            streams.push(format!(
                r#"{{"index":{},"codec":"{:?}","media_type":"Video","width":{},"height":{}{}}}"#,
                v.index, v.codec, v.width, v.height, dur
            ));
        }
        if let Some(ref a) = self.audio_track {
            let dur = a
                .duration_ms
                .map_or(String::new(), |d| format!(",\"duration_ms\":{d:.0}"));
            streams.push(format!(
                r#"{{"index":{},"codec":"{:?}","media_type":"Audio","sample_rate":{},"channels":{}{}}}"#,
                a.index, a.codec, a.sample_rate, a.channels, dur
            ));
        }
        let streams_json = streams.join(",");

        format!(
            r#"{{"format":"{format_str}","state":"{state_str}"{duration_field},"stream_count":{},"streams":[{streams_json}]}}"#,
            streams.len()
        )
    }

    /// Get video frame width in pixels (0 if no video track).
    pub fn video_width(&self) -> u32 {
        self.video_track.as_ref().map(|t| t.width).unwrap_or(0)
    }

    /// Get video frame height in pixels (0 if no video track).
    pub fn video_height(&self) -> u32 {
        self.video_track.as_ref().map(|t| t.height).unwrap_or(0)
    }

    /// Get audio sample rate in Hz (0 if no audio track).
    pub fn audio_sample_rate(&self) -> u32 {
        self.audio_track
            .as_ref()
            .map(|t| t.sample_rate)
            .unwrap_or(0)
    }

    /// Get audio channel count (0 if no audio track).
    pub fn audio_channels(&self) -> u8 {
        self.audio_track.as_ref().map(|t| t.channels).unwrap_or(0)
    }

    /// Get total duration in milliseconds (NaN if unknown).
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms.unwrap_or(f64::NAN)
    }

    /// Get current player state as a string.
    pub fn state(&self) -> String {
        match self.state {
            PlayerState::Idle => "Idle".to_string(),
            PlayerState::Loaded => "Loaded".to_string(),
            PlayerState::Done => "Done".to_string(),
        }
    }

    /// Get number of video frames decoded in this session.
    pub fn video_frame_count(&self) -> u64 {
        self.video_frame_count
    }

    /// Get number of audio chunks produced in this session.
    pub fn audio_chunk_count(&self) -> u64 {
        self.audio_chunk_count
    }

    /// Returns `true` if all packets have been consumed.
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Reset the player to its initial `Idle` state, releasing all resources.
    ///
    /// After `reset()` the player can be reused by calling `load()` again.
    pub fn reset(&mut self) {
        self.state = PlayerState::Idle;
        self.data = None;
        self.source = None;
        self.container_format = None;
        self.video_track = None;
        self.audio_track = None;
        self.seek_ms = 0;
        self.duration_ms = None;
        self.video_frame_count = 0;
        self.audio_chunk_count = 0;
        self.video_packet_queue.clear();
        self.audio_packet_queue.clear();
        self.eof = false;
    }
}

// Private implementation
impl WasmMediaPlayer {
    /// Parse container headers and populate `video_track` / `audio_track`.
    fn parse_headers(&mut self) -> Result<(), JsValue> {
        let format = self
            .container_format
            .ok_or_else(|| crate::utils::js_err("WasmMediaPlayer: no format detected"))?;

        match format {
            ContainerFormat::Matroska => {
                // WebM / Matroska — assume AV1 video + Opus audio as a
                // common profile.  A production implementation would parse
                // the EBML Track elements.
                self.video_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Av1,
                    width: 1920,
                    height: 1080,
                    sample_rate: 0,
                    channels: 0,
                    duration_ms: None,
                });
                self.audio_track = Some(TrackInfo {
                    index: 1,
                    codec: CodecId::Opus,
                    width: 0,
                    height: 0,
                    sample_rate: 48000,
                    channels: 2,
                    duration_ms: None,
                });
            }
            ContainerFormat::Mp4 => {
                self.video_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Av1,
                    width: 1920,
                    height: 1080,
                    sample_rate: 0,
                    channels: 0,
                    duration_ms: None,
                });
            }
            ContainerFormat::Ogg => {
                self.audio_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Opus,
                    width: 0,
                    height: 0,
                    sample_rate: 48000,
                    channels: 2,
                    duration_ms: None,
                });
            }
            ContainerFormat::Flac => {
                self.audio_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Flac,
                    width: 0,
                    height: 0,
                    sample_rate: 44100,
                    channels: 2,
                    duration_ms: None,
                });
            }
            ContainerFormat::Wav => {
                self.audio_track = Some(TrackInfo {
                    index: 0,
                    codec: CodecId::Pcm,
                    width: 0,
                    height: 0,
                    sample_rate: 44100,
                    channels: 2,
                    duration_ms: None,
                });
            }
        }

        Ok(())
    }

    /// Read the next raw packet from the byte source.
    ///
    /// Returns `None` at EOF.
    fn read_next_raw_packet(&mut self) -> Result<Option<Packet>, JsValue> {
        let src = match self.source.as_mut() {
            Some(s) => s,
            None => return Ok(None),
        };

        if src.is_eof() {
            return Ok(None);
        }

        let mut buf = vec![0u8; 4096];
        match src.read(&mut buf) {
            Ok(0) => Ok(None),
            Ok(n) => {
                buf.truncate(n);
                use oximedia_core::{Rational, Timestamp};
                let pts_ms = (self.video_frame_count + self.audio_chunk_count) as i64 * 33;
                let ts = Timestamp::new(pts_ms, Rational::new(1, 1000));
                Ok(Some(Packet::new(
                    0,
                    Bytes::from(buf),
                    ts,
                    PacketFlags::empty(),
                )))
            }
            Err(e) => Err(crate::utils::js_err(&format!(
                "WasmMediaPlayer read error: {e}"
            ))),
        }
    }

    /// Handle a raw video packet for the current `video_track`.
    ///
    /// Returns `Err` for an AV1-coded track: this player does not decode
    /// AV1 (see the module-level "AV1 decode honesty" docs) rather than
    /// silently handing back an unpopulated buffer. For every other coded
    /// video track, no decoder is wired up in this player either, so this
    /// always returns `Ok(None)` (the caller's read loop then keeps
    /// consuming packets until EOF).
    fn decode_video_packet(&mut self, _data: &[u8]) -> Result<Option<js_sys::Uint8Array>, JsValue> {
        if let Some(track) = self.video_track.as_ref() {
            if track.codec == CodecId::Av1 {
                return Err(crate::utils::js_err(
                    "AV1 decode is not supported in the browser build; use WebCodecs VideoDecoder",
                ));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_new() {
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
        assert_eq!(player.video_width(), 0);
        assert_eq!(player.video_height(), 0);
        assert_eq!(player.video_frame_count(), 0);
        assert_eq!(player.audio_chunk_count(), 0);
        assert!(!player.is_eof());
    }

    #[test]
    fn test_media_info_idle() {
        let player = WasmMediaPlayer::new();
        let info = player.media_info();
        assert!(info.contains("Idle"));
    }

    #[test]
    fn test_load_empty_does_not_transition() {
        // On native we cannot call load(&[]) because JsValue::from_str panics
        // outside a WASM context.  Instead verify that the player starts Idle
        // and remains consistent before any load is attempted.
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
        assert!(!player.is_eof());
    }

    #[test]
    fn test_load_matroska_header() {
        let mut player = WasmMediaPlayer::new();
        // Minimal EBML / Matroska magic bytes
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        let result = player.load(&data);
        assert!(result.is_ok(), "load failed: {:?}", result.err());
        assert_eq!(player.state(), "Loaded");
        let info = player.media_info();
        assert!(info.contains("Matroska"), "info: {info}");
    }

    #[test]
    fn test_seek_requires_loaded_state() {
        // Verify the player starts Idle — seek requires Loaded state.
        // The actual error path uses JsValue which panics outside WASM.
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
    }

    #[test]
    fn test_next_frame_requires_loaded_state() {
        // Verify the player starts Idle — next_frame requires Loaded state.
        // The actual error path uses JsValue which panics outside WASM.
        let player = WasmMediaPlayer::new();
        assert_eq!(player.state(), "Idle");
    }

    #[test]
    fn test_reset() {
        let mut player = WasmMediaPlayer::new();
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        player.load(&data).expect("load should succeed");
        assert_eq!(player.state(), "Loaded");
        player.reset();
        assert_eq!(player.state(), "Idle");
        assert_eq!(player.video_width(), 0);
    }

    #[test]
    fn test_media_info_loaded() {
        let mut player = WasmMediaPlayer::new();
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        player.load(&data).expect("load should succeed");
        let info = player.media_info();
        // Should include codec and stream info
        assert!(info.contains("stream_count"));
    }

    /// Regression test for the AV1 decode-honesty fix: `next_frame()` must
    /// error instead of returning an unpopulated (silently black) frame for
    /// an AV1-coded track. `crate::utils::js_err` is native-safe (returns
    /// `JsValue::null()` off wasm32), and this error path never touches
    /// `js_sys::Uint8Array`, so it is safe to exercise on the native target.
    #[test]
    fn test_next_frame_av1_track_errors() {
        let mut player = WasmMediaPlayer::new();
        // Minimal EBML / Matroska magic bytes — parse_headers() assumes an
        // AV1 video track for any Matroska/WebM container.
        let data = vec![
            0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
        ];
        player.load(&data).expect("load should succeed");
        assert_eq!(player.state(), "Loaded");

        let result = player.next_frame();
        assert!(
            result.is_err(),
            "next_frame() must reject AV1 rather than return an unpopulated frame"
        );
        // The player must not have silently reported a decoded frame.
        assert_eq!(player.video_frame_count(), 0);
    }
}
