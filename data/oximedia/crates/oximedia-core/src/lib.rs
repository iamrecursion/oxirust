//! Core types and traits for `OxiMedia`.
//!
//! `oximedia-core` provides the foundational types and traits used throughout
//! the `OxiMedia` multimedia framework. This includes:
//!
//! - **Types**: Rational numbers, timestamps, pixel/sample formats, codec IDs, FourCC
//! - **Traits**: Decoder and demuxer interfaces
//! - **Error handling**: Unified error type with patent violation detection
//! - **Memory management**: Buffer pools for zero-copy operations
//! - **HDR support**: HDR metadata, transfer functions, color primaries, and conversions
//!
//! # Patent-Free Codec Philosophy
//!
//! OxiMedia is committed to a **Green List** policy: only patent-free, royalty-free
//! codecs are supported.  This design decision is principled and deliberate:
//!
//! - **Legal safety**: Patent-encumbered codecs such as H.264, H.265 (HEVC), and AAC
//!   require licence fees from organisations like MPEG LA, Via LA, and Dolby.  Using
//!   them without a licence exposes distributors to significant legal risk.
//! - **Open-source compatibility**: Copyleft (GPL/AGPL) projects cannot legally bundle
//!   patent-restricted code; the Green List ensures frictionless composition.
//! - **Long-term viability**: Patent-free codecs (AV1, VP9, Opus, FLAC, etc.) are
//!   backed by major industry consortia and carry no royalty obligations now or in
//!   the future.
//!
//! ## Green List — supported codecs
//!
//! | Category | Codec | Notes |
//! |----------|-------|-------|
//! | Video | AV1 | AOMedia royalty-free |
//! | Video | VP9 / VP8 | Google/WebM royalty-free |
//! | Video | Theora | Xiph.org royalty-free |
//! | Video | MJPEG | Baseline JPEG patents expired |
//! | Video | ProRes 422 | Decoding is patent-unencumbered |
//! | Video | DNxHD / VC-3 | Decoding is patent-unencumbered |
//! | Video | MPEG-2 | Core patents expired Feb 2023 |
//! | Video | FFV1 | RFC 9043, lossless |
//! | Image | JPEG 2000 | All patents expired 2010 |
//! | Image | JPEG-LS | HP patents expired 2017–2019 |
//! | Image | JPEG XS | Patent-free broadcast codec |
//! | Image | JPEG-XL | ISO/IEC 18181 royalty-free |
//! | Image | WebP / PNG / GIF | Widely royalty-free |
//! | Audio | Opus | IETF/Xiph.org royalty-free |
//! | Audio | Vorbis | Xiph.org royalty-free |
//! | Audio | FLAC | Xiph.org lossless royalty-free |
//! | Audio | ALAC | Apple Apache-2.0 royalty-free |
//! | Audio | MP3 | MPEG-1/2 Layer III patents expired 2017 |
//! | Audio | PCM | Uncompressed, always royalty-free |
//! | Subtitle | WebVTT / ASS / SRT | Open subtitle formats |
//!
//! Attempting to use patent-encumbered codecs (H.264, H.265, AAC, AC-3, DTS, etc.)
//! will result in an [`OxiError::PatentViolation`]
//! error, which carries the codec name for diagnostics.  Parsing those names via
//! [`CodecId::from_str`](types::CodecId) returns `Err` immediately.
//!
//! ```
//! use oximedia_core::error::OxiError;
//!
//! // Directly creating a PatentViolation (e.g. when a container demuxer
//! // encounters an H.264 track):
//! let err = OxiError::patent_violation("H.264/AVC");
//! assert!(err.is_patent_violation());
//! ```
//!
//! ## Type System
//!
//! Key types and their purposes:
//!
//! | Type | Module | Description |
//! |------|--------|-------------|
//! | `Rational` | `types` | Exact rational number (numerator/denominator) |
//! | `Timestamp` | `types` | Media timestamp with configurable timebase |
//! | `PixelFormat` | `types` | Video pixel layout — planar YUV, packed RGB, hardware interop |
//! | `SampleFormat` | `types` | Audio sample encoding — integer, float, planar |
//! | `CodecId` | `types` | Green-list codec identifier with `FromStr` / `Display` |
//! | `MediaType` | `types` | Stream classification (Video / Audio / Subtitle / Data) |
//! | [`FourCc`](types::fourcc::FourCc) | `types::fourcc` | Typed `[u8; 4]` FourCC value with 30+ named constants |
//!
//! ### Hardware-Interop Pixel Formats
//!
//! `PixelFormat` includes semi-planar formats commonly output by hardware decoders:
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `Nv12` | Y + interleaved UV, 8-bit (most common HW decoder output) |
//! | `Nv21` | Y + interleaved VU, 8-bit (Android camera native order) |
//! | `P010` | Y + interleaved UV, 10-bit LE in 16-bit words (10-bit HDR HW) |
//! | `P016` | Y + interleaved UV, 16-bit LE (full 16-bit HW precision) |
//!
//! ### `CodecId` String Parsing
//!
//! `CodecId` implements `std::str::FromStr`, enabling ergonomic
//! configuration from string input. Parsing is case-insensitive and accepts several aliases:
//!
//! ```
//! use oximedia_core::types::CodecId;
//!
//! let a: CodecId = "av1".parse().expect("valid");
//! let b: CodecId = "WEBP".parse().expect("valid");
//! let c: CodecId = "jxl".parse().expect("valid alias for JpegXl");
//! let d: CodecId = "gif".parse().expect("valid");
//!
//! assert_eq!(a, CodecId::Av1);
//! assert_eq!(b, CodecId::WebP);
//! assert_eq!(c, CodecId::JpegXl);
//! assert_eq!(d, CodecId::Gif);
//! assert_eq!(CodecId::JpegXl.canonical_name(), "jpegxl");
//! ```
//!
//! ### FourCC Constants
//!
//! [`FourCc`](types::fourcc::FourCc) is a `repr(transparent)` value type for FourCC codes.
//! It is `Copy`, hashable, `const`-constructible, and implements `FromStr` / `Display`.
//! Thirty named constants cover video codecs, audio codecs, ISOBMFF box types, and RIFF/AVI:
//!
//! ```
//! use oximedia_core::types::fourcc::{FourCc, AVC1, AV01, MP4A, FTYP, MOOV};
//!
//! // Named constants
//! assert_eq!(AV01.as_bytes(), b"av01");
//! assert_eq!(MP4A.as_bytes(), b"mp4a");
//! assert_eq!(FTYP.as_bytes(), b"ftyp");
//!
//! // Parse from a string
//! let code: FourCc = "av01".parse().expect("4-byte string");
//! assert_eq!(code, AV01);
//! ```
//!
//! ### Color Metadata — HDR10 and Beyond
//!
//! The `color_metadata` module exposes four compact enums that encode the ITU-T H.273 / ISO
//! 23001-8 color description parameters stored in video bitstreams and container headers.
//! Every variant has a `const fn to_u8()` / `from_u8()` pair for zero-overhead codec
//! integration.
//!
//! | Enum | Role |
//! |------|------|
//! | `ColorPrimaries` | Primary chromaticities (BT.709, BT.2020, P3-D65, …) |
//! | `TransferCharacteristics` | Electro-optical transfer function (SDR, PQ, HLG, …) |
//! | `MatrixCoefficients` | YCbCr derivation matrix (BT.601, BT.2020 NCL/CL, ICtCp, …) |
//! | `ColorRange` | Luma/chroma quantisation range (`Limited` / `Full`) |
//!
//! `TransferCharacteristics::is_hdr()` returns `true` for the four HDR transfer functions:
//! `Smpte2084` (PQ), `Hlg`, `Bt2020_10`, and `Bt2020_12`.
//!
//! HDR10 stream colour descriptor:
//!
//! ```
//! use oximedia_core::color_metadata::{ColorPrimaries, TransferCharacteristics,
//!                                     MatrixCoefficients, ColorRange};
//!
//! let primaries    = ColorPrimaries::Bt2020;
//! let transfer     = TransferCharacteristics::Smpte2084;   // PQ — HDR10
//! let matrix       = MatrixCoefficients::Bt2020Ncl;
//! let range        = ColorRange::Limited;
//!
//! assert!(transfer.is_hdr());
//! assert_eq!(primaries.to_u8(), 9);
//! assert_eq!(ColorPrimaries::from_u8(9), ColorPrimaries::Bt2020);
//! ```
//!
//! ### `Timestamp` Arithmetic
//!
//! Three convenience methods simplify common timestamp manipulations while preserving
//! the timebase and clamping to `[0, i64::MAX]`:
//!
//! | Method | Description |
//! |--------|-------------|
//! | `duration_add(Duration)` | Advance a timestamp by a wall-clock `Duration` |
//! | `duration_sub(Duration)` | Retreat a timestamp; clamps to 0 on underflow |
//! | `scale_by(num, den)` | Multiply PTS by a rational factor; identity when `den == 0` |
//!
//! ```
//! use oximedia_core::types::{Rational, Timestamp};
//! use std::time::Duration;
//!
//! let ts = Timestamp::new(1000, Rational::new(1, 1000)); // 1 second at 1 ms/tick
//!
//! // Advance by 500 ms
//! let later = ts.duration_add(Duration::from_millis(500));
//! assert_eq!(later.pts, 1500);
//!
//! // Retreat by 200 ms
//! let earlier = ts.duration_sub(Duration::from_millis(200));
//! assert_eq!(earlier.pts, 800);
//!
//! // Scale 3/2 (e.g., 1.5× playback → half the encoded PTS span)
//! let scaled = ts.scale_by(3, 2);
//! assert_eq!(scaled.pts, 1500);
//! ```
//!
//! ### Immersive Audio Channel Layouts
//!
//! `channel_layout::ChannelLayoutKind` enumerates all standard speaker arrangements,
//! including the three high-channel-count layouts added for immersive audio:
//!
//! | Variant | Channels | Speaker bed | Description |
//! |---------|----------|-------------|-------------|
//! | `Surround714` | 11 | 7.1 + 3 height | Dolby Atmos 7.1.4 |
//! | `Surround916` | 16 | 9.1 + 6 height | Auro-3D / Atmos 9.1.6 |
//! | `DolbyAtmosBed9_1_6` | 16 | 9.1 + 6 height | Dolby Atmos bed channel order |
//!
//! All three report `has_height_channels() == true` and expose `height_channel_count()`.
//!
//! ```ignore
//! use oximedia_core::channel_layout::ChannelLayoutKind;
//!
//! let layout = ChannelLayoutKind::Surround714;
//! assert_eq!(layout.channel_count(), 11);
//! assert_eq!(layout.height_channel_count(), 3);
//! assert!(layout.has_lfe());
//! assert_eq!(layout.name(), "7.1.4");
//!
//! let atmos_bed = ChannelLayoutKind::DolbyAtmosBed9_1_6;
//! assert_eq!(atmos_bed.channel_count(), 16);
//! assert_eq!(atmos_bed.name(), "9.1.6 Atmos Bed");
//! ```
//!
//! # Type Conversion Guide
//!
//! ## Timestamp ↔ Seconds
//!
//! [`Timestamp`] stores a PTS (presentation time stamp) as
//! an integer tick count paired with a [`Rational`] time-base
//! (seconds-per-tick).  Converting to and from wall-clock seconds:
//!
//! ```
//! use oximedia_core::types::{Rational, Timestamp};
//!
//! // --- From seconds to Timestamp ---
//! // Time-base 1/90000 (90 kHz MPEG clock):
//! let tb_90k = Rational::new(1, 90_000);
//! let ts = Timestamp::from_seconds(1.5, tb_90k);   // 1.5 seconds
//! assert_eq!(ts.pts, 135_000);                     // 1.5 × 90_000 = 135_000 ticks
//!
//! // --- From Timestamp to seconds ---
//! let ts2 = Timestamp::new(48_000, Rational::new(1, 48_000)); // 1 s at 48 kHz
//! let secs: f64 = ts2.to_seconds();
//! assert!((secs - 1.0).abs() < 1e-9);
//!
//! // --- Rescaling between time-bases ---
//! let tb_ms = Rational::new(1, 1_000);
//! let ts_ms = ts2.rescale(tb_ms);                  // 48 kHz → milliseconds
//! assert_eq!(ts_ms.pts, 1_000);                    // 1 s = 1000 ms
//! ```
//!
//! ## Rational ↔ f64
//!
//! [`Rational`] is an exact integer fraction reduced to
//! lowest terms on construction.  Convert to/from `f64` carefully — floating-
//! point cannot represent all rational values exactly.
//!
//! ```
//! use oximedia_core::types::Rational;
//!
//! // Construct from numerator/denominator (auto-reduces via GCD)
//! let r = Rational::new(3, 6);        // reduces to 1/2
//! assert_eq!(r.num, 1);              // public field: numerator
//! assert_eq!(r.den, 2);              // public field: denominator
//!
//! // Convert to f64 (lossy for non-dyadic fractions)
//! let f: f64 = r.to_f64();
//! assert!((f - 0.5).abs() < f64::EPSILON);
//!
//! // Frame-rate example: 30000/1001 (NTSC 29.97 fps)
//! let ntsc = Rational::new(30_000, 1_001);
//! let fps: f64 = ntsc.to_f64();
//! assert!((fps - 29.970_029_970_029_97).abs() < 1e-9);
//! ```
//!
//! ## Rational ↔ frame count
//!
//! When the time-base is `1/frame_rate`, `Timestamp.pts` equals the frame
//! number (zero-based).  This is the convention used by many container
//! parsers and codec APIs:
//!
//! ```
//! use oximedia_core::types::{Rational, Timestamp};
//!
//! // 24 fps time-base
//! let tb_24 = Rational::new(1, 24);
//! let frame_10 = Timestamp::new(10, tb_24);
//! let secs = frame_10.to_seconds();
//! assert!((secs - 10.0 / 24.0).abs() < 1e-9);
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_core::types::{Rational, Timestamp, PixelFormat, CodecId};
//! use oximedia_core::error::OxiResult;
//!
//! fn example() -> OxiResult<()> {
//!     // Create a timestamp at 1 second with millisecond precision
//!     let ts = Timestamp::new(1000, Rational::new(1, 1000));
//!     assert!((ts.to_seconds() - 1.0).abs() < f64::EPSILON);
//!
//!     // Check codec properties
//!     let codec = CodecId::Av1;
//!     assert!(codec.is_video());
//!
//!     // Check pixel format properties
//!     let format = PixelFormat::Yuv420p;
//!     assert!(format.is_planar());
//!     assert_eq!(format.plane_count(), 3);
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

pub mod alloc;
pub mod bitrate;
pub mod buffer_pool;
pub mod channel_layout;
pub mod codec_info;
pub mod codec_matrix;
pub mod codec_negotiation;
pub mod codec_params;
pub mod color_metadata;
pub mod convert;
pub mod downmix;
pub mod error;
pub mod error_context;
pub mod event_queue;
pub mod event_stream;
pub mod fourcc;
pub mod frame_info;
pub mod frame_pool;
pub mod frame_sharing;
pub mod hdr;
pub mod media_clock;
pub mod media_props;
pub mod media_segment;
pub mod media_time;
pub mod memory;
pub mod pixel_format;
pub mod pixel_format_color;
pub mod rational;
pub mod resource_handle;
pub mod ring_buffer;
pub mod sample_conv;
pub mod sample_format;
pub mod sync;
pub mod timestamp_arith;
pub mod traits;
pub mod type_registry;
pub mod types;
pub mod version;
pub mod work_queue;
pub mod work_queue_ws;
pub mod work_steal;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-export commonly used items at crate root
pub use codec_negotiation::{FormatConversionResult, FormatCost, FormatNegotiator};
pub use error::{OxiError, OxiResult};
pub use error_context::{ErrorContext, ErrorFrame, OxiErrorExt};
pub use types::{CodecId, MediaType, PixelFormat, Rational, SampleFormat, Timestamp};

/// Initialises the OxiMedia WASM module.
///
/// Sets up panic hooks for better error messages in the browser console.
/// Called from `oximedia-wasm` init function.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
}
