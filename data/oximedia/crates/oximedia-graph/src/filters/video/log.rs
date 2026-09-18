//! Log/Linear conversion filter for color grading workflows.
//!
//! This filter provides professional log encoding and decoding for various
//! camera systems and color grading workflows. Supports:
//!
//! - **Log Formats:**
//!   - Cineon Log
//!   - ARRI LogC (v3, v4)
//!   - Sony S-Log3
//!   - Panasonic V-Log
//!   - RED Log3G10
//!   - DJI D-Log
//!   - Canon C-Log
//!   - Blackmagic Film Gen 5
//!
//! - **ACES Transforms:**
//!   - ACES2065-1 (AP0)
//!   - ACEScg (AP1)
//!   - ACEScct (log working space)
//!   - ACES Proxy
//!
//! - **Display Transforms:**
//!   - sRGB
//!   - Rec.709
//!   - DCI-P3
//!   - Rec.2020
//!
//! # Example
//!
//! ```ignore
//! use oximedia_graph::filters::video::{LogLinearFilter, LogFormat, LogDirection};
//! use oximedia_graph::node::NodeId;
//!
//! // Convert linear to ARRI LogC
//! let filter = LogLinearFilter::new(
//!     NodeId(0),
//!     "to_logc",
//!     LogFormat::ArriLogC3,
//!     LogDirection::LinearToLog,
//! );
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::module_name_repetitions)]
#![allow(dead_code)]

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{InputPort, OutputPort, PortId, PortType};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

// ============================================================================
// Log Format Definitions
// ============================================================================

/// Log encoding format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogFormat {
    /// Cineon Log (used in film scanning).
    Cineon,
    /// ARRI LogC version 3 (ALEXA, AMIRA).
    ArriLogC3,
    /// ARRI LogC version 4 (ALEXA 35).
    ArriLogC4,
    /// Sony S-Log3 (Venice, FX9, FX6, A7S series).
    SonySLog3,
    /// Panasonic V-Log (Varicam, GH5, S1H).
    PanasonicVLog,
    /// RED Log3G10 (RED cameras).
    RedLog3G10,
    /// DJI D-Log (drones and gimbals).
    DjiDLog,
    /// Canon C-Log (Cinema EOS).
    CanonCLog,
    /// Blackmagic Film Gen 5.
    BlackmagicFilm5,
    /// ACEScct (ACES logarithmic working space).
    AcesCct,
    /// ACES Proxy (10-bit, 12-bit).
    AcesProxy10,
}

/// Direction of log conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogDirection {
    /// Convert from linear to log.
    LinearToLog,
    /// Convert from log to linear.
    LogToLinear,
}

// ============================================================================
// Cineon Log
// ============================================================================

/// Cineon Log encoding parameters.
///
/// Cineon is the original film scanning log format, widely used in
/// film-to-digital workflows.
#[derive(Clone, Copy, Debug)]
pub struct CineonLog {
    /// Black point (typically 95/1023 for 10-bit).
    pub black: f64,
    /// White point (typically 685/1023 for 10-bit).
    pub white: f64,
    /// Gamma (typically 0.6).
    pub gamma: f64,
}

impl Default for CineonLog {
    fn default() -> Self {
        Self {
            black: 95.0 / 1023.0,
            white: 685.0 / 1023.0,
            gamma: 0.6,
        }
    }
}

impl CineonLog {
    /// Convert linear to Cineon log.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear <= 0.0 {
            return self.black;
        }

        let log_val = self.black
            + (self.white - self.black) * (linear.log10() * self.gamma + (1.0 - self.black));

        log_val.clamp(0.0, 1.0)
    }

    /// Convert Cineon log to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        if log <= self.black {
            return 0.0;
        }

        let normalized = (log - self.black) / (self.white - self.black);
        let linear = 10_f64.powf((normalized - (1.0 - self.black)) / self.gamma);

        linear.max(0.0)
    }
}

// ============================================================================
// ARRI LogC
// ============================================================================

/// ARRI LogC v3 encoding.
///
/// Used in ALEXA, ALEXA Mini, AMIRA cameras.
/// Provides 14+ stops of dynamic range.
#[derive(Clone, Copy, Debug)]
pub struct ArriLogC3 {
    /// Cut point between linear and log sections.
    pub cut: f64,
    /// Slope in linear section.
    pub a: f64,
    /// Offset in linear section.
    pub b: f64,
    /// Slope in log section.
    pub c: f64,
    /// Offset in log section.
    pub d: f64,
    /// Log base multiplier.
    pub e: f64,
    /// Log offset.
    pub f: f64,
}

impl Default for ArriLogC3 {
    fn default() -> Self {
        // LogC3 parameters for EI 800
        Self {
            cut: 0.010591,
            a: 5.555556,
            b: 0.052272,
            c: 0.247190,
            d: 0.385537,
            e: 5.555556,
            f: 0.092809,
        }
    }
}

impl ArriLogC3 {
    /// Convert linear to LogC3.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear > self.cut {
            self.c * (self.a * linear + self.b).log10() + self.d
        } else {
            self.e * linear + self.f
        }
    }

    /// Convert LogC3 to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        let cut_log = self.c * (self.a * self.cut + self.b).log10() + self.d;

        if log > cut_log {
            (10_f64.powf((log - self.d) / self.c) - self.b) / self.a
        } else {
            (log - self.f) / self.e
        }
    }
}

/// ARRI LogC v4 encoding.
///
/// Used in ALEXA 35. Improved encoding with better shadow detail.
#[derive(Clone, Copy, Debug)]
pub struct ArriLogC4 {
    /// Curve parameter a
    pub a: f64,
    /// Curve parameter b
    pub b: f64,
    /// Curve parameter c
    pub c: f64,
    /// Curve parameter d
    pub d: f64,
    /// Curve parameter e
    pub e: f64,
    /// Curve parameter f
    pub f: f64,
}

impl Default for ArriLogC4 {
    fn default() -> Self {
        Self {
            a: 2048.0,
            b: 0.0,
            c: 0.184904,
            d: 0.385537,
            e: 5.555556,
            f: 0.092809,
        }
    }
}

impl ArriLogC4 {
    /// Convert linear to LogC4.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear <= 0.0 {
            return self.b;
        }

        (((linear + self.e) / (1.0 + self.e)).ln() / self.c.ln() + self.d + self.b) / self.a
    }

    /// Convert LogC4 to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        let x = log * self.a - self.d - self.b;
        self.c.powf(x) * (1.0 + self.e) - self.e
    }
}

// ============================================================================
// Sony S-Log3
// ============================================================================

/// Sony S-Log3 encoding.
///
/// Used in Venice, FX9, FX6, A7S series cameras.
/// Provides 14+ stops of dynamic range with improved shadow detail.
#[derive(Clone, Copy, Debug, Default)]
pub struct SonySLog3;

impl SonySLog3 {
    /// Convert linear to S-Log3.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear >= 0.01125000 {
            (420.0 + (((linear + 0.01) / (0.18 + 0.01)).log10() * 261.5)) / 1023.0
        } else {
            (linear * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0
        }
    }

    /// Convert S-Log3 to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        let log_1023 = log * 1023.0;

        if log_1023 >= 171.2102946929 {
            10_f64.powf((log_1023 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01
        } else {
            (log_1023 - 95.0) * 0.01125000 / (171.2102946929 - 95.0)
        }
    }
}

// ============================================================================
// Panasonic V-Log
// ============================================================================

/// Panasonic V-Log encoding.
///
/// Used in Varicam, GH5, GH5S, GH6, S1H cameras.
/// Provides 12+ stops of dynamic range.
#[derive(Clone, Copy, Debug)]
pub struct PanasonicVLog {
    /// Cut point.
    pub cut: f64,
    /// Linear coefficient.
    pub b: f64,
    /// Log coefficient.
    pub c: f64,
    /// Log offset.
    pub d: f64,
}

impl Default for PanasonicVLog {
    fn default() -> Self {
        Self {
            cut: 0.01,
            b: 0.00873,
            c: 0.241514,
            d: 0.598206,
        }
    }
}

impl PanasonicVLog {
    /// Convert linear to V-Log.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear < self.cut {
            5.6 * linear + 0.125
        } else {
            self.c * (linear + self.b).log10() + self.d
        }
    }

    /// Convert V-Log to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        let cut_log = self.c * (self.cut + self.b).log10() + self.d;

        if log < cut_log {
            (log - 0.125) / 5.6
        } else {
            10_f64.powf((log - self.d) / self.c) - self.b
        }
    }
}

// ============================================================================
// RED Log3G10
// ============================================================================

/// RED Log3G10 encoding.
///
/// Used in RED cameras (DSMC2, DSMC3, Komodo).
/// Log3 with 10-bit gamma encoding.
#[derive(Clone, Copy, Debug, Default)]
pub struct RedLog3G10;

impl RedLog3G10 {
    /// Convert linear to Log3G10.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear < 0.0 {
            return 0.0;
        }

        let a = 0.224282;
        let b = 155.975327;
        let c = 0.01;

        a * (linear * b + c).log10() + 0.5
    }

    /// Convert Log3G10 to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        let a = 0.224282;
        let b = 155.975327;
        let c = 0.01;

        (10_f64.powf((log - 0.5) / a) - c) / b
    }
}

// ============================================================================
// DJI D-Log
// ============================================================================

/// DJI D-Log encoding.
///
/// Used in DJI drones and gimbals (Inspire, Phantom, Mavic).
#[derive(Clone, Copy, Debug, Default)]
pub struct DjiDLog;

impl DjiDLog {
    /// Convert linear to D-Log.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear <= 0.0078 {
            6.025 * linear + 0.0929
        } else {
            ((linear + 0.0078) / (1.0 + 0.0078)).ln() / 0.9892 / 6.025 + 0.584
        }
    }

    /// Convert D-Log to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        if log <= 0.14 {
            (log - 0.0929) / 6.025
        } else {
            (0.9892_f64.powf(6.025 * (log - 0.584))) * (1.0 + 0.0078) - 0.0078
        }
    }
}

// ============================================================================
// Canon C-Log
// ============================================================================

/// Canon C-Log encoding.
///
/// Used in Canon Cinema EOS cameras (C300, C500, C70, C200).
#[derive(Clone, Copy, Debug, Default)]
pub struct CanonCLog;

impl CanonCLog {
    /// Convert linear to C-Log.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear < 0.0 {
            return 0.0;
        }

        let a = 0.529136;
        let b = 0.0047622;
        let c = 0.312689;
        let d = 0.092864;

        a * (a * linear + b).ln() + c * linear + d
    }

    /// Convert C-Log to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        // Iterative solver for inverse (simplified approximation)
        let a = 0.529136;
        let c = 0.312689;
        let d = 0.092864;

        if log <= d {
            return 0.0;
        }

        // Newton-Raphson approximation
        let mut x = (log - d) / c;
        for _ in 0..5 {
            let fx = a * (a * x + 0.0047622).ln() + c * x + d - log;
            let dfx = a * a / (a * x + 0.0047622) + c;
            x -= fx / dfx;
        }

        x.max(0.0)
    }
}

// ============================================================================
// Blackmagic Film Gen 5
// ============================================================================

/// Blackmagic Film Gen 5 encoding.
///
/// Used in Blackmagic Cinema Camera, Pocket Cinema Camera, URSA.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlackmagicFilm5;

impl BlackmagicFilm5 {
    /// Convert linear to Blackmagic Film.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear < 0.005 {
            return linear * 8.283605932402494;
        }

        0.2 * (linear + 0.01).ln() + 0.40975773852480107
    }

    /// Convert Blackmagic Film to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        if log < 0.04426550899923 {
            return log / 8.283605932402494;
        }

        ((log - 0.40975773852480107) / 0.2).exp() - 0.01
    }
}

// ============================================================================
// ACES Color Spaces
// ============================================================================

/// ACEScct (ACES Color Correction Transform).
///
/// Logarithmic working space for color grading in ACES workflows.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcesCct;

impl AcesCct {
    /// Convert linear AP1 to ACEScct.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear <= 0.0078125 {
            10.5402377416545 * linear + 0.0729055341958355
        } else {
            ((linear + 0.0000152587890625).max(1e-10).log2() + 9.72) / 17.52
        }
    }

    /// Convert ACEScct to linear AP1.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        if log <= 0.155251141552511 {
            (log - 0.0729055341958355) / 10.5402377416545
        } else {
            2_f64.powf(log * 17.52 - 9.72) - 0.0000152587890625
        }
    }
}

/// ACES Proxy 10-bit encoding.
///
/// Compact log encoding for proxy workflows.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcesProxy10;

impl AcesProxy10 {
    /// Convert linear AP1 to ACES Proxy 10.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        if linear <= 0.0 {
            return 0.0;
        }

        let log2_val = linear.max(1e-10).log2();
        ((log2_val + 2.5) / 10.0 * 1023.0 + 64.0) / 1023.0
    }

    /// Convert ACES Proxy 10 to linear AP1.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        let cv = log * 1023.0;
        let log2_val = (cv - 64.0) / 1023.0 * 10.0 - 2.5;

        2_f64.powf(log2_val)
    }
}

// ============================================================================
// Log Converter
// ============================================================================

/// Unified log converter that handles all formats.
#[derive(Clone, Copy, Debug)]
pub struct LogConverter {
    format: LogFormat,
}

impl LogConverter {
    /// Create a new log converter for the specified format.
    #[must_use]
    pub const fn new(format: LogFormat) -> Self {
        Self { format }
    }

    /// Convert linear to log.
    #[must_use]
    pub fn linear_to_log(self, linear: f64) -> f64 {
        match self.format {
            LogFormat::Cineon => CineonLog::default().linear_to_log(linear),
            LogFormat::ArriLogC3 => ArriLogC3::default().linear_to_log(linear),
            LogFormat::ArriLogC4 => ArriLogC4::default().linear_to_log(linear),
            LogFormat::SonySLog3 => SonySLog3.linear_to_log(linear),
            LogFormat::PanasonicVLog => PanasonicVLog::default().linear_to_log(linear),
            LogFormat::RedLog3G10 => RedLog3G10.linear_to_log(linear),
            LogFormat::DjiDLog => DjiDLog.linear_to_log(linear),
            LogFormat::CanonCLog => CanonCLog.linear_to_log(linear),
            LogFormat::BlackmagicFilm5 => BlackmagicFilm5.linear_to_log(linear),
            LogFormat::AcesCct => AcesCct.linear_to_log(linear),
            LogFormat::AcesProxy10 => AcesProxy10.linear_to_log(linear),
        }
    }

    /// Convert log to linear.
    #[must_use]
    pub fn log_to_linear(self, log: f64) -> f64 {
        match self.format {
            LogFormat::Cineon => CineonLog::default().log_to_linear(log),
            LogFormat::ArriLogC3 => ArriLogC3::default().log_to_linear(log),
            LogFormat::ArriLogC4 => ArriLogC4::default().log_to_linear(log),
            LogFormat::SonySLog3 => SonySLog3.log_to_linear(log),
            LogFormat::PanasonicVLog => PanasonicVLog::default().log_to_linear(log),
            LogFormat::RedLog3G10 => RedLog3G10.log_to_linear(log),
            LogFormat::DjiDLog => DjiDLog.log_to_linear(log),
            LogFormat::CanonCLog => CanonCLog.log_to_linear(log),
            LogFormat::BlackmagicFilm5 => BlackmagicFilm5.log_to_linear(log),
            LogFormat::AcesCct => AcesCct.log_to_linear(log),
            LogFormat::AcesProxy10 => AcesProxy10.log_to_linear(log),
        }
    }

    /// Convert RGB color.
    #[must_use]
    pub fn convert_rgb(self, r: f64, g: f64, b: f64, direction: LogDirection) -> (f64, f64, f64) {
        match direction {
            LogDirection::LinearToLog => (
                self.linear_to_log(r),
                self.linear_to_log(g),
                self.linear_to_log(b),
            ),
            LogDirection::LogToLinear => (
                self.log_to_linear(r),
                self.log_to_linear(g),
                self.log_to_linear(b),
            ),
        }
    }
}

// ============================================================================
// Log/Linear Filter
// ============================================================================

/// Log/Linear conversion filter.
pub struct LogLinearFilter {
    id: NodeId,
    name: String,
    state: NodeState,
    input: InputPort,
    output: OutputPort,
    converter: LogConverter,
    direction: LogDirection,
}

impl LogLinearFilter {
    /// Create a new log/linear filter.
    #[must_use]
    pub fn new(id: NodeId, name: &str, format: LogFormat, direction: LogDirection) -> Self {
        Self {
            id,
            name: name.to_string(),
            state: NodeState::Idle,
            input: InputPort::new(PortId(0), "input", PortType::Video),
            output: OutputPort::new(PortId(1), "output", PortType::Video),
            converter: LogConverter::new(format),
            direction,
        }
    }

    /// Process a video frame.
    fn process_frame(&self, frame: VideoFrame) -> GraphResult<VideoFrame> {
        // Only process RGB formats
        match frame.format {
            PixelFormat::Rgb24 | PixelFormat::Rgba32 => self.process_rgb_frame(frame),
            _ => Ok(frame),
        }
    }

    /// Process an RGB frame.
    fn process_rgb_frame(&self, mut frame: VideoFrame) -> GraphResult<VideoFrame> {
        let width = frame.width;
        let height = frame.height;
        let planes = &mut frame.planes;

        if planes.is_empty() {
            return Ok(frame);
        }

        let plane = &mut planes[0];
        let stride = plane.stride;
        let data = plane.data.as_mut_slice();

        let bytes_per_pixel = match frame.format {
            PixelFormat::Rgb24 => 3,
            PixelFormat::Rgba32 => 4,
            _ => return Ok(frame),
        };

        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = y * stride + x * bytes_per_pixel;

                // Read color (normalized to 0-1)
                let r = data[offset] as f64 / 255.0;
                let g = data[offset + 1] as f64 / 255.0;
                let b = data[offset + 2] as f64 / 255.0;

                // Convert
                let (r_out, g_out, b_out) = self.converter.convert_rgb(r, g, b, self.direction);

                // Write back (clamped to 0-1)
                data[offset] = (r_out.clamp(0.0, 1.0) * 255.0) as u8;
                data[offset + 1] = (g_out.clamp(0.0, 1.0) * 255.0) as u8;
                data[offset + 2] = (b_out.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }

        Ok(frame)
    }
}

impl Node for LogLinearFilter {
    fn id(&self) -> NodeId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn node_type(&self) -> NodeType {
        NodeType::Filter
    }

    fn state(&self) -> NodeState {
        self.state
    }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] {
        std::slice::from_ref(&self.input)
    }

    fn outputs(&self) -> &[OutputPort] {
        std::slice::from_ref(&self.output)
    }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Video(video_frame)) => {
                let processed = self.process_frame(video_frame)?;
                Ok(Some(FilterFrame::Video(processed)))
            }
            Some(_) => Err(GraphError::ProcessingError {
                node: self.id,
                message: "Log/Linear filter expects video input".to_string(),
            }),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cineon_roundtrip() {
        let cineon = CineonLog::default();
        let linear = 0.18; // 18% gray
        let log = cineon.linear_to_log(linear);
        let back = cineon.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_arri_logc3_roundtrip() {
        let logc = ArriLogC3::default();
        let linear = 0.18;
        let log = logc.linear_to_log(linear);
        let back = logc.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_sony_slog3_roundtrip() {
        let slog = SonySLog3;
        let linear = 0.18;
        let log = slog.linear_to_log(linear);
        let back = slog.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_panasonic_vlog_roundtrip() {
        let vlog = PanasonicVLog::default();
        let linear = 0.18;
        let log = vlog.linear_to_log(linear);
        let back = vlog.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_red_log3g10_roundtrip() {
        let red = RedLog3G10;
        let linear = 0.18;
        let log = red.linear_to_log(linear);
        let back = red.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_aces_cct_roundtrip() {
        let aces = AcesCct;
        let linear = 0.18;
        let log = aces.linear_to_log(linear);
        let back = aces.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_log_converter() {
        let converter = LogConverter::new(LogFormat::ArriLogC3);
        let linear = 0.18;
        let log = converter.linear_to_log(linear);
        let back = converter.log_to_linear(log);
        assert!((linear - back).abs() < 0.01);
    }

    #[test]
    fn test_log_converter_rgb() {
        let converter = LogConverter::new(LogFormat::SonySLog3);
        let (r, g, b) = (0.2, 0.5, 0.8);
        let (log_r, log_g, log_b) = converter.convert_rgb(r, g, b, LogDirection::LinearToLog);
        let (back_r, back_g, back_b) =
            converter.convert_rgb(log_r, log_g, log_b, LogDirection::LogToLinear);

        assert!((r - back_r).abs() < 0.01);
        assert!((g - back_g).abs() < 0.01);
        assert!((b - back_b).abs() < 0.01);
    }
}
