//! Output control and HDCP enforcement for DRM playback.
//!
//! Manages output protection policies including HDCP levels, analog output
//! restrictions, screen-capture prevention, and resolution downgrades for
//! non-compliant outputs.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// HDCP
// ---------------------------------------------------------------------------

/// HDCP version requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HdcpVersion {
    /// No HDCP required.
    None,
    /// HDCP 1.x (legacy).
    V1,
    /// HDCP 2.0.
    V2_0,
    /// HDCP 2.1.
    V2_1,
    /// HDCP 2.2 (4K / UHD content).
    V2_2,
    /// HDCP 2.3.
    V2_3,
}

impl HdcpVersion {
    /// Returns a numeric level for ordering: higher means stricter.
    #[allow(clippy::cast_precision_loss)]
    pub fn level(self) -> u32 {
        match self {
            Self::None => 0,
            Self::V1 => 1,
            Self::V2_0 => 2,
            Self::V2_1 => 3,
            Self::V2_2 => 4,
            Self::V2_3 => 5,
        }
    }

    /// Returns `true` if this version satisfies the `required` version.
    pub fn satisfies(self, required: Self) -> bool {
        self.level() >= required.level()
    }
}

impl fmt::Display for HdcpVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::V1 => write!(f, "1.x"),
            Self::V2_0 => write!(f, "2.0"),
            Self::V2_1 => write!(f, "2.1"),
            Self::V2_2 => write!(f, "2.2"),
            Self::V2_3 => write!(f, "2.3"),
        }
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// Type of video output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputType {
    /// HDMI output.
    Hdmi,
    /// DisplayPort output.
    DisplayPort,
    /// Analog component (YPbPr) output.
    AnalogComponent,
    /// Analog composite (CVBS) output.
    AnalogComposite,
    /// Miracast / wireless display.
    Miracast,
    /// AirPlay screen mirroring.
    AirPlay,
    /// Built-in / internal display.
    Internal,
    /// Screen capture or recording software.
    ScreenCapture,
}

impl fmt::Display for OutputType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdmi => write!(f, "HDMI"),
            Self::DisplayPort => write!(f, "DisplayPort"),
            Self::AnalogComponent => write!(f, "Analog Component"),
            Self::AnalogComposite => write!(f, "Analog Composite"),
            Self::Miracast => write!(f, "Miracast"),
            Self::AirPlay => write!(f, "AirPlay"),
            Self::Internal => write!(f, "Internal"),
            Self::ScreenCapture => write!(f, "Screen Capture"),
        }
    }
}

// ---------------------------------------------------------------------------
// Output descriptor
// ---------------------------------------------------------------------------

/// Describes the capabilities and state of a connected display output.
#[derive(Debug, Clone)]
pub struct OutputDescriptor {
    /// A stable identifier for this output (e.g. EDID hash).
    pub id: String,
    /// The type of output.
    pub output_type: OutputType,
    /// The HDCP version supported by the output (if any).
    pub hdcp_version: HdcpVersion,
    /// Maximum resolution supported (width x height).
    pub max_resolution: (u32, u32),
    /// Whether the output is currently active / connected.
    pub active: bool,
}

impl OutputDescriptor {
    /// Create a new output descriptor.
    pub fn new(id: impl Into<String>, output_type: OutputType, hdcp_version: HdcpVersion) -> Self {
        Self {
            id: id.into(),
            output_type,
            hdcp_version,
            max_resolution: (3840, 2160),
            active: true,
        }
    }

    /// Builder: set maximum resolution.
    pub fn with_max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_resolution = (width, height);
        self
    }

    /// Builder: set active state.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Returns the total number of pixels in the maximum resolution.
    pub fn max_pixels(&self) -> u64 {
        u64::from(self.max_resolution.0) * u64::from(self.max_resolution.1)
    }
}

// ---------------------------------------------------------------------------
// Output policy
// ---------------------------------------------------------------------------

/// Action to take when an output does not meet protection requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputAction {
    /// Allow playback without restriction.
    Allow,
    /// Downgrade output resolution to the specified maximum.
    Downgrade(u32, u32),
    /// Block playback on this output entirely.
    Block,
    /// Allow but apply watermarking.
    AllowWithWatermark,
}

impl fmt::Display for OutputAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allow => write!(f, "allow"),
            Self::Downgrade(w, h) => write!(f, "downgrade to {w}x{h}"),
            Self::Block => write!(f, "block"),
            Self::AllowWithWatermark => write!(f, "allow+watermark"),
        }
    }
}

/// Policy that describes output protection requirements for a piece of
/// content.
#[derive(Debug, Clone)]
pub struct OutputPolicy {
    /// Minimum HDCP version required on digital outputs.
    pub min_hdcp: HdcpVersion,
    /// Whether analog outputs are allowed at all.
    pub allow_analog: bool,
    /// Whether screen capture / recording is allowed.
    pub allow_screen_capture: bool,
    /// Whether wireless outputs (Miracast, AirPlay) are allowed.
    pub allow_wireless: bool,
    /// Maximum resolution when output protection is downgraded.
    pub downgrade_resolution: (u32, u32),
    /// Per-output-type overrides (takes precedence over defaults).
    pub overrides: HashMap<OutputType, OutputAction>,
}

impl OutputPolicy {
    /// Create a new, permissive policy (everything allowed).
    pub fn permissive() -> Self {
        Self {
            min_hdcp: HdcpVersion::None,
            allow_analog: true,
            allow_screen_capture: true,
            allow_wireless: true,
            downgrade_resolution: (1920, 1080),
            overrides: HashMap::new(),
        }
    }

    /// Create a strict policy suitable for premium UHD content.
    pub fn strict_uhd() -> Self {
        Self {
            min_hdcp: HdcpVersion::V2_2,
            allow_analog: false,
            allow_screen_capture: false,
            allow_wireless: false,
            downgrade_resolution: (1920, 1080),
            overrides: HashMap::new(),
        }
    }

    /// Builder: set minimum HDCP version.
    pub fn with_min_hdcp(mut self, version: HdcpVersion) -> Self {
        self.min_hdcp = version;
        self
    }

    /// Builder: add a per-output override.
    pub fn with_override(mut self, output_type: OutputType, action: OutputAction) -> Self {
        self.overrides.insert(output_type, action);
        self
    }

    /// Evaluate the policy against a connected output, returning the action
    /// that should be taken.
    pub fn evaluate(&self, output: &OutputDescriptor) -> OutputAction {
        // Check per-type overrides first
        if let Some(action) = self.overrides.get(&output.output_type) {
            return *action;
        }

        match output.output_type {
            OutputType::ScreenCapture => {
                if self.allow_screen_capture {
                    OutputAction::Allow
                } else {
                    OutputAction::Block
                }
            }
            OutputType::AnalogComponent | OutputType::AnalogComposite => {
                if self.allow_analog {
                    OutputAction::Allow
                } else {
                    OutputAction::Block
                }
            }
            OutputType::Miracast | OutputType::AirPlay => {
                if self.allow_wireless {
                    OutputAction::Allow
                } else {
                    OutputAction::Block
                }
            }
            OutputType::Hdmi | OutputType::DisplayPort => {
                if output.hdcp_version.satisfies(self.min_hdcp) {
                    OutputAction::Allow
                } else {
                    OutputAction::Downgrade(
                        self.downgrade_resolution.0,
                        self.downgrade_resolution.1,
                    )
                }
            }
            OutputType::Internal => OutputAction::Allow,
        }
    }

    /// Evaluate the policy against all given outputs. Returns a map from
    /// output ID to the action to take.
    pub fn evaluate_all(&self, outputs: &[OutputDescriptor]) -> HashMap<String, OutputAction> {
        outputs
            .iter()
            .filter(|o| o.active)
            .map(|o| (o.id.clone(), self.evaluate(o)))
            .collect()
    }

    /// Returns `true` if every active output in `outputs` is allowed
    /// (no blocks or downgrades).
    pub fn all_allowed(&self, outputs: &[OutputDescriptor]) -> bool {
        self.evaluate_all(outputs)
            .values()
            .all(|a| *a == OutputAction::Allow)
    }

    /// Returns `true` if any active output would be blocked.
    pub fn any_blocked(&self, outputs: &[OutputDescriptor]) -> bool {
        self.evaluate_all(outputs)
            .values()
            .any(|a| *a == OutputAction::Block)
    }
}

impl Default for OutputPolicy {
    fn default() -> Self {
        Self::permissive()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hdmi_output(hdcp: HdcpVersion) -> OutputDescriptor {
        OutputDescriptor::new("hdmi-1", OutputType::Hdmi, hdcp)
    }

    #[test]
    fn test_hdcp_version_satisfies_same() {
        assert!(HdcpVersion::V2_2.satisfies(HdcpVersion::V2_2));
    }

    #[test]
    fn test_hdcp_version_satisfies_higher() {
        assert!(HdcpVersion::V2_3.satisfies(HdcpVersion::V2_2));
    }

    #[test]
    fn test_hdcp_version_not_satisfies_lower() {
        assert!(!HdcpVersion::V1.satisfies(HdcpVersion::V2_2));
    }

    #[test]
    fn test_hdcp_none_satisfies_none() {
        assert!(HdcpVersion::None.satisfies(HdcpVersion::None));
    }

    #[test]
    fn test_hdcp_display() {
        assert_eq!(HdcpVersion::V2_2.to_string(), "2.2");
        assert_eq!(HdcpVersion::None.to_string(), "none");
    }

    #[test]
    fn test_output_descriptor_new() {
        let out = hdmi_output(HdcpVersion::V2_2);
        assert_eq!(out.output_type, OutputType::Hdmi);
        assert!(out.active);
    }

    #[test]
    fn test_output_descriptor_max_pixels() {
        let out = OutputDescriptor::new("dp-1", OutputType::DisplayPort, HdcpVersion::V2_0)
            .with_max_resolution(1920, 1080);
        assert_eq!(out.max_pixels(), 1920 * 1080);
    }

    #[test]
    fn test_permissive_policy_allows_everything() {
        let policy = OutputPolicy::permissive();
        let out = hdmi_output(HdcpVersion::None);
        assert_eq!(policy.evaluate(&out), OutputAction::Allow);
    }

    #[test]
    fn test_strict_policy_blocks_analog() {
        let policy = OutputPolicy::strict_uhd();
        let analog = OutputDescriptor::new("a1", OutputType::AnalogComponent, HdcpVersion::None);
        assert_eq!(policy.evaluate(&analog), OutputAction::Block);
    }

    #[test]
    fn test_strict_policy_blocks_screen_capture() {
        let policy = OutputPolicy::strict_uhd();
        let sc = OutputDescriptor::new("sc1", OutputType::ScreenCapture, HdcpVersion::None);
        assert_eq!(policy.evaluate(&sc), OutputAction::Block);
    }

    #[test]
    fn test_strict_policy_downgrades_low_hdcp() {
        let policy = OutputPolicy::strict_uhd();
        let out = hdmi_output(HdcpVersion::V1);
        let action = policy.evaluate(&out);
        assert!(matches!(action, OutputAction::Downgrade(_, _)));
    }

    #[test]
    fn test_strict_policy_allows_hdcp22() {
        let policy = OutputPolicy::strict_uhd();
        let out = hdmi_output(HdcpVersion::V2_2);
        assert_eq!(policy.evaluate(&out), OutputAction::Allow);
    }

    #[test]
    fn test_internal_always_allowed() {
        let policy = OutputPolicy::strict_uhd();
        let internal = OutputDescriptor::new("int", OutputType::Internal, HdcpVersion::None);
        assert_eq!(policy.evaluate(&internal), OutputAction::Allow);
    }

    #[test]
    fn test_override_takes_precedence() {
        let policy =
            OutputPolicy::permissive().with_override(OutputType::Hdmi, OutputAction::Block);
        let out = hdmi_output(HdcpVersion::V2_3);
        assert_eq!(policy.evaluate(&out), OutputAction::Block);
    }

    #[test]
    fn test_evaluate_all_filters_inactive() {
        let policy = OutputPolicy::permissive();
        let out1 = hdmi_output(HdcpVersion::V2_2);
        let out2 =
            OutputDescriptor::new("h2", OutputType::Hdmi, HdcpVersion::V1).with_active(false);
        let results = policy.evaluate_all(&[out1, out2]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_all_allowed_returns_true_when_permissive() {
        let policy = OutputPolicy::permissive();
        let outputs = vec![hdmi_output(HdcpVersion::V1)];
        assert!(policy.all_allowed(&outputs));
    }

    #[test]
    fn test_any_blocked_returns_true() {
        let policy = OutputPolicy::strict_uhd();
        let outputs = vec![
            hdmi_output(HdcpVersion::V2_2),
            OutputDescriptor::new("sc", OutputType::ScreenCapture, HdcpVersion::None),
        ];
        assert!(policy.any_blocked(&outputs));
    }

    #[test]
    fn test_output_action_display() {
        assert_eq!(OutputAction::Allow.to_string(), "allow");
        assert_eq!(OutputAction::Block.to_string(), "block");
        assert_eq!(
            OutputAction::Downgrade(1920, 1080).to_string(),
            "downgrade to 1920x1080"
        );
        assert_eq!(
            OutputAction::AllowWithWatermark.to_string(),
            "allow+watermark"
        );
    }

    #[test]
    fn test_output_type_display() {
        assert_eq!(OutputType::Hdmi.to_string(), "HDMI");
        assert_eq!(OutputType::AirPlay.to_string(), "AirPlay");
    }

    #[test]
    fn test_default_policy_is_permissive() {
        let policy = OutputPolicy::default();
        assert!(policy.allow_analog);
        assert!(policy.allow_screen_capture);
    }
}
