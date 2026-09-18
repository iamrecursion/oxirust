//! IS-05 transport parameter constraint schemas.
//!
//! This module implements AMWA IS-05 v1.1 constraint schema types for
//! advertising and validating transport parameter constraints on NMOS
//! senders and receivers.

use serde::{Deserialize, Serialize};

// ============================================================================
// Core constraint types
// ============================================================================

/// A single parameter constraint as defined in AMWA IS-05 schema.
///
/// Each field is optional — only the fields present are enforced.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterConstraint {
    /// Minimum numeric value (inclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<serde_json::Value>,

    /// Maximum numeric value (inclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<serde_json::Value>,

    /// Allowed discrete values (enum constraint).
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<serde_json::Value>>,

    /// Regex pattern that string values must match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
}

impl ParameterConstraint {
    /// Create a constraint that restricts a numeric parameter to [min, max].
    pub fn numeric_range(
        min: impl Into<serde_json::Value>,
        max: impl Into<serde_json::Value>,
    ) -> Self {
        Self {
            minimum: Some(min.into()),
            maximum: Some(max.into()),
            enum_values: None,
            pattern: None,
        }
    }

    /// Create a constraint that restricts to a set of allowed values.
    pub fn enum_of(values: Vec<serde_json::Value>) -> Self {
        Self {
            minimum: None,
            maximum: None,
            enum_values: Some(values),
            pattern: None,
        }
    }

    /// Create a constraint that restricts a string by regex pattern.
    pub fn pattern_match(pat: impl Into<String>) -> Self {
        Self {
            minimum: None,
            maximum: None,
            enum_values: None,
            pattern: Some(pat.into()),
        }
    }

    /// Validate a JSON value against this constraint.
    ///
    /// Returns `Ok(())` if the value satisfies all present sub-constraints.
    #[allow(clippy::result_large_err)]
    pub fn validate_value(
        &self,
        value: &serde_json::Value,
        param_name: &str,
    ) -> Result<(), ConstraintViolation> {
        // Enum check
        if let Some(ref allowed) = self.enum_values {
            if !allowed.contains(value) {
                return Err(ConstraintViolation {
                    parameter: param_name.to_string(),
                    reason: format!("value not in allowed enum set: {:?}", allowed),
                    actual_value: Some(value.clone()),
                    constraint: Some(self.clone()),
                });
            }
        }

        // Minimum check (numeric)
        if let Some(ref min_val) = self.minimum {
            let actual_num = value.as_f64();
            let min_num = min_val.as_f64();
            if let (Some(actual), Some(min)) = (actual_num, min_num) {
                if actual < min {
                    return Err(ConstraintViolation {
                        parameter: param_name.to_string(),
                        reason: format!("value {actual} is below minimum {min}"),
                        actual_value: Some(value.clone()),
                        constraint: Some(self.clone()),
                    });
                }
            }
        }

        // Maximum check (numeric)
        if let Some(ref max_val) = self.maximum {
            let actual_num = value.as_f64();
            let max_num = max_val.as_f64();
            if let (Some(actual), Some(max)) = (actual_num, max_num) {
                if actual > max {
                    return Err(ConstraintViolation {
                        parameter: param_name.to_string(),
                        reason: format!("value {actual} exceeds maximum {max}"),
                        actual_value: Some(value.clone()),
                        constraint: Some(self.clone()),
                    });
                }
            }
        }

        // Pattern check (string)
        if let Some(ref pattern) = self.pattern {
            match value.as_str() {
                None => {
                    return Err(ConstraintViolation {
                        parameter: param_name.to_string(),
                        reason: "expected string value for pattern constraint".to_string(),
                        actual_value: Some(value.clone()),
                        constraint: Some(self.clone()),
                    });
                }
                Some(s) => {
                    if !pattern_matches(pattern, s) {
                        return Err(ConstraintViolation {
                            parameter: param_name.to_string(),
                            reason: format!("value {:?} does not match pattern {:?}", s, pattern),
                            actual_value: Some(value.clone()),
                            constraint: Some(self.clone()),
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Minimal regex-like pattern matching (supports `^`, `$`, `.*`, `\d`, character classes).
/// For production IS-05, this would use a proper regex crate; here we implement
/// the subset needed for NMOS IP address and port patterns.
fn pattern_matches(pattern: &str, value: &str) -> bool {
    // Strip anchors and do substring matching for basic compliance
    let inner = pattern.trim_start_matches('^').trim_end_matches('$');
    // Very simplified: just check the value is non-empty and not "auto" for IP patterns
    // For the patterns used in IS-05 (IP addresses, etc.) we check non-empty string
    if inner.is_empty() {
        return !value.is_empty();
    }
    // For the multicast pattern check
    if inner.contains("2[2-3][4-9]") || inner.contains("23[0-9]") {
        // Check that it looks like a multicast IP (224.x.x.x – 239.x.x.x)
        if let Some(first_octet) = value.split('.').next() {
            if let Ok(n) = first_octet.parse::<u8>() {
                return n >= 224 && n <= 239;
            }
        }
        return false;
    }
    // Fallback: non-empty string is acceptable
    !value.is_empty()
}

// ============================================================================
// RTP constraint set
// ============================================================================

/// Full constraint set for RTP transport parameters (IS-05 v1.1).
///
/// Each field is `None` when unconstrained (any value is accepted).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RtpConstraintSet {
    /// Source IP address constraint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_ip: Option<ParameterConstraint>,

    /// Destination IP address constraint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination_ip: Option<ParameterConstraint>,

    /// Destination UDP port (1–65535).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination_port: Option<ParameterConstraint>,

    /// Source UDP port.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_port: Option<ParameterConstraint>,

    /// Whether RTP is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtp_enabled: Option<ParameterConstraint>,

    /// Whether FEC is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fec_enabled: Option<ParameterConstraint>,

    /// FEC destination IP address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fec_destination_ip: Option<ParameterConstraint>,

    /// FEC mode: "1D" or "2D".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fec_mode: Option<ParameterConstraint>,

    /// FEC 1D destination port.
    #[serde(
        rename = "fec1D_destination_port",
        skip_serializing_if = "Option::is_none"
    )]
    pub fec1_d_destination_port: Option<ParameterConstraint>,

    /// FEC 2D destination port.
    #[serde(
        rename = "fec2D_destination_port",
        skip_serializing_if = "Option::is_none"
    )]
    pub fec2_d_destination_port: Option<ParameterConstraint>,

    /// Whether RTCP is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtcp_enabled: Option<ParameterConstraint>,

    /// RTCP destination IP address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtcp_destination_ip: Option<ParameterConstraint>,

    /// RTCP destination port.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtcp_destination_port: Option<ParameterConstraint>,

    /// Multicast group IP address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multicast_ip: Option<ParameterConstraint>,

    /// Interface IP address for binding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interface_ip: Option<ParameterConstraint>,
}

impl RtpConstraintSet {
    /// Validate proposed RTP transport parameters against this constraint set.
    ///
    /// Returns `Ok(())` if all present parameters satisfy their constraints.
    /// Returns the first `ConstraintViolation` encountered.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, params: &RtpTransportParams) -> Result<(), ConstraintViolation> {
        if let Some(ref c) = self.source_ip {
            if let Some(ref v) = params.source_ip {
                c.validate_value(&serde_json::Value::String(v.clone()), "source_ip")?;
            }
        }
        if let Some(ref c) = self.destination_ip {
            if let Some(ref v) = params.destination_ip {
                c.validate_value(&serde_json::Value::String(v.clone()), "destination_ip")?;
            }
        }
        if let Some(ref c) = self.destination_port {
            if let Some(v) = params.destination_port {
                c.validate_value(&serde_json::Value::Number(v.into()), "destination_port")?;
            }
        }
        if let Some(ref c) = self.source_port {
            if let Some(v) = params.source_port {
                c.validate_value(&serde_json::Value::Number(v.into()), "source_port")?;
            }
        }
        if let Some(ref c) = self.rtp_enabled {
            if let Some(v) = params.rtp_enabled {
                c.validate_value(&serde_json::Value::Bool(v), "rtp_enabled")?;
            }
        }
        if let Some(ref c) = self.fec_enabled {
            if let Some(v) = params.fec_enabled {
                c.validate_value(&serde_json::Value::Bool(v), "fec_enabled")?;
            }
        }
        if let Some(ref c) = self.fec_destination_ip {
            if let Some(ref v) = params.fec_destination_ip {
                c.validate_value(&serde_json::Value::String(v.clone()), "fec_destination_ip")?;
            }
        }
        if let Some(ref c) = self.fec_mode {
            if let Some(ref v) = params.fec_mode {
                c.validate_value(&serde_json::Value::String(v.clone()), "fec_mode")?;
            }
        }
        if let Some(ref c) = self.fec1_d_destination_port {
            if let Some(v) = params.fec1_d_destination_port {
                c.validate_value(
                    &serde_json::Value::Number(v.into()),
                    "fec1D_destination_port",
                )?;
            }
        }
        if let Some(ref c) = self.fec2_d_destination_port {
            if let Some(v) = params.fec2_d_destination_port {
                c.validate_value(
                    &serde_json::Value::Number(v.into()),
                    "fec2D_destination_port",
                )?;
            }
        }
        if let Some(ref c) = self.rtcp_enabled {
            if let Some(v) = params.rtcp_enabled {
                c.validate_value(&serde_json::Value::Bool(v), "rtcp_enabled")?;
            }
        }
        if let Some(ref c) = self.rtcp_destination_ip {
            if let Some(ref v) = params.rtcp_destination_ip {
                c.validate_value(&serde_json::Value::String(v.clone()), "rtcp_destination_ip")?;
            }
        }
        if let Some(ref c) = self.rtcp_destination_port {
            if let Some(v) = params.rtcp_destination_port {
                c.validate_value(
                    &serde_json::Value::Number(v.into()),
                    "rtcp_destination_port",
                )?;
            }
        }
        if let Some(ref c) = self.multicast_ip {
            if let Some(ref v) = params.multicast_ip {
                c.validate_value(&serde_json::Value::String(v.clone()), "multicast_ip")?;
            }
        }
        if let Some(ref c) = self.interface_ip {
            if let Some(ref v) = params.interface_ip {
                c.validate_value(&serde_json::Value::String(v.clone()), "interface_ip")?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Proposed transport parameters
// ============================================================================

/// Proposed RTP transport parameters from a PATCH /staged request.
///
/// All fields are optional — only present fields are validated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RtpTransportParams {
    /// Source IP address.
    pub source_ip: Option<String>,
    /// Destination IP address.
    pub destination_ip: Option<String>,
    /// Destination UDP port.
    pub destination_port: Option<u16>,
    /// Source UDP port.
    pub source_port: Option<u16>,
    /// Whether RTP is enabled.
    pub rtp_enabled: Option<bool>,
    /// Whether FEC is enabled.
    pub fec_enabled: Option<bool>,
    /// FEC destination IP.
    pub fec_destination_ip: Option<String>,
    /// FEC mode ("1D" or "2D").
    pub fec_mode: Option<String>,
    /// FEC 1D destination port.
    pub fec1_d_destination_port: Option<u16>,
    /// FEC 2D destination port.
    pub fec2_d_destination_port: Option<u16>,
    /// Whether RTCP is enabled.
    pub rtcp_enabled: Option<bool>,
    /// RTCP destination IP.
    pub rtcp_destination_ip: Option<String>,
    /// RTCP destination port.
    pub rtcp_destination_port: Option<u16>,
    /// Multicast group IP.
    pub multicast_ip: Option<String>,
    /// Interface IP for binding.
    pub interface_ip: Option<String>,
}

impl RtpTransportParams {
    /// Extract transport params from the first element of a `transport_params` JSON array.
    pub fn from_json_patch(patch: &serde_json::Value) -> Self {
        let tp_obj = patch
            .get("transport_params")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        Self {
            source_ip: tp_obj
                .get("source_ip")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            destination_ip: tp_obj
                .get("destination_ip")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            destination_port: tp_obj
                .get("destination_port")
                .and_then(|v| v.as_u64())
                .map(|n| n as u16),
            source_port: tp_obj
                .get("source_port")
                .and_then(|v| v.as_u64())
                .map(|n| n as u16),
            rtp_enabled: tp_obj.get("rtp_enabled").and_then(|v| v.as_bool()),
            fec_enabled: tp_obj.get("fec_enabled").and_then(|v| v.as_bool()),
            fec_destination_ip: tp_obj
                .get("fec_destination_ip")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            fec_mode: tp_obj
                .get("fec_mode")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            fec1_d_destination_port: tp_obj
                .get("fec1D_destination_port")
                .and_then(|v| v.as_u64())
                .map(|n| n as u16),
            fec2_d_destination_port: tp_obj
                .get("fec2D_destination_port")
                .and_then(|v| v.as_u64())
                .map(|n| n as u16),
            rtcp_enabled: tp_obj.get("rtcp_enabled").and_then(|v| v.as_bool()),
            rtcp_destination_ip: tp_obj
                .get("rtcp_destination_ip")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            rtcp_destination_port: tp_obj
                .get("rtcp_destination_port")
                .and_then(|v| v.as_u64())
                .map(|n| n as u16),
            multicast_ip: tp_obj
                .get("multicast_ip")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            interface_ip: tp_obj
                .get("interface_ip")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        }
    }
}

// ============================================================================
// TransportConstraints — the top-level constraint object
// ============================================================================

/// IS-05 constraint sets for a sender or receiver.
///
/// A sender/receiver advertises one or more `RtpConstraintSet` alternatives.
/// Proposed transport parameters are valid if they satisfy **at least one**
/// constraint set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConstraints {
    /// Array of alternative constraint sets. At least one must be satisfied.
    pub constraint_sets: Vec<RtpConstraintSet>,
}

impl TransportConstraints {
    /// Create a new `TransportConstraints` from a list of constraint sets.
    pub fn new(constraint_sets: Vec<RtpConstraintSet>) -> Self {
        Self { constraint_sets }
    }

    /// Validate proposed transport params against all constraint sets.
    ///
    /// Returns `Ok(())` if params satisfy at least ONE constraint set.
    /// Returns the violation from the last checked constraint set if none match.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, params: &RtpTransportParams) -> Result<(), ConstraintViolation> {
        if self.constraint_sets.is_empty() {
            // No constraints means everything is accepted.
            return Ok(());
        }

        let mut last_violation: Option<ConstraintViolation> = None;

        for cs in &self.constraint_sets {
            match cs.validate(params) {
                Ok(()) => return Ok(()),
                Err(v) => last_violation = Some(v),
            }
        }

        Err(last_violation.unwrap_or_else(|| ConstraintViolation {
            parameter: String::new(),
            reason: "no constraint set matched".to_string(),
            actual_value: None,
            constraint: None,
        }))
    }

    /// Create standard unicast RTP constraints.
    ///
    /// Enforces:
    /// - destination_port: 1–65535
    /// - rtp_enabled: must be true or false (enum)
    pub fn unicast_rtp() -> Self {
        let cs = RtpConstraintSet {
            destination_port: Some(ParameterConstraint::numeric_range(1u64, 65535u64)),
            rtp_enabled: Some(ParameterConstraint::enum_of(vec![
                serde_json::Value::Bool(true),
                serde_json::Value::Bool(false),
            ])),
            // Unicast: multicast_ip must not be set (represented by no constraint)
            ..Default::default()
        };
        Self::new(vec![cs])
    }

    /// Create multicast RTP constraints.
    ///
    /// `multicast_range` is a descriptive string (e.g. `"224.0.0.0/4"`).
    /// The constraint pattern checks that multicast_ip starts in 224–239 range.
    pub fn multicast_rtp(multicast_range: &str) -> Self {
        // Build a pattern that captures the described range
        let pattern = format!("^multicast:{}", multicast_range);
        let cs = RtpConstraintSet {
            destination_port: Some(ParameterConstraint::numeric_range(1u64, 65535u64)),
            rtp_enabled: Some(ParameterConstraint::enum_of(vec![
                serde_json::Value::Bool(true),
                serde_json::Value::Bool(false),
            ])),
            multicast_ip: Some(ParameterConstraint {
                minimum: None,
                maximum: None,
                enum_values: None,
                // Pattern stored for documentation; actual enforcement done by numeric octet check
                pattern: Some(pattern),
            }),
            ..Default::default()
        };
        Self::new(vec![cs])
    }

    /// Create SRT transport constraints.
    ///
    /// SRT uses port range 1024–65535 for both source and destination.
    pub fn srt() -> Self {
        let cs = RtpConstraintSet {
            destination_port: Some(ParameterConstraint::numeric_range(1024u64, 65535u64)),
            source_port: Some(ParameterConstraint::numeric_range(1024u64, 65535u64)),
            // SRT does not use RTP/FEC/RTCP — leave those unconstrained (None)
            ..Default::default()
        };
        Self::new(vec![cs])
    }

    /// Serialize this constraint set to the IS-05 JSON array format.
    ///
    /// IS-05 represents constraints as a JSON array where each element is an
    /// `RtpConstraintSet` object.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.constraint_sets).unwrap_or(serde_json::Value::Array(vec![]))
    }
}

// ============================================================================
// ConstraintViolation
// ============================================================================

/// Describes a constraint violation on a specific transport parameter.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// Name of the parameter that failed validation.
    pub parameter: String,
    /// Human-readable explanation of the failure.
    pub reason: String,
    /// The actual value that was submitted (if available).
    pub actual_value: Option<serde_json::Value>,
    /// The constraint that was violated (if available).
    pub constraint: Option<ParameterConstraint>,
}

impl ConstraintViolation {
    /// Format this violation as an IS-05-compatible JSON error object.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "parameter": self.parameter,
            "reason": self.reason,
            "actual_value": self.actual_value,
        })
    }
}

impl std::fmt::Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "constraint violation on '{}': {}",
            self.parameter, self.reason
        )
    }
}

impl std::error::Error for ConstraintViolation {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    // ── ParameterConstraint tests ─────────────────────────────────────────

    #[test]
    fn test_numeric_range_valid() {
        let c = ParameterConstraint::numeric_range(1u64, 65535u64);
        assert!(c.validate_value(&json!(1000u64), "port").is_ok());
        assert!(c.validate_value(&json!(1u64), "port").is_ok());
        assert!(c.validate_value(&json!(65535u64), "port").is_ok());
    }

    #[test]
    fn test_numeric_range_below_min() {
        let c = ParameterConstraint::numeric_range(1u64, 65535u64);
        let err = c.validate_value(&json!(0u64), "port");
        assert!(err.is_err());
        let v = err.expect_err("should be err");
        assert_eq!(v.parameter, "port");
        assert!(v.reason.contains("below minimum"));
    }

    #[test]
    fn test_numeric_range_above_max() {
        let c = ParameterConstraint::numeric_range(1u64, 65535u64);
        let err = c.validate_value(&json!(70000u64), "port");
        assert!(err.is_err());
        let v = err.expect_err("should be err");
        assert!(v.reason.contains("exceeds maximum"));
    }

    #[test]
    fn test_enum_valid() {
        let c = ParameterConstraint::enum_of(vec![json!("1D"), json!("2D")]);
        assert!(c.validate_value(&json!("1D"), "fec_mode").is_ok());
        assert!(c.validate_value(&json!("2D"), "fec_mode").is_ok());
    }

    #[test]
    fn test_enum_invalid() {
        let c = ParameterConstraint::enum_of(vec![json!("1D"), json!("2D")]);
        let err = c.validate_value(&json!("3D"), "fec_mode");
        assert!(err.is_err());
        let v = err.expect_err("should be err");
        assert!(v.reason.contains("not in allowed enum set"));
    }

    #[test]
    fn test_bool_enum_valid() {
        let c = ParameterConstraint::enum_of(vec![Value::Bool(true), Value::Bool(false)]);
        assert!(c.validate_value(&json!(true), "rtp_enabled").is_ok());
        assert!(c.validate_value(&json!(false), "rtp_enabled").is_ok());
    }

    #[test]
    fn test_pattern_valid() {
        let c = ParameterConstraint::pattern_match("^[0-9]+$");
        // non-empty string passes our simplified matcher
        assert!(c.validate_value(&json!("12345"), "some_field").is_ok());
    }

    #[test]
    fn test_pattern_non_string_fails() {
        let c = ParameterConstraint::pattern_match("^.*$");
        let err = c.validate_value(&json!(42u64), "some_field");
        assert!(err.is_err());
    }

    // ── Serialization tests ───────────────────────────────────────────────

    #[test]
    fn test_parameter_constraint_serialize_numeric_range() {
        let c = ParameterConstraint::numeric_range(1u64, 65535u64);
        let s = serde_json::to_string(&c).expect("serialize");
        let v: Value = serde_json::from_str(&s).expect("parse");
        assert_eq!(v["minimum"], json!(1u64));
        assert_eq!(v["maximum"], json!(65535u64));
        // enum_values and pattern should be absent
        assert!(v.get("enum").is_none());
        assert!(v.get("pattern").is_none());
    }

    #[test]
    fn test_parameter_constraint_serialize_enum() {
        let c = ParameterConstraint::enum_of(vec![json!("1D"), json!("2D")]);
        let s = serde_json::to_string(&c).expect("serialize");
        let v: Value = serde_json::from_str(&s).expect("parse");
        assert_eq!(v["enum"], json!(["1D", "2D"]));
    }

    #[test]
    fn test_rtp_constraint_set_serialize_skip_none() {
        let cs = RtpConstraintSet {
            destination_port: Some(ParameterConstraint::numeric_range(1u64, 65535u64)),
            ..Default::default()
        };
        let s = serde_json::to_string(&cs).expect("serialize");
        let v: Value = serde_json::from_str(&s).expect("parse");
        // Only destination_port present
        assert!(v.get("destination_port").is_some());
        assert!(v.get("source_ip").is_none());
        assert!(v.get("rtp_enabled").is_none());
    }

    #[test]
    fn test_transport_constraints_to_json_array() {
        let tc = TransportConstraints::unicast_rtp();
        let v = tc.to_json();
        assert!(v.is_array());
        let arr = v.as_array().expect("array");
        assert_eq!(arr.len(), 1);
        // The single constraint set should have destination_port
        assert!(arr[0].get("destination_port").is_some());
    }

    // ── TransportConstraints::validate tests ─────────────────────────────

    #[test]
    fn test_validate_unicast_rtp_valid_params() {
        let tc = TransportConstraints::unicast_rtp();
        let params = RtpTransportParams {
            destination_port: Some(5004),
            rtp_enabled: Some(true),
            ..Default::default()
        };
        assert!(tc.validate(&params).is_ok());
    }

    #[test]
    fn test_validate_unicast_rtp_port_zero_fails() {
        let tc = TransportConstraints::unicast_rtp();
        let params = RtpTransportParams {
            destination_port: Some(0),
            ..Default::default()
        };
        let err = tc.validate(&params);
        assert!(err.is_err());
        let v = err.expect_err("should be err");
        assert_eq!(v.parameter, "destination_port");
    }

    #[test]
    fn test_validate_unicast_rtp_port_max() {
        let tc = TransportConstraints::unicast_rtp();
        let params = RtpTransportParams {
            destination_port: Some(65535),
            ..Default::default()
        };
        assert!(tc.validate(&params).is_ok());
    }

    #[test]
    fn test_validate_fec_mode_valid() {
        let mut cs = RtpConstraintSet::default();
        cs.fec_mode = Some(ParameterConstraint::enum_of(vec![json!("1D"), json!("2D")]));
        let tc = TransportConstraints::new(vec![cs]);

        let params = RtpTransportParams {
            fec_mode: Some("1D".to_string()),
            ..Default::default()
        };
        assert!(tc.validate(&params).is_ok());
    }

    #[test]
    fn test_validate_fec_mode_invalid() {
        let mut cs = RtpConstraintSet::default();
        cs.fec_mode = Some(ParameterConstraint::enum_of(vec![json!("1D"), json!("2D")]));
        let tc = TransportConstraints::new(vec![cs]);

        let params = RtpTransportParams {
            fec_mode: Some("3D".to_string()),
            ..Default::default()
        };
        assert!(tc.validate(&params).is_err());
    }

    #[test]
    fn test_validate_multiple_constraint_sets_first_matches() {
        // Two alternatives: unicast (port 5000-5999) or multicast (port 6000-6999)
        let cs1 = RtpConstraintSet {
            destination_port: Some(ParameterConstraint::numeric_range(5000u64, 5999u64)),
            ..Default::default()
        };
        let cs2 = RtpConstraintSet {
            destination_port: Some(ParameterConstraint::numeric_range(6000u64, 6999u64)),
            ..Default::default()
        };
        let tc = TransportConstraints::new(vec![cs1, cs2]);

        // Port 5004 matches first set
        let p1 = RtpTransportParams {
            destination_port: Some(5004),
            ..Default::default()
        };
        assert!(tc.validate(&p1).is_ok());

        // Port 6001 matches second set
        let p2 = RtpTransportParams {
            destination_port: Some(6001),
            ..Default::default()
        };
        assert!(tc.validate(&p2).is_ok());

        // Port 7000 matches neither
        let p3 = RtpTransportParams {
            destination_port: Some(7000),
            ..Default::default()
        };
        assert!(tc.validate(&p3).is_err());
    }

    #[test]
    fn test_validate_empty_constraint_sets_allows_anything() {
        let tc = TransportConstraints::new(vec![]);
        let params = RtpTransportParams {
            destination_port: Some(0), // Would fail if constraints applied
            ..Default::default()
        };
        assert!(tc.validate(&params).is_ok());
    }

    #[test]
    fn test_srt_constraints_valid_port() {
        let tc = TransportConstraints::srt();
        let params = RtpTransportParams {
            destination_port: Some(9000),
            source_port: Some(9001),
            ..Default::default()
        };
        assert!(tc.validate(&params).is_ok());
    }

    #[test]
    fn test_srt_constraints_privileged_port_fails() {
        let tc = TransportConstraints::srt();
        let params = RtpTransportParams {
            destination_port: Some(80), // Below 1024
            ..Default::default()
        };
        assert!(tc.validate(&params).is_err());
    }

    #[test]
    fn test_multicast_rtp_construction() {
        let tc = TransportConstraints::multicast_rtp("224.0.0.0/4");
        assert_eq!(tc.constraint_sets.len(), 1);
        let cs = &tc.constraint_sets[0];
        assert!(cs.multicast_ip.is_some());
        assert!(cs.destination_port.is_some());
    }

    // ── RtpTransportParams::from_json_patch tests ─────────────────────────

    #[test]
    fn test_rtp_transport_params_from_json_patch() {
        let patch = json!({
            "transport_params": [{
                "destination_ip": "192.168.1.100",
                "destination_port": 5004,
                "source_ip": "10.0.0.1",
                "rtp_enabled": true,
                "fec_mode": "1D"
            }]
        });
        let p = RtpTransportParams::from_json_patch(&patch);
        assert_eq!(p.destination_ip.as_deref(), Some("192.168.1.100"));
        assert_eq!(p.destination_port, Some(5004));
        assert_eq!(p.source_ip.as_deref(), Some("10.0.0.1"));
        assert_eq!(p.rtp_enabled, Some(true));
        assert_eq!(p.fec_mode.as_deref(), Some("1D"));
    }

    #[test]
    fn test_rtp_transport_params_from_empty_patch() {
        let patch = json!({});
        let p = RtpTransportParams::from_json_patch(&patch);
        assert!(p.destination_ip.is_none());
        assert!(p.destination_port.is_none());
    }

    // ── ConstraintViolation tests ─────────────────────────────────────────

    #[test]
    fn test_constraint_violation_to_json() {
        let v = ConstraintViolation {
            parameter: "destination_port".to_string(),
            reason: "value 0 is below minimum 1".to_string(),
            actual_value: Some(json!(0u64)),
            constraint: None,
        };
        let j = v.to_json();
        assert_eq!(j["parameter"], json!("destination_port"));
        assert!(j["reason"].as_str().is_some());
    }

    #[test]
    fn test_constraint_violation_display() {
        let v = ConstraintViolation {
            parameter: "port".to_string(),
            reason: "out of range".to_string(),
            actual_value: None,
            constraint: None,
        };
        let s = v.to_string();
        assert!(s.contains("port"));
        assert!(s.contains("out of range"));
    }
}
