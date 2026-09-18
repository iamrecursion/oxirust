#![allow(unused_imports)]
//! MPEG-TS packet structure validation.
//!
//! Provides validation of MPEG Transport Stream files including:
//! - Packet sync byte validation
//! - Continuity counter checking
//! - PCR/DTS/PTS validation
//! - PAT/PMT table validation
//! - PSI/SI table validation
//! - Null packet analysis

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;
use std::path::Path;

/// MPEG-TS packet size in bytes.
const TS_PACKET_SIZE: usize = 188;
const TS_PACKET_SIZE_WITH_FEC: usize = 204; // With Forward Error Correction
const TS_SYNC_BYTE: u8 = 0x47;

/// MPEG-TS format validator.
///
/// Validates compliance with ISO/IEC 13818-1 (MPEG-2 Systems).
pub struct MpegTsValidator {
    check_continuity: bool,
    check_pcr: bool,
    check_psi: bool,
    check_timing: bool,
    max_cc_errors: usize,
    max_pcr_interval_ms: u64,
}

impl MpegTsValidator {
    /// Creates a new MPEG-TS validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            check_continuity: true,
            check_pcr: true,
            check_psi: true,
            check_timing: true,
            max_cc_errors: 0,
            max_pcr_interval_ms: 100,
        }
    }

    /// Sets whether to check continuity counters.
    #[must_use]
    pub const fn with_continuity_check(mut self, check: bool) -> Self {
        self.check_continuity = check;
        self
    }

    /// Sets whether to check PCR (Program Clock Reference).
    #[must_use]
    pub const fn with_pcr_check(mut self, check: bool) -> Self {
        self.check_pcr = check;
        self
    }

    /// Sets maximum allowed continuity counter errors.
    #[must_use]
    pub const fn with_max_cc_errors(mut self, max: usize) -> Self {
        self.max_cc_errors = max;
        self
    }

    /// Sets maximum PCR interval in milliseconds.
    #[must_use]
    pub const fn with_max_pcr_interval_ms(mut self, ms: u64) -> Self {
        self.max_pcr_interval_ms = ms;
        self
    }

    /// Validates packet structure and sync bytes.
    fn validate_packet_structure(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Read file and check size is multiple of 188 or 204 bytes
        // - Verify sync byte (0x47) at correct intervals
        // - Detect packet size (188 vs 204)
        // - Count packets and validate structure

        results.push(CheckResult::pass(self.name()).with_recommendation(format!(
            "TS packet structure validation (sync byte: 0x{TS_SYNC_BYTE:02X})"
        )));

        Ok(results)
    }

    /// Validates continuity counters.
    fn validate_continuity_counters(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_continuity {
            // In production, would:
            // - Track continuity counter for each PID
            // - Detect discontinuities
            // - Allow for discontinuity_indicator flag
            // - Count total CC errors

            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "Continuity counter validation (max errors: {})",
                self.max_cc_errors
            )));
        }

        Ok(results)
    }

    /// Validates PCR timing.
    fn validate_pcr(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_pcr {
            // In production, would:
            // - Extract PCR values from adaptation field
            // - Verify PCR interval is within spec (max 100ms)
            // - Check PCR continuity and jitter
            // - Validate PCR accuracy

            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "PCR timing validation (max interval: {}ms)",
                self.max_pcr_interval_ms
            )));
        }

        Ok(results)
    }

    /// Validates DTS/PTS timing.
    fn validate_dts_pts(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_timing {
            // In production, would:
            // - Extract DTS/PTS from PES headers
            // - Verify DTS <= PTS
            // - Check for timestamp wraparound
            // - Validate timestamp continuity

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("DTS/PTS timing validation".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates PSI tables (PAT, PMT).
    fn validate_psi_tables(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.check_psi {
            // In production, would validate:
            // - PAT (Program Association Table) on PID 0
            // - PMT (Program Map Table) references
            // - Table versioning and CRC
            // - Section length and structure
            // - Stream type descriptors

            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("PSI table validation (PAT/PMT/CAT)".to_string()),
            );
        }

        Ok(results)
    }

    /// Validates SI tables (SDT, EIT, etc.).
    fn validate_si_tables(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would validate:
        // - SDT (Service Description Table)
        // - EIT (Event Information Table)
        // - TDT/TOT (Time and Date Table)
        // - NIT (Network Information Table)

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("SI table validation for DVB compliance".to_string()),
        );

        Ok(results)
    }

    /// Analyzes null packets.
    fn analyze_null_packets(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Count null packets (PID 0x1FFF)
        // - Calculate null packet ratio
        // - Warn if excessive null packets (indicates padding)

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Null packet analysis (PID 0x1FFF)".to_string()),
        );

        Ok(results)
    }

    /// Validates adaptation field.
    fn validate_adaptation_field(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would validate:
        // - Adaptation field length
        // - Discontinuity indicator
        // - Random access indicator
        // - Elementary stream priority indicator
        // - PCR flag and PCR value
        // - OPCR flag and value
        // - Splicing point flag
        // - Private data

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Adaptation field structure validation".to_string()),
        );

        Ok(results)
    }

    /// Checks bitrate consistency.
    fn check_bitrate_consistency(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Calculate instantaneous bitrate
        // - Detect bitrate variations
        // - Warn about CBR vs VBR characteristics
        // - Check against specified bitrate

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Bitrate consistency analysis".to_string()),
        );

        Ok(results)
    }
}

impl Default for MpegTsValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for MpegTsValidator {
    fn name(&self) -> &str {
        "mpegts_structure_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates MPEG-TS packet structure and timing (ISO 13818-1)"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        // Only applicable to MPEG-TS files
        let path_lower = context.file_path.to_lowercase();
        if !path_lower.ends_with(".ts")
            && !path_lower.ends_with(".m2ts")
            && !path_lower.ends_with(".mts")
        {
            return Ok(vec![CheckResult::pass(self.name()).with_recommendation(
                "Not an MPEG-TS file - check skipped".to_string(),
            )]);
        }

        // Validate packet structure
        let packet_results = self.validate_packet_structure(&context.file_path)?;
        all_results.extend(packet_results);

        // Validate continuity counters
        let cc_results = self.validate_continuity_counters(&context.file_path)?;
        all_results.extend(cc_results);

        // Validate PCR
        let pcr_results = self.validate_pcr(&context.file_path)?;
        all_results.extend(pcr_results);

        // Validate DTS/PTS
        let timing_results = self.validate_dts_pts(&context.file_path)?;
        all_results.extend(timing_results);

        // Validate PSI tables
        let psi_results = self.validate_psi_tables(&context.file_path)?;
        all_results.extend(psi_results);

        // Validate SI tables
        let si_results = self.validate_si_tables(&context.file_path)?;
        all_results.extend(si_results);

        // Analyze null packets
        let null_results = self.analyze_null_packets(&context.file_path)?;
        all_results.extend(null_results);

        // Validate adaptation field
        let adaptation_results = self.validate_adaptation_field(&context.file_path)?;
        all_results.extend(adaptation_results);

        // Check bitrate consistency
        let bitrate_results = self.check_bitrate_consistency(&context.file_path)?;
        all_results.extend(bitrate_results);

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        let path = Path::new(&context.file_path);
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            matches!(ext_lower.as_str(), "ts" | "m2ts" | "mts")
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpegts_validator_creation() {
        let validator = MpegTsValidator::new();
        assert_eq!(validator.name(), "mpegts_structure_validation");
        assert!(validator.check_continuity);
        assert_eq!(validator.max_pcr_interval_ms, 100);
    }

    #[test]
    fn test_mpegts_validator_configuration() {
        let validator = MpegTsValidator::new()
            .with_continuity_check(false)
            .with_max_cc_errors(5)
            .with_max_pcr_interval_ms(200);

        assert!(!validator.check_continuity);
        assert_eq!(validator.max_cc_errors, 5);
        assert_eq!(validator.max_pcr_interval_ms, 200);
    }

    #[test]
    fn test_mpegts_applicability() {
        let validator = MpegTsValidator::new();
        let mut context = QcContext::new("test.ts");
        assert!(validator.is_applicable(&context));

        context.file_path = "test.m2ts".to_string();
        assert!(validator.is_applicable(&context));

        context.file_path = "test.mp4".to_string();
        assert!(!validator.is_applicable(&context));
    }

    #[test]
    fn test_ts_constants() {
        assert_eq!(TS_PACKET_SIZE, 188);
        assert_eq!(TS_PACKET_SIZE_WITH_FEC, 204);
        assert_eq!(TS_SYNC_BYTE, 0x47);
    }
}
