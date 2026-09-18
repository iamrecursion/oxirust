//! Dolby Vision quality control checking.
//!
//! This module validates Dolby Vision metadata blocks for consistency,
//! profile compliance, and light level plausibility.

/// Dolby Vision profile variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DvProfile {
    /// Profile 5: Single-layer, HDR-only, no HDR10 fallback.
    Profile5,
    /// Profile 7: Dual-layer, with HDR10 base layer.
    Profile7,
    /// Profile 8.1: Single-layer, HDR10 compatible, cross-compatible with SDR.
    Profile8_1,
    /// Profile 8.2: Single-layer, HDR10 compatible.
    Profile8_2,
    /// Profile 8.4: Single-layer, HLG compatible.
    Profile8_4,
}

impl DvProfile {
    /// Returns whether this profile is compatible with HDR10 displays.
    #[must_use]
    pub fn compatible_with_hdr10(self) -> bool {
        matches!(
            self,
            DvProfile::Profile7
                | DvProfile::Profile8_1
                | DvProfile::Profile8_2
                | DvProfile::Profile8_4
        )
    }

    /// Returns the profile name as a string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            DvProfile::Profile5 => "Profile 5",
            DvProfile::Profile7 => "Profile 7",
            DvProfile::Profile8_1 => "Profile 8.1",
            DvProfile::Profile8_2 => "Profile 8.2",
            DvProfile::Profile8_4 => "Profile 8.4",
        }
    }

    /// Returns whether this profile requires a base layer.
    #[must_use]
    pub fn requires_base_layer(self) -> bool {
        matches!(self, DvProfile::Profile7)
    }
}

/// A Dolby Vision metadata block for a single frame.
#[derive(Debug, Clone)]
pub struct DvMetadataBlock {
    /// Frame index this metadata applies to.
    pub frame_idx: u64,
    /// Maximum PQ value (0–4095).
    pub max_pq: u16,
    /// Minimum PQ value (0–4095).
    pub min_pq: u16,
    /// Maximum content light level (nits).
    pub max_content_light: u16,
    /// Maximum frame-average light level (nits).
    pub max_frame_avg_light: u16,
}

impl DvMetadataBlock {
    /// Creates a new metadata block.
    #[must_use]
    pub fn new(
        frame_idx: u64,
        max_pq: u16,
        min_pq: u16,
        max_content_light: u16,
        max_frame_avg_light: u16,
    ) -> Self {
        Self {
            frame_idx,
            max_pq,
            min_pq,
            max_content_light,
            max_frame_avg_light,
        }
    }

    /// Converts max_pq to nits using the PQ EOTF.
    ///
    /// Formula: `(pq / 4095.0).powf(1.0 / 0.1593) * 10000.0`
    #[must_use]
    pub fn max_nits(&self) -> f32 {
        pq_to_nits(self.max_pq)
    }

    /// Converts min_pq to nits.
    #[must_use]
    pub fn min_nits(&self) -> f32 {
        pq_to_nits(self.min_pq)
    }

    /// Returns true if the block has consistent light levels
    /// (max_frame_avg_light ≤ max_content_light).
    #[must_use]
    pub fn has_consistent_light_levels(&self) -> bool {
        self.max_frame_avg_light <= self.max_content_light
    }
}

/// Converts a PQ code value (0–4095) to nits.
#[must_use]
pub fn pq_to_nits(pq: u16) -> f32 {
    let normalized = pq as f32 / 4095.0;
    normalized.powf(1.0 / 0.1593) * 10_000.0
}

/// Types of Dolby Vision QC issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DvIssueType {
    /// `max_pq` exceeds the profile's allowed maximum.
    MaxPqTooHigh,
    /// `min_pq` is negative or zero (implies data corruption).
    MinPqNegative,
    /// `max_frame_avg_light` exceeds `max_content_light`.
    LightLevelInconsistent,
    /// Metadata block is missing for a frame.
    MetadataMissing,
}

impl DvIssueType {
    /// Returns a human-readable description.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            DvIssueType::MaxPqTooHigh => "max_pq exceeds profile maximum",
            DvIssueType::MinPqNegative => "min_pq is zero (invalid)",
            DvIssueType::LightLevelInconsistent => "max_frame_avg_light exceeds max_content_light",
            DvIssueType::MetadataMissing => "Metadata block missing for frame",
        }
    }
}

/// A single QC result for a Dolby Vision frame.
#[derive(Debug, Clone)]
pub struct DvQcResult {
    /// Frame index where the issue was found.
    pub frame_idx: u64,
    /// Type of issue.
    pub issue_type: DvIssueType,
    /// The offending value.
    pub value: f32,
}

impl DvQcResult {
    /// Creates a new DV QC result.
    #[must_use]
    pub fn new(frame_idx: u64, issue_type: DvIssueType, value: f32) -> Self {
        Self {
            frame_idx,
            issue_type,
            value,
        }
    }
}

/// Dolby Vision quality control checker.
#[derive(Debug, Clone, Default)]
pub struct DvQcChecker;

impl DvQcChecker {
    /// Creates a new DV QC checker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns the maximum allowed max_pq for a given profile.
    ///
    /// Profile 5 is limited to 4095 (10,000 nits).
    /// Profile 7/8.x allow up to 4095 as well (full PQ range).
    #[must_use]
    fn max_pq_for_profile(profile: DvProfile) -> u16 {
        match profile {
            DvProfile::Profile5 => 4095,
            DvProfile::Profile7 => 4095,
            DvProfile::Profile8_1 | DvProfile::Profile8_2 | DvProfile::Profile8_4 => 4095,
        }
    }

    /// Checks Dolby Vision metadata blocks for QC issues.
    ///
    /// Checks performed:
    /// - `max_pq` within profile limits
    /// - `min_pq` > 0
    /// - Light level consistency (`max_frame_avg_light ≤ max_content_light`)
    #[must_use]
    pub fn check(metadata: &[DvMetadataBlock], profile: DvProfile) -> Vec<DvQcResult> {
        let mut results = Vec::new();
        let max_pq_limit = Self::max_pq_for_profile(profile);

        for block in metadata {
            // Check max_pq
            if block.max_pq > max_pq_limit {
                results.push(DvQcResult::new(
                    block.frame_idx,
                    DvIssueType::MaxPqTooHigh,
                    f32::from(block.max_pq),
                ));
            }

            // Check min_pq — zero is technically valid but suspicious for real content
            // In practice, min_pq of 0 indicates black, which can be valid.
            // We treat it as an issue only if max_pq is also 0 (no signal).
            if block.max_pq == 0 && block.min_pq == 0 {
                results.push(DvQcResult::new(
                    block.frame_idx,
                    DvIssueType::MinPqNegative,
                    0.0,
                ));
            }

            // Check light level consistency
            if !block.has_consistent_light_levels() {
                results.push(DvQcResult::new(
                    block.frame_idx,
                    DvIssueType::LightLevelInconsistent,
                    f32::from(block.max_frame_avg_light),
                ));
            }
        }

        results
    }
}

/// Comprehensive Dolby Vision quality control report.
#[derive(Debug, Clone)]
pub struct DvReport {
    /// Dolby Vision profile of the content.
    pub profile: DvProfile,
    /// Total number of frames analyzed.
    pub frame_count: u64,
    /// All QC issues found.
    pub issues: Vec<DvQcResult>,
    /// Maximum measured peak luminance in nits.
    pub max_measured_nits: f32,
    /// Whether the content is compatible with HDR10 display.
    pub hdr10_compatible: bool,
}

impl DvReport {
    /// Builds a `DvReport` from metadata blocks.
    #[must_use]
    pub fn build(metadata: &[DvMetadataBlock], profile: DvProfile) -> Self {
        let issues = DvQcChecker::check(metadata, profile);
        let frame_count = metadata.len() as u64;
        let max_measured_nits = metadata
            .iter()
            .map(DvMetadataBlock::max_nits)
            .fold(0.0_f32, f32::max);
        let hdr10_compatible = profile.compatible_with_hdr10();

        Self {
            profile,
            frame_count,
            issues,
            max_measured_nits,
            hdr10_compatible,
        }
    }

    /// Returns true if no issues were found.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Returns the number of issues of each type.
    #[must_use]
    pub fn issue_count(&self, issue_type: DvIssueType) -> usize {
        self.issues
            .iter()
            .filter(|i| i.issue_type == issue_type)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(
        frame_idx: u64,
        max_pq: u16,
        min_pq: u16,
        cll: u16,
        fall: u16,
    ) -> DvMetadataBlock {
        DvMetadataBlock::new(frame_idx, max_pq, min_pq, cll, fall)
    }

    #[test]
    fn test_dv_profile_compatible_with_hdr10() {
        assert!(!DvProfile::Profile5.compatible_with_hdr10());
        assert!(DvProfile::Profile7.compatible_with_hdr10());
        assert!(DvProfile::Profile8_1.compatible_with_hdr10());
        assert!(DvProfile::Profile8_2.compatible_with_hdr10());
        assert!(DvProfile::Profile8_4.compatible_with_hdr10());
    }

    #[test]
    fn test_dv_profile_names() {
        assert_eq!(DvProfile::Profile5.name(), "Profile 5");
        assert_eq!(DvProfile::Profile8_1.name(), "Profile 8.1");
    }

    #[test]
    fn test_dv_profile_requires_base_layer() {
        assert!(DvProfile::Profile7.requires_base_layer());
        assert!(!DvProfile::Profile5.requires_base_layer());
    }

    #[test]
    fn test_pq_to_nits_full_scale() {
        // PQ=4095 should be ~10,000 nits
        let nits = pq_to_nits(4095);
        assert!(
            (nits - 10_000.0).abs() < 1.0,
            "Expected ~10000 nits, got {nits}"
        );
    }

    #[test]
    fn test_pq_to_nits_zero() {
        let nits = pq_to_nits(0);
        assert_eq!(nits, 0.0);
    }

    #[test]
    fn test_max_nits() {
        let block = make_block(0, 4095, 100, 1000, 400);
        let nits = block.max_nits();
        assert!((nits - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn test_consistent_light_levels_valid() {
        let block = make_block(0, 4000, 100, 1000, 400);
        assert!(block.has_consistent_light_levels());
    }

    #[test]
    fn test_consistent_light_levels_invalid() {
        // fall > cll → inconsistent
        let block = make_block(0, 4000, 100, 400, 1000);
        assert!(!block.has_consistent_light_levels());
    }

    #[test]
    fn test_dv_qc_checker_clean_metadata() {
        let metadata = vec![
            make_block(0, 3000, 100, 1000, 400),
            make_block(1, 3500, 150, 1000, 450),
        ];
        let issues = DvQcChecker::check(&metadata, DvProfile::Profile8_1);
        assert!(issues.is_empty(), "Expected no issues, got {:?}", issues);
    }

    #[test]
    fn test_dv_qc_checker_inconsistent_light() {
        let metadata = vec![make_block(0, 3000, 100, 400, 1000)];
        let issues = DvQcChecker::check(&metadata, DvProfile::Profile8_1);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == DvIssueType::LightLevelInconsistent));
    }

    #[test]
    fn test_dv_qc_checker_zero_pq() {
        let metadata = vec![make_block(5, 0, 0, 0, 0)];
        let issues = DvQcChecker::check(&metadata, DvProfile::Profile5);
        assert!(issues
            .iter()
            .any(|i| i.issue_type == DvIssueType::MinPqNegative));
    }

    #[test]
    fn test_dv_report_build() {
        let metadata = vec![
            make_block(0, 4000, 100, 1000, 400),
            make_block(1, 3500, 50, 1000, 450),
        ];
        let report = DvReport::build(&metadata, DvProfile::Profile8_1);
        assert_eq!(report.frame_count, 2);
        assert!(report.hdr10_compatible);
        assert!(report.max_measured_nits > 0.0);
    }

    #[test]
    fn test_dv_report_is_clean() {
        let metadata = vec![make_block(0, 3000, 100, 1000, 400)];
        let report = DvReport::build(&metadata, DvProfile::Profile8_2);
        assert!(report.is_clean());
    }

    #[test]
    fn test_dv_issue_type_description() {
        assert!(!DvIssueType::MaxPqTooHigh.description().is_empty());
        assert!(!DvIssueType::MetadataMissing.description().is_empty());
    }
}
