//! WCAG 2.1 and 2.2 compliance checking.

use crate::compliance::report::{ComplianceIssue, IssueSeverity};
use serde::{Deserialize, Serialize};

/// WCAG 2.1 conformance level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WcagLevel {
    /// Level A (minimum).
    A,
    /// Level AA (recommended).
    AA,
    /// Level AAA (highest).
    AAA,
}

/// WCAG 2.1 guideline categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WcagGuideline {
    /// Perceivable - Information must be presentable to users.
    Perceivable,
    /// Operable - UI components must be operable.
    Operable,
    /// Understandable - Information and UI must be understandable.
    Understandable,
    /// Robust - Content must be robust enough for assistive technologies.
    Robust,
}

/// Version of the WCAG standard to check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WcagVersion {
    /// WCAG 2.1 (June 2018).
    V2_1,
    /// WCAG 2.2 (October 2023) — superset of 2.1 with nine new SCs.
    #[default]
    V2_2,
}

/// Visibility details for a focus indicator (used for SC 2.4.11 / 2.4.12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusIndicatorParams {
    /// Area of the focus indicator outline in CSS pixels².
    pub outline_area_px2: f64,
    /// Contrast ratio between focused and unfocused state colors.
    pub contrast_ratio: f64,
    /// Whether the indicator completely encloses the component perimeter.
    pub encloses_component: bool,
    /// Perimeter length of the component in CSS pixels.
    pub component_perimeter_px: f64,
    /// Thickness of the focus outline in CSS pixels.
    pub outline_thickness_px: f64,
}

impl FocusIndicatorParams {
    /// Minimum outline area required by WCAG 2.2 SC 2.4.11 (AA).
    /// The spec requires at least: component_perimeter × 2 css px² of focused area.
    #[must_use]
    pub fn meets_minimum_area(&self) -> bool {
        // SC 2.4.11: focus indicator area ≥ perimeter × 2 pixel area
        let required = self.component_perimeter_px * 2.0;
        self.outline_area_px2 >= required
    }

    /// Check SC 2.4.12 Enhanced Focus (AAA): contrast ≥ 3:1 and area ≥ perimeter × 4px².
    #[must_use]
    pub fn meets_enhanced_focus(&self) -> bool {
        let required = self.component_perimeter_px * 4.0;
        self.outline_area_px2 >= required && self.contrast_ratio >= 3.0
    }
}

/// Parameters describing a drag interaction for SC 2.5.7 (Dragging Movements).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragInteractionParams {
    /// Whether the draggable functionality also has a single-pointer alternative.
    pub has_single_pointer_alternative: bool,
    /// Description of the alternative (e.g., "Use the arrow buttons to reorder").
    pub alternative_description: Option<String>,
    /// Whether the drag operation can be cancelled (e.g., pressing Escape).
    pub is_cancellable: bool,
}

/// WCAG compliance checker with WCAG 2.1 and 2.2 support.
pub struct WcagChecker {
    level: WcagLevel,
    /// Version of WCAG to check against.
    version: WcagVersion,
}

impl WcagChecker {
    /// Create a new WCAG checker.
    #[must_use]
    pub const fn new(level: WcagLevel) -> Self {
        Self {
            level,
            version: WcagVersion::V2_2,
        }
    }

    /// Create a checker pinned to a specific WCAG version.
    #[must_use]
    pub const fn with_version(level: WcagLevel, version: WcagVersion) -> Self {
        Self { level, version }
    }

    /// Get the WCAG version.
    #[must_use]
    pub const fn version(&self) -> WcagVersion {
        self.version
    }

    /// Check WCAG compliance.
    #[must_use]
    pub fn check(&self) -> Vec<ComplianceIssue> {
        let mut issues = Vec::new();

        // Check Perceivable guidelines
        issues.extend(self.check_perceivable());

        // Check Operable guidelines
        issues.extend(self.check_operable());

        // Check Understandable guidelines
        issues.extend(self.check_understandable());

        // Check Robust guidelines
        issues.extend(self.check_robust());

        issues
    }

    fn check_perceivable(&self) -> Vec<ComplianceIssue> {
        let issues = Vec::new();

        // 1.1 Text Alternatives
        // 1.2 Time-based Media (captions, audio description, etc.)
        // 1.3 Adaptable
        // 1.4 Distinguishable (contrast, resize text, etc.)

        // Placeholder checks
        match self.level {
            WcagLevel::A | WcagLevel::AA | WcagLevel::AAA => {
                // Check for captions
                // Check for audio descriptions
                // Check contrast ratios
            }
        }

        issues
    }

    fn check_operable(&self) -> Vec<ComplianceIssue> {
        // 2.1 Keyboard Accessible
        // 2.2 Enough Time
        // 2.3 Seizures and Physical Reactions
        // 2.4 Navigable
        // 2.5 Input Modalities

        Vec::new()
    }

    fn check_understandable(&self) -> Vec<ComplianceIssue> {
        // 3.1 Readable
        // 3.2 Predictable
        // 3.3 Input Assistance

        Vec::new()
    }

    fn check_robust(&self) -> Vec<ComplianceIssue> {
        // 4.1 Compatible

        Vec::new()
    }

    /// Check if captions are present (Success Criterion 1.2.2).
    #[must_use]
    pub fn check_captions_present(&self, has_captions: bool) -> Option<ComplianceIssue> {
        if !has_captions {
            return Some(ComplianceIssue::new(
                "WCAG-1.2.2".to_string(),
                "Captions (Prerecorded)".to_string(),
                "Media content must have synchronized captions".to_string(),
                IssueSeverity::Critical,
            ));
        }
        None
    }

    /// Check if audio description is present (Success Criterion 1.2.3).
    #[must_use]
    pub fn check_audio_description(&self, has_audio_desc: bool) -> Option<ComplianceIssue> {
        if matches!(self.level, WcagLevel::AA | WcagLevel::AAA) && !has_audio_desc {
            return Some(ComplianceIssue::new(
                "WCAG-1.2.5".to_string(),
                "Audio Description (Prerecorded)".to_string(),
                "Media content should have audio description".to_string(),
                IssueSeverity::High,
            ));
        }
        None
    }

    /// Check contrast ratio (Success Criterion 1.4.3).
    #[must_use]
    pub fn check_contrast_ratio(&self, ratio: f32) -> Option<ComplianceIssue> {
        let min_ratio = match self.level {
            WcagLevel::A => 3.0,
            WcagLevel::AA => 4.5,
            WcagLevel::AAA => 7.0,
        };

        if ratio < min_ratio {
            return Some(ComplianceIssue::new(
                "WCAG-1.4.3".to_string(),
                "Contrast (Minimum)".to_string(),
                format!("Contrast ratio {ratio:.2}:1 is below required {min_ratio:.1}:1"),
                IssueSeverity::High,
            ));
        }
        None
    }

    /// Get conformance level.
    #[must_use]
    pub const fn level(&self) -> WcagLevel {
        self.level
    }

    // ─── WCAG 2.2 Success Criteria ───────────────────────────────────────────

    /// SC 2.4.11 – Focus Not Obscured (Minimum) — Level AA (WCAG 2.2).
    ///
    /// When a component receives keyboard focus, the focus indicator must be at
    /// least partially visible (not fully hidden behind sticky headers, etc.).
    /// Pass `is_indicator_visible = true` if the indicator can be seen by the user.
    #[must_use]
    pub fn check_focus_not_obscured_minimum(
        &self,
        is_indicator_visible: bool,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None; // Not applicable to 2.1
        }
        if !is_indicator_visible {
            return Some(ComplianceIssue::new(
                "WCAG-2.4.11".to_string(),
                "Focus Not Obscured (Minimum)".to_string(),
                "Keyboard focus indicator must not be fully hidden behind other content"
                    .to_string(),
                IssueSeverity::High,
            ));
        }
        None
    }

    /// SC 2.4.12 – Focus Not Obscured (Enhanced) — Level AAA (WCAG 2.2).
    ///
    /// The focused component must be entirely visible — no part obscured.
    #[must_use]
    pub fn check_focus_not_obscured_enhanced(
        &self,
        is_fully_visible: bool,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 || self.level != WcagLevel::AAA {
            return None;
        }
        if !is_fully_visible {
            return Some(ComplianceIssue::new(
                "WCAG-2.4.12".to_string(),
                "Focus Not Obscured (Enhanced)".to_string(),
                "Focused component must be entirely visible (not partially obscured)".to_string(),
                IssueSeverity::Medium,
            ));
        }
        None
    }

    /// SC 2.4.13 – Focus Appearance — Level AA (WCAG 2.2).
    ///
    /// The keyboard focus indicator must have sufficient area and contrast.
    #[must_use]
    pub fn check_focus_appearance(&self, params: &FocusIndicatorParams) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None;
        }
        if !matches!(self.level, WcagLevel::AA | WcagLevel::AAA) {
            return None;
        }
        let area_ok = params.meets_minimum_area();
        let contrast_ok = params.contrast_ratio >= 3.0;

        if !area_ok || !contrast_ok {
            let detail = if !area_ok && !contrast_ok {
                format!(
                    "Focus indicator area ({:.1}px²) is below required minimum and contrast ({:.2}:1) is below 3:1",
                    params.outline_area_px2, params.contrast_ratio
                )
            } else if !area_ok {
                format!(
                    "Focus indicator area ({:.1}px²) is below minimum (perimeter × 2 = {:.1}px²)",
                    params.outline_area_px2,
                    params.component_perimeter_px * 2.0
                )
            } else {
                format!(
                    "Focus indicator contrast ({:.2}:1) is below required 3:1",
                    params.contrast_ratio
                )
            };

            return Some(ComplianceIssue::new(
                "WCAG-2.4.13".to_string(),
                "Focus Appearance".to_string(),
                detail,
                IssueSeverity::High,
            ));
        }
        None
    }

    /// SC 2.5.7 – Dragging Movements — Level AA (WCAG 2.2).
    ///
    /// Any functionality using a dragging movement must also be achievable
    /// with a single-pointer action.
    #[must_use]
    pub fn check_dragging_alternatives(
        &self,
        drag: &DragInteractionParams,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None;
        }
        if !matches!(self.level, WcagLevel::AA | WcagLevel::AAA) {
            return None;
        }
        if !drag.has_single_pointer_alternative {
            let desc = drag
                .alternative_description
                .as_deref()
                .unwrap_or("no alternative provided");
            return Some(ComplianceIssue::new(
                "WCAG-2.5.7".to_string(),
                "Dragging Movements".to_string(),
                format!("Drag interaction has no single-pointer alternative: {desc}"),
                IssueSeverity::High,
            ));
        }
        None
    }

    /// SC 2.5.8 – Target Size (Minimum) — Level AA (WCAG 2.2).
    ///
    /// Pointer targets must be at least 24 × 24 CSS pixels (or have adequate
    /// spacing so that a 24px circle centred on the target intersects no other target).
    #[must_use]
    pub fn check_target_size_minimum(
        &self,
        target_width_px: f64,
        target_height_px: f64,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None;
        }
        if !matches!(self.level, WcagLevel::AA | WcagLevel::AAA) {
            return None;
        }
        const MIN_SIZE: f64 = 24.0;
        if target_width_px < MIN_SIZE || target_height_px < MIN_SIZE {
            return Some(ComplianceIssue::new(
                "WCAG-2.5.8".to_string(),
                "Target Size (Minimum)".to_string(),
                format!(
                    "Target size {target_width_px:.0}×{target_height_px:.0}px is below the \
                     minimum 24×24 CSS pixels required by WCAG 2.2 SC 2.5.8"
                ),
                IssueSeverity::Medium,
            ));
        }
        None
    }

    /// SC 3.2.6 – Consistent Help — Level A (WCAG 2.2).
    ///
    /// If a page contains a help mechanism (chat widget, contact link, etc.),
    /// it must appear in a consistent location across a set of pages.
    #[must_use]
    pub fn check_consistent_help(
        &self,
        has_help_mechanism: bool,
        is_consistently_placed: bool,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None;
        }
        if has_help_mechanism && !is_consistently_placed {
            return Some(ComplianceIssue::new(
                "WCAG-3.2.6".to_string(),
                "Consistent Help".to_string(),
                "Help mechanisms must appear in consistent locations across pages".to_string(),
                IssueSeverity::Medium,
            ));
        }
        None
    }

    /// SC 3.3.7 – Redundant Entry — Level A (WCAG 2.2).
    ///
    /// Information users have previously entered in the same process must not
    /// be required again unless re-entry is essential.
    #[must_use]
    pub fn check_redundant_entry(
        &self,
        requires_re_entry_of_previous_info: bool,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None;
        }
        if requires_re_entry_of_previous_info {
            return Some(ComplianceIssue::new(
                "WCAG-3.3.7".to_string(),
                "Redundant Entry".to_string(),
                "Previously entered information must not be re-required unless essential"
                    .to_string(),
                IssueSeverity::Medium,
            ));
        }
        None
    }

    /// SC 3.3.8 – Accessible Authentication (Minimum) — Level AA (WCAG 2.2).
    ///
    /// Authentication must not rely solely on a cognitive function test
    /// (e.g., solve a puzzle, remember a passcode without an alternative).
    #[must_use]
    pub fn check_accessible_authentication_minimum(
        &self,
        relies_on_cognitive_test_only: bool,
    ) -> Option<ComplianceIssue> {
        if self.version == WcagVersion::V2_1 {
            return None;
        }
        if !matches!(self.level, WcagLevel::AA | WcagLevel::AAA) {
            return None;
        }
        if relies_on_cognitive_test_only {
            return Some(ComplianceIssue::new(
                "WCAG-3.3.8".to_string(),
                "Accessible Authentication (Minimum)".to_string(),
                "Authentication must not rely solely on cognitive function tests without \
                 providing an alternative method or assistance"
                    .to_string(),
                IssueSeverity::High,
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wcag_checker() {
        let checker = WcagChecker::new(WcagLevel::AA);
        assert_eq!(checker.level(), WcagLevel::AA);
    }

    #[test]
    fn test_check_captions() {
        let checker = WcagChecker::new(WcagLevel::AA);
        assert!(checker.check_captions_present(false).is_some());
        assert!(checker.check_captions_present(true).is_none());
    }

    #[test]
    fn test_check_contrast() {
        let checker = WcagChecker::new(WcagLevel::AA);
        assert!(checker.check_contrast_ratio(3.0).is_some());
        assert!(checker.check_contrast_ratio(5.0).is_none());
    }

    #[test]
    fn test_check_audio_description() {
        let checker = WcagChecker::new(WcagLevel::AA);
        assert!(checker.check_audio_description(false).is_some());

        let checker_a = WcagChecker::new(WcagLevel::A);
        assert!(checker_a.check_audio_description(false).is_none());
    }

    // ============================================================
    // WCAG 2.2 tests
    // ============================================================

    #[test]
    fn test_wcag22_version_default() {
        let checker = WcagChecker::new(WcagLevel::AA);
        assert_eq!(checker.version(), WcagVersion::V2_2);
    }

    #[test]
    fn test_wcag21_new_criteria_not_applicable() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_1);
        // WCAG 2.2 criteria should return None for a 2.1 checker
        assert!(checker.check_focus_not_obscured_minimum(false).is_none());
        assert!(checker
            .check_dragging_alternatives(&DragInteractionParams {
                has_single_pointer_alternative: false,
                alternative_description: None,
                is_cancellable: false,
            })
            .is_none());
        assert!(checker.check_target_size_minimum(10.0, 10.0).is_none());
        assert!(checker.check_consistent_help(true, false).is_none());
        assert!(checker.check_redundant_entry(true).is_none());
        assert!(checker
            .check_accessible_authentication_minimum(true)
            .is_none());
    }

    #[test]
    fn test_focus_not_obscured_minimum_visible() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        assert!(checker.check_focus_not_obscured_minimum(true).is_none());
    }

    #[test]
    fn test_focus_not_obscured_minimum_hidden() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let issue = checker.check_focus_not_obscured_minimum(false);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-2.4.11");
    }

    #[test]
    fn test_focus_appearance_passing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let params = FocusIndicatorParams {
            outline_area_px2: 500.0, // large enough
            contrast_ratio: 4.5,
            encloses_component: true,
            component_perimeter_px: 200.0, // requires 400px² minimum
            outline_thickness_px: 2.0,
        };
        assert!(checker.check_focus_appearance(&params).is_none());
    }

    #[test]
    fn test_focus_appearance_failing_area() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let params = FocusIndicatorParams {
            outline_area_px2: 100.0, // too small (needs 200px² for 100px perimeter)
            contrast_ratio: 4.5,
            encloses_component: true,
            component_perimeter_px: 100.0,
            outline_thickness_px: 1.0,
        };
        let issue = checker.check_focus_appearance(&params);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-2.4.13");
    }

    #[test]
    fn test_focus_appearance_failing_contrast() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let params = FocusIndicatorParams {
            outline_area_px2: 1000.0, // area is fine
            contrast_ratio: 1.5,      // contrast too low
            encloses_component: true,
            component_perimeter_px: 100.0,
            outline_thickness_px: 2.0,
        };
        let issue = checker.check_focus_appearance(&params);
        assert!(issue.is_some());
    }

    #[test]
    fn test_dragging_alternatives_with_alternative() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let drag = DragInteractionParams {
            has_single_pointer_alternative: true,
            alternative_description: Some("Click Move Up/Down buttons".to_string()),
            is_cancellable: true,
        };
        assert!(checker.check_dragging_alternatives(&drag).is_none());
    }

    #[test]
    fn test_dragging_alternatives_without_alternative() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let drag = DragInteractionParams {
            has_single_pointer_alternative: false,
            alternative_description: None,
            is_cancellable: true,
        };
        let issue = checker.check_dragging_alternatives(&drag);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-2.5.7");
    }

    #[test]
    fn test_target_size_minimum_passing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        assert!(checker.check_target_size_minimum(24.0, 24.0).is_none());
        assert!(checker.check_target_size_minimum(44.0, 44.0).is_none());
    }

    #[test]
    fn test_target_size_minimum_failing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let issue = checker.check_target_size_minimum(16.0, 16.0);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-2.5.8");
    }

    #[test]
    fn test_consistent_help_passing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        assert!(checker.check_consistent_help(true, true).is_none());
        assert!(checker.check_consistent_help(false, false).is_none()); // no help = no requirement
    }

    #[test]
    fn test_consistent_help_failing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let issue = checker.check_consistent_help(true, false);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-3.2.6");
    }

    #[test]
    fn test_redundant_entry_passing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        assert!(checker.check_redundant_entry(false).is_none());
    }

    #[test]
    fn test_redundant_entry_failing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let issue = checker.check_redundant_entry(true);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-3.3.7");
    }

    #[test]
    fn test_accessible_authentication_passing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        assert!(checker
            .check_accessible_authentication_minimum(false)
            .is_none());
    }

    #[test]
    fn test_accessible_authentication_failing() {
        let checker = WcagChecker::with_version(WcagLevel::AA, WcagVersion::V2_2);
        let issue = checker.check_accessible_authentication_minimum(true);
        assert!(issue.is_some());
        let issue = issue.expect("issue should be present");
        assert_eq!(issue.id, "WCAG-3.3.8");
    }

    #[test]
    fn test_focus_indicator_params_meets_minimum_area() {
        let params = FocusIndicatorParams {
            outline_area_px2: 400.0,
            contrast_ratio: 3.5,
            encloses_component: true,
            component_perimeter_px: 200.0, // requires 400px²
            outline_thickness_px: 2.0,
        };
        assert!(params.meets_minimum_area());

        let small = FocusIndicatorParams {
            outline_area_px2: 100.0,
            ..params.clone()
        };
        assert!(!small.meets_minimum_area());
    }

    #[test]
    fn test_focus_indicator_params_enhanced() {
        let params = FocusIndicatorParams {
            outline_area_px2: 800.0, // needs 200 * 4 = 800
            contrast_ratio: 3.0,
            encloses_component: true,
            component_perimeter_px: 200.0,
            outline_thickness_px: 2.0,
        };
        assert!(params.meets_enhanced_focus());

        let low_contrast = FocusIndicatorParams {
            contrast_ratio: 2.5,
            ..params.clone()
        };
        assert!(!low_contrast.meets_enhanced_focus());
    }
}
