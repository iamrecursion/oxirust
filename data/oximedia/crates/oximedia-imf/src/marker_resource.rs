#![allow(dead_code)]
//! Marker resource handling for IMF Composition Playlists.
//!
//! Markers in IMF (SMPTE ST 2067-3) annotate specific points or ranges
//! on the CPL timeline. They are used for:
//!
//! - **Chapter points** - Navigation markers for playback
//! - **Commercial breaks** - Ad insertion cue points
//! - **Content ratings** - Age-rating boundaries
//! - **Localization cues** - Language/subtitle switch points
//! - **Custom annotations** - Arbitrary producer-defined markers

use std::collections::HashMap;
use std::fmt;

/// Standard IMF marker labels defined by SMPTE ST 2067-3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StandardMarkerLabel {
    /// FFBT - first frame of brand titles.
    FirstFrameBrandTitles,
    /// FFTC - first frame of title credits.
    FirstFrameTitleCredits,
    /// FFOI - first frame of intermission.
    FirstFrameIntermission,
    /// FFEC - first frame of end credits.
    FirstFrameEndCredits,
    /// FFHS - first frame of hors sujet (off-topic content).
    FirstFrameHorsSujet,
    /// FFMC - first frame of moving credits.
    FirstFrameMovingCredits,
    /// LFMC - last frame of moving credits.
    LastFrameMovingCredits,
    /// FFOC - first frame of content.
    FirstFrameOfContent,
    /// LFOC - last frame of content.
    LastFrameOfContent,
    /// FFSP - first frame of special content.
    FirstFrameSpecialContent,
}

impl StandardMarkerLabel {
    /// Return the SMPTE symbol for this marker label.
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::FirstFrameBrandTitles => "FFBT",
            Self::FirstFrameTitleCredits => "FFTC",
            Self::FirstFrameIntermission => "FFOI",
            Self::FirstFrameEndCredits => "FFEC",
            Self::FirstFrameHorsSujet => "FFHS",
            Self::FirstFrameMovingCredits => "FFMC",
            Self::LastFrameMovingCredits => "LFMC",
            Self::FirstFrameOfContent => "FFOC",
            Self::LastFrameOfContent => "LFOC",
            Self::FirstFrameSpecialContent => "FFSP",
        }
    }

    /// Return the human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::FirstFrameBrandTitles => "First frame of brand titles",
            Self::FirstFrameTitleCredits => "First frame of title credits",
            Self::FirstFrameIntermission => "First frame of intermission",
            Self::FirstFrameEndCredits => "First frame of end credits",
            Self::FirstFrameHorsSujet => "First frame of hors sujet",
            Self::FirstFrameMovingCredits => "First frame of moving credits",
            Self::LastFrameMovingCredits => "Last frame of moving credits",
            Self::FirstFrameOfContent => "First frame of content",
            Self::LastFrameOfContent => "Last frame of content",
            Self::FirstFrameSpecialContent => "First frame of special content",
        }
    }

    /// Try to parse a symbol string into a standard marker label.
    pub fn from_symbol(s: &str) -> Option<Self> {
        match s {
            "FFBT" => Some(Self::FirstFrameBrandTitles),
            "FFTC" => Some(Self::FirstFrameTitleCredits),
            "FFOI" => Some(Self::FirstFrameIntermission),
            "FFEC" => Some(Self::FirstFrameEndCredits),
            "FFHS" => Some(Self::FirstFrameHorsSujet),
            "FFMC" => Some(Self::FirstFrameMovingCredits),
            "LFMC" => Some(Self::LastFrameMovingCredits),
            "FFOC" => Some(Self::FirstFrameOfContent),
            "LFOC" => Some(Self::LastFrameOfContent),
            "FFSP" => Some(Self::FirstFrameSpecialContent),
            _ => None,
        }
    }
}

impl fmt::Display for StandardMarkerLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// A marker label, either standard SMPTE or custom.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MarkerLabel {
    /// A standard SMPTE marker label.
    Standard(StandardMarkerLabel),
    /// A custom marker label with a string identifier.
    Custom(String),
}

impl MarkerLabel {
    /// Create a standard marker label.
    pub fn standard(label: StandardMarkerLabel) -> Self {
        Self::Standard(label)
    }

    /// Create a custom marker label.
    pub fn custom(label: impl Into<String>) -> Self {
        Self::Custom(label.into())
    }

    /// Check if this is a standard SMPTE label.
    pub fn is_standard(&self) -> bool {
        matches!(self, Self::Standard(_))
    }

    /// Get the display name of this label.
    pub fn name(&self) -> String {
        match self {
            Self::Standard(s) => s.symbol().to_string(),
            Self::Custom(c) => c.clone(),
        }
    }
}

impl fmt::Display for MarkerLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard(s) => write!(f, "{s}"),
            Self::Custom(c) => write!(f, "{c}"),
        }
    }
}

/// A single marker on the CPL timeline.
#[derive(Clone, Debug)]
pub struct Marker {
    /// Unique identifier for this marker.
    pub id: String,
    /// The marker label.
    pub label: MarkerLabel,
    /// Offset in edit units from the beginning of the enclosing resource.
    pub offset: u64,
    /// Optional scope / annotation string.
    pub scope: Option<String>,
    /// Optional annotation text.
    pub annotation: Option<String>,
}

impl Marker {
    /// Create a new marker.
    pub fn new(id: impl Into<String>, label: MarkerLabel, offset: u64) -> Self {
        Self {
            id: id.into(),
            label,
            offset,
            scope: None,
            annotation: None,
        }
    }

    /// Set the scope.
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Set the annotation.
    pub fn with_annotation(mut self, annotation: impl Into<String>) -> Self {
        self.annotation = Some(annotation.into());
        self
    }
}

/// A marker resource within a CPL segment.
///
/// Contains an ordered list of markers that belong to a specific
/// marker sequence resource.
#[derive(Clone, Debug)]
pub struct MarkerResource {
    /// Resource UUID.
    pub id: String,
    /// Edit rate for this marker resource.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
    /// Intrinsic duration in edit units.
    pub intrinsic_duration: u64,
    /// Entry point offset.
    pub entry_point: u64,
    /// Source duration in edit units.
    pub source_duration: u64,
    /// Markers within this resource.
    pub markers: Vec<Marker>,
}

impl MarkerResource {
    /// Create a new marker resource.
    pub fn new(
        id: impl Into<String>,
        edit_rate_num: u32,
        edit_rate_den: u32,
        intrinsic_duration: u64,
    ) -> Self {
        Self {
            id: id.into(),
            edit_rate_num,
            edit_rate_den,
            intrinsic_duration,
            entry_point: 0,
            source_duration: intrinsic_duration,
            markers: Vec::new(),
        }
    }

    /// Add a marker to this resource.
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
    }

    /// Get the edit rate as a floating-point value.
    #[allow(clippy::cast_precision_loss)]
    pub fn edit_rate_fps(&self) -> f64 {
        if self.edit_rate_den == 0 {
            0.0
        } else {
            self.edit_rate_num as f64 / self.edit_rate_den as f64
        }
    }

    /// Convert an offset in edit units to seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn offset_to_seconds(&self, offset: u64) -> f64 {
        let fps = self.edit_rate_fps();
        if fps <= 0.0 {
            0.0
        } else {
            offset as f64 / fps
        }
    }

    /// Get all markers of a specific label.
    pub fn markers_by_label(&self, label: &MarkerLabel) -> Vec<&Marker> {
        self.markers.iter().filter(|m| &m.label == label).collect()
    }

    /// Get markers sorted by offset.
    pub fn markers_sorted(&self) -> Vec<&Marker> {
        let mut sorted: Vec<&Marker> = self.markers.iter().collect();
        sorted.sort_by_key(|m| m.offset);
        sorted
    }

    /// Check if a marker offset is within the valid range.
    pub fn is_offset_valid(&self, offset: u64) -> bool {
        offset < self.source_duration
    }

    /// Count markers.
    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    /// Validate all markers in this resource.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.edit_rate_num == 0 || self.edit_rate_den == 0 {
            errors.push("Invalid edit rate (zero numerator or denominator)".to_string());
        }

        if self.source_duration > self.intrinsic_duration {
            errors.push(format!(
                "SourceDuration ({}) exceeds IntrinsicDuration ({})",
                self.source_duration, self.intrinsic_duration
            ));
        }

        for marker in &self.markers {
            if marker.offset >= self.source_duration {
                errors.push(format!(
                    "Marker '{}' offset ({}) exceeds SourceDuration ({})",
                    marker.id, marker.offset, self.source_duration
                ));
            }
        }

        // Check for duplicate marker IDs
        let mut seen_ids: HashMap<&str, usize> = HashMap::new();
        for marker in &self.markers {
            *seen_ids.entry(marker.id.as_str()).or_insert(0) += 1;
        }
        for (id, count) in &seen_ids {
            if *count > 1 {
                errors.push(format!("Duplicate marker ID: {id} (appears {count} times)"));
            }
        }

        errors
    }
}

/// A collection of marker resources for a CPL.
#[derive(Clone, Debug, Default)]
pub struct MarkerResourceSet {
    /// All marker resources.
    pub resources: Vec<MarkerResource>,
}

impl MarkerResourceSet {
    /// Create a new empty set.
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
        }
    }

    /// Add a marker resource.
    pub fn add_resource(&mut self, resource: MarkerResource) {
        self.resources.push(resource);
    }

    /// Get total marker count across all resources.
    pub fn total_marker_count(&self) -> usize {
        self.resources
            .iter()
            .map(MarkerResource::marker_count)
            .sum()
    }

    /// Find all markers with a specific label across all resources.
    pub fn find_by_label(&self, label: &MarkerLabel) -> Vec<(&MarkerResource, &Marker)> {
        let mut results = Vec::new();
        for resource in &self.resources {
            for marker in &resource.markers {
                if &marker.label == label {
                    results.push((resource, marker));
                }
            }
        }
        results
    }

    /// Validate all resources.
    pub fn validate_all(&self) -> Vec<String> {
        let mut all_errors = Vec::new();
        for (idx, resource) in self.resources.iter().enumerate() {
            for error in resource.validate() {
                all_errors.push(format!("Resource[{idx}]: {error}"));
            }
        }
        all_errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_resource() -> MarkerResource {
        let mut res = MarkerResource::new("mr-001", 24, 1, 240);
        res.add_marker(Marker::new(
            "m1",
            MarkerLabel::standard(StandardMarkerLabel::FirstFrameOfContent),
            0,
        ));
        res.add_marker(Marker::new(
            "m2",
            MarkerLabel::standard(StandardMarkerLabel::FirstFrameEndCredits),
            200,
        ));
        res.add_marker(Marker::new(
            "m3",
            MarkerLabel::standard(StandardMarkerLabel::LastFrameOfContent),
            239,
        ));
        res
    }

    #[test]
    fn test_standard_marker_label_symbol() {
        assert_eq!(StandardMarkerLabel::FirstFrameOfContent.symbol(), "FFOC");
        assert_eq!(StandardMarkerLabel::LastFrameOfContent.symbol(), "LFOC");
        assert_eq!(StandardMarkerLabel::FirstFrameEndCredits.symbol(), "FFEC");
    }

    #[test]
    fn test_standard_marker_from_symbol() {
        assert_eq!(
            StandardMarkerLabel::from_symbol("FFOC"),
            Some(StandardMarkerLabel::FirstFrameOfContent)
        );
        assert_eq!(
            StandardMarkerLabel::from_symbol("LFOC"),
            Some(StandardMarkerLabel::LastFrameOfContent)
        );
        assert!(StandardMarkerLabel::from_symbol("INVALID").is_none());
    }

    #[test]
    fn test_marker_label_display() {
        let standard = MarkerLabel::standard(StandardMarkerLabel::FirstFrameOfContent);
        assert_eq!(format!("{standard}"), "FFOC");

        let custom = MarkerLabel::custom("MY_MARKER");
        assert_eq!(format!("{custom}"), "MY_MARKER");
    }

    #[test]
    fn test_marker_label_is_standard() {
        let standard = MarkerLabel::standard(StandardMarkerLabel::FirstFrameOfContent);
        assert!(standard.is_standard());

        let custom = MarkerLabel::custom("test");
        assert!(!custom.is_standard());
    }

    #[test]
    fn test_marker_creation() {
        let marker = Marker::new(
            "mk-001",
            MarkerLabel::standard(StandardMarkerLabel::FirstFrameOfContent),
            100,
        )
        .with_scope("http://www.smpte-ra.org/schemas/2067-3/2016")
        .with_annotation("Content starts here");

        assert_eq!(marker.id, "mk-001");
        assert_eq!(marker.offset, 100);
        assert!(marker.scope.is_some());
        assert!(marker.annotation.is_some());
    }

    #[test]
    fn test_marker_resource_new() {
        let res = MarkerResource::new("mr-001", 24, 1, 240);
        assert_eq!(res.id, "mr-001");
        assert_eq!(res.edit_rate_num, 24);
        assert_eq!(res.edit_rate_den, 1);
        assert_eq!(res.intrinsic_duration, 240);
        assert_eq!(res.marker_count(), 0);
    }

    #[test]
    fn test_marker_resource_add_and_count() {
        let res = make_test_resource();
        assert_eq!(res.marker_count(), 3);
    }

    #[test]
    fn test_edit_rate_fps() {
        let res = MarkerResource::new("mr-001", 24000, 1001, 100);
        let fps = res.edit_rate_fps();
        assert!((fps - 23.976).abs() < 0.1);
    }

    #[test]
    fn test_offset_to_seconds() {
        let res = MarkerResource::new("mr-001", 24, 1, 240);
        let secs = res.offset_to_seconds(48);
        assert!((secs - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_markers_by_label() {
        let res = make_test_resource();
        let ffoc_label = MarkerLabel::standard(StandardMarkerLabel::FirstFrameOfContent);
        let found = res.markers_by_label(&ffoc_label);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "m1");
    }

    #[test]
    fn test_markers_sorted() {
        let mut res = MarkerResource::new("mr-001", 24, 1, 240);
        res.add_marker(Marker::new("m-late", MarkerLabel::custom("LATE"), 200));
        res.add_marker(Marker::new("m-early", MarkerLabel::custom("EARLY"), 10));
        res.add_marker(Marker::new("m-mid", MarkerLabel::custom("MID"), 100));
        let sorted = res.markers_sorted();
        assert_eq!(sorted[0].id, "m-early");
        assert_eq!(sorted[1].id, "m-mid");
        assert_eq!(sorted[2].id, "m-late");
    }

    #[test]
    fn test_is_offset_valid() {
        let res = MarkerResource::new("mr-001", 24, 1, 240);
        assert!(res.is_offset_valid(0));
        assert!(res.is_offset_valid(239));
        assert!(!res.is_offset_valid(240));
    }

    #[test]
    fn test_validate_valid_resource() {
        let res = make_test_resource();
        let errors = res.validate();
        assert!(errors.is_empty(), "Expected no errors: {:?}", errors);
    }

    #[test]
    fn test_validate_marker_out_of_range() {
        let mut res = MarkerResource::new("mr-001", 24, 1, 100);
        res.add_marker(Marker::new("m-bad", MarkerLabel::custom("OOB"), 200));
        let errors = res.validate();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("exceeds SourceDuration"));
    }

    #[test]
    fn test_marker_resource_set() {
        let mut set = MarkerResourceSet::new();
        set.add_resource(make_test_resource());
        set.add_resource(make_test_resource());
        assert_eq!(set.total_marker_count(), 6);
    }

    #[test]
    fn test_find_by_label_across_resources() {
        let mut set = MarkerResourceSet::new();
        set.add_resource(make_test_resource());
        set.add_resource(make_test_resource());
        let ffoc = MarkerLabel::standard(StandardMarkerLabel::FirstFrameOfContent);
        let found = set.find_by_label(&ffoc);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_validate_all_resources() {
        let mut set = MarkerResourceSet::new();
        set.add_resource(make_test_resource());
        // Add one with invalid edit rate
        let bad_res = MarkerResource::new("mr-bad", 0, 0, 100);
        set.add_resource(bad_res);
        let errors = set.validate_all();
        assert!(!errors.is_empty());
    }
}
