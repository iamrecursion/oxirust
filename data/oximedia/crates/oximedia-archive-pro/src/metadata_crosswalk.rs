//! Metadata crosswalk — mapping between different archival metadata schemas.
//!
//! Provides a rule-based engine for transforming metadata fields from one
//! schema (e.g. Dublin Core, PBCore) into another (e.g. PREMIS or METS).
//!
//! ## PBCore 2.1 Support
//!
//! PBCore (Public Broadcasting Core) 2.1 field mappings are registered via
//! [`pbcore_to_internal_crosswalk`] and [`internal_to_pbcore_crosswalk`].
//! The "internal" scheme uses a flat normalized field set aligned to common
//! media metadata vocabulary (title, identifier, description, subject, etc.).

#![allow(dead_code)]

/// Supported archival metadata schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataScheme {
    /// Dublin Core — a simple, 15-element metadata vocabulary.
    DublinCore,
    /// Encoded Archival Description — used for archival finding aids.
    Ead,
    /// PREMIS — Preservation Metadata: Implementation Strategies.
    Premis,
    /// METS — Metadata Encoding and Transmission Standard.
    Mets,
    /// PBCore 2.1 — Public Broadcasting Metadata Dictionary.
    PbCore,
    /// Internal normalized field model (flat key-value for crosswalk targets).
    Internal,
}

impl MetadataScheme {
    /// Returns the XML namespace URI for this scheme.
    #[must_use]
    pub const fn namespace(&self) -> &'static str {
        match self {
            Self::DublinCore => "http://purl.org/dc/elements/1.1/",
            Self::Ead => "urn:isbn:1-931666-22-9",
            Self::Premis => "http://www.loc.gov/premis/v3",
            Self::Mets => "http://www.loc.gov/METS/",
            Self::PbCore => "http://www.pbcore.org/PBCore/PBCoreNamespace.html",
            Self::Internal => "",
        }
    }

    /// Returns the conventional short prefix for this scheme's namespace.
    #[must_use]
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::DublinCore => "dc",
            Self::Ead => "ead",
            Self::Premis => "premis",
            Self::Mets => "mets",
            Self::PbCore => "pb",
            Self::Internal => "internal",
        }
    }
}

/// A single rule mapping a source field to a target field in a destination scheme.
#[derive(Debug, Clone)]
pub struct CrosswalkRule {
    /// The source scheme from which the field is read.
    pub source_scheme: MetadataScheme,
    /// The field name in the source scheme.
    pub source_field: String,
    /// The destination scheme into which the field is written.
    pub target_scheme: MetadataScheme,
    /// The field name in the target scheme.
    pub target_field_name: String,
    /// Optional static prefix prepended to every transformed value.
    pub value_prefix: Option<String>,
}

impl CrosswalkRule {
    /// Creates a new crosswalk rule.
    #[must_use]
    pub fn new(
        source_scheme: MetadataScheme,
        source_field: impl Into<String>,
        target_scheme: MetadataScheme,
        target_field: impl Into<String>,
    ) -> Self {
        Self {
            source_scheme,
            source_field: source_field.into(),
            target_scheme,
            target_field_name: target_field.into(),
            value_prefix: None,
        }
    }

    /// Returns the target field name.
    #[must_use]
    pub fn target_field(&self) -> &str {
        &self.target_field_name
    }

    /// Attaches an optional value prefix to this rule.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.value_prefix = Some(prefix.into());
        self
    }

    /// Applies this rule to a source value, returning the transformed value.
    #[must_use]
    pub fn transform_value(&self, value: &str) -> String {
        match &self.value_prefix {
            Some(prefix) => format!("{prefix}{value}"),
            None => value.to_owned(),
        }
    }
}

/// A metadata crosswalk that holds a collection of rules and applies them to
/// transform a metadata map from one scheme into another.
#[derive(Debug, Default)]
pub struct MetadataCrosswalk {
    rules: Vec<CrosswalkRule>,
}

impl MetadataCrosswalk {
    /// Creates an empty crosswalk.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a rule to this crosswalk.
    pub fn add_rule(&mut self, rule: CrosswalkRule) {
        self.rules.push(rule);
    }

    /// Returns the total number of rules registered.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Transforms a source metadata map (field → value) into a target map using
    /// all rules that match the given source and target schemes.
    ///
    /// Fields present in the source that have no matching rule are silently
    /// dropped from the result.
    #[must_use]
    pub fn transform(
        &self,
        source: &std::collections::HashMap<String, String>,
        source_scheme: MetadataScheme,
        target_scheme: MetadataScheme,
    ) -> std::collections::HashMap<String, String> {
        let mut result = std::collections::HashMap::new();
        for rule in &self.rules {
            if rule.source_scheme != source_scheme || rule.target_scheme != target_scheme {
                continue;
            }
            if let Some(value) = source.get(&rule.source_field) {
                result.insert(rule.target_field_name.clone(), rule.transform_value(value));
            }
        }
        result
    }

    /// Returns a slice of all rules.
    #[must_use]
    pub fn rules(&self) -> &[CrosswalkRule] {
        &self.rules
    }
}

// ---------------------------------------------------------------------------
// PBCore 2.1 crosswalk builders
// ---------------------------------------------------------------------------

/// Builds a [`MetadataCrosswalk`] mapping PBCore 2.1 elements to the internal
/// normalized field model.
///
/// ## PBCore 2.1 → Internal field mapping
///
/// | PBCore 2.1 element                                        | Internal field     |
/// |-----------------------------------------------------------|--------------------|
/// | `pbcoreTitle`                                             | `title`            |
/// | `pbcoreIdentifier`                                        | `identifier`       |
/// | `pbcoreDescription`                                       | `description`      |
/// | `pbcoreSubject`                                           | `subject`          |
/// | `pbcoreInstantiation/instantiationDuration`               | `duration`         |
/// | `pbcoreInstantiation/instantiationMediaType`              | `media_type`       |
/// | `pbcoreInstantiation/instantiationFileSize`               | `file_size`        |
/// | `pbcoreInstantiation/instantiationFrameRate`              | `frame_rate`       |
/// | `pbcoreInstantiation/instantiationAudioSamplingRate`      | `sample_rate`      |
/// | `pbcoreEssenceTrack/essenceTrackType`                     | `track_type`       |
#[must_use]
pub fn pbcore_to_internal_crosswalk() -> MetadataCrosswalk {
    let mut cw = MetadataCrosswalk::new();

    // Descriptive metadata
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreTitle",
        MetadataScheme::Internal,
        "title",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreIdentifier",
        MetadataScheme::Internal,
        "identifier",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreDescription",
        MetadataScheme::Internal,
        "description",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreSubject",
        MetadataScheme::Internal,
        "subject",
    ));

    // Instantiation metadata
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationDuration",
        MetadataScheme::Internal,
        "duration",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationMediaType",
        MetadataScheme::Internal,
        "media_type",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationFileSize",
        MetadataScheme::Internal,
        "file_size",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationFrameRate",
        MetadataScheme::Internal,
        "frame_rate",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationAudioSamplingRate",
        MetadataScheme::Internal,
        "sample_rate",
    ));

    // Essence track metadata
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::PbCore,
        "pbcoreEssenceTrack/essenceTrackType",
        MetadataScheme::Internal,
        "track_type",
    ));

    cw
}

/// Builds a [`MetadataCrosswalk`] mapping the internal normalized field model
/// back to PBCore 2.1 elements.
///
/// This is the inverse of [`pbcore_to_internal_crosswalk`].
#[must_use]
pub fn internal_to_pbcore_crosswalk() -> MetadataCrosswalk {
    let mut cw = MetadataCrosswalk::new();

    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "title",
        MetadataScheme::PbCore,
        "pbcoreTitle",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "identifier",
        MetadataScheme::PbCore,
        "pbcoreIdentifier",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "description",
        MetadataScheme::PbCore,
        "pbcoreDescription",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "subject",
        MetadataScheme::PbCore,
        "pbcoreSubject",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "duration",
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationDuration",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "media_type",
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationMediaType",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "file_size",
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationFileSize",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "frame_rate",
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationFrameRate",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "sample_rate",
        MetadataScheme::PbCore,
        "pbcoreInstantiation/instantiationAudioSamplingRate",
    ));
    cw.add_rule(CrosswalkRule::new(
        MetadataScheme::Internal,
        "track_type",
        MetadataScheme::PbCore,
        "pbcoreEssenceTrack/essenceTrackType",
    ));

    cw
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn dc_to_premis_crosswalk() -> MetadataCrosswalk {
        let mut cw = MetadataCrosswalk::new();
        cw.add_rule(CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "title",
            MetadataScheme::Premis,
            "objectIdentifierValue",
        ));
        cw.add_rule(CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "creator",
            MetadataScheme::Premis,
            "creatingApplicationName",
        ));
        cw.add_rule(CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "date",
            MetadataScheme::Premis,
            "dateCreatedByApplication",
        ));
        cw
    }

    #[test]
    fn test_scheme_namespaces_nonempty() {
        let schemes = [
            MetadataScheme::DublinCore,
            MetadataScheme::Ead,
            MetadataScheme::Premis,
            MetadataScheme::Mets,
        ];
        for s in schemes {
            assert!(!s.namespace().is_empty());
        }
    }

    #[test]
    fn test_scheme_prefixes_nonempty() {
        assert_eq!(MetadataScheme::DublinCore.prefix(), "dc");
        assert_eq!(MetadataScheme::Premis.prefix(), "premis");
        assert_eq!(MetadataScheme::Mets.prefix(), "mets");
        assert_eq!(MetadataScheme::Ead.prefix(), "ead");
    }

    #[test]
    fn test_crosswalk_rule_target_field() {
        let rule = CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "title",
            MetadataScheme::Premis,
            "objectIdentifierValue",
        );
        assert_eq!(rule.target_field(), "objectIdentifierValue");
    }

    #[test]
    fn test_crosswalk_rule_transform_value_no_prefix() {
        let rule = CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "title",
            MetadataScheme::Premis,
            "objectIdentifierValue",
        );
        assert_eq!(rule.transform_value("MyFilm"), "MyFilm");
    }

    #[test]
    fn test_crosswalk_rule_transform_value_with_prefix() {
        let rule = CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "identifier",
            MetadataScheme::Premis,
            "objectIdentifierValue",
        )
        .with_prefix("ark:/12148/");
        assert_eq!(rule.transform_value("abc123"), "ark:/12148/abc123");
    }

    #[test]
    fn test_crosswalk_rule_count() {
        let cw = dc_to_premis_crosswalk();
        assert_eq!(cw.rule_count(), 3);
    }

    #[test]
    fn test_crosswalk_transform_known_fields() {
        let cw = dc_to_premis_crosswalk();
        let mut source = HashMap::new();
        source.insert("title".to_string(), "Sunset Film".to_string());
        source.insert("creator".to_string(), "AcmeCam".to_string());
        source.insert("date".to_string(), "2024-01-15".to_string());
        let result = cw.transform(&source, MetadataScheme::DublinCore, MetadataScheme::Premis);
        assert_eq!(
            result
                .get("objectIdentifierValue")
                .expect("operation should succeed"),
            "Sunset Film"
        );
        assert_eq!(
            result
                .get("creatingApplicationName")
                .expect("operation should succeed"),
            "AcmeCam"
        );
        assert_eq!(
            result
                .get("dateCreatedByApplication")
                .expect("operation should succeed"),
            "2024-01-15"
        );
    }

    #[test]
    fn test_crosswalk_transform_missing_source_field() {
        let cw = dc_to_premis_crosswalk();
        let source: HashMap<String, String> = HashMap::new();
        let result = cw.transform(&source, MetadataScheme::DublinCore, MetadataScheme::Premis);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crosswalk_transform_wrong_scheme_ignored() {
        let cw = dc_to_premis_crosswalk();
        let mut source = HashMap::new();
        source.insert("title".to_string(), "Film".to_string());
        // Using wrong source scheme — no rules should match.
        let result = cw.transform(&source, MetadataScheme::Mets, MetadataScheme::Premis);
        assert!(result.is_empty());
    }

    #[test]
    fn test_crosswalk_rules_slice() {
        let cw = dc_to_premis_crosswalk();
        assert_eq!(cw.rules().len(), 3);
    }

    #[test]
    fn test_crosswalk_empty_default() {
        let cw = MetadataCrosswalk::default();
        assert_eq!(cw.rule_count(), 0);
    }

    #[test]
    fn test_add_multiple_rules() {
        let mut cw = MetadataCrosswalk::new();
        for i in 0..5 {
            cw.add_rule(CrosswalkRule::new(
                MetadataScheme::DublinCore,
                format!("field{i}"),
                MetadataScheme::Mets,
                format!("target{i}"),
            ));
        }
        assert_eq!(cw.rule_count(), 5);
    }

    #[test]
    fn test_scheme_namespace_dc() {
        assert_eq!(
            MetadataScheme::DublinCore.namespace(),
            "http://purl.org/dc/elements/1.1/"
        );
    }

    #[test]
    fn test_transform_partial_match() {
        let mut cw = MetadataCrosswalk::new();
        cw.add_rule(CrosswalkRule::new(
            MetadataScheme::DublinCore,
            "subject",
            MetadataScheme::Mets,
            "dmSubject",
        ));
        let mut source = HashMap::new();
        source.insert("subject".to_string(), "documentary".to_string());
        source.insert("unknown_field".to_string(), "foo".to_string());
        let result = cw.transform(&source, MetadataScheme::DublinCore, MetadataScheme::Mets);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.get("dmSubject").expect("operation should succeed"),
            "documentary"
        );
    }

    // ── PBCore 2.1 tests ────────────────────────────────────────────────────

    #[test]
    fn test_pbcore_scheme_namespace() {
        assert_eq!(
            MetadataScheme::PbCore.namespace(),
            "http://www.pbcore.org/PBCore/PBCoreNamespace.html"
        );
        assert_eq!(MetadataScheme::PbCore.prefix(), "pb");
    }

    #[test]
    fn test_pbcore_crosswalk_to_internal() {
        let cw = pbcore_to_internal_crosswalk();
        let mut source = HashMap::new();
        source.insert(
            "pbcoreTitle".to_string(),
            "The Great Archive Film".to_string(),
        );
        source.insert(
            "pbcoreIdentifier".to_string(),
            "urn:pbcore:test:001".to_string(),
        );
        source.insert(
            "pbcoreDescription".to_string(),
            "A test broadcast recording.".to_string(),
        );
        source.insert("pbcoreSubject".to_string(), "broadcasting".to_string());
        source.insert(
            "pbcoreInstantiation/instantiationMediaType".to_string(),
            "Moving Image".to_string(),
        );
        source.insert(
            "pbcoreInstantiation/instantiationDuration".to_string(),
            "00:01:30.000".to_string(),
        );

        let result = cw.transform(&source, MetadataScheme::PbCore, MetadataScheme::Internal);

        assert_eq!(
            result.get("title").map(String::as_str),
            Some("The Great Archive Film"),
            "pbcoreTitle must map to internal 'title'"
        );
        assert_eq!(
            result.get("identifier").map(String::as_str),
            Some("urn:pbcore:test:001"),
            "pbcoreIdentifier must map to internal 'identifier'"
        );
        assert_eq!(
            result.get("description").map(String::as_str),
            Some("A test broadcast recording."),
            "pbcoreDescription must map to internal 'description'"
        );
        assert_eq!(
            result.get("subject").map(String::as_str),
            Some("broadcasting"),
            "pbcoreSubject must map to internal 'subject'"
        );
        assert_eq!(
            result.get("media_type").map(String::as_str),
            Some("Moving Image"),
            "instantiationMediaType must map to internal 'media_type'"
        );
        assert_eq!(
            result.get("duration").map(String::as_str),
            Some("00:01:30.000"),
            "instantiationDuration must map to internal 'duration'"
        );
    }

    #[test]
    fn test_pbcore_crosswalk_round_trip() {
        let forward = pbcore_to_internal_crosswalk();
        let backward = internal_to_pbcore_crosswalk();

        let mut pbcore_source = HashMap::new();
        pbcore_source.insert("pbcoreTitle".to_string(), "Round-Trip Test".to_string());
        pbcore_source.insert(
            "pbcoreIdentifier".to_string(),
            "urn:pbcore:roundtrip:42".to_string(),
        );
        pbcore_source.insert(
            "pbcoreInstantiation/instantiationFrameRate".to_string(),
            "29.97".to_string(),
        );
        pbcore_source.insert(
            "pbcoreInstantiation/instantiationAudioSamplingRate".to_string(),
            "48000".to_string(),
        );

        // PBCore → Internal
        let internal = forward.transform(
            &pbcore_source,
            MetadataScheme::PbCore,
            MetadataScheme::Internal,
        );
        assert_eq!(
            internal.get("title").map(String::as_str),
            Some("Round-Trip Test")
        );
        assert_eq!(
            internal.get("identifier").map(String::as_str),
            Some("urn:pbcore:roundtrip:42")
        );
        assert_eq!(
            internal.get("frame_rate").map(String::as_str),
            Some("29.97")
        );
        assert_eq!(
            internal.get("sample_rate").map(String::as_str),
            Some("48000")
        );

        // Internal → PBCore
        let pbcore_result =
            backward.transform(&internal, MetadataScheme::Internal, MetadataScheme::PbCore);
        assert_eq!(
            pbcore_result.get("pbcoreTitle").map(String::as_str),
            Some("Round-Trip Test"),
            "title must survive round-trip back to pbcoreTitle"
        );
        assert_eq!(
            pbcore_result.get("pbcoreIdentifier").map(String::as_str),
            Some("urn:pbcore:roundtrip:42"),
            "identifier must survive round-trip back to pbcoreIdentifier"
        );
        assert_eq!(
            pbcore_result
                .get("pbcoreInstantiation/instantiationFrameRate")
                .map(String::as_str),
            Some("29.97"),
            "frame_rate must survive round-trip"
        );
    }

    #[test]
    fn test_pbcore_crosswalk_rule_count() {
        let cw = pbcore_to_internal_crosswalk();
        assert_eq!(
            cw.rule_count(),
            10,
            "PBCore crosswalk must have 10 mapping rules"
        );
        let cw_inv = internal_to_pbcore_crosswalk();
        assert_eq!(
            cw_inv.rule_count(),
            10,
            "Inverse PBCore crosswalk must have 10 rules"
        );
    }

    #[test]
    fn test_pbcore_essence_track_type_mapping() {
        let cw = pbcore_to_internal_crosswalk();
        let mut source = HashMap::new();
        source.insert(
            "pbcoreEssenceTrack/essenceTrackType".to_string(),
            "Video".to_string(),
        );
        let result = cw.transform(&source, MetadataScheme::PbCore, MetadataScheme::Internal);
        assert_eq!(
            result.get("track_type").map(String::as_str),
            Some("Video"),
            "essenceTrackType must map to internal 'track_type'"
        );
    }
}
