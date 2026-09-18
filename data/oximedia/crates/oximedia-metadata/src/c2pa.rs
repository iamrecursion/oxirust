//! C2PA (Coalition for Content Provenance and Authenticity) metadata support.
//!
//! This module implements the C2PA specification for content authenticity
//! and provenance metadata.  C2PA provides a standard for certifying the
//! source and history of media content, enabling trust signals for
//! AI-generated, edited, and original media.
//!
//! # Overview
//!
//! The C2PA manifest store contains one or more *manifests*, each describing
//! a set of *assertions* about the content and optionally a *claim signature*.
//!
//! Key concepts:
//!
//! - **Manifest**: A signed collection of assertions about an asset.
//! - **Assertion**: A statement about the asset (e.g., creation tool,
//!   editing actions, thumbnail, ingredients).
//! - **Claim**: The top-level structure that binds assertions together.
//! - **Claim Signature**: A COSE Sign1 signature over the claim.
//! - **Ingredient**: A reference to a parent/source asset (for derivative works).
//! - **Action**: A specific edit or transformation applied to the asset.
//!
//! # Pure Rust
//!
//! This implementation is 100% pure Rust with no C/Fortran dependencies.

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::collections::HashMap;

// ---- C2PA Action Types (from C2PA specification) ----

/// Well-known C2PA action identifiers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum C2paAction {
    /// The asset was created from scratch.
    Created,
    /// The asset was edited / modified.
    Edited,
    /// The asset was cropped.
    Cropped,
    /// The asset was resized.
    Resized,
    /// Color or tone adjustments were applied.
    ColorAdjusted,
    /// The asset was converted to a different format.
    Converted,
    /// The asset was published or distributed.
    Published,
    /// AI/ML generation or modification.
    AiGenerated,
    /// AI/ML training data usage.
    AiTrained,
    /// Watermark was applied.
    Watermarked,
    /// Redaction was applied (e.g., face blurring).
    Redacted,
    /// Custom / vendor-specific action.
    Custom(String),
}

impl C2paAction {
    /// C2PA action URI string.
    pub fn uri(&self) -> String {
        match self {
            Self::Created => "c2pa.created".to_string(),
            Self::Edited => "c2pa.edited".to_string(),
            Self::Cropped => "c2pa.cropped".to_string(),
            Self::Resized => "c2pa.resized".to_string(),
            Self::ColorAdjusted => "c2pa.color_adjusted".to_string(),
            Self::Converted => "c2pa.converted".to_string(),
            Self::Published => "c2pa.published".to_string(),
            Self::AiGenerated => "c2pa.ai_generated".to_string(),
            Self::AiTrained => "c2pa.ai_trained".to_string(),
            Self::Watermarked => "c2pa.watermarked".to_string(),
            Self::Redacted => "c2pa.redacted".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }

    /// Parse from a URI string.
    pub fn from_uri(uri: &str) -> Self {
        match uri {
            "c2pa.created" => Self::Created,
            "c2pa.edited" => Self::Edited,
            "c2pa.cropped" => Self::Cropped,
            "c2pa.resized" => Self::Resized,
            "c2pa.color_adjusted" => Self::ColorAdjusted,
            "c2pa.converted" => Self::Converted,
            "c2pa.published" => Self::Published,
            "c2pa.ai_generated" => Self::AiGenerated,
            "c2pa.ai_trained" => Self::AiTrained,
            "c2pa.watermarked" => Self::Watermarked,
            "c2pa.redacted" => Self::Redacted,
            other => Self::Custom(other.to_string()),
        }
    }

    /// Returns true if the action involves AI/ML.
    pub fn is_ai_related(&self) -> bool {
        matches!(self, Self::AiGenerated | Self::AiTrained)
    }

    /// Returns true if the action is potentially destructive.
    pub fn is_destructive(&self) -> bool {
        matches!(
            self,
            Self::Edited | Self::Cropped | Self::Resized | Self::Redacted
        )
    }
}

// ---- C2PA Assertion ----

/// A C2PA assertion — a single statement about the asset.
#[derive(Debug, Clone)]
pub struct C2paAssertion {
    /// Assertion label (e.g., "c2pa.actions", "stds.schema_org.CreativeWork").
    pub label: String,
    /// Assertion data as key-value pairs.
    pub data: HashMap<String, String>,
}

impl C2paAssertion {
    /// Create a new assertion with the given label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            data: HashMap::new(),
        }
    }

    /// Set a data field.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.data.insert(key.into(), value.into());
    }

    /// Get a data field.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }

    /// Create an actions assertion from a list of C2PA actions.
    pub fn from_actions(actions: &[C2paActionEntry]) -> Self {
        let mut assertion = Self::new("c2pa.actions");
        for (i, entry) in actions.iter().enumerate() {
            let prefix = format!("action_{i}");
            assertion.set(format!("{prefix}.action"), entry.action.uri());
            if let Some(ref when) = entry.when {
                assertion.set(format!("{prefix}.when"), when.clone());
            }
            if let Some(ref sw) = entry.software_agent {
                assertion.set(format!("{prefix}.softwareAgent"), sw.clone());
            }
            if let Some(ref reason) = entry.reason {
                assertion.set(format!("{prefix}.reason"), reason.clone());
            }
        }
        assertion.set("action_count", actions.len().to_string());
        assertion
    }

    /// Create a creative work assertion (schema.org metadata).
    pub fn creative_work(
        title: Option<&str>,
        author: Option<&str>,
        date_created: Option<&str>,
    ) -> Self {
        let mut assertion = Self::new("stds.schema_org.CreativeWork");
        if let Some(t) = title {
            assertion.set("name", t);
        }
        if let Some(a) = author {
            assertion.set("author", a);
        }
        if let Some(d) = date_created {
            assertion.set("dateCreated", d);
        }
        assertion
    }
}

/// A single action entry within a C2PA actions assertion.
#[derive(Debug, Clone)]
pub struct C2paActionEntry {
    /// The action performed.
    pub action: C2paAction,
    /// ISO 8601 timestamp of when the action was performed.
    pub when: Option<String>,
    /// Software agent that performed the action.
    pub software_agent: Option<String>,
    /// Human-readable reason for the action.
    pub reason: Option<String>,
}

impl C2paActionEntry {
    /// Create a new action entry.
    pub fn new(action: C2paAction) -> Self {
        Self {
            action,
            when: None,
            software_agent: None,
            reason: None,
        }
    }

    /// Set the timestamp.
    pub fn with_when(mut self, when: impl Into<String>) -> Self {
        self.when = Some(when.into());
        self
    }

    /// Set the software agent.
    pub fn with_software_agent(mut self, agent: impl Into<String>) -> Self {
        self.software_agent = Some(agent.into());
        self
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

// ---- C2PA Ingredient ----

/// A reference to a source/parent asset used to create the current asset.
#[derive(Debug, Clone)]
pub struct C2paIngredient {
    /// Title of the ingredient asset.
    pub title: String,
    /// MIME type of the ingredient (e.g., "image/jpeg").
    pub format: String,
    /// Instance ID (unique identifier for this particular version).
    pub instance_id: String,
    /// Document ID (shared across versions of the same document).
    pub document_id: Option<String>,
    /// Relationship to the current asset.
    pub relationship: IngredientRelationship,
    /// Hash of the ingredient data (for integrity verification).
    pub hash: Option<String>,
}

/// How an ingredient relates to the current asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngredientRelationship {
    /// The ingredient is the parent (this asset is derived from it).
    ParentOf,
    /// The ingredient is a component (embedded or referenced).
    ComponentOf,
}

impl C2paIngredient {
    /// Create a new ingredient.
    pub fn new(
        title: impl Into<String>,
        format: impl Into<String>,
        instance_id: impl Into<String>,
        relationship: IngredientRelationship,
    ) -> Self {
        Self {
            title: title.into(),
            format: format.into(),
            instance_id: instance_id.into(),
            document_id: None,
            relationship,
            hash: None,
        }
    }

    /// Set the document ID.
    pub fn with_document_id(mut self, id: impl Into<String>) -> Self {
        self.document_id = Some(id.into());
        self
    }

    /// Set the hash.
    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.hash = Some(hash.into());
        self
    }
}

// ---- C2PA Claim ----

/// A C2PA claim — the top-level structure binding assertions.
#[derive(Debug, Clone)]
pub struct C2paClaim {
    /// The claim generator (software that created the claim).
    pub claim_generator: String,
    /// Title of the asset.
    pub title: Option<String>,
    /// MIME format of the asset.
    pub format: Option<String>,
    /// Instance ID.
    pub instance_id: Option<String>,
    /// Assertions bound by this claim.
    pub assertions: Vec<C2paAssertion>,
    /// Ingredients (source assets).
    pub ingredients: Vec<C2paIngredient>,
}

impl C2paClaim {
    /// Create a new claim.
    pub fn new(claim_generator: impl Into<String>) -> Self {
        Self {
            claim_generator: claim_generator.into(),
            title: None,
            format: None,
            instance_id: None,
            assertions: Vec::new(),
            ingredients: Vec::new(),
        }
    }

    /// Set the asset title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the asset format.
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set the instance ID.
    pub fn with_instance_id(mut self, id: impl Into<String>) -> Self {
        self.instance_id = Some(id.into());
        self
    }

    /// Add an assertion.
    pub fn add_assertion(&mut self, assertion: C2paAssertion) {
        self.assertions.push(assertion);
    }

    /// Add an ingredient.
    pub fn add_ingredient(&mut self, ingredient: C2paIngredient) {
        self.ingredients.push(ingredient);
    }

    /// Check whether any assertion references AI-generated content.
    pub fn has_ai_content(&self) -> bool {
        self.assertions.iter().any(|a| {
            a.label == "c2pa.actions"
                && a.data
                    .values()
                    .any(|v| v == "c2pa.ai_generated" || v == "c2pa.ai_trained")
        })
    }

    /// Return the number of assertions.
    pub fn assertion_count(&self) -> usize {
        self.assertions.len()
    }

    /// Return the number of ingredients.
    pub fn ingredient_count(&self) -> usize {
        self.ingredients.len()
    }
}

// ---- C2PA Manifest ----

/// A C2PA manifest containing a claim and its signature.
#[derive(Debug, Clone)]
pub struct C2paManifest {
    /// The claim.
    pub claim: C2paClaim,
    /// Signature algorithm identifier (e.g., "ES256", "PS256").
    pub signature_algorithm: Option<String>,
    /// The raw signature bytes (COSE Sign1).
    pub signature: Option<Vec<u8>>,
    /// Certificate chain (PEM or DER, for verification).
    pub certificate_chain: Vec<Vec<u8>>,
}

impl C2paManifest {
    /// Create a new manifest with the given claim.
    pub fn new(claim: C2paClaim) -> Self {
        Self {
            claim,
            signature_algorithm: None,
            signature: None,
            certificate_chain: Vec::new(),
        }
    }

    /// Set the signature algorithm.
    pub fn with_algorithm(mut self, alg: impl Into<String>) -> Self {
        self.signature_algorithm = Some(alg.into());
        self
    }

    /// Set the signature bytes.
    pub fn with_signature(mut self, sig: Vec<u8>) -> Self {
        self.signature = Some(sig);
        self
    }

    /// Add a certificate to the chain.
    pub fn add_certificate(&mut self, cert: Vec<u8>) {
        self.certificate_chain.push(cert);
    }

    /// Returns true if the manifest has a signature.
    pub fn is_signed(&self) -> bool {
        self.signature.is_some()
    }
}

// ---- C2PA Manifest Store ----

/// A C2PA manifest store containing one or more manifests.
///
/// The *active manifest* is the one that applies to the current asset.
#[derive(Debug, Clone)]
pub struct C2paManifestStore {
    /// All manifests, keyed by label.
    manifests: HashMap<String, C2paManifest>,
    /// The label of the active manifest.
    active_manifest: Option<String>,
}

impl C2paManifestStore {
    /// Create an empty manifest store.
    pub fn new() -> Self {
        Self {
            manifests: HashMap::new(),
            active_manifest: None,
        }
    }

    /// Add a manifest with the given label and set it as active.
    pub fn add_manifest(&mut self, label: impl Into<String>, manifest: C2paManifest) {
        let lbl = label.into();
        self.active_manifest = Some(lbl.clone());
        self.manifests.insert(lbl, manifest);
    }

    /// Get the active manifest.
    pub fn active(&self) -> Option<&C2paManifest> {
        self.active_manifest
            .as_ref()
            .and_then(|lbl| self.manifests.get(lbl))
    }

    /// Get a manifest by label.
    pub fn get(&self, label: &str) -> Option<&C2paManifest> {
        self.manifests.get(label)
    }

    /// Return the number of manifests.
    pub fn manifest_count(&self) -> usize {
        self.manifests.len()
    }

    /// Returns true if any manifest in the store has AI-related assertions.
    pub fn has_ai_content(&self) -> bool {
        self.manifests.values().any(|m| m.claim.has_ai_content())
    }

    /// Set the active manifest label.
    pub fn set_active(&mut self, label: impl Into<String>) {
        self.active_manifest = Some(label.into());
    }
}

impl Default for C2paManifestStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Conversion to/from Metadata ----

/// Serialize a C2PA manifest store into an `Metadata` container.
///
/// The resulting metadata uses the `MetadataFormat::Xmp` format
/// with C2PA-prefixed keys.
pub fn to_metadata(store: &C2paManifestStore) -> Metadata {
    let mut metadata = Metadata::new(MetadataFormat::Xmp);

    if let Some(active) = store.active() {
        metadata.insert(
            "c2pa:claim_generator".to_string(),
            MetadataValue::Text(active.claim.claim_generator.clone()),
        );

        if let Some(ref title) = active.claim.title {
            metadata.insert("c2pa:title".to_string(), MetadataValue::Text(title.clone()));
        }

        if let Some(ref fmt) = active.claim.format {
            metadata.insert("c2pa:format".to_string(), MetadataValue::Text(fmt.clone()));
        }

        if let Some(ref alg) = active.signature_algorithm {
            metadata.insert(
                "c2pa:signature_algorithm".to_string(),
                MetadataValue::Text(alg.clone()),
            );
        }

        metadata.insert(
            "c2pa:is_signed".to_string(),
            MetadataValue::Boolean(active.is_signed()),
        );

        metadata.insert(
            "c2pa:assertion_count".to_string(),
            MetadataValue::Integer(active.claim.assertion_count() as i64),
        );

        metadata.insert(
            "c2pa:ingredient_count".to_string(),
            MetadataValue::Integer(active.claim.ingredient_count() as i64),
        );

        metadata.insert(
            "c2pa:has_ai_content".to_string(),
            MetadataValue::Boolean(active.claim.has_ai_content()),
        );

        // Serialize assertions as numbered keys
        for (i, assertion) in active.claim.assertions.iter().enumerate() {
            metadata.insert(
                format!("c2pa:assertion_{i}_label"),
                MetadataValue::Text(assertion.label.clone()),
            );
            for (k, v) in &assertion.data {
                metadata.insert(
                    format!("c2pa:assertion_{i}_{k}"),
                    MetadataValue::Text(v.clone()),
                );
            }
        }

        // Serialize ingredients
        for (i, ingredient) in active.claim.ingredients.iter().enumerate() {
            metadata.insert(
                format!("c2pa:ingredient_{i}_title"),
                MetadataValue::Text(ingredient.title.clone()),
            );
            metadata.insert(
                format!("c2pa:ingredient_{i}_format"),
                MetadataValue::Text(ingredient.format.clone()),
            );
            metadata.insert(
                format!("c2pa:ingredient_{i}_instance_id"),
                MetadataValue::Text(ingredient.instance_id.clone()),
            );
            let rel = match ingredient.relationship {
                IngredientRelationship::ParentOf => "parentOf",
                IngredientRelationship::ComponentOf => "componentOf",
            };
            metadata.insert(
                format!("c2pa:ingredient_{i}_relationship"),
                MetadataValue::Text(rel.to_string()),
            );
            if let Some(ref hash) = ingredient.hash {
                metadata.insert(
                    format!("c2pa:ingredient_{i}_hash"),
                    MetadataValue::Text(hash.clone()),
                );
            }
        }
    }

    metadata
}

/// Extract a C2PA claim generator from generic `Metadata` if present.
pub fn claim_generator_from_metadata(metadata: &Metadata) -> Option<String> {
    metadata
        .get("c2pa:claim_generator")
        .and_then(|v| v.as_text())
        .map(|s| s.to_string())
}

/// Check whether metadata indicates AI-generated content.
pub fn is_ai_generated(metadata: &Metadata) -> bool {
    metadata
        .get("c2pa:has_ai_content")
        .and_then(|v| v.as_boolean())
        .unwrap_or(false)
}

/// Validate basic structural requirements of C2PA metadata.
///
/// Returns a list of validation issues (empty means valid).
pub fn validate_manifest(manifest: &C2paManifest) -> Vec<String> {
    let mut issues = Vec::new();

    if manifest.claim.claim_generator.is_empty() {
        issues.push("claim_generator is required".to_string());
    }

    if manifest.claim.assertions.is_empty() {
        issues.push("at least one assertion is required".to_string());
    }

    if manifest.signature.is_none() {
        issues.push("manifest is unsigned (signature recommended)".to_string());
    }

    issues
}

// ---- C2PA Builder (convenience) ----

/// Builder for constructing a C2PA manifest with common patterns.
#[derive(Debug)]
pub struct C2paBuilder {
    claim: C2paClaim,
    algorithm: Option<String>,
}

impl C2paBuilder {
    /// Create a new builder with the given claim generator string.
    pub fn new(claim_generator: impl Into<String>) -> Self {
        Self {
            claim: C2paClaim::new(claim_generator),
            algorithm: None,
        }
    }

    /// Set the asset title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.claim.title = Some(title.into());
        self
    }

    /// Set the asset format.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.claim.format = Some(format.into());
        self
    }

    /// Set the signature algorithm.
    pub fn algorithm(mut self, alg: impl Into<String>) -> Self {
        self.algorithm = Some(alg.into());
        self
    }

    /// Add a creation action.
    pub fn created(mut self, software_agent: impl Into<String>) -> Self {
        let entry = C2paActionEntry::new(C2paAction::Created).with_software_agent(software_agent);
        let assertion = C2paAssertion::from_actions(&[entry]);
        self.claim.add_assertion(assertion);
        self
    }

    /// Add an AI generation assertion.
    pub fn ai_generated(mut self, software_agent: impl Into<String>) -> Self {
        let entry =
            C2paActionEntry::new(C2paAction::AiGenerated).with_software_agent(software_agent);
        let assertion = C2paAssertion::from_actions(&[entry]);
        self.claim.add_assertion(assertion);
        self
    }

    /// Add a creative work assertion.
    pub fn creative_work(
        mut self,
        title: Option<&str>,
        author: Option<&str>,
        date: Option<&str>,
    ) -> Self {
        let assertion = C2paAssertion::creative_work(title, author, date);
        self.claim.add_assertion(assertion);
        self
    }

    /// Add an ingredient.
    pub fn ingredient(mut self, ingredient: C2paIngredient) -> Self {
        self.claim.add_ingredient(ingredient);
        self
    }

    /// Build the manifest (unsigned).
    pub fn build(self) -> C2paManifest {
        let mut manifest = C2paManifest::new(self.claim);
        if let Some(alg) = self.algorithm {
            manifest.signature_algorithm = Some(alg);
        }
        manifest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c2pa_action_uri_round_trip() {
        let actions = [
            C2paAction::Created,
            C2paAction::Edited,
            C2paAction::Cropped,
            C2paAction::Resized,
            C2paAction::ColorAdjusted,
            C2paAction::Converted,
            C2paAction::Published,
            C2paAction::AiGenerated,
            C2paAction::AiTrained,
            C2paAction::Watermarked,
            C2paAction::Redacted,
        ];

        for action in &actions {
            let uri = action.uri();
            let parsed = C2paAction::from_uri(&uri);
            assert_eq!(&parsed, action);
        }
    }

    #[test]
    fn test_c2pa_action_custom() {
        let action = C2paAction::Custom("vendor.my_action".to_string());
        assert_eq!(action.uri(), "vendor.my_action");
        assert_eq!(
            C2paAction::from_uri("vendor.my_action"),
            C2paAction::Custom("vendor.my_action".to_string())
        );
    }

    #[test]
    fn test_c2pa_action_is_ai_related() {
        assert!(C2paAction::AiGenerated.is_ai_related());
        assert!(C2paAction::AiTrained.is_ai_related());
        assert!(!C2paAction::Created.is_ai_related());
        assert!(!C2paAction::Edited.is_ai_related());
    }

    #[test]
    fn test_c2pa_action_is_destructive() {
        assert!(C2paAction::Edited.is_destructive());
        assert!(C2paAction::Cropped.is_destructive());
        assert!(C2paAction::Resized.is_destructive());
        assert!(C2paAction::Redacted.is_destructive());
        assert!(!C2paAction::Created.is_destructive());
        assert!(!C2paAction::Published.is_destructive());
    }

    #[test]
    fn test_c2pa_assertion_new_and_get_set() {
        let mut assertion = C2paAssertion::new("test.label");
        assertion.set("key1", "value1");
        assertion.set("key2", "value2");

        assert_eq!(assertion.label, "test.label");
        assert_eq!(assertion.get("key1"), Some("value1"));
        assert_eq!(assertion.get("key2"), Some("value2"));
        assert_eq!(assertion.get("key3"), None);
    }

    #[test]
    fn test_c2pa_assertion_from_actions() {
        let entries = vec![
            C2paActionEntry::new(C2paAction::Created)
                .with_software_agent("OxiMedia 0.1.2")
                .with_when("2025-01-01T00:00:00Z"),
            C2paActionEntry::new(C2paAction::Edited)
                .with_software_agent("OxiMedia 0.1.2")
                .with_reason("Color correction"),
        ];

        let assertion = C2paAssertion::from_actions(&entries);
        assert_eq!(assertion.label, "c2pa.actions");
        assert_eq!(assertion.get("action_count"), Some("2"));
        assert_eq!(assertion.get("action_0.action"), Some("c2pa.created"));
        assert_eq!(
            assertion.get("action_0.softwareAgent"),
            Some("OxiMedia 0.1.2")
        );
        assert_eq!(assertion.get("action_0.when"), Some("2025-01-01T00:00:00Z"));
        assert_eq!(assertion.get("action_1.action"), Some("c2pa.edited"));
        assert_eq!(assertion.get("action_1.reason"), Some("Color correction"));
    }

    #[test]
    fn test_c2pa_assertion_creative_work() {
        let assertion =
            C2paAssertion::creative_work(Some("My Photo"), Some("Alice"), Some("2025-06-15"));
        assert_eq!(assertion.label, "stds.schema_org.CreativeWork");
        assert_eq!(assertion.get("name"), Some("My Photo"));
        assert_eq!(assertion.get("author"), Some("Alice"));
        assert_eq!(assertion.get("dateCreated"), Some("2025-06-15"));
    }

    #[test]
    fn test_c2pa_action_entry_builder() {
        let entry = C2paActionEntry::new(C2paAction::AiGenerated)
            .with_when("2025-03-01T12:00:00Z")
            .with_software_agent("StableDiffusion 3.0")
            .with_reason("Initial generation");

        assert_eq!(entry.action, C2paAction::AiGenerated);
        assert_eq!(entry.when.as_deref(), Some("2025-03-01T12:00:00Z"));
        assert_eq!(entry.software_agent.as_deref(), Some("StableDiffusion 3.0"));
        assert_eq!(entry.reason.as_deref(), Some("Initial generation"));
    }

    #[test]
    fn test_c2pa_ingredient() {
        let ingredient = C2paIngredient::new(
            "source.jpg",
            "image/jpeg",
            "xmp:iid:1234",
            IngredientRelationship::ParentOf,
        )
        .with_document_id("xmp:did:5678")
        .with_hash("sha256:abcdef");

        assert_eq!(ingredient.title, "source.jpg");
        assert_eq!(ingredient.format, "image/jpeg");
        assert_eq!(ingredient.instance_id, "xmp:iid:1234");
        assert_eq!(ingredient.document_id.as_deref(), Some("xmp:did:5678"));
        assert_eq!(ingredient.hash.as_deref(), Some("sha256:abcdef"));
        assert_eq!(ingredient.relationship, IngredientRelationship::ParentOf);
    }

    #[test]
    fn test_c2pa_claim() {
        let mut claim = C2paClaim::new("OxiMedia/0.1.2")
            .with_title("photo.jpg")
            .with_format("image/jpeg")
            .with_instance_id("xmp:iid:abc123");

        assert_eq!(claim.claim_generator, "OxiMedia/0.1.2");
        assert_eq!(claim.title.as_deref(), Some("photo.jpg"));
        assert_eq!(claim.assertion_count(), 0);
        assert_eq!(claim.ingredient_count(), 0);
        assert!(!claim.has_ai_content());

        claim.add_assertion(C2paAssertion::new("test"));
        assert_eq!(claim.assertion_count(), 1);

        claim.add_ingredient(C2paIngredient::new(
            "src.png",
            "image/png",
            "id1",
            IngredientRelationship::ComponentOf,
        ));
        assert_eq!(claim.ingredient_count(), 1);
    }

    #[test]
    fn test_c2pa_claim_has_ai_content() {
        let mut claim = C2paClaim::new("test");
        let entry = C2paActionEntry::new(C2paAction::AiGenerated).with_software_agent("DALL-E");
        let assertion = C2paAssertion::from_actions(&[entry]);
        claim.add_assertion(assertion);
        assert!(claim.has_ai_content());
    }

    #[test]
    fn test_c2pa_manifest() {
        let claim = C2paClaim::new("OxiMedia/0.1.2");
        let mut manifest = C2paManifest::new(claim);
        assert!(!manifest.is_signed());

        manifest = manifest
            .with_algorithm("ES256")
            .with_signature(vec![0x01, 0x02, 0x03]);
        assert!(manifest.is_signed());
        assert_eq!(manifest.signature_algorithm.as_deref(), Some("ES256"));

        manifest.add_certificate(vec![0xAA, 0xBB]);
        assert_eq!(manifest.certificate_chain.len(), 1);
    }

    #[test]
    fn test_c2pa_manifest_store() {
        let mut store = C2paManifestStore::new();
        assert_eq!(store.manifest_count(), 0);
        assert!(store.active().is_none());

        let claim = C2paClaim::new("Generator1");
        let manifest = C2paManifest::new(claim);
        store.add_manifest("manifest-1", manifest);

        assert_eq!(store.manifest_count(), 1);
        assert!(store.active().is_some());
        assert!(store.get("manifest-1").is_some());
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn test_c2pa_manifest_store_has_ai_content() {
        let mut store = C2paManifestStore::new();

        let mut claim = C2paClaim::new("test");
        let entry = C2paActionEntry::new(C2paAction::AiGenerated).with_software_agent("Midjourney");
        claim.add_assertion(C2paAssertion::from_actions(&[entry]));
        store.add_manifest("m1", C2paManifest::new(claim));

        assert!(store.has_ai_content());
    }

    #[test]
    fn test_c2pa_manifest_store_default() {
        let store = C2paManifestStore::default();
        assert_eq!(store.manifest_count(), 0);
    }

    #[test]
    fn test_c2pa_to_metadata() {
        let mut store = C2paManifestStore::new();

        let mut claim = C2paClaim::new("OxiMedia/0.1.2")
            .with_title("test.mp4")
            .with_format("video/mp4");

        claim.add_assertion(C2paAssertion::creative_work(
            Some("My Video"),
            Some("Director"),
            None,
        ));

        claim.add_ingredient(
            C2paIngredient::new(
                "clip.mov",
                "video/quicktime",
                "id-001",
                IngredientRelationship::ParentOf,
            )
            .with_hash("sha256:deadbeef"),
        );

        let manifest = C2paManifest::new(claim).with_algorithm("ES256");
        store.add_manifest("active", manifest);

        let metadata = to_metadata(&store);

        assert_eq!(
            metadata
                .get("c2pa:claim_generator")
                .and_then(|v| v.as_text()),
            Some("OxiMedia/0.1.2")
        );
        assert_eq!(
            metadata.get("c2pa:title").and_then(|v| v.as_text()),
            Some("test.mp4")
        );
        assert_eq!(
            metadata.get("c2pa:format").and_then(|v| v.as_text()),
            Some("video/mp4")
        );
        assert_eq!(
            metadata
                .get("c2pa:signature_algorithm")
                .and_then(|v| v.as_text()),
            Some("ES256")
        );
        assert_eq!(
            metadata
                .get("c2pa:assertion_count")
                .and_then(|v| v.as_integer()),
            Some(1)
        );
        assert_eq!(
            metadata
                .get("c2pa:ingredient_count")
                .and_then(|v| v.as_integer()),
            Some(1)
        );
        assert_eq!(
            metadata
                .get("c2pa:ingredient_0_title")
                .and_then(|v| v.as_text()),
            Some("clip.mov")
        );
        assert_eq!(
            metadata
                .get("c2pa:ingredient_0_hash")
                .and_then(|v| v.as_text()),
            Some("sha256:deadbeef")
        );
    }

    #[test]
    fn test_claim_generator_from_metadata() {
        let mut metadata = Metadata::new(MetadataFormat::Xmp);
        metadata.insert(
            "c2pa:claim_generator".to_string(),
            MetadataValue::Text("OxiMedia/0.1.2".to_string()),
        );
        assert_eq!(
            claim_generator_from_metadata(&metadata),
            Some("OxiMedia/0.1.2".to_string())
        );
    }

    #[test]
    fn test_is_ai_generated_true() {
        let mut metadata = Metadata::new(MetadataFormat::Xmp);
        metadata.insert(
            "c2pa:has_ai_content".to_string(),
            MetadataValue::Boolean(true),
        );
        assert!(is_ai_generated(&metadata));
    }

    #[test]
    fn test_is_ai_generated_false() {
        let metadata = Metadata::new(MetadataFormat::Xmp);
        assert!(!is_ai_generated(&metadata));
    }

    #[test]
    fn test_validate_manifest_issues() {
        // Empty claim generator
        let claim = C2paClaim::new("");
        let manifest = C2paManifest::new(claim);
        let issues = validate_manifest(&manifest);
        assert!(issues.iter().any(|i| i.contains("claim_generator")));
        assert!(issues.iter().any(|i| i.contains("assertion")));
        assert!(issues.iter().any(|i| i.contains("unsigned")));
    }

    #[test]
    fn test_validate_manifest_minimal_valid() {
        let mut claim = C2paClaim::new("OxiMedia");
        claim.add_assertion(C2paAssertion::new("test"));
        let manifest = C2paManifest::new(claim).with_signature(vec![0x01]);
        let issues = validate_manifest(&manifest);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_c2pa_builder() {
        let manifest = C2paBuilder::new("OxiMedia/0.1.2")
            .title("photo.jpg")
            .format("image/jpeg")
            .algorithm("ES256")
            .created("OxiMedia Camera")
            .creative_work(
                Some("Beach Sunset"),
                Some("Photographer"),
                Some("2025-08-01"),
            )
            .ingredient(C2paIngredient::new(
                "raw.dng",
                "image/x-adobe-dng",
                "id-raw",
                IngredientRelationship::ParentOf,
            ))
            .build();

        assert_eq!(manifest.claim.claim_generator, "OxiMedia/0.1.2");
        assert_eq!(manifest.claim.title.as_deref(), Some("photo.jpg"));
        assert_eq!(manifest.signature_algorithm.as_deref(), Some("ES256"));
        assert_eq!(manifest.claim.assertion_count(), 2);
        assert_eq!(manifest.claim.ingredient_count(), 1);
        assert!(!manifest.is_signed());
    }

    #[test]
    fn test_c2pa_builder_ai_generated() {
        let manifest = C2paBuilder::new("test").ai_generated("DALL-E 3").build();

        assert!(manifest.claim.has_ai_content());
    }

    #[test]
    fn test_c2pa_manifest_store_set_active() {
        let mut store = C2paManifestStore::new();

        let claim1 = C2paClaim::new("Gen1");
        let claim2 = C2paClaim::new("Gen2");

        store.add_manifest("m1", C2paManifest::new(claim1));
        store.add_manifest("m2", C2paManifest::new(claim2));

        // m2 is active (added last)
        assert_eq!(
            store.active().map(|m| m.claim.claim_generator.as_str()),
            Some("Gen2")
        );

        // Switch back to m1
        store.set_active("m1");
        assert_eq!(
            store.active().map(|m| m.claim.claim_generator.as_str()),
            Some("Gen1")
        );
    }

    #[test]
    fn test_ingredient_relationship() {
        assert_ne!(
            IngredientRelationship::ParentOf,
            IngredientRelationship::ComponentOf
        );
    }
}
