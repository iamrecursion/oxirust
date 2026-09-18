//! XMP (Extensible Metadata Platform) metadata for images.
//!
//! Provides structured XMP entry storage, namespace constants, and a
//! builder API for composing image metadata in the XMP model.

#![allow(dead_code)]

/// A single XMP metadata entry with namespace, key and string value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XmpEntry {
    /// XMP namespace URI (e.g. `"http://purl.org/dc/elements/1.1/"`).
    pub namespace: String,
    /// Property key within the namespace (e.g. `"title"`).
    pub key: String,
    /// Property value as a UTF-8 string.
    pub value: String,
}

impl XmpEntry {
    /// Create a new `XmpEntry`.
    #[must_use]
    pub fn new(
        namespace: impl Into<String>,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            key: key.into(),
            value: value.into(),
        }
    }

    /// Returns the qualified name `"<namespace>/<key>"`.
    #[must_use]
    pub fn qualified_name(&self) -> String {
        format!("{}/{}", self.namespace, self.key)
    }
}

/// Well-known XMP namespace URI constants.
pub struct XmpNamespace;

impl XmpNamespace {
    /// Dublin Core namespace URI.
    pub const DUBLIN_CORE: &'static str = "http://purl.org/dc/elements/1.1/";
    /// Exif namespace URI.
    pub const EXIF: &'static str = "http://ns.adobe.com/exif/1.0/";
    /// IPTC Core namespace URI.
    pub const IPTC_CORE: &'static str = "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/";
    /// Photoshop namespace URI.
    pub const PHOTOSHOP: &'static str = "http://ns.adobe.com/photoshop/1.0/";
}

/// Container for XMP metadata entries belonging to an image.
#[derive(Debug, Clone, Default)]
pub struct ImageXmp {
    /// All XMP entries stored in insertion order.
    pub entries: Vec<XmpEntry>,
}

impl ImageXmp {
    /// Create an empty `ImageXmp`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the value of the entry with the given namespace and key, if present.
    #[must_use]
    pub fn get(&self, ns: &str, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.namespace == ns && e.key == key)
            .map(|e| e.value.as_str())
    }

    /// Set (insert or update) an entry with the given namespace, key and value.
    pub fn set(&mut self, ns: impl Into<String>, key: impl Into<String>, value: impl Into<String>) {
        let ns = ns.into();
        let key = key.into();
        let value = value.into();
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|e| e.namespace == ns && e.key == key)
        {
            entry.value = value;
        } else {
            self.entries.push(XmpEntry::new(ns, key, value));
        }
    }

    /// Remove the entry with the given namespace and key.
    ///
    /// Returns `true` if an entry was removed.
    pub fn remove(&mut self, ns: &str, key: &str) -> bool {
        let before = self.entries.len();
        self.entries
            .retain(|e| !(e.namespace == ns && e.key == key));
        self.entries.len() < before
    }

    /// Return a deduplicated list of all namespace URIs present in the entries.
    #[must_use]
    pub fn namespaces(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for e in &self.entries {
            if !seen.contains(&e.namespace.as_str()) {
                seen.push(&e.namespace);
            }
        }
        seen
    }

    /// Return the total number of entries.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

/// Builder for constructing `ImageXmp` with common Dublin Core fields.
#[derive(Debug, Default)]
pub struct XmpBuilder {
    xmp: ImageXmp,
}

impl XmpBuilder {
    /// Create a new `XmpBuilder`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the `dc:creator` field.
    #[must_use]
    pub fn creator(mut self, creator: impl Into<String>) -> Self {
        self.xmp.set(XmpNamespace::DUBLIN_CORE, "creator", creator);
        self
    }

    /// Set the `dc:title` field.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.xmp.set(XmpNamespace::DUBLIN_CORE, "title", title);
        self
    }

    /// Set the `dc:description` field.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.xmp.set(XmpNamespace::DUBLIN_CORE, "description", desc);
        self
    }

    /// Set the `dc:rights` field.
    #[must_use]
    pub fn rights(mut self, rights: impl Into<String>) -> Self {
        self.xmp.set(XmpNamespace::DUBLIN_CORE, "rights", rights);
        self
    }

    /// Append a subject keyword to `dc:subject` (comma-separated accumulation).
    #[must_use]
    pub fn subject(mut self, kw: &str) -> Self {
        let existing = self
            .xmp
            .get(XmpNamespace::DUBLIN_CORE, "subject")
            .map(str::to_owned);
        let new_val = match existing {
            Some(s) if !s.is_empty() => format!("{s},{kw}"),
            _ => kw.to_owned(),
        };
        self.xmp.set(XmpNamespace::DUBLIN_CORE, "subject", new_val);
        self
    }

    /// Consume the builder and return the completed `ImageXmp`.
    #[must_use]
    pub fn build(self) -> ImageXmp {
        self.xmp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xmp_entry_qualified_name() {
        let e = XmpEntry::new("http://purl.org/dc/elements/1.1/", "title", "Test");
        assert_eq!(e.qualified_name(), "http://purl.org/dc/elements/1.1//title");
    }

    #[test]
    fn test_xmp_entry_new_fields() {
        let e = XmpEntry::new("ns", "key", "val");
        assert_eq!(e.namespace, "ns");
        assert_eq!(e.key, "key");
        assert_eq!(e.value, "val");
    }

    #[test]
    fn test_namespace_constants() {
        assert!(XmpNamespace::DUBLIN_CORE.contains("purl.org"));
        assert!(XmpNamespace::EXIF.contains("adobe.com/exif"));
        assert!(XmpNamespace::IPTC_CORE.contains("iptc.org"));
        assert!(XmpNamespace::PHOTOSHOP.contains("photoshop"));
    }

    #[test]
    fn test_image_xmp_set_get() {
        let mut xmp = ImageXmp::new();
        xmp.set(XmpNamespace::DUBLIN_CORE, "title", "My Photo");
        assert_eq!(
            xmp.get(XmpNamespace::DUBLIN_CORE, "title"),
            Some("My Photo")
        );
    }

    #[test]
    fn test_image_xmp_get_missing() {
        let xmp = ImageXmp::new();
        assert!(xmp.get(XmpNamespace::DUBLIN_CORE, "title").is_none());
    }

    #[test]
    fn test_image_xmp_set_updates_existing() {
        let mut xmp = ImageXmp::new();
        xmp.set(XmpNamespace::DUBLIN_CORE, "title", "First");
        xmp.set(XmpNamespace::DUBLIN_CORE, "title", "Second");
        assert_eq!(xmp.get(XmpNamespace::DUBLIN_CORE, "title"), Some("Second"));
        assert_eq!(xmp.count(), 1);
    }

    #[test]
    fn test_image_xmp_remove_returns_true() {
        let mut xmp = ImageXmp::new();
        xmp.set(XmpNamespace::DUBLIN_CORE, "title", "T");
        assert!(xmp.remove(XmpNamespace::DUBLIN_CORE, "title"));
        assert_eq!(xmp.count(), 0);
    }

    #[test]
    fn test_image_xmp_remove_missing_returns_false() {
        let mut xmp = ImageXmp::new();
        assert!(!xmp.remove(XmpNamespace::DUBLIN_CORE, "title"));
    }

    #[test]
    fn test_image_xmp_namespaces_deduped() {
        let mut xmp = ImageXmp::new();
        xmp.set(XmpNamespace::DUBLIN_CORE, "title", "T");
        xmp.set(XmpNamespace::DUBLIN_CORE, "creator", "A");
        xmp.set(XmpNamespace::EXIF, "ExposureTime", "1/200");
        let ns = xmp.namespaces();
        assert_eq!(ns.len(), 2);
    }

    #[test]
    fn test_image_xmp_count() {
        let mut xmp = ImageXmp::new();
        assert_eq!(xmp.count(), 0);
        xmp.set("ns", "k1", "v1");
        xmp.set("ns", "k2", "v2");
        assert_eq!(xmp.count(), 2);
    }

    #[test]
    fn test_xmp_builder_title() {
        let xmp = XmpBuilder::new().title("Sunset").build();
        assert_eq!(xmp.get(XmpNamespace::DUBLIN_CORE, "title"), Some("Sunset"));
    }

    #[test]
    fn test_xmp_builder_creator_description_rights() {
        let xmp = XmpBuilder::new()
            .creator("Alice")
            .description("Nice photo")
            .rights("CC-BY 4.0")
            .build();
        assert_eq!(xmp.get(XmpNamespace::DUBLIN_CORE, "creator"), Some("Alice"));
        assert_eq!(
            xmp.get(XmpNamespace::DUBLIN_CORE, "description"),
            Some("Nice photo")
        );
        assert_eq!(
            xmp.get(XmpNamespace::DUBLIN_CORE, "rights"),
            Some("CC-BY 4.0")
        );
    }

    #[test]
    fn test_xmp_builder_subject_accumulates() {
        let xmp = XmpBuilder::new()
            .subject("nature")
            .subject("landscape")
            .build();
        let subjects = xmp
            .get(XmpNamespace::DUBLIN_CORE, "subject")
            .expect("should succeed in test");
        assert!(subjects.contains("nature"));
        assert!(subjects.contains("landscape"));
    }

    #[test]
    fn test_xmp_builder_build_count() {
        let xmp = XmpBuilder::new()
            .title("T")
            .creator("C")
            .description("D")
            .rights("R")
            .subject("kw1")
            .build();
        assert_eq!(xmp.count(), 5);
    }
}
