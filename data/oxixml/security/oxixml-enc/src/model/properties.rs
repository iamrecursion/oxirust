//! `xenc:EncryptionProperties` and `xenc:EncryptionProperty`.

use crate::model::cipher::XmlFragment;

/// The `xenc:EncryptionProperties` element of
/// [XMLEnc §3.7](https://www.w3.org/TR/xmlenc-core1/#sec-EncryptionProperties).
///
/// A container for out-of-band information about an encryption — when it happened, what serial
/// number the hardware had, what the sender's clock said. XMLEnc §6.3 warns that anything put
/// here is *plaintext*, so a property that says "this element contains a salary" defeats the
/// point of encrypting it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EncryptionProperties {
    /// The optional `Id` attribute.
    pub id: Option<String>,
    /// The `EncryptionProperty` children; the schema requires at least one.
    pub properties: Vec<EncryptionProperty>,
}

impl EncryptionProperties {
    /// Returns an empty container.
    ///
    /// A container with no children is invalid against the schema, so
    /// [`crate::model::EncryptedData`] omits the element entirely when
    /// [`is_empty`](Self::is_empty) holds rather than writing an invalid one.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when there are no properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    /// Adds a property and returns the container, for chaining.
    #[must_use]
    pub fn with(mut self, property: EncryptionProperty) -> Self {
        self.properties.push(property);
        self
    }
}

/// One `xenc:EncryptionProperty`.
///
/// The content model is mixed and wide open (`<any namespace='##other'/>` repeated), so the
/// children are kept verbatim as [`XmlFragment`]s.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EncryptionProperty {
    /// The optional `Id` attribute.
    pub id: Option<String>,
    /// The optional `Target` attribute, which points at the `EncryptedData` or `EncryptedKey`
    /// the property describes.
    pub target: Option<String>,
    /// The children, in document order.
    pub content: Vec<XmlFragment>,
}

impl EncryptionProperty {
    /// Creates a property whose only content is the given text.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            id: None,
            target: None,
            content: vec![XmlFragment::Text(text.into())],
        }
    }

    /// Sets the `Target` attribute.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Sets the `Id` attribute.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Returns the concatenated character data of the property, ignoring element children.
    #[must_use]
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|fragment| match fragment {
                XmlFragment::Text(text) => Some(text.as_str()),
                XmlFragment::Element(_) => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_container_reports_emptiness() {
        assert!(EncryptionProperties::new().is_empty());
        let filled = EncryptionProperties::new().with(EncryptionProperty::text("2026-07-27"));
        assert!(!filled.is_empty());
        assert_eq!(filled.properties.len(), 1);
    }

    #[test]
    fn text_content_ignores_elements() {
        let property = EncryptionProperty {
            id: None,
            target: None,
            content: vec![
                XmlFragment::Text("before".into()),
                XmlFragment::Element("<x/>".into()),
                XmlFragment::Text("after".into()),
            ],
        };
        assert_eq!(property.text_content(), "beforeafter");
    }

    #[test]
    fn builders_set_the_attributes() {
        let property = EncryptionProperty::text("t")
            .with_target("#ed")
            .with_id("p1");
        assert_eq!(property.target.as_deref(), Some("#ed"));
        assert_eq!(property.id.as_deref(), Some("p1"));
        assert_eq!(property.text_content(), "t");
    }
}
