//! Entity resolver module

use std::convert::Infallible;
use std::error::Error;

use crate::escape::resolve_predefined_entity;
use crate::events::BytesText;

/// Used to resolve unknown entities while parsing
///
/// # Example
///
/// ```
/// # use serde::Deserialize;
/// use std::collections::BTreeMap;
/// use std::string::FromUtf8Error;
/// use oxixml_xml::de::{Deserializer, EntityResolver};
/// use oxixml_xml::events::BytesText;
///
/// #[derive(Default)]
/// struct DocTypeEntityResolver {
///     map: BTreeMap<String, String>,
/// }
///
/// impl EntityResolver for DocTypeEntityResolver {
///     type Error = FromUtf8Error;
///
///     fn capture(&mut self, doctype: BytesText) -> Result<(), Self::Error> {
///         // We do not focus on true parsing in this example: this only handles
///         // `<!ENTITY name "value">` declarations. You should use a dedicated
///         // DTD parser in real code.
///         let text = String::from_utf8(doctype.to_vec())?;
///         let mut rest = text.as_str();
///         while let Some(start) = rest.find("<!ENTITY") {
///             rest = rest[start + "<!ENTITY".len()..].trim_start();
///             let name_len = rest.find(char::is_whitespace).unwrap_or(rest.len());
///             let name = rest[..name_len].to_string();
///             rest = &rest[name_len..];
///
///             let Some(open) = rest.find('"') else { break };
///             rest = &rest[open + 1..];
///             let Some(close) = rest.find('"') else { break };
///             self.map.insert(name, rest[..close].to_string());
///             rest = &rest[close + 1..];
///         }
///         Ok(())
///     }
///
///     fn resolve(&self, entity: &str) -> Option<&str> {
///         self.map.get(entity).map(|s| s.as_str())
///     }
/// }
///
/// let xml_reader = br#"
///     <!DOCTYPE dict[ <!ENTITY e1 "entity 1"> ]>
///     <root>
///         <entity_one>&e1;</entity_one>
///     </root>
/// "#.as_ref();
///
/// let mut de = Deserializer::with_resolver(
///     xml_reader,
///     DocTypeEntityResolver::default(),
/// );
/// let data: BTreeMap<String, String> = BTreeMap::deserialize(&mut de).expect("should be Ok");
///
/// assert_eq!(data.get("entity_one"), Some(&"entity 1".to_string()));
/// ```
pub trait EntityResolver {
    /// The error type that represents DTD parse error
    type Error: Error;

    /// Called on contents of [`Event::DocType`] to capture declared entities.
    /// Can be called multiple times, for each parsed `<!DOCTYPE >` declaration.
    ///
    /// [`Event::DocType`]: crate::events::Event::DocType
    fn capture(&mut self, doctype: BytesText) -> Result<(), Self::Error>;

    /// Called when an entity needs to be resolved.
    ///
    /// `None` is returned if a suitable value can not be found.
    /// In that case an [`EscapeError::UnrecognizedEntity`] will be returned by
    /// a deserializer.
    ///
    /// [`EscapeError::UnrecognizedEntity`]: crate::escape::EscapeError::UnrecognizedEntity
    fn resolve(&self, entity: &str) -> Option<&str>;
}

/// An [`EntityResolver`] that resolves only predefined entities:
///
/// | Entity | Resolution
/// |--------|------------
/// |`&lt;`  | `<`
/// |`&gt;`  | `>`
/// |`&amp;` | `&`
/// |`&apos;`| `'`
/// |`&quot;`| `"`
#[derive(Default, Copy, Clone)]
pub struct PredefinedEntityResolver;

impl EntityResolver for PredefinedEntityResolver {
    type Error = Infallible;

    #[inline]
    fn capture(&mut self, _doctype: BytesText) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn resolve(&self, entity: &str) -> Option<&str> {
        resolve_predefined_entity(entity)
    }
}
