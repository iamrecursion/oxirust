//! Addressing a property, optionally down to one component of a compound
//! datatype.

use core::fmt;

use crate::property::id::PropertyId;

/// A property, optionally narrowed to one component of a compound datatype.
///
/// Compound properties are written in a stylesheet either whole
/// (`space-before="4pt"`, the *short form*) or component by component
/// (`space-before.optimum="4pt"`); a `PropertyKey` names either.
///
/// ```
/// use oxixml_fo::property::{PropertyId, PropertyKey};
///
/// let whole = PropertyKey::parse("space-before").expect("a property");
/// assert_eq!(whole.id, PropertyId::SpaceBefore);
/// assert_eq!(whole.component, None);
///
/// let part = PropertyKey::parse("space-before.optimum").expect("a component");
/// assert_eq!(part.component, Some("optimum"));
///
/// assert_eq!(PropertyKey::parse("space-before.nonesuch"), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PropertyKey {
    /// The property being addressed.
    pub id: PropertyId,
    /// The component of the compound datatype, or `None` for the short form.
    pub component: Option<&'static str>,
}

impl PropertyKey {
    /// Addresses a whole property.
    #[must_use]
    pub const fn whole(id: PropertyId) -> Self {
        Self {
            id,
            component: None,
        }
    }

    /// Addresses one component of a compound property.
    ///
    /// Returns `None` when `component` is not a component of `id`'s datatype.
    #[must_use]
    pub fn component(id: PropertyId, component: &str) -> Option<Self> {
        id.compound()
            .components()
            .iter()
            .find(|candidate| **candidate == component)
            .map(|candidate| Self {
                id,
                component: Some(candidate),
            })
    }

    /// Parses an attribute name such as `space-before` or
    /// `space-before.optimum`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        if let Some(id) = PropertyId::parse(name) {
            return Some(Self::whole(id));
        }
        let (base, component) = name.rsplit_once('.')?;
        let id = PropertyId::parse(base)?;
        Self::component(id, component)
    }
}

impl fmt::Display for PropertyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id.name())?;
        if let Some(component) = self.component {
            write!(f, ".{component}")?;
        }
        Ok(())
    }
}

impl From<PropertyId> for PropertyKey {
    fn from(id: PropertyId) -> Self {
        Self::whole(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_components_are_validated() {
        assert!(PropertyKey::parse("keep-together.within-page").is_some());
        assert!(PropertyKey::parse("keep-together.within-galaxy").is_none());
        assert!(PropertyKey::parse("border-before-width.conditionality").is_some());
        assert!(PropertyKey::parse("border-separation.inline-progression-direction").is_some());
        // A non-compound property has no components at all.
        assert!(PropertyKey::parse("color.red").is_none());
    }

    #[test]
    fn keys_render_back_to_their_attribute_name() {
        let key = PropertyKey::parse("space-after.precedence").expect("a component");
        assert_eq!(key.to_string(), "space-after.precedence");
    }
}
