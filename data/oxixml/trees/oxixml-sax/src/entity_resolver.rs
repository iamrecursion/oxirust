//! [`EntityResolver`], allowing an application to redirect external identifiers.

/// Allows an application to redirect external identifiers (`SYSTEM`/`PUBLIC`
/// references found in a document type declaration) to a different location, or
/// to a locally-cached copy.
///
/// Mirrors `org.xml.sax.EntityResolver`. [`XMLReader`](crate::XMLReader) calls
/// [`resolve_entity`](Self::resolve_entity) whenever it encounters an external
/// identifier — on the `<!DOCTYPE ...>` declaration itself, on an external
/// `<!ENTITY>` declaration, and on a `<!NOTATION>` declaration — but this crate
/// never fetches an external DTD subset or external entity, so the resolved
/// value (if any) is informational only: it is not currently used to redirect
/// any I/O.
pub trait EntityResolver {
    /// Called with the public and system identifiers of an external reference.
    /// Returning `Some(resolved_system_id)` indicates the application wants that
    /// identifier used in place of `system_id`; returning `None` leaves the
    /// identifier as found in the document.
    fn resolve_entity(
        &mut self,
        _public_id: Option<&str>,
        _system_id: Option<&str>,
    ) -> Option<String> {
        None
    }
}

/// An [`EntityResolver`] that never redirects anything.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullEntityResolver;

impl EntityResolver for NullEntityResolver {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_resolver_never_redirects() {
        let mut resolver = NullEntityResolver;
        assert_eq!(resolver.resolve_entity(None, Some("a.dtd")), None);
    }
}
