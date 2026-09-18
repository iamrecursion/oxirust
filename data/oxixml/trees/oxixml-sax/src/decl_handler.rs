//! [`DeclHandler`], reporting element, attribute-list and entity declarations
//! from the document's internal DTD subset.

use crate::error::SaxError;

/// Receives notification of `<!ELEMENT>`, `<!ATTLIST>` and general `<!ENTITY>`
/// declarations from the document's internal DTD subset.
///
/// Mirrors `org.xml.sax.ext.DeclHandler`. This crate does not fetch an external
/// DTD subset, so only declarations found in the internal subset (between `[`
/// and `]` in the `<!DOCTYPE ...>`) are reported. Every method has a no-op
/// default implementation.
pub trait DeclHandler {
    /// Called for an `<!ELEMENT name model>` declaration. `model` is the raw text
    /// of the content model (`EMPTY`, `ANY`, or a `Mixed`/`children` group such as
    /// `(#PCDATA|a|b)*` or `(a,b?,c*)`), exactly as written.
    fn element_decl(&mut self, _name: &str, _model: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called once per attribute definition in an `<!ATTLIST element ...>`
    /// declaration.
    ///
    /// `attr_type` is the raw attribute type (`CDATA`, `ID`, `IDREF`, `IDREFS`,
    /// `ENTITY`, `ENTITIES`, `NMTOKEN`, `NMTOKENS`, an enumeration such as
    /// `(a|b|c)`, or `NOTATION (a|b|c)`). `mode` is `Some("#REQUIRED")`,
    /// `Some("#IMPLIED")`, `Some("#FIXED")` or `None` (no keyword — a plain
    /// default value follows). `value` is the default or fixed value, if any.
    fn attribute_decl(
        &mut self,
        _element_name: &str,
        _attribute_name: &str,
        _attr_type: &str,
        _mode: Option<&str>,
        _value: Option<&str>,
    ) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called for an internal general entity declaration
    /// (`<!ENTITY name "value">`). `value` is the *literal entity value* of
    /// [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement) — the
    /// quoted string exactly as it is written, with its character and general
    /// entity references intact — so that a handler can reproduce the
    /// declaration verbatim. The parser resolves those references itself
    /// wherever the entity is used, and reports the result through
    /// [`ContentHandler::characters`](crate::ContentHandler::characters).
    fn internal_entity_decl(&mut self, _name: &str, _value: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called for an external, *parsed* general entity declaration
    /// (`<!ENTITY name SYSTEM "...">` or `<!ENTITY name PUBLIC "..." "...">`,
    /// without `NDATA`). External *unparsed* entities are reported through
    /// [`DtdHandler::unparsed_entity_decl`](crate::DtdHandler::unparsed_entity_decl)
    /// instead.
    fn external_entity_decl(
        &mut self,
        _name: &str,
        _public_id: Option<&str>,
        _system_id: &str,
    ) -> Result<(), SaxError> {
        Ok(())
    }
}

/// A [`DeclHandler`] that ignores every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullDeclHandler;

impl DeclHandler for NullDeclHandler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handler_accepts_every_call() {
        let mut handler = NullDeclHandler;
        assert!(handler.element_decl("a", "EMPTY").is_ok());
        assert!(
            handler
                .attribute_decl("a", "id", "ID", Some("#REQUIRED"), None)
                .is_ok()
        );
        assert!(handler.internal_entity_decl("e", "value").is_ok());
        assert!(handler.external_entity_decl("e", None, "e.xml").is_ok());
    }
}
