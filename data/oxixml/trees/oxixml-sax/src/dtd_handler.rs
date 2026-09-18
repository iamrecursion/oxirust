//! [`DtdHandler`], reporting `NOTATION` declarations and unparsed (`NDATA`)
//! entity declarations.

use crate::error::SaxError;

/// Receives notification of basic DTD-related events: notations and unparsed
/// (external, `NDATA`) entities.
///
/// Mirrors `org.xml.sax.DTDHandler`. Declarations are parsed out of the
/// document's internal DTD subset (this crate does not fetch an external DTD
/// subset, so declarations that live there are not reported). Every method has a
/// no-op default implementation.
pub trait DtdHandler {
    /// Called for a `<!NOTATION name ...>` declaration.
    fn notation_decl(
        &mut self,
        _name: &str,
        _public_id: Option<&str>,
        _system_id: Option<&str>,
    ) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called for a `<!ENTITY name ... NDATA notation>` declaration (an external,
    /// unparsed general entity).
    fn unparsed_entity_decl(
        &mut self,
        _name: &str,
        _public_id: Option<&str>,
        _system_id: &str,
        _notation_name: &str,
    ) -> Result<(), SaxError> {
        Ok(())
    }
}

/// A [`DtdHandler`] that ignores every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullDtdHandler;

impl DtdHandler for NullDtdHandler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handler_accepts_every_call() {
        let mut handler = NullDtdHandler;
        assert!(handler.notation_decl("n", None, Some("n.dtd")).is_ok());
        assert!(
            handler
                .unparsed_entity_decl("e", None, "e.bin", "n")
                .is_ok()
        );
    }
}
