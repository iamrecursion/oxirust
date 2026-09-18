//! [`LexicalHandler`], reporting lexical details that
//! [`ContentHandler`](crate::ContentHandler) alone cannot distinguish.

use crate::error::SaxError;

/// Receives notification of lexical information that is not required by
/// [`ContentHandler`](crate::ContentHandler): comments, the boundary between
/// `CDATA` sections and ordinary text, and the document type declaration.
///
/// Mirrors `org.xml.sax.ext.LexicalHandler`. Every method has a no-op default
/// implementation.
pub trait LexicalHandler {
    /// Called with the text of an XML comment (`<!-- text -->`), excluding the
    /// delimiters.
    fn comment(&mut self, _text: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called immediately before the content of a `CDATA` section
    /// (`<![CDATA[...]]>`) is reported through
    /// [`ContentHandler::characters`](crate::ContentHandler::characters).
    fn start_cdata(&mut self) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called immediately after the content of a `CDATA` section has been
    /// reported.
    fn end_cdata(&mut self) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called at the start of the document type declaration
    /// (`<!DOCTYPE name ...>`), before any internal-subset declarations are
    /// reported to a [`DtdHandler`](crate::DtdHandler) or
    /// [`DeclHandler`](crate::DeclHandler).
    fn start_dtd(
        &mut self,
        _name: &str,
        _public_id: Option<&str>,
        _system_id: Option<&str>,
    ) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called at the end of the document type declaration.
    fn end_dtd(&mut self) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called before a general entity reference is expanded, bracketing the
    /// [`characters`](crate::ContentHandler::characters) call (or calls) its
    /// replacement text produces.
    ///
    /// As in SAX2, the five predefined entities and character references are not
    /// reported this way: they are simply part of the character data. An entity
    /// this reader cannot expand is reported through
    /// [`ContentHandler::skipped_entity`](crate::ContentHandler::skipped_entity)
    /// instead, and produces no `start_entity`/`end_entity` pair.
    fn start_entity(&mut self, _name: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called after a general entity reference has been expanded; see
    /// [`start_entity`](Self::start_entity).
    fn end_entity(&mut self, _name: &str) -> Result<(), SaxError> {
        Ok(())
    }
}

/// A [`LexicalHandler`] that ignores every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullLexicalHandler;

impl LexicalHandler for NullLexicalHandler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handler_accepts_every_call() {
        let mut handler = NullLexicalHandler;
        assert!(handler.comment("c").is_ok());
        assert!(handler.start_cdata().is_ok());
        assert!(handler.end_cdata().is_ok());
        assert!(handler.start_dtd("root", None, None).is_ok());
        assert!(handler.end_dtd().is_ok());
    }
}
