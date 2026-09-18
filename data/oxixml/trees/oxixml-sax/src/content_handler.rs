//! [`ContentHandler`], the primary SAX2 callback interface for element and
//! character data.

use crate::attributes::Attributes;
use crate::error::SaxError;
use crate::locator::Locator;

/// Receives notification of the logical content of a document.
///
/// Mirrors `org.xml.sax.ContentHandler`. Every method has a no-op default
/// implementation, so a handler only needs to override the callbacks it cares
/// about.
///
/// # Method ordering
///
/// For a well-formed document, [`XMLReader::parse`](crate::XMLReader::parse)
/// calls these methods in the following order:
///
/// 1. [`set_document_locator`](Self::set_document_locator), at most once, before
///    anything else.
/// 2. [`start_document`](Self::start_document), exactly once.
/// 3. Any mixture of [`start_prefix_mapping`](Self::start_prefix_mapping)/
///    [`end_prefix_mapping`](Self::end_prefix_mapping),
///    [`start_element`](Self::start_element)/[`end_element`](Self::end_element)
///    (always balanced, and prefix mappings always bracket the element that
///    declares them),
///    [`characters`](Self::characters),
///    [`processing_instruction`](Self::processing_instruction) and
///    [`skipped_entity`](Self::skipped_entity).
/// 4. [`end_document`](Self::end_document), exactly once, unless parsing was
///    aborted by an error or by a handler returning `Err`.
///
/// # Whitespace
///
/// This reader does not perform DTD-based validation, so it cannot reliably tell
/// *ignorable* whitespace (whitespace in element content, per a validated content
/// model) from significant character data. Consequently
/// [`ignorable_whitespace`](Self::ignorable_whitespace) is never called by
/// [`XMLReader`](crate::XMLReader); all character data, including whitespace-only
/// runs, is reported through [`characters`](Self::characters). The method still
/// exists so that handlers written against this trait compile unchanged against a
/// future validating driver.
pub trait ContentHandler {
    /// Called once, before [`start_document`](Self::start_document), with a
    /// snapshot of the document's initial position and identifiers.
    fn set_document_locator(&mut self, _locator: &Locator) {}

    /// Called once, before any other event (except
    /// [`set_document_locator`](Self::set_document_locator)).
    fn start_document(&mut self) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called once, after all other events, unless parsing ended in an error.
    fn end_document(&mut self) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called when a new namespace prefix binding comes into scope, before the
    /// [`start_element`](Self::start_element) call for the element that declares
    /// it. `prefix` is `None` for a default-namespace declaration (`xmlns="..."`).
    ///
    /// # A declaration whose value uses a declared entity
    ///
    /// `uri` here is the declaration's value after the attribute-value
    /// normalization of [XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize),
    /// which *includes* the general entities the document's own internal subset
    /// declares — this reader holds that subset, so it can resolve them. The
    /// `uri` that reaches [`start_element`](Self::start_element) and
    /// [`Attributes`] comes from the underlying reader's namespace resolution
    /// instead, which resolves only the five entities the specification
    /// predefines, deliberately: namespace identity must not depend on which
    /// entity tables a build has compiled in.
    ///
    /// The two therefore disagree for a declaration that names a document-declared
    /// entity. Given
    ///
    /// ```text
    /// <!DOCTYPE r [<!ENTITY u "urn:x">]><r xmlns:p="&u;"><p:c/></r>
    /// ```
    ///
    /// this method reports `urn:x` while `start_element` reports the namespace of
    /// `p:c` as `&u;`. The value reported here is the one §3.3.3 asks for; a
    /// handler that needs the namespace name of an element whose prefix was bound
    /// that way should take it from the binding rather than from `start_element`.
    /// Reconciling the two needs the namespace resolver of `oxixml-xml` to accept
    /// an entity resolver, which would change a public API that mirrors its
    /// upstream, so it is recorded here rather than papered over.
    fn start_prefix_mapping(&mut self, _prefix: Option<&str>, _uri: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called when a namespace prefix binding goes out of scope, after the
    /// matching [`end_element`](Self::end_element) call. Bindings declared on the
    /// same element end in the reverse of the order they began in.
    fn end_prefix_mapping(&mut self, _prefix: Option<&str>) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called at the start of every element, `<uri, local, qname>`-resolved, with
    /// its resolved attributes.
    ///
    /// `uri` is the resolved namespace name (`None` if the element name has no
    /// prefix and there is no default namespace in scope, or if namespace
    /// processing is disabled). `qname` is the raw name as written in the source.
    fn start_element(
        &mut self,
        _uri: Option<&str>,
        _local_name: &str,
        _qname: &str,
        _attributes: &Attributes,
    ) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called at the end of every element, matching a prior
    /// [`start_element`](Self::start_element) call.
    fn end_element(
        &mut self,
        _uri: Option<&str>,
        _local_name: &str,
        _qname: &str,
    ) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called with a run of character data. May be called more than once in a row
    /// for what is logically a single run of text (for example, `CDATA` sections
    /// and character references each produce a separate call). Predefined and
    /// numeric character references are resolved before being reported here.
    fn characters(&mut self, _text: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called with whitespace that a validating parser has determined to be
    /// ignorable, given a DTD content model. Never called by this crate's
    /// [`XMLReader`](crate::XMLReader) — see the trait-level documentation.
    fn ignorable_whitespace(&mut self, _text: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called for a processing instruction `<?target data?>`. `data` is empty
    /// (not `None`) when the instruction has no data.
    fn processing_instruction(&mut self, _target: &str, _data: &str) -> Result<(), SaxError> {
        Ok(())
    }

    /// Called in place of expanding a general entity reference (`&name;`) whose
    /// replacement text this reader does not have: one that is undeclared, one
    /// declared as external (whose replacement text is never retrieved), or one
    /// whose replacement text is markup rather than character data.
    ///
    /// Character references (`&#…;`), the five predefined entities (`amp`, `lt`,
    /// `gt`, `apos`, `quot`) and every general entity declared in the document's
    /// own internal subset are expanded instead, and reported through
    /// [`characters`](Self::characters) — the latter bracketed by
    /// [`LexicalHandler::start_entity`](crate::LexicalHandler::start_entity) and
    /// [`end_entity`](crate::LexicalHandler::end_entity).
    fn skipped_entity(&mut self, _name: &str) -> Result<(), SaxError> {
        Ok(())
    }
}

/// A [`ContentHandler`] that ignores every event. Useful as a placeholder when
/// only, say, the [`LexicalHandler`](crate::LexicalHandler) is of interest.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullContentHandler;

impl ContentHandler for NullContentHandler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_handler_accepts_every_call() {
        let mut handler = NullContentHandler;
        assert!(handler.start_document().is_ok());
        assert!(
            handler
                .start_element(None, "a", "a", &Attributes::new())
                .is_ok()
        );
        assert!(handler.characters("text").is_ok());
        assert!(handler.end_element(None, "a", "a").is_ok());
        assert!(handler.end_document().is_ok());
    }
}
