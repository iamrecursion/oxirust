//! [`XMLReader`], the SAX2-style push-parsing driver.
//!
//! [`XMLReader`] pumps an [`oxixml_xml::reader::NsReader`] and dispatches each
//! event to whichever handlers are installed in the [`Handlers`] bundle passed to
//! [`parse`](XMLReader::parse). See the crate-level documentation for the overall
//! shape of the API and a complete example.

mod tracking;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use oxixml_support::pos::TextPosition;
use oxixml_xml::XmlVersion;
use oxixml_xml::escape::resolve_xml_entity;
use oxixml_xml::events::{BytesEnd, BytesRef, BytesStart, BytesText, Event};
use oxixml_xml::name::{Namespace, PrefixDeclaration, QName, ResolveResult};
use oxixml_xml::reader::{Config, NsReader};
use oxixml_xml::{encoding::EncodingError, errors::Error as XmlError};

use crate::attributes::{Attribute, Attributes};
use crate::dtd_subset::{dispatch_internal_subset, parse_doctype_header, reborrow};
use crate::entity::{ExpandError, Expander, MAX_ENTITY_DEPTH};
use crate::error::{Result, SaxError};
use crate::handlers::{Feature, Handlers};
use crate::locator::{Locator, SaxParseException};

use tracking::PositionTrackingReader;

/// A SAX2-style, push-parsing driver over [`oxixml_xml::reader::NsReader`].
///
/// Configure it (namespace [`Feature`]s, and the underlying
/// [`oxixml_xml::reader::Config`]), then call one of the `parse*` methods with a
/// [`Handlers`] bundle. Because `XMLReader` holds no handler state of its own,
/// the same reader (or a cheap [`Clone`] of its configuration) can be reused to
/// parse many documents, with different handlers each time, from multiple
/// threads if wrapped accordingly.
///
/// # Example
///
/// ```
/// use oxixml_sax::{ContentHandler, Handlers, SaxError, XMLReader};
///
/// struct Print;
/// impl ContentHandler for Print {
///     fn characters(&mut self, text: &str) -> Result<(), SaxError> {
///         print!("{text}");
///         Ok(())
///     }
/// }
///
/// let mut handler = Print;
/// let mut handlers = Handlers::new().with_content(&mut handler);
/// XMLReader::new().parse_str("<msg>hello, world</msg>", &mut handlers)?;
/// # Ok::<(), SaxError>(())
/// ```
#[derive(Debug, Clone)]
pub struct XMLReader {
    xml_config: Config,
    namespaces: bool,
    namespace_prefixes: bool,
}

impl Default for XMLReader {
    fn default() -> Self {
        Self {
            xml_config: Config::default(),
            namespaces: true,
            namespace_prefixes: false,
        }
    }
}

impl XMLReader {
    /// Creates a reader with SAX2 default features: namespace processing on,
    /// namespace-prefix retention off.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The [`oxixml_xml`] parser configuration used for every `parse*` call.
    ///
    /// Note that [`Config::expand_empty_elements`](oxixml_xml::reader::Config::expand_empty_elements)
    /// is always forced to `true` while parsing, regardless of what is set here,
    /// because SAX2 always needs a matched
    /// [`start_element`](crate::ContentHandler::start_element)/
    /// [`end_element`](crate::ContentHandler::end_element) pair, even for a
    /// self-closing `<tag/>`.
    #[must_use]
    pub const fn xml_config(&self) -> &Config {
        &self.xml_config
    }

    /// Mutable access to the [`oxixml_xml`] parser configuration.
    pub fn xml_config_mut(&mut self) -> &mut Config {
        &mut self.xml_config
    }

    /// Returns whether `feature` is currently enabled.
    #[must_use]
    pub const fn feature(&self, feature: Feature) -> bool {
        match feature {
            Feature::Namespaces => self.namespaces,
            Feature::NamespacePrefixes => self.namespace_prefixes,
        }
    }

    /// Enables or disables `feature`.
    pub fn set_feature(&mut self, feature: Feature, value: bool) {
        match feature {
            Feature::Namespaces => self.namespaces = value,
            Feature::NamespacePrefixes => self.namespace_prefixes = value,
        }
    }

    /// Returns whether the feature named by the SAX2 feature URI `uri` is
    /// enabled, or `None` if `uri` does not name a feature this reader supports.
    #[must_use]
    pub fn feature_by_name(&self, uri: &str) -> Option<bool> {
        Feature::from_uri(uri).map(|feature| self.feature(feature))
    }

    /// Enables or disables the feature named by the SAX2 feature URI `uri`.
    ///
    /// # Errors
    ///
    /// Returns [`SaxError::UnknownFeature`] if `uri` does not name a feature this
    /// reader supports.
    pub fn set_feature_by_name(&mut self, uri: &str, value: bool) -> Result<()> {
        match Feature::from_uri(uri) {
            Some(feature) => {
                self.set_feature(feature, value);
                Ok(())
            }
            None => Err(SaxError::UnknownFeature(uri.to_string())),
        }
    }

    /// Parses a document read from `source`, dispatching to `handlers`.
    pub fn parse<R: BufRead>(&self, source: R, handlers: &mut Handlers<'_>) -> Result<()> {
        self.parse_with_system_id(source, None, handlers)
    }

    /// Parses a document held entirely in memory as `xml`.
    pub fn parse_str(&self, xml: &str, handlers: &mut Handlers<'_>) -> Result<()> {
        self.parse(xml.as_bytes(), handlers)
    }

    /// Parses a document read from any [`Read`]er, wrapping it in a
    /// [`BufReader`] internally.
    pub fn parse_reader<R: Read>(&self, source: R, handlers: &mut Handlers<'_>) -> Result<()> {
        self.parse(BufReader::new(source), handlers)
    }

    /// Parses a document read from the file at `path`, which is also reported as
    /// the [`Locator::system_id`] of every position this parse produces.
    ///
    /// # Errors
    ///
    /// Returns [`SaxError::Xml`] if `path` cannot be opened.
    pub fn parse_file<P: AsRef<Path>>(&self, path: P, handlers: &mut Handlers<'_>) -> Result<()> {
        let path = path.as_ref();
        let system_id = path.to_string_lossy().into_owned();
        let file = File::open(path).map_err(|e| SaxError::Xml(XmlError::from(e)))?;
        self.parse_with_system_id(BufReader::new(file), Some(system_id), handlers)
    }

    fn parse_with_system_id<R: BufRead>(
        &self,
        source: R,
        system_id: Option<String>,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        let mut config = self.xml_config.clone();
        // SAX2 always needs a matched start/end pair, even for `<tag/>`.
        config.expand_empty_elements = true;

        let mut reader = NsReader::from_reader(PositionTrackingReader::new(source));
        *reader.config_mut() = config;

        let mut buf = Vec::new();
        let mut driver = Driver::new(system_id, self.namespaces, self.namespace_prefixes);
        driver.run(&mut reader, &mut buf, handlers)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns whether `ch` is one of the four XML 1.0 `S` (white space) characters.
const fn is_xml_whitespace(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\r' | '\n')
}

/// Decodes `bytes` as UTF-8, wrapping any error as a [`SaxError::Xml`].
fn decode_utf8(bytes: &[u8]) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|e| SaxError::Xml(EncodingError::from(e).into()))
}

/// Mutable dispatch state carried across one call to [`XMLReader::parse`] and its
/// siblings: the entity's system identifier, the XML version in effect (for
/// end-of-line normalization), and the stack of namespace prefixes declared by
/// each currently-open element (so that
/// [`ContentHandler::end_prefix_mapping`] can be called in the right order).
struct Driver {
    system_id: Option<String>,
    namespaces: bool,
    namespace_prefixes: bool,
    xml_version: XmlVersion,
    prefix_stack: Vec<Vec<Option<String>>>,
    /// Qualified names of the elements that are currently open, innermost last.
    /// Used to diagnose an entity that ends while an element is still open
    /// (XML 1.0 §3: every start tag must have a matching end tag) and to tell
    /// content inside the document element from content outside it.
    open_elements: Vec<String>,
    /// The internal general entities declared by the document's `<!DOCTYPE>`,
    /// keyed on their name, each holding the replacement text of
    /// [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement).
    entities: HashMap<String, String>,
    /// The expansion budget for this document's entity references.
    expander: Expander,
}

impl Driver {
    fn new(system_id: Option<String>, namespaces: bool, namespace_prefixes: bool) -> Self {
        Self {
            system_id,
            namespaces,
            namespace_prefixes,
            xml_version: XmlVersion::Implicit1_0,
            prefix_stack: Vec::new(),
            open_elements: Vec::new(),
            entities: HashMap::new(),
            expander: Expander::default(),
        }
    }

    /// Converts a 0-based [`TextPosition`] (the `oxixml-support` convention) into a [`Locator`],
    /// whose line and column are 1-based as mandated by the SAX2 `Locator` contract.
    fn locator_at(&self, position: TextPosition) -> Locator {
        Locator::new(
            position.line + 1,
            position.column + 1,
            position.offset,
            None,
            self.system_id.clone(),
        )
    }

    fn run<R: BufRead>(
        &mut self,
        reader: &mut NsReader<PositionTrackingReader<R>>,
        buf: &mut Vec<u8>,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        let start_locator = self.locator_at(reader.get_ref().position());
        if let Some(content) = reborrow(&mut handlers.content) {
            content.set_document_locator(&start_locator);
            content.start_document()?;
        }

        loop {
            buf.clear();
            let position = reader.get_ref().position();
            match reader.read_event_into(buf) {
                Ok(Event::Eof) => break,
                Ok(event) => {
                    if let Err(err) = self.dispatch_event(reader, event, position, handlers) {
                        self.report_fatal_unless_handler_error(&err, position, handlers)?;
                        return Err(err);
                    }
                }
                Err(XmlError::IllFormed(inner)) => {
                    // XML 1.0 §1.2/§5.1: a well-formedness violation is a *fatal*
                    // error and the processor must not continue normal
                    // processing. `oxixml_xml` would happily keep going (and,
                    // with `check_well_formedness` enabled, would report the very
                    // same error again at the very same position forever, because
                    // an `IllFormed` error deliberately leaves the parse state
                    // untouched), so the driver stops here instead.
                    let exception =
                        SaxParseException::new(inner.to_string(), self.locator_at(position));
                    if let Some(error_handler) = reborrow(&mut handlers.error) {
                        error_handler.fatal_error(&exception)?;
                    }
                    return Err(SaxError::Xml(XmlError::IllFormed(inner)));
                }
                Err(other) => {
                    let exception =
                        SaxParseException::new(other.to_string(), self.locator_at(position));
                    if let Some(error_handler) = reborrow(&mut handlers.error) {
                        error_handler.fatal_error(&exception)?;
                    }
                    return Err(SaxError::Xml(other));
                }
            }
        }

        // The entity ended: every element that was opened must have been closed
        // again (XML 1.0 §3, the "element type match" well-formedness
        // constraint). `oxixml_xml` only diagnoses this with
        // `Config::check_well_formedness` enabled, so the driver always checks.
        if let Some(name) = self.open_elements.last() {
            let message = format!("element `{name}` is not closed at the end of the document");
            let position = reader.get_ref().position();
            return Err(self.fatal_not_well_formed(message, position, handlers)?);
        }

        if let Some(content) = reborrow(&mut handlers.content) {
            content.end_document()?;
        }
        Ok(())
    }

    /// Reports a well-formedness violation the driver itself detected to the
    /// installed [`ErrorHandler`] as a fatal error, and builds the
    /// [`SaxError::NotWellFormed`] the caller is expected to return.
    fn fatal_not_well_formed(
        &self,
        message: String,
        position: TextPosition,
        handlers: &mut Handlers<'_>,
    ) -> Result<SaxError> {
        let exception = SaxParseException::new(message.clone(), self.locator_at(position));
        if let Some(error_handler) = reborrow(&mut handlers.error) {
            error_handler.fatal_error(&exception)?;
        }
        Ok(SaxError::NotWellFormed(message))
    }

    /// Reports `err` to the installed [`ErrorHandler`] as a fatal error, unless
    /// it originated from a handler itself (in which case the handler already
    /// knows why it aborted, and there is no useful XML-level position to report),
    /// or the driver already reported it while producing it.
    fn report_fatal_unless_handler_error(
        &self,
        err: &SaxError,
        position: TextPosition,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        if let SaxError::Xml(xml_err) = err {
            let exception = SaxParseException::new(xml_err.to_string(), self.locator_at(position));
            if let Some(error_handler) = reborrow(&mut handlers.error) {
                error_handler.fatal_error(&exception)?;
            }
        }
        Ok(())
    }

    fn dispatch_event<R: BufRead>(
        &mut self,
        reader: &mut NsReader<PositionTrackingReader<R>>,
        event: Event<'_>,
        position: TextPosition,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        match event {
            Event::Decl(decl) => {
                if let Ok(version) = decl.xml_version() {
                    self.xml_version = version;
                    self.expander.set_version(version);
                }
                Ok(())
            }
            Event::Start(start) => self.dispatch_start(reader, &start, handlers),
            Event::End(end) => self.dispatch_end(reader, &end, handlers),
            Event::Empty(start) => {
                // Should not normally be reached (`parse` always forces
                // `expand_empty_elements`), but handled directly anyway so a
                // future change to that default cannot silently drop elements.
                self.dispatch_start(reader, &start, handlers)?;
                let end = start.to_end();
                self.dispatch_end(reader, &end, handlers)
            }
            Event::Text(text) => {
                let decoded = text
                    .xml_content(self.xml_version)
                    .map_err(|e| SaxError::Xml(e.into()))?;
                self.check_character_data_position(&decoded, position, handlers)?;
                if !decoded.is_empty() {
                    if let Some(content) = reborrow(&mut handlers.content) {
                        content.characters(&decoded)?;
                    }
                }
                Ok(())
            }
            Event::CData(cdata) => {
                let decoded = cdata
                    .xml_content(self.xml_version)
                    .map_err(|e| SaxError::Xml(e.into()))?;
                if self.open_elements.is_empty() {
                    let message = "CDATA section outside the document element".to_string();
                    return Err(self.fatal_not_well_formed(message, position, handlers)?);
                }
                if let Some(lexical) = reborrow(&mut handlers.lexical) {
                    lexical.start_cdata()?;
                }
                if let Some(content) = reborrow(&mut handlers.content) {
                    content.characters(&decoded)?;
                }
                if let Some(lexical) = reborrow(&mut handlers.lexical) {
                    lexical.end_cdata()?;
                }
                Ok(())
            }
            Event::Comment(comment) => {
                if let Some(lexical) = reborrow(&mut handlers.lexical) {
                    let decoded = comment.decode().map_err(|e| SaxError::Xml(e.into()))?;
                    lexical.comment(&decoded)?;
                }
                Ok(())
            }
            Event::PI(pi) => {
                if let Some(content) = reborrow(&mut handlers.content) {
                    let target = decode_utf8(pi.target())?;
                    let data = decode_utf8(pi.content())?;
                    content.processing_instruction(&target, data.trim_start())?;
                }
                Ok(())
            }
            Event::DocType(doctype) => self.dispatch_doctype(&doctype, handlers),
            Event::GeneralRef(reference) => {
                self.dispatch_general_ref(&reference, position, handlers)
            }
            Event::Eof => Ok(()),
        }
    }

    /// Rejects character data that appears outside the document element.
    ///
    /// XML 1.0 §2.1 allows only `Misc` (comments, processing instructions and
    /// white space) before and after the document element, so any non-white-space
    /// character there is a well-formedness violation.
    fn check_character_data_position(
        &self,
        text: &str,
        position: TextPosition,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        if !self.open_elements.is_empty() || text.trim_matches(is_xml_whitespace).is_empty() {
            return Ok(());
        }
        let message = "character data outside the document element".to_string();
        Err(self.fatal_not_well_formed(message, position, handlers)?)
    }

    fn dispatch_start<R: BufRead>(
        &mut self,
        reader: &mut NsReader<PositionTrackingReader<R>>,
        start: &BytesStart<'_>,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        let mut declared_prefixes: Vec<Option<String>> = Vec::new();
        if self.namespaces {
            for attribute in start.attributes().with_checks(false) {
                let attribute = attribute.map_err(|e| SaxError::Xml(e.into()))?;
                let Some(declaration) = attribute.key.as_namespace_binding() else {
                    continue;
                };
                let uri = self.normalized_value(&attribute)?.into_owned();
                let prefix = match declaration {
                    PrefixDeclaration::Default => None,
                    PrefixDeclaration::Named(name) => Some(decode_utf8(name)?),
                };
                if let Some(content) = reborrow(&mut handlers.content) {
                    content.start_prefix_mapping(prefix.as_deref(), &uri)?;
                }
                declared_prefixes.push(prefix);
            }
        }
        self.prefix_stack.push(declared_prefixes);

        let (uri, local_name, qname) = self.resolve_element_name(reader, start.name())?;
        let attributes = self.collect_attributes(reader, start)?;
        self.open_elements.push(qname.clone());

        if let Some(content) = reborrow(&mut handlers.content) {
            content.start_element(uri.as_deref(), &local_name, &qname, &attributes)?;
        }
        Ok(())
    }

    fn dispatch_end<R: BufRead>(
        &mut self,
        reader: &mut NsReader<PositionTrackingReader<R>>,
        end: &BytesEnd<'_>,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        let (uri, local_name, qname) = self.resolve_element_name(reader, end.name())?;
        self.open_elements.pop();
        if let Some(content) = reborrow(&mut handlers.content) {
            content.end_element(uri.as_deref(), &local_name, &qname)?;
        }

        let prefixes = self.prefix_stack.pop().unwrap_or_default();
        if let Some(content) = reborrow(&mut handlers.content) {
            for prefix in prefixes.into_iter().rev() {
                content.end_prefix_mapping(prefix.as_deref())?;
            }
        }
        Ok(())
    }

    /// Normalizes an attribute value, resolving the general entities declared by
    /// the document's internal subset as well as the predefined ones.
    ///
    /// XML 1.0 §4.4 has an internal general entity *included in literal* inside
    /// an attribute value, exactly as it is included in content, so an attribute
    /// that uses one must not be rejected as an unknown escape.
    fn normalized_value<'a>(
        &mut self,
        attribute: &oxixml_xml::events::attributes::Attribute<'a>,
    ) -> Result<std::borrow::Cow<'a, str>> {
        let entities = &self.entities;
        let expander = &mut self.expander;
        // The budget is charged from inside the resolver because
        // `normalized_value_with` walks the nested references itself: it is the
        // only point at which each occurrence — including the ones it reaches
        // recursively — is visible. Refusing to resolve is how a spent budget is
        // signalled, and is reported as the well-formedness violation it is.
        let mut over_budget: Option<String> = None;
        let value = attribute.normalized_value_with(self.xml_version, MAX_ENTITY_DEPTH, |name| {
            if let Some(text) = resolve_xml_entity(name) {
                return Some(text);
            }
            let text = entities.get(name)?;
            if over_budget.is_some() || !expander.charge_attribute_expansion(name, text.len()) {
                over_budget.get_or_insert_with(|| name.to_owned());
                return None;
            }
            Some(text.as_str())
        });
        match (value, over_budget) {
            (Ok(value), _) => Ok(value),
            (Err(_), Some(entity)) => Err(SaxError::NotWellFormed(format!(
                "expanding the general entity `&{entity};` would exceed the limits on entity \
                 expansion"
            ))),
            (Err(error), None) => Err(SaxError::Xml(error)),
        }
    }

    /// Resolves an element's (or end tag's) name into `(uri, local_name, qname)`.
    /// Unknown prefixes (a document that declares no binding for a prefix it
    /// uses) are treated permissively as unbound, matching this crate's
    /// generally permissive-by-default parsing.
    fn resolve_element_name<R: BufRead>(
        &self,
        reader: &NsReader<PositionTrackingReader<R>>,
        name: QName<'_>,
    ) -> Result<(Option<String>, String, String)> {
        let qname = decode_utf8(name.as_ref())?;
        if !self.namespaces {
            return Ok((None, qname.clone(), qname));
        }
        let (resolved, local) = reader.resolve(name, false);
        let uri = match resolved {
            ResolveResult::Bound(Namespace(bytes)) => Some(decode_utf8(bytes)?),
            ResolveResult::Unbound | ResolveResult::Unknown(_) => None,
        };
        let local_name = decode_utf8(local.as_ref())?;
        Ok((uri, local_name, qname))
    }

    fn collect_attributes<R: BufRead>(
        &mut self,
        reader: &NsReader<PositionTrackingReader<R>>,
        start: &BytesStart<'_>,
    ) -> Result<Attributes> {
        let mut attributes = Attributes::new();
        for attribute in start.attributes() {
            let attribute = attribute.map_err(|e| SaxError::Xml(e.into()))?;
            let is_namespace_decl = attribute.key.as_namespace_binding().is_some();
            // Namespace declaration attributes are only ever excluded as part of
            // namespace processing itself; with `namespaces` disabled they are
            // just ordinary attributes.
            if self.namespaces && is_namespace_decl && !self.namespace_prefixes {
                continue;
            }

            let qname = decode_utf8(attribute.key.as_ref())?;
            let value = self.normalized_value(&attribute)?.into_owned();

            let (uri, local_name) = if !self.namespaces {
                (None, qname.clone())
            } else if is_namespace_decl {
                (None, decode_utf8(attribute.key.local_name().as_ref())?)
            } else {
                let (resolved, local) = reader.resolve(attribute.key, true);
                let uri = match resolved {
                    ResolveResult::Bound(Namespace(bytes)) => Some(decode_utf8(bytes)?),
                    ResolveResult::Unbound | ResolveResult::Unknown(_) => None,
                };
                (uri, decode_utf8(local.as_ref())?)
            };

            attributes.push(Attribute {
                uri,
                local_name,
                qname,
                value,
                attr_type: Attribute::CDATA.to_string(),
            });
        }
        Ok(attributes)
    }

    /// Reports a `&…;` reference found in content.
    ///
    /// XML 1.0 §4.4 has a non-validating processor *include* an internal general
    /// entity unconditionally: its replacement text is part of the document
    /// itself, so there is nothing to fetch and nothing to skip. Only an entity
    /// that is undeclared, external, or whose replacement text is markup rather
    /// than character data reaches
    /// [`ContentHandler::skipped_entity`](crate::ContentHandler::skipped_entity).
    fn dispatch_general_ref(
        &mut self,
        reference: &BytesRef<'_>,
        position: TextPosition,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        if reference.is_char_ref() {
            if let Some(ch) = reference.resolve_char_ref().map_err(SaxError::Xml)? {
                let mut buf = [0u8; 4];
                let text = ch.encode_utf8(&mut buf);
                if let Some(content) = reborrow(&mut handlers.content) {
                    content.characters(text)?;
                }
            }
            return Ok(());
        }

        let name = reference.decode().map_err(|e| SaxError::Xml(e.into()))?;
        if let Some(replacement) = resolve_xml_entity(&name) {
            if let Some(content) = reborrow(&mut handlers.content) {
                content.characters(replacement)?;
            }
            return Ok(());
        }
        match self.expander.expand(&name, &self.entities) {
            Ok(text) => {
                if let Some(lexical) = reborrow(&mut handlers.lexical) {
                    lexical.start_entity(&name)?;
                }
                if !text.is_empty() {
                    if let Some(content) = reborrow(&mut handlers.content) {
                        content.characters(&text)?;
                    }
                }
                if let Some(lexical) = reborrow(&mut handlers.lexical) {
                    lexical.end_entity(&name)?;
                }
                Ok(())
            }
            Err(ExpandError::Undeclared { .. } | ExpandError::Markup { .. }) => {
                if let Some(content) = reborrow(&mut handlers.content) {
                    content.skipped_entity(&name)?;
                }
                Ok(())
            }
            // Recursion is a well-formedness violation, and an exhausted budget
            // means the document is trying to expand without bound; neither may
            // be reported as a merely skipped entity.
            Err(error) => Err(self.fatal_not_well_formed(error.to_string(), position, handlers)?),
        }
    }

    fn dispatch_doctype(
        &mut self,
        doctype: &BytesText<'_>,
        handlers: &mut Handlers<'_>,
    ) -> Result<()> {
        let content = doctype.decode().map_err(|e| SaxError::Xml(e.into()))?;
        let header = parse_doctype_header(&content);

        if header.public_id.is_some() || header.system_id.is_some() {
            if let Some(resolver) = reborrow(&mut handlers.entity_resolver) {
                let _ = resolver
                    .resolve_entity(header.public_id.as_deref(), header.system_id.as_deref());
            }
        }

        if let Some(lexical) = reborrow(&mut handlers.lexical) {
            lexical.start_dtd(
                header.name,
                header.public_id.as_deref(),
                header.system_id.as_deref(),
            )?;
        }
        if let Some(subset) = header.internal_subset {
            // `dispatch_internal_subset` takes ownership of these (rather than
            // reborrowing the fields in place) to sidestep `&mut` reference
            // invariance; there is only ever one `<!DOCTYPE>` per document, so
            // there is no need to put them back afterwards.
            dispatch_internal_subset(
                subset,
                handlers.dtd.take(),
                handlers.decl.take(),
                handlers.entity_resolver.take(),
                &mut self.entities,
                self.xml_version,
            )?;
        }
        if let Some(lexical) = reborrow(&mut handlers.lexical) {
            lexical.end_dtd()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_features_match_sax2_core() {
        let reader = XMLReader::new();
        assert!(reader.feature(Feature::Namespaces));
        assert!(!reader.feature(Feature::NamespacePrefixes));
    }

    #[test]
    fn set_feature_by_name_round_trips() {
        let mut reader = XMLReader::new();
        reader
            .set_feature_by_name("http://xml.org/sax/features/namespace-prefixes", true)
            .expect("known feature");
        assert!(reader.feature(Feature::NamespacePrefixes));
        assert_eq!(
            reader.feature_by_name("http://xml.org/sax/features/namespace-prefixes"),
            Some(true)
        );
        assert_eq!(reader.feature_by_name("http://example.com/nope"), None);
    }

    #[test]
    fn set_feature_by_name_rejects_unknown_uri() {
        let mut reader = XMLReader::new();
        let err = reader
            .set_feature_by_name("http://example.com/nope", true)
            .expect_err("unknown feature");
        assert!(matches!(err, SaxError::UnknownFeature(_)));
    }
}
