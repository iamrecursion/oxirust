//! The reader-agnostic core of the RDF/XML parser.
//!
//! [`InternalRdfXmlParser`] consumes XML events one at a time and pushes the triples it derives
//! into a caller-owned buffer. The public [`RdfXmlParser`](crate::RdfXmlParser) wrappers only
//! differ in how they obtain those events (buffered reader vs. borrowed slice).

use crate::error::{RdfXmlParseError, RdfXmlSyntaxError};
use crate::parser::state::{
    EntityRegistry, NodeOrText, RDF_ABOUT, RDF_BAG_ID, RDF_DATATYPE, RDF_DESCRIPTION, RDF_ID,
    RDF_LI, RDF_NODE_ID, RDF_PARSE_TYPE, RDF_RDF, RDF_RESOURCE, RESERVED_RDF_ATTRIBUTES,
    RESERVED_RDF_ELEMENTS, RdfXmlState, is_object_defined, is_utf8, is_whitespace,
};
use crate::utils::is_nc_name;
use oxixml_iri::{Iri, IriRef};
use oxixml_langtag::LanguageTag;
use oxixml_model::vocab::rdf;
use oxixml_model::{BlankNode, Literal, NamedNode, NamedOrBlankNode, OxString, Term, Triple};
use oxixml_xml::escape::{EscapeError, unescape_with};
use oxixml_xml::events::attributes::Attribute;
use oxixml_xml::events::{BytesEnd, BytesRef, BytesStart, BytesText, Event};
use oxixml_xml::name::{LocalName, Namespace, PrefixDeclaration, ResolveResult};
use oxixml_xml::{Error, NsReader, Writer, XmlVersion};
use std::borrow::Cow;
use std::collections::HashSet;

/// Maximum depth of nested entity expansions, guarding against billion-laughs style inputs.
const MAX_ENTITY_NESTING: usize = 1024;

/// The streaming RDF/XML state machine.
pub(crate) struct InternalRdfXmlParser<R> {
    /// The namespace-aware XML pull reader.
    pub(crate) reader: NsReader<R>,
    /// The stack of currently open RDF/XML productions.
    pub(crate) state: Vec<RdfXmlState>,
    /// The `<!ENTITY …>` declarations of the internal DTD subset.
    pub(crate) custom_entities: EntityRegistry,
    /// How deep we currently are inside a captured `rdf:parseType="Literal"` subtree.
    pub(crate) in_literal_depth: usize,
    /// Every `rdf:ID`-derived IRI seen so far, to reject reuse.
    pub(crate) known_rdf_id: HashSet<OxString>,
    /// Whether the end of the document has been reached.
    pub(crate) is_end: bool,
    /// Whether validation is skipped.
    pub(crate) lenient: bool,
    /// The XML version declared by the document.
    pub(crate) xml_version: XmlVersion,
    /// Character data accumulated since the last start or end tag.
    pub(crate) text_buffer: String,
}

/// The `rdf:parseType` values the grammar distinguishes.
#[derive(PartialEq, Eq)]
enum RdfXmlParseType {
    /// No `rdf:parseType`.
    Default,
    /// `rdf:parseType="Collection"`.
    Collection,
    /// `rdf:parseType="Literal"`, or — per the `parseTypeOtherPropertyElt` production — any
    /// value other than `Collection`, `Literal` and `Resource`.
    Literal,
    /// `rdf:parseType="Resource"`.
    Resource,
}

/// Which production the grammar allows as the next child element.
enum RdfXmlNextProduction {
    /// The document element.
    Rdf,
    /// A node element.
    NodeElt,
    /// A property element of the given subject.
    PropertyElt {
        /// The subject the property element attaches to.
        subject: NamedOrBlankNode,
    },
}

/// The attributes of a start tag, after validation and base-IRI resolution.
struct ParsedAttributes {
    /// The `xml:lang` value.
    language: Option<OxString>,
    /// The `xml:base` value.
    base_iri: Option<Iri<OxString>>,
    /// The IRI `rdf:ID` resolves to.
    id_attr: Option<NamedNode>,
    /// The blank node `rdf:nodeID` denotes.
    node_id_attr: Option<BlankNode>,
    /// The IRI `rdf:about` resolves to.
    about_attr: Option<NamedNode>,
    /// The (predicate, lexical value) pairs of the property attributes.
    property_attrs: Vec<(NamedNode, OxString)>,
    /// The IRI `rdf:resource` resolves to.
    resource_attr: Option<NamedNode>,
    /// The IRI `rdf:datatype` resolves to.
    datatype_attr: Option<NamedNode>,
    /// The `rdf:parseType` value.
    parse_type: RdfXmlParseType,
    /// The IRI `rdf:type` resolves to.
    type_attr: Option<NamedNode>,
}

impl<R> InternalRdfXmlParser<R> {
    /// Consumes one XML event, appending any derived triples to `results`.
    pub(crate) fn parse_event(
        &mut self,
        event: Event<'_>,
        results: &mut Vec<Triple>,
    ) -> Result<(), RdfXmlParseError> {
        match event {
            Event::Start(event) => {
                // We make sure we always run both, even if one fails, for better error recovery.
                let text_error = self.flush_text_buffer();
                let start_error = self.parse_start_event(&event, results);
                text_error.and(start_error)
            }
            Event::End(event) => {
                // We make sure we always run both, even if one fails, for better error recovery.
                let text_error = self.flush_text_buffer();
                let end_error = self.parse_end_event(&event, results);
                text_error.and(end_error)
            }
            Event::Empty(_) => Err(RdfXmlSyntaxError::msg(
                "The expand_empty_elements option must be enabled",
            )
            .into()),
            Event::Text(event) => {
                self.text_buffer
                    .push_str(&event.xml_content(self.xml_version)?);
                Ok(())
            }
            Event::GeneralRef(event) => {
                self.decode_xml_entity(&event)?;
                Ok(())
            }
            Event::CData(event) => {
                self.text_buffer
                    .push_str(&event.xml_content(self.xml_version)?);
                Ok(())
            }
            Event::Comment(_) | Event::PI(_) => Ok(()),
            Event::Decl(event) => {
                self.xml_version = event.xml_version()?;
                if let Some(encoding) = event.encoding() {
                    if !is_utf8(&encoding?) {
                        return Err(RdfXmlSyntaxError::msg(
                            "Only UTF-8 is supported by the RDF/XML parser",
                        )
                        .into());
                    }
                }
                Ok(())
            }
            Event::DocType(dt) => self.parse_doctype(&dt),
            Event::Eof => {
                self.is_end = true;
                self.flush_text_buffer()
            }
        }
    }

    /// Hands the accumulated character data to the current production, if any.
    fn flush_text_buffer(&mut self) -> Result<(), RdfXmlParseError> {
        if self.text_buffer.is_empty() {
            return Ok(());
        }
        let text = OxString::new_owned(&self.text_buffer);
        self.text_buffer.clear();
        self.parse_text_event(text)
    }

    /// Extracts the `<!ENTITY …>` declarations of an internal DTD subset.
    fn parse_doctype(&mut self, dt: &BytesText<'_>) -> Result<(), RdfXmlParseError> {
        for input in dt.xml_content(self.xml_version)?.split('<').skip(1) {
            if let Some(input) = input.strip_prefix("!ENTITY") {
                let input = input.trim_start().strip_prefix('%').unwrap_or(input);
                let (entity_name, input) = input.trim_start().split_once(|c: char| c.is_ascii_whitespace()).ok_or_else(|| {
                    RdfXmlSyntaxError::msg(
                        "<!ENTITY declarations should contain both an entity name and an entity value",
                    )
                })?;
                let input = input.trim_start().strip_prefix('\"').ok_or_else(|| {
                    RdfXmlSyntaxError::msg("<!ENTITY values should be enclosed in double quotes")
                })?;
                let (entity_value, input) = input.split_once('"').ok_or_else(|| {
                    RdfXmlSyntaxError::msg(
                        "<!ENTITY declarations values should be enclosed in double quotes",
                    )
                })?;
                input.trim_start().strip_prefix('>').ok_or_else(|| {
                    RdfXmlSyntaxError::msg("<!ENTITY declarations values should end with >")
                })?;

                // Resolves custom entities within the current entity definition.
                let entity_value = unescape_with(entity_value, |e| self.custom_entities.resolve(e))
                    .map_err(Error::from)?;
                self.custom_entities
                    .0
                    .insert(entity_name.to_owned(), entity_value.to_string());
            }
        }
        Ok(())
    }

    /// Handles an opening tag, pushing the matching production onto the state stack.
    #[expect(clippy::too_many_lines)]
    fn parse_start_event(
        &mut self,
        event: &BytesStart<'_>,
        results: &mut Vec<Triple>,
    ) -> Result<(), RdfXmlParseError> {
        // `rdf:parseType="Literal"` case: the whole subtree is copied verbatim.
        if matches!(
            self.state.last(),
            Some(RdfXmlState::ParseTypeLiteralPropertyElt { .. })
        ) {
            return self.capture_literal_start_event(event);
        }

        let (tag_namespace, tag_local_name) = self.reader.resolver().resolve_element(event.name());
        let tag_name = self.resolve_ns_name(tag_namespace, tag_local_name)?;

        let ParsedAttributes {
            language,
            base_iri,
            id_attr,
            node_id_attr,
            about_attr,
            property_attrs,
            resource_attr,
            datatype_attr,
            parse_type,
            type_attr,
        } = self.read_attributes(event)?;

        let expected_production = match self.state.last() {
            Some(RdfXmlState::Doc { .. }) => RdfXmlNextProduction::Rdf,
            Some(
                RdfXmlState::Rdf { .. }
                | RdfXmlState::PropertyElt { .. }
                | RdfXmlState::ParseTypeCollectionPropertyElt { .. },
            ) => RdfXmlNextProduction::NodeElt,
            Some(RdfXmlState::NodeElt { subject, .. }) => RdfXmlNextProduction::PropertyElt {
                subject: subject.clone(),
            },
            Some(RdfXmlState::ParseTypeLiteralPropertyElt { .. }) => {
                return Err(RdfXmlSyntaxError::msg(
                    "ParseTypeLiteralPropertyElt production children should never be considered as an RDF/XML content",
                )
                .into());
            }
            None => {
                return Err(RdfXmlSyntaxError::msg(
                    "No state in the stack: the XML is not balanced",
                )
                .into());
            }
        };

        let new_state = match expected_production {
            RdfXmlNextProduction::Rdf => {
                if *tag_name == *RDF_RDF {
                    RdfXmlState::Rdf { base_iri, language }
                } else if RESERVED_RDF_ELEMENTS.contains(&&*tag_name) {
                    return Err(RdfXmlSyntaxError::msg(format!(
                        "Invalid node element tag name: {tag_name}"
                    ))
                    .into());
                } else {
                    self.build_node_elt(
                        self.parse_iri(tag_name)?,
                        base_iri,
                        language,
                        id_attr,
                        node_id_attr,
                        about_attr,
                        type_attr,
                        property_attrs,
                        results,
                    )?
                }
            }
            RdfXmlNextProduction::NodeElt => {
                if RESERVED_RDF_ELEMENTS.contains(&&*tag_name) {
                    return Err(RdfXmlSyntaxError::msg(format!(
                        "Invalid node element tag name: {tag_name}"
                    ))
                    .into());
                }
                self.build_node_elt(
                    self.parse_iri(tag_name)?,
                    base_iri,
                    language,
                    id_attr,
                    node_id_attr,
                    about_attr,
                    type_attr,
                    property_attrs,
                    results,
                )?
            }
            RdfXmlNextProduction::PropertyElt { subject } => {
                let iri = if *tag_name == *RDF_LI {
                    let Some(RdfXmlState::NodeElt { li_counter, .. }) = self.state.last_mut()
                    else {
                        return Err(RdfXmlSyntaxError::msg(format!(
                            "Invalid property element tag name: {tag_name}"
                        ))
                        .into());
                    };
                    *li_counter += 1;
                    NamedNode::new_unchecked(OxString::new_owned(&format!(
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#_{li_counter}"
                    )))
                } else if RESERVED_RDF_ELEMENTS.contains(&&*tag_name)
                    || *tag_name == *RDF_DESCRIPTION
                {
                    return Err(RdfXmlSyntaxError::msg(format!(
                        "Invalid property element tag name: {tag_name}"
                    ))
                    .into());
                } else {
                    self.parse_iri(tag_name)?
                };
                // The `parseType{Literal,Resource,Collection}PropertyElt` productions allow only
                // `rdf:ID` beside `rdf:parseType` itself, so anything that would supply a second,
                // conflicting object is an error (W3C `rdfms-empty-property-elements` error001 and
                // error002 combine `rdf:parseType="Literal"` with `rdf:resource`).
                if parse_type != RdfXmlParseType::Default
                    && (resource_attr.is_some()
                        || node_id_attr.is_some()
                        || datatype_attr.is_some()
                        || type_attr.is_some()
                        || !property_attrs.is_empty())
                {
                    return Err(RdfXmlSyntaxError::msg(
                        "rdf:parseType must not be combined with rdf:resource, rdf:nodeID, \
                         rdf:datatype, rdf:type or a property attribute",
                    )
                    .into());
                }
                match parse_type {
                    RdfXmlParseType::Default => {
                        if resource_attr.is_some()
                            || node_id_attr.is_some()
                            || !property_attrs.is_empty()
                        {
                            let object = match (resource_attr, node_id_attr) {
                                (Some(resource_attr), None) => {
                                    NamedOrBlankNode::from(resource_attr)
                                }
                                (None, Some(node_id_attr)) => node_id_attr.into(),
                                (None, None) => BlankNode::default().into(),
                                (Some(_), Some(_)) => {
                                    return Err(RdfXmlSyntaxError::msg(
                                        "Not both rdf:resource and rdf:nodeID could be set at the same time",
                                    )
                                    .into());
                                }
                            };
                            self.emit_property_attrs(
                                &object,
                                property_attrs,
                                language.as_ref(),
                                results,
                            );
                            if let Some(type_attr) = type_attr {
                                results.push(Triple::new(object.clone(), rdf::TYPE, type_attr));
                            }
                            RdfXmlState::PropertyElt {
                                iri,
                                base_iri,
                                language,
                                subject,
                                object: Some(NodeOrText::Node(object)),
                                id_attr,
                                datatype_attr,
                            }
                        } else {
                            RdfXmlState::PropertyElt {
                                iri,
                                base_iri,
                                language,
                                subject,
                                object: None,
                                id_attr,
                                datatype_attr,
                            }
                        }
                    }
                    RdfXmlParseType::Literal => RdfXmlState::ParseTypeLiteralPropertyElt {
                        iri,
                        base_iri,
                        language,
                        subject,
                        writer: Writer::new(Vec::new()),
                        id_attr,
                    },
                    RdfXmlParseType::Resource => self.build_parse_type_resource_property_elt(
                        iri, base_iri, language, subject, id_attr, results,
                    ),
                    RdfXmlParseType::Collection => RdfXmlState::ParseTypeCollectionPropertyElt {
                        iri,
                        base_iri,
                        language,
                        subject,
                        objects: Vec::new(),
                        id_attr,
                    },
                }
            }
        };
        self.state.push(new_state);
        Ok(())
    }

    /// Copies a start tag verbatim into the XML literal currently being captured.
    ///
    /// At the top level of the literal every in-scope namespace declaration is re-emitted, which
    /// is what makes the captured markup self-contained.
    fn capture_literal_start_event(
        &mut self,
        event: &BytesStart<'_>,
    ) -> Result<(), RdfXmlParseError> {
        let mut clean_event = BytesStart::new(
            self.reader
                .decoder()
                .decode(event.name().as_ref())?
                .to_string(),
        );
        for attr in event.attributes() {
            clean_event.push_attribute(attr.map_err(Error::InvalidAttr)?);
        }
        if self.in_literal_depth == 0 {
            for (prefix, namespace) in self.reader.resolver().bindings() {
                let namespace = self.reader.decoder().decode(namespace.into_inner())?;
                if let Err(error) = Iri::parse(namespace.as_ref()) {
                    return Err(RdfXmlSyntaxError::invalid_iri(
                        OxString::new_owned(&namespace),
                        error,
                    )
                    .into());
                }
                match prefix {
                    PrefixDeclaration::Default => {
                        clean_event.push_attribute(("xmlns".as_bytes(), namespace.as_bytes()));
                    }
                    PrefixDeclaration::Named(name) => {
                        let mut attr = Vec::with_capacity(6 + name.len());
                        attr.extend_from_slice(b"xmlns:");
                        attr.extend_from_slice(name);
                        clean_event.push_attribute((attr.as_slice(), namespace.as_bytes()));
                    }
                }
            }
        }
        let Some(RdfXmlState::ParseTypeLiteralPropertyElt { writer, .. }) = self.state.last_mut()
        else {
            return Err(RdfXmlSyntaxError::msg("No XML literal production is open").into());
        };
        writer.write_event(Event::Start(clean_event))?;
        self.in_literal_depth += 1;
        Ok(())
    }

    /// Reads, validates and base-resolves every attribute of an element.
    #[expect(clippy::too_many_lines)]
    fn read_attributes(
        &mut self,
        event: &BytesStart<'_>,
    ) -> Result<ParsedAttributes, RdfXmlParseError> {
        let mut language = None;
        let mut base_iri = None;
        let mut id_attr = None;
        let mut node_id_attr = None;
        let mut about_attr = None;
        let mut property_attrs = Vec::new();
        let mut resource_attr = None;
        let mut datatype_attr = None;
        let mut parse_type = RdfXmlParseType::Default;
        let mut type_attr = None;

        for attribute in event.attributes() {
            let attribute = attribute.map_err(Error::InvalidAttr)?;
            let (attribute_namespace, attribute_local_name) =
                self.reader.resolver().resolve_attribute(attribute.key);
            if attribute_namespace
                == ResolveResult::Bound(Namespace(b"http://www.w3.org/XML/1998/namespace"))
            {
                match attribute.key.local_name().as_ref() {
                    b"lang" => {
                        let mut tag = OxString::new_owned(&self.convert_attribute(&attribute)?);
                        tag.make_mut().make_ascii_lowercase();
                        // `xml:lang=""` removes the in-scope language, it is not a language tag.
                        language = Some(if self.lenient || tag.is_empty() {
                            tag
                        } else {
                            LanguageTag::parse(tag.clone())
                                .map_err(|error| {
                                    RdfXmlSyntaxError::invalid_language_tag(tag, error)
                                })?
                                .into_inner()
                        });
                    }
                    b"base" => {
                        let iri = OxString::new_owned(&self.convert_attribute(&attribute)?);
                        base_iri = Some(if self.lenient {
                            Iri::parse_unchecked(iri)
                        } else {
                            Iri::parse(iri.clone())
                                .map_err(|error| RdfXmlSyntaxError::invalid_iri(iri, error))?
                        });
                    }
                    _ => (), // We ignore other xml attributes
                }
            } else if attribute.key.as_ref().starts_with(b"xml") {
                // We ignore other xml attributes
            } else {
                let attribute_url =
                    self.resolve_ns_name(attribute_namespace.clone(), attribute_local_name)?;
                if *attribute_url == *RDF_ID {
                    let mut id = self.convert_attribute(&attribute)?.into_owned();
                    if !is_nc_name(&id) {
                        return Err(RdfXmlSyntaxError::msg(format!(
                            "{id} is not a valid rdf:ID value"
                        ))
                        .into());
                    }
                    id.insert(0, '#');
                    id_attr = Some(id);
                } else if *attribute_url == *RDF_BAG_ID {
                    let bag_id = self.convert_attribute(&attribute)?;
                    if !is_nc_name(&bag_id) {
                        return Err(RdfXmlSyntaxError::msg(format!(
                            "{bag_id} is not a valid rdf:bagID value"
                        ))
                        .into());
                    }
                } else if *attribute_url == *RDF_NODE_ID {
                    let id = self.convert_attribute(&attribute)?;
                    if !is_nc_name(&id) {
                        return Err(RdfXmlSyntaxError::msg(format!(
                            "{id} is not a valid rdf:nodeID value"
                        ))
                        .into());
                    }
                    node_id_attr = Some(BlankNode::new_unchecked(OxString::new_owned(&id)));
                } else if *attribute_url == *RDF_ABOUT {
                    about_attr = Some(attribute);
                } else if *attribute_url == *RDF_RESOURCE {
                    resource_attr = Some(attribute);
                } else if *attribute_url == *RDF_DATATYPE {
                    datatype_attr = Some(attribute);
                } else if *attribute_url == *RDF_PARSE_TYPE {
                    parse_type = match self.convert_attribute(&attribute)?.as_ref() {
                        "Collection" => RdfXmlParseType::Collection,
                        "Literal" => RdfXmlParseType::Literal,
                        "Resource" => RdfXmlParseType::Resource,
                        // RDF/XML 1.1 sec. 7.2.20 (parseTypeOtherPropertyElt): "All
                        // rdf:parseType attribute values other than the strings "Resource",
                        // "Literal" or "Collection" are treated as if the value was "Literal"".
                        // That includes a mis-cased "literal", which must not silently drop the
                        // statement.
                        _ => RdfXmlParseType::Literal,
                    };
                } else if attribute_url == rdf::TYPE.as_str() {
                    type_attr = Some(attribute);
                } else if RESERVED_RDF_ATTRIBUTES.contains(&&*attribute_url) {
                    return Err(RdfXmlSyntaxError::msg(format!(
                        "{attribute_url} is not a valid attribute"
                    ))
                    .into());
                } else {
                    property_attrs.push((
                        self.parse_iri(attribute_url)?,
                        OxString::new_owned(&self.convert_attribute(&attribute)?),
                    ));
                }
            }
        }

        // Parsing with the base URI
        let id_attr = if let Some(iri) = id_attr {
            let iri = self.resolve_iri(base_iri.as_ref(), &iri)?;
            if !self.lenient {
                if self.known_rdf_id.contains(iri.as_str()) {
                    return Err(RdfXmlSyntaxError::msg(format!(
                        "{iri} has already been used as rdf:ID value"
                    ))
                    .into());
                }
                self.known_rdf_id.insert(iri.clone().into_string());
            }
            Some(iri)
        } else {
            None
        };
        let about_attr = if let Some(attr) = about_attr {
            Some(self.convert_iri_attribute(base_iri.as_ref(), &attr)?)
        } else {
            None
        };
        let resource_attr = if let Some(attr) = resource_attr {
            Some(self.convert_iri_attribute(base_iri.as_ref(), &attr)?)
        } else {
            None
        };
        let datatype_attr = if let Some(attr) = datatype_attr {
            Some(self.convert_iri_attribute(base_iri.as_ref(), &attr)?)
        } else {
            None
        };
        let type_attr = if let Some(attr) = type_attr {
            Some(self.convert_iri_attribute(base_iri.as_ref(), &attr)?)
        } else {
            None
        };

        Ok(ParsedAttributes {
            language,
            base_iri,
            id_attr,
            node_id_attr,
            about_attr,
            property_attrs,
            resource_attr,
            datatype_attr,
            parse_type,
            type_attr,
        })
    }

    /// Handles a closing tag, popping and finalizing the matching production.
    fn parse_end_event(
        &mut self,
        event: &BytesEnd<'_>,
        results: &mut Vec<Triple>,
    ) -> Result<(), RdfXmlParseError> {
        // `rdf:parseType="Literal"` case.
        if self.in_literal_depth > 0 {
            let name = self
                .reader
                .decoder()
                .decode(event.name().as_ref())?
                .into_owned();
            let Some(RdfXmlState::ParseTypeLiteralPropertyElt { writer, .. }) =
                self.state.last_mut()
            else {
                return Err(RdfXmlSyntaxError::msg(
                    "Unbalanced XML literal: no literal production is open",
                )
                .into());
            };
            writer.write_event(Event::End(BytesEnd::new(name)))?;
            self.in_literal_depth -= 1;
            return Ok(());
        }

        if let Some(current_state) = self.state.pop() {
            self.end_state(current_state, results)?;
        }
        Ok(())
    }

    /// Hands character data to the current production.
    fn parse_text_event(&mut self, text: OxString) -> Result<(), RdfXmlParseError> {
        match self.state.last_mut() {
            Some(RdfXmlState::PropertyElt { object, .. }) => {
                if is_object_defined(object) {
                    if text.bytes().all(is_whitespace) {
                        Ok(()) // whitespace anyway, we ignore
                    } else {
                        Err(
                            RdfXmlSyntaxError::msg(format!("Unexpected text event: '{text}'"))
                                .into(),
                        )
                    }
                } else {
                    *object = Some(NodeOrText::Text(text));
                    Ok(())
                }
            }
            Some(RdfXmlState::ParseTypeLiteralPropertyElt { writer, .. }) => {
                writer.write_event(Event::Text(BytesText::new(&text)))?;
                Ok(())
            }
            _ => {
                if text.bytes().all(is_whitespace) {
                    Ok(())
                } else {
                    Err(RdfXmlSyntaxError::msg(format!("Unexpected text event: '{text}'")).into())
                }
            }
        }
    }

    /// Concatenates a resolved namespace and a local name into an absolute IRI.
    fn resolve_ns_name(
        &self,
        namespace: ResolveResult<'_>,
        local_name: LocalName<'_>,
    ) -> Result<OxString, RdfXmlParseError> {
        match namespace {
            ResolveResult::Bound(ns) => {
                let mut value = Vec::with_capacity(ns.as_ref().len() + local_name.as_ref().len());
                value.extend_from_slice(ns.as_ref());
                value.extend_from_slice(local_name.as_ref());
                Ok(OxString::new_owned(
                    &unescape_with(&self.reader.decoder().decode(&value)?, |e| {
                        self.custom_entities.resolve(e)
                    })
                    .map_err(Error::from)?,
                ))
            }
            ResolveResult::Unbound => {
                Err(RdfXmlSyntaxError::msg("XML namespaces are required in RDF/XML").into())
            }
            ResolveResult::Unknown(v) => Err(RdfXmlSyntaxError::msg(format!(
                "Unknown prefix {}:",
                self.reader.decoder().decode(&v)?
            ))
            .into()),
        }
    }

    /// Builds the `nodeElement` production, emitting its type and property-attribute triples.
    #[expect(clippy::too_many_arguments)]
    fn build_node_elt(
        &mut self,
        iri: NamedNode,
        base_iri: Option<Iri<OxString>>,
        language: Option<OxString>,
        id_attr: Option<NamedNode>,
        node_id_attr: Option<BlankNode>,
        about_attr: Option<NamedNode>,
        type_attr: Option<NamedNode>,
        property_attrs: Vec<(NamedNode, OxString)>,
        results: &mut Vec<Triple>,
    ) -> Result<RdfXmlState, RdfXmlSyntaxError> {
        let subject = match (id_attr, node_id_attr, about_attr) {
            (Some(id_attr), None, None) => NamedOrBlankNode::from(id_attr),
            (None, Some(node_id_attr), None) => node_id_attr.into(),
            (None, None, Some(about_attr)) => about_attr.into(),
            (None, None, None) => BlankNode::default().into(),
            (Some(_), Some(_), _) => {
                return Err(RdfXmlSyntaxError::msg(
                    "Not both rdf:ID and rdf:nodeID could be set at the same time",
                ));
            }
            (_, Some(_), Some(_)) => {
                return Err(RdfXmlSyntaxError::msg(
                    "Not both rdf:nodeID and rdf:about could be set at the same time",
                ));
            }
            (Some(_), _, Some(_)) => {
                return Err(RdfXmlSyntaxError::msg(
                    "Not both rdf:ID and rdf:about could be set at the same time",
                ));
            }
        };

        self.emit_property_attrs(&subject, property_attrs, language.as_ref(), results);

        if let Some(type_attr) = type_attr {
            results.push(Triple::new(subject.clone(), rdf::TYPE, type_attr));
        }

        if iri != *RDF_DESCRIPTION {
            results.push(Triple::new(subject.clone(), rdf::TYPE, iri));
        }
        Ok(RdfXmlState::NodeElt {
            base_iri,
            language,
            subject,
            li_counter: 0,
        })
    }

    /// Builds the `parseTypeResourcePropertyElt` production.
    fn build_parse_type_resource_property_elt(
        &mut self,
        iri: NamedNode,
        base_iri: Option<Iri<OxString>>,
        language: Option<OxString>,
        subject: NamedOrBlankNode,
        id_attr: Option<NamedNode>,
        results: &mut Vec<Triple>,
    ) -> RdfXmlState {
        let object = BlankNode::default();
        let triple = Triple::new(subject, iri, object.clone());
        Self::reify(&triple, id_attr, results);
        results.push(triple);
        RdfXmlState::NodeElt {
            base_iri,
            language,
            subject: object.into(),
            li_counter: 0,
        }
    }

    /// Emits the triples a production yields once its closing tag is seen.
    fn end_state(
        &mut self,
        state: RdfXmlState,
        results: &mut Vec<Triple>,
    ) -> Result<(), RdfXmlSyntaxError> {
        match state {
            RdfXmlState::PropertyElt {
                iri,
                language,
                subject,
                id_attr,
                datatype_attr,
                object,
                ..
            } => {
                let object = match object {
                    Some(NodeOrText::Node(node)) => Term::from(node),
                    Some(NodeOrText::Text(text)) => {
                        self.new_literal(text, language, datatype_attr).into()
                    }
                    None => self
                        .new_literal(OxString::new(""), language, datatype_attr)
                        .into(),
                };
                let triple = Triple::new(subject, iri, object);
                Self::reify(&triple, id_attr, results);
                results.push(triple);
            }
            RdfXmlState::ParseTypeCollectionPropertyElt {
                iri,
                subject,
                id_attr,
                objects,
                ..
            } => {
                let mut current_node = NamedOrBlankNode::from(rdf::NIL);
                for object in objects.into_iter().rev() {
                    let subject = NamedOrBlankNode::from(BlankNode::default());
                    results.push(Triple::new(subject.clone(), rdf::FIRST, object));
                    results.push(Triple::new(subject.clone(), rdf::REST, current_node));
                    current_node = subject;
                }
                let triple = Triple::new(subject, iri, current_node);
                Self::reify(&triple, id_attr, results);
                results.push(triple);
            }
            RdfXmlState::ParseTypeLiteralPropertyElt {
                iri,
                subject,
                id_attr,
                writer,
                ..
            } => {
                // An empty XML literal is legal: the grammar allows a `parseTypeLiteralPropertyElt`
                // with no content at all, and the W3C suite's `rdfms-empty-property-elements`
                // test003 expects `""^^rdf:XMLLiteral` for `<p rdf:parseType="Literal"/>`.
                let object = writer.into_inner();
                let triple = Triple::new(
                    subject,
                    iri,
                    Literal::new_typed_literal(
                        OxString::new_owned(&String::from_utf8(object).map_err(|_| {
                            RdfXmlSyntaxError::msg("The XML literal is not in valid UTF-8")
                        })?),
                        rdf::XML_LITERAL,
                    ),
                );
                Self::reify(&triple, id_attr, results);
                results.push(triple);
            }
            RdfXmlState::NodeElt { subject, .. } => match self.state.last_mut() {
                Some(RdfXmlState::PropertyElt { object, .. }) => {
                    if is_object_defined(object) {
                        return Err(RdfXmlSyntaxError::msg(
                            "Unexpected node, a text value is already present",
                        ));
                    }
                    *object = Some(NodeOrText::Node(subject));
                }
                Some(RdfXmlState::ParseTypeCollectionPropertyElt { objects, .. }) => {
                    objects.push(subject);
                }
                _ => (),
            },
            RdfXmlState::Doc { .. } | RdfXmlState::Rdf { .. } => (),
        }
        Ok(())
    }

    /// Builds a literal, applying the in-scope language when no datatype is given.
    fn new_literal(
        &self,
        value: OxString,
        language: Option<OxString>,
        datatype: Option<NamedNode>,
    ) -> Literal {
        if let Some(datatype) = datatype {
            Literal::new_typed_literal(value, datatype)
        } else if let Some(language) = self.effective_language(language.as_ref()) {
            Literal::new_language_tagged_literal_unchecked(value, language)
        } else {
            Literal::new_simple_literal(value)
        }
    }

    /// Resolves the language that applies to a literal: the element's own `xml:lang` when it
    /// declares one, otherwise the innermost enclosing one. An `xml:lang=""` declaration clears
    /// the language instead of introducing an (invalid) empty tag.
    fn effective_language(&self, declared: Option<&OxString>) -> Option<OxString> {
        match declared {
            Some(language) if language.is_empty() => None,
            Some(language) => Some(language.clone()),
            None => self.current_language().cloned(),
        }
    }

    /// Emits the four reification triples requested by an `rdf:ID` on a property element.
    fn reify(triple: &Triple, statement_id: Option<NamedNode>, results: &mut Vec<Triple>) {
        let Some(statement_id) = statement_id else {
            return;
        };
        results.push(Triple::new(statement_id.clone(), rdf::TYPE, rdf::STATEMENT));
        results.push(Triple::new(
            statement_id.clone(),
            rdf::SUBJECT,
            triple.subject.clone(),
        ));
        results.push(Triple::new(
            statement_id.clone(),
            rdf::PREDICATE,
            triple.predicate.as_ref(),
        ));
        results.push(Triple::new(
            statement_id,
            rdf::OBJECT,
            triple.object.clone(),
        ));
    }

    /// Emits the triples produced by property attributes of a node or property element.
    fn emit_property_attrs(
        &self,
        subject: &NamedOrBlankNode,
        literal_attributes: Vec<(NamedNode, OxString)>,
        language: Option<&OxString>,
        results: &mut Vec<Triple>,
    ) {
        let language = self.effective_language(language);
        for (literal_predicate, literal_value) in literal_attributes {
            let object = if let Some(language) = &language {
                Literal::new_language_tagged_literal_unchecked(literal_value, language.clone())
            } else {
                Literal::new_simple_literal(literal_value)
            };
            results.push(Triple::new(subject.clone(), literal_predicate, object));
        }
    }

    /// Decodes and normalizes an attribute value, expanding custom entities.
    fn convert_attribute<'a>(
        &self,
        attribute: &Attribute<'a>,
    ) -> Result<Cow<'a, str>, RdfXmlParseError> {
        Ok(attribute.decoded_and_normalized_value_with(
            self.xml_version,
            self.reader.decoder(),
            MAX_ENTITY_NESTING,
            |e| self.custom_entities.resolve(e),
        )?)
    }

    /// Decodes an attribute value and resolves it against the in-scope base IRI.
    fn convert_iri_attribute(
        &self,
        base_iri: Option<&Iri<OxString>>,
        attribute: &Attribute<'_>,
    ) -> Result<NamedNode, RdfXmlParseError> {
        Ok(self.resolve_iri(base_iri, &self.convert_attribute(attribute)?)?)
    }

    /// Resolves a possibly relative IRI reference against the in-scope base IRI.
    fn resolve_iri(
        &self,
        base_iri: Option<&Iri<OxString>>,
        relative_iri: &str,
    ) -> Result<NamedNode, RdfXmlSyntaxError> {
        if self.lenient {
            let Some(base_iri) = base_iri.or_else(|| self.current_base_iri()) else {
                return Ok(NamedNode::new_unchecked(OxString::new_owned(relative_iri)));
            };
            let relative_iri = IriRef::parse_unchecked(relative_iri);
            let mut buffer = String::new();
            Ok(NamedNode::new_unchecked(OxString::new_owned(
                if relative_iri.is_absolute() {
                    relative_iri.into_inner()
                } else {
                    base_iri.resolve_into_unchecked(&relative_iri, &mut buffer);
                    &buffer
                },
            )))
        } else {
            let relative_iri = IriRef::parse(relative_iri).map_err(|error| {
                RdfXmlSyntaxError::invalid_iri(OxString::new_owned(relative_iri), error)
            })?;
            let mut buffer = String::new();
            Ok(NamedNode::new_unchecked(OxString::new_owned(
                if relative_iri.is_absolute() {
                    relative_iri.into_inner()
                } else if let Some(base_iri) = base_iri.or_else(|| self.current_base_iri()) {
                    base_iri
                        .resolve_into(&relative_iri, &mut buffer)
                        .map_err(|error| {
                            RdfXmlSyntaxError::invalid_iri(
                                OxString::new_owned(relative_iri.into_inner()),
                                error,
                            )
                        })?;
                    &buffer
                } else {
                    return Err(RdfXmlSyntaxError::msg(format!(
                        "{relative_iri} is a relative IRI but no xml:base is set"
                    )));
                },
            )))
        }
    }

    /// Validates an already-absolute IRI (element and attribute names).
    fn parse_iri(&self, relative_iri: OxString) -> Result<NamedNode, RdfXmlSyntaxError> {
        Ok(NamedNode::new_unchecked(if self.lenient {
            relative_iri
        } else {
            Iri::parse(relative_iri.clone())
                .map_err(|error| RdfXmlSyntaxError::invalid_iri(relative_iri, error))?
                .into_inner()
        }))
    }

    /// The innermost `xml:lang` currently in scope, `None` when the innermost declaration is the
    /// language-clearing `xml:lang=""`.
    fn current_language(&self) -> Option<&OxString> {
        for state in self.state.iter().rev() {
            match state {
                RdfXmlState::Doc { .. } => (),
                RdfXmlState::Rdf { language, .. }
                | RdfXmlState::NodeElt { language, .. }
                | RdfXmlState::PropertyElt { language, .. }
                | RdfXmlState::ParseTypeCollectionPropertyElt { language, .. }
                | RdfXmlState::ParseTypeLiteralPropertyElt { language, .. } => {
                    if let Some(language) = language {
                        return if language.is_empty() {
                            None
                        } else {
                            Some(language)
                        };
                    }
                }
            }
        }
        None
    }

    /// The innermost `xml:base` currently in scope.
    pub(crate) fn current_base_iri(&self) -> Option<&Iri<OxString>> {
        for state in self.state.iter().rev() {
            match state {
                RdfXmlState::Doc { base_iri }
                | RdfXmlState::Rdf { base_iri, .. }
                | RdfXmlState::NodeElt { base_iri, .. }
                | RdfXmlState::PropertyElt { base_iri, .. }
                | RdfXmlState::ParseTypeCollectionPropertyElt { base_iri, .. }
                | RdfXmlState::ParseTypeLiteralPropertyElt { base_iri, .. } => {
                    if let Some(base_iri) = base_iri {
                        return Some(base_iri);
                    }
                }
            }
        }
        None
    }

    /// Expands a character or general entity reference into the text buffer.
    fn decode_xml_entity(&mut self, event: &BytesRef<'_>) -> Result<(), Error> {
        if let Some(char_ref) = event.resolve_char_ref()? {
            self.text_buffer.push(char_ref);
            return Ok(());
        }
        let reference = event.xml_content(self.xml_version)?;
        let Some(value) = self.custom_entities.resolve(&reference) else {
            return Err(EscapeError::UnrecognizedEntity(0..event.len(), reference.into()).into());
        };
        self.text_buffer.push_str(value);
        Ok(())
    }
}
