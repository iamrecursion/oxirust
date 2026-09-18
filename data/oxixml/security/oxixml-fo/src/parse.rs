//! Building an [`FoTree`] from an XSL-FO source document.

use oxixml_xml::events::attributes::Attribute;
use oxixml_xml::events::{BytesStart, Event};
use oxixml_xml::name::ResolveResult;
use oxixml_xml::reader::NsReader;

use crate::error::{FoError, Result, SourcePosition};
use crate::fo::{Category, ContentModel, FoElement};
use crate::property::{PropertyKey, RefineOptions, refine};
use crate::tree::{FoNode, FoTree, ForeignElement, NodeId, NodeKind, ResolvedProperties};

/// The XSL-FO namespace, `http://www.w3.org/1999/XSL/Format`.
pub const FO_NAMESPACE: &str = "http://www.w3.org/1999/XSL/Format";

/// The XML namespace, `http://www.w3.org/XML/1998/namespace`, in which
/// `xml:lang` — a shorthand property in XSL — is written.
pub const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";

/// How an XSL-FO document is turned into a tree.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseOptions {
    /// Options for the refinement pass.
    pub refine: RefineOptions,
    /// Enforce the `Contents:` clause of each formatting object (XSL 1.1 §6).
    pub validate_content_model: bool,
    /// Reject attributes in no namespace that are not XSL-FO properties.
    ///
    /// With this off, unknown attributes are kept as
    /// [foreign attributes](crate::tree::FoNode::foreign_attributes) instead.
    pub reject_unknown_properties: bool,
    /// Drop text nodes that are entirely whitespace from content models that do
    /// not allow `#PCDATA`.
    ///
    /// Turning this off makes such whitespace an error, which is useful for
    /// checking that a stylesheet emits exactly the intended tree.
    pub ignore_structural_whitespace: bool,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            refine: RefineOptions::default(),
            validate_content_model: true,
            reject_unknown_properties: true,
            ignore_structural_whitespace: true,
        }
    }
}

/// Entry point for reading XSL-FO documents.
///
/// ```
/// use oxixml_fo::{FoDocument, fo::FoElement};
///
/// let tree = FoDocument::parse(
///     r#"<fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format">
///          <fo:layout-master-set>
///            <fo:simple-page-master master-name="page">
///              <fo:region-body/>
///            </fo:simple-page-master>
///          </fo:layout-master-set>
///          <fo:page-sequence master-reference="page">
///            <fo:flow flow-name="xsl-region-body">
///              <fo:block>Hello</fo:block>
///            </fo:flow>
///          </fo:page-sequence>
///        </fo:root>"#,
/// )
/// .expect("a well-formed XSL-FO document");
///
/// assert_eq!(tree.element(tree.root()), Some(FoElement::Root));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FoDocument;

impl FoDocument {
    /// Parses, validates and refines an XSL-FO document.
    ///
    /// # Errors
    ///
    /// Returns an [`FoError`] when the source is not well-formed XML, when its
    /// document element is not `fo:root`, when a content model or a property
    /// applicability rule is violated, or when a property value cannot be
    /// evaluated.
    pub fn parse(xml: &str) -> Result<FoTree> {
        Self::parse_with(xml, &ParseOptions::default())
    }

    /// Parses, validates and refines an XSL-FO document with explicit options.
    ///
    /// # Errors
    ///
    /// As [`FoDocument::parse`].
    pub fn parse_with(xml: &str, options: &ParseOptions) -> Result<FoTree> {
        let mut builder = Builder::new(options);
        builder.run(xml)?;
        let mut tree = builder.finish()?;
        refine(&mut tree, &options.refine)?;
        Ok(tree)
    }
}

/// A frame of the element stack while the tree is being built.
struct Frame {
    id: NodeId,
    element: Option<FoElement>,
    children: Vec<FoElement>,
    saw_text: bool,
}

struct Builder<'a> {
    options: &'a ParseOptions,
    nodes: Vec<FoNode>,
    stack: Vec<Frame>,
    root: Option<NodeId>,
    element_order: Vec<NodeId>,
}

impl<'a> Builder<'a> {
    fn new(options: &'a ParseOptions) -> Self {
        Self {
            options,
            nodes: Vec::new(),
            stack: Vec::new(),
            root: None,
            element_order: Vec::new(),
        }
    }

    fn push_node(&mut self, kind: NodeKind) -> NodeId {
        let id = NodeId::from_index(self.nodes.len());
        let parent = self.stack.last().map(|frame| frame.id);
        self.nodes.push(FoNode {
            kind,
            parent,
            children: Vec::new(),
            specified: Vec::new(),
            foreign_attributes: Vec::new(),
            properties: ResolvedProperties::default(),
        });
        if let Some(parent) = parent {
            self.nodes[parent.index()].children.push(id);
        }
        id
    }

    fn current_model(&self) -> Option<&'static ContentModel> {
        self.stack
            .last()
            .and_then(|frame| frame.element)
            .map(FoElement::content_model)
    }

    fn run(&mut self, xml: &str) -> Result<()> {
        let mut reader = NsReader::from_str(xml);
        reader.config_mut().check_well_formedness = true;
        loop {
            let position = reader.buffer_position() as usize;
            let (namespace, event) = reader
                .read_resolved_event()
                .map_err(|error| FoError::xml(&error, position))?;
            let namespace = match namespace {
                ResolveResult::Bound(namespace) => Some(namespace.as_ref().to_vec()),
                _ => None,
            };
            match event {
                Event::Start(start) => {
                    self.start_element(&reader, namespace.as_deref(), &start, position, false)?;
                }
                Event::Empty(start) => {
                    self.start_element(&reader, namespace.as_deref(), &start, position, true)?;
                }
                Event::End(_) => self.end_element(position)?,
                Event::Text(text) => {
                    let decoded = text
                        .xml_content(oxixml_xml::XmlVersion::Explicit1_0)
                        .map_err(|error| FoError::Xml {
                            message: error.to_string(),
                            position: SourcePosition::new(position),
                        })?;
                    self.text(&decoded, position)?;
                }
                Event::CData(cdata) => {
                    let decoded = cdata.decode().map_err(|error| FoError::Xml {
                        message: error.to_string(),
                        position: SourcePosition::new(position),
                    })?;
                    self.text(&decoded, position)?;
                }
                Event::Eof => break,
                // Comments, processing instructions, the XML declaration, the
                // doctype and general entity references carry no formatting
                // semantics.
                _ => {}
            }
        }
        Ok(())
    }

    fn start_element<R>(
        &mut self,
        reader: &NsReader<R>,
        namespace: Option<&[u8]>,
        start: &BytesStart<'_>,
        position: usize,
        empty: bool,
    ) -> Result<()> {
        let local_name = String::from_utf8_lossy(start.local_name().as_ref()).into_owned();
        let is_fo = namespace == Some(FO_NAMESPACE.as_bytes());

        if is_fo {
            let element =
                FoElement::parse(&local_name).ok_or_else(|| FoError::UnknownFormattingObject {
                    name: local_name.clone(),
                    position: SourcePosition::new(position),
                })?;
            self.check_child(element, position)?;
            let id = self.push_node(NodeKind::Fo(element));
            self.element_order.push(id);
            if self.root.is_none() {
                if element != FoElement::Root {
                    return Err(FoError::NotAnFoDocument {
                        found: format!("fo:{local_name}"),
                    });
                }
                self.root = Some(id);
            }
            self.read_properties(reader, element, id, start, position)?;
            if let Some(frame) = self.stack.last_mut() {
                frame.children.push(element);
            }
            self.stack.push(Frame {
                id,
                element: Some(element),
                children: Vec::new(),
                saw_text: false,
            });
            if empty {
                self.end_element(position)?;
            }
            return Ok(());
        }

        if self.root.is_none() {
            return Err(FoError::NotAnFoDocument { found: local_name });
        }

        // A foreign element: allowed where the content model permits one, and
        // always allowed below another foreign element.
        let inside_foreign = self
            .stack
            .last()
            .is_some_and(|frame| frame.element.is_none());
        if !inside_foreign
            && self.options.validate_content_model
            && !self.current_model().is_some_and(|model| model.foreign)
        {
            if let Some(parent) = self.stack.last().and_then(|frame| frame.element) {
                return Err(FoError::ContentModel {
                    parent,
                    expected: format!(
                        "{} does not allow elements outside the XSL-FO namespace",
                        parent.content_model().describe()
                    ),
                    position: SourcePosition::new(position),
                });
            }
        }
        let foreign = self.read_foreign(reader, &local_name, namespace, start)?;
        let id = self.push_node(NodeKind::Foreign(foreign));
        self.stack.push(Frame {
            id,
            element: None,
            children: Vec::new(),
            saw_text: false,
        });
        if empty {
            self.end_element(position)?;
        }
        Ok(())
    }

    fn read_foreign<R>(
        &self,
        reader: &NsReader<R>,
        local_name: &str,
        namespace: Option<&[u8]>,
        start: &BytesStart<'_>,
    ) -> Result<ForeignElement> {
        let mut attributes = Vec::new();
        for attribute in start.attributes() {
            let attribute = attribute.map_err(|error| FoError::Xml {
                message: error.to_string(),
                position: SourcePosition::default(),
            })?;
            if is_namespace_declaration(&attribute) {
                continue;
            }
            let (resolved, local) = reader.resolve_attribute(attribute.key);
            let namespace = match resolved {
                ResolveResult::Bound(namespace) => {
                    Some(String::from_utf8_lossy(namespace.as_ref()).into_owned())
                }
                _ => None,
            };
            attributes.push((
                namespace,
                String::from_utf8_lossy(local.as_ref()).into_owned(),
                decode_attribute(&attribute)?,
            ));
        }
        Ok(ForeignElement {
            namespace: namespace.map(|bytes| String::from_utf8_lossy(bytes).into_owned()),
            local_name: local_name.to_owned(),
            attributes,
        })
    }

    fn read_properties<R>(
        &mut self,
        reader: &NsReader<R>,
        element: FoElement,
        id: NodeId,
        start: &BytesStart<'_>,
        position: usize,
    ) -> Result<()> {
        for attribute in start.attributes() {
            let attribute = attribute.map_err(|error| FoError::Xml {
                message: error.to_string(),
                position: SourcePosition::new(position),
            })?;
            if is_namespace_declaration(&attribute) {
                continue;
            }
            let (resolved, local) = reader.resolve_attribute(attribute.key);
            let local_name = String::from_utf8_lossy(local.as_ref()).into_owned();
            let value = decode_attribute(&attribute)?;
            let key = match resolved {
                ResolveResult::Unbound => PropertyKey::parse(&local_name),
                ResolveResult::Bound(namespace)
                    if namespace.as_ref() == XML_NAMESPACE.as_bytes() =>
                {
                    // `xml:lang` is the only XSL property in a namespace.
                    PropertyKey::parse(&format!("xml:{local_name}"))
                }
                _ => None,
            };
            match key {
                Some(key) => {
                    if self.options.refine.validate_applicability
                        && !element.accepts_property(key.id)
                    {
                        return Err(FoError::InapplicableProperty {
                            fo: element,
                            property: key.to_string(),
                            position: SourcePosition::new(position),
                        });
                    }
                    self.nodes[id.index()].specified.push((key, value));
                }
                None => {
                    let namespace = match resolved {
                        ResolveResult::Bound(namespace) => {
                            Some(String::from_utf8_lossy(namespace.as_ref()).into_owned())
                        }
                        _ => None,
                    };
                    if namespace.is_none() && self.options.reject_unknown_properties {
                        return Err(FoError::UnknownProperty {
                            property: local_name,
                            position: SourcePosition::new(position),
                        });
                    }
                    self.nodes[id.index()]
                        .foreign_attributes
                        .push((namespace, local_name, value));
                }
            }
        }
        Ok(())
    }

    fn check_child(&self, child: FoElement, position: usize) -> Result<()> {
        let Some(frame) = self.stack.last() else {
            return Ok(());
        };
        let Some(parent) = frame.element else {
            // Inside foreign content anything goes.
            return Ok(());
        };
        if !self.options.validate_content_model {
            return Ok(());
        }
        if parent.content_model().accepts(child) {
            return Ok(());
        }
        Err(FoError::InvalidChild {
            parent,
            child,
            position: SourcePosition::new(position),
        })
    }

    fn text(&mut self, text: &str, position: usize) -> Result<()> {
        let Some(frame) = self.stack.last() else {
            // Whitespace around the document element.
            return Ok(());
        };
        let parent = frame.element;
        let allows_text = parent.is_none_or(|element| element.content_model().text);
        let blank = text.trim().is_empty();
        if !allows_text {
            if blank && self.options.ignore_structural_whitespace {
                return Ok(());
            }
            if let Some(parent) = parent {
                return Err(FoError::InvalidText {
                    parent,
                    position: SourcePosition::new(position),
                });
            }
        }
        if let Some(frame) = self.stack.last_mut() {
            frame.saw_text |= !blank;
        }
        self.push_node(NodeKind::Text(text.to_owned()));
        Ok(())
    }

    fn end_element(&mut self, position: usize) -> Result<()> {
        let Some(frame) = self.stack.pop() else {
            return Ok(());
        };
        if let Some(element) = frame.element
            && self.options.validate_content_model
        {
            validate_children(element, &frame.children, frame.saw_text, position)?;
        }
        Ok(())
    }

    fn finish(self) -> Result<FoTree> {
        let root = self.root.ok_or_else(|| FoError::NotAnFoDocument {
            found: "(empty document)".to_owned(),
        })?;
        Ok(FoTree {
            nodes: self.nodes,
            root,
            element_order: self.element_order,
        })
    }
}

fn is_namespace_declaration(attribute: &Attribute<'_>) -> bool {
    let key = attribute.key.as_ref();
    key == b"xmlns" || key.starts_with(b"xmlns:")
}

fn decode_attribute(attribute: &Attribute<'_>) -> Result<String> {
    attribute
        .normalized_value(oxixml_xml::XmlVersion::Explicit1_0)
        .map(std::borrow::Cow::into_owned)
        .map_err(|error| FoError::Xml {
            message: error.to_string(),
            position: SourcePosition::default(),
        })
}

/// Checks the children of a finished element against its content model.
fn validate_children(
    parent: FoElement,
    children: &[FoElement],
    saw_text: bool,
    position: usize,
) -> Result<()> {
    let model = parent.content_model();
    let expected = || FoError::ContentModel {
        parent,
        expected: model.describe(),
        position: SourcePosition::new(position),
    };

    // Leading `fo:marker` children, optionally followed by an
    // `fo:initial-property-set`, are allowed by prose rather than by the
    // content model itself.
    let mut index = 0;
    while index < children.len() && children[index] == FoElement::Marker && model.markers {
        index += 1;
    }
    if model.initial_property_set && children.get(index) == Some(&FoElement::InitialPropertySet) {
        index += 1;
    }
    let rest = &children[index..];
    if rest.contains(&FoElement::Marker) && !model.markers {
        return Err(expected());
    }

    if model.particles.is_empty() {
        if model.block_required && !rest.iter().any(|child| child.category() == Category::Block) {
            return Err(expected());
        }
        return Ok(());
    }

    // Non-structural children (neutral containers, out-of-line objects and
    // change-bar points) do not take part in the particle sequence.
    let structural: Vec<FoElement> = rest
        .iter()
        .copied()
        .filter(|child| {
            model
                .particles
                .iter()
                .any(|particle| particle.accepts(*child))
        })
        .collect();
    if structural.len() != rest.len() {
        return Err(expected());
    }
    if saw_text && !model.text {
        return Err(expected());
    }

    if model.choice {
        let Some(first) = structural.first() else {
            return if model.particles.iter().all(|particle| particle.min == 0) {
                Ok(())
            } else {
                Err(expected())
            };
        };
        let particle = model
            .particles
            .iter()
            .find(|particle| particle.accepts(*first))
            .ok_or_else(expected)?;
        if !structural.iter().all(|child| particle.accepts(*child)) {
            return Err(expected());
        }
        let count = structural.len() as u32;
        if count < particle.min || count > particle.max {
            return Err(expected());
        }
        return Ok(());
    }

    let mut cursor = 0usize;
    for particle in model.particles {
        let mut count = 0u32;
        while cursor < structural.len()
            && particle.accepts(structural[cursor])
            && count < particle.max
        {
            cursor += 1;
            count += 1;
        }
        if count < particle.min {
            return Err(expected());
        }
    }
    if cursor != structural.len() {
        return Err(expected());
    }
    Ok(())
}
