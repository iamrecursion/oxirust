//! Node construction.
//!
//! [XQuery 3.1 §3.9](https://www.w3.org/TR/xquery-31/#id-constructors) says every constructor
//! builds a node with a *new identity*, and that everything placed in its content is copied. This
//! module implements the content-sequence rules — adjacent atomic values become one
//! space-separated text node, attributes must precede other content, a document node is replaced
//! by its children, adjacent text nodes merge and empty ones vanish — together with namespace
//! resolution, namespace fixup and the boundary-space policy.

use super::program::{CompiledConstructor, CompiledContent, CompiledElement, CompiledName};
use super::{Frame, evaluate};
use crate::context::{BoundarySpace, ConstructionMode, CopyNamespaces};
use crate::error::{XQueryError, XQueryErrorCode, XQueryResult};
use crate::nodes::{CopyOptions, Draft, copy_node, fixup, fixup_attribute, push_binding};
use oxixml_infoset::{ExpandedName, NodeKind, XdmNode};
use oxixml_qname::{XML_NAMESPACE, is_ncname};
use oxixml_xpath::{XpathItem, XpathValue};

const XMLNS_NAMESPACE: &str = "http://www.w3.org/2000/xmlns/";

/// Evaluates one node constructor.
pub(crate) fn evaluate_constructor(
    constructor: &CompiledConstructor,
    frame: &Frame,
) -> XQueryResult<XpathValue> {
    match constructor {
        CompiledConstructor::DirectElement(element) => direct_element(element, frame),
        CompiledConstructor::Element { name, content } => {
            let name = element_name(name, frame)?;
            check_constructed_name(&name, false)?;
            let items = match content {
                Some(program) => evaluate(program, frame)?.into_items(),
                None => Vec::new(),
            };
            Ok(XpathValue::singleton(
                build_element(name, Vec::new(), items, frame)?.build(),
            ))
        }
        CompiledConstructor::Attribute { name, content } => {
            let mut name = attribute_name(name, frame)?;
            check_constructed_name(&name, true)?;
            name.invent_attribute_prefix();
            let value = match content {
                Some(program) => atomized_string(&evaluate(program, frame)?)?,
                None => String::new(),
            };
            check_special_attribute(&name.name, &value)?;
            Ok(XpathValue::singleton(
                Draft::attribute(name.spelt(), value).build(),
            ))
        }
        CompiledConstructor::Document(program) => {
            let items = evaluate(program, frame)?.into_items();
            let mut document = Draft::document();
            document.base_uri = frame.construction_base();
            let mut content = Content::new(frame);
            content.document = true;
            content.push_value(items)?;
            let (attributes, namespaces, children) = content.finish()?;
            if !attributes.is_empty() || !namespaces.is_empty() {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xpty0004,
                    "a document node may not have attribute or namespace children",
                ));
            }
            document.children = children;
            Ok(XpathValue::singleton(document.build()))
        }
        CompiledConstructor::Text(content) => {
            // §3.9.3.5: only an *empty sequence* content expression yields no text node. A
            // content expression whose value happens to be the zero-length string yields a text
            // node whose content is the zero-length string, and `count(text {''})` is 1.
            let Some(program) = content else {
                return Ok(XpathValue::default());
            };
            let value = evaluate(program, frame)?;
            if value.is_empty() {
                return Ok(XpathValue::default());
            }
            Ok(XpathValue::singleton(
                Draft::text(atomized_string(&value)?).build(),
            ))
        }
        CompiledConstructor::Comment(content) => {
            let text = match content {
                Some(program) => atomized_string(&evaluate(program, frame)?)?,
                None => String::new(),
            };
            if text.contains("--") || text.ends_with('-') {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0072,
                    "a comment may not contain `--` or end with `-`",
                ));
            }
            Ok(XpathValue::singleton(Draft::comment(text).build()))
        }
        CompiledConstructor::ProcessingInstruction {
            name,
            content,
            literal,
        } => {
            let target = processing_instruction_target(name, frame)?;
            let text = match content {
                Some(program) if *literal => match program {
                    super::program::Program::Literal(value) => value.string_value(),
                    other => atomized_string(&evaluate(other, frame)?)?,
                },
                // XQuery 3.1 §3.9.3.5 removes the *leading white space* of a computed
                // processing instruction's content, and white space there is XML's `S` — a
                // content that begins with a no-break space begins with content.
                Some(program) => atomized_string(&evaluate(program, frame)?)?
                    .trim_start_matches(crate::lexer::XML_SPACE)
                    .to_owned(),
                None => String::new(),
            };
            if text.contains("?>") {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0026,
                    "the content of a processing instruction may not contain `?>`",
                ));
            }
            Ok(XpathValue::singleton(
                Draft::processing_instruction(&target, text).build(),
            ))
        }
        CompiledConstructor::Namespace { prefix, uri } => {
            let prefix = match prefix {
                CompiledName::Lexical(text) => text.clone(),
                CompiledName::Computed(program) => {
                    let value = evaluate(program, frame)?;
                    let atomized = value.atomize()?;
                    match atomized.as_slice() {
                        [] => String::new(),
                        // §3.9.3.7: the prefix expression must yield an `xs:string`,
                        // `xs:untypedAtomic` or `xs:NCName`; anything else is `XPTY0004`.
                        [single]
                            if matches!(
                                single.atomic_type(),
                                oxixml_infoset::AtomicType::String
                                    | oxixml_infoset::AtomicType::UntypedAtomic
                            ) || single
                                .atomic_type()
                                .is_derived_from(oxixml_infoset::AtomicType::String) =>
                        {
                            single.string_value().trim().to_owned()
                        }
                        _ => {
                            return Err(XQueryError::new(
                                XQueryErrorCode::Xpty0004,
                                "a computed namespace prefix must be one xs:string, \
                                 xs:untypedAtomic or xs:NCName",
                            ));
                        }
                    }
                }
            };
            let namespace = match uri {
                Some(program) => atomized_string(&evaluate(program, frame)?)?,
                None => String::new(),
            };
            if !prefix.is_empty() && !is_ncname(&prefix) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0074,
                    format!("`{prefix}` is not a valid namespace prefix"),
                ));
            }
            check_reserved_binding(
                (!prefix.is_empty()).then_some(prefix.as_str()),
                &namespace,
                XQueryErrorCode::Xqdy0101,
            )?;
            // §3.9.3.7: only the default namespace may be bound to a zero-length URI.
            if !prefix.is_empty() && namespace.is_empty() {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0101,
                    format!("`{prefix}` may not be bound to a zero-length namespace URI"),
                ));
            }
            let name = (!prefix.is_empty()).then_some(prefix.as_str());
            Ok(XpathValue::singleton(
                Draft::namespace(name, namespace).build(),
            ))
        }
    }
}

fn direct_element(element: &CompiledElement, frame: &Frame) -> XQueryResult<XpathValue> {
    // The namespace declaration attributes come into scope before anything is resolved.
    let mut declared: Vec<(Option<String>, String)> = Vec::new();
    for (prefix, uri) in &element.namespaces {
        if declared.iter().any(|(bound, _)| bound == prefix) {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0071,
                "a direct element constructor declares the same prefix twice",
            ));
        }
        check_reserved_binding(prefix.as_deref(), uri, XQueryErrorCode::Xqst0070)?;
        if uri.is_empty() && prefix.is_some() {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0085,
                "only the default namespace may be undeclared",
            ));
        }
        declared.push((prefix.clone(), uri.clone()));
    }
    let mut scope = declared.clone();
    for binding in frame.namespaces.iter() {
        push_binding(&mut scope, binding.0.clone(), binding.1.clone());
    }
    let inner = frame.with_namespaces(scope);

    let mut name = resolve_lexical(
        &element.name,
        &inner,
        false,
        XQueryErrorCode::Xpst0081,
        true,
    )?;
    let mut attributes = Vec::with_capacity(element.attributes.len());
    let mut attribute_names = Vec::with_capacity(element.attributes.len());
    for attribute in &element.attributes {
        let attribute_name = resolve_lexical(
            &attribute.name,
            &inner,
            true,
            XQueryErrorCode::Xpst0081,
            true,
        )?;
        if attributes
            .iter()
            .any(|existing: &Draft| existing.name.as_ref() == Some(&attribute_name.name))
        {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqst0040,
                format!(
                    "the element has two attributes named `{}`",
                    attribute_name.name
                ),
            ));
        }
        let mut value = String::new();
        for part in &attribute.value {
            match part {
                CompiledContent::Text(text) | CompiledContent::BoundaryWhitespace(text) => {
                    value.push_str(text);
                }
                // An attribute value has no nested constructors, so `Nested` cannot occur here.
                CompiledContent::Enclosed(program) | CompiledContent::Nested(program) => {
                    value.push_str(&atomized_string(&evaluate(program, &inner)?)?);
                }
            }
        }
        check_special_attribute(&attribute_name.name, &value)?;
        let value = normalized_attribute_value(&attribute_name.name, value);
        attributes.push(Draft::attribute(attribute_name.name.clone(), value));
        attribute_names.push(attribute_name);
    }

    // The element's own in-scope namespaces are what a copied node inherits, so they have to be
    // known before the content is evaluated.
    let mut scope_bindings = declared.clone();
    for binding in frame.namespaces.iter() {
        push_binding(&mut scope_bindings, binding.0.clone(), binding.1.clone());
    }
    fixup(&mut scope_bindings, &mut name);
    for attribute_name in &mut attribute_names {
        fixup_attribute(&mut scope_bindings, attribute_name);
    }
    // A child's base URI is this element's, so it has to be settled before the content runs.
    let own_base = element_base_uri(&attributes, frame);
    let content_frame = inner
        .with_inherited(scope_bindings)
        .with_parent_base(own_base);

    let mut content = Content::new(&content_frame);
    for part in &element.content {
        match part {
            CompiledContent::Text(text) => content.push_literal(text),
            CompiledContent::BoundaryWhitespace(text) => {
                if inner.options.boundary_space == BoundarySpace::Preserve {
                    content.push_literal(text);
                }
            }
            CompiledContent::Enclosed(program) => {
                content.push_value(evaluate(program, &content_frame)?.into_items())?;
            }
            CompiledContent::Nested(program) => {
                content.push_constructed(evaluate(program, &content_frame)?.into_items())?;
            }
        }
    }
    let (extra_attributes, extra_namespaces, children) = content.finish()?;

    for attribute in extra_attributes {
        let Some(attribute_name) = attribute.name.clone() else {
            continue;
        };
        if attributes
            .iter()
            .any(|existing| existing.name.as_ref() == Some(&attribute_name))
        {
            return Err(XQueryError::new(
                XQueryErrorCode::Xqdy0025,
                format!("the element has two attributes named `{attribute_name}`"),
            ));
        }
        attribute_names.push(ConstructedName::retained(attribute_name));
        attributes.push(attribute);
    }
    let mut bindings = declared;
    for binding in extra_namespaces {
        push_binding(&mut bindings, binding.0, binding.1);
    }
    for binding in frame.namespaces.iter() {
        push_binding(&mut bindings, binding.0.clone(), binding.1.clone());
    }
    finish_bindings(&mut bindings, &mut name, &mut attribute_names);
    let mut draft = Draft::element(name.spelt());
    rename_attributes(&mut attributes, &attribute_names);
    draft.namespaces = bindings;
    draft.attributes = attributes;
    draft.children = children;
    annotate(&mut draft, frame);
    Ok(XpathValue::singleton(draft.build()))
}

/// Applies the last two rules of §3.9.4: a binding for every prefix the element's own names use,
/// and a binding for `xml`.
fn finish_bindings(
    bindings: &mut Vec<(Option<String>, String)>,
    name: &mut ConstructedName,
    attributes: &mut [ConstructedName],
) {
    fixup(bindings, name);
    for attribute in attributes {
        fixup_attribute(bindings, attribute);
    }
    push_binding(bindings, Some("xml".to_owned()), XML_NAMESPACE.to_owned());
    bindings.retain(|(prefix, uri)| prefix.is_some() || !uri.is_empty());
}

/// Writes the prefixes namespace fixup settled on back onto the attribute drafts, which were
/// built before the bindings were known. The two vectors are built in lock-step.
fn rename_attributes(attributes: &mut [Draft], names: &[ConstructedName]) {
    for (attribute, name) in attributes.iter_mut().zip(names) {
        attribute.name = Some(name.spelt());
    }
}

/// Gives a constructed element the base URI and type annotation the specification prescribes.
///
/// §3.9.1.3: an `xml:base` attribute on the constructed element sets its base URI, resolved when
/// it is relative "against the base-uri property of its new parent node" — the enclosing
/// constructor's, or the static base URI when there is none. That is what makes
/// `<elem xml:base="fluster/"><a xml:base="now"/></elem>` give `a` the base URI
/// `…/fluster/now` rather than `…/now`.
fn annotate(draft: &mut Draft, frame: &Frame) {
    draft.base_uri = element_base_uri(&draft.attributes, frame);
    // §3.9.1.3 annotates a newly constructed element `xs:anyType` under construction mode
    // preserve — but only a processor with the Schema Import Feature can ever give it anything
    // else, and this one has none: nothing constructed here has been validated. Its annotation
    // is therefore `xs:untyped`, whichever mode is in force, which is what a processor without
    // typed data is required to report (`K2-DirectConElemContent-35b`,
    // `K2-ExternalVariablesWith-22a`). `element(*, xs:anyType)` still matches, because
    // `xs:untyped` derives from it, so the construction-mode cases are unaffected; the mode
    // continues to govern whether a *copied* node keeps the annotation it arrived with.
    draft.type_name = Some(ExpandedName::new(
        Some(oxixml_infoset::XSD_NAMESPACE),
        "untyped",
    ));
}

/// The base URI a constructed element takes from its `xml:base` attribute, if any, and from the
/// base URI it inherits.
fn element_base_uri(attributes: &[Draft], frame: &Frame) -> Option<String> {
    let inherited = frame.construction_base();
    let declared = attributes.iter().find_map(|attribute| {
        attribute
            .name
            .as_ref()
            .filter(|name| name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "base")
            .map(|_| attribute.value.clone())
    });
    match declared {
        Some(value) => Some(resolve_uri(inherited.as_deref(), &value)),
        None => inherited,
    }
}

/// Resolves `reference` against `base`, leaving it alone when there is no base to resolve
/// against or when it is already absolute.
pub(crate) fn resolve_uri(base: Option<&str>, reference: &str) -> String {
    let Some(base) = base else {
        return reference.to_owned();
    };
    oxixml_fn::strings::uri::resolve_uri_against(reference, base)
        .unwrap_or_else(|_| reference.to_owned())
}

/// Builds a computed element from an already-resolved name and an evaluated content sequence.
fn build_element(
    mut name: ConstructedName,
    declared: Vec<(Option<String>, String)>,
    items: Vec<XpathItem>,
    frame: &Frame,
) -> XQueryResult<Draft> {
    let mut scope_bindings = declared.clone();
    for binding in frame.namespaces.iter() {
        push_binding(&mut scope_bindings, binding.0.clone(), binding.1.clone());
    }
    fixup(&mut scope_bindings, &mut name);
    let inner = frame.with_inherited(scope_bindings);

    let mut content = Content::new(&inner);
    content.push_value(items)?;
    let (mut attributes, extra_namespaces, children) = content.finish()?;
    let mut seen: Vec<ExpandedName> = Vec::new();
    let mut attribute_names = Vec::with_capacity(attributes.len());
    for attribute in &attributes {
        if let Some(attribute_name) = &attribute.name {
            if seen.contains(attribute_name) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0025,
                    format!("the element has two attributes named `{attribute_name}`"),
                ));
            }
            seen.push(attribute_name.clone());
            attribute_names.push(ConstructedName::retained(attribute_name.clone()));
        }
    }
    let mut bindings = declared;
    for binding in extra_namespaces {
        push_binding(&mut bindings, binding.0, binding.1);
    }
    // §3.9.4: an element whose own name is in no namespace may not also declare a default
    // namespace, because the declaration would change what the name means.
    if name.name.namespace().is_none()
        && bindings
            .iter()
            .any(|(prefix, uri)| prefix.is_none() && !uri.is_empty())
    {
        return Err(XQueryError::new(
            XQueryErrorCode::Xqdy0102,
            "an element in no namespace may not declare a default namespace",
        ));
    }
    for binding in frame.namespaces.iter() {
        push_binding(&mut bindings, binding.0.clone(), binding.1.clone());
    }
    finish_bindings(&mut bindings, &mut name, &mut attribute_names);
    rename_attributes(&mut attributes, &attribute_names);
    let mut draft = Draft::element(name.spelt());
    draft.namespaces = bindings;
    draft.attributes = attributes;
    draft.children = children;
    annotate(&mut draft, frame);
    Ok(draft)
}

/// `xml:id` is normalized as an `xs:NCName`, so its value is collapsed rather than kept verbatim.
fn normalized_attribute_value(name: &ExpandedName, value: String) -> String {
    if name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "id" {
        return collapse(&value);
    }
    value
}

/// A constructor's finished content: its attributes, its namespace bindings and its children.
type BuiltContent = (Vec<Draft>, Vec<(Option<String>, String)>, Vec<Draft>);

/// Replaces every array in a constructor's content by its flattened members, as
/// [`array:flatten`](https://www.w3.org/TR/xpath-functions-31/#func-array-flatten) does.
fn flatten_arrays(items: Vec<XpathItem>) -> XQueryResult<Vec<XpathItem>> {
    if !items.iter().any(|item| item.as_array().is_some()) {
        return Ok(items);
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item.as_array() {
            Some(array) => {
                let mut members = Vec::new();
                for member in array.members() {
                    members.extend(member.clone().into_items().into_iter().map(XpathItem::from));
                }
                out.extend(flatten_arrays(members)?);
            }
            None => out.push(item),
        }
    }
    Ok(out)
}

/// Accumulates a constructor's content sequence and applies the rules of §3.9.1.3.
struct Content<'a> {
    frame: &'a Frame,
    attributes: Vec<Draft>,
    namespaces: Vec<(Option<String>, String)>,
    children: Vec<Draft>,
    pending: Vec<String>,
    seen_content: bool,
    /// A document constructor rejects attribute and namespace content outright, with
    /// `XPTY0004` rather than the `XQTY0024` an element constructor raises.
    document: bool,
}

impl<'a> Content<'a> {
    fn new(frame: &'a Frame) -> Self {
        Self {
            frame,
            attributes: Vec::new(),
            namespaces: Vec::new(),
            children: Vec::new(),
            pending: Vec::new(),
            seen_content: false,
            document: false,
        }
    }

    fn push_literal(&mut self, text: &str) {
        self.flush();
        self.seen_content = true;
        self.children.push(Draft::text(text));
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let joined = std::mem::take(&mut self.pending).join(" ");
        self.seen_content = true;
        self.children.push(Draft::text(joined));
    }

    fn copy_options(&self) -> CopyOptions {
        CopyOptions {
            copy_namespaces: self.frame.options.copy_namespaces,
            construction: self.frame.options.construction,
            inherited: self.frame.inherited.as_ref().clone(),
            base_uri: self.frame.construction_base(),
        }
    }

    /// Adds the value of *one* enclosed expression to the content sequence.
    ///
    /// §3.9.1.3 works per enclosed expression: each array is flattened first, each run of
    /// adjacent atomic values within the expression becomes one space-separated text node, and
    /// the run ends where the expression does. That is why `<c>{1}{2}{3}</c>` has the single
    /// child `123` while `<c>{1, 2, 3}</c>` has the single child `1 2 3`.
    /// The copy options for a node made by a *directly nested* constructor.
    ///
    /// §3.9.1.3 copies "each node returned by an enclosed expression", and only a copy is
    /// subject to the copy-namespaces mode. A nested direct constructor is not an enclosed
    /// expression: "Each nested direct constructor is evaluated according to the rules in 3.9.1
    /// … resulting in a new element, comment, or processing instruction node. Then: The parent
    /// property of the resulting node is then set to the newly constructed element node." The
    /// node therefore keeps exactly the in-scope namespaces §3.9.4 gave it — which is what makes
    /// `<e xmlns="urn:x"><a xmlns=""/></e>` leave `a` with no default namespace binding, and
    /// `<e a:n1="" b:n1=""><a:n1/></e>` leave `a:n1` without the `b` binding, even under the
    /// default `preserve, inherit`. This engine still has to rebuild the node as a [`Draft`], so
    /// it copies it verbatim: everything preserved, nothing inherited.
    fn verbatim_options(&self) -> CopyOptions {
        CopyOptions {
            copy_namespaces: CopyNamespaces {
                preserve: true,
                inherit: false,
            },
            construction: ConstructionMode::Preserve,
            inherited: Vec::new(),
            base_uri: self.frame.construction_base(),
        }
    }

    /// Adds the node of one directly nested constructor to the content sequence.
    fn push_constructed(&mut self, items: Vec<XpathItem>) -> XQueryResult<()> {
        let options = self.verbatim_options();
        for item in items {
            if let Some(node) = item.as_node() {
                self.push_node(&node.clone(), &options)?;
            }
        }
        Ok(())
    }

    fn push_value(&mut self, items: Vec<XpathItem>) -> XQueryResult<()> {
        let options = self.copy_options();
        for item in flatten_arrays(items)? {
            if let Some(value) = item.as_atomic() {
                self.pending.push(value.string_value());
                continue;
            }
            if let Some(node) = item.as_node() {
                self.push_node(&node.clone(), &options)?;
                continue;
            }
            return Err(XQueryError::new(
                XQueryErrorCode::Xqty0105,
                format!(
                    "{} cannot appear in a constructor's content",
                    item.describe()
                ),
            ));
        }
        self.flush();
        Ok(())
    }

    fn push_node(
        &mut self,
        node: &oxixml_infoset::NodeRef,
        options: &CopyOptions,
    ) -> XQueryResult<()> {
        match node.node_kind() {
            NodeKind::Attribute | NodeKind::Namespace if self.document => {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xpty0004,
                    "a document node may not have attribute or namespace children",
                ));
            }
            NodeKind::Attribute => {
                if self.seen_content || !self.pending.is_empty() {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqty0024,
                        "an attribute node must come before any other content",
                    ));
                }
                self.attributes.push(copy_node(node, options));
            }
            NodeKind::Namespace => {
                if self.seen_content || !self.pending.is_empty() {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xqty0024,
                        "a namespace node must come before any other content",
                    ));
                }
                let prefix = node.node_name().map(|name| name.local_name().to_owned());
                let uri = node.string_value();
                if let Some((_, bound)) = self
                    .namespaces
                    .iter()
                    .find(|(existing, _)| existing == &prefix)
                {
                    if bound != &uri {
                        return Err(XQueryError::new(
                            XQueryErrorCode::Xqdy0102,
                            "two namespace nodes bind the same prefix to different namespaces",
                        ));
                    }
                } else {
                    self.namespaces.push((prefix, uri));
                }
            }
            NodeKind::Document => {
                self.flush();
                self.seen_content = true;
                for child in node.children() {
                    self.children.push(copy_node(&child, options));
                }
            }
            _ => {
                self.flush();
                self.seen_content = true;
                self.children.push(copy_node(node, options));
            }
        }
        Ok(())
    }

    /// Merges adjacent text nodes, drops empty ones, and returns the finished content.
    fn finish(mut self) -> XQueryResult<BuiltContent> {
        self.flush();
        let mut merged: Vec<Draft> = Vec::with_capacity(self.children.len());
        for child in self.children {
            if child.kind == NodeKind::Text {
                if child.value.is_empty() {
                    continue;
                }
                if let Some(previous) = merged.last_mut() {
                    if previous.kind == NodeKind::Text {
                        previous.value.push_str(&child.value);
                        continue;
                    }
                }
            }
            merged.push(child);
        }
        Ok((self.attributes, self.namespaces, merged))
    }
}

// --- names ---------------------------------------------------------------------------------------

/// A constructed node's name, together with the prefix it was *written* with.
///
/// [XQuery 3.1 §3.9.4](https://www.w3.org/TR/xquery-31/#id-in-scope-namespaces) fixes up the
/// in-scope namespaces of a constructed element by looking at the prefix each of its names uses
/// — including the empty prefix, which is what makes `<inner/>` inside a default element
/// namespace produce `xmlns="…"` rather than an invented `xmlns:ns1="…"`. An
/// [`ExpandedName`] has thrown that prefix away, so it is carried alongside.
#[derive(Debug, Clone)]
pub(crate) struct ConstructedName {
    pub(crate) name: ExpandedName,
    /// The prefix as written: `None` for a name with no prefix at all.
    pub(crate) prefix: Option<String>,
}

impl ConstructedName {
    fn new(name: ExpandedName, prefix: Option<String>) -> Self {
        Self { name, prefix }
    }

    /// Recovers the written prefix an already-built node's name retained, so that a node placed
    /// in a constructor's content keeps the spelling its own constructor gave it.
    fn retained(name: ExpandedName) -> Self {
        let prefix = name.prefix().map(str::to_owned);
        Self { name, prefix }
    }

    /// Gives an attribute name that is in a namespace but carries no prefix one, as
    /// [XQuery 3.1 §3.9.3.2](https://www.w3.org/TR/xquery-31/#CompAttrConstructor) requires:
    /// "Attributes have no default namespace. The rules that expand attribute names create an
    /// implementation-dependent prefix if an attribute name has a namespace URI but no prefix is
    /// provided." The `xml` namespace has only one legal spelling, so it is not invented.
    fn invent_attribute_prefix(&mut self) {
        if self.prefix.is_some() {
            return;
        }
        match self.name.namespace() {
            None => {}
            Some(XML_NAMESPACE) => self.prefix = Some("xml".to_owned()),
            Some(_) => self.prefix = Some("ns1".to_owned()),
        }
    }

    /// The name with its prefix attached, which is what the constructed node records — `fn:name`,
    /// `fn:node-name` and `fn:prefix-from-QName` all report it.
    fn spelt(&self) -> ExpandedName {
        match &self.prefix {
            Some(prefix) if self.name.namespace().is_some() => {
                self.name.clone().with_prefix(prefix.as_str())
            }
            _ => self.name.clone(),
        }
    }
}

fn element_name(name: &CompiledName, frame: &Frame) -> XQueryResult<ConstructedName> {
    match name {
        CompiledName::Lexical(lexical) => {
            resolve_lexical(lexical, frame, false, XQueryErrorCode::Xpst0081, true)
        }
        CompiledName::Computed(program) => computed_name(program, frame, false),
    }
}

fn attribute_name(name: &CompiledName, frame: &Frame) -> XQueryResult<ConstructedName> {
    match name {
        CompiledName::Lexical(lexical) => {
            resolve_lexical(lexical, frame, true, XQueryErrorCode::Xpst0081, true)
        }
        CompiledName::Computed(program) => computed_name(program, frame, true),
    }
}

/// Whether an atomic value is an `xs:string` or an `xs:untypedAtomic`, the two types that may
/// spell a constructor's name lexically. A type derived from `xs:string` — `xs:NCName`, say —
/// counts; `xs:anyURI`, which is *not* derived from `xs:string`, does not.
fn is_string_like(value: &oxixml_infoset::AtomicValue) -> bool {
    let kind = value.atomic_type();
    kind == oxixml_infoset::AtomicType::UntypedAtomic
        || kind.is_derived_from(oxixml_infoset::AtomicType::String)
}

fn computed_name(
    program: &super::program::Program,
    frame: &Frame,
    is_attribute: bool,
) -> XQueryResult<ConstructedName> {
    let value = evaluate(program, frame)?;
    let atomized = value.atomize()?;
    let [single] = atomized.as_slice() else {
        return Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            "a computed constructor's name must be exactly one value",
        ));
    };
    if let Some(qname) = single.as_qname() {
        let namespace = qname.namespace().map(str::to_owned);
        return Ok(ConstructedName::new(
            ExpandedName::new(namespace, qname.local_name()),
            qname.prefix().map(str::to_owned),
        ));
    }
    // Only xs:QName, xs:string and xs:untypedAtomic may name a computed constructor.
    if !is_string_like(single) {
        return Err(XQueryError::new(
            XQueryErrorCode::Xpty0004,
            "a computed constructor's name must be an xs:QName, xs:string or xs:untypedAtomic",
        ));
    }
    let lexical = single.string_value();
    resolve_lexical(
        lexical.trim(),
        frame,
        is_attribute,
        XQueryErrorCode::Xqdy0074,
        false,
    )
}

/// Rejects a computed constructor's name that lands in one of the namespaces the specification
/// reserves — `XQDY0096` for an element, `XQDY0044` for an attribute.
fn check_constructed_name(name: &ConstructedName, is_attribute: bool) -> XQueryResult<()> {
    let code = if is_attribute {
        XQueryErrorCode::Xqdy0044
    } else {
        XQueryErrorCode::Xqdy0096
    };
    let namespace = name.name.namespace().unwrap_or_default();
    let prefix = name.prefix.as_deref();
    let reserved = namespace == XMLNS_NAMESPACE
        || prefix == Some("xmlns")
        || (is_attribute && prefix.is_none() && name.name.local_name() == "xmlns")
        || (prefix == Some("xml") && namespace != XML_NAMESPACE)
        || (namespace == XML_NAMESPACE && prefix.is_some_and(|prefix| prefix != "xml"));
    if reserved {
        return Err(XQueryError::new(
            code,
            format!("`{}` is in a reserved namespace", name.name),
        ));
    }
    Ok(())
}

fn processing_instruction_target(name: &CompiledName, frame: &Frame) -> XQueryResult<String> {
    let target = match name {
        CompiledName::Lexical(text) => text.clone(),
        CompiledName::Computed(program) => {
            let value = evaluate(program, frame)?;
            let atomized = value.atomize()?;
            match atomized.as_slice() {
                [single] => match single.as_qname() {
                    Some(qname) => qname.local_name().to_owned(),
                    // §3.9.3.6: only `xs:string`, `xs:untypedAtomic` and `xs:QName` may name a
                    // processing instruction; `xs:anyURI` and friends are `XPTY0004`.
                    None if is_string_like(single) => single.string_value().trim().to_owned(),
                    None => {
                        return Err(XQueryError::new(
                            XQueryErrorCode::Xpty0004,
                            "a processing instruction's target must be an xs:string, \
                             xs:untypedAtomic or xs:QName",
                        ));
                    }
                },
                _ => {
                    return Err(XQueryError::new(
                        XQueryErrorCode::Xpty0004,
                        "a processing instruction's target must be exactly one value",
                    ));
                }
            }
        }
    };
    if !is_ncname(&target) {
        return Err(XQueryError::new(
            XQueryErrorCode::Xqdy0041,
            format!("`{target}` is not a valid processing-instruction target"),
        ));
    }
    if target.eq_ignore_ascii_case("xml") {
        return Err(XQueryError::new(
            XQueryErrorCode::Xqdy0064,
            "`xml` is a reserved processing-instruction target",
        ));
    }
    Ok(target)
}

/// Resolves a lexical name against the constructor's namespace scope, then the static context.
fn resolve_lexical(
    lexical: &str,
    frame: &Frame,
    is_attribute: bool,
    code: XQueryErrorCode,
    from_source: bool,
) -> XQueryResult<ConstructedName> {
    if let Some(rest) = lexical.strip_prefix("Q{") {
        // A name written in the query has already been through the lexer, which rejects a
        // literal brace inside a `BracedURILiteral` but *expands* character references — so
        // `attribute Q{&#x7b;&#x7d;}x {}` legitimately names the namespace `{}`, and the `}`
        // that closes the URI is the last one, never the first. A name computed at run time
        // from a string has had no such scrutiny, so there a brace is exactly what makes
        // `"Q{{}x"` the invalid name `XQDY0074` is for.
        let split = if from_source {
            rest.rsplit_once('}')
        } else {
            rest.split_once('}')
        };
        let Some((uri, local)) = split else {
            return Err(XQueryError::new(code, format!("`{lexical}` is not a name")));
        };
        // The whitespace of a `BracedURILiteral` is normalized as for `xs:anyURI`.
        let uri = collapse(uri);
        if (!from_source && uri.contains(['{', '}'])) || !is_ncname(local) {
            return Err(XQueryError::new(
                code,
                format!("`{lexical}` is not a valid name"),
            ));
        }
        return Ok(ConstructedName::new(
            ExpandedName::new((!uri.is_empty()).then_some(uri), local),
            None,
        ));
    }
    let statik = frame.context.static_context();
    match lexical.split_once(':') {
        None => {
            if !is_ncname(lexical) {
                return Err(XQueryError::new(
                    code,
                    format!("`{lexical}` is not a valid QName"),
                ));
            }
            if is_attribute {
                return Ok(ConstructedName::new(
                    ExpandedName::new(None::<&str>, lexical),
                    None,
                ));
            }
            let declared = frame
                .namespaces
                .iter()
                .find(|(prefix, _)| prefix.is_none())
                .map(|(_, uri)| uri.clone());
            let default = declared.or_else(|| {
                statik
                    .default_element_namespace()
                    .map(std::borrow::ToOwned::to_owned)
            });
            Ok(ConstructedName::new(
                ExpandedName::new(default.filter(|uri| !uri.is_empty()), lexical),
                None,
            ))
        }
        Some((prefix, local)) => {
            if !is_ncname(prefix) || !is_ncname(local) {
                return Err(XQueryError::new(
                    code,
                    format!("`{lexical}` is not a valid QName"),
                ));
            }
            let bound = frame
                .namespaces
                .iter()
                .find(|(bound, _)| bound.as_deref() == Some(prefix))
                .map(|(_, uri)| uri.clone())
                .or_else(|| statik.namespace(prefix).map(std::borrow::ToOwned::to_owned));
            match bound.filter(|uri| !uri.is_empty()) {
                Some(uri) => Ok(ConstructedName::new(
                    ExpandedName::new(Some(uri), local),
                    Some(prefix.to_owned()),
                )),
                None => Err(XQueryError::new(
                    code,
                    format!("the namespace prefix `{prefix}` is not bound"),
                )),
            }
        }
    }
}

/// Collapses whitespace, as the `xs:anyURI` whitespace facet does.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn check_reserved_binding(
    prefix: Option<&str>,
    uri: &str,
    code: XQueryErrorCode,
) -> XQueryResult<()> {
    let reserved = matches!(prefix, Some("xml")) && uri != XML_NAMESPACE
        || matches!(prefix, Some("xmlns"))
        || uri == XMLNS_NAMESPACE
        || (uri == XML_NAMESPACE && !matches!(prefix, Some("xml")));
    if reserved {
        return Err(XQueryError::new(
            code,
            "a namespace declaration may not rebind a reserved prefix or namespace",
        ));
    }
    Ok(())
}

/// `xml:id` and `xml:space` have values a constructor must check.
fn check_special_attribute(name: &ExpandedName, value: &str) -> XQueryResult<()> {
    if name.namespace() != Some(XML_NAMESPACE) {
        return Ok(());
    }
    match name.local_name() {
        "id" => {
            let normalized = value.trim();
            if !is_ncname(normalized) {
                return Err(XQueryError::new(
                    XQueryErrorCode::Xqdy0091,
                    "`xml:id` must be an NCName",
                ));
            }
            Ok(())
        }
        "space" => {
            if value == "preserve" || value == "default" {
                return Ok(());
            }
            Err(XQueryError::new(
                XQueryErrorCode::Xqdy0092,
                "`xml:space` must be `preserve` or `default`",
            ))
        }
        _ => Ok(()),
    }
}

/// The string an attribute, comment, text or PI constructor takes from its content: every item
/// atomized, converted to a string, and joined with single spaces.
fn atomized_string(value: &XpathValue) -> XQueryResult<String> {
    let atomized = value.atomize()?;
    let parts: Vec<String> = atomized
        .iter()
        .map(oxixml_infoset::AtomicValue::string_value)
        .collect();
    Ok(parts.join(" "))
}
