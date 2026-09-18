//! Conditional inclusion — the `vc:` attributes of
//! [XSD 1.1 Part 1 §4.2.6](https://www.w3.org/TR/xmlschema11-1/#cip).
//!
//! A schema document may guard any of its elements with attributes in the
//! *version control* namespace, so that one document can be written for several
//! versions of XML Schema at once: an element whose guard does not hold is
//! removed **before** the document is read, exactly as if it had never been
//! written, and no schema representation constraint is applied to it.
//!
//! | Attribute | The element is kept when |
//! |---|---|
//! | `vc:minVersion` | the processor's version is **at least** the value |
//! | `vc:maxVersion` | the processor's version is **less than** the value |
//! | `vc:typeAvailable` | **every** named type is known |
//! | `vc:typeUnavailable` | **some** named type is unknown |
//! | `vc:facetAvailable` | **every** named facet is known |
//! | `vc:facetUnavailable` | **some** named facet is unknown |
//!
//! All the guards on one element are conjoined, and removing an element removes
//! its descendants with it. The guards themselves are dropped from the document
//! afterwards, so nothing downstream has to know about them.
//!
//! # Both versions apply it
//!
//! The mechanism exists precisely so that a document can be shared between an
//! XSD 1.0 and an XSD 1.1 processor, so the pass runs under **both** — the
//! version each guard is compared against is the one being assembled under, and
//! the types and facets it may ask about are the ones that version defines.
//! `saxonData/VC/vc001` guards an `xs:assert` — a 1.1-only element — with
//! `vc:minVersion="1.1"`, and the suite marks the document valid under 1.0 and
//! its `vc001.n1.xml` (which the assertion would reject) *valid* under 1.0 and
//! *invalid* under 1.1: the guard has to be honoured under 1.0 for either
//! answer to come out right.
//!
//! What a 1.0 processor does **not** do is police the guards themselves. To it
//! they are ordinary attributes from a foreign namespace, which the schema for
//! schemas allows without saying anything about their values, so a guard it
//! cannot interpret is ignored rather than reported. `saxonData/VC/vc902` and
//! `vc903` are the same document — an attribute carrying
//! `vc:minVersion="10g"` — expected *valid* under 1.0 and *invalid* under 1.1.

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_qname::ExpandedName;

use crate::error::SchemaError;
use crate::names::{VC_NS, XSD_NS, XsdVersion};
use crate::parse::{node_path, resolve_qname};

/// The constraining facets XML Schema 1.0 defines
/// ([Part 2 §4.3](https://www.w3.org/TR/xmlschema-2/#rf-facets)).
const FACETS_1_0: &[&str] = &[
    "length",
    "minLength",
    "maxLength",
    "pattern",
    "enumeration",
    "whiteSpace",
    "maxInclusive",
    "maxExclusive",
    "minInclusive",
    "minExclusive",
    "totalDigits",
    "fractionDigits",
];

/// The two constraining facets XML Schema 1.1 adds — `assertion` (Part 2
/// §4.3.12) and `explicitTimezone` (§4.3.13).
///
/// Together with [`FACETS_1_0`] these are the fourteen the rest of the crate
/// accepts inside `xs:restriction`; only conditional inclusion has to know
/// which version each of them belongs to, which is why the split lives here.
const FACETS_1_1: &[&str] = &["assertion", "explicitTimezone"];

/// What conditional inclusion is being asked about: which processor is reading
/// the document.
struct Processor {
    /// The version number the `vc:minVersion` and `vc:maxVersion` bounds are
    /// compared against.
    version: f64,
    /// Whether this is an XSD 1.1 processor, which decides both what
    /// `vc:facetAvailable` may name and whether a guard it cannot interpret is
    /// an error.
    one_one: bool,
    /// The datatypes `vc:typeAvailable` may name: the built-in hierarchy of the
    /// version in force. Only the built-ins can be answered — the pass runs
    /// before any user-defined component exists — which is what the appendix
    /// intends, since the guards are there to hide a *vocabulary* the processor
    /// may not have.
    types: crate::schema::Schema,
}

impl Processor {
    fn new(version: XsdVersion) -> Result<Self, SchemaError> {
        Ok(Self {
            // Each processor is exactly the version it implements.
            version: if version.is_1_1() { 1.1 } else { 1.0 },
            one_one: version.is_1_1(),
            types: crate::builtins::build(version)?,
        })
    }

    fn type_known(&self, name: &ExpandedName) -> bool {
        self.types
            .find_type(name.namespace(), name.local_name())
            .is_some()
    }

    fn facet_known(&self, name: &ExpandedName) -> bool {
        name.namespace() == Some(XSD_NS)
            && (FACETS_1_0.contains(&name.local_name())
                || (self.one_one && FACETS_1_1.contains(&name.local_name())))
    }
}

/// Removes every element of `doc` whose `vc:` guards do not hold.
///
/// Returns `false` when the guard on `xs:schema` *itself* does not hold, in
/// which case the whole schema document contributes nothing — which the
/// appendix makes a legitimate outcome rather than an error
/// (`saxonData/VC/vc006`).
///
/// # Errors
///
/// Returns `src-cip` when a guard of a schema document read as XSD 1.1 is not
/// of the type the appendix gives it — a `minVersion` that is not a decimal, or
/// a `typeAvailable` entry that is not a QName. Under XSD 1.0 such a guard is
/// ignored instead; see the module documentation.
pub(crate) fn prune(
    doc: &mut Document,
    root: NodeId,
    version: XsdVersion,
) -> Result<bool, SchemaError> {
    // Almost every schema document carries no guard at all. Finding that out
    // costs one pass over the attributes and saves constructing the built-in
    // datatype hierarchy that `vc:typeAvailable` would have to be answered
    // from.
    if !carries_guard(doc, root) {
        return Ok(true);
    }
    let processor = Processor::new(version)?;
    let mut doomed = Vec::new();
    let mut guards = Vec::new();
    collect(doc, root, &processor, &mut doomed, &mut guards)?;
    // The document element has no parent to be removed from; a failed guard on
    // it disables the document instead.
    if !holds(doc, root, &processor)? {
        return Ok(false);
    }
    for (parent, child) in doomed {
        let _ = doc.remove_child(parent, child);
    }
    // The guards have done their work; leaving them in would make every
    // element that carries one look as though it had an unknown attribute.
    for (element, attribute) in guards {
        let _ = doc.remove_attribute_node(element, attribute);
    }
    Ok(true)
}

/// Whether any element of the subtree rooted at `node` carries an attribute in
/// the version-control namespace.
///
/// Walked with an explicit stack rather than by recursion: it runs over every
/// schema document ever loaded, whether or not the document has anything to do
/// with conditional inclusion, and there is no reason for the one pass that is
/// always paid to put anything on the stack at all.
fn carries_guard(doc: &Document, node: NodeId) -> bool {
    let mut stack = vec![node];
    while let Some(current) = stack.pop() {
        if let Some(element) = doc.element(current)
            && element.attribute_ids().iter().any(|attribute| {
                doc.attr(*attribute)
                    .is_some_and(|data| data.namespace_uri() == Some(VC_NS))
            })
        {
            return true;
        }
        stack.extend(
            doc.children(current)
                .filter(|child| doc.kind(*child) == Some(NodeKind::Element)),
        );
    }
    false
}

/// Walks the document, recording the elements to remove and the guard
/// attributes to strip.
///
/// The recursion follows the element nesting of the schema document, which
/// [`SchemaSet::push_document`](crate::SchemaSet) has already bounded at
/// [`SchemaSet::MAX_DEPTH`](crate::SchemaSet::MAX_DEPTH) by the time this runs.
fn collect(
    doc: &Document,
    node: NodeId,
    processor: &Processor,
    doomed: &mut Vec<(NodeId, NodeId)>,
    guards: &mut Vec<(NodeId, NodeId)>,
) -> Result<(), SchemaError> {
    let mut carries = false;
    if let Some(element) = doc.element(node) {
        for attribute in element.attribute_ids() {
            let Some(data) = doc.attr(*attribute) else {
                continue;
            };
            if data.namespace_uri() != Some(VC_NS) {
                continue;
            }
            carries = true;
            guards.push((node, *attribute));
        }
    }
    if carries && !holds(doc, node, processor)? {
        if let Some(parent) = doc.parent(node) {
            doomed.push((parent, node));
        }
        // Everything below a removed element goes with it, guards included.
        return Ok(());
    }
    let children: Vec<NodeId> = doc
        .children(node)
        .filter(|child| doc.kind(*child) == Some(NodeKind::Element))
        .collect();
    for child in children {
        collect(doc, child, processor, doomed, guards)?;
    }
    Ok(())
}

/// Whether every `vc:` guard on `node` holds.
///
/// A guard this processor cannot interpret contributes nothing to the
/// conjunction — under XSD 1.1 that case has already been reported as
/// `src-cip`, and under XSD 1.0 it is deliberately not an error at all.
fn holds(doc: &Document, node: NodeId, processor: &Processor) -> Result<bool, SchemaError> {
    let mut kept = true;
    if let Some(text) = doc.get_attribute_ns(node, Some(VC_NS), "minVersion")
        && let Some(bound) = decimal(text, "minVersion", doc, node, processor)?
    {
        kept &= processor.version >= bound;
    }
    if let Some(text) = doc.get_attribute_ns(node, Some(VC_NS), "maxVersion")
        && let Some(bound) = decimal(text, "maxVersion", doc, node, processor)?
    {
        kept &= processor.version < bound;
    }
    if let Some(text) = doc.get_attribute_ns(node, Some(VC_NS), "typeAvailable")
        && let Some(names) = qnames(doc, node, text, processor)?
    {
        kept &= names.iter().all(|name| processor.type_known(name));
    }
    if let Some(text) = doc.get_attribute_ns(node, Some(VC_NS), "typeUnavailable")
        && let Some(names) = qnames(doc, node, text, processor)?
    {
        kept &= names.iter().any(|name| !processor.type_known(name));
    }
    if let Some(text) = doc.get_attribute_ns(node, Some(VC_NS), "facetAvailable")
        && let Some(names) = qnames(doc, node, text, processor)?
    {
        kept &= names.iter().all(|name| processor.facet_known(name));
    }
    if let Some(text) = doc.get_attribute_ns(node, Some(VC_NS), "facetUnavailable")
        && let Some(names) = qnames(doc, node, text, processor)?
    {
        kept &= names.iter().any(|name| !processor.facet_known(name));
    }
    Ok(kept)
}

/// The value of a `vc:minVersion`/`vc:maxVersion` guard, or [`None`] when it is
/// not a decimal and this processor does not police that.
fn decimal(
    text: &str,
    name: &str,
    doc: &Document,
    node: NodeId,
    processor: &Processor,
) -> Result<Option<f64>, SchemaError> {
    let text = crate::value::collapse(text);
    match crate::bigdecimal::BigDecimal::parse(&text)
        .and_then(|value| value.to_string().parse::<f64>().ok())
    {
        Some(value) => Ok(Some(value)),
        None if !processor.one_one => Ok(None),
        None => Err(SchemaError::new(
            "src-cip",
            format!("`vc:{name}` must be an `xs:decimal`, not `{text}`"),
        )
        .at(node_path(doc, node))),
    }
}

/// The names a `vc:typeAvailable`-style guard lists, or [`None`] when one of
/// them is not a `QName` and this processor does not police that.
fn qnames(
    doc: &Document,
    node: NodeId,
    text: &str,
    processor: &Processor,
) -> Result<Option<Vec<ExpandedName>>, SchemaError> {
    let mut names = Vec::new();
    for token in text.split_whitespace() {
        match resolve_qname(doc, node, token) {
            Ok(name) => names.push(name),
            Err(_) if !processor.one_one => return Ok(None),
            Err(error) => return Err(error.or_at(node_path(doc, node))),
        }
    }
    Ok(Some(names))
}
