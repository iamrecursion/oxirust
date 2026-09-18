//! Schema-validity assessment of an instance document.
//!
//! The entry points are [`Schema::validate_document`] and
//! [`Schema::validate_str`]. Assessment follows
//! [XML Schema Part 1 §3.3.4 and §3.4.4](https://www.w3.org/TR/xmlschema-1/):
//! an element is matched against a declaration (or, under a wildcard, against
//! whatever the `processContents` asks for), its type is chosen — possibly
//! overridden by `xsi:type` — its attributes are checked against the type's
//! attribute uses and wildcard, its children against the type's content model,
//! and finally its identity constraints are evaluated over the subtree.
//!
//! The result carries a light **PSVI**: every element and attribute node that
//! was assessed is annotated with the declaration and type that governed it,
//! with the `whiteSpace`-normalized value, and with the value that value
//! denotes.

mod assertions;
mod attrs;
mod content;
mod hints;
mod identity;

use std::collections::HashMap;
use std::path::Path;

use oxixml_dom::{Document, NodeId, NodeKind};

use crate::components::{ContentType, ElementId, ProcessContents, TypeId, TypeKind};
use crate::error::ValidationError;
use crate::names::XSI_NS;
use crate::parse::node_path;
use crate::schema::Schema;
use crate::simple;
use crate::value::{NamespaceContext, Value};

/// How deeply an instance document may nest before assessment gives up.
///
/// Assessment walks the instance tree by recursion — an element, its content
/// model, then each of its element children in turn — so an arbitrarily deep
/// document would otherwise consume an arbitrary amount of stack and abort the
/// process. A document that reaches this depth is reported as invalid with the
/// `cvc-max-depth` code instead, exactly like any other rule failure; nothing
/// panics and nothing aborts.
///
/// The ceiling is generous by the standards of real documents — the whole W3C
/// XML Schema Test Suite peaks at 27 levels — and applies to
/// [`Schema::validate_document`] as well as to [`Schema::validate_str`], since
/// a [`Document`] can be built through the DOM's own mutation API without
/// passing the parser's own nesting cap.
pub const MAX_DEPTH: usize = 256;

/// How thoroughly a node was assessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validity {
    /// The node was assessed and is valid.
    Valid,
    /// The node was assessed and is not valid.
    Invalid,
    /// No declaration or type could be found (`processContents="lax"` or
    /// `"skip"`).
    NotKnown,
}

/// The post-schema-validation annotations of one element node.
#[derive(Debug, Clone)]
pub struct ElementPsvi {
    /// The declaration that governed the element, when one was found.
    pub declaration: Option<ElementId>,
    /// The type definition it was validated against.
    pub type_definition: Option<TypeId>,
    /// Whether it is valid.
    ///
    /// [Part 1 §3.3.4](https://www.w3.org/TR/xmlschema-1/) makes an element
    /// *valid* only when it is locally valid **and** every attribute and every
    /// element child of it is valid, so a failure anywhere in the subtree — a
    /// facet, the content model, an identity constraint — leaves every element
    /// on the path from it to the validation root [`Validity::Invalid`].
    pub validity: Validity,
    /// Whether `xsi:nil="true"` applied.
    pub nil: bool,
    /// The `whiteSpace`-normalized character content, for simple content.
    pub normalized_value: Option<String>,
    /// The value the content denotes, for simple content.
    pub value: Option<Value>,
    /// The union member type that accepted the content, when the governing
    /// simple type is a union — the *\[member type definition\]* property of
    /// Part 1 §3.3.5. [`None`] for every other variety.
    pub member_type: Option<TypeId>,
    /// Whether the value came from a `default` or `fixed` value constraint
    /// rather than from the document.
    pub from_default: bool,
}

/// The post-schema-validation annotations of one attribute node.
#[derive(Debug, Clone)]
pub struct AttributePsvi {
    /// The declaration that governed the attribute.
    pub declaration: Option<crate::components::AttributeId>,
    /// The simple type it was validated against.
    pub type_definition: Option<TypeId>,
    /// Whether it is valid.
    pub validity: Validity,
    /// The `whiteSpace`-normalized value.
    pub normalized_value: Option<String>,
    /// The value it denotes.
    pub value: Option<Value>,
    /// The union member type that accepted the value, when the governing type
    /// is a union — the *\[member type definition\]* property of Part 1
    /// §3.2.5. [`None`] for every other variety.
    pub member_type: Option<TypeId>,
    /// Whether *this occurrence* is `inheritable` (XSD 1.1 §3.2.2/§3.5.2):
    /// the attribute **use** that matched it, when one did and it carries an
    /// explicit `inheritable`, otherwise the governing attribute
    /// **declaration**'s own `{inheritable}`. A `ref`'d use's own
    /// `inheritable` wins over the referenced declaration's
    /// (`CTA/cta0011`'s schema states this outright: "inheritability differs
    /// between the attribute decl and the attribute use: the attribute use
    /// wins"), which is why this is captured per occurrence rather than
    /// re-derived from `declaration` alone.
    pub inheritable: bool,
}

/// The PSVI produced by an assessment.
#[derive(Debug, Clone, Default)]
pub struct Psvi {
    elements: HashMap<NodeId, ElementPsvi>,
    attributes: HashMap<NodeId, AttributePsvi>,
    defaulted_attributes: HashMap<NodeId, Vec<(String, String)>>,
}

impl Psvi {
    /// The annotations of an element node.
    #[must_use]
    pub fn element(&self, node: NodeId) -> Option<&ElementPsvi> {
        self.elements.get(&node)
    }

    /// The annotations of an attribute node.
    #[must_use]
    pub fn attribute(&self, node: NodeId) -> Option<&AttributePsvi> {
        self.attributes.get(&node)
    }

    /// The attributes that a `default` value would add to an element, as
    /// `(qualified name, normalized value)` pairs.
    #[must_use]
    pub fn defaulted_attributes(&self, node: NodeId) -> &[(String, String)] {
        self.defaulted_attributes
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The number of assessed element nodes.
    #[must_use]
    pub fn assessed_element_count(&self) -> usize {
        self.elements.len()
    }
}

/// The outcome of assessing a document.
#[derive(Debug, Clone)]
pub struct ValidationOutcome {
    /// Whether the document is schema-valid.
    pub valid: bool,
    /// Every rule that failed, in document order.
    pub errors: Vec<ValidationError>,
    /// The annotations produced along the way.
    pub psvi: Psvi,
}

impl ValidationOutcome {
    /// The first error, when the document is not valid.
    #[must_use]
    pub fn first_error(&self) -> Option<&ValidationError> {
        self.errors.first()
    }
}

/// One attribute the schema supplied for an element that did not carry it —
/// the *Attribute Default Value* contribution of §3.4.5.
///
/// The public [`Psvi::defaulted_attributes`] reports the same additions as
/// `(name, normalized value)` pairs, which is all a caller serializing the
/// augmented infoset needs. An identity-constraint `field`, on the other hand,
/// has to *select* the addition by expanded name and then compare what it
/// denotes in the value space, so the name is kept in parts and the value is
/// kept as a [`Value`].
#[derive(Debug, Clone)]
pub(crate) struct DefaultedAttribute {
    pub namespace: Option<String>,
    pub local: String,
    pub value: Value,
}

pub(crate) struct Validator<'a> {
    pub schema: &'a Schema,
    /// The components assembled from the instance's own `xsi:schemaLocation`
    /// hints, when the caller asked for them to be followed.
    ///
    /// §4.3.2 makes the hints subordinate to what the processor already has, so
    /// this is consulted **only** where [`Validator::schema`] has nothing: a
    /// declaration it does not hold, a type an `xsi:type` names that it does
    /// not define. Once a subtree is handed over, the whole of it is assessed
    /// against these components rather than against `schema`, because the two
    /// schemas' component identifiers index different arenas and a content
    /// model may not straddle them.
    pub hints: Option<&'a Schema>,
    pub doc: &'a Document,
    pub errors: Vec<ValidationError>,
    pub elements: HashMap<NodeId, ElementPsvi>,
    pub attributes: HashMap<NodeId, AttributePsvi>,
    pub defaulted: HashMap<NodeId, Vec<(String, String)>>,
    /// The same additions as [`Validator::defaulted`], in the shape an
    /// identity-constraint `field` needs.
    pub idc_defaults: HashMap<NodeId, Vec<DefaultedAttribute>>,
    /// The *ID/IDREF table* of §3.17.5.2: for every string used as an `ID` or
    /// an `IDREF`, the set of elements which ·bind· it — that is, which carry
    /// it on an attribute of type `ID` or on a **child** of type `ID`.
    pub ids: HashMap<String, Vec<NodeId>>,
    /// Every `IDREF` value, with the path and the node that carries it.
    pub idrefs: Vec<(String, String, NodeId)>,
    /// `ID` values written on an element whose parent lies outside the
    /// validation root, so that the binding they would make is empty
    /// (§3.3.4.5 clause 1), with the path and the node that carries them.
    pub unbound_ids: Vec<(String, String, NodeId)>,
    pub idc_nodes: Vec<NodeId>,
    /// Elements ·attributed to· a `skip` wildcard. §3.10.4.1 makes an item
    /// *skipped* when it is attributed to such a wildcard **or one of its
    /// ancestor elements is**, so only the outermost node is recorded and
    /// [`Validator::is_skipped`] walks up.
    pub skipped: std::collections::HashSet<NodeId>,
    /// The names of the *unparsed entities* the instance's internal DTD subset
    /// declares, when it has one — what `xs:ENTITY` values must name
    /// (§3.3.4 / Datatypes §3.3.11). [`None`] when the document carries no
    /// internal subset at all, in which case the validator has no entity
    /// information to check against and does not try.
    pub unparsed_entities: Option<Vec<String>>,
    /// How many elements deep the assessment currently is, against
    /// [`MAX_DEPTH`].
    pub depth: usize,
}

/// The names declared by `<!ENTITY name … NDATA notation>` in an internal DTD
/// subset.
///
/// The DOM keeps the internal subset verbatim on the document-type node, so
/// the declarations are read back from there; an *external* subset is never
/// retrieved, which is why a document whose entities live in one cannot be
/// checked (`saxonData/Id/id018` … `id021` all use an internal subset).
fn unparsed_entities(document: &Document) -> Option<Vec<String>> {
    let Some(doctype) = document.doctype() else {
        // No `DOCTYPE` at all: the instance declares no entities, so *every*
        // `xs:ENTITY` value in it is invalid — the empty set is exactly the
        // right answer, not "unknown" (`saxonData/Id/id017`'s `n01`, whose
        // `entity` attribute takes an `xs:ENTITY` default in a document with
        // no declarations).
        return Some(Vec::new());
    };
    let data = document.doctype_data(doctype)?;
    // An *external* subset is never retrieved, so a document that has one may
    // declare entities this validator cannot see; the check is skipped rather
    // than guessed at.
    if data.system_id().is_some() || data.public_id().is_some() {
        return None;
    }
    let Some(subset) = data.internal_subset() else {
        return Some(Vec::new());
    };
    let mut names = Vec::new();
    let mut rest = subset;
    while let Some(start) = rest.find("<!ENTITY") {
        let after = &rest[start + "<!ENTITY".len()..];
        let end = after.find('>').unwrap_or(after.len());
        let declaration = &after[..end];
        rest = &after[end..];
        let mut tokens = declaration.split_whitespace();
        let Some(name) = tokens.next() else { continue };
        // `<!ENTITY % name …>` is a parameter entity, not a general one.
        if name == "%" || name.starts_with('%') {
            continue;
        }
        if declaration.split_whitespace().any(|token| token == "NDATA") {
            names.push(name.to_owned());
        }
    }
    Some(names)
}

impl Schema {
    /// Assesses `document` against this schema.
    ///
    /// The schema is exactly the one that was assembled: any
    /// `xsi:schemaLocation` the instance carries is left alone, and nothing is
    /// read from the file system. Use
    /// [`validate_document_at`](Self::validate_document_at) to follow the
    /// hints.
    #[must_use]
    pub fn validate_document(&self, document: &Document) -> ValidationOutcome {
        self.assess(document, None)
    }

    /// Assesses `document`, following the schema documents it names.
    ///
    /// [§4.3.2](https://www.w3.org/TR/xmlschema-1/#xsi_schemaLocation) lets an
    /// instance point at the schema documents it expects to be assessed
    /// against, through `xsi:schemaLocation` (pairs of a namespace name and a
    /// location) and `xsi:noNamespaceSchemaLocation`. Every location is
    /// resolved against `base_uri` — the URI the instance itself came from —
    /// and read from the **file system**; a location carrying a remote scheme
    /// is ignored rather than fetched, so assessment never reaches the network.
    ///
    /// The hints are subordinate to this schema, as §4.3.2 requires: a
    /// declaration, an attribute or a type that is already here wins, and what
    /// the hints supply is consulted only where it is not. A hint that cannot
    /// be retrieved, or that names documents which do not assemble, is not an
    /// error — the assessment simply proceeds without it and reports whatever
    /// the instance then breaks.
    ///
    /// Because the schema documents are chosen by the *instance*, this reads
    /// files the caller did not name. Point it at instances whose origin you
    /// trust, or confine the process, exactly as you would for any other
    /// document-driven retrieval.
    #[must_use]
    pub fn validate_document_at(&self, document: &Document, base_uri: &str) -> ValidationOutcome {
        let hints = hints::load(self.version(), document, base_uri);
        self.assess(document, hints.as_ref())
    }

    /// Reads an instance from the file system and assesses it, following the
    /// schema documents it names against its own location.
    ///
    /// This is [`validate_document_at`](Self::validate_document_at) over a file:
    /// the same hint handling, with the file's path as the base URI. A file
    /// that cannot be read or is not well-formed is reported as a single
    /// `cvc-document` error rather than as a panic or an `Err`, so that the
    /// result of assessing an instance always has the same shape.
    #[must_use]
    pub fn validate_file(&self, path: impl AsRef<Path>) -> ValidationOutcome {
        let path = path.as_ref();
        let text = std::fs::read(path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| crate::source::decode(&bytes));
        let text = match text {
            Ok(text) => text,
            Err(reason) => {
                return Self::unreadable(format!("cannot read `{}`: {reason}", path.display()));
            }
        };
        match oxixml_dom::build::parse_str(&text) {
            Ok(document) => self.validate_document_at(&document, &path.to_string_lossy()),
            Err(error) => Self::unreadable(format!("the instance is not well-formed: {error}")),
        }
    }

    /// Parses `text` and assesses it.
    ///
    /// A parse failure is reported as a single `cvc-document` error rather than
    /// as a panic, so that this can be used directly on untrusted input. The
    /// text has no location, so `xsi:schemaLocation` is left alone here too.
    #[must_use]
    pub fn validate_str(&self, text: &str) -> ValidationOutcome {
        match oxixml_dom::build::parse_str(text) {
            Ok(document) => self.validate_document(&document),
            Err(error) => Self::unreadable(format!("the instance is not well-formed: {error}")),
        }
    }

    /// The outcome of an instance that could not be assessed at all.
    fn unreadable(message: String) -> ValidationOutcome {
        ValidationOutcome {
            valid: false,
            errors: vec![ValidationError::new("cvc-document", "/", message)],
            psvi: Psvi::default(),
        }
    }

    fn assess<'a>(
        &'a self,
        document: &'a Document,
        hints: Option<&'a Schema>,
    ) -> ValidationOutcome {
        let mut validator = Validator::new(self, document, hints);
        match document.document_element() {
            Some(root) => validator.assess_root(root),
            None => validator.error("cvc-elt.1", "/", "the document has no element"),
        }
        validator.finish()
    }
}

impl<'a> Validator<'a> {
    fn new(schema: &'a Schema, doc: &'a Document, hints: Option<&'a Schema>) -> Self {
        Self {
            schema,
            hints,
            doc,
            errors: Vec::new(),
            elements: HashMap::new(),
            attributes: HashMap::new(),
            defaulted: HashMap::new(),
            idc_defaults: HashMap::new(),
            ids: HashMap::new(),
            idrefs: Vec::new(),
            unbound_ids: Vec::new(),
            idc_nodes: Vec::new(),
            skipped: std::collections::HashSet::new(),
            unparsed_entities: unparsed_entities(doc),
            depth: 0,
        }
    }

    /// A validator for part of a document another one is already walking:
    /// it starts at the depth reached so far, so the nesting ceiling holds over
    /// the instance as a whole rather than being restarted at the hand-over.
    pub(crate) fn new_nested(schema: &'a Schema, doc: &'a Document, depth: usize) -> Self {
        let mut nested = Self::new(schema, doc, None);
        nested.depth = depth;
        nested
    }

    /// The declaration a hinted schema has for a name this one does not.
    pub(crate) fn hinted_element(&self, namespace: Option<&str>, local: &str) -> Option<ElementId> {
        self.hints?.find_element(namespace, local)
    }

    /// Whether the element's `xsi:type` names a type only a hinted schema
    /// defines — the other way an element can belong to the hints rather than
    /// to this schema, since §3.3.4 lets `xsi:type` govern an element that has
    /// no declaration at all.
    pub(crate) fn hinted_xsi_type(&self, node: NodeId) -> bool {
        let Some(hints) = self.hints else {
            return false;
        };
        let Some(text) = self.xsi_type(node) else {
            return false;
        };
        let context = crate::parse::namespace_context(self.doc, node);
        let Some((namespace, local)) = resolve_qname_value(&text, &context) else {
            return false;
        };
        self.schema
            .find_type(namespace.as_deref(), &local)
            .is_none()
            && hints.find_type(namespace.as_deref(), &local).is_some()
    }

    /// Assesses `node` and everything below it against the hinted schema, and
    /// splices what that produced into this assessment.
    ///
    /// The subtree is handed over whole: a content model built from the hinted
    /// components can only be walked with those components, and their
    /// identifiers are meaningless against [`Validator::schema`]. Identity
    /// constraints are therefore evaluated *there*, where the constraint
    /// definitions live; the ID/IDREF table, which is keyed by string rather
    /// than by component, comes back out and is checked over the document as a
    /// whole.
    pub(crate) fn delegate(&mut self, node: NodeId, declaration: Option<ElementId>) {
        let Some(hints) = self.hints else {
            return;
        };
        let mut nested = Validator::new_nested(hints, self.doc, self.depth);
        nested.assess_element(node, declaration, None);
        nested.check_identity_constraints();
        self.absorb(nested);
    }

    /// Takes over everything a nested assessment produced.
    pub(crate) fn absorb(&mut self, nested: Validator<'_>) {
        let Validator {
            errors,
            elements,
            attributes,
            defaulted,
            idc_defaults,
            ids,
            idrefs,
            unbound_ids,
            skipped,
            ..
        } = nested;
        self.errors.extend(errors);
        // A component identifier indexes the arena of the schema that handed it
        // out, so the ones a hinted schema produced are dropped rather than
        // published against this one, where they would name unrelated
        // components or nothing at all. What the node *is* — its validity, its
        // normalized value, the value that denotes — carries over unchanged.
        for (node, mut psvi) in elements {
            psvi.declaration = None;
            psvi.type_definition = None;
            psvi.member_type = None;
            self.elements.insert(node, psvi);
        }
        for (node, mut psvi) in attributes {
            psvi.declaration = None;
            psvi.type_definition = None;
            psvi.member_type = None;
            self.attributes.insert(node, psvi);
        }
        for (node, entries) in defaulted {
            self.defaulted.entry(node).or_default().extend(entries);
        }
        for (node, entries) in idc_defaults {
            self.idc_defaults.entry(node).or_default().extend(entries);
        }
        for (value, owners) in ids {
            self.ids.entry(value).or_default().extend(owners);
        }
        self.idrefs.extend(idrefs);
        self.unbound_ids.extend(unbound_ids);
        self.skipped.extend(skipped);
    }

    pub(crate) fn error(&mut self, code: &str, path: &str, message: impl Into<String>) {
        self.errors
            .push(ValidationError::new(code, path, message.into()));
    }

    pub(crate) fn path(&self, node: NodeId) -> String {
        node_path(self.doc, node)
    }

    fn finish(mut self) -> ValidationOutcome {
        self.check_identity_constraints();
        self.check_id_references();
        let psvi = Psvi {
            elements: self.elements,
            attributes: self.attributes,
            defaulted_attributes: self.defaulted,
        };
        ValidationOutcome {
            valid: self.errors.is_empty(),
            errors: self.errors,
            psvi,
        }
    }

    fn assess_root(&mut self, root: NodeId) {
        let (namespace, local) = match self.doc.element(root) {
            Some(element) => (
                element.namespace_uri().map(str::to_owned),
                element.local_name().to_owned(),
            ),
            None => return,
        };
        match self.schema.find_element(namespace.as_deref(), &local) {
            Some(declaration) => self.assess_element(root, Some(declaration), None),
            None => {
                // §4.3.2: what the instance's own location hints brought in
                // stands behind the assembled schema, so it gets its turn only
                // now — for the declaration first, and failing that for a type
                // an `xsi:type` names.
                if let Some(declaration) = self.hinted_element(namespace.as_deref(), &local) {
                    self.delegate(root, Some(declaration));
                } else if self.hinted_xsi_type(root) {
                    self.delegate(root, None);
                } else if self.xsi_type(root).is_some() {
                    self.assess_element(root, None, None);
                } else {
                    let path = self.path(root);
                    self.error(
                        "cvc-elt.1",
                        &path,
                        format!(
                            "there is no declaration for the element `{}`",
                            qualified(namespace.as_deref(), &local)
                        ),
                    );
                }
            }
        }
    }

    fn xsi_type(&self, node: NodeId) -> Option<String> {
        self.doc
            .get_attribute_ns(node, Some(XSI_NS), "type")
            .map(str::to_owned)
    }

    /// Validates one element against a declaration, or (under a wildcard)
    /// against a type alone.
    ///
    /// This is the outer half of the walk: it bounds the nesting at
    /// [`MAX_DEPTH`] and, once the element and everything below it has been
    /// assessed, records whether anything went wrong. Any error raised while
    /// this call is in progress — the element's own, an attribute's, a
    /// descendant's — belongs to this element too, which is exactly the
    /// nesting §3.3.4 gives the *\[validity\]* property, and the enclosing
    /// calls carry it on up to the validation root.
    pub(crate) fn assess_element(
        &mut self,
        node: NodeId,
        declaration: Option<ElementId>,
        governing_type: Option<TypeId>,
    ) {
        if self.depth >= MAX_DEPTH {
            let path = self.path(node);
            self.error(
                "cvc-max-depth",
                &path,
                format!("the element nesting exceeds the maximum depth of {MAX_DEPTH} elements"),
            );
            self.mark_invalid(node);
            return;
        }
        let before = self.errors.len();
        self.depth += 1;
        self.assess_element_inner(node, declaration, governing_type);
        self.depth -= 1;
        if self.errors.len() != before {
            self.mark_invalid(node);
        }
    }

    /// Marks `node` — or, for an attribute, the element that carries it — and
    /// every ancestor of it as [`Validity::Invalid`].
    ///
    /// Elements not yet recorded in the PSVI are simply skipped: an element
    /// whose assessment has not reached [`Validator::record_element`] gets its
    /// entry once it does, and this is called again on the way out.
    pub(crate) fn mark_invalid(&mut self, node: NodeId) {
        let start = match self.doc.kind(node) {
            Some(NodeKind::Attr) => match self.doc.owner_element(node) {
                Some(owner) => owner,
                None => return,
            },
            _ => node,
        };
        let chain: Vec<NodeId> = std::iter::once(start)
            .chain(self.doc.ancestors(start))
            .collect();
        for element in chain {
            if let Some(psvi) = self.elements.get_mut(&element) {
                psvi.validity = Validity::Invalid;
            }
        }
    }

    fn assess_element_inner(
        &mut self,
        node: NodeId,
        declaration: Option<ElementId>,
        governing_type: Option<TypeId>,
    ) {
        let path = self.path(node);
        let mut type_id = match (declaration, governing_type) {
            (Some(id), _) => {
                let element = self.schema.element_declaration(id);
                if element.abstract_ {
                    self.error(
                        "cvc-elt.2",
                        &path,
                        format!("the element declaration `{}` is abstract", element.name),
                    );
                }
                element.type_id
            }
            (None, Some(id)) => id,
            (None, None) => self.schema.any_type(),
        };

        // XSD 1.1 §3.3.4 (cvc-type / cvc-assess-elt): when the declaration has
        // a {type table} and the instance does not override the type with
        // `xsi:type`, the *governing type* is the one the first satisfied
        // `xs:alternative` selects.
        if let Some(id) = declaration
            && !self.schema.element_declaration(id).alternatives.is_empty()
            && self.xsi_type(node).is_none()
        {
            type_id = self.conditional_type(node, id, type_id);
        }

        // `xsi:type` overrides the declared type (§3.3.4, cvc-elt.4).
        if let Some(text) = self.xsi_type(node) {
            let context = crate::parse::namespace_context(self.doc, node);
            match resolve_qname_value(&text, &context) {
                Some((namespace, local)) => {
                    match self.schema.find_type(namespace.as_deref(), &local) {
                        Some(override_type) => {
                            if !self.schema.is_derived_from(override_type, type_id) {
                                self.error(
                                    "cvc-elt.4.3",
                                    &path,
                                    format!(
                                        "`xsi:type=\"{text}\"` is not derived from the declared \
                                         type `{}`",
                                        self.schema.type_name(type_id)
                                    ),
                                );
                            } else if let Some(used) =
                                self.derivation_methods(override_type, type_id)
                            {
                                let blocked = self.blocking_set(declaration, type_id);
                                if (used.extension && blocked.extension)
                                    || (used.restriction && blocked.restriction)
                                {
                                    self.error(
                                        "cvc-elt.4.3",
                                        &path,
                                        format!("`xsi:type=\"{text}\"` is blocked"),
                                    );
                                }
                            }
                            type_id = override_type;
                        }
                        None => self.error(
                            "cvc-elt.4.2",
                            &path,
                            format!("`xsi:type=\"{text}\"` names no known type"),
                        ),
                    }
                }
                None => self.error(
                    "cvc-elt.4.1",
                    &path,
                    format!("`xsi:type=\"{text}\"` is not a QName"),
                ),
            }
        }

        if let Some(complex) = self.schema.complex_type(type_id)
            && complex.abstract_
        {
            self.error(
                "cvc-type.2",
                &path,
                format!(
                    "the type `{}` is abstract, so it cannot govern an element",
                    self.schema.type_name(type_id)
                ),
            );
        }

        // `xsi:nil` (§3.3.4, cvc-elt.3).
        let nillable = declaration.is_some_and(|id| self.schema.element_declaration(id).nillable);
        let nil_attribute = self.doc.get_attribute_ns(node, Some(XSI_NS), "nil");
        let mut nil = false;
        if let Some(text) = nil_attribute {
            let text = crate::value::collapse(text);
            if !nillable {
                self.error(
                    "cvc-elt.3.1",
                    &path,
                    "`xsi:nil` is only allowed on a nillable element",
                );
            } else if text == "true" || text == "1" {
                nil = true;
            }
        }

        self.assess_attributes(node, type_id);

        if nil {
            if self.doc.children(node).any(|child| {
                matches!(
                    self.doc.kind(child),
                    Some(NodeKind::Element | NodeKind::Text | NodeKind::CdataSection)
                ) && !self.is_ignorable(child)
            }) {
                self.error("cvc-elt.3.2.1", &path, "a nilled element must be empty");
            }
            if let Some(id) = declaration
                && let Some(constraint) = &self.schema.element_declaration(id).value_constraint
                && constraint.fixed
            {
                self.error(
                    "cvc-elt.3.2.2",
                    &path,
                    "a nilled element may not have a `fixed` value constraint",
                );
            }
            self.record_element(
                node,
                declaration,
                Some(type_id),
                true,
                AssessedContent::default(),
                false,
            );
            return;
        }

        // §3.3.4 (cvc-elt.5.1): an element with no content at all takes the
        // `default` (or `fixed`) value of its declaration as its content.
        let substitute = declaration
            .and_then(|id| self.schema.element_declaration(id).value_constraint.clone())
            .filter(|_| self.is_content_empty(node))
            .map(|constraint| constraint.lexical);
        let from_default = substitute.is_some();
        let content = self.assess_content(node, type_id, &path, substitute.as_deref());
        if let Some(id) = declaration {
            self.check_value_constraint(node, id, type_id, &path, content.normalized.as_deref());
        }
        self.record_element(
            node,
            declaration,
            Some(type_id),
            false,
            content,
            from_default,
        );
        self.check_assertions(node, type_id, &path);
        if declaration.is_some_and(|id| {
            !self
                .schema
                .element_declaration(id)
                .identity_constraints
                .is_empty()
        }) {
            self.idc_nodes.push(node);
        }
    }

    /// The set of derivation methods used to reach `base` from `candidate`,
    /// following a union's {member type definitions} as well as the chain of
    /// {base type definition}s.
    ///
    /// [`Schema::derivation_path`](crate::Schema::derivation_path) walks base
    /// types only, so it answers [`None`] for a type that reaches its base
    /// through a union — and *Type Derivation OK (Simple)* clause 2.2.4 makes
    /// exactly that a valid derivation: a member of a union, or anything
    /// derived from one, is derived from the union itself. The blocking check
    /// of *cvc-elt* clause 4.3 has to see the methods used along that route
    /// too, or `block="restriction"` on an element typed by a union would let
    /// a restriction of a member type through (`msData/element/elemT074`).
    fn derivation_methods(
        &self,
        candidate: TypeId,
        base: TypeId,
    ) -> Option<crate::components::DerivationSet> {
        self.derivation_methods_bounded(candidate, base, self.schema.types.len())
    }

    fn derivation_methods_bounded(
        &self,
        candidate: TypeId,
        base: TypeId,
        budget: usize,
    ) -> Option<crate::components::DerivationSet> {
        if let Some(used) = self.schema.derivation_path(candidate, base) {
            return Some(used);
        }
        if budget == 0 {
            return None;
        }
        // The union step itself is not a *derivation* — the member is simply
        // one of the type's alternatives — so only the methods used to reach
        // the member are collected.
        self.schema
            .union_member_types(base)?
            .into_iter()
            .find_map(|member| self.derivation_methods_bounded(candidate, member, budget - 1))
    }

    fn blocking_set(
        &self,
        declaration: Option<ElementId>,
        type_id: TypeId,
    ) -> crate::components::DerivationSet {
        let from_type = self.schema.type_definition(type_id).block;
        match declaration {
            Some(id) => from_type.union_with(self.schema.element_declaration(id).block),
            None => from_type,
        }
    }

    fn record_element(
        &mut self,
        node: NodeId,
        declaration: Option<ElementId>,
        type_id: Option<TypeId>,
        nil: bool,
        content: AssessedContent,
        from_default: bool,
    ) {
        // An element already known to be invalid — one of its attributes or
        // children failed before this ran — keeps that verdict; the enclosing
        // `assess_element` would restore it anyway, but recording the right
        // answer straight away keeps the map consistent at every moment.
        let validity = match self.elements.get(&node).map(|psvi| psvi.validity) {
            Some(Validity::Invalid) => Validity::Invalid,
            _ if type_id.is_some() => Validity::Valid,
            _ => Validity::NotKnown,
        };
        self.elements.insert(
            node,
            ElementPsvi {
                declaration,
                type_definition: type_id,
                validity,
                nil,
                normalized_value: content.normalized,
                value: content.value,
                member_type: content.member_type,
                from_default,
            },
        );
    }

    /// Whether a child node contributes no content (comments and processing
    /// instructions).
    pub(crate) fn is_ignorable(&self, node: NodeId) -> bool {
        matches!(
            self.doc.kind(node),
            Some(NodeKind::Comment | NodeKind::ProcessingInstruction)
        )
    }

    /// The concatenated character content of an element's direct children.
    pub(crate) fn character_content(&self, node: NodeId) -> String {
        let mut text = String::new();
        for child in self.doc.children(node) {
            if matches!(
                self.doc.kind(child),
                Some(NodeKind::Text | NodeKind::CdataSection)
            ) && let Some(data) = self.doc.character_data(child)
            {
                text.push_str(data);
            }
        }
        text
    }

    /// Whether the element has neither element children nor character data.
    pub(crate) fn is_content_empty(&self, node: NodeId) -> bool {
        !self.doc.children(node).any(|child| {
            matches!(
                self.doc.kind(child),
                Some(NodeKind::Element | NodeKind::Text | NodeKind::CdataSection)
            )
        })
    }

    fn assess_content(
        &mut self,
        node: NodeId,
        type_id: TypeId,
        path: &str,
        substitute: Option<&str>,
    ) -> AssessedContent {
        match &self.schema.type_definition(type_id).kind {
            TypeKind::Simple(_) => {
                self.reject_element_children(node, path, "cvc-type.3.1.2");
                let text = match substitute {
                    Some(text) => text.to_owned(),
                    None => self.character_content(node),
                };
                self.validate_simple_content(node, type_id, &text, path)
            }
            TypeKind::Complex(complex) => match complex.content.clone() {
                ContentType::Empty => {
                    // §3.4.4.2 clause 1.1: "If `T . {content type} . {variety}
                    // = empty`, then E has **no character** or element
                    // information item [children]." Unlike element-only
                    // content — clause 1.3, which does tolerate white space —
                    // empty content admits no character data whatever
                    // (`saxonData/Open/open012`).
                    if !self.character_content(node).is_empty() {
                        self.error(
                            "cvc-complex-type.2.1",
                            path,
                            "character content is not allowed by this type",
                        );
                    }
                    self.match_children(node, type_id, path);
                    AssessedContent::default()
                }
                ContentType::Simple(inner) => {
                    self.reject_element_children(node, path, "cvc-complex-type.2.2");
                    let text = match substitute {
                        Some(text) => text.to_owned(),
                        None => self.character_content(node),
                    };
                    self.validate_simple_content(node, inner, &text, path)
                }
                ContentType::ElementOnly(_) => {
                    let text = self.character_content(node);
                    if !text.trim_matches(is_xml_space).is_empty() {
                        self.error(
                            "cvc-complex-type.2.3",
                            path,
                            "character content is not allowed by an element-only content type",
                        );
                    }
                    self.match_children(node, type_id, path);
                    AssessedContent::default()
                }
                ContentType::Mixed(_) => {
                    self.match_children(node, type_id, path);
                    AssessedContent::default()
                }
            },
        }
    }

    fn reject_element_children(&mut self, node: NodeId, path: &str, code: &str) {
        let has_elements = self
            .doc
            .children(node)
            .any(|child| self.doc.kind(child) == Some(NodeKind::Element));
        if has_elements {
            self.error(code, path, "element children are not allowed by this type");
        }
    }

    pub(crate) fn validate_simple_content(
        &mut self,
        node: NodeId,
        type_id: TypeId,
        text: &str,
        path: &str,
    ) -> AssessedContent {
        let context = crate::parse::namespace_context(self.doc, node);
        match simple::validate(self.schema, type_id, text, &context) {
            Ok(result) => {
                // §3.17.5.2: an `ID` written as an element's content binds the
                // element's *parent*, not the element itself.
                let binding = self
                    .doc
                    .parent(node)
                    .filter(|parent| self.doc.kind(*parent) == Some(NodeKind::Element));
                self.note_identifier(type_id, &result.normalized, binding, node, path);
                AssessedContent {
                    normalized: Some(result.normalized),
                    value: Some(result.value),
                    member_type: result.member_type,
                }
            }
            Err(violation) => {
                self.error(violation.code, path, violation.message);
                AssessedContent {
                    normalized: Some(text.to_owned()),
                    value: None,
                    member_type: None,
                }
            }
        }
    }

    fn check_value_constraint(
        &mut self,
        node: NodeId,
        declaration: ElementId,
        type_id: TypeId,
        path: &str,
        normalized: Option<&str>,
    ) {
        let Some(constraint) = self
            .schema
            .element_declaration(declaration)
            .value_constraint
            .clone()
        else {
            return;
        };
        if self.is_content_empty(node) {
            // The default (or fixed) value already stood in for the missing
            // content in `assess_content`.
            return;
        }
        if !constraint.fixed {
            return;
        }
        // §3.3.4 *cvc-elt* clause 5.2.2.1: when the declaration's {value
        // constraint} is `fixed` and the item has content of its own, "the
        // element information item must have **no element information item**
        // \[children\]" — a fixed value fixes the whole content, so a child
        // element can never be part of it, whatever the content type allows
        // (`sunData/ElemDecl/valueConstraint/valueConstraint00701m1`,
        // `msData/complexType/ctZ013c`, `ctZ013d`).
        if self
            .doc
            .children(node)
            .any(|child| self.doc.kind(child) == Some(NodeKind::Element))
        {
            self.error(
                "cvc-elt.5.2.2.1",
                path,
                format!(
                    "an element with the `fixed` value `{}` may have no element children",
                    constraint.lexical
                ),
            );
            return;
        }
        let context = crate::parse::namespace_context(self.doc, node);
        match self.simple_content_type(type_id) {
            // Clause 5.2.2.2.2: with a simple content type the comparison is
            // in the *value* space, so `01` matches a fixed `1` of type
            // `xs:int`.
            Some(simple_id) => {
                let Some(actual) = normalized else { return };
                let matches = match (
                    simple::validate(self.schema, simple_id, actual, &context),
                    simple::validate(
                        self.schema,
                        simple_id,
                        &constraint.lexical,
                        &constraint.namespaces,
                    ),
                ) {
                    (Ok(left), Ok(right)) => left.value == right.value,
                    _ => false,
                };
                if !matches {
                    self.error(
                        "cvc-elt.5.2.2.2.2",
                        path,
                        format!(
                            "the content must equal the fixed value `{}`",
                            constraint.lexical
                        ),
                    );
                }
            }
            // Clause 5.2.2.2.1: with a *mixed* content type there is no value
            // space to compare in — the rule asks that the ·initial value· of
            // the item, its character content exactly as written, match the
            // canonical lexical representation of the {value constraint}
            // value, which for mixed content is the string itself
            // (`sunData/ElemDecl/valueConstraint/valueConstraint00801m1`,
            // `msData/additional/isdefault070`, `isdefault076`).
            None => {
                if !self.is_mixed_content(type_id) {
                    return;
                }
                if self.character_content(node) != constraint.lexical {
                    self.error(
                        "cvc-elt.5.2.2.2.1",
                        path,
                        format!(
                            "the content must equal the fixed value `{}`",
                            constraint.lexical
                        ),
                    );
                }
            }
        }
    }

    /// Whether the type's {content type} is *mixed* — the case clause
    /// 5.2.2.2.1 of *cvc-elt* singles out.
    fn is_mixed_content(&self, type_id: TypeId) -> bool {
        self.schema
            .type_definition(type_id)
            .as_complex()
            .is_some_and(|complex| matches!(complex.content, ContentType::Mixed(_)))
    }

    pub(crate) fn simple_content_type(&self, type_id: TypeId) -> Option<TypeId> {
        match &self.schema.type_definition(type_id).kind {
            TypeKind::Simple(_) => Some(type_id),
            TypeKind::Complex(complex) => match &complex.content {
                ContentType::Simple(inner) => Some(*inner),
                _ => None,
            },
        }
    }

    /// Finds the declaration a wildcard match should use.
    pub(crate) fn wildcard_declaration(
        &mut self,
        node: NodeId,
        process: ProcessContents,
        path: &str,
    ) -> Option<(Option<ElementId>, bool)> {
        let (namespace, local) = match self.doc.element(node) {
            Some(element) => (
                element.namespace_uri().map(str::to_owned),
                element.local_name().to_owned(),
            ),
            None => return None,
        };
        match process {
            ProcessContents::Skip => None,
            ProcessContents::Lax | ProcessContents::Strict => {
                if let Some(id) = self.schema.find_element(namespace.as_deref(), &local) {
                    return Some((Some(id), true));
                }
                // §4.3.2 again: a declaration the instance's own hints supplied
                // is a declaration, so a `lax` wildcard uses it and a `strict`
                // one is satisfied by it. The subtree goes over to the hinted
                // schema whole — `None` tells the caller there is nothing left
                // for it to assess.
                if let Some(id) = self.hinted_element(namespace.as_deref(), &local) {
                    self.delegate(node, Some(id));
                    return None;
                }
                if self.hinted_xsi_type(node) {
                    self.delegate(node, None);
                    return None;
                }
                if process == ProcessContents::Lax || self.xsi_type(node).is_some() {
                    return Some((None, true));
                }
                self.error(
                    "cvc-complex-type.2.4.c",
                    path,
                    format!(
                        "`{}` is matched by a strict wildcard but has no declaration",
                        qualified(namespace.as_deref(), &local)
                    ),
                );
                None
            }
        }
    }
}

/// What assessing an element's character content produced: the three PSVI
/// properties a simple type contributes, kept together so that every path
/// through [`Validator::assess_content`] answers the same question.
#[derive(Debug, Clone, Default)]
pub(crate) struct AssessedContent {
    /// The `whiteSpace`-normalized content, when the type has a value space at
    /// all.
    pub normalized: Option<String>,
    /// The value it denotes.
    pub value: Option<Value>,
    /// The union member type that accepted it, when the type is a union.
    pub member_type: Option<TypeId>,
}

pub(crate) fn is_xml_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

pub(crate) fn qualified(namespace: Option<&str>, local: &str) -> String {
    match namespace {
        Some(namespace) => format!("{{{namespace}}}{local}"),
        None => local.to_owned(),
    }
}

pub(crate) fn resolve_qname_value(
    text: &str,
    context: &NamespaceContext,
) -> Option<(Option<String>, String)> {
    let text = crate::value::collapse(text);
    match text.split_once(':') {
        Some((prefix, local)) => {
            if !crate::value::is_xml_ncname(prefix) || !crate::value::is_xml_ncname(local) {
                return None;
            }
            let namespace = context.lookup(Some(prefix))?;
            Some((Some(namespace.to_owned()), local.to_owned()))
        }
        None => {
            if !crate::value::is_xml_ncname(&text) {
                return None;
            }
            Some((context.lookup(None).map(str::to_owned), text))
        }
    }
}
