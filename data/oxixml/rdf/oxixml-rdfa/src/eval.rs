//! The RDFa 1.1 processing model.
//!
//! This is a direct transcription of [RDFa Core 1.1 §7.5
//! *Sequence*](https://www.w3.org/TR/rdfa-core/#s_sequence), with the extensions of
//! [HTML+RDFa 1.1 §3](https://www.w3.org/TR/html-rdfa/#extensions-to-rdfa-core-1-1) applied when
//! the host language is one of the HTML serializations. The fourteen numbered steps of the
//! specification appear as `// Step N` comments in [`Processor::process_element`], in order.
//!
//! # Deliberate deviations
//!
//! * **Step 13, the skipped-element branch.** RDFa Core says the child context of a skipped
//!   element is "a copy of the current context … with the language and list of IRI mappings values
//!   replaced with the local values", which would throw away a `@vocab` set on that very element —
//!   and an element carrying nothing but `@vocab` *is* skipped, which would make the canonical
//!   RDFa Lite idiom `<div vocab="…"><span property="name">…</span></div>` produce nothing. The
//!   local default vocabulary, term mappings and base are therefore carried over as well, matching
//!   every conforming processor the W3C test suite reports on.
//! * **`@href` and `@src` in the bare-XML host language.** RDFa Core leaves both to the host
//!   language, and generic XML defines neither; they are honoured anyway, because the XML
//!   documents of the W3C RDFa Test Suite are generated from the same sources as the HTML ones and
//!   expect them to be.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::str::FromStr as _;

use oxixml_dom::{Document, NodeId, NodeKind};
use oxixml_iri::Iri;
use oxixml_model::vocab::{rdf, xsd};
use oxixml_model::{BlankNode, Literal, NamedNode, NamedOrBlankNode, Term, Triple};

use crate::attrs::{RdfaAttributes, parse_prefix_attribute};
use crate::context::XHTML_VOCABULARY;
use crate::curie::{
    Prefixes, Reference, Terms, expand_term, initial_prefixes, initial_terms, is_term, resolve_iri,
    safe_curie_or_curie_or_iri, split_values, term_or_curie_or_abs_iri,
};
use crate::error::RdfaSyntaxError;
use crate::host::HostLanguage;
use crate::literal;

/// `http://www.w3.org/ns/rdfa#usesVocabulary`.
const USES_VOCABULARY: &str = "http://www.w3.org/ns/rdfa#usesVocabulary";
/// `http://www.w3.org/ns/rdfa#copy`.
const RDFA_COPY: &str = "http://www.w3.org/ns/rdfa#copy";
/// `http://www.w3.org/ns/rdfa#Pattern`.
const RDFA_PATTERN: &str = "http://www.w3.org/ns/rdfa#Pattern";
/// `http://www.w3.org/1999/xhtml/vocab#role`.
const XHTML_ROLE: &str = "http://www.w3.org/1999/xhtml/vocab#role";

/// A list under construction, shared between the element that instantiated it and every descendant
/// that adds to it.
type List = Rc<RefCell<Vec<Term>>>;

/// The *list mapping* of the evaluation context: an insertion-ordered association from a predicate
/// IRI to the list being collected for it.
///
/// It is deliberately shared by `Rc`: [RDFa Core 1.1 §7.5 step
/// 1](https://www.w3.org/TR/rdfa-core/#s_sequence) says the local list mapping is set to "(a
/// reference of)" the one from the evaluation context, and step 14 emits a list only at the
/// element that instantiated the mapping — which pointer identity detects exactly.
type ListMapping = Rc<RefCell<ListMappingData>>;

/// The contents of a list mapping: the subject the lists belong to, and the lists themselves.
///
/// RDFa Core keys the mapping on the predicate IRI alone, which conflates two `@inlist`
/// collections that share a predicate but not a subject; the W3C RDFa Test Suite's "confusion
/// between multiple implicit collections" cases assert that they stay apart, so the subject the
/// mapping was instantiated for is recorded and a change of subject starts a new mapping.
#[derive(Debug, Default)]
struct ListMappingData {
    owner: Option<NamedOrBlankNode>,
    entries: Vec<(NamedNode, List)>,
}

/// One entry of the *list of incomplete triples*.
#[derive(Debug, Clone)]
enum Incomplete {
    /// `@rel`: the parent subject becomes the subject and the new subject the object.
    Forward(NamedNode),
    /// `@rev`: the new subject becomes the subject and the parent subject the object.
    Reverse(NamedNode),
    /// `@rel` together with `@inlist`: the new subject is appended to a list.
    List(List),
}

/// The evaluation context of [RDFa Core 1.1 §7.2](https://www.w3.org/TR/rdfa-core/#s_model).
#[derive(Debug, Clone)]
struct EvalContext {
    base: Iri<String>,
    parent_subject: NamedOrBlankNode,
    parent_object: Option<NamedOrBlankNode>,
    incomplete: Rc<Vec<Incomplete>>,
    list_mapping: ListMapping,
    language: Option<String>,
    prefixes: Rc<Prefixes>,
    /// Only the prefixes the *document* declared, with `@prefix` or `xmlns:`, which an
    /// `rdf:XMLLiteral` has to reproduce as namespace declarations.
    declared: Rc<BTreeMap<String, String>>,
    terms: Rc<Terms>,
    vocabulary: Option<String>,
    /// How many elements deep `Processor::process_element` has recursed to reach this context:
    /// `0` at the document element, incremented by one per recursive call. Bounded by
    /// `Processor`'s own `max_depth`.
    depth: usize,
}

/// The RDFa processor: a document, a host language, and the graph being built.
pub(crate) struct Processor<'a> {
    document: &'a Document,
    host: HostLanguage,
    /// The IRI of the document itself, which relative CURIE expansions resolve against and which
    /// neither `<base href>` nor `@xml:base` changes.
    document_iri: Iri<String>,
    root: NodeId,
    triples: Vec<Triple>,
    seen: HashSet<Triple>,
    counter: u64,
    named_bnodes: HashMap<String, BlankNode>,
    /// The nesting-depth ceiling; see `EvalContext`'s own `depth`. Set from
    /// [`RdfaParser::max_depth`](crate::RdfaParser::max_depth) by [`Self::extract`]'s caller.
    max_depth: usize,
    /// Set once [`Self::process_element`] or an `rdf:XMLLiteral`/`rdf:HTML` literal serialization
    /// refuses to descend past `max_depth`, so [`Self::extract`] can report it instead of
    /// returning a graph silently missing whatever that subtree would have contributed.
    depth_exceeded: bool,
}

impl<'a> Processor<'a> {
    /// Runs the whole processing model over `document` and returns the output graph, in the order
    /// the triples were generated and with duplicates removed.
    ///
    /// # Errors
    ///
    /// Returns [`RdfaSyntaxError`] if `document` (or an `rdf:XMLLiteral`/`rdf:HTML` literal inside
    /// it) is nested more than `max_depth` elements deep.
    pub(crate) fn extract(
        document: &'a Document,
        host: HostLanguage,
        base: Iri<String>,
        max_depth: usize,
    ) -> Result<Vec<Triple>, RdfaSyntaxError> {
        let Some(root) = document.document_element() else {
            return Ok(Vec::new());
        };
        let mut processor = Self {
            document,
            host,
            document_iri: base.clone(),
            root,
            triples: Vec::new(),
            seen: HashSet::new(),
            counter: 0,
            named_bnodes: HashMap::new(),
            max_depth,
            depth_exceeded: false,
        };
        let base = processor.document_base(base);
        let context = EvalContext {
            parent_subject: NamedNode::new_unchecked(base.as_str().to_owned()).into(),
            base,
            parent_object: None,
            incomplete: Rc::new(Vec::new()),
            list_mapping: Rc::new(RefCell::new(ListMappingData::default())),
            language: None,
            prefixes: Rc::new(initial_prefixes(host.initial_prefixes())),
            declared: Rc::new(BTreeMap::new()),
            terms: Rc::new(initial_terms(host.initial_terms())),
            vocabulary: host.default_vocabulary().map(str::to_owned),
            depth: 0,
        };
        processor.process_element(root, &context);
        if host.is_html() {
            processor.copy_patterns();
        }
        if processor.depth_exceeded {
            return Err(RdfaSyntaxError::msg(format!(
                "document nesting exceeds the limit of {max_depth} elements (configurable via \
                 RdfaParser::with_max_depth)"
            )));
        }
        Ok(processor.triples)
    }

    /// Applies the host language's document-level base rule: `<base href="…">` for the HTML
    /// family, which [HTML+RDFa 1.1 §3.1](https://www.w3.org/TR/html-rdfa/) makes authoritative
    /// for the whole document however late in the stream it appears.
    fn document_base(&self, base: Iri<String>) -> Iri<String> {
        if !self.host.uses_base_element() {
            return base;
        }
        let document = self.document;
        for node in document.descendants(self.root) {
            if document.kind(node) != Some(NodeKind::Element) {
                continue;
            }
            let Some(data) = document.element(node) else {
                continue;
            };
            if !data.local_name().eq_ignore_ascii_case("base") {
                continue;
            }
            let Some(href) = document.get_attribute(node, "href") else {
                continue;
            };
            // A fragment on the base reference is dropped: RFC 3986 §5.1 bases have none.
            let href = href.split('#').next().unwrap_or(href);
            if let Some(resolved) = resolve_iri(href, &base) {
                if let Ok(iri) = Iri::parse(resolved) {
                    return iri;
                }
            }
        }
        base
    }

    // ---------------------------------------------------------------- triples

    fn emit(&mut self, subject: NamedOrBlankNode, predicate: NamedNode, object: Term) {
        let triple = Triple::new(subject, predicate, object);
        if self.seen.insert(triple.clone()) {
            self.triples.push(triple);
        }
    }

    fn fresh_bnode(&mut self) -> BlankNode {
        self.counter += 1;
        BlankNode::new_unchecked(format!("b{}", self.counter))
    }

    /// Maps a `_:label` reference onto a blank node, consistently within the document and without
    /// ever colliding with an automatically generated one — every generated label starts with `b`
    /// and every document-supplied one with `e`.
    fn named_bnode(&mut self, label: &str) -> BlankNode {
        if let Some(node) = self.named_bnodes.get(label) {
            return node.clone();
        }
        let mut id = String::from("e");
        for ch in label.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                id.push(ch);
            } else {
                id.push('_');
            }
        }
        let node = BlankNode::new_unchecked(id);
        self.named_bnodes.insert(label.to_owned(), node.clone());
        node
    }

    fn resource(&mut self, reference: Reference) -> Option<NamedOrBlankNode> {
        match reference {
            Reference::Iri(iri) => NamedNode::new(iri).ok().map(Into::into),
            Reference::Blank(label) => Some(self.named_bnode(&label).into()),
        }
    }

    // ------------------------------------------------------------ resolution

    fn about(&mut self, value: &str, context: &EvalContext) -> Option<NamedOrBlankNode> {
        let reference = safe_curie_or_curie_or_iri(
            value,
            &context.prefixes,
            &context.base,
            &self.document_iri,
        )?;
        self.resource(reference)
    }

    fn plain_iri(&mut self, value: &str, context: &EvalContext) -> Option<NamedOrBlankNode> {
        let iri = resolve_iri(value, &context.base)?;
        NamedNode::new(iri).ok().map(Into::into)
    }

    /// Expands a whitespace-separated `TERMorCURIEorAbsIRIs` attribute into predicates, dropping
    /// values that name a blank node — a blank node cannot be a predicate.
    fn predicates(&self, value: &str, context: &EvalContext) -> Vec<NamedNode> {
        split_values(value)
            .filter_map(|item| {
                match term_or_curie_or_abs_iri(
                    item,
                    &context.prefixes,
                    &context.terms,
                    context.vocabulary.as_deref(),
                    &self.document_iri,
                )? {
                    Reference::Iri(iri) => NamedNode::new(iri).ok(),
                    Reference::Blank(_) => None,
                }
            })
            .collect()
    }

    fn types(&mut self, value: &str, context: &EvalContext) -> Vec<NamedOrBlankNode> {
        let references: Vec<Reference> = split_values(value)
            .filter_map(|item| {
                term_or_curie_or_abs_iri(
                    item,
                    &context.prefixes,
                    &context.terms,
                    context.vocabulary.as_deref(),
                    &self.document_iri,
                )
            })
            .collect();
        references
            .into_iter()
            .filter_map(|reference| self.resource(reference))
            .collect()
    }

    // ------------------------------------------------------------ the sequence

    #[expect(
        clippy::too_many_lines,
        reason = "the fourteen steps of RDFa Core §7.5 read as one sequence"
    )]
    fn process_element(&mut self, element: NodeId, context: &EvalContext) {
        // A depth cap, not a correctness rule: every other check in this function assumes the
        // element is worth processing at all, so this must be the first thing that runs, before
        // recursing one native stack frame deeper would even be possible.
        if context.depth >= self.max_depth {
            self.depth_exceeded = true;
            return;
        }

        let document = self.document;
        let attributes = RdfaAttributes::collect(document, element);
        let is_root = element == self.root;

        // Step 1 — initialize the local values.
        let mut skip = false;
        let mut new_subject: Option<NamedOrBlankNode> = None;
        let mut current_object_resource: Option<NamedOrBlankNode> = None;
        let mut typed_resource: Option<NamedOrBlankNode> = None;
        let mut local_prefixes = Rc::clone(&context.prefixes);
        let mut local_declared = Rc::clone(&context.declared);
        let mut local_incomplete: Vec<Incomplete> = Vec::new();
        let mut list_mapping = Rc::clone(&context.list_mapping);
        let mut language = context.language.clone();
        let local_terms = Rc::clone(&context.terms);
        let mut vocabulary = context.vocabulary.clone();
        let mut base = context.base.clone();

        // A host-language extension rather than a numbered step: `@xml:base` re-bases the subtree
        // in XML and XHTML, and is ignored in `text/html`.
        if self.host.uses_xml_base() {
            if let Some(value) = attributes.xml_base {
                if let Some(resolved) = resolve_iri(value, &base) {
                    if let Ok(iri) = Iri::parse(resolved) {
                        base = iri;
                    }
                }
            }
        }

        // Step 2 — @vocab.
        if let Some(value) = attributes.vocab {
            if value.trim().is_empty() {
                vocabulary = self.host.default_vocabulary().map(str::to_owned);
            } else if let Some(resolved) = resolve_iri(value.trim(), &base) {
                vocabulary = Some(resolved.clone());
                if let Ok(object) = NamedNode::new(resolved) {
                    let subject = NamedNode::new_unchecked(base.as_str().to_owned());
                    self.emit(
                        subject.into(),
                        NamedNode::new_unchecked(USES_VOCABULARY),
                        object.into(),
                    );
                }
            }
        }

        // Step 3 — prefix mappings: the legacy `xmlns:` ones first, then `@prefix`.
        if !attributes.xmlns.is_empty() || attributes.prefix.is_some() {
            let mut prefixes = (*local_prefixes).clone();
            let mut declared = (*local_declared).clone();
            for (prefix, iri) in &attributes.xmlns {
                if *prefix != "_" {
                    prefixes.insert(prefix.to_ascii_lowercase(), (*iri).to_owned());
                    declared.insert(prefix.to_ascii_lowercase(), (*iri).to_owned());
                }
            }
            if let Some(value) = attributes.prefix {
                for (prefix, iri) in parse_prefix_attribute(value) {
                    if prefix != "_" {
                        prefixes.insert(prefix.clone(), iri.clone());
                        declared.insert(prefix, iri);
                    }
                }
            }
            local_prefixes = Rc::new(prefixes);
            local_declared = Rc::new(declared);
        }

        // Step 4 — the natural language of the element. `@xml:lang` wins over `@lang`.
        let declared_language = match (attributes.xml_lang, attributes.lang) {
            (Some(value), _) => Some(value),
            (None, Some(value)) if self.host.uses_plain_lang() => Some(value),
            _ => None,
        };
        if let Some(value) = declared_language {
            let value = value.trim();
            language = if value.is_empty() {
                None
            } else {
                Some(value.to_owned())
            };
        }

        // A scoped context, so that everything below resolves against this element's own prefixes,
        // terms, vocabulary and base. Still `context`'s own depth: this context evaluates `element`
        // itself, one recursive `process_element` call, not a descent into its children.
        let local = EvalContext {
            base: base.clone(),
            parent_subject: context.parent_subject.clone(),
            parent_object: context.parent_object.clone(),
            incomplete: Rc::clone(&context.incomplete),
            list_mapping: Rc::clone(&list_mapping),
            language: language.clone(),
            prefixes: Rc::clone(&local_prefixes),
            declared: Rc::clone(&local_declared),
            terms: Rc::clone(&local_terms),
            vocabulary: vocabulary.clone(),
            depth: context.depth,
        };

        // The `@role` attribute of [Role Attribute 1.0](https://www.w3.org/TR/role-attribute/),
        // which is independent of the RDFa sequence but shares its CURIE machinery.
        self.process_role(&attributes, &local);

        // HTML+RDFa 1.1 §3.1: when `@property` shares an element with `@rel` or `@rev`, the bare
        // terms in those attributes are dropped, and an attribute so emptied counts as absent.
        let filter_terms = attributes.property.is_some();
        let rel = filtered_relation(attributes.rel, filter_terms);
        let rev = filtered_relation(attributes.rev, filter_terms);

        let about = attributes.about.and_then(|value| self.about(value, &local));
        let resource = attributes
            .resource
            .and_then(|value| self.about(value, &local));
        let href = attributes
            .href
            .and_then(|value| self.plain_iri(value, &local));
        let src = attributes
            .src
            .and_then(|value| self.plain_iri(value, &local));
        let base_subject: NamedOrBlankNode =
            NamedNode::new_unchecked(base.as_str().to_owned()).into();
        let is_head_or_body = self.is_head_or_body(element);

        if rel.is_none() && rev.is_none() {
            // Step 5 — the element has neither @rel nor @rev.
            if attributes.property.is_some()
                && attributes.content.is_none()
                && attributes.datatype.is_none()
            {
                // Step 5.1 — @property without @content or @datatype.
                new_subject = about
                    .clone()
                    .or_else(|| is_root.then(|| base_subject.clone()))
                    .or_else(|| context.parent_object.clone());
                if attributes.type_of.is_some() {
                    typed_resource = about
                        .clone()
                        .or_else(|| is_root.then(|| base_subject.clone()))
                        .or_else(|| resource.clone())
                        .or_else(|| href.clone())
                        .or_else(|| src.clone())
                        .or_else(|| Some(self.fresh_bnode().into()));
                    current_object_resource = typed_resource.clone();
                }
            } else {
                // Step 5.2.
                let provided = about
                    .clone()
                    .or_else(|| resource.clone())
                    .or_else(|| href.clone())
                    .or_else(|| src.clone());
                if let Some(provided) = provided {
                    new_subject = Some(provided);
                } else if is_root {
                    new_subject = Some(base_subject.clone());
                } else if is_head_or_body && context.parent_object.is_some() {
                    // HTML+RDFa 1.1 §3.1: `<head>` and `<body>` never introduce a blank node.
                    new_subject = context.parent_object.clone();
                } else if attributes.type_of.is_some() {
                    new_subject = Some(self.fresh_bnode().into());
                } else if let Some(parent_object) = context.parent_object.clone() {
                    new_subject = Some(parent_object);
                    if attributes.property.is_none() {
                        skip = true;
                    }
                }
                if attributes.type_of.is_some() {
                    typed_resource = new_subject.clone();
                }
            }
        } else {
            // Step 6 — the element has @rel or @rev.
            if let Some(about) = about.clone() {
                new_subject = Some(about);
                if attributes.type_of.is_some() {
                    typed_resource = new_subject.clone();
                }
            } else if is_root {
                new_subject = Some(base_subject.clone());
            } else if is_head_or_body && context.parent_object.is_some() {
                new_subject = context.parent_object.clone();
            } else if let Some(parent_object) = context.parent_object.clone() {
                new_subject = Some(parent_object);
            }
            current_object_resource = resource.clone().or_else(|| href.clone()).or_else(|| {
                src.clone().or_else(|| {
                    if attributes.type_of.is_some() && attributes.about.is_none() {
                        Some(self.fresh_bnode().into())
                    } else {
                        None
                    }
                })
            });
            if attributes.type_of.is_some() && attributes.about.is_none() {
                typed_resource = current_object_resource.clone();
            }
        }

        // Step 7 — @typeof generates rdf:type triples on the typed resource.
        if let Some(typed) = typed_resource.clone() {
            if let Some(value) = attributes.type_of {
                for class in self.types(value, &local) {
                    self.emit(typed.clone(), rdf::TYPE.into_owned(), class.into());
                }
            }
        }

        // Step 8 — a new subject different from the parent object starts a fresh list mapping.
        if let Some(subject) = &new_subject {
            let owned_by_someone_else = list_mapping.borrow().owner.as_ref() != Some(subject);
            if owned_by_someone_else {
                list_mapping = Rc::new(RefCell::new(ListMappingData {
                    owner: Some(subject.clone()),
                    entries: Vec::new(),
                }));
            }
        }

        let subject_for_lists = new_subject
            .clone()
            .unwrap_or_else(|| context.parent_subject.clone());

        // Step 9 — a resource object completes the @rel and @rev triples.
        if let Some(object) = current_object_resource.clone() {
            if let Some(rel) = rel.as_deref() {
                if attributes.inlist {
                    for predicate in self.predicates(rel, &local) {
                        list_for(&list_mapping, &predicate)
                            .borrow_mut()
                            .push(object.clone().into());
                    }
                } else {
                    let subject = subject_for_lists.clone();
                    for predicate in self.predicates(rel, &local) {
                        self.emit(subject.clone(), predicate, object.clone().into());
                    }
                }
            }
            if let Some(rev) = rev.as_deref() {
                let target = subject_for_lists.clone();
                for predicate in self.predicates(rev, &local) {
                    self.emit(object.clone(), predicate, target.clone().into());
                }
            }
        } else if rel.is_some() || rev.is_some() {
            // Step 10 — no resource object yet: remember the predicates as incomplete triples.
            if let Some(rel) = rel.as_deref() {
                for predicate in self.predicates(rel, &local) {
                    if attributes.inlist {
                        local_incomplete
                            .push(Incomplete::List(list_for(&list_mapping, &predicate)));
                    } else {
                        local_incomplete.push(Incomplete::Forward(predicate));
                    }
                }
            }
            if let Some(rev) = rev.as_deref() {
                for predicate in self.predicates(rev, &local) {
                    local_incomplete.push(Incomplete::Reverse(predicate));
                }
            }
            if !local_incomplete.is_empty() && !attributes.inlist {
                current_object_resource = Some(self.fresh_bnode().into());
            }
        }

        // Step 11 — the current property value.
        if let Some(value) = attributes.property {
            let predicates = self.predicates(value, &local);
            if !predicates.is_empty() {
                let object = self.property_value(
                    element,
                    &attributes,
                    &local,
                    language.as_deref(),
                    typed_resource.as_ref(),
                    rel.is_some() || rev.is_some(),
                );
                if let Some(object) = object {
                    let subject = subject_for_lists.clone();
                    for predicate in predicates {
                        if attributes.inlist {
                            list_for(&list_mapping, &predicate)
                                .borrow_mut()
                                .push(object.clone());
                        } else {
                            self.emit(subject.clone(), predicate, object.clone());
                        }
                    }
                }
            }
        }

        // Step 12 — complete the incomplete triples inherited from the parent.
        if !skip {
            if let Some(subject) = new_subject.clone() {
                let incomplete = Rc::clone(&context.incomplete);
                for entry in incomplete.iter() {
                    match entry {
                        Incomplete::Forward(predicate) => self.emit(
                            context.parent_subject.clone(),
                            predicate.clone(),
                            subject.clone().into(),
                        ),
                        Incomplete::Reverse(predicate) => self.emit(
                            subject.clone(),
                            predicate.clone(),
                            context.parent_subject.clone().into(),
                        ),
                        Incomplete::List(list) => list.borrow_mut().push(subject.clone().into()),
                    }
                }
            }
        }

        // Step 13 — recurse into the element children.
        let child_context = if skip {
            EvalContext {
                base,
                language,
                prefixes: local_prefixes,
                declared: local_declared,
                terms: local_terms,
                vocabulary,
                depth: context.depth + 1,
                ..context.clone()
            }
        } else {
            EvalContext {
                base,
                parent_subject: new_subject
                    .clone()
                    .unwrap_or_else(|| context.parent_subject.clone()),
                parent_object: current_object_resource
                    .clone()
                    .or_else(|| new_subject.clone())
                    .or_else(|| Some(context.parent_subject.clone())),
                incomplete: Rc::new(local_incomplete),
                list_mapping: Rc::clone(&list_mapping),
                language,
                prefixes: local_prefixes,
                declared: local_declared,
                terms: local_terms,
                vocabulary,
                depth: context.depth + 1,
            }
        };
        let children: Vec<NodeId> = document.children(element).collect();
        for child in children {
            if document.kind(child) == Some(NodeKind::Element) {
                self.process_element(child, &child_context);
            }
        }

        // Step 14 — emit the lists this element instantiated.
        if !Rc::ptr_eq(&list_mapping, &context.list_mapping) {
            let entries = list_mapping.borrow().entries.clone();
            for (predicate, list) in entries {
                let items = list.borrow().clone();
                self.emit_list(&subject_for_lists, &predicate, &items);
            }
        }
    }

    /// Returns `true` when the element is the `head` or `body` of an HTML-family document, which
    /// [HTML+RDFa 1.1 §3.1](https://www.w3.org/TR/html-rdfa/) forbids from introducing a subject
    /// of their own.
    fn is_head_or_body(&self, element: NodeId) -> bool {
        self.document.element(element).is_some_and(|data| {
            data.local_name().eq_ignore_ascii_case("head")
                || data.local_name().eq_ignore_ascii_case("body")
        })
    }

    /// [RDFa Core 1.1 §7.5 step 14](https://www.w3.org/TR/rdfa-core/#s_sequence): turns a
    /// collected list into `rdf:first`/`rdf:rest` triples.
    fn emit_list(&mut self, subject: &NamedOrBlankNode, predicate: &NamedNode, items: &[Term]) {
        if items.is_empty() {
            self.emit(subject.clone(), predicate.clone(), rdf::NIL.into());
            return;
        }
        let nodes: Vec<BlankNode> = (0..items.len()).map(|_| self.fresh_bnode()).collect();
        for (index, item) in items.iter().enumerate() {
            let node: NamedOrBlankNode = nodes[index].clone().into();
            self.emit(node.clone(), rdf::FIRST.into_owned(), item.clone());
            let rest: Term = nodes
                .get(index + 1)
                .map_or_else(|| rdf::NIL.into(), |next| Term::from(next.clone()));
            self.emit(node, rdf::REST.into_owned(), rest);
        }
        self.emit(
            subject.clone(),
            predicate.clone(),
            Term::from(nodes[0].clone()),
        );
    }

    /// [Role Attribute 1.0](https://www.w3.org/TR/role-attribute/): every `@role` value becomes an
    /// `xhv:role` triple whose subject is the element's `@id` as a fragment of the base, or a
    /// fresh blank node when the element has no `@id`. Terms in `@role` always resolve against the
    /// XHTML vocabulary, never against `@vocab`.
    fn process_role(&mut self, attributes: &RdfaAttributes<'_>, context: &EvalContext) {
        let Some(value) = attributes.role else {
            return;
        };
        let references: Vec<Reference> = split_values(value)
            .filter_map(|item| {
                if is_term(item) {
                    expand_term(item, &context.terms, Some(XHTML_VOCABULARY)).map(Reference::Iri)
                } else {
                    term_or_curie_or_abs_iri(
                        item,
                        &context.prefixes,
                        &context.terms,
                        None,
                        &self.document_iri,
                    )
                }
            })
            .collect();
        if references.is_empty() {
            return;
        }
        let subject = match attributes.id {
            Some(id) => resolve_iri(&format!("#{id}"), &context.base)
                .and_then(|iri| NamedNode::new(iri).ok())
                .map(NamedOrBlankNode::from),
            None => Some(self.fresh_bnode().into()),
        };
        let Some(subject) = subject else {
            return;
        };
        for reference in references {
            if let Some(object) = self.resource(reference) {
                self.emit(
                    subject.clone(),
                    NamedNode::new_unchecked(XHTML_ROLE),
                    object.into(),
                );
            }
        }
    }

    /// [RDFa Core 1.1 §7.5 step 11](https://www.w3.org/TR/rdfa-core/#s_sequence), with the
    /// `@datetime`, `<time>` and `rdf:HTML` extensions of
    /// [HTML+RDFa 1.1 §3.1](https://www.w3.org/TR/html-rdfa/).
    fn property_value(
        &mut self,
        element: NodeId,
        attributes: &RdfaAttributes<'_>,
        context: &EvalContext,
        language: Option<&str>,
        typed_resource: Option<&NamedOrBlankNode>,
        has_relation: bool,
    ) -> Option<Term> {
        let datetime = self.datetime_value(element, attributes);
        let text = self.document.text_content(element).unwrap_or_default();
        let explicit = || -> String {
            attributes.content.map_or_else(
                || datetime.clone().unwrap_or_else(|| text.clone()),
                str::to_owned,
            )
        };

        if let Some(raw) = attributes.datatype {
            if raw.trim().is_empty() {
                // An empty @datatype forces a plain literal, language and all.
                return Some(plain(explicit(), language));
            }
            let datatype = term_or_curie_or_abs_iri(
                raw.trim(),
                &context.prefixes,
                &context.terms,
                context.vocabulary.as_deref(),
                &self.document_iri,
            );
            let Some(Reference::Iri(iri)) = datatype else {
                // A `@datatype` that resolves to nothing is "empty according to the section on
                // CURIE and IRI Processing", which RDFa Core §7.5 step 11 makes a plain literal.
                return Some(plain(explicit(), language));
            };
            if iri == rdf::XML_LITERAL.as_str() {
                let mut declarations = literal::in_scope_declarations(self.document, element);
                // HTML+RDFa 1.1 §3.4: the prefixes declared with `@prefix` are namespaces too, as
                // far as an `rdf:XMLLiteral` is concerned.
                for (prefix, namespace) in context.declared.iter() {
                    declarations
                        .entry(Some(prefix.clone()))
                        .or_insert_with(|| namespace.clone());
                }
                let Some(value) =
                    literal::xml_literal(self.document, element, &declarations, self.max_depth)
                else {
                    // `element`'s descendants are nested deeper than `max_depth`: reported by
                    // `Self::extract` rather than risking a truncated, mistagged literal value.
                    self.depth_exceeded = true;
                    return None;
                };
                return Some(Literal::new_typed_literal(value, rdf::XML_LITERAL).into());
            }
            if self.host.is_html() && iri == rdf::HTML.as_str() {
                let Some(value) = literal::html_literal(self.document, element, self.max_depth)
                else {
                    self.depth_exceeded = true;
                    return None;
                };
                return Some(Literal::new_typed_literal(value, rdf::HTML).into());
            }
            let datatype = NamedNode::new(iri).ok()?;
            return Some(Literal::new_typed_literal(explicit(), datatype).into());
        }

        if let Some(content) = attributes.content {
            return Some(plain(content.to_owned(), language));
        }

        if let Some(datetime) = datetime {
            return Some(datetime_literal(&datetime, language));
        }

        if !has_relation {
            let resource = attributes
                .resource
                .and_then(|value| self.about(value, context))
                .or_else(|| {
                    attributes
                        .href
                        .and_then(|value| self.plain_iri(value, context))
                })
                .or_else(|| {
                    attributes
                        .src
                        .and_then(|value| self.plain_iri(value, context))
                });
            if let Some(resource) = resource {
                return Some(resource.into());
            }
        }

        if attributes.type_of.is_some() && attributes.about.is_none() {
            if let Some(typed) = typed_resource {
                return Some(typed.clone().into());
            }
        }

        Some(plain(text, language))
    }

    /// The HTML5 `@datetime` value of an element: the attribute itself, or — for a `<time>`
    /// element with neither `@datetime` nor `@content` — the element's text.
    fn datetime_value(&self, element: NodeId, attributes: &RdfaAttributes<'_>) -> Option<String> {
        if !self.host.is_html() {
            return None;
        }
        if let Some(datetime) = attributes.datetime {
            return Some(datetime.to_owned());
        }
        if attributes.content.is_some() {
            return None;
        }
        let data = self.document.element(element)?;
        if !data.local_name().eq_ignore_ascii_case("time") {
            return None;
        }
        Some(self.document.text_content(element).unwrap_or_default())
    }

    /// [HTML+RDFa 1.1 §3.5](https://www.w3.org/TR/html-rdfa/#property-copying): copies every
    /// property of an `rdfa:Pattern` into each resource that `rdfa:copy`s it, to a fixed point,
    /// and then removes the scaffolding.
    fn copy_patterns(&mut self) {
        let copy = NamedNode::new_unchecked(RDFA_COPY);
        let pattern = Term::from(NamedNode::new_unchecked(RDFA_PATTERN));
        loop {
            let patterns = self.pattern_subjects(&pattern);
            let copies: Vec<(NamedOrBlankNode, NamedOrBlankNode)> = self
                .triples
                .iter()
                .filter(|triple| triple.predicate == copy)
                .filter_map(|triple| {
                    object_as_resource(&triple.object)
                        .map(|target| (triple.subject.clone(), target))
                })
                .collect();
            let mut added = Vec::new();
            for (subject, target) in &copies {
                if !patterns.contains(target) {
                    continue;
                }
                for triple in &self.triples {
                    if &triple.subject != target {
                        continue;
                    }
                    if triple.predicate == rdf::TYPE && triple.object == pattern {
                        continue;
                    }
                    added.push(Triple::new(
                        subject.clone(),
                        triple.predicate.clone(),
                        triple.object.clone(),
                    ));
                }
            }
            let mut grew = false;
            for triple in added {
                if self.seen.insert(triple.clone()) {
                    self.triples.push(triple);
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }

        // pattern-clean: drop the `rdfa:copy` links and everything the patterns themselves said.
        let patterns = self.pattern_subjects(&pattern);
        let used: HashSet<NamedOrBlankNode> = self
            .triples
            .iter()
            .filter(|triple| triple.predicate == copy)
            .filter_map(|triple| object_as_resource(&triple.object))
            .filter(|target| patterns.contains(target))
            .collect();
        self.triples.retain(|triple| {
            if triple.predicate == copy {
                if let Some(target) = object_as_resource(&triple.object) {
                    if used.contains(&target) {
                        return false;
                    }
                }
            }
            !used.contains(&triple.subject)
        });
        self.seen = self.triples.iter().cloned().collect();
    }

    fn pattern_subjects(&self, pattern: &Term) -> HashSet<NamedOrBlankNode> {
        self.triples
            .iter()
            .filter(|triple| triple.predicate == rdf::TYPE && &triple.object == pattern)
            .map(|triple| triple.subject.clone())
            .collect()
    }
}

/// Reinterprets a triple object as a resource, which is what `rdfa:copy` points at.
fn object_as_resource(object: &Term) -> Option<NamedOrBlankNode> {
    match object {
        Term::NamedNode(node) => Some(node.clone().into()),
        Term::BlankNode(node) => Some(node.clone().into()),
        _ => None,
    }
}

/// Builds a plain literal, language-tagged when a language is in scope.
fn plain(value: String, language: Option<&str>) -> Term {
    match language {
        Some(language) => {
            match Literal::new_language_tagged_literal(value.clone(), language.to_owned()) {
                Ok(literal) => literal.into(),
                Err(_) => Literal::new_simple_literal(value).into(),
            }
        }
        None => Literal::new_simple_literal(value).into(),
    }
}

/// [HTML+RDFa 1.1 §3.1](https://www.w3.org/TR/html-rdfa/): an `@datetime` value gets the XSD
/// datatype whose lexical space it falls in, and is a plain literal when it falls in none. The
/// order of the tests is the one the specification recommends.
fn datetime_literal(value: &str, language: Option<&str>) -> Term {
    // The lexical value is used exactly as written: RDFa does not normalize it, and a value that
    // only matches an XSD type after trimming does not match it at all.
    let trimmed = value;
    if oxixml_xsd::Duration::from_str(trimmed).is_ok() {
        return Literal::new_typed_literal(trimmed.to_owned(), xsd::DURATION).into();
    }
    if oxixml_xsd::DateTime::from_str(trimmed).is_ok() {
        return Literal::new_typed_literal(trimmed.to_owned(), xsd::DATE_TIME).into();
    }
    if oxixml_xsd::Date::from_str(trimmed).is_ok() {
        return Literal::new_typed_literal(trimmed.to_owned(), xsd::DATE).into();
    }
    if oxixml_xsd::Time::from_str(trimmed).is_ok() {
        return Literal::new_typed_literal(trimmed.to_owned(), xsd::TIME).into();
    }
    if oxixml_xsd::GYearMonth::from_str(trimmed).is_ok() {
        return Literal::new_typed_literal(trimmed.to_owned(), xsd::G_YEAR_MONTH).into();
    }
    if oxixml_xsd::GYear::from_str(trimmed).is_ok() {
        return Literal::new_typed_literal(trimmed.to_owned(), xsd::G_YEAR).into();
    }
    plain(value.to_owned(), language)
}

/// Drops the bare terms from a `@rel` or `@rev` value, and reports the attribute as absent when
/// nothing survives.
fn filtered_relation(value: Option<&str>, filter_terms: bool) -> Option<String> {
    let value = value?;
    if !filter_terms {
        return Some(value.to_owned());
    }
    let kept: Vec<&str> = split_values(value).filter(|item| !is_term(item)).collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join(" "))
    }
}

/// Looks a predicate's list up in the list mapping, instantiating it when it is not there yet.
fn list_for(mapping: &ListMapping, predicate: &NamedNode) -> List {
    let mut data = mapping.borrow_mut();
    if let Some((_, list)) = data.entries.iter().find(|(iri, _)| iri == predicate) {
        return Rc::clone(list);
    }
    let list: List = Rc::new(RefCell::new(Vec::new()));
    data.entries.push((predicate.clone(), Rc::clone(&list)));
    list
}
