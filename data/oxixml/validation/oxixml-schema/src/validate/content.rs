//! Matching an element's children against the effective content model.

use oxixml_dom::{NodeId, NodeKind};

use crate::automaton::{AllModel, ContentModel, Fsm, LeafTerm};
use crate::components::{
    Derivation, DerivationSet, ElementId, MaxOccurs, NamespaceConstraint, OpenContent,
    ProcessContents, TypeId, Wildcard,
};
use crate::schema::Schema;

use super::{Validator, qualified};

/// What a single child was attributed to.
#[derive(Debug, Clone, Copy)]
enum Attribution {
    /// The child matched an element declaration.
    Declared(ElementId),
    /// The child matched a wildcard.
    Wild(ProcessContents),
}

impl<'a> Validator<'a> {
    /// Validates the element children of `node` against the content model of
    /// `type_id`, then assesses each child in turn.
    pub(crate) fn match_children(&mut self, node: NodeId, type_id: TypeId, path: &str) {
        let children: Vec<NodeId> = self
            .doc
            .children(node)
            .filter(|child| self.doc.kind(*child) == Some(NodeKind::Element))
            .collect();
        let Some(complex) = self.schema.complex_type(type_id) else {
            return;
        };
        let model = complex.content_model.clone().unwrap_or(ContentModel::Empty);

        let names: Vec<(Option<String>, String)> = children
            .iter()
            .map(|child| match self.doc.element(*child) {
                Some(element) => (
                    element.namespace_uri().map(str::to_owned),
                    element.local_name().to_owned(),
                ),
                None => (None, String::new()),
            })
            .collect();

        // XML Schema 1.1 §3.4.4.2: an `xs:openContent` (or the
        // `xs:defaultOpenContent` of the schema document) adds a wildcard that
        // is either *interleaved* through the content model or appended as a
        // *suffix*. The wildcard is offered whatever the model itself cannot
        // consume at that point.
        //
        // §3.4.2.2: for a type derived by extension this is not simply its
        // own `open_content` — an extension inherits, and cannot cancel, the
        // open content of its base, so the *effective* one has to be resolved
        // by walking `{derivation}` (`effective_open_content`).
        let effective_open = effective_open_content(self.schema, type_id);
        let open = effective_open.as_ref().and_then(Open::of);
        let attribution = match &model {
            ContentModel::Empty => {
                let mut attribution = Vec::new();
                for (index, (namespace, local)) in names.iter().enumerate() {
                    match open.as_ref().filter(|open| {
                        open.admits(self.schema, &(namespace.clone(), local.clone()), &[])
                    }) {
                        Some(open) => attribution.push(Attribution::Wild(open.process_contents)),
                        None => {
                            if index == 0 {
                                self.error(
                                    "cvc-complex-type.2.1",
                                    path,
                                    format!(
                                        "`{}` is not allowed: this type has empty content",
                                        qualified(namespace.as_deref(), local)
                                    ),
                                );
                            }
                            attribution.clear();
                            break;
                        }
                    }
                }
                attribution
            }
            ContentModel::Fsm(fsm) => {
                let greedy = match_fsm(self.schema, fsm, &names, open.as_ref());
                // The greedy walk is exact for a model that satisfies Unique
                // Particle Attribution, which is nearly all of them; when it
                // fails, the model may still accept the children along a run
                // the walk did not take, so the exact subset simulation gets
                // the last word.
                let outcome = match greedy {
                    Ok(attribution) => Ok(attribution),
                    Err(failure) if open.is_none() => {
                        match_fsm_exact(self.schema, fsm, &names).map_err(|_| failure)
                    }
                    Err(failure) => Err(failure),
                };
                match outcome {
                    Ok(attribution) => attribution,
                    Err(failure) => {
                        self.report_content_failure(path, &names, failure);
                        Vec::new()
                    }
                }
            }
            ContentModel::All(all) => match match_all(self.schema, all, &names, open.as_ref()) {
                Ok(attribution) => attribution,
                Err(failure) => {
                    self.report_content_failure(path, &names, failure);
                    Vec::new()
                }
            },
        };

        self.check_dynamic_edc(path, &names, &attribution);

        let assessed_children = children.clone();
        for (index, child) in children.iter().enumerate() {
            let child_path = self.path(*child);
            match attribution.get(index).copied() {
                Some(Attribution::Declared(head)) => {
                    let declaration = self.substituted_declaration(head, &names[index]);
                    match declaration {
                        Some(id) => {
                            if id != head {
                                self.check_substitution(head, id, &child_path);
                            }
                            self.assess_element(*child, Some(id), None);
                        }
                        None => self.assess_element(*child, Some(head), None),
                    }
                }
                Some(Attribution::Wild(process)) => {
                    // §3.10.4.1: an item ·attributed to· a `skip` wildcard —
                    // and everything below it — is ·skipped·, which takes it
                    // out of identity-constraint evaluation (§3.11.4).
                    if process == ProcessContents::Skip {
                        self.skipped.insert(*child);
                    }
                    if let Some((declaration, _)) =
                        self.wildcard_declaration(*child, process, &child_path)
                    {
                        self.assess_element(*child, declaration, None);
                    }
                }
                None => {
                    // The model did not match; the children are still assessed
                    // against their global declarations when they have one, so
                    // that further errors are reported too.
                    if let Some(id) = self
                        .schema
                        .find_element(names[index].0.as_deref(), &names[index].1)
                    {
                        self.assess_element(*child, Some(id), None);
                    }
                }
            }
        }
        self.check_locally_declared_types(type_id, &assessed_children, &names);
    }

    /// Clause 5 of *Element Locally Valid (Complex Type)* (§3.4.4.2): "For each
    /// element information item in `E . [children]` … if neither the
    /// ·governing type definition· nor the ·locally declared type· is ·absent·,
    /// then the ·governing type definition· is the same as, or is ·validly
    /// substitutable· for, the ·locally declared type·, ·without limitation·."
    ///
    /// This is the instance-time half of what XSD 1.1's appendix G.1.6 calls
    /// "more consistency in type assignment when elements with the same
    /// expanded name may match both a local element declaration and a wildcard
    /// in the same content model" — an element admitted by a wildcard is
    /// assessed against its *global* declaration (or `xsi:type`), and that
    /// type has to be a specialization of whatever the content model, or a type
    /// it restricts, declares for that name (`saxonData/Wild/wild062` …
    /// `wild064`, `wild068`).
    fn check_locally_declared_types(
        &mut self,
        type_id: TypeId,
        children: &[NodeId],
        names: &[(Option<String>, String)],
    ) {
        if !self.schema.version().is_1_1() {
            return;
        }
        for (index, child) in children.iter().enumerate() {
            let Some(name) = names.get(index) else {
                continue;
            };
            let Some(governing) = self
                .elements
                .get(child)
                .and_then(|psvi| psvi.type_definition)
            else {
                continue;
            };
            let Some(local) = self.locally_declared_type(type_id, name) else {
                continue;
            };
            if governing == local || self.schema.is_derived_from(governing, local) {
                continue;
            }
            let child_path = self.path(*child);
            self.error(
                "cvc-complex-type.5",
                &child_path,
                format!(
                    "`{}` is governed by `{}`, which is not validly substitutable for `{}`, the \
                     type this content model declares for that name",
                    qualified(name.0.as_deref(), &name.1),
                    self.schema.type_name(governing),
                    self.schema.type_name(local)
                ),
            );
        }
    }

    /// The ·locally declared type· of an element of this name within `type_id`
    /// (§3.4.4.1): the declared `{type definition}` of an element declaration
    /// of the same expanded name ·contained· by the type's content model —
    /// "whether ·directly·, ·indirectly·, or ·implicitly·" — falling back to
    /// the same question asked of the `{base type definition}`, and ·absent·
    /// once `xs:anyType` is reached.
    fn locally_declared_type(
        &self,
        type_id: TypeId,
        name: &(Option<String>, String),
    ) -> Option<TypeId> {
        let mut current = type_id;
        let mut guard = 0usize;
        loop {
            guard += 1;
            if current == self.schema.any_type() || guard > 64 {
                return None;
            }
            if let Some(complex) = self.schema.complex_type(current)
                && let Some(model) = &complex.content_model
                && let Some(declaration) = self.model_declaration(model, name)
            {
                return Some(self.schema.element_declaration(declaration).type_id);
            }
            let base = self.schema.type_definition(current).base?;
            if base == current {
                return None;
            }
            current = base;
        }
    }

    /// The element declaration of this expanded name that a content model
    /// ·contains·, counting the members of the ·substitution group· of every
    /// declaration it contains (§3.8.6.3's ·implicit· containment).
    fn model_declaration(
        &self,
        model: &ContentModel,
        name: &(Option<String>, String),
    ) -> Option<ElementId> {
        let matches = |candidate: ElementId| {
            let declaration = self.schema.element_declaration(candidate);
            declaration.name.local_name() == name.1
                && declaration.name.namespace() == name.0.as_deref()
        };
        for head in model_element_declarations(model) {
            if matches(head) {
                return Some(head);
            }
            let declaration = self.schema.element_declaration(head);
            if declaration.block.substitution {
                continue;
            }
            if let Some(member) = declaration
                .substitutable_by
                .iter()
                .copied()
                .filter(|member| self.schema.is_actual_substitution_member(*member))
                .find(|member| matches(*member))
            {
                return Some(member);
            }
        }
        None
    }

    /// The *dynamic* Element Declarations Consistent check of XSD 1.1
    /// (§3.8.4, *cvc-complex-type.2.4.d*).
    ///
    /// XSD 1.0 made it a schema constraint that no content model could contain
    /// two element declarations of the same name with different types, and
    /// exempted wildcards from it. XSD 1.1 keeps the static rule for declared
    /// particles but adds an instance-time one for wildcards: an element
    /// matched by a **wildcard** must not have the same expanded name as a
    /// sibling matched by an element *declaration* unless the two are governed
    /// by the same type (`saxonData/Wild/wild061`).
    ///
    /// `processContents="skip"` puts an element outside assessment altogether,
    /// so it takes no part in the comparison.
    fn check_dynamic_edc(
        &mut self,
        path: &str,
        names: &[(Option<String>, String)],
        attribution: &[Attribution],
    ) {
        if !self.schema.version().is_1_1() {
            return;
        }
        // `(name, type, matched by a wildcard)` for every assessed child.
        type Governed<'name> = (&'name (Option<String>, String), TypeId, bool);
        let mut governing: Vec<Governed<'_>> = Vec::new();
        for (index, name) in names.iter().enumerate() {
            let (declaration, wild) = match attribution.get(index) {
                Some(Attribution::Declared(head)) => (
                    self.substituted_declaration(*head, name).or(Some(*head)),
                    false,
                ),
                Some(Attribution::Wild(ProcessContents::Skip)) | None => continue,
                Some(Attribution::Wild(_)) => {
                    (self.schema.find_element(name.0.as_deref(), &name.1), true)
                }
            };
            let Some(declaration) = declaration else {
                continue;
            };
            let type_id = self.schema.element_declaration(declaration).type_id;
            let clash = governing.iter().any(|(candidate, seen, seen_wild)| {
                *candidate == name
                    && *seen != type_id
                    && (*seen_wild || wild)
                    && !self.types_agree(*seen, type_id)
            });
            if clash {
                self.error(
                    "cvc-complex-type.2.4.d",
                    path,
                    format!(
                        "`{}` is matched both by a wildcard and by an element declaration, with \
                         different types",
                        qualified(name.0.as_deref(), &name.1)
                    ),
                );
                return;
            }
            governing.push((name, type_id, wild));
        }
    }

    /// Whether two governing types are close enough that XSD 1.1's dynamic
    /// *Element Declarations Consistent* check is satisfied.
    ///
    /// The constraint is about an item whose governing type could not be
    /// determined; when one of the two types is ·validly substitutable· for the
    /// other — derived from it either way, or a member of it when the other is
    /// a union — the item is assessed the same way whichever declaration wins,
    /// so there is nothing to report (`saxonData/Wild/wild063`, `wild064`,
    /// `wild066`; `wild061`, where `xs:date` and `xs:time` are unrelated,
    /// still is).
    fn types_agree(&self, left: TypeId, right: TypeId) -> bool {
        if self.schema.is_derived_from(left, right) || self.schema.is_derived_from(right, left) {
            return true;
        }
        let member_of = |union: TypeId, other: TypeId| {
            self.schema
                .union_member_types(union)
                .is_some_and(|members| {
                    members.iter().any(|member| {
                        *member == other || self.schema.is_derived_from(other, *member)
                    })
                })
        };
        member_of(left, right) || member_of(right, left)
    }

    fn report_content_failure(
        &mut self,
        path: &str,
        names: &[(Option<String>, String)],
        failure: MatchFailure,
    ) {
        match failure {
            MatchFailure::Unexpected(index) => {
                let (namespace, local) = &names[index];
                self.error(
                    "cvc-complex-type.2.4.a",
                    path,
                    format!(
                        "`{}` is not expected at this position in the content model",
                        qualified(namespace.as_deref(), local)
                    ),
                );
            }
            MatchFailure::Incomplete => self.error(
                "cvc-complex-type.2.4.b",
                path,
                "the content is not complete: the content model expects more children",
            ),
        }
    }

    fn substituted_declaration(
        &self,
        head: ElementId,
        name: &(Option<String>, String),
    ) -> Option<ElementId> {
        let declaration = self.schema.element_declaration(head);
        if declaration.name.namespace() == name.0.as_deref()
            && declaration.name.local_name() == name.1
        {
            return Some(head);
        }
        declaration
            .substitutable_by
            .iter()
            .copied()
            .filter(|member| self.schema.is_actual_substitution_member(*member))
            .find(|member| {
                let member = self.schema.element_declaration(*member);
                member.name.namespace() == name.0.as_deref() && member.name.local_name() == name.1
            })
    }

    /// *Substitution Group OK (Transitive)* (§3.3.6): whether `member` may
    /// stand in for `head`.
    ///
    /// Clause 3 measures the derivation of the member's type from the head's
    /// against more than the head declaration's own `block`: it takes "the
    /// union of the blocking constraint, `C`'s {prohibited substitutions} (if
    /// `C` is typed by a complex type definition), and the {prohibited
    /// substitutions} … of any intermediate {type definition}s in the
    /// derivation". So a head whose *type* carries `block="restriction"`
    /// blocks a member derived by restriction even when the head element
    /// itself blocks nothing
    /// (`sunData/ElemDecl/disallowedSubst/disallowedSubst00503m3` … `m5`).
    fn check_substitution(&mut self, head: ElementId, member: ElementId, path: &str) {
        let head_declaration = self.schema.element_declaration(head);
        let member_declaration = self.schema.element_declaration(member);
        let blocked = head_declaration.block.union_with(type_prohibitions(
            self.schema
                .type_definition(head_declaration.type_id)
                .block
                .union_with(
                    self.prohibited_between(member_declaration.type_id, head_declaration.type_id),
                ),
        ));
        if blocked.substitution {
            self.error(
                "cvc-complex-type.2.4.a",
                path,
                format!(
                    "`{}` may not substitute for `{}`: substitution is blocked",
                    member_declaration.name, head_declaration.name
                ),
            );
            return;
        }
        if let Some(used) = self
            .schema
            .derivation_path(member_declaration.type_id, head_declaration.type_id)
            && ((used.extension && blocked.extension) || (used.restriction && blocked.restriction))
        {
            self.error(
                "cvc-complex-type.2.4.a",
                path,
                format!(
                    "`{}` may not substitute for `{}`: the derivation is blocked",
                    member_declaration.name, head_declaration.name
                ),
            );
        }
    }

    /// The union of the {prohibited substitutions} of every type strictly
    /// between `from` and `to` on the chain of {base type definition}s — the
    /// "intermediate type definitions" of §3.3.6 clause 3.
    ///
    /// The endpoints are excluded: the derived type's own `block` says nothing
    /// about how it was reached, and the head type's is added by the caller.
    /// An empty set is the answer whenever `from` does not derive from `to`.
    fn prohibited_between(&self, from: TypeId, to: TypeId) -> DerivationSet {
        let mut prohibited = DerivationSet::empty();
        let mut current = from;
        for _ in 0..self.schema.types.len() {
            if current == to {
                return prohibited;
            }
            let Some(next) = self.schema.type_definition(current).base else {
                return DerivationSet::empty();
            };
            if next == current || next == to {
                return prohibited;
            }
            current = next;
            prohibited = prohibited.union_with(self.schema.type_definition(current).block);
        }
        DerivationSet::empty()
    }
}

/// The {prohibited substitutions} a *type definition* contributes, which
/// §3.4.1 confines to a subset of `{extension, restriction}`.
///
/// `block="#all"` is parsed into the whole [`DerivationSet`], `substitution`
/// included, because one parser serves `block`, `final` and `blockDefault`
/// alike. Only an *element declaration*'s `{disallowed substitutions}` may
/// really carry `substitution`, so a type's contribution to the blocking
/// constraint of §3.3.6 clause 3 is narrowed here — otherwise a head typed by
/// a `block="#all"` complex type would refuse every member of its
/// substitution group outright.
fn type_prohibitions(block: DerivationSet) -> DerivationSet {
    DerivationSet {
        extension: block.extension,
        restriction: block.restriction,
        ..DerivationSet::empty()
    }
}

/// The open-content wildcard actually in force for `type_id`'s content model.
///
/// This is not simply `type_id`'s own `open_content` (which
/// `compile::types::Compiler::build_complex_content` leaves exactly as
/// written, or defaulted from `xs:defaultOpenContent` — deliberately *not*
/// merged with the base's, because `compile::restriction::
/// check_open_content_extension`/`check_open_content_restriction` need that
/// raw value, in a later compile pass, to detect a derivation that conflicts
/// with its base).
///
/// [XSD 1.1 §3.4.2.2][spec]: an extension cannot cancel open content it
/// inherits — there is deliberately no way to, not even by writing an
/// explicit `xs:openContent mode="none"` in the extension, which only means
/// "this step contributes nothing of its own" (`Open/open031`, where the
/// base's `interleave` open content survives an extension that also carries
/// a `mode="none"` and an — inapplicable, because it is not written on an
/// *empty* type — `xs:defaultOpenContent`). So when `type_id` derives by
/// extension, its own open content and its base's *effective* one (computed
/// the same way, recursively) are merged: the base's propagates unchanged
/// when the extension adds none of its own (`Open/open027`'s `suffix`
/// reaching past the particle the extension appends), and when both apply
/// the two wildcards are **united**, not nested (`Open/open047`: one
/// namespace from the base, everything but that namespace and one other from
/// the extension, net effect their union) — exactly as `compile::types::
/// Compiler::merge_attributes` unions the attribute wildcard for the same
/// derivation. The mode is `interleave` unless both sides are `suffix` (an
/// `interleave` base forcing a non-`interleave` extension is already
/// rejected by `check_open_content_extension`).
///
/// A restriction's open content, by contrast, always stands on its own
/// (`check_open_content_restriction` is what constrains it against the
/// base's, not this), so the recursion only ever follows an `Extension` step
/// — by the time validation runs every type is already known to be a valid
/// extension/restriction of its base, so there is nothing left here to
/// re-check.
///
/// [spec]: https://www.w3.org/TR/xmlschema11-1/#Complex_Type_Definition_details
fn effective_open_content(schema: &Schema, type_id: TypeId) -> Option<OpenContent> {
    let complex = schema.complex_type(type_id)?;
    let own = complex
        .open_content
        .clone()
        .filter(|open| open.wildcard.is_some());
    if complex.derivation != Derivation::Extension {
        return own;
    }
    let base = schema.type_definition(type_id).base?;
    if base == type_id {
        return own;
    }
    let base_open = effective_open_content(schema, base);
    match (base_open, own) {
        (None, own) => own,
        (Some(base_open), None) => Some(base_open),
        (Some(base_open), Some(own)) => {
            let wildcard = match (&base_open.wildcard, &own.wildcard) {
                (Some(base_wildcard), Some(own_wildcard)) => Some(Wildcard {
                    namespace: base_wildcard
                        .namespace
                        .union_with(&own_wildcard.namespace)
                        .unwrap_or(NamespaceConstraint::Any),
                    process_contents: base_wildcard.process_contents,
                    // §3.10.6.3: the union disallows a name only when
                    // *neither* wildcard would have allowed it.
                    disallowed: crate::components::union_disallowed_names(
                        base_wildcard,
                        own_wildcard,
                    ),
                }),
                (Some(only), None) | (None, Some(only)) => Some(only.clone()),
                (None, None) => None,
            };
            Some(OpenContent {
                interleave: base_open.interleave || own.interleave,
                wildcard,
            })
        }
    }
}

/// The open-content wildcard in force for one complex type.
#[derive(Debug, Clone)]
struct Open {
    wildcard: Wildcard,
    /// `mode="interleave"`; `false` is `mode="suffix"`.
    interleave: bool,
    process_contents: ProcessContents,
}

impl Open {
    fn of(open: &OpenContent) -> Option<Self> {
        let wildcard = open.wildcard.clone()?;
        let process_contents = wildcard.process_contents;
        Some(Self {
            wildcard,
            interleave: open.interleave,
            process_contents,
        })
    }

    fn admits(
        &self,
        schema: &Schema,
        name: &(Option<String>, String),
        siblings: &[ElementId],
    ) -> bool {
        wildcard_matches(schema, &self.wildcard, name, siblings)
    }
}

/// Why a child sequence did not match.
#[derive(Debug, Clone, Copy)]
enum MatchFailure {
    /// The child at this index could not be attributed to any particle.
    Unexpected(usize),
    /// The sequence ended before the model was satisfied.
    Incomplete,
}

fn term_matches(
    schema: &Schema,
    term: &LeafTerm,
    name: &(Option<String>, String),
    siblings: &[ElementId],
) -> bool {
    match term {
        LeafTerm::Element(id) => {
            let declaration = schema.element_declaration(*id);
            if declaration.name.namespace() == name.0.as_deref()
                && declaration.name.local_name() == name.1
            {
                return true;
            }
            // Only the ·actual· substitution group can match: under XSD 1.0 an
            // `abstract` member is not part of it, so a particle referring to
            // the head does not match that name at all and some other particle
            // of the model may (`wgData/sg/upa.xsd`, where a local `e1`
            // competes with an abstract global one).
            declaration
                .substitutable_by
                .iter()
                .filter(|member| schema.is_actual_substitution_member(**member))
                .any(|member| {
                    let member = schema.element_declaration(*member);
                    let namespace_matches = member.name.namespace() == name.0.as_deref();
                    namespace_matches && member.name.local_name() == name.1
                })
        }
        LeafTerm::Wildcard(wildcard) => wildcard_matches(schema, wildcard, name, siblings),
    }
}

/// Whether one of `siblings` — the element declarations of the content model
/// the wildcard sits in — is declared with `name`, substitution groups
/// included.
///
/// This is what `notQName="##definedSibling"` excludes (XSD 1.1 §3.10.1). It
/// is answered here rather than at assembly time because a substitution group
/// is only complete once the whole schema is (`saxonData/Wild/wild071`).
fn is_defined_sibling(
    schema: &Schema,
    siblings: &[ElementId],
    name: &(Option<String>, String),
) -> bool {
    siblings.iter().any(|id| {
        let declaration = schema.element_declaration(*id);
        let matches_name = |candidate: &crate::components::ElementDeclaration| {
            candidate.name.namespace() == name.0.as_deref() && candidate.name.local_name() == name.1
        };
        matches_name(declaration)
            || declaration
                .substitutable_by
                .iter()
                .filter(|member| schema.is_actual_substitution_member(**member))
                .any(|member| matches_name(schema.element_declaration(*member)))
    })
}

/// Whether a wildcard matches an element name, `notQName` included.
///
/// `##defined` (XSD 1.1 §3.10.1) asks whether the name is that of a *global*
/// element declaration, which only the whole schema can answer.
fn wildcard_matches(
    schema: &Schema,
    wildcard: &crate::components::Wildcard,
    name: &(Option<String>, String),
    siblings: &[ElementId],
) -> bool {
    if wildcard.disallowed.defined_sibling && is_defined_sibling(schema, siblings, name) {
        return false;
    }
    let defined =
        wildcard.disallowed.defined && schema.find_element(name.0.as_deref(), &name.1).is_some();
    wildcard.matches(name.0.as_deref(), &name.1, defined)
}

fn attribution_of(term: &LeafTerm) -> Attribution {
    match term {
        LeafTerm::Element(id) => Attribution::Declared(*id),
        LeafTerm::Wildcard(wildcard) => Attribution::Wild(wildcard.process_contents),
    }
}

/// Walks the Glushkov automaton. A content model that passed the Unique
/// Particle Attribution check is deterministic, so a single position is chosen
/// at every step; where a model is ambiguous anyway (which the schema author
/// was told about at assembly time) an element particle wins over a wildcard,
/// as §3.8.6 prescribes.
fn match_fsm(
    schema: &Schema,
    fsm: &Fsm,
    names: &[(Option<String>, String)],
    open: Option<&Open>,
) -> Result<Vec<Attribution>, MatchFailure> {
    let siblings = fsm_siblings(fsm);
    let siblings = siblings.as_slice();
    let mut active: &[usize] = &fsm.first;
    let mut attribution = Vec::with_capacity(names.len());
    let mut last_position: Option<usize> = None;
    let mut in_suffix = false;
    for (index, name) in names.iter().enumerate() {
        let mut chosen: Option<usize> = None;
        if in_suffix {
            match open.filter(|open| open.admits(schema, name, siblings)) {
                Some(open) => {
                    attribution.push(Attribution::Wild(open.process_contents));
                    continue;
                }
                None => return Err(MatchFailure::Unexpected(index)),
            }
        }
        for position in active {
            if term_matches(schema, &fsm.positions[*position], name, siblings) {
                let is_element = matches!(fsm.positions[*position], LeafTerm::Element(_));
                match chosen {
                    Some(existing)
                        if is_element
                            && !matches!(fsm.positions[existing], LeafTerm::Element(_)) =>
                    {
                        chosen = Some(*position);
                    }
                    Some(_) => {}
                    None => chosen = Some(*position),
                }
            }
        }
        let Some(position) = chosen else {
            let admitted = open.filter(|open| open.admits(schema, name, siblings));
            let Some(open) = admitted else {
                return Err(MatchFailure::Unexpected(index));
            };
            if !open.interleave {
                // `mode="suffix"`: the model must already be satisfied before
                // the wildcard may take over, and it keeps the rest.
                let satisfied = match last_position {
                    Some(position) => fsm.last[position],
                    None => fsm.nullable,
                };
                if !satisfied {
                    return Err(MatchFailure::Unexpected(index));
                }
                in_suffix = true;
            }
            attribution.push(Attribution::Wild(open.process_contents));
            continue;
        };
        attribution.push(attribution_of(&fsm.positions[position]));
        active = &fsm.follow[position];
        last_position = Some(position);
    }
    let accepted = match last_position {
        Some(position) => fsm.last[position],
        None => fsm.nullable,
    };
    if accepted {
        Ok(attribution)
    } else {
        Err(MatchFailure::Incomplete)
    }
}

/// Matches the model *exactly*, by simulating the automaton as the
/// nondeterministic machine it is.
///
/// The greedy walk above commits to one position per child, which is right for
/// a model that satisfies Unique Particle Attribution but wrong for one that
/// does not: `(a+){2,unbounded}` accepts `a a` only if the walk is willing to
/// end the first `a+` after one `a`, and the greedy walk is not
/// (`msData/complexType/ctZ006`). This routine keeps the whole reachable
/// *set* of positions at each step — a subset simulation, so it is
/// `O(children × positions²)` rather than exponential — and then reads one
/// accepting run back out of it, which is what the attribution needs.
///
/// It is used only after the greedy walk has failed and only when there is no
/// open content to interleave, so the fast path is untouched.
fn match_fsm_exact(
    schema: &Schema,
    fsm: &Fsm,
    names: &[(Option<String>, String)],
) -> Result<Vec<Attribution>, MatchFailure> {
    let siblings = fsm_siblings(fsm);
    let siblings = siblings.as_slice();
    if names.is_empty() {
        return if fsm.nullable {
            Ok(Vec::new())
        } else {
            Err(MatchFailure::Incomplete)
        };
    }
    // `reachable[i]` holds the positions that can consume `names[i]`.
    let mut reachable: Vec<Vec<usize>> = Vec::with_capacity(names.len());
    let mut current: Vec<usize> = fsm.first.clone();
    for (index, name) in names.iter().enumerate() {
        let matched: Vec<usize> = current
            .iter()
            .copied()
            .filter(|position| term_matches(schema, &fsm.positions[*position], name, siblings))
            .collect();
        if matched.is_empty() {
            return Err(MatchFailure::Unexpected(index));
        }
        let mut next: Vec<usize> = Vec::new();
        for position in &matched {
            for follower in &fsm.follow[*position] {
                if !next.contains(follower) {
                    next.push(*follower);
                }
            }
        }
        reachable.push(matched);
        current = next;
    }
    // Read one accepting run back: the last step must be able to end the
    // model, and each earlier step must be able to reach the one after it.
    let mut run = vec![0_usize; names.len()];
    let last = reachable
        .last()
        .and_then(|positions| choose(fsm, positions.iter().copied().filter(|p| fsm.last[*p])))
        .ok_or(MatchFailure::Incomplete)?;
    run[names.len() - 1] = last;
    for index in (0..names.len() - 1).rev() {
        let successor = run[index + 1];
        let position = choose(
            fsm,
            reachable[index]
                .iter()
                .copied()
                .filter(|position| fsm.follow[*position].contains(&successor)),
        )
        .ok_or(MatchFailure::Incomplete)?;
        run[index] = position;
    }
    Ok(run
        .into_iter()
        .map(|position| attribution_of(&fsm.positions[position]))
        .collect())
}

/// Picks one position out of a set, preferring an element particle over a
/// wildcard as §3.8.6 prescribes when a model is ambiguous anyway.
fn choose(fsm: &Fsm, positions: impl Iterator<Item = usize>) -> Option<usize> {
    let mut chosen: Option<usize> = None;
    for position in positions {
        let is_element = matches!(fsm.positions[position], LeafTerm::Element(_));
        match chosen {
            Some(existing)
                if is_element && !matches!(fsm.positions[existing], LeafTerm::Element(_)) =>
            {
                chosen = Some(position);
            }
            Some(_) => {}
            None => chosen = Some(position),
        }
    }
    chosen
}

/// Matches an `xs:all` group: order does not matter, so each child takes the
/// first item that can still accept it.
///
/// Open content is handled here exactly as [`match_fsm`] handles it for an
/// ordered model. `mode="interleave"` lets a wildcard match take an item at any
/// point; `mode="suffix"` (§3.4.2.4) does not — "the content is valid if it
/// consists of a sequence which is valid with respect to the content model,
/// **followed by** a sequence of items matched by the wildcard" — so the model
/// must already be satisfied before the first wildcard match, and once it has
/// been taken nothing but further wildcard matches may follow
/// (`saxonData/Open/open009`, whose `n1` instance writes a content-model
/// element after two wildcard matches).
fn match_all(
    schema: &Schema,
    model: &AllModel,
    names: &[(Option<String>, String)],
    open: Option<&Open>,
) -> Result<Vec<Attribution>, MatchFailure> {
    let siblings: Vec<ElementId> = model
        .items
        .iter()
        .filter_map(|item| match &item.term {
            LeafTerm::Element(id) => Some(*id),
            LeafTerm::Wildcard(_) => None,
        })
        .collect();
    let siblings = siblings.as_slice();
    let mut counts = vec![0_u32; model.items.len()];
    let mut attribution = Vec::with_capacity(names.len());
    let mut in_suffix = false;
    for (index, name) in names.iter().enumerate() {
        if in_suffix {
            match open.filter(|open| open.admits(schema, name, siblings)) {
                Some(open) => {
                    attribution.push(Attribution::Wild(open.process_contents));
                    continue;
                }
                None => return Err(MatchFailure::Unexpected(index)),
            }
        }
        let mut chosen: Option<usize> = None;
        for (position, item) in model.items.iter().enumerate() {
            if !item.max_occurs.allows(counts[position] + 1) {
                continue;
            }
            if term_matches(schema, &item.term, name, siblings) {
                let is_element = matches!(item.term, LeafTerm::Element(_));
                match chosen {
                    Some(existing)
                        if is_element
                            && !matches!(model.items[existing].term, LeafTerm::Element(_)) =>
                    {
                        chosen = Some(position);
                    }
                    Some(_) => {}
                    None => chosen = Some(position),
                }
            }
        }
        let Some(position) = chosen else {
            let Some(open) = open.filter(|open| open.admits(schema, name, siblings)) else {
                return Err(MatchFailure::Unexpected(index));
            };
            if !open.interleave {
                // `mode="suffix"`: the model must already be satisfied before
                // the wildcard may take over, and it keeps the rest.
                let satisfied = model
                    .items
                    .iter()
                    .enumerate()
                    .all(|(position, item)| counts[position] >= item.min_occurs)
                    || (counts.iter().all(|count| *count == 0) && model.emptiable);
                if !satisfied {
                    return Err(MatchFailure::Unexpected(index));
                }
                in_suffix = true;
            }
            attribution.push(Attribution::Wild(open.process_contents));
            continue;
        };
        counts[position] += 1;
        attribution.push(attribution_of(&model.items[position].term));
    }
    if names.is_empty() && model.emptiable {
        return Ok(attribution);
    }
    for (position, item) in model.items.iter().enumerate() {
        if counts[position] < item.min_occurs {
            return Err(MatchFailure::Incomplete);
        }
        if let MaxOccurs::Bounded(max) = item.max_occurs
            && counts[position] > max
        {
            return Err(MatchFailure::Incomplete);
        }
    }
    Ok(attribution)
}

/// The element declarations of a compiled content model, for
/// `##definedSibling`.
fn fsm_siblings(fsm: &Fsm) -> Vec<ElementId> {
    let mut out = Vec::new();
    for term in &fsm.positions {
        if let LeafTerm::Element(id) = term
            && !out.contains(id)
        {
            out.push(*id);
        }
    }
    out
}

/// Every element declaration a content model holds, in model order.
fn model_element_declarations(model: &ContentModel) -> Vec<ElementId> {
    match model {
        ContentModel::Empty => Vec::new(),
        ContentModel::All(all) => all
            .items
            .iter()
            .filter_map(|item| match &item.term {
                LeafTerm::Element(id) => Some(*id),
                LeafTerm::Wildcard(_) => None,
            })
            .collect(),
        ContentModel::Fsm(fsm) => fsm
            .positions
            .iter()
            .filter_map(|term| match term {
                LeafTerm::Element(id) => Some(*id),
                LeafTerm::Wildcard(_) => None,
            })
            .collect(),
    }
}
