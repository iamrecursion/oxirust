//! Identity constraints (`xs:key`, `xs:keyref`, `xs:unique`) and the
//! `ID`/`IDREF` rules of `cvc-id`.

use oxixml_dom::{NodeId, NodeKind};

use crate::components::{ContentType, IdcCategory, TypeId};
use crate::value::Value;
use crate::xpath::{NameTest, PathSet, Step};

use super::Validator;

impl<'a> Validator<'a> {
    /// Records an `ID` value, or an `IDREF`/`IDREFS` reference, when the type
    /// of the value calls for it.
    ///
    /// `binding` is the element the *ID/IDREF binding* of §3.17.5.2 binds the
    /// value to: the set it builds consists of "every element information item
    /// … \[which\] has an attribute information item in its \[attributes\] **or
    /// an element information item in its \[children\]** whose ·actual value·
    /// is or contains the \[id\] of this ID/IDREF binding". So an `ID` written
    /// on an attribute binds the element that carries the attribute, and an
    /// `ID` written as an element's *content* binds that element's **parent** —
    /// §3.3.4.5 spells this out: "an element governed by type `xs:ID` provides
    /// a unique identifier for the element's parent element … In the document
    /// `<DOC><X>abcd</X></DOC>`, the ID value 'abcd' will normally be bound to
    /// the `DOC` element."
    ///
    /// [`None`] means the value binds nothing — an `ID`-typed element whose
    /// parent lies outside the ·validation root· — which §3.3.4.5 clause 1
    /// rejects just as it rejects a dangling `IDREF`.
    ///
    /// `carrier` is the node the value was *written on*, which is what a
    /// failure has to be reported against: the dangling-reference checks run
    /// after the whole document has been walked, so the node cannot be
    /// recovered from the walk itself any more.
    pub(crate) fn note_identifier(
        &mut self,
        type_id: TypeId,
        normalized: &str,
        binding: Option<NodeId>,
        carrier: NodeId,
        path: &str,
    ) {
        let mut ids = Vec::new();
        let mut references = Vec::new();
        let mut entities = Vec::new();
        self.classify_identifier(
            type_id,
            normalized,
            &mut ids,
            &mut references,
            &mut entities,
        );
        // Datatypes §3.3.11: an `xs:ENTITY` value must be the name of an
        // *unparsed entity* the instance declares. The check runs only when the
        // instance carries an internal DTD subset, because an external subset
        // is never retrieved and the validator would otherwise reject every
        // legal value (`saxonData/Id/id018` … `id021`).
        if let Some(declared) = self.unparsed_entities.clone() {
            for value in entities {
                if !declared.contains(&value) {
                    self.error(
                        "cvc-datatype-valid.1.2.1",
                        path,
                        format!("`{value}` is not the name of an unparsed entity"),
                    );
                    self.mark_invalid(carrier);
                }
            }
        }
        // XSD 1.1 §3.17.5.2 counts *elements* rather than occurrences: an
        // element that carries the same `ID` value twice — on two attributes,
        // or on two `xs:ID`-typed children of the same parent — still makes a
        // one-member ·binding·, and clause 2 is about the size of that set.
        // XSD 1.0 asked instead that every `ID` value in a document be
        // distinct, so the repeat is an error there
        // (`msData/element/elemZ016`, invalid under 1.0 and valid under 1.1,
        // and the suite says so in as many words: "two child elements of type
        // xs:ID are permitted if the ID is the same").
        let one_binding_per_element = self.schema.version().is_1_1();
        for value in ids {
            let Some(owner) = binding else {
                // The string still gets an entry in the table — with an empty
                // ·binding·, which clause 1 of §3.3.4.5 rejects unless some
                // *other* item binds the same string.
                self.ids.entry(value.clone()).or_default();
                self.unbound_ids.push((value, path.to_owned(), carrier));
                continue;
            };
            let owners = self.ids.entry(value.clone()).or_default();
            if one_binding_per_element && owners.contains(&owner) {
                continue;
            }
            owners.push(owner);
            // §3.3.4.5 clause 2: the ·binding· must not have more than one
            // member.
            if owners.len() > 1 {
                self.error(
                    "cvc-id.2",
                    path,
                    format!("the ID `{value}` is not unique in the document"),
                );
                self.mark_invalid(carrier);
            }
        }
        for value in references {
            self.idrefs.push((value, path.to_owned(), carrier));
        }
    }

    /// Splits a validated value into the `ID` values it binds and the `IDREF`
    /// values it references.
    ///
    /// A *list* contributes one entry per item and a *union* contributes
    /// whatever its first accepting member type contributes, which is what
    /// §3.3.4 means by "the declared type is or is derived from `ID`" once the
    /// 1.1 varieties are taken into account.
    fn classify_identifier(
        &self,
        type_id: TypeId,
        normalized: &str,
        ids: &mut Vec<String>,
        references: &mut Vec<String>,
        entities: &mut Vec<String>,
    ) {
        if let Some(id_type) = self.schema.builtin("ID")
            && self.schema.is_derived_from(type_id, id_type)
        {
            ids.push(normalized.to_owned());
            return;
        }
        if let Some(idref) = self.schema.builtin("IDREF")
            && self.schema.is_derived_from(type_id, idref)
        {
            references.push(normalized.to_owned());
            return;
        }
        if let Some(entity) = self.schema.builtin("ENTITY")
            && self.schema.is_derived_from(type_id, entity)
        {
            entities.push(normalized.to_owned());
            return;
        }
        if let Some(item) = self.schema.list_item_type(type_id) {
            for token in crate::value::split_tokens(normalized) {
                self.classify_identifier(item, token, ids, references, entities);
            }
            return;
        }
        if let Some(members) = self.schema.union_member_types(type_id) {
            let context = crate::value::NamespaceContext::new();
            for member in members {
                if crate::simple::validate(self.schema, member, normalized, &context).is_ok() {
                    self.classify_identifier(member, normalized, ids, references, entities);
                    return;
                }
            }
        }
    }

    /// `cvc-id.1` (§3.3.4.5 clause 1): no ID/IDREF binding may have an empty
    /// ·binding· — every `IDREF` must match some `ID`, and every `ID` must
    /// actually identify an element inside the ·validation root·.
    pub(crate) fn check_id_references(&mut self) {
        let references = std::mem::take(&mut self.idrefs);
        for (value, path, carrier) in references {
            if !self.is_bound(&value) {
                self.error(
                    "cvc-id.1",
                    &path,
                    format!("there is no ID matching the IDREF `{value}`"),
                );
                self.mark_invalid(carrier);
            }
        }
        let unbound = std::mem::take(&mut self.unbound_ids);
        for (value, path, carrier) in unbound {
            if !self.is_bound(&value) {
                self.error(
                    "cvc-id.1",
                    &path,
                    format!(
                        "the ID `{value}` identifies no element: an element of type `xs:ID` \
                         identifies its parent, and this one has none within the validation root"
                    ),
                );
                self.mark_invalid(carrier);
            }
        }
    }

    /// Whether some element ·binds· `value` (§3.17.5.2).
    fn is_bound(&self, value: &str) -> bool {
        self.ids.get(value).is_some_and(|owners| !owners.is_empty())
    }

    /// Evaluates every identity constraint that was collected during
    /// assessment.
    pub(crate) fn check_identity_constraints(&mut self) {
        let nodes = std::mem::take(&mut self.idc_nodes);
        // Each entry is one *{identity-constraint table}* entry in the sense of XSD 1.0 §3.11.5:
        // the constraint, the element the table was built for, and its node table. Keeping the
        // element is what makes §3.11.5's recursive assembly work — the node table of an element
        // for a constraint is the union of the tables produced at that element and at every
        // descendant of it, and *only* those, so a `keyref` can never be satisfied by a key that
        // exists solely in an unrelated sibling's subtree.
        let mut key_tables: Vec<(u32, NodeId, Vec<Vec<Value>>)> = Vec::new();
        let mut references: Vec<(u32, NodeId, Vec<Value>, String, NodeId)> = Vec::new();

        for node in nodes {
            let Some(declaration) = self.elements.get(&node).and_then(|psvi| psvi.declaration)
            else {
                continue;
            };
            let constraints = self
                .schema
                .element_declaration(declaration)
                .identity_constraints
                .clone();
            for constraint_id in constraints {
                let constraint = self.schema.identity_constraint(constraint_id).clone();
                // §3.11.4 clause 1: the ·target node set· is the sequence the
                // selector yields "after omitting all element nodes
                // corresponding to element information items that are
                // ·skipped·" — content under a `processContents="skip"`
                // wildcard takes no part in an identity constraint
                // (`saxonData/Wild/wild101` … `wild104`).
                let targets: Vec<NodeId> = self
                    .evaluate(node, &constraint.selector)
                    .into_iter()
                    .filter(|target| !self.is_skipped(*target))
                    .collect();
                let mut table: Vec<Vec<Value>> = Vec::new();
                for target in targets {
                    let target_path = self.path(target);
                    let mut sequence = Vec::new();
                    let mut complete = true;
                    for field in &constraint.fields {
                        // §3.11.4 clause 3: the sequence a field yields "contains
                        // zero or more ·skipped· nodes, at most one node whose
                        // ·governing· type definition is … simple, and no other
                        // nodes" — the skipped ones simply do not contribute.
                        let matched: Vec<FieldMatch> = self
                            .evaluate_field(target, field)
                            .into_iter()
                            .filter(|selected| match selected {
                                FieldMatch::Node(node) => !self.is_skipped(*node),
                                FieldMatch::Defaulted(_) => true,
                            })
                            .collect();
                        if matched.len() > 1 {
                            self.error(
                                "cvc-identity-constraint.3",
                                &target_path,
                                format!(
                                    "the field `{}` of `{}` selects more than one node",
                                    field.source, constraint.name
                                ),
                            );
                            self.mark_invalid(target);
                            complete = false;
                            break;
                        }
                        // cvc-identity-constraint.3: a field must select a node with a
                        // *simple* type. An element with element-only or mixed content has
                        // none, so synthesizing a string value from its character data would
                        // silently accept a schema-invalid constraint.
                        if let Some(FieldMatch::Node(selected)) = matched.first()
                            && self.has_complex_content(*selected)
                        {
                            self.error(
                                "cvc-identity-constraint.3",
                                &target_path,
                                format!(
                                    "the field `{}` of `{}` selects a node whose type is not \
                                     simple",
                                    field.source, constraint.name
                                ),
                            );
                            self.mark_invalid(target);
                            complete = false;
                            break;
                        }
                        match matched.first().and_then(|selected| match selected {
                            FieldMatch::Node(node) => self.node_value(*node),
                            FieldMatch::Defaulted(value) => Some(value.clone()),
                        }) {
                            Some(value) => sequence.push(value),
                            None => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if !complete {
                        if constraint.category == IdcCategory::Key {
                            self.error(
                                "cvc-identity-constraint.4.2.1",
                                &target_path,
                                format!(
                                    "every field of the key `{}` must select exactly one node",
                                    constraint.name
                                ),
                            );
                            self.mark_invalid(target);
                        }
                        continue;
                    }
                    if constraint.category == IdcCategory::KeyRef {
                        references.push((constraint_id.0, node, sequence, target_path, target));
                        continue;
                    }
                    if table
                        .iter()
                        .any(|existing| sequences_equal(existing, &sequence))
                    {
                        let code = if constraint.category == IdcCategory::Key {
                            "cvc-identity-constraint.4.2.2"
                        } else {
                            "cvc-identity-constraint.4.1"
                        };
                        self.error(
                            code,
                            &target_path,
                            format!("duplicate key sequence for `{}`", constraint.name),
                        );
                        self.mark_invalid(target);
                    } else {
                        table.push(sequence);
                    }
                }
                if constraint.category != IdcCategory::KeyRef {
                    key_tables.push((constraint_id.0, node, table));
                }
            }
        }

        for (constraint_raw, scope, sequence, path, target) in references {
            let constraint = &self.schema.identity_constraints[constraint_raw as usize];
            let name = constraint.name.clone();
            let Some(refer) = constraint.refer else {
                continue;
            };
            // cvc-identity-constraint.4.3: the key sequence must appear in the node table of the
            // {referenced key} *of this element* — which, per §3.11.5, is assembled from this
            // element and its descendants only.
            let found = key_tables
                .iter()
                .filter(|(id, owner, _)| {
                    *id == refer.0 && self.is_descendant_or_self(*owner, scope)
                })
                .any(|(_, _, table)| {
                    table
                        .iter()
                        .any(|candidate| sequences_equal(candidate, &sequence))
                });
            if !found {
                self.error(
                    "cvc-identity-constraint.4.3",
                    &path,
                    format!("the key reference `{name}` matches no key"),
                );
                self.mark_invalid(target);
            }
        }
    }

    /// Whether `first` is `second` or a descendant of it — the containment test
    /// behind the recursive assembly of {identity-constraint table}s
    /// (XSD 1.0 §3.11.5).
    fn is_descendant_or_self(&self, first: NodeId, second: NodeId) -> bool {
        first == second || self.doc.ancestors(first).any(|node| node == second)
    }

    /// Whether a node is ·skipped· (§3.10.4.1): "an element or attribute
    /// information item is skipped if it is ·attributed to· a skip wildcard or
    /// if one of its ancestor elements is".
    pub(crate) fn is_skipped(&self, node: NodeId) -> bool {
        if self.skipped.is_empty() {
            return false;
        }
        // An *attribute* is skipped in its own right when it is the one
        // ·attributed to· a skip wildcard; the walk below then answers the
        // other half of §3.10.4.1, "or one of its ancestor elements is".
        if self.skipped.contains(&node) {
            return true;
        }
        let start = match self.doc.kind(node) {
            Some(NodeKind::Attr) => match self.doc.owner_element(node) {
                Some(owner) => owner,
                None => return false,
            },
            _ => node,
        };
        self.skipped.contains(&start)
            || self
                .doc
                .ancestors(start)
                .any(|ancestor| self.skipped.contains(&ancestor))
    }

    /// Whether the element `node` was assessed against a complex type whose
    /// content is *not* simple, so that it has no typed value a `field` could
    /// select (XSD 1.0 §3.11.4 clause 3).
    fn has_complex_content(&self, node: NodeId) -> bool {
        if self.doc.kind(node) != Some(NodeKind::Element) {
            return false;
        }
        let Some(psvi) = self.elements.get(&node) else {
            return false;
        };
        // `xsi:nil` empties an element's content but leaves its type alone: a
        // nilled element governed by a complex type still has no simple type
        // for a field to select, which is what clause 3 asks for
        // (`msData/identityConstraint/idF018`, where one of the two nilled
        // elements carries `xsi:type="xsd:string"` and the other keeps the
        // declaration's `xs:anyType`).
        if psvi.value.is_some() {
            return false;
        }
        let Some(type_id) = psvi.type_definition else {
            return false;
        };
        self.schema
            .type_definition(type_id)
            .as_complex()
            .is_some_and(|complex| !matches!(complex.content, ContentType::Simple(_)))
    }

    /// The typed value of a node selected by a `field`, falling back to its
    /// string value when it was not assessed.
    fn node_value(&self, node: NodeId) -> Option<Value> {
        match self.doc.kind(node) {
            Some(NodeKind::Attr) => {
                if let Some(psvi) = self.attributes.get(&node)
                    && let Some(value) = &psvi.value
                {
                    return Some(value.clone());
                }
                self.doc
                    .attr(node)
                    .map(|data| Value::String(data.value().to_owned()))
            }
            Some(NodeKind::Element) => {
                if let Some(psvi) = self.elements.get(&node) {
                    if psvi.nil {
                        return None;
                    }
                    if let Some(value) = &psvi.value {
                        return Some(value.clone());
                    }
                }
                Some(Value::String(self.character_content(node)))
            }
            _ => None,
        }
    }

    /// Evaluates a `field` path set from `from`, over the **augmented**
    /// infoset.
    ///
    /// §3.11.4 evaluates selectors and fields against the
    /// post-schema-validation infoset, and §3.4.5 has already added an
    /// attribute information item to \[attributes\] for every attribute use
    /// with a `default` or `fixed` {value constraint} the element did not
    /// carry. Those additions have no node in the source document, so they are
    /// yielded as values rather than as node identifiers: an element that
    /// leaves `val` to its `default="test"` contributes the very same key as
    /// one that writes `val="test"` out (`msData/identityConstraint/idF016`,
    /// `idG011`, `idZ011_a`).
    ///
    /// Only the attribute step can select an addition — a defaulted
    /// *element*, unlike a defaulted attribute, is not added to \[children\] —
    /// so every other step still walks the document itself.
    fn evaluate_field(&self, from: NodeId, set: &PathSet) -> Vec<FieldMatch> {
        let mut result: Vec<FieldMatch> = Vec::new();
        for path in &set.paths {
            let owners = self.walk_steps(from, path);
            let Some(test) = &path.attribute else {
                for node in owners {
                    let matched = FieldMatch::Node(node);
                    if !result
                        .iter()
                        .any(|existing| existing.is_same_node(&matched))
                    {
                        result.push(matched);
                    }
                }
                continue;
            };
            for owner in owners {
                for attribute in self.doc.attributes(owner) {
                    if let Some(data) = self.doc.attr(attribute)
                        && attribute_matches(test, data.namespace_uri(), data.local_name())
                    {
                        let matched = FieldMatch::Node(attribute);
                        if !result
                            .iter()
                            .any(|existing| existing.is_same_node(&matched))
                        {
                            result.push(matched);
                        }
                    }
                }
                // An attribute is defaulted only when the element does not
                // carry one of that name, so this can never double up with
                // the loop above.
                for defaulted in self.idc_defaults.get(&owner).into_iter().flatten() {
                    if !attribute_matches(test, defaulted.namespace.as_deref(), &defaulted.local) {
                        continue;
                    }
                    result.push(FieldMatch::Defaulted(defaulted.value.clone()));
                }
            }
        }
        result
    }

    /// The nodes a path's steps reach from `from`, before its optional
    /// attribute step.
    fn walk_steps(&self, from: NodeId, path: &crate::xpath::Path) -> Vec<NodeId> {
        let mut current: Vec<NodeId> = if path.descendant_or_self {
            let mut nodes = vec![from];
            nodes.extend(
                self.doc
                    .descendants(from)
                    .filter(|node| self.doc.kind(*node) == Some(NodeKind::Element)),
            );
            nodes
        } else {
            vec![from]
        };
        for step in &path.steps {
            match step {
                Step::SelfNode => {}
                Step::Child(test) => {
                    let mut next = Vec::new();
                    for node in &current {
                        for child in self.doc.children(*node) {
                            if self.doc.kind(child) != Some(NodeKind::Element) {
                                continue;
                            }
                            if let Some(element) = self.doc.element(child)
                                && test.matches(element.namespace_uri(), element.local_name())
                            {
                                next.push(child);
                            }
                        }
                    }
                    current = next;
                }
            }
        }
        current
    }

    /// Evaluates a restricted-XPath path set from `from`, over the document as
    /// written — which is what a `selector` needs, since it can only ever
    /// yield element nodes.
    fn evaluate(&self, from: NodeId, set: &PathSet) -> Vec<NodeId> {
        let mut result: Vec<NodeId> = Vec::new();
        for path in &set.paths {
            let mut current = self.walk_steps(from, path);
            if let Some(test) = &path.attribute {
                let mut next = Vec::new();
                for node in &current {
                    for attribute in self.doc.attributes(*node) {
                        if let Some(data) = self.doc.attr(attribute)
                            && attribute_matches(test, data.namespace_uri(), data.local_name())
                        {
                            next.push(attribute);
                        }
                    }
                }
                current = next;
            }
            for node in current {
                if !result.contains(&node) {
                    result.push(node);
                }
            }
        }
        result
    }
}

/// One member of the node-set a `field` evaluates to.
///
/// A field selects either something written in the instance — an element or an
/// attribute node — or an attribute the schema supplied through a `default` or
/// `fixed` {value constraint}, which exists only in the augmented infoset and
/// therefore has no node identifier to name it by.
#[derive(Debug, Clone)]
enum FieldMatch {
    Node(NodeId),
    Defaulted(Value),
}

impl FieldMatch {
    /// Whether two matches are the same *node* — used to keep the result of a
    /// path set duplicate-free, as `evaluate` does. Two defaulted values are
    /// never merged: they come from different elements even when they are
    /// equal, which is exactly what makes them a duplicate key.
    fn is_same_node(&self, other: &Self) -> bool {
        matches!((self, other), (Self::Node(left), Self::Node(right)) if left == right)
    }
}

fn attribute_matches(test: &NameTest, namespace: Option<&str>, local: &str) -> bool {
    if namespace == Some("http://www.w3.org/2000/xmlns/") {
        return false;
    }
    test.matches(namespace, local)
}

fn sequences_equal(left: &[Value], right: &[Value]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| values_equal(a, b))
}

/// Equality of two key-sequence members.
///
/// §3.11.4 ends with: "For purposes of checking identity-constraints, single
/// atomic values are not distinguished from lists with single items. An atomic
/// value `V` and a list `L` with a single item are treated as equal … if `V` is
/// equal to the atomic value which is the single item of `L`." So a `keyref`
/// whose field has a list type can match a `key` whose field is atomic
/// (`saxonData/Id/id022`).
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::List(items), other) | (other, Value::List(items))
            if !matches!(other, Value::List(_)) =>
        {
            match items.as_slice() {
                [single] => values_equal(single, other),
                _ => false,
            }
        }
        // Datatypes §3.3.x: for the seven date/time datatypes, a value that
        // carries a timezone offset and one that does not are *incomparable* —
        // the un-timezoned value stands for the whole interval [+14:00,
        // −14:00], which merely overlaps the other rather than equalling it —
        // so they are never `equal or identical` in the sense §3.11.4 clause
        // 4.1/4.2.2/4.3 appeals to (`saxonData/Zone/zone206`'s `n01`, where
        // `12:00:00` must not match the key `12:00:00Z`).
        _ if timezoned(left) != timezoned(right) => false,
        _ => left == right,
    }
}

/// Whether a date/time value carries an explicit timezone offset;
/// [`None`] for every other datatype, where the question does not arise.
fn timezoned(value: &Value) -> Option<bool> {
    Some(match value {
        Value::DateTime(inner) => inner.timezone_offset().is_some(),
        Value::Time(inner) => inner.timezone_offset().is_some(),
        Value::Date(inner) => inner.timezone_offset().is_some(),
        Value::GYearMonth(inner) => inner.timezone_offset().is_some(),
        Value::GYear(inner) => inner.timezone_offset().is_some(),
        Value::GMonthDay(inner) => inner.timezone_offset().is_some(),
        Value::GMonth(inner) => inner.timezone_offset().is_some(),
        Value::GDay(inner) => inner.timezone_offset().is_some(),
        _ => return None,
    })
}
