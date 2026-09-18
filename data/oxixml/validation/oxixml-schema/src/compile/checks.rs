//! The schema component constraints that can only be checked once every
//! component exists: Unique Particle Attribution, Element Declarations
//! Consistent, and the substitution-group rules.

use std::collections::HashMap;

use oxixml_qname::ExpandedName;

use crate::automaton::{ContentModel, LeafTerm};
use crate::components::{ElementId, TypeId, TypeKind};
use crate::error::SchemaError;
use crate::schema::Schema;

use super::Compiler;

impl Compiler {
    pub(crate) fn check_all(&mut self) -> Result<(), SchemaError> {
        self.check_substitution_groups()?;
        self.check_restrictions()?;
        for index in 0..self.schema.types.len() {
            let id = TypeId(u32::try_from(index).unwrap_or(u32::MAX));
            let TypeKind::Complex(complex) = &self.schema.types[index].kind else {
                continue;
            };
            let Some(model) = complex.content_model.clone() else {
                continue;
            };
            check_element_declarations_consistent(&self.schema, id, &model)?;
            if self.schema.version().is_1_1() {
                check_wildcard_type_tables(&self.schema, id, &model)?;
            }
            check_unique_particle_attribution(&self.schema, id, &model)?;
        }
        self.check_single_id_attribute()?;
        self.check_notation_use()?;
        self.check_type_alternatives()?;
        Ok(())
    }

    /// *Element Declaration Properties Correct* clause 7 (§3.3.6.1): "If
    /// `E . {type table}` exists, then for each `{type definition}` T in
    /// `E . {type table} . {alternatives}`, and also for
    /// `E . {type table} . {default type definition} . {type definition}`, one
    /// of the following is true: 7.1 T is ·validly substitutable· for
    /// `E . {type definition}`, subject to the blocking keywords of
    /// `E . {disallowed substitutions}`; 7.2 T is the type `xs:error`."
    ///
    /// (`ibmData/schema_invalid/S3_12/s3_12si03`, `saxonData/CTA/cta9008err`.)
    fn check_type_alternatives(&self) -> Result<(), SchemaError> {
        if !self.schema.version().is_1_1() {
            return Ok(());
        }
        let error_type = self.schema.builtin("error");
        for element in &self.schema.elements {
            for alternative in &element.alternatives {
                if Some(alternative.type_id) == error_type
                    || alternative.type_id == element.type_id
                    || self
                        .schema
                        .is_derived_from(alternative.type_id, element.type_id)
                {
                    continue;
                }
                return Err(SchemaError::new(
                    "e-props-correct.7",
                    format!(
                        "the type `{}` of an `xs:alternative` of `{}` is not validly substitutable \
                         for the element's declared type `{}`",
                        self.schema.type_name(alternative.type_id),
                        element.name,
                        self.schema.type_name(element.type_id)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// *enumeration-required-notation* (Datatypes §3.2.19): `xs:NOTATION` may
    /// not be used directly in a schema — only a type derived from it by
    /// enumerating notation names may be, and each enumerated value must name a
    /// notation the schema declares
    /// (`saxonData/Simple/simple090` … `simple095`).
    fn check_notation_use(&self) -> Result<(), SchemaError> {
        let Some(notation) = self.schema.builtin("NOTATION") else {
            return Ok(());
        };
        let mut used: Vec<TypeId> = Vec::new();
        for element in &self.schema.elements {
            used.push(element.type_id);
        }
        for attribute in &self.schema.attributes {
            used.push(attribute.type_id);
        }
        for type_id in used {
            self.check_notation_type(type_id, notation, 0)?;
        }
        Ok(())
    }

    fn check_notation_type(
        &self,
        type_id: TypeId,
        notation: TypeId,
        depth: usize,
    ) -> Result<(), SchemaError> {
        if depth > 32 {
            return Ok(());
        }
        let Some(simple) = self.schema.types[type_id.index()].as_simple() else {
            return Ok(());
        };
        match &simple.variety {
            crate::components::Variety::List(item) => {
                return self.check_notation_type(*item, notation, depth + 1);
            }
            crate::components::Variety::Union(members) => {
                for member in members {
                    self.check_notation_type(*member, notation, depth + 1)?;
                }
                return Ok(());
            }
            crate::components::Variety::Absent => return Ok(()),
            crate::components::Variety::Atomic(_) => {}
        }
        if !self.schema.is_derived_from(type_id, notation) {
            return Ok(());
        }
        let Some(values) = simple.facets.enumeration.as_ref() else {
            return Err(SchemaError::new(
                "enumeration-required-notation",
                format!(
                    "`{}` may only be used in a schema through a type that enumerates notation \
                     names",
                    self.schema.type_name(type_id)
                ),
            ));
        };
        for value in values {
            let crate::value::Value::QName(namespace, local) = value else {
                continue;
            };
            if self
                .schema
                .find_notation(namespace.as_deref(), local)
                .is_none()
            {
                return Err(SchemaError::new(
                    "enumeration-required-notation",
                    format!("`{local}` is not the name of a declared notation"),
                ));
            }
        }
        Ok(())
    }

    /// *ct-props-correct.5* (§3.4.6) and *ag-props-correct.3* (§3.6.6) of XML
    /// Schema **1.0**: a complex type — and an attribute group definition —
    /// may carry at most one attribute whose type is `xs:ID` or derived from
    /// it. XML Schema 1.1 dropped both clauses, which is why the check is
    /// version-dependent: `msData/complexType/ctM004` is invalid under 1.0 and
    /// valid under 1.1.
    fn check_single_id_attribute(&self) -> Result<(), SchemaError> {
        if self.schema.version().is_1_1() {
            return Ok(());
        }
        let Some(id_type) = self.schema.builtin("ID") else {
            return Ok(());
        };
        let count_ids = |uses: &[crate::components::AttributeUse]| {
            uses.iter()
                .filter(|use_| {
                    use_.kind != crate::components::AttributeUseKind::Prohibited
                        && self.schema.is_derived_from(
                            self.schema.attributes[use_.attribute.index()].type_id,
                            id_type,
                        )
                })
                .count()
        };
        for definition in &self.schema.types {
            let TypeKind::Complex(complex) = &definition.kind else {
                continue;
            };
            if count_ids(&complex.attribute_uses) > 1 {
                return Err(SchemaError::new(
                    "ct-props-correct.5",
                    format!(
                        "`{}` has more than one attribute of type `xs:ID`",
                        definition
                            .name
                            .as_ref()
                            .map_or_else(|| "#anonymous".to_owned(), ToString::to_string)
                    ),
                ));
            }
        }
        for group in &self.schema.attribute_groups {
            if count_ids(&group.attribute_uses) > 1 {
                return Err(SchemaError::new(
                    "ag-props-correct.3",
                    "an attribute group has more than one attribute of type `xs:ID`",
                ));
            }
        }
        Ok(())
    }

    fn check_substitution_groups(&self) -> Result<(), SchemaError> {
        // §3.3.6 *e-props-correct.6*: the `{substitution group affiliation}`
        // chain must not be circular (`msData/additional/test111869`).
        for (index, element) in self.schema.elements.iter().enumerate() {
            let start = ElementId(u32::try_from(index).unwrap_or(u32::MAX));
            let mut frontier: Vec<ElementId> = element.substitution_group.clone();
            let mut seen: Vec<ElementId> = vec![start];
            while let Some(head) = frontier.pop() {
                if head == start {
                    return Err(SchemaError::new(
                        "e-props-correct.6",
                        format!(
                            "`{}` is, through its substitution group, its own head",
                            element.name
                        ),
                    ));
                }
                if seen.contains(&head) {
                    continue;
                }
                seen.push(head);
                frontier.extend(
                    self.schema
                        .element_declaration(head)
                        .substitution_group
                        .iter(),
                );
            }
        }
        for element in &self.schema.elements {
            for head in &element.substitution_group {
                let head_declaration = self.schema.element_declaration(*head);
                if head_declaration.final_.substitution {
                    return Err(SchemaError::new(
                        "cos-equiv-derived-ok-rec.1",
                        format!(
                            "`{}` is `final=\"substitution\"`, so `{}` may not substitute for it",
                            head_declaration.name, element.name
                        ),
                    ));
                }
                if element.type_id != head_declaration.type_id
                    && !self
                        .schema
                        .is_derived_from(element.type_id, head_declaration.type_id)
                {
                    return Err(SchemaError::new(
                        "cos-equiv-derived-ok-rec.2",
                        format!(
                            "the type of `{}` is not derived from the type of its substitution \
                             group head `{}`",
                            element.name, head_declaration.name
                        ),
                    ));
                }
                if let Some(used) = self
                    .schema
                    .derivation_path(element.type_id, head_declaration.type_id)
                {
                    let blocked = head_declaration.final_;
                    if (used.extension && blocked.extension)
                        || (used.restriction && blocked.restriction)
                    {
                        return Err(SchemaError::new(
                            "cos-equiv-derived-ok-rec.3",
                            format!(
                                "the derivation of `{}` from `{}` is blocked by `final`",
                                element.name, head_declaration.name
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// *Element Declarations Consistent* (§3.4.6, `cos-element-consistent`): two
/// element declarations in one content model that share an expanded name must
/// share a type definition.
fn check_element_declarations_consistent(
    schema: &Schema,
    owner: TypeId,
    model: &ContentModel,
) -> Result<(), SchemaError> {
    // The key is the expanded name itself, not a rendering of it: Clark
    // notation is unescaped, so two different `(namespace, local)` pairs can
    // render alike whenever a namespace URI contains a `}`.
    let mut seen: HashMap<ExpandedName, (ElementId, TypeId)> = HashMap::new();
    for element in implicitly_contained(schema, model) {
        let declaration = schema.element_declaration(element);
        let key = declaration.name.clone();
        // XSD 1.1 adds the {type table} to what must agree: two particles of
        // the same name whose conditional type assignment differs would give
        // the same element two governing types
        // (`saxonData/CTA/cta9009err`, `cta9010err`).
        if let Some((first, _)) = seen.get(&key)
            && schema.version().is_1_1()
            && !same_type_table(
                &schema.element_declaration(*first).alternatives,
                &declaration.alternatives,
            )
        {
            return Err(SchemaError::new(
                "cos-element-consistent",
                format!(
                    "`{}` appears twice with different type tables in the content model of `{}`",
                    declaration.name,
                    schema.type_name(owner)
                ),
            ));
        }
        match seen.get(&key) {
            Some((_, type_id)) if *type_id != declaration.type_id => {
                return Err(SchemaError::new(
                    "cos-element-consistent",
                    format!(
                        "`{}` appears twice with different types in the content model of `{}`",
                        declaration.name,
                        schema.type_name(owner)
                    ),
                ));
            }
            Some(_) => {}
            None => {
                seen.insert(key, (element, declaration.type_id));
            }
        }
    }
    Ok(())
}

/// *Element Declarations Consistent* (§3.4.6), the XSD 1.1 clause about
/// wildcards: an element information item that a wildcard admits is assessed
/// against the **global** declaration of its name, so if the same content model
/// also declares that name locally the two must agree about *conditional type
/// assignment* — otherwise the item's governing type would depend on which
/// particle happened to match it.
///
/// Only the {type table} is compared here, which is what the constraint adds
/// over the 1.0 rule and what `saxonData/Wild/wild078.bad`, `wild079.bad` and
/// `wild081.bad` exercise; a plain type mismatch between a local and a global
/// declaration is decided at instance time by *cvc-complex-type.2.4.d*, which
/// XSD 1.1 introduced precisely so that the static rule need not reject it.
fn check_wildcard_type_tables(
    schema: &Schema,
    owner: TypeId,
    model: &ContentModel,
) -> Result<(), SchemaError> {
    let wildcards: Vec<&crate::components::Wildcard> = model_wildcards(model)
        .into_iter()
        .filter(|wildcard| wildcard.process_contents != crate::components::ProcessContents::Skip)
        .collect();
    if wildcards.is_empty() {
        return Ok(());
    }
    for element in model_elements(model) {
        let local = schema.element_declaration(element);
        let Some(global) = schema.find_element(local.name.namespace(), local.name.local_name())
        else {
            continue;
        };
        if global == element {
            continue;
        }
        let global_declaration = schema.element_declaration(global);
        if same_type_table(&local.alternatives, &global_declaration.alternatives) {
            continue;
        }
        let admitted = wildcards.iter().any(|wildcard| {
            wildcard.matches(local.name.namespace(), local.name.local_name(), true)
        });
        if admitted {
            return Err(SchemaError::new(
                "cos-element-consistent",
                format!(
                    "in the content model of `{}`, `{}` is matched both by a wildcard and by an \
                     element declaration whose type table differs from the global declaration's",
                    schema.type_name(owner),
                    local.name
                ),
            ));
        }
    }
    Ok(())
}

/// Whether two {type table}s select the same types under the same tests.
fn same_type_table(
    left: &[crate::components::TypeAlternative],
    right: &[crate::components::TypeAlternative],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.type_id == b.type_id
                && a.test.as_ref().map(|test| &test.test) == b.test.as_ref().map(|test| &test.test)
        })
}

/// Every wildcard of a content model.
fn model_wildcards(model: &ContentModel) -> Vec<&crate::components::Wildcard> {
    let terms: Box<dyn Iterator<Item = &LeafTerm>> = match model {
        ContentModel::Empty => Box::new(std::iter::empty()),
        ContentModel::All(all) => Box::new(all.items.iter().map(|item| &item.term)),
        ContentModel::Fsm(fsm) => Box::new(fsm.positions.iter()),
    };
    terms
        .filter_map(|term| match term {
            LeafTerm::Wildcard(wildcard) => Some(wildcard),
            LeafTerm::Element(_) => None,
        })
        .collect()
}

/// *Unique Particle Attribution* (§3.8.6, `cos-nonambig`).
///
/// An `xs:all` group is not compiled into an automaton, so its members are
/// compared directly: every child of the group competes with every other,
/// because the group imposes no order at all
/// (`saxonData/All/all240.n` … `all243.n`).
fn check_unique_particle_attribution(
    schema: &Schema,
    owner: TypeId,
    model: &ContentModel,
) -> Result<(), SchemaError> {
    if let ContentModel::All(all) = model {
        for (offset, left) in all.items.iter().enumerate() {
            for right in &all.items[offset + 1..] {
                if terms_compete(schema, &left.term, &right.term) {
                    return Err(SchemaError::new(
                        "cos-nonambig",
                        format!(
                            "the `all` group of `{}` is ambiguous: {} and {} can both match the \
                             same element",
                            schema.type_name(owner),
                            describe(schema, &left.term),
                            describe(schema, &right.term)
                        ),
                    ));
                }
            }
        }
        return Ok(());
    }
    for set in model.competing_sets() {
        for (offset, left) in set.iter().enumerate() {
            for right in &set[offset + 1..] {
                if model.position_particle(*left) == model.position_particle(*right) {
                    // Two unrolled copies of one particle: `maxOccurs` does not
                    // create an attribution problem.
                    continue;
                }
                let (Some(a), Some(b)) = (model.position_term(*left), model.position_term(*right))
                else {
                    continue;
                };
                if terms_compete(schema, a, b) {
                    return Err(SchemaError::new(
                        "cos-nonambig",
                        format!(
                            "the content model of `{}` is ambiguous: {} and {} can both match the \
                             same element",
                            schema.type_name(owner),
                            describe(schema, a),
                            describe(schema, b)
                        ),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn terms_compete(schema: &Schema, left: &LeafTerm, right: &LeafTerm) -> bool {
    match (left, right) {
        (LeafTerm::Element(a), LeafTerm::Element(b)) => {
            let left_names = substitution_names(schema, *a);
            let right_names = substitution_names(schema, *b);
            left_names.iter().any(|name| right_names.contains(name))
        }
        // XSD 1.1 §3.8.6.4 makes wildcard particles *weak*: an element
        // information item that matches both an element declaration and a
        // wildcard is attributed to the declaration, so the two never compete
        // and the model stays deterministic. XSD 1.0 had no such rule, and
        // rejected the overlap outright (`ibmData/valid/S3_10_1/s3_10_1v04`,
        // `saxonData/Wild/wild063`).
        (LeafTerm::Element(_), LeafTerm::Wildcard(_))
        | (LeafTerm::Wildcard(_), LeafTerm::Element(_))
            if schema.version().is_1_1() =>
        {
            false
        }
        (LeafTerm::Element(element), LeafTerm::Wildcard(wildcard))
        | (LeafTerm::Wildcard(wildcard), LeafTerm::Element(element)) => {
            substitution_namespaces(schema, *element)
                .into_iter()
                .any(|namespace| wildcard.namespace.allows(namespace.as_deref()))
        }
        (LeafTerm::Wildcard(a), LeafTerm::Wildcard(b)) => a.namespace.intersects(&b.namespace),
    }
}

/// The members of `element`'s substitution group that can actually stand in for
/// it — its *actual* substitution group.
///
/// A head whose `{disallowed substitutions}` contains `substitution` cannot be
/// stood in for at all (§3.3.6 *cvc-elt.4*), so it contributes no members and
/// cannot make a content model ambiguous (`msData/element/elemZ028a`). What
/// remains is filtered by [`Schema::is_actual_substitution_member`], which is
/// where XML Schema 1.0 and 1.1 part company over `abstract` members:
/// `wgData/sg/upa.xsd` is written to pin exactly that down — an abstract global
/// `e1` in the substitution group of `e`, competing with an unrelated local
/// `e1`, is a legal content model under 1.0 and not under 1.1.
fn actual_substitution_members(schema: &Schema, element: ElementId) -> Vec<ElementId> {
    let declaration = schema.element_declaration(element);
    if declaration.block.substitution {
        return Vec::new();
    }
    declaration
        .substitutable_by
        .iter()
        .copied()
        .filter(|member| schema.is_actual_substitution_member(*member))
        .collect()
}

/// The names a particle referring to `element` can match.
fn substitution_names(schema: &Schema, element: ElementId) -> Vec<ExpandedName> {
    let mut names = vec![schema.element_declaration(element).name.clone()];
    names.extend(
        actual_substitution_members(schema, element)
            .into_iter()
            .map(|member| schema.element_declaration(member).name.clone()),
    );
    names
}

fn substitution_namespaces(schema: &Schema, element: ElementId) -> Vec<Option<String>> {
    let declaration = schema.element_declaration(element);
    let mut namespaces = vec![declaration.name.namespace().map(str::to_owned)];
    for member in actual_substitution_members(schema, element) {
        namespaces.push(
            schema
                .element_declaration(member)
                .name
                .namespace()
                .map(str::to_owned),
        );
    }
    namespaces
}

fn describe(schema: &Schema, term: &LeafTerm) -> String {
    match term {
        LeafTerm::Element(id) => format!("`{}`", schema.element_declaration(*id).name),
        LeafTerm::Wildcard(_) => "a wildcard".to_owned(),
    }
}

/// Every element declaration a content model ·contains·, directly or
/// ·implicitly· (§3.8.6.3): "a list of particles **implicitly contains** an
/// element declaration if and only if a member of the list contains that
/// element declaration in its ·substitution group·".
///
/// This is what makes a locally declared `n` conflict with a global `n` that
/// stands in a substitution group the same model refers to
/// (`saxonData/Subsgroup/subsgroup901`). Only the ·actual· substitution group
/// counts, exactly as in [`substitution_names`].
fn implicitly_contained(schema: &Schema, model: &ContentModel) -> Vec<ElementId> {
    let mut elements = Vec::new();
    for element in model_elements(model) {
        elements.push(element);
        for member in actual_substitution_members(schema, element) {
            if !elements.contains(&member) {
                elements.push(member);
            }
        }
    }
    elements
}

/// Every element declaration reachable from a content model.
fn model_elements(model: &ContentModel) -> Vec<ElementId> {
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
