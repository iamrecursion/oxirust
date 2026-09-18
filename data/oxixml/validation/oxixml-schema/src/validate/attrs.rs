//! Attribute assessment: `cvc-attribute`, `cvc-complex-type.3` and
//! `cvc-complex-type.4`.

use oxixml_dom::NodeId;

use crate::components::{AttributeUse, AttributeUseKind, ProcessContents, TypeId, TypeKind};
use crate::names::{XSD_NS, XSI_NS};
use crate::simple;

use super::{AttributePsvi, DefaultedAttribute, Validator, Validity, qualified};

/// The namespace that namespace declarations themselves live in.
const XMLNS_NS: &str = "http://www.w3.org/2000/xmlns/";

/// The four attributes an instance may carry from the XSI namespace.
const XSI_ATTRIBUTES: &[&str] = &["type", "nil", "schemaLocation", "noNamespaceSchemaLocation"];

impl<'a> Validator<'a> {
    pub(crate) fn assess_attributes(&mut self, node: NodeId, type_id: TypeId) {
        let path = self.path(node);
        let instance: Vec<NodeId> = self.doc.attributes(node).collect();

        let (uses, wildcard) = match &self.schema.type_definition(type_id).kind {
            TypeKind::Simple(_) => (Vec::new(), None),
            TypeKind::Complex(complex) => (
                complex.attribute_uses.clone(),
                complex.attribute_wildcard.clone(),
            ),
        };
        let is_simple = self.schema.type_definition(type_id).is_simple();

        let mut matched = vec![false; uses.len()];
        // §3.4.4 clause 5 (XSD 1.0 only): the ·wild IDs· of the element — the
        // attributes clause 3.2 let a wildcard admit whose names resolve to a
        // declaration of a type derived from `ID`.
        let mut wild_ids: Vec<String> = Vec::new();
        for attribute in instance {
            let Some(data) = self.doc.attr(attribute) else {
                continue;
            };
            let namespace = data.namespace_uri().map(str::to_owned);
            let local = data.local_name().to_owned();
            let value = data.value().to_owned();
            if namespace.as_deref() == Some(XMLNS_NS) || local == "xmlns" {
                continue;
            }
            if namespace.as_deref() == Some(XSI_NS) {
                self.check_xsi_attribute(attribute, &local);
                // The four `xsi:` attributes are exempt from the attribute
                // rules, but a type may still *declare* one — `<xs:attribute
                // ref="xsi:type" use="required"/>` — and then the requirement
                // is satisfied by the attribute being here
                // (`saxonData/Complex/complex009`, `complex010`).
                if let Some(position) = uses.iter().position(|use_| {
                    let declaration = self.schema.attribute_declaration(use_.attribute);
                    declaration.name.namespace() == Some(XSI_NS)
                        && declaration.name.local_name() == local
                }) {
                    matched[position] = true;
                }
                continue;
            }
            let attribute_path = self.path(attribute);
            if is_simple {
                self.error(
                    "cvc-type.3.1.1",
                    &attribute_path,
                    "an element with a simple type may not carry attributes",
                );
                continue;
            }

            // §3.4.1: an attribute use with `use="prohibited"` contributes no
            // *attribute declaration* to `{attribute uses}`, so it does not
            // match the attribute at all — the attribute wildcard, if there is
            // one, gets its turn (`msData/attribute/attZ002`), and only when
            // there is none does *cvc-complex-type.3.2.2* apply.
            let position = uses.iter().position(|use_| {
                if use_.kind == AttributeUseKind::Prohibited {
                    return false;
                }
                let declaration = self.schema.attribute_declaration(use_.attribute);
                declaration.name.namespace() == namespace.as_deref()
                    && declaration.name.local_name() == local
            });
            match position {
                Some(position) => {
                    matched[position] = true;
                    self.validate_attribute_use(
                        attribute,
                        &uses[position],
                        &value,
                        &attribute_path,
                    );
                }
                None => match &wildcard {
                    Some(wildcard)
                        if wildcard.matches(
                            namespace.as_deref(),
                            &local,
                            wildcard.disallowed.defined
                                && self.is_declared_attribute(namespace.as_deref(), &local),
                        ) =>
                    {
                        if wildcard.process_contents != ProcessContents::Skip
                            && self.is_wild_id(namespace.as_deref(), &local)
                        {
                            wild_ids.push(attribute_path.clone());
                        }
                        self.validate_wildcard_attribute(
                            attribute,
                            wildcard.process_contents,
                            namespace.as_deref(),
                            &local,
                            &value,
                            &attribute_path,
                        );
                    }
                    _ => self.error(
                        "cvc-complex-type.3.2.2",
                        &attribute_path,
                        format!(
                            "the attribute `{}` is not allowed here",
                            qualified(namespace.as_deref(), &local)
                        ),
                    ),
                },
            }
        }

        self.check_wild_ids(&wild_ids, &uses, &path);

        for (position, use_) in uses.iter().enumerate() {
            if matched[position] {
                continue;
            }
            // §3.4.1 again, from the other side: a `use="prohibited"` use is
            // not an *attribute use* of the type at all, so it neither demands
            // an attribute nor supplies one. A `default` or `fixed` written
            // beside the prohibition is inert — nothing of it may reach the
            // augmented infoset, where it would otherwise show up as a
            // defaulted attribute, feed the ID/IDREF table and be selected by
            // an identity-constraint `field`.
            if use_.kind == AttributeUseKind::Prohibited {
                continue;
            }
            let declaration = self.schema.attribute_declaration(use_.attribute);
            if use_.kind == AttributeUseKind::Required {
                self.error(
                    "cvc-complex-type.4",
                    &path,
                    format!("the required attribute `{}` is missing", declaration.name),
                );
                continue;
            }
            let constraint = use_
                .value_constraint
                .as_ref()
                .or(declaration.value_constraint.as_ref());
            if let Some(constraint) = constraint {
                self.defaulted.entry(node).or_default().push((
                    declaration.name.local_name().to_owned(),
                    constraint.normalized.clone(),
                ));
                // §3.4.5 *Attribute Default Value* adds the defaulted
                // attribute to the element's \[attributes\] in the
                // post-schema-validation infoset, which is what an
                // identity-constraint `field` runs over — so the value has to
                // be recorded where the field evaluator can see it, together
                // with the name that selects it and the value it denotes
                // (`msData/identityConstraint/idF016`, `idF017`, `idZ011_a`).
                let value = constraint.value.clone().or_else(|| {
                    simple::validate(
                        self.schema,
                        declaration.type_id,
                        &constraint.lexical,
                        &constraint.namespaces,
                    )
                    .ok()
                    .map(|result| result.value)
                });
                if let Some(value) = value {
                    self.idc_defaults
                        .entry(node)
                        .or_default()
                        .push(DefaultedAttribute {
                            namespace: declaration.name.namespace().map(str::to_owned),
                            local: declaration.name.local_name().to_owned(),
                            value,
                        });
                }
                // §3.3.4 *cvc-id*: a defaulted attribute contributes its value
                // to the ID/IDREF table exactly as a written one does, so a
                // `default` of type `xs:ID` can clash with a real one
                // (`saxonData/Id/id010`).
                let type_id = declaration.type_id;
                let normalized = constraint.normalized.clone();
                self.note_identifier(type_id, &normalized, Some(node), node, &path);
            }
        }
    }

    /// Whether an attribute admitted by a wildcard is one of the element's
    /// ·wild IDs· — whether its name "resolves … to an attribute declaration
    /// whose {type definition} is or is derived from `ID`" (§3.4.4 clause 5).
    fn is_wild_id(&self, namespace: Option<&str>, local: &str) -> bool {
        let Some(id_type) = self.schema.builtin("ID") else {
            return false;
        };
        self.schema
            .find_attribute(namespace, local)
            .is_some_and(|declaration| {
                let type_id = self.schema.attribute_declaration(declaration).type_id;
                self.schema.is_derived_from(type_id, id_type)
            })
    }

    /// *cvc-complex-type* clause 5, which XSD 1.0 imposes and XSD 1.1 drops.
    ///
    /// "5.1 There must be no more than one item in ·wild IDs·. 5.2 If ·wild
    /// IDs· is non-empty, there must not be any attribute uses among the
    /// {attribute uses} whose {attribute declaration}'s {type definition} is
    /// or is derived from `ID`." Both exist because XSD 1.0 lets an element
    /// carry at most one `ID`, and a wildcard is the one way an attribute can
    /// reach an element without the type knowing about it. XSD 1.1 allows an
    /// element several `ID` attributes outright, so the clause is gone there
    /// (`msData/attribute/attZ014a`, `attZ014b`, both invalid under 1.0 and
    /// valid under 1.1).
    fn check_wild_ids(&mut self, wild_ids: &[String], uses: &[AttributeUse], path: &str) {
        if wild_ids.is_empty() || self.schema.version().is_1_1() {
            return;
        }
        if wild_ids.len() > 1 {
            self.error(
                "cvc-complex-type.5.1",
                &wild_ids[1],
                "an element may carry only one attribute of type `ID`",
            );
            return;
        }
        let declared_id = uses.iter().any(|use_| {
            if use_.kind == AttributeUseKind::Prohibited {
                return false;
            }
            let type_id = self.schema.attribute_declaration(use_.attribute).type_id;
            self.schema
                .builtin("ID")
                .is_some_and(|id_type| self.schema.is_derived_from(type_id, id_type))
        });
        if declared_id {
            self.error(
                "cvc-complex-type.5.2",
                path,
                "an attribute of type `ID` reached this element through a wildcard, and its type \
                 already declares one",
            );
        }
    }

    /// Whether an attribute of this name is *declared* by the schema, for
    /// `notQName="##defined"` (XSD 1.1 §3.10.1).
    ///
    /// The `xml:` and `xsi:` attributes this crate makes available to every
    /// schema are not declared *by* a schema document, so `##defined` does not
    /// exclude them (`saxonData/Wild/wild054`).
    fn is_declared_attribute(&self, namespace: Option<&str>, local: &str) -> bool {
        if matches!(namespace, Some(crate::names::XML_NS | XSI_NS)) {
            return false;
        }
        self.schema.find_attribute(namespace, local).is_some()
    }

    /// The four `xsi:` attributes are exempt from the attribute rules of a
    /// complex type (§3.4.4.2, *cvc-complex-type.3*).
    ///
    /// XSD 1.0 §3.2.7 also makes *any other* attribute in that namespace an
    /// error; 1.1 dropped that clause, so under 1.1 an unrecognized `xsi:`
    /// attribute is simply not one of the exempt four and is otherwise ignored
    /// (`saxonData/Wild/wild042`).
    fn check_xsi_attribute(&mut self, attribute: NodeId, local: &str) {
        if self.schema.version().is_1_1() {
            return;
        }
        if !XSI_ATTRIBUTES.contains(&local) {
            let path = self.path(attribute);
            self.error(
                "cvc-complex-type.3.2.2",
                &path,
                format!("`xsi:{local}` is not one of the four instance attributes"),
            );
        }
    }

    fn validate_attribute_use(
        &mut self,
        attribute: NodeId,
        use_: &AttributeUse,
        value: &str,
        path: &str,
    ) {
        if use_.kind == AttributeUseKind::Prohibited {
            self.error(
                "cvc-complex-type.3.2.1",
                path,
                "this attribute is prohibited by the type",
            );
            return;
        }
        let declaration = self.schema.attribute_declaration(use_.attribute).clone();
        let context = crate::parse::namespace_context(self.doc, attribute);
        let outcome = simple::validate(self.schema, declaration.type_id, value, &context);
        match outcome {
            Ok(result) => {
                // §3.3.4 binds an ID to the *element* that carries the
                // attribute, not to the attribute node.
                let owner = self.doc.owner_element(attribute).unwrap_or(attribute);
                self.note_identifier(
                    declaration.type_id,
                    &result.normalized,
                    Some(owner),
                    attribute,
                    path,
                );
                let constraint = use_
                    .value_constraint
                    .as_ref()
                    .or(declaration.value_constraint.as_ref());
                if let Some(constraint) = constraint
                    && constraint.fixed
                {
                    let expected = simple::validate(
                        self.schema,
                        declaration.type_id,
                        &constraint.lexical,
                        &constraint.namespaces,
                    );
                    let equal = match expected {
                        Ok(expected) => expected.value == result.value,
                        Err(_) => false,
                    };
                    if !equal {
                        self.error(
                            "cvc-attribute.4",
                            path,
                            format!(
                                "the attribute must equal its fixed value `{}`",
                                constraint.lexical
                            ),
                        );
                    }
                }
                self.attributes.insert(
                    attribute,
                    AttributePsvi {
                        declaration: Some(use_.attribute),
                        type_definition: Some(declaration.type_id),
                        validity: Validity::Valid,
                        normalized_value: Some(result.normalized),
                        value: Some(result.value),
                        member_type: result.member_type,
                        inheritable: use_.inheritable,
                    },
                );
            }
            Err(violation) => {
                self.error(violation.code, path, violation.message);
                self.attributes.insert(
                    attribute,
                    AttributePsvi {
                        declaration: Some(use_.attribute),
                        type_definition: Some(declaration.type_id),
                        validity: Validity::Invalid,
                        normalized_value: Some(value.to_owned()),
                        value: None,
                        member_type: None,
                        inheritable: use_.inheritable,
                    },
                );
            }
        }
    }

    fn validate_wildcard_attribute(
        &mut self,
        attribute: NodeId,
        process: ProcessContents,
        namespace: Option<&str>,
        local: &str,
        value: &str,
        path: &str,
    ) {
        match process {
            ProcessContents::Skip => {
                // §3.10.4.1: "an element or attribute information item is
                // skipped if it is ·attributed to· a skip wildcard" — and a
                // ·skipped· node takes no part in an identity constraint
                // (§3.11.4 clause 3), so a `field` that resolves to an
                // attribute only a skip wildcard allows selects nothing
                // (`msData/identityConstraint/idZ015`, whose `xs:key` is then
                // short of a field).
                self.skipped.insert(attribute);
                self.attributes.insert(
                    attribute,
                    AttributePsvi {
                        declaration: None,
                        type_definition: None,
                        validity: Validity::NotKnown,
                        normalized_value: Some(value.to_owned()),
                        value: None,
                        member_type: None,
                        inheritable: false,
                    },
                );
            }
            ProcessContents::Lax | ProcessContents::Strict => {
                match self.schema.find_attribute(namespace, local) {
                    Some(id) => {
                        // A wildcard has no `xs:attribute` particle of its own
                        // to carry an `inheritable` override, so the use
                        // simply takes the declaration's own `{inheritable}`.
                        let use_ = AttributeUse {
                            attribute: id,
                            kind: AttributeUseKind::Optional,
                            value_constraint: None,
                            inheritable: self.schema.attribute_declaration(id).inheritable,
                        };
                        self.validate_attribute_use(attribute, &use_, value, path);
                    }
                    None => {
                        // §4.3.2: a declaration the instance's own location
                        // hints supplied stands behind the assembled schema,
                        // and satisfies a `strict` wildcard just as one of this
                        // schema's own would.
                        if self.delegate_attribute(attribute, namespace, local, value, path) {
                            return;
                        }
                        if process == ProcessContents::Strict && namespace != Some(XSD_NS) {
                            self.error(
                                "cvc-complex-type.3.2.2",
                                path,
                                format!(
                                    "`{}` is matched by a strict wildcard but has no declaration",
                                    qualified(namespace, local)
                                ),
                            );
                        }
                        self.attributes.insert(
                            attribute,
                            AttributePsvi {
                                declaration: None,
                                type_definition: None,
                                validity: Validity::NotKnown,
                                normalized_value: Some(value.to_owned()),
                                value: None,
                                member_type: None,
                                inheritable: false,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Assesses one attribute against a declaration the instance's own location
    /// hints supplied, when they have one for this name.
    ///
    /// Reports whether it did. Unlike an element, an attribute has no subtree,
    /// so the hand-over is a single value check — but it still runs against the
    /// hinted schema, whose type definitions are the only ones its declaration
    /// can name.
    fn delegate_attribute(
        &mut self,
        attribute: NodeId,
        namespace: Option<&str>,
        local: &str,
        value: &str,
        path: &str,
    ) -> bool {
        let Some(hints) = self.hints else {
            return false;
        };
        let Some(id) = hints.find_attribute(namespace, local) else {
            return false;
        };
        let mut nested = Validator::new_nested(hints, self.doc, self.depth);
        let use_ = AttributeUse {
            attribute: id,
            kind: AttributeUseKind::Optional,
            value_constraint: None,
            inheritable: hints.attribute_declaration(id).inheritable,
        };
        nested.validate_attribute_use(attribute, &use_, value, path);
        self.absorb(nested);
        true
    }
}
