//! Building particles, element and attribute declarations, model groups,
//! attribute groups and identity constraints.

use std::rc::Rc;

use oxixml_dom::{Document, NodeId};
use oxixml_qname::ExpandedName;

use crate::components::{
    AttrGroupId, AttributeDeclaration, AttributeId, AttributeUse, AttributeUseKind, Compositor,
    DerivationSet, ElementDeclaration, ElementId, GroupId, IdcCategory, IdcId,
    IdentityConstraintDefinition, MaxOccurs, ModelGroup, Particle, Term, TypeId, ValueConstraint,
    Wildcard,
};
use crate::error::SchemaError;
use crate::parse::{
    attr, attr_token, bool_attr, content_children, derivation_set, max_occurs, min_occurs,
    namespace_context, node_path, resolve_qname, xsd_name,
};
use crate::xpath::PathSet;

use super::{BuildState, Compiler, DocInfo, Redefine, RedefineKind};

impl Compiler {
    /// Builds the particle written at `node`, which must be one of
    /// `xs:element`, `xs:group`, `xs:all`, `xs:choice`, `xs:sequence` or
    /// `xs:any`.
    pub(crate) fn build_particle(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<Particle, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let min = min_occurs(&doc, node)?;
        let max = max_occurs(&doc, node)?;
        if let MaxOccurs::Bounded(bound) = max
            && bound < min
        {
            return Err(SchemaError::new(
                "p-props-correct.2.1",
                format!("`minOccurs` ({min}) is greater than `maxOccurs` ({bound})"),
            )
            .at(node_path(&doc, node)));
        }
        let term = match xsd_name(&doc, node) {
            Some("element") => Term::Element(self.build_element_particle(info, node, redefine)?),
            Some("any") => Term::Wildcard(crate::parse::wildcard(
                &doc,
                node,
                info.target_namespace.as_deref(),
                self.schema.version(),
            )?),
            Some("group") => {
                let reference = attr(&doc, node, "ref").ok_or_else(|| {
                    SchemaError::new("src-element", "a `group` particle needs a `ref`")
                        .at(node_path(&doc, node))
                })?;
                let name = self.reference(info, node, reference, redefine, RedefineKind::Group)?;
                let id = self
                    .schema
                    .global_groups
                    .get(&name)
                    .copied()
                    .ok_or_else(|| {
                        SchemaError::new(
                            "src-resolve",
                            format!("the model group `{name}` is not defined"),
                        )
                        .at(node_path(&doc, node))
                    })?;
                self.ensure_group(id)?;
                let group = self.schema.groups[id.index()]
                    .model_group
                    .clone()
                    .unwrap_or(ModelGroup {
                        compositor: Compositor::Sequence,
                        particles: Vec::new(),
                    });
                Term::Group(group)
            }
            Some(compositor @ ("all" | "choice" | "sequence")) => {
                let compositor = match compositor {
                    "all" => Compositor::All,
                    "choice" => Compositor::Choice,
                    _ => Compositor::Sequence,
                };
                let mut particles = Vec::new();
                for child in content_children(&doc, node) {
                    particles.push(self.build_particle(info, child, redefine)?);
                }
                Term::Group(ModelGroup {
                    compositor,
                    particles,
                })
            }
            other => {
                return Err(SchemaError::new(
                    "src-element",
                    format!("`xs:{}` is not a particle", other.unwrap_or("?")),
                )
                .at(node_path(&doc, node)));
            }
        };
        Ok(Particle {
            min_occurs: min,
            max_occurs: max,
            term,
        })
    }

    /// Builds the named model group definition `id`, if it has not been built.
    pub(crate) fn ensure_group(&mut self, id: GroupId) -> Result<(), SchemaError> {
        match self.group_state[id.index()] {
            BuildState::Built => return Ok(()),
            BuildState::Building => {
                return Err(SchemaError::new(
                    "src-group",
                    format!(
                        "the model group `{}` refers to itself",
                        self.schema.groups[id.index()].name
                    ),
                ));
            }
            BuildState::NotBuilt => {}
        }
        let Some(source) = self.group_source.get(&id.0).cloned() else {
            self.group_state[id.index()] = BuildState::Built;
            return Ok(());
        };
        self.group_state[id.index()] = BuildState::Building;
        let info = self.info(source.doc);
        let doc = Rc::clone(&info.doc);
        let children = content_children(&doc, source.node);
        let [child] = children.as_slice() else {
            return Err(SchemaError::new(
                "src-group",
                "a top-level `xs:group` needs exactly one `all`, `choice` or `sequence`",
            )
            .at(node_path(&doc, source.node)));
        };
        self.enter_definition("model group")?;
        let particle = self.build_particle(&info, *child, source.redefine.as_ref());
        self.leave_definition();
        let particle = particle?;
        let group = match particle.term {
            Term::Group(group) => group,
            other => ModelGroup {
                compositor: Compositor::Sequence,
                particles: vec![Particle {
                    min_occurs: particle.min_occurs,
                    max_occurs: particle.max_occurs,
                    term: other,
                }],
            },
        };
        self.schema.groups[id.index()].model_group = Some(group);
        self.group_state[id.index()] = BuildState::Built;
        Ok(())
    }

    /// Builds the attribute group definition `id`, if it has not been built.
    pub(crate) fn ensure_attribute_group(&mut self, id: AttrGroupId) -> Result<(), SchemaError> {
        match self.attr_group_state[id.index()] {
            BuildState::Built => return Ok(()),
            BuildState::Building => {
                // XSD 1.1 §3.6.6 dropped the 1.0 prohibition (`src-attribute_group.3`)
                // on an attribute group referring to itself (W3C bug 15795). The
                // {attribute uses} of an attribute group are the least fixed point
                // of the reference graph, so a reference that closes a cycle
                // contributes nothing the surrounding collection is not already
                // gathering — the reference is simply skipped
                // (`msData/attributeGroup/attgC010`, `attgC020`, `attgC031`,
                // `attgD015`). Under 1.0 it stays an error.
                if self.schema.version().is_1_1() {
                    return Ok(());
                }
                return Err(SchemaError::new(
                    "src-attribute_group.3",
                    "an attribute group refers to itself",
                ));
            }
            BuildState::NotBuilt => {}
        }
        let Some(source) = self.attr_group_source.get(&id.0).cloned() else {
            self.attr_group_state[id.index()] = BuildState::Built;
            return Ok(());
        };
        self.attr_group_state[id.index()] = BuildState::Building;
        let info = self.info(source.doc);
        let doc = Rc::clone(&info.doc);
        let children = content_children(&doc, source.node);
        self.enter_definition("attribute group")?;
        // Record references this group makes so a cycle broken above can be
        // closed to its fixed point once every group is built.
        let outer = self.current_attr_group.replace(id);
        let built = self.build_attribute_uses(&info, &children, source.redefine.as_ref());
        self.current_attr_group = outer;
        self.leave_definition();
        let (uses, wildcard) = built?;
        let definition = &mut self.schema.attribute_groups[id.index()];
        definition.attribute_uses = uses;
        definition.attribute_wildcard = wildcard;
        self.attr_group_state[id.index()] = BuildState::Built;
        Ok(())
    }

    /// Completes the {attribute uses} of attribute groups that take part in an
    /// XSD 1.1 reference cycle.
    ///
    /// The first build breaks a cycle by treating the back-reference as empty
    /// (see [`Compiler::ensure_attribute_group`]), which leaves the group
    /// entered *second* short of the attributes reachable only past the
    /// back-edge. Unioning each group's referents into it and iterating to
    /// stability repairs that: §3.6.6 makes the uses the least fixed point of
    /// the reference graph, and since uses only ever grow — bounded by the
    /// number of declarations — the iteration terminates. An attribute already
    /// present by name is left as it is, so a declaration reached twice around a
    /// cycle is not duplicated; a genuine clash of two distinct declarations
    /// sharing a name has already been reported against the group that reaches
    /// both directly. Under XSD 1.0 a cycle is an error, so there is nothing to
    /// close.
    pub(crate) fn close_attribute_group_cycles(&mut self) {
        if !self.schema.version().is_1_1() || self.attr_group_edges.is_empty() {
            return;
        }
        let edges = self.attr_group_edges.clone();
        loop {
            let mut changed = false;
            for &(from, to) in &edges {
                if from == to {
                    continue;
                }
                let members = self.schema.attribute_groups[to.index()]
                    .attribute_uses
                    .clone();
                for member in members {
                    if member.kind == AttributeUseKind::Prohibited {
                        continue;
                    }
                    let name = self.schema.attributes[member.attribute.index()]
                        .name
                        .clone();
                    let present = self.schema.attribute_groups[from.index()]
                        .attribute_uses
                        .iter()
                        .any(|existing| {
                            self.schema.attributes[existing.attribute.index()].name == name
                        });
                    if present {
                        continue;
                    }
                    self.schema.attribute_groups[from.index()]
                        .attribute_uses
                        .push(member);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Collects the attribute uses and the attribute wildcard contributed by a
    /// run of `attribute`, `attributeGroup` and `anyAttribute` children.
    pub(crate) fn build_attribute_uses(
        &mut self,
        info: &DocInfo,
        nodes: &[NodeId],
        redefine: Option<&Redefine>,
    ) -> Result<(Vec<AttributeUse>, Option<Wildcard>), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let mut uses: Vec<AttributeUse> = Vec::new();
        // §3.4.2 keeps the *local wildcard* — what an `<anyAttribute>` written
        // here says — apart from the wildcards reached through
        // `<attributeGroup>` references, because the *complete wildcard* is
        // their intersection and takes its {process contents} from the local
        // one alone. Accumulating them in one variable would let whichever
        // came last in document order overwrite the other.
        let mut group_wildcard: Option<Wildcard> = None;
        let mut local_wildcard: Option<Wildcard> = None;
        for node in nodes {
            match xsd_name(&doc, *node) {
                Some("attribute") => {
                    let use_ = self.build_attribute_use(info, *node, redefine)?;
                    let name = self.schema.attributes[use_.attribute.index()].name.clone();
                    if uses.iter().any(|existing| {
                        self.schema.attributes[existing.attribute.index()].name == name
                    }) {
                        return Err(SchemaError::new(
                            "ct-props-correct.4",
                            format!("the attribute `{name}` is used twice"),
                        )
                        .at(node_path(&doc, *node)));
                    }
                    uses.push(use_);
                }
                Some("attributeGroup") => {
                    let reference = attr(&doc, *node, "ref").ok_or_else(|| {
                        SchemaError::new(
                            "src-attribute_group",
                            "an `attributeGroup` reference needs a `ref`",
                        )
                        .at(node_path(&doc, *node))
                    })?;
                    let name = self.reference(
                        info,
                        *node,
                        reference,
                        redefine,
                        RedefineKind::AttributeGroup,
                    )?;
                    let id = self
                        .schema
                        .global_attribute_groups
                        .get(&name)
                        .copied()
                        .ok_or_else(|| {
                            SchemaError::new(
                                "src-resolve",
                                format!("the attribute group `{name}` is not defined"),
                            )
                            .at(node_path(&doc, *node))
                        })?;
                    if let Some(from) = self.current_attr_group {
                        self.attr_group_edges.push((from, id));
                    }
                    self.ensure_attribute_group(id)?;
                    let definition = self.schema.attribute_groups[id.index()].clone();
                    for member in definition.attribute_uses {
                        // §3.4.2 clause 3 excludes the base type's attribute
                        // uses only for the `<attribute>` **children** of the
                        // derivation that carry `use="prohibited"`; one
                        // reached through an `<attributeGroup>` reference does
                        // not prohibit anything, so it is simply dropped here
                        // (`msData/attribute/attZ015`).
                        if member.kind == AttributeUseKind::Prohibited {
                            continue;
                        }
                        let member_name = self.schema.attributes[member.attribute.index()]
                            .name
                            .clone();
                        // §3.4.6 *ct-props-correct.4* and §3.6.6
                        // *ag-props-correct.2*: two **distinct** attribute
                        // declarations in one `{attribute uses}` may not share
                        // an expanded name. Reaching the *same* declaration
                        // twice — through two attribute groups that both refer
                        // to a third — is not an error, and is exactly what
                        // the wording "two distinct attribute declarations"
                        // permits.
                        if let Some(existing) = uses.iter().find(|existing| {
                            self.schema.attributes[existing.attribute.index()].name == member_name
                        }) {
                            if existing.attribute == member.attribute {
                                continue;
                            }
                            return Err(SchemaError::new(
                                "ct-props-correct.4",
                                format!(
                                    "two different attribute declarations named `{member_name}` \
                                     are in scope here"
                                ),
                            )
                            .at(node_path(&doc, *node)));
                        }
                        uses.push(member);
                    }
                    if let Some(referenced) = definition.attribute_wildcard {
                        group_wildcard = Some(match group_wildcard {
                            // §3.4.2 clause 2.2.3 takes the {process contents}
                            // of the first non-absent pointed-to wildcard when
                            // there is no local one to take it from.
                            Some(existing) => intersect_wildcards(&existing, &referenced),
                            None => referenced,
                        });
                    }
                }
                Some("anyAttribute") => {
                    let own = crate::parse::wildcard(
                        &doc,
                        *node,
                        info.target_namespace.as_deref(),
                        self.schema.version(),
                    )?;
                    local_wildcard = Some(own);
                }
                _ => {}
            }
        }
        // §3.4.2 clause 2, the *complete wildcard*: with attribute groups in
        // play it is the intersection of the local wildcard with every
        // wildcard they contribute, and its {process contents} is the local
        // wildcard's whenever there is one (`sunData/combined/007`,
        // `msData/wildcards/wildZ011`).
        let wildcard = match (local_wildcard, group_wildcard) {
            (None, referenced) => referenced,
            (Some(local), None) => Some(local),
            (Some(local), Some(referenced)) => Some(intersect_wildcards(&local, &referenced)),
        };
        Ok((uses, wildcard))
    }

    fn build_attribute_use(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<AttributeUse, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let declared_use = attr_token(&doc, node, "use");
        let kind = match declared_use.as_deref() {
            None | Some("optional") => AttributeUseKind::Optional,
            Some("required") => AttributeUseKind::Required,
            Some("prohibited") => AttributeUseKind::Prohibited,
            Some(other) => {
                return Err(SchemaError::new(
                    "src-attribute",
                    format!("`{other}` is not an attribute `use`"),
                )
                .at(node_path(&doc, node)));
            }
        };
        let default = attr(&doc, node, "default").map(str::to_owned);
        let fixed = attr(&doc, node, "fixed").map(str::to_owned);
        if default.is_some() && fixed.is_some() {
            return Err(SchemaError::new(
                "src-attribute.1",
                "an attribute may not have both `default` and `fixed`",
            )
            .at(node_path(&doc, node)));
        }
        if default.is_some() && !matches!(declared_use.as_deref(), None | Some("optional")) {
            return Err(SchemaError::new(
                "src-attribute.2",
                "an attribute with a `default` must have `use=\"optional\"`",
            )
            .at(node_path(&doc, node)));
        }
        // XSD 1.1 §3.2.3 *src-attribute.4*: an attribute use that prohibits the
        // attribute has no value to constrain, so a `fixed` on it says nothing
        // and is an error. XSD 1.0 tolerated the combination — it simply
        // ignored the value constraint — which is why the suite splits the
        // verdict by version (`msData/attribute/attKb009`, `attKc009`,
        // `attP029`; W3C bug 14245).
        if self.schema.version().is_1_1() && kind == AttributeUseKind::Prohibited && fixed.is_some()
        {
            return Err(SchemaError::new(
                "src-attribute.4",
                "an attribute with `use=\"prohibited\"` may not also have a `fixed` value",
            )
            .at(node_path(&doc, node)));
        }
        let value_constraint = match (&default, &fixed) {
            (Some(text), _) => Some(ValueConstraint::new(
                false,
                text.clone(),
                namespace_context(&doc, node),
            )),
            (_, Some(text)) => Some(ValueConstraint::new(
                true,
                text.clone(),
                namespace_context(&doc, node),
            )),
            _ => None,
        };
        let inheritable = bool_attr(&doc, node, "inheritable", false)?;

        if let Some(reference) = attr(&doc, node, "ref") {
            if attr(&doc, node, "name").is_some() {
                return Err(SchemaError::new(
                    "src-attribute.3.1",
                    "an attribute reference may not also carry a `name`",
                )
                .at(node_path(&doc, node)));
            }
            // §3.2.3 *src-attribute.3.2*: a reference fixes the declaration,
            // so `form`, `type`, `targetNamespace` and an inline
            // `xs:simpleType` are all absent (`constraintsOnAttribute/
            // s3_2_3si10`'s element analogue is the same rule: a reference
            // names an existing declaration rather than writing a new one, so
            // there is no local declaration for `targetNamespace` to attach
            // to).
            let inline = content_children(&doc, node)
                .into_iter()
                .any(|child| xsd_name(&doc, child) == Some("simpleType"));
            if inline
                || attr(&doc, node, "form").is_some()
                || attr(&doc, node, "type").is_some()
                || attr(&doc, node, "targetNamespace").is_some()
            {
                return Err(SchemaError::new(
                    "src-attribute.3.2",
                    "an attribute reference may not also carry `form`, `type`, \
                     `targetNamespace` or an inline `xs:simpleType`",
                )
                .at(node_path(&doc, node)));
            }
            let name = self.reference(info, node, reference, redefine, RedefineKind::Type)?;
            let id = self
                .schema
                .global_attributes
                .get(&name)
                .copied()
                .ok_or_else(|| {
                    SchemaError::new(
                        "src-resolve",
                        format!("the attribute `{name}` is not declared"),
                    )
                    .at(node_path(&doc, node))
                })?;
            // §3.5.6 *au-props-correct.2*: when the referenced declaration
            // fixes a value, a use may only repeat that very value
            // (`msData/additional/addB108`, `msData/attribute/attO025`). The
            // declaration is built first: a use written inside a *named*
            // complex type is reached before the pass over the global
            // attributes, so its {value constraint} would otherwise still be
            // absent and the rule would pass over a violation.
            self.build_global_attribute(id)?;
            if let Some(global) = self.schema.attributes[id.index()].value_constraint.clone()
                && global.fixed
                && let Some(own) = &value_constraint
                && (!own.fixed || own.lexical.trim() != global.lexical.trim())
            {
                return Err(SchemaError::new(
                    "au-props-correct.2",
                    format!(
                        "`{}` is fixed to `{}` by its declaration",
                        self.schema.attributes[id.index()].name,
                        global.lexical
                    ),
                )
                .at(node_path(&doc, node)));
            }
            return Ok(AttributeUse {
                attribute: id,
                kind,
                value_constraint,
                inheritable,
            });
        }

        let local = attr_token(&doc, node, "name").ok_or_else(|| {
            SchemaError::new(
                "src-attribute.3.1",
                "an attribute needs a `name` or a `ref`",
            )
            .at(node_path(&doc, node))
        })?;
        if local == "xmlns" {
            return Err(SchemaError::new(
                "no-xmlns",
                "an attribute declaration may not be named `xmlns`",
            )
            .at(node_path(&doc, node)));
        }
        let qualified = match attr_token(&doc, node, "form").as_deref() {
            Some("qualified") => true,
            Some("unqualified") => false,
            Some(other) => {
                return Err(SchemaError::new(
                    "src-attribute",
                    format!("`{other}` is not a `form` value"),
                )
                .at(node_path(&doc, node)));
            }
            None => info.attribute_form_qualified,
        };
        // XSD 1.1 §3.2.2: an explicit `targetNamespace` on a local attribute
        // declaration overrides `form`/`attributeFormDefault` outright, and
        // may only narrow a wildcard a restriction is entitled to narrow
        // (`targetNamespace/targetNamespace_005`, `constraintsOnAttribute/
        // s3_2_3v01`; `TargetNS/target004` is the same feature rejected for
        // sitting inside an `xs:extension` instead). `form` and `targetNamespace`
        // may not appear together — `form` is how a declaration with no
        // `targetNamespace` opts in or out of `attributeFormDefault`, which is
        // moot once `targetNamespace` states the namespace outright
        // (`constraintsOnAttribute/s3_2_3si03`).
        let namespace = if let Some(explicit) = attr(&doc, node, "targetNamespace") {
            if attr(&doc, node, "form").is_some() {
                return Err(SchemaError::new(
                    "src-attribute",
                    "an attribute may not have both a `form` and a `targetNamespace`",
                )
                .at(node_path(&doc, node)));
            }
            let context = local_derivation_context(&doc, node);
            if !target_namespace_override_allowed(
                &context,
                explicit,
                info.target_namespace.as_deref(),
            ) {
                return Err(SchemaError::new(
                    "src-attribute",
                    "`targetNamespace` on a local attribute declaration is only allowed as \
                     part of an `xs:restriction` narrowing a wildcard, or as a no-op restating \
                     the schema's own target namespace",
                )
                .at(node_path(&doc, node)));
            }
            (!explicit.is_empty()).then(|| explicit.to_owned())
        } else if qualified {
            info.target_namespace.clone()
        } else {
            None
        };
        // §3.2.6 *no-xsi*: the XSI namespace holds exactly the four attributes
        // the Recommendation defines, so no schema document may put another
        // one there — including by way of `attributeFormDefault="qualified"` in
        // a document whose target namespace *is* the XSI namespace
        // (`msData/attribute/attKb018a`). The same rule already guards
        // top-level declarations in `register_attribute`.
        if namespace.as_deref() == Some(crate::names::XSI_NS) {
            return Err(SchemaError::new(
                "no-xsi",
                format!("`{local}` would be in the XSI namespace, which is reserved"),
            )
            .at(node_path(&doc, node)));
        }
        let type_id = self.attribute_type(info, node, redefine)?;
        let id = AttributeId(u32::try_from(self.schema.attributes.len()).unwrap_or(u32::MAX));
        self.schema.attributes.push(AttributeDeclaration {
            name: ExpandedName::new(namespace, local),
            type_id,
            global: false,
            value_constraint: value_constraint.clone(),
            inheritable,
        });
        Ok(AttributeUse {
            attribute: id,
            kind,
            value_constraint,
            inheritable,
        })
    }

    fn attribute_type(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<TypeId, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let inline = content_children(&doc, node)
            .into_iter()
            .find(|child| xsd_name(&doc, *child) == Some("simpleType"));
        match (attr(&doc, node, "type"), inline) {
            (Some(_), Some(_)) => Err(SchemaError::new(
                "src-attribute.4",
                "an attribute may not have both a `type` and a `simpleType` child",
            )
            .at(node_path(&doc, node))),
            (Some(text), None) => {
                let name = self.reference(info, node, text, redefine, RedefineKind::Type)?;
                let id = self.type_by_name(&name, &doc, node)?;
                self.ensure_type(id)?;
                if !self.schema.types[id.index()].is_simple() {
                    return Err(SchemaError::new(
                        "src-resolve",
                        format!("the type of an attribute must be simple, but `{name}` is complex"),
                    )
                    .at(node_path(&doc, node)));
                }
                Ok(id)
            }
            (None, Some(child)) => self.build_anonymous_type(info, child, redefine),
            (None, None) => Ok(self.schema.any_simple_type),
        }
    }

    // ------------------------------------------------------------- elements

    fn build_element_particle(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<ElementId, SchemaError> {
        let doc = Rc::clone(&info.doc);
        if let Some(reference) = attr(&doc, node, "ref") {
            if attr(&doc, node, "name").is_some() {
                return Err(SchemaError::new(
                    "src-element.2.1",
                    "an element reference may not also carry a `name`",
                )
                .at(node_path(&doc, node)));
            }
            // §3.3.3 *src-element.2.2*: when `ref` is present the only other
            // things an `<xs:element>` particle may carry are `minOccurs`,
            // `maxOccurs`, `id` and an `<xs:annotation>` — a reference names
            // an existing declaration rather than writing a new one, so there
            // is no local declaration for `targetNamespace` to attach to
            // (`constraintsOnAttribute/s3_2_3si10`).
            const REF_EXCLUSIVE: &[&str] = &[
                "nillable",
                "default",
                "fixed",
                "form",
                "block",
                "type",
                "abstract",
                "final",
                "substitutionGroup",
                "targetNamespace",
            ];
            if let Some(offender) = REF_EXCLUSIVE
                .iter()
                .find(|name| attr(&doc, node, name).is_some())
            {
                return Err(SchemaError::new(
                    "src-element.2.2",
                    format!("an element reference may not also carry `{offender}`"),
                )
                .at(node_path(&doc, node)));
            }
            if let Some(child) = content_children(&doc, node).first() {
                return Err(SchemaError::new(
                    "src-element.2.2",
                    format!(
                        "an element reference may not have an `xs:{}` child",
                        xsd_name(&doc, *child).unwrap_or("?")
                    ),
                )
                .at(node_path(&doc, node)));
            }
            let name = self.reference(info, node, reference, redefine, RedefineKind::Type)?;
            return self
                .schema
                .global_elements
                .get(&name)
                .copied()
                .ok_or_else(|| {
                    SchemaError::new(
                        "src-resolve",
                        format!("the element `{name}` is not declared"),
                    )
                    .at(node_path(&doc, node))
                });
        }
        let local = attr_token(&doc, node, "name").ok_or_else(|| {
            SchemaError::new("src-element.2.1", "an element needs a `name` or a `ref`")
                .at(node_path(&doc, node))
        })?;
        let qualified = match attr_token(&doc, node, "form").as_deref() {
            Some("qualified") => true,
            Some("unqualified") => false,
            Some(other) => {
                return Err(SchemaError::new(
                    "src-element",
                    format!("`{other}` is not a `form` value"),
                )
                .at(node_path(&doc, node)));
            }
            None => info.element_form_qualified,
        };
        // XSD 1.1 §3.3.2: an explicit `targetNamespace` on a local element
        // declaration overrides `form`/`elementFormDefault` outright, and may
        // only narrow a wildcard a restriction is entitled to narrow
        // (`TargetNS/target001`; `target002` is the same feature rejected for
        // sitting inside an `xs:extension` instead). `form` and
        // `targetNamespace` may not appear together (`constraintsOnAttribute/
        // s3_2_3si06`), for the same reason as the attribute case above.
        let namespace = if let Some(explicit) = attr(&doc, node, "targetNamespace") {
            if attr(&doc, node, "form").is_some() {
                return Err(SchemaError::new(
                    "src-element",
                    "an element may not have both a `form` and a `targetNamespace`",
                )
                .at(node_path(&doc, node)));
            }
            let context = local_derivation_context(&doc, node);
            if !target_namespace_override_allowed(
                &context,
                explicit,
                info.target_namespace.as_deref(),
            ) {
                return Err(SchemaError::new(
                    "src-element",
                    "`targetNamespace` on a local element declaration is only allowed as part \
                     of an `xs:restriction` narrowing a wildcard, or as a no-op restating the \
                     schema's own target namespace",
                )
                .at(node_path(&doc, node)));
            }
            (!explicit.is_empty()).then(|| explicit.to_owned())
        } else if qualified {
            info.target_namespace.clone()
        } else {
            None
        };
        let id = ElementId(u32::try_from(self.schema.elements.len()).unwrap_or(u32::MAX));
        self.schema.elements.push(ElementDeclaration {
            name: ExpandedName::new(namespace, local),
            type_id: self.schema.any_type,
            global: false,
            nillable: false,
            abstract_: false,
            value_constraint: None,
            identity_constraints: Vec::new(),
            substitution_group: Vec::new(),
            block: info.block_default,
            final_: DerivationSet::empty(),
            substitutable_by: Vec::new(),
            alternatives: Vec::new(),
        });
        self.fill_element(info, node, id, redefine)?;
        Ok(id)
    }

    /// Fills in a *global* element declaration registered earlier.
    ///
    /// An element with no `type` of its own takes the type of its
    /// substitution-group head (§3.3.2), so the head is built first; the state
    /// map stops a cyclic `substitutionGroup` from recursing forever.
    pub(crate) fn build_global_element(&mut self, id: ElementId) -> Result<(), SchemaError> {
        if self
            .element_state
            .insert(id.0, BuildState::Building)
            .is_some()
        {
            return Ok(());
        }
        let Some(source) = self.element_source.get(&id.0).cloned() else {
            self.element_state.insert(id.0, BuildState::Built);
            return Ok(());
        };
        let info = self.info(source.doc);
        let outcome = self.fill_element(&info, source.node, id, source.redefine.as_ref());
        self.element_state.insert(id.0, BuildState::Built);
        outcome
    }

    fn fill_element(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        id: ElementId,
        redefine: Option<&Redefine>,
    ) -> Result<(), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let nillable = bool_attr(&doc, node, "nillable", false)?;
        let abstract_ = bool_attr(&doc, node, "abstract", false)?;
        let block = match attr(&doc, node, "block") {
            Some(text) => {
                derivation_set(text, &["extension", "restriction", "substitution"], "block")?
            }
            None => info.block_default,
        };
        let final_ = match attr(&doc, node, "final") {
            Some(text) => derivation_set(text, &["extension", "restriction"], "final")?,
            None => info.final_default,
        };
        let default = attr(&doc, node, "default").map(str::to_owned);
        let fixed = attr(&doc, node, "fixed").map(str::to_owned);
        if default.is_some() && fixed.is_some() {
            return Err(SchemaError::new(
                "src-element.1",
                "an element may not have both `default` and `fixed`",
            )
            .at(node_path(&doc, node)));
        }
        let value_constraint = match (&default, &fixed) {
            (Some(text), _) => Some(ValueConstraint::new(
                false,
                text.clone(),
                namespace_context(&doc, node),
            )),
            (_, Some(text)) => Some(ValueConstraint::new(
                true,
                text.clone(),
                namespace_context(&doc, node),
            )),
            _ => None,
        };

        let mut substitution_group = Vec::new();
        if let Some(text) = attr(&doc, node, "substitutionGroup") {
            for token in crate::value::split_tokens(text) {
                let name = self.reference(info, node, token, redefine, RedefineKind::Type)?;
                let head = self
                    .schema
                    .global_elements
                    .get(&name)
                    .copied()
                    .ok_or_else(|| {
                        SchemaError::new(
                            "src-resolve",
                            format!("the substitution group head `{name}` is not declared"),
                        )
                        .at(node_path(&doc, node))
                    })?;
                substitution_group.push(head);
            }
        }

        let inline = content_children(&doc, node)
            .into_iter()
            .find(|child| matches!(xsd_name(&doc, *child), Some("simpleType" | "complexType")));
        let type_id = match (attr(&doc, node, "type"), inline) {
            (Some(_), Some(_)) => {
                return Err(SchemaError::new(
                    "src-element.3",
                    "an element may not have both a `type` and an inline type definition",
                )
                .at(node_path(&doc, node)));
            }
            (Some(text), None) => {
                // Only the identifier is needed here: an element declaration may
                // legitimately name the very type it is nested in
                // (`<complexType name="C"><choice><element name="e" type="C"/>`),
                // so forcing the type to be built would report a false cycle.
                let name = self.reference(info, node, text, redefine, RedefineKind::Type)?;
                self.type_by_name(&name, &doc, node)?
            }
            // A type written inside an element declaration is built once every
            // named component is: what it refers to belongs to the element,
            // not to the group or type the element is written in, so building
            // it here would report their definitions as circular when they are
            // not (see `Compiler::pending_element_types`). Naming the type
            // instead — the branch above — was always deferred for the same
            // reason.
            (None, Some(child)) => self.defer_element_type(info, child, redefine),
            (None, None) => match substitution_group.first() {
                // §3.3.2: with no type of its own, an element takes the type of
                // its substitution-group head, or `xs:anyType`.
                Some(head) => {
                    let head = *head;
                    self.build_global_element(head)?;
                    self.schema.elements[head.index()].type_id
                }
                None => self.schema.any_type,
            },
        };

        let mut identity_constraints = Vec::new();
        let mut alternatives = Vec::new();
        for child in content_children(&doc, node) {
            match xsd_name(&doc, child) {
                Some(category @ ("key" | "keyref" | "unique")) => {
                    // XSD 1.1 §3.11.2: an identity constraint may be written
                    // once and *referred to* from other element declarations
                    // with `ref`, in which case it carries nothing else.
                    if let Some(text) = attr(&doc, child, "ref") {
                        if !self.schema.version().is_1_1() {
                            return Err(SchemaError::new(
                                "src-identity-constraint",
                                "`ref` on an identity constraint needs XML Schema 1.1",
                            )
                            .at(node_path(&doc, child)));
                        }
                        // §3.11.3 (*Identity-constraint Definition
                        // Representation OK*) clause 1: "One of `ref` or
                        // `name` is present, but not both"; clause 4: "If
                        // `ref` is present, then only `id` and `<annotation>`
                        // are allowed to appear together with `ref`"
                        // (`ibmData/schema_invalid/S2_2_4/s2_2_4si07`).
                        if attr(&doc, child, "name").is_some() {
                            return Err(SchemaError::new(
                                "src-identity-constraint.1",
                                "an identity constraint may have `ref` or `name`, not both",
                            )
                            .at(node_path(&doc, child)));
                        }
                        if content_children(&doc, child)
                            .into_iter()
                            .any(|member| xsd_name(&doc, member) != Some("annotation"))
                        {
                            return Err(SchemaError::new(
                                "src-identity-constraint.4",
                                "an identity constraint written with `ref` may have no children                                  but `xs:annotation`",
                            )
                            .at(node_path(&doc, child)));
                        }
                        if let Some(element) = doc.element(child)
                            && element.attribute_ids().iter().any(|attribute| {
                                doc.attr(*attribute).is_some_and(|data| {
                                    data.namespace_uri().is_none()
                                        && !matches!(data.local_name(), "ref" | "id")
                                })
                            })
                        {
                            return Err(SchemaError::new(
                                "src-identity-constraint.4",
                                "an identity constraint written with `ref` may carry no attribute \
                                 but `id`",
                            )
                            .at(node_path(&doc, child)));
                        }
                        let name =
                            self.reference(info, child, text, redefine, RedefineKind::Type)?;
                        self.idc_ref_pending.push((
                            id,
                            name,
                            Self::idc_category(category),
                            node_path(&doc, child),
                        ));
                        continue;
                    }
                    let idc = self.build_identity_constraint(info, child, category)?;
                    identity_constraints.push(idc);
                }
                Some("alternative") => {
                    let alternative =
                        self.build_type_alternative(info, child, type_id, redefine)?;
                    alternatives.push(alternative);
                }
                _ => continue,
            }
        }

        // XSD 1.1 §3.3.3: an `xs:alternative` written without a `test` is the
        // *default* one of the {type table}, and only the last may be written
        // that way — an earlier one would select its type for every element
        // information item and leave the alternatives after it unreachable
        // (`saxonData/CTA/cta9001err`).
        if alternatives
            .iter()
            .rev()
            .skip(1)
            .any(|alternative| alternative.test.is_none())
        {
            return Err(SchemaError::new(
                "src-element",
                "only the last `xs:alternative` of an element declaration may omit its `test`",
            )
            .at(node_path(&doc, node)));
        }

        let element = &mut self.schema.elements[id.index()];
        element.alternatives = alternatives;
        element.type_id = type_id;
        element.nillable = nillable;
        element.abstract_ = abstract_;
        element.block = block;
        element.final_ = final_;
        element.value_constraint = value_constraint;
        element.substitution_group = substitution_group;
        element.identity_constraints = identity_constraints;
        Ok(())
    }

    /// Builds one `xs:alternative` of an element declaration's {type table}
    /// (XSD 1.1 §3.3.2).
    ///
    /// The type is named by `type` or written inline; a `test` makes the
    /// alternative conditional and its absence makes it the *default*
    /// alternative, which §3.3.3 allows only as the last one — a rule
    /// [`crate::compile::checks`] enforces once the whole table is known.
    fn build_type_alternative(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        declared: TypeId,
        redefine: Option<&Redefine>,
    ) -> Result<crate::components::TypeAlternative, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let inline = content_children(&doc, node)
            .into_iter()
            .find(|child| matches!(xsd_name(&doc, *child), Some("simpleType" | "complexType")));
        let type_id = match (attr(&doc, node, "type"), inline) {
            (Some(_), Some(_)) => {
                return Err(SchemaError::new(
                    "src-type-alternative",
                    "an `xs:alternative` may not have both a `type` and an inline type definition",
                )
                .at(node_path(&doc, node)));
            }
            (Some(text), None) => {
                let name = self.reference(info, node, text, redefine, RedefineKind::Type)?;
                self.type_by_name(&name, &doc, node)?
            }
            (None, Some(child)) => self.build_anonymous_type(info, child, redefine)?,
            // §3.3.3 *src-type-alternative*: an alternative must name a type
            // one way or the other. The declared type is used as a fallback so
            // that assembly continues; the representation checks report it.
            (None, None) => declared,
        };
        let test = attr(&doc, node, "test").map(|text| crate::components::Assertion {
            test: text.to_owned(),
            namespaces: namespace_context(&doc, node),
            xpath_default_namespace: crate::parse::xpath_default_namespace(
                &doc,
                node,
                info.target_namespace.as_deref(),
            ),
        });
        if let Some(test) = &test {
            crate::assertion::check_static(test, false, "src-type-alternative")
                .map_err(|error| error.or_at(node_path(&doc, node)))?;
        }
        Ok(crate::components::TypeAlternative { test, type_id })
    }

    /// The `{identity-constraint category}` an `xs:key`, `xs:keyref` or
    /// `xs:unique` element maps to.
    fn idc_category(name: &str) -> IdcCategory {
        match name {
            "key" => IdcCategory::Key,
            "keyref" => IdcCategory::KeyRef,
            _ => IdcCategory::Unique,
        }
    }

    fn build_identity_constraint(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        category: &str,
    ) -> Result<IdcId, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let local = attr_token(&doc, node, "name").ok_or_else(|| {
            SchemaError::new(
                "src-identity-constraint",
                "an identity constraint needs a `name`",
            )
            .at(node_path(&doc, node))
        })?;
        let name = ExpandedName::new(info.target_namespace.as_deref(), local);
        let category = Self::idc_category(category);
        let mut selector = None;
        let mut fields = Vec::new();
        for child in content_children(&doc, node) {
            match xsd_name(&doc, child) {
                Some("selector") => {
                    let text = attr(&doc, child, "xpath").ok_or_else(|| {
                        SchemaError::new("src-identity-constraint", "a `selector` needs an `xpath`")
                            .at(node_path(&doc, child))
                    })?;
                    let default = crate::parse::xpath_default_namespace(
                        &doc,
                        child,
                        info.target_namespace.as_deref(),
                    );
                    selector = Some(
                        PathSet::parse_selector_in(
                            text,
                            &namespace_context(&doc, child),
                            default.as_deref(),
                        )
                        .map_err(|error| error.or_at(node_path(&doc, child)))?,
                    );
                }
                Some("field") => {
                    let text = attr(&doc, child, "xpath").ok_or_else(|| {
                        SchemaError::new("src-identity-constraint", "a `field` needs an `xpath`")
                            .at(node_path(&doc, child))
                    })?;
                    let default = crate::parse::xpath_default_namespace(
                        &doc,
                        child,
                        info.target_namespace.as_deref(),
                    );
                    fields.push(
                        PathSet::parse_field_in(
                            text,
                            &namespace_context(&doc, child),
                            default.as_deref(),
                        )
                        .map_err(|error| error.or_at(node_path(&doc, child)))?,
                    );
                }
                _ => {}
            }
        }
        let selector = selector.ok_or_else(|| {
            SchemaError::new(
                "src-identity-constraint",
                "an identity constraint needs a `selector`",
            )
            .at(node_path(&doc, node))
        })?;
        if fields.is_empty() {
            return Err(SchemaError::new(
                "src-identity-constraint",
                "an identity constraint needs at least one `field`",
            )
            .at(node_path(&doc, node)));
        }
        let refer_name = match attr(&doc, node, "refer") {
            Some(text) => Some(self.reference(info, node, text, None, RedefineKind::Type)?),
            None => None,
        };
        if category == IdcCategory::KeyRef && refer_name.is_none() {
            return Err(
                SchemaError::new("src-identity-constraint", "a `keyref` needs a `refer`")
                    .at(node_path(&doc, node)),
            );
        }
        let id = IdcId(u32::try_from(self.schema.identity_constraints.len()).unwrap_or(u32::MAX));
        if self
            .schema
            .global_identity_constraints
            .insert(name.clone(), id)
            .is_some()
        {
            return Err(SchemaError::new(
                "c-props-correct.1",
                format!("the identity constraint `{name}` is defined more than once"),
            )
            .at(node_path(&doc, node)));
        }
        self.schema
            .identity_constraints
            .push(IdentityConstraintDefinition {
                name,
                category,
                selector,
                fields,
                refer: None,
                refer_name: refer_name.clone(),
            });
        if let Some(refer) = refer_name {
            self.keyref_pending.push((id, refer));
        }
        Ok(id)
    }

    /// Fills in a *global* attribute declaration registered earlier.
    pub(crate) fn build_global_attribute(&mut self, id: AttributeId) -> Result<(), SchemaError> {
        // An attribute use may need the declaration it refers to before the
        // pass over the global attributes reaches it, so the state map both
        // lets it be built early and keeps it from being built twice — a
        // second build would push a duplicate of any inline `xs:simpleType`.
        if self
            .attribute_state
            .insert(id.0, BuildState::Built)
            .is_some()
        {
            return Ok(());
        }
        let Some(source) = self.attribute_source.get(&id.0).cloned() else {
            return Ok(());
        };
        let info = self.info(source.doc);
        let doc = Rc::clone(&info.doc);
        let node = source.node;
        let default = attr(&doc, node, "default").map(str::to_owned);
        let fixed = attr(&doc, node, "fixed").map(str::to_owned);
        if default.is_some() && fixed.is_some() {
            return Err(SchemaError::new(
                "src-attribute.1",
                "an attribute may not have both `default` and `fixed`",
            )
            .at(node_path(&doc, node)));
        }
        let value_constraint = match (&default, &fixed) {
            (Some(text), _) => Some(ValueConstraint::new(
                false,
                text.clone(),
                namespace_context(&doc, node),
            )),
            (_, Some(text)) => Some(ValueConstraint::new(
                true,
                text.clone(),
                namespace_context(&doc, node),
            )),
            _ => None,
        };
        let type_id = self.attribute_type(&info, node, source.redefine.as_ref())?;
        let inheritable = bool_attr(&doc, node, "inheritable", false)?;
        let attribute = &mut self.schema.attributes[id.index()];
        attribute.type_id = type_id;
        attribute.value_constraint = value_constraint;
        attribute.inheritable = inheritable;
        Ok(())
    }
}

/// The derivation `node` — a local `xs:element` or `xs:attribute` particle —
/// sits inside, for the purposes of its `targetNamespace` attribute.
enum LocalDerivationContext {
    /// Inside an `xs:extension`, or no `xs:restriction`/`xs:extension`/bare
    /// `xs:complexType` ancestor is found at all (a component, such as an
    /// `xs:attributeGroup` definition, written independently of where it is
    /// referenced): `targetNamespace` is never justified here, since nothing
    /// is being narrowed.
    Extension,
    /// Inside a restriction whose base is literally `xs:anyType` — including
    /// a bare `xs:complexType` with no `xs:complexContent`/`xs:simpleContent`
    /// wrapper, which *is* an implicit restriction of `xs:anyType` (§3.4.2) —
    /// so there is no inherited wildcard to narrow either; `targetNamespace`
    /// is tolerated only as a no-op that restates the enclosing schema's own
    /// target namespace.
    RestrictsAnyType,
    /// Inside a restriction of some other, named base type, which may have a
    /// wildcard `targetNamespace` is narrowing one item out of.
    RestrictsNamedType,
}

/// The *attribute wildcard intersection* of §3.10.6, as §3.4.2 and §3.6.2 use
/// it to fold several wildcards into one.
///
/// The {process contents} of the result is `first`'s: the callers order their
/// arguments so that the wildcard entitled to supply it — the local
/// `<anyAttribute>`, or the first attribute group to contribute one — comes
/// first. A namespace intersection that cannot be written as one namespace
/// constraint leaves `first`'s constraint in place rather than widening the
/// result.
fn intersect_wildcards(first: &Wildcard, second: &Wildcard) -> Wildcard {
    Wildcard {
        namespace: first
            .namespace
            .intersection(&second.namespace)
            .unwrap_or_else(|| first.namespace.clone()),
        process_contents: first.process_contents,
        // §3.10.6: intersecting two wildcards disallows everything either of
        // them disallowed.
        disallowed: first.disallowed.union_with(&second.disallowed),
    }
}

/// Classifies the derivation `node` sits inside, per [`LocalDerivationContext`].
///
/// XSD 1.1 §3.2.2/§3.3.2: `targetNamespace` on a local declaration exists so a
/// restriction can narrow one specific item out of a wildcard the base type
/// admits. An `xs:extension` merely appends content, so it narrows nothing
/// (`TargetNS/target002`, `target004`: "Invalid because not part of an
/// `xs:restriction`"); nor does a restriction of `xs:anyType` itself, which
/// has no wildcard at all (`constraintsOnAttribute/s3_2_3si01`, `si05`,
/// `si08`: valid only when the given namespace is a no-op, equal to the
/// schema's own target namespace).
fn local_derivation_context(doc: &Document, node: NodeId) -> LocalDerivationContext {
    for ancestor in doc.ancestors(node) {
        match xsd_name(doc, ancestor) {
            Some("restriction") => {
                let restricts_any_type = attr(doc, ancestor, "base").is_some_and(|base| {
                    resolve_qname(doc, ancestor, base).is_ok_and(|name| {
                        name.namespace() == Some(crate::names::XSD_NS)
                            && name.local_name() == "anyType"
                    })
                });
                return if restricts_any_type {
                    LocalDerivationContext::RestrictsAnyType
                } else {
                    LocalDerivationContext::RestrictsNamedType
                };
            }
            Some("extension") => return LocalDerivationContext::Extension,
            // No `xs:complexContent`/`xs:simpleContent` wrapper: XSD's
            // shorthand for `<xs:complexContent><xs:restriction
            // base="xs:anyType">`.
            Some("complexType") => return LocalDerivationContext::RestrictsAnyType,
            _ => {}
        }
    }
    LocalDerivationContext::Extension
}

/// Whether `targetNamespace="explicit"` on a local declaration inside
/// `context` is allowed, given the enclosing document's own target namespace.
fn target_namespace_override_allowed(
    context: &LocalDerivationContext,
    explicit: &str,
    schema_target_namespace: Option<&str>,
) -> bool {
    match context {
        LocalDerivationContext::Extension => false,
        LocalDerivationContext::RestrictsAnyType => schema_target_namespace == Some(explicit),
        LocalDerivationContext::RestrictsNamedType => true,
    }
}
