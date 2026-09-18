//! Building simple and complex type definitions out of their schema-document
//! representation.

use std::rc::Rc;

use oxixml_dom::NodeId;

use crate::automaton::ContentModel;
use crate::components::{
    Assertion, AttributeUse, AttributeUseKind, ComplexTypeInfo, Compositor, ContentType,
    Derivation, DerivationSet, Facets, MaxOccurs, ModelGroup, Particle, SimpleTypeInfo, Term,
    TypeDefinition, TypeId, TypeKind, Variety, Wildcard,
};
use crate::error::SchemaError;
use crate::names::XsdVersion;
use crate::parse::{attr, attr_token, content_children, derivation_set, node_path, xsd_name};
use crate::simple::{RawFacet, apply_facets};
use crate::value::WhiteSpace;

use super::{BuildState, Compiler, DocInfo, Redefine, RedefineKind};

/// The constraining facets that may appear inside `xs:restriction`.
const FACET_NAMES: &[&str] = &[
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
    "assertion",
    "explicitTimezone",
];

impl Compiler {
    /// Builds the body of a named type definition, if it has not been built
    /// yet.
    pub(crate) fn ensure_type(&mut self, id: TypeId) -> Result<(), SchemaError> {
        match self.type_state[id.index()] {
            BuildState::Built => return Ok(()),
            BuildState::Building => {
                return Err(SchemaError::new(
                    "src-type-cycle",
                    format!(
                        "the type `{}` is defined in terms of itself",
                        self.schema.type_name(id)
                    ),
                ));
            }
            BuildState::NotBuilt => {}
        }
        let Some(source) = self.type_source.get(&id.0).cloned() else {
            self.type_state[id.index()] = BuildState::Built;
            return Ok(());
        };
        self.type_state[id.index()] = BuildState::Building;
        let info = self.info(source.doc);
        let name = self.schema.types[id.index()].name.clone();
        self.enter_definition("type")?;
        let built = self.build_type(&info, source.node, source.redefine.as_ref(), name);
        self.leave_definition();
        self.schema.types[id.index()] = built?;
        self.type_state[id.index()] = BuildState::Built;
        Ok(())
    }

    /// Builds an anonymous `xs:simpleType` or `xs:complexType` and returns its
    /// identifier.
    pub(crate) fn build_anonymous_type(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<TypeId, SchemaError> {
        let definition = self.build_type(info, node, redefine, None)?;
        let id = self.push_type_stub(None);
        self.schema.types[id.index()] = definition;
        self.type_state[id.index()] = BuildState::Built;
        Ok(id)
    }

    fn build_type(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
        name: Option<oxixml_qname::ExpandedName>,
    ) -> Result<TypeDefinition, SchemaError> {
        match xsd_name(&info.doc, node) {
            Some("simpleType") => self.build_simple_type(info, node, redefine, name),
            Some("complexType") => self.build_complex_type(info, node, redefine, name),
            other => Err(SchemaError::new(
                "src-element",
                format!(
                    "expected a type definition, found `xs:{}`",
                    other.unwrap_or("?")
                ),
            )
            .at(node_path(&info.doc, node))),
        }
    }

    // ---------------------------------------------------------------- simple

    fn build_simple_type(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
        name: Option<oxixml_qname::ExpandedName>,
    ) -> Result<TypeDefinition, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let final_ = match attr(&doc, node, "final") {
            Some(text) => derivation_set(
                text,
                &["restriction", "list", "union", "extension"],
                "final",
            )?,
            None => info.final_default,
        };
        let children = content_children(&doc, node);
        let [child] = children.as_slice() else {
            return Err(SchemaError::new(
                "src-simple-type",
                "an `xs:simpleType` needs exactly one of `restriction`, `list` or `union`",
            )
            .at(node_path(&doc, node)));
        };
        let child = *child;
        let (base, derivation, variety, primitive, facets) = match xsd_name(&doc, child) {
            Some("restriction") => self.build_simple_restriction(info, child, redefine)?,
            Some("list") => self.build_simple_list(info, child, redefine)?,
            Some("union") => self.build_simple_union(info, child, redefine)?,
            other => {
                return Err(SchemaError::new(
                    "src-simple-type",
                    format!(
                        "`xs:{}` may not appear inside `xs:simpleType`",
                        other.unwrap_or("?")
                    ),
                )
                .at(node_path(&doc, child)));
            }
        };
        Ok(TypeDefinition {
            name,
            base: Some(base),
            derivation,
            final_,
            block: DerivationSet::empty(),
            kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                variety,
                primitive,
                facets,
            })),
        })
    }

    #[allow(clippy::type_complexity)]
    fn build_simple_restriction(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<
        (
            TypeId,
            Derivation,
            Variety,
            Option<crate::value::Primitive>,
            Facets,
        ),
        SchemaError,
    > {
        let doc = Rc::clone(&info.doc);
        let children = content_children(&doc, node);
        let mut base = None;
        let mut facet_nodes = Vec::new();
        for child in &children {
            match xsd_name(&doc, *child) {
                Some("simpleType") => {
                    if base.is_some() {
                        return Err(SchemaError::new(
                            "src-restriction",
                            "a `restriction` may not have both a `base` and a `simpleType`",
                        )
                        .at(node_path(&doc, *child)));
                    }
                    base = Some(self.build_anonymous_type(info, *child, redefine)?);
                }
                Some(name) if FACET_NAMES.contains(&name) => facet_nodes.push(*child),
                other => {
                    return Err(SchemaError::new(
                        "src-restriction",
                        format!("`xs:{}` is not a constraining facet", other.unwrap_or("?")),
                    )
                    .at(node_path(&doc, *child)));
                }
            }
        }
        let base = match base {
            Some(id) => id,
            None => {
                let text = attr(&doc, node, "base").ok_or_else(|| {
                    SchemaError::new(
                        "src-restriction",
                        "a `restriction` needs a `base` attribute or a `simpleType` child",
                    )
                    .at(node_path(&doc, node))
                })?;
                let name = self.reference(info, node, text, redefine, RedefineKind::Type)?;
                self.type_by_name(&name, &doc, node)?
            }
        };
        self.ensure_type(base)?;
        let base_definition = &self.schema.types[base.index()];
        let base_simple = base_definition.as_simple().ok_or_else(|| {
            SchemaError::new(
                "src-restriction",
                format!(
                    "`{}` is a complex type, so a simple type may not restrict it",
                    self.schema.type_name(base)
                ),
            )
            .at(node_path(&doc, node))
        })?;
        if base_definition.final_.restriction {
            return Err(SchemaError::new(
                "st-props-correct.3",
                format!(
                    "`{}` is `final=\"restriction\"`",
                    self.schema.type_name(base)
                ),
            )
            .at(node_path(&doc, node)));
        }
        // Datatypes §3.4.1: `xs:anySimpleType` is the ·simple ur-type
        // definition·, and has no {variety} for a restriction to inherit and
        // no facets for one to constrain. It is therefore not available as the
        // base of a user-defined simple type — the derivation would leave the
        // new type with no variety at all (`msData/additional/addB110`).
        if base == self.schema.any_simple_type() {
            return Err(SchemaError::new(
                "cos-st-restricts",
                "`xs:anySimpleType` has no variety, so a simple type may not restrict it",
            )
            .at(node_path(&doc, node)));
        }
        let variety = base_simple.variety.clone();
        let primitive = base_simple.primitive;
        let base_facets = base_simple.facets.clone();
        let raw = self.collect_facets(info, &facet_nodes)?;
        let facets = apply_facets(&self.schema, base, &base_facets, &variety, &raw)
            .map_err(|error| error.or_at(node_path(&doc, node)))?;
        Ok((base, Derivation::Restriction, variety, primitive, facets))
    }

    #[allow(clippy::type_complexity)]
    fn build_simple_list(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<
        (
            TypeId,
            Derivation,
            Variety,
            Option<crate::value::Primitive>,
            Facets,
        ),
        SchemaError,
    > {
        let doc = Rc::clone(&info.doc);
        let children = content_children(&doc, node);
        let inline = children.first().copied();
        let item = match (attr(&doc, node, "itemType"), inline) {
            (Some(_), Some(child)) if xsd_name(&doc, child) == Some("simpleType") => {
                return Err(SchemaError::new(
                    "src-list-itemType-or-simpleType",
                    "a `list` may not have both an `itemType` and a `simpleType` child",
                )
                .at(node_path(&doc, node)));
            }
            (Some(text), _) => {
                let name = self.reference(info, node, text, redefine, RedefineKind::Type)?;
                let id = self.type_by_name(&name, &doc, node)?;
                self.ensure_type(id)?;
                id
            }
            (None, Some(child)) if xsd_name(&doc, child) == Some("simpleType") => {
                self.build_anonymous_type(info, child, redefine)?
            }
            _ => {
                return Err(SchemaError::new(
                    "src-list-itemType-or-simpleType",
                    "a `list` needs an `itemType` or a `simpleType` child",
                )
                .at(node_path(&doc, node)));
            }
        };
        let item_definition = &self.schema.types[item.index()];
        let item_simple = item_definition.as_simple().ok_or_else(|| {
            SchemaError::new(
                "cos-list-of-atomic",
                format!(
                    "the item type `{}` of a list must be a simple type",
                    self.schema.type_name(item)
                ),
            )
            .at(node_path(&doc, node))
        })?;
        if matches!(item_simple.variety, Variety::List(_)) {
            return Err(SchemaError::new(
                "cos-list-of-atomic",
                "the item type of a list may not itself be a list",
            )
            .at(node_path(&doc, node)));
        }
        if item_definition.final_.list {
            return Err(SchemaError::new(
                "st-props-correct.3",
                format!("`{}` is `final=\"list\"`", self.schema.type_name(item)),
            )
            .at(node_path(&doc, node)));
        }
        let facets = Facets {
            white_space: WhiteSpace::Collapse,
            white_space_fixed: true,
            ..Facets::default()
        };
        Ok((
            self.schema.any_simple_type,
            Derivation::List,
            Variety::List(item),
            None,
            facets,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn build_simple_union(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<
        (
            TypeId,
            Derivation,
            Variety,
            Option<crate::value::Primitive>,
            Facets,
        ),
        SchemaError,
    > {
        let doc = Rc::clone(&info.doc);
        let mut members = Vec::new();
        if let Some(text) = attr(&doc, node, "memberTypes") {
            for token in crate::value::split_tokens(text) {
                let name = self.reference(info, node, token, redefine, RedefineKind::Type)?;
                let id = self.type_by_name(&name, &doc, node)?;
                self.ensure_type(id)?;
                members.push(id);
            }
        }
        for child in content_children(&doc, node) {
            if xsd_name(&doc, child) != Some("simpleType") {
                return Err(SchemaError::new(
                    "src-union-memberTypes-or-simpleTypes",
                    "only `simpleType` children may appear inside `xs:union`",
                )
                .at(node_path(&doc, child)));
            }
            members.push(self.build_anonymous_type(info, child, redefine)?);
        }
        if members.is_empty() {
            return Err(SchemaError::new(
                "src-union-memberTypes-or-simpleTypes",
                "a `union` needs at least one member type",
            )
            .at(node_path(&doc, node)));
        }
        for member in &members {
            let definition = &self.schema.types[member.index()];
            if !definition.is_simple() {
                return Err(SchemaError::new(
                    "src-union-memberTypes-or-simpleTypes",
                    format!(
                        "the member type `{}` of a union must be a simple type",
                        self.schema.type_name(*member)
                    ),
                )
                .at(node_path(&doc, node)));
            }
            if definition.final_.union {
                return Err(SchemaError::new(
                    "st-props-correct.3",
                    format!("`{}` is `final=\"union\"`", self.schema.type_name(*member)),
                )
                .at(node_path(&doc, node)));
            }
            // Datatypes §3.4.1: XSD 1.1 makes `xs:anySimpleType` ·special· —
            // it has no {variety} and so no value space for a union to take a
            // member from — and forbids naming it as a member type. XSD 1.0
            // had no such rule, and the suite splits the verdict by version
            // (`msData/simpleType/stE053`).
            if self.schema.version().is_1_1()
                && matches!(
                    definition.as_simple().map(|simple| &simple.variety),
                    Some(Variety::Absent)
                )
            {
                return Err(SchemaError::new(
                    "src-union-memberTypes-or-simpleTypes",
                    format!(
                        "`{}` has no variety, so it may not be a member type of a union",
                        self.schema.type_name(*member)
                    ),
                )
                .at(node_path(&doc, node)));
            }
        }
        let facets = Facets {
            white_space: WhiteSpace::Collapse,
            white_space_fixed: true,
            ..Facets::default()
        };
        Ok((
            self.schema.any_simple_type,
            Derivation::Union,
            Variety::Union(members),
            None,
            facets,
        ))
    }

    fn collect_facets(
        &mut self,
        info: &DocInfo,
        nodes: &[NodeId],
    ) -> Result<Vec<RawFacet>, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let mut raw = Vec::with_capacity(nodes.len());
        for node in nodes {
            let name = xsd_name(&doc, *node).unwrap_or_default().to_owned();
            let value_attribute = if name == "assertion" { "test" } else { "value" };
            let value = attr(&doc, *node, value_attribute)
                .ok_or_else(|| {
                    SchemaError::new(
                        "src-facet",
                        format!("`xs:{name}` needs a `{value_attribute}` attribute"),
                    )
                    .at(node_path(&doc, *node))
                })?
                .to_owned();
            let fixed = crate::parse::bool_attr(&doc, *node, "fixed", false)?;
            raw.push(RawFacet {
                name,
                value,
                fixed,
                namespaces: crate::parse::namespace_context(&doc, *node),
                xpath_default_namespace: crate::parse::xpath_default_namespace(
                    &doc,
                    *node,
                    info.target_namespace.as_deref(),
                ),
            });
        }
        Ok(raw)
    }

    pub(crate) fn type_by_name(
        &self,
        name: &oxixml_qname::ExpandedName,
        doc: &oxixml_dom::Document,
        node: NodeId,
    ) -> Result<TypeId, SchemaError> {
        self.schema.global_types.get(name).copied().ok_or_else(|| {
            SchemaError::new("src-resolve", format!("the type `{name}` is not defined"))
                .at(node_path(doc, node))
        })
    }

    // --------------------------------------------------------------- complex

    fn build_complex_type(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
        name: Option<oxixml_qname::ExpandedName>,
    ) -> Result<TypeDefinition, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let abstract_ = crate::parse::bool_attr(&doc, node, "abstract", false)?;
        let mixed_attribute = crate::parse::bool_attr(&doc, node, "mixed", false)?;
        let mixed_written = crate::parse::attr(&doc, node, "mixed").map(|_| mixed_attribute);
        let final_ = match attr(&doc, node, "final") {
            Some(text) => derivation_set(text, &["restriction", "extension"], "final")?,
            None => info.final_default,
        };
        let block = match attr(&doc, node, "block") {
            Some(text) => derivation_set(text, &["restriction", "extension"], "block")?,
            None => info.block_default,
        };
        let children = content_children(&doc, node);
        let first = children.first().copied();
        let (base, derivation, mut complex) = match first.and_then(|c| xsd_name(&doc, c)) {
            Some("simpleContent") => {
                let child = first.unwrap_or(node);
                self.build_simple_content(info, child, redefine)?
            }
            Some("complexContent") => {
                let child = first.unwrap_or(node);
                self.build_complex_content(info, child, redefine, mixed_attribute, mixed_written)?
            }
            _ => {
                let (content, uses, wildcard, assertions, open) =
                    self.build_content_shorthand(info, node, &children, redefine, mixed_attribute)?;
                (
                    self.schema.any_type,
                    Derivation::Restriction,
                    ComplexTypeInfo {
                        derivation: Derivation::Restriction,
                        abstract_,
                        attribute_uses: uses,
                        attribute_wildcard: wildcard,
                        content,
                        content_model: None,
                        assertions,
                        open_content: open,
                    },
                )
            }
        };
        complex.abstract_ = abstract_;
        complex.derivation = derivation;
        self.apply_default_attributes(info, node, &mut complex, redefine)?;
        complex.content_model = Some(match complex.content.particle() {
            Some(particle) => ContentModel::build_for(particle, self.schema.version())
                .map_err(|error| error.or_at(node_path(&doc, node)))?,
            None => ContentModel::Empty,
        });
        Ok(TypeDefinition {
            name,
            base: Some(base),
            derivation,
            final_,
            block,
            kind: TypeKind::Complex(Box::new(complex)),
        })
    }

    /// The particle, attribute uses, wildcard and assertions written directly
    /// inside `xs:complexType`, `xs:restriction` or `xs:extension`.
    #[allow(clippy::type_complexity)]
    fn build_content_shorthand(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        children: &[NodeId],
        redefine: Option<&Redefine>,
        mixed: bool,
    ) -> Result<
        (
            ContentType,
            Vec<AttributeUse>,
            Option<Wildcard>,
            Vec<Assertion>,
            Option<crate::components::OpenContent>,
        ),
        SchemaError,
    > {
        let doc = Rc::clone(&info.doc);
        let mut particle = None;
        let mut attribute_nodes = Vec::new();
        let mut assertions = Vec::new();
        let mut open_content = None;
        for child in children {
            match xsd_name(&doc, *child) {
                Some("sequence" | "choice" | "all" | "group") => {
                    if particle.is_some() {
                        return Err(SchemaError::new(
                            "src-ct",
                            "a complex type may contain only one particle",
                        )
                        .at(node_path(&doc, *child)));
                    }
                    particle = Some(self.build_particle(info, *child, redefine)?);
                }
                Some("attribute" | "attributeGroup" | "anyAttribute") => {
                    attribute_nodes.push(*child);
                }
                Some("assert") => {
                    let test = attr(&doc, *child, "test").unwrap_or_default().to_owned();
                    let assertion = Assertion {
                        test,
                        namespaces: crate::parse::namespace_context(&doc, *child),
                        xpath_default_namespace: crate::parse::xpath_default_namespace(
                            &doc,
                            *child,
                            info.target_namespace.as_deref(),
                        ),
                    };
                    crate::assertion::check_static(&assertion, true, "src-assert")
                        .map_err(|error| error.or_at(node_path(&doc, *child)))?;
                    assertions.push(assertion);
                }
                Some("openContent") => {
                    open_content = Some(self.build_open_content(info, *child)?);
                }
                other => {
                    return Err(SchemaError::new(
                        "src-ct",
                        format!(
                            "`xs:{}` may not appear in a complex type definition",
                            other.unwrap_or("?")
                        ),
                    )
                    .at(node_path(&doc, *child)));
                }
            }
        }
        let (uses, wildcard) = self.build_attribute_uses(info, &attribute_nodes, redefine)?;
        let content = match particle {
            Some(particle) if !particle_is_empty(&particle) => {
                if mixed {
                    ContentType::Mixed(Box::new(particle))
                } else {
                    ContentType::ElementOnly(Box::new(particle))
                }
            }
            _ => {
                if mixed {
                    ContentType::Mixed(Box::new(empty_particle()))
                } else {
                    ContentType::Empty
                }
            }
        };
        let _ = node;
        // XML Schema 1.1 §3.4.2: a schema document's `xs:defaultOpenContent`
        // supplies the open content of every complex type in it that has no
        // `xs:openContent` of its own — and, unless it says
        // `appliesToEmpty="true"`, only of those whose content is not empty.
        //
        // §3.4.2.3.3 clause 5.2 speaks of "the `<schema>` ancestor", which for
        // a type written inside `xs:override` is the *overridden* document's
        // (`saxonData/Open/open043`) — see `defaults_document`.
        let host = self.defaults_document(info, node);
        let host_doc = Rc::clone(&host.doc);
        if open_content.is_none()
            && let Some(default) = default_open_content(&host_doc, host.root)
            && (!matches!(content, ContentType::Empty)
                || crate::parse::attr_token(&host_doc, default, "appliesToEmpty").as_deref()
                    == Some("true"))
        {
            open_content = Some(self.build_open_content(&host, default)?);
        }
        // §3.4.2.3.3 clause 6.2: once a ·wildcard element· applies, the
        // {variety} is "the {variety} of the ·explicit content type· if it's
        // not `empty`; otherwise **element-only**", with an empty sequence for
        // its {particle}. So a type whose own content is empty but which
        // carries open content is element-only, and white space in an instance
        // is allowed by clause 1.3 rather than forbidden by clause 1.1
        // (`saxonData/Open/open010`, `open011` — against `open012`, where the
        // `appliesToEmpty="false"` default open content does *not* apply and
        // the variety stays `empty`).
        let content = match (&content, &open_content) {
            (ContentType::Empty, Some(open)) if open.wildcard.is_some() => {
                ContentType::ElementOnly(Box::new(empty_particle()))
            }
            _ => content,
        };
        Ok((content, uses, wildcard, assertions, open_content))
    }

    fn build_open_content(
        &mut self,
        info: &DocInfo,
        node: NodeId,
    ) -> Result<crate::components::OpenContent, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let mode = attr_token(&doc, node, "mode").unwrap_or_else(|| "interleave".to_owned());
        let any = content_children(&doc, node)
            .into_iter()
            .find(|child| xsd_name(&doc, *child) == Some("any"));
        // The schema for schemas declares the `any` inside `xs:openContent`
        // and `xs:defaultOpenContent` with type `xs:wildcard`, which — unlike
        // the type of a `<xs:any>` particle — carries no occurrence
        // attributes. Open content is not a particle: it says which elements
        // may appear beside the content model, not how many times, so
        // `minOccurs` and `maxOccurs` have nothing to say here
        // (`saxonData/Open/open048`; W3C bug 15618).
        if let Some(child) = any
            && let Some(offender) = ["minOccurs", "maxOccurs"]
                .into_iter()
                .find(|name| attr(&doc, child, name).is_some())
        {
            return Err(SchemaError::new(
                "src-openContent",
                format!("the `xs:any` of an open content may not carry `{offender}`"),
            )
            .at(node_path(&doc, child)));
        }
        let wildcard = any
            .map(|child| {
                crate::parse::wildcard(
                    &doc,
                    child,
                    info.target_namespace.as_deref(),
                    self.schema.version(),
                )
            })
            .transpose()?;
        // XSD 1.1 §3.4.2: `mode="none"` says there is no open content at all,
        // so an `xs:any` child would have nothing to describe — writing one is
        // a schema-representation error (`saxonData/Open/open036.bad`).
        if mode == "none" && wildcard.is_some() {
            return Err(SchemaError::new(
                "src-openContent",
                "`xs:openContent` with `mode=\"none\"` may not have an `xs:any` child",
            )
            .at(node_path(&doc, node)));
        }
        Ok(crate::components::OpenContent {
            interleave: mode != "suffix",
            wildcard: if mode == "none" { None } else { wildcard },
        })
    }

    fn build_simple_content(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> Result<(TypeId, Derivation, ComplexTypeInfo), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let children = content_children(&doc, node);
        let [child] = children.as_slice() else {
            return Err(SchemaError::new(
                "src-ct",
                "`xs:simpleContent` needs exactly one `restriction` or `extension`",
            )
            .at(node_path(&doc, node)));
        };
        let child = *child;
        let derivation = match xsd_name(&doc, child) {
            Some("restriction") => Derivation::Restriction,
            Some("extension") => Derivation::Extension,
            other => {
                return Err(SchemaError::new(
                    "src-ct",
                    format!(
                        "`xs:{}` may not appear inside `xs:simpleContent`",
                        other.unwrap_or("?")
                    ),
                )
                .at(node_path(&doc, child)));
            }
        };
        let base_name_text = attr(&doc, child, "base").ok_or_else(|| {
            SchemaError::new("src-ct", "a derivation needs a `base` attribute")
                .at(node_path(&doc, child))
        })?;
        let base_name =
            self.reference(info, child, base_name_text, redefine, RedefineKind::Type)?;
        let base = self.type_by_name(&base_name, &doc, child)?;
        self.ensure_type(base)?;
        self.check_derivation_allowed(base, derivation, &doc, child)?;

        let grand_children = content_children(&doc, child);
        let mut attribute_nodes = Vec::new();
        let mut facet_nodes = Vec::new();
        let mut inline_simple = None;
        let mut assertions = Vec::new();
        for member in &grand_children {
            match xsd_name(&doc, *member) {
                Some("attribute" | "attributeGroup" | "anyAttribute") => {
                    attribute_nodes.push(*member);
                }
                Some("simpleType") => inline_simple = Some(*member),
                Some("assert") => {
                    let assertion = Assertion {
                        test: attr(&doc, *member, "test").unwrap_or_default().to_owned(),
                        namespaces: crate::parse::namespace_context(&doc, *member),
                        xpath_default_namespace: crate::parse::xpath_default_namespace(
                            &doc,
                            *member,
                            info.target_namespace.as_deref(),
                        ),
                    };
                    crate::assertion::check_static(&assertion, true, "src-assert")
                        .map_err(|error| error.or_at(node_path(&doc, *member)))?;
                    assertions.push(assertion);
                }
                Some(name) if FACET_NAMES.contains(&name) => facet_nodes.push(*member),
                other => {
                    return Err(SchemaError::new(
                        "src-ct",
                        format!(
                            "`xs:{}` may not appear inside a `simpleContent` derivation",
                            other.unwrap_or("?")
                        ),
                    )
                    .at(node_path(&doc, *member)));
                }
            }
        }

        let base_definition = &self.schema.types[base.index()];
        let base_simple_content = match &base_definition.kind {
            TypeKind::Complex(complex) => match &complex.content {
                ContentType::Simple(inner) => Some(*inner),
                ContentType::Mixed(particle)
                    if derivation == Derivation::Restriction && particle.is_emptiable() =>
                {
                    // Restricting a mixed, ·emptiable· content type down to a
                    // simple one is allowed (§3.4.6 derivation-ok, case 1.4.2).
                    // ·Emptiable· is "can match the empty sequence", not "can
                    // match nothing else": a particle of
                    // `<xs:any minOccurs="0" maxOccurs="unbounded"/>` is
                    // emptiable and its type may be restricted to a simple one
                    // (`saxonData/CTA/cta0001`).
                    //
                    // §3.4.2.2 clause 1.2 grants this case only to a
                    // restriction that writes the simple type itself: there is
                    // nothing in a mixed base for the {content type} to be
                    // derived *from*, so without an `xs:simpleType` child the
                    // derivation has no simple type to name
                    // (`msData/complexType/ctD004`, restricting `xs:anyType`
                    // with nothing at all).
                    if inline_simple.is_none() {
                        return Err(SchemaError::new(
                            "src-ct.2.2",
                            format!(
                                "`{}` has mixed content, so a `simpleContent` restriction of it \
                                 must write an `xs:simpleType`",
                                self.schema.type_name(base)
                            ),
                        )
                        .at(node_path(&doc, child)));
                    }
                    None
                }
                _ => {
                    return Err(SchemaError::new(
                        "src-ct.2",
                        format!(
                            "`{}` does not have simple content, so `simpleContent` may not \
                             derive from it",
                            self.schema.type_name(base)
                        ),
                    )
                    .at(node_path(&doc, child)));
                }
            },
            TypeKind::Simple(_) => {
                if derivation == Derivation::Extension {
                    Some(base)
                } else {
                    return Err(SchemaError::new(
                        "src-ct.2",
                        "a `simpleContent` restriction must have a complex base type",
                    )
                    .at(node_path(&doc, child)));
                }
            }
        };

        let (own_uses, own_wildcard) =
            self.build_attribute_uses(info, &attribute_nodes, redefine)?;
        let (attribute_uses, attribute_wildcard) =
            self.merge_attributes(base, derivation, own_uses, own_wildcard, &doc, child)?;

        let content = match derivation {
            Derivation::Extension => match base_simple_content {
                Some(inner) => ContentType::Simple(inner),
                None => ContentType::Empty,
            },
            _ => {
                let restriction_base = match inline_simple {
                    Some(inline) => self.build_anonymous_type(info, inline, redefine)?,
                    None => base_simple_content.unwrap_or(self.schema.any_simple_type),
                };
                self.ensure_type(restriction_base)?;
                // Datatypes §3.2.1.3: "When a new datatype is defined by
                // facet-based restriction, `anySimpleType` must not be used as
                // the base type." §3.4.2.2 maps an `xs:simpleContent`
                // `xs:restriction` with no `xs:simpleType` child onto exactly
                // such a definition — one restricting the {content type} of
                // the base type, with the facets written here, of which there
                // may be none (Datatypes §2.4.2.1). A base whose simple
                // content is the ·simple ur-type definition· therefore leaves
                // nothing to restrict. XSD 1.0 had no such rule, and the suite
                // splits the verdict by version (`msData/simpleType/stZ007`,
                // `stZ047`, `stZ055`; W3C bug 14559).
                if self.schema.version().is_1_1()
                    && inline_simple.is_none()
                    && base_simple_content == Some(self.schema.any_simple_type())
                {
                    return Err(SchemaError::new(
                        "cos-st-restricts",
                        format!(
                            "the simple content of `{}` is `xs:anySimpleType`, which has no \
                             variety for a restriction to constrain",
                            self.schema.type_name(base)
                        ),
                    )
                    .at(node_path(&doc, child)));
                }
                // §3.4.6 *derivation-ok-restriction* clause 5.1.1: when both
                // the restriction and its base have simple content, the
                // restriction's simple type must be validly ·derived· from the
                // base's given the empty set. An `xs:simpleType` written
                // inside the `xs:restriction` replaces the base's outright, so
                // nothing else checks it: without this a `list` of `xs:int`
                // passes for a restriction of `xs:decimal`
                // (`msData/particles/particlesZ018`).
                if let Some(inner) = base_simple_content
                    && !self.schema.is_derived_from(restriction_base, inner)
                {
                    return Err(SchemaError::new(
                        "derivation-ok-restriction.5.1.1",
                        format!(
                            "`{}` is not derived from `{}`, which is the simple content of the \
                             base type",
                            self.schema.type_name(restriction_base),
                            self.schema.type_name(inner)
                        ),
                    )
                    .at(node_path(&doc, child)));
                }
                if facet_nodes.is_empty() {
                    ContentType::Simple(restriction_base)
                } else {
                    let base_simple = self.schema.types[restriction_base.index()]
                        .as_simple()
                        .ok_or_else(|| {
                            SchemaError::new(
                                "src-ct.2",
                                "the base of a `simpleContent` restriction has no simple type",
                            )
                            .at(node_path(&doc, child))
                        })?;
                    let variety = base_simple.variety.clone();
                    let primitive = base_simple.primitive;
                    let base_facets = base_simple.facets.clone();
                    let raw = self.collect_facets(info, &facet_nodes)?;
                    let facets =
                        apply_facets(&self.schema, restriction_base, &base_facets, &variety, &raw)
                            .map_err(|error| error.or_at(node_path(&doc, child)))?;
                    let derived = self.push_type_stub(None);
                    self.schema.types[derived.index()] = TypeDefinition {
                        name: None,
                        base: Some(restriction_base),
                        derivation: Derivation::Restriction,
                        final_: DerivationSet::empty(),
                        block: DerivationSet::empty(),
                        kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                            variety,
                            primitive,
                            facets,
                        })),
                    };
                    ContentType::Simple(derived)
                }
            }
        };

        // §3.4.2: the {assertions} of a derived complex type are the base's,
        // in order, followed by the `xs:assert` children written here — a
        // restriction may add assertions but never drop one.
        let assertions = inherited_assertions(&self.schema, base, assertions);
        Ok((
            base,
            derivation,
            ComplexTypeInfo {
                derivation,
                abstract_: false,
                attribute_uses,
                attribute_wildcard,
                content,
                content_model: None,
                assertions,
                open_content: None,
            },
        ))
    }

    /// XSD 1.1 §3.4.2: `xs:schema/@defaultAttributes` names an attribute group
    /// whose uses every complex type of the document acquires, unless the type
    /// says `defaultAttributesApply="false"`.
    ///
    /// The group's uses are added *after* the type's own, so an attribute the
    /// type declares itself keeps its own declaration.
    fn apply_default_attributes(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        complex: &mut ComplexTypeInfo,
        redefine: Option<&Redefine>,
    ) -> Result<(), SchemaError> {
        if !self.schema.version().is_1_1() {
            return Ok(());
        }
        let doc = Rc::clone(&info.doc);
        if !crate::parse::bool_attr(&doc, node, "defaultAttributesApply", true)? {
            return Ok(());
        }
        // §3.4.2.4 speaks of "the `<schema>` ancestor", which for a component
        // written inside `xs:override` is the ancestor it *ends up* with —
        // §F.2 copies the element into the overridden document, whose
        // `<schema>` attributes D2′ keeps
        // (`ibmData/instance_invalid/S3_4_2_4/s3_4_2_4ii08`). `xs:redefine`
        // performs no such transformation, so a redefinition keeps the
        // redefining document's defaults (`saxonData/Open/open044`).
        let host = self.defaults_document(info, node);
        let host_doc = Rc::clone(&host.doc);
        let Some(text) = attr(&host_doc, host.root, "defaultAttributes") else {
            return Ok(());
        };
        let name = self.reference(
            &host,
            host.root,
            text,
            redefine,
            RedefineKind::AttributeGroup,
        )?;
        let Some(group) = self.schema.global_attribute_groups.get(&name).copied() else {
            return Err(SchemaError::new(
                "src-schema",
                format!("`defaultAttributes` names the undeclared attribute group `{name}`"),
            )
            .at(node_path(&host_doc, host.root)));
        };
        self.ensure_attribute_group(group)?;
        // §3.4.2.4: the {attribute uses} are computed "as if there were an
        // `<attributeGroup>` [child] with empty content and a `ref`
        // [attribute] whose ·actual value· is the same as that of the
        // `defaultAttributes` [attribute] appearing after any other
        // `<attributeGroup>` [children]". Being an ordinary attribute-group
        // reference, it can collide with one — *Complex Type Definition
        // Properties Correct* clause 4 forbids "two distinct members of the
        // {attribute uses}" with the same expanded name
        // (`ibmData/schema_invalid/S3_4_2_4/s3_4_2_4si02`) — and it
        // contributes its {attribute wildcard} as well (§3.4.2.5,
        // `saxonData/Open/open044`).
        let uses = self.schema.attribute_groups[group.index()]
            .attribute_uses
            .clone();
        let group_wildcard = self.schema.attribute_groups[group.index()]
            .attribute_wildcard
            .clone();
        for use_ in uses {
            let name = self
                .schema
                .attribute_declaration(use_.attribute)
                .name
                .clone();
            let clash = complex.attribute_uses.iter().any(|existing| {
                self.schema.attribute_declaration(existing.attribute).name == name
                    && existing.attribute != use_.attribute
            });
            if clash {
                return Err(SchemaError::new(
                    "ct-props-correct.4",
                    format!(
                        "the default attribute group and this type both declare the attribute                          `{name}`"
                    ),
                )
                .at(node_path(&doc, node)));
            }
            let already = complex
                .attribute_uses
                .iter()
                .any(|existing| self.schema.attribute_declaration(existing.attribute).name == name);
            if !already {
                complex.attribute_uses.push(use_);
            }
        }
        if let Some(group_wildcard) = group_wildcard {
            complex.attribute_wildcard = Some(match complex.attribute_wildcard.take() {
                Some(own) => Wildcard {
                    namespace: own
                        .namespace
                        .union_with(&group_wildcard.namespace)
                        .unwrap_or(crate::components::NamespaceConstraint::Any),
                    process_contents: own.process_contents,
                    disallowed: crate::components::union_disallowed_names(&own, &group_wildcard),
                },
                None => group_wildcard,
            });
        }
        Ok(())
    }

    /// The schema document whose `<schema>` attributes supply the
    /// `defaultAttributes` and `xs:defaultOpenContent` of a component written
    /// at `node`.
    ///
    /// That is the document the component is written in — except inside
    /// `xs:override`, where §F.2's transformation copies the element into the
    /// *overridden* document, so that document's `<schema>` governs.
    fn defaults_document(&self, info: &DocInfo, node: NodeId) -> DocInfo {
        let doc = Rc::clone(&info.doc);
        let Some(parent) = doc.parent(node) else {
            return info.clone();
        };
        if xsd_name(&doc, parent) != Some("override") {
            return info.clone();
        }
        let Some(index) = self.docs.iter().position(|other| other.uri == info.uri) else {
            return info.clone();
        };
        let Some(location) = self.locations.get(&(index, parent)) else {
            return info.clone();
        };
        match self.docs.iter().find(|other| other.uri == *location) {
            Some(target) => target.clone(),
            None => info.clone(),
        }
    }

    fn build_complex_content(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
        outer_mixed: bool,
        outer_mixed_written: Option<bool>,
    ) -> Result<(TypeId, Derivation, ComplexTypeInfo), SchemaError> {
        let doc = Rc::clone(&info.doc);
        // §3.4.2.3.3 clause 1 (·effective mixed·): `complexContent/@mixed`
        // wins over `complexType/@mixed`. §3.4.3 (*Complex Type Definition
        // Representation OK*) clause 5 makes the two disagreeing an error, so
        // the precedence can never actually be observed in a conforming
        // schema document (`saxonData/Complex/complex002`,
        // `ibmData/schema_invalid/S3_4_1/s3_4_1si09`).
        let mixed = match crate::parse::attr(&doc, node, "mixed") {
            Some(_) => {
                let here = crate::parse::bool_attr(&doc, node, "mixed", outer_mixed)?;
                if let Some(outer) = outer_mixed_written
                    && outer != here
                {
                    return Err(SchemaError::new(
                        "src-ct.5",
                        "the `mixed` attribute is written on both `xs:complexType` and \
                         `xs:complexContent` with different values",
                    )
                    .at(node_path(&doc, node)));
                }
                here
            }
            None => outer_mixed,
        };
        let children = content_children(&doc, node);
        let [child] = children.as_slice() else {
            return Err(SchemaError::new(
                "src-ct",
                "`xs:complexContent` needs exactly one `restriction` or `extension`",
            )
            .at(node_path(&doc, node)));
        };
        let child = *child;
        let derivation = match xsd_name(&doc, child) {
            Some("restriction") => Derivation::Restriction,
            Some("extension") => Derivation::Extension,
            other => {
                return Err(SchemaError::new(
                    "src-ct",
                    format!(
                        "`xs:{}` may not appear inside `xs:complexContent`",
                        other.unwrap_or("?")
                    ),
                )
                .at(node_path(&doc, child)));
            }
        };
        let base_name_text = attr(&doc, child, "base").ok_or_else(|| {
            SchemaError::new("src-ct", "a derivation needs a `base` attribute")
                .at(node_path(&doc, child))
        })?;
        let base_name =
            self.reference(info, child, base_name_text, redefine, RedefineKind::Type)?;
        let base = self.type_by_name(&base_name, &doc, child)?;
        self.ensure_type(base)?;
        if self.schema.types[base.index()].as_complex().is_none() {
            return Err(SchemaError::new(
                "src-ct.1",
                format!(
                    "`{}` is a simple type, so `complexContent` may not derive from it",
                    self.schema.type_name(base)
                ),
            )
            .at(node_path(&doc, child)));
        }
        self.check_derivation_allowed(base, derivation, &doc, child)?;

        let grand_children = content_children(&doc, child);
        // The schema for schemas writes the content model of an
        // `xs:restriction` as
        //
        // ```xml
        // <xs:choice minOccurs="0">
        //   <xs:sequence>
        //     <xs:element ref="xs:openContent" minOccurs="0"/>
        //     <xs:group ref="xs:typeDefParticle"/>
        //   </xs:sequence>
        //   <xs:group ref="xs:simpleRestrictionModel"/>
        // </xs:choice>
        // ```
        //
        // — the particle is *required* once an `xs:openContent` is written.
        // `xs:extension` has the two as independent optional members, so only
        // a restriction is bound by this (`saxonData/Complex/complex018`, W3C
        // bug 16786: the XML representation summary appeared to allow it and
        // the schema for schemas did not, which the schema for schemas won).
        if derivation == Derivation::Restriction
            && let Some(written) = grand_children
                .iter()
                .find(|member| xsd_name(&doc, **member) == Some("openContent"))
            && !grand_children.iter().any(|member| {
                matches!(
                    xsd_name(&doc, *member),
                    Some("sequence" | "choice" | "all" | "group")
                )
            })
        {
            return Err(SchemaError::new(
                "src-ct",
                "an `xs:restriction` that writes an `xs:openContent` must write its content \
                 model too",
            )
            .at(node_path(&doc, *written)));
        }
        let (own_content, own_uses, own_wildcard, assertions, open_content) =
            self.build_content_shorthand(info, child, &grand_children, redefine, mixed)?;
        let (attribute_uses, attribute_wildcard) =
            self.merge_attributes(base, derivation, own_uses, own_wildcard, &doc, child)?;

        let content = match derivation {
            Derivation::Restriction => own_content,
            _ => self.extend_content(base, own_content, mixed, &doc, child)?,
        };
        // §3.4.2.3.3: the document's `xs:defaultOpenContent` reaches a type
        // that writes no `xs:openContent` of its own, and — when it says
        // `appliesToEmpty="false"` — only one whose {content type} is not
        // empty. That is the content type the definition *ends up* with: an
        // extension whose own particle is empty takes the base's content, and
        // the base's may be anything but. `build_content_shorthand` decided
        // this on the explicit content alone, which is all it can see, so the
        // question is asked again here now that the derivation has been
        // carried out (`saxonData/Open/open046.bad`; its neighbour `open031`
        // is the same schema with an `<xs:openContent mode="none"/>` on the
        // extension, which keeps the default out and is valid).
        let open_content = match open_content {
            Some(written) => Some(written),
            None if !matches!(content, ContentType::Empty) => {
                let host = self.defaults_document(info, child);
                let host_doc = Rc::clone(&host.doc);
                match default_open_content(&host_doc, host.root) {
                    Some(default) => Some(self.build_open_content(&host, default)?),
                    None => None,
                }
            }
            None => None,
        };

        // §3.4.2: the {assertions} of a derived complex type are the base's,
        // in order, followed by the `xs:assert` children written here — a
        // restriction may add assertions but never drop one.
        let assertions = inherited_assertions(&self.schema, base, assertions);
        // `open_content` here stays exactly what *this* definition wrote (or
        // defaulted to) — `check_open_content_extension`/
        // `check_open_content_restriction` (run in a later pass, over the
        // fully built `self.schema.types`) need that raw value to detect an
        // extension or restriction that conflicts with its base, which they
        // cannot do once it has already been folded together with the
        // base's. The *effective*, base-merged open content a derived
        // extension's instances are actually validated against — the base's
        // wildcard reaching past whatever particles this extension appends —
        // is instead computed on demand by `validate::content::
        // effective_open_content`, which is free to recurse through
        // `{derivation}` because by the time validation runs, every type is
        // already known to be a valid extension/restriction of its base.
        Ok((
            base,
            derivation,
            ComplexTypeInfo {
                derivation,
                abstract_: false,
                attribute_uses,
                attribute_wildcard,
                content,
                content_model: None,
                assertions,
                open_content,
            },
        ))
    }

    fn extend_content(
        &mut self,
        base: TypeId,
        own: ContentType,
        mixed: bool,
        doc: &oxixml_dom::Document,
        node: NodeId,
    ) -> Result<ContentType, SchemaError> {
        let base_content = self.schema.types[base.index()]
            .as_complex()
            .map(|complex| complex.content.clone())
            .unwrap_or(ContentType::Empty);
        let base_particle = base_content.particle().cloned();
        let own_particle = own.particle().cloned();
        Ok(match (base_particle, own_particle) {
            (None, None) => {
                if matches!(base_content, ContentType::Simple(_)) {
                    // §3.4.3 *src-ct.1*: XSD 1.1 narrowed which base an
                    // `xs:complexContent` may name to one whose {content type}
                    // is not simple. Under 1.0 the derivation is legal and
                    // clause 4.2.2 hands the base's simple content type
                    // straight through, which is why the suite splits the
                    // verdict by version (`msData/particles/particlesZ031`;
                    // W3C bug 15629).
                    if self.schema.version().is_1_1() {
                        return Err(SchemaError::new(
                            "src-ct.1",
                            format!(
                                "`{}` has simple content, so `complexContent` may not derive \
                                 from it in XML Schema 1.1",
                                self.schema.type_name(base)
                            ),
                        )
                        .at(node_path(doc, node)));
                    }
                    base_content
                } else if mixed {
                    ContentType::Mixed(Box::new(empty_particle()))
                } else {
                    ContentType::Empty
                }
            }
            (None, Some(particle)) => {
                // §3.4.6 *cos-ct-extends* clause 1.4.3.2: an extension that
                // brings a particle of its own may only sit on a base whose
                // {content type} is empty (clause 1.4.3.2.1) or is mixed or
                // element-only to match (clause 1.4.3.2.2.1). A base with
                // **simple** content is neither — its instances carry a value
                // where the extension wants elements — so element content
                // cannot be added to it (`msData/additional/addB033`,
                // `addB036`).
                if matches!(base_content, ContentType::Simple(_)) {
                    return Err(SchemaError::new(
                        "cos-ct-extends.1.4.3.2",
                        format!(
                            "`{}` has simple content, so an extension of it may not add element \
                             content",
                            self.schema.type_name(base)
                        ),
                    )
                    .at(node_path(doc, node)));
                }
                wrap_content(particle, mixed)
            }
            (Some(particle), None) => {
                let base_mixed = matches!(base_content, ContentType::Mixed(_));
                // §3.4.2.3.3 clause 3.1: with no particle of its own the
                // ·effective content· is empty only when the ·effective
                // mixed· is false; then clause 4.2.2 hands the base's
                // {content type} straight through. When the ·effective mixed·
                // *is* true the ·effective content· is a synthesized empty
                // sequence, clause 4.2.3 applies, and the derived variety is
                // `mixed` — which clause 1.4.3.2.2.1 of *Derivation Valid
                // (Extension)* forbids over an element-only base
                // (`saxonData/Complex/complex025`).
                if mixed && !base_mixed {
                    return Err(mixed_extension_error(base_mixed, doc, node));
                }
                wrap_content(particle, mixed || base_mixed)
            }
            (Some(base_particle), Some(own_particle)) => {
                // §3.4.6.2 clause 1.4.3.2.2.1: "Both B and T have {content
                // type} . {variety} = mixed or both have {content type} .
                // {variety} = element-only." T's variety comes from the
                // ·effective mixed· alone (clause 4.2.3), so a `mixed` that
                // disagrees with the base is an error rather than something
                // to paper over (`saxonData/Complex/complex026`,
                // `ibmData/schema_invalid/S3_4_1/s3_4_1si08`).
                let base_mixed = matches!(base_content, ContentType::Mixed(_));
                if mixed != base_mixed {
                    return Err(mixed_extension_error(base_mixed, doc, node));
                }
                if matches!(base_particle.term, Term::Group(ref group) if group.compositor == Compositor::All)
                    || matches!(own_particle.term, Term::Group(ref group) if group.compositor == Compositor::All)
                {
                    if self.schema.version() == XsdVersion::V1_0 {
                        return Err(SchemaError::new(
                            "cos-all-limited.1.2",
                            "an `all` group cannot be extended in XML Schema 1.0",
                        )
                        .at(node_path(doc, node)));
                    }
                    let merged = merge_all_groups(&base_particle, &own_particle)?;
                    return Ok(wrap_content(merged, mixed));
                }
                let joined = Particle {
                    min_occurs: 1,
                    max_occurs: MaxOccurs::Bounded(1),
                    term: Term::Group(ModelGroup {
                        compositor: Compositor::Sequence,
                        particles: vec![base_particle, own_particle],
                    }),
                };
                wrap_content(joined, mixed)
            }
        })
    }

    fn check_derivation_allowed(
        &self,
        base: TypeId,
        derivation: Derivation,
        doc: &oxixml_dom::Document,
        node: NodeId,
    ) -> Result<(), SchemaError> {
        let definition = &self.schema.types[base.index()];
        if definition.final_.contains(derivation) {
            return Err(SchemaError::new(
                "cos-ct-derived-ok",
                format!(
                    "`{}` is `final=\"{derivation}\"`",
                    self.schema.type_name(base)
                ),
            )
            .at(node_path(doc, node)));
        }
        Ok(())
    }

    fn merge_attributes(
        &mut self,
        base: TypeId,
        derivation: Derivation,
        own_uses: Vec<AttributeUse>,
        own_wildcard: Option<Wildcard>,
        doc: &oxixml_dom::Document,
        node: NodeId,
    ) -> Result<(Vec<AttributeUse>, Option<Wildcard>), SchemaError> {
        let base_complex = self.schema.types[base.index()].as_complex();
        let (base_uses, base_wildcard) = match base_complex {
            Some(complex) => (
                complex.attribute_uses.clone(),
                complex.attribute_wildcard.clone(),
            ),
            None => (Vec::new(), None),
        };
        let mut merged: Vec<AttributeUse> = Vec::new();
        for base_use in base_uses {
            let name = self.schema.attributes[base_use.attribute.index()]
                .name
                .clone();
            let overridden = own_uses
                .iter()
                .any(|own| self.schema.attributes[own.attribute.index()].name == name);
            if !overridden {
                merged.push(base_use);
            }
        }
        for own in own_uses {
            if own.kind == AttributeUseKind::Prohibited {
                continue;
            }
            let name = self.schema.attributes[own.attribute.index()].name.clone();
            if merged
                .iter()
                .any(|existing| self.schema.attributes[existing.attribute.index()].name == name)
            {
                return Err(SchemaError::new(
                    "ct-props-correct.4",
                    format!("the attribute `{name}` is used twice"),
                )
                .at(node_path(doc, node)));
            }
            merged.push(own);
        }
        merged.retain(|use_| use_.kind != AttributeUseKind::Prohibited);
        let wildcard = match derivation {
            Derivation::Extension => match (base_wildcard, own_wildcard) {
                (Some(base), Some(own)) => {
                    let namespace = base
                        .namespace
                        .union_with(&own.namespace)
                        .unwrap_or(crate::components::NamespaceConstraint::Any);
                    self.check_union_expressible(&namespace, doc, node)?;
                    Some(Wildcard {
                        namespace,
                        process_contents: base.process_contents,
                        // §3.10.6.3: the union disallows a name only when
                        // *neither* wildcard would have allowed it.
                        disallowed: crate::components::union_disallowed_names(&base, &own),
                    })
                }
                (Some(only), None) | (None, Some(only)) => Some(only),
                (None, None) => None,
            },
            _ => own_wildcard,
        };
        Ok((merged, wildcard))
    }

    /// §3.10.6 *Attribute Wildcard Union*, as XSD 1.0 Errata E1-10 restates it:
    /// clause 5.3 has no value to give.
    ///
    /// A 1.0 namespace constraint is `any`, a set of namespace names and
    /// ·absent·, a pair of `not` and a namespace name — which does not match
    /// ·absent· either — or (E1-10's addition) a pair of `not` and ·absent·.
    /// Uniting `not(N)` with a set that contains ·absent· but not `N` would
    /// have to say "anything but `N`, ·absent· included", and none of those
    /// four shapes says that: the union is *not expressible*, and a schema that
    /// asks for one is in error. XSD 1.1 replaced the whole scheme with a
    /// negation of a *set*, which expresses it, so the suite splits the verdict
    /// by version (`msData/wildcards/wildZ013`).
    fn check_union_expressible(
        &self,
        namespace: &crate::components::NamespaceConstraint,
        doc: &oxixml_dom::Document,
        node: NodeId,
    ) -> Result<(), SchemaError> {
        if self.schema.version().is_1_1() {
            return Ok(());
        }
        if let crate::components::NamespaceConstraint::Not(excluded) = namespace
            && !excluded.contains(&None)
        {
            return Err(SchemaError::new(
                "cos-aw-union",
                "the union of the two attribute wildcards is not expressible in XML Schema 1.0",
            )
            .at(node_path(doc, node)));
        }
        Ok(())
    }
}

/// The failure of clause 1.4.3.2.2.1 of *Derivation Valid (Extension)*
/// (§3.4.6.2): an extension whose content variety does not agree with its
/// base's.
fn mixed_extension_error(
    base_mixed: bool,
    doc: &oxixml_dom::Document,
    node: NodeId,
) -> SchemaError {
    let (base, derived) = if base_mixed {
        ("mixed", "element-only")
    } else {
        ("element-only", "mixed")
    };
    SchemaError::new(
        "cos-ct-extends.1.4.3.2.2.1",
        format!(
            "an extension must have the same content variety as its base: the base is {base} but \
             this type is {derived}"
        ),
    )
    .at(node_path(doc, node))
}

fn wrap_content(particle: Particle, mixed: bool) -> ContentType {
    if mixed {
        ContentType::Mixed(Box::new(particle))
    } else {
        ContentType::ElementOnly(Box::new(particle))
    }
}

fn merge_all_groups(base: &Particle, own: &Particle) -> Result<Particle, SchemaError> {
    let all_group = |particle: &Particle| match &particle.term {
        Term::Group(group) if group.compositor == Compositor::All => Some(group.particles.clone()),
        _ => None,
    };
    // §3.4.6.2 *cos-ct-extends* clause 1.4.3.2.2.3: the two particles are
    // joined into an `all` group only when *both* of them are one; a base whose
    // ·effective content· is the synthesized empty **sequence** of clause
    // 3.1.2 — which is what a `mixed` base with no content model has — cannot
    // be extended by an `all` group, because the alternative is a `sequence`
    // holding it, and an `all` group may not sit inside one
    // (`saxonData/All/all308`; W3C bug 6202).
    let (Some(base_members), Some(own_members)) = (all_group(base), all_group(own)) else {
        return Err(SchemaError::new(
            "cos-ct-extends.1.4.3.2",
            "an `all` group can only be extended by, and only extend, another `all` group",
        ));
    };
    // Same clause: the joined group keeps one {min occurs}, so the two must
    // agree — an `xs:all minOccurs="0"` extending an `xs:all` that is required
    // says two different things at once (`saxonData/All/all313`).
    if base.min_occurs != own.min_occurs {
        return Err(SchemaError::new(
            "cos-ct-extends.1.4.3.2",
            format!(
                "an `all` group extending an `all` group must have the same `minOccurs`, but \
                 the base has {} and the extension {}",
                base.min_occurs, own.min_occurs
            ),
        ));
    }
    let mut particles = base_members;
    particles.extend(own_members);
    Ok(Particle {
        min_occurs: base.min_occurs,
        max_occurs: MaxOccurs::Bounded(1),
        term: Term::Group(ModelGroup {
            compositor: Compositor::All,
            particles,
        }),
    })
}

/// An empty `sequence`, used as the particle of an empty mixed content type.
pub(crate) fn empty_particle() -> Particle {
    Particle {
        min_occurs: 1,
        max_occurs: MaxOccurs::Bounded(1),
        term: Term::Group(ModelGroup {
            compositor: Compositor::Sequence,
            particles: Vec::new(),
        }),
    }
}

/// Whether a particle can match nothing but the empty sequence.
pub(crate) fn particle_is_empty(particle: &Particle) -> bool {
    if particle.max_occurs.is_zero() {
        return true;
    }
    match &particle.term {
        Term::Element(_) | Term::Wildcard(_) => false,
        // §3.4.2.3.3 clause 2.1.3: a branch-less `xs:choice` counts as ·empty
        // content· only when its `minOccurs` is 0. With `minOccurs` 1 it is a
        // real particle that accepts the *empty language*, so even an empty
        // element is invalid against it (`saxonData/Complex/complex022`) —
        // clause 2.1.2 grants the exemption to `all` and `sequence` alone.
        Term::Group(group) if group.compositor == Compositor::Choice => {
            if group.particles.is_empty() {
                return particle.min_occurs == 0;
            }
            group.particles.iter().all(particle_is_empty)
        }
        Term::Group(group) => group.particles.iter().all(particle_is_empty),
    }
}

/// The `xs:defaultOpenContent` of a schema document, if it has one.
fn default_open_content(doc: &oxixml_dom::Document, root: NodeId) -> Option<NodeId> {
    crate::parse::child_elements(doc, root)
        .into_iter()
        .find(|child| xsd_name(doc, *child) == Some("defaultOpenContent"))
}

/// The {assertions} of a type derived from `base`: the base's, then `own`.
fn inherited_assertions(
    schema: &crate::schema::Schema,
    base: TypeId,
    own: Vec<Assertion>,
) -> Vec<Assertion> {
    match schema.type_definition(base).as_complex() {
        Some(complex) if !complex.assertions.is_empty() => {
            let mut all = complex.assertions.clone();
            all.extend(own);
            all
        }
        _ => own,
    }
}
