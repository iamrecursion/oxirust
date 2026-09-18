//! Schema assembly: loading schema documents, registering their global names,
//! and turning them into the component model.

mod checks;
mod document;
mod licensing;
mod particles;
mod restriction;
mod subsume;
mod types;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use oxixml_dom::{Document, NodeId};
use oxixml_qname::ExpandedName;

use crate::components::{
    AttrGroupId, AttributeDeclaration, AttributeGroupDefinition, AttributeId, DerivationSet,
    ElementDeclaration, ElementId, Facets, GroupId, ModelGroupDefinition, NotationDeclaration,
    NotationId, SimpleTypeInfo, TypeDefinition, TypeId, TypeKind, ValueConstraint, Variety,
};
use crate::error::SchemaError;
use crate::names::XsdVersion;
use crate::parse::{
    attr, attr_token, child_elements, derivation_set, node_path, resolve_qname, xsd_name,
};
use crate::resolver::{FileSystemResolver, SchemaResolver};
use crate::schema::Schema;
use crate::{builtins, simple};

/// How far a component has got through assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BuildState {
    /// Registered, body not built yet.
    NotBuilt,
    /// Currently being built (a cycle would come back here).
    Building,
    /// Finished.
    Built,
}

/// What a `redefine` replaced, so that a self-reference inside it resolves to
/// the original component rather than to the replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RedefineKind {
    /// `xs:simpleType` or `xs:complexType`.
    Type,
    /// `xs:group`.
    Group,
    /// `xs:attributeGroup`.
    AttributeGroup,
}

/// The self-reference remapping in force inside one `xs:redefine` child.
#[derive(Debug, Clone)]
pub(crate) struct Redefine {
    pub kind: RedefineKind,
    pub real: ExpandedName,
    pub hidden: ExpandedName,
}

/// Where a component's definition is written.
#[derive(Debug, Clone)]
pub(crate) struct Source {
    pub doc: usize,
    pub node: NodeId,
    pub redefine: Option<Redefine>,
}

/// A loaded schema document together with the defaults its `xs:schema` element
/// establishes.
#[derive(Debug, Clone)]
pub(crate) struct DocInfo {
    /// Where the document sits in [`Compiler::docs`], so that a component
    /// reached through it can record a [`Source`] pointing back at it.
    pub index: usize,
    pub uri: String,
    pub doc: Rc<Document>,
    pub root: NodeId,
    pub target_namespace: Option<String>,
    /// Which namespaces this document may name in a `QName` reference — see
    /// [`licensing`]. Shared rather than copied, because [`Compiler::info`]
    /// clones a whole `DocInfo` for every component it builds.
    pub licence: Rc<licensing::Licence>,
    /// Set when the document was pulled in by a *chameleon* include: it has no
    /// `targetNamespace` of its own, so unqualified references inside it name
    /// components of the including namespace.
    pub chameleon: bool,
    /// Set when an `xs:import` named a namespace the document does not have:
    /// the reference is in error and the document contributes nothing.
    pub disabled: bool,
    pub element_form_qualified: bool,
    pub attribute_form_qualified: bool,
    pub block_default: DerivationSet,
    pub final_default: DerivationSet,
}

/// A set of schema documents that assemble into one [`Schema`].
///
/// # Examples
///
/// ```
/// use oxixml_schema::SchemaSet;
///
/// let mut set = SchemaSet::new();
/// set.add_document(None, r#"
///   <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
///     <xs:element name="greeting" type="xs:string"/>
///   </xs:schema>"#)?;
/// let schema = set.compile()?;
/// assert!(schema.validate_str("<greeting>hello</greeting>").valid);
/// assert!(!schema.validate_str("<greeting><b/></greeting>").valid);
/// # Ok::<(), oxixml_schema::SchemaError>(())
/// ```
#[derive(Debug)]
pub struct SchemaSet {
    version: XsdVersion,
    resolver: Box<dyn SchemaResolver>,
    docs: Vec<DocInfo>,
    seen: HashSet<String>,
    /// The text of every schema document taken in so far, paired with the
    /// namespace a *chameleon* include would absorb it into, so that a second
    /// `xs:import` reaching the same document under a different spelling of
    /// its location is recognised as the document it already has.
    contents: HashSet<(Option<String>, String)>,
    inline_count: usize,
    /// Where every `xs:include`, `xs:redefine` and `xs:override` resolved to,
    /// keyed by the document it is written in and the element itself. The
    /// resolver runs once, at load time, and *src-redefine.1* reads the answer
    /// back here rather than resolving a second time.
    locations: HashMap<(usize, NodeId), String>,
}

impl Default for SchemaSet {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaSet {
    /// How deeply the elements of a schema document may nest before assembly
    /// refuses to read it.
    ///
    /// Everything that reads a schema document — conditional inclusion, the
    /// schema representation constraints, the particle and type builders —
    /// walks its element tree by recursion, so an arbitrarily deep document
    /// would otherwise consume an arbitrary amount of stack and abort the
    /// process. Checking the depth once, as the document is taken in, bounds
    /// every one of those walks at the door; a document that nests deeper is
    /// reported as a `src-max-depth` error like any other malformed one.
    ///
    /// The ceiling is far beyond what a schema document written by hand or by a
    /// tool reaches — the deepest in the W3C XML Schema Test Suite nests
    /// sixteen elements — and is deliberately the same as the one instance
    /// assessment observes,
    /// [`validate::MAX_DEPTH`](crate::validate::MAX_DEPTH).
    pub const MAX_DEPTH: usize = 256;

    /// How long a chain of named definitions may get before assembly refuses to
    /// follow it.
    ///
    /// A type whose base is a type whose base is a type, or a model group
    /// referring to a model group referring to a model group, is built by
    /// recursion: one turn for each link. A *cycle* among them is already
    /// caught — the component half-built when the chain comes back to it is
    /// still marked as being built — but a chain is perfectly legal and as long
    /// as the document cares to make it, and 20 000 `xs:group` definitions fit
    /// comfortably in a megabyte.
    ///
    /// The ceiling is half of [`MAX_DEPTH`](Self::MAX_DEPTH) because a link
    /// costs far more stack than an element of the document tree does: each one
    /// carries a whole particle and the definition it is being read into, over
    /// several frames, and an unoptimized build overflows an ordinary thread
    /// stack somewhere under two hundred links. Schemas written to be read
    /// nest a handful of links deep, so the ceiling is out of their way.
    pub const MAX_DEFINITION_CHAIN: usize = Self::MAX_DEPTH / 2;

    /// A new, empty set targeting XML Schema 1.0 and resolving
    /// `schemaLocation`s against the file system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: XsdVersion::V1_0,
            resolver: Box::new(FileSystemResolver::new()),
            docs: Vec::new(),
            seen: HashSet::new(),
            contents: HashSet::new(),
            inline_count: 0,
            locations: HashMap::new(),
        }
    }

    /// Selects the version of XML Schema to assemble under.
    #[must_use]
    pub fn with_version(mut self, version: XsdVersion) -> Self {
        self.version = version;
        self
    }

    /// Installs a custom resolver for `schemaLocation`s.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Box<dyn SchemaResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// The version this set assembles under.
    #[must_use]
    pub fn version(&self) -> XsdVersion {
        self.version
    }

    /// Adds a schema document, given its text and (optionally) the URI it came
    /// from — which becomes the base for the `schemaLocation`s inside it.
    ///
    /// References the document makes are followed immediately through the
    /// resolver.
    ///
    /// # Errors
    ///
    /// Returns a `src-…` error when the text is not well-formed XML or its
    /// document element is not `xs:schema`.
    pub fn add_document(&mut self, uri: Option<&str>, text: &str) -> Result<(), SchemaError> {
        let uri = match uri {
            Some(uri) => uri.to_owned(),
            None => {
                self.inline_count += 1;
                format!("#inline{}", self.inline_count)
            }
        };
        if let Some(index) = self.push_document(uri, text, None, false)? {
            self.load_references(index)?;
        }
        Ok(())
    }

    /// Adds a schema document read from the file system.
    ///
    /// # Errors
    ///
    /// Returns `src-include` when the file cannot be read, and the same errors
    /// as [`add_document`](Self::add_document) otherwise.
    pub fn add_file(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), SchemaError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| {
            SchemaError::new(
                "src-include",
                format!("cannot read `{}`: {error}", path.display()),
            )
        })?;
        // A schema document is an XML document, so its encoding is whatever
        // XML 1.0 Appendix F says it is — not necessarily UTF-8.
        let text = crate::source::decode(&bytes).map_err(|reason| {
            SchemaError::new(
                "src-include",
                format!("cannot read `{}`: {reason}", path.display()),
            )
        })?;
        self.add_document(Some(&path.to_string_lossy()), &text)
    }

    /// Whether any schema document has been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// The URIs of the loaded documents, in load order.
    pub fn document_uris(&self) -> impl Iterator<Item = &str> {
        self.docs.iter().map(|info| info.uri.as_str())
    }

    /// Assembles every document into one compiled [`Schema`].
    ///
    /// # Errors
    ///
    /// Returns the first schema-representation or schema-component constraint
    /// that the documents break.
    pub fn compile(self) -> Result<Schema, SchemaError> {
        let schema = builtins::build(self.version)?;
        let mut compiler = Compiler {
            schema,
            docs: self.docs,
            type_source: HashMap::new(),
            type_state: Vec::new(),
            element_source: HashMap::new(),
            element_state: HashMap::new(),
            attribute_source: HashMap::new(),
            attribute_state: HashMap::new(),
            group_source: HashMap::new(),
            group_state: Vec::new(),
            attr_group_source: HashMap::new(),
            attr_group_state: Vec::new(),
            notation_source: HashMap::new(),
            keyref_pending: Vec::new(),
            hidden_counter: 0,
            overridden: HashSet::new(),
            redefine_restrictions: Vec::new(),
            locations: self.locations,
            idc_ref_pending: Vec::new(),
            chain_depth: 0,
            attr_group_edges: Vec::new(),
            current_attr_group: None,
            pending_element_types: Vec::new(),
        };
        // The built-in components occupy the first slots of every arena and are
        // finished before assembly starts.
        compiler.type_state = vec![BuildState::Built; compiler.schema.types.len()];
        compiler.group_state = vec![BuildState::Built; compiler.schema.groups.len()];
        compiler.attr_group_state = vec![BuildState::Built; compiler.schema.attribute_groups.len()];
        compiler.run()?;
        Ok(compiler.schema)
    }

    fn push_document(
        &mut self,
        uri: String,
        text: &str,
        chameleon_namespace: Option<String>,
        deduplicate: bool,
    ) -> Result<Option<usize>, SchemaError> {
        // A document reached a second time contributes nothing new, whichever
        // of `xs:include`, `xs:import`, `xs:redefine` or `xs:override` led back
        // to it, so the load simply stops here rather than going round again.
        //
        // Circularity is not by itself an error — `saxonData/Override/over023`
        // is titled "A permissible circular `xs:override`" — and for the one
        // shape where it might be, a circular `xs:redefine`, the test suite
        // contradicts itself: `msData/schema/schU1` is a pair of documents that
        // redefine each other and is expected **valid** (as a test the working
        // group recorded as disputed, W3C bug 4135), while
        // `ibmData/schema_invalid/S4_2_4/s4_2_4si01` is the same shape and is
        // expected **invalid**. Closing the cycle silently answers the first
        // and not the second; that is the trade, recorded here rather than
        // hidden.
        if !self.seen.insert(uri.clone()) {
            return Ok(None);
        }
        // §4.2.6.2: the `schemaLocation` of an `xs:import` is a *hint*, and a
        // processor that already has the schema components the import asks for
        // is free to ignore it. Two spellings of one location — `Schz012_b.xsd`
        // beside `schZ012_b.xsd`, which name the same file on a case-folding
        // file system — otherwise read the same document twice and collide with
        // themselves over every global it declares
        // (`msData/schema/schZ012_b`, `schZ012_c`). Identical text means
        // identical `targetNamespace` and identical components, so the second
        // reading contributes nothing; `xs:include`, `xs:redefine` and
        // `xs:override` are left alone, since what they bring in depends on the
        // document that names them rather than only on the text.
        //
        // The namespace a *chameleon* include would absorb the document into is
        // part of the key, because the same text read into two namespaces is
        // two different sets of components.
        let known = !self
            .contents
            .insert((chameleon_namespace.clone(), text.to_owned()));
        if deduplicate && known {
            return Ok(None);
        }
        let mut doc = oxixml_dom::build::parse_str(text).map_err(|error| {
            SchemaError::new("src-schema", format!("`{uri}` is not well-formed: {error}"))
        })?;
        // The depth is checked before anything walks the tree, so that every
        // recursion below this point is bounded by construction.
        if let Some(root) = doc.document_element()
            && nesting_exceeds(&doc, root, Self::MAX_DEPTH)
        {
            return Err(SchemaError::new(
                "src-max-depth",
                format!(
                    "the elements of `{uri}` nest deeper than the maximum of {} elements",
                    Self::MAX_DEPTH
                ),
            ));
        }
        // Conditional inclusion runs before anything else looks at the
        // document: a guarded element that this version does not want must not
        // even be checked against the schema for schemas.
        let mut disabled = false;
        if let Some(root) = doc.document_element() {
            disabled = !crate::versioning::prune(&mut doc, root, self.version)?;
        }
        let doc = doc;
        let root = doc.document_element().ok_or_else(|| {
            SchemaError::new("src-schema", format!("`{uri}` has no document element"))
        })?;
        if xsd_name(&doc, root) != Some("schema") {
            return Err(SchemaError::new(
                "src-schema",
                format!("the document element of `{uri}` is not `xs:schema`"),
            ));
        }
        let declared_namespace = attr(&doc, root, "targetNamespace").map(str::to_owned);
        let chameleon = declared_namespace.is_none() && chameleon_namespace.is_some();
        let target_namespace = declared_namespace.or(chameleon_namespace);
        let block_default = match attr(&doc, root, "blockDefault") {
            Some(text) => derivation_set(
                text,
                &["extension", "restriction", "substitution"],
                "blockDefault",
            )?,
            None => DerivationSet::empty(),
        };
        let final_default = match attr(&doc, root, "finalDefault") {
            Some(text) => derivation_set(
                text,
                &["extension", "restriction", "list", "union"],
                "finalDefault",
            )?,
            None => DerivationSet::empty(),
        };
        crate::represent::check_document(&doc, root, self.version)?;
        document::check(&doc, root)?;
        let element_form_qualified =
            attr_token(&doc, root, "elementFormDefault").as_deref() == Some("qualified");
        let attribute_form_qualified =
            attr_token(&doc, root, "attributeFormDefault").as_deref() == Some("qualified");
        let licence = Rc::new(licensing::Licence::of(
            &doc,
            root,
            target_namespace.as_deref(),
        ));
        self.docs.push(DocInfo {
            index: self.docs.len(),
            uri,
            doc: Rc::new(doc),
            root,
            target_namespace,
            licence,
            chameleon,
            disabled,
            element_form_qualified,
            attribute_form_qualified,
            block_default,
            final_default,
        });
        Ok(Some(self.docs.len() - 1))
    }

    fn load_references(&mut self, index: usize) -> Result<(), SchemaError> {
        let mut queue = vec![index];
        while let Some(current) = queue.pop() {
            let info = self.docs[current].clone();
            if info.disabled {
                continue;
            }
            for child in child_elements(&info.doc, info.root) {
                let Some(name) = xsd_name(&info.doc, child) else {
                    continue;
                };
                let importing = name == "import";
                let chameleon = match name {
                    "include" | "redefine" | "override" => info.target_namespace.clone(),
                    "import" => None,
                    _ => continue,
                };
                let imported_namespace = attr(&info.doc, child, "namespace").map(str::to_owned);
                // §4.2.3 *src-import.1.2*: an `xs:import` that names no
                // namespace may only appear in a schema document that has a
                // `targetNamespace` of its own
                // (`msData/additional/test65472`).
                if importing && imported_namespace.is_none() && info.target_namespace.is_none() {
                    return Err(SchemaError::new(
                        "src-import.1.2",
                        "an `xs:import` without a `namespace` needs the importing document to                          have a `targetNamespace`",
                    )
                    .at(node_path(&info.doc, child)));
                }
                if importing && imported_namespace.as_deref() == info.target_namespace.as_deref() {
                    return Err(SchemaError::new(
                        "src-import.1.1",
                        "an `xs:import` may not name the importing document's own target                          namespace",
                    )
                    .at(node_path(&info.doc, child)));
                }
                let Some(location) = attr_token(&info.doc, child, "schemaLocation") else {
                    continue;
                };
                let base = if info.uri.starts_with('#') {
                    None
                } else {
                    Some(info.uri.as_str())
                };
                let Some(resolved) = self.resolver.resolve(&location, base)? else {
                    continue;
                };
                if !importing {
                    self.locations
                        .insert((current, child), resolved.uri.clone());
                }
                if let Some(added) =
                    self.push_document(resolved.uri, &resolved.text, chameleon, importing)?
                {
                    // §4.2.3 *src-import.3.1/3.2*: what an `xs:import` finds
                    // must carry exactly the `namespace` the import names —
                    // and no `targetNamespace` at all when the import names
                    // none.
                    if importing && self.docs[added].target_namespace != imported_namespace {
                        return Err(SchemaError::new(
                            "src-import.3.1",
                            format!(
                                "`{}` has target namespace {} but is imported for {}",
                                self.docs[added].uri,
                                describe_namespace(self.docs[added].target_namespace.as_deref()),
                                describe_namespace(imported_namespace.as_deref())
                            ),
                        ));
                    }
                    // §4.2.1 *src-include.2.1*: what an `xs:include`,
                    // `xs:redefine` or `xs:override` finds must have either the
                    // including document's target namespace or none at all —
                    // the *chameleon* case (`msData/schema/schC2`).
                    if !importing
                        && !self.docs[added].chameleon
                        && self.docs[added].target_namespace != info.target_namespace
                    {
                        return Err(SchemaError::new(
                            "src-include.2.1",
                            format!(
                                "`{}` has target namespace {}, which is not the including \
                                 document's {}",
                                self.docs[added].uri,
                                describe_namespace(self.docs[added].target_namespace.as_deref()),
                                describe_namespace(info.target_namespace.as_deref())
                            ),
                        ));
                    }
                    queue.push(added);
                }
            }
        }
        Ok(())
    }
}

/// The keys of a source map in identifier order — which, since identifiers are
/// handed out as globals are registered, is document order.
fn sorted_keys(sources: &HashMap<u32, Source>) -> Vec<u32> {
    let mut keys: Vec<u32> = sources.keys().copied().collect();
    keys.sort_unstable();
    keys
}

/// Whether the element tree rooted at `root` nests deeper than `limit`.
///
/// Walked with an explicit stack, so that measuring a hostile document cannot
/// itself do what the measurement is there to prevent, and stopped as soon as
/// the answer is known rather than walking the whole tree for a depth nobody
/// asked for.
fn nesting_exceeds(doc: &Document, root: NodeId, limit: usize) -> bool {
    let mut stack = vec![(root, 1usize)];
    while let Some((node, depth)) = stack.pop() {
        if depth > limit {
            return true;
        }
        stack.extend(
            doc.children(node)
                .filter(|child| doc.kind(*child) == Some(oxixml_dom::NodeKind::Element))
                .map(|child| (child, depth + 1)),
        );
    }
    false
}

/// The assembly driver: owns the growing [`Schema`] and every component's
/// source location.
pub(crate) struct Compiler {
    pub schema: Schema,
    pub docs: Vec<DocInfo>,
    pub type_source: HashMap<u32, Source>,
    pub type_state: Vec<BuildState>,
    pub element_source: HashMap<u32, Source>,
    pub element_state: HashMap<u32, BuildState>,
    pub attribute_source: HashMap<u32, Source>,
    pub attribute_state: HashMap<u32, BuildState>,
    pub group_source: HashMap<u32, Source>,
    pub group_state: Vec<BuildState>,
    pub attr_group_source: HashMap<u32, Source>,
    pub attr_group_state: Vec<BuildState>,
    pub notation_source: HashMap<u32, Source>,
    pub keyref_pending: Vec<(crate::components::IdcId, ExpandedName)>,
    pub hidden_counter: usize,
    /// The `xs:redefine` children that contained no reference to themselves,
    /// as `(kind, redefinition, original)`. §4.2.2 *src-redefine.6.2.2* and
    /// *7.2.2* make each of them a restriction of the component it replaces,
    /// which can only be checked once both have been built.
    pub redefine_restrictions: Vec<(RedefineKind, ExpandedName, ExpandedName)>,
    /// See [`SchemaSet::locations`].
    pub locations: HashMap<(usize, NodeId), String>,
    /// `<xs:key ref="…"/>` and its siblings, as `(element, name, location)`.
    /// The referent may be written after the reference, so they are resolved
    /// once every declaration is built.
    pub idc_ref_pending: Vec<(
        ElementId,
        ExpandedName,
        crate::components::IdcCategory,
        String,
    )>,
    /// How many named definitions are being built inside one another right
    /// now — the length of the chain of `ref`s and `base`s that reached the
    /// one currently in hand.
    ///
    /// A *cycle* is already caught by [`BuildState::Building`], but a chain is
    /// perfectly legal and as long as the document cares to make it: a group
    /// referring to a group referring to a group is built by recursion, one
    /// turn per link, so a document holding tens of thousands of links would
    /// exhaust the stack. The chain is therefore bounded like everything else
    /// that reads a schema document, at [`SchemaSet::MAX_DEFINITION_CHAIN`].
    pub chain_depth: usize,
    /// The components an `xs:override` has already replaced, as
    /// `(element type, expanded name)`.
    ///
    /// §F.2 clause 4 lets an `xs:override` reach *through* an `xs:override` of
    /// the document it names, and clause 4.1 says the outer one wins: "if a
    /// child of `E2` and a child of `O1` match …, then `E2′` has a child which
    /// is a copy of the child of **`O1`**". Documents are visited overriding
    /// document first, so the first replacement seen is the outer one and any
    /// later one for the same component is discarded
    /// (`saxonData/Override/over009`, `over026.bad`).
    pub overridden: HashSet<(String, ExpandedName)>,
    /// Direct references between attribute groups, as `(from, to)` pairs,
    /// recorded while each group's definition is built. XSD 1.1 §3.6.6 permits
    /// an attribute group to refer to itself directly or through a cycle; its
    /// {attribute uses} are then the least fixed point over these edges, which
    /// [`Compiler::close_attribute_group_cycles`] computes once every group has
    /// been built, so a group reached only through the far side of a cycle still
    /// contributes its attributes.
    pub attr_group_edges: Vec<(AttrGroupId, AttrGroupId)>,
    /// The attribute group whose definition is being built right now, so a
    /// reference it makes can be recorded in [`Compiler::attr_group_edges`].
    pub current_attr_group: Option<AttrGroupId>,
    /// Anonymous type definitions written inside an element declaration whose
    /// construction has been put off until every named component is built.
    ///
    /// An element declaration is a component in its own right: the model group
    /// or type definition it is written inside reaches it, but what its *type*
    /// is written in terms of belongs to the element, not to them. §3.8.6
    /// *Model Group Correct* clause 2 forbids a group that contains itself
    /// "within the {particles} of the group", and §3.4.6 *ct-props-correct*
    /// clause 3 forbids a type reachable from itself by {base type
    /// definition}s — neither of which a reference made from inside an element
    /// declaration's own type takes part in. Building that type eagerly would
    /// walk straight back into the definition still in hand and report a cycle
    /// that is not there (`msData/additional/addB077`,
    /// `msData/particles/particlesZ010`, `particlesEb041`). Naming the type
    /// instead of writing it inline was always allowed for the same reason, so
    /// deferring gives the inline spelling the behaviour the named one has.
    pub pending_element_types: Vec<TypeId>,
}

impl Compiler {
    fn run(&mut self) -> Result<(), SchemaError> {
        self.register_globals()?;
        self.register_redefinitions()?;

        // The sources are held in hash maps, whose iteration order varies from
        // run to run. Components are built in identifier order — which is the
        // order they were registered in, and so document order — so that a
        // schema document with more than one thing wrong with it reports the
        // *same* one every time. Whether assembly succeeds does not depend on
        // the order; which of several errors it reports otherwise would.
        // Attribute groups are built — and their XSD 1.1 reference cycles closed
        // to a fixed point — before the complex types that reference them, so a
        // type inlines a group's *complete* {attribute uses} rather than the
        // partial set a group entered second in a cycle would otherwise hold.
        // An attribute group depends only on simple types (an attribute cannot
        // carry a complex type) and other attribute groups, so nothing it needs
        // is left unbuilt by taking it first.
        for raw in sorted_keys(&self.attr_group_source) {
            self.ensure_attribute_group(AttrGroupId(raw))?;
        }
        self.close_attribute_group_cycles();
        for raw in sorted_keys(&self.type_source) {
            self.ensure_type(TypeId(raw))?;
        }
        for raw in sorted_keys(&self.group_source) {
            self.ensure_group(GroupId(raw))?;
        }
        for raw in sorted_keys(&self.attribute_source) {
            self.build_global_attribute(AttributeId(raw))?;
        }
        for raw in sorted_keys(&self.element_source) {
            self.build_global_element(ElementId(raw))?;
        }
        for raw in sorted_keys(&self.notation_source) {
            self.build_notation(NotationId(raw))?;
        }
        self.build_deferred_element_types()?;

        self.resolve_keyrefs()?;
        self.resolve_value_constraints()?;
        self.compute_substitution_groups()?;
        self.resolve_idc_references()?;
        self.check_redefine_restrictions()?;
        self.check_all()?;
        Ok(())
    }

    /// The document a source belongs to.
    pub(crate) fn info(&self, index: usize) -> DocInfo {
        self.docs[index].clone()
    }

    /// Enters one more link of a chain of named definitions, refusing to go
    /// past [`SchemaSet::MAX_DEFINITION_CHAIN`] of them.
    ///
    /// `what` names the kind of definition, for the message.
    pub(crate) fn enter_definition(&mut self, what: &str) -> Result<(), SchemaError> {
        if self.chain_depth >= SchemaSet::MAX_DEFINITION_CHAIN {
            return Err(SchemaError::new(
                "src-max-depth",
                format!(
                    "the chain of definitions reaching this {what} is longer than the maximum of \
                     {} links",
                    SchemaSet::MAX_DEFINITION_CHAIN
                ),
            ));
        }
        self.chain_depth += 1;
        Ok(())
    }

    /// Leaves the link [`Compiler::enter_definition`] entered.
    ///
    /// Called on the way out whether the definition was built or not, so that
    /// one failure cannot leave the count permanently raised.
    pub(crate) fn leave_definition(&mut self) {
        self.chain_depth = self.chain_depth.saturating_sub(1);
    }

    /// Resolves a `QName` reference written at `node` in document `index`,
    /// honouring both chameleon inclusion and a `redefine` self-reference, and
    /// refusing a namespace the referring document was never licensed to name.
    ///
    /// # Errors
    ///
    /// Reports `src-resolve` when the `QName` does not resolve at all, and
    /// when it resolves into a namespace the document neither declares as its
    /// target namespace nor imports ([`licensing`]).
    pub(crate) fn reference(
        &self,
        info: &DocInfo,
        node: NodeId,
        value: &str,
        redefine: Option<&Redefine>,
        kind: RedefineKind,
    ) -> Result<ExpandedName, SchemaError> {
        let name = self.unchecked_reference(info, node, value, redefine, kind)?;
        // A `redefine` self-reference is remapped to a hidden name in the very
        // same namespace, so checking the licence after the remapping asks the
        // same question as checking it before.
        if !info.licence.allows(name.namespace()) {
            return Err(SchemaError::new(
                "src-resolve",
                licensing::message(name.namespace(), &info.uri),
            )
            .at(node_path(&info.doc, node)));
        }
        Ok(name)
    }

    /// Resolves a `QName` reference **without** the licence check.
    ///
    /// Two places only compare a resolved name against another rather than
    /// looking a component up, and both belong to
    /// `check_redefinition_shape`: the search for the self-reference that
    /// makes a redefinition well-formed. A
    /// name written there that no document may name is not a use of a foreign
    /// component; it simply is not the self-reference being looked for, and
    /// reporting it as an unlicensed reference would replace the
    /// `src-redefine` diagnosis the reader wants with an unrelated one.
    pub(crate) fn unchecked_reference(
        &self,
        info: &DocInfo,
        node: NodeId,
        value: &str,
        redefine: Option<&Redefine>,
        kind: RedefineKind,
    ) -> Result<ExpandedName, SchemaError> {
        let mut name = resolve_qname(&info.doc, node, value)?;
        if name.namespace().is_none()
            && info.chameleon
            && let Some(target) = &info.target_namespace
        {
            name = ExpandedName::new(Some(target.as_str()), name.local_name());
        }
        if let Some(redefine) = redefine
            && redefine.kind == kind
            && redefine.real == name
        {
            return Ok(redefine.hidden.clone());
        }
        Ok(name)
    }

    fn register_globals(&mut self) -> Result<(), SchemaError> {
        for index in 0..self.docs.len() {
            let info = self.info(index);
            if info.disabled {
                continue;
            }
            for child in child_elements(&info.doc, info.root) {
                let Some(name) = xsd_name(&info.doc, child) else {
                    continue;
                };
                match name {
                    "simpleType" | "complexType" => {
                        self.register_type(&info, index, child, None)?;
                    }
                    "element" => self.register_element(&info, index, child)?,
                    "attribute" => self.register_attribute(&info, index, child)?,
                    "group" => {
                        self.register_group(&info, index, child, None)?;
                    }
                    "attributeGroup" => {
                        self.register_attribute_group(&info, index, child, None)?;
                    }
                    "notation" => self.register_notation(&info, index, child)?,
                    "annotation" | "include" | "import" | "redefine" | "override"
                    | "defaultOpenContent" => {}
                    other => {
                        return Err(SchemaError::new(
                            "src-schema",
                            format!("`xs:{other}` may not appear at the top level"),
                        )
                        .at(node_path(&info.doc, child)));
                    }
                }
            }
        }
        Ok(())
    }

    fn global_name(
        &self,
        info: &DocInfo,
        doc: &Document,
        node: NodeId,
    ) -> Result<ExpandedName, SchemaError> {
        let local = attr_token(doc, node, "name").ok_or_else(|| {
            SchemaError::new("src-element", "a top-level definition needs a `name`")
                .at(node_path(doc, node))
        })?;
        if !crate::value::is_xml_ncname(&local) {
            return Err(
                SchemaError::new("src-element", format!("`{local}` is not an NCName"))
                    .at(node_path(doc, node)),
            );
        }
        Ok(ExpandedName::new(info.target_namespace.as_deref(), local))
    }

    fn register_type(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
        redefine: Option<Redefine>,
    ) -> Result<TypeId, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        if redefine.is_none() && self.schema.global_types.contains_key(&name) {
            return Err(SchemaError::new(
                "sch-props-correct.2",
                format!("`{name}` is defined more than once"),
            )
            .at(node_path(&doc, node)));
        }
        let id = self.push_type_stub(Some(name.clone()));
        self.schema.global_types.insert(name, id);
        self.type_source.insert(
            id.0,
            Source {
                doc: index,
                node,
                redefine,
            },
        );
        self.type_state[id.index()] = BuildState::NotBuilt;
        Ok(id)
    }

    fn register_element(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
    ) -> Result<(), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        if self.schema.global_elements.contains_key(&name) {
            return Err(SchemaError::new(
                "sch-props-correct.2",
                format!("the element `{name}` is declared more than once"),
            )
            .at(node_path(&doc, node)));
        }
        let id = ElementId(self.next_index(self.schema.elements.len())?);
        self.schema.elements.push(ElementDeclaration {
            name: name.clone(),
            type_id: self.schema.any_type,
            global: true,
            nillable: false,
            abstract_: false,
            value_constraint: None,
            identity_constraints: Vec::new(),
            substitution_group: Vec::new(),
            block: DerivationSet::empty(),
            final_: DerivationSet::empty(),
            substitutable_by: Vec::new(),
            alternatives: Vec::new(),
        });
        self.schema.global_elements.insert(name, id);
        self.element_source.insert(
            id.0,
            Source {
                doc: index,
                node,
                redefine: None,
            },
        );
        Ok(())
    }

    fn register_attribute(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
    ) -> Result<(), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        // §3.2.6 *no-xsi*: the XSI namespace is reserved for the four
        // attributes the Recommendation itself defines, so a schema document
        // may not declare an attribute in it
        // (`saxonData/Wild/wild041.bad`).
        if name.namespace() == Some(crate::names::XSI_NS) {
            return Err(SchemaError::new(
                "no-xsi",
                format!("`{name}` is in the XSI namespace, which is reserved"),
            )
            .at(node_path(&doc, node)));
        }
        if let Some(existing) = self.schema.global_attributes.get(&name).copied() {
            // A schema may re-declare the attributes of the XML namespace that
            // this crate provides out of the box; its declaration wins.
            if self.attribute_source.contains_key(&existing.0) {
                return Err(SchemaError::new(
                    "sch-props-correct.2",
                    format!("the attribute `{name}` is declared more than once"),
                )
                .at(node_path(&doc, node)));
            }
        }
        let id = AttributeId(self.next_index(self.schema.attributes.len())?);
        self.schema.attributes.push(AttributeDeclaration {
            name: name.clone(),
            type_id: self.schema.any_simple_type,
            global: true,
            value_constraint: None,
            inheritable: false,
        });
        self.schema.global_attributes.insert(name, id);
        self.attribute_source.insert(
            id.0,
            Source {
                doc: index,
                node,
                redefine: None,
            },
        );
        Ok(())
    }

    fn register_group(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
        redefine: Option<Redefine>,
    ) -> Result<GroupId, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        if redefine.is_none() && self.schema.global_groups.contains_key(&name) {
            return Err(SchemaError::new(
                "sch-props-correct.2",
                format!("the group `{name}` is defined more than once"),
            )
            .at(node_path(&doc, node)));
        }
        let id = GroupId(self.next_index(self.schema.groups.len())?);
        self.schema.groups.push(ModelGroupDefinition {
            name: name.clone(),
            model_group: None,
        });
        self.group_state.push(BuildState::NotBuilt);
        self.schema.global_groups.insert(name, id);
        self.group_source.insert(
            id.0,
            Source {
                doc: index,
                node,
                redefine,
            },
        );
        Ok(id)
    }

    fn register_attribute_group(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
        redefine: Option<Redefine>,
    ) -> Result<AttrGroupId, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        if redefine.is_none()
            && let Some(existing) = self.schema.global_attribute_groups.get(&name).copied()
        {
            // A schema may supply its own definitions for the XML namespace,
            // whose components — `xml:lang`, `xml:space`, `xml:base`, `xml:id`
            // and the `xml:specialAttrs` group — this crate provides out of the
            // box so that a schema referring to them need not import anything.
            // Those built-ins have no source document, and a real declaration
            // replaces them rather than colliding with them
            // (`msData/additional/addB187`, `addB188`, where the user's schema
            // for the XML namespace adds `xml:blah` to `xml:specialAttrs`).
            // `register_attribute` says the same thing about the attributes.
            if self.attr_group_source.contains_key(&existing.0) {
                return Err(SchemaError::new(
                    "sch-props-correct.2",
                    format!("the attribute group `{name}` is defined more than once"),
                )
                .at(node_path(&doc, node)));
            }
        }
        let id = AttrGroupId(self.next_index(self.schema.attribute_groups.len())?);
        self.schema.attribute_groups.push(AttributeGroupDefinition {
            name: Some(name.clone()),
            ..AttributeGroupDefinition::default()
        });
        self.attr_group_state.push(BuildState::NotBuilt);
        self.schema.global_attribute_groups.insert(name, id);
        self.attr_group_source.insert(
            id.0,
            Source {
                doc: index,
                node,
                redefine,
            },
        );
        Ok(id)
    }

    fn register_notation(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
    ) -> Result<(), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        if self.schema.global_notations.contains_key(&name) {
            return Err(SchemaError::new(
                "sch-props-correct.2",
                format!("the notation `{name}` is declared more than once"),
            )
            .at(node_path(&doc, node)));
        }
        let id = NotationId(self.next_index(self.schema.notations.len())?);
        self.schema.notations.push(NotationDeclaration {
            name: name.clone(),
            public_id: None,
            system_id: None,
        });
        self.schema.global_notations.insert(name, id);
        self.notation_source.insert(
            id.0,
            Source {
                doc: index,
                node,
                redefine: None,
            },
        );
        Ok(())
    }

    /// The order [`register_redefinitions`](Self::register_redefinitions)
    /// visits the documents in.
    ///
    /// §4.2.2 gives an `xs:redefine` the components of *the document its
    /// `schemaLocation` names*, so when redefinitions are chained — a document
    /// redefining a document that itself redefines a third — each level must
    /// see the level below it already redefined. Load order is the opposite of
    /// that: the outermost document is read first and the document it names
    /// second, so processing in load order would give the outer redefinition
    /// the *original* component and the inner one the outer's replacement, and
    /// the chain would come out inside-out
    /// (`msData/simpleType/stZ033`, `stZ034`, where three documents redefine
    /// one enumeration in turn).
    ///
    /// The visit order is therefore a topological sort over the `xs:redefine`
    /// edges, so that a redefined document is always visited before the
    /// document redefining it, with ties broken by load order so that
    /// everything else is visited exactly as before. A cycle among redefines
    /// leaves its documents unordered, and they are appended in load order
    /// rather than dropped.
    ///
    /// `xs:override` is deliberately not an edge here: §F.2 clause 4.1 wants
    /// the *outer* override applied first, which is what load order already
    /// gives.
    fn redefinition_order(&self) -> Vec<usize> {
        let count = self.docs.len();
        let mut by_uri: HashMap<&str, usize> = HashMap::new();
        for (index, info) in self.docs.iter().enumerate() {
            by_uri.entry(info.uri.as_str()).or_insert(index);
        }
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut waiting = vec![0_usize; count];
        for ((source, node), uri) in &self.locations {
            let Some(&target) = by_uri.get(uri.as_str()) else {
                continue;
            };
            if target == *source || *source >= count {
                continue;
            }
            let info = &self.docs[*source];
            if xsd_name(&info.doc, *node) != Some("redefine") {
                continue;
            }
            successors[target].push(*source);
            waiting[*source] += 1;
        }
        let mut ready: Vec<usize> = (0..count).filter(|index| waiting[*index] == 0).collect();
        let mut order = Vec::with_capacity(count);
        while !ready.is_empty() {
            ready.sort_unstable();
            let next = ready.remove(0);
            order.push(next);
            for successor in successors[next].clone() {
                waiting[successor] -= 1;
                if waiting[successor] == 0 {
                    ready.push(successor);
                }
            }
        }
        // Whatever a cycle left behind, in load order.
        let leftover: Vec<usize> = (0..count).filter(|index| !order.contains(index)).collect();
        order.extend(leftover);
        order
    }

    /// Processes `xs:redefine` (and, in 1.1, `xs:override`) once every ordinary
    /// global name is registered: the original component is re-registered under
    /// a hidden name so that the redefinition can refer to it as its base.
    fn register_redefinitions(&mut self) -> Result<(), SchemaError> {
        for index in self.redefinition_order() {
            let info = self.info(index);
            if info.disabled {
                continue;
            }
            // §4.2.5 *src-override.1*: one schema document may not override the
            // same component twice — neither twice inside one `xs:override`
            // nor from two of them — because the result would depend on the
            // order they were read in (`saxonData/Override/over021.bad`,
            // `over022.bad`).
            let mut overridden: HashSet<(String, String)> = HashSet::new();
            for child in child_elements(&info.doc, info.root) {
                let overriding = match xsd_name(&info.doc, child) {
                    Some("redefine") => false,
                    Some("override") => true,
                    _ => continue,
                };
                let mut redefined: Vec<(&str, ExpandedName)> = Vec::new();
                let mut pending_restrictions: Vec<(RedefineKind, ExpandedName)> = Vec::new();
                // §4.2.2 *src-redefine.1*: what a redefinition redefines must
                // be a component of the document the `schemaLocation` points
                // at — or of one that document includes — and not merely a
                // component of the same namespace
                // (`msData/schema/schS1`).
                let provenance = self.redefine_provenance(index, child);
                for member in child_elements(&info.doc, child) {
                    let Some(kind_name) = xsd_name(&info.doc, member) else {
                        continue;
                    };
                    if overriding
                        && kind_name != "annotation"
                        && let Some(local) = attr(&info.doc, member, "name")
                        && !overridden.insert((kind_name.to_owned(), local.to_owned()))
                    {
                        return Err(SchemaError::new(
                            "src-override.1",
                            format!("`{local}` is overridden more than once"),
                        )
                        .at(node_path(&info.doc, member)));
                    }
                    if !overriding
                        && matches!(
                            kind_name,
                            "simpleType" | "complexType" | "group" | "attributeGroup"
                        )
                    {
                        let name = self.global_name(&info, &Rc::clone(&info.doc), member)?;
                        let family = if kind_name.ends_with("Type") {
                            "type"
                        } else {
                            kind_name
                        };
                        // §4.2.2 *src-redefine.2*: one `xs:redefine` may hold
                        // at most one redefinition of any one component
                        // (`msData/schema/schT2`).
                        if redefined
                            .iter()
                            .any(|(seen, seen_name)| *seen == family && *seen_name == name)
                        {
                            return Err(SchemaError::new(
                                "src-redefine.2",
                                format!("`{name}` is redefined twice in one `xs:redefine`"),
                            )
                            .at(node_path(&info.doc, member)));
                        }
                        redefined.push((family, name.clone()));
                        if let Some(provenance) = &provenance
                            && !provenance
                                .iter()
                                .any(|(seen, seen_name)| seen == family && *seen_name == name)
                        {
                            return Err(SchemaError::new(
                                "src-redefine.1",
                                format!(
                                    "`{name}` is redefined, but the document the `xs:redefine` \
                                     names does not define it"
                                ),
                            )
                            .at(node_path(&info.doc, member)));
                        }
                        if !self.check_redefinition_shape(&info, member, kind_name)? {
                            let kind = if kind_name == "group" {
                                RedefineKind::Group
                            } else {
                                RedefineKind::AttributeGroup
                            };
                            pending_restrictions.push((kind, name));
                        }
                    }
                    match kind_name {
                        "simpleType" | "complexType" => {
                            self.redefine_component(
                                &info,
                                index,
                                member,
                                RedefineKind::Type,
                                overriding,
                            )?;
                        }
                        "group" => {
                            let hidden = self.redefine_component(
                                &info,
                                index,
                                member,
                                RedefineKind::Group,
                                overriding,
                            )?;
                            record_pending(
                                &mut self.redefine_restrictions,
                                &pending_restrictions,
                                RedefineKind::Group,
                                hidden,
                            );
                        }
                        "attributeGroup" => {
                            let hidden = self.redefine_component(
                                &info,
                                index,
                                member,
                                RedefineKind::AttributeGroup,
                                overriding,
                            )?;
                            record_pending(
                                &mut self.redefine_restrictions,
                                &pending_restrictions,
                                RedefineKind::AttributeGroup,
                                hidden,
                            );
                        }
                        "annotation" => {}
                        "element" | "attribute" | "notation" if overriding => {
                            // 1.1 `override` may replace these outright; the
                            // original was registered already, so replace its
                            // source and rebuild it from here.
                            self.override_declaration(&info, index, member, kind_name)?;
                        }
                        other => {
                            return Err(SchemaError::new(
                                "src-redefine",
                                format!("`xs:{other}` may not appear inside `xs:redefine`"),
                            )
                            .at(node_path(&info.doc, member)));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The top-level `(family, name)` pairs the document an `xs:redefine`
    /// points at defines, transitively through its own `xs:include`s.
    ///
    /// Returns [`None`] when the document cannot be identified — an
    /// `xs:redefine` with no `schemaLocation`, or one whose resolver returned
    /// nothing — in which case *src-redefine.1* is not checked rather than
    /// guessed at.
    fn redefine_provenance(
        &self,
        index: usize,
        node: NodeId,
    ) -> Option<Vec<(String, ExpandedName)>> {
        let start = self.document_at(index, node)?;
        let mut names: Vec<(String, ExpandedName)> = Vec::new();
        let mut seen = vec![start];
        let mut queue = vec![start];
        while let Some(current) = queue.pop() {
            let doc = &self.docs[current];
            for child in child_elements(&doc.doc, doc.root) {
                match xsd_name(&doc.doc, child) {
                    Some(kind @ ("simpleType" | "complexType" | "group" | "attributeGroup")) => {
                        let Some(local) = attr_token(&doc.doc, child, "name") else {
                            continue;
                        };
                        let family = if kind.ends_with("Type") { "type" } else { kind };
                        names.push((
                            family.to_owned(),
                            ExpandedName::new(doc.target_namespace.as_deref(), local),
                        ));
                    }
                    // An `xs:include` brings the included document's names in,
                    // and a nested `xs:redefine` both includes and defines.
                    Some("include" | "redefine" | "override") => {
                        if let Some(next) = self.document_at(current, child)
                            && !seen.contains(&next)
                        {
                            seen.push(next);
                            queue.push(next);
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(names)
    }

    /// The document index the `xs:include`/`xs:redefine`/`xs:override` written
    /// at `node` of document `index` resolved to.
    fn document_at(&self, index: usize, node: NodeId) -> Option<usize> {
        let uri = self.locations.get(&(index, node))?;
        self.docs.iter().position(|doc| doc.uri == *uri)
    }

    /// The shape rules a redefinition must have, from §4.2.2.
    ///
    /// * *src-redefine.5*: a `<simpleType>` or `<complexType>` inside an
    ///   `xs:redefine` must derive from *itself* — its `<restriction>` or
    ///   `<extension>` must name the very component being redefined
    ///   (`msData/schema/schJ2`, `msData/schema/schP3`).
    /// * *src-redefine.6.1.1* and *6.1.2*: a `<group>` may contain at most one
    ///   reference to itself, and that reference must have
    ///   `minOccurs = maxOccurs = 1` (`msData/schema/schR3`).
    /// * *src-redefine.7.1*: an `<attributeGroup>` may likewise contain at
    ///   most one reference to itself.
    fn check_redefinition_shape(
        &self,
        info: &DocInfo,
        node: NodeId,
        kind_name: &str,
    ) -> Result<bool, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        match kind_name {
            "simpleType" | "complexType" => {
                let derivation = find_self_derivation(&doc, node);
                let Some(derivation) = derivation else {
                    return Err(SchemaError::new(
                        "src-redefine.5",
                        format!("the redefinition of `{name}` does not derive from anything"),
                    )
                    .at(node_path(&doc, node)));
                };
                let base = attr(&doc, derivation, "base").unwrap_or_default();
                let resolved =
                    self.unchecked_reference(info, derivation, base, None, RedefineKind::Type)?;
                if resolved != name {
                    return Err(SchemaError::new(
                        "src-redefine.5",
                        format!(
                            "the redefinition of `{name}` derives from `{resolved}` rather than \
                             from `{name}` itself"
                        ),
                    )
                    .at(node_path(&doc, derivation)));
                }
                return Ok(true);
            }
            "group" | "attributeGroup" => {
                let kind = if kind_name == "group" {
                    RedefineKind::Group
                } else {
                    RedefineKind::AttributeGroup
                };
                let mut count = 0;
                for reference in self_references(&doc, node, kind_name) {
                    let Some(text) = attr(&doc, reference, "ref") else {
                        continue;
                    };
                    if self.unchecked_reference(info, reference, text, None, kind)? != name {
                        continue;
                    }
                    count += 1;
                    if count > 1 {
                        return Err(SchemaError::new(
                            "src-redefine.6.1.1",
                            format!("the redefinition of `{name}` refers to itself twice"),
                        )
                        .at(node_path(&doc, reference)));
                    }
                    if kind == RedefineKind::Group
                        && (attr_token(&doc, reference, "minOccurs")
                            .as_deref()
                            .unwrap_or("1")
                            != "1"
                            || attr_token(&doc, reference, "maxOccurs")
                                .as_deref()
                                .unwrap_or("1")
                                != "1")
                    {
                        return Err(SchemaError::new(
                            "src-redefine.6.1.2",
                            format!(
                                "the self-reference in the redefinition of `{name}` must occur \
                                 exactly once"
                            ),
                        )
                        .at(node_path(&doc, reference)));
                    }
                }
                return Ok(count > 0);
            }
            _ => {}
        }
        Ok(true)
    }

    fn redefine_component(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
        kind: RedefineKind,
        overriding: bool,
    ) -> Result<Option<(ExpandedName, ExpandedName)>, SchemaError> {
        let doc = Rc::clone(&info.doc);
        let real = self.global_name(info, &doc, node)?;
        // §F.2 clause 4.1: an outer `xs:override` wins over one nested in the
        // document it names, and the outer document is visited first.
        if overriding && !self.overridden.insert((format!("{kind:?}"), real.clone())) {
            return Ok(None);
        }
        self.hidden_counter += 1;
        let hidden = ExpandedName::new(
            real.namespace(),
            format!("{}#redefined{}", real.local_name(), self.hidden_counter),
        );
        let redefine = if overriding {
            None
        } else {
            Some(Redefine {
                kind,
                real: real.clone(),
                hidden: hidden.clone(),
            })
        };
        // §F.2 (*Transformation for xs:override*) replaces an element of the
        // overridden document only when `O1` has "a child … with the same
        // element type and the same value for its `name` attribute"; a child
        // of the `xs:override` that matches nothing is copied nowhere, so it
        // simply has no effect. That is what `saxonData/Override/over007`'s
        // own comment says of its second group: "the following declaration
        // overrides nothing, so it is ignored". `xs:redefine` is the
        // opposite — §4.2.2 *src-redefine.1* requires the redefined component
        // to exist.
        match kind {
            RedefineKind::Type => {
                let Some(original) = self.schema.global_types.remove(&real) else {
                    if overriding {
                        return Ok(None);
                    }
                    return Err(SchemaError::new(
                        "src-redefine.1",
                        format!("`{real}` is redefined but never defined"),
                    )
                    .at(node_path(&doc, node)));
                };
                if !overriding {
                    self.schema.global_types.insert(hidden.clone(), original);
                }
                self.register_type(info, index, node, redefine)?;
            }
            RedefineKind::Group => {
                let Some(original) = self.schema.global_groups.remove(&real) else {
                    if overriding {
                        return Ok(None);
                    }
                    return Err(SchemaError::new(
                        "src-redefine.1",
                        format!("the group `{real}` is redefined but never defined"),
                    )
                    .at(node_path(&doc, node)));
                };
                if !overriding {
                    self.schema.global_groups.insert(hidden.clone(), original);
                }
                self.register_group(info, index, node, redefine)?;
            }
            RedefineKind::AttributeGroup => {
                let Some(original) = self.schema.global_attribute_groups.remove(&real) else {
                    if overriding {
                        return Ok(None);
                    }
                    return Err(SchemaError::new(
                        "src-redefine.1",
                        format!("the attribute group `{real}` is redefined but never defined"),
                    )
                    .at(node_path(&doc, node)));
                };
                if !overriding {
                    self.schema
                        .global_attribute_groups
                        .insert(hidden.clone(), original);
                }
                self.register_attribute_group(info, index, node, redefine)?;
            }
        }
        Ok(if overriding {
            None
        } else {
            Some((real, hidden))
        })
    }

    fn override_declaration(
        &mut self,
        info: &DocInfo,
        index: usize,
        node: NodeId,
        kind: &str,
    ) -> Result<(), SchemaError> {
        let doc = Rc::clone(&info.doc);
        let name = self.global_name(info, &doc, node)?;
        // §F.2 clause 4.1, as in `redefine_component`: the first override of a
        // component is the outermost one, and it wins.
        if !self.overridden.insert((kind.to_owned(), name.clone())) {
            return Ok(());
        }
        let source = Source {
            doc: index,
            node,
            redefine: None,
        };
        match kind {
            "element" => {
                if let Some(id) = self.schema.global_elements.get(&name).copied() {
                    self.element_source.insert(id.0, source);
                } else {
                    self.register_element(info, index, node)?;
                }
            }
            "attribute" => {
                if let Some(id) = self.schema.global_attributes.get(&name).copied() {
                    self.attribute_source.insert(id.0, source);
                } else {
                    self.register_attribute(info, index, node)?;
                }
            }
            _ => {
                if let Some(id) = self.schema.global_notations.get(&name).copied() {
                    self.notation_source.insert(id.0, source);
                } else {
                    self.register_notation(info, index, node)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn push_type_stub(&mut self, name: Option<ExpandedName>) -> TypeId {
        let id = TypeId(
            self.next_index(self.schema.types.len())
                .unwrap_or(TypeId(u32::MAX).0),
        );
        self.schema.types.push(TypeDefinition {
            name,
            base: Some(self.schema.any_type),
            derivation: crate::components::Derivation::Restriction,
            final_: DerivationSet::empty(),
            block: DerivationSet::empty(),
            kind: TypeKind::Simple(Box::new(SimpleTypeInfo {
                variety: Variety::Absent,
                primitive: None,
                facets: Facets::default(),
            })),
        });
        self.type_state.push(BuildState::Built);
        id
    }

    /// Reserves an identifier for the anonymous type written at `node` inside
    /// an element declaration, and puts its construction off until every named
    /// component is built — see [`Compiler::pending_element_types`].
    ///
    /// The identifier is usable straight away: the element declaration keeps
    /// it as its {type definition}, and everything that looks the type up does
    /// so once assembly is over.
    pub(crate) fn defer_element_type(
        &mut self,
        info: &DocInfo,
        node: NodeId,
        redefine: Option<&Redefine>,
    ) -> TypeId {
        let id = self.push_type_stub(None);
        self.type_state[id.index()] = BuildState::NotBuilt;
        self.type_source.insert(
            id.0,
            Source {
                doc: info.index,
                node,
                redefine: redefine.cloned(),
            },
        );
        self.pending_element_types.push(id);
        id
    }

    /// Builds every type [`Compiler::defer_element_type`] put off, including
    /// the ones deferred while building them.
    fn build_deferred_element_types(&mut self) -> Result<(), SchemaError> {
        while !self.pending_element_types.is_empty() {
            for id in std::mem::take(&mut self.pending_element_types) {
                self.ensure_type(id)?;
            }
        }
        Ok(())
    }

    fn next_index(&self, length: usize) -> Result<u32, SchemaError> {
        u32::try_from(length)
            .map_err(|_| SchemaError::new("src-schema", "the schema has too many components"))
    }

    fn build_notation(&mut self, id: NotationId) -> Result<(), SchemaError> {
        let Some(source) = self.notation_source.get(&id.0).cloned() else {
            return Ok(());
        };
        let info = self.info(source.doc);
        let public_id = attr(&info.doc, source.node, "public").map(str::to_owned);
        let system_id = attr(&info.doc, source.node, "system").map(str::to_owned);
        let notation = &mut self.schema.notations[id.index()];
        notation.public_id = public_id;
        notation.system_id = system_id;
        Ok(())
    }

    /// Attaches every `<xs:key ref="…"/>` to its element declaration.
    fn resolve_idc_references(&mut self) -> Result<(), SchemaError> {
        let pending = std::mem::take(&mut self.idc_ref_pending);
        for (element, name, category, location) in pending {
            let target = self
                .schema
                .global_identity_constraints
                .get(&name)
                .copied()
                .ok_or_else(|| {
                    SchemaError::new(
                        "src-resolve",
                        format!("the identity constraint `{name}` is not defined"),
                    )
                    .at(location.clone())
                })?;
            // §3.11.3 clause 5: "If `ref` is present, then the
            // {identity-constraint category} of the identity-constraint
            // definition ·resolved· to by the ·actual value· of the `ref`
            // [attribute] **matches the name of the element information
            // item**" — an `xs:key ref="…"` may not point at an `xs:unique`
            // (`saxonData/Id/id041`, `ibmData/schema_invalid/S2_2_4/
            // s2_2_4si02`).
            let target_category = self.schema.identity_constraints[target.index()].category;
            if target_category != category {
                return Err(SchemaError::new(
                    "src-identity-constraint.5",
                    format!("`{name}` is not the kind of identity constraint this reference names"),
                )
                .at(location));
            }
            let constraints = &mut self.schema.elements[element.index()].identity_constraints;
            if !constraints.contains(&target) {
                constraints.push(target);
            }
        }
        Ok(())
    }

    fn resolve_keyrefs(&mut self) -> Result<(), SchemaError> {
        let pending = std::mem::take(&mut self.keyref_pending);
        for (id, name) in pending {
            let target = self
                .schema
                .global_identity_constraints
                .get(&name)
                .copied()
                .ok_or_else(|| {
                    SchemaError::new(
                        "src-resolve",
                        format!("the key `{name}` referred to by a `keyref` is not defined"),
                    )
                })?;
            let target_category = self.schema.identity_constraints[target.index()].category;
            if target_category == crate::components::IdcCategory::KeyRef {
                return Err(SchemaError::new(
                    "c-props-correct.1",
                    format!("`{name}` is a `keyref`, so a `keyref` may not refer to it"),
                ));
            }
            let expected = self.schema.identity_constraints[target.index()]
                .fields
                .len();
            let actual = self.schema.identity_constraints[id.index()].fields.len();
            if expected != actual {
                return Err(SchemaError::new(
                    "c-props-correct.2",
                    format!("the `keyref` has {actual} fields but `{name}` has {expected}"),
                ));
            }
            self.schema.identity_constraints[id.index()].refer = Some(target);
        }
        Ok(())
    }

    /// *e-props-correct.4* and *a-props-correct.3* of XML Schema 1.0 §3.3.6
    /// and §3.2.6: a declaration whose type is or is derived from `xs:ID` may
    /// carry no value constraint, because a DTD `ID` attribute may have no
    /// default. XML Schema 1.1 dropped both clauses, which is why the check is
    /// version-dependent — `sunData/ElemDecl/valueConstraint/valueConstraint01001m`
    /// is invalid under 1.0 and valid under 1.1.
    fn is_id_typed(&self, type_id: TypeId) -> bool {
        !self.schema.version().is_1_1()
            && self
                .schema
                .builtin("ID")
                .is_some_and(|id| self.schema.is_derived_from(type_id, id))
    }

    fn resolve_value_constraints(&mut self) -> Result<(), SchemaError> {
        for index in 0..self.schema.elements.len() {
            if self.schema.elements[index].value_constraint.is_some()
                && self.is_id_typed(self.schema.elements[index].type_id)
            {
                return Err(SchemaError::new(
                    "e-props-correct.4",
                    format!(
                        "`{}` is of type `xs:ID`, so it may have no `default` or `fixed`",
                        self.schema.elements[index].name
                    ),
                ));
            }
        }
        for index in 0..self.schema.attributes.len() {
            if self.schema.attributes[index].value_constraint.is_some()
                && self.is_id_typed(self.schema.attributes[index].type_id)
            {
                return Err(SchemaError::new(
                    "a-props-correct.3",
                    format!(
                        "`{}` is of type `xs:ID`, so it may have no `default` or `fixed`",
                        self.schema.attributes[index].name
                    ),
                ));
            }
        }
        for index in 0..self.schema.elements.len() {
            let Some(constraint) = self.schema.elements[index].value_constraint.clone() else {
                continue;
            };
            let type_id = self.schema.elements[index].type_id;
            let resolved = self.evaluate_constraint(type_id, constraint, "e-props-correct.2")?;
            self.schema.elements[index].value_constraint = Some(resolved);
        }
        for index in 0..self.schema.attributes.len() {
            let Some(constraint) = self.schema.attributes[index].value_constraint.clone() else {
                continue;
            };
            let type_id = self.schema.attributes[index].type_id;
            let resolved = self.evaluate_constraint(type_id, constraint, "a-props-correct.2")?;
            self.schema.attributes[index].value_constraint = Some(resolved);
        }
        for index in 0..self.schema.types.len() {
            let TypeKind::Complex(complex) = &self.schema.types[index].kind else {
                continue;
            };
            let uses = complex.attribute_uses.clone();
            let mut resolved_uses = Vec::with_capacity(uses.len());
            for mut use_ in uses {
                if let Some(constraint) = use_.value_constraint.take() {
                    let type_id = self.schema.attributes[use_.attribute.index()].type_id;
                    use_.value_constraint =
                        Some(self.evaluate_constraint(type_id, constraint, "a-props-correct.2")?);
                }
                resolved_uses.push(use_);
            }
            if let TypeKind::Complex(complex) = &mut self.schema.types[index].kind {
                complex.attribute_uses = resolved_uses;
            }
        }
        Ok(())
    }

    fn evaluate_constraint(
        &self,
        type_id: TypeId,
        mut constraint: ValueConstraint,
        code: &str,
    ) -> Result<ValueConstraint, SchemaError> {
        let simple_id = match &self.schema.types[type_id.index()].kind {
            TypeKind::Simple(_) => Some(type_id),
            TypeKind::Complex(complex) => match &complex.content {
                crate::components::ContentType::Simple(inner) => Some(*inner),
                crate::components::ContentType::Mixed(_) => None,
                _ => {
                    return Err(SchemaError::new(
                        code,
                        format!(
                            "a `default` or `fixed` value needs simple or mixed content, but \
                             `{}` has neither",
                            self.schema.type_name(type_id)
                        ),
                    ));
                }
            },
        };
        match simple_id {
            Some(id) => {
                let value = simple::validate(
                    &self.schema,
                    id,
                    &constraint.lexical,
                    &constraint.namespaces,
                )
                .map_err(|error| {
                    SchemaError::new(
                        code,
                        format!(
                            "the value `{}` is not valid for `{}`: {error}",
                            constraint.lexical,
                            self.schema.type_name(id)
                        ),
                    )
                })?;
                constraint.normalized = value.normalized;
                constraint.value = Some(value.value);
            }
            None => {
                constraint.normalized = constraint.lexical.clone();
            }
        }
        Ok(constraint)
    }

    fn compute_substitution_groups(&mut self) -> Result<(), SchemaError> {
        // Direct members first…
        let mut direct: HashMap<u32, Vec<ElementId>> = HashMap::new();
        for (index, element) in self.schema.elements.iter().enumerate() {
            let member = ElementId(u32::try_from(index).unwrap_or(u32::MAX));
            for head in &element.substitution_group {
                direct.entry(head.0).or_default().push(member);
            }
        }
        // …then the transitive closure, breadth-first with a visited set so
        // that a cyclic `substitutionGroup` cannot loop.
        for index in 0..self.schema.elements.len() {
            let head = ElementId(u32::try_from(index).unwrap_or(u32::MAX));
            let mut seen: HashSet<u32> = HashSet::new();
            let mut queue: Vec<ElementId> = direct.get(&head.0).cloned().unwrap_or_default();
            let mut members = Vec::new();
            while let Some(member) = queue.pop() {
                if member == head || !seen.insert(member.0) {
                    continue;
                }
                members.push(member);
                if let Some(next) = direct.get(&member.0) {
                    queue.extend(next.iter().copied());
                }
            }
            self.schema.elements[index].substitutable_by = members;
        }
        Ok(())
    }
}

/// Names a target namespace for a diagnostic.
fn describe_namespace(namespace: Option<&str>) -> String {
    match namespace {
        Some(uri) => format!("`{uri}`"),
        None => "none".to_owned(),
    }
}

/// The `<restriction>` or `<extension>` element that gives a redefinition its
/// base, wherever it sits under a `<simpleType>` or `<complexType>`.
fn find_self_derivation(doc: &Document, node: NodeId) -> Option<NodeId> {
    for child in child_elements(doc, node) {
        match xsd_name(doc, child) {
            Some("restriction" | "extension") => return Some(child),
            Some("simpleContent" | "complexContent") => {
                return find_self_derivation(doc, child);
            }
            _ => {}
        }
    }
    None
}

/// Every `<group ref>` (or `<attributeGroup ref>`) anywhere below `node`.
fn self_references(doc: &Document, node: NodeId, wanted: &str) -> Vec<NodeId> {
    let mut found = Vec::new();
    let mut stack = child_elements(doc, node);
    while let Some(current) = stack.pop() {
        if xsd_name(doc, current) == Some(wanted)
            && doc.get_attribute_ns(current, None, "ref").is_some()
        {
            found.push(current);
        }
        stack.extend(child_elements(doc, current));
    }
    found
}

/// Remembers a redefinition that has no self-reference, so that
/// *src-redefine.6.2.2* / *7.2.2* can be checked once both components exist.
fn record_pending(
    out: &mut Vec<(RedefineKind, ExpandedName, ExpandedName)>,
    pending: &[(RedefineKind, ExpandedName)],
    kind: RedefineKind,
    names: Option<(ExpandedName, ExpandedName)>,
) {
    let Some((real, hidden)) = names else {
        return;
    };
    if pending
        .iter()
        .any(|(candidate, name)| *candidate == kind && *name == real)
    {
        out.push((kind, real, hidden));
    }
}
