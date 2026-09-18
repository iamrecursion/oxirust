//! A small, tolerant scanner for the document type declaration.
//!
//! [`oxixml_xml`] hands back the whole `<!DOCTYPE …>` as one
//! [`Event::DocType`](oxixml_xml::events::Event::DocType) without interpreting
//! it. Six things in it belong to the data model: the internal general
//! entities declared in the *internal* subset, so that a `&name;` in content can
//! be replaced by what it stands for; the external *parsed* general entities,
//! which a caller-supplied
//! [`ExternalEntityResolver`](crate::ExternalEntityResolver) can retrieve; the
//! unparsed (`NDATA`) entities, which
//! [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) makes
//! a property of the document node; the notations those entities name; the
//! attribute declarations of `<!ATTLIST …>`, whose defaults
//! [XML 1.0 §3.3.2](https://www.w3.org/TR/xml/#sec-attr-defaults) has the
//! processor add to every element that leaves the attribute out — including when
//! the attribute so supplied is a namespace declaration — and whose declared
//! types decide both how a value is normalized
//! ([§3.3.3](https://www.w3.org/TR/xml/#AVNormalize)) and which attributes are
//! IDs and references to them; and the content models of `<!ELEMENT …>`, which
//! decide which white space is *element content white space*
//! ([XML 1.0 §2.10](https://www.w3.org/TR/xml/#sec-white-space)) and therefore
//! never becomes a text node.
//!
//! This is not a DTD parser: parameter entities are not expanded, and a content
//! model is classified only as far as [`ContentModel`] draws the one distinction
//! the data model needs — the particle inside it belongs to validation, which
//! this crate does not do. The external subset is retrieved only when the caller asks
//! for it and supplies a resolver — a non-validating processor is explicitly
//! allowed not to read it ([XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types))
//! — and is then scanned by the very same walk, so a declaration means the same
//! thing wherever it was written. Everything the scanner does not understand is
//! skipped rather than rejected; a genuine syntax error in the declaration is
//! the reader's business, not this module's.

use std::collections::{BTreeMap, HashMap};

use oxixml_xml::XmlVersion;

use super::entity::replacement_text;

/// An unparsed general entity: one declared with an `NDATA` clause, whose
/// content is *not* XML and is therefore never retrieved or parsed.
///
/// [XDM 3.1 §6.1](https://www.w3.org/TR/xpath-datamodel-31/#DocumentNode) makes
/// the set of them an accessor of the document node, which is what
/// `fn:unparsed-entity-uri` and `fn:unparsed-entity-public-id` read.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::{ParseOptions, Tree};
///
/// let tree = Tree::parse_str_with(
///     r#"<!DOCTYPE r [
///          <!NOTATION png SYSTEM "image/png">
///          <!ENTITY logo SYSTEM "img/logo.png" NDATA png>
///        ]><r/>"#,
///     ParseOptions::default().document_uri("http://example.com/a/b.xml"),
/// )?;
///
/// let logo = tree.unparsed_entity("logo").expect("a declared unparsed entity");
/// assert_eq!(logo.system_id, "img/logo.png");
/// assert_eq!(logo.uri.as_deref(), Some("http://example.com/a/img/logo.png"));
/// assert_eq!(logo.public_id, None);
/// assert_eq!(logo.notation, "png");
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnparsedEntity {
    /// The system identifier exactly as the declaration wrote it, which the
    /// grammar allows to be a relative reference.
    pub system_id: String,
    /// The system identifier resolved against the base URI in force where the
    /// declaration appeared — the *absolute* URI XDM 3.1 §6.1 asks for.
    ///
    /// `None` when [`system_id`](Self::system_id) is relative and the tree has
    /// no base URI to resolve it against, or when either is not a valid IRI.
    pub uri: Option<String>,
    /// The public identifier, when the declaration gave one.
    pub public_id: Option<String>,
    /// The name of the notation the declaration's `NDATA` clause names.
    pub notation: String,
}

/// A notation declared by a `<!NOTATION …>` in the internal subset.
///
/// A notation names an external format; an unparsed entity refers to one, and
/// so may an attribute of type `NOTATION`. XML 1.0 §4.7 allows the declaration
/// to carry a system identifier, a public identifier, or both.
///
/// # Examples
///
/// ```
/// use oxixml_infoset::Tree;
///
/// let tree = Tree::parse_str(
///     r#"<!DOCTYPE r [<!NOTATION gif PUBLIC "-//x//NOTATION GIF//EN">]><r/>"#,
/// )?;
///
/// let gif = tree.notation("gif").expect("a declared notation");
/// assert_eq!(gif.public_id.as_deref(), Some("-//x//NOTATION GIF//EN"));
/// assert_eq!(gif.system_id, None);
/// # Ok::<(), oxixml_infoset::XdmError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notation {
    /// The system identifier exactly as the declaration wrote it, when it gave
    /// one. The `PublicID` form of the declaration has none.
    pub system_id: Option<String>,
    /// The system identifier resolved against the base URI in force where the
    /// declaration appeared, on the same terms as
    /// [`UnparsedEntity::uri`](crate::UnparsedEntity::uri).
    ///
    /// The resolution is mechanical: a `SystemLiteral` is a URI reference as far
    /// as the grammar is concerned, and it is resolved as one whatever it holds.
    /// A notation is only a *name* for a format, so a resolved value here is no
    /// claim that anything can be retrieved from it — `SYSTEM "image/png"` under
    /// a document URI of `http://example.com/a/b.xml` resolves to
    /// `http://example.com/a/image/png`, which locates nothing.
    pub uri: Option<String>,
    /// The public identifier, when the declaration gave one.
    pub public_id: Option<String>,
}

/// The external identifiers an `<!ENTITY … NDATA …>` declaration carries, before
/// the system identifier is resolved against a base URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct UnparsedEntityDeclaration {
    /// The public identifier, when the declaration gave one.
    pub(super) public_id: Option<String>,
    /// The system identifier as written.
    pub(super) system_id: String,
    /// The notation named by the `NDATA` clause.
    pub(super) notation: String,
    /// The absolute URI, when this declaration was carried over from a tree that had already
    /// resolved it. `None` — which is what a declaration read from a document type declaration
    /// carries — leaves [`system_id`](Self::system_id) to be resolved against the tree's own
    /// document URI.
    pub(super) uri: Option<String>,
}

/// The external identifiers an `<!ENTITY … SYSTEM "…">` declaration without an
/// `NDATA` clause carries: an external *parsed* general entity, whose
/// replacement text is XML and is retrieved only when the caller supplies an
/// [`ExternalEntityResolver`](crate::ExternalEntityResolver).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ExternalEntityDeclaration {
    /// The public identifier, when the declaration gave one.
    pub(super) public_id: Option<String>,
    /// The system identifier as written.
    pub(super) system_id: String,
}

/// The external identifiers a `<!NOTATION …>` declaration carries, before the
/// system identifier is resolved against a base URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NotationDeclaration {
    /// The public identifier, when the declaration gave one.
    pub(super) public_id: Option<String>,
    /// The system identifier as written, when the declaration gave one.
    pub(super) system_id: Option<String>,
    /// The absolute URI, on the same terms as
    /// [`UnparsedEntityDeclaration::uri`](UnparsedEntityDeclaration::uri).
    pub(super) uri: Option<String>,
}

/// The `AttType` of an `<!ATTLIST …>` definition, reduced to the three things
/// the data model takes from it.
///
/// ```text
/// [54] AttType        ::= StringType | TokenizedType | EnumeratedType
/// [56] TokenizedType  ::= 'ID' | 'IDREF' | 'IDREFS' | 'ENTITY' | 'ENTITIES' | 'NMTOKEN' | 'NMTOKENS'
/// ```
///
/// Every type but `CDATA` is *tokenized*, which decides how the value is
/// normalized ([XML 1.0 §3.3.3](https://www.w3.org/TR/xml/#AVNormalize)); `ID`
/// and the two reference types additionally decide the `is-id` and `is-idref`
/// properties [XDM 3.1
/// §5.10](https://www.w3.org/TR/xpath-datamodel-31/#dm-is-id) gives the
/// attribute node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeclaredType {
    /// `CDATA`: not tokenized, and neither an ID nor a reference to one.
    String,
    /// `ID`.
    Id,
    /// `IDREF` or `IDREFS`.
    IdRef,
    /// Any other tokenized type: `ENTITY`, `ENTITIES`, `NMTOKEN`, `NMTOKENS`,
    /// `NOTATION (…)` or an enumeration.
    Tokenized,
}

impl DeclaredType {
    /// Whether the value is normalized as a token list rather than as `CDATA`.
    pub(super) const fn is_tokenized(self) -> bool {
        !matches!(self, Self::String)
    }

    /// Whether an attribute so declared is an ID.
    pub(super) const fn is_id(self) -> bool {
        matches!(self, Self::Id)
    }

    /// Whether an attribute so declared refers to an ID.
    pub(super) const fn is_idref(self) -> bool {
        matches!(self, Self::IdRef)
    }
}

/// The `contentspec` of an `<!ELEMENT …>` declaration, reduced to the one
/// distinction the data model draws.
///
/// ```text
/// [46] contentspec ::= 'EMPTY' | 'ANY' | Mixed | children
/// [47] children    ::= (choice | seq) ('?' | '*' | '+')?
/// [51] Mixed       ::= '(' S? '#PCDATA' (S? '|' S? Name)* S? ')*'
///                    | '(' S? '#PCDATA' S? ')'
/// ```
///
/// An element type declared with `children` has *element content*
/// ([XML 1.0 §3.2.1](https://www.w3.org/TR/xml/#sec-element-content)): its
/// content is child elements "optionally separated by white space", and that
/// white space is the `[element content whitespace]` of
/// [XML 1.0 §2.10](https://www.w3.org/TR/xml/#sec-white-space). Every other
/// content specification either allows character data (`ANY`, `Mixed`) or
/// allows no content at all (`EMPTY`), so white space written inside it is
/// ordinary character data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContentModel {
    /// `children`: child elements, optionally separated by white space.
    ElementOnly,
    /// `EMPTY`, `ANY`, or a `Mixed` model — anything that is not element
    /// content.
    Other,
}

impl ContentModel {
    /// Whether white space written directly inside an element so declared is
    /// element content white space.
    pub(super) const fn is_element_only(self) -> bool {
        matches!(self, Self::ElementOnly)
    }
}

/// One attribute an `<!ATTLIST …>` declares.
///
/// Every definition is recorded, whether or not it supplies a default value:
/// the declared type belongs to the data model on its own, because it is what
/// makes an attribute an ID and what decides how its value is normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AttributeDeclaration {
    /// The attribute's name exactly as the declaration wrote it. It is matched
    /// against the start tag lexically, so a prefix takes part in the match.
    pub(super) name: String,
    /// The declared `AttType`.
    pub(super) declared_type: DeclaredType,
    /// The `AttValue` between the quotation marks, before normalization: what
    /// the references inside it stand for can only be decided where the
    /// document's entities are all known.
    ///
    /// Only the third form of `DefaultDecl` — `(('#FIXED' S)? AttValue)` —
    /// supplies one, so `#REQUIRED` and `#IMPLIED` leave it `None`. `#FIXED` and
    /// a bare `AttValue` are the same as far as the data model is concerned: a
    /// non-validating processor supplies the value either way, and the extra
    /// constraint `#FIXED` places on a written value is a validity constraint.
    pub(super) default: Option<String>,
}

/// Everything the data model takes from a document type declaration.
///
/// The maps are ordered so that a tree built from the same document twice
/// reports its declarations in the same order.
#[derive(Debug, Default)]
pub(super) struct InternalSubset {
    /// Internal general entities, each holding the *replacement text* of
    /// [XML 1.0 §4.5](https://www.w3.org/TR/xml/#intern-replacement).
    pub(super) entities: HashMap<String, String>,
    /// External *parsed* general entities: the ones a resolver can retrieve.
    pub(super) external_entities: BTreeMap<String, ExternalEntityDeclaration>,
    /// Unparsed (`NDATA`) general entities.
    pub(super) unparsed_entities: BTreeMap<String, UnparsedEntityDeclaration>,
    /// Notations.
    pub(super) notations: BTreeMap<String, NotationDeclaration>,
    /// Attribute declarations, keyed on the element name the declaration wrote
    /// and held in declaration order.
    pub(super) attribute_declarations: BTreeMap<String, Vec<AttributeDeclaration>>,
    /// The content model each `<!ELEMENT …>` declaration gives an element type,
    /// keyed on the element name the declaration wrote.
    pub(super) content_models: BTreeMap<String, ContentModel>,
}

impl InternalSubset {
    /// Records `declaration` for `element`, unless an earlier declaration
    /// already declared that attribute.
    ///
    /// [XML 1.0 §3.3](https://www.w3.org/TR/xml/#attdecls) makes a second
    /// declaration of the same attribute of the same element a validity error,
    /// which a non-validating processor reports by keeping the first — the same
    /// rule §2.8 gives the internal subset over the external one, since the
    /// internal subset is scanned first. The type and the default value travel
    /// together, so the winning declaration decides both.
    fn declare_attribute(&mut self, element: &str, declaration: AttributeDeclaration) {
        let declared = self
            .attribute_declarations
            .entry(element.to_owned())
            .or_default();
        if declared
            .iter()
            .any(|existing| existing.name == declaration.name)
        {
            return;
        }
        declared.push(declaration);
    }
}

/// Collects the entities, notations and attribute defaults declared by the
/// *internal* subset of a `<!DOCTYPE …>`.
///
/// `body` is the text of the declaration with `<!DOCTYPE` and the closing `>`
/// already stripped, which is exactly what the `DocType` event carries.
/// `version` decides which characters a character reference inside an
/// `EntityValue` may denote.
pub(super) fn scan_internal_subset(body: &str, version: XmlVersion) -> InternalSubset {
    let mut subset_items = InternalSubset::default();
    if let Some(subset) = internal_subset(body) {
        scan_declarations(&mut subset_items, subset, version);
    }
    subset_items
}

/// Adds everything `declarations` declares to `subset_items`, which lets the
/// external subset be scanned into the same collection the internal one
/// produced. A name the internal subset already bound keeps its first
/// declaration, as [XML 1.0 §2.8](https://www.w3.org/TR/xml/#sec-prolog-dtd)
/// requires.
///
/// `subset` is a bare `markupdecl*`: the internal subset without its brackets,
/// or the whole text of an external subset.
pub(super) fn scan_declarations(
    subset_items: &mut InternalSubset,
    subset: &str,
    version: XmlVersion,
) {
    let bytes = subset.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if subset[cursor..].starts_with("<!--") {
            cursor = subset[cursor..]
                .find("-->")
                .map_or(bytes.len(), |offset| cursor + offset + "-->".len());
            continue;
        }
        if subset[cursor..].starts_with("<?") {
            cursor = subset[cursor..]
                .find("?>")
                .map_or(bytes.len(), |offset| cursor + offset + "?>".len());
            continue;
        }
        // `conditionalSect ::= includeSect | ignoreSect` — only the external
        // subset may hold one. The declarations inside an `INCLUDE` are
        // ordinary declarations, so only the section's own delimiters are
        // skipped; everything inside an `IGNORE` is not part of the DTD at all.
        if subset[cursor..].starts_with("<![") {
            cursor = conditional_section(subset, cursor);
            continue;
        }
        if subset[cursor..].starts_with("]]>") {
            cursor += "]]>".len();
            continue;
        }
        if subset[cursor..].starts_with("<!ATTLIST") {
            let (declaration, next) = read_declaration(subset, cursor + "<!ATTLIST".len());
            if let Some((element, attributes)) = parse_attlist(declaration) {
                for attribute in attributes {
                    subset_items.declare_attribute(element, attribute);
                }
            }
            cursor = next;
            continue;
        }
        if subset[cursor..].starts_with("<!ENTITY") {
            let (declaration, next) = read_declaration(subset, cursor + "<!ENTITY".len());
            // XML 1.0 §4.2: when an entity is declared more than once, the
            // *first* declaration binds — whichever kind of declaration it was,
            // which is why each arm looks in the other general-entity map too.
            match parse_entity(declaration) {
                Some(EntityDeclaration::Internal { name, literal })
                    if !subset_items.external_entities.contains_key(name) =>
                {
                    subset_items
                        .entities
                        .entry(name.to_owned())
                        .or_insert_with(|| replacement_text(literal, version));
                }
                Some(EntityDeclaration::External {
                    name,
                    public_id,
                    system_id,
                }) if !subset_items.entities.contains_key(name) => {
                    subset_items
                        .external_entities
                        .entry(name.to_owned())
                        .or_insert_with(|| ExternalEntityDeclaration {
                            public_id: public_id.map(str::to_owned),
                            system_id: system_id.to_owned(),
                        });
                }
                Some(EntityDeclaration::Unparsed {
                    name,
                    public_id,
                    system_id,
                    notation,
                }) => {
                    subset_items
                        .unparsed_entities
                        .entry(name.to_owned())
                        .or_insert_with(|| UnparsedEntityDeclaration {
                            public_id: public_id.map(str::to_owned),
                            system_id: system_id.to_owned(),
                            notation: notation.to_owned(),
                            uri: None,
                        });
                }
                // A general entity whose name the other kind of declaration already bound, a
                // parameter entity, and a declaration this scanner does not understand.
                Some(EntityDeclaration::Internal { .. } | EntityDeclaration::External { .. })
                | None => {}
            }
            cursor = next;
            continue;
        }
        if subset[cursor..].starts_with("<!ELEMENT") {
            let (declaration, next) = read_declaration(subset, cursor + "<!ELEMENT".len());
            if let Some((element, model)) = parse_elementdecl(declaration) {
                // XML 1.0 VC: Unique Element Type Declaration — a second
                // declaration of the same element type is a validity error, and
                // a non-validating processor reports it by keeping the first.
                subset_items
                    .content_models
                    .entry(element.to_owned())
                    .or_insert(model);
            }
            cursor = next;
            continue;
        }
        if subset[cursor..].starts_with("<!NOTATION") {
            let (declaration, next) = read_declaration(subset, cursor + "<!NOTATION".len());
            if let Some((name, public_id, system_id)) = parse_notation(declaration) {
                // XML 1.0 VC: Unique Notation Name makes a second declaration a
                // validity error, which a non-validating processor reports by
                // keeping the first, as it does for entities.
                subset_items
                    .notations
                    .entry(name.to_owned())
                    .or_insert_with(|| NotationDeclaration {
                        public_id: public_id.map(str::to_owned),
                        system_id: system_id.map(str::to_owned),
                        uri: None,
                    });
            }
            cursor = next;
            continue;
        }
        cursor += char_len(bytes, cursor);
    }
}

/// Returns the text strictly between the internal subset's `[` and its `]`.
fn internal_subset(body: &str) -> Option<&str> {
    let start = body.find('[')?;
    let end = body.rfind(']')?;
    (end > start).then(|| &body[start + 1..end])
}

/// The external identifier a `<!DOCTYPE …>` gives for its external subset, if it
/// gives one.
///
/// `[28] doctypedecl ::= '<!DOCTYPE' S Name (S ExternalID)? S? ('[' intSubset ']' S?)? '>'`
/// — the identifier, when present, always precedes the internal subset, so
/// everything from the `[` on is left out of the match.
pub(super) fn external_subset_id(body: &str) -> Option<ExternalEntityDeclaration> {
    let head = body.split_once('[').map_or(body, |(head, _)| head);
    let head = head.trim_start_matches(is_space);
    // The document type name is followed by white space exactly when an
    // external identifier follows it.
    let after_name = head.find(is_space)?;
    let (public_id, system_id, rest) =
        external_id(head[after_name..].trim_start_matches(is_space))?;
    let system_id = system_id?;
    rest.trim_matches(is_space)
        .is_empty()
        .then(|| ExternalEntityDeclaration {
            public_id: public_id.map(str::to_owned),
            system_id: system_id.to_owned(),
        })
}

/// Skips the delimiters of a conditional section beginning at `cursor`,
/// returning where scanning resumes.
///
/// An `INCLUDE` section contributes its declarations, so only its
/// `<![ S? INCLUDE S? [` header is consumed and the walk carries on inside it.
/// An `IGNORE` section contributes nothing, so the whole of it — including any
/// sections nested within — is skipped. A header this scanner cannot classify
/// (one written with a parameter-entity reference, which nothing here expands)
/// is treated as `INCLUDE`, which keeps a declaration rather than losing one.
fn conditional_section(subset: &str, cursor: usize) -> usize {
    let after_bracket = cursor + "<![".len();
    let keyword_start = after_bracket + count_space(&subset[after_bracket..]);
    let header = &subset[keyword_start..];
    let Some(after_keyword) = header.strip_prefix("IGNORE") else {
        // `INCLUDE`, or a header this scanner does not understand: step past the
        // `[` that opens the section and carry on scanning inside it.
        let Some(after_keyword) = header.strip_prefix("INCLUDE") else {
            return after_bracket;
        };
        let opening = keyword_start + "INCLUDE".len() + count_space(after_keyword);
        return match subset[opening..].starts_with('[') {
            true => opening + 1,
            false => after_bracket,
        };
    };
    let mut index = keyword_start + "IGNORE".len() + count_space(after_keyword);
    // A header without its `[` is malformed; skipping the keyword is as far as
    // this scanner can safely go.
    if !subset[index..].starts_with('[') {
        return index;
    }
    index += 1;
    let mut depth = 1_usize;
    let bytes = subset.as_bytes();
    while index < bytes.len() {
        if subset[index..].starts_with("<![") {
            depth += 1;
            index += "<![".len();
            continue;
        }
        if subset[index..].starts_with("]]>") {
            index += "]]>".len();
            depth -= 1;
            if depth == 0 {
                return index;
            }
            continue;
        }
        index += char_len(bytes, index);
    }
    index
}

/// The number of bytes of white space at the start of `text`.
fn count_space(text: &str) -> usize {
    text.len() - text.trim_start_matches(is_space).len()
}

/// Parses the body of an `<!ATTLIST …>` declaration into the element name it
/// applies to and the attributes it declares.
///
/// ```text
/// [52] AttlistDecl ::= '<!ATTLIST' S Name AttDef* S? '>'
/// [53] AttDef      ::= S Name S AttType S DefaultDecl
/// ```
///
/// The whole declaration is abandoned as soon as one `AttDef` cannot be read:
/// the definitions are positional, so a scanner that guessed its way past a
/// malformed one would attach the *following* type and default to the wrong
/// attribute.
fn parse_attlist(body: &str) -> Option<(&str, Vec<AttributeDeclaration>)> {
    let (element, mut rest) = split_name(body.trim_start_matches(is_space))?;
    let mut attributes = Vec::new();
    loop {
        rest = rest.trim_start_matches(is_space);
        if rest.is_empty() {
            return Some((element, attributes));
        }
        let (name, after_name) = split_name(rest)?;
        let (declared_type, after_type) = attribute_type(after_name)?;
        let (literal, after_default) = attribute_default(after_type)?;
        attributes.push(AttributeDeclaration {
            name: name.to_owned(),
            declared_type,
            default: literal.map(str::to_owned),
        });
        rest = after_default;
    }
}

/// Parses the body of an `<!ELEMENT …>` declaration into the element type it
/// declares and the content model it gives it.
///
/// ```text
/// [45] elementdecl ::= '<!ELEMENT' S Name S contentspec S? '>'
/// ```
///
/// Only the *shape* of the `contentspec` matters here, not the particle inside
/// it: a model that opens with `#PCDATA` is `Mixed` and every other
/// parenthesized model is `children`. A specification written with a
/// parameter-entity reference is classified as nothing at all — this scanner
/// expands no parameter entity, so it cannot tell `(%x;)` apart from
/// `(#PCDATA|%x;)`, and guessing would strip white space that belongs to the
/// document.
fn parse_elementdecl(body: &str) -> Option<(&str, ContentModel)> {
    let (element, rest) = split_name(body.trim_start_matches(is_space))?;
    let spec = rest.trim_matches(is_space);
    if spec == "EMPTY" || spec == "ANY" {
        return Some((element, ContentModel::Other));
    }
    let group = spec.strip_prefix('(')?;
    if spec.contains('%') {
        return None;
    }
    let model = match group.trim_start_matches(is_space).starts_with("#PCDATA") {
        true => ContentModel::Other,
        false => ContentModel::ElementOnly,
    };
    Some((element, model))
}

/// Parses an `AttType` at the start of `text`, returning what it declares and
/// what follows it.
///
/// ```text
/// [54] AttType        ::= StringType | TokenizedType | EnumeratedType
/// [55] StringType     ::= 'CDATA'
/// [56] TokenizedType  ::= 'ID' | 'IDREF' | 'IDREFS' | 'ENTITY' | 'ENTITIES' | 'NMTOKEN' | 'NMTOKENS'
/// [57] EnumeratedType ::= NotationType | Enumeration
/// [58] NotationType   ::= 'NOTATION' S '(' S? Name (S? '|' S? Name)* S? ')'
/// [59] Enumeration    ::= '(' S? Nmtoken (S? '|' S? Nmtoken)* S? ')'
/// ```
///
/// Both parenthesized forms hold white space and `|`, neither of which may end
/// the type: only the closing parenthesis does, and no `Name` or `Nmtoken` can
/// contain one.
fn attribute_type(text: &str) -> Option<(DeclaredType, &str)> {
    let text = text.trim_start_matches(is_space);
    if let Some(rest) = keyword(text, "NOTATION") {
        let rest = rest.trim_start_matches(is_space).strip_prefix('(')?;
        let end = rest.find(')')?;
        return Some((DeclaredType::Tokenized, &rest[end + 1..]));
    }
    if let Some(rest) = text.strip_prefix('(') {
        let end = rest.find(')')?;
        return Some((DeclaredType::Tokenized, &rest[end + 1..]));
    }
    let (name, rest) = split_name(text)?;
    match name {
        "CDATA" => Some((DeclaredType::String, rest)),
        "ID" => Some((DeclaredType::Id, rest)),
        "IDREF" | "IDREFS" => Some((DeclaredType::IdRef, rest)),
        "ENTITY" | "ENTITIES" | "NMTOKEN" | "NMTOKENS" => Some((DeclaredType::Tokenized, rest)),
        _ => None,
    }
}

/// Parses a `DefaultDecl` at the start of `text`, returning the `AttValue` it
/// supplies — `None` for `#REQUIRED` and `#IMPLIED`, which supply nothing — and
/// what follows it.
///
/// `[60] DefaultDecl ::= '#REQUIRED' | '#IMPLIED' | (('#FIXED' S)? AttValue)`
fn attribute_default(text: &str) -> Option<(Option<&str>, &str)> {
    let text = text.trim_start_matches(is_space);
    for word in ["#REQUIRED", "#IMPLIED"] {
        if let Some(rest) = text.strip_prefix(word)
            && (rest.is_empty() || rest.starts_with(is_space))
        {
            return Some((None, rest));
        }
    }
    let text = match keyword(text, "#FIXED") {
        Some(rest) => rest.trim_start_matches(is_space),
        None => text,
    };
    let (literal, rest) = quoted(text)?;
    Some((Some(literal), rest))
}

/// Reads a markup declaration body up to the terminating `>`, honouring quoted
/// literals so that a `>` inside one does not end it. Returns the body and the
/// offset just past the `>`.
fn read_declaration(text: &str, start: usize) -> (&str, usize) {
    let bytes = text.as_bytes();
    let mut cursor = start;
    let mut quote: Option<u8> = None;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        match quote {
            Some(open) if byte == open => quote = None,
            Some(_) => {}
            None => match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'>' => return (&text[start..cursor], cursor + 1),
                _ => {}
            },
        }
        cursor += char_len(bytes, cursor);
    }
    (&text[start..], bytes.len())
}

/// What an `<!ENTITY …>` declaration the scanner understands declares.
///
/// A parameter entity (`% name`) is none of these: [`parse_entity`] returns
/// `None` for it, because the data model gets nothing from a declaration this
/// crate never expands.
#[derive(Debug)]
enum EntityDeclaration<'a> {
    /// An internal general entity, with the literal entity value between the
    /// quotes of its declaration.
    Internal {
        /// The entity's name.
        name: &'a str,
        /// The literal entity value.
        literal: &'a str,
    },
    /// An external general entity without an `NDATA` clause: a *parsed* entity,
    /// whose replacement text is XML.
    External {
        /// The entity's name.
        name: &'a str,
        /// The public identifier, when the declaration gave one.
        public_id: Option<&'a str>,
        /// The system identifier.
        system_id: &'a str,
    },
    /// An external general entity with an `NDATA` clause: an unparsed entity.
    Unparsed {
        /// The entity's name.
        name: &'a str,
        /// The public identifier, when the declaration gave one.
        public_id: Option<&'a str>,
        /// The system identifier.
        system_id: &'a str,
        /// The notation the `NDATA` clause names.
        notation: &'a str,
    },
}

/// Parses the body of an `<!ENTITY …>` declaration.
fn parse_entity(body: &str) -> Option<EntityDeclaration<'_>> {
    let body = body.trim_start_matches(is_space);
    if body.starts_with('%') {
        return None;
    }
    let (name, rest) = split_name(body)?;
    let rest = rest.trim_start_matches(is_space);
    if let Some(quote) = quote_of(rest) {
        let literal_end = rest[1..].find(quote)? + 1;
        // Anything after the closing quote of an `EntityValue` — in practice
        // only an `NDATA` clause, which is not even legal here — means this is
        // not a plain internal entity.
        if !rest[literal_end + 1..].trim_matches(is_space).is_empty() {
            return None;
        }
        return Some(EntityDeclaration::Internal {
            name,
            literal: &rest[1..literal_end],
        });
    }

    // `EntityDef ::= EntityValue | (ExternalID NDataDecl?)`. Without the
    // `NDataDecl` the entity is an external *parsed* one, whose replacement text
    // is XML and is retrieved only through a caller-supplied
    // [`ExternalEntityResolver`](crate::ExternalEntityResolver).
    let (public_id, system_id, rest) = external_id(rest)?;
    let system_id = system_id?;
    let rest = rest.trim_start_matches(is_space);
    let Some(rest) = keyword(rest, "NDATA") else {
        return rest.is_empty().then_some(EntityDeclaration::External {
            name,
            public_id,
            system_id,
        });
    };
    let (notation, rest) = split_name(rest.trim_start_matches(is_space))?;
    rest.trim_matches(is_space)
        .is_empty()
        .then_some(EntityDeclaration::Unparsed {
            name,
            public_id,
            system_id,
            notation,
        })
}

/// Parses the body of a `<!NOTATION …>` declaration into
/// `(name, public id, system id)`.
///
/// `[82] NotationDecl ::= '<!NOTATION' S Name S (ExternalID | PublicID) S? '>'`
/// — the `PublicID` alternative carries no system identifier.
fn parse_notation(body: &str) -> Option<(&str, Option<&str>, Option<&str>)> {
    let body = body.trim_start_matches(is_space);
    let (name, rest) = split_name(body)?;
    let (public_id, system_id, rest) = external_id(rest.trim_start_matches(is_space))?;
    rest.trim_matches(is_space)
        .is_empty()
        .then_some((name, public_id, system_id))
}

/// Parses an `ExternalID` or a `PublicID` at the start of `text`, returning
/// `(public id, system id, the rest of the declaration)`.
///
/// ```text
/// [75] ExternalID ::= 'SYSTEM' S SystemLiteral | 'PUBLIC' S PubidLiteral S SystemLiteral
/// [83] PublicID   ::= 'PUBLIC' S PubidLiteral
/// ```
///
/// The two `PUBLIC` forms are told apart by what follows the public identifier:
/// a quoted literal is the system identifier of an `ExternalID`, anything else
/// ends a `PublicID`.
fn external_id(text: &str) -> Option<(Option<&str>, Option<&str>, &str)> {
    if let Some(rest) = keyword(text, "SYSTEM") {
        let (system_id, rest) = quoted(rest.trim_start_matches(is_space))?;
        return Some((None, Some(system_id), rest));
    }
    let rest = keyword(text, "PUBLIC")?;
    let (public_id, rest) = quoted(rest.trim_start_matches(is_space))?;
    let after = rest.trim_start_matches(is_space);
    match quote_of(after) {
        Some(_) => {
            let (system_id, rest) = quoted(after)?;
            Some((Some(public_id), Some(system_id), rest))
        }
        None => Some((Some(public_id), None, rest)),
    }
}

/// Strips `word` from the start of `text`, requiring the white space the
/// grammar puts after it, and returns what follows.
fn keyword<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let rest = text.strip_prefix(word)?;
    rest.starts_with(is_space).then_some(rest)
}

/// Splits a `Name` off the front of `text`, returning it and the rest.
///
/// The name is delimited by white space or by the quotation mark that opens the
/// next literal; the characters inside it are the reader's business, and a name
/// this scanner mis-splits ends up as a declaration nothing refers to.
fn split_name(text: &str) -> Option<(&str, &str)> {
    let end = text
        .find(|character| is_space(character) || matches!(character, '"' | '\''))
        .unwrap_or(text.len());
    (end > 0).then(|| (&text[..end], &text[end..]))
}

/// Returns the quotation mark `text` begins with, if it begins with one.
fn quote_of(text: &str) -> Option<char> {
    text.chars().next().filter(|&c| matches!(c, '"' | '\''))
}

/// Reads a quoted literal at the start of `text`, returning its contents and
/// what follows the closing quotation mark.
fn quoted(text: &str) -> Option<(&str, &str)> {
    let quote = quote_of(text)?;
    let body = &text[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some((&body[..end], &body[end + quote.len_utf8()..]))
}

/// Returns whether `character` is one of the four XML 1.0 `S` (white space)
/// characters.
pub(super) const fn is_space(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\r' | '\n')
}

/// Returns the length of the UTF-8 sequence starting at `cursor`, so that byte
/// offsets always land on character boundaries.
fn char_len(bytes: &[u8], cursor: usize) -> usize {
    match bytes.get(cursor) {
        None => 1,
        Some(&byte) if byte < 0x80 => 1,
        Some(&byte) if byte < 0xE0 => 2,
        Some(&byte) if byte < 0xF0 => 3,
        Some(_) => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(body: &str) -> InternalSubset {
        scan_internal_subset(body, XmlVersion::Implicit1_0)
    }

    #[test]
    fn no_internal_subset_declares_nothing() {
        assert!(scan("root").entities.is_empty());
        assert!(scan(r#"root SYSTEM "root.dtd""#).entities.is_empty());
    }

    #[test]
    fn internal_entities_are_collected_with_their_replacement_text() {
        let entities =
            scan(r#"root [ <!ENTITY a "plain"> <!ENTITY b '&#169;'> <!ENTITY c "&a; and more"> ]"#)
                .entities;
        assert_eq!(entities.get("a").map(String::as_str), Some("plain"));
        assert_eq!(entities.get("b").map(String::as_str), Some("\u{a9}"));
        assert_eq!(entities.get("c").map(String::as_str), Some("&a; and more"));
    }

    #[test]
    fn a_replacement_text_follows_the_documents_version() {
        let subset = r#"root [<!ENTITY bel "&#x7;">]"#;
        assert_eq!(
            scan_internal_subset(subset, XmlVersion::Explicit1_1)
                .entities
                .get("bel")
                .map(String::as_str),
            Some("\u{7}")
        );
        // `#x7` is not a `Char` in XML 1.0, so the reference is not one either.
        assert_eq!(
            scan(subset).entities.get("bel").map(String::as_str),
            Some("&#x7;")
        );
    }

    #[test]
    fn entities_without_replacement_text_are_left_out() {
        let subset = scan(
            r#"root [
                <!ENTITY % param "not a general entity">
                <!ENTITY external SYSTEM "e.xml">
                <!ENTITY public PUBLIC "-//x//EN" "e.xml">
                <!ELEMENT root ANY>
                <!ATTLIST root id ID #REQUIRED>
            ]"#,
        );
        assert!(subset.entities.is_empty(), "{:?}", subset.entities);
        assert!(subset.unparsed_entities.is_empty());
    }

    #[test]
    fn unparsed_entities_and_notations_are_collected() {
        let subset = scan(
            r#"root [
                <!NOTATION png SYSTEM "image/png">
                <!NOTATION gif PUBLIC "-//x//NOTATION GIF//EN">
                <!NOTATION both PUBLIC "-//x//NOTATION BOTH//EN" "both.exe">
                <!ENTITY logo SYSTEM "logo.png" NDATA png>
                <!ENTITY badge PUBLIC "-//x//BADGE//EN" "badge.gif" NDATA gif>
            ]"#,
        );
        assert!(subset.entities.is_empty(), "{:?}", subset.entities);

        let logo = subset.unparsed_entities.get("logo").expect("`logo`");
        assert_eq!(logo.public_id, None);
        assert_eq!(logo.system_id, "logo.png");
        assert_eq!(logo.notation, "png");

        let badge = subset.unparsed_entities.get("badge").expect("`badge`");
        assert_eq!(badge.public_id.as_deref(), Some("-//x//BADGE//EN"));
        assert_eq!(badge.system_id, "badge.gif");
        assert_eq!(badge.notation, "gif");

        let png = subset.notations.get("png").expect("`png`");
        assert_eq!(png.public_id, None);
        assert_eq!(png.system_id.as_deref(), Some("image/png"));

        // The `PublicID` form of the declaration has no system identifier.
        let gif = subset.notations.get("gif").expect("`gif`");
        assert_eq!(gif.public_id.as_deref(), Some("-//x//NOTATION GIF//EN"));
        assert_eq!(gif.system_id, None);

        let both = subset.notations.get("both").expect("`both`");
        assert_eq!(both.public_id.as_deref(), Some("-//x//NOTATION BOTH//EN"));
        assert_eq!(both.system_id.as_deref(), Some("both.exe"));
    }

    #[test]
    fn declarations_hidden_in_comments_and_quotes_are_not_declarations() {
        let subset = scan(
            r#"root [ <!-- <!ENTITY hidden "no"> --> <?pi <!ENTITY also "no"> ?>
               <!-- <!NOTATION hidden SYSTEM "no"> -->
               <!ENTITY bracket "a > b"> ]"#,
        );
        assert_eq!(subset.entities.len(), 1, "{:?}", subset.entities);
        assert_eq!(
            subset.entities.get("bracket").map(String::as_str),
            Some("a > b")
        );
        assert!(subset.notations.is_empty(), "{:?}", subset.notations);
    }

    #[test]
    fn the_first_declaration_of_a_name_binds() {
        let subset = scan(
            r#"root [<!ENTITY e "first"><!ENTITY e "second">
                 <!NOTATION n SYSTEM "first.exe"><!NOTATION n SYSTEM "second.exe">
                 <!ENTITY u SYSTEM "first.bin" NDATA n><!ENTITY u SYSTEM "second.bin" NDATA n>]"#,
        );
        assert_eq!(subset.entities.get("e").map(String::as_str), Some("first"));
        assert_eq!(
            subset
                .notations
                .get("n")
                .and_then(|notation| notation.system_id.as_deref()),
            Some("first.exe")
        );
        assert_eq!(
            subset
                .unparsed_entities
                .get("u")
                .map(|entity| entity.system_id.as_str()),
            Some("first.bin")
        );
    }

    #[test]
    fn malformed_external_identifiers_are_skipped_rather_than_guessed() {
        let subset = scan(
            r#"root [
                <!ENTITY missing SYSTEM NDATA png>
                <!ENTITY unquoted SYSTEM logo.png NDATA png>
                <!ENTITY nameless SYSTEM "logo.png" NDATA>
                <!NOTATION lonely>
                <!NOTATION unclosed SYSTEM "png>
            ]"#,
        );
        assert!(
            subset.unparsed_entities.is_empty(),
            "{:?}",
            subset.unparsed_entities
        );
        assert!(subset.notations.is_empty(), "{:?}", subset.notations);
    }

    #[test]
    fn an_apostrophe_quoted_system_identifier_is_read_like_any_other() {
        let subset = scan(r#"root [<!ENTITY logo SYSTEM 'a "quoted" name.png' NDATA png>]"#);
        assert_eq!(
            subset
                .unparsed_entities
                .get("logo")
                .map(|entity| entity.system_id.as_str()),
            Some(r#"a "quoted" name.png"#)
        );
    }

    /// The attributes `scan` found for `element`, as `(name, default, tokenized)`.
    fn defaults(subset: &InternalSubset, element: &str) -> Vec<(String, String, bool)> {
        declarations(subset, element)
            .into_iter()
            .filter_map(|declaration| {
                declaration.default.map(|default| {
                    (
                        declaration.name,
                        default,
                        declaration.declared_type.is_tokenized(),
                    )
                })
            })
            .collect()
    }

    /// Every attribute `scan` found declared for `element`, in declaration order.
    fn declarations(subset: &InternalSubset, element: &str) -> Vec<AttributeDeclaration> {
        subset
            .attribute_declarations
            .get(element)
            .cloned()
            .unwrap_or_default()
    }

    #[test]
    fn one_attribute_list_may_define_several_attributes() {
        // The two parenthesized forms of `AttType` hold white space and `|`; a walk that took
        // either for the end of the type would attach `'x'` to the wrong attribute.
        let subset = scan(
            r#"doc [<!ATTLIST e a CDATA #IMPLIED b (x | y) 'x' c NOTATION (n) #REQUIRED d CDATA #FIXED "v">]"#,
        );
        assert_eq!(
            defaults(&subset, "e"),
            [
                ("b".to_owned(), "x".to_owned(), true),
                ("d".to_owned(), "v".to_owned(), false)
            ]
        );
        // The two definitions that supply no default are still declarations: their types are
        // what make an attribute written in the start tag an ID or a token list.
        assert_eq!(
            declarations(&subset, "e")
                .into_iter()
                .map(|declaration| declaration.name)
                .collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
    }

    #[test]
    fn every_declared_type_is_recognized() {
        let subset = scan(
            r#"doc [<!ATTLIST e
                     s CDATA "1" i ID "2" r IDREF "3" rs IDREFS "4"
                     n ENTITY "5" ns ENTITIES "6" t NMTOKEN "7" ts NMTOKENS "8"
                     o NOTATION (a | b) "9" u (p|q) "10">]"#,
        );
        assert_eq!(
            declarations(&subset, "e")
                .into_iter()
                .map(|declaration| (declaration.name, declaration.declared_type))
                .collect::<Vec<_>>(),
            [
                ("s".to_owned(), DeclaredType::String),
                ("i".to_owned(), DeclaredType::Id),
                ("r".to_owned(), DeclaredType::IdRef),
                ("rs".to_owned(), DeclaredType::IdRef),
                ("n".to_owned(), DeclaredType::Tokenized),
                ("ns".to_owned(), DeclaredType::Tokenized),
                ("t".to_owned(), DeclaredType::Tokenized),
                ("ts".to_owned(), DeclaredType::Tokenized),
                ("o".to_owned(), DeclaredType::Tokenized),
                ("u".to_owned(), DeclaredType::Tokenized),
            ]
        );
        assert!(!DeclaredType::String.is_tokenized());
        assert!(DeclaredType::Id.is_id() && !DeclaredType::Id.is_idref());
        assert!(DeclaredType::IdRef.is_idref() && !DeclaredType::IdRef.is_id());
        assert!(!DeclaredType::Tokenized.is_id() && !DeclaredType::Tokenized.is_idref());
    }

    #[test]
    fn a_malformed_definition_abandons_its_whole_declaration() {
        // Attribute definitions are positional, so a scanner that guessed its way past one it
        // could not read would attach the following type and default to the wrong attribute.
        for subset in [
            r#"doc [<!ATTLIST e a NOTHING "v" b CDATA "w">]"#,
            r#"doc [<!ATTLIST e a CDATA>]"#,
            r#"doc [<!ATTLIST e a (x|y "v">]"#,
        ] {
            assert!(
                scan(subset).attribute_declarations.is_empty(),
                "{subset}: {:?}",
                scan(subset).attribute_declarations
            );
        }
    }

    #[test]
    fn the_first_declaration_of_an_attribute_binds() {
        let subset =
            scan(r#"doc [<!ATTLIST e a CDATA "first" b CDATA "b"><!ATTLIST e a CDATA "second">]"#);
        assert_eq!(
            defaults(&subset, "e"),
            [
                ("a".to_owned(), "first".to_owned(), false),
                ("b".to_owned(), "b".to_owned(), false)
            ]
        );
    }

    #[test]
    fn the_first_declaration_of_an_attribute_decides_its_type() {
        // XML 1.0 §2.8 scans the internal subset first, and `scan_declarations` adds the
        // external one to the same collection, so a redeclaration cannot take an attribute's
        // ID-ness away.
        let subset =
            scan(r#"doc [<!ATTLIST e a ID #IMPLIED b CDATA "b"><!ATTLIST e a CDATA "second">]"#);
        assert_eq!(
            declarations(&subset, "e")
                .into_iter()
                .map(|declaration| (
                    declaration.name,
                    declaration.declared_type,
                    declaration.default
                ))
                .collect::<Vec<_>>(),
            [
                ("a".to_owned(), DeclaredType::Id, None),
                ("b".to_owned(), DeclaredType::String, Some("b".to_owned())),
            ]
        );
    }

    #[test]
    fn an_external_identifier_is_found_only_before_the_internal_subset() {
        let system = external_subset_id(r#"Section SYSTEM "s.dtd""#).expect("a system identifier");
        assert_eq!(system.system_id, "s.dtd");
        assert_eq!(system.public_id, None);

        let public = external_subset_id(r#"r PUBLIC "-//x//DTD R//EN" "r.dtd" ["#)
            .expect("both identifiers");
        assert_eq!(public.public_id.as_deref(), Some("-//x//DTD R//EN"));
        assert_eq!(public.system_id, "r.dtd");

        // A `PublicID` without a system identifier names nothing to retrieve, and neither does a
        // declaration with an internal subset and no external identifier at all.
        assert!(external_subset_id(r#"r PUBLIC "-//x//DTD R//EN""#).is_none());
        assert!(external_subset_id("svg[<!ELEMENT svg EMPTY>]").is_none());
        assert!(external_subset_id("r").is_none());
    }

    #[test]
    fn element_declarations_record_which_models_are_element_content() {
        let subset = scan(
            "doc [<!ELEMENT chapter (title, para*)><!ELEMENT title (#PCDATA)>\
             <!ELEMENT para (#PCDATA|em)*><!ELEMENT em EMPTY><!ELEMENT note ANY>\
             <!ELEMENT list ( item )+ >]",
        );
        let models: Vec<(&str, ContentModel)> = subset
            .content_models
            .iter()
            .map(|(name, model)| (name.as_str(), *model))
            .collect();
        assert_eq!(
            models,
            [
                ("chapter", ContentModel::ElementOnly),
                ("em", ContentModel::Other),
                ("list", ContentModel::ElementOnly),
                ("note", ContentModel::Other),
                ("para", ContentModel::Other),
                ("title", ContentModel::Other),
            ]
        );
        assert!(ContentModel::ElementOnly.is_element_only());
        assert!(!ContentModel::Other.is_element_only());
    }

    #[test]
    fn a_model_this_scanner_cannot_read_is_recorded_as_nothing() {
        // A content specification assembled out of parameter entities could be either kind, and
        // nothing here expands one.
        let subset = scan("doc [<!ELEMENT a (%inner;)><!ELEMENT b %model;><!ELEMENT c>]");
        assert!(
            subset.content_models.is_empty(),
            "{:?}",
            subset.content_models
        );
    }

    #[test]
    fn the_first_element_declaration_binds() {
        // XML 1.0 VC: Unique Element Type Declaration — a non-validating processor reports the
        // duplicate by keeping the first, as it does for entities and attributes.
        let subset = scan("doc [<!ELEMENT e (f*)><!ELEMENT e (#PCDATA)>]");
        assert_eq!(
            subset.content_models.get("e"),
            Some(&ContentModel::ElementOnly)
        );
    }

    #[test]
    fn an_ignored_section_declares_nothing() {
        let mut subset = InternalSubset::default();
        scan_declarations(
            &mut subset,
            r#"<![ INCLUDE [ <!ENTITY kept "yes"> ]]>
               <![IGNORE[ <!ENTITY dropped "no">
                 <![INCLUDE[ <!ENTITY nested "no"> ]]>
               ]]>
               <!ENTITY after "yes">"#,
            XmlVersion::Implicit1_0,
        );
        let mut names: Vec<&str> = subset.entities.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(names, ["after", "kept"]);
    }
}
