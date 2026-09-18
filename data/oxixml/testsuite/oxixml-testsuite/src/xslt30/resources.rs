//! The external resources one XSLT 3.0 test case can reach: the documents `fn:doc`,
//! `fn:document` and `xsl:include` retrieve, the collections `fn:collection` enumerates, and the
//! external subset and general entities the documents themselves declare.
//!
//! The engine has no filesystem access of its own — retrieval is the application's to install —
//! so everything a case can reach is what this module hands it. Two kinds of resource are
//! declared, and they are declared differently:
//!
//! * a *file beside the test set*, named by a relative reference the stylesheet writes
//!   (`doc('mode-14.xml')`), which is retrieved from disk on demand;
//! * a resource the `<environment>` **registers** — a `<source file="…" uri="…"/>` that is not
//!   the principal source, or a `<collection uri="…">` and its members — which is bound to the URI
//!   the catalogue names it under, whatever it is called on disk.
//!
//! Both are served by [`SuiteResolver`], whose retrievals are parsed exactly as the principal
//! source document is: the announced encoding, the external subset of the document type
//! declaration, and the general entities that subset declares. A document reached through `doc()`
//! is the same document reached through `<source role=".">`, and the `base-uri` set says so in as
//! many words — `base-uri-051` supplies its document as the source, `-051a` retrieves the very
//! same file with `doc()`, and both assert the base URIs of elements that only exist because an
//! external entity was read.
//!
//! Two resources are neither of those kinds, and each has a function of its own here. A source
//! document the catalogue marks `xinclude="true"` is **merged** before the transformation sees it
//! ([`expand_inclusions`]); and the principal *stylesheet module* is read here too
//! ([`parse_stylesheet_module`]), because a module is an XML document like any other and one of
//! them declares an external general entity of its own.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use oxixml_infoset::{NodeRef, ParseOptions, Tree, XdmNode};
use oxixml_xinclude::{LoadError, ResourceLoader};
use oxixml_xslt::{FileResolver, Resolver};

use super::{file_uri, read_xml_file};

/// Retrieves the external subset of a source document's document type declaration — and any
/// external general entity it declares — from the files beside the test set.
///
/// Wrapping [`oxixml_infoset::FileEntityResolver`] rather than using it directly turns a
/// retrieval *failure* into a decline. The catalogue holds documents whose `DOCTYPE` names a DTD
/// that is not in the checkout at all (`http://docbook.org/xml/4.4/docbookx.dtd`), and [XML 1.0
/// §5.1](https://www.w3.org/TR/xml/#proc-types) lets a non-validating processor not read the
/// external subset, so a subset that cannot be retrieved leaves the document exactly as it was
/// without one instead of rejecting a case the catalogue expects to run.
pub(super) struct SuiteEntities {
    files: oxixml_infoset::FileEntityResolver,
}

impl SuiteEntities {
    pub(super) fn new(directory: &Path) -> Self {
        Self {
            files: oxixml_infoset::FileEntityResolver::new(directory.to_path_buf()),
        }
    }
}

impl oxixml_infoset::ExternalEntityResolver for SuiteEntities {
    fn resolve(
        &mut self,
        entity: &oxixml_infoset::ExternalEntity<'_>,
    ) -> Result<Option<String>, oxixml_infoset::ExternalEntityError> {
        Ok(self.files.resolve(entity).unwrap_or_default())
    }
}

/// The resources an `xi:include` can reach: the files beside the test set, under the `file:` URIs
/// the including document resolves its `href` references to.
///
/// The crate's own default posture is [`NullLoader`](oxixml_xinclude::NullLoader), which resolves
/// nothing; retrieval is the application's to install, exactly as it is for [`SuiteResolver`].
struct SuiteFiles;

impl ResourceLoader for SuiteFiles {
    fn load(
        &self,
        iri: &str,
        _accept: Option<&str>,
        _accept_language: Option<&str>,
    ) -> Result<Vec<u8>, LoadError> {
        let path = iri.strip_prefix("file://").ok_or(LoadError::NotFound)?;
        std::fs::read(PathBuf::from(percent_decode(path))).map_err(|_| LoadError::NotFound)
    }
}

/// The text of `document` with every `xi:include` it holds replaced by what it points at, or
/// `None` when the document is not one this can be done to.
///
/// A `<source role="." xinclude="true">` asks for [XInclude
/// 1.0](https://www.w3.org/TR/xinclude/) processing *before* the transformation starts, so the
/// document the stylesheet sees is the merged one — and, because
/// [§4.5.5](https://www.w3.org/TR/xinclude/#base_uri_fixup)'s base-URI fixup writes an explicit
/// `xml:base` on every top-level included item, each grafted element keeps resolving relative
/// references against the file it was acquired from rather than against the including document.
/// That fixup is the whole point of the one case in the catalogue that asks for this.
///
/// The merged document is handed back as *text* rather than as a tree because the processing is
/// [`oxixml_dom`](oxixml_dom)'s and everything downstream of it is
/// [`oxixml_infoset`](oxixml_infoset)'s: re-reading the text is what puts the result in the data
/// model the engines consume, under the same reading (`document_uri`, external subset, entities)
/// every other source document gets. What the round trip cannot carry is the document type
/// declaration, since the merged document is serialized without one — a limitation with no reach
/// here, because the only source document the catalogue XIncludes declares none.
pub(super) fn expand_inclusions(text: &str, base_uri: &str) -> Option<String> {
    let mut document = oxixml_dom::build::parse_str(text).ok()?;
    oxixml_xinclude::process_document(&mut document, base_uri, &SuiteFiles).ok()?;
    Some(document.to_xml())
}

/// One member of a catalogue `<collection>`.
///
/// A member is usually a whole document, but it need not be: the catalogue writes
/// `<source file="doc15.xml#frag2" uri="doc15.xml#frag2"/>`, and [F&O 3.1
/// §14.7.3](https://www.w3.org/TR/xpath-functions-31/#func-collection) maps a collection URI to a
/// sequence of *items*, so a member with a fragment identifier is the one element that fragment
/// names rather than the document containing it. `collection-004` copies such a collection and
/// expects exactly `<a xml:id="frag2"/>` where that member is.
#[derive(Debug, Clone)]
pub(super) struct CollectionMember {
    /// The file the member is read from, with any fragment identifier already removed.
    path: PathBuf,
    /// The fragment identifier that narrows the document to one element, when the catalogue
    /// wrote one.
    fragment: Option<String>,
}

impl CollectionMember {
    /// The member a `<source file="…"/>` inside a `<collection>` declares, resolved against the
    /// test set's own directory.
    pub(super) fn new(file: &str, directory: &Path) -> Self {
        match file.split_once('#') {
            Some((path, fragment)) => Self {
                path: directory.join(path),
                fragment: Some(fragment.to_owned()),
            },
            None => Self {
                path: directory.join(file),
                fragment: None,
            },
        }
    }
}

/// The documents and collections one test case can retrieve.
///
/// The `<environment>`'s registrations are consulted first, then the test set's own directory —
/// the order the catalogue assumes, since a registration binds a URI to a file that need not be
/// named after it.
#[derive(Debug)]
pub(super) struct SuiteResolver {
    /// The stylesheet modules `xsl:include`/`xsl:import` retrieve, left to the engine's own file
    /// resolver: a module is read as a stylesheet, not as a source document, and nothing about
    /// the two rules below applies to one.
    modules: FileResolver,
    /// The test set's directory, which a relative reference is resolved against.
    directory: PathBuf,
    /// Documents the environment registered under a URI of the catalogue's choosing.
    documents: BTreeMap<String, PathBuf>,
    /// Collections the environment declared, by the URI they answer to; the empty string is the
    /// default collection, the one `fn:collection()` with no argument asks for.
    collections: BTreeMap<String, Vec<CollectionMember>>,
    /// The trees already built, by the URI they were asked for under.
    ///
    /// A cache here is a correctness requirement rather than an optimization: `fn:doc` must
    /// answer the *same* node for the same URI throughout a transformation, so that
    /// `doc('x') is doc('x')` holds and a collection member and the document behind it are one
    /// node. A URI that names nothing is cached as such, so a repeated miss stays one stat call.
    trees: Mutex<HashMap<String, Option<Tree>>>,
}

impl SuiteResolver {
    /// A resolver over `directory`, with the documents and collections `environment` registered.
    pub(super) fn new(
        directory: &Path,
        documents: BTreeMap<String, PathBuf>,
        collections: BTreeMap<String, Vec<CollectionMember>>,
    ) -> Self {
        Self {
            modules: FileResolver::new(directory.to_path_buf()),
            directory: directory.to_path_buf(),
            documents,
            collections,
            trees: Mutex::new(HashMap::new()),
        }
    }

    /// The file `uri` names: the one registered under it, or the one it names below the test
    /// set's directory. `None` when nothing of that name is there.
    fn path_for(&self, uri: &str) -> Option<PathBuf> {
        if let Some(path) = self.documents.get(uri) {
            return path.exists().then(|| path.clone());
        }
        let trimmed = uri.strip_prefix("file://").unwrap_or(uri);
        let trimmed = trimmed.split(['#', '?']).next().unwrap_or(trimmed);
        let decoded = percent_decode(trimmed);
        // A registration may name the file with the very reference the stylesheet writes, in
        // which case the URI the engine hands over has already been resolved against the
        // stylesheet's base URI and no longer matches it. The decoded form is tried too.
        if let Some(path) = self.documents.get(decoded.as_str()) {
            return path.exists().then(|| path.clone());
        }
        let candidate = Path::new(&decoded);
        let path = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.directory.join(candidate)
        };
        path.exists().then_some(path)
    }

    /// The tree `uri` names, built at most once per URI. See [`SuiteResolver::trees`].
    fn tree_for(&self, uri: &str) -> Option<Tree> {
        if let Ok(trees) = self.trees.lock()
            && let Some(built) = trees.get(uri)
        {
            return built.clone();
        }
        let built = self
            .path_for(uri)
            .and_then(|path| parse_document(&path, uri, &self.directory));
        if let Ok(mut trees) = self.trees.lock() {
            trees.insert(uri.to_owned(), built.clone());
        }
        built
    }

    /// The node one collection member contributes: the document, or the element its fragment
    /// identifier names inside it.
    fn member_node(&self, member: &CollectionMember) -> Option<NodeRef> {
        let root = self.tree_for(&file_uri(&member.path))?.root();
        match &member.fragment {
            None => Some(root),
            Some(fragment) => element_with_xml_id(&root, fragment),
        }
    }

    /// The collection `uri` names, whether or not it has any members.
    ///
    /// The difference between an *empty* collection and one that was never declared is the whole
    /// of `collection-003`, which asserts that the default collection exists (`exists(/result)`)
    /// and holds nothing (`empty(/result/node())`); an undeclared collection is a resource that
    /// cannot be retrieved, and `fn:collection` reports `FODC0002` for it.
    fn members(&self, uri: Option<&str>) -> Option<&Vec<CollectionMember>> {
        let key = uri.unwrap_or_default();
        if let Some(members) = self.collections.get(key) {
            return Some(members);
        }
        // The engine offers the reference as written before resolving it, but a stylesheet may
        // also write one that is already absolute; a registration keyed on the relative name
        // still answers it.
        let trimmed = key.strip_prefix("file://").unwrap_or(key);
        let relative = trimmed.strip_prefix(self.directory.to_str()?)?;
        self.collections.get(relative.trim_start_matches('/'))
    }
}

impl Resolver for SuiteResolver {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        self.tree_for(uri).map(|tree| tree.root())
    }

    /// A stylesheet module is retrieved the way the engine's own file resolver retrieves one.
    fn stylesheet(&self, uri: &str) -> Option<NodeRef> {
        self.modules.stylesheet(uri)
    }

    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        self.modules.unparsed_text(uri, encoding)
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        let members = self.members(uri)?;
        Some(
            members
                .iter()
                .filter_map(|member| self.member_node(member))
                .collect(),
        )
    }
}

/// Parses one retrieved document exactly as the principal source document is parsed: the encoding
/// it announces, the external subset of its document type declaration, and the general entities
/// that subset declares, under the URI it was retrieved from.
///
/// `fallback_base` is the directory a relative system identifier is resolved against for a
/// document that has no directory of its own to offer. It is only ever a fallback: a system
/// identifier in a document type declaration is relative to the document that declares it
/// ([XML 1.0 §4.2.2](https://www.w3.org/TR/xml/#sec-external-ent)), which is not always the test
/// set's own directory — a case that walks the catalogue retrieves stylesheet modules from every
/// other test set, and the files *their* declarations name sit beside them rather than beside the
/// walking case. [`parse_stylesheet_module`] resolves them the same way, for the same reason.
fn parse_document(path: &Path, uri: &str, fallback_base: &Path) -> Option<Tree> {
    let text = read_xml_file(path)?;
    let options = ParseOptions::default()
        .document_uri(uri.to_owned())
        .read_external_subset(true);
    let mut entities = SuiteEntities::new(path.parent().unwrap_or(fallback_base));
    Tree::parse_str_with_resolver(&text, options, &mut entities).ok()
}

/// Parses one stylesheet module the way this module parses a source document, so that the two
/// readings agree.
///
/// [`oxixml_xslt::Stylesheet::compile_str_with`] parses the text itself, with the compiler's own
/// options and no entity resolver — which is exactly right for an application that has none to
/// give, and wrong for this one: a stylesheet module is an XML document like any other, and one
/// of them (`use-when-0136.xsl`) declares an external general entity in its internal subset and
/// references it at the top level of `xsl:stylesheet`. Retrieval has to happen while the
/// *module* is parsed, so the module is parsed here and
/// [`oxixml_xslt::Stylesheet::compile_with`] is handed the tree.
///
/// The two properties the compiler's own parse sets are kept, because a module compiled from a
/// caller's tree carries only what that caller asked for: source positions, which an `xsl:catch`
/// body reads back as `$err:line-number`, and the document URI, which is the module's base URI
/// and what an external entity's system identifier is resolved against.
///
/// `None` when the text is not well-formed — the caller then hands the *text* to the compiler,
/// whose own parse reports the error under the code the catalogue grades (`XTSE0165`), which no
/// application can construct on the engine's behalf.
pub(super) fn parse_stylesheet_module(text: &str, path: &Path, directory: &Path) -> Option<Tree> {
    let options = ParseOptions::default()
        .record_source_positions(true)
        .document_uri(file_uri(path));
    // A system identifier in a module is relative to the module, which is not always the test
    // set's own directory: the docbook set's principal modules are two directories below it.
    let mut entities = SuiteEntities::new(path.parent().unwrap_or(directory));
    Tree::parse_str_with_resolver(text, options, &mut entities).ok()
}

/// The element inside `root` whose `xml:id` is `id`.
///
/// [xml:id §4](https://www.w3.org/TR/xml-id/#processing) makes the attribute an identifier
/// without any need for a document type declaration, which is exactly why the catalogue uses it
/// here: `collection-004` was rewritten in 2021 to name its fragment with `xml:id` rather than
/// with a DTD-declared `ID` attribute, "to remove dependency".
fn element_with_xml_id(root: &NodeRef, id: &str) -> Option<NodeRef> {
    /// The namespace `xml:` is bound to, which no document may rebind.
    const XML_NAMESPACE: &str = "http://www.w3.org/XML/1998/namespace";
    if root.attributes().into_iter().any(|attribute| {
        attribute.node_name().is_some_and(|name| {
            name.namespace() == Some(XML_NAMESPACE) && name.local_name() == "id"
        }) && attribute.string_value() == id
    }) {
        return Some(root.clone());
    }
    root.children()
        .into_iter()
        .find_map(|child| element_with_xml_id(&child, id))
}

/// Undoes the percent-encoding a `file:` URI applies to a path, leaving anything that is not an
/// escape exactly as it was written.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && let Some(hex) = value.get(index + 1..index + 3)
            && hex.bytes().all(|digit| digit.is_ascii_hexdigit())
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            index += 3;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The documents `fn:doc` reaches while the harness evaluates a catalogue `select` expression:
/// whatever the test-set directory holds under the name the expression writes.
///
/// The engine has no filesystem access of its own — a document resolver is the application's to
/// install — and a catalogue selector such as `<initial-mode select="doc('mode-14.xml')//v"/>`
/// names a fixture beside the test set, so this is the mapping the catalogue assumes. It reads
/// the same documents [`SuiteResolver`] does, and reads them the same way: a selector and a
/// stylesheet naming one file must see one document.
#[derive(Debug)]
pub(super) struct SuiteDocuments {
    pub(super) directory: PathBuf,
}

impl oxixml_xpath::DocumentResolver for SuiteDocuments {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        let path = match uri.strip_prefix("file://") {
            Some(absolute) => PathBuf::from(percent_decode(absolute)),
            None => self.directory.join(percent_decode(uri)),
        };
        parse_document(&path, uri, &self.directory).map(|tree| tree.root())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document retrieved by URI is parsed like the principal source document: the external
    /// subset is read, so the `<!ATTLIST>` defaults it declares are on the tree that comes back.
    #[test]
    fn a_retrieved_document_reads_its_external_subset() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-xslt30-resources-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(
            directory.join("defaults.dtd"),
            "<!ELEMENT doc (#PCDATA)>\n<!ATTLIST doc kind CDATA \"declared\">\n",
        )
        .expect("fixture subset");
        std::fs::write(
            directory.join("doc.xml"),
            "<!DOCTYPE doc SYSTEM \"defaults.dtd\"><doc/>",
        )
        .expect("fixture document");

        let resolver = SuiteResolver::new(&directory, BTreeMap::new(), BTreeMap::new());
        let root = resolver
            .document(&file_uri(&directory.join("doc.xml")))
            .expect("the document is retrievable");
        let element = root
            .children()
            .into_iter()
            .next()
            .expect("a document element");
        assert_eq!(
            element
                .attributes()
                .into_iter()
                .map(|attribute| attribute.string_value())
                .collect::<Vec<_>>(),
            vec!["declared".to_owned()],
            "the default declared in the external subset reaches the tree"
        );

        // The same URI answers the same node, which is what `doc('x') is doc('x')` requires.
        let again = resolver
            .document(&file_uri(&directory.join("doc.xml")))
            .expect("the document is retrievable");
        assert!(root.is_same_node(&again));
        std::fs::remove_dir_all(&directory).ok();
    }

    /// A stylesheet module is read with the same entity resolver a source document is: an
    /// external general entity its *internal subset* declares is retrieved while the module is
    /// parsed, and the nodes its replacement text contributes carry the entity's own base URI
    /// ([XML 1.0 §4.2.2](https://www.w3.org/TR/xml/#sec-external-ent)).
    ///
    /// This is the shape `use-when-0136.xsl` has: `<!ENTITY child SYSTEM "dir/…ent">` referenced
    /// as `&child;` at the top level of `xsl:stylesheet`. Handing the module's *text* to the
    /// compiler cannot answer it, because the compiler parses with no resolver and the reference
    /// is then an undeclared entity — a document that is not well-formed.
    #[test]
    fn a_stylesheet_module_retrieves_the_entities_it_declares() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-xslt30-module-entities-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(directory.join("dir")).expect("scratch directory");
        std::fs::write(
            directory.join("dir/declared.ent"),
            r#"<xsl:variable xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                             name="sbu" select="static-base-uri()" static="yes"/>"#,
        )
        .expect("fixture entity");
        let module = directory.join("module.xsl");
        let text = concat!(
            "<!DOCTYPE xsl:stylesheet [<!ENTITY child SYSTEM \"dir/declared.ent\">]>",
            "<xsl:stylesheet xmlns:xsl=\"http://www.w3.org/1999/XSL/Transform\" version=\"3.0\">",
            "&child;</xsl:stylesheet>"
        );
        std::fs::write(&module, text).expect("fixture module");

        let tree = parse_stylesheet_module(text, &module, &directory)
            .expect("the module parses once its entity can be retrieved");
        let stylesheet = tree.document_element().expect("a document element");
        let variable = stylesheet
            .children()
            .into_iter()
            .find(|child| child.node_kind() == oxixml_infoset::NodeKind::Element)
            .expect("the entity contributed an element");
        assert_eq!(
            variable
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default(),
            "variable"
        );
        let base = variable.base_uri().unwrap_or_default();
        assert!(
            base.ends_with("dir/declared.ent"),
            "the entity's content keeps the entity's own base URI, got `{base}`"
        );

        // Without a resolver the same text is not a document at all, which is what the compiler's
        // own parse sees and why the retrieval has to happen here.
        assert!(Tree::parse_str(text).is_err());
        std::fs::remove_dir_all(&directory).ok();
    }

    /// A document the catalogue marks `xinclude="true"` is merged before the transformation sees
    /// it, and [XInclude 1.0 §4.5.5](https://www.w3.org/TR/xinclude/#base_uri_fixup)'s base-URI
    /// fixup is what makes the merge observable: each top-level included item carries an explicit
    /// `xml:base`, so an element keeps the base URI of the file it came from.
    #[test]
    fn an_xincluded_source_keeps_the_base_uri_of_each_included_file() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-xslt30-xinclude-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(directory.join("dir")).expect("scratch directory");
        std::fs::write(directory.join("dir/part.xml"), "<para>included</para>")
            .expect("fixture part");
        let including = directory.join("doc.xml");
        let text = concat!(
            "<doc><xi:include href=\"dir/part.xml\" ",
            "xmlns:xi=\"http://www.w3.org/2001/XInclude\"/></doc>"
        );
        std::fs::write(&including, text).expect("fixture document");

        let merged = expand_inclusions(text, &file_uri(&including)).expect("the document merges");
        let tree = Tree::parse_str_with(
            &merged,
            ParseOptions::default().document_uri(file_uri(&including)),
        )
        .expect("the merged document parses");
        let paragraph = tree
            .document_element()
            .expect("a document element")
            .children()
            .into_iter()
            .find(|child| child.node_kind() == oxixml_infoset::NodeKind::Element)
            .expect("the inclusion was expanded");
        assert_eq!(paragraph.string_value(), "included");
        let base = paragraph.base_uri().unwrap_or_default();
        assert!(
            base.ends_with("dir/part.xml"),
            "an included element keeps its own file's base URI, got `{base}`"
        );

        // A resource error with no `xi:fallback` beside it aborts the processing (XInclude 1.0
        // [§4.4](https://www.w3.org/TR/xinclude/#fallback)), so there is no merged document to
        // hand back and the caller keeps the document exactly as it was read.
        let missing = concat!(
            "<doc><xi:include href=\"dir/absent.xml\" ",
            "xmlns:xi=\"http://www.w3.org/2001/XInclude\"/></doc>"
        );
        assert!(expand_inclusions(missing, &file_uri(&including)).is_none());
        std::fs::remove_dir_all(&directory).ok();
    }

    /// A collection the environment declared with no members is not the same as one it never
    /// declared: the first is an empty sequence, the second is a resource that cannot be
    /// retrieved. `collection-003` decides the difference.
    #[test]
    fn a_declared_collection_answers_even_when_it_is_empty() {
        let directory = std::env::temp_dir().join(format!(
            "oxixml-xslt30-collections-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("scratch directory");
        std::fs::write(directory.join("m.xml"), "<doc><a xml:id=\"f\"/></doc>")
            .expect("fixture member");

        let mut collections = BTreeMap::new();
        collections.insert(String::new(), Vec::new());
        collections.insert(
            "named".to_owned(),
            vec![
                CollectionMember::new("m.xml", &directory),
                CollectionMember::new("m.xml#f", &directory),
            ],
        );
        let resolver = SuiteResolver::new(&directory, BTreeMap::new(), collections);

        assert_eq!(
            resolver.collection(None).map(|members| members.len()),
            Some(0),
            "the default collection was declared empty"
        );
        assert_eq!(
            resolver.collection(Some("absent")),
            None,
            "an undeclared collection is not retrievable"
        );

        let members = resolver
            .collection(Some("named"))
            .expect("the named collection");
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].node_kind(), oxixml_infoset::NodeKind::Document);
        assert_eq!(
            members[1]
                .node_name()
                .map(|name| name.local_name().to_owned()),
            Some("a".to_owned()),
            "a member with a fragment identifier is the element it names"
        );
        std::fs::remove_dir_all(&directory).ok();
    }
}
