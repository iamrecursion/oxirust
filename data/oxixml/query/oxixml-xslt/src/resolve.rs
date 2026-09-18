//! [`Resolver`] — how a stylesheet reaches outside itself.
//!
//! `xsl:include`, `xsl:import`, `document()`, `fn:doc()` and `fn:unparsed-text()` all need
//! something the library cannot invent: the bytes behind a URI. A Pure-Rust library does not
//! read the filesystem or the network on its caller's behalf, so nothing is reachable until an
//! application installs a resolver.
//!
//! Two are supplied: [`MemoryResolver`], which serves documents an application registered by
//! URI, and [`FileResolver`], which an application can opt into to read from a directory.

use oxixml_infoset::{NodeRef, ParseOptions, Tree, XdmNode};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Supplies documents and text to a transformation.
///
/// # Examples
///
/// ```
/// use oxixml_xslt::{MemoryResolver, Resolver};
///
/// let resolver = MemoryResolver::new().with_document("urn:x", "<a/>");
/// assert!(resolver.document("urn:x").is_some());
/// assert!(resolver.document("urn:missing").is_none());
/// ```
pub trait Resolver: fmt::Debug + Send + Sync {
    /// The document `uri` names, already parsed.
    fn document(&self, uri: &str) -> Option<NodeRef>;

    /// The stylesheet module `uri` names. Defaults to [`Resolver::document`].
    fn stylesheet(&self, uri: &str) -> Option<NodeRef> {
        self.document(uri)
    }

    /// The text `uri` names, for `fn:unparsed-text`.
    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        let _ = (uri, encoding);
        None
    }

    /// The nodes `fn:collection(uri)` returns.
    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        let _ = uri;
        None
    }

    /// Resolves `href` against `base`, which the default implementation does with
    /// [`oxixml_iri`].
    fn resolve_uri(&self, href: &str, base: Option<&str>) -> String {
        resolve_uri(href, base)
    }
}

/// Resolves `href` against `base`, falling back to `href` when either is not a legal IRI.
#[must_use]
pub fn resolve_uri(href: &str, base: Option<&str>) -> String {
    let Some(base) = base else {
        return href.to_owned();
    };
    let Ok(base) = oxixml_iri::Iri::parse(base.to_owned()) else {
        return href.to_owned();
    };
    let Ok(reference) = oxixml_iri::IriRef::parse(href.to_owned()) else {
        return href.to_owned();
    };
    match base.resolve(&reference) {
        Ok(resolved) => resolved.into_inner(),
        Err(_) => href.to_owned(),
    }
}

/// A resolver serving documents an application registered up front.
#[derive(Debug, Default)]
pub struct MemoryResolver {
    documents: Mutex<HashMap<String, Tree>>,
    texts: Mutex<HashMap<String, String>>,
}

impl MemoryResolver {
    /// An empty resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `xml` under `uri`, ignoring a document that does not parse.
    #[must_use]
    pub fn with_document(self, uri: impl Into<String>, xml: &str) -> Self {
        let uri = uri.into();
        if let Ok(tree) =
            Tree::parse_str_with(xml, ParseOptions::default().document_uri(uri.clone()))
            && let Ok(mut documents) = self.documents.lock()
        {
            documents.insert(uri, tree);
        }
        self
    }

    /// Registers `text` under `uri`, for `fn:unparsed-text`.
    #[must_use]
    pub fn with_text(self, uri: impl Into<String>, text: impl Into<String>) -> Self {
        if let Ok(mut texts) = self.texts.lock() {
            texts.insert(uri.into(), text.into());
        }
        self
    }

    /// Registers an already-parsed tree.
    pub fn insert_tree(&self, uri: impl Into<String>, tree: Tree) {
        if let Ok(mut documents) = self.documents.lock() {
            documents.insert(uri.into(), tree);
        }
    }
}

impl Resolver for MemoryResolver {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        let documents = self.documents.lock().ok()?;
        documents.get(uri).map(Tree::root)
    }

    fn unparsed_text(&self, uri: &str, _encoding: Option<&str>) -> Option<String> {
        let texts = self.texts.lock().ok()?;
        texts.get(uri).cloned()
    }
}

/// A resolver that reads from a directory on the local filesystem.
///
/// An application opts into filesystem access by constructing one; nothing in this crate does
/// so on its own.
#[derive(Debug)]
pub struct FileResolver {
    root: PathBuf,
    cache: Mutex<HashMap<String, Option<Tree>>>,
}

impl FileResolver {
    /// A resolver rooted at `directory`. A `file:` URI or a relative path is resolved beneath
    /// it; an absolute path outside it is rejected.
    #[must_use]
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            root: directory.into(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn path_for(&self, uri: &str) -> Option<PathBuf> {
        let trimmed = uri.strip_prefix("file://").unwrap_or(uri);
        let trimmed = trimmed.split(['#', '?']).next().unwrap_or(trimmed);
        let decoded = percent_decode(trimmed);
        let candidate = Path::new(&decoded);
        let path = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.root.join(candidate)
        };
        path.exists().then_some(path)
    }

    fn tree_for(&self, uri: &str) -> Option<Tree> {
        if let Ok(cache) = self.cache.lock()
            && let Some(entry) = cache.get(uri)
        {
            return entry.clone();
        }
        let parsed = self.path_for(uri).and_then(|path| {
            let text = std::fs::read_to_string(&path).ok()?;
            Tree::parse_str_with(
                &text,
                ParseOptions::default().document_uri(format!("file://{}", path.display())),
            )
            .ok()
        });
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(uri.to_owned(), parsed.clone());
        }
        parsed
    }
}

impl Resolver for FileResolver {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        self.tree_for(uri).map(|tree| tree.root())
    }

    fn unparsed_text(&self, uri: &str, _encoding: Option<&str>) -> Option<String> {
        let path = self.path_for(uri)?;
        std::fs::read_to_string(path).ok()
    }
}

/// Undoes the percent-encoding a `file:` URI applies to a path.
fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = (bytes[index + 1] as char).to_digit(16);
            let low = (bytes[index + 2] as char).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                out.push((high * 16 + low) as u8);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Applies the stylesheet's whitespace-stripping rules to every source document a
/// transformation loads, not only to the initial one.
///
/// [XSLT 3.0 §4.4](https://www.w3.org/TR/xslt-30/#strip) scopes `xsl:strip-space` to "all
/// documents in the input", which includes the trees `fn:doc`, `fn:document` and
/// `fn:collection` return. The stripped copy is cached per URI so that two references to the
/// same document keep node identity — `doc('x') is doc('x')` has to hold.
pub(crate) struct DocumentStripper {
    ir: Arc<crate::compile::StylesheetIr>,
    cache: Mutex<HashMap<String, NodeRef>>,
}

impl fmt::Debug for DocumentStripper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocumentStripper").finish_non_exhaustive()
    }
}

impl DocumentStripper {
    pub(crate) fn new(ir: Arc<crate::compile::StylesheetIr>) -> Self {
        Self {
            ir,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The stripped counterpart of `node`, or `node` itself when the stylesheet strips
    /// nothing.
    pub(crate) fn apply(&self, uri: &str, node: NodeRef) -> NodeRef {
        if !self.ir.strips_any_space {
            return node;
        }
        if let Ok(cache) = self.cache.lock()
            && let Some(found) = cache.get(uri)
        {
            return found.clone();
        }
        let Ok(tree) = crate::run::strip_whitespace(&self.ir, &node.root()) else {
            return node;
        };
        let stripped = tree.root();
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(uri.to_owned(), stripped.clone());
        }
        stripped
    }
}

/// Adapts a [`Resolver`] to the XPath engine's document resolver.
#[derive(Debug)]
pub(crate) struct XPathResolver {
    pub(crate) inner: Arc<dyn Resolver>,
    pub(crate) base: Option<String>,
    pub(crate) stripper: Arc<DocumentStripper>,
}

impl oxixml_xpath::DocumentResolver for XPathResolver {
    fn document(&self, uri: &str) -> Option<NodeRef> {
        let resolved = self.inner.resolve_uri(uri, self.base.as_deref());
        let found = self
            .inner
            .document(&resolved)
            .or_else(|| self.inner.document(uri))?;
        Some(self.stripper.apply(&resolved, found))
    }

    fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
        self.inner.collection(uri)
    }

    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        let resolved = self.inner.resolve_uri(uri, self.base.as_deref());
        self.inner
            .unparsed_text(&resolved, encoding)
            .or_else(|| self.inner.unparsed_text(uri, encoding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memory_resolver_serves_what_it_was_given() {
        let resolver = MemoryResolver::new()
            .with_document("urn:a", "<a/>")
            .with_text("urn:t", "hello");
        assert!(resolver.document("urn:a").is_some());
        assert_eq!(
            resolver.unparsed_text("urn:t", None).as_deref(),
            Some("hello")
        );
        assert!(resolver.document("urn:b").is_none());
    }

    #[test]
    fn relative_references_resolve_against_the_base() {
        assert_eq!(
            resolve_uri("b.xml", Some("http://example.com/a/c.xml")),
            "http://example.com/a/b.xml"
        );
        assert_eq!(resolve_uri("b.xml", None), "b.xml");
    }

    #[test]
    fn percent_escapes_are_decoded() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("a%zzb"), "a%zzb");
    }
}
