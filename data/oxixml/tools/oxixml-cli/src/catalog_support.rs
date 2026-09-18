//! Wires `--catalog` (OASIS XML Catalog resolution, via [`oxixml_catalog`]) into the handful of
//! places this workspace lets an application plug in its own resource retrieval: XInclude's
//! [`ResourceLoader`](oxixml_xinclude::ResourceLoader), RELAX NG's
//! [`SchemaResolver`](oxixml_relaxng::SchemaResolver), XML Schema's
//! [`SchemaResolver`](oxixml_schema::SchemaResolver), and XSLT's
//! [`Resolver`](oxixml_xslt::Resolver).
//!
//! Every adapter here follows the same shape: consult the catalog (if one was loaded) for a
//! rewritten URI, then read *that* — or, when there is no catalog or no matching entry, the
//! original URI or path — from the local filesystem. None of this crate's dependencies perform
//! network access, so a `http:`/`https:` catalog entry can be *matched* but not *fetched*; only
//! `file:` URIs and plain paths ever actually resolve to bytes.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use oxixml_catalog::Catalog;

use crate::io_util;

/// Reads and parses the OASIS XML Catalog document at `path`.
///
/// # Errors
///
/// A human-readable message when the file cannot be read or is not a well-formed catalog
/// document.
pub fn load(path: &str, stdin: &mut dyn Read) -> Result<Catalog, String> {
    let text = io_util::read_string(path, stdin)
        .map_err(|error| format!("cannot read catalog `{path}`: {error}"))?;
    let base = io_util::file_base_uri(path).unwrap_or_else(|| "file:///catalog.xml".to_owned());
    Catalog::parse(&text, &base).map_err(|error| format!("invalid catalog `{path}`: {error}"))
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

/// Turns a `file:` URI or a plain (possibly relative) path into a filesystem path, joining a
/// relative one against `base_dir`.
fn uri_to_path(uri: &str, base_dir: &Path) -> PathBuf {
    let trimmed = uri.strip_prefix("file://").unwrap_or(uri);
    let trimmed = trimmed.split(['#', '?']).next().unwrap_or(trimmed);
    let decoded = percent_decode(trimmed);
    let candidate = Path::new(&decoded);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base_dir.join(candidate)
    }
}

/// Reads the bytes `uri` names: rewritten through `catalog` first when one is given and it knows
/// `uri`, otherwise `uri` itself, resolved as a `file:` URI or a path relative to `base_dir`.
///
/// # Errors
///
/// Any [`io::Error`] the underlying read raises.
fn read_via_catalog(
    catalog: Option<&Catalog>,
    uri: &str,
    base_dir: &Path,
) -> io::Result<(PathBuf, Vec<u8>)> {
    let rewritten = catalog.and_then(|catalog| catalog.resolve_uri(uri));
    let target = rewritten.as_deref().unwrap_or(uri);
    let path = uri_to_path(target, base_dir);
    let bytes = fs::read(&path)?;
    Ok((path, bytes))
}

/// An [`oxixml_xinclude::ResourceLoader`] that reads `file:` URIs (and plain paths) from the
/// local filesystem, first offering each one to an optional catalog for rewriting.
#[derive(Debug, Clone, Default)]
pub struct XIncludeLoader {
    catalog: Option<Catalog>,
}

impl XIncludeLoader {
    /// Builds a loader, with or without a catalog.
    #[must_use]
    pub const fn new(catalog: Option<Catalog>) -> Self {
        Self { catalog }
    }
}

impl oxixml_xinclude::ResourceLoader for XIncludeLoader {
    fn load(
        &self,
        iri: &str,
        _accept: Option<&str>,
        _accept_language: Option<&str>,
    ) -> Result<Vec<u8>, oxixml_xinclude::LoadError> {
        read_via_catalog(self.catalog.as_ref(), iri, Path::new("."))
            .map(|(_, bytes)| bytes)
            .map_err(|error| match error.kind() {
                io::ErrorKind::NotFound => oxixml_xinclude::LoadError::NotFound,
                io::ErrorKind::PermissionDenied => oxixml_xinclude::LoadError::PermissionDenied,
                _ => oxixml_xinclude::LoadError::Other(error.to_string()),
            })
    }
}

/// An [`oxixml_relaxng::SchemaResolver`] for `include`/`externalRef`, catalog-aware.
#[derive(Debug, Clone, Default)]
pub struct RelaxNgCatalogResolver {
    catalog: Option<Catalog>,
}

impl RelaxNgCatalogResolver {
    /// Builds a resolver, with or without a catalog.
    #[must_use]
    pub const fn new(catalog: Option<Catalog>) -> Self {
        Self { catalog }
    }
}

impl oxixml_relaxng::SchemaResolver for RelaxNgCatalogResolver {
    fn resolve(&self, uri: &str) -> Result<String, oxixml_relaxng::ResolveError> {
        let (path, bytes) =
            read_via_catalog(self.catalog.as_ref(), uri, Path::new(".")).map_err(|error| {
                oxixml_relaxng::ResolveError::new(format!("cannot read `{uri}`: {error}"))
            })?;
        String::from_utf8(bytes).map_err(|error| {
            oxixml_relaxng::ResolveError::new(format!(
                "`{}` is not valid UTF-8: {error}",
                path.display()
            ))
        })
    }
}

/// An [`oxixml_schema::SchemaResolver`] for `xs:include`/`xs:import`/`xs:redefine`, catalog-aware
/// and falling back to [`oxixml_schema::FileSystemResolver`] when the catalog has nothing to say.
#[derive(Debug)]
pub struct SchemaCatalogResolver {
    catalog: Option<Catalog>,
    fallback: oxixml_schema::FileSystemResolver,
}

impl SchemaCatalogResolver {
    /// Builds a resolver, with or without a catalog.
    #[must_use]
    pub fn new(catalog: Option<Catalog>) -> Self {
        Self {
            catalog,
            fallback: oxixml_schema::FileSystemResolver::new(),
        }
    }
}

impl oxixml_schema::SchemaResolver for SchemaCatalogResolver {
    fn resolve(
        &mut self,
        location: &str,
        base: Option<&str>,
    ) -> Result<Option<oxixml_schema::ResolvedSchema>, oxixml_schema::SchemaError> {
        if let Some(catalog) = &self.catalog
            && let Some(rewritten) = catalog.resolve_uri(location)
        {
            let path = uri_to_path(&rewritten, Path::new("."));
            if let Ok(bytes) = fs::read(&path) {
                let text = String::from_utf8(bytes).map_err(|error| {
                    oxixml_schema::SchemaError::new(
                        "src-include",
                        format!("`{}` is not valid UTF-8: {error}", path.display()),
                    )
                })?;
                return Ok(Some(oxixml_schema::ResolvedSchema {
                    uri: path.to_string_lossy().into_owned(),
                    text,
                }));
            }
        }
        self.fallback.resolve(location, base)
    }
}

/// An [`oxixml_xslt::Resolver`] for `xsl:include`/`xsl:import`/`document()`, catalog-aware and
/// falling back to [`oxixml_xslt::FileResolver`] when the catalog has nothing to say.
#[derive(Debug)]
pub struct XsltCatalogResolver {
    catalog: Option<Catalog>,
    fallback: oxixml_xslt::FileResolver,
}

impl XsltCatalogResolver {
    /// Builds a resolver rooted at `directory`, with or without a catalog.
    #[must_use]
    pub fn new(catalog: Option<Catalog>, directory: impl Into<PathBuf>) -> Self {
        Self {
            catalog,
            fallback: oxixml_xslt::FileResolver::new(directory),
        }
    }

    fn via_catalog(&self, uri: &str) -> Option<(PathBuf, Vec<u8>)> {
        let catalog = self.catalog.as_ref()?;
        let rewritten = catalog.resolve_uri(uri)?;
        let path = uri_to_path(&rewritten, Path::new("."));
        let bytes = fs::read(&path).ok()?;
        Some((path, bytes))
    }
}

impl oxixml_xslt::Resolver for XsltCatalogResolver {
    fn document(&self, uri: &str) -> Option<oxixml_infoset::NodeRef> {
        if let Some((path, bytes)) = self.via_catalog(uri) {
            let text = String::from_utf8(bytes).ok()?;
            let tree = oxixml_infoset::Tree::parse_str_with(
                &text,
                oxixml_infoset::ParseOptions::default()
                    .document_uri(io_util::path_to_file_uri(&path)),
            )
            .ok()?;
            return Some(tree.root());
        }
        self.fallback.document(uri)
    }

    fn unparsed_text(&self, uri: &str, encoding: Option<&str>) -> Option<String> {
        if let Some((_, bytes)) = self.via_catalog(uri) {
            return String::from_utf8(bytes).ok();
        }
        self.fallback.unparsed_text(uri, encoding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_relaxng::SchemaResolver as _;
    use oxixml_xinclude::ResourceLoader as _;
    use std::env;

    fn temp_file(name: &str, contents: &str) -> PathBuf {
        let path = env::temp_dir().join(format!(
            "oxixml-cli-catalog-support-{}-{name}",
            std::process::id()
        ));
        fs::write(&path, contents).expect("write temp file");
        path
    }

    #[test]
    fn xinclude_loader_reads_a_plain_absolute_path_without_a_catalog() {
        let path = temp_file("a.xml", "<p>hi</p>");
        let loader = XIncludeLoader::new(None);
        let bytes = loader
            .load(path.to_str().expect("utf8 path"), None, None)
            .expect("reads");
        assert_eq!(bytes, b"<p>hi</p>");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn xinclude_loader_reports_not_found() {
        let loader = XIncludeLoader::new(None);
        let error = loader
            .load("/no/such/file/oxixml-cli-test", None, None)
            .expect_err("missing file");
        assert_eq!(error, oxixml_xinclude::LoadError::NotFound);
    }

    #[test]
    fn relaxng_resolver_reads_plain_paths() {
        let path = temp_file("schema.rng", "<grammar/>");
        let resolver = RelaxNgCatalogResolver::new(None);
        let text = resolver
            .resolve(path.to_str().expect("utf8 path"))
            .expect("reads");
        assert_eq!(text, "<grammar/>");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn catalog_rewrites_take_priority() {
        let target = temp_file("rewritten.xml", "<rewritten/>");
        let catalog_xml = format!(
            r#"<catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
                 <uri name="urn:example:thing" uri="{}"/>
               </catalog>"#,
            io_util::path_to_file_uri(&target)
        );
        let catalog = Catalog::parse(&catalog_xml, "file:///catalog.xml").expect("valid catalog");
        let loader = XIncludeLoader::new(Some(catalog));
        let bytes = loader.load("urn:example:thing", None, None).expect("reads");
        assert_eq!(bytes, b"<rewritten/>");
        let _ = fs::remove_file(&target);
    }
}
