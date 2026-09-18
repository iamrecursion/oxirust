//! Pluggable, opt-in resolution of external entities.
//!
//! An XML processor that silently dereferences every `SYSTEM` identifier it
//! meets is an SSRF gadget and a file-disclosure gadget. This crate therefore
//! never fetches anything by itself: the external subset, external parameter
//! entities and external general entities are only read when the caller
//! installs an [`EntityResolver`] that agrees to supply them.
//!
//! The default, [`DenyExternalEntities`], supplies nothing, so
//! [`DtdParser`](crate::DtdParser) out of the box cannot touch the network or
//! the file system.
//!
//! Two ready-made resolvers cover the common cases:
//!
//! * [`MemoryResolver`] — a fixed table of system identifiers to text; ideal
//!   for tests and for embedding a known DTD in the binary.
//! * [`FilesystemResolver`] — reads from a directory tree, refusing to escape
//!   the root it was given.
//!
//! # Examples
//!
//! ```
//! use oxixml_dtd::{DtdParser, MemoryResolver};
//!
//! let mut resolver = MemoryResolver::new();
//! resolver.insert("common.ent", "<!ELEMENT note (#PCDATA)>");
//!
//! let dtd = DtdParser::new()
//!     .with_resolver(Box::new(resolver))
//!     .parse_internal_subset(r#"<!ENTITY % common SYSTEM "common.ent"> %common;"#)?;
//! assert!(dtd.element("note").is_some());
//! # Ok::<(), oxixml_dtd::DtdError>(())
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};

/// What is being resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExternalKind {
    /// The external DTD subset named by the document type declaration.
    Subset,
    /// An external parameter entity (`<!ENTITY % name SYSTEM "…">`).
    ParameterEntity,
    /// An external *parsed* general entity (`<!ENTITY name SYSTEM "…">`).
    GeneralEntity,
}

impl fmt::Display for ExternalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Subset => "external subset",
            Self::ParameterEntity => "external parameter entity",
            Self::GeneralEntity => "external general entity",
        };
        f.write_str(text)
    }
}

/// Everything a resolver is told about the entity it is being asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolveRequest<'a> {
    /// Whether this is the external subset, a parameter entity or a general
    /// entity.
    pub kind: ExternalKind,
    /// The entity's name, absent for the external subset.
    pub name: Option<&'a str>,
    /// The public identifier, when the declaration gave one.
    pub public_id: Option<&'a str>,
    /// The system identifier, exactly as written in the declaration — *not*
    /// resolved against any base. Resolvers that understand relative references
    /// should combine it with [`base`](Self::base) themselves.
    pub system_id: &'a str,
    /// The base URI of the entity that contained the declaration, when known.
    pub base: Option<&'a str>,
}

/// Text supplied by a resolver, plus the base URI to use for anything nested
/// inside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEntity {
    /// The entity's replacement text, already decoded to UTF-8.
    pub text: String,
    /// The base URI of the resolved entity, used to resolve relative system
    /// identifiers that occur inside it.
    pub base: Option<String>,
}

impl ResolvedEntity {
    /// Builds a resolved entity with no base URI.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            base: None,
        }
    }

    /// Sets the base URI used for nested relative system identifiers.
    #[must_use]
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = Some(base.into());
        self
    }
}

/// Why a resolver could not supply an entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolverError {
    /// The resolver refuses to dereference this identifier at all. Surfaces as
    /// [`DtdErrorKind::ExternalEntityDenied`](crate::DtdErrorKind::ExternalEntityDenied).
    Denied,
    /// The resolver tried and failed. Surfaces as
    /// [`DtdErrorKind::ExternalEntityUnavailable`](crate::DtdErrorKind::ExternalEntityUnavailable).
    Unavailable(String),
}

impl fmt::Display for ResolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Denied => f.write_str("denied by the entity resolver"),
            Self::Unavailable(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for ResolverError {}

/// Supplies the text of external entities.
///
/// Implementations receive `&mut self` so that a resolver may cache, count or
/// rate-limit what it hands out.
pub trait EntityResolver {
    /// Resolves one external entity.
    ///
    /// # Errors
    ///
    /// Return [`ResolverError::Denied`] to refuse, or
    /// [`ResolverError::Unavailable`] when resolution was attempted and failed.
    fn resolve(&mut self, request: &ResolveRequest<'_>) -> Result<ResolvedEntity, ResolverError>;
}

impl<T: EntityResolver + ?Sized> EntityResolver for Box<T> {
    fn resolve(&mut self, request: &ResolveRequest<'_>) -> Result<ResolvedEntity, ResolverError> {
        (**self).resolve(request)
    }
}

/// The default resolver: refuses everything, so nothing external is ever read.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{DtdErrorKind, DtdParser};
///
/// let error = DtdParser::new()
///     .parse_internal_subset(r#"<!ENTITY % ext SYSTEM "http://example.invalid/x.ent"> %ext;"#)
///     .expect_err("external entities are denied by default");
/// assert!(matches!(error.kind(), DtdErrorKind::ExternalEntityDenied { .. }));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct DenyExternalEntities;

impl EntityResolver for DenyExternalEntities {
    fn resolve(&mut self, _request: &ResolveRequest<'_>) -> Result<ResolvedEntity, ResolverError> {
        Err(ResolverError::Denied)
    }
}

/// A resolver backed by an in-memory table.
///
/// Lookup tries the public identifier first (when the declaration gave one and
/// the table has an entry for it), then the system identifier. Matching is
/// exact — no relative-reference resolution is attempted, which keeps the
/// behaviour predictable for tests.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{EntityResolver, ExternalKind, MemoryResolver, ResolveRequest};
///
/// let mut resolver = MemoryResolver::new();
/// resolver.insert("urn:x", "<!ELEMENT a EMPTY>");
///
/// let request = ResolveRequest {
///     kind: ExternalKind::ParameterEntity,
///     name: Some("p"),
///     public_id: None,
///     system_id: "urn:x",
///     base: None,
/// };
/// assert_eq!(resolver.resolve(&request)?.text, "<!ELEMENT a EMPTY>");
/// # Ok::<(), oxixml_dtd::ResolverError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryResolver {
    entries: BTreeMap<String, String>,
}

impl MemoryResolver {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `text` under `identifier` (a system *or* public identifier).
    pub fn insert(&mut self, identifier: impl Into<String>, text: impl Into<String>) {
        self.entries.insert(identifier.into(), text.into());
    }

    /// Builder-style [`insert`](Self::insert).
    #[must_use]
    pub fn with(mut self, identifier: impl Into<String>, text: impl Into<String>) -> Self {
        self.insert(identifier, text);
        self
    }

    /// The number of registered identifiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl EntityResolver for MemoryResolver {
    fn resolve(&mut self, request: &ResolveRequest<'_>) -> Result<ResolvedEntity, ResolverError> {
        if let Some(public_id) = request.public_id
            && let Some(text) = self.entries.get(public_id)
        {
            return Ok(ResolvedEntity::new(text.clone()));
        }
        self.entries
            .get(request.system_id)
            .map(|text| ResolvedEntity::new(text.clone()).with_base(request.system_id))
            .ok_or(ResolverError::Denied)
    }
}

/// A resolver that reads files from below a fixed root directory.
///
/// Relative system identifiers are resolved against the requesting entity's
/// [base](ResolveRequest::base); the result is then normalised and checked to
/// still be inside the root, so a system identifier of `../../etc/passwd`
/// is refused rather than followed. Absolute paths, `file:` URLs pointing
/// outside the root, and any other scheme are refused as well.
///
/// The file is decoded as UTF-8; a byte-order mark is stripped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemResolver {
    root: PathBuf,
}

impl FilesystemResolver {
    /// Creates a resolver rooted at `root`.
    ///
    /// Nothing outside `root` will ever be read.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory this resolver is confined to.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolves `system_id` against `base`, returning a path relative to the
    /// root, or `None` when the result escapes the root or is not a plain
    /// relative reference.
    fn locate(&self, system_id: &str, base: Option<&str>) -> Option<PathBuf> {
        let reference = system_id.strip_prefix("file://").unwrap_or(system_id);
        if reference.contains("://") {
            return None;
        }
        let mut combined = if reference.starts_with('/') {
            PathBuf::from(reference.trim_start_matches('/'))
        } else {
            let mut directory = match base {
                Some(base) => {
                    let base = base.strip_prefix("file://").unwrap_or(base);
                    let mut path = PathBuf::from(base);
                    path.pop();
                    path
                }
                None => PathBuf::new(),
            };
            directory.push(reference);
            directory
        };
        combined = normalize_relative(&combined)?;
        Some(combined)
    }
}

/// Collapses `.` and `..` components, returning `None` when the path would
/// escape its starting directory or is not relative.
fn normalize_relative(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            Component::Normal(part) => out.push(part),
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

impl EntityResolver for FilesystemResolver {
    fn resolve(&mut self, request: &ResolveRequest<'_>) -> Result<ResolvedEntity, ResolverError> {
        let relative = self
            .locate(request.system_id, request.base)
            .ok_or_else(|| {
                ResolverError::Unavailable(format!(
                    "`{}` resolves outside the resolver root",
                    request.system_id
                ))
            })?;
        let absolute = self.root.join(&relative);
        // `locate`/`normalize_relative` above only collapse `.`/`..` *lexically* in the
        // requested path string; they say nothing about symlinks the filesystem itself might
        // follow. A `root/escape.dtd` that is a symlink pointing outside `root` would otherwise
        // be read straight through, silently breaking the "nothing outside root will ever be
        // read" guarantee. Canonicalizing both sides resolves every symlink (and any remaining
        // `.`/`..`) at the OS level, so the containment check that follows is real rather than
        // purely textual — and the root itself must be canonicalized too, not just the target,
        // since a root that is itself reached through a symlink would otherwise make even a
        // legitimate, non-hostile request fail the prefix check.
        let canonical_root = std::fs::canonicalize(&self.root).map_err(|error| {
            ResolverError::Unavailable(format!(
                "the resolver root `{}` could not be canonicalized: {error}",
                self.root.display()
            ))
        })?;
        let canonical_target = std::fs::canonicalize(&absolute)
            .map_err(|error| ResolverError::Unavailable(error.to_string()))?;
        if !canonical_target.starts_with(&canonical_root) {
            return Err(ResolverError::Unavailable(format!(
                "`{}` resolves outside the resolver root",
                request.system_id
            )));
        }
        let bytes = std::fs::read(&canonical_target)
            .map_err(|error| ResolverError::Unavailable(error.to_string()))?;
        let text = decode_utf8(&bytes).map_err(ResolverError::Unavailable)?;
        let base = relative.to_string_lossy().replace('\\', "/");
        Ok(ResolvedEntity {
            text,
            base: Some(base),
        })
    }
}

/// Decodes UTF-8 (or UTF-16, when a byte-order mark says so), stripping any BOM.
fn decode_utf8(bytes: &[u8]) -> Result<String, String> {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8(rest.to_vec()).map_err(|error| error.to_string());
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return decode_utf16(rest, true);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return decode_utf16(rest, false);
    }
    String::from_utf8(bytes.to_vec()).map_err(|error| error.to_string())
}

/// Decodes UTF-16 code units of the given endianness.
fn decode_utf16(bytes: &[u8], big_endian: bool) -> Result<String, String> {
    if bytes.len() % 2 != 0 {
        return Err("truncated UTF-16 input".to_owned());
    }
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16(&units).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_resolver_refuses_everything() {
        let request = ResolveRequest {
            kind: ExternalKind::Subset,
            name: None,
            public_id: None,
            system_id: "x.dtd",
            base: None,
        };
        assert_eq!(
            DenyExternalEntities.resolve(&request),
            Err(ResolverError::Denied)
        );
    }

    #[test]
    fn memory_resolver_prefers_the_public_identifier() {
        let mut resolver = MemoryResolver::new()
            .with("-//X//DTD//EN", "public")
            .with("x.dtd", "system");
        let request = ResolveRequest {
            kind: ExternalKind::Subset,
            name: None,
            public_id: Some("-//X//DTD//EN"),
            system_id: "x.dtd",
            base: None,
        };
        assert_eq!(
            resolver.resolve(&request).map(|entity| entity.text),
            Ok("public".to_owned())
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_symlink_inside_root_pointing_outside_is_refused_not_followed() {
        // `locate`/`normalize_relative` only collapse `.`/`..` lexically in the *requested*
        // path string; a symlink placed inside `root` that points elsewhere on disk bypasses
        // that check entirely unless the actual read path is also verified against the root
        // after the OS resolves every symlink.
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "oxixml-dtd-resolver-escape-{}-{}",
            std::process::id(),
            line!()
        ));
        let root = base.join("root");
        let outside = base.join("outside");
        std::fs::create_dir_all(&root).expect("create the root directory");
        std::fs::create_dir_all(&outside).expect("create the outside directory");
        std::fs::write(outside.join("secret.dtd"), "TOP SECRET OUTSIDE ROOT")
            .expect("write the file outside root");
        symlink(outside.join("secret.dtd"), root.join("escape.dtd"))
            .expect("create the escaping symlink");

        let mut resolver = FilesystemResolver::new(&root);
        let request = ResolveRequest {
            kind: ExternalKind::Subset,
            name: None,
            public_id: None,
            system_id: "escape.dtd",
            base: None,
        };
        let error = resolver
            .resolve(&request)
            .expect_err("a symlink escaping the root must be refused, not followed");
        assert!(matches!(error, ResolverError::Unavailable(_)), "{error:?}");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    #[cfg(unix)]
    fn a_symlink_that_stays_inside_root_still_resolves() {
        // The containment check must reject an escape without also rejecting the common,
        // entirely legitimate case of a symlink that happens to stay inside the root (shared
        // DTD directories built by unpacking archives commonly contain exactly this shape).
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!(
            "oxixml-dtd-resolver-inroot-{}-{}",
            std::process::id(),
            line!()
        ));
        // Canonicalized up front so the assertions below compare like with like even on a
        // platform (macOS) where the system temporary directory is itself reached through a
        // symlink (`/var` -> `/private/var`) — the resolver canonicalizes its own root
        // internally regardless, so this only sharpens what the test expects.
        std::fs::create_dir_all(&base).expect("create the base directory");
        let base = std::fs::canonicalize(&base).expect("canonicalize the base directory");
        let root = base.join("root");
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).expect("create the nested directory");
        std::fs::write(nested.join("real.dtd"), "<!ELEMENT a EMPTY>").expect("write the real file");
        symlink(nested.join("real.dtd"), root.join("alias.dtd"))
            .expect("create the in-root symlink");

        let mut resolver = FilesystemResolver::new(&root);
        let request = ResolveRequest {
            kind: ExternalKind::Subset,
            name: None,
            public_id: None,
            system_id: "alias.dtd",
            base: None,
        };
        let resolved = resolver
            .resolve(&request)
            .expect("a symlink that stays inside the root must still resolve");
        assert_eq!(resolved.text, "<!ELEMENT a EMPTY>");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn filesystem_resolver_refuses_to_escape_its_root() {
        let resolver = FilesystemResolver::new("/srv/dtds");
        assert_eq!(resolver.locate("../../etc/passwd", None), None);
        assert_eq!(resolver.locate("http://example.invalid/x", None), None);
        assert_eq!(
            resolver.locate("common.ent", Some("sub/main.dtd")),
            Some(PathBuf::from("sub/common.ent"))
        );
    }

    #[test]
    fn utf16_input_is_decoded() {
        let mut bytes = vec![0xFF, 0xFE];
        bytes.extend_from_slice(&[b'a', 0, b'b', 0]);
        assert_eq!(decode_utf8(&bytes), Ok("ab".to_owned()));
    }
}
