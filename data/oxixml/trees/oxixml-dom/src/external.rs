//! Optional retrieval of external parsed general entities.
//!
//! A non-validating XML processor is explicitly allowed not to retrieve the
//! text of an external entity ([XML 1.0 §5.1](https://www.w3.org/TR/xml/#proc-types)),
//! and by default [`build::parse_str`](crate::build::parse_str) does not: an
//! `&name;` whose declaration is `<!ENTITY name SYSTEM "…">` keeps its
//! [`EntityReference`](crate::NodeKind::EntityReference) node, exactly as an
//! undeclared one does. Retrieving it means reading whatever the system
//! identifier points at, which is a decision only the application can make —
//! so it is opt-in, through
//! [`build::parse_str_with_resolver`](crate::build::parse_str_with_resolver)
//! and an [`ExternalEntityResolver`] the caller supplies.
//!
//! [`FileEntityResolver`] is the resolver most callers want: it resolves a
//! relative or `file:` system identifier against one base directory, reads the
//! file, and refuses everything else. It never opens a network connection and
//! never escapes the base directory.
//!
//! # Example
//!
//! ```
//! use oxixml_dom::build::{self, BuilderOptions};
//! use oxixml_dom::external::{ExternalEntity, ExternalEntityError, ExternalEntityResolver};
//!
//! /// A resolver backed by an in-memory table, which is all a test needs.
//! struct Table;
//! impl ExternalEntityResolver for Table {
//!     fn resolve(&mut self, entity: &ExternalEntity<'_>) -> Result<Option<String>, ExternalEntityError> {
//!         Ok((entity.system_id == "greeting.ent").then(|| "<hi>there</hi>".to_owned()))
//!     }
//! }
//!
//! let xml = r#"<!DOCTYPE doc [<!ENTITY g SYSTEM "greeting.ent">]><doc>&g;</doc>"#;
//! // Without a resolver the reference stays a node of its own.
//! assert_eq!(build::parse_str(xml)?.to_xml(), "<!DOCTYPE doc [<!ENTITY g SYSTEM \"greeting.ent\">]><doc>&g;</doc>");
//! // With one, its content is spliced into the tree.
//! let doc = build::parse_str_with_resolver(xml, &BuilderOptions::new(), &mut Table)?;
//! assert!(doc.to_xml().ends_with("<doc><hi>there</hi></doc>"));
//! # Ok::<(), oxixml_dom::BuildError>(())
//! ```

use std::fmt;
use std::path::{Component, Path, PathBuf};

/// The declaration of an external parsed general entity, as the resolver sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ExternalEntity<'a> {
    /// The entity name, without the `&` and `;` of a reference.
    pub name: &'a str,
    /// The public identifier of the declaration, if it had one.
    pub public_id: Option<&'a str>,
    /// The system identifier of the declaration.
    pub system_id: &'a str,
}

/// Why an external parsed general entity could not be retrieved.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExternalEntityError {
    /// The system identifier does not name something this resolver will read —
    /// a non-`file:` scheme, or a path outside the base directory.
    Refused {
        /// The system identifier as written in the declaration.
        system_id: String,
        /// Why it was refused.
        detail: &'static str,
    },
    /// Reading the entity failed.
    Io {
        /// The system identifier as written in the declaration.
        system_id: String,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// The entity is larger than the resolver is willing to read.
    TooLarge {
        /// The system identifier as written in the declaration.
        system_id: String,
        /// The ceiling, in bytes.
        limit: usize,
    },
}

impl fmt::Display for ExternalEntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused { system_id, detail } => {
                write!(f, "refused to retrieve {system_id:?}: {detail}")
            }
            Self::Io { system_id, source } => {
                write!(f, "could not read {system_id:?}: {source}")
            }
            Self::TooLarge { system_id, limit } => write!(
                f,
                "{system_id:?} is larger than the {limit}-byte limit on an external entity"
            ),
        }
    }
}

impl std::error::Error for ExternalEntityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Retrieves the replacement text of an external parsed general entity.
///
/// Returning `Ok(None)` means "I do not know this entity", which leaves the
/// reference in the tree as an
/// [`EntityReference`](crate::NodeKind::EntityReference) node — the same
/// outcome as parsing without a resolver at all. Returning `Ok(Some(text))`
/// hands over the entity's replacement text, which is parsed as
/// `content` ([XML 1.0 §4.3.2](https://www.w3.org/TR/xml/#wf-entities)) in the
/// namespace scope of the reference and spliced into the tree in its place.
pub trait ExternalEntityResolver {
    /// Retrieves `entity`, or declines to.
    ///
    /// # Errors
    ///
    /// Returns [`ExternalEntityError`] when retrieval was attempted and failed.
    /// Declining is `Ok(None)`, not an error.
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError>;
}

/// The default ceiling, in bytes, on one external entity read by a
/// [`FileEntityResolver`].
pub const DEFAULT_MAX_ENTITY_BYTES: usize = 8 * 1024 * 1024;

/// An [`ExternalEntityResolver`] that reads UTF-8 files below one base
/// directory.
///
/// A system identifier is accepted when it is a relative reference or a `file:`
/// URI, and the path it denotes stays inside the base directory. Every other
/// scheme — `http:`, `https:`, `ftp:`, … — is refused without any I/O, so a
/// document cannot make the parser open a network connection. An absolute path,
/// a `..` that climbs out of the base directory and a host-bearing `file://`
/// authority are refused for the same reason.
#[derive(Debug, Clone)]
pub struct FileEntityResolver {
    base: PathBuf,
    max_bytes: usize,
}

impl FileEntityResolver {
    /// Creates a resolver that reads files below `base`.
    #[must_use]
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self {
            base: base.into(),
            max_bytes: DEFAULT_MAX_ENTITY_BYTES,
        }
    }

    /// Sets the ceiling, in bytes, on one entity. The default is
    /// [`DEFAULT_MAX_ENTITY_BYTES`].
    #[must_use]
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// The base directory system identifiers are resolved against.
    #[must_use]
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Turns a system identifier into the path it names below the base
    /// directory, or explains why it names nothing this resolver will read.
    fn locate(&self, system_id: &str) -> Result<PathBuf, &'static str> {
        let relative = strip_file_scheme(system_id)?;
        let relative = percent_decode(relative);
        if relative.is_empty() {
            return Err("the system identifier is empty");
        }
        let candidate = Path::new(&relative);
        if candidate.is_absolute() {
            return Err("an absolute path is outside the base directory");
        }
        // Resolve `.` and `..` textually: the result must still be *below* the
        // base directory, and a symlink must not be able to change that answer
        // after the fact, so no canonicalization of the target is done.
        let mut resolved = PathBuf::new();
        for component in candidate.components() {
            match component {
                Component::CurDir => {}
                Component::ParentDir => {
                    if !resolved.pop() {
                        return Err("the system identifier climbs out of the base directory");
                    }
                }
                Component::Normal(part) => resolved.push(part),
                Component::RootDir | Component::Prefix(_) => {
                    return Err("an absolute path is outside the base directory");
                }
            }
        }
        Ok(self.base.join(resolved))
    }
}

impl ExternalEntityResolver for FileEntityResolver {
    fn resolve(
        &mut self,
        entity: &ExternalEntity<'_>,
    ) -> Result<Option<String>, ExternalEntityError> {
        let path =
            self.locate(entity.system_id)
                .map_err(|detail| ExternalEntityError::Refused {
                    system_id: entity.system_id.to_owned(),
                    detail,
                })?;
        let bytes = std::fs::read(&path).map_err(|source| ExternalEntityError::Io {
            system_id: entity.system_id.to_owned(),
            source,
        })?;
        if bytes.len() > self.max_bytes {
            return Err(ExternalEntityError::TooLarge {
                system_id: entity.system_id.to_owned(),
                limit: self.max_bytes,
            });
        }
        let text = String::from_utf8(bytes).map_err(|error| ExternalEntityError::Io {
            system_id: entity.system_id.to_owned(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, error),
        })?;
        Ok(Some(text))
    }
}

/// Strips a `file:` scheme from `system_id`, refusing every other scheme.
///
/// `file:///a/b` and `file:/a/b` both denote the absolute path `/a/b`, which
/// this resolver rejects later; `file://host/a` names another machine and is
/// refused here.
fn strip_file_scheme(system_id: &str) -> Result<&str, &'static str> {
    let Some((scheme, rest)) = split_scheme(system_id) else {
        return Ok(system_id);
    };
    if !scheme.eq_ignore_ascii_case("file") {
        return Err("only relative and `file:` system identifiers are retrieved");
    }
    let Some(after_slashes) = rest.strip_prefix("//") else {
        return Ok(rest);
    };
    // `file://host/path`: only an empty authority means "this machine", and the
    // path keeps its leading `/`, which makes it absolute and so out of bounds.
    match after_slashes.find('/') {
        Some(0) => Ok(after_slashes),
        Some(_) | None => Err("a `file:` URI naming a host is not retrieved"),
    }
}

/// Splits `reference` into its scheme and the rest, when it begins with one.
///
/// `scheme ::= ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )` followed by `:`, as
/// [RFC 3986 §3.1](https://www.rfc-editor.org/rfc/rfc3986#section-3.1) defines
/// it. A single letter counts: a system identifier is a URI reference, not a
/// file-system path, so `c:/windows/win.ini` names the scheme `c` and is refused
/// on every platform rather than being read as a Windows drive letter on one.
fn split_scheme(reference: &str) -> Option<(&str, &str)> {
    let colon = reference.find(':')?;
    let scheme = &reference[..colon];
    if scheme.is_empty() {
        return None;
    }
    let mut chars = scheme.chars();
    if !chars.next()?.is_ascii_alphabetic() {
        return None;
    }
    if !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '-' | '.')) {
        return None;
    }
    Some((scheme, &reference[colon + 1..]))
}

/// Decodes the `%XX` escapes of a URI reference. A malformed escape is left
/// alone rather than being an error: it is then just a character of the path.
fn percent_decode(text: &str) -> String {
    if !text.contains('%') {
        return text.to_owned();
    }
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && let Some(pair) = text.get(index + 1..index + 3)
            && let Ok(byte) = u8::from_str_radix(pair, 16)
        {
            out.push(byte);
            index += 3;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| text.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_identifiers_resolve_below_the_base() {
        let resolver = FileEntityResolver::new("/base");
        assert_eq!(resolver.locate("a.ent"), Ok(PathBuf::from("/base/a.ent")));
        assert_eq!(
            resolver.locate("sub/./a.ent"),
            Ok(PathBuf::from("/base/sub/a.ent"))
        );
        assert_eq!(
            resolver.locate("sub/../a.ent"),
            Ok(PathBuf::from("/base/a.ent"))
        );
        assert_eq!(
            resolver.locate("file:a%20b.ent"),
            Ok(PathBuf::from("/base/a b.ent"))
        );
    }

    #[test]
    fn identifiers_outside_the_base_are_refused() {
        let resolver = FileEntityResolver::new("/base");
        assert!(resolver.locate("../secret").is_err());
        assert!(resolver.locate("sub/../../secret").is_err());
        assert!(resolver.locate("/etc/passwd").is_err());
        assert!(resolver.locate("file:///etc/passwd").is_err());
        assert!(resolver.locate("").is_err());
    }

    #[test]
    fn only_the_file_scheme_is_retrieved() {
        let resolver = FileEntityResolver::new("/base");
        assert!(resolver.locate("http://example.invalid/a.ent").is_err());
        assert!(resolver.locate("https://example.invalid/a.ent").is_err());
        assert!(resolver.locate("ftp://example.invalid/a.ent").is_err());
        assert!(resolver.locate("file://host/a.ent").is_err());
        // A single letter is a scheme, so a drive-letter-shaped identifier is
        // refused rather than read as a path.
        assert!(resolver.locate("c:/windows/win.ini").is_err());
    }

    #[test]
    fn a_missing_file_is_an_io_error_not_a_refusal() {
        let mut resolver = FileEntityResolver::new(std::env::temp_dir());
        let error = resolver
            .resolve(&ExternalEntity {
                name: "e",
                public_id: None,
                system_id: "oxixml-does-not-exist.ent",
            })
            .expect_err("the file does not exist");
        assert!(matches!(error, ExternalEntityError::Io { .. }), "{error}");
    }

    #[test]
    fn a_file_below_the_base_is_read() {
        let dir = std::env::temp_dir().join("oxixml-dom-external-entity");
        std::fs::create_dir_all(&dir).expect("a writable temporary directory");
        let path = dir.join("greeting.ent");
        std::fs::write(&path, "<hi>there</hi>").expect("a writable temporary file");

        let mut resolver = FileEntityResolver::new(&dir);
        let text = resolver
            .resolve(&ExternalEntity {
                name: "g",
                public_id: None,
                system_id: "greeting.ent",
            })
            .expect("the file is readable");
        assert_eq!(text.as_deref(), Some("<hi>there</hi>"));

        let mut small = FileEntityResolver::new(&dir).with_max_bytes(4);
        let error = small
            .resolve(&ExternalEntity {
                name: "g",
                public_id: None,
                system_id: "greeting.ent",
            })
            .expect_err("the file is over the limit");
        assert!(
            matches!(error, ExternalEntityError::TooLarge { limit: 4, .. }),
            "{error}"
        );

        let _ = std::fs::remove_file(&path);
    }
}
