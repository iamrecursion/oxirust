//! Locating the schema documents that `xs:include`, `xs:import`,
//! `xs:redefine` and `xs:override` point at.

use std::path::{Path, PathBuf};

use crate::error::SchemaError;

/// A schema document that a [`SchemaResolver`] found.
#[derive(Debug, Clone)]
pub struct ResolvedSchema {
    /// The absolute location the document was read from; it becomes the base
    /// URI for the references *inside* it.
    pub uri: String,
    /// The document text.
    pub text: String,
}

/// Turns a `schemaLocation` into a schema document.
///
/// Implement this to serve schema documents from anywhere — a catalogue, a
/// bundle compiled into the binary, the network. The default,
/// [`FileSystemResolver`], reads them from the file system.
pub trait SchemaResolver: std::fmt::Debug {
    /// Resolves `location`, which is relative to `base` when `base` is given.
    ///
    /// Returning `Ok(None)` means "not found, and that is not an error" — which
    /// is what the specification requires of `xs:import` without a
    /// `schemaLocation`, and what most processors do for an unreachable hint.
    ///
    /// # Errors
    ///
    /// Return an error only when the document was found but could not be used.
    fn resolve(
        &mut self,
        location: &str,
        base: Option<&str>,
    ) -> Result<Option<ResolvedSchema>, SchemaError>;
}

/// The default resolver: reads schema documents from the file system,
/// interpreting `schemaLocation` as a relative path against the including
/// document.
#[derive(Debug, Default, Clone)]
pub struct FileSystemResolver {
    _private: (),
}

impl FileSystemResolver {
    /// Builds a resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl SchemaResolver for FileSystemResolver {
    fn resolve(
        &mut self,
        location: &str,
        base: Option<&str>,
    ) -> Result<Option<ResolvedSchema>, SchemaError> {
        let path = join_location(location, base);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let text = crate::source::decode(&bytes).map_err(|reason| {
                    SchemaError::new(
                        "src-include",
                        format!("`{}` cannot be decoded: {reason}", path.display()),
                    )
                })?;
                Ok(Some(ResolvedSchema {
                    uri: path.to_string_lossy().into_owned(),
                    text,
                }))
            }
            Err(_) => Ok(None),
        }
    }
}

/// A resolver that never finds anything: every `schemaLocation` is ignored.
///
/// Useful when every schema document is supplied by hand through
/// [`SchemaSet::add_document`](crate::SchemaSet::add_document).
#[derive(Debug, Default, Clone)]
pub struct NullResolver;

impl SchemaResolver for NullResolver {
    fn resolve(
        &mut self,
        _location: &str,
        _base: Option<&str>,
    ) -> Result<Option<ResolvedSchema>, SchemaError> {
        Ok(None)
    }
}

/// Joins a (possibly relative) `location` onto the directory of `base`.
#[must_use]
pub fn join_location(location: &str, base: Option<&str>) -> PathBuf {
    let stripped = location.strip_prefix("file://").unwrap_or(location);
    let candidate = Path::new(stripped);
    if candidate.is_absolute() {
        return candidate.to_path_buf();
    }
    match base {
        Some(base) => {
            let base = base.strip_prefix("file://").unwrap_or(base);
            let directory = Path::new(base).parent().unwrap_or(Path::new(""));
            normalize(&directory.join(candidate))
        }
        None => candidate.to_path_buf(),
    }
}

/// Removes `.` and `..` segments so that two spellings of one path compare
/// equal.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_locations_join_onto_the_base() {
        assert_eq!(
            join_location("b/c.xsd", Some("/tmp/a/root.xsd")),
            PathBuf::from("/tmp/a/b/c.xsd")
        );
        assert_eq!(
            join_location("../c.xsd", Some("/tmp/a/root.xsd")),
            PathBuf::from("/tmp/c.xsd")
        );
        assert_eq!(
            join_location("/abs.xsd", Some("/tmp/a/root.xsd")),
            PathBuf::from("/abs.xsd")
        );
    }

    #[test]
    fn the_null_resolver_finds_nothing() -> Result<(), SchemaError> {
        assert!(NullResolver.resolve("x.xsd", None)?.is_none());
        Ok(())
    }
}
