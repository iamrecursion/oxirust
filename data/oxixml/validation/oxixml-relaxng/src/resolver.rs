//! Retrieving the resources that `include` and `externalRef` point at.
//!
//! [§4.5](https://relaxng.org/spec-20011203.html#IDA4ZR) leaves the retrieval
//! mechanism to the implementation, and §8 explicitly allows a conforming
//! validator to document what it can and cannot fetch. This crate ships a
//! filesystem resolver — which covers `file:` URIs and plain paths — and lets an
//! application plug in anything else, including an in-memory map, by implementing
//! [`SchemaResolver`].

use std::fmt;
use std::path::PathBuf;

/// Why a resource could not be retrieved.
#[derive(Debug)]
pub struct ResolveError {
    message: String,
}

impl ResolveError {
    /// Builds an error with a human-readable explanation.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ResolveError {}

impl From<std::io::Error> for ResolveError {
    fn from(error: std::io::Error) -> Self {
        Self::new(error.to_string())
    }
}

/// Retrieves the text of the schema identified by an absolute URI.
pub trait SchemaResolver {
    /// Returns the text of the resource `uri` identifies.
    ///
    /// The URI has already been made absolute against the base URI in force, and
    /// has already had its disallowed characters escaped.
    ///
    /// # Errors
    ///
    /// Returns a [`ResolveError`] when the resource cannot be retrieved.
    fn resolve(&self, uri: &str) -> Result<String, ResolveError>;
}

/// A resolver that refuses everything.
///
/// This is the default for schemas given as a string with no base URI: without
/// one, a relative `href` cannot be made absolute anyway.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoResolver;

impl SchemaResolver for NoResolver {
    fn resolve(&self, uri: &str) -> Result<String, ResolveError> {
        Err(ResolveError::new(format!(
            "this schema was loaded without a resolver, so `{uri}` cannot be retrieved"
        )))
    }
}

/// A resolver that reads `file:` URIs from the local filesystem.
#[derive(Debug, Default, Clone, Copy)]
pub struct FileResolver;

impl SchemaResolver for FileResolver {
    fn resolve(&self, uri: &str) -> Result<String, ResolveError> {
        let path = file_uri_to_path(uri).ok_or_else(|| {
            ResolveError::new(format!(
                "only `file:` URIs can be retrieved, and `{uri}` is not one"
            ))
        })?;
        std::fs::read_to_string(&path).map_err(|error| {
            ResolveError::new(format!("cannot read `{}`: {error}", path.display()))
        })
    }
}

/// Converts a `file:` URI into a filesystem path, undoing percent-encoding.
#[must_use]
pub(crate) fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://").or_else(|| {
        uri.strip_prefix("file:")
            .filter(|rest| rest.starts_with('/'))
    })?;
    // An empty authority or `localhost` is allowed; anything else is a remote
    // host, which the filesystem cannot serve.
    let path = if let Some(path) = rest.strip_prefix("localhost/") {
        format!("/{path}")
    } else if rest.starts_with('/') {
        rest.to_owned()
    } else {
        return None;
    };
    Some(PathBuf::from(percent_decode(&path)))
}

/// Builds the `file:` URI of a filesystem path.
#[must_use]
pub(crate) fn path_to_file_uri(path: &std::path::Path) -> String {
    let mut uri = String::from("file://");
    for byte in path.to_string_lossy().bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' | b':' => {
                uri.push(char::from(byte));
            }
            other => {
                uri.push('%');
                uri.push(hex(other >> 4));
                uri.push(hex(other & 0x0f));
            }
        }
    }
    uri
}

fn hex(nibble: u8) -> char {
    char::from(match nibble {
        0..=9 => b'0' + nibble,
        _ => b'A' + (nibble - 10),
    })
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let high = char::from(bytes[index + 1]).to_digit(16);
            let low = char::from(bytes[index + 2]).to_digit(16);
            if let (Some(high), Some(low)) = (high, low) {
                let byte = u8::try_from(high * 16 + low).unwrap_or(b'?');
                out.push(byte);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_uris_round_trip() {
        let path = std::path::Path::new("/tmp/a b/c.rng");
        let uri = path_to_file_uri(path);
        assert_eq!(uri, "file:///tmp/a%20b/c.rng");
        assert_eq!(file_uri_to_path(&uri).as_deref(), Some(path));
    }

    #[test]
    fn only_local_file_uris_are_accepted() {
        assert!(file_uri_to_path("http://example.com/x").is_none());
        assert!(file_uri_to_path("file://example.com/x").is_none());
        assert_eq!(
            file_uri_to_path("file://localhost/tmp/x"),
            Some(PathBuf::from("/tmp/x"))
        );
    }

    #[test]
    fn the_no_resolver_always_fails() {
        assert!(NoResolver.resolve("file:///x").is_err());
    }
}
