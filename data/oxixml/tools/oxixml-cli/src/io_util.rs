//! Reading command-line input (honoring the `-` = stdin convention every subcommand shares) and
//! deriving a `file:` base URI from a real path.

use std::io::{self, Read, Write};
use std::path::Path;

/// The path argument that means "read from stdin".
pub const STDIN: &str = "-";

/// Reads `path` as bytes: `stdin` verbatim when `path` is [`STDIN`], the named file otherwise.
///
/// # Errors
///
/// Any [`io::Error`] the underlying read raises.
pub fn read_bytes(path: &str, stdin: &mut dyn Read) -> io::Result<Vec<u8>> {
    if path == STDIN {
        let mut buffer = Vec::new();
        stdin.read_to_end(&mut buffer)?;
        Ok(buffer)
    } else {
        std::fs::read(path)
    }
}

/// Reads `path` as UTF-8 text: `stdin` verbatim when `path` is [`STDIN`], the named file
/// otherwise.
///
/// # Errors
///
/// Any [`io::Error`] the underlying read raises, including [`io::ErrorKind::InvalidData`] when
/// the bytes are not valid UTF-8.
pub fn read_string(path: &str, stdin: &mut dyn Read) -> io::Result<String> {
    let bytes = read_bytes(path, stdin)?;
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// The label used in diagnostics and error messages for `path`: `<stdin>` for [`STDIN`], the
/// path itself otherwise.
#[must_use]
pub fn display_path(path: &str) -> &str {
    if path == STDIN { "<stdin>" } else { path }
}

/// Writes `output`'s bytes to `path`: the named file when `path` is `Some`, `stdout` otherwise.
///
/// # Errors
///
/// Any [`io::Error`] the underlying write raises.
pub fn write_output(path: Option<&str>, output: &[u8], stdout: &mut dyn Write) -> io::Result<()> {
    match path {
        Some(path) => std::fs::write(path, output),
        None => stdout.write_all(output),
    }
}

/// Derives a `file:` base URI for `path`, for resolving relative references inside the document
/// it names. Returns `None` for [`STDIN`] (there is nothing to resolve against) and for a path
/// whose absolute form cannot be computed.
#[must_use]
pub fn file_base_uri(path: &str) -> Option<String> {
    if path == STDIN {
        return None;
    }
    let absolute = std::path::absolute(Path::new(path)).ok()?;
    Some(path_to_file_uri(&absolute))
}

/// Renders an absolute filesystem path as a `file:` URI, percent-encoding the handful of octets
/// that are not allowed unescaped in a URI path segment.
#[must_use]
pub fn path_to_file_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for component in path.components() {
        let text = component.as_os_str().to_string_lossy();
        if text == "/" {
            continue;
        }
        uri.push('/');
        for byte in text.as_bytes().iter().copied() {
            match byte {
                b'A'..=b'Z'
                | b'a'..=b'z'
                | b'0'..=b'9'
                | b'-'
                | b'_'
                | b'.'
                | b'~'
                | b'/'
                | b':' => uri.push(byte as char),
                other => uri.push_str(&format!("%{other:02X}")),
            }
        }
    }
    uri
}

/// The directory a path lives in, as a string usable as a resolver root — `.` for a bare file
/// name and for [`STDIN`].
#[must_use]
pub fn parent_dir(path: &str) -> String {
    if path == STDIN {
        return ".".to_owned();
    }
    Path::new(path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(
            || ".".to_owned(),
            |parent| parent.to_string_lossy().into_owned(),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdin_reads_from_the_provided_reader() {
        let mut input: &[u8] = b"hello";
        let bytes = read_bytes(STDIN, &mut input).expect("reads");
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn a_real_path_is_read_from_disk() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "oxixml-cli-io-util-test-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, b"disk").expect("write temp file");
        let mut input: &[u8] = &[];
        let bytes = read_bytes(path.to_str().expect("utf8 path"), &mut input).expect("reads");
        assert_eq!(bytes, b"disk");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn display_path_labels_stdin() {
        assert_eq!(display_path(STDIN), "<stdin>");
        assert_eq!(display_path("a.xml"), "a.xml");
    }

    #[test]
    fn file_base_uri_is_none_for_stdin() {
        assert_eq!(file_base_uri(STDIN), None);
    }

    #[test]
    fn file_base_uri_starts_with_the_file_scheme() {
        let uri = file_base_uri("a.xml").expect("resolvable");
        assert!(uri.starts_with("file:///"), "{uri}");
        assert!(uri.ends_with("a.xml"), "{uri}");
    }

    #[test]
    fn parent_dir_of_a_bare_name_is_the_current_directory() {
        assert_eq!(parent_dir("a.xml"), ".");
        assert_eq!(parent_dir(STDIN), ".");
    }

    #[test]
    fn parent_dir_of_a_nested_path_is_its_directory() {
        assert_eq!(parent_dir("dir/a.xml"), "dir");
    }
}
