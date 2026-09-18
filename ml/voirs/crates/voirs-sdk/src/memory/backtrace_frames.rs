//! Structured stack frames from [`std::backtrace::Backtrace`].
//!
//! The standard library exposes a captured backtrace only through its
//! `Display` output, so this module parses that text back into frames. It
//! replaces the `backtrace` crate, whose symbolizer dependency chain pulled
//! `miniz_oxide` into every default build (COOLJAPAN policy: compression only
//! via OxiARC).
//!
//! The `Display` format is one header line per symbol followed by an optional
//! location line:
//!
//! ```text
//!    0: my_crate::module::function
//!              at ./src/module.rs:42:9
//!    1: std::rt::lang_start
//! ```
//!
//! Parsing is tolerant: lines that do not fit the format are ignored, a
//! symbol without a location keeps `file`/`line`/`column` as `None`, and
//! Windows paths (`C:\...`) are handled by splitting the location from the
//! right. An inlined call site produces several frames with the same
//! `index`, exactly as the standard library prints them.

use std::backtrace::{Backtrace, BacktraceStatus};

/// One symbol of a captured backtrace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacktraceFrame {
    /// Frame index as printed by the standard library.
    pub index: usize,
    /// Demangled symbol name (`<unknown>` when unresolved).
    pub symbol: String,
    /// Source file, when debug info is available.
    pub file: Option<String>,
    /// 1-based source line, when available.
    pub line: Option<u32>,
    /// 1-based source column, when available.
    pub column: Option<u32>,
}

/// Capture the current stack unconditionally (independent of
/// `RUST_BACKTRACE`) and return its frames.
///
/// Returns an empty vector on platforms where backtraces are unsupported.
#[must_use]
pub fn capture_frames() -> Vec<BacktraceFrame> {
    frames_of(&Backtrace::force_capture())
}

/// Frames of an already captured backtrace; empty unless its status is
/// [`BacktraceStatus::Captured`].
#[must_use]
pub fn frames_of(backtrace: &Backtrace) -> Vec<BacktraceFrame> {
    if backtrace.status() == BacktraceStatus::Captured {
        // `{:#}` prints every frame (no short-backtrace trimming).
        parse_backtrace(&format!("{backtrace:#}"))
    } else {
        Vec::new()
    }
}

/// Parse the `Display` output of a [`Backtrace`] into frames.
#[must_use]
pub fn parse_backtrace(text: &str) -> Vec<BacktraceFrame> {
    let mut frames: Vec<BacktraceFrame> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(location) = trimmed.strip_prefix("at ") {
            if let Some(frame) = frames.last_mut().filter(|frame| frame.file.is_none()) {
                let (file, line, column) = parse_location(location);
                frame.file = Some(file);
                frame.line = line;
                frame.column = column;
            }
            continue;
        }

        let Some((index, symbol)) = trimmed.split_once(": ") else {
            continue;
        };
        let Ok(index) = index.parse::<usize>() else {
            continue;
        };
        frames.push(BacktraceFrame {
            index,
            symbol: symbol.trim().to_string(),
            file: None,
            line: None,
            column: None,
        });
    }

    frames
}

/// Split `path:line:column` (or `path:line`) from the right, so colons inside
/// the path (Windows drive letters) are preserved.
fn parse_location(location: &str) -> (String, Option<u32>, Option<u32>) {
    let location = location.trim();
    let mut parts = location.rsplitn(3, ':');
    let last = parts.next();
    let middle = parts.next();
    let rest = parts.next();

    match (rest, middle, last) {
        (Some(path), Some(line), Some(column)) => match (line.parse(), column.parse()) {
            (Ok(line), Ok(column)) => (path.to_string(), Some(line), Some(column)),
            // Only `path:line`, with a colon inside the path.
            (Err(_), Ok(line)) => (
                format!("{path}:{}", middle.unwrap_or_default()),
                Some(line),
                None,
            ),
            _ => (location.to_string(), None, None),
        },
        (None, Some(path), Some(line)) => match line.parse() {
            Ok(line) => (path.to_string(), Some(line), None),
            Err(_) => (location.to_string(), None, None),
        },
        _ => (location.to_string(), None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_backtrace_symbols_and_locations() {
        let text = "   0: my_crate::alloc_here\n\
                    \x20            at ./src/lib.rs:42:9\n\
                    \x20  1: <T as core::ops::FnOnce<()>>::call_once\n\
                    \x20            at /rustc/abc/library/core/src/ops/function.rs:250:5\n\
                    \x20  1: std::rt::lang_start_internal\n\
                    \x20  2: __libc_start_main\n\
                    \x20  3: <unknown>\n";
        let frames = parse_backtrace(text);
        assert_eq!(frames.len(), 5);
        assert_eq!(
            frames[0],
            BacktraceFrame {
                index: 0,
                symbol: "my_crate::alloc_here".to_string(),
                file: Some("./src/lib.rs".to_string()),
                line: Some(42),
                column: Some(9),
            }
        );
        assert_eq!(frames[1].symbol, "<T as core::ops::FnOnce<()>>::call_once");
        assert_eq!(frames[1].line, Some(250));
        // Inlined frames keep the printed index; a frame without a location
        // line has no file.
        assert_eq!(frames[2].index, 1);
        assert_eq!(frames[2].file, None);
        assert_eq!(frames[4].symbol, "<unknown>");
    }

    #[test]
    fn test_parse_location_variants() {
        assert_eq!(
            parse_location(r"C:\src\lib.rs:10:3"),
            (r"C:\src\lib.rs".to_string(), Some(10), Some(3))
        );
        assert_eq!(
            parse_location("src/lib.rs:7"),
            ("src/lib.rs".to_string(), Some(7), None)
        );
        assert_eq!(
            parse_location(r"C:\src\lib.rs:7"),
            (r"C:\src\lib.rs".to_string(), Some(7), None)
        );
        assert_eq!(parse_location("weird"), ("weird".to_string(), None, None));
    }

    #[test]
    fn test_parse_ignores_non_frame_text() {
        assert!(parse_backtrace("disabled backtrace").is_empty());
        assert!(parse_backtrace("unsupported backtrace").is_empty());
        assert!(parse_backtrace("").is_empty());
    }

    /// The capture path produces frames that name this very test when the
    /// platform can symbolize; it never panics either way.
    #[test]
    fn test_capture_frames_sees_caller() {
        let frames = capture_frames();
        if Backtrace::force_capture().status() == BacktraceStatus::Captured {
            assert!(!frames.is_empty());
            let symbolized = frames
                .iter()
                .any(|frame| frame.symbol.contains("test_capture_frames_sees_caller"));
            let unresolved = frames.iter().all(|frame| frame.symbol == "<unknown>");
            assert!(symbolized || unresolved, "unexpected frames: {frames:?}");
        } else {
            assert!(frames.is_empty());
        }
    }
}
