//! The `join-URI-References` function used for `xml:base` fixup.
//!
//! [Canonical XML 1.1 §2.4](https://www.w3.org/TR/xml-c14n11/#DocSubsets)
//! introduces `xml:base` fixup: when an element carrying `xml:base` is dropped
//! from a document subset, the base URI it contributed has to be folded into the
//! `xml:base` of the nearest surviving descendant, so that the canonical form
//! still resolves relative references the same way.
//!
//! The folding is done by *join-URI-References*, "a simple method … similar to
//! that found in sections 5.2.1, 5.2.2 and 5.2.4 of
//! [RFC 3986](https://www.rfc-editor.org/rfc/rfc3986)" with four modifications:
//!
//! 1. the scheme of the base is optional — `xml:base` values are usually
//!    relative paths;
//! 2. a trailing `..` segment of the base becomes `../` before processing;
//! 3. *Remove Dot Segments* keeps leading `../` segments, collapses runs of `/`
//!    into one, and appends `/` to a trailing `..` segment, so that joining two
//!    relative paths yields a relative path again;
//! 4. the fragment of the reference is discarded.
//!
//! The Recommendation gives three worked results, all of which are covered by
//! the tests at the bottom of this module: `"abc/"` ∘ `"../"` is `""`, `"../"` ∘
//! `"../"` is `"../../"`, and `".."` ∘ `".."` is `"../../"`.

/// The pieces of a URI reference, per RFC 3986 §3.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct UriReference<'a> {
    scheme: Option<&'a str>,
    authority: Option<&'a str>,
    path: &'a str,
    query: Option<&'a str>,
}

impl<'a> UriReference<'a> {
    /// Splits a URI reference into its components, dropping the fragment.
    fn parse(text: &'a str) -> Self {
        let text = text.split_once('#').map_or(text, |(before, _)| before);
        let (text, query) = match text.split_once('?') {
            Some((before, after)) => (before, Some(after)),
            None => (text, None),
        };
        let (scheme, rest) = split_scheme(text);
        let (authority, path) = match rest.strip_prefix("//") {
            Some(after) => {
                let end = after.find(['/', '?', '#']).unwrap_or(after.len());
                (Some(&after[..end]), &after[end..])
            }
            None => (None, rest),
        };
        Self {
            scheme,
            authority,
            path,
            query,
        }
    }

    /// Recomposes the reference per RFC 3986 §5.3, always without a fragment.
    fn recompose(
        scheme: Option<&str>,
        authority: Option<&str>,
        path: &str,
        query: Option<&str>,
    ) -> String {
        let mut out = String::new();
        if let Some(scheme) = scheme {
            out.push_str(scheme);
            out.push(':');
        }
        if let Some(authority) = authority {
            out.push_str("//");
            out.push_str(authority);
        }
        out.push_str(path);
        if let Some(query) = query {
            out.push('?');
            out.push_str(query);
        }
        out
    }
}

/// Splits off an RFC 3986 `scheme ":"` prefix, if the text has one.
fn split_scheme(text: &str) -> (Option<&str>, &str) {
    let mut end = None;
    for (index, ch) in text.char_indices() {
        if ch == ':' {
            end = Some(index);
            break;
        }
        if index == 0 {
            if !ch.is_ascii_alphabetic() {
                return (None, text);
            }
        } else if !(ch.is_ascii_alphanumeric() || ch == '+' || ch == '-' || ch == '.') {
            return (None, text);
        }
    }
    match end {
        Some(0) | None => (None, text),
        Some(end) => (Some(&text[..end]), &text[end + 1..]),
    }
}

/// Joins a base `xml:base` value with a relative one, per Canonical XML 1.1.
///
/// Both arguments come from `xml:base` attributes, so either may be a full URI,
/// a path, or a bare relative reference.
#[must_use]
pub fn join_uri_references(base: &str, reference: &str) -> String {
    let reference = UriReference::parse(reference);

    if reference.scheme.is_some() {
        return UriReference::recompose(
            reference.scheme,
            reference.authority,
            &remove_dot_segments(reference.path),
            reference.query,
        );
    }

    // §5.2.1 pre-parse of the base, with the trailing "`..` becomes `../`" fix.
    let base_owned;
    let base_text = if base.ends_with("..") && (base.len() == 2 || base.ends_with("/..")) {
        base_owned = format!("{base}/");
        base_owned.as_str()
    } else {
        base
    };
    let base = UriReference::parse(base_text);

    if reference.authority.is_some() {
        return UriReference::recompose(
            base.scheme,
            reference.authority,
            &remove_dot_segments(reference.path),
            reference.query,
        );
    }

    if reference.path.is_empty() {
        let query = reference.query.or(base.query);
        return UriReference::recompose(base.scheme, base.authority, base.path, query);
    }

    let path = if reference.path.starts_with('/') {
        remove_dot_segments(reference.path)
    } else {
        remove_dot_segments(&merge(&base, reference.path))
    };
    UriReference::recompose(base.scheme, base.authority, &path, reference.query)
}

/// RFC 3986 §5.2.3 *Merge Paths*.
fn merge(base: &UriReference<'_>, reference_path: &str) -> String {
    if base.authority.is_some() && base.path.is_empty() {
        return format!("/{reference_path}");
    }
    match base.path.rfind('/') {
        Some(slash) => format!("{}{reference_path}", &base.path[..=slash]),
        None => reference_path.to_owned(),
    }
}

/// RFC 3986 §5.2.4 *Remove Dot Segments*, with the three Canonical XML 1.1
/// modifications described in the [module documentation](self).
fn remove_dot_segments(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut trailing_slash = path.ends_with('/');

    let segments: Vec<&str> = path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    let mut kept: Vec<&str> = Vec::with_capacity(segments.len());
    let mut leading_parents = 0_usize;

    for (index, segment) in segments.iter().enumerate() {
        let is_last = index + 1 == segments.len();
        match *segment {
            "." => {
                if is_last {
                    trailing_slash = true;
                }
            }
            ".." => {
                if kept.pop().is_none() && !absolute {
                    leading_parents += 1;
                }
                if is_last {
                    trailing_slash = true;
                }
            }
            other => kept.push(other),
        }
    }

    let mut out = String::with_capacity(path.len());
    if absolute {
        out.push('/');
    }
    for _ in 0..leading_parents {
        out.push_str("../");
    }
    for (index, segment) in kept.iter().enumerate() {
        if index > 0 {
            out.push('/');
        }
        out.push_str(segment);
    }
    if trailing_slash && !kept.is_empty() {
        out.push('/');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three results spelled out in Canonical XML 1.1 §2.4.
    #[test]
    fn recommendation_examples() {
        assert_eq!(join_uri_references("abc/", "../"), "");
        assert_eq!(join_uri_references("../", "../"), "../../");
        assert_eq!(join_uri_references("..", ".."), "../../");
    }

    /// The `<a xml:base="foo/bar"><b xml:base=".."><c xml:base=".."><d
    /// xml:base="x">` walk-through of Canonical XML 1.1 §2.4: removing `b` and
    /// `c` must leave `d` with `xml:base="../../x"`.
    #[test]
    fn nested_parent_segments() {
        let inner = join_uri_references("..", "x");
        assert_eq!(inner, "../x");
        assert_eq!(join_uri_references("..", &inner), "../../x");
    }

    /// The chain used by the §3.8 example.
    #[test]
    fn document_subset_example() {
        assert_eq!(join_uri_references("bar/", "foo"), "bar/foo");
        assert_eq!(
            join_uri_references("something/else", "bar/foo"),
            "something/bar/foo"
        );
    }

    #[test]
    fn absolute_references_win() {
        assert_eq!(
            join_uri_references("http://example.com/a/b", "http://other.example/c"),
            "http://other.example/c"
        );
        assert_eq!(
            join_uri_references("http://example.com/a/b", "/c/d"),
            "http://example.com/c/d"
        );
        assert_eq!(
            join_uri_references("http://example.com/a/b", "c/d"),
            "http://example.com/a/c/d"
        );
    }

    #[test]
    fn fragments_are_discarded() {
        assert_eq!(join_uri_references("a/b", "c#frag"), "a/c");
    }

    #[test]
    fn runs_of_slashes_collapse() {
        assert_eq!(remove_dot_segments("a//b///c"), "a/b/c");
    }

    #[test]
    fn empty_reference_keeps_the_base() {
        assert_eq!(join_uri_references("a/b?q", ""), "a/b?q");
    }

    #[test]
    fn scheme_detection() {
        assert_eq!(split_scheme("http://x"), (Some("http"), "//x"));
        assert_eq!(split_scheme("foo/bar"), (None, "foo/bar"));
        assert_eq!(split_scheme("1http:x"), (None, "1http:x"));
        assert_eq!(split_scheme(":x"), (None, ":x"));
    }
}
