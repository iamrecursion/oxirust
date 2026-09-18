//! The insertion modes, one module per group.

mod body;
mod end;
mod head;
mod table;
mod template;

/// Splits a run of characters into its leading white space and the rest.
///
/// Several insertion modes ignore (or insert) leading white space and then
/// reprocess whatever follows, which a run-carrying character token has to do
/// by splitting.
pub(crate) fn split_whitespace_prefix(text: &str) -> (&str, &str) {
    let end = text
        .find(|ch: char| !matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' '))
        .unwrap_or(text.len());
    text.split_at(end)
}
