//! The [JSON-LD 1.1 Streaming](https://www.w3.org/TR/json-ld11-streaming/) profile: the key-order
//! constraint a *streaming document* must satisfy.
//!
//! A streaming processor sees each map entry once, in document order, so a document written for
//! the streaming profile must present the entries that change how later entries are interpreted
//! *before* those entries. Inside every node object:
//!
//! 1. `@context` must precede every other entry; and
//! 2. `@type` must precede every entry other than `@context`.
//!
//! `@id` carries no ordering requirement of its own — a streaming processor can buffer a node's
//! statements until its subject is known — and value objects are unconstrained.
//!
//! The check runs *during expansion*, so keyword aliases (`"type": {"@id": "@type"}`) are ranked by
//! what they expand to, not by their spelling. Documents that break the ordering are rejected with
//! [`JsonLdErrorCode::InvalidStreamingKeyOrder`].
//!
//! This crate's parser buffers the whole document, so requesting the profile changes *validation*
//! only: the RDF produced for a well-ordered document is identical either way.

use crate::error::{JsonLdErrorCode, JsonLdSyntaxError};

/// The position an entry must occupy in a streaming node object; lower sorts earlier.
///
/// `expanded_key` is the result of IRI-expanding the entry's key, so aliases rank with the keyword
/// they alias. Anything that is not `@context` or `@type` — including `@id` — shares the last rank
/// and may therefore appear in any order among itself.
#[must_use]
pub(crate) fn key_rank(key: &str, expanded_key: Option<&str>) -> u8 {
    if key == "@context" {
        return 0;
    }
    match expanded_key {
        Some("@type") => 1,
        _ => 2,
    }
}

/// Builds the error raised when `key` appears too late in a streaming node object.
#[must_use]
pub(crate) fn key_order_error(key: &str) -> JsonLdSyntaxError {
    JsonLdSyntaxError::msg(
        format!("{key} must precede the entries it follows in the streaming document form"),
        JsonLdErrorCode::InvalidStreamingKeyOrder,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_ranks_first_and_type_second() {
        assert_eq!(key_rank("@context", Some("@context")), 0);
        assert_eq!(key_rank("type", Some("@type")), 1);
        assert_eq!(key_rank("@type", Some("@type")), 1);
        assert_eq!(key_rank("@id", Some("@id")), 2);
        assert_eq!(key_rank("name", Some("http://example.com/name")), 2);
    }

    #[test]
    fn the_error_carries_the_specification_code() {
        assert_eq!(
            key_order_error("@type").code(),
            Some(JsonLdErrorCode::InvalidStreamingKeyOrder)
        );
    }
}
