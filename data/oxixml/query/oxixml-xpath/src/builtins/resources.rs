//! `fn:doc`, `fn:collection` and `fn:unparsed-text` — the functions that reach outside the
//! query.
//!
//! [F&O 3.1 §14.5](https://www.w3.org/TR/xpath-functions-31/#func-doc) makes the set of
//! *available documents* implementation-defined, and a library has no business reading the
//! filesystem or the network on its caller's behalf. Nothing is available until the application
//! installs a [`DocumentResolver`](crate::DocumentResolver) on the dynamic context; until then
//! these functions raise `FODC0002`, which is what a processor reports for a resource it cannot
//! retrieve.

use super::string;
use crate::context::{DynamicContext, TextRetrievalError};
use crate::error::{XPathError, XPathErrorCode, XPathResult};
use crate::value::XpathValue;
use oxixml_infoset::AtomicValue;

/// Invokes one of the external-resource functions.
///
/// # Errors
///
/// [`XPathErrorCode::Fodc0002`] when no resolver is installed or the URI names nothing.
pub(crate) fn call(
    context: &DynamicContext,
    local: &str,
    arguments: &[XpathValue],
) -> XPathResult<XpathValue> {
    match local {
        "doc" => {
            let Some(uri) = optional_uri(&arguments[0])? else {
                return Ok(XpathValue::empty());
            };
            // F&O 3.1 §14.5.4: "If $uri is not a valid URI reference, an error is raised
            // [err:FODC0005]" — and that is decided before anything is retrieved, so a
            // malformed reference is not reported as a resource that happens to be missing.
            if !is_uri_reference(&uri) {
                return Err(XPathError::new(
                    XPathErrorCode::Fodc0005,
                    format!("`{uri}` is not a valid URI reference"),
                ));
            }
            let uri = absolute(context, &uri);
            let node = resolve(context, &uri)?;
            Ok(XpathValue::from(node))
        }
        "doc-available" => {
            let Some(uri) = optional_uri(&arguments[0])? else {
                return Ok(XpathValue::from(AtomicValue::boolean(false)));
            };
            let available = context
                .document_resolver()
                .and_then(|resolver| resolver.document(&uri))
                .is_some();
            Ok(XpathValue::from(AtomicValue::boolean(available)))
        }
        "collection" | "uri-collection" => {
            let uri = match arguments.first() {
                Some(argument) => optional_uri(argument)?,
                None => None,
            };
            let Some(resolver) = context.document_resolver() else {
                return Err(unavailable(uri.as_deref()));
            };
            // F&O 3.1 §14.6.6 resolves `$uri` against the static base URI before consulting the
            // available collections, exactly as `fn:doc` does; the unresolved form is tried
            // first so that a resolver keyed on the relative name still answers
            // (`collection-006`, `-007`).
            let items = resolver
                .collection_items(uri.as_deref())
                .or_else(|| {
                    let absolute = uri.as_deref().map(|uri| absolute(context, uri))?;
                    resolver.collection_items(Some(absolute.as_str()))
                })
                .ok_or_else(|| unavailable(uri.as_deref()))?;
            if local == "uri-collection" {
                // Only a node has a document URI; an atomic member of a collection contributes
                // nothing, which is why this is a filter and not an error.
                let uris: Vec<AtomicValue> = items
                    .iter()
                    .filter_map(|item| {
                        oxixml_infoset::XdmNode::document_uri(item.as_node()?)
                            .map(AtomicValue::any_uri)
                    })
                    .collect();
                return Ok(XpathValue::from_iter(uris));
            }
            Ok(XpathValue::from_items(
                items.into_iter().map(crate::value::XpathItem::from),
            ))
        }
        "unparsed-text" | "unparsed-text-lines" | "unparsed-text-available" => {
            let Some(uri) = require_text(&arguments[0], "the $href of fn:unparsed-text")? else {
                return Ok(if local == "unparsed-text-available" {
                    XpathValue::from(AtomicValue::boolean(false))
                } else {
                    XpathValue::empty()
                });
            };
            let uri = absolute(context, &uri);
            let encoding = match arguments.get(1) {
                Some(argument) => require_text(argument, "the $encoding of fn:unparsed-text")?,
                None => None,
            };
            if let Some(encoding) = &encoding {
                if !is_known_encoding(encoding) {
                    if local == "unparsed-text-available" {
                        return Ok(XpathValue::from(AtomicValue::boolean(false)));
                    }
                    return Err(XPathError::new(
                        XPathErrorCode::Fout1190,
                        format!("the encoding `{encoding}` is not supported"),
                    ));
                }
            }
            let text = retrieve_text(context, &uri, encoding.as_deref())
                .map_err(|reason| retrieval_error(&uri, reason))
                .and_then(|text| require_xml_characters(&uri, text));
            match local {
                "unparsed-text-available" => {
                    Ok(XpathValue::from(AtomicValue::boolean(text.is_ok())))
                }
                "unparsed-text-lines" => Ok(XpathValue::from_iter(
                    split_lines(&text?).into_iter().map(AtomicValue::string),
                )),
                _ => Ok(string(text?)),
            }
        }
        // `fn:json-doc` is `fn:parse-json` applied to a retrieved resource, and the retrieval is
        // what keeps it out of the function library.
        "json-doc" => {
            let Some(uri) = require_text(&arguments[0], "the $href of fn:json-doc")? else {
                return Ok(XpathValue::empty());
            };
            let uri = absolute(context, &uri);
            let text = retrieve_json(context, &uri).map_err(|reason| retrieval_error(&uri, reason));
            super::json::with_options(
                context,
                arguments.get(1),
                oxixml_fn::json::JsonFunction::ParseJson,
                |options| {
                    Ok(XpathValue::from(oxixml_fn::json::parse_json(
                        text.as_deref().map_err(Clone::clone)?,
                        options,
                    )?))
                },
            )
        }
        other => Err(XPathError::type_error(format!(
            "`fn:{other}` is not implemented"
        ))),
    }
}

/// Resolves a relative `$href` against the static base URI, which is what every one of these
/// functions is defined to do before it consults the available resources. A resolver that keyed
/// its resources on the relative form still sees it: the *unresolved* URI is tried first by the
/// caller, because `absolute` leaves the URI alone when there is no base to resolve against.
fn absolute(context: &DynamicContext, uri: &str) -> String {
    if uri.contains(':') {
        return uri.to_owned();
    }
    let Some(base) = context.static_context().base_uri() else {
        return uri.to_owned();
    };
    match oxixml_fn::strings::resolve_uri_against(uri, base) {
        Ok(resolved) => resolved,
        Err(_) => uri.to_owned(),
    }
}

/// Whether `name` is an encoding this processor recognises, for the `FOUT1190` check.
fn is_known_encoding(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    matches!(
        lowered.as_str(),
        "utf-8"
            | "utf8"
            | "utf-16"
            | "utf16"
            | "utf-16le"
            | "utf-16be"
            | "utf-32"
            | "utf-32le"
            | "utf-32be"
            | "us-ascii"
            | "ascii"
            | "iso-8859-1"
            | "latin1"
            | "windows-1252"
    )
}

/// Drops a leading byte-order mark, which is not part of the text a resource denotes.
fn strip_bom(text: &str) -> String {
    text.strip_prefix('\u{FEFF}').unwrap_or(text).to_owned()
}

fn resolve(context: &DynamicContext, uri: &str) -> XPathResult<oxixml_infoset::NodeRef> {
    context
        .document_resolver()
        .and_then(|resolver| resolver.document(uri))
        .ok_or_else(|| unavailable(Some(uri)))
}

fn optional_uri(value: &XpathValue) -> XPathResult<Option<String>> {
    Ok(value
        .atomize_at_most_one("a URI argument")?
        .map(|value| value.string_value()))
}

/// A `$href` or `$encoding` argument, which must be a string-typed value: `fn:unparsed-text`
/// declares them `xs:string?`, so a number is `XPTY0004` rather than a silent conversion.
fn require_text(value: &XpathValue, what: &str) -> XPathResult<Option<String>> {
    let Some(atomic) = value.atomize_at_most_one(what)? else {
        return Ok(None);
    };
    if !atomic.atomic_type().is_text() {
        return Err(XPathError::type_error(format!(
            "{what} must be an xs:string, not an {}",
            atomic.atomic_type()
        )));
    }
    Ok(Some(atomic.string_value()))
}

/// Retrieves a text resource through the context's resolver, keeping the reason for a failure.
fn retrieve_text(
    context: &DynamicContext,
    uri: &str,
    encoding: Option<&str>,
) -> Result<String, TextRetrievalError> {
    let Some(resolver) = context.document_resolver() else {
        return Err(TextRetrievalError::Unretrievable);
    };
    resolver
        .unparsed_text_result(uri, encoding)
        .map(|text| strip_bom(&text))
}

/// Retrieves the resource behind `fn:json-doc`, which is *not* the same retrieval as
/// `fn:unparsed-text`'s: see [`DocumentResolver::json_text_result`].
fn retrieve_json(context: &DynamicContext, uri: &str) -> Result<String, TextRetrievalError> {
    let Some(resolver) = context.document_resolver() else {
        return Err(TextRetrievalError::Unretrievable);
    };
    resolver.json_text_result(uri).map(|text| strip_bom(&text))
}

/// Applies [F&O 3.1 §14.8.1](https://www.w3.org/TR/xpath-functions-31/#func-unparsed-text)'s
/// rule that the characters a text resource decodes to must be permitted in XML.
///
/// The check belongs to `fn:unparsed-text` and not to the resolver: `fn:json-doc` retrieves
/// through the same door and is *not* subject to it — JSON may carry ` ` through an escape,
/// and `fn:parse-json`'s own `fallback` option exists precisely to deal with it.
fn require_xml_characters(uri: &str, text: String) -> XPathResult<String> {
    match text
        .chars()
        .find(|character| !oxixml_fn::json::parser::is_xml_character(*character))
    {
        None => Ok(text),
        Some(character) => Err(XPathError::new(
            XPathErrorCode::Fout1190,
            format!(
                "the text resource at `{uri}` contains U+{:04X}, which is not a permitted XML character",
                character as u32
            ),
        )),
    }
}

/// Maps a retrieval failure onto the `FOUT` code F&O 3.1 §14.8.2 gives it.
fn retrieval_error(uri: &str, reason: TextRetrievalError) -> XPathError {
    match reason {
        TextRetrievalError::UnsupportedEncoding => XPathError::new(
            XPathErrorCode::Fout1190,
            format!("the encoding of the text resource at `{uri}` is not supported"),
        ),
        TextRetrievalError::InvalidOctets => XPathError::new(
            XPathErrorCode::Fout1200,
            format!("the text resource at `{uri}` cannot be decoded in its encoding"),
        ),
        _ => XPathError::new(
            XPathErrorCode::Fout1170,
            format!("no text resource is available at `{uri}`"),
        ),
    }
}

/// Whether `text` has the shape of an RFC 3986 `URI-reference`, as far as `fn:doc` has to decide.
///
/// A full grammar is not needed and would be actively unhelpful: the set of *available
/// documents* is implementation-defined, so a resolver is free to key itself on names a strict
/// reading would reject, and rejecting those would turn "this document is not there" into a
/// syntax error. What is checked is the one part of the syntax that can never be a name at all:
/// a reference whose first segment contains a `:` is an absolute URI, and its `scheme` must then
/// be `ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )`. That is what makes `:/` — a colon with no
/// scheme in front of it — invalid rather than merely unavailable.
fn is_uri_reference(text: &str) -> bool {
    let Some(colon) = text.find(':') else {
        return true;
    };
    // A `:` after the first `/`, `?` or `#` belongs to a path, query or fragment, where it is an
    // ordinary character.
    if text[..colon].contains(['/', '?', '#']) {
        return true;
    }
    let mut characters = text[..colon].chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
}

fn unavailable(uri: Option<&str>) -> XPathError {
    XPathError::new(
        XPathErrorCode::Fodc0002,
        match uri {
            Some(uri) => format!("no resource is available at `{uri}`"),
            None => "no default collection is available".to_owned(),
        },
    )
}

/// Splits on `\r\n`, `\r` or `\n`, discarding a single trailing line terminator, as
/// `fn:unparsed-text-lines` requires.
fn split_lines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\r' => {
                if characters.peek() == Some(&'\n') {
                    characters.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            '\n' => lines.push(std::mem::take(&mut current)),
            other => current.push(other),
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{DocumentResolver, StaticContext};
    use oxixml_infoset::{NodeRef, Tree};
    use std::sync::Arc;

    #[derive(Debug)]
    struct Fixture {
        tree: Tree,
    }

    impl DocumentResolver for Fixture {
        fn document(&self, uri: &str) -> Option<NodeRef> {
            (uri == "urn:doc").then(|| self.tree.root())
        }

        fn collection(&self, uri: Option<&str>) -> Option<Vec<NodeRef>> {
            uri.is_none().then(|| vec![self.tree.root()])
        }

        fn unparsed_text(&self, uri: &str, _encoding: Option<&str>) -> Option<String> {
            (uri == "urn:text").then(|| "one\r\ntwo\n".to_owned())
        }
    }

    fn context() -> XPathResult<DynamicContext> {
        let tree = Tree::parse_str("<a/>")?;
        Ok(DynamicContext::new(Arc::new(StaticContext::new()))
            .with_document_resolver(Arc::new(Fixture { tree })))
    }

    #[test]
    fn without_a_resolver_everything_is_fodc0002() -> XPathResult<()> {
        let context = DynamicContext::new(Arc::new(StaticContext::new()));
        let failure = call(&context, "doc", &[string("urn:doc")]).expect_err("no resolver");
        assert_eq!(failure.code(), Some(XPathErrorCode::Fodc0002));
        assert_eq!(
            call(&context, "doc-available", &[string("urn:doc")])?.to_string(),
            "false"
        );
        Ok(())
    }

    #[test]
    fn a_resolver_supplies_documents_and_text() -> XPathResult<()> {
        let context = context()?;
        assert_eq!(call(&context, "doc", &[string("urn:doc")])?.len(), 1);
        assert_eq!(
            call(&context, "doc-available", &[string("urn:doc")])?.to_string(),
            "true"
        );
        assert!(call(&context, "doc", &[string("urn:missing")]).is_err());
        assert_eq!(call(&context, "collection", &[])?.len(), 1);
        assert_eq!(
            call(&context, "unparsed-text", &[string("urn:text")])?.to_string(),
            "one\r\ntwo\n"
        );
        assert_eq!(
            call(&context, "unparsed-text-lines", &[string("urn:text")])?.len(),
            2
        );
        Ok(())
    }

    #[test]
    fn lines_split_on_every_terminator() {
        assert_eq!(split_lines("a\nb\r\nc\rd"), ["a", "b", "c", "d"]);
        assert_eq!(split_lines("a\n"), ["a"]);
        assert!(split_lines("").is_empty());
    }
}
