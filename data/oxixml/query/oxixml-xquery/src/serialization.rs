//! The serialization parameters a prolog's *output declarations* set.
//!
//! [XQuery 3.1 §2.2.4](https://www.w3.org/TR/xquery-31/#id-serialization-declarations) lets a
//! main module set a serialization parameter with `declare option output:name "value"`, where
//! `output` is bound to the serialization namespace. The engine itself never serializes — the
//! result of a query is a sequence, and what a host does with it is the host's business — so the
//! declarations are validated while compiling (`XQST0108`, `XQST0109`, `XQST0110`) and then kept
//! for whoever does serialize.
//!
//! Reading them back out of the query *text* is what a caller had to do before
//! [`CompiledQuery::serialization_parameters`](crate::CompiledQuery::serialization_parameters)
//! existed, and it cannot be done correctly without re-implementing the prolog grammar: an option
//! name is an `EQName` whose prefix is bound by the same prolog, a declared value is a string
//! literal that may carry character and entity references, and both may be separated by nested
//! comments.
//!
//! The declared values are turned into an [`oxixml_fn::xmlio::SerializationParameters`] by
//! Serialization 3.1 §3.1's own route — an `output:serialization-parameters` element handed to
//! [`parameters_from_element_in_context`] — rather than by mapping each name onto a setter here.
//! That keeps one implementation of what a parameter's value may be, and it is what lets a
//! QName-valued parameter such as `cdata-section-elements` resolve its prefixes: the element
//! carries the query's own namespace bindings. The one thing it cannot carry is the module's
//! `declare default element namespace`, which travels beside it.

use crate::context::ERROR_NAMESPACE;
use crate::error::{XQueryError, XQueryResult};
use oxixml_fn::xmlio::{
    SERIALIZATION_NAMESPACE, SerializationParameters, parameters_from_element_in_context,
};
use oxixml_infoset::{ExpandedName, Tree};
use std::collections::BTreeMap;

/// `SEPM0017`, which Serialization 3.1 §3.1 raises for a parameter document that cannot be read.
///
/// The code is not one [`XQueryErrorCode`](crate::XQueryErrorCode) enumerates — it belongs to the
/// serialization specification rather than to XQuery — so it is raised by name, exactly as the
/// codes reaching this crate from [`oxixml_fn`] are.
fn invalid_parameters(message: impl Into<String>) -> XQueryError {
    XQueryError::with_name(
        ExpandedName::new(Some(ERROR_NAMESPACE), "SEPM0017"),
        message,
    )
}

/// The `declare option` declarations of `options` that are in the serialization namespace, as
/// `(parameter name, declared value)` pairs in declaration order.
///
/// `namespaces` is the module's in-scope prefix map, which is what an option's `EQName` prefix
/// is resolved against. An option in any other namespace is not an output declaration at all and
/// is left out; §4.16 lets a processor ignore an option it does not recognize.
pub(crate) fn output_declarations(
    options: &[(String, String)],
    namespaces: &BTreeMap<String, String>,
) -> Vec<(String, String)> {
    options
        .iter()
        .filter_map(|(lexical, value)| {
            let (namespace, local) = expanded(lexical, namespaces)?;
            (namespace == SERIALIZATION_NAMESPACE).then(|| (local, value.clone()))
        })
        .collect()
}

/// Splits an option's `EQName` into `(namespace, local name)`, or `None` when it has no prefix
/// (an unprefixed option name is in no namespace, so it is never an output declaration) or when
/// its prefix is not bound.
fn expanded(lexical: &str, namespaces: &BTreeMap<String, String>) -> Option<(String, String)> {
    if let Some(rest) = lexical.strip_prefix("Q{") {
        let (uri, local) = rest.split_once('}')?;
        return Some((uri.to_owned(), local.to_owned()));
    }
    let (prefix, local) = lexical.split_once(':')?;
    let uri = namespaces.get(prefix)?;
    Some((uri.clone(), local.to_owned()))
}

/// Resolves `declarations` into the parameter set they describe, over
/// [`SerializationParameters::default()`] — which is `fn:serialize`'s default set, agreeing with
/// Serialization 3.1 §3 except that `omit-xml-declaration` starts at `yes`.
///
/// `default_element_namespace` is the module's `declare default element namespace`, which is
/// what an *unprefixed* name in `cdata-section-elements` or `suppress-indentation` denotes: those
/// two parameters carry lexical `QName`s, and a name without a prefix is a name in the default
/// element namespace of the context it was written in rather than a name in no namespace. The
/// declaration cannot travel in the parameter document — an empty prefix would have to be written
/// `xmlns:=""`, which is not a namespace declaration at all — so it is passed alongside it.
/// `None` means the module declared none, and an unprefixed name is then in no namespace.
///
/// # Errors
///
/// `SEPM0017` when a declared value is not admissible for its parameter, or when the declarations
/// cannot be expressed as a parameter document at all. That is the code Serialization 3.1 §3.1
/// gives a fault inside an `output:serialization-parameters` element, and the values are checked
/// by that element's own reader rather than by a second implementation here — so the diagnostic
/// is the one `fn:serialize` gives the same value.
pub(crate) fn resolve(
    declarations: &[(String, String)],
    namespaces: &BTreeMap<String, String>,
    default_element_namespace: Option<&str>,
) -> XQueryResult<SerializationParameters> {
    if declarations.is_empty() {
        return Ok(SerializationParameters::default());
    }
    let document = parameters_document(declarations, namespaces)?;
    let tree = Tree::parse_str(&document)?;
    let element = tree.document_element().ok_or_else(|| {
        invalid_parameters("the output declarations do not form a serialization parameter document")
    })?;
    Ok(parameters_from_element_in_context(
        &element,
        &SerializationParameters::default(),
        default_element_namespace,
    )?)
}

/// The `output:serialization-parameters` element of Serialization 3.1 §3.1 that carries
/// `declarations`, with every namespace the module bound in scope on it.
///
/// The module's *default* namespace is deliberately not among them — see
/// [`resolve`]'s `default_element_namespace`.
fn parameters_document(
    declarations: &[(String, String)],
    namespaces: &BTreeMap<String, String>,
) -> XQueryResult<String> {
    let mut document = format!("<serialization-parameters xmlns=\"{SERIALIZATION_NAMESPACE}\"");
    for (prefix, uri) in namespaces {
        // `xml` is bound by definition and `xmlns` may not be re-bound; an empty URI would be
        // an undeclaration, which the module's own bindings never are.
        if prefix == "xml" || prefix == "xmlns" || prefix.is_empty() || uri.is_empty() {
            continue;
        }
        document.push_str(" xmlns:");
        document.push_str(prefix);
        document.push_str("=\"");
        document.push_str(&escape(uri)?);
        document.push('"');
    }
    document.push('>');
    for (parameter, value) in declarations {
        document.push('<');
        document.push_str(parameter);
        document.push_str(" value=\"");
        document.push_str(&escape(value)?);
        document.push_str("\"/>");
    }
    document.push_str("</serialization-parameters>");
    Ok(document)
}

/// `text` as an XML attribute value.
///
/// Whitespace is written as a character reference rather than literally, because attribute-value
/// normalization would otherwise turn the tab of `declare option output:item-separator "&#9;"`
/// into a space.
fn escape(text: &str) -> XQueryResult<String> {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\t' | '\n' | '\r' => escaped.push_str(&format!("&#{};", character as u32)),
            other if crate::lexer::is_xml_char(other as u32) => escaped.push(other),
            other => {
                return Err(invalid_parameters(format!(
                    "the declared value holds U+{:04X}, which no serialization parameter may \
                     carry",
                    other as u32
                )));
            }
        }
    }
    Ok(escaped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_fn::xmlio::OutputMethod;

    fn namespaces() -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        map.insert("output".to_owned(), SERIALIZATION_NAMESPACE.to_owned());
        map.insert("p".to_owned(), "urn:p".to_owned());
        map
    }

    #[test]
    fn only_options_in_the_serialization_namespace_are_declarations() {
        let options = vec![
            ("output:method".to_owned(), "html".to_owned()),
            ("p:thing".to_owned(), "x".to_owned()),
            ("bare".to_owned(), "x".to_owned()),
            ("unbound:thing".to_owned(), "x".to_owned()),
            (
                format!("Q{{{SERIALIZATION_NAMESPACE}}}indent"),
                "yes".to_owned(),
            ),
        ];
        assert_eq!(
            output_declarations(&options, &namespaces()),
            vec![
                ("method".to_owned(), "html".to_owned()),
                ("indent".to_owned(), "yes".to_owned())
            ]
        );
    }

    #[test]
    fn declared_values_reach_the_parameter_set() -> XQueryResult<()> {
        let declarations = vec![
            ("method".to_owned(), "html".to_owned()),
            ("indent".to_owned(), "yes".to_owned()),
            ("version".to_owned(), "5.0".to_owned()),
            ("item-separator".to_owned(), "\t".to_owned()),
        ];
        let parameters = resolve(&declarations, &namespaces(), None)?;
        assert_eq!(parameters.method(), OutputMethod::Html);
        assert!(parameters.indent());
        assert_eq!(parameters.version(), "5.0");
        assert_eq!(parameters.item_separator(), Some("\t".to_owned()));
        // A parameter the prolog does not declare keeps `fn:serialize`'s default rather than
        // Serialization 3.1 §3's, which differ over exactly this one.
        assert!(parameters.omit_xml_declaration());
        Ok(())
    }

    #[test]
    fn a_qname_valued_parameter_resolves_against_the_modules_prefixes() -> XQueryResult<()> {
        let declarations = vec![("cdata-section-elements".to_owned(), "p:a p:b".to_owned())];
        let parameters = resolve(&declarations, &namespaces(), None)?;
        assert_eq!(
            parameters.cdata_section_elements(),
            [
                ExpandedName::new(Some("urn:p"), "a"),
                ExpandedName::new(Some("urn:p"), "b")
            ]
        );
        Ok(())
    }

    #[test]
    fn a_markup_bearing_value_survives_the_round_trip() -> XQueryResult<()> {
        let declarations = vec![("doctype-system".to_owned(), "a<b&c\"d".to_owned())];
        assert_eq!(
            resolve(&declarations, &namespaces(), None)?.doctype_system(),
            Some("a<b&c\"d".to_owned())
        );
        Ok(())
    }

    #[test]
    fn nothing_declared_is_the_default_parameter_set() -> XQueryResult<()> {
        assert_eq!(
            resolve(&[], &namespaces(), None)?,
            SerializationParameters::default()
        );
        Ok(())
    }

    /// QT3 `method-xml/K2-Serialization-28` and `-31`: an unprefixed name in one of the two
    /// QName-valued parameters is a name in the module's *default element namespace*, which no
    /// part of the parameter document can carry, and a name in no namespace only when the module
    /// declared none.
    #[test]
    fn an_unprefixed_qname_takes_the_modules_default_element_namespace() -> XQueryResult<()> {
        let declarations = vec![
            ("cdata-section-elements".to_owned(), "b p:a".to_owned()),
            ("suppress-indentation".to_owned(), "p para".to_owned()),
        ];

        let declared = resolve(&declarations, &namespaces(), Some("urn:default"))?;
        assert_eq!(
            declared.cdata_section_elements(),
            [
                ExpandedName::new(Some("urn:default"), "b"),
                ExpandedName::new(Some("urn:p"), "a"),
            ],
            "only the unprefixed name moves"
        );
        assert_eq!(
            declared.suppress_indentation(),
            [
                ExpandedName::new(Some("urn:default"), "p"),
                ExpandedName::new(Some("urn:default"), "para"),
            ]
        );

        let undeclared = resolve(&declarations, &namespaces(), None)?;
        assert_eq!(
            undeclared.cdata_section_elements(),
            [
                ExpandedName::new(None::<String>, "b"),
                ExpandedName::new(Some("urn:p"), "a"),
            ],
            "a module that declares none leaves an unprefixed name in no namespace"
        );
        Ok(())
    }

    #[test]
    fn an_inadmissible_value_is_reported_rather_than_ignored() {
        let declarations = vec![("indent".to_owned(), "maybe".to_owned())];
        let failure =
            resolve(&declarations, &namespaces(), None).expect_err("`maybe` is not yes/no");
        assert_eq!(failure.name().local_name(), "SEPM0017");
    }
}
