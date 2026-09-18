//! The serialization parameters a QT3 case's query declares in its prolog.
//!
//! `serialization-matches` asserts a regular expression against the *serialized* result, and
//! `catalog-schema.xsd` is explicit about which serialization: "the result of the query must be
//! serialized using the serialization options specified within the query (if any)". Those
//! options are the `declare option output:…` declarations of XQuery 3.1 §2.2.4, and
//! [`oxixml_xquery`] deliberately leaves them to its caller — it records an output declaration
//! while compiling (so `XQST0108`/`XQST0109`/`XQST0110` are still raised) but never applies one,
//! because nothing in the engine serializes. Applying them is therefore this harness's job, and
//! this module is the half of it that reads them back out of the query text.
//!
//! The reading is a small tokenizer rather than a regular expression, because the two shapes
//! that have to be recognized — `declare namespace ncname = "uri"` and `declare option eqname
//! "value"` — are only meaningful once XQuery comments and string literals have been taken out
//! of the way, and because a string literal may carry the XML character and entity references
//! that [XQuery 3.1 §3.1.1](https://www.w3.org/TR/xquery-31/#id-literals) admits inside one.
//!
//! # When a case cannot be decided
//!
//! An assertion this harness cannot *reproduce* is not one the engine failed: if the query asks
//! for a serialization feature [`oxixml_fn::xmlio`] does not provide, the text to compare
//! against was never produced, so [`declared_parameters`] answers `None` and the caller skips
//! the case rather than reporting a failure the engine did not earn. Two things are left that
//! qualify: a `parameter-document`, which points at parameters held in a resource only the
//! *caller* can resolve, and an `encoding` other than UTF-8 — the serializer's result is a Rust
//! `String`, so a narrower repertoire's escaping is not something it can apply.

use oxixml_fn::xmlio::{
    SERIALIZATION_NAMESPACE, SerializationParameters, parameters_from_element_in_context,
};
use oxixml_infoset::Tree;

/// Serialization parameters whose declared value changes the serialized text and which this
/// harness cannot supply the effect of. A query that sets one asks for a serialization this
/// harness cannot produce — see the module documentation.
///
/// Only `parameter-document` qualifies: [`oxixml_fn::xmlio::serialize`] implements every other
/// parameter Serialization 3.1 defines, `parameter-document` included in the sense that it
/// *reads* one — but the URI it names is resolved by the calling application, and this harness
/// has no mapping from a catalogue-relative URI to the parameter set behind it.
const UNREPRODUCIBLE_PARAMETERS: &[&str] = &["parameter-document"];

/// The serialization parameters `query` declares, or `None` when they are not ones this harness
/// can apply.
///
/// Whatever the query declares is read *over* [`SerializationParameters::serialization_defaults`]
/// rather than over `fn:serialize`'s set: the text a `serialization-matches` is asserted against
/// is the result of serializing the query's own result, which XQuery 3.1 §2.2.4 hands to the
/// Serialization specification with its §3 defaults in force — where `omit-xml-declaration` is
/// `no`. `fn:serialize`'s `yes` belongs to that function, not to the host language, and using it
/// here suppressed the XML declaration four `method-xml` cases assert the presence of.
///
/// A query with no output declaration at all therefore serializes with those defaults, which is
/// what makes the `ser/` cases that declare nothing decidable.
#[must_use]
pub(crate) fn declared_parameters(query: &str) -> Option<SerializationParameters> {
    // Every case in the catalogue comes through here, and only a few hundred declare anything:
    // a query with no `option` anywhere in it — comments and literals included — cannot hold an
    // option declaration, so it does not need tokenizing to establish that.
    if !query.contains("option") {
        return Some(SerializationParameters::serialization_defaults());
    }
    let tokens = tokenize(query)?;
    let prefixes = namespace_declarations(&tokens);
    let mut declared: Vec<(String, String)> = Vec::new();
    for (name, value) in option_declarations(&tokens) {
        let Some(parameter) = serialization_parameter(&name, &prefixes) else {
            continue; // an option in another namespace, which §4.16 lets a processor ignore
        };
        if UNREPRODUCIBLE_PARAMETERS.contains(&parameter.as_str()) {
            return None;
        }
        if parameter == "encoding" && !value.trim().eq_ignore_ascii_case("utf-8") {
            return None;
        }
        declared.push((parameter, value));
    }
    if declared.is_empty() {
        return Some(SerializationParameters::serialization_defaults());
    }
    // The declarations are handed to `oxixml_fn`'s own parameter-document reader rather than
    // mapped name by name here: it already validates every value, and building an element lets
    // the *query's* namespace bindings come with them, which a `cdata-section-elements` (or any
    // other QName-valued) parameter needs to resolve its prefixes against. The query's default
    // element namespace travels beside them rather than on the element, because a namespace
    // *declaration* on the parameter document would bind the parameter names themselves.
    let tree = Tree::parse_str(&parameters_document(&prefixes, &declared)?).ok()?;
    parameters_from_element_in_context(
        &tree.document_element()?,
        &SerializationParameters::serialization_defaults(),
        default_element_namespace(&tokens).as_deref(),
    )
    .ok()
}

/// The `output:serialization-parameters` element of Serialization 3.1 §3.1 that carries
/// `declared`, with every namespace the query bound in scope on it.
///
/// `None` when a declared value holds a character XML cannot carry, which no serialization
/// parameter admits anyway.
fn parameters_document(
    prefixes: &[(String, String)],
    declared: &[(String, String)],
) -> Option<String> {
    let mut document = format!("<serialization-parameters xmlns=\"{SERIALIZATION_NAMESPACE}\"");
    for (prefix, uri) in prefixes {
        if prefix == "xml" || prefix == "xmlns" || uri.is_empty() {
            continue;
        }
        document.push_str(&format!(" xmlns:{prefix}=\"{}\"", escape(uri)?));
    }
    document.push('>');
    for (parameter, value) in declared {
        document.push_str(&format!("<{parameter} value=\"{}\"/>", escape(value)?));
    }
    document.push_str("</serialization-parameters>");
    Some(document)
}

/// `text` as an XML attribute value, with the whitespace written as character references so
/// attribute-value normalization does not turn it into spaces. `None` for a character XML 1.0
/// cannot represent.
fn escape(text: &str) -> Option<String> {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\t' | '\n' | '\r' => escaped.push_str(&format!("&#{};", character as u32)),
            control if (control as u32) < 0x20 => return None,
            other => escaped.push(other),
        }
    }
    Some(escaped)
}

/// The local name of the serialization parameter `name` (an option declaration's `EQName`)
/// sets, or `None` when the option is not in the serialization namespace.
fn serialization_parameter(name: &str, prefixes: &[(String, String)]) -> Option<String> {
    if let Some(rest) = name.strip_prefix("Q{") {
        let (uri, local) = rest.split_once('}')?;
        return (uri == SERIALIZATION_NAMESPACE).then(|| local.to_owned());
    }
    let (prefix, local) = name.split_once(':')?;
    let (_, uri) = prefixes.iter().rev().find(|(bound, _)| bound == prefix)?;
    (uri == SERIALIZATION_NAMESPACE).then(|| local.to_owned())
}

/// One lexical unit of a query, reduced to what reading the prolog needs.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// A name-shaped run — a keyword, an `NCName`, a lexical `QName`, or a `Q{uri}local`.
    Word(String),
    /// A string literal, with its quoting and its character references resolved.
    Literal(String),
    /// Any other single character.
    Symbol(char),
}

/// Splits `query` into [`Token`]s, discarding comments and whitespace.
///
/// `None` for an unterminated comment, an unterminated literal or a malformed character
/// reference — none of which a query that compiled can contain, so the caller treats them the
/// same way it treats a serialization it cannot reproduce.
fn tokenize(query: &str) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let characters: Vec<char> = query.chars().collect();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if character.is_whitespace() {
            index += 1;
            continue;
        }
        // `(: … :)` nests, so the depth is counted rather than the first `:)` taken.
        if character == '(' && characters.get(index + 1) == Some(&':') {
            let mut depth = 1;
            index += 2;
            while depth > 0 {
                match (characters.get(index), characters.get(index + 1)) {
                    (Some('('), Some(':')) => {
                        depth += 1;
                        index += 2;
                    }
                    (Some(':'), Some(')')) => {
                        depth -= 1;
                        index += 2;
                    }
                    (Some(_), _) => index += 1,
                    (None, _) => return None,
                }
            }
            continue;
        }
        if character == '"' || character == '\'' {
            let (literal, next) = read_literal(&characters, index, character)?;
            tokens.push(Token::Literal(literal));
            index = next;
            continue;
        }
        if is_name_start(character) {
            let (word, next) = read_word(&characters, index)?;
            tokens.push(Token::Word(word));
            index = next;
            continue;
        }
        tokens.push(Token::Symbol(character));
        index += 1;
    }
    Some(tokens)
}

/// Reads the string literal that starts at `start` (whose quote is `quote`), resolving the
/// doubled quote that escapes one and the character and entity references XQuery 3.1 §3.1.1
/// admits inside one. Answers the literal's value and the index just past its closing quote.
fn read_literal(characters: &[char], start: usize, quote: char) -> Option<(String, usize)> {
    let mut value = String::new();
    let mut index = start + 1;
    loop {
        let character = *characters.get(index)?;
        if character == quote {
            if characters.get(index + 1) == Some(&quote) {
                value.push(quote);
                index += 2;
                continue;
            }
            return Some((value, index + 1));
        }
        if character == '&' {
            let (resolved, next) = read_reference(characters, index)?;
            value.push(resolved);
            index = next;
            continue;
        }
        value.push(character);
        index += 1;
    }
}

/// Reads the character or predefined entity reference that starts at `start`, answering the
/// character it stands for and the index just past its `;`.
fn read_reference(characters: &[char], start: usize) -> Option<(char, usize)> {
    let mut body = String::new();
    let mut index = start + 1;
    loop {
        let character = *characters.get(index)?;
        index += 1;
        if character == ';' {
            break;
        }
        body.push(character);
        if body.len() > 16 {
            return None;
        }
    }
    let code = match body.as_str() {
        "amp" => return Some(('&', index)),
        "quot" => return Some(('"', index)),
        "apos" => return Some(('\'', index)),
        "lt" => return Some(('<', index)),
        "gt" => return Some(('>', index)),
        hexadecimal if hexadecimal.starts_with("#x") || hexadecimal.starts_with("#X") => {
            u32::from_str_radix(hexadecimal.get(2..)?, 16).ok()?
        }
        decimal if decimal.starts_with('#') => decimal.get(1..)?.parse::<u32>().ok()?,
        _ => return None,
    };
    Some((char::from_u32(code)?, index))
}

/// Reads the name-shaped run that starts at `start`, including the `Q{uri}` of a braced
/// `EQName`. Answers the run and the index just past it.
fn read_word(characters: &[char], start: usize) -> Option<(String, usize)> {
    let mut word = String::new();
    let mut index = start;
    if characters.get(index) == Some(&'Q') && characters.get(index + 1) == Some(&'{') {
        word.push('Q');
        word.push('{');
        index += 2;
        loop {
            let character = *characters.get(index)?;
            index += 1;
            word.push(character);
            if character == '}' {
                break;
            }
        }
    }
    while let Some(&character) = characters.get(index) {
        if !is_name_character(character) {
            break;
        }
        word.push(character);
        index += 1;
    }
    Some((word, index))
}

/// Whether `character` may start a name-shaped run. `Q` is included for the braced `EQName`
/// form, which [`read_word`] recognizes by the `{` that follows it.
fn is_name_start(character: char) -> bool {
    character.is_alphabetic() || character == '_'
}

/// Whether `character` may continue a name-shaped run: the XML name characters, plus the `:` of
/// a lexical `QName`.
fn is_name_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
}

/// Every `declare namespace prefix = "uri"` in `tokens`, in the order they were written.
fn namespace_declarations(tokens: &[Token]) -> Vec<(String, String)> {
    let mut bindings = Vec::new();
    for window in tokens.windows(5) {
        if let [
            Token::Word(declare),
            Token::Word(namespace),
            Token::Word(prefix),
            Token::Symbol('='),
            Token::Literal(uri),
        ] = window
            && declare == "declare"
            && namespace == "namespace"
            && !prefix.contains(':')
        {
            bindings.push((prefix.clone(), uri.clone()));
        }
    }
    bindings
}

/// The `declare default element namespace "uri"` of the query, or `None` when it declares none
/// (or declares the zero-length URI, which XQuery 3.1 §4.12 makes "no default element
/// namespace").
///
/// The two parameters whose value is a lexical `QName` — `cdata-section-elements` and
/// `suppress-indentation` — resolve an *unprefixed* name against this, not against no namespace:
/// see [`oxixml_fn::xmlio::parameters_from_element_in_context`] and the resolution of bug #28486
/// that `K2-Serialization-28` and `-31` are written to. No parameter document can carry it, since
/// it belongs to the module the parameters were written in rather than to the parameters.
fn default_element_namespace(tokens: &[Token]) -> Option<String> {
    for window in tokens.windows(5) {
        if let [
            Token::Word(declare),
            Token::Word(default),
            Token::Word(element),
            Token::Word(namespace),
            Token::Literal(uri),
        ] = window
            && declare == "declare"
            && default == "default"
            && element == "element"
            && namespace == "namespace"
            && !uri.is_empty()
        {
            return Some(uri.clone());
        }
    }
    None
}

/// Every `declare option eqname "value"` in `tokens`, in the order they were written.
fn option_declarations(tokens: &[Token]) -> Vec<(String, String)> {
    let mut options = Vec::new();
    for window in tokens.windows(4) {
        if let [
            Token::Word(declare),
            Token::Word(option),
            Token::Word(name),
            Token::Literal(value),
        ] = window
            && declare == "declare"
            && option == "option"
        {
            options.push((name.clone(), value.clone()));
        }
    }
    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_fn::xmlio::OutputMethod;

    /// `ser/method-adaptive.xml`'s own preamble, which is the shape almost every case in the
    /// `ser/` sets is written in.
    #[test]
    fn an_output_declaration_reaches_the_serialization_parameters() {
        let parameters = declared_parameters(
            r#"declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare option output:method "adaptive";
               declare option output:indent "no";
               xs:anyURI('"')"#,
        )
        .expect("a reproducible serialization");
        assert_eq!(parameters.method(), OutputMethod::Adaptive);
        assert!(!parameters.indent());
    }

    /// The defaults are the host language's — Serialization 3.1 §3, where the XML declaration is
    /// written — and not `fn:serialize`'s, which omits it.
    #[test]
    fn a_query_without_an_output_declaration_serializes_with_the_host_defaults() {
        let parameters = declared_parameters("<a>{1 + 1}</a>").expect("the defaults");
        assert_eq!(
            parameters,
            SerializationParameters::serialization_defaults()
        );
        assert!(!parameters.omit_xml_declaration());
    }

    /// A declaration the query *does* make overrides the host default it names and leaves the
    /// rest alone — `K2-Serialization-24` declares `standalone` and still asserts that the XML
    /// declaration is there.
    #[test]
    fn a_declared_parameter_overrides_only_itself() {
        let parameters = declared_parameters(
            r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare option o:standalone "yes"; <a/>"#,
        )
        .expect("a reproducible serialization");
        assert_eq!(parameters.standalone(), Some(true));
        assert!(!parameters.omit_xml_declaration());
    }

    /// The braced `EQName` form of an option name names the same parameter as the prefixed one.
    #[test]
    fn a_braced_option_name_is_resolved_without_a_prefix() {
        let parameters = declared_parameters(
            r#"declare option Q{http://www.w3.org/2010/xslt-xquery-serialization}method "text"; 1"#,
        )
        .expect("a reproducible serialization");
        assert_eq!(parameters.method(), OutputMethod::Text);
    }

    /// An unprefixed name in `cdata-section-elements` or `suppress-indentation` is a name in the
    /// *query's* default element namespace (bug #28486), which is what `K2-Serialization-31`
    /// tests: `"b"` under `declare default element namespace` means the `b` of that namespace,
    /// and the no-namespace `b` in the same result must not be wrapped.
    #[test]
    fn an_unprefixed_qname_resolves_in_the_default_element_namespace() {
        let parameters = declared_parameters(
            r#"declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare default element namespace "http://www.example.org/ns/p";
               declare option output:cdata-section-elements "b";
               <chapter/>"#,
        )
        .expect("a reproducible serialization");
        let names = parameters.cdata_section_elements();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].namespace(), Some("http://www.example.org/ns/p"));
        assert_eq!(names[0].local_name(), "b");
    }

    /// With no such declaration the same name is in no namespace, which `K2-Serialization-30`
    /// requires — so the reader must not invent a default element namespace for a query that
    /// declares only ordinary prefixes.
    #[test]
    fn an_unprefixed_qname_stays_in_no_namespace_without_a_declaration() {
        let parameters = declared_parameters(
            r#"declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare namespace p = "http://www.example.org/ns/p";
               declare option output:cdata-section-elements "b";
               <chapter/>"#,
        )
        .expect("a reproducible serialization");
        assert_eq!(parameters.cdata_section_elements()[0].namespace(), None);
    }

    /// The three declarations that begin the same way are told apart, and the zero-length URI is
    /// XQuery 3.1 §4.12's "no default element namespace" rather than a namespace named `""`.
    #[test]
    fn only_a_default_element_namespace_declaration_is_read_as_one() {
        let cases: &[(&str, Option<&str>)] = &[
            (
                r#"declare default element namespace "urn:e"; 1"#,
                Some("urn:e"),
            ),
            (r#"declare default function namespace "urn:f"; 1"#, None),
            (r#"declare default collation "urn:c"; 1"#, None),
            (r#"declare namespace e = "urn:e"; 1"#, None),
            (r#"declare default element namespace ""; 1"#, None),
            ("1", None),
        ];
        for (query, expected) in cases {
            let tokens = tokenize(query).expect("the query tokenizes");
            assert_eq!(
                default_element_namespace(&tokens).as_deref(),
                *expected,
                "{query}"
            );
        }
    }

    /// An option outside the serialization namespace is one §4.16 lets a processor ignore, and
    /// must not make the case undecidable.
    #[test]
    fn an_option_in_another_namespace_is_ignored() {
        let parameters = declared_parameters(
            r#"declare namespace ext = "http://example.com/ns";
               declare option ext:method "nonsense";
               1"#,
        )
        .expect("the defaults");
        assert_eq!(
            parameters,
            SerializationParameters::serialization_defaults()
        );
    }

    #[test]
    fn a_serialization_this_harness_cannot_reproduce_is_declined() {
        for query in [
            r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare option o:parameter-document "params.xml"; 1"#,
            r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare option o:encoding "us-ascii"; 1"#,
            r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare option o:indent "perhaps"; 1"#,
        ] {
            assert!(declared_parameters(query).is_none(), "{query}");
        }
    }

    /// The five parameters this list used to hold are implemented now, so a query declaring one
    /// is decidable rather than skipped — which is the whole of what removing them from
    /// [`UNREPRODUCIBLE_PARAMETERS`] means for the `ser/` test sets. `method` is here for the
    /// same reason: every method Serialization 3.1 defines is implemented, `xhtml` included, so
    /// a query declaring one is decided rather than declined.
    #[test]
    fn the_implemented_text_shaping_parameters_keep_a_case_decidable() {
        for parameter in [
            ("byte-order-mark", "yes"),
            ("escape-uri-attributes", "no"),
            ("normalization-form", "NFC"),
            ("suppress-indentation", "p"),
            ("undeclare-prefixes", "yes"),
            ("method", "xhtml"),
        ] {
            let (name, value) = parameter;
            let query = format!(
                r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
                   declare option o:{name} "{value}"; 1"#
            );
            assert!(declared_parameters(&query).is_some(), "{name}");
        }
    }

    /// `UTF-8` is the encoding the serializer's `String` result already is, so declaring it
    /// leaves the case decidable — 33 of the `ser/` cases do.
    #[test]
    fn declaring_utf_8_keeps_the_case_decidable() {
        assert!(
            declared_parameters(
                r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
                   declare option o:encoding "UTF-8"; 1"#,
            )
            .is_some()
        );
    }

    /// A QName-valued parameter resolves its prefixes against the *query's* bindings, which is
    /// why the declarations are carried onto the synthesized parameter element.
    #[test]
    fn a_qname_valued_parameter_resolves_against_the_querys_prefixes() {
        let parameters = declared_parameters(
            r#"declare namespace o = "http://www.w3.org/2010/xslt-xquery-serialization";
               declare namespace p = "http://example.org/ns/p";
               declare option o:cdata-section-elements "b p:em";
               1"#,
        )
        .expect("a reproducible serialization");
        let names: Vec<String> = parameters
            .cdata_section_elements()
            .iter()
            .map(|name| format!("{:?}/{}", name.namespace(), name.local_name()))
            .collect();
        assert_eq!(
            names,
            vec![
                "None/b".to_owned(),
                "Some(\"http://example.org/ns/p\")/em".to_owned()
            ]
        );
    }

    /// A comment or a string literal may hold anything, including the very keywords the reader
    /// is looking for; the tokenizer takes both out of the way first.
    #[test]
    fn comments_and_literals_do_not_declare_anything() {
        let parameters = declared_parameters(
            r#"(: declare option output:method "text" :)
               "declare option output:method 'text'""#,
        )
        .expect("the defaults");
        assert_eq!(
            parameters,
            SerializationParameters::serialization_defaults()
        );
    }

    /// A string literal carries the doubled quote that escapes one, and the character and
    /// entity references XQuery admits inside one.
    #[test]
    fn a_literal_resolves_its_escapes() {
        let tokens = tokenize(r#"'it''s' "&amp;&#x9;&#65;""#).expect("a tokenizable query");
        assert_eq!(
            tokens,
            vec![
                Token::Literal("it's".to_owned()),
                Token::Literal("&\tA".to_owned()),
            ]
        );
    }

    /// `(: … :)` nests, so a comment that opens one inside itself ends where the outer one does.
    #[test]
    fn a_nested_comment_ends_with_the_outer_one() {
        let tokens = tokenize("(: outer (: inner :) still outer :) body").expect("a comment");
        assert_eq!(tokens, vec![Token::Word("body".to_owned())]);
    }

    #[test]
    fn an_unterminated_comment_or_literal_is_not_tokenizable() {
        assert!(tokenize("(: unterminated").is_none());
        assert!(tokenize("'unterminated").is_none());
        assert!(tokenize("\"&nonsense;\"").is_none());
    }

    /// An item separator written as a tab survives the trip through the parameter element,
    /// which attribute-value normalization would otherwise turn into a space.
    #[test]
    fn whitespace_in_a_value_survives_the_parameter_element() {
        let parameters = declared_parameters(
            "declare namespace o = \"http://www.w3.org/2010/xslt-xquery-serialization\";
             declare option o:item-separator \"&#x9;\"; 1",
        )
        .expect("a reproducible serialization");
        assert_eq!(parameters.item_separator(), Some("\t".to_owned()));
    }
}
