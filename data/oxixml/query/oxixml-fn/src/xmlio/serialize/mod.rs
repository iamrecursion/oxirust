//! [`fn:serialize`](https://www.w3.org/TR/xpath-functions-31/#func-serialize) and the subset of
//! [XSLT and XQuery Serialization 3.1](https://www.w3.org/TR/xslt-xquery-serialization-31/) that
//! a function library needs.
//!
//! All six output methods are implemented:
//!
//! * `xml` — the full element/attribute/namespace/comment/processing-instruction serialization,
//!   with optional indentation and an optional XML declaration and document type declaration;
//! * `text` — the concatenated string values of the items;
//! * `adaptive` — the diagnostic method of Serialization 3.1 §11, which renders atomic values,
//!   maps and arrays in the XPath 3.1 constructor syntax and nodes as XML;
//! * `json` — the JSON output method of Serialization 3.1 §4.7, which writes a map, an array or
//!   a single atomic value as JSON text;
//! * `html` — HTML output, which differs from `xml` in that no XML declaration is written, an
//!   element the HTML vocabulary knows to be empty is written with a start tag alone, every other
//!   element keeps an explicit end tag rather than being self-closed, a boolean attribute is
//!   minimized, a processing instruction is closed by `>`, `script` and `style` content is not
//!   escaped, and a URI-valued attribute is percent-encoded;
//! * `xhtml` — XML output with the three concessions Serialization 3.1 §8 makes to HTML readers:
//!   an element of the XHTML vocabulary whose content model is empty is minimized to `<br />`,
//!   every other empty element is written `<div></div>` rather than self-closed, and the
//!   vocabularies HTML 5 embeds without a prefix — XHTML, SVG and MathML — are written under a
//!   default namespace declaration. It shares §7's document type declaration and content-type
//!   `meta`, and keeps XML's escaping, `?>` and CDATA sections throughout.
//!
//! # What is not implemented
//!
//! `escape-uri-attributes` reaches the `html` method alone. The parameter percent-encodes a
//! URI-valued attribute of an element the HTML 4.01 `%URI` table names, and no QT3 case asks the
//! `xhtml` method for it — so rather than guess at the vocabulary an XHTML document's URI
//! attributes come from, the `xhtml` method leaves the value as it stands.
//!
//! Two parameters are *accepted and inert* — the name is recognised and the value is validated,
//! but the output is the same whatever the value says:
//!
//! | Parameter | Why it has no effect here |
//! |---|---|
//! | `media-type` | metadata for the caller, not for the text — the content-type `meta` the `html` method writes declares that method's own default media type |
//! | `parameter-document` | the parameters live in a resource only the caller can resolve and retrieve |
//!
//! Two more are honoured with a limit worth naming. `normalization-form` implements `NFC`, `NFD`,
//! `NFKC`, `NFKD` and `none`; anything else, `FULLY-NORMALIZED` included, is `SESU0011`. And the
//! normalization is applied to the finished result as a whole, so a replacement injected by
//! `use-character-maps` is normalized along with everything around it rather than being held
//! back. `byte-order-mark` puts U+FEFF at the start of the result, which is the character an
//! encoder writes the mark's octets for — the result itself is a Rust `String`, so there are no
//! octets here to mark.
//!
//! Everything else Serialization 3.1 defines for these six methods — `indent`,
//! `suppress-indentation`, `omit-xml-declaration`, `version`, `standalone`, `item-separator`,
//! `use-character-maps`, `cdata-section-elements`, the document type declaration parameters,
//! `html-version`, `include-content-type`, `escape-uri-attributes`, `undeclare-prefixes`, and the
//! `json` method's `allow-duplicate-names` and `json-node-output-method` — is honoured.

mod adaptive;
mod params;
mod xml;

pub use params::{
    OutputMethod, SERIALIZATION_NAMESPACE, SerializationParameters, parameters_from_element,
    parameters_from_element_in_context, parameters_from_element_with_defaults, qnames_from_option,
};

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{Item, NodeKind, NodeRef, Sequence, XdmNode};

/// [`fn:serialize`](https://www.w3.org/TR/xpath-functions-31/#func-serialize).
///
/// The six output methods this honours and the parameters it accepts without acting on are
/// listed in the [module documentation](self).
///
/// # Errors
///
/// [`FnErrorCode::Senr0001`] when an item cannot be serialized by the chosen method — an
/// attribute or namespace node, a map, or a function item; [`FnErrorCode::Sepm0004`] when
/// `doctype-system` or a `standalone` other than `omit` is asked for while the value is not a
/// well-formed document; [`FnErrorCode::Sepm0009`] when `omit-xml-declaration` suppresses the
/// declaration those same parameters need; [`FnErrorCode::Sesu0011`] for an unsupported
/// `normalization-form`; [`FnErrorCode::Xpdy0130`] when the value nests deeper than
/// [`SerializationParameters::max_depth`].
///
/// # Examples
///
/// ```
/// use oxixml_fn::xmlio::{serialize, SerializationParameters};
/// use oxixml_infoset::{Sequence, Tree};
///
/// let tree = Tree::parse_str("<a b='1'><c>text</c><!-- note --></a>")?;
/// let value = Sequence::from(tree.document_element().expect("a document element"));
/// assert_eq!(
///     serialize(&value, &SerializationParameters::default())?,
///     r#"<a b="1"><c>text</c><!-- note --></a>"#
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn serialize(sequence: &Sequence, parameters: &SerializationParameters) -> FnResult<String> {
    let normalization = normalization_form(parameters)?;
    check_declaration_parameters(parameters)?;
    let mut out = write(sequence, parameters)?;
    check_output_characters(&out, parameters)?;
    if let Some(form) = normalization {
        out = crate::strings::normalize_unicode(&out, form);
    }
    if parameters.byte_order_mark() {
        // The result is a `String`, so the mark is the character U+FEFF the encoder would write
        // the octets of; a caller that encodes the string keeps it (Serialization 3.1 §3).
        out.insert(0, '\u{FEFF}');
    }
    Ok(out)
}

/// `SERE0006` — the result would hold a character the XML version being written cannot spell.
///
/// [Serialization 3.1 §5.1.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#xml-output)
/// gives the `xml` method no way to write a C0 control other than tab, line feed and carriage
/// return when `version` is `1.0`: XML 1.0 excludes them from its `Char` production, so neither
/// the character nor a reference to it is well formed. XML 1.1 admits them as references, which
/// the escaper writes, so nothing is left here to object to.
///
/// The check is on the finished text, where the escapers have already replaced everything they
/// can: whatever survives came from a value being written, or from a character map's replacement,
/// and either way the result would not parse. It applies to the two methods whose output is XML —
/// the `text` method has no XML version, and the `html` method has its own vocabulary (the XSLT
/// 3.0 suite's `xml-version-029`/`-030`, which copy an XML 1.1 document to a `version="1.0"`
/// result and want this error).
fn check_output_characters(out: &str, parameters: &SerializationParameters) -> FnResult<()> {
    if !matches!(parameters.method(), OutputMethod::Xml | OutputMethod::Xhtml) {
        return Ok(());
    }
    let Some(offending) = out.chars().find(|character| {
        crate::xmlio::serialize::xml::is_restricted_c0(*character) && !parameters.is_xml_11()
    }) else {
        return Ok(());
    };
    Err(FnError::new(
        FnErrorCode::Sere0006,
        format!(
            "the character #x{:X} cannot be written by the `xml` output method at version {}",
            u32::from(offending),
            parameters.version().trim()
        ),
    ))
}

/// The Unicode normalization `normalization-form` asks for, `None` when it asks for none.
///
/// # Errors
///
/// [`FnErrorCode::Sesu0011`] for a form this serializer does not implement — `FULLY-NORMALIZED`
/// included, which needs the *starter* analysis of UAX #15 that `fn:normalize-unicode` does not
/// provide either.
fn normalization_form(
    parameters: &SerializationParameters,
) -> FnResult<Option<crate::strings::NormalizationForm>> {
    use crate::strings::NormalizationForm;
    let requested = parameters.normalization_form();
    match requested.trim().to_uppercase().as_str() {
        "" | "NONE" => Ok(None),
        "NFC" => Ok(Some(NormalizationForm::Nfc)),
        "NFD" => Ok(Some(NormalizationForm::Nfd)),
        "NFKC" => Ok(Some(NormalizationForm::Nfkc)),
        "NFKD" => Ok(Some(NormalizationForm::Nfkd)),
        other => Err(FnError::new(
            FnErrorCode::Sesu0011,
            format!("the normalization form `{other}` is not supported by this serializer"),
        )),
    }
}

/// `SEPM0009` — the parameters ask for an XML declaration to carry something while also asking
/// for the declaration to be omitted.
///
/// [Serialization 3.1 §5.1.1](https://www.w3.org/TR/xslt-xquery-serialization-31/#xml-output)
/// makes both spellings an error: `standalone` other than `omit`, and a `doctype-system` under a
/// `version` other than `1.0`, each need a declaration that `omit-xml-declaration="yes"` would
/// suppress (QT3 `Serialization-031`, `-032`).
///
/// The contradiction has to be one the caller asked for, so the omission must have been
/// *specified*: `fn:serialize`'s own default set omits the declaration, and a `standalone` handed
/// to it is a request to write one rather than a request that conflicts with anything (QT3
/// `Serialization-026`, `-027`, which give `standalone` alone and want the `SEPM0004` of a
/// fragment, not this).
///
/// The rule reaches the `xhtml` method too, whose output is XML and carries the same declaration;
/// the `html` method writes none at all, so nothing about it can contradict.
fn check_declaration_parameters(parameters: &SerializationParameters) -> FnResult<()> {
    if !matches!(parameters.method(), OutputMethod::Xml | OutputMethod::Xhtml)
        || !parameters.omit_xml_declaration()
        || !parameters.omit_xml_declaration_specified()
    {
        return Ok(());
    }
    if parameters.standalone().is_some() {
        return Err(FnError::new(
            FnErrorCode::Sepm0009,
            "`standalone` needs an XML declaration, which `omit-xml-declaration` suppresses",
        ));
    }
    if parameters.doctype_system().is_some() && parameters.version().trim() != "1.0" {
        return Err(FnError::new(
            FnErrorCode::Sepm0009,
            "`doctype-system` past XML 1.0 needs an XML declaration, which \
             `omit-xml-declaration` suppresses",
        ));
    }
    Ok(())
}

fn write(sequence: &Sequence, parameters: &SerializationParameters) -> FnResult<String> {
    match parameters.method() {
        OutputMethod::Text => serialize_text(sequence, parameters),
        OutputMethod::Adaptive => adaptive::serialize_adaptive(sequence, parameters),
        OutputMethod::Json => {
            // Serialization 3.1 §10: a node inside the value is serialized with the
            // `json-node-output-method` and its lexical form written as a JSON string.
            //
            // §5.1 fixes `omit-xml-declaration` to `yes` for that nested serialization whatever
            // the outer set says: the node's text goes inside a JSON string in the middle of a
            // JSON text, where an XML declaration would be neither legal nor meaningful (QT3
            // `method-json/Serialization-json-51`, `-64`, which want
            // `"xml-content":"<simple>text-value<\/simple>"` and nothing before the start tag).
            let node_parameters = parameters
                .clone()
                .with_method(parameters.json_node_output_method())
                .with_omit_xml_declaration(true);
            let node_serializer = |node: &NodeRef| -> FnResult<String> {
                // The nested serialization is part of one JSON text, so it is written and left
                // alone: the byte-order mark and the Unicode normalization belong to the result
                // as a whole, and `serialize` applies both once, at the end.
                write(&Sequence::from(node.clone()), &node_parameters)
            };
            let options = crate::json::JsonOutputOptions {
                allow_duplicate_names: parameters.allow_duplicate_names(),
                node_serializer: Some(&node_serializer),
                escape_beyond: crate::json::escapes_non_ascii(&parameters.encoding()),
                // The nesting ceiling is one setting, whichever method ends up writing the
                // value, so the `json` writer inherits the one the caller set here rather than
                // falling back to its own default.
                max_depth: parameters.max_depth(),
                // `use-character-maps` is a parameter of the *serialization*, not of one output
                // method, so the `json` writer honours it where the `xml` writer does — the XSLT
                // 3.0 suite's `output-0708` maps the solidus to itself to suppress the `\/` this
                // method would otherwise write.
                character_map: parameters.character_map(),
                ..crate::json::JsonOutputOptions::new(parameters.indent())
            };
            crate::json::serialize_json_with(sequence, &options)
        }
        OutputMethod::Xml | OutputMethod::Html | OutputMethod::Xhtml => {
            xml::serialize_xml(sequence, parameters)
        }
    }
}

fn separator(parameters: &SerializationParameters, default: &str) -> String {
    parameters
        .item_separator()
        .unwrap_or_else(|| default.to_owned())
}

fn serialize_text(sequence: &Sequence, parameters: &SerializationParameters) -> FnResult<String> {
    let items = normalized_items(sequence, parameters)?;
    let separator = parameters.item_separator();
    let mut out = String::new();
    let mut previous_atomic = false;
    for (index, item) in items.iter().enumerate() {
        let atomic = matches!(item, Item::Atomic(_));
        if index > 0 {
            match &separator {
                Some(separator) => out.push_str(separator),
                // Serialization 3.1 §2: with no `item-separator` the normalization joins
                // *adjacent strings* with a single space, and only an atomic value becomes a
                // string — two nodes stay hard against each other (QT3
                // `method-text/Serialization-text-19` against `-14`).
                None if atomic && previous_atomic => out.push(' '),
                None => {}
            }
        }
        previous_atomic = atomic;
        out.push_str(&item.string_value()?);
    }
    Ok(out)
}

/// The *sequence normalization* of
/// [Serialization 3.1 §2](https://www.w3.org/TR/xslt-xquery-serialization-31/#serdm), as far as
/// it decides which items reach a writer at all.
///
/// Two of its steps are the whole of this function. An **array** is replaced by the sequence of
/// its members, recursively — so `[1, 2, 3]` serializes exactly as `1, 2, 3` would and an array
/// of nodes serializes as those nodes (QT3 `method-xml/Serialization-xml-01`,
/// `method-html/Serialization-html-45`, `-47`, `-48`). Any **other** function item, a map
/// included, has no serialization at all and is `SENR0001`; so is an **attribute** or
/// **namespace** node, which has no meaning outside the start tag it belongs to
/// (`method-xml/K2-Serialization-1`…`-4`, `method-text/Serialization-text-1`…`-4`).
///
/// The `json` and `adaptive` methods define their own treatment of maps, arrays and function
/// items and do not come through here.
fn normalized_items(
    sequence: &Sequence,
    parameters: &SerializationParameters,
) -> FnResult<Vec<Item>> {
    let mut items = Vec::with_capacity(sequence.len());
    flatten_arrays(sequence, parameters, &mut items)?;
    Ok(items)
}

fn flatten_arrays(
    sequence: &Sequence,
    parameters: &SerializationParameters,
    out: &mut Vec<Item>,
) -> FnResult<()> {
    // An array may hold arrays, and the value is the caller's, so the flattening runs out of the
    // same budget the writers do rather than out of stack.
    let _guard = DepthGuard::enter(parameters.max_depth())?;
    for item in sequence.iter() {
        match item {
            Item::Array(array) => {
                for member in array.members() {
                    flatten_arrays(member, parameters, out)?;
                }
            }
            Item::Map(_) | Item::Function(_) => {
                return Err(unserializable_item(parameters.method()));
            }
            Item::Node(node)
                if matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace) =>
            {
                return Err(unserializable_node(parameters.method(), node.node_kind()));
            }
            other => out.push(other.clone()),
        }
    }
    Ok(())
}

/// `SENR0001` for a map or a function item, which no output method but `adaptive` and `json`
/// can write.
fn unserializable_item(method: OutputMethod) -> FnError {
    FnError::new(
        FnErrorCode::Senr0001,
        format!(
            "a function item cannot be serialized with method=\"{}\"",
            method.name()
        ),
    )
}

/// `SENR0001` for an attribute or namespace node, which has no serialization outside the start
/// tag it belongs to.
fn unserializable_node(method: OutputMethod, kind: NodeKind) -> FnError {
    let what = if kind == NodeKind::Namespace {
        "a namespace node"
    } else {
        "an attribute node"
    };
    FnError::new(
        FnErrorCode::Senr0001,
        format!(
            "{what} cannot be serialized with method=\"{}\"",
            method.name()
        ),
    )
}

thread_local! {
    /// How many levels of the value being written are open on this thread.
    ///
    /// The `depth` the writers pass around is an *indentation* depth — it stands still when
    /// `indent` is off, and inside an element with text content — so it cannot double as the
    /// structural count the recursion guard needs.
    static WRITE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Holds one level of writing depth for as long as it is alive.
struct DepthGuard;

impl DepthGuard {
    /// Opens one level, refusing to descend past `limit`.
    fn enter(limit: usize) -> FnResult<Self> {
        let depth = WRITE_DEPTH.with(|depth| {
            let next = depth.get() + 1;
            depth.set(next);
            next
        });
        if depth > limit {
            WRITE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
            return Err(FnError::new(
                FnErrorCode::Xpdy0130,
                format!(
                    "the value nests more than {limit} levels deep, which is this serializer's configurable limit"
                ),
            ));
        }
        Ok(Self)
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        WRITE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FnErrorCode;
    use oxixml_infoset::{AtomicValue, Tree, XdmArray, XdmMap, XdmNode};

    fn document(xml: &str) -> Result<Sequence, Box<dyn std::error::Error>> {
        let tree = Tree::parse_str(xml)?;
        Ok(Sequence::from(
            tree.document_element().ok_or("no document element")?,
        ))
    }

    /// Serialization 3.1 §2: the sequence normalization joins *adjacent strings* with a single
    /// space when no `item-separator` is given — and an atomic value is the only item that
    /// becomes a string, so two nodes stay hard against each other (QT3
    /// `method-text/Serialization-text-19` against `-14`).
    #[test]
    fn the_text_method_joins_adjacent_strings_with_a_space()
    -> Result<(), Box<dyn std::error::Error>> {
        let value = document("<a>one<b>two</b></a>")?;
        let parameters = SerializationParameters::default().with_method(OutputMethod::Text);
        assert_eq!(serialize(&value, &parameters)?, "onetwo");

        let atoms = Sequence::from_iter([AtomicValue::string("a"), AtomicValue::string("b")]);
        assert_eq!(serialize(&atoms, &parameters)?, "a b");
        let separated = parameters.clone().with_item_separator(Some("-".to_owned()));
        assert_eq!(serialize(&atoms, &separated)?, "a-b");

        // Two nodes are not strings, so nothing is inserted between them — but an explicit
        // separator still is.
        let tree = Tree::parse_str("<r><a>1 2 3</a><b>4 5 6</b></r>")?;
        let root = tree.document_element().ok_or("no document element")?;
        let nodes = Sequence::from_iter(root.children().into_iter().map(Item::from));
        assert_eq!(serialize(&nodes, &parameters)?, "1 2 34 5 6");
        assert_eq!(
            serialize(
                &nodes,
                &parameters.clone().with_item_separator(Some("|".to_owned()))
            )?,
            "1 2 3|4 5 6"
        );
        Ok(())
    }

    /// Serialization 3.1 §2: an array is replaced by the sequence of its members before any
    /// method but `json` and `adaptive` writes it, recursively (QT3
    /// `method-xml/Serialization-xml-01`, `method-html/Serialization-html-45`, `-47`, `-48`).
    #[test]
    fn an_array_is_replaced_by_its_members() -> Result<(), Box<dyn std::error::Error>> {
        let numbers = Sequence::singleton(Item::from(XdmArray::from_iter([
            AtomicValue::integer(1_i64),
            AtomicValue::integer(2_i64),
            AtomicValue::integer(3_i64),
            AtomicValue::integer(4_i64),
            AtomicValue::integer(5_i64),
        ])));
        let piped = SerializationParameters::default().with_item_separator(Some("|".to_owned()));
        assert_eq!(serialize(&numbers, &piped)?, "1|2|3|4|5");

        // Recursively, and for the `text` and `html` methods as well as `xml`.
        let nested = Sequence::singleton(Item::from(XdmArray::from_members([
            Sequence::from(AtomicValue::integer(1_i64)),
            Sequence::singleton(Item::from(XdmArray::from_iter([
                AtomicValue::integer(2_i64),
                AtomicValue::integer(3_i64),
            ]))),
        ])));
        assert_eq!(serialize(&nested, &piped)?, "1|2|3");
        // The members are atomic, so with no separator they are the adjacent strings §2 joins
        // with a single space (QT3 `method-text/Serialization-text-19`).
        assert_eq!(
            serialize(
                &nested,
                &SerializationParameters::default().with_method(OutputMethod::Text)
            )?,
            "1 2 3"
        );

        // An array of nodes serializes as those nodes.
        let tree = Tree::parse_str("<a><b/><c/></a>")?;
        let root = tree.document_element().ok_or("no document element")?;
        let children = XdmArray::from_members(
            root.children()
                .into_iter()
                .map(Sequence::from)
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            serialize(
                &Sequence::singleton(Item::from(children)),
                &SerializationParameters::default().with_method(OutputMethod::Html)
            )?,
            "<b></b><c></c>"
        );
        Ok(())
    }

    /// Serialization 3.1 §5.1.1: `doctype-system`, or a `standalone` other than `omit`, needs a
    /// well-formed document — exactly one element, and no character data of its own (QT3
    /// `Serialization-026`, `-028`).
    #[test]
    fn a_fragment_cannot_carry_a_doctype_or_a_standalone_declaration()
    -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<r><a/><b/></r>")?;
        let root = tree.document_element().ok_or("no document element")?;
        let two_elements = Sequence::from_iter(root.children().into_iter().map(Item::from));
        let text = Sequence::from(AtomicValue::string("banana"));
        let one_element = document("<a/>")?;

        for parameters in [
            // Both parameter sets, because a `standalone` that arrives on its own must reach
            // this error rather than the `SEPM0009` of a suppressed declaration.
            SerializationParameters::serialization_defaults().with_standalone(Some(true)),
            SerializationParameters::default().with_standalone(Some(true)),
            SerializationParameters::default()
                .with_doctype_system(Some("http://www.example.com/".to_owned())),
        ] {
            for value in [&two_elements, &text] {
                assert!(
                    serialize(value, &parameters)
                        .expect_err("not a well-formed document")
                        .is(FnErrorCode::Sepm0004)
                );
            }
            // One element and nothing else is well-formed, and so is one element among the
            // comments, processing instructions and whitespace XML admits around it.
            assert!(serialize(&one_element, &parameters).is_ok());
        }

        let with_misc = document("<a/>")?;
        let padded = Sequence::from_iter(
            with_misc
                .iter()
                .cloned()
                .chain([Item::from(AtomicValue::string("  "))]),
        );
        assert!(
            serialize(
                &padded,
                &SerializationParameters::default()
                    .with_doctype_system(Some("about:legacy-compat".to_owned()))
            )
            .is_ok()
        );
        // `standalone="omit"` asks for nothing, so a fragment is no error at all.
        assert!(
            serialize(
                &two_elements,
                &SerializationParameters::serialization_defaults()
            )
            .is_ok()
        );
        Ok(())
    }

    /// Serialization 3.1 §5.1.1: an *asked-for* `omit-xml-declaration="yes"` may not be combined
    /// with a `standalone` other than `omit`, nor with a `doctype-system` past XML 1.0 — the
    /// declaration each of those needs is the one being omitted (QT3 `Serialization-031`, `-032`).
    /// The omission has to be the caller's own: `fn:serialize` omits the declaration by default,
    /// and a `standalone` handed to it contradicts nothing (`Serialization-026`, `-027`).
    #[test]
    fn omitting_the_declaration_conflicts_with_standalone() -> Result<(), Box<dyn std::error::Error>>
    {
        let value = document("<a/>")?;
        for standalone in [true, false] {
            let asked_for = SerializationParameters::default()
                .with_omit_xml_declaration(true)
                .with_standalone(Some(standalone));
            assert!(
                serialize(&value, &asked_for)
                    .expect_err("standalone with the declaration suppressed")
                    .is(FnErrorCode::Sepm0009)
            );
            // Writing the declaration resolves it, and so does leaving the omission to the
            // default parameter set.
            assert!(serialize(&value, &asked_for.with_omit_xml_declaration(false)).is_ok());
            assert!(
                serialize(
                    &value,
                    &SerializationParameters::default().with_standalone(Some(standalone))
                )
                .is_ok()
            );
        }
        let past_1_0 = SerializationParameters::default()
            .with_omit_xml_declaration(true)
            .with_version("1.1")
            .with_doctype_system(Some("about:legacy-compat".to_owned()));
        assert!(
            serialize(&value, &past_1_0)
                .expect_err("a 1.1 doctype with the declaration suppressed")
                .is(FnErrorCode::Sepm0009)
        );
        // XML 1.0 needs no declaration for a document type declaration.
        assert!(
            serialize(
                &value,
                &SerializationParameters::default()
                    .with_omit_xml_declaration(true)
                    .with_doctype_system(Some("about:legacy-compat".to_owned()))
            )
            .is_ok()
        );
        Ok(())
    }

    /// Serialization 3.1 §5.1: a node nested inside a JSON value is serialized with
    /// `omit-xml-declaration="yes"` whatever the outer parameter set says — its text goes inside
    /// a JSON string, where a declaration would be neither legal nor meaningful (QT3
    /// `method-json/Serialization-json-51`, `-64`, `-29`).
    #[test]
    fn a_node_inside_a_json_value_carries_no_xml_declaration()
    -> Result<(), Box<dyn std::error::Error>> {
        let element = document("<e>hi</e>")?;
        // `Serialization-json-51`: `json-node-output-method "xml"` under a set that writes the
        // declaration for a top-level node.
        let json = SerializationParameters::serialization_defaults()
            .with_method(OutputMethod::Json)
            .with_json_node_output_method(OutputMethod::Xml);
        assert_eq!(serialize(&element, &json)?, r#""<e>hi<\/e>""#);
        // The same set at the top level does write one, so the suppression is the nesting's.
        assert!(
            serialize(&element, &SerializationParameters::serialization_defaults())?
                .starts_with("<?xml")
        );
        // An explicit `omit-xml-declaration="no"` does not reach into the JSON value either.
        assert_eq!(
            serialize(&element, &json.clone().with_omit_xml_declaration(false))?,
            r#""<e>hi<\/e>""#
        );

        // `Serialization-json-64`: the node sits under a key in a map.
        let mut map = XdmMap::new();
        map.insert(
            AtomicValue::string("xml-content"),
            document("<simple>text-value</simple>")?,
        );
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(map)), &json)?,
            r#"{"xml-content":"<simple>text-value<\/simple>"}"#
        );

        // `Serialization-json-29`: an element and the document node wrapping it, which the case
        // expects to serialize alike.
        let tree = Tree::parse_str("<a>b</a>")?;
        let members = XdmArray::from_members([
            Sequence::from(tree.document_element().ok_or("no document element")?),
            Sequence::from(tree.root()),
        ]);
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(members)), &json)?,
            r#"["<a>b<\/a>","<a>b<\/a>"]"#
        );
        Ok(())
    }

    /// Serialization 3.1 §3: `normalization-form` normalizes the whole serialized result, and a
    /// form this serializer does not implement is `SESU0011` (QT3
    /// `method-json/Serialization-json-31`, `-32`, `prod/Serialization-018`).
    #[test]
    fn the_result_is_normalized_on_request() -> Result<(), Box<dyn std::error::Error>> {
        // U+0063 U+0327 (c + combining cedilla) composes to U+00E7.
        let value = Sequence::from(AtomicValue::string("suc\u{327}on"));
        let json = SerializationParameters::default().with_method(OutputMethod::Json);
        assert_eq!(
            serialize(&value, &json.clone().with_normalization_form("NFC"))?,
            "\"su\u{e7}on\""
        );
        assert_eq!(
            serialize(&value, &json.clone().with_normalization_form("none"))?,
            "\"suc\u{327}on\""
        );
        assert_eq!(serialize(&value, &json)?, "\"suc\u{327}on\"");
        // The name is matched after trimming and upper-casing, as F&O §5.4.7 does.
        assert_eq!(
            serialize(&value, &json.clone().with_normalization_form(" nfc "))?,
            "\"su\u{e7}on\""
        );
        // NFD decomposes again, and the markup of an XML result is untouched either way.
        let element = document("<a>\u{e7}</a>")?;
        assert_eq!(
            serialize(
                &element,
                &SerializationParameters::default().with_normalization_form("NFD")
            )?,
            "<a>c\u{327}</a>"
        );
        for unsupported in ["FULLY-NORMALIZED", "__NOT_SUPPORTED__"] {
            assert!(
                serialize(&value, &json.clone().with_normalization_form(unsupported))
                    .expect_err(unsupported)
                    .is(FnErrorCode::Sesu0011)
            );
        }
        Ok(())
    }

    /// Serialization 3.1 §3: `byte-order-mark` puts U+FEFF at the start of the result, whatever
    /// the output method — the character an encoder writes the mark's octets for.
    #[test]
    fn a_byte_order_mark_can_be_asked_for() -> Result<(), Box<dyn std::error::Error>> {
        let value = document("<a/>")?;
        for method in [
            OutputMethod::Xml,
            OutputMethod::Html,
            OutputMethod::Xhtml,
            OutputMethod::Text,
            OutputMethod::Adaptive,
        ] {
            let parameters = SerializationParameters::default()
                .with_method(method)
                .with_byte_order_mark(true);
            assert!(
                serialize(&value, &parameters)?.starts_with('\u{FEFF}'),
                "{method:?}"
            );
        }
        assert!(!serialize(&value, &SerializationParameters::default())?.starts_with('\u{FEFF}'));
        Ok(())
    }

    /// Serialization 3.1 §2: a map, a function item, and a parentless attribute or namespace
    /// node have no serialization under any method that normalizes — the `text` method
    /// included, which used to write an attribute's string value instead (QT3
    /// `method-text/Serialization-text-1`…`-4`, `method-xml/K2-Serialization-1`…`-4`).
    #[test]
    fn normalization_refuses_what_no_method_can_write() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<a b='1'/>")?;
        let element = tree.document_element().ok_or("no document element")?;
        let attribute = Sequence::from(
            element
                .attributes()
                .first()
                .cloned()
                .ok_or("no attribute")?,
        );
        let map = Sequence::singleton(Item::from(XdmMap::new()));
        for method in [OutputMethod::Xml, OutputMethod::Html, OutputMethod::Text] {
            let parameters = SerializationParameters::default().with_method(method);
            assert!(
                serialize(&attribute, &parameters)
                    .expect_err("a parentless attribute")
                    .is(FnErrorCode::Senr0001)
            );
            assert!(
                serialize(&map, &parameters)
                    .expect_err("a map")
                    .is(FnErrorCode::Senr0001)
            );
        }
        Ok(())
    }
}
