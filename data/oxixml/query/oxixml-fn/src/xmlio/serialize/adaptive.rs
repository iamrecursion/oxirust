//! The `adaptive` output method of
//! [Serialization 3.1 §11](https://www.w3.org/TR/xslt-xquery-serialization-31/#adaptive-output).
//!
//! The method exists so that *any* XDM value can be written down and read back: it renders a
//! value in the constructor syntax of an XPath 3.1 expression that would build the same value
//! again. That single idea decides every rule below — a string is quoted, a boolean is
//! `true()`, a `QName` is an `EQName`, a value of any other type is `xs:TYPE("lexical")`, and a
//! number is written the way a *literal* of its own type is written, so that reading it back
//! does not silently change `xs:double` into `xs:integer`.

use super::params::SerializationParameters;
use super::xml::{escape_plain_attribute, write_top_level_node};
use super::{DepthGuard, separator};
use crate::error::FnResult;
use oxixml_infoset::{AtomicType, AtomicValue, Item, NodeKind, Sequence, XdmFunction, XdmNode};

pub(super) fn serialize_adaptive(
    sequence: &Sequence,
    parameters: &SerializationParameters,
) -> FnResult<String> {
    let separator = separator(parameters, "\n");
    let mut out = String::new();
    for (index, item) in sequence.iter().enumerate() {
        if index > 0 {
            out.push_str(&separator);
        }
        out.push_str(&adaptive_item(item, parameters)?);
    }
    Ok(out)
}

fn adaptive_item(item: &Item, parameters: &SerializationParameters) -> FnResult<String> {
    let _guard = DepthGuard::enter(parameters.max_depth())?;
    match item {
        // Serialization 3.1 §11 writes an atomic value in the *constructor* syntax wherever it
        // appears — at the top level exactly as inside a `map{…}` or `[…]`, so that what the
        // method prints could be read back as an expression. A string, an `xs:untypedAtomic` or
        // an `xs:anyURI` therefore comes out in quotation marks, with any quotation mark of its
        // own doubled (QT3 `method-adaptive/Serialization-adaptive-05`, `-30`, `-77`, `-80`,
        // `-84`).
        Item::Atomic(value) => Ok(adaptive_atomic(value, parameters)),
        Item::Node(node) => {
            // Serialization 3.1 §11: the adaptive method is defined for *every* node kind, and
            // writes an attribute or namespace node the way it appears in a start tag — where
            // the `xml` output method has no serialization for it at all.
            if matches!(node.node_kind(), NodeKind::Attribute | NodeKind::Namespace) {
                let name = node
                    .node_name()
                    .map_or_else(String::new, |name| name.local_name().to_owned());
                let name = if node.node_kind() == NodeKind::Namespace {
                    if name.is_empty() {
                        "xmlns".to_owned()
                    } else {
                        format!("xmlns:{name}")
                    }
                } else {
                    name
                };
                let mut out = format!("{name}=\"");
                escape_plain_attribute(&mut out, &node.string_value(), parameters);
                out.push('"');
                return Ok(out);
            }
            // Serialization 3.1 §11 writes a node with the `xml` output method, which is what
            // the writers below do for any method that is not `html`.
            let mut out = String::new();
            write_top_level_node(&mut out, node, parameters)?;
            Ok(out)
        }
        Item::Array(array) => {
            let mut parts = Vec::with_capacity(array.len());
            for member in array.members() {
                parts.push(adaptive_sequence(member, parameters)?);
            }
            Ok(format!("[{}]", parts.join(",")))
        }
        Item::Map(map) => {
            let mut parts = Vec::with_capacity(map.len());
            for (key, value) in map.entries() {
                parts.push(format!(
                    "{}:{}",
                    adaptive_atomic(key, parameters),
                    adaptive_sequence(value, parameters)?
                ));
            }
            Ok(format!("map{{{}}}", parts.join(",")))
        }
        Item::Function(function) => Ok(adaptive_function(function)),
    }
}

fn adaptive_sequence(
    sequence: &Sequence,
    parameters: &SerializationParameters,
) -> FnResult<String> {
    let mut parts = Vec::with_capacity(sequence.len());
    for item in sequence.iter() {
        // An atomic value nested in a `map{…}` or `[…]` constructor does not spend a level of the
        // descent budget the way a node, map or array does, so it is written directly rather than
        // through `adaptive_item`.
        match item {
            Item::Atomic(value) => parts.push(adaptive_atomic(value, parameters)),
            other => parts.push(adaptive_item(other, parameters)?),
        }
    }
    match parts.len() {
        1 => Ok(parts.into_iter().next().unwrap_or_default()),
        _ => Ok(format!("({})", parts.join(","))),
    }
}

/// One atomic value in the adaptive method's constructor syntax
/// ([Serialization 3.1 §11](https://www.w3.org/TR/xslt-xquery-serialization-31/#adaptive-output)).
///
/// The five cases are decided by the value's *primitive* type, not by its own: a
/// `xs:dateTimeStamp` is written as `xs:dateTime(…)` and a `xs:dayTimeDuration` as
/// `xs:duration(…)`, because those are the constructors that exist (QT3
/// `method-adaptive/Serialization-adaptive-37`, `-41`, `-42`).
///
/// * `xs:string`, `xs:untypedAtomic` and `xs:anyURI` — and everything derived from them — are
///   written in quotation marks, with each quotation mark of their own doubled (`-64`…`-73`,
///   `-77`, `-80`, `-85`);
/// * `xs:boolean` as `true()` or `false()` (`-74`);
/// * `xs:decimal` (so `xs:integer` and its restrictions too) and `xs:double` as a bare numeric
///   literal (`-44`…`-58`);
/// * `xs:QName` and `xs:NOTATION` as the `Q{uri}local` EQName, with empty braces for a name in
///   no namespace (`-19`, `-78`);
/// * everything else — `xs:float`, the date/time types, the durations, the binary types — in the
///   constructor form `xs:TYPE("lexical")` (`-36`…`-43`, `-59`…`-63`, `-75`, `-76`).
pub(super) fn adaptive_atomic(value: &AtomicValue, parameters: &SerializationParameters) -> String {
    let primitive = value.atomic_type().primitive();
    match primitive {
        AtomicType::String | AtomicType::AnyUri | AtomicType::UntypedAtomic => {
            quoted(&value.string_value(), parameters)
        }
        AtomicType::Boolean => match value.as_bool() {
            Some(true) => "true()".to_owned(),
            Some(false) => "false()".to_owned(),
            // A value labelled `xs:boolean` always carries a boolean payload; falling back on
            // the lexical form keeps this total rather than asserting the invariant.
            None => value.string_value(),
        },
        AtomicType::Decimal => value.string_value(),
        AtomicType::Double => value
            .as_f64()
            .map_or_else(|| value.string_value(), double_literal),
        AtomicType::QName | AtomicType::Notation => match value.as_qname() {
            Some(qname) => format!(
                "Q{{{}}}{}",
                qname.namespace().unwrap_or_default(),
                qname.local_name()
            ),
            None => quoted(&value.string_value(), parameters),
        },
        other => format!("{other}({})", quoted(&value.string_value(), parameters)),
    }
}

/// A string in the adaptive method's quotation marks, with each quotation mark of its own
/// doubled so that the result reads back as the same string literal.
///
/// `use-character-maps` is applied here, and only here. XSLT 3.0 §25.1 makes a mapped character
/// immune to the escaping that would otherwise apply — so a map from `'` to `&apos;` replaces the
/// apostrophe of a *string* (the suite's `character-map-026`, whose three assertions cover a text
/// node, a bare string and a string inside a map) — while the constructor syntax this method
/// generates around the value is untouched: `true()`, `map{`, `Q{…}` and a numeric literal are
/// the method's own spelling of an expression, not characters of the value.
fn quoted(value: &str, parameters: &SerializationParameters) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match parameters.mapped(character) {
            Some(replacement) => out.push_str(replacement),
            // The doubled quotation mark is what makes the result read back as this same string.
            None if character == '"' => out.push_str("\"\""),
            None => out.push(character),
        }
    }
    out.push('"');
    out
}

/// A finite `xs:double` as the XPath 3.1 `DoubleLiteral` that denotes it.
///
/// The adaptive method's whole purpose is that its output can be read back as an expression
/// yielding the same value, and the type of a numeric literal is decided by its *shape*: `1`
/// reads back as `xs:integer` and `1.0` as `xs:decimal`, so an `xs:double` has to carry an
/// exponent. `xs:double(1e0)` is therefore written `1.0e0` rather than the `1` that
/// `fn:string` gives it (QT3 `method-adaptive/Serialization-adaptive-44`). `INF`, `-INF` and
/// `NaN` have no literal form at all and keep their lexical one (`-22`, `-24`, `-27`).
fn double_literal(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_owned();
    }
    if value == f64::INFINITY {
        return "INF".to_owned();
    }
    if value == f64::NEG_INFINITY {
        return "-INF".to_owned();
    }
    // `{:e}` is Rust's shortest round-tripping exponential form: `1e0`, `-1.5e-3`. A mantissa
    // without a fraction digit is not yet a `DoubleLiteral`, so one is added.
    let formatted = format!("{value:e}");
    match formatted.split_once('e') {
        Some((mantissa, exponent)) if !mantissa.contains('.') => {
            format!("{mantissa}.0e{exponent}")
        }
        _ => formatted,
    }
}

/// A function item as its name and arity
/// ([Serialization 3.1 §11](https://www.w3.org/TR/xslt-xquery-serialization-31/#adaptive-output)).
///
/// A named function is written with the conventional prefix of its namespace — `fn:exists#1`,
/// not `Q{http://www.w3.org/2005/xpath-functions}exists#1` — and an inline function, which has
/// no name to write, as `(anonymous-function)#1` (QT3
/// `method-adaptive/Serialization-adaptive-12`, `-13`, `-26`, `-27`).
fn adaptive_function(function: &XdmFunction) -> String {
    let arity = function.arity();
    match function.name() {
        None => format!("(anonymous-function)#{arity}"),
        Some(name) => {
            let local = name.local_name();
            match name.namespace() {
                None => format!("{local}#{arity}"),
                Some(namespace) => match conventional_prefix(namespace) {
                    Some(prefix) => format!("{prefix}:{local}#{arity}"),
                    None => format!("Q{{{namespace}}}{local}#{arity}"),
                },
            }
        }
    }
}

/// The prefix specification prose conventionally binds to `namespace`, if it has one.
///
/// [XPath 3.1 §1.2](https://www.w3.org/TR/xpath-31/#dt-namespace-prefix) fixes these six
/// bindings for the namespaces the language itself defines; every other namespace has no
/// conventional prefix and is written as a `Q{…}` EQName instead.
fn conventional_prefix(namespace: &str) -> Option<&'static str> {
    match namespace {
        "http://www.w3.org/2005/xpath-functions" => Some("fn"),
        "http://www.w3.org/2001/XMLSchema" => Some("xs"),
        "http://www.w3.org/2005/xpath-functions/math" => Some("math"),
        "http://www.w3.org/2005/xpath-functions/map" => Some("map"),
        "http://www.w3.org/2005/xpath-functions/array" => Some("array"),
        "http://www.w3.org/2005/xqt-errors" => Some("err"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{OutputMethod, SerializationParameters, serialize};
    use super::*;
    use oxixml_infoset::{QNameValue, Tree, XdmArray, XdmMap};

    fn adaptive() -> SerializationParameters {
        SerializationParameters::default().with_method(OutputMethod::Adaptive)
    }

    fn rendered(value: AtomicValue) -> FnResult<String> {
        serialize(&Sequence::from(value), &adaptive())
    }

    /// The character map reaches the strings this method writes, and stops at the constructor
    /// syntax it generates around them.
    ///
    /// The XSLT 3.0 suite's `character-map-026` is transcribed here: with `'` mapped to
    /// `&apos;`, one transformation writes a text node, a bare string and a string inside a map,
    /// and all three of its assertions want the mapping. XSLT 3.0 §25.1: "If a character is
    /// mapped, then it is not subjected to XML or HTML escaping."
    #[test]
    fn a_character_map_reaches_the_strings_the_adaptive_method_writes() -> FnResult<()> {
        let parameters = adaptive().with_character_map(vec![('\'', "&apos;".to_owned())]);
        assert_eq!(
            serialize(&Sequence::from(AtomicValue::string("\\'E4")), &parameters)?,
            "\"\\&apos;E4\""
        );
        let map = XdmMap::new().put(
            AtomicValue::string("\u{e4}"),
            Sequence::from(AtomicValue::string("\\'E4")),
        );
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(map)), &parameters)?,
            "map{\"\u{e4}\":\"\\&apos;E4\"}"
        );
        // A node inside the value is written by the `xml` writer, which maps it too, so the
        // three assertions of the case agree.
        let tree = Tree::parse_str("<test>\\'E4</test>")?;
        assert_eq!(
            serialize(&Sequence::from(tree.root()), &parameters)?,
            "<test>\\&apos;E4</test>"
        );
        // A mapped quotation mark is replaced rather than doubled: the map is applied instead of
        // the escaping, not before it.
        let quotes = adaptive().with_character_map(vec![('"', "'".to_owned())]);
        assert_eq!(
            serialize(&Sequence::from(AtomicValue::string("a\"b")), &quotes)?,
            "\"a'b\""
        );
        // With no map, nothing moves.
        assert_eq!(
            serialize(&Sequence::from(AtomicValue::string("a\"b")), &adaptive())?,
            "\"a\"\"b\""
        );
        Ok(())
    }

    #[test]
    fn the_adaptive_method_renders_maps_and_arrays() -> FnResult<()> {
        let parameters = adaptive();
        let array = XdmArray::from_iter([AtomicValue::integer(1_i64), AtomicValue::integer(2_i64)]);
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(array)), &parameters)?,
            "[1,2]"
        );

        let map = XdmMap::new().put(
            AtomicValue::string("k"),
            Sequence::from(AtomicValue::integer(1_i64)),
        );
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(map)), &parameters)?,
            "map{\"k\":1}"
        );
        Ok(())
    }

    /// The adaptive method writes an atomic value in the constructor syntax wherever it stands,
    /// so a top-level string, `xs:untypedAtomic` or `xs:anyURI` is quoted and has each quotation
    /// mark of its own doubled (QT3 `method-adaptive/Serialization-adaptive-05`, `-30`, `-77`,
    /// `-80`, `-82`, `-84`, `-85`).
    #[test]
    fn the_adaptive_method_quotes_atomic_values_at_the_top_level() -> FnResult<()> {
        assert_eq!(
            rendered(AtomicValue::string("simple string"))?,
            "\"simple string\""
        );
        assert_eq!(rendered(AtomicValue::string("\""))?, "\"\"\"\"");
        assert_eq!(rendered(AtomicValue::string("\"\""))?, "\"\"\"\"\"\"");
        assert_eq!(rendered(AtomicValue::string("'"))?, "\"'\"");
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::UntypedAtomic, "\"")?)?,
            "\"\"\"\""
        );
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::AnyUri, "\"")?)?,
            "\"\"\"\""
        );
        assert_eq!(
            rendered(AtomicValue::parse(
                AtomicType::AnyUri,
                "http://www.example.org/"
            )?)?,
            "\"http://www.example.org/\""
        );
        // Numbers and booleans keep the forms they already had.
        assert_eq!(rendered(AtomicValue::integer(5_i64))?, "5");
        assert_eq!(rendered(AtomicValue::boolean(true))?, "true()");
        // The item separator sits between the quoted items, not inside them.
        let pair = Sequence::from_iter([
            AtomicValue::string("the quick"),
            AtomicValue::string("brown fox"),
        ]);
        assert_eq!(
            serialize(&pair, &adaptive().with_item_separator(Some(" ".to_owned())))?,
            "\"the quick\" \"brown fox\""
        );
        Ok(())
    }

    /// Serialization 3.1 §11 writes an atomic value that is neither text, nor boolean, nor a
    /// decimal or double number, nor a name in the constructor form `xs:TYPE("lexical")` — under
    /// the name of its *primitive* type (QT3 `method-adaptive/Serialization-adaptive-36`, `-37`,
    /// `-38`, `-39`, `-40`, `-41`, `-42`, `-43`, `-59`, `-60`, `-61`, `-62`, `-63`, `-75`,
    /// `-76`).
    #[test]
    fn the_adaptive_method_writes_other_types_in_constructor_form() -> FnResult<()> {
        for (atomic_type, lexical, expected) in [
            (
                AtomicType::DateTime,
                "1999-05-31T13:20:00-05:00",
                "xs:dateTime(\"1999-05-31T13:20:00-05:00\")",
            ),
            (
                AtomicType::DateTimeStamp,
                "2011-07-28T12:34:56-08:00",
                "xs:dateTime(\"2011-07-28T12:34:56-08:00\")",
            ),
            (AtomicType::Date, "1999-05-31", "xs:date(\"1999-05-31\")"),
            (AtomicType::Time, "12:00:00", "xs:time(\"12:00:00\")"),
            (
                AtomicType::Duration,
                "P1Y2M3DT10H30M23S",
                "xs:duration(\"P1Y2M3DT10H30M23S\")",
            ),
            (
                AtomicType::YearMonthDuration,
                "P1Y2M",
                "xs:duration(\"P1Y2M\")",
            ),
            (
                AtomicType::DayTimeDuration,
                "P3DT10H30M",
                "xs:duration(\"P3DT10H30M\")",
            ),
            (AtomicType::Float, "1e0", "xs:float(\"1\")"),
            (AtomicType::Float, "INF", "xs:float(\"INF\")"),
            (AtomicType::Float, "NaN", "xs:float(\"NaN\")"),
            (
                AtomicType::GYearMonth,
                "2000-01",
                "xs:gYearMonth(\"2000-01\")",
            ),
            (AtomicType::GYear, "2000", "xs:gYear(\"2000\")"),
            (
                AtomicType::GMonthDay,
                "--01-01",
                "xs:gMonthDay(\"--01-01\")",
            ),
            (AtomicType::GDay, "---01", "xs:gDay(\"---01\")"),
            (AtomicType::GMonth, "--01", "xs:gMonth(\"--01\")"),
            (
                AtomicType::Base64Binary,
                "01001010",
                "xs:base64Binary(\"01001010\")",
            ),
            (
                AtomicType::HexBinary,
                "D74D35D35D35",
                "xs:hexBinary(\"D74D35D35D35\")",
            ),
        ] {
            let value = AtomicValue::parse(atomic_type, lexical)?;
            assert_eq!(rendered(value)?, expected, "{atomic_type} {lexical}");
        }
        Ok(())
    }

    /// A number keeps the shape a *literal* of its own type has, so that the adaptive output
    /// reads back as the same value: `xs:decimal` and `xs:integer` bare, `xs:double` with an
    /// exponent (QT3 `method-adaptive/Serialization-adaptive-22`, `-24`, `-27`, `-44`, `-45`,
    /// `-46`, `-47`…`-58`).
    #[test]
    fn numbers_keep_the_shape_of_their_own_literal() -> FnResult<()> {
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::Double, "1e0")?)?,
            "1.0e0"
        );
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::Double, "1.5e0")?)?,
            "1.5e0"
        );
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::Double, "INF")?)?,
            "INF"
        );
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::Double, "-INF")?)?,
            "-INF"
        );
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::Double, "NaN")?)?,
            "NaN"
        );
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::Decimal, "1.2")?)?,
            "1.2"
        );
        for integral in [
            AtomicType::Integer,
            AtomicType::NonPositiveInteger,
            AtomicType::Long,
            AtomicType::Int,
            AtomicType::Short,
            AtomicType::Byte,
            AtomicType::NonNegativeInteger,
            AtomicType::UnsignedLong,
            AtomicType::UnsignedInt,
            AtomicType::UnsignedShort,
            AtomicType::UnsignedByte,
        ] {
            assert_eq!(rendered(AtomicValue::parse(integral, "0")?)?, "0");
        }
        assert_eq!(
            rendered(AtomicValue::parse(AtomicType::PositiveInteger, "1")?)?,
            "1"
        );

        // `[ xs:float("INF"), xs:double("-INF") ]` — the float carries its constructor, the
        // double does not (`Serialization-adaptive-22`).
        let array = XdmArray::from_iter([
            AtomicValue::parse(AtomicType::Float, "INF")?,
            AtomicValue::parse(AtomicType::Double, "-INF")?,
        ]);
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(array)), &adaptive())?,
            "[xs:float(\"INF\"),-INF]"
        );
        Ok(())
    }

    /// A `QName` is its `Q{uri}local` EQName, with empty braces when it is in no namespace
    /// (QT3 `method-adaptive/Serialization-adaptive-19`, `-78`).
    #[test]
    fn a_qname_is_written_as_an_eqname() -> FnResult<()> {
        assert_eq!(
            rendered(AtomicValue::qname(QNameValue::new(
                Some("xs"),
                Some("http://www.w3.org/2001/XMLSchema"),
                "integer",
            )))?,
            "Q{http://www.w3.org/2001/XMLSchema}integer"
        );

        let map = XdmMap::new()
            .put(
                AtomicValue::string("one"),
                Sequence::from(AtomicValue::integer(1_i64)),
            )
            .put(
                AtomicValue::qname(QNameValue::new(None::<&str>, None::<&str>, "one")),
                Sequence::from(AtomicValue::integer(1_i64)),
            );
        assert_eq!(
            serialize(&Sequence::singleton(Item::from(map)), &adaptive())?,
            "map{\"one\":1,Q{}one:1}"
        );
        Ok(())
    }

    /// A function item is its name and arity, with the conventional prefix for a namespace the
    /// language defines and `(anonymous-function)` for an inline function (QT3
    /// `method-adaptive/Serialization-adaptive-12`, `-13`, `-26`, `-27`).
    #[test]
    fn function_items_are_named_with_their_conventional_prefix() {
        use oxixml_infoset::{ExpandedName, FunctionBody};
        use std::any::Any;
        use std::sync::Arc;

        #[derive(Debug)]
        struct Body(Option<ExpandedName>, usize);

        impl FunctionBody for Body {
            fn function_name(&self) -> Option<ExpandedName> {
                self.0.clone()
            }

            fn arity(&self) -> usize {
                self.1
            }

            fn as_any(&self) -> &(dyn Any + Send + Sync) {
                self
            }
        }

        let named = |namespace: Option<&str>, local: &str, arity: usize| {
            XdmFunction::new(Arc::new(Body(
                Some(ExpandedName::new(namespace, local)),
                arity,
            )))
        };
        assert_eq!(
            adaptive_function(&named(
                Some("http://www.w3.org/2005/xpath-functions"),
                "exists",
                1
            )),
            "fn:exists#1"
        );
        assert_eq!(
            adaptive_function(&named(
                Some("http://www.w3.org/2005/xpath-functions/array"),
                "size",
                1
            )),
            "array:size#1"
        );
        assert_eq!(
            adaptive_function(&named(Some("urn:example"), "f", 2)),
            "Q{urn:example}f#2"
        );
        assert_eq!(adaptive_function(&named(None, "local", 0)), "local#0");
        assert_eq!(
            adaptive_function(&XdmFunction::new(Arc::new(Body(None, 1)))),
            "(anonymous-function)#1"
        );
    }

    /// An attribute node written by the adaptive method is escaped the way it would be in a
    /// start tag, `>` included (QT3 `method-adaptive/Serialization-adaptive-88`).
    #[test]
    fn an_attribute_node_is_escaped_as_in_a_start_tag() -> Result<(), Box<dyn std::error::Error>> {
        let tree = Tree::parse_str("<e att-name='att&lt;&amp;&gt;value'/>")?;
        let element = tree.document_element().ok_or("no document element")?;
        let attribute = element
            .attributes()
            .first()
            .cloned()
            .ok_or("no attribute")?;
        assert_eq!(
            serialize(&Sequence::from(attribute), &adaptive())?,
            "att-name=\"att&lt;&amp;&gt;value\""
        );
        Ok(())
    }
}
