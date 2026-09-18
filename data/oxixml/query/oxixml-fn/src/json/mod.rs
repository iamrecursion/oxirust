//! The JSON functions of
//! [F&O 3.1 §17.5–§17.6](https://www.w3.org/TR/xpath-functions-31/#json-functions):
//! `fn:parse-json`, `fn:json-doc`, `fn:json-to-xml` and `fn:xml-to-json`, plus the `json`
//! output method the serializer uses.
//!
//! The three directions are all expressed over one scanner, [`parser`]:
//!
//! * [`parse_json`] builds the XDM value — maps, arrays, `xs:double`, `xs:string`, `xs:boolean`
//!   and the empty sequence for `null`;
//! * [`json_to_xml`] builds the XML representation of §17.6, in the
//!   `http://www.w3.org/2005/xpath-functions` namespace;
//! * [`xml_to_json`] walks that representation back to JSON text.
//!
//! # Options
//!
//! [`JsonOptions`] carries the option map the specification defines. `fallback` is a *function*,
//! which the data model has no item for, so it is supplied here as a Rust closure and wired up by
//! whichever evaluator owns function items.
//!
//! # Examples
//!
//! ```
//! use oxixml_fn::json::{json_to_xml, parse_json, xml_to_json, JsonOptions};
//! use oxixml_infoset::XdmNode;
//!
//! let value = parse_json("{\"a\":[1,true,null]}", &JsonOptions::default())?;
//! assert_eq!(value.len(), 1);
//!
//! let document = json_to_xml("[1]", &JsonOptions::default())?;
//! assert_eq!(document.string_value(), "1");
//!
//! let root = document.children().into_iter().next().expect("the root element");
//! assert_eq!(xml_to_json(&root, &JsonOptions::default())?, "[1]");
//! # Ok::<(), oxixml_fn::FnError>(())
//! ```

pub mod parser;

mod xml;

use crate::error::{FnError, FnErrorCode, FnResult};
use oxixml_infoset::{AtomicType, AtomicValue, Item, Sequence, XdmArray, XdmMap};
use parser::{Duplicates, JsonValue, StringPolicy};

pub use xml::{json_to_xml, json_to_xml_with_base, xml_to_json};

/// Which function's option rules apply.
///
/// The two families differ in the `duplicates` values they accept:
/// [F&O 3.1 §17.5.1](https://www.w3.org/TR/xpath-functions-31/#func-parse-json) gives
/// `fn:parse-json` and `fn:json-doc` `reject`, `use-first` and `use-last` — `retain` cannot be
/// represented by a map — while
/// [§17.5.3](https://www.w3.org/TR/xpath-functions-31/#func-json-to-xml) gives `fn:json-to-xml`
/// `reject`, `use-first` and `retain`, since its representation preserves order and has no way
/// to express "the last one wins".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsonFunction {
    /// `fn:parse-json` and `fn:json-doc`.
    #[default]
    ParseJson,
    /// `fn:json-to-xml`.
    JsonToXml,
}

impl JsonFunction {
    /// Whether this function accepts `value` as its `duplicates` option.
    const fn accepts(self, duplicates: Duplicates) -> bool {
        match self {
            Self::ParseJson => !matches!(duplicates, Duplicates::Retain),
            Self::JsonToXml => !matches!(duplicates, Duplicates::UseLast),
        }
    }

    /// Whether this function defines a `validate` option.
    ///
    /// Only `fn:json-to-xml` does — it is the only one whose result is a tree a schema could
    /// describe. An option no function defines is ignored, which is what
    /// [F&O 3.1 §2.1.1](https://www.w3.org/TR/xpath-functions-31/#options) prescribes for a key
    /// the function specification does not list, so `fn:parse-json` neither type-checks the
    /// value nor raises anything for it.
    const fn defines_validate(self) -> bool {
        matches!(self, Self::JsonToXml)
    }
}

/// The namespace the XML representation of JSON lives in,
/// `http://www.w3.org/2005/xpath-functions`.
pub const JSON_XML_NAMESPACE: &str = "http://www.w3.org/2005/xpath-functions";

/// The options the JSON functions accept.
///
/// The defaults are the specification's: `liberal=false`, `escape=false`, `indent=false`, and
/// `duplicates=use-first` — `fn:json-to-xml` overrides the last with
/// [`Duplicates::Retain`], which its caller sets explicitly.
pub struct JsonOptions<'a> {
    /// The `liberal` option. Deviations from RFC 7159 are implementation-defined, and this
    /// implementation defines them as *none*: the grammar is the strict one either way.
    pub liberal: bool,
    /// The `duplicates` option.
    pub duplicates: Duplicates,
    /// The `escape` option.
    pub escape: bool,
    /// The `indent` option, which only `fn:xml-to-json` and the `json` output method read.
    pub indent: bool,
    /// The `fallback` option: what replaces a character the XDM string type cannot hold.
    pub fallback: Option<parser::Fallback<'a>>,
    /// Which function these options belong to, which decides the `duplicates` domain.
    pub function: JsonFunction,
    /// How deeply arrays and objects may nest, in either direction.
    ///
    /// This is not one of the specification's options — it is this implementation's guard
    /// against unbounded recursion over hostile input, which
    /// [RFC 8259 §9](https://www.rfc-editor.org/rfc/rfc8259#section-9) expressly permits. It
    /// bounds both the scanner ([`parse_json`], [`json_to_xml`]) and the writers
    /// ([`xml_to_json`], [`serialize_json`]); text or a tree that nests deeper is rejected with
    /// an error rather than overflowing the stack. The default is
    /// [`JsonOptions::DEFAULT_MAX_DEPTH`].
    pub max_depth: usize,
}

impl Default for JsonOptions<'_> {
    fn default() -> Self {
        Self {
            liberal: false,
            duplicates: Duplicates::UseFirst,
            escape: false,
            indent: false,
            fallback: None,
            function: JsonFunction::ParseJson,
            max_depth: Self::DEFAULT_MAX_DEPTH,
        }
    }
}

impl std::fmt::Debug for JsonOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonOptions")
            .field("liberal", &self.liberal)
            .field("duplicates", &self.duplicates)
            .field("escape", &self.escape)
            .field("indent", &self.indent)
            .field("fallback", &self.fallback.is_some())
            .field("function", &self.function)
            .field("max_depth", &self.max_depth)
            .finish()
    }
}

impl<'a> JsonOptions<'a> {
    /// The default value of [`JsonOptions::max_depth`]: 256 levels of nesting.
    ///
    /// Deep enough that no JSON a document interchange format realistically produces comes near
    /// it, shallow enough that the recursive descent it bounds cannot exhaust an ordinary
    /// thread's stack.
    pub const DEFAULT_MAX_DEPTH: usize = 256;

    /// The string handling these options prescribe.
    #[must_use]
    pub fn string_policy(&self) -> StringPolicy<'a> {
        StringPolicy {
            escape: self.escape,
            fallback: self.fallback,
        }
    }

    /// Reads the option map `fn:parse-json` and friends take as their second argument.
    ///
    /// Unrecognised options are ignored, as [F&O 3.1
    /// §2.1.1](https://www.w3.org/TR/xpath-functions-31/#options) requires; `escape` and
    /// `fallback` may not both be supplied.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fojs0005`] for an option value outside the set the specification defines,
    /// [`FnErrorCode::Xpty0004`] for one of the wrong type, and [`FnErrorCode::Fojs0004`] when
    /// `fn:json-to-xml` is asked to validate its result.
    pub fn from_map(map: &XdmMap, mut defaults: Self) -> FnResult<Self> {
        let mut escape_given = false;
        for (key, value) in map.entries() {
            let name = key.string_value();
            match name.as_str() {
                "liberal" => defaults.liberal = boolean_option(&name, value)?,
                "escape" => {
                    defaults.escape = boolean_option(&name, value)?;
                    escape_given = true;
                }
                "indent" => defaults.indent = boolean_option(&name, value)?,
                "validate" if defaults.function.defines_validate() => {
                    validate_option(&name, value)?;
                }
                "duplicates" => {
                    let duplicates = Duplicates::parse(&string_option(&name, value)?)?;
                    if !defaults.function.accepts(duplicates) {
                        return Err(FnError::new(
                            FnErrorCode::Fojs0005,
                            format!(
                                "`duplicates={}` is not a value this function accepts",
                                string_option(&name, value)?
                            ),
                        ));
                    }
                    defaults.duplicates = duplicates;
                }
                _ => {}
            }
        }
        // The `fallback` option is a function item, which this crate can inspect but not call;
        // validating it here means an ill-typed one is reported even by a caller — `fn:xml-to-json`
        // among them — that never invokes it.
        let fallback = fallback_is_supplied(map)?;
        // F&O 3.1 §17.5.1: supplying `fallback` together with `escape=true` is FOJS0005 — the
        // two options prescribe different treatments of the same characters.
        if fallback && escape_given && defaults.escape {
            return Err(FnError::new(
                FnErrorCode::Fojs0005,
                "the `fallback` option cannot be combined with `escape=true`",
            ));
        }
        Ok(defaults)
    }
}

/// Whether the option map supplies a `fallback` function, checking that it is one.
///
/// [F&O 3.1 §17.5.1](https://www.w3.org/TR/xpath-functions-31/#func-parse-json) types the option
/// `function(xs:string) as xs:string`, so anything that is not a single function item of arity
/// one — a string, or a function of the wrong arity — is a type error, whether or not the input
/// ever makes the processor call it.
///
/// The item itself stays where it is: a general function item is
/// [`oxixml_infoset::XdmFunction`], an opaque handle only its own evaluator can invoke, so the
/// caller that owns function items builds the [`parser::Fallback`] closure.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the option's value is not a single arity-one function item.
pub fn fallback_is_supplied(map: &XdmMap) -> FnResult<bool> {
    for (key, value) in map.entries() {
        if key.string_value() != "fallback" {
            continue;
        }
        let acceptable = match value.as_slice() {
            [Item::Function(function)] => function.arity() == 1,
            // XDM 3.1 §2.8.2 makes maps and arrays function items of arity one too.
            [Item::Map(_) | Item::Array(_)] => true,
            _ => false,
        };
        return if acceptable {
            Ok(true)
        } else {
            Err(FnError::new(
                FnErrorCode::Xpty0004,
                "the `fallback` option must be a single function of arity one",
            ))
        };
    }
    Ok(false)
}

/// Turns the sequence a `fallback` function returned into the replacement text.
///
/// # Errors
///
/// [`FnErrorCode::Xpty0004`] when the result is not a single string.
pub fn fallback_result(result: &Sequence) -> FnResult<String> {
    match result.as_slice() {
        [Item::Atomic(atomic)] if atomic.atomic_type().is_text() => Ok(atomic.string_value()),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            "the `fallback` function must return a single xs:string",
        )),
    }
}

/// Whether the value carries no type information of its own, so that the option's required type
/// governs how its lexical form is read.
fn is_untyped(atomic: &AtomicValue) -> bool {
    atomic.atomic_type() == AtomicType::UntypedAtomic
}

/// Reads `fn:json-to-xml`'s `validate` option, which this processor can honour in one direction
/// only.
///
/// [F&O 3.1 §17.5.3](https://www.w3.org/TR/xpath-functions-31/#func-json-to-xml) has `validate`
/// ask for the result to be validated against the schema for the XML representation of JSON,
/// "and it is a dynamic error [err:FOJS0004] if the processor is not schema-aware". This one is
/// not: it has no schema processor to hand and no way to attach the type annotations a validated
/// tree would carry, so the honest answer is the error rather than a tree that only looks
/// validated. `validate=false` asks for nothing and is accepted.
fn validate_option(name: &str, value: &Sequence) -> FnResult<()> {
    if boolean_option(name, value)? {
        return Err(FnError::new(
            FnErrorCode::Fojs0004,
            "the `validate` option asks for schema validation, which this processor is not \
             schema-aware enough to perform",
        ));
    }
    Ok(())
}

fn boolean_option(name: &str, value: &Sequence) -> FnResult<bool> {
    let atomic = single_atomic(name, value)?;
    if let Some(flag) = atomic.as_bool() {
        return Ok(flag);
    }
    // An untyped value — a node's content, say — is cast to the option's required type.
    if is_untyped(&atomic) {
        match atomic.string_value().trim() {
            "true" | "1" => return Ok(true),
            "false" | "0" => return Ok(false),
            _ => {}
        }
    }
    Err(FnError::new(
        FnErrorCode::Xpty0004,
        format!("the `{name}` option must be an xs:boolean"),
    ))
}

fn string_option(name: &str, value: &Sequence) -> FnResult<String> {
    let atomic = single_atomic(name, value)?;
    if atomic.atomic_type().is_text() || is_untyped(&atomic) {
        Ok(atomic.string_value())
    } else {
        Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("the `{name}` option must be an xs:string"),
        ))
    }
}

fn single_atomic(name: &str, value: &Sequence) -> FnResult<AtomicValue> {
    let atomized = value.clone().atomize()?;
    match atomized.as_slice() {
        [Item::Atomic(atomic)] => Ok(atomic.clone()),
        _ => Err(FnError::new(
            FnErrorCode::Xpty0004,
            format!("the `{name}` option must be a single atomic value"),
        )),
    }
}

/// [`fn:parse-json`](https://www.w3.org/TR/xpath-functions-31/#func-parse-json).
///
/// # Errors
///
/// [`FnErrorCode::Fojs0001`] for a syntax error — including a text nesting deeper than
/// [`JsonOptions::max_depth`] — [`FnErrorCode::Fojs0003`] for a duplicate key under
/// `duplicates=reject`, and [`FnErrorCode::Fojs0005`] when `duplicates=retain`, which only
/// `fn:json-to-xml` can honour.
///
/// # Examples
///
/// ```
/// use oxixml_fn::json::{parse_json, JsonOptions};
///
/// let value = parse_json("[1,2]", &JsonOptions::default())?;
/// assert_eq!(value.len(), 1);
/// assert!(parse_json("[1,", &JsonOptions::default()).is_err());
/// # Ok::<(), oxixml_fn::FnError>(())
/// ```
pub fn parse_json(json: &str, options: &JsonOptions<'_>) -> FnResult<Sequence> {
    if options.duplicates == Duplicates::Retain {
        return Err(FnError::new(
            FnErrorCode::Fojs0005,
            "`duplicates=retain` is only defined for fn:json-to-xml",
        ));
    }
    let value = parser::parse(
        json,
        options.duplicates,
        &options.string_policy(),
        options.max_depth,
    )?;
    Ok(to_sequence(&value))
}

/// Converts one parsed JSON value into the XDM value the specification prescribes.
fn to_sequence(value: &JsonValue) -> Sequence {
    match value {
        JsonValue::Null => Sequence::empty(),
        JsonValue::Bool(flag) => Sequence::from(AtomicValue::boolean(*flag)),
        JsonValue::Number(number, _) => Sequence::from(AtomicValue::double(*number)),
        JsonValue::String(text) => Sequence::from(AtomicValue::string(text.clone())),
        JsonValue::Array(members) => Sequence::from(Item::Array(XdmArray::from_members(
            members.iter().map(to_sequence),
        ))),
        JsonValue::Object(entries) => {
            let mut map = XdmMap::new();
            for (key, member) in entries {
                map.insert(AtomicValue::string(key.clone()), to_sequence(member));
            }
            Sequence::from(Item::Map(map))
        }
    }
}

/// The `json` output method's serialization of an arbitrary XDM value, as
/// [Serialization 3.1 §4.7](https://www.w3.org/TR/xslt-xquery-serialization-31/#json-output)
/// defines it.
///
/// # Errors
///
/// [`FnErrorCode::Sere0023`] when the value contains an item the JSON output method cannot
/// represent — a node, or a sequence of more than one item.
pub fn serialize_json(value: &Sequence, indent: bool) -> FnResult<String> {
    serialize_json_with(value, &JsonOutputOptions::new(indent))
}

/// Writes one node the way the `json-node-output-method` parameter asks for.
pub type NodeSerializer<'a> = dyn Fn(&oxixml_infoset::NodeRef) -> FnResult<String> + 'a;

/// How the `json` output method behaves, beyond the `indent` flag
/// [`serialize_json`] takes.
///
/// The defaults are Serialization 3.1's: duplicate keys are an error, nodes are written with the
/// `xml` output method, and the output is Unicode, so nothing outside JSON's own escape set is
/// escaped.
pub struct JsonOutputOptions<'a> {
    /// The `indent` parameter.
    pub indent: bool,
    /// The `allow-duplicate-names` parameter. When it is `false` — the default — two keys whose
    /// *string values* are equal are [`FnErrorCode::Sere0022`], even though the map itself holds
    /// them as distinct keys (`xs:QName("foo")` and `"foo"`, say).
    pub allow_duplicate_names: bool,
    /// Writes a node the way the `json-node-output-method` parameter asks for; the result is
    /// wrapped in a JSON string. `None` restores the pre-3.1 behaviour of refusing nodes.
    pub node_serializer: Option<&'a NodeSerializer<'a>>,
    /// Whether a character outside the target encoding's repertoire must be written as a
    /// `\uXXXX` escape (a surrogate *pair* above the BMP). See [`escapes_non_ascii`].
    pub escape_beyond: Option<u32>,
    /// How deeply maps and arrays may nest in the value being written.
    ///
    /// The writer descends one native stack frame per level, and the value comes from the
    /// caller, so a value nested past this ceiling is reported as [`FnErrorCode::Sere0023`]
    /// rather than allowed to overflow the stack. The default is
    /// [`JsonOptions::DEFAULT_MAX_DEPTH`].
    pub max_depth: usize,
    /// The `use-character-maps` parameter: one entry per mapped character, in the order the
    /// output definition settled on.
    ///
    /// A character with an entry is replaced by its string **verbatim**, in place of every
    /// escape this method would otherwise apply — XSLT 3.0 §25.1: "If a character is mapped,
    /// then it is not subjected to XML or HTML escaping", and the same holds of JSON's own
    /// escaping (the XSLT 3.0 suite's `output-0708` maps `/` to `/` precisely to suppress the
    /// `\/` this method writes). It applies to the characters of the *value*: to a string, to
    /// a map's keys, and not to the punctuation the writer generates around them.
    pub character_map: &'a [(char, String)],
}

impl JsonOutputOptions<'_> {
    /// The defaults, with `indent` as given.
    #[must_use]
    pub const fn new(indent: bool) -> Self {
        Self {
            indent,
            allow_duplicate_names: false,
            node_serializer: None,
            escape_beyond: None,
            max_depth: JsonOptions::DEFAULT_MAX_DEPTH,
            character_map: &[],
        }
    }
}

/// The highest code point an encoding can write directly, for the encodings whose repertoire is
/// decidable without a full transcoder; `None` means "Unicode, so all of them".
///
/// Serialization 3.1 §10 requires a character the target encoding cannot represent to be written
/// as a JSON `\uXXXX` escape. Only the encodings below are treated as limited: guessing at the
/// repertoire of an arbitrary named encoding would escape characters that need not be escaped.
#[must_use]
pub fn escapes_non_ascii(encoding: &str) -> Option<u32> {
    let normalized = encoding.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "us-ascii" | "ascii" => Some(0x7f),
        "iso-8859-1" | "iso8859-1" | "latin1" | "latin-1" => Some(0xff),
        _ => None,
    }
}

/// [`serialize_json`] with the rest of the `json` output method's parameters.
///
/// # Errors
///
/// As [`serialize_json`], plus [`FnErrorCode::Sere0022`] for a duplicate key,
/// [`FnErrorCode::Sere0020`] for `NaN` or an infinity, and [`FnErrorCode::Sere0023`] for a value
/// nested deeper than [`JsonOutputOptions::max_depth`].
pub fn serialize_json_with(value: &Sequence, options: &JsonOutputOptions) -> FnResult<String> {
    let mut out = String::new();
    match value.as_slice() {
        // Serialization 3.1 §10: the empty sequence is the JSON `null`, not the empty string.
        [] => return Ok("null".to_owned()),
        [item] => write_json_item(&mut out, item, options, 0)?,
        _ => {
            return Err(FnError::new(
                FnErrorCode::Sere0023,
                "the JSON output method serializes at most one item",
            ));
        }
    }
    Ok(out)
}

/// Writes one item, `depth` maps and arrays deep inside the value.
///
/// The descent is recursive and the value is the caller's, so the ceiling is checked here: a
/// map or array nested past it is an error rather than an exhausted stack.
fn write_json_item(
    out: &mut String,
    item: &Item,
    options: &JsonOutputOptions,
    depth: usize,
) -> FnResult<()> {
    if depth > options.max_depth {
        return Err(FnError::new(
            FnErrorCode::Sere0023,
            format!(
                "the value nests more than {} levels deep, which is this serializer's configurable limit",
                options.max_depth
            ),
        ));
    }
    let indent = options.indent;
    match item {
        Item::Atomic(atomic) => out.push_str(&json_output_atomic(atomic, options)?),
        Item::Array(array) => {
            let members: Vec<&Sequence> = array.members().collect();
            write_json_group(out, '[', ']', members.len(), indent, depth, |out, index| {
                write_json_sequence(out, members[index], options, depth + 1)
            })?;
        }
        Item::Map(map) => {
            let entries: Vec<(&AtomicValue, &Sequence)> = map.entries().collect();
            if !options.allow_duplicate_names {
                let mut seen: Vec<String> = Vec::with_capacity(entries.len());
                for (key, _) in &entries {
                    let rendered = key.string_value();
                    if seen.contains(&rendered) {
                        return Err(FnError::new(
                            FnErrorCode::Sere0022,
                            format!("`{rendered}` is used as a key twice"),
                        ));
                    }
                    seen.push(rendered);
                }
            }
            write_json_group(out, '{', '}', entries.len(), indent, depth, |out, index| {
                let (key, value) = entries[index];
                out.push_str(&json_output_string(&key.string_value(), options));
                out.push(':');
                if indent {
                    out.push(' ');
                }
                write_json_sequence(out, value, options, depth + 1)
            })?;
        }
        Item::Function(_) => {
            return Err(FnError::new(
                FnErrorCode::Sere0021,
                "the JSON output method cannot serialize a function item",
            ));
        }
        Item::Node(node) => {
            // Serialization 3.1 §10: a node is serialized with the `json-node-output-method`
            // and the resulting *lexical XML* is written as a JSON string.
            let Some(serializer) = options.node_serializer else {
                return Err(FnError::new(
                    FnErrorCode::Sere0023,
                    "the JSON output method cannot serialize a node",
                ));
            };
            let text = serializer(node)?;
            // The nested serialization ran with the same parameters, so the `xml` writer has
            // already applied the character map to the node's own characters; mapping the
            // result again would replace what a mapping produced.
            out.push_str(&json_string_literal(&text, options, &[]));
        }
    }
    Ok(())
}

fn write_json_sequence(
    out: &mut String,
    value: &Sequence,
    options: &JsonOutputOptions,
    depth: usize,
) -> FnResult<()> {
    match value.as_slice() {
        [] => out.push_str("null"),
        [item] => write_json_item(out, item, options, depth)?,
        _ => {
            return Err(FnError::new(
                FnErrorCode::Sere0023,
                "the JSON output method serializes at most one item per entry",
            ));
        }
    }
    Ok(())
}

/// The `json` output method's spelling of an atomic value.
///
/// `NaN` and the infinities are [`FnErrorCode::Sere0020`], and the string escaping is the
/// output method's — the solidus escaped, and whatever the encoding cannot carry escaped too.
fn json_output_atomic(value: &AtomicValue, options: &JsonOutputOptions) -> FnResult<String> {
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { "true" } else { "false" }.to_owned());
    }
    if value.atomic_type().is_numeric() {
        let rendered = value.string_value();
        if rendered == "NaN" || rendered.ends_with("INF") {
            return Err(FnError::new(
                FnErrorCode::Sere0020,
                "the JSON output method cannot serialize NaN or infinity",
            ));
        }
        return Ok(rendered);
    }
    Ok(json_output_string(&value.string_value(), options))
}

/// A JSON string literal as the `json` output method writes it: the character map is applied
/// first, the solidus is escaped, and a character the target encoding cannot carry becomes a
/// `\uXXXX` escape — a surrogate pair above the BMP, since JSON's escape is a UTF-16 code unit.
fn json_output_string(text: &str, options: &JsonOutputOptions) -> String {
    json_string_literal(text, options, options.character_map)
}

/// [`json_output_string`] with the character map given explicitly, so that text which has
/// *already* been through it — the lexical XML of a nested node, written by the
/// `json-node-output-method` with the same parameter set — is not mapped a second time.
fn json_string_literal(
    text: &str,
    options: &JsonOutputOptions,
    character_map: &[(char, String)],
) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        // A mapped character is replaced by its string as it stands, in place of every escape
        // below: that is what makes a map able to suppress one.
        if let Some((_, replacement)) = character_map.iter().find(|(from, _)| *from == character) {
            out.push_str(replacement);
            continue;
        }
        let code = character as u32;
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            // Serialization 3.1 §10 lists the solidus among the characters this method escapes.
            '/' => out.push_str("\\/"),
            _ if code < 0x20
                || !parser::is_xml_character(character)
                || options.escape_beyond.is_some_and(|limit| code > limit) =>
            {
                out.push_str(&xml::json_escape(character));
            }
            _ => out.push(character),
        }
    }
    out.push('"');
    out
}

fn write_json_group(
    out: &mut String,
    open: char,
    close: char,
    count: usize,
    indent: bool,
    depth: usize,
    mut member: impl FnMut(&mut String, usize) -> FnResult<()>,
) -> FnResult<()> {
    out.push(open);
    for index in 0..count {
        if index > 0 {
            out.push(',');
        }
        if indent {
            out.push('\n');
            for _ in 0..=depth {
                out.push_str("  ");
            }
        }
        member(out, index)?;
    }
    if indent && count > 0 {
        out.push('\n');
        for _ in 0..depth {
            out.push_str("  ");
        }
    }
    out.push(close);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::XdmNode;

    fn string_of(value: &Sequence) -> String {
        value
            .first()
            .and_then(Item::as_atomic)
            .map(AtomicValue::string_value)
            .unwrap_or_default()
    }

    /// The `json` output method honours `use-character-maps`, and a mapped character is written
    /// as its replacement in place of the escape this method would otherwise apply.
    ///
    /// The XSLT 3.0 suite's `output-0708` maps the solidus to itself for exactly that reason:
    /// `{"a":"http://www.w3.org/2001/XMLSchema"}` where the unmapped writer gives
    /// `{"a":"http:\/\/www.w3.org\/2001\/XMLSchema"}`.
    #[test]
    fn a_character_map_suppresses_this_methods_own_escaping() -> FnResult<()> {
        let map = XdmMap::new().put(
            AtomicValue::string("a"),
            Sequence::from(AtomicValue::string("http://www.w3.org/2001/XMLSchema")),
        );
        let value = Sequence::singleton(Item::from(map));
        assert_eq!(
            serialize_json_with(&value, &JsonOutputOptions::new(false))?,
            r#"{"a":"http:\/\/www.w3.org\/2001\/XMLSchema"}"#
        );
        let mapping = [('/', "/".to_owned())];
        let mapped = JsonOutputOptions {
            character_map: &mapping,
            ..JsonOutputOptions::new(false)
        };
        assert_eq!(
            serialize_json_with(&value, &mapped)?,
            r#"{"a":"http://www.w3.org/2001/XMLSchema"}"#
        );
        // The map reaches a *key* as readily as a value, and its replacement is written as it
        // stands rather than escaped in turn.
        let keys = XdmMap::new().put(
            AtomicValue::string("a/b"),
            Sequence::from(AtomicValue::string("v")),
        );
        let quoting = [('/', "\\\"".to_owned())];
        let mapped = JsonOutputOptions {
            character_map: &quoting,
            ..JsonOutputOptions::new(false)
        };
        assert_eq!(
            serialize_json_with(&Sequence::singleton(Item::from(keys)), &mapped)?,
            r#"{"a\"b":"v"}"#
        );
        Ok(())
    }

    #[test]
    fn scalars_map_onto_the_data_model() -> FnResult<()> {
        let options = JsonOptions::default();
        assert!(parse_json("null", &options)?.is_empty());
        assert_eq!(string_of(&parse_json("true", &options)?), "true");
        assert_eq!(string_of(&parse_json("93.7", &options)?), "93.7");
        assert_eq!(string_of(&parse_json("4e2", &options)?), "400");
        Ok(())
    }

    fn option(name: &str, value: AtomicValue) -> XdmMap {
        XdmMap::new().put(AtomicValue::string(name), Sequence::from(value))
    }

    /// The two families accept different `duplicates` values: `retain` cannot be represented by
    /// a map, and `use-last` cannot be represented by the ordered XML vocabulary.
    #[test]
    fn each_function_rejects_the_duplicates_value_it_cannot_express() {
        let json_to_xml = JsonOptions {
            duplicates: Duplicates::Retain,
            function: JsonFunction::JsonToXml,
            ..JsonOptions::default()
        };
        let error = JsonOptions::from_map(
            &option("duplicates", AtomicValue::string("use-last")),
            json_to_xml,
        )
        .expect_err("use-last is not a fn:json-to-xml value");
        assert!(error.is(FnErrorCode::Fojs0005));

        let error = JsonOptions::from_map(
            &option("duplicates", AtomicValue::string("retain")),
            JsonOptions::default(),
        )
        .expect_err("retain is not a fn:parse-json value");
        assert!(error.is(FnErrorCode::Fojs0005));

        // The values each function *does* accept still read through.
        let options = JsonOptions::from_map(
            &option("duplicates", AtomicValue::string("use-first")),
            JsonOptions::default(),
        )
        .expect("use-first is a fn:parse-json value");
        assert_eq!(options.duplicates, Duplicates::UseFirst);
    }

    /// The options `fn:json-to-xml` is given when the registry calls it.
    fn json_to_xml_options() -> JsonOptions<'static> {
        JsonOptions {
            duplicates: Duplicates::Retain,
            function: JsonFunction::JsonToXml,
            ..JsonOptions::default()
        }
    }

    /// F&O 3.1 §17.5.3: `validate=true` is `FOJS0004` on a processor that is not schema-aware,
    /// which this one is not.
    #[test]
    fn asking_json_to_xml_to_validate_is_an_error() {
        let error = JsonOptions::from_map(
            &option("validate", AtomicValue::boolean(true)),
            json_to_xml_options(),
        )
        .expect_err("this processor cannot validate");
        assert!(error.is(FnErrorCode::Fojs0004));

        // `validate=false` asks for nothing.
        let options = JsonOptions::from_map(
            &option("validate", AtomicValue::boolean(false)),
            json_to_xml_options(),
        )
        .expect("validate=false asks for no validation");
        assert_eq!(options.function, JsonFunction::JsonToXml);
    }

    /// The option's value is an `xs:boolean`, so anything else is a type error — including the
    /// empty sequence and a two-item one, neither of which is a single boolean.
    #[test]
    fn an_ill_typed_validate_option_is_a_type_error() {
        for value in [
            Sequence::empty(),
            Sequence::from(AtomicValue::string("ECMA-262")),
            Sequence::from_iter([AtomicValue::boolean(true), AtomicValue::boolean(true)]),
        ] {
            let map = XdmMap::new().put(AtomicValue::string("validate"), value);
            let error = JsonOptions::from_map(&map, json_to_xml_options())
                .expect_err("the value is not a single xs:boolean");
            assert!(error.is(FnErrorCode::Xpty0004), "{error}");
        }
    }

    /// `validate` is not one of `fn:parse-json`'s options, and F&O 3.1 §2.1.1 ignores an option
    /// the function does not define — value and all.
    #[test]
    fn parse_json_ignores_the_validate_option() -> FnResult<()> {
        for value in [
            Sequence::from(AtomicValue::boolean(true)),
            Sequence::from(AtomicValue::string("not even a boolean")),
        ] {
            let map = XdmMap::new().put(AtomicValue::string("validate"), value);
            let options = JsonOptions::from_map(&map, JsonOptions::default())?;
            assert_eq!(options.function, JsonFunction::ParseJson);
        }
        Ok(())
    }

    #[test]
    fn an_ill_typed_fallback_is_a_type_error() {
        let error = fallback_is_supplied(&option("fallback", AtomicValue::string("dummy")))
            .expect_err("a string is not a function item");
        assert!(error.is(FnErrorCode::Xpty0004));
        assert!(!fallback_is_supplied(&XdmMap::new()).expect("no fallback was supplied"));
    }

    /// W3C bug #28995: `escape=true` escapes the backslash and whatever XML cannot spell, and
    /// leaves the quotation mark — which needs an escape only inside a JSON literal — alone.
    #[test]
    fn the_escape_option_does_not_re_escape_a_quotation_mark() -> FnResult<()> {
        let options = JsonOptions {
            escape: true,
            ..JsonOptions::default()
        };
        assert_eq!(
            string_of(&parse_json("\"a\\\"b\\\\c\\td\"", &options)?),
            "a\"b\\\\c\\td"
        );
        Ok(())
    }

    #[test]
    fn retain_is_rejected_for_parse_json() {
        let options = JsonOptions {
            duplicates: Duplicates::Retain,
            ..JsonOptions::default()
        };
        let error = parse_json("{}", &options).expect_err("retain is not allowed");
        assert!(error.is(FnErrorCode::Fojs0005));
    }

    #[test]
    fn the_xml_representation_round_trips() -> FnResult<()> {
        let options = JsonOptions::default();
        let document = json_to_xml("{\"a\":[1,\"x\",true,null]}", &options)?;
        let root = document
            .children()
            .into_iter()
            .next()
            .expect("the root element");
        assert_eq!(xml_to_json(&root, &options)?, "{\"a\":[1,\"x\",true,null]}");
        Ok(())
    }

    /// Both directions used to recurse once per level of nesting with nothing to stop them.
    #[test]
    fn deeply_nested_text_is_rejected_by_both_directions() {
        let deep = format!("{}{}", "[".repeat(50_000), "]".repeat(50_000));
        let options = JsonOptions::default();
        assert!(
            parse_json(&deep, &options)
                .expect_err("past the nesting ceiling")
                .is(FnErrorCode::Fojs0001)
        );
        assert!(
            json_to_xml(&deep, &options)
                .expect_err("past the nesting ceiling")
                .is(FnErrorCode::Fojs0001)
        );
    }

    /// `fn:xml-to-json` walks a tree the caller supplies, so its own ceiling has to hold even
    /// when the tree did not come from [`json_to_xml`].
    #[test]
    fn xml_to_json_stops_at_its_ceiling() -> FnResult<()> {
        let shallow = JsonOptions {
            max_depth: 3,
            ..JsonOptions::default()
        };
        let default = JsonOptions::default();
        let document = json_to_xml("[[[1]]]", &default)?;
        let root = document
            .children()
            .into_iter()
            .next()
            .expect("the root element");
        assert_eq!(xml_to_json(&root, &shallow)?, "[[[1]]]");

        let deeper = json_to_xml("[[[[1]]]]", &default)?;
        let root = deeper
            .children()
            .into_iter()
            .next()
            .expect("the root element");
        let error = xml_to_json(&root, &shallow).expect_err("past the ceiling");
        assert!(error.is(FnErrorCode::Fojs0006));
        assert!(error.to_string().contains('3'));
        Ok(())
    }

    /// The `json` output method descends the value it is handed, which an evaluator can nest
    /// arbitrarily deep through higher-order functions.
    #[test]
    fn the_json_output_method_stops_at_its_ceiling() {
        let mut nested = Sequence::from(AtomicValue::integer(1_i64));
        for _ in 0..1_000 {
            nested = Sequence::from(Item::Array(XdmArray::from_members([nested])));
        }
        let error = serialize_json(&nested, false).expect_err("past the ceiling");
        assert!(error.is(FnErrorCode::Sere0023));
        assert!(
            error
                .to_string()
                .contains(&JsonOptions::DEFAULT_MAX_DEPTH.to_string())
        );
    }
}
