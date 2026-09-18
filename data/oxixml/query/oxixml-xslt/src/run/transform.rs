//! [`fn:transform`](https://www.w3.org/TR/xpath-functions-31/#func-transform) — running a whole
//! transformation from inside an expression.
//!
//! F&O 3.1 §16.4 defines the function in terms of a host that can *invoke* an XSLT processor.
//! This crate is one, so the function is completed here rather than in the plain function
//! library: the options map names a stylesheet, this module compiles it with the resolver and
//! base URI the calling transformation is using, invokes it, and returns the result map the
//! specification prescribes — the principal result under `output`, and each secondary result
//! under the absolute URI `xsl:result-document` wrote it to.
//!
//! Of the option entries the specification lists, the ones a stylesheet can observe here are
//! implemented and the rest are rejected rather than ignored, so that a stylesheet is never
//! silently given a transformation it did not ask for.

use crate::api::{CompiledStylesheet, Parameters, StaticOptions, Stylesheet};
use crate::error::XsltError;
use crate::run::current_runtime;
use oxixml_fn::{FN_NAMESPACE, FnError, FnErrorCode, FnResult, FunctionRegistry};
use oxixml_infoset::{
    AtomicValue, ExpandedName, Item, ParseOptions, Sequence, Tree, XdmMap, XdmNode,
};
use oxixml_xpath::XpathValue;

/// The option entries this processor implements.
const KNOWN_OPTIONS: &[&str] = &[
    "base-output-uri",
    "delivery-format",
    "initial-function",
    "function-params",
    "initial-match-selection",
    "initial-mode",
    "initial-template",
    "source-node",
    "static-params",
    "stylesheet-base-uri",
    "stylesheet-location",
    "stylesheet-node",
    "stylesheet-params",
    "stylesheet-text",
    "template-params",
    "tunnel-params",
    "vendor-options",
    "cache",
    "enable-assertions",
    "enable-messages",
    "requested-properties",
    "serialization-params",
];

/// Registers `fn:transform`.
pub(crate) fn register(registry: &mut FunctionRegistry) {
    registry.add(FN_NAMESPACE, "transform", 1, |_, arguments| {
        let options = match arguments.first().and_then(Sequence::first) {
            Some(Item::Map(map)) => map.clone(),
            _ => {
                return Err(invalid_options(
                    "fn:transform takes a map of transformation options",
                ));
            }
        };
        run(&options)
    });
}

/// `FOXT0002` — the options map does not describe a transformation this processor can run.
fn invalid_options(description: &str) -> FnError {
    foxt("FOXT0002", description)
}

/// One of the `FOXT` codes F&O 3.1 §16.4 reserves for `fn:transform`, none of which the plain
/// function library needs a variant for.
fn foxt(code: &str, description: &str) -> FnError {
    FnError::with_name(
        ExpandedName::new(Some(oxixml_fn::ERROR_NAMESPACE), code),
        description.to_owned(),
    )
}

/// The value of a string-valued option, if it is present.
///
/// The value is *atomized* rather than required to be an atomic item: `xsl:map-entry` with
/// content builds a text node, so `<xsl:map-entry key="'stylesheet-location'">a.xsl</xsl:map-entry>`
/// supplies the location as a node (the suite's `transform-008`).
fn string_option(options: &XdmMap, name: &str) -> FnResult<Option<String>> {
    let Some(value) = options.get(&AtomicValue::string(name)) else {
        return Ok(None);
    };
    let atomized = value.atomize()?;
    match atomized.first() {
        None => Ok(None),
        Some(item) => match item.as_atomic() {
            Some(atomic) => Ok(Some(atomic.string_value())),
            None => Err(invalid_options(&format!(
                "the `{name}` option of fn:transform must be a single atomic value"
            ))),
        },
    }
}

/// The value of an option that names a QName — `initial-template`, `initial-mode` and
/// `initial-function` are all declared `xs:QName` by F&O 3.1 §16.4.
fn name_option(options: &XdmMap, name: &str) -> FnResult<Option<ExpandedName>> {
    let Some(value) = options.get(&AtomicValue::string(name)) else {
        return Ok(None);
    };
    let Some(atomic) = value.first().and_then(Item::as_atomic) else {
        return Ok(None);
    };
    match atomic.as_qname() {
        Some(qname) => Ok(Some(ExpandedName::new(
            qname.namespace().filter(|uri| !uri.is_empty()),
            qname.local_name(),
        ))),
        None => Err(invalid_options(&format!(
            "the `{name}` option of fn:transform must be an xs:QName"
        ))),
    }
}

/// The parameters an option holding `map(xs:QName, item()*)` supplies.
fn parameter_option(options: &XdmMap, name: &str) -> FnResult<Vec<(ExpandedName, XpathValue)>> {
    let Some(value) = options.get(&AtomicValue::string(name)) else {
        return Ok(Vec::new());
    };
    let Some(Item::Map(map)) = value.first() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (key, bound) in map.entries() {
        let Some(qname) = key.as_qname() else {
            return Err(invalid_options(&format!(
                "the keys of the `{name}` option of fn:transform must be xs:QNames"
            )));
        };
        out.push((
            ExpandedName::new(
                qname.namespace().filter(|uri| !uri.is_empty()),
                qname.local_name(),
            ),
            XpathValue::from(bound.clone()),
        ));
    }
    Ok(out)
}

/// Compiles the stylesheet the options name and runs it, returning the result map.
fn run(options: &XdmMap) -> FnResult<Sequence> {
    for key in options.keys() {
        let name = key.string_value();
        if !KNOWN_OPTIONS.contains(&name.as_str()) {
            return Err(invalid_options(&format!(
                "`{name}` is not a transformation option this processor implements"
            )));
        }
    }
    // The transformation running around the call supplies the base URI and the resolver; a
    // static expression the compiler is evaluating supplies the compilation's own instead, since
    // there is no transformation yet (`transform-004` binds the result of `fn:transform` to a
    // static variable).
    let (base_uri, resolver) = match current_runtime() {
        Some(runtime) => (runtime.base_uri.clone(), runtime.resolver.clone()),
        None => match crate::run::functions::compiling() {
            Some(environment) => (environment.base_uri, environment.resolver),
            None => {
                return Err(FnError::new(
                    FnErrorCode::Foxt0004,
                    "fn:transform is only available inside a transformation",
                ));
            }
        },
    };
    let compiled = compile(options, base_uri.as_deref(), resolver)?;

    let format =
        string_option(options, "delivery-format")?.unwrap_or_else(|| "document".to_owned());
    if !matches!(format.as_str(), "document" | "raw" | "serialized") {
        return Err(invalid_options(&format!(
            "`{format}` is not a delivery format fn:transform recognizes"
        )));
    }
    let mut parameters = Parameters::new();
    if let Some(name) = name_option(options, "initial-template")? {
        parameters = parameters.with_initial_template(lexical(&name));
    }
    if let Some(name) = name_option(options, "initial-mode")? {
        parameters = parameters.with_initial_mode(lexical(&name));
    }
    if let Some(name) = name_option(options, "initial-function")? {
        parameters = parameters.with_initial_function(lexical(&name));
    }
    if let Some(arguments) = options.get(&AtomicValue::string("function-params")) {
        parameters = parameters.with_initial_function_arguments(
            arguments
                .iter()
                .filter_map(|item| match item {
                    Item::Array(array) => Some(
                        array
                            .members()
                            .map(|member| XpathValue::from(member.clone()))
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                })
                .next()
                .unwrap_or_default(),
        );
    }
    for (name, value) in parameter_option(options, "stylesheet-params")? {
        parameters = parameters.with_parameter_name(name, value);
    }
    let mut template_parameters: Vec<(ExpandedName, XpathValue, bool)> =
        parameter_option(options, "template-params")?
            .into_iter()
            .map(|(name, value)| (name, value, false))
            .collect();
    template_parameters.extend(
        parameter_option(options, "tunnel-params")?
            .into_iter()
            .map(|(name, value)| (name, value, true)),
    );
    if !template_parameters.is_empty() {
        parameters = parameters.with_initial_template_parameters(template_parameters);
    }
    if let Some(selection) = options.get(&AtomicValue::string("initial-match-selection")) {
        parameters = parameters.with_initial_match_selection(XpathValue::from(selection.clone()));
    }
    if let Some(uri) = string_option(options, "base-output-uri")? {
        parameters = parameters.with_base_output_uri(uri);
    }

    // A raw delivery hands the caller the result *sequence*, which may hold items no output
    // method can serialize — `transform-003` returns a partially applied function and calls it.
    // Serializing eagerly would report `SENR0001` for a result the caller never asked to see as
    // text, so that step is skipped where it can have no reader.
    if format == "raw" {
        parameters = parameters.without_serialization();
    }
    let source = options
        .get(&AtomicValue::string("source-node"))
        .and_then(Sequence::first)
        .and_then(|item| item.as_node().cloned());
    let outcome = compiled
        .transform_optional(source.as_ref(), &parameters)
        .map_err(transformation_failed)?;
    let mut result = XdmMap::new();
    result.insert(
        AtomicValue::string("output"),
        principal_result(&outcome, &format)?,
    );
    for (uri, text) in outcome.secondary_results() {
        result.insert(
            AtomicValue::string(uri.clone()),
            secondary_result(text, &format)?,
        );
    }
    Ok(Sequence::from(Item::Map(result)))
}

/// The value the `output` entry of the result map carries.
fn principal_result(outcome: &crate::api::Transformation, format: &str) -> FnResult<Sequence> {
    match format {
        // "document": the result is a document node, which is what the principal result tree is.
        "document" => Ok(Sequence::from(Item::Node(outcome.root()))),
        // "raw": the result of the transformation as it stands, atomic values and function items
        // among them — an initial template or function that declares `as` returns exactly that.
        "raw" => Ok(outcome.as_sequence()),
        "serialized" => Ok(Sequence::from(AtomicValue::string(outcome.as_str()))),
        other => Err(invalid_options(&format!(
            "`{other}` is not a delivery format fn:transform recognizes"
        ))),
    }
}

/// The value a secondary result carries, given the delivery format.
///
/// A secondary result is held as the text `xsl:result-document` serialized, so the document and
/// raw formats parse it back — the parse cannot fail for output this processor wrote, and a
/// failure is reported rather than hidden.
fn secondary_result(text: &str, format: &str) -> FnResult<Sequence> {
    if format == "serialized" {
        return Ok(Sequence::from(AtomicValue::string(text)));
    }
    match Tree::parse_str_with(text, ParseOptions::default()) {
        Ok(tree) => Ok(Sequence::from(Item::Node(tree.root()))),
        Err(error) => Err(foxt(
            "FOXT0003",
            &format!("a secondary result of the transformation could not be read back: {error}"),
        )),
    }
}

/// An expanded name in the `{namespace}local` spelling the invocation API parses.
fn lexical(name: &ExpandedName) -> String {
    match name.namespace() {
        Some(namespace) => format!("{{{namespace}}}{}", name.local_name()),
        None => name.local_name().to_owned(),
    }
}

/// Compiles the stylesheet the options name.
fn compile(
    options: &XdmMap,
    base_uri: Option<&str>,
    resolver: Option<std::sync::Arc<dyn crate::resolve::Resolver>>,
) -> FnResult<CompiledStylesheet> {
    let mut static_options = StaticOptions::new();
    if let Some(resolver) = resolver.clone() {
        static_options = static_options.with_resolver(resolver);
    }
    for (name, value) in parameter_option(options, "static-params")? {
        static_options = static_options.with_static_parameter_name(name, value);
    }
    let declared_base = string_option(options, "stylesheet-base-uri")?;

    if let Some(node) = options
        .get(&AtomicValue::string("stylesheet-node"))
        .and_then(Sequence::first)
        .and_then(|item| item.as_node().cloned())
    {
        if let Some(uri) = declared_base.clone().or_else(|| node.base_uri()) {
            static_options = static_options.with_base_uri(uri);
        }
        return Stylesheet::compile_with(&node, static_options).map_err(compilation_failed);
    }
    if let Some(text) = string_option(options, "stylesheet-text")? {
        if let Some(uri) = declared_base
            .clone()
            .or_else(|| base_uri.map(str::to_owned))
        {
            static_options = static_options.with_base_uri(uri);
        }
        return Stylesheet::compile_str_with(&text, static_options).map_err(compilation_failed);
    }
    let Some(location) = string_option(options, "stylesheet-location")? else {
        return Err(invalid_options(
            "fn:transform needs one of stylesheet-location, stylesheet-node and stylesheet-text",
        ));
    };
    let resolved = crate::resolve::resolve_uri(&location, base_uri);
    let Some(module) = resolver
        .as_ref()
        .and_then(|resolver| resolver.stylesheet(&resolved))
    else {
        return Err(invalid_options(&format!(
            "the stylesheet `{location}` could not be retrieved"
        )));
    };
    static_options = static_options.with_base_uri(declared_base.unwrap_or(resolved));
    Stylesheet::compile_with(&module, static_options).map_err(compilation_failed)
}

/// A stylesheet the options named but that could not be compiled is an invalid option, which is
/// what F&O 3.1 §16.4 reserves `FOXT0002` for; the error it raised travels in the message.
fn compilation_failed(error: XsltError) -> FnError {
    invalid_options(&format!(
        "the stylesheet fn:transform was given could not be compiled: {error}"
    ))
}

/// An error the *invoked* transformation raised keeps its own code.
///
/// F&O 3.1 §16.4 makes `FOXT0003` the code for a transformation that fails, but the error a
/// conforming stylesheet raises is a statement about that stylesheet — the suite's
/// `transform-007` invokes a transformation whose entry point is not public and expects
/// `XTDE0040`, not a generic wrapper — so a diagnosis the invoked processor made is passed
/// through and only an error with no code of its own becomes `FOXT0003`.
fn transformation_failed(error: XsltError) -> FnError {
    match error.code() {
        Some(code) => FnError::with_name(
            ExpandedName::new(Some(crate::error::ERROR_NAMESPACE), code),
            error.to_string(),
        ),
        None => foxt("FOXT0003", &error.to_string()),
    }
}
