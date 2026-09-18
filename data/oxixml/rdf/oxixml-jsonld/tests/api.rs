//! Tests of the crate's public surface: the `oxjsonld`-compatible streaming API and the JSON-LD
//! 1.1 API operations.
//!
//! Everything here is self-contained — no test-suite checkout and no I/O of any kind is required.

use oxixml_jsonld::{
    ClosureLoader, JsonLdErrorCode, JsonLdOptions, JsonLdParser, JsonLdProcessingMode,
    JsonLdProfile, JsonLdProfileSet, JsonLdSerializer, JsonObject, JsonValue, LoadDocumentOptions,
    RdfDirection, RemoteDocument, compact, expand, flatten, from_rdf, to_rdf,
};
use oxixml_model::vocab::rdf;
use oxixml_model::{NamedNode, Quad, Term};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn json(text: &str) -> Result<JsonValue, Box<dyn std::error::Error>> {
    Ok(JsonValue::from_slice(text.as_bytes())?)
}

fn quads_of(document: &str) -> Result<Vec<Quad>, Box<dyn std::error::Error>> {
    let options = JsonLdOptions::new().with_base("http://example.com/");
    Ok(to_rdf(json(document)?, &options)?)
}

#[test]
fn parser_yields_quads_from_a_slice() -> TestResult {
    let document = br#"{
        "@context": {"schema": "http://schema.org/"},
        "@id": "http://example.com/foo",
        "@type": "schema:Person",
        "schema:name": "Foo"
    }"#;
    let quads = JsonLdParser::new()
        .for_slice(document)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 2);
    let person = NamedNode::new("http://schema.org/Person")?;
    assert!(
        quads
            .iter()
            .any(|quad| quad.predicate == rdf::TYPE && quad.object == person.clone())
    );
    Ok(())
}

#[test]
fn parser_yields_quads_from_a_reader() -> TestResult {
    let document = r#"{"@id": "http://example.com/s", "http://example.com/p": "o"}"#;
    let quads = JsonLdParser::new()
        .for_reader(document.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 1);
    Ok(())
}

#[test]
fn parser_reports_base_iri_and_prefixes() -> TestResult {
    let document = r#"{
        "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
        "@id": "foo",
        "schema:name": "Foo"
    }"#;
    let mut parser = JsonLdParser::new().for_reader(document.as_bytes());
    assert_eq!(parser.prefixes().collect::<Vec<_>>(), []);
    assert!(parser.base_iri().is_none());
    parser.next().transpose()?;
    assert_eq!(
        parser.prefixes().collect::<Vec<_>>(),
        [("schema", "http://schema.org/")]
    );
    assert_eq!(parser.base_iri(), Some("http://example.com/"));
    Ok(())
}

#[test]
fn parser_honours_an_explicit_base_iri() -> TestResult {
    let document = br#"{"@id": "s", "http://example.com/p": {"@id": "o"}}"#;
    let quads = JsonLdParser::new()
        .with_base_iri("http://example.org/base/")?
        .for_slice(document)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 1);
    let quad = quads.first().ok_or("expected one quad")?;
    assert_eq!(quad.subject.to_string(), "<http://example.org/base/s>");
    assert_eq!(quad.object.to_string(), "<http://example.org/base/o>");
    Ok(())
}

#[test]
fn parser_rejects_an_invalid_base_iri() {
    assert!(JsonLdParser::new().with_base_iri("not an IRI").is_err());
}

#[test]
fn parser_surfaces_json_syntax_errors() {
    let mut parser = JsonLdParser::new().for_slice(b"{ not json");
    let error = parser.next().and_then(Result::err);
    assert!(error.is_some(), "a malformed document must raise an error");
}

#[test]
fn streaming_profile_rejects_a_late_context() -> TestResult {
    let document = br#"{"@id": "http://example.com/s", "@context": {}, "p": "o"}"#;
    let error = JsonLdParser::new()
        .with_profile(JsonLdProfile::Streaming)
        .for_slice(document)
        .find_map(Result::err)
        .ok_or("expected a streaming key-order error")?;
    assert!(error.to_string().contains("streaming"));
    Ok(())
}

#[test]
fn streaming_profile_accepts_a_well_ordered_document() -> TestResult {
    let document = br#"{"@context": {"p": "http://example.com/p"}, "@type": "http://example.com/T",
        "@id": "http://example.com/s", "p": "o"}"#;
    let quads = JsonLdParser::new()
        .with_profile(JsonLdProfile::Streaming | JsonLdProfile::Expanded)
        .for_slice(document)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 2);
    Ok(())
}

#[test]
fn processing_mode_1_0_rejects_1_1_only_features() -> TestResult {
    let document = json(r#"{"@context": {"@version": 1.1}, "http://example.com/p": "o"}"#)?;
    let options = JsonLdOptions {
        processing_mode: JsonLdProcessingMode::JsonLd1_0,
        ..JsonLdOptions::new()
    };
    let error = expand(document, &options)
        .err()
        .ok_or("expected an error")?;
    assert_eq!(
        error.code(),
        Some(JsonLdErrorCode::ProcessingModeConflict),
        "{error}"
    );
    Ok(())
}

#[test]
fn expansion_produces_the_expanded_document_form() -> TestResult {
    let document = json(
        r#"{"@context": {"name": "http://schema.org/name"},
            "@id": "http://example.com/s", "name": "Foo"}"#,
    )?;
    let expanded = expand(document, &JsonLdOptions::new())?;
    assert_eq!(
        expanded.to_json_string(),
        r#"[{"@id":"http://example.com/s","http://schema.org/name":[{"@value":"Foo"}]}]"#
    );
    Ok(())
}

#[test]
fn compaction_reapplies_a_context() -> TestResult {
    let expanded =
        json(r#"[{"@id":"http://example.com/s","http://schema.org/name":[{"@value":"Foo"}]}]"#)?;
    let context = json(r#"{"@context": {"name": "http://schema.org/name"}}"#)?;
    let compacted = compact(expanded, context, &JsonLdOptions::new())?;
    let object = compacted.as_object().ok_or("expected a map")?;
    assert_eq!(
        object.get("name").and_then(JsonValue::as_str),
        Some("Foo"),
        "{}",
        compacted.to_json_string()
    );
    Ok(())
}

#[test]
fn flattening_collects_nested_nodes() -> TestResult {
    let document = json(
        r#"{"@id": "http://example.com/s",
            "http://example.com/p": {"@id": "http://example.com/o",
                                     "http://example.com/q": "v"}}"#,
    )?;
    let flattened = flatten(document, None, &JsonLdOptions::new())?;
    let items = flattened.as_array().ok_or("expected an array")?;
    assert_eq!(items.len(), 2);
    Ok(())
}

#[test]
fn round_trip_through_rdf_preserves_the_statements() -> TestResult {
    let quads = quads_of(
        r#"{"@id": "http://example.com/s",
            "http://example.com/p": [{"@value": "a"}, {"@value": "b"}]}"#,
    )?;
    assert_eq!(quads.len(), 2);
    let document = from_rdf(&quads, &JsonLdOptions::new())?;
    let again = to_rdf(document, &JsonLdOptions::new())?;
    assert_eq!(again.len(), 2);
    Ok(())
}

#[test]
fn lists_round_trip_through_rdf() -> TestResult {
    let quads = quads_of(
        r#"{"@id": "http://example.com/s",
            "http://example.com/p": {"@list": ["a", "b", "c"]}}"#,
    )?;
    // Three cells: rdf:first/rdf:rest each, plus the head statement.
    assert_eq!(quads.len(), 7);
    let document = from_rdf(&quads, &JsonLdOptions::new())?;
    let text = document.to_json_string();
    assert!(text.contains(r#""@list""#), "{text}");
    assert!(text.contains(r#"{"@value":"c"}"#), "{text}");
    Ok(())
}

#[test]
fn json_literals_keep_their_internal_representation() -> TestResult {
    let quads = quads_of(
        r#"{"@context": {"@version": 1.1, "e": {"@id": "http://example.com/e", "@type": "@json"}},
            "@id": "http://example.com/s", "e": {"b": 1, "a": [true, null]}}"#,
    )?;
    let quad = quads.first().ok_or("expected one quad")?;
    let Term::Literal(literal) = &quad.object else {
        return Err("expected a literal".into());
    };
    assert_eq!(
        literal.datatype().as_str(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#JSON"
    );
    // Canonical JSON: object keys sorted, numbers in ECMAScript form.
    assert_eq!(literal.value(), r#"{"a":[true,null],"b":1}"#);
    Ok(())
}

#[test]
fn direction_can_be_folded_into_an_i18n_datatype() -> TestResult {
    let document = json(
        r#"{"@context": {"@version": 1.1}, "@id": "http://example.com/s",
            "http://example.com/p": {"@value": "x", "@language": "en", "@direction": "rtl"}}"#,
    )?;
    let options = JsonLdOptions {
        rdf_direction: RdfDirection::I18nDatatype,
        ..JsonLdOptions::new()
    };
    let quads = to_rdf(document, &options)?;
    let quad = quads.first().ok_or("expected one quad")?;
    assert_eq!(
        quad.object.to_string(),
        r#""x"^^<https://www.w3.org/ns/i18n#en_rtl>"#
    );
    Ok(())
}

#[test]
fn native_types_are_optional_in_from_rdf() -> TestResult {
    let quads = quads_of(r#"{"@id": "http://example.com/s", "http://example.com/p": 7}"#)?;
    let plain = from_rdf(&quads, &JsonLdOptions::new())?;
    assert!(plain.to_json_string().contains(r#""@value":"7""#));
    let options = JsonLdOptions {
        use_native_types: true,
        ..JsonLdOptions::new()
    };
    let native = from_rdf(&quads, &options)?;
    assert!(native.to_json_string().contains(r#""@value":7"#));
    Ok(())
}

#[test]
fn the_default_loader_refuses_remote_contexts() -> TestResult {
    let document = json(r#"{"@context": "http://example.com/context.jsonld", "p": "o"}"#)?;
    let error = expand(document, &JsonLdOptions::new())
        .err()
        .ok_or("expected a loading error")?;
    assert_eq!(
        error.code(),
        Some(JsonLdErrorCode::LoadingRemoteContextFailed)
    );
    Ok(())
}

#[test]
fn a_closure_loader_resolves_remote_contexts() -> TestResult {
    let loader = ClosureLoader::new(|url: &str, _options: &LoadDocumentOptions| {
        assert_eq!(url, "http://example.com/context.jsonld");
        Ok(RemoteDocument::new(
            br#"{"@context": {"p": "http://example.com/p"}}"#.to_vec(),
            url,
        ))
    });
    let document = json(r#"{"@context": "http://example.com/context.jsonld", "p": "o"}"#)?;
    let options = JsonLdOptions::new().with_loader(&loader);
    let expanded = expand(document, &options)?;
    assert_eq!(
        expanded.to_json_string(),
        r#"[{"http://example.com/p":[{"@value":"o"}]}]"#
    );
    Ok(())
}

#[test]
fn the_parser_accepts_a_load_document_callback() -> TestResult {
    let document = br#"{"@context": "http://example.com/context.jsonld",
                        "@id": "http://example.com/s", "p": "o"}"#;
    let quads = JsonLdParser::new()
        .for_slice(document)
        .with_load_document_callback(|url: &str, _options: &LoadDocumentOptions| {
            Ok(RemoteDocument::new(
                br#"{"@context": {"p": "http://example.com/p"}}"#.to_vec(),
                url,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(quads.len(), 1);
    Ok(())
}

/// The parsers must be `Send`, exactly as `oxjsonld`'s are: a consumer that reads a document on a
/// worker thread moves the parser across the thread boundary.
#[test]
fn the_parsers_are_send() -> TestResult {
    fn assert_send<T: Send>() {}

    assert_send::<JsonLdParser>();
    assert_send::<oxixml_jsonld::SliceJsonLdParser<'_>>();
    assert_send::<oxixml_jsonld::ReaderJsonLdParser<&[u8]>>();
    assert_send::<oxixml_jsonld::ReaderJsonLdParser<std::fs::File>>();
    #[cfg(feature = "async-tokio")]
    assert_send::<oxixml_jsonld::TokioAsyncReaderJsonLdParser<&[u8]>>();

    // A parser carrying a custom loader must stay `Send` too — the loader is the part that used
    // to poison it.
    let document = br#"{"@context": "http://example.com/context.jsonld",
                        "@id": "http://example.com/s", "p": "o"}"#;
    let parser = JsonLdParser::new()
        .for_reader(document.as_slice())
        .with_load_document_callback(|url: &str, _options: &LoadDocumentOptions| {
            Ok(RemoteDocument::new(
                br#"{"@context": {"p": "http://example.com/p"}}"#.to_vec(),
                url,
            ))
        });
    let quads = std::thread::spawn(move || parser.collect::<Result<Vec<_>, _>>())
        .join()
        .expect("worker thread panicked")?;
    assert_eq!(quads.len(), 1);

    // The same for `with_loader`.
    let loader = ClosureLoader::new(|url: &str, _options: &LoadDocumentOptions| {
        Ok(RemoteDocument::new(
            br#"{"@context": {"p": "http://example.com/p"}}"#.to_vec(),
            url,
        ))
    });
    let parser = JsonLdParser::new().for_slice(document).with_loader(loader);
    assert_send::<oxixml_jsonld::SliceJsonLdParser<'_>>();
    assert_eq!(parser.collect::<Result<Vec<_>, _>>()?.len(), 1);
    Ok(())
}

/// The serializers are `Send` as well, so a dataset can be written from a worker thread.
#[test]
fn the_serializers_are_send() {
    fn assert_send<T: Send>() {}

    assert_send::<JsonLdSerializer>();
    assert_send::<oxixml_jsonld::WriterJsonLdSerializer<Vec<u8>>>();
    #[cfg(feature = "async-tokio")]
    assert_send::<oxixml_jsonld::TokioAsyncWriterJsonLdSerializer<Vec<u8>>>();
}

/// Serializes `quads` with the options `configure` applies.
fn serialize_with(
    quads: &[Quad],
    configure: impl FnOnce(JsonLdSerializer) -> Result<JsonLdSerializer, Box<dyn std::error::Error>>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut serializer = configure(JsonLdSerializer::new())?.for_writer(Vec::new());
    for quad in quads {
        serializer.serialize_quad(quad)?;
    }
    Ok(String::from_utf8(serializer.finish()?)?)
}

/// Parses `document` back with the crate's own parser and returns the quads, sorted so that two
/// datasets can be compared regardless of the order they came out in.
fn reparse_sorted(document: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut quads = JsonLdParser::new()
        .for_slice(document)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    quads.sort();
    Ok(quads)
}

fn sorted(quads: &[Quad]) -> Vec<String> {
    let mut sorted = quads.iter().map(ToString::to_string).collect::<Vec<_>>();
    sorted.sort();
    sorted
}

#[test]
fn the_serializer_writes_an_expanded_document() -> TestResult {
    let quads = quads_of(r#"{"@id": "http://example.com/s", "http://example.com/p": "o"}"#)?;
    // With no base and no prefix there is no context to write, so the document stays the bare
    // expanded array the *fromRdf* algorithm produces.
    let text = serialize_with(&quads, Ok)?;
    assert_eq!(
        text,
        r#"[{"@id":"http://example.com/s","http://example.com/p":[{"@value":"o"}]}]"#
    );
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn a_declared_prefix_becomes_a_context_entry() -> TestResult {
    let quads = quads_of(r#"{"@id": "http://example.com/s", "http://schema.org/name": "o"}"#)?;
    let text = serialize_with(&quads, |s| {
        Ok(s.with_prefix("schema", "http://schema.org/")?)
    })?;
    // Same envelope `oxjsonld` writes: the prefix is declared, property keys stay expanded.
    assert_eq!(
        text,
        r#"{"@context":{"schema":"http://schema.org/"},"@graph":[{"@id":"http://example.com/s","http://schema.org/name":[{"@value":"o"}]}]}"#
    );
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn the_empty_prefix_name_is_written_as_vocab_and_names_are_sorted() -> TestResult {
    let quads = quads_of(r#"{"@id": "http://example.com/s", "http://schema.org/name": "o"}"#)?;
    let text = serialize_with(&quads, |s| {
        Ok(s.with_prefix("zz", "http://example.com/z#")?
            .with_prefix("", "http://schema.org/")?
            .with_prefix("aa", "http://example.com/a#")?)
    })?;
    assert!(
        text.starts_with(
            r#"{"@context":{"@vocab":"http://schema.org/","aa":"http://example.com/a#","zz":"http://example.com/z#"},"@graph":["#
        ),
        "{text}"
    );
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn a_prefix_declared_twice_keeps_its_last_iri() -> TestResult {
    let quads = quads_of(r#"{"@id": "http://example.com/s", "http://schema.org/name": "o"}"#)?;
    let text = serialize_with(&quads, |s| {
        Ok(s.with_prefix("ex", "http://example.com/first#")?
            .with_prefix("ex", "http://example.com/second#")?)
    })?;
    assert!(
        text.starts_with(r#"{"@context":{"ex":"http://example.com/second#"},"#),
        "{text}"
    );
    Ok(())
}

#[test]
fn a_base_iri_is_declared_and_identifiers_are_made_relative() -> TestResult {
    let quads = quads_of(
        r#"{"@id": "http://example.com/s", "http://example.com/p": {"@id": "http://example.com/o"}}"#,
    )?;
    let text = serialize_with(&quads, |s| Ok(s.with_base_iri("http://example.com/")?))?;
    assert_eq!(
        text,
        r#"{"@context":{"@base":"http://example.com/"},"@graph":[{"@id":"s","http://example.com/p":[{"@id":"o"}]}]}"#
    );
    // The whole point of relativizing: it has to expand back to exactly the same dataset.
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn a_base_iri_shortens_a_foreign_authority_to_a_network_path() -> TestResult {
    let quads = quads_of(
        r#"{"@id": "http://elsewhere.example/s", "http://example.com/p": {"@id": "http://example.com/o"}}"#,
    )?;
    let text = serialize_with(&quads, |s| Ok(s.with_base_iri("http://example.com/")?))?;
    // A different authority under the same scheme relativizes to a network-path reference, which
    // resolves back to the original IRI. `oxiri` — and so `oxjsonld` — does exactly the same.
    assert!(text.contains(r#""@id":"//elsewhere.example/s""#), "{text}");
    assert!(text.contains(r#""@id":"o""#), "{text}");
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn a_base_iri_leaves_a_foreign_scheme_absolute() -> TestResult {
    let quads = quads_of(
        r#"{"@id": "urn:uuid:9f1c0f2e-0000-4000-8000-000000000000", "http://example.com/p": "o"}"#,
    )?;
    let text = serialize_with(&quads, |s| Ok(s.with_base_iri("http://example.com/")?))?;
    assert!(
        text.contains(r#""@id":"urn:uuid:9f1c0f2e-0000-4000-8000-000000000000""#),
        "{text}"
    );
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn a_base_iri_does_not_touch_types_or_blank_nodes() -> TestResult {
    let quads = quads_of(
        r#"{"@id": "http://example.com/s", "@type": "http://example.com/Type",
            "http://example.com/p": [{"@value": "1", "@type": "http://example.com/Datatype"},
                                     {"@id": "_:b0"}]}"#,
    )?;
    let text = serialize_with(&quads, |s| Ok(s.with_base_iri("http://example.com/")?))?;
    // A type is expanded against `@vocab`, not `@base`, so it has to stay absolute.
    assert!(
        text.contains(r#""@type":["http://example.com/Type"]"#),
        "{text}"
    );
    assert!(
        text.contains(r#""@type":"http://example.com/Datatype""#),
        "{text}"
    );
    // A blank node label is never an IRI.
    assert!(text.contains(r#""@id":"_:b0""#), "{text}");
    assert!(text.contains(r#""@id":"s""#), "{text}");
    assert_eq!(reparse_sorted(&text)?.len(), quads.len());
    Ok(())
}

#[test]
fn a_base_and_a_prefix_round_trip_together_across_named_graphs() -> TestResult {
    let quads = quads_of(
        r#"[{"@id": "http://example.com/g",
             "@graph": [{"@id": "http://example.com/s", "http://schema.org/name": "o"}]},
            {"@id": "http://example.com/s2", "http://schema.org/name": {"@id": "http://example.com/o2"}}]"#,
    )?;
    let text = serialize_with(&quads, |s| {
        Ok(s.with_base_iri("http://example.com/")?
            .with_prefix("schema", "http://schema.org/")?)
    })?;
    assert!(
        text.starts_with(
            r#"{"@context":{"@base":"http://example.com/","schema":"http://schema.org/"},"@graph":["#
        ),
        "{text}"
    );
    assert_eq!(reparse_sorted(&text)?, sorted(&quads));
    Ok(())
}

#[test]
fn profiles_form_a_set() {
    let mut profiles = JsonLdProfileSet::empty();
    assert!(profiles.is_empty());
    profiles |= JsonLdProfile::Expanded;
    profiles |= JsonLdProfile::Streaming;
    assert!(profiles.contains(JsonLdProfile::Streaming));
    assert!(!profiles.contains(JsonLdProfile::Framed));
    assert_eq!(
        profiles.into_iter().collect::<Vec<_>>(),
        vec![JsonLdProfile::Expanded, JsonLdProfile::Streaming]
    );
    assert_eq!(
        JsonLdProfile::from_iri(JsonLdProfile::Streaming.iri()),
        Some(JsonLdProfile::Streaming)
    );
}

#[test]
fn processing_modes_round_trip_through_their_identifier() {
    for mode in [
        JsonLdProcessingMode::JsonLd1_0,
        JsonLdProcessingMode::JsonLd1_1,
    ] {
        assert_eq!(JsonLdProcessingMode::from_id(mode.as_str()), Some(mode));
    }
    assert_eq!(JsonLdProcessingMode::from_id("json-ld-2.0"), None);
}

#[test]
fn error_codes_render_the_specification_phrase() {
    assert_eq!(
        JsonLdErrorCode::InvalidIdValue.to_string(),
        "invalid @id value"
    );
    assert_eq!(
        JsonLdErrorCode::InvalidStreamingKeyOrder.to_string(),
        "invalid streaming key order"
    );
}

#[test]
fn json_objects_preserve_insertion_order() {
    let mut object = JsonObject::new();
    object.insert("b", JsonValue::Number("1".to_owned()));
    object.insert("a", JsonValue::Bool(true));
    assert_eq!(object.keys().collect::<Vec<_>>(), vec!["b", "a"]);
    assert_eq!(object.sorted_keys(), vec!["a", "b"]);
    assert!(object.remove("b").is_some());
    assert_eq!(object.len(), 1);
}

#[test]
fn generalized_rdf_drops_statements_the_model_cannot_hold() -> TestResult {
    let document =
        json(r#"{"@id": "http://example.com/s", "_:p": "o", "http://example.com/q": "v"}"#)?;
    let options = JsonLdOptions {
        produce_generalized_rdf: true,
        ..JsonLdOptions::new()
    };
    let quads = to_rdf(document, &options)?;
    // Only the well-formed statement survives: `Quad` has no blank-node predicate.
    assert_eq!(quads.len(), 1);
    Ok(())
}

/// Builds `{"@context":{"p":"http://example.com/p"},"p":{"p":{...:"leaf"}}}`, `depth` levels deep.
fn deeply_nested_document(depth: usize) -> String {
    let mut document = String::from(r#"{"@context":{"p":"http://example.com/p"},"p":"#);
    for _ in 1..depth {
        document.push_str(r#"{"p":"#);
    }
    document.push_str(r#""leaf""#);
    for _ in 0..depth {
        document.push('}');
    }
    document
}

/// Regression test: a deeply nested but otherwise valid JSON-LD document used to abort the
/// process with an uncatchable stack overflow (the algorithms recurse once per nesting level).
/// It must now be a plain, catchable error on every public entry point.
#[test]
fn deeply_nested_documents_are_rejected_rather_than_overflowing_the_stack() {
    for depth in [1_000usize, 2_000, 20_000] {
        let document = deeply_nested_document(depth);

        // The streaming parser: the entry point `oxixml-io` drives for `RdfFormat::JsonLd`.
        assert!(
            JsonLdParser::new()
                .for_slice(document.as_bytes())
                .collect::<Result<Vec<Quad>, _>>()
                .is_err(),
            "depth {depth}"
        );

        // And the JSON-LD API operations, which refuse the document before ever parsing it.
        assert!(
            JsonValue::from_slice(document.as_bytes()).is_err(),
            "depth {depth}"
        );
    }
}

/// The expansion algorithm carries its own depth guard, independent of the one applied while
/// building the value tree: a tree handed to it directly is bounded too.
#[test]
fn expansion_has_its_own_depth_guard() -> TestResult {
    let document = deeply_nested_document(64);
    // Parsed with a bound that lets the document through...
    let value = JsonValue::from_slice(document.as_bytes())?;
    let options = JsonLdOptions::default().with_max_nesting_depth(16);
    // ... but expansion still refuses to recurse that deep.
    let error = expand(value.clone(), &options).expect_err("expansion must fail");
    assert_eq!(error.code(), Some(JsonLdErrorCode::DocumentNestingTooDeep));
    assert!(to_rdf(value.clone(), &options).is_err());
    assert!(flatten(value.clone(), None, &options).is_err());
    assert!(compact(value, JsonValue::Object(JsonObject::new()), &options).is_err());
    Ok(())
}

/// A document at the configured limit still expands normally.
#[test]
fn documents_within_the_nesting_limit_still_expand() -> TestResult {
    let document = deeply_nested_document(32);
    let value = json(&document)?;
    let quads = to_rdf(value, &JsonLdOptions::default())?;
    assert_eq!(quads.len(), 32);
    Ok(())
}

/// The nesting limit is configurable on both the API options and the streaming parser.
#[test]
fn the_nesting_limit_is_configurable() -> TestResult {
    let document = deeply_nested_document(8);
    assert!(
        expand(
            json(&document)?,
            &JsonLdOptions::default().with_max_nesting_depth(3)
        )
        .is_err()
    );
    assert!(
        JsonLdParser::new()
            .with_max_nesting_depth(3)
            .for_slice(document.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .is_err()
    );
    Ok(())
}
