//! End-to-end tests of the public API: the DOM that comes out, the encoding
//! sniffing that goes in, serialization, fragments, quirks mode and errors.

use oxixml_encoding::Encoding;
use oxixml_html::{
    EncodingConfidence, HTML_NAMESPACE, MATHML_NAMESPACE, Namespace, ParseOptions, QuirksMode,
    SVG_NAMESPACE, SerializeOptions, dump_tree, parse_bytes, parse_document, parse_document_with,
    parse_fragment, parse_fragment_with, serialize, serialize_node, sniff_encoding,
};

fn element(parsed: &oxixml_html::ParseOutput, namespace: &str, name: &str) -> oxixml_dom::NodeId {
    let document = parsed.document();
    document
        .get_elements_by_tag_name_ns(document.root(), Some(namespace), name)
        .next()
        .unwrap_or_else(|| panic!("no <{name}> element in {}", dump_tree(parsed)))
}

#[test]
fn the_result_is_an_ordinary_dom_document() {
    let parsed =
        parse_document("<!DOCTYPE html><html lang=en><title>T</title><body><p id=x>Hello<b>!</b>");
    let document = parsed.document();

    let html = element(&parsed, HTML_NAMESPACE, "html");
    assert_eq!(document.get_attribute(html, "lang"), Some("en"));
    assert_eq!(
        document
            .doctype()
            .and_then(|node| document.node_name(node))
            .as_deref(),
        Some("html")
    );

    let paragraph = element(&parsed, HTML_NAMESPACE, "p");
    assert_eq!(document.get_attribute(paragraph, "id"), Some("x"));
    assert_eq!(document.text_content(paragraph).as_deref(), Some("Hello!"));

    // Elements really are in the XHTML namespace, so namespace-aware consumers
    // such as XPath or canonicalization see what they expect.
    assert_eq!(
        document
            .element(paragraph)
            .and_then(|data| data.namespace_uri()),
        Some(HTML_NAMESPACE)
    );
}

#[test]
fn implied_elements_are_created() {
    let parsed = parse_document("<p>only a paragraph");
    assert_eq!(
        dump_tree(&parsed),
        "\
| <html>
|   <head>
|   <body>
|     <p>
|       \"only a paragraph\"
"
    );
}

#[test]
fn foreign_content_lands_in_the_right_namespaces() {
    let parsed = parse_document("<!DOCTYPE html><svg viewbox='0 0 1 1'><circle/></svg><math><mi>x");
    let document = parsed.document();

    let svg = element(&parsed, SVG_NAMESPACE, "svg");
    // The attribute name's camel case is restored by "adjust SVG attributes".
    assert_eq!(document.get_attribute(svg, "viewBox"), Some("0 0 1 1"));
    let _ = element(&parsed, SVG_NAMESPACE, "circle");
    let _ = element(&parsed, MATHML_NAMESPACE, "mi");
}

#[test]
fn xlink_attributes_get_their_namespace() {
    let parsed = parse_document("<!DOCTYPE html><svg><a xlink:href='u' xml:lang='en'>");
    let document = parsed.document();
    let anchor = element(&parsed, SVG_NAMESPACE, "a");
    assert_eq!(
        document.get_attribute_ns(anchor, Some("http://www.w3.org/1999/xlink"), "href"),
        Some("u")
    );
    assert_eq!(
        document.get_attribute_ns(anchor, Some(oxixml_html::XML_NAMESPACE), "lang"),
        Some("en")
    );
}

#[test]
fn template_contents_live_in_a_fragment() {
    let parsed = parse_document("<!DOCTYPE html><body><template><td>cell</td></template>");
    let document = parsed.document();
    let template = element(&parsed, HTML_NAMESPACE, "template");
    assert_eq!(document.children(template).count(), 0);

    let content = parsed
        .template_contents(template)
        .expect("the template has contents");
    // A `<td>` inside a template is kept, because the template's insertion mode
    // is chosen from its contents.
    assert_eq!(document.text_content(content).as_deref(), Some("cell"));
    assert_eq!(parsed.all_template_contents().len(), 1);
}

#[test]
fn foster_parenting_moves_content_out_of_a_table() {
    let parsed = parse_document("<!DOCTYPE html><table><b>stray</b><tr><td>ok");
    assert_eq!(
        dump_tree(&parsed),
        "\
| <!DOCTYPE html>
| <html>
|   <head>
|   <body>
|     <b>
|       \"stray\"
|     <table>
|       <tbody>
|         <tr>
|           <td>
|             \"ok\"
"
    );
}

#[test]
fn quirks_mode_is_reported() {
    assert_eq!(
        parse_document("<!DOCTYPE html><p>x").quirks_mode(),
        QuirksMode::NoQuirks
    );
    assert_eq!(parse_document("<p>x").quirks_mode(), QuirksMode::Quirks);
    assert_eq!(
        parse_document(
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"x\"><p>x"
        )
        .quirks_mode(),
        QuirksMode::LimitedQuirks
    );
    assert!(QuirksMode::Quirks.is_quirks());
}

#[test]
fn parse_errors_are_collected_but_never_fatal() {
    let parsed = parse_document("<p><b>unclosed");
    assert!(!parsed.errors().is_empty());
    assert!(
        parsed
            .errors()
            .iter()
            .any(|error| error.code() == "expected-doctype")
    );
    // A well-formed document has none.
    assert!(
        parse_document("<!DOCTYPE html><title>t</title><p>x</p>")
            .errors()
            .is_empty()
    );
}

#[test]
fn the_scripting_flag_changes_how_noscript_is_parsed() {
    let with_scripts = parse_document_with(
        "<!DOCTYPE html><body><noscript><p>x</p></noscript>",
        &ParseOptions::new().with_scripting(true),
    );
    // With scripting on, the content of <noscript> is raw text.
    let noscript = element(&with_scripts, HTML_NAMESPACE, "noscript");
    assert_eq!(
        with_scripts.document().text_content(noscript).as_deref(),
        Some("<p>x</p>")
    );

    let without = parse_document("<!DOCTYPE html><body><noscript><p>x</p></noscript>");
    let _ = element(&without, HTML_NAMESPACE, "p");
    assert!(!ParseOptions::new().scripting());
}

#[test]
fn fragments_depend_on_their_context_element() {
    assert_eq!(
        dump_tree(&parse_fragment("<td>x", "tr")),
        "| <td>\n|   \"x\"\n"
    );
    assert_eq!(dump_tree(&parse_fragment("<td>x", "div")), "| \"x\"\n");
    assert!(parse_fragment("<p>x", "body").is_fragment());

    // A foreign context element starts the parser in foreign content.
    let parsed = parse_fragment_with("<circle/>", Namespace::Svg, "svg", &ParseOptions::new());
    assert_eq!(dump_tree(&parsed), "| <svg circle>\n");
}

#[test]
fn fragments_of_raw_text_elements_are_not_markup() {
    let parsed = parse_fragment("a<b>c", "title");
    assert_eq!(dump_tree(&parsed), "| \"a<b>c\"\n");
}

#[test]
fn serializing_round_trips() {
    let source = "<!DOCTYPE html><html><head><title>T</title></head><body><p>a &amp; b<br>c</p></body></html>";
    let parsed = parse_document(source);
    let html = serialize(&parsed);
    assert_eq!(html, source);

    // And parsing the output again gives the same tree.
    let again = parse_document(&html);
    assert_eq!(dump_tree(&parsed), dump_tree(&again));
}

#[test]
fn serializing_a_single_node() {
    let parsed = parse_document("<!DOCTYPE html><ul><li>one<li>two");
    let list = element(&parsed, HTML_NAMESPACE, "ul");
    assert_eq!(
        serialize_node(parsed.document(), list, &SerializeOptions::inner()),
        "<li>one</li><li>two</li>"
    );
    assert_eq!(
        serialize_node(parsed.document(), list, &SerializeOptions::outer()),
        "<ul><li>one</li><li>two</li></ul>"
    );
}

#[test]
fn raw_text_elements_are_serialized_without_escaping() {
    let parsed = parse_document("<!DOCTYPE html><body><script>if (a < b) { }</script>");
    assert!(serialize(&parsed).contains("<script>if (a < b) { }</script>"));
}

#[test]
fn unrepresentable_names_and_data_are_remembered() {
    // `a<b` is a legal HTML tag name and an illegal XML one.
    let parsed = parse_document("<a<b>");
    let document = parsed.document();
    let body = element(&parsed, HTML_NAMESPACE, "body");
    let child = document.children(body).next().expect("one child");
    assert_eq!(
        document
            .element(child)
            .map(oxixml_dom::ElementData::local_name),
        Some("a_b")
    );
    assert_eq!(parsed.original_html(child), Some("a<b"));
    assert!(
        parsed
            .errors()
            .iter()
            .any(|error| error.code() == "oxixml-unrepresentable-name")
    );
    // The dump and the serializer both use the original spelling.
    assert!(dump_tree(&parsed).contains("<a<b>"));
    assert!(serialize(&parsed).contains("<a<b>"));
}

#[test]
fn comments_with_double_hyphens_survive_the_round_trip() {
    let parsed = parse_document("<!--a--b-->");
    assert!(dump_tree(&parsed).starts_with("| <!-- a--b -->"));
    assert!(serialize(&parsed).starts_with("<!--a--b-->"));
    // The DOM itself holds an XML-legal rewriting.
    let document = parsed.document();
    let comment = document
        .children(document.root())
        .next()
        .expect("a comment");
    assert_eq!(document.character_data(comment), Some("a- -b"));
}

#[test]
fn processing_instructions_are_bogus_comments_by_default() {
    // HTML § 13.2.5.6 tag open state, U+003F: an
    // `unexpected-question-mark-instead-of-tag-name` parse error reconsumed in
    // the bogus comment state.
    let parsed = parse_document("<body><?target some data?>");
    let document = parsed.document();
    let body = element(&parsed, HTML_NAMESPACE, "body");
    let node = document.children(body).next().expect("a child");
    assert_eq!(document.character_data(node), Some("?target some data?"));
    assert!(
        parsed
            .errors()
            .iter()
            .any(|error| error.code() == "unexpected-question-mark-instead-of-tag-name")
    );
}

#[test]
fn processing_instructions_become_nodes_when_enabled() {
    let options = ParseOptions::new().with_processing_instructions(true);
    let parsed = parse_document_with("<body><?target some data?>", &options);
    let document = parsed.document();
    let body = element(&parsed, HTML_NAMESPACE, "body");
    let node = document.children(body).next().expect("a child");
    let instruction = document
        .processing_instruction(node)
        .expect("a processing instruction");
    assert_eq!(instruction.target(), "target");
    assert_eq!(instruction.data(), "some data");
}

// ------------------------------------------------------------------ encoding

#[test]
fn a_byte_order_mark_is_certain() {
    let sniffed = sniff_encoding(b"\xff\xfe<\0h\0", None);
    assert_eq!(sniffed.encoding, Encoding::Utf16Le);
    assert_eq!(sniffed.confidence, EncodingConfidence::Certain);
    assert_eq!(sniffed.bom_length, 2);
}

#[test]
fn meta_charset_is_honoured() {
    let bytes = b"<!DOCTYPE html><meta charset=\"iso-8859-1\"><p>caf\xe9";
    let parsed = parse_bytes(bytes, None);
    let paragraph = element(&parsed, HTML_NAMESPACE, "p");
    assert_eq!(
        parsed.document().text_content(paragraph).as_deref(),
        Some("caf\u{e9}")
    );
}

#[test]
fn the_transport_encoding_wins() {
    let bytes = b"<meta charset=\"utf-8\"><p>\x93x\x94";
    let parsed = parse_bytes(bytes, Some("windows-1252"));
    let paragraph = element(&parsed, HTML_NAMESPACE, "p");
    assert_eq!(
        parsed.document().text_content(paragraph).as_deref(),
        Some("\u{201c}x\u{201d}")
    );
}

#[test]
fn a_utf8_bom_is_stripped() {
    let parsed = parse_bytes("\u{feff}<p>x".as_bytes(), None);
    let paragraph = element(&parsed, HTML_NAMESPACE, "p");
    assert_eq!(
        parsed.document().text_content(paragraph).as_deref(),
        Some("x")
    );
}

#[test]
fn malformed_bytes_do_not_panic() {
    // Invalid UTF-8 decodes lossily rather than failing.
    let parsed = parse_bytes(b"<p>\xff\xfe\x00ok", Some("utf-8"));
    assert!(!dump_tree(&parsed).is_empty());
}

#[test]
fn enormous_nesting_does_not_overflow_the_stack() {
    let input = "<div>".repeat(2000);
    let parsed = parse_document(&input);
    // Dropping the document must not recurse either.
    drop(parsed);
}

#[test]
fn dump_tree_does_not_overflow_the_stack_on_deep_nesting() {
    // dump_node/dump_element used to recurse one native stack frame per DOM
    // level, the same class of bug write_tree's own doc comment describes
    // for the HTML serializer: a document this deep aborted the process
    // with a stack overflow before dump_tree adopted the same explicit
    // work-list write_tree already used.
    const DEPTH: usize = 8_000;
    let mut source = String::from("<!DOCTYPE html><html><body>");
    source.push_str(&"<div>".repeat(DEPTH));
    source.push('x');
    source.push_str(&"</div>".repeat(DEPTH));
    source.push_str("</body></html>");

    let parsed = parse_document(&source);
    let dumped = dump_tree(&parsed);
    assert_eq!(dumped.matches("<div>\n").count(), DEPTH);
    assert!(dumped.contains("\"x\"\n"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Regression: HTML § 13.3 fragment serialization, and depth robustness
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn template_contents_are_serialized() {
    // A `template` element's content lives in a separate fragment, not in its
    // child list; writing the (always empty) child list discarded every
    // template body.
    let parsed = parse_document("<template>hello<b>x</b></template>");
    assert_eq!(
        serialize(&parsed),
        "<html><head><template>hello<b>x</b></template></head><body></body></html>"
    );

    // And the result is a fixed point.
    let again = parse_document(&serialize(&parsed));
    assert_eq!(serialize(&again), serialize(&parsed));
}

#[test]
fn a_leading_newline_in_pre_survives_a_round_trip() {
    // HTML § 13.3: tree construction swallows one U+000A right after `<pre>`,
    // `<textarea>` and `<listing>`, so the serializer has to put it back or the
    // document loses one newline per round trip.
    for tag in ["pre", "textarea", "listing"] {
        let source = format!("<body><{tag}>\n\nfoo</{tag}>");
        let first = serialize(&parse_document(&source));
        let second = serialize(&parse_document(&first));
        assert!(
            first.contains(&format!("<{tag}>\n\nfoo</{tag}>")),
            "{tag}: {first}"
        );
        assert_eq!(second, first, "{tag} serialization is not a fixed point");
    }

    // An element whose text does not start with a newline is untouched.
    let once = serialize(&parse_document("<body><pre>foo</pre>"));
    assert!(once.contains("<pre>foo</pre>"), "{once}");
    assert_eq!(serialize(&parse_document(&once)), once);
}

#[test]
fn a_deeply_nested_document_parses_and_serializes() {
    // Both tree construction and serialization used to fall over here: the
    // former was quadratic in the nesting depth, the latter recursed once per
    // level and aborted the process with a stack overflow.
    const DEPTH: usize = 100_000;
    let mut source = String::from("<!DOCTYPE html><html><body>");
    source.push_str(&"<div>".repeat(DEPTH));
    source.push('x');
    source.push_str(&"</div>".repeat(DEPTH));
    source.push_str("</body></html>");

    let parsed = parse_document(&source);
    let serialized = serialize(&parsed);
    assert_eq!(serialized.matches("<div>").count(), DEPTH);
    assert_eq!(serialized.matches("</div>").count(), DEPTH);
    assert!(serialized.contains("x"));
}

#[test]
fn unmatched_end_tags_over_deep_non_special_nesting_do_not_blow_up() {
    // any_other_end_tag() used to scan the *entire* stack of open elements
    // for every end tag that does not match anything open, so a document
    // that nests deeply in non-"special" (e.g. formatting) elements and then
    // closes with a name that never matches cost time proportional to
    // depth * end-tag-count. ParseOptions::DEFAULT_MAX_DEPTH bounds the
    // stack -- and therefore that scan -- to a constant regardless of how
    // deep the source claims to nest.
    //
    // "a" is a formatting element, so `</a>` (never opened) reaches
    // adoption_agency() first, which itself falls back to the same scan
    // once it fails its own fast check -- exactly the reported shape. This
    // is a correctness check: `depth` only needs to be safely past the
    // ceiling to exercise the bounded path, not large enough to make a
    // regression's cost observable on the wall clock, which is not a
    // reliable signal on a shared, variably-loaded machine.
    let depth = 2_000;
    let end_tags = 200;
    let mut source = String::with_capacity(depth * 6 + end_tags * 4);
    source.push_str(&"<b><i>".repeat(depth));
    source.push_str(&"</a>".repeat(end_tags));

    let parsed = parse_document(&source);

    // This exercised real tree construction (rather than short-circuiting
    // on something else entirely) and actually reached the depth ceiling,
    // since `depth` is well past it.
    assert!(
        parsed
            .errors()
            .iter()
            .any(|e| e.code() == oxixml_html::error::codes::MAX_DEPTH_EXCEEDED)
    );
}

#[test]
fn max_depth_defaults_to_the_documented_ceiling_and_is_configurable() {
    assert_eq!(
        ParseOptions::new().max_depth(),
        ParseOptions::DEFAULT_MAX_DEPTH
    );
    assert_eq!(ParseOptions::new().with_max_depth(10).max_depth(), 10);
}

#[test]
fn ordinary_depth_never_reaches_the_default_max_depth_ceiling() {
    // A document nested one level short of the default ceiling must parse
    // exactly as it always did: the limit exists for pathological depth, not
    // for anything a real document could plausibly reach.
    let depth = ParseOptions::DEFAULT_MAX_DEPTH - 4; // headroom for html/body already on the stack
    let mut source = String::from("<!DOCTYPE html><html><body>");
    source.push_str(&"<div>".repeat(depth));
    source.push('x');
    source.push_str(&"</div>".repeat(depth));
    source.push_str("</body></html>");

    let parsed = parse_document(&source);
    assert!(
        !parsed
            .errors()
            .iter()
            .any(|e| e.code() == oxixml_html::error::codes::MAX_DEPTH_EXCEEDED),
        "a document safely under the default ceiling must not trip it"
    );
    let serialized = serialize(&parsed);
    assert_eq!(serialized.matches("<div>").count(), depth);
    assert_eq!(serialized.matches("</div>").count(), depth);
}

#[test]
fn exceeding_max_depth_is_reported_once_and_keeps_parsing() {
    let depth = 300;
    let mut source = String::from("<!DOCTYPE html><html><body>");
    source.push_str(&"<div>".repeat(depth));
    source.push('x');
    source.push_str(&"</div>".repeat(depth));
    source.push_str("</body></html>");

    let options = ParseOptions::new().with_max_depth(10);
    let parsed = parse_document_with(&source, &options);

    let exceeded_count = parsed
        .errors()
        .iter()
        .filter(|e| e.code() == oxixml_html::error::codes::MAX_DEPTH_EXCEEDED)
        .count();
    assert_eq!(
        exceeded_count, 1,
        "the ceiling must be reported once, not once per element past it"
    );

    // Parsing kept going rather than stopping cold: every div element is
    // still created and inserted (see insert_element's documentation for
    // why), so the content is not lost even though the tracked nesting
    // stopped growing.
    let document = parsed.document();
    let div_count = document
        .get_elements_by_tag_name_ns(document.root(), Some(HTML_NAMESPACE), "div")
        .count();
    assert_eq!(div_count, depth);
    assert!(dump_tree(&parsed).contains("\"x\""));
}

#[test]
fn differently_attributed_formatting_elements_do_not_blow_up_the_formatting_list() {
    // Noah's Ark only bounds repeats of the *same* name and attributes, so
    // opening enormously many *differently*-attributed formatting elements
    // defeats it without ever growing the stack of open elements past
    // ParseOptions::DEFAULT_MAX_DEPTH: an element no longer tracked as open
    // there was still pushed onto the list of active formatting elements
    // unconditionally, so that list -- which the Noah's Ark scan and
    // adoption_agency's lookup both walk -- could still grow without bound
    // and reproduce the same depth * end-tag-count blowup through a
    // different piece of state.
    //
    // This is a correctness check on the depth ceiling itself; the
    // performance property (that push_formatting_element's own Noah's Ark
    // scan stops running once the ceiling freezes the list, rather than
    // continuing to cost O(max_depth) per subsequent element) is pinned by
    // a deterministic comparison-count regression test next to
    // push_formatting_element itself, not by wall-clock time here, which is
    // not a reliable signal on a shared, variably-loaded machine.
    let depth = 2_000;
    let end_tags = 200;
    let mut source = String::with_capacity(depth * 20 + end_tags * 4);
    for i in 0..depth {
        source.push_str(&format!("<b class=\"{i}\">"));
    }
    source.push_str(&"</a>".repeat(end_tags));

    let parsed = parse_document(&source);

    assert!(
        parsed
            .errors()
            .iter()
            .any(|e| e.code() == oxixml_html::error::codes::MAX_DEPTH_EXCEEDED)
    );
}

#[test]
fn markers_do_not_need_the_same_cap_as_formatting_elements() {
    // Unlike formatting elements, a marker (pushed directly by `applet`,
    // `object`, `marquee`, `template`, `td`, `th` and `caption`, bypassing
    // push_formatting_element's cap entirely) never makes a lookup slow: the
    // Noah's Ark scan and formatting_entry_named both only ever look *since
    // the last marker*, so a marker can only shrink that window, never grow
    // it, regardless of how many of them accumulate -- markers therefore
    // never need push_formatting_element's own ceiling.
    //
    // `applet` is still an ordinary open element in its own right, though,
    // so it is `self.open`'s ceiling (unrelated to markers) this correctness
    // check actually exercises: `depth` only needs to be safely past it.
    let depth = 2_000;
    let end_tags = 200;
    let mut source = String::with_capacity(depth * 20 + end_tags * 4);
    for i in 0..depth {
        source.push_str(&format!("<applet name=\"{i}\">"));
    }
    source.push_str(&"</a>".repeat(end_tags));

    let parsed = parse_document(&source);

    assert!(
        parsed
            .errors()
            .iter()
            .any(|e| e.code() == oxixml_html::error::codes::MAX_DEPTH_EXCEEDED)
    );
}
