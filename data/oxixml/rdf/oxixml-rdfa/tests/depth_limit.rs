//! Regression tests for the document-nesting depth cap.
//!
//! `Processor::process_element` recurses once per nested element, and the `rdf:XMLLiteral`/
//! `rdf:HTML` literal serializers in `literal.rs` recurse once per nested descendant of the
//! element carrying the literal; neither used to be bounded, so a deeply nested document (or a
//! single deeply nested `rdf:XMLLiteral`/`rdf:HTML` literal) could abort the process with a stack
//! overflow rather than fail cleanly. A stack overflow aborts the process, so it cannot be
//! caught with `#[should_panic]`: every case here instead asserts the bounded `Err(..)` these
//! inputs must now produce.
//!
//! The hostile-input cases build the document tree directly with [`DocumentBuilder`] rather than
//! parsing markup, so the result depends only on this crate's own recursion, not on whatever
//! depth some other crate's XML/HTML parser happens to tolerate. They also run on an explicit,
//! generously sized thread stack: the default stack `cargo test` gives each test thread is
//! smaller than a typical process's main-thread stack, and this cap is sized for the latter (a
//! panic inside the spawned thread still propagates through `join`, but a stack overflow is a
//! process-wide abort no `join` call can turn into a normal `Result` -- an abort here would still
//! fail the test binary, just not through this file's own assertions).

use oxixml_dom::Document;
use oxixml_dom::build::DocumentBuilder;
use oxixml_rdfa::{HostLanguage, RdfaParser};

/// A stack size comparable to a typical process main thread, well above what `cargo test`
/// gives a spawned test thread by default.
const BIG_STACK: usize = 8 * 1024 * 1024;

/// Runs `body` on a thread with [`BIG_STACK`] bytes of stack, propagating a panic from inside it
/// as a panic of the calling test.
fn on_a_big_stack<F: FnOnce() + Send + 'static>(body: F) {
    std::thread::Builder::new()
        .stack_size(BIG_STACK)
        .spawn(body)
        .expect("spawning the probe thread")
        .join()
        .expect("the probe thread must not panic");
}

/// Builds `<div>` `depth` levels deep, holding the text `"leaf"`, with no RDFa attributes at all.
fn deeply_nested_plain_divs(depth: usize) -> Document {
    let mut builder = DocumentBuilder::new();
    for _ in 0..depth {
        builder
            .push_element_start(None, "div")
            .expect("push_element_start");
    }
    builder.push_text("leaf");
    for _ in 0..depth {
        builder.push_element_end().expect("push_element_end");
    }
    builder.finish().expect("finish")
}

/// Builds a document only two elements deep at the outer level -- `<div><p about=... property=...
/// datatype="rdf:{XMLLiteral,HTML}">` -- whose `<p>` holds `depth` levels of nested `<span>`, so
/// that `process_element`'s own recursion never approaches any cap and the only risk is inside
/// the literal serializer walking the `<p>`'s content.
fn shallow_document_with_deep_literal_content(datatype_local_name: &str, depth: usize) -> Document {
    let mut builder = DocumentBuilder::new();
    builder
        .push_element_start(None, "div")
        .expect("push div start");
    builder
        .push_attribute(
            None,
            "prefix",
            "rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        )
        .expect("push prefix");
    builder.push_element_start(None, "p").expect("push p start");
    builder
        .push_attribute(None, "about", "http://example.com/a")
        .expect("push about");
    builder
        .push_attribute(None, "property", "http://example.com/p")
        .expect("push property");
    builder
        .push_attribute(None, "datatype", &format!("rdf:{datatype_local_name}"))
        .expect("push datatype");
    for _ in 0..depth {
        builder
            .push_element_start(None, "span")
            .expect("push span start");
    }
    builder.push_text("leaf");
    for _ in 0..depth {
        builder.push_element_end().expect("push span end");
    }
    builder.push_element_end().expect("push p end");
    builder.push_element_end().expect("push div end");
    builder.finish().expect("finish")
}

#[test]
fn deeply_nested_document_is_rejected_instead_of_crashing() {
    on_a_big_stack(|| {
        // Comfortably past the default cap (256); the point is only that this used to abort the
        // process and now returns a normal, bounded error instead.
        let document = deeply_nested_plain_divs(5_000);
        let result: Result<Vec<_>, _> = RdfaParser::new()
            .with_host_language(HostLanguage::Xml)
            .for_document(&document)
            .collect();
        let error = result.expect_err("document nesting must be rejected, not silently walked");
        assert!(error.to_string().contains("nesting"), "{error}");
    });
}

#[test]
fn shallow_document_with_a_deeply_nested_xml_literal_is_rejected_instead_of_crashing() {
    on_a_big_stack(|| {
        // The outer document is only two elements deep, so `process_element`'s own recursion
        // never comes close to any cap; the danger is entirely inside `literal::xml_literal`,
        // which walked the `<p>`'s descendants with its own, independently unbounded recursion
        // prior to the fix.
        let document = shallow_document_with_deep_literal_content("XMLLiteral", 400);
        let result: Result<Vec<_>, _> = RdfaParser::new()
            .with_host_language(HostLanguage::Xml)
            .for_document(&document)
            .collect();
        let error =
            result.expect_err("an oversized rdf:XMLLiteral must be rejected, not walked raw");
        assert!(error.to_string().contains("nesting"), "{error}");
    });
}

#[test]
fn shallow_document_with_a_deeply_nested_html_literal_is_rejected_instead_of_crashing() {
    on_a_big_stack(|| {
        // Same as above, but through the `rdf:HTML` literal path (`literal::html_literal`), a
        // separate function with its own recursion; `rdf:HTML` only applies to the HTML family.
        let document = shallow_document_with_deep_literal_content("HTML", 400);
        let result: Result<Vec<_>, _> = RdfaParser::new()
            .with_host_language(HostLanguage::Html5)
            .for_document(&document)
            .collect();
        let error =
            result.expect_err("an oversized rdf:HTML literal must be rejected, not walked raw");
        assert!(error.to_string().contains("nesting"), "{error}");
    });
}

/// Wraps a body fragment into a minimal HTML5 document.
fn html(body: &str) -> String {
    format!("<!DOCTYPE html>\n<html>\n<head><title>t</title></head>\n<body>{body}</body>\n</html>")
}

/// `<div about=... property=...><div>...<div></div>...</div></div>`, `depth` levels of plain
/// `<div>` deep. Small and shallow by construction (used only for the configurability checks
/// below), so -- unlike the hostile-input cases above -- there is no need to build it directly
/// or to run it on a dedicated stack.
fn nested_divs(depth: usize) -> String {
    let mut out =
        String::from(r#"<div about="http://example.com/a" property="http://example.com/p">"#);
    for _ in 0..depth {
        out.push_str("<div>");
    }
    out.push_str("leaf");
    for _ in 0..depth {
        out.push_str("</div>");
    }
    out.push_str("</div>");
    out
}

#[test]
fn max_depth_is_configurable_and_names_the_limit() {
    // The same document, ten `<div>`s deep (plus whatever the `<html>/<head>/<body>` skeleton
    // adds): a limit of 2 is clearly too small for it, and a limit of 1000 is clearly enough,
    // regardless of the skeleton's exact contribution to the count.
    let markup = html(&nested_divs(10));

    let error = RdfaParser::new()
        .with_host_language(HostLanguage::Html5)
        .with_max_depth(2)
        .for_slice(&markup)
        .collect::<Result<Vec<_>, _>>()
        .expect_err("the document must not fit under a limit of 2");
    let message = error.to_string();
    assert!(message.contains('2'), "{message}");
    assert!(message.contains("configurable"), "{message}");

    let triples = RdfaParser::new()
        .with_host_language(HostLanguage::Html5)
        .with_max_depth(1_000)
        .for_slice(&markup)
        .collect::<Result<Vec<_>, _>>()
        .expect("the same document fits comfortably under a limit of 1000");
    assert_eq!(triples.len(), 1);
}

#[test]
fn max_depth_defaults_to_256() {
    assert_eq!(RdfaParser::new().max_depth(), 256);
    assert_eq!(RdfaParser::DEFAULT_MAX_DEPTH, 256);
}
