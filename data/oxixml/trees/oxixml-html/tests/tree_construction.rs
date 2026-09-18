//! Tree construction fixtures: a curated subset of the HTML standard's own
//! test suite, embedded so that it runs everywhere.
//!
//! Every case here is copied verbatim from the `.dat` files of the
//! [web-platform-tests](https://github.com/web-platform-tests/wpt/tree/master/html/syntax/parsing/resources)
//! tree construction suite — the input and the expected html5lib tree dump are
//! upstream's, not this crate's, which is what makes them worth having. The
//! full suite (about two thousand cases) runs from `tests/html5lib.rs` when
//! `OXIXML_HTML5LIB_TESTS_DIR` points at a checkout.
//!
//! The cases are grouped by the parser feature they exercise: implied tags,
//! the adoption agency algorithm, foster parenting, character references,
//! foreign content, `<template>` contents, `<select>`, script data escaping,
//! quirks-mode DOCTYPEs, processing instructions and fragment parsing.

use oxixml_html::{
    Namespace, ParseOptions, dump_tree, parse_document, parse_document_with, parse_fragment_with,
};

/// One fixture: an input, the expected dump, and the fragment context if any.
struct Case {
    input: &'static str,
    context: Option<(Namespace, &'static str)>,
    expected: &'static str,
}

fn check(group: &str, cases: &[Case]) {
    let mut failures = Vec::new();
    for case in cases {
        let actual = match case.context {
            None => dump_tree(&parse_document(case.input)),
            Some((namespace, name)) => dump_tree(&parse_fragment_with(
                case.input,
                namespace,
                name,
                &ParseOptions::new(),
            )),
        };
        if actual != case.expected {
            failures.push(format!(
                "input {:?}\n--- expected ---\n{}--- actual ---\n{}",
                case.input, case.expected, actual
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{group}: {} of {} cases failed\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// Implicit tags and basic structure (from `tests1.dat`).
#[test]
fn tests1() {
    check(
        "implicit tags and basic structure",
        &[
            Case {
                input: "Test",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"Test\"\n",
            },
            Case {
                input: "<p>One<p>Two",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <p>\n|       \"One\"\n|     <p>\n|       \"Two\"\n",
            },
            Case {
                input: "Line1<br>Line2<br>Line3<br>Line4",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"Line1\"\n|     <br>\n|     \"Line2\"\n|     <br>\n|     \"Line3\"\n|     <br>\n|     \"Line4\"\n",
            },
            Case {
                input: "<html>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n",
            },
            Case {
                input: "<head>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n",
            },
        ],
    );
}

/// Block elements closing an open p (from `blocks.dat`).
#[test]
fn blocks() {
    check(
        "block elements closing an open p",
        &[
            Case {
                input: "<!doctype html><p>foo<address>bar<p>baz",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <p>\n|       \"foo\"\n|     <address>\n|       \"bar\"\n|       <p>\n|         \"baz\"\n",
            },
            Case {
                input: "<!doctype html><address><p>foo</address>bar",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <address>\n|       <p>\n|         \"foo\"\n|     \"bar\"\n",
            },
            Case {
                input: "<!doctype html><p>foo<article>bar<p>baz",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <p>\n|       \"foo\"\n|     <article>\n|       \"bar\"\n|       <p>\n|         \"baz\"\n",
            },
            Case {
                input: "<!doctype html><article><p>foo</article>bar",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <article>\n|       <p>\n|         \"foo\"\n|     \"bar\"\n",
            },
        ],
    );
}

/// Misnested formatting: the adoption agency algorithm (from `adoption01.dat`).
#[test]
fn adoption01() {
    check(
        "misnested formatting: the adoption agency algorithm",
        &[
            Case {
                input: "<a><p></a></p>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <a>\n|     <p>\n|       <a>\n",
            },
            Case {
                input: "<a>1<p>2</a>3</p>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <a>\n|       \"1\"\n|     <p>\n|       <a>\n|         \"2\"\n|       \"3\"\n",
            },
            Case {
                input: "<a>1<button>2</a>3</button>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <a>\n|       \"1\"\n|     <button>\n|       <a>\n|         \"2\"\n|       \"3\"\n",
            },
            Case {
                input: "<a>1<b>2</a>3</b>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <a>\n|       \"1\"\n|       <b>\n|         \"2\"\n|     <b>\n|       \"3\"\n",
            },
            Case {
                input: "<a>1<div>2<div>3</a>4</div>5</div>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <a>\n|       \"1\"\n|     <div>\n|       <a>\n|         \"2\"\n|       <div>\n|         <a>\n|           \"3\"\n|         \"4\"\n|       \"5\"\n",
            },
        ],
    );
}

/// The adoption agency algorithm across a table (from `adoption02.dat`).
#[test]
fn adoption02() {
    check(
        "the adoption agency algorithm across a table",
        &[
            Case {
                input: "<b>1<i>2<p>3</b>4",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <b>\n|       \"1\"\n|       <i>\n|         \"2\"\n|     <i>\n|       <p>\n|         <b>\n|           \"3\"\n|         \"4\"\n",
            },
            Case {
                input: "<a><div><style></style><address><a>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <a>\n|     <div>\n|       <a>\n|         <style>\n|       <address>\n|         <a>\n|         <a>\n",
            },
        ],
    );
}

/// Tables (from `tables01.dat`).
#[test]
fn tables01() {
    check(
        "tables",
        &[
            Case {
                input: "<table><th>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <table>\n|       <tbody>\n|         <tr>\n|           <th>\n",
            },
            Case {
                input: "<table><td>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <table>\n|       <tbody>\n|         <tr>\n|           <td>\n",
            },
            Case {
                input: "<table><col foo='bar'>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <table>\n|       <colgroup>\n|         <col>\n|           foo=\"bar\"\n",
            },
            Case {
                input: "<table><colgroup></html>foo",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"foo\"\n|     <table>\n|       <colgroup>\n",
            },
        ],
    );
}

/// Foreign content: MathML (from `tests9.dat`).
#[test]
fn tests9() {
    check(
        "foreign content: MathML",
        &[
            Case {
                input: "<!DOCTYPE html><math></math>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <math math>\n",
            },
            Case {
                input: "<!DOCTYPE html><body><math></math>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <math math>\n",
            },
            Case {
                input: "<!DOCTYPE html><math><mi>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <math math>\n|       <math mi>\n",
            },
            Case {
                input: "<!DOCTYPE html><math><annotation-xml><svg><u>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <math math>\n|       <math annotation-xml>\n|         <svg svg>\n|     <u>\n",
            },
        ],
    );
}

/// Foreign content: SVG (from `tests10.dat`).
#[test]
fn tests10() {
    check(
        "foreign content: SVG",
        &[
            Case {
                input: "<!DOCTYPE html><svg></svg>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <svg svg>\n",
            },
            Case {
                input: "<!DOCTYPE html><svg></svg><![CDATA[a]]>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <svg svg>\n|     <!-- [CDATA[a]] -->\n",
            },
            Case {
                input: "<!DOCTYPE html><body><svg></svg>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <svg svg>\n",
            },
            Case {
                input: "<!DOCTYPE html><body><select><svg></svg></select>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     <select>\n|       <svg svg>\n",
            },
        ],
    );
}

/// Character references (from `entities01.dat`).
#[test]
fn entities01() {
    check(
        "character references",
        &[
            Case {
                input: "FOO&gt;BAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO>BAR\"\n",
            },
            Case {
                input: "FOO&gtBAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO>BAR\"\n",
            },
            Case {
                input: "FOO&gt BAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO> BAR\"\n",
            },
            Case {
                input: "FOO&gt;;;BAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO>;;BAR\"\n",
            },
            Case {
                input: "I'm &notit; I tell you",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"I'm ¬it; I tell you\"\n",
            },
        ],
    );
}

/// Character references in attribute values (from `entities02.dat`).
#[test]
fn entities02() {
    check(
        "character references in attribute values",
        &[
            Case {
                input: "<div bar=\"ZZ&gt;YY\"></div>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <div>\n|       bar=\"ZZ>YY\"\n",
            },
            Case {
                input: "<div bar=\"ZZ&\"></div>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <div>\n|       bar=\"ZZ&\"\n",
            },
            Case {
                input: "<div bar='ZZ&'></div>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <div>\n|       bar=\"ZZ&\"\n",
            },
            Case {
                input: "<div bar=ZZ&></div>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <div>\n|       bar=\"ZZ&\"\n",
            },
        ],
    );
}

/// Comments (from `comments01.dat`).
#[test]
fn comments01() {
    check(
        "comments",
        &[
            Case {
                input: "FOO<!-- BAR -->BAZ",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <!--  BAR  -->\n|     \"BAZ\"\n",
            },
            Case {
                input: "FOO<!-- BAR --!>BAZ",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <!--  BAR  -->\n|     \"BAZ\"\n",
            },
            Case {
                input: "FOO<!-- BAR --! >BAZ",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <!--  BAR --! >BAZ -->\n",
            },
            Case {
                input: "FOO<!-- BAR --!\n>BAZ",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <!--  BAR --!\n>BAZ -->\n",
            },
        ],
    );
}

/// DOCTYPEs (from `doctype01.dat`).
#[test]
fn doctype01() {
    check(
        "DOCTYPEs",
        &[
            Case {
                input: "<!DOCTYPE html>Hello",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     \"Hello\"\n",
            },
            Case {
                input: "<!dOctYpE HtMl>Hello",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     \"Hello\"\n",
            },
            Case {
                input: "<!DOCTYPEhtml>Hello",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|   <body>\n|     \"Hello\"\n",
            },
            Case {
                input: "<!DOCTYPE>Hello",
                context: None,
                expected: "| <!DOCTYPE >\n| <html>\n|   <head>\n|   <body>\n|     \"Hello\"\n",
            },
        ],
    );
}

/// Quirks mode DOCTYPEs (from `quirks01.dat`).
#[test]
fn quirks01() {
    check(
        "quirks mode DOCTYPEs",
        &[
            Case {
                input: "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Frameset//EN\"\n\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-frameset.dtd\"><p><table>",
                context: None,
                expected: "| <!DOCTYPE html \"-//W3C//DTD XHTML 1.0 Frameset//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-frameset.dtd\">\n| <html>\n|   <head>\n|   <body>\n|     <p>\n|     <table>\n",
            },
            Case {
                input: "<!DOCTYPE html SYSTEM \"http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd\"><p><table>",
                context: None,
                expected: "| <!DOCTYPE html \"\" \"http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd\">\n| <html>\n|   <head>\n|   <body>\n|     <p>\n|       <table>\n",
            },
            Case {
                input: "<!DOCTYPE html PUBLIC \"html\"><p><table>",
                context: None,
                expected: "| <!DOCTYPE html \"html\" \"\">\n| <html>\n|   <head>\n|   <body>\n|     <p>\n|       <table>\n",
            },
        ],
    );
}

/// Template contents (from `template.dat`).
#[test]
fn template() {
    check(
        "template contents",
        &[
            Case {
                input: "<body><template>Hello</template>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <template>\n|       content\n|         \"Hello\"\n",
            },
            Case {
                input: "<template>Hello</template>",
                context: None,
                expected: "| <html>\n|   <head>\n|     <template>\n|       content\n|         \"Hello\"\n|   <body>\n",
            },
            Case {
                input: "<template></template><div></div>",
                context: None,
                expected: "| <html>\n|   <head>\n|     <template>\n|       content\n|   <body>\n|     <div>\n",
            },
            Case {
                input: "<html><template>Hello</template>",
                context: None,
                expected: "| <html>\n|   <head>\n|     <template>\n|       content\n|         \"Hello\"\n|   <body>\n",
            },
        ],
    );
}

/// Select and option (from `webkit02.dat`).
#[test]
fn webkit02() {
    check(
        "select and option",
        &[
            Case {
                input: "<foo bar=qux/>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <foo>\n|       bar=\"qux/\"\n",
            },
            Case {
                input: "<p id=\"status\"><noscript><strong>A</strong></noscript><span>B</span></p>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <p>\n|       id=\"status\"\n|       <noscript>\n|         <strong>\n|           \"A\"\n|       <span>\n|         \"B\"\n",
            },
            Case {
                input: "<div><sarcasm><div></div></sarcasm></div>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <div>\n|       <sarcasm>\n|         <div>\n",
            },
        ],
    );
}

/// Script data escaping (from `scriptdata01.dat`).
#[test]
fn scriptdata01() {
    check(
        "script data escaping",
        &[
            Case {
                input: "FOO<script>'Hello'</script>BAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <script>\n|       \"'Hello'\"\n|     \"BAR\"\n",
            },
            Case {
                input: "FOO<script></script>BAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <script>\n|     \"BAR\"\n",
            },
            Case {
                input: "FOO<script></script >BAR",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"FOO\"\n|     <script>\n|     \"BAR\"\n",
            },
        ],
    );
}

/// Fragment parsing (from `tests_innerHTML_1.dat`).
#[test]
fn tests_inner_html_1() {
    check(
        "fragment parsing",
        &[
            Case {
                input: "<body><span>",
                context: Some((Namespace::Html, "body")),
                expected: "| <span>\n",
            },
            Case {
                input: "<span><body>",
                context: Some((Namespace::Html, "body")),
                expected: "| <span>\n",
            },
            Case {
                input: "<span><body>",
                context: Some((Namespace::Html, "div")),
                expected: "| <span>\n",
            },
            Case {
                input: "<body><span>",
                context: Some((Namespace::Html, "html")),
                expected: "| <head>\n| <body>\n|   <span>\n",
            },
        ],
    );
}

/// Fragment parsing in foreign content (from `foreign-fragment.dat`).
#[test]
fn foreign_fragment() {
    check(
        "fragment parsing in foreign content",
        &[
            Case {
                input: "<nobr>X",
                context: Some((Namespace::Svg, "path")),
                expected: "| <nobr>\n|   \"X\"\n",
            },
            Case {
                input: "<font color></font>X",
                context: Some((Namespace::Svg, "path")),
                expected: "| <font>\n|   color=\"\"\n| \"X\"\n",
            },
            Case {
                input: "<font></font>X",
                context: Some((Namespace::Svg, "path")),
                expected: "| <svg font>\n| \"X\"\n",
            },
            Case {
                input: "<g></path>X",
                context: Some((Namespace::Svg, "path")),
                expected: "| <svg g>\n|   \"X\"\n",
            },
        ],
    );
}

/// Processing instructions (from `processing-instructions.dat`).
#[test]
fn processing_instructions() {
    // By default `<?` is the HTML standard's bogus comment: HTML § 13.2.5.6
    // tag open state, U+003F, reconsumed in the bogus comment state.
    check(
        "processing instructions are bogus comments by default",
        &[
            Case {
                input: "<body><?something>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <!-- ?something -->\n",
            },
            Case {
                input: "<body><?something><span>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <!-- ?something -->\n|     <span>\n",
            },
        ],
    );

    // This crate's opt-in extension keeps them as processing instructions.
    let options = ParseOptions::new().with_processing_instructions(true);
    for (input, expected) in [
        (
            "<body><?something>",
            "| <html>\n|   <head>\n|   <body>\n|     <?something ?>\n",
        ),
        (
            "<body><?something><span>",
            "| <html>\n|   <head>\n|   <body>\n|     <?something ?>\n|     <span>\n",
        ),
        (
            "<body><?something good>",
            "| <html>\n|   <head>\n|   <body>\n|     <?something good?>\n",
        ),
    ] {
        assert_eq!(
            dump_tree(&parse_document_with(input, &options)),
            expected,
            "input {input:?}"
        );
    }
}

/// Deeply misnested markup (from `tricky01.dat`).
#[test]
fn tricky01() {
    check(
        "deeply misnested markup",
        &[
            Case {
                input: "<b><p>Bold </b> Not bold</p>\nAlso not bold.",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <b>\n|     <p>\n|       <b>\n|         \"Bold \"\n|       \" Not bold\"\n|     \"\nAlso not bold.\"\n",
            },
            Case {
                input: "<html>\n<font color=red><i>Italic and Red<p>Italic and Red </font> Just italic.</p> Italic only.</i> Plain\n<p>I should not be red. <font color=red>Red. <i>Italic and red.</p>\n<p>Italic and red. </i> Red.</font> I should not be red.</p>\n<b>Bold <i>Bold and italic</b> Only Italic </i> Plain",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     <font>\n|       color=\"red\"\n|       <i>\n|         \"Italic and Red\"\n|     <i>\n|       <p>\n|         <font>\n|           color=\"red\"\n|           \"Italic and Red \"\n|         \" Just italic.\"\n|       \" Italic only.\"\n|     \" Plain\n\"\n|     <p>\n|       \"I should not be red. \"\n|       <font>\n|         color=\"red\"\n|         \"Red. \"\n|         <i>\n|           \"Italic and red.\"\n|     <font>\n|       color=\"red\"\n|       <i>\n|         \"\n\"\n|     <p>\n|       <font>\n|         color=\"red\"\n|         <i>\n|           \"Italic and red. \"\n|         \" Red.\"\n|       \" I should not be red.\"\n|     \"\n\"\n|     <b>\n|       \"Bold \"\n|       <i>\n|         \"Bold and italic\"\n|     <i>\n|       \" Only Italic \"\n|     \" Plain\"\n",
            },
            Case {
                input: "<html><body>\n<p><font size=\"7\">First paragraph.</p>\n<p>Second paragraph.</p></font>\n<b><p><i>Bold and Italic</b> Italic</p>",
                context: None,
                expected: "| <html>\n|   <head>\n|   <body>\n|     \"\n\"\n|     <p>\n|       <font>\n|         size=\"7\"\n|         \"First paragraph.\"\n|     <font>\n|       size=\"7\"\n|       \"\n\"\n|       <p>\n|         \"Second paragraph.\"\n|     \"\n\"\n|     <b>\n|     <p>\n|       <b>\n|         <i>\n|           \"Bold and Italic\"\n|       <i>\n|         \" Italic\"\n",
            },
        ],
    );
}

/// Raw text and CDATA edge cases (from `tests16.dat`).
#[test]
fn tests16() {
    check(
        "raw text and CDATA edge cases",
        &[
            Case {
                input: "<!doctype html><script>",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|     <script>\n|   <body>\n",
            },
            Case {
                input: "<!doctype html><script>a",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|     <script>\n|       \"a\"\n|   <body>\n",
            },
            Case {
                input: "<!doctype html><script><",
                context: None,
                expected: "| <!DOCTYPE html>\n| <html>\n|   <head>\n|     <script>\n|       \"<\"\n|   <body>\n",
            },
        ],
    );
}
