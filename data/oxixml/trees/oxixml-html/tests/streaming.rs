//! The chunk-fed parser has to agree with the one-shot parser, exactly.
//!
//! [`oxixml_html::StreamingParser`] runs the same tokenizer and the same tree
//! construction stage as [`oxixml_html::parse_document`]; the only difference
//! is that its input can still grow while the state machine is running. This
//! harness is the evidence for that claim: every document below is parsed both
//! ways — in one piece, and split at every adversarial boundary this file can
//! think of, down to one byte at a time — and the two trees have to be the
//! same string.
//!
//! The interesting boundaries are the ones inside a construct the tokenizer
//! cannot finish without more input: inside a tag name, between the `<` and
//! the `!--` of a comment, in the middle of `&amp;` or `&#x27;`, between the
//! two bytes of a UTF-8 sequence, between the `<!` and the `[CDATA[`, and
//! between a `CR` and the `LF` that turns it into a single line break.

use oxixml_encoding::Encoding;
use oxixml_html::{
    Namespace, ParseOptions, StreamingParser, dump_tree, parse_bytes_with, parse_document,
    parse_fragment_with,
};

/// A varied corpus: every one of these is fed to the parser both whole and in
/// pieces.
const DOCUMENTS: &[&str] = [
    // Ordinary structure, implied tags, attributes.
    "<!DOCTYPE html><html><head><title>Hi</title></head><body><p class=greeting id=\"a b\">Hello<p>World</body></html>",
    // Malformed nesting, the adoption agency algorithm's bread and butter.
    "<p>1<b>2<i>3</b>4</i>5</p>",
    "<a href=x><a href=y>text</a>",
    "<div><span><em>unclosed",
    // Tables, foster parenting and the pending-character-token buffer.
    "<table><tr><td>cell</td><td>two</td></tr></table>",
    "<table>  stray  <tr><td>a<table><tr><td>inner</table>outer",
    "<table><colgroup><col><col>  <tbody><tr><th>h</th></tr></table>",
    // Foreign content, including a CDATA section and attribute adjustment.
    "<svg><circle cx=\"1\" cy=\"2\"/><foreignObject><div>html</div></foreignObject></svg>",
    "<svg><![CDATA[ raw < & ]]>after</svg>",
    "<math><mi>x</mi><annotation-xml encoding=\"text/html\"><p>p</p></annotation-xml></math>",
    "<svg><desc><table><tr><td>t</td></table></desc></svg>",
    // Character references: named, legacy without the semicolon, numeric,
    // hexadecimal, unknown, and one inside an attribute value.
    "<p>&amp; &AMP &lt;&gt; &#38; &#x26; &notit; &nope; &CounterClockwiseContourIntegral;",
    "<a title=\"&amp;&notit;&#x41;\" href=?a&amp=b>x</a>",
    "<p>&#x0; &#xD800; &#x110000; &#128169; &#x1F389;",
    // Comments, including the tricky nested and abrupt forms.
    "<!-- a comment --><p>x<!--><!--- --><!--a--!>b<!--c",
    "<!doctype html><!-- <!-- nested --> -->",
    // Raw text and script data escaping.
    "<script>var a = 1 < 2; document.write(\"<!--\"); </script><p>after",
    "<script><!--<script>inner</script>-->still</script>",
    "<style>p { color: red } /* < > */</style><textarea>&amp; <b>raw</b></textarea>",
    "<title>a &amp; b</title><xmp><p>raw</xmp>",
    // DOCTYPEs, including quirks-triggering and malformed ones.
    "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01//EN\" \"http://www.w3.org/TR/html4/strict.dtd\"><p>x",
    "<!DOCTYPE html SYSTEM 'about:legacy-compat'><p>x",
    "<!doctypehtml><p>x",
    "<!DOCTYPE><p>x",
    // Templates and select, which keep state of their own.
    "<template><tr><td>in template</td></tr></template><p>out",
    "<select><option>a<option selected>b<optgroup label=g><option>c</select>",
    // The head/noscript and frameset modes, whose character handling used to
    // depend on a whole run arriving at once.
    "<head><noscript> \tx<link></noscript><body>y",
    "<!DOCTYPE html><frameset><frame src=a></frameset> trailing x",
    "<!DOCTYPE html><frameset></frameset></html> tail y",
    // Line-break preprocessing, including the `<pre>` newline that is dropped.
    "<p>a\r\nb\rc\nd</p><pre>\nkept</pre><pre>\r\nalso kept</pre>",
    // Multi-byte characters everywhere they can appear.
    "<p title=\"caf\u{e9} \u{1f389}\">caf\u{e9} \u{2192} \u{1f389} \u{a5}</p>",
    "<\u{e9}lement>text</\u{e9}lement>",
    // Bogus comments, processing-instruction-looking markup, stray end tags.
    "<?php echo 1 ?><p>x</p></nope><!bogus>",
    // Nothing at all, and only white space.
    "",
    " \n\t ",
]
.as_slice();

/// Parses `html` in one piece, as the reference.
fn one_shot(html: &str) -> String {
    dump_tree(&parse_document(html))
}

/// Parses `html` by writing `size` bytes at a time.
fn in_chunks_of(html: &str, size: usize) -> String {
    let mut parser = StreamingParser::new();
    for chunk in html.as_bytes().chunks(size) {
        parser.write(chunk);
    }
    dump_tree(&parser.finish())
}

/// Parses `html` split at the given byte offsets.
fn split_at(html: &str, offsets: &[usize]) -> String {
    split_at_with(html, offsets, &ParseOptions::new())
}

/// Parses `html` split at the given byte offsets, with explicit options.
fn split_at_with(html: &str, offsets: &[usize], options: &ParseOptions) -> String {
    let bytes = html.as_bytes();
    let mut parser = StreamingParser::with_options(options);
    let mut start = 0;
    for &offset in offsets {
        let offset = offset.min(bytes.len());
        if offset > start {
            parser.write(&bytes[start..offset]);
            start = offset;
        }
    }
    parser.write(&bytes[start..]);
    dump_tree(&parser.finish())
}

/// A deterministic pseudo-random sequence, so that a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn next_below(&mut self, limit: usize) -> usize {
        // xorshift64*, which needs no dependency and is more than good enough
        // for choosing split points.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        if limit == 0 {
            0
        } else {
            (self.0 % limit as u64) as usize
        }
    }
}

#[test]
fn one_byte_at_a_time_parses_identically() {
    for document in DOCUMENTS {
        assert_eq!(
            in_chunks_of(document, 1),
            one_shot(document),
            "one byte at a time diverged for {document:?}"
        );
    }
}

#[test]
fn every_fixed_chunk_size_parses_identically() {
    for document in DOCUMENTS {
        let expected = one_shot(document);
        for size in [1, 2, 3, 4, 5, 7, 11, 16, 33, 64, 1024] {
            assert_eq!(
                in_chunks_of(document, size),
                expected,
                "chunks of {size} diverged for {document:?}"
            );
        }
    }
}

#[test]
fn every_single_split_point_parses_identically() {
    for document in DOCUMENTS {
        let expected = one_shot(document);
        for offset in 0..=document.len() {
            assert_eq!(
                split_at(document, &[offset]),
                expected,
                "a split at byte {offset} diverged for {document:?}"
            );
        }
    }
}

#[test]
fn random_splits_parse_identically() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for document in DOCUMENTS {
        let expected = one_shot(document);
        for _ in 0..20 {
            let mut offsets: Vec<usize> =
                (0..8).map(|_| rng.next_below(document.len() + 1)).collect();
            offsets.sort_unstable();
            assert_eq!(
                split_at(document, &offsets),
                expected,
                "random splits {offsets:?} diverged for {document:?}"
            );
        }
    }
}

#[test]
fn splits_inside_a_character_reference_parse_identically() {
    // Every offset inside every reference, including the legacy semicolon-less
    // forms and the longest name in the table.
    let cases = [
        "<p>a&amp;b",
        "<p>a&ampb",
        "<p>a&am",
        "<p>a&#38;b",
        "<p>a&#x26;b",
        "<p>a&#X26;b",
        "<p>a&#x1F389;b",
        "<p>&CounterClockwiseContourIntegral;",
        "<p>&notit;",
        "<p>&notin;",
        "<a href=\"?a&amp=b&amp;c=d\">x</a>",
        "<a href=?x&am>y</a>",
        "<textarea>&amp;</textarea>",
        "<title>&lt;</title>",
    ];
    for case in cases {
        let expected = one_shot(case);
        for offset in 0..=case.len() {
            assert_eq!(
                split_at(case, &[offset]),
                expected,
                "a split at byte {offset} diverged for {case:?}"
            );
        }
        assert_eq!(
            in_chunks_of(case, 1),
            expected,
            "one byte at a time: {case:?}"
        );
    }
}

#[test]
fn splits_inside_a_utf8_sequence_parse_identically() {
    // Two-, three- and four-byte sequences, in text, in a tag name and in an
    // attribute value, split at every byte of the sequence.
    let cases = [
        "<p>\u{e9}",
        "<p>\u{2192}",
        "<p>\u{1f389}",
        "<p title=\"\u{1f389}\">x",
        "<\u{1f389}>x",
        "<p>a\u{fffd}b",
        "<!--\u{1f389}-->",
        "<script>\u{1f389}</script>",
    ];
    for case in cases {
        let expected = one_shot(case);
        for offset in 0..=case.len() {
            assert_eq!(
                split_at(case, &[offset]),
                expected,
                "a split at byte {offset} diverged for {case:?}"
            );
        }
    }
}

#[test]
fn splits_inside_markup_declarations_parse_identically() {
    // `<!--`, `<!DOCTYPE`, `<![CDATA[` and the DOCTYPE keywords are decided by
    // looking ahead several bytes, which is exactly what a chunk boundary can
    // take away.
    let cases = [
        "<!--comment-->x",
        "<!-",
        "<!--",
        "<!--->x",
        "<!DOCTYPE html>x",
        "<!DOCTYP",
        "<!DOCTYPE html PUBLIC \"p\" \"s\">x",
        "<!DOCTYPE html SYSTEM \"s\">x",
        "<!DOCTYPE html PUBLI",
        "<!DOCTYPE html SYSTE",
        "<svg><![CDATA[x]]></svg>",
        "<svg><![CDAT",
        "<p><![CDATA[x]]>",
        "<!ELEMENT>x",
    ];
    for case in cases {
        let expected = one_shot(case);
        for offset in 0..=case.len() {
            assert_eq!(
                split_at(case, &[offset]),
                expected,
                "a split at byte {offset} diverged for {case:?}"
            );
        }
    }
}

#[test]
fn a_cr_lf_pair_split_across_chunks_is_one_line_break() {
    // HTML § 13.2.3.5 turns `CRLF` and a lone `CR` alike into a single `LF`,
    // which the streaming input has to do across the boundary between the two
    // halves of the pair.
    for case in [
        "<pre>\r\nkept",
        "a\r\nb",
        "a\rb",
        "a\r",
        "<textarea>\r\nkept</textarea>",
        "<p>a\r\r\nb",
    ] {
        let expected = one_shot(case);
        for offset in 0..=case.len() {
            assert_eq!(
                split_at(case, &[offset]),
                expected,
                "a split at byte {offset} diverged for {case:?}"
            );
        }
    }
}

#[test]
fn splits_inside_a_processing_instruction_parse_identically() {
    // The `<?…?>` extension's states are this crate's own, so the html5lib
    // corpus never reaches them and the differential over it says nothing
    // about them: every one of their resumption points needs a boundary of its
    // own here. The cases cover a well-formed instruction, an end of stream in
    // each of the five states, a disallowed target and an invalid first
    // character of one — the last two falling back to a bogus comment.
    let options = ParseOptions::new().with_processing_instructions(true);
    let cases = [
        "<?php echo 1 ?><p>after",
        "<?php    echo 1 ?>",
        "<?php?>",
        "<?php ? > ?>",
        "<?",
        "<?php",
        "<?php ",
        "<?php echo 1 ?",
        "<?xml version=\"1.0\" ?><p>x",
        "<?xml-stylesheet href=x ?>",
        "<?9bad ?>",
        "<?-nope ?>",
        "<?a-b_c data ?>",
    ];
    for case in cases {
        let expected = dump_tree(&oxixml_html::parse_document_with(case, &options));
        for offset in 0..=case.len() {
            assert_eq!(
                split_at_with(case, &[offset], &options),
                expected,
                "a split at byte {offset} diverged for {case:?}"
            );
        }
    }
}

#[test]
fn an_empty_chunk_does_not_resolve_a_held_back_cr() {
    // A `CR` at the end of a chunk waits for the next *character*, not for the
    // next call: an empty write in between must not turn `CRLF` into two line
    // breaks.
    let mut parser = StreamingParser::new();
    parser.write(b"<pre>a\r");
    parser.write(b"");
    parser.write_str("");
    parser.write(b"\nb");
    assert_eq!(dump_tree(&parser.finish()), one_shot("<pre>a\r\nb"));
}

#[test]
fn a_stream_with_nothing_in_it_is_still_a_document() {
    let expected = one_shot("");
    assert_eq!(dump_tree(&StreamingParser::new().finish()), expected);

    let mut parser = StreamingParser::new();
    parser.write(b"");
    parser.write(b"");
    parser.write_str("");
    assert_eq!(dump_tree(&parser.finish()), expected);

    // The same for the constructors that have their own start-up state to get
    // through: a sniffing parser that never sees a byte to sniff, and a
    // fragment parser with no fragment.
    let options = ParseOptions::new();
    assert_eq!(
        dump_tree(&StreamingParser::sniffing(None, &options).finish()),
        dump_tree(&parse_bytes_with(b"", None, &options))
    );
    assert_eq!(
        dump_tree(&StreamingParser::fragment(Namespace::Html, "div", &options).finish()),
        dump_tree(&parse_fragment_with("", Namespace::Html, "div", &options))
    );
    assert_eq!(
        dump_tree(&StreamingParser::with_encoding(Encoding::EucJp, &options).finish()),
        expected
    );
}

#[test]
fn a_truncated_multi_byte_character_at_the_end_of_a_stream_is_replaced() {
    // The decoder's `last` flush has to reach the tokenizer before the input
    // is closed, or a sequence cut short by the end of the stream would
    // vanish instead of becoming U+FFFD.
    let mut parser = StreamingParser::new();
    parser.write(b"<p>a\xe2\x82");
    let streamed = dump_tree(&parser.finish());
    assert_eq!(streamed, one_shot("<p>a\u{fffd}"));
}

#[test]
fn write_str_and_write_agree() {
    for document in DOCUMENTS {
        let expected = one_shot(document);

        let mut parser = StreamingParser::new();
        parser.write_str(document);
        assert_eq!(dump_tree(&parser.finish()), expected, "{document:?}");

        // Interleaved, at character boundaries.
        let mut parser = StreamingParser::new();
        let middle = (document.len() / 2..=document.len())
            .find(|&index| document.is_char_boundary(index))
            .unwrap_or(document.len());
        parser.write(&document.as_bytes()[..middle]);
        parser.write_str(&document[middle..]);
        assert_eq!(dump_tree(&parser.finish()), expected, "{document:?}");
    }
}

#[test]
fn options_are_honoured_by_the_streaming_parser() {
    let options = ParseOptions::new().with_processing_instructions(true);
    let mut parser = StreamingParser::with_options(&options);
    parser.write(b"<?ph");
    parser.write(b"p echo 1 ?><p>x");
    let streamed = dump_tree(&parser.finish());
    assert!(streamed.starts_with("| <?php echo 1 ?>"), "{streamed}");

    // And the depth ceiling a deep document trips.
    let deep = "<div>".repeat(600);
    let options = ParseOptions::new().with_max_depth(10);
    let mut parser = StreamingParser::with_options(&options);
    for chunk in deep.as_bytes().chunks(7) {
        parser.write(chunk);
    }
    let streamed = parser.finish();
    assert!(
        streamed
            .errors()
            .iter()
            .any(|error| error.code() == oxixml_html::error::codes::MAX_DEPTH_EXCEEDED)
    );
    assert_eq!(
        dump_tree(&streamed),
        dump_tree(&oxixml_html::parse_document_with(&deep, &options))
    );
}

#[test]
fn quirks_mode_matches_the_one_shot_parse() {
    for document in DOCUMENTS {
        let expected = parse_document(document);
        let mut parser = StreamingParser::new();
        for chunk in document.as_bytes().chunks(3) {
            parser.write(chunk);
        }
        let streamed = parser.finish();
        assert_eq!(
            streamed.quirks_mode(),
            expected.quirks_mode(),
            "quirks mode diverged for {document:?}"
        );
        assert_eq!(streamed.is_fragment(), expected.is_fragment());
    }
}

#[test]
fn fragments_stream_the_same_way_documents_do() {
    let cases: &[(&str, Namespace, &str)] = &[
        ("<td>cell", Namespace::Html, "tr"),
        ("<td>cell", Namespace::Html, "div"),
        ("a &amp; b", Namespace::Html, "title"),
        ("<b>x</b>", Namespace::Html, "textarea"),
        ("<circle/>", Namespace::Svg, "svg"),
        ("<tr><td>x", Namespace::Html, "template"),
        ("<option>a<option>b", Namespace::Html, "select"),
    ];
    for &(fragment, namespace, context) in cases {
        let options = ParseOptions::new();
        let expected = dump_tree(&parse_fragment_with(fragment, namespace, context, &options));
        for size in [1, 2, 3, 5, 1024] {
            let mut parser = StreamingParser::fragment(namespace, context, &options);
            for chunk in fragment.as_bytes().chunks(size) {
                parser.write(chunk);
            }
            assert_eq!(
                dump_tree(&parser.finish()),
                expected,
                "fragment {fragment:?} in {context} diverged at chunk size {size}"
            );
        }
    }
}

#[test]
fn an_explicit_encoding_decodes_across_chunk_boundaries() {
    // windows-1252 is single-byte, so a boundary can only split a construct;
    // EUC-JP is multi-byte, so it can split a character too.
    let cases: &[(Encoding, &[u8])] = &[
        (Encoding::Windows1252, b"<p>\x93quoted\x94 caf\xe9</p>"),
        (
            Encoding::EucJp,
            b"<p title=\"\xa4\xa2\">\xa4\xa4\xa4\xa6&amp;\xa4\xa8</p>",
        ),
        (Encoding::Utf8, "<p>caf\u{e9} \u{1f389}</p>".as_bytes()),
    ];
    for &(encoding, bytes) in cases {
        let options = ParseOptions::new();
        let expected = dump_tree(&parse_bytes_with(bytes, Some(encoding.name()), &options));
        for size in [1, 2, 3, 5, 8, 1024] {
            let mut parser = StreamingParser::with_encoding(encoding, &options);
            for chunk in bytes.chunks(size) {
                parser.write(chunk);
            }
            assert_eq!(
                dump_tree(&parser.finish()),
                expected,
                "{encoding:?} diverged at chunk size {size}"
            );
        }
    }
}

#[test]
fn sniffing_finds_the_encoding_a_short_stream_declares() {
    let cases: &[&[u8]] = &[
        b"\xef\xbb\xbf<p>caf\xc3\xa9",
        b"<meta charset=utf-8><p>caf\xc3\xa9",
        b"<meta charset=iso-8859-2><p>\xe9",
        b"<meta http-equiv=\"content-type\" content=\"text/html; charset=utf-8\"><p>\xc3\xa9",
        b"<p>no declaration at all: \x93",
    ];
    for &bytes in cases {
        let options = ParseOptions::new();
        let expected = dump_tree(&parse_bytes_with(bytes, None, &options));
        for size in [1, 3, 7, 64, 4096] {
            let mut parser = StreamingParser::sniffing(None, &options);
            for chunk in bytes.chunks(size) {
                parser.write(chunk);
            }
            assert_eq!(
                dump_tree(&parser.finish()),
                expected,
                "sniffing diverged at chunk size {size} for {bytes:?}"
            );
        }
    }
}

#[test]
fn sniffing_honours_a_transport_charset() {
    let bytes = b"<meta charset=utf-8><p>caf\xe9";
    let options = ParseOptions::new();
    let expected = dump_tree(&parse_bytes_with(bytes, Some("windows-1252"), &options));
    let mut parser = StreamingParser::sniffing(Some("windows-1252"), &options);
    for chunk in bytes.chunks(4) {
        parser.write(chunk);
    }
    assert_eq!(dump_tree(&parser.finish()), expected);
}

#[test]
fn sniffing_commits_after_the_prescan_limit() {
    // The documented difference from `parse_bytes`: a `<meta charset>` beyond
    // the read-ahead budget is not seen by a stream, because the bytes before
    // it have already been decoded and handed to the tokenizer.
    let mut bytes = b"<!-- ".to_vec();
    bytes.extend(std::iter::repeat_n(
        b'x',
        StreamingParser::PRESCAN_LIMIT * 2,
    ));
    // Byte 0xE8 is `\u{e8}` in windows-1252 and `\u{10d}` in iso-8859-2, so
    // which encoding won is visible in the tree.
    bytes.extend_from_slice(b" --><meta charset=iso-8859-2><p>\xe8");

    let options = ParseOptions::new();
    let mut parser = StreamingParser::sniffing(None, &options);
    for chunk in bytes.chunks(128) {
        parser.write(chunk);
    }
    let streamed = dump_tree(&parser.finish());
    let buffered = dump_tree(&parse_bytes_with(&bytes, None, &options));
    assert_ne!(
        streamed, buffered,
        "the prescan budget is documented to make this case differ"
    );
    // What the stream produced is the windows-1252 default reading of the same
    // bytes, which is what a browser that gave up reading ahead would show.
    assert_eq!(
        streamed,
        dump_tree(&parse_bytes_with(&bytes, Some("windows-1252"), &options))
    );
}

#[test]
fn nodes_are_attached_before_the_stream_ends() {
    let mut parser = StreamingParser::new();
    parser.write(b"<!DOCTYPE html><body><p>one</p><p>two</p><p>thr");

    let document = parser.document();
    let paragraphs: Vec<_> = document
        .get_elements_by_tag_name_ns(document.root(), Some(oxixml_html::HTML_NAMESPACE), "p")
        .collect();
    assert_eq!(
        paragraphs.len(),
        3,
        "every started paragraph is in the tree"
    );
    assert_eq!(
        document.text_content(paragraphs[0]).as_deref(),
        Some("one"),
        "and the finished ones already have their text"
    );

    parser.write(b"ee</p>");
    let parsed = parser.finish();
    assert_eq!(
        dump_tree(&parsed),
        one_shot("<!DOCTYPE html><body><p>one</p><p>two</p><p>three</p>")
    );
}

#[test]
fn a_large_document_does_not_accumulate_its_input() {
    // A megabyte of ordinary markup, fed in four-kilobyte chunks: what the
    // parser holds on to is the tail of the construct it is in the middle of,
    // never the document. (The tree it is building of course keeps growing —
    // that is the output, not a buffer.)
    const CHUNK: usize = 4096;
    let mut html = String::from("<!DOCTYPE html><body>");
    for index in 0..40_000 {
        html.push_str("<p class=row>row ");
        html.push_str(&index.to_string());
        html.push_str(" &amp; more</p>\n");
    }
    assert!(html.len() > 1_000_000, "the corpus should be big enough");

    let mut parser = StreamingParser::new();
    let mut high_water = 0;
    for chunk in html.as_bytes().chunks(CHUNK) {
        parser.write(chunk);
        high_water = high_water.max(parser.buffered_bytes());
    }
    assert!(
        high_water < CHUNK,
        "the parser retained {high_water} bytes of a {} byte document",
        html.len()
    );
    assert_eq!(dump_tree(&parser.finish()), one_shot(&html));
}

#[test]
fn an_enormous_unclosed_comment_does_not_grow_the_input_buffer() {
    // The retained-input bound is "whatever the current construct still needs",
    // so a comment that never closes is the pathological case — and it is
    // bounded by the comment, not by the document.
    const CHUNK: usize = 4096;
    let mut html = String::from("<p>before<!--");
    html.push_str(&"c".repeat(200_000));
    html.push_str("--><p>after");

    let mut parser = StreamingParser::new();
    let mut high_water = 0;
    for chunk in html.as_bytes().chunks(CHUNK) {
        parser.write(chunk);
        high_water = high_water.max(parser.buffered_bytes());
    }
    // The comment's own data lives in the tokenizer's comment buffer, not in
    // the input buffer, so even this stays bounded by a chunk.
    assert!(
        high_water < CHUNK,
        "the parser retained {high_water} bytes while one comment was open"
    );
    assert_eq!(dump_tree(&parser.finish()), one_shot(&html));
}

#[test]
fn positions_stay_absolute_after_the_consumed_input_is_released() {
    // Errors carry a position into the *stream*, not into whatever part of it
    // the parser still happens to be holding.
    let html = "<p>line one\n<p>line two\n<p>a &nope; here\n";
    let expected = parse_document(html);
    let mut parser = StreamingParser::new();
    for chunk in html.as_bytes().chunks(3) {
        parser.write(chunk);
    }
    let streamed = parser.finish();

    let expected_positions: Vec<_> = expected
        .errors()
        .iter()
        .map(|error| (error.code(), error.position()))
        .collect();
    let streamed_positions: Vec<_> = streamed
        .errors()
        .iter()
        .map(|error| (error.code(), error.position()))
        .collect();
    assert_eq!(streamed_positions, expected_positions);
    assert!(
        !expected_positions.is_empty(),
        "the input should report at least one parse error"
    );
}
