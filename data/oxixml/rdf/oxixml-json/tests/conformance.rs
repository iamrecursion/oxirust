//! Conformance corpus (JSONTestSuite-style `y_`/`n_` vectors), incremental-feeding equivalence,
//! serializer round-trip, and error-position-accuracy tests for `oxixml-json`.
//!
//! The `y_`/`n_` naming convention and many of the specific edge cases below are drawn from
//! [nst/JSONTestSuite](https://github.com/nst/JSONTestSuite) (a widely used JSON conformance
//! corpus): `y_` vectors must be accepted, `n_` vectors must be rejected. Each vector is spelled
//! out inline (rather than vendoring the upstream files) so the crate stays dependency- and
//! network-free.

use oxixml_json::{
    JsonEvent, LowLevelJsonParser, LowLevelJsonParserResult, SliceJsonParser, WriterJsonSerializer,
};

/// Drains a [`SliceJsonParser`] to completion, returning `Ok(())` if the whole document parsed
/// (through [`JsonEvent::Eof`]) without error, or the first [`Err`] encountered.
fn parse_to_end(input: &[u8]) -> Result<(), String> {
    let mut parser = SliceJsonParser::new(input);
    loop {
        match parser.parse_next() {
            Ok(JsonEvent::Eof) => return Ok(()),
            Ok(_) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// JSONTestSuite-style vectors that MUST be accepted.
const ACCEPT: &[(&str, &[u8])] = &[
    ("y_array_arraysWithSpaces", b"[[]   ]"),
    ("y_array_empty-string", b"[\"\"]"),
    ("y_array_empty", b"[]"),
    ("y_array_ending_with_newline", b"[\"x\"]\n"),
    ("y_array_false", b"[false]"),
    ("y_array_heterogeneous", b"[null, 1, \"1\", {}]"),
    ("y_array_null", b"[null]"),
    ("y_array_with_1_and_newline", b"[1\n]"),
    ("y_array_with_leading_space", b" [1]"),
    ("y_array_with_several_null", b"[1,null,null,null,2]"),
    ("y_array_with_trailing_space", b"[2] "),
    ("y_number", b"[123e65]"),
    ("y_number_0e+1", b"[0e+1]"),
    ("y_number_0e1", b"[0e1]"),
    ("y_number_after_space", b"[ 4]"),
    (
        "y_number_double_close_to_zero",
        b"[-0.00000000000000000000000000000000000000001]",
    ),
    ("y_number_int_with_exp", b"[20e1]"),
    ("y_number_minus_zero", b"[-0]"),
    ("y_number_negative_int", b"[-123]"),
    ("y_number_negative_one", b"[-1]"),
    ("y_number_real_capital_e", b"[1E22]"),
    ("y_number_real_capital_e_neg_exp", b"[1E-2]"),
    ("y_number_real_capital_e_pos_exp", b"[1E+2]"),
    ("y_number_real_exponent", b"[123e45]"),
    ("y_number_real_fraction_exponent", b"[123.456e78]"),
    ("y_number_real_neg_exp", b"[1e-2]"),
    ("y_number_real_pos_exponent", b"[1e+2]"),
    ("y_number_simple_int", b"[123]"),
    ("y_number_simple_real", b"[123.456789]"),
    ("y_object", b"{\"asd\":\"sdf\", \"dfg\":\"fgh\"}"),
    ("y_object_basic", b"{\"asd\":\"sdf\"}"),
    ("y_object_duplicated_key", b"{\"a\":\"b\",\"a\":\"c\"}"),
    (
        "y_object_duplicated_key_and_value",
        b"{\"a\":\"b\",\"a\":\"b\"}",
    ),
    ("y_object_empty", b"{}"),
    ("y_object_empty_key", b"{\"\":0}"),
    ("y_object_escaped_null_in_key", b"{\"foo\\u0000bar\": 42}"),
    (
        "y_object_extreme_numbers",
        b"{\"min\": -1.0e+28, \"max\": 1.0e+28}",
    ),
    ("y_object_simple", b"{\"a\":[]}"),
    ("y_object_with_newlines", b"{\n\"a\": \"b\"\n}"),
    ("y_string_accepted_surrogate_pair", b"[\"\\uD801\\udc37\"]"),
    (
        "y_string_accepted_surrogate_pairs",
        b"[\"\\ud83d\\ude39\\ud83d\\udc8d\"]",
    ),
    (
        "y_string_allowed_escapes",
        b"[\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"]",
    ),
    ("y_string_backslash_and_u_escaped_zero", b"[\"\\\\u0000\"]"),
    ("y_string_backslash_doublequotes", b"[\"\\\"\"]"),
    ("y_string_comments", b"[\"a/*b*/c/*d//e\"]"),
    ("y_string_double_escape_a", b"[\"\\\\a\"]"),
    ("y_string_double_escape_n", b"[\"\\\\n\"]"),
    ("y_string_escaped_control_character", b"[\"\\u0012\"]"),
    ("y_string_escaped_noncharacter", b"[\"\\uFFFF\"]"),
    ("y_string_in_array", b"[\"asd\"]"),
    ("y_string_in_array_with_leading_space", b"[ \"asd\"]"),
    ("y_string_last_surrogates_1_and_2", b"[\"\\uDBFF\\uDFFF\"]"),
    ("y_string_nbsp_uescaped", b"[\"new\\u00A0line\"]"),
    ("y_string_null_escape", b"[\"\\u0000\"]"),
    ("y_string_pi", b"[\"\\u03c0\"]"),
    ("y_string_simple_ascii", b"[\"asd \"]"),
    ("y_string_space", b"\"  \""),
    ("y_string_three_byte_utf8", b"[\"\\u0821\"]"),
    ("y_string_two_byte_utf8", b"[\"\\u0123\"]"),
    ("y_string_u+2028_line_sep", b"[\"\\u2028\"]"),
    ("y_string_u+2029_par_sep", b"[\"\\u2029\"]"),
    ("y_string_uescape", b"[\"\\u0061\\u30af\\u5947\\u0021\"]"),
    ("y_string_unescaped_char_delete", b"[\"a\x7fa\"]"),
    ("y_string_unicode", b"[\"\\uA66D\"]"),
    ("y_string_unicode_escaped_backslash", b"[\"\\u005C\"]"),
    ("y_string_unicode_2", b"[\"\\u0022\"]"),
    ("y_string_utf8", b"[\"\\u2603\"]"),
    ("y_string_with_del_character", b"[\"a\x7fa\"]"),
    ("y_structure_lonely_false", b"false"),
    ("y_structure_lonely_int", b"42"),
    ("y_structure_lonely_negative_real", b"-0.1"),
    ("y_structure_lonely_null", b"null"),
    ("y_structure_lonely_string", b"\"asd\""),
    ("y_structure_lonely_true", b"true"),
    ("y_structure_string_empty", b"\"\""),
    ("y_structure_trailing_newline", b"[1]\n"),
    ("y_structure_true_in_array", b"[true]"),
    ("y_structure_whitespace_array", b" [] "),
];

/// JSONTestSuite-style vectors that MUST be rejected (a [`oxixml_json::JsonSyntaxError`] must
/// surface somewhere in the event stream).
const REJECT: &[(&str, &[u8])] = &[
    ("n_array_1_true_without_comma", b"[1 true]"),
    ("n_array_colon_instead_of_comma", b"[\"\": 1]"),
    ("n_array_comma_after_close", b"[\"\"],"),
    ("n_array_comma_and_number", b"[,1]"),
    ("n_array_double_comma", b"[1,,2]"),
    ("n_array_double_extra_comma", b"[1,2,,]"),
    ("n_array_extra_close", b"[\"x\"]]"),
    ("n_array_extra_comma", b"[1,]"),
    ("n_array_incomplete", b"[x"),
    ("n_array_inner_array_no_comma", b"[3[4]]"),
    ("n_array_items_separated_by_semicolon", b"[1:2]"),
    ("n_array_just_comma", b"[,]"),
    ("n_array_just_minus", b"[-]"),
    ("n_array_missing_value", b"[   , \"\"]"),
    ("n_array_number_and_several_commas", b"[1,,]"),
    ("n_array_star_inside", b"[*]"),
    ("n_array_unclosed", b"[\"\""),
    ("n_array_unclosed_trailing_comma", b"[1,"),
    ("n_array_unclosed_with_object_inside", b"[{}"),
    ("n_incomplete_false", b"[fals]"),
    ("n_incomplete_null", b"[nul]"),
    ("n_incomplete_true", b"[tru]"),
    ("n_number_plus_plus", b"[++1234]"),
    ("n_number_plus_1", b"[+1]"),
    ("n_number_plus_Inf", b"[+Inf]"),
    ("n_number_neg_leading_zero", b"[-01]"),
    ("n_number_neg_double_dot", b"[-1.0.]"),
    ("n_number_neg_trailing_dot", b"[-2.]"),
    ("n_number_neg_NaN", b"[-NaN]"),
    ("n_number_dot_neg_1", b"[.-1]"),
    ("n_number_dot_2e_neg_3", b"[.2e-3]"),
    ("n_number_double_dot", b"[0.1.2]"),
    ("n_number_0_3e_plus", b"[0.3e+]"),
    ("n_number_0_3e", b"[0.3e]"),
    ("n_number_0_dot_e1", b"[0.e1]"),
    ("n_number_0_capital_e_plus", b"[0E+]"),
    ("n_number_0_capital_e", b"[0E]"),
    ("n_number_0e_plus", b"[0e+]"),
    ("n_number_0e", b"[0e]"),
    ("n_number_1_0e_plus", b"[1.0e+]"),
    ("n_number_1_0e_minus", b"[1.0e-]"),
    ("n_number_1_0e", b"[1.0e]"),
    ("n_number_1_000", b"[1 000.0]"),
    ("n_number_1eE2", b"[1eE2]"),
    ("n_number_2_e_plus_3", b"[2.e+3]"),
    ("n_number_2_e_minus_3", b"[2.e-3]"),
    ("n_number_2_e3", b"[2.e3]"),
    ("n_number_9_e_plus", b"[9.e+]"),
    ("n_number_expression", b"[1+2]"),
    ("n_number_hex_1_digit", b"[0x1]"),
    ("n_number_hex_2_digits", b"[0x42]"),
    ("n_number_Inf", b"[Inf]"),
    ("n_number_infinity", b"[Infinity]"),
    ("n_number_invalid_plus_minus", b"[0e+-1]"),
    ("n_number_minus_infinity", b"[-Infinity]"),
    ("n_number_minus_trailing_garbage", b"[-foo]"),
    ("n_number_minus_space_1", b"[- 1]"),
    ("n_number_NaN", b"[NaN]"),
    ("n_number_neg_int_leading_zero", b"[-012]"),
    ("n_number_neg_real_without_int_part", b"[-.123]"),
    ("n_number_neg_with_garbage_at_end", b"[-1x]"),
    ("n_number_real_garbage_after_e", b"[1ea]"),
    ("n_number_real_without_fractional_part", b"[1.]"),
    ("n_number_starting_with_dot", b"[.123]"),
    ("n_number_with_alpha", b"[1.2a-3]"),
    ("n_number_with_leading_zero", b"[012]"),
    ("n_object_bad_value", b"[\"x\", truth]"),
    ("n_object_bracket_key", b"{[: \"x\"}"),
    ("n_object_comma_instead_of_colon", b"{\"x\", null}"),
    ("n_object_double_colon", b"{\"x\"::\"b\"}"),
    ("n_object_garbage_at_end", b"{\"a\":\"a\" 123}"),
    ("n_object_key_with_single_quotes", b"{key: 'value'}"),
    ("n_object_missing_colon", b"{\"a\" b}"),
    ("n_object_missing_key", b"{:\"b\"}"),
    ("n_object_missing_semicolon", b"{\"a\" \"b\"}"),
    ("n_object_no_colon", b"{\"a\"}"),
    ("n_object_non_string_key", b"{1:1}"),
    ("n_object_repeated_null_null", b"{null:null,null:null}"),
    ("n_object_single_quote", b"{'a':0}"),
    ("n_object_trailing_comma", b"{\"id\":0,}"),
    ("n_object_unquoted_key", b"{a: \"b\"}"),
    (
        "n_object_with_single_string",
        b"{ \"foo\" : \"bar\", \"a\" }",
    ),
    ("n_single_space", b" "),
    ("n_string_single_quote", b"['single quote']"),
    ("n_string_single_string_no_double_quotes", b"abc"),
    ("n_string_incomplete_escape", b"[\"\\\""),
    ("n_string_incomplete_surrogate", b"[\"\\uD834\\uDd\"]"),
    (
        "n_string_invalid_two_high_surrogates",
        b"[\"\\uD800\\uD800\"]",
    ),
    ("n_string_invalid_backslash_esc", b"[\"\\a\"]"),
    ("n_string_invalid_unicode_escape", b"[\"\\uqqqq\"]"),
    ("n_string_no_quotes_with_bad_escape", b"[\\n]"),
    ("n_string_single_doublequote", b"\""),
    ("n_string_start_escape_unclosed", b"[\"\\"),
    ("n_string_unescaped_ctrl_char_tab", b"[\"a\tb\"]"),
    ("n_string_unescaped_ctrl_char_nul", b"[\"a\x00b\"]"),
    ("n_string_unescaped_newline", b"[\"a\nb\"]"),
    ("n_string_invalid_utf8_lead_byte", b"[\"\xff\"]"),
    ("n_string_invalid_utf8_continuation", b"[\"\xc3\x28\"]"),
    ("n_structure_angle_bracket", b"<.>"),
    ("n_structure_array_trailing_garbage", b"[1]x"),
    ("n_structure_array_with_extra_array_close", b"[1]]"),
    ("n_structure_array_with_unclosed_string", b"[\"asd]"),
    ("n_structure_capitalized_True", b"[True]"),
    ("n_structure_close_unopened_array", b"1]"),
    (
        "n_structure_comma_instead_of_closing_brace",
        b"{\"x\": true,",
    ),
    ("n_structure_double_array", b"[][]"),
    ("n_structure_end_array", b"]"),
    ("n_structure_lone_open_bracket", b"["),
    ("n_structure_no_data", b""),
    ("n_structure_null_byte_outside_string", b"[\0]"),
    ("n_structure_number_with_trailing_garbage", b"2@"),
    ("n_structure_object_followed_by_closing_object", b"{}}"),
    ("n_structure_object_unclosed_no_value", b"{\"\":"),
    (
        "n_structure_object_with_trailing_garbage",
        b"{\"a\": true} \"x\"",
    ),
    ("n_structure_open_array_apostrophe", b"['"),
    ("n_structure_open_array_comma", b"[,"),
    ("n_structure_open_array_object", b"[{\"\":"),
    ("n_structure_open_array_open_object", b"[{"),
    ("n_structure_open_array_open_string", b"[\"a"),
    ("n_structure_open_array_string", b"[\"a\""),
    ("n_structure_open_object", b"{"),
    ("n_structure_open_object_close_array", b"{]"),
    ("n_structure_open_object_comma", b"{,"),
    ("n_structure_open_object_open_array", b"{["),
    ("n_structure_open_object_open_string", b"{\"a"),
    ("n_structure_single_star", b"*"),
    ("n_structure_trailing_hash", b"{\"a\":\"b\"}#"),
    ("n_structure_unclosed_array", b"[1"),
    ("n_structure_unclosed_array_partial_null", b"[ false, nul"),
    (
        "n_structure_unclosed_array_unfinished_false",
        b"[ true, fals",
    ),
    (
        "n_structure_unclosed_array_unfinished_true",
        b"[ false, tru",
    ),
    ("n_structure_unclosed_object", b"{\"asd\":\"asd\""),
    ("n_structure_whitespace_formfeed", b"[\x0c]"),
];

#[test]
fn accept_vectors_are_accepted() {
    assert!(ACCEPT.len() >= 60, "need at least 60 accept vectors");
    let mut failures = Vec::new();
    for (name, input) in ACCEPT {
        if let Err(message) = parse_to_end(input) {
            failures.push(format!("{name}: expected accept, got error: {message}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn reject_vectors_are_rejected() {
    assert!(REJECT.len() >= 60, "need at least 60 reject vectors");
    let mut failures = Vec::new();
    for (name, input) in REJECT {
        if parse_to_end(input).is_ok() {
            failures.push(format!(
                "{name}: expected a rejection, but it parsed cleanly"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Collects every event a [`SliceJsonParser`] produces for a well-formed document, panicking on
/// the first error (the corpus documents here are all known-good).
fn events_via_slice(input: &[u8]) -> Vec<JsonEvent<'static>> {
    let mut parser = SliceJsonParser::new(input);
    let mut out = Vec::new();
    loop {
        match parser.parse_next() {
            Ok(JsonEvent::Eof) => {
                out.push(JsonEvent::Eof);
                return out;
            }
            Ok(event) => out.push(event.into_owned()),
            Err(e) => panic!("unexpected syntax error in known-good document: {e}"),
        }
    }
}

/// Collects every event produced by feeding [`LowLevelJsonParser`] exactly one byte at a time.
fn events_via_byte_at_a_time(input: &[u8]) -> Vec<JsonEvent<'static>> {
    let mut parser = LowLevelJsonParser::new();
    let mut out = Vec::new();
    let mut consumed = 0usize;
    let mut fed_len = 0usize;
    loop {
        fed_len = (fed_len + 1).max(consumed).min(input.len());
        let is_ending = fed_len >= input.len();
        let LowLevelJsonParserResult {
            consumed_bytes,
            event,
        } = parser.parse_next(&input[consumed..fed_len], is_ending);
        consumed += consumed_bytes;
        if let Some(event) = event {
            let event = event
                .unwrap_or_else(|e| panic!("unexpected syntax error in known-good document: {e}"));
            let is_eof = event == JsonEvent::Eof;
            out.push(event.into_owned());
            if is_eof {
                return out;
            }
        }
    }
}

const REPRESENTATIVE_DOCUMENTS: &[&[u8]] = &[
    br#"{"key": [1, 2.5, true, null, "x\ny", -3.14e-10], "nested": {"a": [[]], "b": {}}}"#,
    "[\"\u{1F600}\", \"caf\u{e9}\", \"line1\\nline2\\ttab\"]".as_bytes(),
    br#"[1,2,3,4,5,6,7,8,9,10,{"a":{"b":{"c":{"d":[1,2,3]}}}}]"#,
    br#"{"":"empty key","dup":"a","dup":"b"}"#,
    b"null",
    b"3.14159",
    b"[]",
    b"{}",
];

#[test]
fn incremental_one_byte_feeding_matches_whole_buffer_parsing() {
    for input in REPRESENTATIVE_DOCUMENTS {
        let whole = events_via_slice(input);
        let incremental = events_via_byte_at_a_time(input);
        assert_eq!(
            whole,
            incremental,
            "byte-at-a-time feeding diverged for {:?}",
            String::from_utf8_lossy(input)
        );
    }
}

#[test]
fn serializer_round_trips_representative_documents() {
    for input in REPRESENTATIVE_DOCUMENTS {
        let events = events_via_slice(input);

        let mut writer = WriterJsonSerializer::new(Vec::new());
        for event in &events {
            if *event == JsonEvent::Eof {
                continue;
            }
            writer
                .serialize_event(event.clone())
                .unwrap_or_else(|e| panic!("serialization failed for {input:?}: {e}"));
        }
        let output = writer
            .finish()
            .unwrap_or_else(|e| panic!("finish failed for {input:?}: {e}"));

        // Re-parsing the serializer's output must reproduce the exact same event sequence.
        let round_tripped = events_via_slice(&output);
        assert_eq!(
            events,
            round_tripped,
            "round trip diverged for {:?} -> {:?}",
            String::from_utf8_lossy(input),
            String::from_utf8_lossy(&output)
        );
    }
}

/// (input, expected error start line, expected error start column) triples. Lines/columns are
/// **0-based**, matching the storage of [`oxixml_support::pos::TextPosition`]; columns count
/// Unicode scalar values, not bytes. (Rendered error messages add one to both.)
const POSITION_CASES: &[(&[u8], u64, u64)] = &[
    (b"[1, 2,]", 0, 6),
    (b"{\"a\": 1,}", 0, 8),
    (b"[1\n 2]", 1, 1),
    (b"{\n  \"a\": tru\n}", 1, 7),
    // The unterminated string itself never becomes a token (the lexer keeps waiting for a
    // closing quote), so the *outer* parser is the one that reports "unexpected end of file",
    // anchored at the last confirmed position (right after the `[`).
    (b"[\"unterminated", 0, 1),
    (b"nul", 0, 0),
];

/// Regression test for the 0-based storage of [`oxixml_json::TextPosition`]: reading
/// `.line` / `.column` programmatically on a known input must not be off by one, while the
/// rendered message keeps the human-facing 1-based numbering.
#[test]
fn error_position_fields_are_zero_based() {
    // Two lines; the bad token `tru` starts at 0-based line 1, column 2.
    let mut parser = SliceJsonParser::new(b"[\n  tru\n]");
    let mut last_error = None;
    loop {
        match parser.parse_next() {
            Ok(JsonEvent::Eof) => break,
            Ok(_) => {}
            Err(e) => {
                last_error = Some(e);
                break;
            }
        }
    }
    let error = last_error.expect("expected a syntax error");
    let location = error.location();
    assert_eq!(location.start.line, 1, "{error}");
    assert_eq!(location.start.column, 2, "{error}");
    assert_eq!(location.start.offset, 4, "{error}");
    // Display renders the same position 1-based.
    assert!(error.to_string().contains("line 2"), "{error}");
    assert!(error.to_string().contains("columns 3"), "{error}");
}

#[test]
fn error_positions_are_accurate() {
    for (input, expected_line, expected_column) in POSITION_CASES {
        let mut parser = SliceJsonParser::new(input);
        let mut last_error = None;
        loop {
            match parser.parse_next() {
                Ok(JsonEvent::Eof) => break,
                Ok(_) => {}
                Err(e) => {
                    last_error = Some(e);
                    break;
                }
            }
        }
        let error = last_error.unwrap_or_else(|| {
            panic!(
                "expected a syntax error for {:?}",
                String::from_utf8_lossy(input)
            )
        });
        let location = error.location();
        assert_eq!(
            location.start.line,
            *expected_line,
            "line mismatch for {:?}: {error}",
            String::from_utf8_lossy(input)
        );
        assert_eq!(
            location.start.column,
            *expected_column,
            "column mismatch for {:?}: {error}",
            String::from_utf8_lossy(input)
        );
    }
}
