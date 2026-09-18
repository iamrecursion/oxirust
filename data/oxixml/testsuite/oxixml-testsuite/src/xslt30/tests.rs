//! Unit tests for the XSLT 3.0 runner.

use super::*;

/// The three `year_component_values` capabilities the catalogue names are all answered yes,
/// so a case gated `satisfied="true"` runs and its `satisfied="false"` twin — which asserts
/// the *absence* of the same capability — is skipped. `date-094a`/`date-094n` are that pair:
/// both drive `date-094.xsl`, one requiring an error for year `0000` and the other requiring
/// `0000-02-29` to succeed.
#[test]
fn a_year_component_gate_admits_only_the_half_that_matches_this_processor() {
    let cases = [
        (
            r#"<year_component_values satisfied="true" value="support year zero"/>"#,
            None,
        ),
        (
            r#"<year_component_values satisfied="false" value="support year zero"/>"#,
            Some("year_component_values:support year zero"),
        ),
        (
            r#"<year_component_values satisfied="true" value="support negative year"/>"#,
            None,
        ),
        (
            r#"<year_component_values satisfied="false" value="support negative year"/>"#,
            Some("year_component_values:support negative year"),
        ),
        (
            r#"<year_component_values satisfied="true" value="support year above 9999"/>"#,
            None,
        ),
        (
            r#"<year_component_values satisfied="false" value="support year above 9999"/>"#,
            Some("year_component_values:support year above 9999"),
        ),
        // `satisfied` defaults to true, which is how the 56 `format-date-en`/`match`/`as`
        // cases that need negative years are written.
        (
            r#"<year_component_values value="support negative year"/>"#,
            None,
        ),
        // A capability this processor cannot answer for is undecidable in either direction.
        (
            r#"<year_component_values satisfied="true" value="support the julian calendar"/>"#,
            Some("year_component_values:support the julian calendar"),
        ),
        (
            r#"<year_component_values satisfied="false" value="support the julian calendar"/>"#,
            Some("year_component_values:support the julian calendar"),
        ),
    ];
    for (gate, expected) in cases {
        let tree = Tree::parse_str(&format!("<dependencies>{gate}</dependencies>"))
            .expect("well-formed fixture");
        let dependencies = tree.document_element().expect("a document element");
        assert_eq!(
            unsupported_dependency(&dependencies).as_deref(),
            expected,
            "{gate}"
        );
    }
}

/// A case may declare several of them — `date-094m` needs both year zero and negative
/// years — and every one has to hold for the case to run.
#[test]
fn several_year_component_gates_must_all_hold() {
    let tree = Tree::parse_str(
        r#"<dependencies>
             <spec value="XSLT30+"/>
             <year_component_values satisfied="true" value="support negative year"/>
             <year_component_values satisfied="true" value="support year zero"/>
           </dependencies>"#,
    )
    .expect("well-formed fixture");
    let dependencies = tree.document_element().expect("a document element");
    assert_eq!(unsupported_dependency(&dependencies), None);

    let tree = Tree::parse_str(
        r#"<dependencies>
             <year_component_values satisfied="true" value="support negative year"/>
             <year_component_values satisfied="false" value="support year above 9999"/>
           </dependencies>"#,
    )
    .expect("well-formed fixture");
    let dependencies = tree.document_element().expect("a document element");
    assert_eq!(
        unsupported_dependency(&dependencies).as_deref(),
        Some("year_component_values:support year above 9999")
    );
}

#[test]
fn literal_value_recognizes_quoted_strings_and_integers() {
    assert_eq!(literal_value("'hi'").to_string(), "hi");
    assert_eq!(literal_value("\"hi\"").to_string(), "hi");
    assert_eq!(literal_value("42").to_string(), "42");
    assert_eq!(literal_value("not-quoted").to_string(), "not-quoted");
}

/// `decl-function/function-0701` asserts `count(//td) = 64` and
/// `insn-call-template/call-template-1003` asserts `/out = "…"`: neither names `$result`,
/// because this catalogue evaluates an `<assert>` with the result document node as the
/// context item (`admin/catalog-schema.xsd`).
#[test]
fn an_xpath_assertion_is_evaluated_against_the_result_document() {
    let result = Tree::parse_str("<out><td>a</td><td>b</td></out>").expect("well-formed");
    let root = result.root();
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::from(Sequence::from(root.clone())));
    let tree = Tree::parse_str("<assert>count(//td) = 2</assert>").expect("well-formed");
    let assertion = tree.document_element().expect("a document element");
    assert_eq!(
        check_case(
            &assertion,
            Some(produced("<out><td>a</td><td>b</td></out>")),
            &outcome,
            Path::new(".")
        ),
        Outcome::Pass
    );
    assert_eq!(
        result_document(Some(produced("<out/>")), &outcome).map(NodeRef::node_kind),
        Some(NodeKind::Document)
    );
}

/// A successful transformation's `<result>` is a *wrapper*: the assertion inside it has to be
/// unwrapped before the checker sees it, exactly as the error path already did. Handing the
/// wrapper over instead decides every such case "unsupported assertion kind", which reads as
/// a skip and silently measures nothing — so this drives the whole of [`run_case`] over a
/// stylesheet of its own and insists on a real verdict.
#[test]
fn a_successful_transformation_decides_its_assertion() {
    let directory = std::env::temp_dir().join(format!(
        "oxixml-xslt30-result-wrapper-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    std::fs::write(
        directory.join("case.xsl"),
        r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
             <xsl:template name="xsl:initial-template"><out>ok</out></xsl:template>
           </xsl:stylesheet>"#,
    )
    .expect("fixture stylesheet");
    let tree = Tree::parse_str(
        r#"<test-case name="wrapper">
             <test><stylesheet file="case.xsl"/></test>
             <result><assert>/out = "ok"</assert></result>
           </test-case>"#,
    )
    .expect("well-formed fixture");
    let case = tree.document_element().expect("a document element");
    let environments = std::collections::BTreeMap::new();
    assert_eq!(
        run_case(&case, "wrapper", &directory, &environments, unhurried()),
        Outcome::Pass
    );

    // And a wrong expectation is a failure, not a skip: the case really was decided.
    let tree = Tree::parse_str(
        r#"<test-case name="wrapper">
             <test><stylesheet file="case.xsl"/></test>
             <result><assert>/out = "wrong"</assert></result>
           </test-case>"#,
    )
    .expect("well-formed fixture");
    let case = tree.document_element().expect("a document element");
    assert!(matches!(
        run_case(&case, "wrapper", &directory, &environments, unhurried()),
        Outcome::Fail(_)
    ));
    std::fs::remove_dir_all(&directory).ok();
}

/// A [`Produced`] over just a serialized principal result: the shape almost every assertion
/// under test here reads, with no `xsl:message` log and no secondary result documents.
fn produced(serialized: &str) -> Produced<'_> {
    Produced {
        serialized,
        messages: &[],
        secondary: &[],
    }
}

/// The outcome [`run_case`] reaches for one named `<test-case>` of a test-set file.
fn verdict_of(path: &Path, case_name: &str) -> Outcome {
    let text = std::fs::read_to_string(path).expect("the test-set file");
    let tree = Tree::parse_str(&text).expect("the test set parses");
    let set = tree.document_element().expect("a document element");
    let directory = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut environments = std::collections::BTreeMap::new();
    for child in element_children(&set) {
        if local_name(&child) == "environment"
            && let Some(name) = attribute(&child, "name")
        {
            environments.insert(name, Environment::parse(&child, &directory));
        }
    }
    let case = element_children(&set)
        .into_iter()
        .find(|child| {
            local_name(child) == "test-case"
                && attribute(child, "name").as_deref() == Some(case_name)
        })
        .expect("the named test case");
    run_case(&case, case_name, &directory, &environments, unhurried())
}

/// A deadline no test can reach: these fixtures measure verdicts, not time, and a transformation
/// that stops early because a shared machine was busy would fail one of them for the wrong reason.
/// The driver's own deadline is the runner's budget — see [`run_test_set`].
fn unhurried() -> std::time::Instant {
    std::time::Instant::now() + std::time::Duration::from_secs(3600)
}

/// The two catalogue cases whose `<assert>` names no `$result` and can only be decided with
/// the result document as the context item — `decl-function/function-0701`
/// (`count(//td) = 64`) and `insn-call-template/call-template-1003` (`/out = "…"`) — driven
/// through the real driver, so the rule is pinned end to end and not only at [`check_case`].
///
/// Needs the provisioned suite; like every other suite-dependent test here it decides nothing
/// when the suite is not there.
#[test]
fn the_context_item_cases_pass_against_the_real_catalogue() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    for (file, case) in [
        (
            "tests/decl/function/_function-test-set.xml",
            "function-0701",
        ),
        (
            "tests/insn/call-template/_call-template-test-set.xml",
            "call-template-1003",
        ),
    ] {
        let path = root.join(file);
        // On a worker sized like the driver's own: `run_case` declares `WORKER_STACK` to the
        // processor, and these two stylesheets recurse deeply on purpose.
        let worker = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || verdict_of(&path, case))
            .expect("a worker thread");
        assert_eq!(
            worker.join().expect("the case ran to completion"),
            Outcome::Pass,
            "{case}"
        );
    }
}

/// A transformation that raised instead of producing a tree has no result document for an
/// assertion to be focused on.
#[test]
fn a_failed_transformation_supplies_no_context_item() {
    let outcome: Result<XpathValue, ErrorInfo> = Err(ErrorInfo::bare("XTDE0040", "no such"));
    assert!(result_document(None, &outcome).is_none());
    let value: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    assert!(
        result_document(Some(produced("")), &value).is_none(),
        "an empty result is not a document node"
    );
}

#[test]
fn canonical_ignores_attribute_order_and_the_xml_declaration() {
    let a = canonical(r#"<?xml version="1.0"?><r a="1" b="2"/>"#);
    let b = canonical(r#"<r b="2" a="1"/>"#);
    assert_eq!(a, b);
}

/// A byte-order mark in front of an expectation file is an encoding signature, not a character of
/// the document: left in place it lands inside the wrapper element the canonical form is built
/// with, which then does not parse and drops the comparison to its whitespace-collapsed fallback.
/// `date-061.out` is written that way, and six more `.out` files in the catalogue are.
#[test]
fn canonical_ignores_a_byte_order_mark_in_front_of_the_declaration() {
    let plain = canonical(r#"<?xml version="1.0" encoding="UTF-8"?><d><e>P6D</e></d>"#);
    assert_eq!(
        plain, "<d><e>P6D</e></d>",
        "the fixture parses to begin with"
    );
    for marked in [
        "\u{feff}<?xml version=\"1.0\" encoding=\"UTF-8\"?><d><e>P6D</e></d>",
        "\u{feff}<d><e>P6D</e></d>",
    ] {
        assert_eq!(canonical(marked), plain, "{marked:?}");
    }
}

/// `assert-xml` is decided against the result *tree*: the catalogue defines it over a
/// `method="xml"` serialization of the result, so what the stylesheet's own `xsl:output` made of
/// it — an `html`/`xhtml` method's inserted `<meta http-equiv="Content-Type">`, a `<!DOCTYPE>`,
/// a CDATA section — is not part of the comparison. `bug-1901`'s only difference from its
/// expectation is that inserted element.
#[test]
fn assert_xml_compares_the_result_tree_and_not_the_serialization() {
    let result = Tree::parse_str("<html><head><title>test</title></head></html>")
        .expect("well-formed fixture");
    let outcome: Result<XpathValue, ErrorInfo> =
        Ok(XpathValue::from(Sequence::from(result.root())));
    let tree = Tree::parse_str(
        "<assert-xml><![CDATA[<html><head><title>test</title></head></html>]]></assert-xml>",
    )
    .expect("well-formed fixture");
    let assertion = tree.document_element().expect("a document element");
    let serialized = concat!(
        "<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\">",
        "<title>test</title></head></html>"
    );
    assert_eq!(
        check_case(
            &assertion,
            Some(produced(serialized)),
            &outcome,
            Path::new(".")
        ),
        Outcome::Pass
    );

    // A real difference in the tree is still a failure: the rule narrows what is compared, not
    // how strictly.
    let other = Tree::parse_str("<html><head><title>wrong</title></head></html>")
        .expect("well-formed fixture");
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::from(Sequence::from(other.root())));
    assert!(matches!(
        check_case(
            &assertion,
            Some(produced(serialized)),
            &outcome,
            Path::new(".")
        ),
        Outcome::Fail(_)
    ));

    // A result tree whose top level begins and ends with the stylesheet's own layout whitespace
    // still matches an expectation the catalogue wrote indented inside its element: the framing
    // is trimmed off both sides, exactly as it was when the comparison read the serialization.
    // `select-2304` and the `tunnel-04*` family are that shape.
    let framed = Tree::parse_str("<w>\n<html><head><title>test</title></head></html>\n</w>")
        .expect("well-formed fixture");
    let element = framed.document_element().expect("a document element");
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::from(Sequence::from(element)));
    assert_eq!(
        check_case(
            &assertion,
            Some(produced(serialized)),
            &outcome,
            Path::new(".")
        ),
        Outcome::Pass
    );

    // With no result tree to walk — the raw-sequence shape a `<output tree="no"/>` asks for —
    // the serialized comparison is what is left, and it still decides the case.
    let raw: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    assert_eq!(
        check_case(
            &assertion,
            Some(produced("<html><head><title>test</title></head></html>")),
            &raw,
            Path::new(".")
        ),
        Outcome::Pass
    );
}

/// A `<param>` child of `<initial-template>` supplies the *entry point's* own parameter, and
/// the catalogue writes its name against a namespace bound on the `<param>` itself
/// (`initial-template-002`).
#[test]
fn an_entry_point_parameter_carries_its_namespace_and_tunnel_flag() {
    let tree = Tree::parse_str(
        r#"<initial-template name="param1">
             <param name="a" select="1234"/>
             <param xmlns:my="http://my.net/" name="my:b" tunnel="yes" select="999"/>
             <param name="c"/>
           </initial-template>"#,
    )
    .expect("well-formed fixture");
    let entry = tree.document_element().expect("a document element");
    let parameters = entry_point_parameters(&entry, Path::new("."));
    assert_eq!(
        parameters.len(),
        2,
        "a `<param>` with no `select` binds nothing"
    );
    assert_eq!(parameters[0].0, ExpandedName::new(None::<&str>, "a"));
    assert_eq!(
        crate::assertions::joined_string_value(&parameters[0].1).as_deref(),
        Some("1234")
    );
    assert!(!parameters[0].2, "no `tunnel` attribute is not a tunnel");
    assert_eq!(
        parameters[1].0,
        ExpandedName::new(Some("http://my.net/"), "b")
    );
    assert!(parameters[1].2, "`tunnel=\"yes\"` picks the tunnel channel");
}

/// The trace is line-oriented, so an engine message spanning several lines has to stay one
/// record — otherwise the `GAP` line for a case cannot be paired with its `SKIP` line.
#[test]
fn a_trace_line_holds_a_multi_line_detail_on_one_line() {
    let line = trace_line("output/output-001", "GAP", "first\nsecond\tthird\r\n");
    assert_eq!(line, "output/output-001\tGAP\tfirst second third  ");
    assert_eq!(line.lines().count(), 1);
}

#[test]
fn a_documented_gap_is_reported_by_name() {
    let tree = Tree::parse_str(
        r#"<dependencies><feature value="streaming" satisfied="true"/></dependencies>"#,
    )
    .expect("well-formed fixture");
    let root = tree.document_element().expect("a document element");
    assert_eq!(unsupported_dependency(&root).as_deref(), Some("streaming"));
}

#[test]
fn an_unsatisfied_gap_feature_is_no_obstacle() {
    let tree = Tree::parse_str(
        r#"<dependencies><feature value="streaming" satisfied="false"/></dependencies>"#,
    )
    .expect("well-formed fixture");
    let root = tree.document_element().expect("a document element");
    assert_eq!(unsupported_dependency(&root), None);
}

/// `serialization` is a capability this processor has: `oxixml_fn::xmlio` implements the
/// Serialization 3.1 parameters, and [`crate::xslt30`] adjudicates the assertion kinds that
/// read serialized output (`assert-serialization`, `serialization-matches`). A case that
/// declares the feature is therefore run, not gated — which is what makes the `output`,
/// `character-map` and `result-document` sets measurable rather than invisible.
#[test]
fn the_serialization_feature_is_not_a_gap() {
    let tree = Tree::parse_str(
        r#"<dependencies><feature value="serialization" satisfied="true"/></dependencies>"#,
    )
    .expect("well-formed fixture");
    let root = tree.document_element().expect("a document element");
    assert_eq!(unsupported_dependency(&root), None);
}

/// The value `system-property()` reports for `name`, read out of the processor this runner drives.
///
/// Written as a transformation rather than as a call into the engine's internals because that is
/// how the catalogue reads it: the `system-property` test set asks the question from inside a
/// stylesheet, so the answer this helper returns is the one those cases assert on.
fn system_property_answer(name: &str) -> String {
    let stylesheet = format!(
        r#"<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="3.0">
             <xsl:output method="text"/>
             <xsl:template match="/"><xsl:value-of select="system-property('{name}')"/></xsl:template>
           </xsl:stylesheet>"#
    );
    let compiled = Stylesheet::compile_str(&stylesheet).expect("a well-formed stylesheet compiles");
    let source = Tree::parse_str("<source/>").expect("well-formed fixture");
    let transformation = compiled
        .transform(&source.root())
        .expect("the transformation reads a property and writes nothing else");
    transformation.as_str().to_owned()
}

/// [`UNSUPPORTED_FEATURES`] is a claim about the processor, and `system-property()` is the same
/// claim made by the processor itself — so a name may appear in the list exactly when the engine
/// answers `no` for the property that asks the same question.
///
/// Disagreement is not a cosmetic inconsistency. A feature declared missing here gates every case
/// that depends on it out of the run, *and* leaves the cases that interrogate the property
/// directly (`system-property-018` asks for `xsl:supports-higher-order-functions`) failing against
/// an engine that says the opposite. This test is what stops the two drifting apart again: adding
/// a name back to the list, or flipping an answer in the engine's table, fails it.
///
/// Only the properties that answer `yes` or `no` can be paired this way, which is why `XSD_1.1` is
/// not among them: `xsl:xsd-version` reports a version, and the catalogue's feature asks about the
/// *schema* processor rather than about the type library the version names. That entry is settled
/// by measurement instead — see [`super::UNSUPPORTED_FEATURES`].
#[test]
fn the_gap_list_agrees_with_the_processor_it_describes() {
    // The catalogue's feature name, and the system property that asks the same question.
    let paired = [
        (
            "higher_order_functions",
            "xsl:supports-higher-order-functions",
        ),
        ("namespace_axis", "xsl:supports-namespace-axis"),
        ("streaming", "xsl:supports-streaming"),
        ("dynamic_evaluation", "xsl:supports-dynamic-evaluation"),
        ("schema_aware", "xsl:is-schema-aware"),
        (
            "backwards_compatibility",
            "xsl:supports-backwards-compatibility",
        ),
    ];
    for (feature, property) in paired {
        let answer = system_property_answer(property);
        assert!(
            answer == "yes" || answer == "no",
            "`{property}` answered `{answer}`, which is neither `yes` nor `no`"
        );
        let declared_missing = UNSUPPORTED_FEATURES.contains(&feature);
        assert_eq!(
            declared_missing,
            answer == "no",
            "the gap list says `{feature}` is {}, the engine says `{property}` = `{answer}`",
            if declared_missing {
                "missing"
            } else {
                "present"
            }
        );
    }
}

/// The consequence of the agreement above, at the point the catalogue's dependency filter reads
/// it: a case gated on either of the two features the engine claims is run, not skipped.
///
/// Both spellings matter — `satisfied="true"` is the ordinary form and `satisfied="false"`
/// describes a processor without the feature, which this one is not.
#[test]
fn the_two_features_the_engine_claims_are_not_gaps() {
    for feature in ["higher_order_functions", "namespace_axis"] {
        let tree = Tree::parse_str(&format!(
            r#"<dependencies><feature value="{feature}" satisfied="true"/></dependencies>"#
        ))
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(unsupported_dependency(&root), None, "{feature}");

        let tree = Tree::parse_str(&format!(
            r#"<dependencies><feature value="{feature}" satisfied="false"/></dependencies>"#
        ))
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            unsupported_dependency(&root).as_deref(),
            Some(format!("feature-present:{feature}").as_str()),
            "{feature}"
        );
    }
}

#[test]
fn a_pre_30_only_spec_is_reported_as_a_spec_gap() {
    let tree = Tree::parse_str(r#"<dependencies><spec value="XSLT20"/></dependencies>"#)
        .expect("well-formed fixture");
    let root = tree.document_element().expect("a document element");
    assert_eq!(
        unsupported_dependency(&root).as_deref(),
        Some("spec:XSLT20")
    );
}

/// Every skip this runner records names *why*, so that the dependency-gated ones (a gap the
/// catalogue itself declares) can be told apart from the ones that hide an unknown verdict.
#[test]
fn a_skipped_case_carries_a_specific_reason() {
    let environments = std::collections::BTreeMap::new();
    let directory = Path::new(".");
    for (fixture, expected) in [
        (
            r#"<test-case><dependencies><feature value="streaming"/></dependencies></test-case>"#,
            "dependency:streaming",
        ),
        (r#"<test-case/>"#, "no-test"),
        (
            r#"<test-case><test><stylesheet file="x.xsl"/></test></test-case>"#,
            "no-result",
        ),
        (
            r#"<test-case><test/><result><assert>true()</assert></result></test-case>"#,
            "no-stylesheet",
        ),
        (
            r#"<test-case><test><stylesheet/></test><result><assert>true()</assert></result></test-case>"#,
            "no-stylesheet-file",
        ),
        (
            r#"<test-case><test><stylesheet file="absent.xsl"/></test><result><assert>true()</assert></result></test-case>"#,
            "unreadable-stylesheet",
        ),
    ] {
        let tree = Tree::parse_str(fixture).expect("well-formed fixture");
        let case = tree.document_element().expect("a document element");
        assert_eq!(
            run_case(&case, "fixture", directory, &environments, unhurried()),
            Outcome::Skip(expected.to_owned()),
            "{fixture}"
        );
    }
}

/// `assert-serialization` compares the serialized result as text: the framing whitespace an
/// inline expectation carries from the catalogue's own indentation is removed, and nothing
/// else is — an attribute written in a different order is a real difference here, where
/// `assert-xml` would forgive it.
#[test]
fn assert_serialization_compares_the_serialized_text() {
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    // The catalogue writes an inline expectation as CDATA (or escaped text), indented inside
    // its own element — the framing whitespace this fixture reproduces on purpose.
    let tree = Tree::parse_str(
        "<assert-serialization>\n   <![CDATA[<r a=\"1\" b=\"2\"/>]]>\n</assert-serialization>",
    )
    .expect("well-formed fixture");
    let assertion = tree.document_element().expect("a document element");
    assert_eq!(
        check_case(
            &assertion,
            Some(produced(r#"<r a="1" b="2"/>"#)),
            &outcome,
            Path::new(".")
        ),
        Outcome::Pass
    );
    assert!(matches!(
        check_case(
            &assertion,
            Some(produced(r#"<r b="2" a="1"/>"#)),
            &outcome,
            Path::new(".")
        ),
        Outcome::Fail(_)
    ));
    // With no result tree there is nothing serialized to compare, and the skip says so.
    assert_eq!(
        check_case(&assertion, None, &outcome, Path::new(".")),
        Outcome::Skip("no-result-tree:assert-serialization".to_owned())
    );
}

/// `assert-serialization-error` is graded by the code, whenever the processor raised it — a
/// processor that rejects an impossible `xsl:output` while compiling has still reported the
/// fault the case names. Serializing without complaint is the failure.
#[test]
fn assert_serialization_error_accepts_the_error_however_early_it_was_raised() {
    let tree = Tree::parse_str(r#"<assert-serialization-error code="SEPM0016"/>"#)
        .expect("well-formed fixture");
    let assertion = tree.document_element().expect("a document element");

    let raised: Result<XpathValue, ErrorInfo> = Err(ErrorInfo::bare("SEPM0016", "bad value"));
    assert_eq!(
        check_case(&assertion, None, &raised, Path::new(".")),
        Outcome::Pass
    );

    let other: Result<XpathValue, ErrorInfo> = Err(ErrorInfo::bare("XTSE0020", "elsewhere"));
    assert!(matches!(
        check_case(&assertion, None, &other, Path::new(".")),
        Outcome::Fail(_)
    ));

    let clean: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    assert!(matches!(
        check_case(&assertion, Some(produced("<out/>")), &clean, Path::new(".")),
        Outcome::Fail(_)
    ));
}

/// `assert-result-document` decides its nested assertion against the *secondary* result, both
/// as text (`assert-xml`) and as a document to evaluate XPath over (`assert`).
#[test]
fn assert_result_document_checks_the_named_secondary_output() {
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    let secondary = [(
        "file:///suite/tests/out.xml".to_owned(),
        "<root>account</root>".to_owned(),
    )];
    let written = Produced {
        serialized: "<principal/>",
        messages: &[],
        secondary: &secondary,
    };
    for body in [
        "<assert-xml><![CDATA[<root>account</root>]]></assert-xml>",
        "<assert>/root = \"account\"</assert>",
    ] {
        let tree = Tree::parse_str(&format!(
            r#"<assert-result-document uri="out.xml">{body}</assert-result-document>"#
        ))
        .expect("well-formed fixture");
        let assertion = tree.document_element().expect("a document element");
        assert_eq!(
            check_case(&assertion, Some(written), &outcome, Path::new(".")),
            Outcome::Pass,
            "{body}"
        );
    }

    // A URI nothing was written to is a failure, and the message says what *was* written.
    let tree = Tree::parse_str(
        r#"<assert-result-document uri="absent.xml"><assert>true()</assert></assert-result-document>"#,
    )
    .expect("well-formed fixture");
    let assertion = tree.document_element().expect("a document element");
    let Outcome::Fail(reason) = check_case(&assertion, Some(written), &outcome, Path::new("."))
    else {
        panic!("a missing result document is a failure, not a skip");
    };
    assert!(reason.contains("out.xml"), "{reason}");
}

#[test]
fn a_result_document_uri_matches_by_its_trailing_segments() {
    assert!(names_result_document("out.xml", "out.xml"));
    assert!(names_result_document("file:///a/b/out.xml", "out.xml"));
    assert!(names_result_document(
        "file:///a/b/sub/out.xml",
        "sub/out.xml"
    ));
    assert!(!names_result_document("file:///a/b/other.xml", "out.xml"));
    assert!(
        !names_result_document("file:///a/b/notout.xml", "out.xml"),
        "a suffix must start at a path segment"
    );
}

/// An `all-of` whose one undecidable leaf is what stopped the case says which leaf that was,
/// rather than collapsing into a generic "unsupported assertion kind".
#[test]
fn a_connective_propagates_the_reason_of_the_leaf_it_could_not_decide() {
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    let tree = Tree::parse_str(r#"<all-of><assert-empty/><assert-warning/></all-of>"#)
        .expect("well-formed fixture");
    let assertion = tree.document_element().expect("a document element");
    assert_eq!(
        check_case(&assertion, None, &outcome, Path::new(".")),
        Outcome::Skip("unsupported-assertion:assert-warning".to_owned())
    );
}

/// A whole family of cases shares one `<environment>` and states the stylesheet there once,
/// leaving each `<test>` to name only an entry point — `regex-070a`'s shape. Taking the
/// environment's module is what turns those from `no-stylesheet` skips into real verdicts.
///
/// A `role="secondary"` module is *not* one: it is a module the principal one imports, and
/// compiling it as the entry point would run a different stylesheet than the case describes.
#[test]
fn an_environment_supplies_the_stylesheet_a_test_omits() {
    let directory = Path::new("/suite/tests");
    let cases: &[(&str, &str, Result<&str, &str>)] = &[
        (
            r#"<environment><stylesheet file="regex-070.xsl"/></environment>"#,
            r#"<test><initial-mode name="a"/></test>"#,
            Ok("/suite/tests/regex-070.xsl"),
        ),
        // The `<test>`'s own module always wins over the environment's.
        (
            r#"<environment><stylesheet file="shared.xsl"/></environment>"#,
            r#"<test><stylesheet file="own.xsl"/></test>"#,
            Ok("/suite/tests/own.xsl"),
        ),
        // A secondary module on the `<test>` is passed over for the principal one after it.
        (
            r#"<environment/>"#,
            r#"<test><stylesheet role="secondary" file="lib.xsl"/><stylesheet file="main.xsl"/></test>"#,
            Ok("/suite/tests/main.xsl"),
        ),
        // A secondary module on the environment is not a fallback at all.
        (
            r#"<environment><stylesheet role="secondary" file="lib.xsl"/></environment>"#,
            r#"<test/>"#,
            Err("no-stylesheet"),
        ),
        // A package is a stylesheet module, so a principal one is the entry point to compile.
        (
            r#"<environment/>"#,
            r#"<test><package role="principal" file="p.xsl"/></test>"#,
            Ok("/suite/tests/p.xsl"),
        ),
        (r#"<environment/>"#, r#"<test/>"#, Err("no-stylesheet")),
    ];
    for (environment_text, test_text, expected) in cases {
        let tree = Tree::parse_str(&format!("<w>{environment_text}{test_text}</w>"))
            .expect("well-formed fixture");
        let wrapper = tree.document_element().expect("a document element");
        let environment_element =
            find_child(&wrapper, "environment").expect("the fixture's environment");
        let test = find_child(&wrapper, "test").expect("the fixture's test");
        let environment = Environment::parse(&environment_element, directory);
        let resolved = principal_stylesheet(&test, &environment, directory);
        match expected {
            Ok(path) => assert_eq!(
                resolved,
                Ok(Module::File(PathBuf::from(path))),
                "{environment_text}{test_text}"
            ),
            Err(reason) => assert_eq!(
                resolved,
                Err((*reason).to_owned()),
                "{environment_text}{test_text}"
            ),
        }
    }
}

/// A `<source defines-stylesheet="true">` names no file to compile: the module is an element of
/// the source document, which is a different kind of answer and not the absence of one.
#[test]
fn a_defines_stylesheet_source_names_an_embedded_module() {
    let directory = Path::new("/suite/tests");
    let tree = Tree::parse_str(
        r#"<w><environment><source role="." defines-stylesheet="true"/></environment><test/></w>"#,
    )
    .expect("well-formed fixture");
    let wrapper = tree.document_element().expect("a document element");
    let environment_element =
        find_child(&wrapper, "environment").expect("the fixture's environment");
    let test = find_child(&wrapper, "test").expect("the fixture's test");
    let environment = Environment::parse(&environment_element, directory);
    assert!(environment.defines_stylesheet);
    assert_eq!(
        principal_stylesheet(&test, &environment, directory),
        Ok(Module::Embedded)
    );

    // A `<test>` that names its own module keeps it: `embedded-stylesheet-016` shares an
    // environment with the embedded cases and compiles a file of its own.
    let tree = Tree::parse_str(
        r#"<w><environment><source role="." defines-stylesheet="true"/></environment>
             <test><stylesheet file="own.xsl"/></test></w>"#,
    )
    .expect("well-formed fixture");
    let wrapper = tree.document_element().expect("a document element");
    let environment_element =
        find_child(&wrapper, "environment").expect("the fixture's environment");
    let test = find_child(&wrapper, "test").expect("the fixture's test");
    let environment = Environment::parse(&environment_element, directory);
    assert_eq!(
        principal_stylesheet(&test, &environment, directory),
        Ok(Module::File(directory.join("own.xsl")))
    );
}

/// A `<source role="." file="…">` is a document retrieved from a URI, and the tree has to carry
/// it: the base URI every relative reference inside the document resolves against — an
/// `xml:base`, an `NDATA` entity's system identifier — is that URI and nothing else.
///
/// An inline `<content>` source names no file, so it has no URI to carry, and the options it is
/// parsed with say so rather than inventing one.
#[test]
fn a_source_document_carries_the_uri_it_was_read_from() {
    let directory = std::env::temp_dir().join("oxixml-xslt30-source-uri");
    std::fs::create_dir_all(&directory).expect("a writable temporary directory");
    let path = directory.join("source.xml");
    std::fs::write(&path, "<doc/>").expect("the fixture is writable");

    let tree =
        Tree::parse_str(r#"<environment><source role="." file="source.xml"/></environment>"#)
            .expect("well-formed fixture");
    let element = tree.document_element().expect("a document element");
    let environment = Environment::parse(&element, &directory);
    assert_eq!(environment.source.as_deref(), Some("<doc/>"));
    assert_eq!(
        environment.source_uri.as_deref(),
        Some(format!("file://{}", path.display()).as_str())
    );
    let parsed =
        Tree::parse_str_with("<doc/>", source_options(&environment)).expect("the fixture parses");
    assert_eq!(
        parsed.document_uri(),
        environment.source_uri.as_deref(),
        "the document URI reaches the tree"
    );

    let tree = Tree::parse_str(
        r#"<environment><source role="."><content>&lt;doc/&gt;</content></source></environment>"#,
    )
    .expect("well-formed fixture");
    let element = tree.document_element().expect("a document element");
    let inline = Environment::parse(&element, &directory);
    assert_eq!(inline.source.as_deref(), Some("<doc/>"));
    assert_eq!(inline.source_uri, None);
    let parsed =
        Tree::parse_str_with("<doc/>", source_options(&inline)).expect("the fixture parses");
    assert_eq!(parsed.document_uri(), None);

    std::fs::remove_file(&path).ok();
}

/// A `<source role="." select="…">` starts the transformation at a node *inside* the document.
/// `mode-1105` names `/doc` so that a rule matching `doc` in the initial mode is the first to
/// fire; without it the transformation starts at the document node and runs different rules.
/// An expression naming no single node leaves the document node in place rather than declining
/// the case.
#[test]
fn a_source_select_names_the_node_the_transformation_starts_from() {
    let tree = Tree::parse_str("<doc><str>brown-fox</str></doc>").expect("well-formed fixture");
    let root = tree.root();
    let directory = Path::new(".");
    assert_eq!(
        initial_context_node("/doc", &root, directory)
            .node_name()
            .map(|name| name.local_name().to_owned()),
        Some("doc".to_owned())
    );
    assert_eq!(
        initial_context_node("/doc/str/text()", &root, directory).string_value(),
        "brown-fox"
    );
    for absent in [
        "/doc/missing",
        "(/doc, /doc/str)",
        "'not a node'",
        "'unterminated",
    ] {
        assert_eq!(
            initial_context_node(absent, &root, directory).node_kind(),
            NodeKind::Document,
            "{absent}"
        );
    }

    // And the `select` reaches the environment the driver builds from the catalogue.
    let tree = Tree::parse_str(
        r#"<environment><source role="." select="/doc"><content>&lt;doc/&gt;</content></source></environment>"#,
    )
    .expect("well-formed fixture");
    let element = tree.document_element().expect("a document element");
    let environment = Environment::parse(&element, directory);
    assert_eq!(environment.source_select.as_deref(), Some("/doc"));
    assert_eq!(environment.source.as_deref(), Some("<doc/>"));
}

/// The `unparsed-entity-uri` set asserts the whole path from a `<!ENTITY … NDATA …>` declaration
/// in the source document's internal subset to the *absolute* URI `fn:unparsed-entity-uri`
/// answers: `unparsed-entity-01` requires `unparsed-entity-uri('hatch-pic')` — declared
/// `SYSTEM "../grafix/OpenHatch.gif"` — to end in `/fn/grafix/OpenHatch.gif`, which is only
/// true when the system identifier was resolved against the document's own URI.
///
/// Three cases of the set are not named here because they read the property back through a
/// *copied* document node that does not carry it: `-06` copies with `<xsl:copy/>`, `-07` with
/// `fn:copy-of()` and `-08` with `fn:snapshot()`. `<xsl:copy-of/>` does carry it, which is why
/// `-05` is checked here — the three are a processor gap, not a gap in what this drives.
/// `-09`/`-10` need higher-order functions and `-50` the external DTD subset, all of which the
/// runner declines earlier.
///
/// Needs the provisioned suite; decides nothing when it is not there.
#[test]
fn the_unparsed_entity_cases_pass_against_the_real_catalogue() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    let path = root.join("tests/fn/unparsed-entity-uri/_unparsed-entity-uri-test-set.xml");
    if !path.exists() {
        return;
    }
    let mut unexpected = Vec::new();
    for case in [
        "unparsed-entity-01",
        "unparsed-entity-02",
        "unparsed-entity-03",
        "unparsed-entity-04",
        "unparsed-entity-05",
        "unparsed-entity-11",
    ] {
        let path = path.clone();
        let worker = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || verdict_of(&path, case))
            .expect("a worker thread");
        let outcome = worker.join().expect("the case ran to completion");
        if outcome != Outcome::Pass {
            unexpected.push(format!("{case}: {outcome:?}"));
        }
    }
    assert!(unexpected.is_empty(), "{}", unexpected.join("; "));
}

/// `<initial-function>` is the third entry point, and `error-0041a` is the only case in an
/// allowed test set that uses it: it names a function no stylesheet declares and requires
/// `XTDE0041`. Reaching that error means the entry point was really selected — a runner that
/// ignores the element transforms through the default entry point instead and the case succeeds
/// where it must fail.
///
/// Needs the provisioned suite; decides nothing when it is not there.
#[test]
fn the_initial_function_entry_point_is_selected() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    let path = root.join("tests/misc/error/_error-test-set.xml");
    if !path.exists() {
        return;
    }
    let worker = std::thread::Builder::new()
        .stack_size(WORKER_STACK)
        .spawn(move || verdict_of(&path, "error-0041a"))
        .expect("a worker thread");
    assert_eq!(
        worker.join().expect("the case ran to completion"),
        Outcome::Pass
    );
}

/// The raw-result path and the `<initial-function>` argument list end to end: a transformation
/// started at a stylesheet *function* called with two arguments, whose result is a sequence of
/// two decimals, asserted with `deep-equal` under a `<output tree="no" result-var="result"/>`.
///
/// Three things are pinned at once, and each fails distinctly without its half. The arguments
/// have to be supplied positionally *and* counted, because how many there are chooses the arity
/// — an empty list would select no function and raise `XTDE0041`. And the result has to reach
/// the assertion as a sequence: bound as a result tree instead, `deep-equal` compares a document
/// node against two decimals and cannot hold.
///
/// Written here rather than driven from the catalogue because `error-0041a` — the one allowed
/// case that names an `<initial-function>` — supplies no arguments and asserts only the error,
/// the `initial-function` test set is not one this runner attempts, and `initial-template-004`
/// (the one allowed case writing `result-var`) is waiting on the engine to record the sequence
/// for the *template* entry point as it already does for this one.
#[test]
fn an_initial_function_is_called_with_its_arguments_and_answers_a_sequence() {
    let directory = std::env::temp_dir().join("oxixml-xslt30-raw-result");
    std::fs::create_dir_all(&directory).expect("a writable temporary directory");
    std::fs::write(
        directory.join("raw.xsl"),
        r#"<?xml version="1.0"?>
           <xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                           xmlns:xs="http://www.w3.org/2001/XMLSchema"
                           xmlns:f="urn:oxixml:raw" version="3.0">
             <xsl:function name="f:pair" as="xs:decimal*" visibility="public">
               <xsl:param name="a" as="xs:decimal"/>
               <xsl:param name="b" as="xs:decimal"/>
               <xsl:sequence select="($a, $b)"/>
             </xsl:function>
           </xsl:stylesheet>"#,
    )
    .expect("the stylesheet is writable");
    let set = directory.join("_raw-test-set.xml");
    std::fs::write(
        &set,
        r#"<?xml version="1.0"?>
           <test-set name="raw">
             <test-case name="raw-sequence">
               <test>
                 <stylesheet file="raw.xsl"/>
                 <initial-function xmlns:f="urn:oxixml:raw" name="f:pair">
                   <param select="1.5"/>
                   <param select="2.5"/>
                 </initial-function>
                 <output tree="no" result-var="result"/>
               </test>
               <result><assert>deep-equal($result, (1.5, 2.5))</assert></result>
             </test-case>
             <test-case name="raw-sequence-unnamed-variable">
               <test>
                 <stylesheet file="raw.xsl"/>
                 <initial-function xmlns:f="urn:oxixml:raw" name="f:pair">
                   <param select="1.5"/>
                   <param select="2.5"/>
                 </initial-function>
                 <output tree="no" result-var="other"/>
               </test>
               <result><assert>deep-equal($other, (1.5, 2.5))</assert></result>
             </test-case>
             <test-case name="raw-sequence-wrong-arity">
               <test>
                 <stylesheet file="raw.xsl"/>
                 <initial-function xmlns:f="urn:oxixml:raw" name="f:pair">
                   <param select="1.5"/>
                 </initial-function>
               </test>
               <result><error code="XTDE0041"/></result>
             </test-case>
           </test-set>"#,
    )
    .expect("the test set is writable");

    assert_eq!(verdict_of(&set, "raw-sequence"), Outcome::Pass);
    assert_eq!(
        verdict_of(&set, "raw-sequence-wrong-arity"),
        Outcome::Pass,
        "how many arguments were supplied chooses the arity"
    );
    assert_eq!(
        verdict_of(&set, "raw-sequence-unnamed-variable"),
        Outcome::Skip("unsupported-result-var:other".to_owned()),
        "a variable the assertion checker cannot bind leaves the case undecided"
    );

    std::fs::remove_file(&set).ok();
    std::fs::remove_file(directory.join("raw.xsl")).ok();
}

/// `<output tree="no">` decides what the assertions see: the raw result sequence, under the
/// `$result` the catalogue's convention names it by, whether or not a `result-var` also names a
/// binding. A `result-var` naming any *other* variable leaves the case undecided under its own
/// reason rather than answering it against a binding it did not ask for.
#[test]
fn an_output_element_says_whether_the_raw_result_sequence_is_wanted() {
    let cases: &[(&str, Result<bool, &str>)] = &[
        (r#"<test/>"#, Ok(false)),
        (r#"<test><output serialize="yes"/></test>"#, Ok(false)),
        (
            r#"<test><output tree="no" serialize="no"/></test>"#,
            Ok(true),
        ),
        (
            r#"<test><output tree="no" result-var="result"/></test>"#,
            Ok(true),
        ),
        // A result tree is not what the case asked for, and `$other` is a name the assertion
        // checker has no way to bind.
        (
            r#"<test><output tree="no" result-var="other"/></test>"#,
            Err("unsupported-result-var:other"),
        ),
    ];
    for (fixture, expected) in cases {
        let tree = Tree::parse_str(fixture).expect("well-formed fixture");
        let test = tree.document_element().expect("a document element");
        assert_eq!(
            raw_result_wanted(&test)
                .as_ref()
                .copied()
                .map_err(String::as_str),
            *expected,
            "{fixture}"
        );
    }
}

/// A `<param>`'s `select` is XPath, not a literal: reading it as text would bind the *string*
/// `"true()"` where the catalogue means the boolean, which is precisely what a static
/// parameter feeding `use-when` or a shadow attribute cannot survive.
#[test]
fn a_selector_is_evaluated_as_xpath_rather_than_read_as_a_literal() {
    let directory = Path::new(".");
    assert_eq!(
        selector_value("true()", None, directory).to_string(),
        "true"
    );
    assert_eq!(
        selector_value("'f:square1'", None, directory).to_string(),
        "f:square1"
    );
    assert_eq!(selector_value("1 to 5", None, directory).iter().len(), 5);
    assert_eq!(selector_value("2 + 3", None, directory).to_string(), "5");
    // An expression the engine cannot compile still binds the literal it looks like, so a gap
    // in the XPath engine cannot leave a parameter unsupplied.
    assert_eq!(
        selector_value("'unterminated", None, directory).to_string(),
        "'unterminated"
    );
}

/// The `static="yes"` a `<param>` carries decides *which* channel supplies it: a static
/// parameter has to reach the compiler, an ordinary one the transformation.
#[test]
fn read_parameter_records_whether_a_parameter_is_static() {
    let tree = Tree::parse_str(
        r#"<test><param name="a" select="1"/><param name="b" select="2" static="yes"/><param name="c"/></test>"#,
    )
    .expect("well-formed fixture");
    let test = tree.document_element().expect("a document element");
    let parameters: Vec<CaseParameter> = element_children(&test)
        .iter()
        .filter_map(read_parameter)
        .collect();
    assert_eq!(
        parameters.len(),
        2,
        "a `<param>` with no `select` binds nothing"
    );
    assert_eq!(parameters[0].name, "a");
    assert!(!parameters[0].is_static);
    assert_eq!(parameters[1].name, "b");
    assert!(parameters[1].is_static);
}

/// `result-document-0102` asserts `ends-with(base-uri(/), "/out/second.xml")` against a
/// *secondary* result: re-parsing the serialization has to keep the URI the engine wrote it
/// to, or every base-URI assertion over one is decided against an absent base URI.
#[test]
fn a_secondary_result_document_keeps_the_uri_it_was_written_to() {
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    let secondary = [(
        "file:///suite/tests/out/second.xml".to_owned(),
        r#"<two><k>Kilo</k><l xml:base="in/third.xml">Lima</l></two>"#.to_owned(),
    )];
    let written = Produced {
        serialized: "<principal/>",
        messages: &[],
        secondary: &secondary,
    };
    for body in [
        r#"<assert>ends-with(base-uri(/), "/out/second.xml")</assert>"#,
        r#"<assert>ends-with(base-uri(/*), "/out/second.xml")</assert>"#,
        r#"<assert>ends-with(base-uri(/*/k), "/out/second.xml")</assert>"#,
        // An `xml:base` inside the document resolves against that same URI.
        r#"<assert>ends-with(base-uri(/*/l), "/out/in/third.xml")</assert>"#,
    ] {
        let tree = Tree::parse_str(&format!(
            r#"<assert-result-document uri="out/second.xml">{body}</assert-result-document>"#
        ))
        .expect("well-formed fixture");
        let assertion = tree.document_element().expect("a document element");
        assert_eq!(
            check_case(&assertion, Some(written), &outcome, Path::new(".")),
            Outcome::Pass,
            "{body}"
        );
    }
}

/// The three shapes this runner learned to drive, each pinned against the catalogue case that
/// motivated it: a stylesheet named only on the environment (`regex-070a`), a static parameter
/// that a shadow attribute reads (`date-094g`, whose whole stylesheet is
/// `<xsl:value-of _select="{$test-case}"/>`), and a secondary result document's base URI
/// (`result-document-0102`).
///
/// Needs the provisioned suite; decides nothing when it is not there.
#[test]
fn the_newly_driven_shapes_pass_against_the_real_catalogue() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    for (file, case) in [
        ("tests/misc/regex/_regex-test-set.xml", "regex-070a"),
        ("tests/type/date/_date-test-set.xml", "date-094g"),
        (
            "tests/insn/result-document/_result-document-test-set.xml",
            "result-document-0102",
        ),
        // A static parameter read by `use-when` rather than by a shadow attribute.
        ("tests/insn/copy/_copy-test-set.xml", "copy-0614"),
    ] {
        let path = root.join(file);
        if !path.exists() {
            continue;
        }
        let worker = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || verdict_of(&path, case))
            .expect("a worker thread");
        assert_eq!(
            worker.join().expect("the case ran to completion"),
            Outcome::Pass,
            "{case}"
        );
    }
}

/// The entry point's own parameters, pinned against the catalogue cases that supply them:
/// `initial-template-002` binds a plain and a tunnel parameter, `-003` binds a stylesheet
/// parameter and an entry-point parameter of the *same* name to different values and expects
/// both to be visible where each applies. Without the entry-point channel the initial
/// template's required `xsl:param` is unsupplied and the transformation raises `XTDE0700`
/// instead of running — which `-902` (whose `<param>` really is missing) still expects, and
/// `-901` still expects `XTDE0040` for a template that does not exist, so the two error cases
/// are here to pin that supplying the channel did not make the errors unreachable.
///
/// Needs the provisioned suite; decides nothing when it is not there.
#[test]
fn the_initial_template_parameter_cases_pass_against_the_real_catalogue() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    let path = root.join("tests/misc/initial-template/_initial-template-test-set.xml");
    if !path.exists() {
        return;
    }
    for case in [
        "initial-template-002",
        "initial-template-003",
        "initial-template-901",
        "initial-template-902",
    ] {
        let path = path.clone();
        let worker = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || verdict_of(&path, case))
            .expect("a worker thread");
        assert_eq!(
            worker.join().expect("the case ran to completion"),
            Outcome::Pass,
            "{case}"
        );
    }
}

/// A `<feature satisfied="false">` names a feature the case wants *absent*: it describes what a
/// processor lacking it does, so a processor that has the feature is out of scope rather than
/// wrong. The complement is deliberately not the rule — a feature named in neither list is
/// unanswered, and answering it "present" would decline cases this processor passes.
#[test]
fn a_feature_a_case_wants_absent_declines_it_only_when_this_processor_has_it() {
    let cases: &[(&str, Option<&str>)] = &[
        // `format-number-069b` and the five `disabling_output_escaping` cases.
        (
            r#"<feature value="XPath_3.1" satisfied="false"/>"#,
            Some("feature-present:XPath_3.1"),
        ),
        (
            r#"<feature value="disabling_output_escaping" satisfied="false"/>"#,
            Some("feature-present:disabling_output_escaping"),
        ),
        // Having the feature is the whole point of the `satisfied="true"` direction, which stays
        // exactly as it was: not a gap, so not a skip.
        (r#"<feature value="XPath_3.1" satisfied="true"/>"#, None),
        (r#"<feature value="XPath_3.1"/>"#, None),
        // A feature this processor does not have, wanted absent, is no obstacle either way.
        (r#"<feature value="streaming" satisfied="false"/>"#, None),
        // `catalog-005b`/`catalog-006b` write this one and pass: `XML_1.1` is in neither list, so
        // the gate must leave it alone rather than infer that an unlisted feature is present.
        (r#"<feature value="XML_1.1" satisfied="false"/>"#, None),
    ];
    for (gate, expected) in cases {
        let tree = Tree::parse_str(&format!("<dependencies>{gate}</dependencies>"))
            .expect("well-formed fixture");
        let dependencies = tree.document_element().expect("a document element");
        assert_eq!(
            unsupported_dependency(&dependencies).as_deref(),
            *expected,
            "{gate}"
        );
    }
}

/// `<default_html_version>` asks which HTML version the `html`/`xhtml` methods serialize as when
/// the stylesheet states none, and the catalogue gates both directions on the answer — the same
/// shape as `year_component_values`.
#[test]
fn the_default_html_version_gate_admits_only_the_half_that_matches_this_processor() {
    let cases: &[(&str, Option<&str>)] = &[
        (r#"<default_html_version value="5"/>"#, None),
        (
            r#"<default_html_version value="4"/>"#,
            Some("default_html_version:4"),
        ),
        (
            r#"<default_html_version value="5" satisfied="false"/>"#,
            Some("default_html_version:5"),
        ),
        (
            r#"<default_html_version value="4" satisfied="false"/>"#,
            None,
        ),
    ];
    for (gate, expected) in cases {
        let tree = Tree::parse_str(&format!("<dependencies>{gate}</dependencies>"))
            .expect("well-formed fixture");
        let dependencies = tree.document_element().expect("a document element");
        assert_eq!(
            unsupported_dependency(&dependencies).as_deref(),
            *expected,
            "{gate}"
        );
    }
}

/// `<initial-mode>`'s own `<param>` children, pinned against the catalogue case that supplies
/// them: `initial-mode-004` applies the `flobble` mode to the source document and binds `a` and a
/// tunnelled `my:b` that the matching template rule declares as its own `xsl:param`s. Its
/// paramless siblings are checked beside it so that opening the channel is shown not to disturb
/// the entry point that does not use it.
///
/// Needs the provisioned suite; decides nothing when it is not there.
#[test]
fn the_initial_mode_parameter_case_passes_against_the_real_catalogue() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    let path = root.join("tests/misc/initial-mode/_initial-mode-test-set.xml");
    if !path.exists() {
        return;
    }
    for case in [
        "initial-mode-004",
        "initial-mode-001",
        "initial-mode-002",
        "initial-mode-003",
        "initial-mode-005",
    ] {
        let path = path.clone();
        let worker = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || verdict_of(&path, case))
            .expect("a worker thread");
        assert_eq!(
            worker.join().expect("the case ran to completion"),
            Outcome::Pass,
            "{case}"
        );
    }
}

/// The two dependency gates added for cases this processor is not the audience of, driven through
/// the real catalogue so that the *label* the tally carries is pinned and not only the
/// disposition: `output-0195` and `output-0195a` serialize one stylesheet under a bare
/// `<xsl:output method="html"/>` and disagree only about the default HTML version, and
/// `format-number-069b` asks for the error a processor without XPath 3.1 raises.
///
/// Needs the provisioned suite; decides nothing when it is not there.
#[test]
fn the_out_of_audience_cases_are_declined_by_name_against_the_real_catalogue() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    let expectations: [(&'static str, &'static str, Outcome); 5] = [
        (
            "tests/decl/output/_output-test-set.xml",
            "output-0195",
            Outcome::Skip("dependency:default_html_version:4".to_owned()),
        ),
        (
            "tests/decl/output/_output-test-set.xml",
            "output-0195a",
            Outcome::Pass,
        ),
        (
            "tests/decl/output/_output-test-set.xml",
            "output-0195b",
            Outcome::Pass,
        ),
        (
            "tests/fn/format-number/_format-number-test-set.xml",
            "format-number-069b",
            Outcome::Skip("dependency:feature-present:XPath_3.1".to_owned()),
        ),
        (
            "tests/misc/error/_error-test-set.xml",
            "error-1620b",
            Outcome::Skip("dependency:feature-present:disabling_output_escaping".to_owned()),
        ),
    ];
    for (file, case, expected) in expectations {
        let path = root.join(file);
        if !path.exists() {
            continue;
        }
        let worker = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || verdict_of(&path, case))
            .expect("a worker thread");
        assert_eq!(
            worker.join().expect("the case ran to completion"),
            expected,
            "{case}"
        );
    }
}

#[test]
fn unavailable_when_env_is_unset() {
    if std::env::var_os(env::XML_SUITES_DIR_VAR).is_some() {
        return;
    }
    let reports = run();
    assert_eq!(reports.len(), 1);
    assert!(!reports[0].is_available());
}

/// Streamability analysis, asked for as a capability of its own. A `<posture-and-sweep>` case
/// carries no stylesheet at all — it hands the processor an XPath expression and a context
/// posture — so a processor that performs no streamability analysis has nothing to answer with,
/// whichever way the dependency is written.
#[test]
fn a_streamability_assessment_is_a_gap_in_both_directions() {
    for satisfied in ["true", "false"] {
        let tree = Tree::parse_str(&format!(
            r#"<dependencies><sweep_and_posture satisfied="{satisfied}"
                 value="supports-sweep-and-posture-assessments"/></dependencies>"#
        ))
        .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(
            unsupported_dependency(&root).as_deref(),
            Some("sweep_and_posture:supports-sweep-and-posture-assessments"),
            "satisfied={satisfied}"
        );
    }
}

/// A capability pinned to a version says *which* version in its label, for the ones this
/// processor declines: it implements one version of each and does not offer to behave as another,
/// so a tally that merged them into one bare line could not say which version was asked for.
#[test]
fn a_version_pinned_capability_names_the_version_it_was_asked_for() {
    let expectations = [
        (r#"<xsd-version value="1.1"/>"#, "xsd-version:1.1"),
        (
            r#"<unicode-normalization-form value="FULLY-NORMALIZED"/>"#,
            "unicode-normalization-form:FULLY-NORMALIZED",
        ),
        // No value to carry: the bare category is all there is to say.
        (r#"<calendar/>"#, "calendar"),
    ];
    for (gate, expected) in expectations {
        let tree = Tree::parse_str(&format!("<dependencies>{gate}</dependencies>"))
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(unsupported_dependency(&root).as_deref(), Some(expected));
    }
}

/// The Unicode version is the one version-pinned capability that is *answered*. The catalogue asks
/// about two — `unicode-90` (1460 cases) declares `9.0` and `regex-classes` (120) declares `6.0` —
/// and this workspace's character tables are newer than both, so all 1580 cases are adjudicated
/// rather than hidden behind one blanket label. A version this driver has not weighed stays
/// undecidable, and a case wanting a processor *older* than the version it names is declined,
/// because that is a processor this is not.
#[test]
fn a_unicode_version_the_processor_subsumes_is_answered_rather_than_declined() {
    let admitted = [
        r#"<unicode-version value="9.0" satisfied="true"/>"#,
        r#"<unicode-version value="6.0" satisfied="true"/>"#,
        r#"<unicode-version value=" 9.0 " satisfied="true"/>"#,
    ];
    for gate in admitted {
        let tree = Tree::parse_str(&format!("<dependencies>{gate}</dependencies>"))
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(unsupported_dependency(&root), None, "{gate}");
    }

    let declined = [
        // The inverted gate: a case written for a processor whose tables are *not* Unicode 9.0's.
        (
            r#"<unicode-version value="9.0" satisfied="false"/>"#,
            "unicode-version:9.0",
        ),
        // A version this driver has not weighed against the tables it ships.
        (r#"<unicode-version value="3.1"/>"#, "unicode-version:3.1"),
    ];
    for (gate, expected) in declined {
        let tree = Tree::parse_str(&format!("<dependencies>{gate}</dependencies>"))
            .expect("well-formed fixture");
        let root = tree.document_element().expect("a document element");
        assert_eq!(unsupported_dependency(&root).as_deref(), Some(expected));
    }
}

/// A package is a stylesheet module (XSLT 3.0 §3.5), so a `<package role="principal">` entry point
/// is compiled exactly like a `<stylesheet>` one and the engine gets to give its own answer.
#[test]
fn a_package_entry_point_is_compiled_like_a_stylesheet() {
    let expectations = [
        (
            r#"<test><package file="package-001.xsl" role="principal"/></test>"#,
            Ok("package-001.xsl"),
        ),
        (
            r#"<test><stylesheet file="s.xsl"/><package file="p.xsl" role="secondary"/></test>"#,
            Ok("s.xsl"),
        ),
        (
            r#"<test><package role="secondary" file="p.xsl"/></test>"#,
            Err("no-stylesheet".to_owned()),
        ),
        (
            r#"<test><package role="principal"/></test>"#,
            Err("no-package-file".to_owned()),
        ),
    ];
    let directory = Path::new("tests/decl/package");
    for (fixture, expected) in expectations {
        let tree = Tree::parse_str(fixture).expect("well-formed fixture");
        let test = tree.document_element().expect("a document element");
        let outcome = principal_stylesheet(&test, &Environment::default(), directory);
        match expected {
            Ok(file) => assert_eq!(outcome, Ok(Module::File(directory.join(file)))),
            Err(reason) => assert_eq!(outcome, Err(reason)),
        }
    }
}

/// [`SETS_VAR`] is a triage narrowing, and an accidentally empty setting must mean "the whole
/// catalogue" rather than "nothing" — a run that silently measures no cases is exactly the shape
/// of failure this driver exists to make impossible. (What a *deliberate* narrowing leaves out is
/// recorded as `catalogue:not-selected`, so its report says so too.)
#[test]
fn an_empty_set_selection_means_the_whole_catalogue() {
    assert_eq!(parse_set_selection(""), None);
    assert_eq!(parse_set_selection("  \t\n "), None);
    assert_eq!(parse_set_selection(",,"), None);
    assert_eq!(
        parse_set_selection("base-uri, initial-function"),
        Some(vec!["base-uri".to_owned(), "initial-function".to_owned()])
    );
    assert_eq!(
        parse_set_selection("axes\tposition"),
        Some(vec!["axes".to_owned(), "position".to_owned()])
    );
}

/// Writes `text` to a uniquely named file under the system temporary directory and answers its
/// path. The suffix keeps two tests in one process from colliding.
fn temporary_file(suffix: &str, text: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("oxixml-xslt30-{suffix}.xml"));
    std::fs::write(&path, text).expect("a writable temporary directory");
    path
}

/// A `<dependencies>` on the `<test-set>` governs every case in it — the catalogue schema says so
/// ("A dependency at the level of a test-set applies to all test cases in that test-set") and the
/// suite's reference runner reads the union `$test/(.|..)/dependencies`. Without it the ~2 600
/// streaming cases, whose only declaration of streaming is the one their test set makes, would be
/// compiled and counted as failures against a processor the catalogue never meant them for.
#[test]
fn a_test_set_dependency_governs_every_case_in_it() {
    let set = r#"<test-set xmlns="http://www.w3.org/2012/10/xslt-test-catalog" name="fixture">
          <dependencies><feature value="streaming"/></dependencies>
          <test-case name="one">
             <test><stylesheet file="no-such-file.xsl"/></test>
             <result><assert-xml><![CDATA[<out/>]]></assert-xml></result>
          </test-case>
          <test-case name="two">
             <dependencies><spec value="XSLT30+"/></dependencies>
             <test><stylesheet file="no-such-file.xsl"/></test>
             <result><assert-xml><![CDATA[<out/>]]></assert-xml></result>
          </test-case>
       </test-set>"#;
    let path = temporary_file("set-level-dependency", set);
    let mut report = SuiteReport::new("fixture");
    let mut runner = CaseRunner::new();
    run_test_set(&path, "fixture", &mut report, &mut runner, true);
    assert_eq!((report.total, report.passed, report.failed), (2, 0, 0));
    assert_eq!(
        report.skip_reasons.get("dependency:streaming"),
        Some(&2),
        "both cases carry the set's own declaration: {:?}",
        report.skip_reasons
    );
    let _ = std::fs::remove_file(&path);
}

/// The converse, so that the rule above cannot degrade into "a test set with any dependency is
/// skipped": a declaration this processor satisfies leaves every case to be adjudicated on its
/// merits, and a case's own declaration still gates it.
#[test]
fn a_test_set_dependency_this_processor_satisfies_gates_nothing() {
    let set = r#"<test-set xmlns="http://www.w3.org/2012/10/xslt-test-catalog" name="fixture">
          <dependencies><spec value="XSLT30+"/><feature value="serialization"/></dependencies>
          <test-case name="one">
             <test><stylesheet file="no-such-file.xsl"/></test>
             <result><assert-xml><![CDATA[<out/>]]></assert-xml></result>
          </test-case>
          <test-case name="two">
             <dependencies><feature value="schema_aware"/></dependencies>
             <test><stylesheet file="no-such-file.xsl"/></test>
             <result><assert-xml><![CDATA[<out/>]]></assert-xml></result>
          </test-case>
       </test-set>"#;
    let path = temporary_file("set-level-dependency-satisfied", set);
    let mut report = SuiteReport::new("fixture");
    let mut runner = CaseRunner::new();
    run_test_set(&path, "fixture", &mut report, &mut runner, true);
    assert_eq!(report.total, 2);
    assert_eq!(
        report.skip_reasons.get("dependency:schema_aware"),
        Some(&1),
        "{:?}",
        report.skip_reasons
    );
    assert_eq!(
        report.skip_reasons.get("unreadable-stylesheet"),
        Some(&1),
        "the ungated case was driven and declined for its own reason: {:?}",
        report.skip_reasons
    );
    let _ = std::fs::remove_file(&path);
}

/// A test set the catalogue names but whose file cannot be read is not a set with no cases: it is
/// a set whose cases nobody counted. Both halves are recorded, because a total that is quietly
/// short is the one kind of wrong number no reader can detect.
#[test]
fn a_test_set_that_cannot_be_read_is_recorded_rather_than_dropped() {
    let mut report = SuiteReport::new("fixture");
    let mut runner = CaseRunner::new();
    let missing = std::env::temp_dir().join("oxixml-xslt30-no-such-test-set.xml");
    let _ = std::fs::remove_file(&missing);
    run_test_set(&missing, "absent", &mut report, &mut runner, true);
    assert_eq!(
        report
            .skip_reasons
            .get("catalogue:unreadable-test-set:absent"),
        Some(&1),
        "{:?}",
        report.skip_reasons
    );

    let path = temporary_file("malformed-test-set", "<test-set><unclosed></test-set>");
    run_test_set(&path, "malformed", &mut report, &mut runner, true);
    assert_eq!(
        report
            .skip_reasons
            .get("catalogue:test-set-did-not-parse:malformed"),
        Some(&1),
        "{:?}",
        report.skip_reasons
    );
    let _ = std::fs::remove_file(&path);
}

/// The admission invariant, pinned against the real catalogue: every `<test-set>` it lists is
/// reachable — the file exists, decodes, parses, and answers to the name the catalogue gave it.
/// A set that fails any of those would previously have vanished from the run without trace; this
/// is the test that says the driver's `total` is the catalogue's own size and not a selection.
#[test]
fn every_test_set_the_catalogue_lists_is_reachable() {
    let Some(root) = env::xml_suite_subdir("xslt30-test", "catalog.xml") else {
        return;
    };
    let text = read_xml_file(&root.join("catalog.xml")).expect("the catalogue is readable");
    let tree = Tree::parse_str(&text).expect("the catalogue parses");
    let catalog = tree.document_element().expect("a document element");
    let mut sets = 0;
    for entry in element_children(&catalog) {
        if local_name(&entry) != "test-set" {
            continue;
        }
        let name = attribute(&entry, "name").expect("a named test set");
        let file = attribute(&entry, "file").expect("a test set file");
        let path = root.join(&file);
        let text = read_xml_file(&path).unwrap_or_else(|| panic!("{name}: {file} is readable"));
        let tree =
            Tree::parse_str(&text).unwrap_or_else(|_| panic!("{name}: {file} parses as XML"));
        let set = tree
            .document_element()
            .unwrap_or_else(|| panic!("{name}: {file} has a document element"));
        assert_eq!(
            attribute(&set, "name").as_deref(),
            Some(name.as_str()),
            "{file} answers to the name the catalogue gave it"
        );
        sets += 1;
    }
    assert!(sets > 200, "the catalogue lists {sets} test sets");
}

/// An expectation this parser cannot read is not evidence about the processor. Thirteen
/// `assert-xml` expectations in the catalogue are not fragments it accepts: ten undeclare a
/// prefixed namespace (`<b xmlns:n=""/>`, legal only in XML 1.1) and three write C0 control
/// characters that XML 1.0 excludes outright. Comparing the canonical form of a result against
/// the *collapsed text* of such an expectation can only ever mismatch, whatever the processor
/// produced.
#[test]
fn an_expectation_that_is_not_a_readable_fragment_decides_nothing() {
    assert_eq!(
        canonical_fragment(r#"<a xmlns:n="http://n1.com/"><b xmlns:n=""/></a>"#),
        None,
        "a prefixed namespace undeclaration is not XML 1.0"
    );
    assert_eq!(
        canonical_fragment(r#"<a><b/></a>"#).as_deref(),
        Some("<a><b></b></a>"),
        "an ordinary fragment still canonicalizes"
    );

    let assertion = Tree::parse_str(
        r#"<assert-xml><![CDATA[<a xmlns:n="http://n1.com/"><b xmlns:n=""/></a>]]></assert-xml>"#,
    )
    .expect("well-formed fixture");
    let assertion = assertion.document_element().expect("a document element");
    let outcome: Result<XpathValue, ErrorInfo> = Ok(XpathValue::default());
    assert_eq!(
        check_case(
            &assertion,
            Some(produced("<a><b/></a>")),
            &outcome,
            Path::new(".")
        ),
        Outcome::Skip("unparseable-expectation:assert-xml".to_owned())
    );
}

/// The base output URI the catalogue names is where the principal result *would* be written, and
/// the three spellings mean three different things: a file below the test set, the test set's own
/// directory, and no base output URI at all. `current-output-uri-014` and `-015` are the pair that
/// pins the last two apart — the first requires the URI to contain the test set's path, the second
/// requires `fn:current-output-uri()` to be empty.
#[test]
fn the_base_output_uri_reads_all_three_spellings_the_catalogue_uses() {
    let directory = Path::new("tests/fn/current-output-uri");
    let cases = [
        (
            r#"<test><output file="results/current-output-uri-002.xml"/></test>"#,
            Some("file://tests/fn/current-output-uri/results/current-output-uri-002.xml"),
        ),
        // An empty reference resolves to the base it is resolved against: the test set itself.
        (
            r#"<test><output file=""/></test>"#,
            Some("file://tests/fn/current-output-uri/"),
        ),
        // The catalogue's sentinel for "this transformation has no base output URI".
        (r##"<test><output file="#absent"/></test>"##, None),
        // No `<output>` at all, and an `<output>` that names no file, say nothing about it.
        (r#"<test><stylesheet file="s.xsl"/></test>"#, None),
        (r#"<test><output tree="no"/></test>"#, None),
    ];
    for (fixture, expected) in cases {
        let tree = Tree::parse_str(fixture).expect("well-formed fixture");
        let test = tree.document_element().expect("a document element");
        assert_eq!(
            base_output_uri(&test, directory).as_deref(),
            expected,
            "{fixture}"
        );
    }
}

/// A set declined for its cost is declined by *name*, under a reason that is not a `dependency:`
/// one — the cases are in this processor's audience and it answers them, so a label saying the
/// catalogue put them out of scope would be untrue. A run that names the set gets it anyway.
#[test]
fn the_one_set_declined_for_its_cost_says_so_in_its_own_words() {
    assert_eq!(costly_set("unicode-90").as_deref(), Some("cost:unicode-90"));
    assert_eq!(costly_set("regex-classes"), None);
    assert_eq!(costly_set("axes"), None);
}

/// An `<environment>` registers the documents and collections a case can retrieve. The three
/// shapes the `collection` set writes all have to survive the reading: a collection with members,
/// one declared empty (which is not the same as an absent one), and a source that is *not* the
/// principal one and is therefore a document to be retrieved rather than the input.
#[test]
fn an_environment_registers_its_documents_and_collections() {
    let directory = Path::new("tests/fn/collection");
    let tree = Tree::parse_str(
        r#"<environment name="e">
             <source role="."><content>&lt;doc/&gt;</content></source>
             <source file="a.xml" uri="a.xml"/>
             <source file="collectinput/b.xml" uri="urn:b"/>
             <collection uri=""/>
             <collection uri="catalog1.xml">
               <source file="a.xml" uri="a.xml"/>
               <source file="doc15.xml#frag2" uri="doc15.xml#frag2"/>
             </collection>
           </environment>"#,
    )
    .expect("well-formed fixture");
    let element = tree.document_element().expect("a document element");
    let environment = Environment::parse(&element, directory);

    assert_eq!(
        environment.source.as_deref(),
        Some("<doc/>"),
        "the principal source is still the input, not a registration"
    );
    assert_eq!(
        environment.documents.get("a.xml").map(PathBuf::as_path),
        Some(directory.join("a.xml").as_path())
    );
    assert_eq!(
        environment.documents.get("urn:b").map(PathBuf::as_path),
        Some(directory.join("collectinput/b.xml").as_path()),
        "a registration binds the URI the catalogue chose, not the file's own name"
    );
    assert_eq!(
        environment.collections.get("").map(Vec::len),
        Some(0),
        "a collection declared with no members is empty, not absent"
    );
    assert_eq!(
        environment.collections.get("catalog1.xml").map(Vec::len),
        Some(2)
    );
    assert!(
        !environment.collections.contains_key("absent"),
        "nothing declares a collection under that URI"
    );
}
