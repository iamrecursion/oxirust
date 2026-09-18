//! Integration tests exercising every major ISO Schematron feature end to end, through the
//! public API only.

use std::collections::HashMap;

use oxixml_schematron::parse::IncludeResolver;
use oxixml_schematron::{Schematron, SchematronError};

type TestResult = Result<(), SchematronError>;

#[test]
fn assert_fires_when_the_test_is_false() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <pattern id="p1">
               <rule context="book">
                 <assert test="title" id="needs-title">a book needs a title</assert>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;

    let report = compiled.validate("<book/>")?;
    assert!(!report.is_valid());
    let failure = report.failed_asserts().next().expect("one failure");
    assert_eq!(failure.id.as_deref(), Some("needs-title"));
    assert_eq!(failure.message, "a book needs a title");
    assert_eq!(failure.location, "/book[1]");

    let report = compiled.validate("<book><title>Dune</title></book>")?;
    assert!(report.is_valid());
    Ok(())
}

#[test]
fn report_fires_when_the_test_is_true_and_never_invalidates() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <pattern id="p1">
               <rule context="book">
                 <report test="@out-of-print = 'true'" role="warning">out of print</report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;

    let report = compiled.validate(r#"<book out-of-print="true"/>"#)?;
    assert!(
        report.is_valid(),
        "a successful report never invalidates a document"
    );
    let fired = report.successful_reports().next().expect("one report");
    assert_eq!(fired.role.as_deref(), Some("warning"));
    assert_eq!(fired.message, "out of print");
    Ok(())
}

#[test]
fn value_of_and_name_interpolate_the_message() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <pattern>
               <rule context="item">
                 <report test="true()">
                   <name/> costs <value-of select="@price"/>
                 </report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    let report = compiled.validate(r#"<item price="9.99"/>"#)?;
    let fired = report.successful_reports().next().expect("one report");
    assert_eq!(fired.message.trim(), "item costs 9.99");
    Ok(())
}

#[test]
fn let_variables_shadow_from_schema_to_rule() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <let name="threshold" value="10"/>
             <pattern>
               <let name="threshold" value="20"/>
               <rule context="item">
                 <let name="threshold" value="30"/>
                 <report test="@n = $threshold">at the rule-scoped threshold</report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    assert!(
        compiled
            .validate(r#"<item n="10"/>"#)?
            .successful_reports()
            .next()
            .is_none()
    );
    assert!(
        compiled
            .validate(r#"<item n="20"/>"#)?
            .successful_reports()
            .next()
            .is_none()
    );
    assert!(
        compiled
            .validate(r#"<item n="30"/>"#)?
            .successful_reports()
            .next()
            .is_some()
    );
    Ok(())
}

#[test]
fn the_first_matching_rule_wins() -> TestResult {
    // Both rules' contexts match `item`; the pattern-order-first rule fires, and the second
    // never gets a chance to.
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <pattern>
               <rule context="item">
                 <report test="true()" id="first">first rule fired</report>
               </rule>
               <rule context="item">
                 <report test="true()" id="second">second rule fired</report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    let report = compiled.validate("<item/>")?;
    let fired: Vec<_> = report.successful_reports().collect();
    assert_eq!(fired.len(), 1);
    assert_eq!(fired[0].id.as_deref(), Some("first"));
    Ok(())
}

#[test]
fn phases_activate_only_their_own_patterns() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron" defaultPhase="strict">
        <phase id="strict">
          <active pattern="required-title"/>
        </phase>
        <phase id="lenient">
          <active pattern="warn-only"/>
        </phase>
        <pattern id="required-title">
          <rule context="book">
            <assert test="title">needs a title</assert>
          </rule>
        </pattern>
        <pattern id="warn-only">
          <rule context="book">
            <report test="not(title)">missing a title</report>
          </rule>
        </pattern>
      </schema>"#;

    let strict = Schematron::compile(schema, None)?; // defaultPhase
    assert!(!strict.validate("<book/>")?.is_valid());

    let lenient = Schematron::compile(schema, Some("lenient"))?;
    let report = lenient.validate("<book/>")?;
    assert!(report.is_valid());
    assert_eq!(report.successful_reports().count(), 1);

    let all = Schematron::compile(schema, Some("#ALL"))?;
    let report = all.validate("<book/>")?;
    assert_eq!(report.active_patterns.len(), 2);
    Ok(())
}

#[test]
fn unknown_phase_is_reported() {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern id="p1"><rule context="a"><assert test="b">x</assert></rule></pattern>
      </schema>"#;
    let error = Schematron::compile(schema, Some("nope")).expect_err("no such phase");
    assert!(matches!(
        error,
        SchematronError::UnknownReference { kind: "phase", .. }
    ));
}

#[test]
fn abstract_patterns_instantiate_with_param_substitution() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern id="unique-abstract" abstract="true">
          <rule context="$element">
            <let name="selfId" value="@id"/>
            <assert test="count(//$element[@id = $selfId]) = 1">
              duplicate <name/> id
            </assert>
          </rule>
        </pattern>
        <pattern id="unique-books" is-a="unique-abstract">
          <param name="element" value="book"/>
        </pattern>
      </schema>"#;
    let compiled = Schematron::compile(schema, None)?;
    assert_eq!(
        compiled.active_pattern_ids().collect::<Vec<_>>(),
        vec![Some("unique-books")]
    );

    let good = r#"<r><book id="a"/><book id="b"/></r>"#;
    assert!(compiled.validate(good)?.is_valid());

    let bad = r#"<r><book id="a"/><book id="a"/></r>"#;
    let report = compiled.validate(bad)?;
    assert!(!report.is_valid());
    assert_eq!(report.failed_asserts().count(), 2);
    Ok(())
}

#[test]
fn rule_extends_splices_in_lets_and_checks() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern>
          <rule abstract="true" id="has-id">
            <assert test="@id" id="needs-id">every element needs an id</assert>
          </rule>
          <rule context="book">
            <extends rule="has-id"/>
            <assert test="title">a book needs a title too</assert>
          </rule>
        </pattern>
      </schema>"#;
    let compiled = Schematron::compile(schema, None)?;
    let report = compiled.validate("<book/>")?;
    let ids: Vec<_> = report
        .failed_asserts()
        .map(|check| check.id.as_deref())
        .collect();
    assert_eq!(ids, vec![Some("needs-id"), None]);
    Ok(())
}

#[test]
fn extends_cycles_are_rejected() {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern>
          <rule context="a" id="r1"><extends rule="r2"/></rule>
          <rule abstract="true" id="r2"><extends rule="r1"/></rule>
        </pattern>
      </schema>"#;
    let error = Schematron::compile(schema, None).expect_err("a cycle");
    assert!(matches!(error, SchematronError::ExtendsCycle { .. }));
}

#[test]
fn a_long_non_repeating_extends_chain_is_rejected_not_a_stack_overflow() {
    // `r0` extends `r1` extends `r2` … each one a *different*, never-before-seen id, so the
    // cycle detector — which only catches a rule extending itself, directly or through
    // intermediaries — has nothing to reject here; only a separate depth guard does. Without
    // one, this recurses the native call stack without bound.
    let count = 300;
    let mut rules = String::from(r#"<rule context="a" id="r0"><extends rule="r1"/></rule>"#);
    for i in 1..count {
        if i + 1 < count {
            rules.push_str(&format!(
                r#"<rule abstract="true" id="r{i}"><extends rule="r{}"/></rule>"#,
                i + 1
            ));
        } else {
            rules.push_str(&format!(
                r#"<rule abstract="true" id="r{i}"><assert test="true()">ok</assert></rule>"#
            ));
        }
    }
    let schema = format!(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron"><pattern>{rules}</pattern></schema>"#
    );
    let error = Schematron::compile(&schema, None).expect_err("a chain this deep must be rejected");
    assert!(
        matches!(error, SchematronError::ExtendsDepth { limit: 256 }),
        "{error:?}"
    );
}

#[test]
fn an_extends_chain_within_the_depth_cap_still_compiles() -> TestResult {
    // The guard must track how deeply `extends` nests, not how many rules a schema declares in
    // total, and must not reject a chain that lands exactly on the cap.
    let count = 256;
    let mut rules = String::from(r#"<rule context="a" id="r0"><extends rule="r1"/></rule>"#);
    for i in 1..count {
        if i + 1 < count {
            rules.push_str(&format!(
                r#"<rule abstract="true" id="r{i}"><extends rule="r{}"/></rule>"#,
                i + 1
            ));
        } else {
            rules.push_str(&format!(
                r#"<rule abstract="true" id="r{i}"><assert test="true()" id="deepest">ok</assert></rule>"#
            ));
        }
    }
    let schema = format!(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron"><pattern>{rules}</pattern></schema>"#
    );
    let compiled = Schematron::compile(&schema, None)?;
    let report = compiled.validate("<a/>")?;
    assert!(report.is_valid());
    Ok(())
}

#[test]
fn diagnostics_are_attached_to_a_firing_check() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern>
          <rule context="item">
            <assert test="@price" diagnostics="no-price-diag">needs a price</assert>
          </rule>
        </pattern>
        <diagnostics>
          <diagnostic id="no-price-diag">the item id was "<value-of select="@id"/>"</diagnostic>
        </diagnostics>
      </schema>"#;
    let compiled = Schematron::compile(schema, None)?;
    let report = compiled.validate(r#"<item id="x9"/>"#)?;
    let failure = report.failed_asserts().next().expect("one failure");
    assert_eq!(failure.diagnostics, vec!["the item id was \"x9\""]);
    Ok(())
}

#[test]
fn subject_relocates_the_reported_location() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern>
          <rule context="book">
            <assert test="price" subject="@id">this book is missing a price</assert>
          </rule>
        </pattern>
      </schema>"#;
    let compiled = Schematron::compile(schema, None)?;
    let report = compiled.validate(r#"<book id="b1"/>"#)?;
    let failure = report.failed_asserts().next().expect("one failure");
    assert_eq!(failure.location, "/book[1]/@id");
    Ok(())
}

#[test]
fn query_binding_selects_the_xpath_version() {
    let xpath2_only = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt2">
        <pattern><rule context="a"><assert test="(1 to 3) = 2">present</assert></rule></pattern>
      </schema>"#;
    assert!(Schematron::compile(xpath2_only, None).is_ok());

    let under_xpath1 = xpath2_only.replace("queryBinding=\"xslt2\"", "");
    let error = Schematron::compile(&under_xpath1, None).expect_err("`to` needs XPath 2.0+");
    assert!(matches!(error, SchematronError::XPath { .. }));
}

#[test]
fn context_unions_match_either_alternative() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern>
          <rule context="book | cd">
            <report test="true()">seen</report>
          </rule>
        </pattern>
      </schema>"#;
    let compiled = Schematron::compile(schema, None)?;
    let report = compiled.validate("<catalog><book/><cd/><dvd/></catalog>")?;
    assert_eq!(report.successful_reports().count(), 2);
    Ok(())
}

#[test]
fn to_svrl_contains_the_expected_structure() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <pattern id="p1">
               <rule context="book">
                 <assert test="title" id="needs-title">needs a title</assert>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    let svrl = compiled.validate("<book/>")?.to_svrl();
    assert!(svrl.contains("svrl:schematron-output"));
    assert!(svrl.contains("xmlns:svrl=\"http://purl.oclc.org/dsdl/svrl\""));
    assert!(svrl.contains("svrl:active-pattern"));
    assert!(svrl.contains("svrl:fired-rule"));
    assert!(svrl.contains("svrl:failed-assert"));
    assert!(svrl.contains("needs-title"));
    assert!(svrl.contains("needs a title"));
    Ok(())
}

#[derive(Debug)]
struct MapResolver(HashMap<&'static str, &'static str>);

impl IncludeResolver for MapResolver {
    fn resolve(&self, href: &str) -> Result<String, SchematronError> {
        self.0
            .get(href)
            .map(|text| (*text).to_owned())
            .ok_or_else(|| SchematronError::UnresolvedInclude {
                href: href.to_owned(),
            })
    }
}

#[test]
fn include_splices_in_the_referenced_element() -> TestResult {
    let mut fragments = HashMap::new();
    fragments.insert(
        "shared-pattern.xml",
        r#"<pattern xmlns="http://purl.oclc.org/dsdl/schematron" id="shared">
             <rule context="book"><assert test="title">needs a title</assert></rule>
           </pattern>"#,
    );
    let resolver = MapResolver(fragments);

    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <include href="shared-pattern.xml"/>
      </schema>"#;
    let compiled = Schematron::compile_with(schema, None, Some(&resolver))?;
    assert_eq!(
        compiled.active_pattern_ids().collect::<Vec<_>>(),
        vec![Some("shared")]
    );
    assert!(!compiled.validate("<book/>")?.is_valid());
    Ok(())
}

#[test]
fn an_include_without_a_resolver_is_an_error() {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <include href="anywhere.xml"/>
      </schema>"#;
    let error = Schematron::compile(schema, None).expect_err("no resolver was supplied");
    assert!(matches!(error, SchematronError::UnresolvedInclude { .. }));
}

#[test]
fn a_missing_required_context_is_reported() {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <pattern><rule><assert test="true()">no context</assert></rule></pattern>
      </schema>"#;
    let error = Schematron::compile(schema, None).expect_err("the rule has no context");
    assert!(matches!(
        error,
        SchematronError::MissingAttribute {
            name: "context",
            ..
        }
    ));
}

#[test]
fn namespaces_bind_prefixes_used_in_expressions() -> TestResult {
    let schema = r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
        <ns prefix="c" uri="urn:example:catalog"/>
        <pattern>
          <rule context="c:book">
            <assert test="c:title">needs a title</assert>
          </rule>
        </pattern>
      </schema>"#;
    let compiled = Schematron::compile(schema, None)?;
    let document = r#"<c:catalog xmlns:c="urn:example:catalog"><c:book/></c:catalog>"#;
    assert!(!compiled.validate(document)?.is_valid());
    Ok(())
}

#[test]
fn schema_level_let_is_a_global_evaluated_once_at_the_document_root() -> TestResult {
    // ISO/IEC 19757-3 §5.4.5: a schema-level `sch:let` is a *global* XSLT variable, so its
    // value is computed once with the document root as the context node — not once per rule
    // firing with the firing node in focus. `name(.)` on the root node is the empty string and
    // `count(../*)` is zero, and both rule firings must see exactly that.
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <let name="ctx" value="name(.)"/>
             <let name="sibs" value="count(../*)"/>
             <pattern>
               <rule context="item">
                 <report test="true()">ctx=<value-of select="$ctx"/> sibs=<value-of select="$sibs"/></report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    let report = compiled.validate("<doc><item/><item/></doc>")?;
    let fired: Vec<_> = report.successful_reports().collect();
    assert_eq!(fired.len(), 2);
    for result in fired {
        assert_eq!(result.message, "ctx= sibs=0");
    }
    Ok(())
}

#[test]
fn a_global_let_may_refer_to_an_earlier_global_let() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron">
             <let name="items" value="count(//item)"/>
             <let name="doubled" value="$items * 2"/>
             <pattern>
               <rule context="doc">
                 <report test="true()">n=<value-of select="$doubled"/></report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    let report = compiled.validate("<doc><item/><item/><item/></doc>")?;
    let fired = report.successful_reports().next().expect("one report");
    assert_eq!(fired.message, "n=6");
    Ok(())
}

#[test]
fn phase_level_let_is_global_too() -> TestResult {
    let compiled = Schematron::compile(
        r#"<schema xmlns="http://purl.oclc.org/dsdl/schematron" defaultPhase="only">
             <phase id="only">
               <let name="root" value="name(/*)"/>
               <active pattern="p"/>
             </phase>
             <pattern id="p">
               <rule context="item">
                 <report test="true()">root=<value-of select="$root"/></report>
               </rule>
             </pattern>
           </schema>"#,
        None,
    )?;
    let report = compiled.validate("<doc><item/><item/></doc>")?;
    let fired: Vec<_> = report.successful_reports().collect();
    assert_eq!(fired.len(), 2);
    for result in fired {
        assert_eq!(result.message, "root=doc");
    }
    Ok(())
}
