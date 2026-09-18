//! The expected-failure ledger: every `xslt30-test` case this processor does not pass, the class
//! it belongs to, and the evidence behind that class.
//!
//! A conformance percentage is a summary, and a summary is not a claim anybody can check. This
//! table is the claim: each of the cases the full catalogue reports as failing is named here once,
//! with a [`FailureClass`] and a sentence of evidence — a reference-processor record, a
//! specification citation, or the mechanism a reader can follow to the code. [`audit`] then
//! compares a run's failing set against it and answers whether the two are *equal*, in both
//! directions.
//!
//! Equality in both directions is the point, and it is what makes the ledger a gate rather than a
//! excuse list. A case that starts failing and is not in the table is a regression. A case that is
//! in the table and stops failing is a **stale entry**, and it fails the gate exactly as loudly:
//! an entry nobody removed is a claim the processor has a defect it no longer has, and a table
//! that is allowed to over-state its own failures is one nobody can trust to under-state them
//! either.
//!
//! # What a class means
//!
//! Two of the five say the case is not this processor's to pass, and three say it is:
//!
//! * [`FailureClass::SuiteDefect`] and [`FailureClass::Dependency`] — the catalogue, not the
//!   processor. Each entry carries the citation that establishes it: a published
//!   reference-processor record, a passage of the specification read against the case's own files,
//!   or a conflict the catalogue states in its own words.
//! * [`FailureClass::UnicodeVersion`] — the case pins its expectation to a Unicode version this
//!   workspace has moved past. Not excused as a suite defect, because the case is well-formed and
//!   was right when it was written; not an engine gap either, because closing it would mean
//!   shipping historical character tables.
//! * [`FailureClass::Performance`] — the case does not finish inside the runner's per-case budget.
//! * [`FailureClass::EngineGap`] — a genuine gap in this workspace, named as roadmap work. These
//!   are **not excused**: the evidence says what is missing and where, so the entry reads as a
//!   backlog item rather than as an exemption.
//!
//! # What is deliberately not in the table
//!
//! `catalog-003` and `catalog-007` each parse the entire catalogue, come close to the per-case
//! budget doing it, and cross it or not depending on what else the machine is doing; driven alone
//! they pass. A borderline case is a fact about the run, not about the processor, so neither is
//! ledgered — and a run in which one of them times out therefore **fails** this gate, which is the
//! intended outcome: [`crate::ledger::Discrepancy`] says so in as many words and names the remedy
//! (drive that test set alone) rather than letting a load artefact accumulate into a permanent
//! entry.

use crate::ledger::{ExpectedFailure, FailureClass, LedgerVerdict};
use crate::report::SuiteReport;

/// The dependency capabilities a [`FailureClass::Dependency`] entry is allowed to rest on.
///
/// A dependency entry claims the catalogue put the case outside this processor's audience. That is
/// only true of a capability the processor actually declares missing, or of a resource the run
/// cannot reach at all — so the vocabulary is closed and
/// [`tests::a_dependency_entry_names_a_capability_this_processor_really_lacks`] checks every entry
/// against it and, for the feature names, against the engine's own answer.
#[cfg(test)]
const DEPENDENCY_GROUNDS: &[&str] = &["streaming", "network"];

/// Every case the full catalogue reports as failing — 80 of them, exactly once each.
///
/// Grouped by class in reporting order and sorted by id inside each group, so that a reader can
/// take in one class at a time and still find an id by eye; a run's failing list comes out sorted,
/// so the two are diffable within a class. See the module documentation for what the classes mean
/// and for the two cases that are deliberately absent.
pub const EXPECTED_FAILURES: &[ExpectedFailure] = &[
    // ---------------------------------------------------------------------------------------
    // suite-defect — the catalogue, not the processor.
    // ---------------------------------------------------------------------------------------
    ExpectedFailure {
        id: "as/as-1303",
        class: FailureClass::SuiteDefect,
        evidence: "The expectation and the produced result differ only in whitespace, and the \
                   whitespace is whitespace no processor may emit: as-1303.xsl separates the \
                   <e1>…<e6> children of its literal result element with whitespace-only text \
                   nodes in the stylesheet, which XSLT 3.0 §4.2 strips, and the seven runs the \
                   expectation carries are exactly the seven this processor omits. Created \
                   2019-02-06, after both published reference-results files, so neither records \
                   a verdict either way.",
    },
    ExpectedFailure {
        id: "as/as-1304",
        class: FailureClass::SuiteDefect,
        evidence: "The same stylesheet shape and the same §4.2 argument as as-1303, with the same \
                   creation date and the same absence from both published reference-results files.",
    },
    ExpectedFailure {
        id: "attribute-set/attribute-set-1508",
        class: FailureClass::SuiteDefect,
        evidence: "Whitespace-only difference. The reference processor's published results record \
                   this case as result=\"pass\" comment=\"OK after ignoring whitespace text\" — a \
                   normalization the catalogue licenses nowhere: admin/catalog-schema.xsd defines \
                   assert-xml as a byte-for-byte comparison of the canonicalized forms, and \
                   runner/compare.xsl compares text nodes with string(.) = string($b). The \
                   environment source is <doc><foo>a</foo></doc>, which holds no whitespace at all \
                   for the expectation's newline-and-two-spaces to have come from.",
    },
    ExpectedFailure {
        id: "attribute-set/attribute-set-1509",
        class: FailureClass::SuiteDefect,
        evidence: "The same mechanism as attribute-set-1508, with its own \
                   comment=\"OK after ignoring whitespace text\" record in the published \
                   reference results.",
    },
    ExpectedFailure {
        id: "base-uri/base-uri-052",
        class: FailureClass::SuiteDefect,
        evidence: "Four of the five assertions hold. The fifth wants dir5/data.xml resolved as if \
                   the xml:base already on dir/data2.xml's root had survived XInclude processing, \
                   and XInclude 1.0 §4.5.5 base-URI fixup replaces it — which is what this \
                   workspace's XInclude processor does. The catalogue concedes the deviation in \
                   its own inline comment beside the assertion: the specification says an existing \
                   xml:base \"is replaced by the new attribute\", \"but Xerces doesn't appear to do \
                   this\".",
    },
    ExpectedFailure {
        id: "for-each-group/for-each-group-085",
        class: FailureClass::SuiteDefect,
        evidence: "The mirror image of the other whitespace cases and the one with no reference \
                   record in either direction: the source begins <c>, newline, three spaces, \
                   <t n=\"T\"/>, the stylesheet declares no xsl:strip-space, and the built-in rule \
                   copies text nodes — so XSLT 3.0 §4.4 preserves the whitespace this processor \
                   emits and the expectation omits. Absent from both published reference-results \
                   files, so it rests on §4.4 alone.",
    },
    ExpectedFailure {
        id: "function/function-1701",
        class: FailureClass::SuiteDefect,
        evidence: "Whitespace-only difference, with the reference processor's own \
                   result=\"pass\" comment=\"OK after ignoring whitespace text\" record. Exactly \
                   twenty ids in that file carry the comment, so the signal selects precisely \
                   rather than covering a class of cases.",
    },
    ExpectedFailure {
        id: "math/math-3702",
        class: FailureClass::SuiteDefect,
        evidence: "Unsatisfiable together with extension-functions-0105. Both describe \
                   xsl:extension-element-prefixes naming a reserved namespace and demand different \
                   codes — XTSE0085 here, XTSE0800 there — and an error assertion compares codes \
                   exactly, so no processor passes both. Only extension-functions-0105 has a \
                   reference-processor record (result=\"pass\"); math-3702 was created 2019-01-21 \
                   and appears in neither published results file. This processor passes the case \
                   with the record.",
    },
    ExpectedFailure {
        id: "on-non-empty/on-non-empty-004",
        class: FailureClass::SuiteDefect,
        evidence: "Whitespace-only difference, with the reference processor's own \
                   result=\"pass\" comment=\"OK after ignoring whitespace text\" record.",
    },
    ExpectedFailure {
        id: "on-non-empty/on-non-empty-005",
        class: FailureClass::SuiteDefect,
        evidence: "Whitespace-only difference, with the reference processor's own \
                   result=\"pass\" comment=\"OK after ignoring whitespace text\" record.",
    },
    // ---------------------------------------------------------------------------------------
    // dependency — a capability this processor declares it does not have, or a resource the run
    // cannot reach.
    // ---------------------------------------------------------------------------------------
    ExpectedFailure {
        id: "stream-available/stream-available-002",
        class: FailureClass::Dependency,
        evidence: "streaming — fn:stream-available answering true is a claim only a streaming \
                   processor can make, and this one declares streaming unsupported (2899 cases are \
                   gated on it). The discriminator is the whole argument: this processor answers \
                   false for every URI, so it passes exactly the three cases of the set that \
                   assert false and fails exactly the three that assert true. These three escape \
                   the catalogue's own gate only because _stream-available-test-set.xml declares \
                   <spec value=\"XSLT30+\"/> and no <feature value=\"streaming\"/>.",
    },
    ExpectedFailure {
        id: "stream-available/stream-available-004",
        class: FailureClass::Dependency,
        evidence: "streaming — the same set and the same discriminator as stream-available-002: \
                   an assertion that fn:stream-available is true.",
    },
    ExpectedFailure {
        id: "stream-available/stream-available-005",
        class: FailureClass::Dependency,
        evidence: "streaming — the same set and the same discriminator as stream-available-002: \
                   an assertion that fn:stream-available is true.",
    },
    ExpectedFailure {
        id: "unparsed-text/unparsed-text-2003",
        class: FailureClass::Dependency,
        evidence: "network — only the first of four assertions differs, and it needs \
                   unparsed-text-available('http://www.w3.org/Consortium/mission.html') to be \
                   true, which is a retrieval over the network. The case declares no remote_http \
                   dependency for the catalogue's gate to read, and the reference processor's own \
                   two published runs disagree about it: fail in the 2.0 results file, pass in the \
                   3.0 one.",
    },
    // ---------------------------------------------------------------------------------------
    // unicode-version — an expectation pinned to a Unicode version this workspace has moved past.
    //
    // reclass.xsl takes one character class per case, enumerates the members of \p{class}+ and
    // \P{class}+ over the whole Unicode range and compares them against reference results for
    // Unicode 3.1, 5.2 and 6.0, answering <true/> if the class matches any of the three. Its own
    // comment says why: "Alternative results are acceptable depending on which version of Unicode
    // the processor supports." A failing case is therefore a class whose membership changed after
    // 6.0, measured against this workspace's Unicode 17.0.0 tables. 90 of the 119 classes still
    // agree; these are the ones that do not. The reference processor recorded all 120 cases of the
    // set as result="notRun" comment="requires excessive resources", so none has a reference
    // verdict to appeal to.
    // ---------------------------------------------------------------------------------------
    ExpectedFailure {
        id: "regex-classes/regex-classes-001",
        class: FailureClass::UnicodeVersion,
        evidence: "General category C, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-003",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Cf, whose membership changed after Unicode 6.0 — worked out \
                   from the reference documents themselves: the 3.1 reference starts a run at \
                   8204 where 5.2 and 6.0 start it at 8203, so the three the case accepts do not \
                   even agree with each other.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-004",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Cn, whose membership changed after Unicode 6.0. Every \
                   codepoint assigned since is one this class has lost.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-006",
        class: FailureClass::UnicodeVersion,
        evidence: "General category L, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-007",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Ll, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-008",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Lm, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-009",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Lo, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-011",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Lu, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-012",
        class: FailureClass::UnicodeVersion,
        evidence: "General category M, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-013",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Mc, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-014",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Me, whose membership changed after Unicode 6.0 — worked out \
                   from the reference documents themselves: 3.1 has 1758 and not 42608–42610, 5.2 \
                   has both, 6.0 has 42608–42610 and not 1758.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-015",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Mn, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-016",
        class: FailureClass::UnicodeVersion,
        evidence: "General category N, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-017",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Nd, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-018",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Nl, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-019",
        class: FailureClass::UnicodeVersion,
        evidence: "General category No, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-020",
        class: FailureClass::UnicodeVersion,
        evidence: "General category P, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-022",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Pd, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-023",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Pe, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-024",
        class: FailureClass::UnicodeVersion,
        evidence: "General category C — the set tests it twice, this being the second case, and \
                   both fail for the one reason regex-classes-001 gives.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-026",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Po, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-027",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Ps, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-028",
        class: FailureClass::UnicodeVersion,
        evidence: "General category S, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-029",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Sc, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-030",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Sk, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-031",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Sm, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-032",
        class: FailureClass::UnicodeVersion,
        evidence: "General category So, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-033",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Z, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-036",
        class: FailureClass::UnicodeVersion,
        evidence: "General category Zs, whose membership changed after Unicode 6.0.",
    },
    ExpectedFailure {
        id: "regex-classes/regex-classes-111",
        class: FailureClass::UnicodeVersion,
        evidence: "The block name IsPrivateUse, and the one case of the set that is a conflict \
                   between two W3C suites rather than drift. The xslt30 reference results require \
                   the block to be the union of U+E000–U+F8FF, U+F0000–U+FFFFD and \
                   U+100000–U+10FFFD, byte-identical across the 3.1, 5.2 and 6.0 reference files \
                   — XSD 1.0 Appendix F copying Unicode 3.1's Blocks.txt. The XML Schema test \
                   suite says the opposite for the same escape: msData/regex reL98, reL99, reM98 \
                   and reN99 each carry <expected validity=\"invalid\"/> for exactly those \
                   characters, and all four pass today. Extending the block closes this case and \
                   breaks those four.",
    },
    // ---------------------------------------------------------------------------------------
    // engine-gap — this workspace's own backlog, named rather than excused.
    // ---------------------------------------------------------------------------------------
    ExpectedFailure {
        id: "accessor/accessor-008",
        class: FailureClass::EngineGap,
        evidence: "A document-identity question over a constructed or retrieved document: expects \
                   <t1>true</t1>, produces <t1>false</t1>.",
    },
    ExpectedFailure {
        id: "accessor/accessor-026",
        class: FailureClass::EngineGap,
        evidence: "The same document-identity area as accessor-008, reported through the case's \
                   own diagnostic string: two of its six accessors answer \
                   \"WROONG TYPE returned\" (the case's spelling) where a node is wanted.",
    },
    ExpectedFailure {
        id: "catalog/catalog-004",
        class: FailureClass::EngineGap,
        evidence: "The case walks the whole catalogue and retrieves every stylesheet module in it \
                   with fn:document, so one module this workspace cannot parse fails the case. Two \
                   remain, both in the XML parser rather than in the XSLT engine or this driver: \
                   whitespace-011.xsl declares an external parameter entity \
                   (<!ENTITY % ext SYSTEM \"entity.ent\">%ext;) and parameter entities are not \
                   expanded; include-0101b.xsl declares its general entities inside an external \
                   subset, whose own system identifiers are relative to that subset rather than to \
                   the document (XML 1.0 §4.2.2), and an ExternalEntity carries the system \
                   identifier as written with no base URI for a resolver to chain from. The base \
                   URI of the retrieved document itself is correct: this case previously failed on \
                   use-when-0136.xsl and no longer does.",
    },
    ExpectedFailure {
        id: "catalog/catalog-006",
        class: FailureClass::EngineGap,
        evidence: "The catalogue's own way of measuring this workspace's compile-time backlog: the \
                   case walks the whole instruction inventory and reports every construct \
                   element-available() answers no to. The list it produces is exactly the \
                   not-implemented:compile tally — xsl:package, xsl:merge, xsl:source-document, \
                   xsl:accumulator, xsl:use-package, xsl:fork, xsl:evaluate — so it closes when \
                   that backlog does.",
    },
    ExpectedFailure {
        id: "catalog/catalog-006b",
        class: FailureClass::EngineGap,
        evidence: "The same inventory walk as catalog-006, over the same construct list.",
    },
    ExpectedFailure {
        id: "catalog/catalog-012",
        class: FailureClass::EngineGap,
        evidence: "The same whole-catalogue walk and the same two unparseable modules as \
                   catalog-004; only the property the case asks each retrieved stylesheet about \
                   differs.",
    },
    ExpectedFailure {
        id: "current-output-uri/current-output-uri-017",
        class: FailureClass::EngineGap,
        evidence: "XSLT 3.0 §24.3 clears the current output URI for the evaluation of a dynamic \
                   function call, and this case makes the call through an anonymous function body \
                   — function() { current-output-uri() } — where the two other cases of the rule \
                   name the function directly and now pass. The signal is missing rather than \
                   ignored: oxixml-xpath builds an inline function as a native function item, and \
                   the native arm of Function::call publishes no call site, so the inner call's \
                   own static invocation frame is the innermost one and the host function cannot \
                   tell it is inside a function-item invocation. Needs either a dynamic call site \
                   on that arm plus an any-frame predicate (not a change to \
                   invoked_through_function_item, which fn:current, fn:current-group and \
                   fn:current-grouping-key depend on being innermost-only), or a public reader of \
                   the open-call depth.",
    },
    ExpectedFailure {
        id: "higher-order-functions/higher-order-functions-066",
        class: FailureClass::EngineGap,
        evidence: "An XTTE0780-versus-XPTY0004 disagreement over an xsl:function result type: the \
                   engine reports the stylesheet function's declared result type as violated \
                   where the case wants the type error attributed to the expression.",
    },
    ExpectedFailure {
        id: "id/id-016",
        class: FailureClass::EngineGap,
        evidence: "NOT a suite defect, however much it resembles the whitespace cluster: the \
                   reference processor records it as a plain result=\"pass\" with no \
                   normalization comment, so it produced the whitespace this processor omits. Ten \
                   runs of three spaces are missing from the result and nothing else differs. \
                   XSLT 3.0 §4.4 preserves whitespace-only text nodes from a source document \
                   unless xsl:strip-space says otherwise.",
    },
    ExpectedFailure {
        id: "initial-function/initial-function-905",
        class: FailureClass::EngineGap,
        evidence: "The entry-point rules: expects XTDE0041 and the transformation succeeds \
                   instead.",
    },
    ExpectedFailure {
        id: "json-to-xml/json-to-xml-typed-010",
        class: FailureClass::EngineGap,
        evidence: "Expects XTDE3245 where this processor declines xsl:import-schema at compile \
                   time with XTSE1650. The reference processor records the case notRun \
                   (\"dependency not satisfied: !feature:schema_aware\") because it is \
                   schema-aware; this processor is not, so the case does apply here and the code \
                   is the disagreement.",
    },
    ExpectedFailure {
        id: "merge/merge-085",
        class: FailureClass::EngineGap,
        evidence: "xsl:merge over several sources concatenates where it must interleave. The \
                   construct is 90 of the not-implemented:compile skips; this is the one case of \
                   it that compiles far enough to produce a wrong answer.",
    },
    ExpectedFailure {
        id: "namespace/namespace-2633",
        class: FailureClass::EngineGap,
        evidence: "Wants XTSE0020 for xsl:inherit-namespaces=\" \" and the transformation \
                   succeeds. The attribute-permission table in the compiler's validation decides \
                   465 passing error cases and needs a scope-aware rewrite to admit this one — the \
                   same area as shadow-008 and static-025/-026.",
    },
    ExpectedFailure {
        id: "namespace/namespace-6202",
        class: FailureClass::EngineGap,
        evidence: "Wants XPST0081 for a prefix XSLT-hosted XPath must not have in scope: XSLT 3.0 \
                   §5.1 gives an expression only the in-scope namespaces of its containing \
                   element, and this workspace's static context seeds fn, xs, math, map, array and \
                   err unconditionally. One case against a workspace-wide change of behaviour.",
    },
    ExpectedFailure {
        id: "package/package-001r",
        class: FailureClass::EngineGap,
        evidence: "Returns a raw sequence where the case expects a tree. xsl:package is the \
                   largest single construct in the compile backlog (432 skips).",
    },
    ExpectedFailure {
        id: "package/package-100",
        class: FailureClass::EngineGap,
        evidence: "A function a used package exposes must be visible to the using package; the \
                   call reports XPST0017 instead. Needs xsl:package and xsl:use-package.",
    },
    ExpectedFailure {
        id: "package/package-101",
        class: FailureClass::EngineGap,
        evidence: "The same package-visibility requirement as package-100.",
    },
    ExpectedFailure {
        id: "shadow/shadow-001",
        class: FailureClass::EngineGap,
        evidence: "Writes _select=\"${}{$N}\", whose {} is an empty expression this parser reports \
                   as XPST0003. Nobody has yet established the intended reading from the \
                   specification, so the entry records an open question rather than a known fix.",
    },
    ExpectedFailure {
        id: "shadow/shadow-008",
        class: FailureClass::EngineGap,
        evidence: "An unknown xsl:-prefixed attribute on a literal result element must be \
                   XTSE0805; the transformation succeeds. The attribute-permission table again — \
                   see namespace-2633.",
    },
    ExpectedFailure {
        id: "snapshot/snapshot-0102a",
        class: FailureClass::EngineGap,
        evidence: "Function coercion of a function(node()) as node() argument, reported as \
                   XPTY0004 — oxixml-xpath's coercion rules rather than the XSLT engine's \
                   snapshot implementation.",
    },
    ExpectedFailure {
        id: "snapshot/snapshot-0103",
        class: FailureClass::EngineGap,
        evidence: "The transformation runs to completion and reports one orphan kind whose \
                   snapshot differs from the specification's own f:snapshot implementation. The \
                   case emits the disagreement in a <wrong> element, so the next reader starts \
                   from the case's own diagnosis.",
    },
    ExpectedFailure {
        id: "snapshot/snapshot-0103a",
        class: FailureClass::EngineGap,
        evidence: "The same orphan-kind disagreement as snapshot-0103, over the same f:snapshot \
                   comparison.",
    },
    ExpectedFailure {
        id: "static/static-012",
        class: FailureClass::EngineGap,
        evidence: "Expects XTDE0700 and this processor raises XPTY0004 — which is what the \
                   reference processor raises too: its published results record this case as a \
                   wrongError, \"Expected XTDE0700; got XPTY0004\". The two processors agree with \
                   each other and disagree with the catalogue.",
    },
    ExpectedFailure {
        id: "static/static-013",
        class: FailureClass::EngineGap,
        evidence: "Expects XTDE0050 and raises XPTY0004; the reference processor records the \
                   identical wrongError, \"Expected XTDE0050; got XPTY0004\".",
    },
    ExpectedFailure {
        id: "static/static-013c",
        class: FailureClass::EngineGap,
        evidence: "Expects XTTE0590 and raises XTTE0570; the reference processor records the \
                   identical wrongError, \"Expected XTTE0590; got XTTE0570\".",
    },
    ExpectedFailure {
        id: "static/static-025",
        class: FailureClass::EngineGap,
        evidence: "Wants XTSE0090 for static=\"yes\" on a non-global xsl:param. The \
                   attribute-permission table again — see namespace-2633.",
    },
    ExpectedFailure {
        id: "static/static-026",
        class: FailureClass::EngineGap,
        evidence: "Wants XTSE0090 for static=\"yes\" beside visibility=\"final\". The \
                   attribute-permission table again — see namespace-2633.",
    },
    ExpectedFailure {
        id: "transform/transform-004",
        class: FailureClass::EngineGap,
        evidence: "A stylesheet-function item must capture the runtime that created it: the \
                   closure outlives the inner transformation fn:transform ran, so the later call \
                   reports XPST0017 (\"a stylesheet function is only available inside a \
                   transformation\").",
    },
    ExpectedFailure {
        id: "transform/transform-005",
        class: FailureClass::EngineGap,
        evidence: "Needs the package-name / package-version options of fn:transform, which are \
                   xsl:package — reported honestly as FOXT0002, an option this processor does not \
                   implement.",
    },
    ExpectedFailure {
        id: "transform/transform-006",
        class: FailureClass::EngineGap,
        evidence: "The same fn:transform package options as transform-005.",
    },
    ExpectedFailure {
        id: "transform/transform-007",
        class: FailureClass::EngineGap,
        evidence: "The same fn:transform package options as transform-005.",
    },
    ExpectedFailure {
        id: "use-when/use-when-0134",
        class: FailureClass::EngineGap,
        evidence: "A consistency judgement across import groups: the engine reports XTSE3450 for \
                   static declarations the case holds to be consistent.",
    },
    ExpectedFailure {
        id: "use-when/use-when-0136",
        class: FailureClass::EngineGap,
        evidence: "The module's internal-subset external general entity is retrieved and spliced \
                   in, and the spliced element carries the entity's own base URI (XML 1.0 §4.2.2) \
                   — the case now fails one step further on, with XTDE0040. What is left is the \
                   static base URI: XSLT 3.0 §5.4.3 makes the static base URI of an element the \
                   base URI of that element, and the compiler consults a node's own base URI only \
                   when the module has none, so the module URI shadows the entity's and \
                   static-base-uri() inside the entity's content answers the wrong one, excluding \
                   the template the case names.",
    },
    ExpectedFailure {
        id: "use-when/use-when-0501",
        class: FailureClass::EngineGap,
        evidence: "A sort/output-shape difference rather than a use-when question: the items are \
                   all present and the order they come out in is not the one the expectation \
                   carries.",
    },
    ExpectedFailure {
        id: "xml-version/xml-version-021",
        class: FailureClass::EngineGap,
        evidence: "NOT a suite defect, for the same reason id-016 is not: the reference processor \
                   records a plain result=\"pass\" with no normalization comment. One text node is \
                   missing and its content is a LINE SEPARATOR between two spaces — XML 1.1 \
                   normalizes U+2028 to a line feed, after which the node is whitespace-only and \
                   this processor drops it, while every neighbouring element whose content is not \
                   whitespace-only survives intact.",
    },
    ExpectedFailure {
        id: "xml-version/xml-version-041",
        class: FailureClass::EngineGap,
        evidence: "Expects XTSE0020 and raises XTSE0165 — the right diagnosis (\"undeclaring a \
                   prefixed namespace binding with an empty URI requires Namespaces in XML 1.1\") \
                   under the wrong code, on a stylesheet the parser is entitled to reject.",
    },
];

/// The skip reason a narrowed run records for every test set it passed over.
const NOT_SELECTED: &str = "catalogue:not-selected";

/// Compares `report`'s failing cases against [`EXPECTED_FAILURES`], in both directions.
///
/// A run narrowed with [`super::SETS_VAR`] is declined rather than compared: it records a
/// `catalogue:not-selected` skip for every test set it passed over, so its failing set is a subset
/// of the catalogue's by construction and every unselected entry would be reported stale.
#[must_use]
pub fn audit(report: &SuiteReport) -> LedgerVerdict {
    let narrowed = report.skip_reasons.get(NOT_SELECTED).map(|passed_over| {
        format!(
            "the run was narrowed and passed over {passed_over} case(s); only a whole-catalogue \
             run has a failing set the ledger describes"
        )
    });
    crate::ledger::compare(report, EXPECTED_FAILURES, narrowed)
}

/// How many ledger entries each class holds, in reporting order.
#[must_use]
pub fn tally_by_class() -> Vec<(FailureClass, usize)> {
    crate::ledger::tally_by_class(EXPECTED_FAILURES)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::table_problems;

    /// A report whose failing set is exactly `ids`, shaped like a whole-catalogue run.
    fn report_failing(ids: &[&str]) -> SuiteReport {
        let mut report = SuiteReport::new("xslt30");
        for id in ids {
            report.record_fail(*id, "a failure message");
        }
        report
    }

    /// A report whose failing set is exactly the ledger's.
    fn report_matching_the_ledger() -> SuiteReport {
        let ids: Vec<&str> = EXPECTED_FAILURES.iter().map(|entry| entry.id).collect();
        report_failing(&ids)
    }

    /// No duplicate id, no entry filed out of its class, no entry without evidence — the rules
    /// [`crate::ledger::table_problems`] states once for every suite's table.
    #[test]
    fn the_table_is_well_formed() {
        assert_eq!(table_problems(EXPECTED_FAILURES), Vec::<String>::new());
    }

    /// Every id is `test-set/test-case`, the form this runner records a failure under. An id in
    /// any other shape can never match, so the gate would report it stale for ever.
    #[test]
    fn every_id_is_written_the_way_a_failure_is_recorded() {
        for entry in EXPECTED_FAILURES {
            let (set, case) = entry
                .id
                .split_once('/')
                .unwrap_or_else(|| panic!("`{}` is not `test-set/test-case`", entry.id));
            assert!(!set.is_empty() && !case.is_empty(), "{}", entry.id);
            assert!(
                case.starts_with(set),
                "`{}`: the catalogue names a case after the set it is in",
                entry.id
            );
        }
    }

    #[test]
    fn the_class_tally_accounts_for_every_entry() {
        assert_eq!(
            tally_by_class()
                .iter()
                .map(|(_, count)| count)
                .sum::<usize>(),
            EXPECTED_FAILURES.len()
        );
    }

    #[test]
    fn a_whole_catalogue_run_that_matches_the_ledger_agrees_with_it() {
        assert_eq!(
            audit(&report_matching_the_ledger()),
            LedgerVerdict::Agreed(EXPECTED_FAILURES.len())
        );
    }

    /// The one unledgered failure this machine really produces: a borderline case that crossed the
    /// per-case budget. It must fail the gate — and say what to do about it.
    #[test]
    fn a_borderline_case_that_times_out_fails_the_gate_and_names_the_remedy() {
        let mut report = report_matching_the_ledger();
        report.record_fail("catalog/catalog-007", "TIMEOUT: did not finish within 30s");
        let verdict = audit(&report);
        assert!(verdict.is_a_gate_failure(), "{verdict}");
        let reported = verdict.to_string();
        assert!(reported.contains("catalog/catalog-007"), "{reported}");
        assert!(reported.contains("driven alone"), "{reported}");
    }

    /// A narrowed run is the triage mode this driver is used in every day, and it must not trip the
    /// ship gate: a gate that fires on triage runs is one everybody learns to ignore.
    #[test]
    fn a_narrowed_run_is_not_compared_with_the_ledger() {
        let mut report = report_matching_the_ledger();
        report.record_skip(NOT_SELECTED);
        let verdict = audit(&report);
        assert!(matches!(verdict, LedgerVerdict::Narrowed(_)), "{verdict}");
        assert!(!verdict.is_a_gate_failure(), "{verdict}");
        assert!(verdict.to_string().contains("narrowed"), "{verdict}");
    }

    /// The seam between this table and the gap list beside it: a `dependency` entry claims the
    /// catalogue put the case outside this processor's audience, which is only true of a capability
    /// the processor really declares missing. Claiming it for a feature the engine answers `yes` to
    /// would be an exemption dressed as a declaration.
    #[test]
    fn a_dependency_entry_names_a_capability_this_processor_really_lacks() {
        for entry in EXPECTED_FAILURES {
            if entry.class != FailureClass::Dependency {
                continue;
            }
            let ground = entry
                .evidence
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_end_matches([',', '.']);
            assert!(
                DEPENDENCY_GROUNDS.contains(&ground),
                "`{}` rests on `{ground}`, which is not one of {DEPENDENCY_GROUNDS:?}",
                entry.id
            );
            if ground != "network" {
                assert!(
                    super::super::UNSUPPORTED_FEATURES.contains(&ground),
                    "`{}` claims the catalogue gates it on `{ground}`, but this processor does \
                     not declare that feature missing",
                    entry.id
                );
            }
        }
    }

    /// One class is empty, and it is empty on purpose — see the assertion's own message.
    #[test]
    fn the_performance_class_is_empty_and_says_why() {
        let performance = EXPECTED_FAILURES
            .iter()
            .filter(|entry| entry.class == FailureClass::Performance)
            .count();
        assert_eq!(
            performance, 0,
            "normalize-unicode-008 was the only performance entry this table ever needed, and it \
             passes: a case that finishes inside the budget must be removed from the ledger rather \
             than left in it under a class that no longer describes it"
        );
    }
}
