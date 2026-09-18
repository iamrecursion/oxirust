//! The expected-failure ledger for the W3C XML Schema Test Suite: every slot this processor does
//! not pass, in both passes over the suite, with the class it belongs to and the evidence behind
//! that class.
//!
//! See [`crate::ledger`] for what the classes mean and why a stale entry fails the gate as loudly
//! as a regression. What is particular to this suite is the *granularity* and the headline: the
//! residue is pinned per **slot** — one `schemaTest` or one `instanceTest` of one `testGroup`,
//! under one XSD version — rather than per test group, because a group whose schema assembles and
//! whose instances then disagree is a different fact from one whose schema is refused, and only the
//! slot says which. `oxixml-schema`'s own runner pins the same residue independently, from the
//! other side of the same engine.
//!
//! # The headline: none of it is an engine gap
//!
//! Every entry below is a [`FailureClass::SuiteDefect`] or a [`FailureClass::Dependency`] — 66 of
//! the 67 slots and 1 respectively. That is
//! not an assumption anybody inherited. Four of these ids were once classed as engine gaps;
//! all four were investigated against the suite sources and the running engine, **no** code change
//! was made, and each was reclassified — the four are `introspection`, `attP031`,
//! `particlesZ033_g` and `sg-abstract-upa2`, and each carries its own reasoning below.
//!
//! # Two things not to "fix"
//!
//! Both are cases where the change that would close a slot is precisely the evidence that it should
//! not be made:
//!
//! * **`attP031`** — the only change that closes it lands with zero pass-to-fail, because only four
//!   groups in the whole suite combine `use="prohibited"` with `fixed` and only this one supplies
//!   the attribute. Passing would therefore be evidence that no other test exercises the rule, not
//!   that the rule is right.
//! * **`introspection`** — making it assemble means relaxing schema-component resolution, which
//!   flips the ordinary "reference to a component that does not exist in a licensed namespace"
//!   shape from reject to accept and costs dozens of currently-passing assertions. Scoped narrowly
//!   to "the `schemaLocation` could not be retrieved" it silently accepts every mistyped
//!   `schemaLocation` instead.

use crate::ledger::{ExpectedFailure, FailureClass, LedgerVerdict};
use crate::report::SuiteReport;

/// Slots that fail in the XSD 1.0 pass (`xsts/1.0`).
///
/// Filed by class and sorted by id inside each class, the order
/// [`crate::ledger::table_problems`] requires.
pub const EXPECTED_FAILURES_1_0: &[ExpectedFailure] = &[
    ExpectedFailure {
        id: "Attribute_w3c/attP031/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Both of the group's schema slots already pass. Satisfying the 1.0 instance\
                   expectation — <x:elem x:att=\"37\"/> valid against <xsd:attribute name=\"att\"\
                   use=\"prohibited\" fixed=\"37\"/> — needs a rule under which a fixed value\
                   constraint makes a prohibited attribute use non-prohibiting. XSD 1.0 maps use\
                   to {required} only, and the same catalogue pins the opposite three times over\
                   (attF001, attJ002, attJ003), all of which this engine passes. Do not fix this;\
                   see the module documentation.",
    },
    ExpectedFailure {
        id: "DataTypes_w3c/anyURI_a004_1339/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Expects an xs:anyURI lexical form to be rejected. XSD 1.0's second edition made\
                   the anyURI lexical space effectively unconstrained — an erratum this suite\
                   predates — so a conformant processor now accepts every string, which is what\
                   this engine does.",
    },
    ExpectedFailure {
        id: "DataTypes_w3c/gMonth002_2061/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Expects `--03--` to be a valid xs:gMonth. That is the pre-erratum lexical form\
                   the XSD 1.0 second edition withdrew; the lexical space is `--MM`, which this\
                   engine implements.",
    },
    ExpectedFailure {
        id: "DataTypes_w3c/gMonth004_2063/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Expects `--05---05:00` to be a valid xs:gMonth — the same withdrawn pre-erratum\
                   lexical form as gMonth002_2061, with a timezone after it.",
    },
    ExpectedFailure {
        id: "Element_w3c/elemZ026/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects assembly to be refused for a complex-type derivation XSD 1.0 permits;\
                   the schema assembles here, and the group's own instance slots — which the suite\
                   grades against the schema it says must not exist — pass.",
    },
    ExpectedFailure {
        id: "Missing/missing001/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a schema that references the type `absent`, which no document in the\
                   group declares, to assemble. Assembling it means resolving a reference to a\
                   component that does not exist in a licensed namespace, which src-resolve\
                   forbids and which dozens of passing groups of this suite depend on being\
                   forbidden.",
    },
    ExpectedFailure {
        id: "Missing/missing002/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a schema that references the substitution group head `rotten`, which no\
                   document in the group declares, to assemble. Assembling it means resolving a\
                   reference to a component that does not exist in a licensed namespace, which\
                   src-resolve forbids and which dozens of passing groups of this suite depend on\
                   being forbidden.",
    },
    ExpectedFailure {
        id: "Missing/missing003/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a schema that references the substitution group head `missing`, which\
                   no document in the group declares, to assemble. Assembling it means resolving a\
                   reference to a component that does not exist in a licensed namespace, which\
                   src-resolve forbids and which dozens of passing groups of this suite depend on\
                   being forbidden.",
    },
    ExpectedFailure {
        id: "Missing/missing006/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a schema that references the type `absent` named by an xs:list\
                   itemType, which no document in the group declares, to assemble. Assembling it\
                   means resolving a reference to a component that does not exist in a licensed\
                   namespace, which src-resolve forbids and which dozens of passing groups of this\
                   suite depend on being forbidden.",
    },
    ExpectedFailure {
        id: "Particles_w3c/particlesZ001/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a restriction to be valid that the rcase-Recurse.2 mapping of XSD 1.0\
                   Part 1 §3.9.6 refuses: the derived content model does not map onto the base\
                   type's. The engine's diagnosis names the rule it applied.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ11/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Lu}` and the value the expectation requires it not to accept holds\
                   U+1D7A8, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ13/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Ll}` and the value the expectation requires it not to accept holds\
                   U+1D7C9, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ19/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Lo}` and the value the expectation requires it not to accept holds\
                   U+2FA1D, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ21/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{M}` and the value the expectation requires it not to accept holds U+1D1AD\
                   and U+1D172, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ23/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Mn}` and the value the expectation requires it not to accept holds\
                   U+1D1AD, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ25/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Mc}` and the value the expectation requires it not to accept holds\
                   U+1D172, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ29/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{N}` and the value the expectation requires it not to accept holds U+1D7FF,\
                   U+1034A and U+10323, outside the Basic Multilingual Plane. That expectation\
                   follows the original (2004) XSD 1.0 reading, under which a regular expression\
                   matches over UCS-2 code units and such a character is a surrogate pair no\
                   single-character escape can match; this engine matches over code points, which\
                   is what XSD 1.1 Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this\
                   set fail for that one reason and every one of them carries a supplementary-\
                   plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ31/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Nd}` and the value the expectation requires it not to accept holds\
                   U+1D7FF, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ33/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Nl}` and the value the expectation requires it not to accept holds\
                   U+1034A, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ35/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{No}` and the value the expectation requires it not to accept holds\
                   U+10323, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ61/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{S}` and the value the expectation requires it not to accept holds U+1D1DD,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ69/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{So}` and the value the expectation requires it not to accept holds\
                   U+1D1DD, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ75/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Cf}` and the value the expectation requires it not to accept holds\
                   U+E0078, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ77/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Co}` and the value the expectation requires it not to accept holds\
                   U+100000, U+F0000, U+FFFFD and U+10FFFD, outside the Basic Multilingual Plane.\
                   That expectation follows the original (2004) XSD 1.0 reading, under which a\
                   regular expression matches over UCS-2 code units and such a character is a\
                   surrogate pair no single-character escape can match; this engine matches over\
                   code points, which is what XSD 1.1 Part 2 Appendix G and F&O 3.1 require.\
                   Eighteen groups of this set fail for that one reason and every one of them\
                   carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reS21/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\d` and the value the expectation requires it not to accept holds U+1D7CE,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reS42/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\d` and the value the expectation requires it not to accept holds U+1D7FF,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reT63/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\D` and the value the expectation requires it not to accept holds U+1D7CD,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reT84/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\D` and the value the expectation requires it not to accept holds U+1D800,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "SimpleType_w3c/ste110/schema",
        class: FailureClass::SuiteDefect,
        evidence: "A union whose member types include the union itself. The group expects the\
                   circularity to be accepted; XSD 1.0 Part 1 §3.14.6 (src-type-cycle) forbids a\
                   simple type from being defined in terms of itself, which is the rule this\
                   engine reports.",
    },
    ExpectedFailure {
        id: "anyAttribute/s3_10_6ii01/schema",
        class: FailureClass::SuiteDefect,
        evidence: "The schema writes `notQName` on `xs:anyAttribute`, an XML Schema 1.1 attribute,\
                   and the group is graded under 1.0 because neither it nor its test set declares\
                   a version selector. A 1.0 processor must refuse it, which is what this engine\
                   does; the suite's expectation describes a 1.1 processor.",
    },
    ExpectedFailure {
        id: "anyAttribute/s3_10_6ii02/schema",
        class: FailureClass::SuiteDefect,
        evidence: "The schema writes `notNamespace` on `xs:anyAttribute`, an XML Schema 1.1\
                   attribute, and the group is graded under 1.0 because neither it nor its test\
                   set declares a version selector. A 1.0 processor must refuse it, which is what\
                   this engine does; the suite's expectation describes a 1.1 processor.",
    },
    ExpectedFailure {
        id: "anyAttribute/s3_10_6ii04/schema",
        class: FailureClass::SuiteDefect,
        evidence: "The schema writes `notNamespace` on `xs:anyAttribute`, an XML Schema 1.1\
                   attribute, and the group is graded under 1.0 because neither it nor its test\
                   set declares a version selector. A 1.0 processor must refuse it, which is what\
                   this engine does; the suite's expectation describes a 1.1 processor.",
    },
    ExpectedFailure {
        id: "cyclicRedefineIncludeImportOverride/s4_2_4si01/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a cyclic xs:redefine to be refused. The suite contradicts itself here:\
                   msData/schema's schU1 requires the identical shape to assemble, and it does.\
                   Accepting the cycle is a deliberate trade in favour of the group the suite\
                   grades the other way, not an unnoticed gap.",
    },
    ExpectedFailure {
        id: "cyclicRedefineIncludeImportOverride/s4_2_4si01b/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a cyclic xs:redefine to be refused. The suite contradicts itself here:\
                   msData/schema's schU1 requires the identical shape to assemble, and it does.\
                   Accepting the cycle is a deliberate trade in favour of the group the suite\
                   grades the other way, not an unnoticed gap.",
    },
    ExpectedFailure {
        id: "substitution-groups/sg-abstract-upa2/instance-7",
        class: FailureClass::SuiteDefect,
        evidence: "Unsatisfiable by construction: wgMeta/substitution-groups.testSet declares two\
                   instanceTests over the same href with opposite expectations — e1.xml invalid\
                   and e1bis.xml, the same document, valid. Exactly one slot must fail for any\
                   processor, and this engine is at that floor.",
    },
    ExpectedFailure {
        id: "wildcard/s3_10_1ii08/schema",
        class: FailureClass::SuiteDefect,
        evidence: "The schema writes `notQName` on `xs:any`, an XML Schema 1.1 attribute, and the\
                   group is graded under 1.0 because neither it nor its test set declares a\
                   version selector. Same mechanism as anyAttribute/s3_10_6ii01.",
    },
    ExpectedFailure {
        id: "wildcard/s3_10_1ii09/schema",
        class: FailureClass::SuiteDefect,
        evidence: "The schema writes `notQName` on `xs:any`, an XML Schema 1.1 attribute, and the\
                   group is graded under 1.0 because neither it nor its test set declares a\
                   version selector. Same mechanism as anyAttribute/s3_10_6ii01.",
    },
    ExpectedFailure {
        id: "introspection/introspection/schema",
        class: FailureClass::Dependency,
        evidence: "Retrieval. common/xsts.xsd writes <xsd:attribute ref=\"xlink:type\"\
                   default=\"locator\"/>, and the only source for that namespace is an\
                   <xsd:import> naming an absolute http: URI\
                   (http://www.w3.org/XML/2008/06/xlink.xsd). A filesystem resolver cannot read it\
                   and reports not-found rather than an error, so no XLink components exist and\
                   src-resolve refuses the reference. Measured with a stand-in XLink schema in\
                   place: xsts.xsd then assembles and all 91 instance documents the group names\
                   validate. No engine work sits behind the wall. Do not fix this by relaxing\
                   resolution; see the module documentation.",
    },
];

/// Slots that fail in the XSD 1.1 pass (`xsts/1.1`) — 29 of them.
///
/// Nine shorter than the 1.0 list, and every one of the differences is a rule that changed between
/// the two versions rather than a slot this pass does not reach: the `notQName`/`notNamespace`
/// wildcard groups and `attP031` are 1.1 syntax or 1.0-only rules, `missing001` … `missing006` and
/// `sg-abstract-upa2` are 1.0-versioned, and `particlesZ026a`, `particlesZ033_g` and
/// `defaultAttributesApply/s3_4_2_4si03` appear only here for the same reason in reverse.
pub const EXPECTED_FAILURES_1_1: &[ExpectedFailure] = &[
    ExpectedFailure {
        id: "DataTypes_w3c/anyURI_a004_1339/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Expects an xs:anyURI lexical form to be rejected. XSD 1.0's second edition made\
                   the anyURI lexical space effectively unconstrained — an erratum this suite\
                   predates — so a conformant processor now accepts every string, which is what\
                   this engine does.",
    },
    ExpectedFailure {
        id: "DataTypes_w3c/gMonth002_2061/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Expects `--03--` to be a valid xs:gMonth. That is the pre-erratum lexical form\
                   the XSD 1.0 second edition withdrew; the lexical space is `--MM`, which this\
                   engine implements.",
    },
    ExpectedFailure {
        id: "DataTypes_w3c/gMonth004_2063/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Expects `--05---05:00` to be a valid xs:gMonth — the same withdrawn pre-erratum\
                   lexical form as gMonth002_2061, with a timezone after it.",
    },
    ExpectedFailure {
        id: "Element_w3c/elemZ026/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects assembly to be refused for a complex-type derivation XSD 1.0 permits;\
                   the schema assembles here, and the group's own instance slots — which the suite\
                   grades against the schema it says must not exist — pass.",
    },
    ExpectedFailure {
        id: "Particles_w3c/particlesZ026a/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects assembly to be refused under a Unique Particle Attribution reading XSD\
                   1.1 Part 1 §3.8.6 relaxes, and which the suite's own 1.1 categories declare\
                   relaxed (common/XSD1_1TestCategories.xml, xsd1_1-Wildcards-RelaxationOfUPA).",
    },
    ExpectedFailure {
        id: "Particles_w3c/particlesZ033_g/schema",
        class: FailureClass::SuiteDefect,
        evidence: "The 1.0 verdict is already correct. msMeta/Particles_w3c.xml carries <expected\
                   validity=\"invalid\"/> with no version selector and a <current\
                   date=\"2006-07-16\"/> that predates XSD 1.1. Under 1.1 the only residual\
                   conflict is <xsd:element ref='m1'/> followed by <xsd:any/>, which the suite\
                   itself declares legal — common/XSD1_1TestCategories.xml names xsd1_1-Wildcards-\
                   RelaxationOfUPA, and ibmMeta/wildcard.testSet's s3_10_1v04 expects the\
                   identical shape to be valid.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ11/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Lu}` and the value the expectation requires it not to accept holds\
                   U+1D7A8, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ13/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Ll}` and the value the expectation requires it not to accept holds\
                   U+1D7C9, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ19/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Lo}` and the value the expectation requires it not to accept holds\
                   U+2FA1D, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ21/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{M}` and the value the expectation requires it not to accept holds U+1D1AD\
                   and U+1D172, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ23/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Mn}` and the value the expectation requires it not to accept holds\
                   U+1D1AD, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ25/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Mc}` and the value the expectation requires it not to accept holds\
                   U+1D172, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ29/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{N}` and the value the expectation requires it not to accept holds U+1D7FF,\
                   U+1034A and U+10323, outside the Basic Multilingual Plane. That expectation\
                   follows the original (2004) XSD 1.0 reading, under which a regular expression\
                   matches over UCS-2 code units and such a character is a surrogate pair no\
                   single-character escape can match; this engine matches over code points, which\
                   is what XSD 1.1 Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this\
                   set fail for that one reason and every one of them carries a supplementary-\
                   plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ31/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Nd}` and the value the expectation requires it not to accept holds\
                   U+1D7FF, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ33/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Nl}` and the value the expectation requires it not to accept holds\
                   U+1034A, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ35/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{No}` and the value the expectation requires it not to accept holds\
                   U+10323, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ61/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{S}` and the value the expectation requires it not to accept holds U+1D1DD,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ69/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{So}` and the value the expectation requires it not to accept holds\
                   U+1D1DD, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ75/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Cf}` and the value the expectation requires it not to accept holds\
                   U+E0078, outside the Basic Multilingual Plane. That expectation follows the\
                   original (2004) XSD 1.0 reading, under which a regular expression matches over\
                   UCS-2 code units and such a character is a surrogate pair no single-character\
                   escape can match; this engine matches over code points, which is what XSD 1.1\
                   Part 2 Appendix G and F&O 3.1 require. Eighteen groups of this set fail for\
                   that one reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reJ77/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\p{Co}` and the value the expectation requires it not to accept holds\
                   U+100000, U+F0000, U+FFFFD and U+10FFFD, outside the Basic Multilingual Plane.\
                   That expectation follows the original (2004) XSD 1.0 reading, under which a\
                   regular expression matches over UCS-2 code units and such a character is a\
                   surrogate pair no single-character escape can match; this engine matches over\
                   code points, which is what XSD 1.1 Part 2 Appendix G and F&O 3.1 require.\
                   Eighteen groups of this set fail for that one reason and every one of them\
                   carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reS21/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\d` and the value the expectation requires it not to accept holds U+1D7CE,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reS42/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\d` and the value the expectation requires it not to accept holds U+1D7FF,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reT63/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\D` and the value the expectation requires it not to accept holds U+1D7CD,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "Regex_w3c/reT84/instance-3",
        class: FailureClass::SuiteDefect,
        evidence: "Supplementary-plane characters in a character-class escape. The pattern is\
                   `\\D` and the value the expectation requires it not to accept holds U+1D800,\
                   outside the Basic Multilingual Plane. That expectation follows the original\
                   (2004) XSD 1.0 reading, under which a regular expression matches over UCS-2\
                   code units and such a character is a surrogate pair no single-character escape\
                   can match; this engine matches over code points, which is what XSD 1.1 Part 2\
                   Appendix G and F&O 3.1 require. Eighteen groups of this set fail for that one\
                   reason and every one of them carries a supplementary-plane value.",
    },
    ExpectedFailure {
        id: "SimpleType_w3c/ste110/schema",
        class: FailureClass::SuiteDefect,
        evidence: "A union whose member types include the union itself. The group expects the\
                   circularity to be accepted; XSD 1.0 Part 1 §3.14.6 (src-type-cycle) forbids a\
                   simple type from being defined in terms of itself, which is the rule this\
                   engine reports.",
    },
    ExpectedFailure {
        id: "cyclicRedefineIncludeImportOverride/s4_2_4si01/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a cyclic xs:redefine to be refused. The suite contradicts itself here:\
                   msData/schema's schU1 requires the identical shape to assemble, and it does.\
                   Accepting the cycle is a deliberate trade in favour of the group the suite\
                   grades the other way, not an unnoticed gap.",
    },
    ExpectedFailure {
        id: "cyclicRedefineIncludeImportOverride/s4_2_4si01b/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects a cyclic xs:redefine to be refused. The suite contradicts itself here:\
                   msData/schema's schU1 requires the identical shape to assemble, and it does.\
                   Accepting the cycle is a deliberate trade in favour of the group the suite\
                   grades the other way, not an unnoticed gap.",
    },
    ExpectedFailure {
        id: "defaultAttributesApply/s3_4_2_4si03/schema",
        class: FailureClass::SuiteDefect,
        evidence: "Expects assembly to be refused for an xs:defaultAttributesApply combination XSD\
                   1.1 Part 1 §3.4.2 permits; the schema assembles here.",
    },
    ExpectedFailure {
        id: "introspection/introspection/schema",
        class: FailureClass::Dependency,
        evidence: "Retrieval. common/xsts.xsd writes <xsd:attribute ref=\"xlink:type\"\
                   default=\"locator\"/>, and the only source for that namespace is an\
                   <xsd:import> naming an absolute http: URI\
                   (http://www.w3.org/XML/2008/06/xlink.xsd). A filesystem resolver cannot read it\
                   and reports not-found rather than an error, so no XLink components exist and\
                   src-resolve refuses the reference. Measured with a stand-in XLink schema in\
                   place: xsts.xsd then assembles and all 91 instance documents the group names\
                   validate. No engine work sits behind the wall. Do not fix this by relaxing\
                   resolution; see the module documentation.",
    },
];

/// Compares one XSTS report against the table for the version it ran under, in both directions.
///
/// The two passes are graded separately because they *are* separate runs over the same suite, with
/// different version selectors in force: a slot that fails only under 1.0 would look stale in the
/// 1.1 report and vice versa. A report whose suite name is neither pass is not comparable.
#[must_use]
pub fn audit(report: &SuiteReport) -> LedgerVerdict {
    let expected = match report.suite.as_str() {
        "xsts/1.0" => EXPECTED_FAILURES_1_0,
        "xsts/1.1" => EXPECTED_FAILURES_1_1,
        other => {
            return LedgerVerdict::Unreadable(format!(
                "`{other}` is neither of the two passes this ledger describes"
            ));
        }
    };
    crate::ledger::compare(report, expected, None)
}

/// Every distinct test group named in either pass, which is the count the release documentation
/// quotes beside the two slot counts.
#[must_use]
pub fn distinct_groups() -> usize {
    let mut groups: Vec<&str> = EXPECTED_FAILURES_1_0
        .iter()
        .chain(EXPECTED_FAILURES_1_1)
        .filter_map(|entry| entry.id.rsplit_once('/').map(|(group, _)| group))
        .collect();
    groups.sort_unstable();
    groups.dedup();
    groups.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::table_problems;

    fn report_failing(suite: &str, ids: &[&str]) -> SuiteReport {
        let mut report = SuiteReport::new(suite);
        for id in ids {
            report.record_fail(*id, "expected valid: a diagnosis");
        }
        report
    }

    #[test]
    fn both_tables_are_well_formed() {
        assert_eq!(table_problems(EXPECTED_FAILURES_1_0), Vec::<String>::new());
        assert_eq!(table_problems(EXPECTED_FAILURES_1_1), Vec::<String>::new());
    }

    /// Every id is `test-set/test-group/slot`, the three-part form
    /// [`super::run_group`](super::super::run_group) records a failing slot under, with the slot
    /// either `schema` or `instance-N`.
    #[test]
    fn every_id_is_written_the_way_a_failing_slot_is_recorded() {
        for entry in EXPECTED_FAILURES_1_0.iter().chain(EXPECTED_FAILURES_1_1) {
            let parts: Vec<&str> = entry.id.split('/').collect();
            assert_eq!(parts.len(), 3, "`{}` is not `set/group/slot`", entry.id);
            assert!(parts.iter().all(|part| !part.is_empty()), "{}", entry.id);
            assert!(
                parts[2] == "schema"
                    || parts[2]
                        .strip_prefix("instance-")
                        .is_some_and(|position| position.parse::<usize>().is_ok()),
                "`{}` names no slot this runner records",
                entry.id
            );
        }
    }

    /// The headline of the module documentation, held to by a test rather than by a sentence:
    /// nothing in this suite's residue is this workspace's own gap.
    #[test]
    fn no_slot_of_this_suite_is_an_engine_gap() {
        for entry in EXPECTED_FAILURES_1_0.iter().chain(EXPECTED_FAILURES_1_1) {
            assert!(
                matches!(
                    entry.class,
                    FailureClass::SuiteDefect | FailureClass::Dependency
                ),
                "`{}` is classed {}, and the claim this suite's residue holds no engine gap is \
                 either wrong or out of date",
                entry.id,
                entry.class
            );
        }
    }

    #[test]
    fn each_pass_is_graded_against_its_own_table() {
        let ids: Vec<&str> = EXPECTED_FAILURES_1_0.iter().map(|e| e.id).collect();
        assert_eq!(
            audit(&report_failing("xsts/1.0", &ids)),
            LedgerVerdict::Agreed(EXPECTED_FAILURES_1_0.len())
        );
        // The same failing set graded against the other pass's table is a disagreement, which is
        // the whole reason the two are separate.
        assert!(audit(&report_failing("xsts/1.1", &ids)).is_a_gate_failure());
    }

    #[test]
    fn a_report_from_neither_pass_is_not_comparable() {
        let verdict = audit(&report_failing("xsts/1.2", &[]));
        assert!(matches!(verdict, LedgerVerdict::Unreadable(_)), "{verdict}");
    }

    #[test]
    fn the_group_count_is_computed_from_the_tables() {
        assert!(distinct_groups() >= EXPECTED_FAILURES_1_1.len());
        assert!(distinct_groups() <= EXPECTED_FAILURES_1_0.len() + EXPECTED_FAILURES_1_1.len());
    }
}
