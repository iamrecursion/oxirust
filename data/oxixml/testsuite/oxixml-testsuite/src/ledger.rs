//! The shared vocabulary for an **expected-failure ledger**: a suite's complete residue, named
//! case by case, with the class each failure belongs to and the evidence behind that class.
//!
//! A conformance percentage is a summary, and a summary is not a claim anybody can check. A ledger
//! is the claim. Each suite that has one keeps its own table — [`crate::xslt30::ledger`] for the
//! XSLT 3.0 catalogue, [`crate::xsts`] for the XML Schema suite — and this module holds what they
//! have in common: the class vocabulary, the comparison, and what the comparison can conclude.
//!
//! # Equality, in both directions
//!
//! [`compare`] does not ask whether the failures are a subset of the table. It asks whether the two
//! are **equal**, and reports both differences:
//!
//! * a case that fails and is not in the table is a **regression**;
//! * a case that is in the table and does not fail is a **stale entry**, and it fails the gate just
//!   as loudly. An entry nobody removed is a claim that the processor has a defect it no longer
//!   has, and a table allowed to over-state its own failures is one nobody can trust to under-state
//!   them either.
//!
//! That symmetry is the whole difference between a ledger and an allowlist.

use std::collections::BTreeSet;
use std::fmt;

use crate::report::{CaseFailure, MAX_RECORDED_FAILURES, SuiteReport};

/// Why a ledgered case does not pass.
///
/// The order is the order the classes are reported in, and it runs from "not this processor's to
/// pass" to "this processor's backlog".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailureClass {
    /// The case, or the expectation it carries, is wrong or unsatisfiable — established by a
    /// published reference-processor record, by the specification read against the case's own
    /// files, or by a conflict the suite states itself.
    SuiteDefect,
    /// The case needs a capability this processor declares it does not have, or a resource (the
    /// network, a document that is not in the checkout) the run cannot reach.
    Dependency,
    /// The case's expectation is pinned to a Unicode version older than the one this workspace's
    /// character tables implement.
    UnicodeVersion,
    /// The case does not finish inside the runner's per-case wall-clock budget.
    Performance,
    /// A genuine gap in this workspace. Roadmap work, named rather than excused.
    EngineGap,
}

impl FailureClass {
    /// The label this class is reported under.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SuiteDefect => "suite-defect",
            Self::Dependency => "dependency",
            Self::UnicodeVersion => "unicode-version",
            Self::Performance => "performance",
            Self::EngineGap => "engine-gap",
        }
    }

    /// Every class, in reporting order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::SuiteDefect,
            Self::Dependency,
            Self::UnicodeVersion,
            Self::Performance,
            Self::EngineGap,
        ]
    }
}

impl fmt::Display for FailureClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One ledgered case: what fails, which class it belongs to, and why that class is the right one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedFailure {
    /// The identifier the runner records the failure under, spelled exactly as the runner spells
    /// it — an id in any other shape can never match, so the gate would report it stale for ever.
    pub id: &'static str,
    /// Which of the five kinds of not-passing this is.
    pub class: FailureClass,
    /// The evidence for that class: a reference-processor record, a specification citation, or the
    /// mechanism and the file it lives in.
    pub evidence: &'static str,
}

/// How a run's failing set differs from a ledger.
///
/// Both directions are recorded and both are failures of the gate — see the module documentation
/// for why a stale entry is treated as seriously as a regression.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Discrepancy {
    /// Cases that failed and are not in the ledger.
    pub unexpected: Vec<CaseFailure>,
    /// Ledger entries whose case did not fail in this run.
    pub stale: Vec<&'static str>,
}

impl Discrepancy {
    /// Whether the run and the ledger agree exactly.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.unexpected.is_empty() && self.stale.is_empty()
    }

    /// Whether `message` reads as a case that ran out of the runner's per-case budget rather than
    /// as a wrong answer.
    ///
    /// `XPDY0130` is the engine's own report of the same fact — the deadline the driver declared
    /// fired a hair before the driver stopped waiting — so the two spellings are one condition.
    fn is_a_budget_failure(message: &str) -> bool {
        message.contains("TIMEOUT") || message.contains("XPDY0130")
    }
}

impl fmt::Display for Discrepancy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("the failing set is exactly the ledger");
        }
        if !self.unexpected.is_empty() {
            writeln!(
                f,
                "{} case(s) failed that the ledger does not name:",
                self.unexpected.len()
            )?;
            for failure in &self.unexpected {
                writeln!(f, "  + {failure}")?;
            }
            if self
                .unexpected
                .iter()
                .any(|failure| Self::is_a_budget_failure(&failure.message))
            {
                writeln!(
                    f,
                    "  note: at least one of those ran out of the per-case budget. A case whose \
                     cost is a large fraction of the budget measures the machine as much as the \
                     processor. Re-drive its test set on its own before reading this as a \
                     regression, and do not ledger a case that passes when driven alone."
                )?;
            }
        }
        if !self.stale.is_empty() {
            writeln!(
                f,
                "{} ledger entr(y/ies) name a case that no longer fails — remove them:",
                self.stale.len()
            )?;
            for id in &self.stale {
                writeln!(f, "  - {id}")?;
            }
        }
        Ok(())
    }
}

/// What comparing a run against a ledger established.
///
/// A comparison that could not be made must never look like one that succeeded, which is why the
/// two ways of not comparing are variants of their own rather than an empty [`Discrepancy`] — and
/// why they are *two* variants rather than one. They call for opposite responses: a narrowed run is
/// a deliberate triage mode and simply has no failing set the ledger describes, while a report that
/// cannot be read as its run's failing set is a fault in the instrument and must stop whatever is
/// reading it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerVerdict {
    /// The run's failing set is exactly the ledger's.
    Agreed(usize),
    /// The run and the ledger differ, in one direction or both.
    Disagreed(Discrepancy),
    /// The run was narrowed to part of its suite, so its failing set is a *subset* of the whole
    /// one by construction: comparing it would report every unselected entry as stale. Not a gate
    /// failure — there is no gate to fail.
    Narrowed(String),
    /// The report cannot be read as the run's failing set at all.
    Unreadable(String),
}

impl LedgerVerdict {
    /// Whether this verdict means the run and the ledger were compared and disagreed, or could not
    /// be compared for a reason that is the instrument's fault.
    ///
    /// [`LedgerVerdict::Narrowed`] is the one verdict that is neither: it says the run was not the
    /// kind of run the ledger describes.
    #[must_use]
    pub const fn is_a_gate_failure(&self) -> bool {
        matches!(self, Self::Disagreed(_) | Self::Unreadable(_))
    }
}

impl fmt::Display for LedgerVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Agreed(count) => {
                write!(f, "the failing set is exactly the ledger's {count} case(s)")
            }
            Self::Disagreed(discrepancy) => discrepancy.fmt(f),
            Self::Narrowed(reason) => write!(f, "no ledger comparison was made: {reason}"),
            Self::Unreadable(reason) => write!(f, "the report is not a failing set: {reason}"),
        }
    }
}

/// Compares `report`'s failing cases against `expected`, in both directions.
///
/// `narrowed` is the reason the run covered only part of its suite, when it did: a caller that can
/// tell (the XSLT catalogue runner reads its own `catalogue:not-selected` skips) passes it, and the
/// comparison is declined rather than made against a table that describes the whole suite.
#[must_use]
pub fn compare(
    report: &SuiteReport,
    expected: &[ExpectedFailure],
    narrowed: Option<String>,
) -> LedgerVerdict {
    if let Some(reason) = report.unavailable.as_ref() {
        return LedgerVerdict::Unreadable(format!("the suite was not discovered ({reason})"));
    }
    if let Some(reason) = narrowed {
        return LedgerVerdict::Narrowed(reason);
    }
    if report.failures.len() != report.failed {
        return LedgerVerdict::Unreadable(format!(
            "the report counted {} failure(s) but kept {} description(s) — the list is truncated \
             at MAX_RECORDED_FAILURES ({MAX_RECORDED_FAILURES}) and is not the failing set",
            report.failed,
            report.failures.len()
        ));
    }
    let ledgered: BTreeSet<&str> = expected.iter().map(|entry| entry.id).collect();
    let failing: BTreeSet<&str> = report
        .failures
        .iter()
        .map(|failure| failure.id.as_str())
        .collect();
    let discrepancy = Discrepancy {
        unexpected: report
            .failures
            .iter()
            .filter(|failure| !ledgered.contains(failure.id.as_str()))
            .cloned()
            .collect(),
        stale: expected
            .iter()
            .map(|entry| entry.id)
            .filter(|id| !failing.contains(id))
            .collect(),
    };
    if discrepancy.is_empty() {
        LedgerVerdict::Agreed(expected.len())
    } else {
        LedgerVerdict::Disagreed(discrepancy)
    }
}

/// How many entries of `expected` each class holds, in reporting order.
///
/// The figure the release documentation quotes beside a conformance line, computed from the table
/// rather than kept beside it.
#[must_use]
pub fn tally_by_class(expected: &[ExpectedFailure]) -> Vec<(FailureClass, usize)> {
    FailureClass::all()
        .iter()
        .map(|class| {
            (
                *class,
                expected
                    .iter()
                    .filter(|entry| entry.class == *class)
                    .count(),
            )
        })
        .collect()
}

/// Everything wrong with `expected` as a *table*, independently of any run: duplicate ids, an
/// entry filed out of order, an entry with no evidence worth the name.
///
/// Written as a function rather than as a test so that every suite's table is held to it by a test
/// of its own, and so that the rules are stated once. An empty answer means the table is well
/// formed.
#[must_use]
pub fn table_problems(expected: &[ExpectedFailure]) -> Vec<String> {
    let mut problems = Vec::new();
    let mut seen = BTreeSet::new();
    for entry in expected {
        if entry.id.is_empty() {
            problems.push("an entry has an empty id".to_owned());
        }
        if !seen.insert(entry.id) {
            problems.push(format!("`{}` is ledgered more than once", entry.id));
        }
        // An entry with no evidence is an exemption, which is the one thing a ledger must not be.
        if entry.evidence.len() <= 40 {
            problems.push(format!(
                "`{}` carries no evidence worth the name: {:?}",
                entry.id, entry.evidence
            ));
        }
        // A doubled space is how a continuation line that lost its trailing backslash shows up.
        if entry.evidence.contains("  ") {
            problems.push(format!("`{}`'s evidence has a doubled space", entry.id));
        }
    }
    // Grouped by class in reporting order, sorted by id inside each group: what lets a reader take
    // in one class at a time and still find an id by eye. An entry appended to the end rather than
    // filed under its class shows up here.
    let keys: Vec<(FailureClass, &str)> = expected
        .iter()
        .map(|entry| (entry.class, entry.id))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    if keys != sorted {
        problems.push(
            "the table is not grouped by class and sorted by id within each group".to_owned(),
        );
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &[ExpectedFailure] = &[
        ExpectedFailure {
            id: "set/first",
            class: FailureClass::SuiteDefect,
            evidence: "The suite's own reference results record this case as passing after a \
                       normalization its schema licenses nowhere.",
        },
        ExpectedFailure {
            id: "set/second",
            class: FailureClass::EngineGap,
            evidence: "A construct this workspace's compiler declines, named in the backlog \
                       rather than excused.",
        },
    ];

    fn report_failing(ids: &[&str]) -> SuiteReport {
        let mut report = SuiteReport::new("suite");
        for id in ids {
            report.record_fail(*id, "a failure message");
        }
        report
    }

    #[test]
    fn a_well_formed_table_has_no_problems() {
        assert_eq!(table_problems(TABLE), Vec::<String>::new());
    }

    #[test]
    fn a_duplicate_entry_is_a_problem() {
        let doubled = [TABLE[0], TABLE[0]];
        assert_eq!(table_problems(&doubled).len(), 1);
        assert!(table_problems(&doubled)[0].contains("more than once"));
    }

    #[test]
    fn an_entry_with_no_evidence_is_a_problem() {
        let bare = [ExpectedFailure {
            evidence: "known issue",
            ..TABLE[0]
        }];
        assert!(table_problems(&bare)[0].contains("no evidence"));
    }

    #[test]
    fn a_table_filed_out_of_order_is_a_problem() {
        let reversed = [TABLE[1], TABLE[0]];
        assert!(table_problems(&reversed)[0].contains("grouped by class"));
    }

    #[test]
    fn a_run_that_matches_the_table_agrees_with_it() {
        let verdict = compare(&report_failing(&["set/first", "set/second"]), TABLE, None);
        assert_eq!(verdict, LedgerVerdict::Agreed(2));
        assert!(!verdict.is_a_gate_failure());
    }

    #[test]
    fn a_case_that_fails_and_is_not_ledgered_is_a_regression() {
        let verdict = compare(
            &report_failing(&["set/first", "set/second", "set/third"]),
            TABLE,
            None,
        );
        let LedgerVerdict::Disagreed(discrepancy) = &verdict else {
            panic!("{verdict}");
        };
        assert!(discrepancy.stale.is_empty());
        assert_eq!(discrepancy.unexpected.len(), 1);
        assert_eq!(discrepancy.unexpected[0].id, "set/third");
        assert!(verdict.is_a_gate_failure());
    }

    /// The half a gate is usually written without, and the half that keeps a table honest.
    #[test]
    fn a_ledgered_case_that_passes_is_a_stale_entry() {
        let verdict = compare(&report_failing(&["set/first"]), TABLE, None);
        let LedgerVerdict::Disagreed(discrepancy) = &verdict else {
            panic!("{verdict}");
        };
        assert!(discrepancy.unexpected.is_empty());
        assert_eq!(discrepancy.stale, vec!["set/second"]);
        assert!(verdict.to_string().contains("no longer fails"));
        assert!(verdict.is_a_gate_failure());
    }

    /// A budget failure is the one an eight-core machine shared with other work produces, so the
    /// report names the remedy rather than leaving the next reader to rediscover it.
    #[test]
    fn an_unexpected_budget_failure_names_its_remedy() {
        let mut report = report_failing(&["set/first", "set/second"]);
        report.record_fail("set/heavy", "TIMEOUT: did not finish within 30s");
        let reported = compare(&report, TABLE, None).to_string();
        assert!(reported.contains("set/heavy"), "{reported}");
        assert!(reported.contains("driven alone"), "{reported}");
    }

    #[test]
    fn a_narrowed_run_is_not_compared_and_is_not_a_gate_failure() {
        let verdict = compare(
            &report_failing(&["set/first"]),
            TABLE,
            Some("the run passed over 400 case(s)".to_owned()),
        );
        assert!(matches!(verdict, LedgerVerdict::Narrowed(_)), "{verdict}");
        assert!(!verdict.is_a_gate_failure(), "{verdict}");
    }

    /// A truncated failure list is a subset too, but for a reason that is the instrument's fault
    /// and that nothing in the report's counts makes obvious — so it *is* a gate failure.
    #[test]
    fn a_truncated_failure_list_is_a_gate_failure() {
        let mut report = report_failing(&["set/first", "set/second"]);
        report.failed += 1;
        let verdict = compare(&report, TABLE, None);
        assert!(matches!(verdict, LedgerVerdict::Unreadable(_)), "{verdict}");
        assert!(verdict.to_string().contains("truncated"), "{verdict}");
        assert!(verdict.is_a_gate_failure());
    }

    #[test]
    fn an_undiscovered_suite_is_a_gate_failure() {
        let verdict = compare(&SuiteReport::unavailable("suite", "not found"), TABLE, None);
        assert!(matches!(verdict, LedgerVerdict::Unreadable(_)), "{verdict}");
        assert!(verdict.is_a_gate_failure());
    }

    #[test]
    fn the_class_tally_accounts_for_every_entry() {
        let tally = tally_by_class(TABLE);
        assert_eq!(tally.len(), FailureClass::all().len());
        assert_eq!(
            tally.iter().map(|(_, count)| count).sum::<usize>(),
            TABLE.len()
        );
        assert_eq!(tally[0], (FailureClass::SuiteDefect, 1));
        assert_eq!(tally[4], (FailureClass::EngineGap, 1));
    }
}
