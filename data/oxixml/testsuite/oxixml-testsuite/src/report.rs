//! The shared report type every suite runner returns.
//!
//! A [`SuiteReport`] is deliberately dumb: it counts outcomes and remembers a bounded number of
//! failure descriptions. Nothing in this module knows how to discover a suite, parse a manifest,
//! or run a case — that is each runner module's job. This module only knows how to tally what a
//! runner found and to print the tally.

use std::collections::BTreeMap;
use std::fmt;

/// One failing test case, kept for a human to read in a failure list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseFailure {
    /// The manifest identifier of the case (a suite-relative name, not necessarily unique
    /// across suites).
    pub id: String,
    /// A human-readable description of how the case failed.
    pub message: String,
}

impl CaseFailure {
    /// Builds a failure record.
    pub fn new(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for CaseFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.message)
    }
}

/// The maximum number of [`CaseFailure`]s a report keeps.
///
/// A suite that fails wholesale (a compile error in the harness itself, say) must not produce an
/// unboundedly large report; the count fields stay exact regardless of the cap. The cap is set
/// high enough that every real suite's full failure list fits, so gap analysis never truncates.
pub const MAX_RECORDED_FAILURES: usize = 10_000;

/// The outcome of running one conformance suite (or one facet of a suite, such as one QT3
/// test-set family or one JSON-LD entry point).
///
/// `total` is always `passed + failed + skipped`; the four counts are kept in lock-step by the
/// `record_*` methods rather than computed on demand, so that a partially-built report is never
/// inconsistent.
#[derive(Debug, Clone)]
pub struct SuiteReport {
    /// The suite's report name, e.g. `"xmlconf/valid"` or `"qt3-xpath"`.
    pub suite: String,
    /// `passed + failed + skipped`.
    pub total: usize,
    /// Cases whose actual outcome matched the manifest's expectation.
    pub passed: usize,
    /// Cases whose actual outcome did not match the manifest's expectation.
    pub failed: usize,
    /// Cases the runner deliberately did not attempt (see `skip_reasons` for why).
    pub skipped: usize,
    /// How many cases were skipped for each reason, keyed by a short human-readable label.
    pub skip_reasons: BTreeMap<String, usize>,
    /// Up to [`MAX_RECORDED_FAILURES`] descriptions of the failing cases.
    pub failures: Vec<CaseFailure>,
    /// `Some(reason)` when the suite itself could not be discovered at all — the environment
    /// variable that names its root was unset, or the expected layout was not found under it.
    ///
    /// An unavailable suite always has every count at zero; it is a distinct state from "the
    /// suite was found and had zero applicable cases", which is a legitimate (if unusual)
    /// [`SuiteReport::new`] outcome.
    pub unavailable: Option<String>,
}

impl SuiteReport {
    /// Starts an empty, available report for `suite`.
    #[must_use]
    pub fn new(suite: impl Into<String>) -> Self {
        Self {
            suite: suite.into(),
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            skip_reasons: BTreeMap::new(),
            failures: Vec::new(),
            unavailable: None,
        }
    }

    /// Builds a report for a suite that could not even be located, e.g. because the environment
    /// variable that names its root is unset.
    #[must_use]
    pub fn unavailable(suite: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            unavailable: Some(reason.into()),
            ..Self::new(suite)
        }
    }

    /// Whether the suite was actually discovered and (possibly partially) run.
    ///
    /// The reciprocal of [`SuiteReport::unavailable`] being `Some`, spelled out for callers that
    /// would rather not match on the field directly.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.unavailable.is_none()
    }

    /// Records one passing case.
    pub fn record_pass(&mut self) {
        self.passed += 1;
        self.total += 1;
    }

    /// Records one failing case, keeping at most [`MAX_RECORDED_FAILURES`] descriptions.
    pub fn record_fail(&mut self, id: impl Into<String>, message: impl Into<String>) {
        self.failed += 1;
        self.total += 1;
        if self.failures.len() < MAX_RECORDED_FAILURES {
            self.failures.push(CaseFailure::new(id, message));
        }
    }

    /// Records one deliberately-skipped case under `reason`.
    pub fn record_skip(&mut self, reason: impl Into<String>) {
        self.skipped += 1;
        self.total += 1;
        *self.skip_reasons.entry(reason.into()).or_insert(0) += 1;
    }

    /// Folds `other`'s counts and failures into `self`, prefixing `other`'s failure ids with
    /// `other.suite` so that a merged report's failure list stays traceable to its source.
    ///
    /// Used by runners that drive several sub-suites (e.g. one QT3 test-set family per call)
    /// but want to publish a single combined [`SuiteReport`] alongside the per-family ones.
    pub fn merge(&mut self, other: &Self) {
        self.total += other.total;
        self.passed += other.passed;
        self.failed += other.failed;
        self.skipped += other.skipped;
        for (reason, count) in &other.skip_reasons {
            *self.skip_reasons.entry(reason.clone()).or_insert(0) += count;
        }
        for failure in &other.failures {
            if self.failures.len() >= MAX_RECORDED_FAILURES {
                break;
            }
            self.failures.push(CaseFailure::new(
                format!("{}/{}", other.suite, failure.id),
                failure.message.clone(),
            ));
        }
    }

    /// The number of cases actually judged, i.e. excluding the ones skipped.
    #[must_use]
    pub fn attempted(&self) -> usize {
        self.passed + self.failed
    }

    /// The percentage of attempted cases that passed, or `None` when nothing was attempted
    /// (either the suite is unavailable, or every discovered case was skipped).
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "case counts never approach 2^53"
    )]
    pub fn pass_rate(&self) -> Option<f64> {
        let attempted = self.attempted();
        if attempted == 0 {
            None
        } else {
            Some(self.passed as f64 * 100.0 / attempted as f64)
        }
    }
}

impl fmt::Display for SuiteReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.unavailable {
            Some(reason) => write!(f, "{}: unavailable ({reason})", self.suite),
            None => write!(
                f,
                "{}: {}/{} passed, {} failed, {} skipped ({})",
                self.suite,
                self.passed,
                self.attempted(),
                self.failed,
                self.skipped,
                match self.pass_rate() {
                    Some(rate) => format!("{rate:.1}%"),
                    None => "n/a".to_owned(),
                }
            ),
        }
    }
}

/// A collection of [`SuiteReport`]s, printed as one aligned table.
///
/// This is what the `report` binary and the aggregate `#[test]`s work with: a single value that
/// prints the whole workspace's conformance picture in one shot.
#[derive(Debug, Clone, Default)]
pub struct ConformanceReport(pub Vec<SuiteReport>);

impl ConformanceReport {
    /// An empty report.
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Appends one suite's report.
    pub fn push(&mut self, report: SuiteReport) {
        self.0.push(report);
    }

    /// Appends every report of `reports`, in order.
    pub fn extend(&mut self, reports: impl IntoIterator<Item = SuiteReport>) {
        self.0.extend(reports);
    }

    /// The reports for suites that were actually discovered and run.
    pub fn available(&self) -> impl Iterator<Item = &SuiteReport> {
        self.0.iter().filter(|report| report.is_available())
    }

    /// The reports for suites that could not be discovered at all.
    pub fn unavailable(&self) -> impl Iterator<Item = &SuiteReport> {
        self.0.iter().filter(|report| !report.is_available())
    }

    /// The sum of every available suite's `failed` count — the number an automated caller
    /// should treat as "did anything regress".
    #[must_use]
    pub fn total_failed(&self) -> usize {
        self.available().map(|report| report.failed).sum()
    }

    /// The sum of every available suite's `passed` count.
    #[must_use]
    pub fn total_passed(&self) -> usize {
        self.available().map(|report| report.passed).sum()
    }

    /// The sum of every available suite's `attempted()` count.
    #[must_use]
    pub fn total_attempted(&self) -> usize {
        self.available().map(SuiteReport::attempted).sum()
    }

    /// A per-suite tally of *why* cases were skipped, most frequent reason first.
    ///
    /// A skip count on its own says nothing: a case skipped because the catalogue declares a
    /// dependency this workspace does not implement is accounted for, while one skipped because
    /// the runner could not check an assertion kind is a verdict nobody knows. The two are
    /// indistinguishable in the summary table and are exactly what this breaks apart, so that a
    /// reader of the report can see which skips are gaps in the *measurement* rather than in the
    /// implementation. Suites that skipped nothing are omitted; an empty string means nothing
    /// anywhere was skipped.
    #[must_use]
    pub fn skip_breakdown(&self) -> String {
        use fmt::Write;

        let mut out = String::new();
        for report in self.available() {
            if report.skip_reasons.is_empty() {
                continue;
            }
            let mut reasons: Vec<(&String, &usize)> = report.skip_reasons.iter().collect();
            // Most frequent first, ties broken by the reason itself so the order is stable.
            reasons.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
            let _ = writeln!(out, "{} — {} skipped:", report.suite, report.skipped);
            for (reason, count) in reasons {
                let _ = writeln!(out, "  {count:>8}  {reason}");
            }
        }
        out
    }
}

impl FromIterator<SuiteReport> for ConformanceReport {
    fn from_iter<T: IntoIterator<Item = SuiteReport>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl fmt::Display for ConformanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name_width = self
            .0
            .iter()
            .map(|report| report.suite.len())
            .max()
            .unwrap_or(5)
            .max("Suite".len());

        writeln!(
            f,
            "{:<name_width$}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}",
            "Suite", "Total", "Passed", "Failed", "Skipped", "Pass%"
        )?;
        let rule_width = name_width + 2 + 8 * 4 + 2 * 4 + 7;
        writeln!(f, "{:-<rule_width$}", "")?;

        for report in &self.0 {
            if let Some(reason) = &report.unavailable {
                writeln!(
                    f,
                    "{:<name_width$}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}  ({reason})",
                    report.suite, "-", "-", "-", "-", "-"
                )?;
                continue;
            }
            let rate = match report.pass_rate() {
                Some(rate) => format!("{rate:.1}%"),
                None => "n/a".to_owned(),
            };
            writeln!(
                f,
                "{:<name_width$}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}",
                report.suite, report.total, report.passed, report.failed, report.skipped, rate
            )?;
        }

        writeln!(f, "{:-<rule_width$}", "")?;
        let total_attempted = self.total_attempted();
        let total_passed = self.total_passed();
        #[allow(
            clippy::cast_precision_loss,
            reason = "case counts never approach 2^53"
        )]
        let overall_rate = if total_attempted == 0 {
            "n/a".to_owned()
        } else {
            format!(
                "{:.1}%",
                total_passed as f64 * 100.0 / total_attempted as f64
            )
        };
        writeln!(
            f,
            "{:<name_width$}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}",
            "TOTAL",
            self.available().map(|r| r.total).sum::<usize>(),
            total_passed,
            self.total_failed(),
            self.available().map(|r| r.skipped).sum::<usize>(),
            overall_rate
        )?;

        let unavailable: Vec<&str> = self
            .unavailable()
            .map(|report| report.suite.as_str())
            .collect();
        if !unavailable.is_empty() {
            writeln!(f, "\nunavailable suites: {}", unavailable.join(", "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_methods_keep_total_in_lock_step() {
        let mut report = SuiteReport::new("demo");
        report.record_pass();
        report.record_pass();
        report.record_fail("case-1", "boom");
        report.record_skip("unsupported feature");
        report.record_skip("unsupported feature");

        assert_eq!(report.total, 5);
        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 1);
        assert_eq!(report.skipped, 2);
        assert_eq!(report.skip_reasons.get("unsupported feature"), Some(&2));
        assert_eq!(report.attempted(), 3);
        let rate = report.pass_rate().expect("cases were attempted");
        assert!((rate - 66.666_666_666).abs() < 1e-6);
    }

    #[test]
    fn unavailable_report_has_no_pass_rate() {
        let report = SuiteReport::unavailable("demo", "OXIXML_DEMO_DIR is not set");
        assert!(!report.is_available());
        assert_eq!(report.pass_rate(), None);
        assert_eq!(report.total, 0);
        assert_eq!(
            format!("{report}"),
            "demo: unavailable (OXIXML_DEMO_DIR is not set)"
        );
    }

    #[test]
    fn empty_but_available_report_has_no_pass_rate() {
        let report = SuiteReport::new("demo");
        assert!(report.is_available());
        assert_eq!(report.pass_rate(), None);
    }

    #[test]
    fn failure_cap_stops_growing_but_counts_stay_exact() {
        let mut report = SuiteReport::new("demo");
        for i in 0..(MAX_RECORDED_FAILURES + 50) {
            report.record_fail(format!("case-{i}"), "boom");
        }
        assert_eq!(report.failed, MAX_RECORDED_FAILURES + 50);
        assert_eq!(report.failures.len(), MAX_RECORDED_FAILURES);
    }

    #[test]
    fn merge_prefixes_failure_ids_with_the_source_suite() {
        let mut combined = SuiteReport::new("rdf-tests");
        let mut turtle = SuiteReport::new("rdf-tests/turtle");
        turtle.record_pass();
        turtle.record_fail("t01", "graphs differ");
        combined.merge(&turtle);

        assert_eq!(combined.passed, 1);
        assert_eq!(combined.failed, 1);
        assert_eq!(combined.failures[0].id, "rdf-tests/turtle/t01");
    }

    #[test]
    fn conformance_report_renders_an_aligned_table() {
        let mut passing = SuiteReport::new("alpha");
        passing.record_pass();
        passing.record_pass();
        passing.record_fail("a1", "mismatch");

        let missing = SuiteReport::unavailable("beta", "OXIXML_BETA_DIR is not set");

        let report: ConformanceReport = [passing, missing].into_iter().collect();
        let rendered = report.to_string();
        assert!(rendered.contains("alpha"));
        assert!(rendered.contains("beta"));
        assert!(rendered.contains("unavailable suites: beta"));
        assert_eq!(report.total_failed(), 1);
        assert_eq!(report.total_passed(), 2);
        assert_eq!(report.total_attempted(), 3);
    }

    #[test]
    fn skip_breakdown_lists_each_reason_most_frequent_first() {
        let mut suite = SuiteReport::new("alpha");
        suite.record_pass();
        suite.record_skip("dependency:streaming");
        suite.record_skip("dependency:streaming");
        suite.record_skip("unsupported-assertion:assert-warning");
        let quiet = SuiteReport::new("beta");

        let report: ConformanceReport = [suite, quiet].into_iter().collect();
        let breakdown = report.skip_breakdown();
        assert!(breakdown.starts_with("alpha — 3 skipped:"), "{breakdown}");
        let streaming = breakdown
            .find("dependency:streaming")
            .expect("the frequent reason is listed");
        let assertion = breakdown
            .find("unsupported-assertion:assert-warning")
            .expect("the rare reason is listed");
        assert!(streaming < assertion, "most frequent first: {breakdown}");
        assert!(
            !breakdown.contains("beta"),
            "a suite that skipped nothing is omitted: {breakdown}"
        );
    }

    #[test]
    fn skip_breakdown_is_empty_when_nothing_was_skipped() {
        let mut suite = SuiteReport::new("alpha");
        suite.record_pass();
        let report: ConformanceReport = [suite].into_iter().collect();
        assert!(report.skip_breakdown().is_empty());
    }

    #[test]
    fn conformance_report_handles_the_empty_case() {
        let report = ConformanceReport::new();
        // Must not panic on the width computation or divide by zero on the overall rate.
        let rendered = report.to_string();
        assert!(rendered.contains("TOTAL"));
    }
}
