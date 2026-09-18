//! [`SchematronReport`] — the structured result of [`crate::Compiled::validate`], with an
//! [`SVRL`](https://schematron.com/document/321.html) serialization.

use std::io::{self, Write};

use oxixml_xml::events::BytesText;
use oxixml_xml::writer::Writer;

use crate::model::SVRL_NS;

/// Whether a firing check is a failed requirement or a positive observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckOutcome {
    /// A `sch:assert` whose test evaluated to `false`.
    FailedAssert,
    /// A `sch:report` whose test evaluated to `true`.
    SuccessfulReport,
}

impl CheckOutcome {
    /// The SVRL element name this outcome serializes as.
    #[must_use]
    pub const fn svrl_element_name(self) -> &'static str {
        match self {
            Self::FailedAssert => "svrl:failed-assert",
            Self::SuccessfulReport => "svrl:successful-report",
        }
    }
}

/// One `sch:assert` or `sch:report` that fired: a failed assertion or a successful report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckResult {
    /// Whether this is a failed assertion or a successful report.
    pub outcome: CheckOutcome,
    /// The check's `id`, if it had one.
    pub id: Option<String>,
    /// The check's `role`, inherited from its rule when the check did not set its own.
    pub role: Option<String>,
    /// The check's `flag`, inherited from its rule when the check did not set its own.
    pub flag: Option<String>,
    /// The `test` expression's source text.
    pub test: String,
    /// The SVRL-style absolute path of the node the message pertains to — the check's
    /// `subject`, if it had one and it selected a node, otherwise the rule's context node.
    pub location: String,
    /// The interpolated message.
    pub message: String,
    /// The rendered text of every `sch:diagnostic` this check referenced, in the order they
    /// were listed in `diagnostics`.
    pub diagnostics: Vec<String>,
}

/// One context node a pattern's rules fired on, with the checks that fired there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FiredRule {
    /// The firing rule's `id`, if it had one.
    pub rule_id: Option<String>,
    /// The rule's `context` attribute, as written in the schema.
    pub context: String,
    /// The rule's `role`, if it had one.
    pub role: Option<String>,
    /// The rule's `flag`, if it had one.
    pub flag: Option<String>,
    /// The SVRL-style absolute path of the node the rule fired on.
    pub location: String,
    /// The checks that fired at this context node, in schema order.
    pub checks: Vec<CheckResult>,
}

/// One active pattern's results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivePatternReport {
    /// The pattern's `id`, if it had one.
    pub pattern_id: Option<String>,
    /// The pattern's `sch:title`, if it had one.
    pub title: Option<String>,
    /// Every context node the pattern's rules fired on, in document order.
    pub fired_rules: Vec<FiredRule>,
}

/// The structured result of validating one document against a [`crate::Compiled`] schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchematronReport {
    /// The schema's title, carried through for [`SchematronReport::to_svrl`].
    pub title: Option<String>,
    /// The phase that was validated, `None` for `#ALL`.
    pub phase: Option<String>,
    /// Every active pattern's results, in schema order.
    pub active_patterns: Vec<ActivePatternReport>,
}

impl SchematronReport {
    /// Every [`CheckResult`], across every pattern and rule, in validation order.
    pub fn checks(&self) -> impl Iterator<Item = &CheckResult> {
        self.active_patterns
            .iter()
            .flat_map(|pattern| pattern.fired_rules.iter())
            .flat_map(|rule| rule.checks.iter())
    }

    /// Every failed assertion.
    pub fn failed_asserts(&self) -> impl Iterator<Item = &CheckResult> {
        self.checks()
            .filter(|check| check.outcome == CheckOutcome::FailedAssert)
    }

    /// Every successful report.
    pub fn successful_reports(&self) -> impl Iterator<Item = &CheckResult> {
        self.checks()
            .filter(|check| check.outcome == CheckOutcome::SuccessfulReport)
    }

    /// Whether the document is valid: no `sch:assert` failed.
    ///
    /// A firing `sch:report` never makes a document invalid — it is an observation, not a
    /// requirement, exactly as [ISO/IEC 19757-3](https://www.iso.org/standard/55982.html)
    /// defines it.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.failed_asserts().next().is_none()
    }

    /// Serializes this report as an [SVRL](https://schematron.com/document/321.html)
    /// `svrl:schematron-output` document.
    ///
    /// Serialization to an in-memory buffer cannot fail; a malformed result is impossible by
    /// construction, so this returns `String` rather than a `Result`.
    #[must_use]
    pub fn to_svrl(&self) -> String {
        match self.write_svrl() {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(_) => String::new(),
        }
    }

    fn write_svrl(&self) -> io::Result<Vec<u8>> {
        let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);
        let mut root = writer
            .create_element("svrl:schematron-output")
            .with_attribute(("xmlns:svrl", SVRL_NS));
        if let Some(title) = self.title.as_deref() {
            root = root.with_attribute(("title", title));
        }
        if let Some(phase) = self.phase.as_deref() {
            root = root.with_attribute(("phase", phase));
        }
        root.write_inner_content(|writer| write_active_patterns(writer, &self.active_patterns))?;
        Ok(writer.into_inner())
    }
}

fn write_active_patterns<W: Write>(
    writer: &mut Writer<W>,
    patterns: &[ActivePatternReport],
) -> io::Result<()> {
    for pattern in patterns {
        let mut element = writer.create_element("svrl:active-pattern");
        if let Some(id) = pattern.pattern_id.as_deref() {
            element = element.with_attribute(("id", id));
        }
        element.write_inner_content(|writer| write_fired_rules(writer, &pattern.fired_rules))?;
    }
    Ok(())
}

fn write_fired_rules<W: Write>(writer: &mut Writer<W>, rules: &[FiredRule]) -> io::Result<()> {
    for rule in rules {
        let mut element = writer
            .create_element("svrl:fired-rule")
            .with_attribute(("context", rule.context.as_str()))
            .with_attribute(("location", rule.location.as_str()));
        if let Some(id) = rule.rule_id.as_deref() {
            element = element.with_attribute(("id", id));
        }
        if let Some(role) = rule.role.as_deref() {
            element = element.with_attribute(("role", role));
        }
        if let Some(flag) = rule.flag.as_deref() {
            element = element.with_attribute(("flag", flag));
        }
        element.write_empty()?;
        write_checks(writer, &rule.checks)?;
    }
    Ok(())
}

fn write_checks<W: Write>(writer: &mut Writer<W>, checks: &[CheckResult]) -> io::Result<()> {
    for check in checks {
        let mut element = writer
            .create_element(check.outcome.svrl_element_name())
            .with_attribute(("test", check.test.as_str()))
            .with_attribute(("location", check.location.as_str()));
        if let Some(id) = check.id.as_deref() {
            element = element.with_attribute(("id", id));
        }
        if let Some(role) = check.role.as_deref() {
            element = element.with_attribute(("role", role));
        }
        if let Some(flag) = check.flag.as_deref() {
            element = element.with_attribute(("flag", flag));
        }
        element.write_inner_content(|writer| {
            writer
                .create_element("svrl:text")
                .write_text_content(BytesText::new(&check.message))?;
            for diagnostic in &check.diagnostics {
                writer
                    .create_element("svrl:diagnostic-reference")
                    .write_text_content(BytesText::new(diagnostic))?;
            }
            Ok(())
        })?;
    }
    Ok(())
}
