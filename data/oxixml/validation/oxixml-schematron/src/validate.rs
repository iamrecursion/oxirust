//! [`Compiled::validate`] and [`Compiled::validate_tree`] — running a compiled schema over a
//! document.
//!
//! # Algorithm
//!
//! [ISO/IEC 19757-3 §5.3](https://www.iso.org/standard/55982.html) says a pattern's rules fire
//! "for each node selected by exactly one rule of the pattern, namely the first rule (in
//! document order of the rules within the pattern) whose context matches that node". This is
//! implemented directly:
//!
//! 1. For every rule of the pattern, in schema order, evaluate its (already-rewritten, see
//!    [`crate::compile::context_selecting_expression`]) context expression from the document
//!    root, giving the set of nodes it matches.
//! 2. Walk the rules in schema order, and for every node a rule matches that has not already
//!    been claimed by an earlier rule, claim it for this rule. A node claimed by no rule is
//!    never visited.
//! 3. Process the claimed `(node, rule)` pairs in document order of the node — not rule order —
//!    which is what makes the SVRL output read top-to-bottom the way the source document does.
//!
//! Each firing binds the schema's, the phase's, the pattern's and finally the rule's `let`
//! variables, in that order (so a rule's own binding shadows a pattern's binding of the same
//! name), then evaluates every `sch:assert`/`sch:report` test with the context node in focus.
//!
//! The schema's and the phase's `let`s are *global* variables ([ISO/IEC 19757-3 §5.4.5]): they
//! are evaluated **once per validation run, with the document root as the context node**, not
//! once per firing with the rule's context node — so a top-level `<let name="n"
//! value="count(//item)"/>` means the same thing here as it does under an XSLT-skeleton
//! implementation. Only the pattern's and the rule's own `let`s are context-dependent.
//!
//! [ISO/IEC 19757-3 §5.4.5]: https://www.iso.org/standard/55982.html

use std::collections::HashMap;

use oxixml_infoset::{NodeRef, Tree};
use oxixml_xpath::XpathValue;

use crate::compile::{Compiled, CompiledCheck, CompiledPattern, CompiledRule};
use crate::error::{SchematronError, SchematronResult};
use crate::location::location_path;
use crate::message::render_message;
use crate::model::CheckKind;
use crate::report::{ActivePatternReport, CheckOutcome, CheckResult, FiredRule, SchematronReport};

impl Compiled {
    /// Parses `document` and validates it.
    ///
    /// # Errors
    ///
    /// [`SchematronError::Xml`] if `document` is not well-formed, or
    /// [`SchematronError::Evaluation`] if a context, test, `let`, `subject`, `value-of` or
    /// `name` expression fails at run time (an XPath dynamic error, not a validation failure —
    /// those are reported through [`SchematronReport`], never as an `Err`).
    pub fn validate(&self, document: &str) -> SchematronResult<SchematronReport> {
        let tree = Tree::parse_str(document)?;
        self.validate_tree(&tree)
    }

    /// Validates an already-parsed document, so that the same [`Tree`] can be reused across
    /// several schemas without reparsing.
    ///
    /// # Errors
    ///
    /// As [`Compiled::validate`], minus [`SchematronError::Xml`].
    pub fn validate_tree(&self, document: &Tree) -> SchematronResult<SchematronReport> {
        let root = document.root();
        let globals = self.evaluate_globals(&root)?;
        let active_patterns = self
            .patterns
            .iter()
            .map(|pattern| self.validate_pattern(pattern, &root, &globals))
            .collect::<SchematronResult<Vec<_>>>()?;
        Ok(SchematronReport {
            title: self.title.clone(),
            phase: self.phase.clone(),
            active_patterns,
        })
    }

    /// Evaluates every schema- and phase-level `sch:let` once, with the document root as the
    /// context node, in declaration order (so a later global may refer to an earlier one).
    fn evaluate_globals(&self, root: &NodeRef) -> SchematronResult<Vec<(String, XpathValue)>> {
        let mut bound: Vec<(String, XpathValue)> = Vec::new();
        for (name, expr) in &self.globals {
            let mut context = expr.context_with(root.clone());
            for (previous, value) in &bound {
                context = context.with_variable(previous.clone(), value.clone());
            }
            let value = expr
                .evaluate_value_in(&context)
                .map_err(|error| SchematronError::Evaluation(Box::new(error)))?;
            bound.push((name.clone(), value));
        }
        Ok(bound)
    }

    fn validate_pattern(
        &self,
        pattern: &CompiledPattern,
        root: &NodeRef,
        globals: &[(String, XpathValue)],
    ) -> SchematronResult<ActivePatternReport> {
        let mut claimed: HashMap<NodeRef, usize> = HashMap::new();
        for (rule_index, rule) in pattern.rules.iter().enumerate() {
            let matched = rule
                .context_expr
                .evaluate_to_nodes(root.clone())
                .map_err(|error| SchematronError::Evaluation(Box::new(error)))?;
            for node in matched {
                claimed.entry(node).or_insert(rule_index);
            }
        }

        let mut firings: Vec<(NodeRef, usize)> = claimed.into_iter().collect();
        firings.sort_by(|left, right| left.0.cmp(&right.0));

        let fired_rules = firings
            .into_iter()
            .map(|(node, rule_index)| self.fire_rule(&pattern.rules[rule_index], &node, globals))
            .collect::<SchematronResult<Vec<_>>>()?;

        Ok(ActivePatternReport {
            pattern_id: pattern.id.clone(),
            title: pattern.title.clone(),
            fired_rules,
        })
    }

    fn fire_rule(
        &self,
        rule: &CompiledRule,
        node: &NodeRef,
        globals: &[(String, XpathValue)],
    ) -> SchematronResult<FiredRule> {
        let checks = rule
            .checks
            .iter()
            .map(|check| self.evaluate_check(check, node, globals))
            .collect::<SchematronResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(FiredRule {
            rule_id: rule.id.clone(),
            context: rule.context_source.clone(),
            role: rule.role.clone(),
            flag: rule.flag.clone(),
            location: location_path(node),
            checks,
        })
    }

    fn evaluate_check(
        &self,
        check: &CompiledCheck,
        node: &NodeRef,
        globals: &[(String, XpathValue)],
    ) -> SchematronResult<Option<CheckResult>> {
        let mut dynamic_context = check.test.context_with(node.clone());
        // The globals are seeded first, so a pattern's or a rule's `let` of the same name
        // shadows them (§5.4.5).
        for (name, value) in globals {
            dynamic_context = dynamic_context.with_variable(name.clone(), value.clone());
        }
        for (name, expr) in &check.lets {
            let value = expr
                .evaluate_value_in(&dynamic_context)
                .map_err(|error| SchematronError::Evaluation(Box::new(error)))?;
            dynamic_context = dynamic_context.with_variable(name.clone(), value);
        }

        let holds = check
            .test
            .evaluate_to_boolean_in(&dynamic_context)
            .map_err(|error| SchematronError::Evaluation(Box::new(error)))?;
        let fires = match check.kind {
            CheckKind::Assert => !holds,
            CheckKind::Report => holds,
        };
        if !fires {
            return Ok(None);
        }

        let outcome = match check.kind {
            CheckKind::Assert => CheckOutcome::FailedAssert,
            CheckKind::Report => CheckOutcome::SuccessfulReport,
        };

        let subject_node = match &check.subject {
            Some(subject) => subject
                .evaluate_to_nodes_in(&dynamic_context)
                .map_err(|error| SchematronError::Evaluation(Box::new(error)))?
                .into_iter()
                .next()
                .unwrap_or_else(|| node.clone()),
            None => node.clone(),
        };

        let message = render_message(&check.message, &dynamic_context, node)?;
        let mut diagnostics = Vec::new();
        for id in &check.diagnostics {
            if let Some(diagnostic) = self.diagnostics.get(id) {
                diagnostics.push(render_message(&diagnostic.message, &dynamic_context, node)?);
            }
        }

        Ok(Some(CheckResult {
            outcome,
            id: check.id.clone(),
            role: check.role.clone(),
            flag: check.flag.clone(),
            test: check.test_source.clone(),
            location: location_path(&subject_node),
            message,
            diagnostics,
        }))
    }
}
