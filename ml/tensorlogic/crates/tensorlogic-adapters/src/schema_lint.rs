//! Schema validation and linting for SymbolTable definitions.
//!
//! Detects common issues in schema definitions: unused domains, orphan predicates,
//! naming convention violations, empty domains, and arity inconsistencies.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::SymbolTable;

/// Severity level for lint issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LintSeverity {
    Info,
    Warning,
    Error,
}

impl std::fmt::Display for LintSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LintSeverity::Info => write!(f, "INFO"),
            LintSeverity::Warning => write!(f, "WARN"),
            LintSeverity::Error => write!(f, "ERROR"),
        }
    }
}

/// A lint rule code identifying the check that triggered it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LintCode {
    /// Domain is defined but not referenced by any predicate
    UnusedDomain,
    /// Predicate references a domain that doesn't exist
    OrphanPredicate,
    /// Domain name doesn't follow PascalCase convention
    DomainNamingConvention,
    /// Predicate name doesn't follow snake_case convention
    PredicateNamingConvention,
    /// Domain has zero cardinality
    EmptyDomain,
    /// Predicate has zero arity (no arguments)
    ZeroArityPredicate,
}

impl LintCode {
    /// Returns the default severity for this lint code.
    pub fn default_severity(&self) -> LintSeverity {
        match self {
            LintCode::OrphanPredicate => LintSeverity::Error,
            LintCode::EmptyDomain | LintCode::ZeroArityPredicate => LintSeverity::Warning,
            LintCode::UnusedDomain
            | LintCode::DomainNamingConvention
            | LintCode::PredicateNamingConvention => LintSeverity::Info,
        }
    }

    /// Returns the short name for this lint code.
    pub fn name(&self) -> &'static str {
        match self {
            LintCode::UnusedDomain => "unused-domain",
            LintCode::OrphanPredicate => "orphan-predicate",
            LintCode::DomainNamingConvention => "domain-naming",
            LintCode::PredicateNamingConvention => "predicate-naming",
            LintCode::EmptyDomain => "empty-domain",
            LintCode::ZeroArityPredicate => "zero-arity",
        }
    }
}

/// A single lint issue found in the schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintIssue {
    pub severity: LintSeverity,
    pub code: LintCode,
    pub message: String,
    /// The domain or predicate name this issue relates to.
    pub location: String,
}

impl std::fmt::Display for LintIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}: {} ({})",
            self.severity,
            self.code.name(),
            self.message,
            self.location
        )
    }
}

/// Result of linting a schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LintResult {
    pub issues: Vec<LintIssue>,
}

impl LintResult {
    /// Count of issues with Error severity.
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == LintSeverity::Error)
            .count()
    }

    /// Count of issues with Warning severity.
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == LintSeverity::Warning)
            .count()
    }

    /// Count of issues with Info severity.
    pub fn info_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == LintSeverity::Info)
            .count()
    }

    /// Total number of issues.
    pub fn total_count(&self) -> usize {
        self.issues.len()
    }

    /// Returns true if no issues were found.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Returns true if any error-level issues were found.
    pub fn has_errors(&self) -> bool {
        self.error_count() > 0
    }

    /// Filter issues by minimum severity.
    pub fn filter_by_severity(&self, min_severity: LintSeverity) -> Vec<&LintIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity >= min_severity)
            .collect()
    }

    /// Summary string describing issue counts.
    pub fn summary(&self) -> String {
        format!(
            "{} errors, {} warnings, {} infos",
            self.error_count(),
            self.warning_count(),
            self.info_count()
        )
    }
}

/// Configuration for the schema linter.
#[derive(Debug, Clone)]
pub struct LinterConfig {
    pub check_unused_domains: bool,
    pub check_orphan_predicates: bool,
    pub check_domain_naming: bool,
    pub check_predicate_naming: bool,
    pub check_empty_domains: bool,
    pub check_zero_arity: bool,
}

impl Default for LinterConfig {
    fn default() -> Self {
        LinterConfig {
            check_unused_domains: true,
            check_orphan_predicates: true,
            check_domain_naming: true,
            check_predicate_naming: true,
            check_empty_domains: true,
            check_zero_arity: true,
        }
    }
}

impl LinterConfig {
    /// Returns a config with all rules enabled.
    pub fn all_enabled() -> Self {
        Self::default()
    }

    /// Returns a config with all rules disabled.
    pub fn all_disabled() -> Self {
        LinterConfig {
            check_unused_domains: false,
            check_orphan_predicates: false,
            check_domain_naming: false,
            check_predicate_naming: false,
            check_empty_domains: false,
            check_zero_arity: false,
        }
    }
}

/// Schema linter for SymbolTable definitions.
///
/// Runs configurable checks against a SymbolTable and collects
/// all lint issues into a `LintResult`.
pub struct SchemaLinter {
    config: LinterConfig,
}

impl SchemaLinter {
    /// Create a linter with the given configuration.
    pub fn new(config: LinterConfig) -> Self {
        SchemaLinter { config }
    }

    /// Create a linter with all rules enabled.
    pub fn with_all_rules() -> Self {
        Self::new(LinterConfig::all_enabled())
    }

    /// Lint a SymbolTable and return all issues found.
    pub fn lint(&self, table: &SymbolTable) -> LintResult {
        let mut result = LintResult::default();

        if self.config.check_unused_domains {
            self.check_unused_domains(table, &mut result);
        }
        if self.config.check_orphan_predicates {
            self.check_orphan_predicates(table, &mut result);
        }
        if self.config.check_domain_naming {
            self.check_domain_naming(table, &mut result);
        }
        if self.config.check_predicate_naming {
            self.check_predicate_naming(table, &mut result);
        }
        if self.config.check_empty_domains {
            self.check_empty_domains(table, &mut result);
        }
        if self.config.check_zero_arity {
            self.check_zero_arity(table, &mut result);
        }

        result
    }

    /// Check for domains that are not referenced by any predicate.
    fn check_unused_domains(&self, table: &SymbolTable, result: &mut LintResult) {
        let mut referenced: HashSet<&str> = HashSet::new();
        for pred in table.predicates.values() {
            for domain_name in &pred.arg_domains {
                referenced.insert(domain_name.as_str());
            }
        }
        // Also count variable bindings as references
        for domain_name in table.variables.values() {
            referenced.insert(domain_name.as_str());
        }

        for domain_name in table.domains.keys() {
            if !referenced.contains(domain_name.as_str()) {
                result.issues.push(LintIssue {
                    severity: LintCode::UnusedDomain.default_severity(),
                    code: LintCode::UnusedDomain,
                    message: format!(
                        "Domain '{}' is defined but not referenced by any predicate or variable",
                        domain_name
                    ),
                    location: domain_name.clone(),
                });
            }
        }
    }

    /// Check for predicates referencing domains that do not exist.
    fn check_orphan_predicates(&self, table: &SymbolTable, result: &mut LintResult) {
        for pred in table.predicates.values() {
            for domain_name in &pred.arg_domains {
                if !table.domains.contains_key(domain_name) {
                    result.issues.push(LintIssue {
                        severity: LintCode::OrphanPredicate.default_severity(),
                        code: LintCode::OrphanPredicate,
                        message: format!(
                            "Predicate '{}' references nonexistent domain '{}'",
                            pred.name, domain_name
                        ),
                        location: pred.name.clone(),
                    });
                }
            }
        }
    }

    /// Check that domain names follow PascalCase convention.
    fn check_domain_naming(&self, table: &SymbolTable, result: &mut LintResult) {
        for domain_name in table.domains.keys() {
            if !is_pascal_case(domain_name) {
                result.issues.push(LintIssue {
                    severity: LintCode::DomainNamingConvention.default_severity(),
                    code: LintCode::DomainNamingConvention,
                    message: format!(
                        "Domain '{}' does not follow PascalCase naming convention",
                        domain_name
                    ),
                    location: domain_name.clone(),
                });
            }
        }
    }

    /// Check that predicate names follow snake_case convention.
    fn check_predicate_naming(&self, table: &SymbolTable, result: &mut LintResult) {
        for pred_name in table.predicates.keys() {
            if !is_snake_case(pred_name) {
                result.issues.push(LintIssue {
                    severity: LintCode::PredicateNamingConvention.default_severity(),
                    code: LintCode::PredicateNamingConvention,
                    message: format!(
                        "Predicate '{}' does not follow snake_case naming convention",
                        pred_name
                    ),
                    location: pred_name.clone(),
                });
            }
        }
    }

    /// Check for domains with zero cardinality.
    fn check_empty_domains(&self, table: &SymbolTable, result: &mut LintResult) {
        for (domain_name, domain_info) in &table.domains {
            if domain_info.cardinality == 0 {
                result.issues.push(LintIssue {
                    severity: LintCode::EmptyDomain.default_severity(),
                    code: LintCode::EmptyDomain,
                    message: format!("Domain '{}' has zero cardinality", domain_name),
                    location: domain_name.clone(),
                });
            }
        }
    }

    /// Check for predicates with zero arity (no arguments).
    fn check_zero_arity(&self, table: &SymbolTable, result: &mut LintResult) {
        for (pred_name, pred_info) in &table.predicates {
            if pred_info.arity == 0 {
                result.issues.push(LintIssue {
                    severity: LintCode::ZeroArityPredicate.default_severity(),
                    code: LintCode::ZeroArityPredicate,
                    message: format!("Predicate '{}' has zero arity (no arguments)", pred_name),
                    location: pred_name.clone(),
                });
            }
        }
    }
}

/// Check if a string follows PascalCase convention.
///
/// PascalCase requires the first character to be uppercase and no underscores.
fn is_pascal_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    first.is_uppercase() && !s.contains('_')
}

/// Check if a string follows snake_case convention.
///
/// snake_case requires all characters to be lowercase, digits, or underscores.
fn is_snake_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars()
        .all(|c| c.is_lowercase() || c == '_' || c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DomainInfo, PredicateInfo};

    /// Helper to build a clean, well-formed SymbolTable for testing.
    fn make_clean_table() -> SymbolTable {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("failed to add domain");
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .expect("failed to add predicate");
        table
    }

    #[test]
    fn test_lint_clean_schema() {
        let table = make_clean_table();
        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);
        assert!(
            result.is_clean(),
            "Expected clean schema, got: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_lint_unused_domain() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("failed to add domain");
        table
            .add_domain(DomainInfo::new("Animal", 50))
            .expect("failed to add domain");
        // Only Person is used by the predicate
        table
            .add_predicate(PredicateInfo::new(
                "knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .expect("failed to add predicate");

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let unused: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::UnusedDomain)
            .collect();
        assert_eq!(unused.len(), 1);
        assert_eq!(unused[0].location, "Animal");
        assert_eq!(unused[0].severity, LintSeverity::Info);
    }

    #[test]
    fn test_lint_orphan_predicate() {
        let mut table = SymbolTable::new();
        // Manually insert predicate without domain validation
        table.predicates.insert(
            "likes".to_string(),
            PredicateInfo::new("likes", vec!["Ghost".to_string()]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let orphans: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::OrphanPredicate)
            .collect();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].severity, LintSeverity::Error);
        assert!(orphans[0].message.contains("Ghost"));
    }

    #[test]
    fn test_lint_domain_naming_bad() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("person", 100))
            .expect("failed to add domain");
        // Add a predicate to avoid unused-domain noise
        table.predicates.insert(
            "exists_in".to_string(),
            PredicateInfo::new("exists_in", vec!["person".to_string()]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let naming: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::DomainNamingConvention)
            .collect();
        assert_eq!(naming.len(), 1);
        assert_eq!(naming[0].location, "person");
    }

    #[test]
    fn test_lint_domain_naming_good() {
        let table = make_clean_table();
        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let naming: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::DomainNamingConvention)
            .collect();
        assert!(naming.is_empty());
    }

    #[test]
    fn test_lint_predicate_naming_bad() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("failed to add domain");
        table
            .add_predicate(PredicateInfo::new(
                "Knows",
                vec!["Person".to_string(), "Person".to_string()],
            ))
            .expect("failed to add predicate");

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let naming: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::PredicateNamingConvention)
            .collect();
        assert_eq!(naming.len(), 1);
        assert_eq!(naming[0].location, "Knows");
    }

    #[test]
    fn test_lint_predicate_naming_good() {
        let table = make_clean_table();
        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let naming: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::PredicateNamingConvention)
            .collect();
        assert!(naming.is_empty());
    }

    #[test]
    fn test_lint_empty_domain() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Empty", 0))
            .expect("failed to add domain");
        table.predicates.insert(
            "check".to_string(),
            PredicateInfo::new("check", vec!["Empty".to_string()]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let empty: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::EmptyDomain)
            .collect();
        assert_eq!(empty.len(), 1);
        assert_eq!(empty[0].severity, LintSeverity::Warning);
    }

    #[test]
    fn test_lint_zero_arity() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("failed to add domain");
        table.predicates.insert(
            "tautology".to_string(),
            PredicateInfo::new("tautology", vec![]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let zero: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::ZeroArityPredicate)
            .collect();
        assert_eq!(zero.len(), 1);
        assert_eq!(zero[0].severity, LintSeverity::Warning);
    }

    #[test]
    fn test_lint_multiple_issues() {
        let mut table = SymbolTable::new();
        // 1. Empty domain (Warning)
        table
            .add_domain(DomainInfo::new("Empty", 0))
            .expect("failed to add domain");
        // 2. Orphan predicate referencing nonexistent domain (Error)
        table.predicates.insert(
            "orphan".to_string(),
            PredicateInfo::new("orphan", vec!["Ghost".to_string()]),
        );
        // 3. Zero arity predicate (Warning)
        table
            .predicates
            .insert("nullary".to_string(), PredicateInfo::new("nullary", vec![]));

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        // At least 3 issues: unused Empty, orphan predicate, zero-arity
        assert!(
            result.total_count() >= 3,
            "Expected at least 3 issues, got {}",
            result.total_count()
        );
    }

    #[test]
    fn test_lint_severity_filter() {
        let mut table = SymbolTable::new();
        // Info: unused domain
        table
            .add_domain(DomainInfo::new("Unused", 10))
            .expect("failed to add domain");
        // Warning: empty domain
        table
            .add_domain(DomainInfo::new("Empty", 0))
            .expect("failed to add domain");
        // Error: orphan predicate
        table.predicates.insert(
            "orphan".to_string(),
            PredicateInfo::new("orphan", vec!["Missing".to_string()]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let warnings_and_above = result.filter_by_severity(LintSeverity::Warning);
        // Should not include Info-level issues
        for issue in &warnings_and_above {
            assert!(issue.severity >= LintSeverity::Warning);
        }
        assert!(!warnings_and_above.is_empty());
    }

    #[test]
    fn test_lint_summary() {
        let mut table = SymbolTable::new();
        table.predicates.insert(
            "orphan".to_string(),
            PredicateInfo::new("orphan", vec!["Missing".to_string()]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        let summary = result.summary();
        assert!(summary.contains("errors"));
        assert!(summary.contains("warnings"));
        assert!(summary.contains("infos"));
    }

    #[test]
    fn test_lint_error_count() {
        let mut table = SymbolTable::new();
        // Two orphan predicates referencing nonexistent domains
        table.predicates.insert(
            "pred_a".to_string(),
            PredicateInfo::new("pred_a", vec!["Phantom".to_string()]),
        );
        table.predicates.insert(
            "pred_b".to_string(),
            PredicateInfo::new("pred_b", vec!["Specter".to_string()]),
        );

        let linter = SchemaLinter::with_all_rules();
        let result = linter.lint(&table);

        assert_eq!(result.error_count(), 2);
    }

    #[test]
    fn test_lint_config_disabled() {
        let mut table = SymbolTable::new();
        // This would normally trigger UnusedDomain
        table
            .add_domain(DomainInfo::new("Lonely", 50))
            .expect("failed to add domain");

        let mut config = LinterConfig::all_enabled();
        config.check_unused_domains = false;

        let linter = SchemaLinter::new(config);
        let result = linter.lint(&table);

        let unused: Vec<_> = result
            .issues
            .iter()
            .filter(|i| i.code == LintCode::UnusedDomain)
            .collect();
        assert!(unused.is_empty());
    }

    #[test]
    fn test_lint_config_all_disabled() {
        let mut table = SymbolTable::new();
        // Add problematic entries that would normally trigger issues
        table
            .add_domain(DomainInfo::new("unused", 0))
            .expect("failed to add domain");
        table.predicates.insert(
            "Orphan".to_string(),
            PredicateInfo::new("Orphan", vec!["Ghost".to_string()]),
        );

        let linter = SchemaLinter::new(LinterConfig::all_disabled());
        let result = linter.lint(&table);

        assert!(result.is_clean());
    }

    #[test]
    fn test_is_pascal_case() {
        assert!(is_pascal_case("Person"));
        assert!(is_pascal_case("MyDomain"));
        assert!(is_pascal_case("A"));
        assert!(!is_pascal_case("person"));
        assert!(!is_pascal_case("my_domain"));
        assert!(!is_pascal_case(""));
    }

    #[test]
    fn test_is_snake_case() {
        assert!(is_snake_case("knows"));
        assert!(is_snake_case("knows_about"));
        assert!(is_snake_case("pred2"));
        assert!(!is_snake_case("Knows"));
        assert!(!is_snake_case("knowsAbout"));
        assert!(!is_snake_case(""));
    }

    #[test]
    fn test_lint_code_names() {
        let codes = vec![
            LintCode::UnusedDomain,
            LintCode::OrphanPredicate,
            LintCode::DomainNamingConvention,
            LintCode::PredicateNamingConvention,
            LintCode::EmptyDomain,
            LintCode::ZeroArityPredicate,
        ];
        for code in &codes {
            let name = code.name();
            assert!(!name.is_empty(), "LintCode {:?} has empty name", code);
        }
    }
}
