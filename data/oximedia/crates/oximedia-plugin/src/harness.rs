//! Standardised plugin test harness.
//!
//! The harness provides a battery of tests that plugin authors can run against
//! their implementation to verify correctness, API compliance, and performance
//! expectations.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oximedia_plugin::harness::PluginHarness;
//! use std::sync::Arc;
//!
//! let result = PluginHarness::new(Arc::new(my_plugin))
//!     .expect_codec("h264")
//!     .expect_decode("h264")
//!     .run();
//!
//! result.assert_all_passed();
//! ```
//!
//! # Design
//!
//! [`PluginHarness`] accumulates *expectations* and *checks* via the builder
//! pattern.  When [`PluginHarness::run`] is called, each check is evaluated and
//! a [`HarnessReport`] is returned containing individual [`CheckResult`]s.
//!
//! The harness is intentionally synchronous and allocation-light so it can be
//! used in `#[test]` functions without async runtimes.

use crate::traits::{CodecPlugin, CodecPluginInfo, PLUGIN_API_VERSION};
use std::sync::Arc;

// ── CheckResult ───────────────────────────────────────────────────────────────

/// Outcome of a single harness check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Descriptive name of the check.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Optional details about why the check failed.
    pub message: Option<String>,
}

impl CheckResult {
    fn pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            message: None,
        }
    }

    fn fail(name: impl Into<String>, msg: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            message: Some(msg.into()),
        }
    }
}

impl std::fmt::Display for CheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(f, "[{status}] {}", self.name)?;
        if let Some(ref msg) = self.message {
            write!(f, " — {msg}")?;
        }
        Ok(())
    }
}

// ── HarnessReport ─────────────────────────────────────────────────────────────

/// A summary of all checks run by a [`PluginHarness`].
#[derive(Debug, Clone)]
pub struct HarnessReport {
    /// Plugin name from `info()`.
    pub plugin_name: String,
    /// Individual check results in the order they were registered.
    pub checks: Vec<CheckResult>,
}

impl HarnessReport {
    /// Returns `true` if every check passed.
    pub fn all_passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    /// Number of checks that passed.
    pub fn pass_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    /// Number of checks that failed.
    pub fn fail_count(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }

    /// Panics with a diagnostic message if any check failed.
    ///
    /// Intended for use inside `#[test]` functions.
    #[track_caller]
    pub fn assert_all_passed(&self) {
        if !self.all_passed() {
            let failures: Vec<String> = self
                .checks
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.to_string())
                .collect();
            panic!(
                "PluginHarness: {} check(s) failed for '{}':\n{}",
                self.fail_count(),
                self.plugin_name,
                failures.join("\n")
            );
        }
    }
}

impl std::fmt::Display for HarnessReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "PluginHarness report for '{}': {}/{} passed",
            self.plugin_name,
            self.pass_count(),
            self.checks.len()
        )?;
        for c in &self.checks {
            writeln!(f, "  {c}")?;
        }
        Ok(())
    }
}

// ── HarnessCheck ─────────────────────────────────────────────────────────────

/// An internal check function.
type CheckFn = Box<dyn Fn(&dyn CodecPlugin) -> CheckResult + Send + Sync>;

// ── PluginHarness ─────────────────────────────────────────────────────────────

/// A configurable test harness for [`CodecPlugin`] implementations.
///
/// Build up expectations via the builder methods, then call [`run`](Self::run)
/// to execute all checks and obtain a [`HarnessReport`].
pub struct PluginHarness {
    plugin: Arc<dyn CodecPlugin>,
    checks: Vec<CheckFn>,
}

impl PluginHarness {
    /// Create a new harness for the given plugin.
    pub fn new(plugin: Arc<dyn CodecPlugin>) -> Self {
        Self {
            plugin,
            checks: Vec::new(),
        }
    }

    // ── Built-in checks ──────────────────────────────────────────────────────

    /// Add a check that `info().name` is non-empty.
    #[must_use]
    pub fn check_name_nonempty(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            let info = p.info();
            if info.name.is_empty() {
                CheckResult::fail("name_nonempty", "Plugin name must not be empty")
            } else {
                CheckResult::pass("name_nonempty")
            }
        }));
        self
    }

    /// Add a check that `info().version` is a valid semver string.
    #[must_use]
    pub fn check_version_semver(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            let info = p.info();
            match crate::version_resolver::SemVer::parse(&info.version) {
                Ok(_) => CheckResult::pass("version_semver"),
                Err(e) => CheckResult::fail(
                    "version_semver",
                    format!("Version '{}' is not valid semver: {e}", info.version),
                ),
            }
        }));
        self
    }

    /// Add a check that the plugin's `api_version` matches the host.
    #[must_use]
    pub fn check_api_version(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            let info = p.info();
            if info.api_version != PLUGIN_API_VERSION {
                CheckResult::fail(
                    "api_version",
                    format!(
                        "api_version is {}, expected {PLUGIN_API_VERSION}",
                        info.api_version
                    ),
                )
            } else {
                CheckResult::pass("api_version")
            }
        }));
        self
    }

    /// Add a check that the plugin has at least one capability.
    #[must_use]
    pub fn check_has_capabilities(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            if p.capabilities().is_empty() {
                CheckResult::fail("has_capabilities", "Plugin reports no capabilities")
            } else {
                CheckResult::pass("has_capabilities")
            }
        }));
        self
    }

    /// Add a check that `supports_codec(codec_name)` returns `true`.
    #[must_use]
    pub fn expect_codec(mut self, codec_name: impl Into<String>) -> Self {
        let name = codec_name.into();
        self.checks.push(Box::new(move |p: &dyn CodecPlugin| {
            if p.supports_codec(&name) {
                CheckResult::pass(format!("supports_codec:{name}"))
            } else {
                CheckResult::fail(
                    format!("supports_codec:{name}"),
                    format!("Plugin does not support codec '{name}'"),
                )
            }
        }));
        self
    }

    /// Add a check that `can_decode(codec_name)` returns `true`.
    #[must_use]
    pub fn expect_decode(mut self, codec_name: impl Into<String>) -> Self {
        let name = codec_name.into();
        self.checks.push(Box::new(move |p: &dyn CodecPlugin| {
            if p.can_decode(&name) {
                CheckResult::pass(format!("can_decode:{name}"))
            } else {
                CheckResult::fail(
                    format!("can_decode:{name}"),
                    format!("Plugin cannot decode '{name}'"),
                )
            }
        }));
        self
    }

    /// Add a check that `can_encode(codec_name)` returns `true`.
    #[must_use]
    pub fn expect_encode(mut self, codec_name: impl Into<String>) -> Self {
        let name = codec_name.into();
        self.checks.push(Box::new(move |p: &dyn CodecPlugin| {
            if p.can_encode(&name) {
                CheckResult::pass(format!("can_encode:{name}"))
            } else {
                CheckResult::fail(
                    format!("can_encode:{name}"),
                    format!("Plugin cannot encode '{name}'"),
                )
            }
        }));
        self
    }

    /// Add a check that `info().author` is non-empty.
    #[must_use]
    pub fn check_author_nonempty(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            let info = p.info();
            if info.author.is_empty() {
                CheckResult::fail("author_nonempty", "Plugin author must not be empty")
            } else {
                CheckResult::pass("author_nonempty")
            }
        }));
        self
    }

    /// Add a check that `info().license` is non-empty.
    #[must_use]
    pub fn check_license_nonempty(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            let info = p.info();
            if info.license.is_empty() {
                CheckResult::fail("license_nonempty", "Plugin license must not be empty")
            } else {
                CheckResult::pass("license_nonempty")
            }
        }));
        self
    }

    /// Add a check that each capability has a non-empty `codec_name`.
    #[must_use]
    pub fn check_capability_names(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            for cap in p.capabilities() {
                if cap.codec_name.is_empty() {
                    return CheckResult::fail(
                        "capability_names",
                        "At least one capability has an empty codec_name",
                    );
                }
            }
            CheckResult::pass("capability_names")
        }));
        self
    }

    /// Add a check that each capability declares at least decode or encode.
    #[must_use]
    pub fn check_capability_operations(mut self) -> Self {
        self.checks.push(Box::new(|p: &dyn CodecPlugin| {
            for cap in p.capabilities() {
                if !cap.can_decode && !cap.can_encode {
                    return CheckResult::fail(
                        "capability_operations",
                        format!(
                            "Capability '{}' declares neither decode nor encode",
                            cap.codec_name
                        ),
                    );
                }
            }
            CheckResult::pass("capability_operations")
        }));
        self
    }

    /// Add a check that `patent_encumbered` is explicitly set (either value).
    ///
    /// This just verifies the field is present; the harness does not enforce
    /// which value is correct.
    #[must_use]
    pub fn check_patent_flag_present(mut self) -> Self {
        self.checks.push(Box::new(|_p: &dyn CodecPlugin| {
            // The field always exists on CodecPluginInfo (bool, not Option<bool>),
            // so this check is always valid but serves as an explicit audit point.
            CheckResult::pass("patent_flag_present")
        }));
        self
    }

    /// Add a custom check closure.
    ///
    /// The closure receives a reference to the plugin and must return a
    /// [`CheckResult`].
    #[must_use]
    pub fn add_check<F>(mut self, f: F) -> Self
    where
        F: Fn(&dyn CodecPlugin) -> CheckResult + Send + Sync + 'static,
    {
        self.checks.push(Box::new(f));
        self
    }

    /// Apply a standard compliance suite: name, version, API version,
    /// capabilities, capability names, and capability operations.
    #[must_use]
    pub fn standard_compliance(self) -> Self {
        self.check_name_nonempty()
            .check_version_semver()
            .check_api_version()
            .check_has_capabilities()
            .check_capability_names()
            .check_capability_operations()
            .check_author_nonempty()
            .check_license_nonempty()
            .check_patent_flag_present()
    }

    /// Execute all registered checks and return a [`HarnessReport`].
    pub fn run(&self) -> HarnessReport {
        let info: CodecPluginInfo = self.plugin.info();
        let checks: Vec<CheckResult> = self
            .checks
            .iter()
            .map(|check| check(self.plugin.as_ref()))
            .collect();
        HarnessReport {
            plugin_name: info.name,
            checks,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_plugin::StaticPlugin;
    use crate::traits::{CodecPluginInfo, PluginCapability};
    use std::collections::HashMap;

    fn compliant_plugin() -> Arc<dyn CodecPlugin> {
        let info = CodecPluginInfo {
            name: "compliant-plugin".to_string(),
            version: "1.2.3".to_string(),
            author: "Test Author".to_string(),
            description: "A fully compliant plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        let cap = PluginCapability {
            codec_name: "test-codec".to_string(),
            can_decode: true,
            can_encode: true,
            pixel_formats: vec![],
            properties: HashMap::new(),
        };
        Arc::new(StaticPlugin::new(info).add_capability(cap))
    }

    fn bad_api_plugin() -> Arc<dyn CodecPlugin> {
        let info = CodecPluginInfo {
            name: "bad-api".to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Bad API plugin".to_string(),
            api_version: 999,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        Arc::new(StaticPlugin::new(info))
    }

    // 1. Standard compliance passes for a well-formed plugin.
    #[test]
    fn test_standard_compliance_passes() {
        let report = PluginHarness::new(compliant_plugin())
            .standard_compliance()
            .run();
        report.assert_all_passed();
    }

    // 2. check_api_version fails for wrong api_version.
    #[test]
    fn test_check_api_version_fails() {
        let report = PluginHarness::new(bad_api_plugin())
            .check_api_version()
            .run();
        assert_eq!(report.fail_count(), 1);
    }

    // 3. expect_codec passes when plugin supports the codec.
    #[test]
    fn test_expect_codec_passes() {
        let report = PluginHarness::new(compliant_plugin())
            .expect_codec("test-codec")
            .run();
        assert!(report.all_passed());
    }

    // 4. expect_codec fails when plugin does not support the codec.
    #[test]
    fn test_expect_codec_fails() {
        let report = PluginHarness::new(compliant_plugin())
            .expect_codec("nonexistent-codec")
            .run();
        assert_eq!(report.fail_count(), 1);
    }

    // 5. expect_decode.
    #[test]
    fn test_expect_decode_passes() {
        let report = PluginHarness::new(compliant_plugin())
            .expect_decode("test-codec")
            .run();
        assert!(report.all_passed());
    }

    // 6. expect_encode.
    #[test]
    fn test_expect_encode_passes() {
        let report = PluginHarness::new(compliant_plugin())
            .expect_encode("test-codec")
            .run();
        assert!(report.all_passed());
    }

    // 7. HarnessReport counts pass/fail.
    #[test]
    fn test_report_counts() {
        let report = PluginHarness::new(bad_api_plugin())
            .check_name_nonempty() // passes
            .check_api_version() // fails
            .run();
        assert_eq!(report.pass_count(), 1);
        assert_eq!(report.fail_count(), 1);
    }

    // 8. Custom check added via add_check.
    #[test]
    fn test_add_custom_check() {
        let report = PluginHarness::new(compliant_plugin())
            .add_check(|p| {
                if p.info().patent_encumbered {
                    CheckResult::fail("no_patent", "Plugin is patent encumbered")
                } else {
                    CheckResult::pass("no_patent")
                }
            })
            .run();
        assert!(report.all_passed());
    }

    // 9. HarnessReport display.
    #[test]
    fn test_report_display() {
        let report = PluginHarness::new(compliant_plugin())
            .check_name_nonempty()
            .run();
        let s = report.to_string();
        assert!(s.contains("compliant-plugin"));
        assert!(s.contains("PASS"));
    }

    // 10. CheckResult display includes FAIL and message.
    #[test]
    fn test_check_result_display_fail() {
        let cr = CheckResult::fail("my-check", "something went wrong");
        let s = cr.to_string();
        assert!(s.contains("FAIL"));
        assert!(s.contains("my-check"));
        assert!(s.contains("something went wrong"));
    }

    // 11. check_has_capabilities fails when no capabilities.
    #[test]
    fn test_check_has_capabilities_fails_empty() {
        let info = CodecPluginInfo {
            name: "no-caps".to_string(),
            version: "1.0.0".to_string(),
            author: "T".to_string(),
            description: "D".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        let p: Arc<dyn CodecPlugin> = Arc::new(StaticPlugin::new(info));
        let report = PluginHarness::new(p).check_has_capabilities().run();
        assert_eq!(report.fail_count(), 1);
    }

    // 12. assert_all_passed does not panic on full pass.
    #[test]
    fn test_assert_all_passed_no_panic() {
        let report = PluginHarness::new(compliant_plugin())
            .standard_compliance()
            .run();
        // Should not panic.
        report.assert_all_passed();
    }

    // 13. check_version_semver fails on bad version.
    #[test]
    fn test_check_version_semver_fails() {
        let info = CodecPluginInfo {
            name: "bad-ver".to_string(),
            version: "NOT-SEMVER".to_string(),
            author: "T".to_string(),
            description: "D".to_string(),
            api_version: PLUGIN_API_VERSION,
            license: "MIT".to_string(),
            patent_encumbered: false,
        };
        let p: Arc<dyn CodecPlugin> = Arc::new(StaticPlugin::new(info));
        let report = PluginHarness::new(p).check_version_semver().run();
        assert_eq!(report.fail_count(), 1);
    }
}
