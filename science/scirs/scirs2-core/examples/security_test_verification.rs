//! Security Testing Framework Verification
//!
//! This example demonstrates and verifies that the security testing framework
//! is working correctly by running various security tests.

#[cfg(feature = "testing")]
use scirs2_core::error::CoreResult;
#[cfg(feature = "testing")]
use scirs2_core::testing::security::{
    InputValidationTester, MemorySafetyTester, SecurityLevel, SecurityTestConfig,
    VulnerabilityAssessment,
};
#[cfg(feature = "testing")]
use std::time::Duration;

#[cfg(feature = "testing")]
#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("🔒 Security Testing Framework Verification");
    println!("==========================================");

    // Create security test configuration
    let config = SecurityTestConfig::new()
        .with_max_input_size(1024)
        .with_malicious_patterns(50)
        .with_test_timeout(Duration::from_secs(5));

    println!("✅ Security test configuration created");

    // Test 1: Input Validation Testing
    println!("\n📝 Test 1: Input Validation");
    let input_tester = InputValidationTester::new(config.clone());

    let validation_result = input_tester.test_malicious_inputs(|input| {
        // Simple validation function for testing
        if input.len() > 500 {
            return Err(scirs2_core::error::CoreError::ValidationError(
                scirs2_core::error::ErrorContext::new("Input too large"),
            ));
        }
        if input.is_empty() {
            return Err(scirs2_core::error::CoreError::ValidationError(
                scirs2_core::error::ErrorContext::new("Input cannot be empty"),
            ));
        }
        Ok(())
    })?;

    println!("   Tests executed: {}", validation_result.tests_executed);
    println!(
        "   Vulnerabilities found: {}",
        validation_result.vulnerabilities_found
    );
    println!("   Security level: {:?}", validation_result.security_level);
    println!("   Duration: {:?}", validation_result.duration);

    // Test 2: Bounds Checking
    println!("\n🔢 Test 2: Bounds Checking");
    let bounds_result = input_tester.test_bounds_checking(|start, length| {
        // Test bounds checking with overflow detection
        let end = start.checked_add(length).ok_or_else(|| {
            scirs2_core::error::CoreError::ValidationError(scirs2_core::error::ErrorContext::new(
                "Integer overflow in bounds calculation",
            ))
        })?;

        if end > 1000 {
            return Err(scirs2_core::error::CoreError::ValidationError(
                scirs2_core::error::ErrorContext::new("Bounds exceed maximum allowed"),
            ));
        }
        Ok(())
    })?;

    println!("   Tests executed: {}", bounds_result.tests_executed);
    println!(
        "   Vulnerabilities found: {}",
        bounds_result.vulnerabilities_found
    );
    println!("   Security level: {:?}", bounds_result.security_level);

    // Test 3: Memory Safety Testing
    println!("\n🧠 Test 3: Memory Safety");
    let memory_tester = MemorySafetyTester::new(config.clone());

    let memory_result = memory_tester.test_memory_leaks(|| {
        // Test function that should not leak memory
        let test_data = vec![0u8; 1000];
        std::thread::sleep(Duration::from_millis(1)); // Simulate some work
        Ok(())
    })?;

    println!("   Tests executed: {}", memory_result.tests_executed);
    println!(
        "   Vulnerabilities found: {}",
        memory_result.vulnerabilities_found
    );
    println!("   Security level: {:?}", memory_result.security_level);

    // Test 4: Use-after-free test (Rust-specific)
    println!("\n🚫 Test 4: Use-After-Free Detection");
    let uaf_result = memory_tester.test_use_after_free()?;
    println!("   Tests executed: {}", uaf_result.tests_executed);
    println!("   Security level: {:?}", uaf_result.security_level);
    println!("   Note: Rust prevents use-after-free vulnerabilities");

    // Test 5: Comprehensive Vulnerability Assessment
    println!("\n🔍 Test 5: Comprehensive Vulnerability Assessment");
    let assessment = VulnerabilityAssessment::new(config);
    let audit_report = assessment.perform_security_audit()?;

    println!("   Total tests: {}", audit_report.total_tests);
    println!("   Tests passed: {}", audit_report.passed_tests);
    println!("   Tests failed: {}", audit_report.failed_tests);
    println!(
        "   Vulnerabilities found: {}",
        audit_report.vulnerabilities.len()
    );
    println!("   Security score: {:.1}/100.0", audit_report.overall_score);
    println!("   Security level: {:?}", audit_report.security_level);
    println!("   Audit duration: {:?}", audit_report.duration);

    // Display any vulnerabilities found
    if !audit_report.vulnerabilities.is_empty() {
        println!("\n⚠️  Vulnerabilities Found:");
        for vuln in &audit_report.vulnerabilities {
            println!(
                "   - {} ({:?}): {}",
                vuln.title, vuln.severity, vuln.description
            );
        }
    }

    // Display recommendations
    if !audit_report.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for rec in &audit_report.recommendations {
            println!("   - {}", rec);
        }
    }

    // Final assessment
    println!("\n🎯 Final Assessment");
    println!("==================");

    let overall_secure = validation_result.security_level != SecurityLevel::Insecure
        && bounds_result.security_level != SecurityLevel::Insecure
        && memory_result.security_level != SecurityLevel::Insecure
        && audit_report.security_level != SecurityLevel::Insecure;

    if overall_secure {
        println!("✅ Security Testing Framework: OPERATIONAL");
        println!("✅ All security tests passed successfully");
        println!("✅ No critical vulnerabilities found");
    } else {
        println!("⚠️  Security Testing Framework: ISSUES DETECTED");
        println!("⚠️  Some security tests failed or found vulnerabilities");
    }

    println!("\n🔒 Security testing framework verification completed!");

    Ok(())
}

#[cfg(not(feature = "testing"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'testing' feature to be enabled.");
    println!("Run with: cargo run --example security_test_verification --features testing");
}
