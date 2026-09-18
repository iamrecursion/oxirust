//! Demonstrate API compatibility checking
//!
//! This example shows how to check if required APIs are available.

use scirs2_core::api_freeze::{
    check_apis_available, current_libraryversion, initialize_api_freeze, is_api_available,
    ApiCompatibilityChecker,
};
use scirs2_core::apiversioning::Version;

#[allow(dead_code)]
fn main() {
    // Initialize the API freeze registry
    initialize_api_freeze();

    println!("Current library version: {}", current_libraryversion());
    println!();

    // Check individual API availability
    println!("=== Individual API Checks ===");
    check_api("CoreError", "error");
    check_api("check_finite", "validation");
    check_api("SimdOps", "simd_ops");
    check_api("NonExistentAPI", "error");
    println!();

    // Check multiple APIs at once
    println!("=== Batch API Check ===");
    let required_apis = vec![
        ("CoreError", "error"),
        ("check_finite", "validation"),
        ("Config", "config"),
    ];

    match check_apis_available(&required_apis) {
        Ok(()) => println!("✓ All required APIs are available"),
        Err(e) => println!("✗ Some APIs are missing: {e}"),
    }
    println!();

    // Use the compatibility checker builder
    println!("=== Compatibility Checker ===");
    let checker = ApiCompatibilityChecker::new()
        .require_api("CoreError", "error")
        .require_api("check_finite", "validation")
        .require_api("SystemResources", "resource")
        .minimum_version(Version::new(1, 0, 0));

    match checker.check() {
        Ok(()) => println!("✓ All compatibility requirements met"),
        Err(e) => println!("✗ Compatibility check failed: {e}"),
    }

    // Check with missing API
    println!("\n=== Checking with Missing API ===");
    let checker_fail = ApiCompatibilityChecker::new()
        .require_api("NonExistentAPI", "fake_module")
        .minimum_version(Version::new(1, 0, 0));

    match checker_fail.check() {
        Ok(()) => println!("✓ All compatibility requirements met"),
        Err(e) => println!("✗ Compatibility check failed: {e}"),
    }
}

#[allow(dead_code)]
fn check_api(name: &str, module: &str) {
    if is_api_available(name, module) {
        println!("✓ {module}::{name} is available");
    } else {
        println!("✗ {module}::{name} is NOT available");
    }
}
