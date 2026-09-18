//! Generate API freeze report for scirs2-core 1.0
//!
//! This example generates a comprehensive report of all frozen APIs.

use scirs2_core::api_freeze::{generate_frozen_api_report, initialize_api_freeze};

#[allow(dead_code)]
fn main() {
    // Initialize the API freeze registry
    initialize_api_freeze();

    // Generate the report
    let report = generate_frozen_api_report();

    // Print the report
    println!("{report}");

    // Also write to file
    if let Err(e) = std::fs::write("frozen_api_report.md", &report) {
        eprintln!("Failed to write report to file: {e}");
    } else {
        println!("\nReport also written to frozen_api_report.md");
    }
}
