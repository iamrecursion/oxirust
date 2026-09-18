// Build script for scirs workspace
// Optimizes build for CI environments with limited disk space

use std::env;

#[allow(dead_code)]
fn main() {
    // Check if we're in a CI environment
    let is_ci = env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok();
    
    if is_ci {
        println!("cargo:rustc-env=SCIRS_CI_BUILD=1");
        
        // Enable optimizations for CI builds
        println!("cargo:rustc-link-arg=-Wl,--compress-debug-sections=zlib");
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
        
        // Reduce debug info in CI
        if env::var("CARGO_CFG_DEBUG_ASSERTIONS").is_ok() {
            println!("cargo:rustc-env=CARGO_PROFILE_DEV_DEBUG=1");
        }
    }

    // NOTE: BLAS/LAPACK configuration removed - now using pure Rust OxiBLAS
    // All linear algebra operations are handled by OxiBLAS (no system dependencies)

    // Only rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
