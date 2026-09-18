fn main() {
    // NOTE: Pure Rust implementation using OxiBLAS
    // All system BLAS/LAPACK dependencies have been removed
    // Linear algebra operations now use Pure Rust OxiBLAS (enabled via 'linalg' feature)

    #[cfg(target_os = "macos")]
    {
        // Compile MPSGraph Objective-C wrapper if mpsgraph feature is enabled
        // Note: This is a GPU-specific feature and unavoidable for Metal acceleration
        #[cfg(feature = "mpsgraph")]
        {
            cc::Build::new()
                .file("src/gpu/backends/mpsgraph_wrapper.m")
                .flag("-fobjc-arc") // Enable ARC for automatic memory management
                .flag("-Wno-deprecated-declarations")
                .compile("mpsgraph_wrapper");

            // Link Metal frameworks
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");

            // Rerun if wrapper source changes
            println!("cargo:rerun-if-changed=src/gpu/backends/mpsgraph_wrapper.m");
        }
    }
}
