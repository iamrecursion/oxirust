//! Cross-Platform Advanced Validation Demo
//!
//! This example demonstrates the cross-platform validation capabilities
//! that ensure optimizations work consistently across different platforms.

use scirs2_core::validation::cross_platform::{get_platform_info, CrossPlatformValidator};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Cross-Platform Validation Demo");
    println!("=================================\n");

    // Get platform information
    let platform_info = get_platform_info()?;
    println!("📊 Platform Information:");
    println!("   OS Family: {:?}", platform_info.os_family);
    println!("   Architecture: {:?}", platform_info.arch);
    println!("   SIMD Support: {:?}", platform_info.simd_support);
    println!("   Endianness: {:?}", platform_info.endianness);
    println!("   Path Separator: '{}'", platform_info.path_separator);
    println!("   Max Path Length: {}", platform_info.max_path_length);
    println!("   Page Size: {} bytes", platform_info.page_size);

    // Create validator
    let mut validator = CrossPlatformValidator::new()?;
    println!("\n✅ Cross-platform validator created successfully");

    // Test file path validation
    println!("\n🗂️  File Path Validation:");
    let test_paths = [
        "/tmp/test.txt",
        "C:\\temp\\test.txt",
        "data/input.csv",
        "/very/long/path/that/might/exceed/platform/limits/test.txt",
    ];

    for path in &test_paths {
        let result = validator.validate_file_path(path);
        println!(
            "   {} Path '{}' -> valid: {}",
            if result.is_valid { "✅" } else { "❌" },
            path,
            result.is_valid
        );
    }

    // Test SIMD operation validation
    println!("\n🚀 SIMD Operation Validation:");
    let simd_ops = [
        ("vector_add", 1024),
        ("matrix_multiply", 4096),
        ("convolution", 2048),
    ];

    for (op, size) in &simd_ops {
        let result = validator.validate_simd_operation(op, *size, 256);
        println!(
            "   {} Operation '{}' (size: {}) -> valid: {}",
            if result.is_valid { "✅" } else { "❌" },
            op,
            size,
            result.is_valid
        );
    }

    // Test memory allocation validation
    println!("\n💾 Memory Allocation Validation:");
    let memory_tests = [
        (1024 * 1024, "small_buffer"),       // 1MB
        (100 * 1024 * 1024, "large_buffer"), // 100MB
        (1024 * 1024 * 1024, "huge_buffer"), // 1GB
    ];

    for (size, purpose) in &memory_tests {
        let result = validator.validate_memory_allocation(*size, purpose);
        println!(
            "   {} Allocation {} bytes for '{}' -> valid: {}",
            if result.is_valid { "✅" } else { "❌" },
            size,
            purpose,
            result.is_valid
        );
    }

    println!("\n✅ Cross-platform validation demo completed successfully!");
    Ok(())
}
