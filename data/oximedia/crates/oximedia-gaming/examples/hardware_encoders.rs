//! Hardware encoder detection and selection example.

use oximedia_gaming::encode::{NvencEncoder, QsvEncoder, VceEncoder};

fn main() {
    println!("=== OxiMedia Gaming - Hardware Encoder Detection ===\n");

    // Check NVIDIA NVENC
    println!("🎮 NVIDIA NVENC");
    if NvencEncoder::is_available() {
        println!("   Status: ✓ Available");

        match NvencEncoder::get_capabilities() {
            Ok(caps) => {
                println!("   GPU: {}", caps.gpu_name);
                println!("   Max Resolution: {}x{}", caps.max_width, caps.max_height);
                println!("   Max FPS: {}", caps.max_framerate);
                println!(
                    "   AV1 Support: {}",
                    if caps.supports_av1 { "Yes" } else { "No" }
                );
                println!(
                    "   VP9 Support: {}",
                    if caps.supports_vp9 { "Yes" } else { "No" }
                );
                println!(
                    "   B-frames: {}",
                    if caps.supports_b_frames { "Yes" } else { "No" }
                );
            }
            Err(e) => println!("   Error: {}", e),
        }
    } else {
        println!("   Status: ✗ Not available");
    }
    println!();

    // Check Intel Quick Sync
    println!("🎮 Intel Quick Sync Video");
    if QsvEncoder::is_available() {
        println!("   Status: ✓ Available");

        match QsvEncoder::get_capabilities() {
            Ok(caps) => {
                println!("   GPU: {}", caps.gpu_name);
                println!("   Generation: {:?}", caps.generation);
                println!("   Max Resolution: {}x{}", caps.max_width, caps.max_height);
                println!(
                    "   AV1 Support: {}",
                    if caps.supports_av1 { "Yes" } else { "No" }
                );
                println!(
                    "   VP9 Support: {}",
                    if caps.supports_vp9 { "Yes" } else { "No" }
                );
            }
            Err(e) => println!("   Error: {}", e),
        }
    } else {
        println!("   Status: ✗ Not available");
    }
    println!();

    // Check AMD VCE
    println!("🎮 AMD VCE/VCN");
    if VceEncoder::is_available() {
        println!("   Status: ✓ Available");

        match VceEncoder::get_capabilities() {
            Ok(caps) => {
                println!("   GPU: {}", caps.gpu_name);
                println!("   Version: {:?}", caps.vcn_version);
                println!("   Max Resolution: {}x{}", caps.max_width, caps.max_height);
                println!(
                    "   AV1 Support: {}",
                    if caps.supports_av1 { "Yes" } else { "No" }
                );
                println!(
                    "   VP9 Support: {}",
                    if caps.supports_vp9 { "Yes" } else { "No" }
                );
            }
            Err(e) => println!("   Error: {}", e),
        }
    } else {
        println!("   Status: ✗ Not available");
    }
    println!();

    // Recommendation
    println!("=== Recommended Encoder ===");
    if NvencEncoder::is_available() {
        println!("✓ Use NVIDIA NVENC (best quality/performance)");
    } else if QsvEncoder::is_available() {
        println!("✓ Use Intel Quick Sync Video (good efficiency)");
    } else if VceEncoder::is_available() {
        println!("✓ Use AMD VCE (hardware acceleration)");
    } else {
        println!("⚠ No hardware encoder available - using software encoding");
    }
}
