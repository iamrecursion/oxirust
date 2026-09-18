//! Simple example demonstrating image scaling with hardware acceleration.

use oximedia_accel::{AccelContext, HardwareAccel, ScaleFilter};
use oximedia_core::PixelFormat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // Create acceleration context (GPU if available, otherwise CPU)
    let accel = AccelContext::new()?;
    println!("Using backend: {}", accel.backend_name());
    println!("GPU accelerated: {}", accel.is_gpu_accelerated());

    // Create a simple test image (640x480 RGB)
    let src_width = 640;
    let src_height = 480;
    let dst_width = 320;
    let dst_height = 240;

    // Generate a gradient test pattern
    let mut input = vec![0u8; (src_width * src_height * 3) as usize];
    for y in 0..src_height {
        for x in 0..src_width {
            let idx = ((y * src_width + x) * 3) as usize;
            input[idx] = (x * 255 / src_width) as u8; // Red gradient
            input[idx + 1] = (y * 255 / src_height) as u8; // Green gradient
            input[idx + 2] = 128; // Blue constant
        }
    }

    // Scale the image
    println!("Scaling image from {src_width}x{src_height} to {dst_width}x{dst_height}...");

    let output = accel.scale_image(
        &input,
        src_width,
        src_height,
        dst_width,
        dst_height,
        PixelFormat::Rgb24,
        ScaleFilter::Bilinear,
    )?;

    println!("Output size: {} bytes", output.len());
    println!("Expected size: {} bytes", dst_width * dst_height * 3);

    Ok(())
}
