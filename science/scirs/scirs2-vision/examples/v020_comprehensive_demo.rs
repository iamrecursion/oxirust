//! Comprehensive demo of scirs2-vision v0.2.0 features
//!
//! This example demonstrates all major new features added in v0.2.0:
//! - SURF feature detection with GPU acceleration
//! - Advanced optical flow (Farneback, TVL1, DualTVL1)
//! - Semantic segmentation with deep learning
//! - 3D vision (stereo matching, depth estimation)
//! - Enhanced tracking capabilities

use image::{DynamicImage, RgbImage};
use scirs2_vision::error::Result;

// Feature detection
use scirs2_vision::feature::{
    detect_and_compute_orb, detect_and_compute_surf, detect_surf, OrbConfig, SurfConfig,
};

// Advanced optical flow
use scirs2_vision::feature::{
    dual_tvl1_flow, farneback_flow, tvl1_flow, DenseFlow, FarnebackParams, TVL1Params,
};

// Semantic segmentation
use scirs2_vision::segmentation::{
    create_cityscapes_classes, create_pascal_voc_classes, DeepLabV3Plus, FCNVariant, UNet, FCN,
};

// 3D vision
use scirs2_vision::vision_3d::{
    stereo_bm, stereo_sgbm, triangulate_points, BMParams, DisparityMap, SGBMParams,
};

fn main() -> Result<()> {
    println!("=== SciRS2-Vision v0.2.0 Comprehensive Demo ===\n");

    // Create test images
    let test_img = create_test_image(512, 512);
    let test_img_small = create_test_image(256, 256);

    // ===================================================================
    // 1. SURF Feature Detection
    // ===================================================================
    println!("1. SURF Feature Detection");
    println!("   -----------------------");

    let surf_config = SurfConfig {
        num_octaves: 4,
        num_scales: 4,
        hessian_threshold: 100.0,
        extended: false,
        upright: false,
        initial_step: 2,
    };

    println!("   Detecting SURF keypoints...");
    match detect_surf(&test_img, &surf_config) {
        Ok(keypoints) => {
            println!("   ✓ Detected {} SURF keypoints", keypoints.len());
        }
        Err(e) => println!("   ✗ SURF detection error: {}", e),
    }

    println!("   Computing SURF descriptors...");
    match detect_and_compute_surf(&test_img, &surf_config) {
        Ok(descriptors) => {
            println!("   ✓ Computed {} SURF descriptors", descriptors.len());
            if !descriptors.is_empty() {
                println!("   ✓ Descriptor dimension: {}", descriptors[0].vector.len());
            }
        }
        Err(e) => println!("   ✗ SURF descriptor error: {}", e),
    }

    println!("   Testing extended SURF (128-dim descriptors)...");
    let surf_config_ext = SurfConfig {
        extended: true,
        ..surf_config
    };
    match detect_and_compute_surf(&test_img, &surf_config_ext) {
        Ok(descriptors) => {
            if !descriptors.is_empty() {
                println!(
                    "   ✓ Extended descriptor dimension: {}",
                    descriptors[0].vector.len()
                );
            }
        }
        Err(e) => println!("   ✗ Extended SURF error: {}", e),
    }

    // ===================================================================
    // 2. ORB Feature Detection (Enhanced)
    // ===================================================================
    println!("\n2. ORB Feature Detection");
    println!("   ---------------------");

    let orb_config = OrbConfig::default();
    println!("   Detecting ORB features...");
    match detect_and_compute_orb(&test_img, &orb_config) {
        Ok(descriptors) => {
            println!("   ✓ Detected {} ORB features", descriptors.len());
        }
        Err(e) => println!("   ✗ ORB detection error: {}", e),
    }

    // ===================================================================
    // 3. Advanced Optical Flow
    // ===================================================================
    println!("\n3. Advanced Optical Flow Algorithms");
    println!("   ----------------------------------");

    let frame1 = create_test_image(128, 128);
    let frame2 = create_shifted_image(128, 128, 5, 5);

    // Farneback dense optical flow
    println!("   a) Farneback Dense Optical Flow");
    let farneback_params = FarnebackParams {
        pyr_scale: 0.5,
        levels: 3,
        winsize: 15,
        iterations: 3,
        poly_n: 5,
        poly_sigma: 1.2,
    };

    match farneback_flow(&frame1, &frame2, &farneback_params) {
        Ok(flow) => {
            let mag = flow.magnitude();
            let max_mag = mag.iter().fold(0.0f32, |a, &b| a.max(b));
            println!("      ✓ Farneback flow computed");
            println!("      ✓ Maximum flow magnitude: {:.2}", max_mag);
        }
        Err(e) => println!("      ✗ Farneback flow error: {}", e),
    }

    // TVL1 optical flow
    println!("   b) TVL1 Optical Flow");
    let tvl1_params = TVL1Params {
        tau: 0.25,
        lambda: 0.15,
        theta: 0.3,
        warps: 5,
        scales: 3,
        ..TVL1Params::default()
    };

    match tvl1_flow(&frame1, &frame2, &tvl1_params) {
        Ok(flow) => {
            let mag = flow.magnitude();
            let max_mag = mag.iter().fold(0.0f32, |a, &b| a.max(b));
            println!("      ✓ TVL1 flow computed");
            println!("      ✓ Maximum flow magnitude: {:.2}", max_mag);
        }
        Err(e) => println!("      ✗ TVL1 flow error: {}", e),
    }

    // Dual TVL1 optical flow
    println!("   c) Dual TVL1 Optical Flow");
    match dual_tvl1_flow(&frame1, &frame2, &tvl1_params) {
        Ok(flow) => {
            let mag = flow.magnitude();
            let max_mag = mag.iter().fold(0.0f32, |a, &b| a.max(b));
            println!("      ✓ Dual TVL1 flow computed");
            println!("      ✓ Maximum flow magnitude: {:.2}", max_mag);
        }
        Err(e) => println!("      ✗ Dual TVL1 flow error: {}", e),
    }

    // ===================================================================
    // 4. Semantic Segmentation with Deep Learning
    // ===================================================================
    println!("\n4. Semantic Segmentation with Deep Learning");
    println!("   -----------------------------------------");

    // DeepLab v3+
    println!("   a) DeepLab v3+ (PASCAL VOC classes)");
    let pascal_classes = create_pascal_voc_classes();
    let deeplab = DeepLabV3Plus::new(21, (256, 256), pascal_classes.clone());

    match deeplab.segment(&test_img_small) {
        Ok(result) => {
            println!("      ✓ Segmentation completed");
            println!("      ✓ Output dimensions: {:?}", result.class_map.dim());
            println!("      ✓ Number of classes: {}", result.classes.len());

            // Visualize segmentation
            match result.to_color_image() {
                Ok(_color_img) => {
                    println!("      ✓ Color visualization created");
                }
                Err(e) => println!("      ✗ Visualization error: {}", e),
            }
        }
        Err(e) => println!("      ✗ DeepLab segmentation error: {}", e),
    }

    // U-Net
    println!("   b) U-Net (Binary segmentation)");
    let unet_classes = vec![
        scirs2_vision::segmentation::SegmentationClass {
            id: 0,
            name: "background".to_string(),
            color: (0, 0, 0),
        },
        scirs2_vision::segmentation::SegmentationClass {
            id: 1,
            name: "foreground".to_string(),
            color: (255, 255, 255),
        },
    ];
    let unet = UNet::new(2, (256, 256), unet_classes);

    match unet.segment(&test_img_small) {
        Ok(result) => {
            println!("      ✓ U-Net segmentation completed");
            println!("      ✓ Output dimensions: {:?}", result.class_map.dim());
        }
        Err(e) => println!("      ✗ U-Net segmentation error: {}", e),
    }

    // FCN
    println!("   c) FCN (Fully Convolutional Network)");
    let fcn = FCN::new(21, FCNVariant::FCN8s, pascal_classes);

    match fcn.segment(&test_img_small) {
        Ok(result) => {
            println!("      ✓ FCN segmentation completed");
            println!("      ✓ Output dimensions: {:?}", result.class_map.dim());
        }
        Err(e) => println!("      ✗ FCN segmentation error: {}", e),
    }

    // ===================================================================
    // 5. 3D Vision - Stereo Matching and Depth Estimation
    // ===================================================================
    println!("\n5. 3D Vision - Stereo Matching and Depth Estimation");
    println!("   ------------------------------------------------");

    let left_img = create_test_image(256, 256);
    let right_img = create_shifted_image(256, 256, 10, 0); // Simulate stereo shift

    // Semi-Global Block Matching (SGBM)
    println!("   a) Semi-Global Block Matching (SGBM)");
    let sgbm_params = SGBMParams {
        min_disparity: 0,
        num_disparities: 64,
        block_size: 5,
        p1: 8 * 3 * 5 * 5,
        p2: 32 * 3 * 5 * 5,
        disp_12_max_diff: 1,
        pre_filter_cap: 63,
        uniqueness_ratio: 10,
        speckle_window_size: 100,
        speckle_range: 32,
    };

    match stereo_sgbm(&left_img, &right_img, &sgbm_params) {
        Ok(disparity) => {
            println!("      ✓ SGBM disparity computed");
            println!(
                "      ✓ Disparity map size: {:?}",
                disparity.disparity.dim()
            );

            // Convert to depth map
            let depth = disparity.to_depth_map(500.0, 0.1);
            println!("      ✓ Depth map created");
            let max_depth = depth
                .iter()
                .fold(0.0f32, |a, &b| if b > 0.0 { a.max(b) } else { a });
            println!("      ✓ Maximum depth: {:.2}m", max_depth);

            // Visualize disparity
            match disparity.visualize() {
                Ok(_vis_img) => {
                    println!("      ✓ Disparity visualization created");
                }
                Err(e) => println!("      ✗ Visualization error: {}", e),
            }
        }
        Err(e) => println!("      ✗ SGBM error: {}", e),
    }

    // Block Matching (BM)
    println!("   b) Block Matching (BM)");
    let bm_params = BMParams {
        min_disparity: 0,
        num_disparities: 64,
        block_size: 15,
    };

    match stereo_bm(&left_img, &right_img, &bm_params) {
        Ok(disparity) => {
            println!("      ✓ BM disparity computed");
            println!(
                "      ✓ Disparity map size: {:?}",
                disparity.disparity.dim()
            );
        }
        Err(e) => println!("      ✗ BM error: {}", e),
    }

    // 3D Point Cloud Triangulation
    println!("   c) 3D Point Cloud Triangulation");
    match stereo_sgbm(&left_img, &right_img, &sgbm_params) {
        Ok(disparity) => match triangulate_points(&disparity, 500.0, 0.1, 128.0, 128.0) {
            Ok(point_cloud) => {
                println!("      ✓ Point cloud created");
                println!("      ✓ Number of points: {}", point_cloud.points.nrows());
            }
            Err(e) => println!("      ✗ Triangulation error: {}", e),
        },
        Err(_) => println!("      ✗ Could not compute disparity for triangulation"),
    }

    // ===================================================================
    // Summary
    // ===================================================================
    println!("\n=== v0.2.0 Feature Summary ===");
    println!("✓ SURF feature detection with configurable parameters");
    println!("✓ Advanced optical flow algorithms (Farneback, TVL1, DualTVL1)");
    println!("✓ Semantic segmentation with deep learning (DeepLab, U-Net, FCN)");
    println!("✓ 3D vision capabilities (SGBM, BM stereo matching)");
    println!("✓ Depth estimation and point cloud generation");
    println!("✓ Production-ready computer vision capabilities");

    println!("\n=== Demo Complete ===");

    Ok(())
}

/// Create a test image with gradient pattern
fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;

            // Add some features
            let checker = ((x / 32) + (y / 32)) % 2;
            let intensity = if checker == 0 { 1.0 } else { 0.7 };

            img.put_pixel(
                x,
                y,
                image::Rgb([
                    (r as f32 * intensity) as u8,
                    (g as f32 * intensity) as u8,
                    (b as f32 * intensity) as u8,
                ]),
            );
        }
    }

    DynamicImage::ImageRgb8(img)
}

/// Create a shifted version of test image to simulate motion
fn create_shifted_image(width: u32, height: u32, shift_x: i32, shift_y: i32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let src_x = (x as i32 - shift_x).max(0).min(width as i32 - 1) as u32;
            let src_y = (y as i32 - shift_y).max(0).min(height as i32 - 1) as u32;

            let r = ((src_x as f32 / width as f32) * 255.0) as u8;
            let g = ((src_y as f32 / height as f32) * 255.0) as u8;
            let b = (((src_x + src_y) as f32 / (width + height) as f32) * 255.0) as u8;

            let checker = ((src_x / 32) + (src_y / 32)) % 2;
            let intensity = if checker == 0 { 1.0 } else { 0.7 };

            img.put_pixel(
                x,
                y,
                image::Rgb([
                    (r as f32 * intensity) as u8,
                    (g as f32 * intensity) as u8,
                    (b as f32 * intensity) as u8,
                ]),
            );
        }
    }

    DynamicImage::ImageRgb8(img)
}
