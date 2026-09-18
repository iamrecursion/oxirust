//! Color management and LUT processing demonstration.
//!
//! This example showcases:
//! - Standard color space enumeration (sRGB, Rec.709, Rec.2020, DCI-P3)
//! - Delta-E CIE76 and CIEDE2000 perceptual color difference
//! - 3D LUT creation (identity, Size17) with tetrahedral interpolation
//! - HDR-to-SDR tone mapping pipeline with PQ transfer function
//!
//! # Usage
//!
//! ```bash
//! cargo run --example color_pipeline --features "colormgmt,lut" -p oximedia
//! ```

use oximedia::colormgmt::{
    colorspaces::ColorSpace,
    delta_e::{delta_e_1976, delta_e_2000},
    xyz::Lab,
};
use oximedia::lut::{
    HdrPipeline, HdrToSdrParams, Lut3d, LutInterpolation, LutSize, ToneMappingAlgorithm,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Color Management & LUT Pipeline Demo");
    println!("==============================================\n");

    // ── Color space survey ────────────────────────────────────────────────────

    let spaces: &[(&str, Result<ColorSpace, _>)] = &[
        ("sRGB", ColorSpace::srgb()),
        ("Rec.709", ColorSpace::rec709()),
        ("Rec.2020", ColorSpace::rec2020()),
        ("DCI-P3", ColorSpace::dci_p3()),
        ("Display P3", ColorSpace::display_p3()),
    ];

    println!("Available color spaces:");
    for (label, result) in spaces.iter() {
        match result {
            Ok(cs) => println!("  {label:12} → name='{}'", cs.name),
            Err(e) => println!("  {label:12} → error: {e}"),
        }
    }

    // ACES spaces are defined in the aces sub-module (conceptual listing)
    println!("  ACES AP0      → ACES2065-1 (scene-linear, full-gamut)");
    println!("  ACES AP1      → ACEScg (scene-linear, working space)");

    // ── Delta-E color difference ──────────────────────────────────────────────

    // Crimson red vs. a slightly desaturated version in Lab
    let red_lab = Lab::new(40.0, 55.0, 30.0);
    let desaturated_lab = Lab::new(42.0, 35.0, 20.0);

    let de76 = delta_e_1976(&red_lab, &desaturated_lab);
    let de2000 = delta_e_2000(&red_lab, &desaturated_lab);

    println!("\nDelta-E color difference (crimson vs. desaturated red):");
    println!(
        "  Lab 1 : L={:.1} a={:.1} b={:.1}",
        red_lab.l, red_lab.a, red_lab.b
    );
    println!(
        "  Lab 2 : L={:.1} a={:.1} b={:.1}",
        desaturated_lab.l, desaturated_lab.a, desaturated_lab.b
    );
    println!("  ΔE 1976 (CIE76)    : {de76:.4}  (> 2 = noticeable)");
    println!("  ΔE 2000 (CIEDE2000): {de2000:.4}  (industry standard)");

    // Near-identical colors — should be imperceptible
    let near1 = Lab::new(50.0, 10.0, 5.0);
    let near2 = Lab::new(50.2, 10.1, 5.1);
    let de_near = delta_e_2000(&near1, &near2);
    println!("\n  Near-identical pair ΔE 2000: {de_near:.4}  (< 1.0 = imperceptible)");

    // ── 3D LUT — identity with tetrahedral interpolation ─────────────────────

    let lut = Lut3d::identity(LutSize::Size17);
    println!(
        "\n3D LUT (identity, {}³ = {} entries):",
        LutSize::Size17.as_usize(),
        LutSize::Size17.total_entries()
    );

    let sample_rgb = [0.6, 0.35, 0.15];
    let trilinear_out = lut.apply(&sample_rgb, LutInterpolation::Trilinear);
    let tetrahedral_out = lut.apply(&sample_rgb, LutInterpolation::Tetrahedral);

    println!(
        "  Input         : R={:.3} G={:.3} B={:.3}",
        sample_rgb[0], sample_rgb[1], sample_rgb[2]
    );
    println!(
        "  Trilinear out : R={:.4} G={:.4} B={:.4}",
        trilinear_out[0], trilinear_out[1], trilinear_out[2]
    );
    println!(
        "  Tetrahedral out: R={:.4} G={:.4} B={:.4}  (cinema-grade accuracy)",
        tetrahedral_out[0], tetrahedral_out[1], tetrahedral_out[2]
    );

    // ── HDR Pipeline — PQ to SDR tone mapping ────────────────────────────────

    let params = HdrToSdrParams::new()
        .with_algorithm(ToneMappingAlgorithm::AcesFilmic)
        .with_white_point(1000.0); // 1000-nit HDR10 mastering display

    let pipeline = HdrPipeline::new(params);

    println!("\nHDR-to-SDR Pipeline (PQ, 1000 nit → SDR 100 nit, ACES Filmic):");
    println!("  Source: BT.2020 PQ (HDR10)  →  Target: BT.709 (SDR)");

    // Representative HDR pixels: near-black, mid-grey, bright highlight
    let hdr_pixels: &[(&str, f32, f32, f32)] = &[
        ("near-black (0.1 PQ)", 0.10, 0.10, 0.10),
        ("mid-grey   (0.5 PQ)", 0.50, 0.50, 0.50),
        ("warm highlight (PQ)", 0.80, 0.55, 0.20),
        ("peak specular (0.9 PQ)", 0.90, 0.90, 0.90),
    ];

    for (label, r, g, b) in hdr_pixels {
        let (ro, go, bo) = pipeline.process_pixel(*r, *g, *b);
        println!("  {label:28} → R={ro:.3} G={go:.3} B={bo:.3}");
    }

    println!("\nColor pipeline demo complete.");
    Ok(())
}
