//! Accessibility compliance checking example.

use oximedia_access::compliance::{
    ComplianceChecker, EbuChecker, Section508Checker, WcagChecker, WcagLevel,
};
use oximedia_access::visual::contrast::ContrastEnhancer;

fn main() {
    println!("Accessibility Compliance Checking Example\n");
    println!("==========================================\n");

    // Create compliance checker
    let checker = ComplianceChecker::new();

    // Example 1: WCAG compliance checking
    println!("1. WCAG 2.1 Level AA Compliance\n");

    let wcag_checker = WcagChecker::new(WcagLevel::AA);

    // Check if captions are present
    let has_captions = true; // Example: content has captions
    if let Some(issue) = wcag_checker.check_captions_present(has_captions) {
        println!("   Issue: {} - {}", issue.title, issue.description);
    } else {
        println!("   ✓ Captions present (WCAG 1.2.2)");
    }

    // Check if audio description is present
    let has_audio_desc = false; // Example: content lacks audio description
    if let Some(issue) = wcag_checker.check_audio_description(has_audio_desc) {
        println!("   ✗ {} - {}", issue.title, issue.description);
    } else {
        println!("   ✓ Audio description present (WCAG 1.2.5)");
    }

    // Check contrast ratios
    println!("\n   Contrast Ratio Checks:");

    let test_cases = vec![
        ((0, 0, 0), (255, 255, 255), "Black on White"),
        ((255, 255, 255), (0, 0, 0), "White on Black"),
        ((128, 128, 128), (255, 255, 255), "Gray on White"),
        ((0, 0, 255), (255, 255, 255), "Blue on White"),
    ];

    for (bg, fg, description) in test_cases {
        let ratio = ContrastEnhancer::contrast_ratio(fg, bg);
        let meets_aa = ContrastEnhancer::meets_wcag_aa(fg, bg);
        let meets_aaa = ContrastEnhancer::meets_wcag_aaa(fg, bg);

        println!("   {description} - Ratio: {ratio:.2}:1");
        println!("      AA:  {}", if meets_aa { "✓ Pass" } else { "✗ Fail" });
        println!("      AAA: {}", if meets_aaa { "✓ Pass" } else { "✗ Fail" });
    }

    // Example 2: Section 508 compliance
    println!("\n2. Section 508 Compliance\n");

    let section508 = Section508Checker::new();

    if let Some(issue) = section508.check_synchronized_captions(has_captions) {
        println!("   ✗ {} - {}", issue.title, issue.description);
    } else {
        println!("   ✓ Synchronized captions present");
    }

    if let Some(issue) = section508.check_audio_descriptions(has_audio_desc) {
        println!("   ✗ {} - {}", issue.title, issue.description);
    } else {
        println!("   ✓ Audio descriptions present");
    }

    // Example 3: EBU compliance
    println!("\n3. EBU (European Broadcasting Union) Compliance\n");

    let ebu = EbuChecker::new();

    // Check loudness normalization (EBU R128)
    let loudness_tests = vec![
        (-23.0, "Correct loudness"),
        (-20.0, "Too loud"),
        (-26.0, "Too quiet"),
    ];

    println!("   Loudness (EBU R128 target: -23.0 LUFS ±1.0):");
    for (loudness, description) in loudness_tests {
        if let Some(issue) = ebu.check_loudness(loudness) {
            println!(
                "   ✗ {} - {:.1} LUFS: {}",
                description, loudness, issue.description
            );
        } else {
            println!("   ✓ {description} - {loudness:.1} LUFS");
        }
    }

    // Check subtitle format (EBU-TT-D)
    println!("\n   Subtitle Format (EBU-TT-D):");

    let subtitle_tests = vec![(35, "Normal subtitle"), (50, "Too long subtitle")];

    for (chars, description) in subtitle_tests {
        if let Some(issue) = ebu.check_subtitle_format(chars) {
            println!(
                "   ✗ {} ({} chars): {}",
                description, chars, issue.description
            );
        } else {
            println!("   ✓ {description} ({chars} chars)");
        }
    }

    // Check subtitle duration
    println!("\n   Subtitle Duration (1-7 seconds):");

    let duration_tests = vec![
        (500, "Too short"),
        (3000, "Good duration"),
        (8000, "Too long"),
    ];

    for (duration_ms, description) in duration_tests {
        if let Some(issue) = ebu.check_subtitle_duration(duration_ms) {
            println!(
                "   ✗ {} ({}ms): {}",
                description, duration_ms, issue.description
            );
        } else {
            println!("   ✓ {description} ({duration_ms}ms)");
        }
    }

    // Example 4: Generate compliance report
    println!("\n4. Complete Compliance Report\n");

    let report = checker.check_all();

    println!("{}", report.to_text());

    // Export as JSON
    if let Ok(json) = report.to_json() {
        println!("\nJSON Report ({} bytes):", json.len());
        println!("{json}");
    }

    println!("\nCompliance checking complete!");
}
