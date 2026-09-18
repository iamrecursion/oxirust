//! Usage examples and integration patterns for the benchmarking suite.
//!
//! This module provides comprehensive examples of how to use the benchmarking
//! suite for various scenarios.

use crate::*;
use oximedia_core::types::{CodecId, Rational};

/// Example: Quick benchmark comparing AV1 and VP9.
///
/// # Errors
///
/// Returns an error if the benchmark fails.
pub fn example_quick_comparison() -> BenchResult<BenchmarkResults> {
    // Create a quick benchmark configuration
    let config = BenchmarkPresets::quick();

    // Create and run the benchmark suite
    let suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Example: Comprehensive quality-focused benchmark.
///
/// # Errors
///
/// Returns an error if the benchmark fails.
pub fn example_quality_benchmark() -> BenchResult<BenchmarkResults> {
    let config = BenchmarkPresets::quality_focus();
    let suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Example: Custom codec configuration with specific parameters.
///
/// # Errors
///
/// Returns an error if the benchmark fails.
pub fn example_custom_codec_config() -> BenchResult<BenchmarkResults> {
    let av1_config = CodecConfig::new(CodecId::Av1)
        .with_preset("medium")
        .with_bitrate(2000)
        .with_passes(2)
        .with_rate_control(true)
        .with_param("cpu-used", "4")
        .with_param("threads", "8");

    let vp9_config = CodecConfig::new(CodecId::Vp9)
        .with_preset("good")
        .with_bitrate(2000)
        .with_param("cpu-used", "2");

    let config = BenchmarkConfig::builder()
        .add_codec(av1_config)
        .add_codec(vp9_config)
        .add_sequence("./sequences/test_1080p.y4m")
        .add_sequence("./sequences/test_4k.y4m")
        .parallel_jobs(4)
        .enable_psnr(true)
        .enable_ssim(true)
        .enable_vmaf(true)
        .cache_dir("./bench_cache")
        .output_dir("./bench_custom")
        .max_frames(300)
        .warmup_iterations(1)
        .measurement_iterations(3)
        .build()?;

    let suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Example: Exporting results in multiple formats.
///
/// # Errors
///
/// Returns an error if export fails.
pub fn example_export_results(results: &BenchmarkResults) -> BenchResult<()> {
    // Export to JSON
    results.export_json("results.json")?;

    // Export to CSV
    results.export_csv("results.csv")?;

    // Export to HTML
    results.export_html("results.html")?;

    // Use advanced HTML report
    let advanced_report = report::AdvancedHtmlReport::new(results)
        .with_charts(true)
        .with_detailed_stats(true);
    advanced_report.write_to_file("results_advanced.html")?;

    // Generate Markdown report
    let md_report = report::MarkdownReport::new(results);
    md_report.write_to_file("results.md")?;

    Ok(())
}

/// Example: Comparing two codecs.
///
/// # Errors
///
/// Returns an error if comparison fails.
pub fn example_codec_comparison(results: &BenchmarkResults) -> Option<ComparisonResult> {
    results.compare_codecs(CodecId::Av1, CodecId::Vp9)
}

/// Example: Filtering benchmark results.
///
/// # Errors
///
/// Returns an error if filtering fails.
pub fn example_filter_results(results: &BenchmarkResults) -> Vec<&CodecBenchmarkResult> {
    let filter = BenchmarkFilter::new()
        .with_min_encoding_fps(30.0)
        .with_min_psnr(35.0)
        .with_codec_ids(vec![CodecId::Av1, CodecId::Vp9]);

    filter.apply(results)
}

/// Example: Using test sequences.
///
/// # Errors
///
/// Returns an error if sequence operations fail.
pub fn example_test_sequences() -> BenchResult<()> {
    // Create a standard set of test sequences
    let _sequence_set = sequences::SequenceSet::standard_set();

    // Create a custom sequence
    let custom_seq = sequences::TestSequence::new(
        "my_test_sequence",
        "./sequences/my_test.y4m",
        1920,
        1080,
        Rational::new(30, 1),
    )
    .with_frame_count(300)
    .with_content_type(sequences::ContentType::Sports)
    .with_motion(sequences::MotionCharacteristics::High)
    .with_complexity(sequences::SceneComplexity::High);

    // Validate the sequence
    custom_seq.validate()?;

    // Use sequence database
    let mut db = sequences::SequenceDatabase::new();
    db.add(custom_seq);

    // Get sequences by resolution
    let hd_sequences = db.get_by_resolution(1920, 1080);
    println!("Found {} HD sequences", hd_sequences.len());

    // Export database
    db.export_to_json("sequences.json")?;

    Ok(())
}

/// Example: Calculating custom metrics.
///
/// # Errors
///
/// Returns an error if metric calculation fails.
pub fn example_custom_metrics() -> BenchResult<()> {
    // Calculate bitrate
    let bitrate = BenchmarkUtils::calculate_bitrate(10_000_000, 100.0);
    println!("Bitrate: {bitrate:.2} kbps");

    // Calculate bits per pixel
    let bpp = BenchmarkUtils::calculate_bpp(10_000_000, 1920, 1080, 100);
    println!("BPP: {bpp:.4}");

    // Calculate compression ratio
    let ratio = BenchmarkUtils::calculate_compression_ratio(100_000_000, 10_000_000);
    println!("Compression ratio: {ratio:.2}:1");

    // Format bytes
    let formatted = BenchmarkUtils::format_bytes(10_485_760);
    println!("File size: {formatted}");

    Ok(())
}

/// Example: Preset comparison.
///
/// # Errors
///
/// Returns an error if comparison fails.
pub fn example_preset_comparison() -> BenchResult<BenchmarkResults> {
    let config = BenchmarkConfig::builder()
        .add_codec(CodecConfig::new(CodecId::Av1).with_preset("ultrafast"))
        .add_codec(CodecConfig::new(CodecId::Av1).with_preset("fast"))
        .add_codec(CodecConfig::new(CodecId::Av1).with_preset("medium"))
        .add_codec(CodecConfig::new(CodecId::Av1).with_preset("slow"))
        .add_sequence("./sequences/test_1080p.y4m")
        .parallel_jobs(4)
        .enable_psnr(true)
        .enable_ssim(true)
        .build()?;

    let suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Example: Rate-distortion analysis.
///
/// # Errors
///
/// Returns an error if analysis fails.
pub fn example_rd_analysis() -> BenchResult<BenchmarkResults> {
    let bitrates = [500, 1000, 2000, 4000, 8000];

    let mut builder = BenchmarkConfig::builder();

    for bitrate in &bitrates {
        builder = builder.add_codec(
            CodecConfig::new(CodecId::Av1)
                .with_bitrate(*bitrate)
                .with_preset("medium"),
        );
    }

    let config = builder
        .add_sequence("./sequences/test_1080p.y4m")
        .parallel_jobs(4)
        .enable_psnr(true)
        .enable_ssim(true)
        .enable_vmaf(true)
        .build()?;

    let suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Example: Parallel execution of benchmarks.
///
/// # Errors
///
/// Returns an error if parallel execution fails.
pub fn example_parallel_execution() -> BenchResult<BenchmarkResults> {
    let config = BenchmarkConfig::builder()
        .add_codec(CodecConfig::new(CodecId::Av1))
        .add_codec(CodecConfig::new(CodecId::Vp9))
        .add_sequence("./sequences/seq1.y4m")
        .add_sequence("./sequences/seq2.y4m")
        .add_sequence("./sequences/seq3.y4m")
        .add_sequence("./sequences/seq4.y4m")
        .parallel_jobs(8) // Use 8 parallel jobs
        .enable_psnr(true)
        .enable_ssim(true)
        .build()?;

    let suite = BenchmarkSuite::new(config);
    suite.run_all()
}

/// Example: Using result cache for incremental benchmarks.
///
/// # Errors
///
/// Returns an error if caching fails.
pub fn example_cached_benchmarks() -> BenchResult<BenchmarkResults> {
    let config = BenchmarkConfig::builder()
        .add_codec(CodecConfig::new(CodecId::Av1))
        .add_sequence("./sequences/test_1080p.y4m")
        .cache_dir("./bench_cache") // Enable caching
        .enable_psnr(true)
        .enable_ssim(true)
        .build()?;

    let suite = BenchmarkSuite::new(config);

    // Load cache from previous runs
    suite.runner.load_cache()?;

    // Run benchmarks (will use cache when possible)
    let results = suite.run_all()?;

    // Save cache for next run
    suite.runner.save_cache()?;

    Ok(results)
}

/// Example: Statistical analysis of results.
///
/// # Errors
///
/// Returns an error if analysis fails.
pub fn example_statistical_analysis(results: &BenchmarkResults) -> BenchResult<()> {
    for codec_result in &results.codec_results {
        println!("Codec: {:?}", codec_result.codec_id);
        println!("Statistics:");
        println!(
            "  Mean Encoding FPS: {:.2}",
            codec_result.statistics.mean_encoding_fps
        );
        println!(
            "  Median Encoding FPS: {:.2}",
            codec_result.statistics.median_encoding_fps
        );
        println!(
            "  Std Dev Encoding FPS: {:.2}",
            codec_result.statistics.std_dev_encoding_fps
        );
        println!(
            "  95th Percentile: {:.2}",
            codec_result.statistics.p95_encoding_fps
        );
        println!(
            "  99th Percentile: {:.2}",
            codec_result.statistics.p99_encoding_fps
        );

        if let Some(mean_psnr) = codec_result.statistics.mean_psnr {
            println!("  Mean PSNR: {mean_psnr:.2} dB");
        }

        if let Some(mean_ssim) = codec_result.statistics.mean_ssim {
            println!("  Mean SSIM: {mean_ssim:.4}");
        }

        println!();
    }

    Ok(())
}

/// Example: Generating a comprehensive benchmark report.
///
/// # Errors
///
/// Returns an error if report generation fails.
pub fn example_comprehensive_report(results: &BenchmarkResults) -> BenchResult<()> {
    // Generate text summary
    let summary = BenchmarkUtils::generate_summary(results);
    std::fs::write("summary.txt", summary)?;

    // Generate advanced HTML report with charts
    let html_report = report::AdvancedHtmlReport::new(results)
        .with_charts(true)
        .with_detailed_stats(true);
    html_report.write_to_file("report_advanced.html")?;

    // Generate JSON for programmatic access
    let json_report = report::JsonReport::new(results).with_pretty(true);
    json_report.write_to_file("results.json")?;

    // Generate Markdown for documentation
    let md_report = report::MarkdownReport::new(results);
    md_report.write_to_file("BENCHMARK_RESULTS.md")?;

    Ok(())
}

/// Example: Command-line interface helper usage.
///
/// # Errors
///
/// Returns an error if CLI operations fail.
pub fn example_cli_helpers() -> BenchResult<()> {
    // Parse codec from string
    let codec = CliHelpers::parse_codec("av1")?;
    println!("Parsed codec: {codec:?}");

    // Generate example configuration
    let example_config = CliHelpers::generate_example_config();
    std::fs::write("example_config.json", example_config)?;

    // Show progress
    for i in 0..=100 {
        CliHelpers::print_progress(i, 100, 50);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    CliHelpers::clear_progress();
    println!("Done!");

    Ok(())
}

/// Example: Complete workflow from configuration to reporting.
///
/// # Errors
///
/// Returns an error if any step fails.
pub fn example_complete_workflow() -> BenchResult<()> {
    // Step 1: Create configuration
    let config = BenchmarkConfig::builder()
        .add_codec(CodecConfig::new(CodecId::Av1).with_preset("medium"))
        .add_codec(CodecConfig::new(CodecId::Vp9).with_preset("good"))
        .add_sequence("./sequences/test_1080p.y4m")
        .parallel_jobs(4)
        .enable_psnr(true)
        .enable_ssim(true)
        .enable_vmaf(false)
        .cache_dir("./bench_cache")
        .output_dir("./bench_results")
        .max_frames(300)
        .warmup_iterations(1)
        .measurement_iterations(3)
        .build()?;

    // Step 2: Run benchmarks
    let suite = BenchmarkSuite::new(config);
    let results = suite.run_all()?;

    // Step 3: Analyze results
    example_statistical_analysis(&results)?;

    // Step 4: Compare codecs
    if let Some(comparison) = example_codec_comparison(&results) {
        println!(
            "Encoding speed ratio: {:.2}",
            comparison.encoding_speed_ratio
        );
        println!(
            "Decoding speed ratio: {:.2}",
            comparison.decoding_speed_ratio
        );
    }

    // Step 5: Filter results
    let filtered = example_filter_results(&results);
    println!("Filtered results: {} codecs meet criteria", filtered.len());

    // Step 6: Generate reports
    example_comprehensive_report(&results)?;

    // Step 7: Export in multiple formats
    example_export_results(&results)?;

    println!("Complete workflow finished successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_preset_names() {
        let config = BenchmarkPresets::quick();
        assert!(!config.codecs.is_empty());
    }

    #[test]
    fn test_example_custom_metrics() {
        assert!(example_custom_metrics().is_ok());
    }

    #[test]
    fn test_example_cli_parse() {
        assert!(CliHelpers::parse_codec("av1").is_ok());
        assert!(CliHelpers::parse_codec("vp9").is_ok());
        assert!(CliHelpers::parse_codec("invalid").is_err());
    }

    #[test]
    fn test_example_filter() {
        let results = BenchmarkResults {
            codec_results: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            total_duration: Duration::from_secs(100),
            config: BenchmarkConfig::default(),
        };

        let filtered = example_filter_results(&results);
        assert_eq!(filtered.len(), 0);
    }
}
