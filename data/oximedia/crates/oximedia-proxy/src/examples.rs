//! Examples and usage demonstrations for OxiMedia Proxy.
#![allow(dead_code)]
use crate::*;
/// Example: Basic proxy generation.
///
/// ```no_run
/// use oximedia_proxy::{ProxyGenerator, ProxyPreset};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let generator = ProxyGenerator::new();
///
/// // Generate a quarter-resolution proxy
/// generator
///     .generate("input.mov", "proxy.mp4", ProxyPreset::QuarterResH264)
///     .await?;
/// # Ok(())
/// # }
/// ```
pub mod basic_generation {
    use super::*;
    /// Generate a single proxy with default settings.
    pub async fn generate_simple_proxy(input: &str, output: &str) -> Result<ProxyEncodeResult> {
        let generator = ProxyGenerator::new();
        generator
            .generate(input, output, ProxyPreset::QuarterResH264)
            .await
    }
    /// Generate a proxy with custom settings.
    pub async fn generate_custom_proxy(
        input: &str,
        output: &str,
        scale_factor: f32,
        bitrate: u64,
    ) -> Result<ProxyEncodeResult> {
        let settings = ProxyGenerationSettings::default()
            .with_scale_factor(scale_factor)
            .with_bitrate(bitrate);
        let generator = ProxyGenerator::new();
        generator
            .generate_with_settings(input, output, settings)
            .await
    }
}
/// Example: Batch proxy generation.
///
/// ```no_run
/// use oximedia_proxy::{BatchProxyGenerator, ProxyGenerationSettings};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let settings = ProxyGenerationSettings::quarter_res_h264();
/// let batch_generator = BatchProxyGenerator::new(settings);
///
/// let inputs = vec![
///     (std::path::PathBuf::from("clip1.mov"), std::path::PathBuf::from("proxy1.mp4")),
///     (std::path::PathBuf::from("clip2.mov"), std::path::PathBuf::from("proxy2.mp4")),
/// ];
///
/// let results = batch_generator.generate_batch(&inputs).await?;
/// # Ok(())
/// # }
/// ```
pub mod batch_generation {
    use super::*;
    use std::path::PathBuf;
    /// Generate proxies for multiple files in parallel.
    pub async fn batch_generate(
        inputs: &[(PathBuf, PathBuf)],
        preset: ProxyPreset,
    ) -> Result<Vec<BatchResult>> {
        let settings = preset.to_settings();
        let generator = BatchProxyGenerator::new(settings);
        generator.generate_batch(inputs).await
    }
    /// Generate proxies with progress reporting.
    pub async fn batch_generate_with_progress<F>(
        inputs: &[(PathBuf, PathBuf)],
        preset: ProxyPreset,
        progress_callback: F,
    ) -> Result<Vec<BatchResult>>
    where
        F: FnMut(usize, usize) + Send,
    {
        let settings = preset.to_settings();
        let generator = BatchProxyGenerator::new(settings);
        generator
            .generate_batch_with_progress(inputs, progress_callback)
            .await
    }
}
/// Example: Proxy linking and management.
///
/// ```no_run
/// use oximedia_proxy::ProxyLinkManager;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut manager = ProxyLinkManager::new("links.db").await?;
///
/// // Link proxy to original
/// manager.link_proxy("proxy.mp4", "original.mov").await?;
///
/// // Get original path from proxy
/// let original = manager.get_original("proxy.mp4")?;
/// println!("Original: {}", original.display());
/// # Ok(())
/// # }
/// ```
pub mod link_management {
    use super::*;
    use std::collections::HashMap;
    /// Create and manage proxy links.
    pub async fn create_links(db_path: &str, proxies: &[(String, String)]) -> Result<()> {
        let mut manager = ProxyLinkManager::new(db_path).await?;
        for (proxy, original) in proxies {
            manager
                .link_proxy_with_metadata(proxy, original, 0.25, "h264", 0.0, None, HashMap::new())
                .await?;
        }
        Ok(())
    }
    /// Verify all proxy links.
    pub fn verify_all_links(manager: &mut ProxyLinkManager) -> Result<usize> {
        let all_links = manager.all_links();
        let mut valid_count = 0;
        for link in &all_links {
            if manager.verify_link(&link.proxy_path)? {
                valid_count += 1;
            }
        }
        Ok(valid_count)
    }
}
/// Example: Complete offline editing workflow.
///
/// ```no_run
/// use oximedia_proxy::{OfflineWorkflow, ProxyPreset};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut workflow = OfflineWorkflow::new("project.db").await?;
///
/// // Step 1: Ingest and create proxies
/// workflow.ingest(
///     "camera/clip001.mov",
///     "proxies/clip001.mp4",
///     ProxyPreset::QuarterResH264
/// ).await?;
///
/// // Step 2: Edit with proxies (use your NLE)
///
/// // Step 3: Conform to original
/// workflow.conform("edit.edl", "final.mov").await?;
/// # Ok(())
/// # }
/// ```
pub mod offline_workflow {
    use super::*;
    /// Complete offline-to-online workflow.
    pub async fn complete_workflow(
        db_path: &str,
        camera_files: &[&str],
        proxy_dir: &str,
        edl_path: &str,
        output: &str,
    ) -> Result<ConformResult> {
        let mut workflow = OfflineWorkflow::new(db_path).await?;
        // Phase 1: Ingest
        for (i, camera_file) in camera_files.iter().enumerate() {
            let proxy_path = format!("{}/proxy_{:03}.mp4", proxy_dir, i);
            workflow
                .ingest(camera_file, &proxy_path, ProxyPreset::QuarterResH264)
                .await?;
        }
        // Phase 2: Edit (external)
        tracing::info!("Edit your proxies in your NLE");
        // Phase 3: Conform
        workflow.conform(edl_path, output).await
    }
}
/// Example: Workflow planning and estimation.
///
/// ```no_run
/// use oximedia_proxy::{WorkflowPlanner, MediaInfo, ProxyGenerationSettings};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let planner = WorkflowPlanner::new();
///
/// let inputs = vec![
///     MediaInfo::from_path("clip1.mov")?,
///     MediaInfo::from_path("clip2.mov")?,
/// ];
///
/// let settings = ProxyGenerationSettings::quarter_res_h264();
/// let plan = planner.plan_generation(&inputs, settings)?;
///
/// println!("Estimated encoding time: {:.1} minutes", plan.estimated_encoding_time / 60.0);
/// println!("Space savings: {:.1}%", (1.0 - plan.compression_ratio) * 100.0);
/// # Ok(())
/// # }
/// ```
pub mod workflow_planning {
    use super::*;
    /// Plan a workflow and get estimates.
    pub fn plan_workflow(media_files: &[MediaInfo], preset: ProxyPreset) -> Result<WorkflowPlan> {
        let planner = WorkflowPlanner::new();
        let settings = preset.to_settings();
        planner.plan_generation(media_files, settings)
    }
    /// Estimate storage requirements.
    pub fn estimate_storage(
        media_files: &[MediaInfo],
        preset: ProxyPreset,
        keep_originals: bool,
    ) -> StorageEstimate {
        let planner = WorkflowPlanner::new();
        let settings = preset.to_settings();
        planner.estimate_storage(media_files, &settings, keep_originals)
    }
}
/// Example: Validation and quality assurance.
///
/// ```no_run
/// use oximedia_proxy::{ValidationChecker, ProxyLinkManager};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let manager = ProxyLinkManager::new("links.db").await?;
/// let checker = ValidationChecker::new(&manager);
///
/// let report = checker.validate()?;
///
/// if report.is_valid() {
///     println!("All proxies are valid!");
/// } else {
///     println!("Found {} errors", report.error_count());
///     for error in &report.errors {
///         println!("  - {}", error);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub mod validation {
    use super::*;
    /// Validate all proxy links and report issues.
    pub fn validate_workflow(manager: &ProxyLinkManager) -> Result<ValidationReport> {
        let checker = ValidationChecker::new(manager);
        checker.validate()
    }
    /// Perform strict validation with comprehensive checks.
    pub fn strict_validation(manager: &ProxyLinkManager) -> Result<ValidationReport> {
        let validator = WorkflowValidator::new(manager).strict();
        validator.validate_all()
    }
}
/// Example: Cache management.
///
/// ```no_run
/// use oximedia_proxy::{CacheManager, CacheStrategy};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let cache_dir = std::path::PathBuf::from("/var/cache/proxies");
/// let max_size = 10 * 1024 * 1024 * 1024; // 10 GB
///
/// let mut cache = CacheManager::new(cache_dir, max_size);
/// cache.set_strategy(CacheStrategy::Lru);
///
/// // Add proxies to cache
/// cache.add(std::path::PathBuf::from("proxy1.mp4"), 100_000_000);
/// cache.add(std::path::PathBuf::from("proxy2.mp4"), 150_000_000);
///
/// println!("Cache utilization: {:.1}%", cache.utilization());
/// # Ok(())
/// # }
/// ```
pub mod cache_management {
    use super::*;
    /// Set up and manage proxy cache.
    pub fn setup_cache(
        cache_dir: std::path::PathBuf,
        max_size: u64,
        strategy: CacheStrategy,
    ) -> CacheManager {
        let mut cache = CacheManager::new(cache_dir, max_size);
        cache.set_strategy(strategy);
        cache
    }
    /// Clean up cache based on policy.
    pub fn cleanup_cache(cache_dir: std::path::PathBuf, policy: CleanupPolicy) -> Result<()> {
        let cleanup = CacheCleanup::new(cache_dir);
        let result = cleanup.cleanup(policy)?;
        tracing::info!(
            "Cleaned up {} files, freed {} bytes",
            result.files_removed,
            result.bytes_freed
        );
        Ok(())
    }
}
/// Example: Multi-resolution proxy management.
///
/// ```no_run
/// use oximedia_proxy::{ResolutionManager, ProxyResolution, ProxyVariant};
/// use std::path::PathBuf;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut manager = ResolutionManager::new();
///
/// let original = PathBuf::from("original.mov");
///
/// // Add quarter resolution variant
/// manager.add_variant(
///     original.clone(),
///     ProxyVariant {
///         resolution: ProxyResolution::Quarter,
///         path: PathBuf::from("proxy_quarter.mp4"),
///         file_size: 50_000_000,
///         codec: "h264".to_string(),
///     },
/// );
///
/// // Add half resolution variant
/// manager.add_variant(
///     original.clone(),
///     ProxyVariant {
///         resolution: ProxyResolution::Half,
///         path: PathBuf::from("proxy_half.mp4"),
///         file_size: 150_000_000,
///         codec: "h264".to_string(),
///     },
/// );
///
/// // Get best variant for target resolution
/// let variant = manager.get_best_variant(&original, ProxyResolution::Quarter);
/// # Ok(())
/// # }
/// ```
pub mod resolution_management {
    use super::*;
    use std::path::PathBuf;
    /// Create multi-resolution proxy set.
    pub fn create_multiresolution_proxies(
        original: &str,
        output_dir: &str,
    ) -> Result<ResolutionManager> {
        let mut manager = ResolutionManager::new();
        let original_path = PathBuf::from(original);
        let resolutions = vec![
            (ProxyResolution::Quarter, "quarter"),
            (ProxyResolution::Half, "half"),
            (ProxyResolution::Full, "full"),
        ];
        for (resolution, suffix) in resolutions {
            let proxy_path = PathBuf::from(format!("{}/proxy_{}.mp4", output_dir, suffix));
            manager.add_variant(
                original_path.clone(),
                ProxyVariant {
                    resolution,
                    path: proxy_path,
                    file_size: 0,
                    codec: "h264".to_string(),
                },
            );
        }
        Ok(manager)
    }
}
/// Example: Statistics and analytics.
///
/// ```no_run
/// use oximedia_proxy::{LinkStatistics, LinkDatabase};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let db = LinkDatabase::new("links.db").await?;
/// let stats = LinkStatistics::new(&db);
///
/// let report = stats.collect();
///
/// println!("Total links: {}", report.total_links);
/// println!("Compression ratio: {:.2}:1", 1.0 / report.compression_ratio);
/// println!("Space saved: {} bytes", report.space_saved);
/// println!("\n{}", report.summary());
/// # Ok(())
/// # }
/// ```
pub mod statistics {
    use super::*;
    /// Collect and display comprehensive statistics.
    pub async fn show_statistics(db_path: &str) -> Result<()> {
        let db = LinkDatabase::new(db_path).await?;
        let collector = LinkStatistics::new(&db);
        let stats = collector.collect();
        println!("=== Proxy Statistics ===");
        println!("{}", stats.summary());
        if let Some(codec) = stats.most_used_codec() {
            let codec_stats = collector.codec_statistics(&codec);
            println!("\nMost used codec: {}", codec);
            println!("  Files: {}", codec_stats.count);
            println!(
                "  Avg bitrate: {} Mbps",
                codec_stats.avg_bitrate / 1_000_000
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn test_examples_compile() {
        // This test just ensures all example modules compile
        assert!(true);
    }
}
