//! Proxy workflow for efficient editing.
//!
//! Enables low-resolution editing with full-resolution export. Proxy
//! files are lightweight stand-ins for high-resolution source media,
//! allowing smooth playback and editing on modest hardware. At export
//! time the renderer seamlessly switches back to original media.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::clip::ClipId;
use crate::error::{EditError, EditResult};

/// Resolution preset for proxy generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyResolution {
    /// Quarter of original resolution.
    Quarter,
    /// Half of original resolution.
    Half,
    /// 720p (1280x720).
    Hd720,
    /// 480p (854x480).
    Sd480,
    /// Custom resolution.
    Custom(u32, u32),
}

impl ProxyResolution {
    /// Compute the proxy dimensions given the original width and height.
    #[must_use]
    pub fn dimensions(&self, original_width: u32, original_height: u32) -> (u32, u32) {
        match self {
            Self::Quarter => ((original_width / 4).max(1), (original_height / 4).max(1)),
            Self::Half => ((original_width / 2).max(1), (original_height / 2).max(1)),
            Self::Hd720 => (1280, 720),
            Self::Sd480 => (854, 480),
            Self::Custom(w, h) => (*w, *h),
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Quarter => "1/4 Resolution",
            Self::Half => "1/2 Resolution",
            Self::Hd720 => "720p",
            Self::Sd480 => "480p",
            Self::Custom(_, _) => "Custom",
        }
    }
}

/// Status of a proxy file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyStatus {
    /// Proxy has not been generated yet.
    NotGenerated,
    /// Proxy generation is in progress.
    Generating,
    /// Proxy is ready to use.
    Ready,
    /// Proxy generation failed.
    Failed,
    /// Proxy is outdated (source changed since generation).
    Outdated,
}

impl ProxyStatus {
    /// Returns `true` if the proxy can be used for editing.
    #[must_use]
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// A mapping between an original media file and its proxy.
#[derive(Debug, Clone)]
pub struct ProxyMapping {
    /// Original (full-resolution) file path.
    pub original_path: PathBuf,
    /// Proxy (low-resolution) file path.
    pub proxy_path: PathBuf,
    /// Resolution of the proxy.
    pub resolution: ProxyResolution,
    /// Status of the proxy.
    pub status: ProxyStatus,
    /// Original file dimensions.
    pub original_width: u32,
    /// Original file height.
    pub original_height: u32,
    /// Proxy dimensions.
    pub proxy_width: u32,
    /// Proxy height.
    pub proxy_height: u32,
}

impl ProxyMapping {
    /// Create a new proxy mapping.
    #[must_use]
    pub fn new(
        original_path: PathBuf,
        proxy_path: PathBuf,
        resolution: ProxyResolution,
        original_width: u32,
        original_height: u32,
    ) -> Self {
        let (pw, ph) = resolution.dimensions(original_width, original_height);
        Self {
            original_path,
            proxy_path,
            resolution,
            status: ProxyStatus::NotGenerated,
            original_width,
            original_height,
            proxy_width: pw,
            proxy_height: ph,
        }
    }

    /// Get the scale factor (proxy / original).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scale_factor(&self) -> f64 {
        if self.original_width == 0 {
            return 1.0;
        }
        self.proxy_width as f64 / self.original_width as f64
    }
}

/// Proxy editing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyMode {
    /// Use original files for playback and export.
    Original,
    /// Use proxy files for playback, originals for export.
    ProxyPlayback,
    /// Use proxy files for everything (fastest).
    ProxyOnly,
}

impl ProxyMode {
    /// Returns `true` if proxies are used for playback.
    #[must_use]
    pub fn uses_proxy_for_playback(self) -> bool {
        matches!(self, Self::ProxyPlayback | Self::ProxyOnly)
    }

    /// Returns `true` if originals are used for export.
    #[must_use]
    pub fn uses_original_for_export(self) -> bool {
        matches!(self, Self::Original | Self::ProxyPlayback)
    }
}

/// Manages proxy files for a project.
#[derive(Debug)]
pub struct ProxyManager {
    /// Proxy mappings keyed by original path string.
    mappings: HashMap<String, ProxyMapping>,
    /// Clip to original path mapping.
    clip_sources: HashMap<ClipId, String>,
    /// Current proxy mode.
    pub mode: ProxyMode,
    /// Default proxy resolution.
    pub default_resolution: ProxyResolution,
    /// Proxy storage directory.
    pub proxy_dir: PathBuf,
}

impl ProxyManager {
    /// Create a new proxy manager.
    #[must_use]
    pub fn new(proxy_dir: PathBuf) -> Self {
        Self {
            mappings: HashMap::new(),
            clip_sources: HashMap::new(),
            mode: ProxyMode::ProxyPlayback,
            default_resolution: ProxyResolution::Half,
            proxy_dir,
        }
    }

    /// Register a source file for proxy management.
    pub fn register_source(
        &mut self,
        original_path: PathBuf,
        original_width: u32,
        original_height: u32,
    ) -> EditResult<()> {
        let key = original_path
            .to_str()
            .ok_or_else(|| EditError::InvalidEdit("Invalid path encoding".to_string()))?
            .to_string();

        let source_name = original_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        let proxy_filename = format!("proxy_{source_name}");
        let proxy_path = self.proxy_dir.join(proxy_filename);

        let mapping = ProxyMapping::new(
            original_path,
            proxy_path,
            self.default_resolution,
            original_width,
            original_height,
        );

        self.mappings.insert(key, mapping);
        Ok(())
    }

    /// Associate a clip with a source file.
    pub fn associate_clip(&mut self, clip_id: ClipId, original_path: &str) {
        self.clip_sources.insert(clip_id, original_path.to_string());
    }

    /// Get the appropriate file path for a clip given the current mode.
    ///
    /// For playback: returns proxy path if mode uses proxies and proxy is ready.
    /// For export: returns original path unless in ProxyOnly mode.
    #[must_use]
    pub fn resolve_path_for_playback(&self, clip_id: ClipId) -> Option<&PathBuf> {
        let source_key = self.clip_sources.get(&clip_id)?;
        let mapping = self.mappings.get(source_key)?;
        if self.mode.uses_proxy_for_playback() && mapping.status.is_usable() {
            Some(&mapping.proxy_path)
        } else {
            Some(&mapping.original_path)
        }
    }

    /// Get the path to use during export.
    #[must_use]
    pub fn resolve_path_for_export(&self, clip_id: ClipId) -> Option<&PathBuf> {
        let source_key = self.clip_sources.get(&clip_id)?;
        let mapping = self.mappings.get(source_key)?;
        if self.mode.uses_original_for_export() {
            Some(&mapping.original_path)
        } else if mapping.status.is_usable() {
            Some(&mapping.proxy_path)
        } else {
            Some(&mapping.original_path)
        }
    }

    /// Mark a proxy as ready.
    pub fn mark_ready(&mut self, original_path: &str) -> bool {
        if let Some(mapping) = self.mappings.get_mut(original_path) {
            mapping.status = ProxyStatus::Ready;
            true
        } else {
            false
        }
    }

    /// Mark a proxy as failed.
    pub fn mark_failed(&mut self, original_path: &str) -> bool {
        if let Some(mapping) = self.mappings.get_mut(original_path) {
            mapping.status = ProxyStatus::Failed;
            true
        } else {
            false
        }
    }

    /// Mark a proxy as outdated.
    pub fn mark_outdated(&mut self, original_path: &str) -> bool {
        if let Some(mapping) = self.mappings.get_mut(original_path) {
            mapping.status = ProxyStatus::Outdated;
            true
        } else {
            false
        }
    }

    /// Get the proxy mapping for a source file.
    #[must_use]
    pub fn get_mapping(&self, original_path: &str) -> Option<&ProxyMapping> {
        self.mappings.get(original_path)
    }

    /// Get all mappings that need proxy generation.
    #[must_use]
    pub fn pending_generation(&self) -> Vec<&ProxyMapping> {
        self.mappings
            .values()
            .filter(|m| matches!(m.status, ProxyStatus::NotGenerated | ProxyStatus::Outdated))
            .collect()
    }

    /// Get total number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.mappings.len()
    }

    /// Get count of ready proxies.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.mappings
            .values()
            .filter(|m| m.status.is_usable())
            .count()
    }
}

/// Codec to use for proxy files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyCodec {
    /// VP9 (default, patent-free).
    Vp9,
    /// AV1 (smaller, slower encode).
    Av1,
    /// VP8 (fastest encode).
    Vp8,
}

impl ProxyCodec {
    /// Human-readable label for this codec.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
            Self::Vp8 => "VP8",
        }
    }
}

/// Configuration for proxy generation workflow.
#[derive(Debug, Clone)]
pub struct ProxyWorkflowConfig {
    /// Target resolution.
    pub resolution: ProxyResolution,
    /// Codec for proxy encoding.
    pub codec: ProxyCodec,
    /// Quality / CRF parameter (0–63, lower = better).
    pub quality: u8,
    /// Whether to preserve audio in proxy.
    pub include_audio: bool,
    /// Maximum concurrent generation tasks.
    pub max_concurrent: usize,
}

impl Default for ProxyWorkflowConfig {
    fn default() -> Self {
        Self {
            resolution: ProxyResolution::Half,
            codec: ProxyCodec::Vp9,
            quality: 35,
            include_audio: true,
            max_concurrent: 4,
        }
    }
}

/// Progress of a single proxy generation job.
#[derive(Debug, Clone)]
pub struct ProxyJobProgress {
    /// Original file path key.
    pub source_key: String,
    /// 0.0 – 1.0 progress fraction.
    pub fraction: f64,
    /// Estimated remaining seconds (-1 if unknown).
    pub eta_seconds: f64,
    /// Current stage description.
    pub stage: String,
}

impl ProxyJobProgress {
    /// Create a new progress entry.
    #[must_use]
    pub fn new(source_key: String) -> Self {
        Self {
            source_key,
            fraction: 0.0,
            eta_seconds: -1.0,
            stage: "queued".to_string(),
        }
    }

    /// Whether the job is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        (self.fraction - 1.0).abs() < 1e-9
    }
}

/// Multi-resolution proxy chain for zoom-dependent switching.
#[derive(Debug, Clone)]
pub struct ProxyChain {
    /// Ordered from lowest to highest resolution proxy mappings.
    entries: Vec<ProxyChainEntry>,
    /// Original source path.
    pub source_path: String,
}

/// One resolution level in the proxy chain.
#[derive(Debug, Clone)]
pub struct ProxyChainEntry {
    /// Resolution enum for this level.
    pub resolution: ProxyResolution,
    /// Proxy file path.
    pub proxy_path: PathBuf,
    /// Scale factor relative to original (0.0–1.0).
    pub scale: f64,
    /// Status of this level.
    pub status: ProxyStatus,
}

impl ProxyChain {
    /// Create a new proxy chain for a given source.
    #[must_use]
    pub fn new(source_path: String) -> Self {
        Self {
            entries: Vec::new(),
            source_path,
        }
    }

    /// Add a resolution level.
    pub fn add_level(&mut self, resolution: ProxyResolution, proxy_path: PathBuf, scale: f64) {
        let entry = ProxyChainEntry {
            resolution,
            proxy_path,
            scale: scale.clamp(0.0, 1.0),
            status: ProxyStatus::NotGenerated,
        };
        self.entries.push(entry);
        // Sort by scale ascending (lowest res first).
        self.entries.sort_by(|a, b| {
            a.scale
                .partial_cmp(&b.scale)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Select the best proxy for a given zoom level (0.0–1.0 where 1.0 = full res).
    ///
    /// Returns the entry whose scale is >= zoom that is ready, or falls back
    /// to the highest-resolution ready entry.
    #[must_use]
    pub fn select_for_zoom(&self, zoom: f64) -> Option<&ProxyChainEntry> {
        // First try: smallest ready proxy that is >= zoom
        let candidate = self
            .entries
            .iter()
            .find(|e| e.status.is_usable() && e.scale >= zoom);
        if candidate.is_some() {
            return candidate;
        }
        // Fallback: highest-resolution ready proxy
        self.entries.iter().rev().find(|e| e.status.is_usable())
    }

    /// Mark a level as ready by scale.
    pub fn mark_ready_by_scale(&mut self, scale: f64) -> bool {
        for entry in &mut self.entries {
            if (entry.scale - scale).abs() < 1e-6 {
                entry.status = ProxyStatus::Ready;
                return true;
            }
        }
        false
    }

    /// Number of levels.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of ready levels.
    #[must_use]
    pub fn ready_level_count(&self) -> usize {
        self.entries.iter().filter(|e| e.status.is_usable()).count()
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[ProxyChainEntry] {
        &self.entries
    }
}

/// Relink result when matching proxies back to originals.
#[derive(Debug, Clone)]
pub struct RelinkResult {
    /// Proxy path that was relinked.
    pub proxy_path: PathBuf,
    /// Original path it was matched to.
    pub original_path: PathBuf,
    /// How the match was determined.
    pub match_method: RelinkMethod,
    /// Confidence score 0.0–1.0.
    pub confidence: f64,
}

/// Method used to match proxy to original.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelinkMethod {
    /// Matched by filename hash.
    FilenameHash,
    /// Matched by metadata (resolution, duration, etc.).
    Metadata,
    /// Matched by exact filename pattern.
    FilenamePattern,
}

/// Simple FNV-1a-style hash for deterministic filename matching.
fn fnv_hash_str(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in s.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Manager for proxy-to-original relinking.
#[derive(Debug)]
pub struct ProxyRelinker {
    /// Known original files keyed by filename hash.
    originals_by_hash: HashMap<u64, PathBuf>,
    /// Known original files keyed by stem (filename without extension).
    originals_by_stem: HashMap<String, PathBuf>,
}

impl ProxyRelinker {
    /// Create a new relinker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            originals_by_hash: HashMap::new(),
            originals_by_stem: HashMap::new(),
        }
    }

    /// Register an original file for matching.
    pub fn register_original(&mut self, path: PathBuf) {
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let hash = fnv_hash_str(filename);
        self.originals_by_hash.insert(hash, path.clone());

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if !stem.is_empty() {
            self.originals_by_stem.insert(stem, path);
        }
    }

    /// Try to relink a proxy path to its original.
    ///
    /// Tries filename-hash match first, then stem-pattern match.
    #[must_use]
    pub fn relink(&self, proxy_path: &PathBuf) -> Option<RelinkResult> {
        let proxy_filename = proxy_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Strip "proxy_" prefix if present to recover original filename.
        let original_filename = proxy_filename
            .strip_prefix("proxy_")
            .unwrap_or(proxy_filename);

        // Try filename hash match.
        let hash = fnv_hash_str(original_filename);
        if let Some(orig) = self.originals_by_hash.get(&hash) {
            return Some(RelinkResult {
                proxy_path: proxy_path.clone(),
                original_path: orig.clone(),
                match_method: RelinkMethod::FilenameHash,
                confidence: 1.0,
            });
        }

        // Try stem pattern match: extract stem from proxy (minus proxy_ prefix).
        let proxy_stem = std::path::Path::new(original_filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !proxy_stem.is_empty() {
            if let Some(orig) = self.originals_by_stem.get(proxy_stem) {
                return Some(RelinkResult {
                    proxy_path: proxy_path.clone(),
                    original_path: orig.clone(),
                    match_method: RelinkMethod::FilenamePattern,
                    confidence: 0.8,
                });
            }
        }

        None
    }

    /// Number of registered originals.
    #[must_use]
    pub fn original_count(&self) -> usize {
        self.originals_by_hash.len()
    }
}

impl Default for ProxyRelinker {
    fn default() -> Self {
        Self::new()
    }
}

/// Background proxy generation queue with progress tracking.
#[derive(Debug)]
pub struct ProxyGenerationQueue {
    /// Pending jobs (source key, config).
    pending: Vec<(String, ProxyWorkflowConfig)>,
    /// In-progress jobs.
    in_progress: HashMap<String, ProxyJobProgress>,
    /// Completed jobs (source key).
    completed: Vec<String>,
    /// Failed jobs (source key, error message).
    failed: Vec<(String, String)>,
    /// Maximum concurrent jobs.
    max_concurrent: usize,
}

impl ProxyGenerationQueue {
    /// Create a new generation queue.
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: Vec::new(),
            in_progress: HashMap::new(),
            completed: Vec::new(),
            failed: Vec::new(),
            max_concurrent: max_concurrent.max(1),
        }
    }

    /// Enqueue a proxy generation job.
    pub fn enqueue(&mut self, source_key: String, config: ProxyWorkflowConfig) {
        self.pending.push((source_key, config));
    }

    /// Start the next pending job if capacity allows.
    ///
    /// Returns the source key of the started job, or None if queue is
    /// empty or at capacity.
    pub fn start_next(&mut self) -> Option<String> {
        if self.in_progress.len() >= self.max_concurrent {
            return None;
        }
        let (key, _config) = self.pending.pop()?;
        let progress = ProxyJobProgress::new(key.clone());
        self.in_progress.insert(key.clone(), progress);
        Some(key)
    }

    /// Update progress for an in-progress job.
    pub fn update_progress(
        &mut self,
        source_key: &str,
        fraction: f64,
        stage: &str,
        eta: f64,
    ) -> bool {
        if let Some(prog) = self.in_progress.get_mut(source_key) {
            prog.fraction = fraction.clamp(0.0, 1.0);
            prog.stage = stage.to_string();
            prog.eta_seconds = eta;
            true
        } else {
            false
        }
    }

    /// Mark a job as completed.
    pub fn mark_completed(&mut self, source_key: &str) -> bool {
        if self.in_progress.remove(source_key).is_some() {
            self.completed.push(source_key.to_string());
            true
        } else {
            false
        }
    }

    /// Mark a job as failed.
    pub fn mark_job_failed(&mut self, source_key: &str, error: String) -> bool {
        if self.in_progress.remove(source_key).is_some() {
            self.failed.push((source_key.to_string(), error));
            true
        } else {
            false
        }
    }

    /// Get progress for a specific job.
    #[must_use]
    pub fn get_progress(&self, source_key: &str) -> Option<&ProxyJobProgress> {
        self.in_progress.get(source_key)
    }

    /// Total overall progress (0.0–1.0) across all jobs.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn overall_progress(&self) -> f64 {
        let total =
            self.pending.len() + self.in_progress.len() + self.completed.len() + self.failed.len();
        if total == 0 {
            return 1.0;
        }
        let done = self.completed.len() as f64;
        let in_prog: f64 = self.in_progress.values().map(|p| p.fraction).sum();
        (done + in_prog) / total as f64
    }

    /// Number of pending jobs.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of in-progress jobs.
    #[must_use]
    pub fn in_progress_count(&self) -> usize {
        self.in_progress.len()
    }

    /// Number of completed jobs.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Number of failed jobs.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.failed.len()
    }

    /// Get failed jobs with their error messages.
    #[must_use]
    pub fn failed_jobs(&self) -> &[(String, String)] {
        &self.failed
    }

    /// Whether the queue has no more work (pending or in-progress).
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.in_progress.is_empty()
    }
}

/// Workflow manager tying together proxy generation, relinking, and chain selection.
#[derive(Debug)]
pub struct ProxyWorkflowManager {
    /// The underlying proxy manager.
    pub proxy_manager: ProxyManager,
    /// Multi-resolution chains per source.
    chains: HashMap<String, ProxyChain>,
    /// Generation queue.
    pub queue: ProxyGenerationQueue,
    /// Relinker.
    pub relinker: ProxyRelinker,
    /// Workflow config.
    pub config: ProxyWorkflowConfig,
}

impl ProxyWorkflowManager {
    /// Create a new workflow manager.
    #[must_use]
    pub fn new(proxy_dir: PathBuf, config: ProxyWorkflowConfig) -> Self {
        let max_concurrent = config.max_concurrent;
        Self {
            proxy_manager: ProxyManager::new(proxy_dir),
            chains: HashMap::new(),
            queue: ProxyGenerationQueue::new(max_concurrent),
            relinker: ProxyRelinker::new(),
            config,
        }
    }

    /// Register a source and create a multi-resolution proxy chain.
    ///
    /// Creates quarter, half, and full-scale chain entries.
    pub fn register_with_chain(
        &mut self,
        original_path: PathBuf,
        original_width: u32,
        original_height: u32,
    ) -> EditResult<()> {
        let key = original_path
            .to_str()
            .ok_or_else(|| EditError::InvalidEdit("Invalid path encoding".to_string()))?
            .to_string();

        // Register in underlying manager.
        self.proxy_manager.register_source(
            original_path.clone(),
            original_width,
            original_height,
        )?;

        // Register in relinker.
        self.relinker.register_original(original_path.clone());

        // Build chain with standard resolutions.
        let mut chain = ProxyChain::new(key.clone());
        let filename = original_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let quarter_path = self
            .proxy_manager
            .proxy_dir
            .join(format!("proxy_quarter_{filename}"));
        chain.add_level(ProxyResolution::Quarter, quarter_path, 0.25);

        let half_path = self
            .proxy_manager
            .proxy_dir
            .join(format!("proxy_half_{filename}"));
        chain.add_level(ProxyResolution::Half, half_path, 0.5);

        self.chains.insert(key, chain);
        Ok(())
    }

    /// Get the chain for a source.
    #[must_use]
    pub fn get_chain(&self, source_key: &str) -> Option<&ProxyChain> {
        self.chains.get(source_key)
    }

    /// Select the best proxy for a source at a given zoom level.
    #[must_use]
    pub fn select_for_zoom(&self, source_key: &str, zoom: f64) -> Option<&ProxyChainEntry> {
        self.chains.get(source_key)?.select_for_zoom(zoom)
    }

    /// Enqueue proxy generation for all pending sources.
    pub fn enqueue_all_pending(&mut self) {
        let pending: Vec<String> = self
            .proxy_manager
            .pending_generation()
            .iter()
            .map(|m| m.original_path.to_str().unwrap_or("unknown").to_string())
            .collect();
        for key in pending {
            self.queue.enqueue(key, self.config.clone());
        }
    }

    /// Mark a chain level as ready.
    pub fn mark_chain_ready(&mut self, source_key: &str, scale: f64) -> bool {
        self.chains
            .get_mut(source_key)
            .map_or(false, |c| c.mark_ready_by_scale(scale))
    }

    /// Number of registered chains.
    #[must_use]
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_resolution_dimensions() {
        assert_eq!(ProxyResolution::Quarter.dimensions(3840, 2160), (960, 540));
        assert_eq!(ProxyResolution::Half.dimensions(1920, 1080), (960, 540));
        assert_eq!(ProxyResolution::Hd720.dimensions(3840, 2160), (1280, 720));
        assert_eq!(ProxyResolution::Sd480.dimensions(1920, 1080), (854, 480));
        assert_eq!(
            ProxyResolution::Custom(640, 360).dimensions(1920, 1080),
            (640, 360)
        );
    }

    #[test]
    fn test_proxy_resolution_zero_dimensions() {
        // Should clamp to minimum 1
        assert_eq!(ProxyResolution::Quarter.dimensions(2, 2), (1, 1));
    }

    #[test]
    fn test_proxy_resolution_label() {
        assert_eq!(ProxyResolution::Quarter.label(), "1/4 Resolution");
        assert_eq!(ProxyResolution::Half.label(), "1/2 Resolution");
        assert_eq!(ProxyResolution::Hd720.label(), "720p");
    }

    #[test]
    fn test_proxy_status_is_usable() {
        assert!(ProxyStatus::Ready.is_usable());
        assert!(!ProxyStatus::NotGenerated.is_usable());
        assert!(!ProxyStatus::Generating.is_usable());
        assert!(!ProxyStatus::Failed.is_usable());
        assert!(!ProxyStatus::Outdated.is_usable());
    }

    #[test]
    fn test_proxy_mapping_scale_factor() {
        let mapping = ProxyMapping::new(
            PathBuf::from("/src/video.mp4"),
            PathBuf::from("/proxy/video.mp4"),
            ProxyResolution::Half,
            1920,
            1080,
        );
        assert!((mapping.scale_factor() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_proxy_mapping_zero_original_width() {
        let mapping = ProxyMapping::new(
            PathBuf::from("/src/video.mp4"),
            PathBuf::from("/proxy/video.mp4"),
            ProxyResolution::Half,
            0,
            0,
        );
        assert!((mapping.scale_factor() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_proxy_mode_logic() {
        assert!(!ProxyMode::Original.uses_proxy_for_playback());
        assert!(ProxyMode::Original.uses_original_for_export());
        assert!(ProxyMode::ProxyPlayback.uses_proxy_for_playback());
        assert!(ProxyMode::ProxyPlayback.uses_original_for_export());
        assert!(ProxyMode::ProxyOnly.uses_proxy_for_playback());
        assert!(!ProxyMode::ProxyOnly.uses_original_for_export());
    }

    #[test]
    fn test_proxy_manager_register_and_resolve() {
        let dir = std::env::temp_dir().join("oximedia_proxy_test");
        let mut mgr = ProxyManager::new(dir);

        let path = "/media/footage.mp4";
        mgr.register_source(PathBuf::from(path), 1920, 1080)
            .expect("registration should succeed");

        mgr.associate_clip(1, path);

        // Not ready yet, should return original
        let playback = mgr.resolve_path_for_playback(1);
        assert!(playback.is_some());
        assert_eq!(
            playback.expect("should resolve").to_str(),
            Some("/media/footage.mp4")
        );

        // Mark ready
        assert!(mgr.mark_ready(path));

        // Now should return proxy
        let playback = mgr.resolve_path_for_playback(1);
        assert!(playback.is_some());
        let p = playback.expect("should resolve");
        assert!(p
            .to_str()
            .map_or(false, |s| s.contains("proxy_footage.mp4")));

        // Export should still return original
        let export = mgr.resolve_path_for_export(1);
        assert!(export.is_some());
        assert_eq!(
            export.expect("should resolve").to_str(),
            Some("/media/footage.mp4")
        );
    }

    #[test]
    fn test_proxy_manager_pending_generation() {
        let dir = std::env::temp_dir().join("oximedia_proxy_test2");
        let mut mgr = ProxyManager::new(dir);
        mgr.register_source(PathBuf::from("/a.mp4"), 1920, 1080)
            .expect("ok");
        mgr.register_source(PathBuf::from("/b.mp4"), 1920, 1080)
            .expect("ok");
        assert_eq!(mgr.pending_generation().len(), 2);

        mgr.mark_ready("/a.mp4");
        assert_eq!(mgr.pending_generation().len(), 1);
        assert_eq!(mgr.ready_count(), 1);
    }

    #[test]
    fn test_proxy_manager_mark_outdated() {
        let dir = std::env::temp_dir().join("oximedia_proxy_test3");
        let mut mgr = ProxyManager::new(dir);
        mgr.register_source(PathBuf::from("/a.mp4"), 1920, 1080)
            .expect("ok");
        mgr.mark_ready("/a.mp4");
        assert_eq!(mgr.ready_count(), 1);
        mgr.mark_outdated("/a.mp4");
        assert_eq!(mgr.ready_count(), 0);
        assert_eq!(mgr.pending_generation().len(), 1);
    }

    #[test]
    fn test_proxy_manager_unknown_path_returns_false() {
        let dir = std::env::temp_dir().join("oximedia_proxy_test4");
        let mgr = ProxyManager::new(dir);
        assert!(mgr.resolve_path_for_playback(999).is_none());
        assert!(mgr.resolve_path_for_export(999).is_none());
    }

    #[test]
    fn test_proxy_manager_source_count() {
        let dir = std::env::temp_dir().join("oximedia_proxy_test5");
        let mut mgr = ProxyManager::new(dir);
        assert_eq!(mgr.source_count(), 0);
        mgr.register_source(PathBuf::from("/x.mp4"), 1920, 1080)
            .expect("ok");
        assert_eq!(mgr.source_count(), 1);
    }

    // ── Proxy codec tests ──────────────────────────────────────────────

    #[test]
    fn test_proxy_codec_labels() {
        assert_eq!(ProxyCodec::Vp9.label(), "VP9");
        assert_eq!(ProxyCodec::Av1.label(), "AV1");
        assert_eq!(ProxyCodec::Vp8.label(), "VP8");
    }

    #[test]
    fn test_proxy_workflow_config_defaults() {
        let cfg = ProxyWorkflowConfig::default();
        assert_eq!(cfg.codec, ProxyCodec::Vp9);
        assert_eq!(cfg.quality, 35);
        assert!(cfg.include_audio);
        assert_eq!(cfg.max_concurrent, 4);
    }

    // ── Proxy job progress tests ───────────────────────────────────────

    #[test]
    fn test_proxy_job_progress_new() {
        let p = ProxyJobProgress::new("test.mp4".to_string());
        assert!(!p.is_complete());
        assert_eq!(p.stage, "queued");
    }

    #[test]
    fn test_proxy_job_progress_complete() {
        let mut p = ProxyJobProgress::new("test.mp4".to_string());
        p.fraction = 1.0;
        assert!(p.is_complete());
    }

    // ── Proxy chain tests ──────────────────────────────────────────────

    #[test]
    fn test_proxy_chain_add_and_select() {
        let mut chain = ProxyChain::new("/src/video.mp4".to_string());
        chain.add_level(ProxyResolution::Quarter, PathBuf::from("/p/q.mp4"), 0.25);
        chain.add_level(ProxyResolution::Half, PathBuf::from("/p/h.mp4"), 0.5);
        assert_eq!(chain.level_count(), 2);
        assert_eq!(chain.ready_level_count(), 0);

        // Mark half ready
        assert!(chain.mark_ready_by_scale(0.5));
        assert_eq!(chain.ready_level_count(), 1);

        // Select for zoom 0.3 → should get half (smallest >= 0.3 that is ready)
        let selected = chain.select_for_zoom(0.3);
        assert!(selected.is_some());
        assert!((selected.map(|s| s.scale).unwrap_or(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_proxy_chain_fallback_to_highest_ready() {
        let mut chain = ProxyChain::new("/src/video.mp4".to_string());
        chain.add_level(ProxyResolution::Quarter, PathBuf::from("/p/q.mp4"), 0.25);
        chain.add_level(ProxyResolution::Half, PathBuf::from("/p/h.mp4"), 0.5);
        chain.mark_ready_by_scale(0.25);

        // Request zoom 0.8 → nothing >= 0.8 is ready, fallback to 0.25
        let selected = chain.select_for_zoom(0.8);
        assert!(selected.is_some());
        assert!((selected.map(|s| s.scale).unwrap_or(0.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_proxy_chain_no_ready() {
        let chain = ProxyChain::new("/src/video.mp4".to_string());
        assert!(chain.select_for_zoom(0.5).is_none());
    }

    #[test]
    fn test_proxy_chain_sorted_by_scale() {
        let mut chain = ProxyChain::new("src".to_string());
        chain.add_level(ProxyResolution::Half, PathBuf::from("/h"), 0.5);
        chain.add_level(ProxyResolution::Quarter, PathBuf::from("/q"), 0.25);
        let entries = chain.entries();
        assert!(entries[0].scale < entries[1].scale);
    }

    // ── Relinker tests ─────────────────────────────────────────────────

    #[test]
    fn test_relinker_hash_match() {
        let mut relinker = ProxyRelinker::new();
        relinker.register_original(PathBuf::from("/media/footage.mp4"));
        assert_eq!(relinker.original_count(), 1);

        let result = relinker.relink(&PathBuf::from("/proxies/proxy_footage.mp4"));
        assert!(result.is_some());
        let r = result.expect("should match");
        assert_eq!(r.match_method, RelinkMethod::FilenameHash);
        assert!((r.confidence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_relinker_stem_match() {
        let mut relinker = ProxyRelinker::new();
        relinker.register_original(PathBuf::from("/media/clip01.mov"));

        // Proxy has different extension but same stem after stripping prefix
        let result = relinker.relink(&PathBuf::from("/proxies/proxy_clip01.webm"));
        assert!(result.is_some());
        let r = result.expect("should match");
        assert_eq!(r.match_method, RelinkMethod::FilenamePattern);
    }

    #[test]
    fn test_relinker_no_match() {
        let relinker = ProxyRelinker::new();
        let result = relinker.relink(&PathBuf::from("/proxies/proxy_unknown.mp4"));
        assert!(result.is_none());
    }

    // ── Generation queue tests ─────────────────────────────────────────

    #[test]
    fn test_generation_queue_basic_flow() {
        let mut queue = ProxyGenerationQueue::new(2);
        assert!(queue.is_idle());

        queue.enqueue("a.mp4".to_string(), ProxyWorkflowConfig::default());
        queue.enqueue("b.mp4".to_string(), ProxyWorkflowConfig::default());
        queue.enqueue("c.mp4".to_string(), ProxyWorkflowConfig::default());
        assert_eq!(queue.pending_count(), 3);

        // Start two (max concurrent)
        let j1 = queue.start_next();
        assert!(j1.is_some());
        let j2 = queue.start_next();
        assert!(j2.is_some());
        let j3 = queue.start_next();
        assert!(j3.is_none()); // at capacity
        assert_eq!(queue.in_progress_count(), 2);

        // Complete one
        assert!(queue.mark_completed(j1.as_deref().unwrap_or("")));
        assert_eq!(queue.completed_count(), 1);

        // Now we can start another
        let j3 = queue.start_next();
        assert!(j3.is_some());
    }

    #[test]
    fn test_generation_queue_progress() {
        let mut queue = ProxyGenerationQueue::new(4);
        queue.enqueue("x.mp4".to_string(), ProxyWorkflowConfig::default());
        let key = queue.start_next().expect("should start");
        assert!(queue.update_progress(&key, 0.5, "encoding", 10.0));
        let prog = queue.get_progress(&key);
        assert!(prog.is_some());
        assert!((prog.map(|p| p.fraction).unwrap_or(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_generation_queue_overall_progress() {
        let mut queue = ProxyGenerationQueue::new(4);
        queue.enqueue("a.mp4".to_string(), ProxyWorkflowConfig::default());
        queue.enqueue("b.mp4".to_string(), ProxyWorkflowConfig::default());
        let k1 = queue.start_next().expect("ok");
        let _k2 = queue.start_next().expect("ok");
        queue.mark_completed(&k1);
        // 1 completed, 1 in progress at 0.0 = 1.0/2.0 = 0.5
        assert!((queue.overall_progress() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_generation_queue_failure() {
        let mut queue = ProxyGenerationQueue::new(4);
        queue.enqueue("bad.mp4".to_string(), ProxyWorkflowConfig::default());
        let key = queue.start_next().expect("ok");
        assert!(queue.mark_job_failed(&key, "codec error".to_string()));
        assert_eq!(queue.failed_count(), 1);
        assert!(queue.is_idle());
    }

    #[test]
    fn test_generation_queue_idle_after_drain() {
        let mut queue = ProxyGenerationQueue::new(4);
        assert!(queue.is_idle());
        queue.enqueue("z.mp4".to_string(), ProxyWorkflowConfig::default());
        assert!(!queue.is_idle());
        let key = queue.start_next().expect("ok");
        assert!(!queue.is_idle());
        queue.mark_completed(&key);
        assert!(queue.is_idle());
    }

    // ── Workflow manager tests ─────────────────────────────────────────

    #[test]
    fn test_workflow_manager_register_with_chain() {
        let dir = std::env::temp_dir().join("oximedia_wf_test1");
        let mut wf = ProxyWorkflowManager::new(dir, ProxyWorkflowConfig::default());
        wf.register_with_chain(PathBuf::from("/media/clip.mp4"), 3840, 2160)
            .expect("should register");
        assert_eq!(wf.chain_count(), 1);
        let chain = wf.get_chain("/media/clip.mp4");
        assert!(chain.is_some());
        assert_eq!(chain.map(|c| c.level_count()).unwrap_or(0), 2);
    }

    #[test]
    fn test_workflow_manager_select_for_zoom() {
        let dir = std::env::temp_dir().join("oximedia_wf_test2");
        let mut wf = ProxyWorkflowManager::new(dir, ProxyWorkflowConfig::default());
        wf.register_with_chain(PathBuf::from("/media/clip.mp4"), 1920, 1080)
            .expect("ok");
        // Nothing ready yet
        assert!(wf.select_for_zoom("/media/clip.mp4", 0.3).is_none());

        wf.mark_chain_ready("/media/clip.mp4", 0.25);
        let entry = wf.select_for_zoom("/media/clip.mp4", 0.2);
        assert!(entry.is_some());
    }

    #[test]
    fn test_workflow_manager_enqueue_pending() {
        let dir = std::env::temp_dir().join("oximedia_wf_test3");
        let mut wf = ProxyWorkflowManager::new(dir, ProxyWorkflowConfig::default());
        wf.register_with_chain(PathBuf::from("/media/a.mp4"), 1920, 1080)
            .expect("ok");
        wf.register_with_chain(PathBuf::from("/media/b.mp4"), 1920, 1080)
            .expect("ok");
        wf.enqueue_all_pending();
        assert_eq!(wf.queue.pending_count(), 2);
    }

    #[test]
    fn test_fnv_hash_deterministic() {
        let h1 = fnv_hash_str("test.mp4");
        let h2 = fnv_hash_str("test.mp4");
        assert_eq!(h1, h2);
        let h3 = fnv_hash_str("other.mp4");
        assert_ne!(h1, h3);
    }
}
