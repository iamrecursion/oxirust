//! HLS streaming client with adaptive bitrate support.
//!
//! This module provides the [`HlsClient`] type which manages HLS playback,
//! including segment fetching, buffering, and adaptive bitrate switching.

#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::if_not_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::single_match_else)]
#![allow(clippy::redundant_slicing)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::format_collect)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unused_async)]
#![allow(clippy::identity_op)]

use crate::error::{NetError, NetResult};
use crate::hls::{
    AbrController, ByteRange, MasterPlaylist, MediaPlaylist, QualityLevel, Segment, SegmentCache,
    SegmentFetcher, ThroughputBasedAbr,
};
use bytes::Bytes;
use reqwest::Client;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};

/// Maps a desired media sequence number to an index into the current media
/// playlist's segment list.
///
/// Returns `None` when `seq` is outside the playlist's sliding window. A live
/// server can raise `media_sequence` above `seq` after evicting segments; using
/// `checked_sub` here turns that "we fell behind" case into a clean `None`
/// instead of a `u64` underflow panic on `seq - media_sequence`.
fn segment_index(seq: u64, media_sequence: u64, segment_count: usize) -> Option<usize> {
    match seq.checked_sub(media_sequence) {
        Some(idx) if (idx as usize) < segment_count => Some(idx as usize),
        _ => None,
    }
}

/// Configuration for the HLS client.
#[derive(Debug, Clone)]
pub struct HlsClientConfig {
    /// Initial buffer size in seconds.
    pub initial_buffer: Duration,
    /// Target buffer size in seconds.
    pub target_buffer: Duration,
    /// Maximum buffer size in seconds.
    pub max_buffer: Duration,
    /// Minimum buffer before rebuffering.
    pub min_buffer: Duration,
    /// Maximum number of segments to prefetch.
    pub max_prefetch: usize,
    /// Interval for refreshing live playlists.
    pub playlist_refresh_interval: Duration,
    /// Enable segment caching.
    pub enable_cache: bool,
    /// Cache size in bytes.
    pub cache_size: usize,
    /// Maximum number of cached segments.
    pub max_cached_segments: usize,
    /// Maximum number of concurrent segment downloads.
    pub max_concurrent_downloads: usize,
    /// Enable adaptive bitrate switching.
    pub enable_abr: bool,
    /// Request timeout.
    pub request_timeout: Duration,
    /// Maximum retries for failed requests.
    pub max_retries: u32,
}

impl Default for HlsClientConfig {
    fn default() -> Self {
        Self {
            initial_buffer: Duration::from_secs(5),
            target_buffer: Duration::from_secs(15),
            max_buffer: Duration::from_secs(30),
            min_buffer: Duration::from_secs(3),
            max_prefetch: 10,
            playlist_refresh_interval: Duration::from_secs(5),
            enable_cache: true,
            cache_size: 100 * 1024 * 1024, // 100MB
            max_cached_segments: 50,
            max_concurrent_downloads: 3,
            enable_abr: true,
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// State of the HLS client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientState {
    /// Client is idle, not yet started.
    Idle,
    /// Loading initial playlist and segments.
    Loading,
    /// Playing back segments.
    Playing,
    /// Paused playback.
    Paused,
    /// Buffering more segments.
    Buffering,
    /// Switching to a different variant.
    Switching,
    /// Stream has ended.
    Ended,
    /// Error occurred.
    Error,
}

/// A buffered segment ready for playback.
#[derive(Debug, Clone)]
pub struct BufferedSegment {
    /// Segment metadata.
    pub segment: Segment,
    /// Segment data.
    pub data: Bytes,
    /// Sequence number.
    pub sequence: u64,
    /// Quality level index.
    pub quality_level: usize,
    /// When the segment was buffered.
    pub buffered_at: Instant,
}

/// Statistics about the HLS client.
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    /// Total bytes downloaded.
    pub bytes_downloaded: u64,
    /// Total segments downloaded.
    pub segments_downloaded: u64,
    /// Total segments dropped (due to quality switch).
    pub segments_dropped: u64,
    /// Number of quality switches.
    pub quality_switches: u64,
    /// Number of rebuffering events.
    pub rebuffer_events: u64,
    /// Total rebuffering time.
    pub rebuffer_time: Duration,
    /// Estimated throughput in bits per second.
    pub estimated_throughput: f64,
    /// Current buffer level.
    pub buffer_level: Duration,
}

/// HLS streaming client with adaptive bitrate support.
pub struct HlsClient {
    /// Client configuration.
    config: HlsClientConfig,
    /// HTTP client.
    http_client: Client,
    /// Current state.
    state: Arc<RwLock<ClientState>>,
    /// Master playlist (if available).
    master_playlist: Arc<RwLock<Option<MasterPlaylist>>>,
    /// Current media playlist.
    media_playlist: Arc<RwLock<Option<MediaPlaylist>>>,
    /// Available quality levels.
    quality_levels: Arc<RwLock<Vec<QualityLevel>>>,
    /// Current quality level index.
    current_quality: Arc<RwLock<usize>>,
    /// Segment buffer queue.
    buffer_queue: Arc<Mutex<VecDeque<BufferedSegment>>>,
    /// Segment fetcher.
    fetcher: Arc<Mutex<SegmentFetcher>>,
    /// Segment cache.
    cache: Option<Arc<SegmentCache>>,
    /// ABR controller.
    abr_controller: Arc<Mutex<Box<dyn AbrController>>>,
    /// Client statistics.
    stats: Arc<RwLock<ClientStats>>,
    /// Next segment sequence number to fetch.
    next_sequence: Arc<Mutex<u64>>,
    /// Last playlist refresh time.
    last_playlist_refresh: Arc<Mutex<Option<Instant>>>,
    /// Playback position (in seconds).
    playback_position: Arc<RwLock<f64>>,
    /// Base URL for resolving relative URLs.
    base_url: Arc<RwLock<Option<String>>>,
}

impl HlsClient {
    /// Creates a new HLS client with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(HlsClientConfig::default())
    }

    /// Creates a new HLS client with the given configuration.
    #[must_use]
    pub fn with_config(config: HlsClientConfig) -> Self {
        let http_client = Client::builder()
            .timeout(config.request_timeout)
            .build()
            .unwrap_or_else(|_| Client::new());

        let cache = if config.enable_cache {
            Some(Arc::new(SegmentCache::new(
                config.cache_size,
                config.max_cached_segments,
            )))
        } else {
            None
        };

        let abr_controller: Box<dyn AbrController> = Box::new(ThroughputBasedAbr::new());

        Self {
            config,
            http_client: http_client.clone(),
            state: Arc::new(RwLock::new(ClientState::Idle)),
            master_playlist: Arc::new(RwLock::new(None)),
            media_playlist: Arc::new(RwLock::new(None)),
            quality_levels: Arc::new(RwLock::new(Vec::new())),
            current_quality: Arc::new(RwLock::new(0)),
            buffer_queue: Arc::new(Mutex::new(VecDeque::new())),
            fetcher: Arc::new(Mutex::new(SegmentFetcher::with_client(http_client))),
            cache,
            abr_controller: Arc::new(Mutex::new(abr_controller)),
            stats: Arc::new(RwLock::new(ClientStats::default())),
            next_sequence: Arc::new(Mutex::new(0)),
            last_playlist_refresh: Arc::new(Mutex::new(None)),
            playback_position: Arc::new(RwLock::new(0.0)),
            base_url: Arc::new(RwLock::new(None)),
        }
    }

    /// Sets a custom ABR controller.
    pub async fn set_abr_controller(&self, controller: Box<dyn AbrController>) {
        let mut abr = self.abr_controller.lock().await;
        *abr = controller;
    }

    /// Returns the current client state.
    pub async fn state(&self) -> ClientState {
        *self.state.read().await
    }

    /// Returns the current statistics.
    pub async fn stats(&self) -> ClientStats {
        self.stats.read().await.clone()
    }

    /// Loads a master playlist from a URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the playlist cannot be fetched or parsed.
    pub async fn load_master_playlist(&self, url: &str) -> NetResult<()> {
        self.set_state(ClientState::Loading).await;

        // Fetch the playlist
        let response =
            self.http_client.get(url).send().await.map_err(|e| {
                NetError::connection(format!("Failed to fetch master playlist: {e}"))
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(NetError::http(
                status.as_u16(),
                format!("Failed to fetch master playlist: {url}"),
            ));
        }

        let text = response
            .text()
            .await
            .map_err(|e| NetError::connection(format!("Failed to read playlist: {e}")))?;

        // Parse the playlist
        let mut master = MasterPlaylist::parse(&text)?;

        // Set base URL
        master.base_uri = Some(url.to_string());
        let mut base_url = self.base_url.write().await;
        *base_url = Some(url.to_string());
        drop(base_url);

        // Extract quality levels from variants
        let quality_levels: Vec<QualityLevel> = master
            .variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| {
                let mut level = QualityLevel::new(idx, variant.stream_inf.bandwidth);
                if let Some((w, h)) = variant.stream_inf.resolution {
                    level = level.with_resolution(w, h);
                }
                if let Some(ref codecs) = variant.stream_inf.codecs {
                    level = level.with_codecs(codecs.clone());
                }
                level
            })
            .collect();

        // Store the master playlist and quality levels
        let mut playlist = self.master_playlist.write().await;
        *playlist = Some(master);
        drop(playlist);

        let mut levels = self.quality_levels.write().await;
        *levels = quality_levels;
        drop(levels);

        // Select initial quality level (lowest to start)
        let mut current = self.current_quality.write().await;
        *current = 0;
        drop(current);

        Ok(())
    }

    /// Loads a media playlist from a URL (for direct media playlist playback).
    ///
    /// # Errors
    ///
    /// Returns an error if the playlist cannot be fetched or parsed.
    pub async fn load_media_playlist(&self, url: &str) -> NetResult<()> {
        self.set_state(ClientState::Loading).await;

        let playlist = self.fetch_media_playlist(url).await?;

        // Set base URL
        let mut base_url = self.base_url.write().await;
        *base_url = Some(url.to_string());
        drop(base_url);

        // Store the media playlist
        let mut media = self.media_playlist.write().await;
        *media = Some(playlist);
        drop(media);

        // Initialize sequence number
        let media_ref = self.media_playlist.read().await;
        if let Some(ref playlist) = *media_ref {
            let mut seq = self.next_sequence.lock().await;
            *seq = playlist.media_sequence;
        }

        Ok(())
    }

    /// Starts playback.
    ///
    /// # Errors
    ///
    /// Returns an error if playback cannot be started.
    pub async fn start(&self) -> NetResult<()> {
        let current_state = self.state().await;
        if current_state != ClientState::Idle && current_state != ClientState::Loading {
            return Err(NetError::invalid_state(format!(
                "Cannot start from state: {current_state:?}"
            )));
        }

        // Load the initial media playlist if we have a master playlist
        let master = self.master_playlist.read().await;
        if let Some(ref master_pl) = *master {
            let current_quality = *self.current_quality.read().await;
            if let Some(variant) = master_pl.variants.get(current_quality) {
                let base = master_pl
                    .base_uri
                    .as_ref()
                    .ok_or_else(|| NetError::invalid_state("Master playlist has no base URI"))?;
                let url = self.resolve_variant_url(base, &variant.uri);
                drop(master);
                self.fetch_and_update_media_playlist(&url).await?;
            }
        }

        self.set_state(ClientState::Playing).await;

        // Start the download loop
        self.start_download_loop().await;

        Ok(())
    }

    /// Pauses playback.
    pub async fn pause(&self) {
        self.set_state(ClientState::Paused).await;
    }

    /// Resumes playback.
    pub async fn resume(&self) {
        self.set_state(ClientState::Playing).await;
    }

    /// Stops playback and clears buffers.
    pub async fn stop(&self) {
        self.set_state(ClientState::Idle).await;

        // Clear buffers
        let mut queue = self.buffer_queue.lock().await;
        queue.clear();

        // Clear cache
        if let Some(ref cache) = self.cache {
            cache.clear().await;
        }

        // Reset statistics
        let mut stats = self.stats.write().await;
        *stats = ClientStats::default();

        // Reset ABR controller
        let mut abr = self.abr_controller.lock().await;
        abr.reset();
    }

    /// Gets the next buffered segment for playback.
    pub async fn get_next_segment(&self) -> Option<BufferedSegment> {
        let mut queue = self.buffer_queue.lock().await;
        let segment = queue.pop_front();

        if let Some(ref seg) = segment {
            // Update playback position
            let mut pos = self.playback_position.write().await;
            *pos += seg.segment.duration.as_secs_f64();

            // Update buffer level
            let buffer_duration: Duration = queue.iter().map(|s| s.segment.duration).sum();
            let mut stats = self.stats.write().await;
            stats.buffer_level = buffer_duration;

            // Report buffer level to ABR controller
            let mut abr = self.abr_controller.lock().await;
            abr.report_buffer(buffer_duration);
        }

        // Check if we need to rebuffer
        let current_state = self.state().await;
        if current_state == ClientState::Playing {
            let buffer_level = self.get_buffer_level().await;
            if buffer_level < self.config.min_buffer {
                self.set_state(ClientState::Buffering).await;

                let mut stats = self.stats.write().await;
                stats.rebuffer_events += 1;
            }
        }

        segment
    }

    /// Returns the current buffer level in seconds.
    pub async fn get_buffer_level(&self) -> Duration {
        let queue = self.buffer_queue.lock().await;
        queue.iter().map(|s| s.segment.duration).sum()
    }

    /// Returns the number of buffered segments.
    pub async fn buffered_segment_count(&self) -> usize {
        let queue = self.buffer_queue.lock().await;
        queue.len()
    }

    /// Manually switches to a different quality level.
    ///
    /// # Errors
    ///
    /// Returns an error if the quality level is invalid.
    pub async fn switch_quality(&self, level: usize) -> NetResult<()> {
        let levels = self.quality_levels.read().await;
        if level >= levels.len() {
            return Err(NetError::invalid_state(format!(
                "Invalid quality level: {level}"
            )));
        }
        drop(levels);

        let mut current = self.current_quality.write().await;
        if *current == level {
            return Ok(()); // Already at this level
        }

        *current = level;
        drop(current);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.quality_switches += 1;

        self.set_state(ClientState::Switching).await;

        Ok(())
    }

    /// Returns the available quality levels.
    pub async fn available_quality_levels(&self) -> Vec<QualityLevel> {
        self.quality_levels.read().await.clone()
    }

    /// Returns the current quality level index.
    pub async fn current_quality_level(&self) -> usize {
        *self.current_quality.read().await
    }

    // Private helper methods

    async fn set_state(&self, state: ClientState) {
        let mut current_state = self.state.write().await;
        *current_state = state;
    }

    async fn fetch_media_playlist(&self, url: &str) -> NetResult<MediaPlaylist> {
        let response =
            self.http_client.get(url).send().await.map_err(|e| {
                NetError::connection(format!("Failed to fetch media playlist: {e}"))
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(NetError::http(
                status.as_u16(),
                format!("Failed to fetch media playlist: {url}"),
            ));
        }

        let text = response
            .text()
            .await
            .map_err(|e| NetError::connection(format!("Failed to read playlist: {e}")))?;

        let mut playlist = MediaPlaylist::parse(&text)?;
        playlist.base_uri = Some(url.to_string());

        Ok(playlist)
    }

    async fn fetch_and_update_media_playlist(&self, url: &str) -> NetResult<()> {
        let playlist = self.fetch_media_playlist(url).await?;

        let mut media = self.media_playlist.write().await;
        *media = Some(playlist);

        // Update last refresh time
        let mut last_refresh = self.last_playlist_refresh.lock().await;
        *last_refresh = Some(Instant::now());

        Ok(())
    }

    fn resolve_variant_url(&self, base: &str, uri: &str) -> String {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            return uri.to_string();
        }

        if uri.starts_with('/') {
            // Absolute path
            if let Some(pos) = base.find("://") {
                if let Some(slash_pos) = base[pos + 3..].find('/') {
                    return format!("{}{uri}", &base[..pos + 3 + slash_pos]);
                }
            }
            format!("{base}{uri}")
        } else {
            // Relative path
            if let Some(last_slash) = base.rfind('/') {
                format!("{}/{uri}", &base[..last_slash])
            } else {
                format!("{base}/{uri}")
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    async fn start_download_loop(&self) {
        let state = self.state.clone();
        let config = self.config.clone();
        let buffer_queue = self.buffer_queue.clone();
        let fetcher = self.fetcher.clone();
        let cache = self.cache.clone();
        let media_playlist = self.media_playlist.clone();
        let next_sequence = self.next_sequence.clone();
        let stats = self.stats.clone();
        let abr_controller = self.abr_controller.clone();
        let quality_levels = self.quality_levels.clone();
        let current_quality = self.current_quality.clone();
        let master_playlist = self.master_playlist.clone();
        let base_url_arc = self.base_url.clone();
        let last_refresh = self.last_playlist_refresh.clone();
        let http_client = self.http_client.clone();

        tokio::spawn(async move {
            loop {
                let current_state = *state.read().await;

                // Exit if stopped
                if current_state == ClientState::Idle || current_state == ClientState::Error {
                    break;
                }

                // Skip if paused
                if current_state == ClientState::Paused {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }

                // Check if we need to refresh the playlist (for live streams)
                let should_refresh = {
                    let last = last_refresh.lock().await;
                    match *last {
                        Some(t) => t.elapsed() >= config.playlist_refresh_interval,
                        None => false,
                    }
                };

                if should_refresh {
                    let media_guard = media_playlist.read().await;
                    if let Some(ref playlist) = *media_guard {
                        if playlist.is_live() {
                            // Refresh the playlist
                            let url = playlist.base_uri.clone();
                            drop(media_guard);

                            if let Some(url) = url {
                                let response = http_client.get(&url).send().await;
                                if let Ok(response) = response {
                                    if let Ok(text) = response.text().await {
                                        if let Ok(mut new_playlist) = MediaPlaylist::parse(&text) {
                                            new_playlist.base_uri = Some(url);
                                            let mut media = media_playlist.write().await;
                                            *media = Some(new_playlist);
                                        }
                                    }
                                }

                                let mut last = last_refresh.lock().await;
                                *last = Some(Instant::now());
                            }
                        }
                    }
                }

                // Check buffer level
                let buffer_level: Duration = {
                    let queue = buffer_queue.lock().await;
                    queue.iter().map(|s| s.segment.duration).sum()
                };

                // Check if we have enough buffer
                if buffer_level >= config.max_buffer {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }

                // Check if we're buffering and have reached target
                if current_state == ClientState::Buffering && buffer_level >= config.initial_buffer
                {
                    let mut s = state.write().await;
                    *s = ClientState::Playing;
                }

                // Perform ABR decision if enabled
                if config.enable_abr {
                    let levels = quality_levels.read().await;
                    let current = *current_quality.read().await;
                    let abr = abr_controller.lock().await;
                    let decision = abr.select_quality(&levels, current);
                    drop(abr);
                    drop(levels);

                    if let Some(target) = decision.target_level() {
                        if target != current {
                            // Switch quality level
                            let mut qual = current_quality.write().await;
                            *qual = target;
                            drop(qual);

                            // Update stats
                            let mut stats_guard = stats.write().await;
                            stats_guard.quality_switches += 1;

                            // Reload media playlist for new variant
                            let master_guard = master_playlist.read().await;
                            if let Some(ref master) = *master_guard {
                                if let Some(variant) = master.variants.get(target) {
                                    let base = master.base_uri.as_ref();
                                    if let Some(base) = base {
                                        let url = Self::resolve_url_static(base, &variant.uri);
                                        drop(master_guard);

                                        let response = http_client.get(&url).send().await;
                                        if let Ok(response) = response {
                                            if let Ok(text) = response.text().await {
                                                if let Ok(mut new_playlist) =
                                                    MediaPlaylist::parse(&text)
                                                {
                                                    new_playlist.base_uri = Some(url);
                                                    let mut media = media_playlist.write().await;
                                                    *media = Some(new_playlist);

                                                    // Reset sequence to start of new playlist
                                                    let seq = media
                                                        .as_ref()
                                                        .map(|p| p.media_sequence)
                                                        .unwrap_or(0);
                                                    let mut next_seq = next_sequence.lock().await;
                                                    *next_seq = seq;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Fetch next segment
                let segment_info = {
                    let media_guard = media_playlist.read().await;
                    if let Some(ref playlist) = *media_guard {
                        let seq = *next_sequence.lock().await;
                        // A live server can raise `media_sequence` past our
                        // `seq` (segments we still wanted were evicted from the
                        // sliding window); `segment_index` uses `checked_sub` so
                        // that becomes a clean wait rather than an underflow
                        // panic. Out-of-range / fell-behind both yield None.
                        segment_index(seq, playlist.media_sequence, playlist.segments.len()).map(
                            |idx| {
                                (
                                    playlist.segments[idx].clone(),
                                    seq,
                                    playlist.base_uri.clone(),
                                )
                            },
                        )
                    } else {
                        None
                    }
                };

                if let Some((segment, seq, base_uri)) = segment_info {
                    // Resolve segment URL
                    let segment_url = if let Some(ref base) = base_uri {
                        Self::resolve_url_static(base, &segment.uri)
                    } else if let Some(ref base) = *base_url_arc.read().await {
                        Self::resolve_url_static(base, &segment.uri)
                    } else {
                        segment.uri.clone()
                    };

                    // Check cache first
                    let cached_data = if let Some(ref cache) = cache {
                        cache.get(&segment_url).await
                    } else {
                        None
                    };

                    let segment_data = if let Some(data) = cached_data {
                        // Cache hit
                        data
                    } else {
                        // Cache miss - fetch segment
                        let byte_range = segment.byte_range.map(|(len, offset)| {
                            if let Some(off) = offset {
                                ByteRange::from_offset_length(off, len)
                            } else {
                                ByteRange::from_offset_length(0, len)
                            }
                        });

                        let mut fetcher_guard = fetcher.lock().await;
                        let fetch_result = fetcher_guard
                            .fetch_with_retry(&segment_url, byte_range)
                            .await;

                        match fetch_result {
                            Ok(result) => {
                                // Update ABR controller
                                let mut abr = abr_controller.lock().await;
                                abr.report_download(result.content_length, result.fetch_time);

                                // Update stats
                                let mut stats_guard = stats.write().await;
                                stats_guard.bytes_downloaded += result.content_length as u64;
                                stats_guard.segments_downloaded += 1;
                                stats_guard.estimated_throughput = abr.estimated_throughput();

                                // Cache the segment
                                if let Some(ref cache) = cache {
                                    cache.put(segment_url.clone(), result.data.clone()).await;
                                }

                                result.data
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, segment = seq, "failed to fetch segment; retrying");
                                tokio::time::sleep(Duration::from_secs(1)).await;
                                continue;
                            }
                        }
                    };

                    // Add to buffer queue
                    let buffered = BufferedSegment {
                        segment: segment.clone(),
                        data: segment_data,
                        sequence: seq,
                        quality_level: *current_quality.read().await,
                        buffered_at: Instant::now(),
                    };

                    let mut queue = buffer_queue.lock().await;
                    queue.push_back(buffered);

                    // Increment sequence
                    let mut next_seq = next_sequence.lock().await;
                    *next_seq += 1;
                } else {
                    // No more segments available
                    let media_guard = media_playlist.read().await;
                    if let Some(ref playlist) = *media_guard {
                        if playlist.ended {
                            // Stream has ended
                            let mut s = state.write().await;
                            *s = ClientState::Ended;
                            break;
                        }
                    }

                    // Wait before checking again
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }

                // Small delay to prevent tight loop
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
    }

    fn resolve_url_static(base: &str, uri: &str) -> String {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            return uri.to_string();
        }

        if uri.starts_with('/') {
            // Absolute path
            if let Some(pos) = base.find("://") {
                if let Some(slash_pos) = base[pos + 3..].find('/') {
                    return format!("{}{uri}", &base[..pos + 3 + slash_pos]);
                }
            }
            format!("{base}{uri}")
        } else {
            // Relative path
            if let Some(last_slash) = base.rfind('/') {
                format!("{}/{uri}", &base[..last_slash])
            } else {
                format!("{base}/{uri}")
            }
        }
    }
}

impl Default for HlsClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring an HLS client.
pub struct HlsClientBuilder {
    config: HlsClientConfig,
}

impl HlsClientBuilder {
    /// Creates a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HlsClientConfig::default(),
        }
    }

    /// Sets the initial buffer duration.
    #[must_use]
    pub const fn initial_buffer(mut self, duration: Duration) -> Self {
        self.config.initial_buffer = duration;
        self
    }

    /// Sets the target buffer duration.
    #[must_use]
    pub const fn target_buffer(mut self, duration: Duration) -> Self {
        self.config.target_buffer = duration;
        self
    }

    /// Sets the maximum buffer duration.
    #[must_use]
    pub const fn max_buffer(mut self, duration: Duration) -> Self {
        self.config.max_buffer = duration;
        self
    }

    /// Sets the minimum buffer duration before rebuffering.
    #[must_use]
    pub const fn min_buffer(mut self, duration: Duration) -> Self {
        self.config.min_buffer = duration;
        self
    }

    /// Sets the maximum number of segments to prefetch.
    #[must_use]
    pub const fn max_prefetch(mut self, count: usize) -> Self {
        self.config.max_prefetch = count;
        self
    }

    /// Sets the playlist refresh interval for live streams.
    #[must_use]
    pub const fn playlist_refresh_interval(mut self, interval: Duration) -> Self {
        self.config.playlist_refresh_interval = interval;
        self
    }

    /// Enables or disables segment caching.
    #[must_use]
    pub const fn enable_cache(mut self, enable: bool) -> Self {
        self.config.enable_cache = enable;
        self
    }

    /// Sets the cache size in bytes.
    #[must_use]
    pub const fn cache_size(mut self, size: usize) -> Self {
        self.config.cache_size = size;
        self
    }

    /// Sets the maximum number of concurrent downloads.
    #[must_use]
    pub const fn max_concurrent_downloads(mut self, count: usize) -> Self {
        self.config.max_concurrent_downloads = count;
        self
    }

    /// Enables or disables adaptive bitrate switching.
    #[must_use]
    pub const fn enable_abr(mut self, enable: bool) -> Self {
        self.config.enable_abr = enable;
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = timeout;
        self
    }

    /// Sets the maximum number of retries for failed requests.
    #[must_use]
    pub const fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Builds the HLS client.
    #[must_use]
    pub fn build(self) -> HlsClient {
        HlsClient::with_config(self.config)
    }
}

impl Default for HlsClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::segment_index;

    #[test]
    fn segment_index_normal_and_boundary() {
        assert_eq!(segment_index(10, 8, 5), Some(2));
        assert_eq!(segment_index(8, 8, 5), Some(0));
        assert_eq!(segment_index(12, 8, 5), Some(4));
    }

    #[test]
    fn segment_index_fallen_behind_is_none_not_panic() {
        // seq < media_sequence: the live window slipped past us. Must be a
        // clean None (the old `seq - media_sequence` panicked on underflow).
        assert_eq!(segment_index(5, 8, 5), None);
        assert_eq!(segment_index(0, u64::MAX, 5), None);
    }

    #[test]
    fn segment_index_out_of_range_is_none() {
        assert_eq!(segment_index(100, 8, 5), None);
    }
}
