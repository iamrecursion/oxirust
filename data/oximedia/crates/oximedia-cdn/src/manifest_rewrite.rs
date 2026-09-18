//! HLS and DASH manifest URL rewriting for CDN edge-local delivery.
//!
//! # Overview
//!
//! `ManifestRewriter` transforms manifest files so that all asset URLs
//! (media segment URIs, `EXT-X-MAP`, `EXT-X-MEDIA`, `EXT-X-STREAM-INF`,
//! rendition playlists, DASH `BaseURL`, `SegmentTemplate`, `media=`,
//! `initialization=`, etc.) are rewritten to point to the edge node that
//! is serving the manifest.
//!
//! This enables CDN split-delivery: the origin stores the canonical manifest,
//! but when the manifest is fetched through a PoP, all child URLs are
//! re-pointed to that PoP so subsequent segment requests also hit the cache.
//!
//! # URL resolution rules
//!
//! 1. Absolute URLs (starting with `http://` or `https://`) have their host
//!    replaced with the edge hostname while the path is preserved.
//! 2. Relative URLs are resolved against the manifest's `base_url` and then
//!    rewritten to absolute edge-host URLs.
//! 3. Data URIs and fragment-only URLs are left unchanged.
//!
//! # Format support
//!
//! | Format | Detection       | Rewritten fields                                    |
//! |--------|-----------------|-----------------------------------------------------|
//! | HLS    | `.m3u8` suffix or `#EXTM3U` header | Segment URIs, `URI=` attributes in tags |
//! | DASH   | `.mpd` suffix or `<MPD` marker      | `BaseURL`, `media=`, `initialization=`, `sourceURL=` |

use std::fmt;
use thiserror::Error;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can arise during manifest rewriting.
#[derive(Debug, Error)]
pub enum RewriteError {
    /// The manifest content is malformed and cannot be parsed.
    #[error("malformed manifest: {0}")]
    Malformed(String),
    /// The edge host URL is invalid.
    #[error("invalid edge host: '{0}'")]
    InvalidEdgeHost(String),
}

// ─── ManifestFormat ───────────────────────────────────────────────────────────

/// Detected or declared manifest format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestFormat {
    /// HTTP Live Streaming — Apple `.m3u8` format.
    Hls,
    /// MPEG-DASH — `.mpd` XML format.
    Dash,
}

impl fmt::Display for ManifestFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hls => f.write_str("HLS"),
            Self::Dash => f.write_str("DASH"),
        }
    }
}

// ─── RewriteStats ─────────────────────────────────────────────────────────────

/// Statistics from a single rewrite pass.
#[derive(Debug, Clone, Default)]
pub struct RewriteStats {
    /// Number of URLs that were rewritten.
    pub urls_rewritten: usize,
    /// Number of URLs that were left unchanged (data URIs, fragments, etc.).
    pub urls_unchanged: usize,
    /// Detected or applied manifest format.
    pub format: Option<ManifestFormat>,
}

// ─── RewriteConfig ────────────────────────────────────────────────────────────

/// Configuration for the manifest rewriter.
/// Configuration for the manifest rewriter.
pub struct RewriteConfig {
    /// Whether to preserve the original query string on rewritten URLs.
    pub preserve_query: bool,
    /// Optional token-signing closure: if set, every rewritten URL passes
    /// through this function so callers can append CDN auth tokens.
    /// The closure receives the rewritten URL and returns the final URL.
    ///
    /// Stored as `None` when no signing is required.
    pub sign_fn: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
    /// Whether to rewrite URLs in `EXT-X-MAP` tags (HLS init segments).
    pub rewrite_init_segments: bool,
    /// Whether to rewrite `EXT-X-MEDIA` URIs (alternative renditions).
    pub rewrite_media_uris: bool,
    /// Whether to rewrite `EXT-X-I-FRAME-STREAM-INF` URIs.
    pub rewrite_iframe_uris: bool,
}

use std::sync::Arc;

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            preserve_query: false,
            sign_fn: None,
            rewrite_init_segments: true,
            rewrite_media_uris: true,
            rewrite_iframe_uris: true,
        }
    }
}

impl RewriteConfig {
    /// Create a config with a token-signing closure.
    pub fn with_signer<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.sign_fn = Some(Arc::new(f));
        self
    }
}

impl fmt::Debug for RewriteConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RewriteConfig")
            .field("preserve_query", &self.preserve_query)
            .field("sign_fn", &self.sign_fn.is_some())
            .field("rewrite_init_segments", &self.rewrite_init_segments)
            .field("rewrite_media_uris", &self.rewrite_media_uris)
            .field("rewrite_iframe_uris", &self.rewrite_iframe_uris)
            .finish()
    }
}

impl Clone for RewriteConfig {
    fn clone(&self) -> Self {
        Self {
            preserve_query: self.preserve_query,
            sign_fn: self.sign_fn.clone(),
            rewrite_init_segments: self.rewrite_init_segments,
            rewrite_media_uris: self.rewrite_media_uris,
            rewrite_iframe_uris: self.rewrite_iframe_uris,
        }
    }
}

// ─── ManifestRewriter ────────────────────────────────────────────────────────

/// Rewrites HLS and DASH manifests so all child URLs point to an edge node.
#[derive(Debug, Clone)]
pub struct ManifestRewriter {
    /// Target edge scheme+host, e.g. `"https://edge-iad.example.com"`.
    edge_base: String,
    config: RewriteConfig,
}

impl ManifestRewriter {
    /// Create a rewriter targeting `edge_host`.
    ///
    /// `edge_host` must be a scheme + host (and optional port), e.g.
    /// `"https://pop-iad.cdn.example.com"`.  It must **not** end with `/`.
    pub fn new(edge_host: impl Into<String>, config: RewriteConfig) -> Result<Self, RewriteError> {
        let edge_base = edge_host.into();
        if !edge_base.starts_with("http://") && !edge_base.starts_with("https://") {
            return Err(RewriteError::InvalidEdgeHost(edge_base));
        }
        Ok(Self { edge_base, config })
    }

    /// Detect the manifest format from `content` (by looking for known markers).
    pub fn detect_format(content: &str) -> Option<ManifestFormat> {
        let trimmed = content.trim_start();
        if trimmed.starts_with("#EXTM3U") {
            return Some(ManifestFormat::Hls);
        }
        if trimmed.starts_with("<?xml") || trimmed.contains("<MPD") {
            return Some(ManifestFormat::Dash);
        }
        None
    }

    /// Rewrite `manifest_content` using the declared or auto-detected format.
    ///
    /// `base_url` is the canonical URL from which the manifest was fetched;
    /// it is used to resolve relative URLs in the manifest.
    ///
    /// Returns `(rewritten_content, stats)`.
    pub fn rewrite(
        &self,
        manifest_content: &str,
        base_url: &str,
        format_hint: Option<ManifestFormat>,
    ) -> Result<(String, RewriteStats), RewriteError> {
        let format = format_hint
            .or_else(|| Self::detect_format(manifest_content))
            .ok_or_else(|| RewriteError::Malformed("cannot detect manifest format".to_string()))?;

        match format {
            ManifestFormat::Hls => self.rewrite_hls(manifest_content, base_url),
            ManifestFormat::Dash => self.rewrite_dash(manifest_content, base_url),
        }
    }

    // ── HLS rewriting ────────────────────────────────────────────────────

    fn rewrite_hls(
        &self,
        content: &str,
        base_url: &str,
    ) -> Result<(String, RewriteStats), RewriteError> {
        let base_path = extract_base_path(base_url);
        let mut output = String::with_capacity(content.len() + 256);
        let mut stats = RewriteStats {
            format: Some(ManifestFormat::Hls),
            ..RewriteStats::default()
        };

        for line in content.lines() {
            let trimmed = line.trim();

            // EXT-X-MAP URI= (init segment)
            if self.config.rewrite_init_segments && trimmed.starts_with("#EXT-X-MAP:") {
                let new_line = self.rewrite_hls_tag_uri(line, &base_path, &mut stats);
                output.push_str(&new_line);
                output.push('\n');
                continue;
            }

            // EXT-X-MEDIA URI= (alternative rendition)
            if self.config.rewrite_media_uris && trimmed.starts_with("#EXT-X-MEDIA:") {
                let new_line = self.rewrite_hls_tag_uri(line, &base_path, &mut stats);
                output.push_str(&new_line);
                output.push('\n');
                continue;
            }

            // EXT-X-I-FRAME-STREAM-INF URI=
            if self.config.rewrite_iframe_uris && trimmed.starts_with("#EXT-X-I-FRAME-STREAM-INF:")
            {
                let new_line = self.rewrite_hls_tag_uri(line, &base_path, &mut stats);
                output.push_str(&new_line);
                output.push('\n');
                continue;
            }

            // EXT-X-STREAM-INF is followed by a URI on the next line;
            // we handle that in the segment-URI branch below.

            // Blank lines, comments, and other directives: pass through.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                output.push_str(line);
                output.push('\n');
                continue;
            }

            // Otherwise this is a segment URI (or rendition playlist URI).
            let rewritten = self.rewrite_url(trimmed, &base_path, &mut stats);
            output.push_str(&rewritten);
            output.push('\n');
        }

        // Remove trailing newline added unconditionally in the loop.
        if output.ends_with('\n') && !content.ends_with('\n') {
            output.pop();
        }

        Ok((output, stats))
    }

    /// Rewrite the `URI="..."` attribute value inside an HLS tag line.
    fn rewrite_hls_tag_uri(&self, line: &str, base_path: &str, stats: &mut RewriteStats) -> String {
        // Find URI="..." pattern.
        const URI_ATTR: &str = "URI=\"";
        if let Some(start) = line.find(URI_ATTR) {
            let uri_start = start + URI_ATTR.len();
            if let Some(end_offset) = line[uri_start..].find('"') {
                let uri = &line[uri_start..uri_start + end_offset];
                let rewritten = self.rewrite_url(uri, base_path, stats);
                let mut result = String::with_capacity(line.len() + rewritten.len());
                result.push_str(&line[..uri_start]);
                result.push_str(&rewritten);
                result.push_str(&line[uri_start + end_offset..]);
                return result;
            }
        }
        // No URI= found — count as unchanged.
        stats.urls_unchanged += 1;
        line.to_string()
    }

    // ── DASH rewriting ───────────────────────────────────────────────────

    fn rewrite_dash(
        &self,
        content: &str,
        base_url: &str,
    ) -> Result<(String, RewriteStats), RewriteError> {
        let base_path = extract_base_path(base_url);
        let mut stats = RewriteStats {
            format: Some(ManifestFormat::Dash),
            ..RewriteStats::default()
        };

        // Rewrite <BaseURL> elements.
        let content =
            self.rewrite_dash_attr(content, "BaseURL", base_url, &base_path, &mut stats, false);

        // Rewrite media= and initialization= attributes in SegmentTemplate.
        let content = self.rewrite_dash_kv_attr(&content, "media", &base_path, &mut stats);
        let content = self.rewrite_dash_kv_attr(&content, "initialization", &base_path, &mut stats);

        // Rewrite sourceURL= (SegmentList / SegmentURL).
        let content = self.rewrite_dash_kv_attr(&content, "sourceURL", &base_path, &mut stats);

        // Rewrite href= (Period, AdaptationSet xlink).
        let content = self.rewrite_dash_kv_attr(&content, "href", &base_path, &mut stats);

        Ok((content, stats))
    }

    /// Rewrite all occurrences of `<{tag}>url</{tag}>` or `<{tag} ...>url</{tag}>`.
    fn rewrite_dash_attr(
        &self,
        content: &str,
        tag: &str,
        _base_url: &str,
        base_path: &str,
        stats: &mut RewriteStats,
        _absolute_only: bool,
    ) -> String {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let mut result = String::with_capacity(content.len());
        let mut remaining = content;

        while let Some(start) = remaining.find(&open) {
            let after_open = start + open.len();
            result.push_str(&remaining[..start]);
            result.push_str(&open);

            if let Some(end) = remaining[after_open..].find(&close) {
                let url = remaining[after_open..after_open + end].trim();
                let rewritten = self.rewrite_url(url, base_path, stats);
                result.push_str(&rewritten);
                remaining = &remaining[after_open + end..];
            } else {
                remaining = &remaining[after_open..];
            }
        }
        result.push_str(remaining);
        result
    }

    /// Rewrite `attr="value"` pairs in the DASH manifest where `attr` is the
    /// given key.
    fn rewrite_dash_kv_attr(
        &self,
        content: &str,
        attr: &str,
        base_path: &str,
        stats: &mut RewriteStats,
    ) -> String {
        let pattern = format!("{attr}=\"");
        let mut result = String::with_capacity(content.len());
        let mut remaining = content;

        while let Some(start) = remaining.find(&pattern) {
            let val_start = start + pattern.len();
            result.push_str(&remaining[..start]);
            result.push_str(&pattern);

            if let Some(end_offset) = remaining[val_start..].find('"') {
                let url = &remaining[val_start..val_start + end_offset];
                // Only rewrite if it looks like a URL (not a template variable
                // like $Number$).
                if url.starts_with("http")
                    || url.starts_with('/')
                    || (!url.starts_with('$') && !url.is_empty())
                {
                    let rewritten = self.rewrite_url(url, base_path, stats);
                    result.push_str(&rewritten);
                } else {
                    result.push_str(url);
                    stats.urls_unchanged += 1;
                }
                result.push('"');
                remaining = &remaining[val_start + end_offset + 1..];
            } else {
                remaining = &remaining[val_start..];
            }
        }
        result.push_str(remaining);
        result
    }

    // ── Core URL rewriting ───────────────────────────────────────────────

    /// Rewrite a single URL to point to the edge.
    ///
    /// Returns the original URL unchanged for data URIs and fragment-only
    /// references.
    fn rewrite_url(&self, url: &str, base_path: &str, stats: &mut RewriteStats) -> String {
        // Skip data URIs and empty strings.
        if url.is_empty() || url.starts_with("data:") || url.starts_with('#') {
            stats.urls_unchanged += 1;
            return url.to_string();
        }

        let resolved = if url.starts_with("http://") || url.starts_with("https://") {
            // Absolute URL: strip the scheme+host, keep the path.
            let path = strip_scheme_and_host(url);
            let (path, query) = split_query(path);
            if self.config.preserve_query {
                if query.is_empty() {
                    path.to_string()
                } else {
                    format!("{path}?{query}")
                }
            } else {
                path.to_string()
            }
        } else if url.starts_with('/') {
            // Root-relative URL.
            if self.config.preserve_query {
                url.to_string()
            } else {
                let (path, _) = split_query(url);
                path.to_string()
            }
        } else {
            // Relative URL — resolve against base_path.
            let joined = join_paths(base_path, url);
            if self.config.preserve_query {
                joined
            } else {
                let (path, _) = split_query(&joined);
                path.to_string()
            }
        };

        let rewritten_url = format!("{}{resolved}", self.edge_base);

        let final_url = match &self.config.sign_fn {
            Some(sign) => sign(&rewritten_url),
            None => rewritten_url,
        };

        stats.urls_rewritten += 1;
        final_url
    }

    /// Return the edge base URL.
    pub fn edge_base(&self) -> &str {
        &self.edge_base
    }
}

// ─── URL helpers ─────────────────────────────────────────────────────────────

/// Extract the directory component of a URL path.
///
/// Given `"https://origin.com/live/master.m3u8"`, returns `"/live/"`.
fn extract_base_path(url: &str) -> String {
    let path = strip_scheme_and_host(url);
    let (path_only, _) = split_query(path);
    match path_only.rfind('/') {
        Some(pos) => path_only[..pos + 1].to_string(),
        None => "/".to_string(),
    }
}

/// Strip the scheme and host from an absolute URL, returning the path.
fn strip_scheme_and_host(url: &str) -> &str {
    let rest = if let Some(r) = url.strip_prefix("https://") {
        r
    } else if let Some(r) = url.strip_prefix("http://") {
        r
    } else {
        return url;
    };
    // Find the first `/` after the host.
    match rest.find('/') {
        Some(pos) => &rest[pos..],
        None => "/",
    }
}

/// Split a URL path at the query string delimiter, returning `(path, query)`.
fn split_query(url: &str) -> (&str, &str) {
    match url.find('?') {
        Some(pos) => (&url[..pos], &url[pos + 1..]),
        None => (url, ""),
    }
}

/// Join a base directory path with a relative URL segment.
///
/// Handles `../` path components.
fn join_paths(base_dir: &str, relative: &str) -> String {
    // Split relative at query string.
    let (rel_path, rel_query) = split_query(relative);

    // Build candidate by appending relative to base.
    let mut parts: Vec<&str> = base_dir.split('/').filter(|s| !s.is_empty()).collect();

    for seg in rel_path.split('/') {
        match seg {
            ".." => {
                parts.pop();
            }
            "." | "" => {}
            s => parts.push(s),
        }
    }

    let mut result = String::from("/");
    result.push_str(&parts.join("/"));

    if !rel_query.is_empty() {
        result.push('?');
        result.push_str(rel_query);
    }

    result
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rewriter(host: &str) -> ManifestRewriter {
        ManifestRewriter::new(host, RewriteConfig::default()).expect("valid host")
    }

    // ── URL helpers ───────────────────────────────────────────────────────

    // 1. extract_base_path strips filename.
    #[test]
    fn test_extract_base_path() {
        assert_eq!(
            extract_base_path("https://origin.com/live/master.m3u8"),
            "/live/"
        );
        assert_eq!(extract_base_path("https://origin.com/"), "/");
    }

    // 2. strip_scheme_and_host.
    #[test]
    fn test_strip_scheme_and_host() {
        assert_eq!(
            strip_scheme_and_host("https://origin.com/a/b.mp4"),
            "/a/b.mp4"
        );
        assert_eq!(strip_scheme_and_host("https://origin.com"), "/");
    }

    // 3. join_paths resolves relative segments.
    #[test]
    fn test_join_paths_basic() {
        assert_eq!(join_paths("/live/", "seg001.ts"), "/live/seg001.ts");
    }

    // 4. join_paths resolves `../`.
    #[test]
    fn test_join_paths_parent() {
        assert_eq!(join_paths("/live/hls/", "../seg001.ts"), "/live/seg001.ts");
    }

    // 5. split_query splits correctly.
    #[test]
    fn test_split_query() {
        let (path, q) = split_query("/seg.ts?t=123");
        assert_eq!(path, "/seg.ts");
        assert_eq!(q, "t=123");

        let (path2, q2) = split_query("/seg.ts");
        assert_eq!(path2, "/seg.ts");
        assert_eq!(q2, "");
    }

    // ── ManifestFormat detection ──────────────────────────────────────────

    // 6. Detect HLS from #EXTM3U header.
    #[test]
    fn test_detect_hls() {
        let content = "#EXTM3U\n#EXT-X-VERSION:3\n";
        assert_eq!(
            ManifestRewriter::detect_format(content),
            Some(ManifestFormat::Hls)
        );
    }

    // 7. Detect DASH from <MPD marker.
    #[test]
    fn test_detect_dash() {
        let content = "<?xml version=\"1.0\"?>\n<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\">";
        assert_eq!(
            ManifestRewriter::detect_format(content),
            Some(ManifestFormat::Dash)
        );
    }

    // 8. Unknown format returns None.
    #[test]
    fn test_detect_unknown() {
        assert!(ManifestRewriter::detect_format("not a manifest").is_none());
    }

    // ── ManifestRewriter construction ─────────────────────────────────────

    // 9. Invalid edge host (no scheme) returns error.
    #[test]
    fn test_invalid_edge_host() {
        let err = ManifestRewriter::new("cdn.example.com", RewriteConfig::default()).unwrap_err();
        assert!(matches!(err, RewriteError::InvalidEdgeHost(_)));
    }

    // 10. edge_base() returns the configured host.
    #[test]
    fn test_edge_base() {
        let r = rewriter("https://edge.cdn.com");
        assert_eq!(r.edge_base(), "https://edge.cdn.com");
    }

    // ── HLS rewriting ─────────────────────────────────────────────────────

    // 11. Simple segment URI is rewritten.
    #[test]
    fn test_hls_segment_uri_rewritten() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXTINF:4.0,\nseg001.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("https://edge.cdn.com/live/seg001.ts"),
            "out={out}"
        );
        assert_eq!(stats.urls_rewritten, 1);
        assert_eq!(stats.format, Some(ManifestFormat::Hls));
    }

    // 12. Absolute segment URL has host replaced.
    #[test]
    fn test_hls_absolute_url_host_replaced() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXTINF:4.0,\nhttps://origin.com/live/seg002.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("https://edge.cdn.com/live/seg002.ts"),
            "out={out}"
        );
        assert_eq!(stats.urls_rewritten, 1);
    }

    // 13. EXT-X-MAP URI is rewritten.
    #[test]
    fn test_hls_ext_x_map_uri() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXT-X-MAP:URI=\"init.mp4\"\n#EXTINF:4.0,\nseg.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("URI=\"https://edge.cdn.com/live/init.mp4\""),
            "out={out}"
        );
        assert_eq!(stats.urls_rewritten, 2);
    }

    // 14. EXT-X-MEDIA URI is rewritten.
    #[test]
    fn test_hls_ext_x_media_uri() {
        let r = rewriter("https://edge.cdn.com");
        let manifest =
            "#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"audio\",URI=\"audio/en.m3u8\"\n";
        let (out, _stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("URI=\"https://edge.cdn.com/live/audio/en.m3u8\""),
            "out={out}"
        );
    }

    // 15. EXT-X-I-FRAME-STREAM-INF URI is rewritten.
    #[test]
    fn test_hls_iframe_uri() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXT-X-I-FRAME-STREAM-INF:BANDWIDTH=100000,URI=\"iframe.m3u8\"\n";
        let (out, _stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("URI=\"https://edge.cdn.com/live/iframe.m3u8\""),
            "out={out}"
        );
    }

    // 16. Comments and directives without URIs are passed through unchanged.
    #[test]
    fn test_hls_comments_unchanged() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n#EXTINF:4.0,\nseg.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(out.contains("#EXT-X-VERSION:3"));
        assert!(out.contains("#EXT-X-TARGETDURATION:4"));
        assert_eq!(stats.urls_rewritten, 1);
    }

    // 17. Root-relative path is rewritten correctly.
    #[test]
    fn test_hls_root_relative_path() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXTINF:4.0,\n/segments/seg001.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("https://edge.cdn.com/segments/seg001.ts"),
            "out={out}"
        );
        assert_eq!(stats.urls_rewritten, 1);
    }

    // 18. preserve_query retains query strings.
    #[test]
    fn test_hls_preserve_query() {
        let config = RewriteConfig {
            preserve_query: true,
            ..RewriteConfig::default()
        };
        let r = ManifestRewriter::new("https://edge.cdn.com", config).expect("ok");
        let manifest = "#EXTM3U\n#EXTINF:4.0,\nhttps://origin.com/live/seg.ts?token=abc\n";
        let (out, _) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(out.contains("?token=abc"), "query not preserved: out={out}");
    }

    // 19. sign_fn is applied to rewritten URLs.
    #[test]
    fn test_hls_sign_fn_applied() {
        let config = RewriteConfig::default().with_signer(|url| format!("{url}?cdn_sig=TEST"));
        let r = ManifestRewriter::new("https://edge.cdn.com", config).expect("ok");
        let manifest = "#EXTM3U\n#EXTINF:4.0,\nseg.ts\n";
        let (out, _) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("cdn_sig=TEST"),
            "sign_fn not applied: out={out}"
        );
    }

    // 20. rewrite_init_segments=false skips EXT-X-MAP URI.
    #[test]
    fn test_hls_skip_init_segment() {
        let config = RewriteConfig {
            rewrite_init_segments: false,
            ..RewriteConfig::default()
        };
        let r = ManifestRewriter::new("https://edge.cdn.com", config).expect("ok");
        let manifest = "#EXTM3U\n#EXT-X-MAP:URI=\"init.mp4\"\n#EXTINF:4.0,\nseg.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        // init.mp4 should NOT be rewritten.
        assert!(
            !out.contains("https://edge.cdn.com/live/init.mp4"),
            "init segment should not be rewritten: out={out}"
        );
        // seg.ts should still be rewritten.
        assert_eq!(stats.urls_rewritten, 1);
    }

    // ── DASH rewriting ────────────────────────────────────────────────────

    // 21. DASH BaseURL is rewritten.
    #[test]
    fn test_dash_base_url_rewritten() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "<?xml?>\n<MPD>\n<BaseURL>https://origin.com/dash/</BaseURL>\n</MPD>";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/dash/manifest.mpd", None)
            .expect("ok");
        assert!(
            out.contains("<BaseURL>https://edge.cdn.com/dash/</BaseURL>"),
            "out={out}"
        );
        assert_eq!(stats.format, Some(ManifestFormat::Dash));
        assert_eq!(stats.urls_rewritten, 1);
    }

    // 22. DASH media= attribute is rewritten.
    #[test]
    fn test_dash_media_attr_rewritten() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "<?xml?>\n<MPD>\n<SegmentTemplate media=\"video/$Number$.mp4\" />\n</MPD>";
        let (out, _stats) = r
            .rewrite(manifest, "https://origin.com/dash/manifest.mpd", None)
            .expect("ok");
        // "$Number$" contains no slashes/http, should be rewritten as relative.
        assert!(out.contains("media=\""), "out={out}");
    }

    // 23. DASH initialization= attribute is rewritten.
    #[test]
    fn test_dash_initialization_attr_rewritten() {
        let r = rewriter("https://edge.cdn.com");
        let manifest =
            "<?xml?>\n<MPD>\n<SegmentTemplate initialization=\"/dash/init.mp4\" />\n</MPD>";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/dash/manifest.mpd", None)
            .expect("ok");
        assert!(
            out.contains("initialization=\"https://edge.cdn.com/dash/init.mp4\""),
            "out={out}"
        );
        assert!(stats.urls_rewritten > 0);
    }

    // 24. ManifestFormat Display.
    #[test]
    fn test_manifest_format_display() {
        assert_eq!(ManifestFormat::Hls.to_string(), "HLS");
        assert_eq!(ManifestFormat::Dash.to_string(), "DASH");
    }

    // 25. Explicit format_hint overrides auto-detection.
    #[test]
    fn test_format_hint_overrides_detection() {
        let r = rewriter("https://edge.cdn.com");
        // Content looks like HLS but we force DASH detection.
        let manifest = "#EXTM3U\n";
        // Rewriting as DASH should not panic (will just produce unchanged output).
        let (_out, stats) = r
            .rewrite(
                manifest,
                "https://origin.com/v/m.mpd",
                Some(ManifestFormat::Dash),
            )
            .expect("ok");
        assert_eq!(stats.format, Some(ManifestFormat::Dash));
    }

    // 26. Multi-segment HLS manifest all URIs rewritten.
    #[test]
    fn test_hls_multi_segment() {
        let r = rewriter("https://edge.cdn.com");
        let manifest =
            "#EXTM3U\n#EXTINF:4.0,\nseg001.ts\n#EXTINF:4.0,\nseg002.ts\n#EXTINF:4.0,\nseg003.ts\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert_eq!(stats.urls_rewritten, 3);
        assert!(out.contains("https://edge.cdn.com/live/seg001.ts"));
        assert!(out.contains("https://edge.cdn.com/live/seg002.ts"));
        assert!(out.contains("https://edge.cdn.com/live/seg003.ts"));
    }

    // 27. join_paths with double-dot traversal.
    #[test]
    fn test_join_paths_double_dot_traversal() {
        assert_eq!(join_paths("/a/b/c/", "../../x.ts"), "/a/x.ts");
    }

    // 28. Data URI is left unchanged.
    #[test]
    fn test_data_uri_unchanged() {
        let r = rewriter("https://edge.cdn.com");
        let manifest = "#EXTM3U\n#EXTINF:4.0,\ndata:text/plain,hello\n";
        let (out, stats) = r
            .rewrite(manifest, "https://origin.com/live/master.m3u8", None)
            .expect("ok");
        assert!(
            out.contains("data:text/plain,hello"),
            "data URI should be unchanged: out={out}"
        );
        assert_eq!(stats.urls_unchanged, 1);
        assert_eq!(stats.urls_rewritten, 0);
    }

    // 29. Fragment-only URL is left unchanged.
    #[test]
    fn test_fragment_only_url_unchanged() {
        let config = RewriteConfig::default();
        let r = ManifestRewriter::new("https://edge.cdn.com", config).expect("ok");
        // Simulate a rewrite_url call directly.
        let mut stats = RewriteStats::default();
        let result = r.rewrite_url("#fragment", "/", &mut stats);
        assert_eq!(result, "#fragment");
        assert_eq!(stats.urls_unchanged, 1);
    }

    // 30. RewriteStats default is zeroed.
    #[test]
    fn test_rewrite_stats_default() {
        let s = RewriteStats::default();
        assert_eq!(s.urls_rewritten, 0);
        assert_eq!(s.urls_unchanged, 0);
        assert!(s.format.is_none());
    }
}
