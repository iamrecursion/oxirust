//! Link statistics and analytics.

use super::database::LinkDatabase;
use std::collections::HashMap;

/// Link statistics collector.
pub struct LinkStatistics<'a> {
    database: &'a LinkDatabase,
}

impl<'a> LinkStatistics<'a> {
    /// Create a new statistics collector.
    #[must_use]
    pub const fn new(database: &'a LinkDatabase) -> Self {
        Self { database }
    }

    /// Get comprehensive statistics about all links.
    #[must_use]
    pub fn collect(&self) -> Statistics {
        let all_links = self.database.all_links();
        let total_links = all_links.len();

        if total_links == 0 {
            return Statistics::default();
        }

        let mut total_duration = 0.0;
        let mut total_original_size = 0u64;
        let mut total_proxy_size = 0u64;
        let mut codec_distribution = HashMap::new();
        let mut scale_distribution = HashMap::new();
        let mut verified_count = 0;
        let mut unverified_count = 0;

        for link in &all_links {
            total_duration += link.duration;

            // Get file sizes if files exist
            if let Ok(metadata) = std::fs::metadata(&link.original_path) {
                total_original_size += metadata.len();
            }
            if let Ok(metadata) = std::fs::metadata(&link.proxy_path) {
                total_proxy_size += metadata.len();
            }

            // Codec distribution
            *codec_distribution.entry(link.codec.clone()).or_insert(0) += 1;

            // Scale factor distribution
            let scale_key = format!("{:.0}%", link.scale_factor * 100.0);
            *scale_distribution.entry(scale_key).or_insert(0) += 1;

            // Verification status
            if link.verified_at.is_some() {
                verified_count += 1;
            } else {
                unverified_count += 1;
            }
        }

        let compression_ratio = if total_original_size > 0 {
            total_proxy_size as f64 / total_original_size as f64
        } else {
            0.0
        };

        let space_saved = if total_original_size > total_proxy_size {
            total_original_size - total_proxy_size
        } else {
            0
        };

        Statistics {
            total_links,
            total_duration,
            total_original_size,
            total_proxy_size,
            compression_ratio,
            space_saved,
            codec_distribution,
            scale_distribution,
            verified_count,
            unverified_count,
        }
    }

    /// Get statistics for a specific codec.
    #[must_use]
    pub fn codec_statistics(&self, codec: &str) -> CodecStatistics {
        let all_links = self.database.all_links();
        let codec_links: Vec<_> = all_links
            .iter()
            .filter(|link| link.codec == codec)
            .collect();

        let count = codec_links.len();
        if count == 0 {
            return CodecStatistics::default();
        }

        let mut total_size = 0u64;
        let mut total_duration = 0.0;

        for link in &codec_links {
            if let Ok(metadata) = std::fs::metadata(&link.proxy_path) {
                total_size += metadata.len();
            }
            total_duration += link.duration;
        }

        let avg_bitrate = if total_duration > 0.0 {
            (total_size as f64 * 8.0 / total_duration) as u64
        } else {
            0
        };

        CodecStatistics {
            codec: codec.to_string(),
            count,
            total_size,
            total_duration,
            avg_bitrate,
        }
    }

    /// Get the most popular proxy settings.
    #[must_use]
    pub fn popular_settings(&self) -> Vec<(f32, String, usize)> {
        let all_links = self.database.all_links();
        let mut settings_map: HashMap<(String, String), usize> = HashMap::new();

        for link in &all_links {
            let key = (format!("{:.2}", link.scale_factor), link.codec.clone());
            *settings_map.entry(key).or_insert(0) += 1;
        }

        let mut settings_vec: Vec<_> = settings_map
            .into_iter()
            .map(|((scale, codec), count)| {
                let scale_f32 = scale.parse::<f32>().unwrap_or(0.0);
                (scale_f32, codec, count)
            })
            .collect();

        settings_vec.sort_by(|a, b| b.2.cmp(&a.2));
        settings_vec
    }
}

/// Comprehensive link statistics.
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Total number of links.
    pub total_links: usize,

    /// Total duration of all media in seconds.
    pub total_duration: f64,

    /// Total size of original files in bytes.
    pub total_original_size: u64,

    /// Total size of proxy files in bytes.
    pub total_proxy_size: u64,

    /// Average compression ratio (proxy size / original size).
    pub compression_ratio: f64,

    /// Total space saved in bytes.
    pub space_saved: u64,

    /// Distribution of codecs used.
    pub codec_distribution: HashMap<String, usize>,

    /// Distribution of scale factors.
    pub scale_distribution: HashMap<String, usize>,

    /// Number of verified links.
    pub verified_count: usize,

    /// Number of unverified links.
    pub unverified_count: usize,
}

impl Statistics {
    /// Get a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Total Links: {}\n\
             Total Duration: {:.2} hours\n\
             Original Size: {}\n\
             Proxy Size: {}\n\
             Space Saved: {} ({:.1}%)\n\
             Compression Ratio: {:.2}:1\n\
             Verified: {} / Unverified: {}",
            self.total_links,
            self.total_duration / 3600.0,
            format_bytes(self.total_original_size),
            format_bytes(self.total_proxy_size),
            format_bytes(self.space_saved),
            (1.0 - self.compression_ratio) * 100.0,
            1.0 / self.compression_ratio,
            self.verified_count,
            self.unverified_count
        )
    }

    /// Get the most used codec.
    #[must_use]
    pub fn most_used_codec(&self) -> Option<String> {
        self.codec_distribution
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(codec, _)| codec.clone())
    }

    /// Get verification percentage.
    #[must_use]
    pub fn verification_percentage(&self) -> f64 {
        if self.total_links == 0 {
            0.0
        } else {
            (self.verified_count as f64 / self.total_links as f64) * 100.0
        }
    }
}

/// Codec-specific statistics.
#[derive(Debug, Clone, Default)]
pub struct CodecStatistics {
    /// Codec name.
    pub codec: String,

    /// Number of proxies using this codec.
    pub count: usize,

    /// Total size of all proxies with this codec.
    pub total_size: u64,

    /// Total duration of all proxies with this codec.
    pub total_duration: f64,

    /// Average bitrate in bits per second.
    pub avg_bitrate: u64,
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::database::ProxyLinkRecord;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_statistics_collection() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_stats.json");

        let mut db = LinkDatabase::new(&db_path)
            .await
            .expect("should succeed in test");

        let record = ProxyLinkRecord {
            proxy_path: PathBuf::from("proxy.mp4"),
            original_path: PathBuf::from("original.mov"),
            scale_factor: 0.25,
            codec: "h264".to_string(),
            duration: 60.0,
            timecode: None,
            created_at: 123456789,
            verified_at: Some(123456800),
            metadata: HashMap::new(),
        };

        db.add_link(record).expect("should succeed in test");

        let stats_collector = LinkStatistics::new(&db);
        let stats = stats_collector.collect();

        assert_eq!(stats.total_links, 1);
        assert_eq!(stats.total_duration, 60.0);
        assert_eq!(stats.verified_count, 1);
        assert_eq!(stats.unverified_count, 0);

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_statistics_summary() {
        let stats = Statistics {
            total_links: 10,
            total_duration: 600.0,
            total_original_size: 1_000_000_000,
            total_proxy_size: 100_000_000,
            compression_ratio: 0.1,
            space_saved: 900_000_000,
            codec_distribution: HashMap::new(),
            scale_distribution: HashMap::new(),
            verified_count: 8,
            unverified_count: 2,
        };

        let summary = stats.summary();
        assert!(summary.contains("Total Links: 10"));
        assert!(summary.contains("Verified: 8"));
    }

    #[test]
    fn test_verification_percentage() {
        let stats = Statistics {
            total_links: 10,
            verified_count: 7,
            ..Default::default()
        };

        assert_eq!(stats.verification_percentage(), 70.0);
    }
}
