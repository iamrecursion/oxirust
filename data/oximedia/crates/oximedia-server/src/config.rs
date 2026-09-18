//! Server configuration.

use std::path::PathBuf;

/// Server configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// Database connection URL
    pub database_url: String,

    /// Media storage directory
    pub media_dir: PathBuf,

    /// Thumbnail storage directory
    pub thumbnail_dir: PathBuf,

    /// Temporary file directory
    pub temp_dir: PathBuf,

    /// JWT secret key
    pub jwt_secret: String,

    /// JWT token expiration in seconds
    pub jwt_expiration: u64,

    /// Maximum upload size in bytes
    pub max_upload_size: usize,

    /// Maximum concurrent transcoding jobs
    pub max_concurrent_jobs: usize,

    /// Enable bandwidth throttling
    pub enable_throttling: bool,

    /// Maximum bandwidth per stream in bytes/sec
    pub max_stream_bandwidth: usize,

    /// Rate limit: requests per minute per user
    pub rate_limit_per_minute: u32,

    /// HLS segment duration in seconds
    pub hls_segment_duration: u64,

    /// DASH segment duration in seconds
    pub dash_segment_duration: u64,

    /// Thumbnail sprite columns
    pub sprite_columns: u32,

    /// Thumbnail sprite rows
    pub sprite_rows: u32,

    /// Thumbnail width
    pub thumbnail_width: u32,

    /// Thumbnail height
    pub thumbnail_height: u32,

    /// Preview duration in seconds
    pub preview_duration: u64,

    /// Enable CORS
    pub enable_cors: bool,

    /// CORS allowed origins
    pub cors_origins: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database_url: "sqlite:oximedia.db".to_string(),
            media_dir: PathBuf::from("media"),
            thumbnail_dir: PathBuf::from("thumbnails"),
            temp_dir: PathBuf::from("temp"),
            jwt_secret: Self::generate_secret(),
            jwt_expiration: 3600 * 24,               // 24 hours
            max_upload_size: 5 * 1024 * 1024 * 1024, // 5GB
            max_concurrent_jobs: 4,
            enable_throttling: false,
            max_stream_bandwidth: 10 * 1024 * 1024, // 10 MB/s
            rate_limit_per_minute: 60,
            hls_segment_duration: 6,
            dash_segment_duration: 4,
            sprite_columns: 10,
            sprite_rows: 10,
            thumbnail_width: 160,
            thumbnail_height: 90,
            preview_duration: 30,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

impl Config {
    /// Creates a new configuration from environment variables.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:oximedia.db".to_string()),
            media_dir: std::env::var("MEDIA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("media")),
            thumbnail_dir: std::env::var("THUMBNAIL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("thumbnails")),
            temp_dir: std::env::var("TEMP_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("temp")),
            jwt_secret: std::env::var("JWT_SECRET").unwrap_or_else(|_| Self::generate_secret()),
            jwt_expiration: std::env::var("JWT_EXPIRATION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3600 * 24),
            max_upload_size: std::env::var("MAX_UPLOAD_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5 * 1024 * 1024 * 1024),
            max_concurrent_jobs: std::env::var("MAX_CONCURRENT_JOBS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4),
            enable_throttling: std::env::var("ENABLE_THROTTLING")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(false),
            max_stream_bandwidth: std::env::var("MAX_STREAM_BANDWIDTH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10 * 1024 * 1024),
            rate_limit_per_minute: std::env::var("RATE_LIMIT_PER_MINUTE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60),
            hls_segment_duration: std::env::var("HLS_SEGMENT_DURATION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(6),
            dash_segment_duration: std::env::var("DASH_SEGMENT_DURATION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4),
            sprite_columns: std::env::var("SPRITE_COLUMNS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            sprite_rows: std::env::var("SPRITE_ROWS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            thumbnail_width: std::env::var("THUMBNAIL_WIDTH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(160),
            thumbnail_height: std::env::var("THUMBNAIL_HEIGHT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(90),
            preview_duration: std::env::var("PREVIEW_DURATION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
            enable_cors: std::env::var("ENABLE_CORS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(true),
            cors_origins: std::env::var("CORS_ORIGINS")
                .map(|s| s.split(',').map(String::from).collect())
                .unwrap_or_else(|_| vec!["*".to_string()]),
        }
    }

    /// Generates a random secret key.
    fn generate_secret() -> String {
        use rand::Rng;
        let mut key = [0u8; 64];
        rand::rng().fill_bytes(&mut key);
        hex::encode(key)
    }
}
