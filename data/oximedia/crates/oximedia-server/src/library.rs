//! Media library management.

use crate::{
    config::Config,
    db::Database,
    error::{ServerError, ServerResult},
    models::media::{Media, MediaMetadata, MediaStatus},
};
use std::path::Path;
use walkdir::WalkDir;

/// Media library manager.
#[derive(Clone)]
pub struct MediaLibrary {
    db: Database,
    config: Config,
}

impl MediaLibrary {
    /// Creates a new media library manager.
    #[must_use]
    pub fn new(db: Database, config: Config) -> Self {
        Self { db, config }
    }

    /// Adds a new media file to the library.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn add_media(&self, media: &Media) -> ServerResult<()> {
        crate::db::query(
            r"
            INSERT INTO media (
                id, user_id, filename, original_filename, mime_type, file_size,
                duration, width, height, codec_video, codec_audio,
                bitrate, framerate, sample_rate, channels,
                thumbnail_path, sprite_path, preview_path,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&media.id)
        .bind(&media.user_id)
        .bind(&media.filename)
        .bind(&media.original_filename)
        .bind(&media.mime_type)
        .bind(media.file_size)
        .bind(media.duration)
        .bind(media.width)
        .bind(media.height)
        .bind(&media.codec_video)
        .bind(&media.codec_audio)
        .bind(media.bitrate)
        .bind(media.framerate)
        .bind(media.sample_rate)
        .bind(media.channels)
        .bind(&media.thumbnail_path)
        .bind(&media.sprite_path)
        .bind(&media.preview_path)
        .bind(media.status.to_string())
        .bind(media.created_at)
        .bind(media.updated_at)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Gets a media file by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the media is not found or database query fails.
    pub async fn get_media(&self, media_id: &str) -> ServerResult<Media> {
        let row = crate::db::query(
            r"
            SELECT * FROM media WHERE id = ?
            ",
        )
        .bind(media_id)
        .fetch_one(self.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Media not found: {media_id}")))?;

        Ok(self.row_to_media(&row)?)
    }

    /// Updates media metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn update_media(&self, media: &Media) -> ServerResult<()> {
        crate::db::query(
            r"
            UPDATE media SET
                duration = ?, width = ?, height = ?,
                codec_video = ?, codec_audio = ?,
                bitrate = ?, framerate = ?, sample_rate = ?, channels = ?,
                thumbnail_path = ?, sprite_path = ?, preview_path = ?,
                status = ?, updated_at = ?
            WHERE id = ?
            ",
        )
        .bind(media.duration)
        .bind(media.width)
        .bind(media.height)
        .bind(&media.codec_video)
        .bind(&media.codec_audio)
        .bind(media.bitrate)
        .bind(media.framerate)
        .bind(media.sample_rate)
        .bind(media.channels)
        .bind(&media.thumbnail_path)
        .bind(&media.sprite_path)
        .bind(&media.preview_path)
        .bind(media.status.to_string())
        .bind(chrono::Utc::now().timestamp())
        .bind(&media.id)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Deletes a media file.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation or file deletion fails.
    pub async fn delete_media(&self, media_id: &str) -> ServerResult<()> {
        let media = self.get_media(media_id).await?;

        // Delete physical file
        let file_path = self.config.media_dir.join(&media.filename);
        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
        }

        // Delete thumbnails and previews
        if let Some(thumb) = &media.thumbnail_path {
            let thumb_path = self.config.thumbnail_dir.join(thumb);
            if thumb_path.exists() {
                std::fs::remove_file(thumb_path).ok();
            }
        }

        if let Some(sprite) = &media.sprite_path {
            let sprite_path = self.config.thumbnail_dir.join(sprite);
            if sprite_path.exists() {
                std::fs::remove_file(sprite_path).ok();
            }
        }

        if let Some(preview) = &media.preview_path {
            let preview_path = self.config.media_dir.join(preview);
            if preview_path.exists() {
                std::fs::remove_file(preview_path).ok();
            }
        }

        // Delete from database (cascades to metadata, collection items, etc.)
        crate::db::query("DELETE FROM media WHERE id = ?")
            .bind(media_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Lists media files for a user.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn list_media(
        &self,
        user_id: &str,
        limit: i64,
        offset: i64,
    ) -> ServerResult<Vec<Media>> {
        let rows = crate::db::query(
            r"
            SELECT * FROM media
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            ",
        )
        .bind(user_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter()
            .map(|row| self.row_to_media(row))
            .collect::<ServerResult<Vec<_>>>()
    }

    /// Searches media files.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn search_media(
        &self,
        user_id: &str,
        query: &str,
        limit: i64,
    ) -> ServerResult<Vec<Media>> {
        let search_pattern = format!("%{query}%");
        let rows = crate::db::query(
            r"
            SELECT * FROM media
            WHERE user_id = ?
            AND (original_filename LIKE ? OR filename LIKE ?)
            ORDER BY created_at DESC
            LIMIT ?
            ",
        )
        .bind(user_id)
        .bind(&search_pattern)
        .bind(&search_pattern)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        rows.iter()
            .map(|row| self.row_to_media(row))
            .collect::<ServerResult<Vec<_>>>()
    }

    /// Scans a directory and adds media files to the library.
    ///
    /// # Errors
    ///
    /// Returns an error if directory scanning or database operations fail.
    pub async fn scan_directory(&self, user_id: &str, dir: &Path) -> ServerResult<usize> {
        let mut count = 0;

        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // Detect MIME type
            let mime_type = mime_guess::from_path(path)
                .first_or_octet_stream()
                .to_string();

            // Only process media files
            if !mime_type.starts_with("video/")
                && !mime_type.starts_with("audio/")
                && !mime_type.starts_with("image/")
            {
                continue;
            }

            // `path.is_file()` was already checked above, so a missing
            // file-name component is not expected — but we skip the entry
            // instead of panicking rather than trust that invariant blindly.
            let Some(filename_os) = path.file_name() else {
                continue;
            };
            let filename = filename_os.to_string_lossy().to_string();
            let file_size = entry
                .metadata()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
                .len() as i64;

            // Check if already in library
            let exists = crate::db::query_scalar::<i64, _>(
                "SELECT COUNT(*) FROM media WHERE user_id = ? AND filename = ?",
            )
            .bind(user_id)
            .bind(&filename)
            .fetch_one(self.db.pool())
            .await?;

            if exists > 0 {
                continue;
            }

            // Add to library
            let media = Media::new(
                user_id.to_string(),
                filename.clone(),
                filename,
                mime_type,
                file_size,
            );

            self.add_media(&media).await?;
            count += 1;
        }

        Ok(count)
    }

    /// Adds metadata to a media file.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn add_metadata(&self, media_id: &str, key: &str, value: &str) -> ServerResult<()> {
        crate::db::query(
            r"
            INSERT OR REPLACE INTO media_metadata (media_id, key, value)
            VALUES (?, ?, ?)
            ",
        )
        .bind(media_id)
        .bind(key)
        .bind(value)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Gets metadata for a media file.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub async fn get_metadata(&self, media_id: &str) -> ServerResult<MediaMetadata> {
        let rows = crate::db::query(
            r"
            SELECT key, value FROM media_metadata
            WHERE media_id = ?
            ",
        )
        .bind(media_id)
        .fetch_all(self.db.pool())
        .await?;

        let mut metadata = MediaMetadata::new(media_id.to_string());
        for row in rows {
            let key: String = row.get("key");
            let value: String = row.get("value");
            metadata.add(key, value);
        }

        Ok(metadata)
    }

    /// Helper to convert a database row to a Media struct.
    fn row_to_media(&self, row: &crate::db::Row) -> ServerResult<Media> {
        let status_str: String = row.get("status");
        let status = status_str
            .parse::<MediaStatus>()
            .map_err(|e| ServerError::Internal(e))?;

        Ok(Media {
            id: row.get("id"),
            user_id: row.get("user_id"),
            filename: row.get("filename"),
            original_filename: row.get("original_filename"),
            mime_type: row.get("mime_type"),
            file_size: row.get("file_size"),
            duration: row.get("duration"),
            width: row.get("width"),
            height: row.get("height"),
            codec_video: row.get("codec_video"),
            codec_audio: row.get("codec_audio"),
            bitrate: row.get("bitrate"),
            framerate: row.get("framerate"),
            sample_rate: row.get("sample_rate"),
            channels: row.get("channels"),
            thumbnail_path: row.get("thumbnail_path"),
            sprite_path: row.get("sprite_path"),
            preview_path: row.get("preview_path"),
            status,
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
    }
}
