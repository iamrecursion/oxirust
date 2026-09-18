//! Main clip manager that integrates all functionality.

use crate::clip::{Clip, ClipId};
use crate::database::ClipDatabase;
use crate::error::{ClipError, ClipResult};
use crate::export::{ClipListExporter, EdlExporter};
use crate::group::{
    Bin, BinId, ClipField, Collection, CollectionId, Folder, FolderId, SmartCollection,
};
use crate::import::{BatchImporter, MediaScanner};
use crate::marker::{Marker, MarkerId, MarkerManager};
use crate::proxy::{ProxyLink, ProxyManager};
use crate::search::{ClipFilter, SearchEngine};
use crate::take::{Take, TakeManager};
use oximedia_core::types::Rational;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Main clip management system.
pub struct ClipManager {
    database: ClipDatabase,
    marker_manager: MarkerManager,
    take_manager: TakeManager,
    proxy_manager: ProxyManager,
    #[allow(dead_code)]
    search_engine: SearchEngine,
    bins: HashMap<BinId, Bin>,
    folders: HashMap<FolderId, Folder>,
    collections: HashMap<CollectionId, Collection>,
    /// Registered smart collections.
    ///
    /// Wrapped in a [`Mutex`] for interior mutability so that `&self` methods
    /// (notably [`ClipManager::update_clip`], which is shared across concurrent
    /// writers via `Arc<ClipManager>`) can invalidate dependent caches without
    /// requiring `&mut self`.
    smart_collections: Mutex<Vec<SmartCollection>>,
}

impl ClipManager {
    /// Creates a new clip manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be initialized.
    pub async fn new(database_url: impl AsRef<str>) -> ClipResult<Self> {
        let database = ClipDatabase::new(database_url).await?;

        Ok(Self {
            database,
            marker_manager: MarkerManager::new(),
            take_manager: TakeManager::new(),
            proxy_manager: ProxyManager::new(),
            search_engine: SearchEngine::new(),
            bins: HashMap::new(),
            folders: HashMap::new(),
            collections: HashMap::new(),
            smart_collections: Mutex::new(Vec::new()),
        })
    }

    // Clip operations

    /// Adds a clip to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be saved.
    pub async fn add_clip(&self, clip: Clip) -> ClipResult<ClipId> {
        let clip_id = clip.id;
        self.database.save_clip(&clip).await?;
        Ok(clip_id)
    }

    /// Gets a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip is not found.
    pub async fn get_clip(&self, clip_id: &ClipId) -> ClipResult<Clip> {
        self.database.get_clip(clip_id).await
    }

    /// Updates a clip, auto-invalidating dependent smart-collection caches.
    ///
    /// Before saving, the previous version of the clip is loaded so that the
    /// set of changed fields can be computed. Only the smart collections whose
    /// rules depend on a changed field have their caches invalidated (field
    /// dependency). When the clip did not previously exist (i.e. this is an
    /// insert), every smart-collection cache is invalidated, because a new clip
    /// may newly match any rule.
    ///
    /// This method takes `&self` (interior mutability) so it remains safe to
    /// share a `ClipManager` across concurrent writers via `Arc`.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be saved or if the smart-collection
    /// registry lock is poisoned.
    pub async fn update_clip(&self, clip: Clip) -> ClipResult<()> {
        // Load the previous version (if any) to diff against. `get_clip`
        // returns `Err(ClipNotFound)` for a brand-new clip, so `.ok()` cleanly
        // distinguishes update (`Some`) from insert (`None`).
        let previous = self.database.get_clip(&clip.id).await.ok();

        // Persist first so the saved state is authoritative.
        self.database.save_clip(&clip).await?;

        match previous {
            Some(old) => {
                let changed = Self::changed_fields(&old, &clip);
                for field in changed {
                    self.invalidate_smart_collections_for(field)?;
                }
            }
            None => {
                // Insert: a new clip may match any rule, so invalidate all.
                self.invalidate_all_smart_collections()?;
            }
        }

        Ok(())
    }

    /// Deletes a clip.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be deleted.
    pub async fn delete_clip(&self, clip_id: &ClipId) -> ClipResult<()> {
        self.database.delete_clip(clip_id).await
    }

    /// Gets all clips.
    ///
    /// # Errors
    ///
    /// Returns an error if clips cannot be loaded.
    pub async fn get_all_clips(&self) -> ClipResult<Vec<Clip>> {
        self.database.get_all_clips().await
    }

    /// Returns the number of clips.
    ///
    /// # Errors
    ///
    /// Returns an error if the count fails.
    pub async fn clip_count(&self) -> ClipResult<i64> {
        self.database.count_clips().await
    }

    /// Adds multiple clips in a single database transaction (bulk import).
    ///
    /// Significantly more efficient than calling `add_clip()` in a loop.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction or any individual save fails.
    pub async fn add_clips(&self, clips: Vec<Clip>) -> ClipResult<Vec<ClipId>> {
        let ids: Vec<ClipId> = clips.iter().map(|c| c.id).collect();
        self.database.batch_save_clips(&clips).await?;
        Ok(ids)
    }

    /// Lists clips with pagination.
    ///
    /// `page` is 0-indexed; `page_size` is the number of clips per page.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_clips(&self, page: i64, page_size: i64) -> ClipResult<Vec<Clip>> {
        self.database.get_clips_page(page, page_size).await
    }

    /// Searches clips by query string with pagination.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search_paged(
        &self,
        query: &str,
        page: i64,
        page_size: i64,
    ) -> ClipResult<Vec<Clip>> {
        self.database
            .search_clips_page(query, page, page_size)
            .await
    }

    // Search operations

    /// Searches clips by query string.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search(&self, query: &str) -> ClipResult<Vec<Clip>> {
        self.database.search_clips(query).await
    }

    /// Filters clips using advanced criteria.
    ///
    /// # Errors
    ///
    /// Returns an error if the filter operation fails.
    pub async fn filter(&self, filter: &ClipFilter) -> ClipResult<Vec<Clip>> {
        let clips = self.database.get_all_clips().await?;
        Ok(filter.apply(&clips).into_iter().cloned().collect())
    }

    // Bin operations

    /// Creates a new bin.
    pub fn create_bin(&mut self, name: impl Into<String>) -> BinId {
        let bin = Bin::new(name);
        let bin_id = bin.id;
        self.bins.insert(bin_id, bin);
        bin_id
    }

    /// Gets a bin by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the bin is not found.
    pub fn get_bin(&self, bin_id: &BinId) -> ClipResult<&Bin> {
        self.bins
            .get(bin_id)
            .ok_or_else(|| ClipError::BinNotFound(bin_id.to_string()))
    }

    /// Gets a mutable bin by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the bin is not found.
    pub fn get_bin_mut(&mut self, bin_id: &BinId) -> ClipResult<&mut Bin> {
        self.bins
            .get_mut(bin_id)
            .ok_or_else(|| ClipError::BinNotFound(bin_id.to_string()))
    }

    /// Adds a clip to a bin.
    ///
    /// # Errors
    ///
    /// Returns an error if the bin is not found.
    pub fn add_clip_to_bin(&mut self, bin_id: &BinId, clip_id: ClipId) -> ClipResult<()> {
        let bin = self.get_bin_mut(bin_id)?;
        bin.add_clip(clip_id);
        Ok(())
    }

    /// Lists all bins.
    #[must_use]
    pub fn list_bins(&self) -> Vec<&Bin> {
        self.bins.values().collect()
    }

    // Folder operations

    /// Creates a new folder.
    pub fn create_folder(&mut self, name: impl Into<String>) -> FolderId {
        let folder = Folder::new(name);
        let folder_id = folder.id;
        self.folders.insert(folder_id, folder);
        folder_id
    }

    /// Creates a child folder.
    pub fn create_child_folder(
        &mut self,
        name: impl Into<String>,
        parent_id: FolderId,
    ) -> FolderId {
        let folder = Folder::new_child(name, parent_id);
        let folder_id = folder.id;
        self.folders.insert(folder_id, folder);
        folder_id
    }

    /// Gets a folder by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the folder is not found.
    pub fn get_folder(&self, folder_id: &FolderId) -> ClipResult<&Folder> {
        self.folders
            .get(folder_id)
            .ok_or_else(|| ClipError::FolderNotFound(folder_id.to_string()))
    }

    // Collection operations

    /// Creates a new collection.
    pub fn create_collection(&mut self, name: impl Into<String>) -> CollectionId {
        let collection = Collection::new(name);
        let collection_id = collection.id;
        self.collections.insert(collection_id, collection);
        collection_id
    }

    /// Gets a collection by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not found.
    pub fn get_collection(&self, collection_id: &CollectionId) -> ClipResult<&Collection> {
        self.collections
            .get(collection_id)
            .ok_or_else(|| ClipError::CollectionNotFound(collection_id.to_string()))
    }

    /// Adds a clip to a collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not found.
    pub fn add_clip_to_collection(
        &mut self,
        collection_id: &CollectionId,
        clip_id: ClipId,
    ) -> ClipResult<()> {
        let collection = self
            .collections
            .get_mut(collection_id)
            .ok_or_else(|| ClipError::CollectionNotFound(collection_id.to_string()))?;
        collection.add_clip(clip_id);
        Ok(())
    }

    // Smart collection operations

    /// Locks the smart-collection registry, mapping a poisoned lock to a
    /// recoverable [`ClipError`] instead of panicking.
    fn lock_smart_collections(
        &self,
    ) -> ClipResult<std::sync::MutexGuard<'_, Vec<SmartCollection>>> {
        self.smart_collections
            .lock()
            .map_err(|_| ClipError::Serialization("smart_collections mutex poisoned".into()))
    }

    /// Creates a new smart collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the smart-collection registry lock is poisoned.
    pub fn create_smart_collection(&self, smart_collection: SmartCollection) -> ClipResult<()> {
        self.lock_smart_collections()?.push(smart_collection);
        Ok(())
    }

    /// Updates all smart collections by re-evaluating every clip.
    ///
    /// # Errors
    ///
    /// Returns an error if clips cannot be loaded or the registry lock is
    /// poisoned.
    pub async fn update_smart_collections(&self) -> ClipResult<()> {
        let clips = self.database.get_all_clips().await?;

        let mut guard = self.lock_smart_collections()?;
        for smart_collection in guard.iter_mut() {
            smart_collection.update(&clips);
        }

        Ok(())
    }

    /// Invalidates the caches of every smart collection whose rules depend on
    /// `field`.
    ///
    /// Returns the number of collections that were invalidated.
    ///
    /// # Errors
    ///
    /// Returns an error if the smart-collection registry lock is poisoned.
    pub fn invalidate_smart_collections_for(&self, field: ClipField) -> ClipResult<usize> {
        let mut guard = self.lock_smart_collections()?;
        let mut count = 0;
        for smart_collection in guard.iter_mut() {
            if smart_collection.dependency_fields().contains(&field) {
                smart_collection.invalidate_cache();
                count += 1;
            }
        }
        Ok(count)
    }

    /// Invalidates the caches of all registered smart collections.
    ///
    /// # Errors
    ///
    /// Returns an error if the smart-collection registry lock is poisoned.
    pub fn invalidate_all_smart_collections(&self) -> ClipResult<()> {
        let mut guard = self.lock_smart_collections()?;
        for smart_collection in guard.iter_mut() {
            smart_collection.invalidate_cache();
        }
        Ok(())
    }

    /// Returns the number of registered smart collections.
    ///
    /// # Errors
    ///
    /// Returns an error if the smart-collection registry lock is poisoned.
    pub fn smart_collection_count(&self) -> ClipResult<usize> {
        Ok(self.lock_smart_collections()?.len())
    }

    /// Provides read-only access to the registered smart collections via a
    /// closure, holding the registry lock for the duration of the call.
    ///
    /// This is the inspection counterpart to the cache-mutating methods; it
    /// avoids exposing the interior [`Mutex`] while letting callers query
    /// cache state (e.g. [`SmartCollection::needs_refresh`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the smart-collection registry lock is poisoned.
    pub fn with_smart_collections<F, R>(&self, f: F) -> ClipResult<R>
    where
        F: FnOnce(&[SmartCollection]) -> R,
    {
        let guard = self.lock_smart_collections()?;
        Ok(f(guard.as_slice()))
    }

    /// Computes the set of clip fields that differ between `old` and `new`.
    ///
    /// This is the basis for fine-grained smart-collection cache invalidation:
    /// only collections whose rules depend on a changed field need refreshing.
    fn changed_fields(old: &Clip, new: &Clip) -> HashSet<ClipField> {
        let mut changed = HashSet::new();
        if old.name != new.name {
            changed.insert(ClipField::Name);
        }
        if old.rating != new.rating {
            changed.insert(ClipField::Rating);
        }
        if old.is_favorite != new.is_favorite {
            changed.insert(ClipField::Favorite);
        }
        if old.is_rejected != new.is_rejected {
            changed.insert(ClipField::Rejected);
        }
        if old.keywords != new.keywords {
            changed.insert(ClipField::Keywords);
        }
        if old.file_path != new.file_path {
            changed.insert(ClipField::FilePath);
        }
        if old.duration != new.duration
            || old.in_point != new.in_point
            || old.out_point != new.out_point
        {
            // `effective_duration` (what the `Duration` rule matches on) is a
            // function of duration plus the in/out points.
            changed.insert(ClipField::Duration);
        }
        if old.created_at != new.created_at {
            changed.insert(ClipField::CreatedDate);
        }
        if old.modified_at != new.modified_at {
            changed.insert(ClipField::ModifiedDate);
        }
        if old.markers.len() != new.markers.len() {
            changed.insert(ClipField::Markers);
        }
        if old.custom_metadata != new.custom_metadata {
            changed.insert(ClipField::CustomMetadata);
        }
        changed
    }

    // Marker operations

    /// Adds a marker to a clip.
    pub fn add_marker(&mut self, clip_id: ClipId, marker: Marker) {
        self.marker_manager.add_marker(clip_id, marker);
    }

    /// Gets markers for a clip.
    #[must_use]
    pub fn get_markers(&self, clip_id: &ClipId) -> Vec<&Marker> {
        self.marker_manager.get_markers(clip_id)
    }

    /// Removes a marker.
    ///
    /// # Errors
    ///
    /// Returns an error if the marker is not found.
    pub fn remove_marker(&mut self, clip_id: &ClipId, marker_id: &MarkerId) -> ClipResult<()> {
        self.marker_manager.remove_marker(clip_id, marker_id)
    }

    // Take operations

    /// Adds a take.
    pub fn add_take(&mut self, take: Take) {
        self.take_manager.add_take(take);
    }

    /// Gets takes for a scene.
    #[must_use]
    pub fn get_scene_takes(&self, scene: &str) -> Vec<&Take> {
        self.take_manager.get_scene_takes(scene)
    }

    /// Gets takes for a clip.
    #[must_use]
    pub fn get_clip_takes(&self, clip_id: &ClipId) -> Vec<&Take> {
        self.take_manager.get_clip_takes(clip_id)
    }

    // Proxy operations

    /// Adds a proxy link.
    pub fn add_proxy(&mut self, link: ProxyLink) {
        self.proxy_manager.add_link(link);
    }

    /// Gets proxy links for a clip.
    #[must_use]
    pub fn get_proxies(&self, clip_id: &ClipId) -> Vec<&ProxyLink> {
        self.proxy_manager.get_links(clip_id)
    }

    /// Gets the best proxy for a clip.
    #[must_use]
    pub fn get_best_proxy(&self, clip_id: &ClipId) -> Option<&ProxyLink> {
        self.proxy_manager.get_best_proxy(clip_id)
    }

    // Import operations

    /// Scans a directory for media files.
    ///
    /// # Errors
    ///
    /// Returns an error if the scan fails.
    pub async fn scan_directory(&self, path: impl AsRef<Path>) -> ClipResult<Vec<Clip>> {
        let scanner = MediaScanner::new();
        scanner.scan(path).await
    }

    /// Imports clips from file paths.
    #[must_use]
    pub fn import_clips(&self, paths: Vec<PathBuf>) -> Vec<Clip> {
        let importer = BatchImporter::default();
        importer.import(paths)
    }

    // Export operations

    /// Exports clips to CSV.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub async fn export_csv(&self, clip_ids: &[ClipId]) -> ClipResult<String> {
        let mut clips = Vec::new();
        for clip_id in clip_ids {
            clips.push(self.database.get_clip(clip_id).await?);
        }

        let exporter = ClipListExporter::new();
        exporter.to_csv(&clips)
    }

    /// Exports clips to EDL.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub async fn export_edl(
        &self,
        clip_ids: &[ClipId],
        frame_rate: Rational,
    ) -> ClipResult<String> {
        let mut clips = Vec::new();
        for clip_id in clip_ids {
            clips.push(self.database.get_clip(clip_id).await?);
        }

        let exporter = EdlExporter::new(frame_rate);
        exporter.to_edl(&clips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clip_manager() {
        let manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");
        let count = manager
            .clip_count()
            .await
            .expect("clip_count should succeed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_add_and_get_clip() {
        let manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");

        let clip = Clip::new(PathBuf::from("/test.mov"));
        let clip_id = manager
            .add_clip(clip.clone())
            .await
            .expect("add_clip should succeed");

        let loaded = manager
            .get_clip(&clip_id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(loaded.id, clip_id);
    }

    #[tokio::test]
    async fn test_bins() {
        let mut manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");

        let bin_id = manager.create_bin("Test Bin");
        let bin = manager.get_bin(&bin_id).expect("get_bin should succeed");
        assert_eq!(bin.name, "Test Bin");

        let clip = Clip::new(PathBuf::from("/test.mov"));
        let clip_id = manager
            .add_clip(clip)
            .await
            .expect("add_clip should succeed");

        manager
            .add_clip_to_bin(&bin_id, clip_id)
            .expect("add_clip_to_bin should succeed");
        let bin = manager.get_bin(&bin_id).expect("get_bin should succeed");
        assert_eq!(bin.count(), 1);
    }

    #[tokio::test]
    async fn test_search() {
        let manager = ClipManager::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Interview");
        manager
            .add_clip(clip)
            .await
            .expect("operation should succeed");

        let results = manager
            .search("interview")
            .await
            .expect("search should succeed");
        assert_eq!(results.len(), 1);
    }
}
