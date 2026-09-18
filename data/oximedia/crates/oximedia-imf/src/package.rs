//! IMF Package creation and management
//!
//! This module provides structures and builders for creating and managing
//! complete IMF packages (IMPs).

use crate::{Asset as AssetMapAsset, Chunk};
use crate::{
    AssetMap, CompositionPlaylist, EditRate, HashAlgorithm, ImfError, ImfResult, MxfEssence,
    OutputProfileList, PackingList, PklAsset,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Returns true if `asset` is explicitly identifiable as a Composition
/// Playlist rather than merely XML-shaped: either via the `asdcpKind=CPL`
/// PKL `Type` convention (e.g. `text/xml;asdcpKind=CPL`), or via this
/// crate's own `CPL_<uuid>.xml` filename convention (see
/// [`ImfPackageBuilder::build`]).
fn is_cpl_asset(asset: &PklAsset) -> bool {
    if asset.asset_type().to_ascii_lowercase().contains("cpl") {
        return true;
    }
    asset
        .original_filename()
        .is_some_and(|name| name.starts_with("CPL_"))
}

/// IMF Package (IMP)
///
/// Represents a complete IMF package including CPL, PKL, ASSETMAP, and essence files.
#[derive(Debug)]
pub struct ImfPackage {
    root_path: PathBuf,
    asset_map: AssetMap,
    packing_lists: Vec<PackingList>,
    composition_playlists: Vec<CompositionPlaylist>,
    output_profile_lists: Vec<OutputProfileList>,
    essence_files: HashMap<Uuid, MxfEssence>,
}

impl ImfPackage {
    /// Open an existing IMF package from a directory
    pub fn open<P: AsRef<Path>>(path: P) -> ImfResult<Self> {
        let root_path = path.as_ref().to_path_buf();

        if !root_path.exists() {
            return Err(ImfError::FileNotFound(
                root_path.to_string_lossy().to_string(),
            ));
        }

        // Load ASSETMAP
        let assetmap_path = root_path.join("ASSETMAP.xml");
        if !assetmap_path.exists() {
            return Err(ImfError::FileNotFound("ASSETMAP.xml".to_string()));
        }

        let asset_map = {
            let file = fs::File::open(&assetmap_path)?;
            let reader = std::io::BufReader::new(file);
            AssetMap::from_xml(reader)?
        };

        // Load PKLs
        let mut packing_lists = Vec::new();
        for asset in asset_map.packing_lists() {
            if let Some(path) = asset.primary_path() {
                let pkl_path = root_path.join(path);
                if pkl_path.exists() {
                    let file = fs::File::open(&pkl_path)?;
                    let reader = std::io::BufReader::new(file);
                    let pkl = PackingList::from_xml(reader)?;
                    packing_lists.push(pkl);
                }
            }
        }

        // Load CPLs. The ASSETMAP (SMPTE ST 429-9) does not carry per-asset
        // MIME types -- it only distinguishes "is this asset the PKL". Asset
        // *types* live on the PKL's own Asset entries (SMPTE ST 429-8), so
        // the PKL is the source of truth for "is this an XML asset" while
        // the ASSETMAP resolves an asset ID to a file path.
        let composition_playlists =
            Self::load_composition_playlists(&root_path, &asset_map, &packing_lists)?;

        // Create package
        Ok(Self {
            root_path,
            asset_map,
            packing_lists,
            composition_playlists,
            output_profile_lists: Vec::new(),
            essence_files: HashMap::new(),
        })
    }

    /// Scans every loaded [`PackingList`]'s asset entries for XML-typed
    /// assets, resolves each one's file path via the [`AssetMap`], and
    /// attempts to parse it as a [`CompositionPlaylist`].
    ///
    /// An asset is treated as *authoritatively* a CPL — meaning a resolution
    /// or parse failure becomes a hard [`ImfError`] instead of being skipped
    /// — when the PKL marks it with the `asdcpKind=CPL` convention (e.g.
    /// `text/xml;asdcpKind=CPL`) commonly used by IMF tooling, or when its
    /// filename matches this crate's own `CPL_<uuid>.xml` naming convention
    /// (see [`ImfPackageBuilder::build`]). Other XML-typed assets (e.g. an
    /// OPL, or a sidecar with a generic `text/xml`/`application/xml` type)
    /// that simply fail to parse as a CPL are skipped rather than treated as
    /// package corruption, since not every XML asset in a package is a CPL.
    fn load_composition_playlists(
        root_path: &Path,
        asset_map: &AssetMap,
        packing_lists: &[PackingList],
    ) -> ImfResult<Vec<CompositionPlaylist>> {
        let mut composition_playlists = Vec::new();

        for pkl in packing_lists {
            for asset in pkl.assets() {
                if !asset.asset_type().to_ascii_lowercase().contains("xml") {
                    continue;
                }

                let explicitly_cpl = is_cpl_asset(asset);

                let Some(am_asset) = asset_map.find_asset(asset.id()) else {
                    if explicitly_cpl {
                        return Err(ImfError::AssetNotFound(asset.id().to_string()));
                    }
                    continue;
                };
                let Some(rel_path) = am_asset.primary_path() else {
                    if explicitly_cpl {
                        return Err(ImfError::InvalidStructure(format!(
                            "CPL asset {} has no chunk path in ASSETMAP",
                            asset.id()
                        )));
                    }
                    continue;
                };
                let cpl_path = root_path.join(rel_path);

                let file = match fs::File::open(&cpl_path) {
                    Ok(f) => f,
                    Err(e) => {
                        if explicitly_cpl {
                            return Err(ImfError::from(e));
                        }
                        continue;
                    }
                };

                match CompositionPlaylist::from_xml(std::io::BufReader::new(file)) {
                    Ok(cpl) => composition_playlists.push(cpl),
                    Err(e) => {
                        if explicitly_cpl {
                            return Err(e);
                        }
                    }
                }
            }
        }

        Ok(composition_playlists)
    }

    /// Get the root path
    pub fn root_path(&self) -> &Path {
        &self.root_path
    }

    /// Get the asset map
    pub fn asset_map(&self) -> &AssetMap {
        &self.asset_map
    }

    /// Get packing lists
    pub fn packing_lists(&self) -> &[PackingList] {
        &self.packing_lists
    }

    /// Get the primary packing list
    pub fn primary_packing_list(&self) -> Option<&PackingList> {
        self.packing_lists.first()
    }

    /// Get composition playlists
    pub fn composition_playlists(&self) -> &[CompositionPlaylist] {
        &self.composition_playlists
    }

    /// Get the primary CPL
    pub fn primary_cpl(&self) -> Option<&CompositionPlaylist> {
        self.composition_playlists.first()
    }

    /// Get output profile lists
    pub fn output_profile_lists(&self) -> &[OutputProfileList] {
        &self.output_profile_lists
    }

    /// Get essence files
    pub fn essence_files(&self) -> &HashMap<Uuid, MxfEssence> {
        &self.essence_files
    }

    /// Get essence file by ID
    pub fn get_essence(&self, id: Uuid) -> Option<&MxfEssence> {
        self.essence_files.get(&id)
    }

    /// Add a composition playlist
    pub fn add_composition_playlist(&mut self, cpl: CompositionPlaylist) {
        self.composition_playlists.push(cpl);
    }

    /// Add an output profile list
    pub fn add_output_profile_list(&mut self, opl: OutputProfileList) {
        self.output_profile_lists.push(opl);
    }

    /// Add an essence file
    pub fn add_essence(&mut self, essence: MxfEssence) {
        self.essence_files.insert(essence.file_id(), essence);
    }

    /// Validate the package
    pub fn validate(&self) -> ImfResult<Vec<String>> {
        let mut warnings = Vec::new();

        // Check for required files
        if self.packing_lists.is_empty() {
            warnings.push("No packing lists found".to_string());
        }

        if self.composition_playlists.is_empty() {
            warnings.push("No composition playlists found".to_string());
        }

        // Validate PKL/ASSETMAP consistency
        for pkl in &self.packing_lists {
            for asset in pkl.assets() {
                if self.asset_map.find_asset(asset.id()).is_none() {
                    warnings.push(format!("Asset {} in PKL not found in ASSETMAP", asset.id()));
                }
            }
        }

        Ok(warnings)
    }

    /// Get package info as string
    pub fn info_string(&self) -> String {
        let mut info = String::new();

        info.push_str(&format!("Package root: {}\n", self.root_path.display()));
        info.push_str(&format!("Packing lists: {}\n", self.packing_lists.len()));
        info.push_str(&format!(
            "Composition playlists: {}\n",
            self.composition_playlists.len()
        ));
        info.push_str(&format!("Essence files: {}\n", self.essence_files.len()));

        if let Some(pkl) = self.primary_packing_list() {
            info.push_str(&format!("Total assets: {}\n", pkl.assets().len()));
            info.push_str(&format!("Total size: {} bytes\n", pkl.total_size()));
        }

        if let Some(cpl) = self.primary_cpl() {
            info.push_str(&format!("Title: {}\n", cpl.content_title()));
            info.push_str(&format!("Duration: {} frames\n", cpl.total_duration()));
        }

        info
    }
}

/// Supplemental package for updates
#[derive(Debug)]
pub struct SupplementalPackage {
    base_package: ImfPackage,
    supplemental_assets: Vec<Uuid>,
}

impl SupplementalPackage {
    /// Create a new supplemental package
    pub fn new(base_package: ImfPackage) -> Self {
        Self {
            base_package,
            supplemental_assets: Vec::new(),
        }
    }

    /// Get the base package
    pub fn base_package(&self) -> &ImfPackage {
        &self.base_package
    }

    /// Get supplemental assets
    pub fn supplemental_assets(&self) -> &[Uuid] {
        &self.supplemental_assets
    }

    /// Add a supplemental asset
    pub fn add_supplemental_asset(&mut self, asset_id: Uuid) {
        self.supplemental_assets.push(asset_id);
    }

    /// Check if an asset is supplemental
    pub fn is_supplemental(&self, asset_id: Uuid) -> bool {
        self.supplemental_assets.contains(&asset_id)
    }
}

/// Builder for creating IMF packages
pub struct ImfPackageBuilder {
    root_path: PathBuf,
    title: String,
    creator: Option<String>,
    issuer: Option<String>,
    edit_rate: EditRate,
    assets: Vec<(PathBuf, String)>, // (path, type)
    hash_algorithm: HashAlgorithm,
}

impl ImfPackageBuilder {
    /// Create a new package builder
    pub fn new<P: AsRef<Path>>(root_path: P) -> Self {
        Self {
            root_path: root_path.as_ref().to_path_buf(),
            title: "Untitled".to_string(),
            creator: None,
            issuer: None,
            edit_rate: EditRate::fps_24(),
            assets: Vec::new(),
            hash_algorithm: HashAlgorithm::Sha1,
        }
    }

    /// Set title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Set creator
    pub fn with_creator(mut self, creator: String) -> Self {
        self.creator = Some(creator);
        self
    }

    /// Set issuer
    pub fn with_issuer(mut self, issuer: String) -> Self {
        self.issuer = Some(issuer);
        self
    }

    /// Set edit rate
    pub fn with_edit_rate(mut self, edit_rate: EditRate) -> Self {
        self.edit_rate = edit_rate;
        self
    }

    /// Set hash algorithm
    pub fn with_hash_algorithm(mut self, algorithm: HashAlgorithm) -> Self {
        self.hash_algorithm = algorithm;
        self
    }

    /// Add a video track
    pub fn add_video_track<P: AsRef<Path>>(mut self, path: P) -> ImfResult<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(ImfError::FileNotFound(path.to_string_lossy().to_string()));
        }
        self.assets.push((path, "video/mxf".to_string()));
        Ok(self)
    }

    /// Add an audio track
    pub fn add_audio_track<P: AsRef<Path>>(mut self, path: P) -> ImfResult<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(ImfError::FileNotFound(path.to_string_lossy().to_string()));
        }
        self.assets.push((path, "audio/mxf".to_string()));
        Ok(self)
    }

    /// Add a subtitle track
    pub fn add_subtitle_track<P: AsRef<Path>>(mut self, path: P) -> ImfResult<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(ImfError::FileNotFound(path.to_string_lossy().to_string()));
        }
        self.assets.push((path, "subtitle/xml".to_string()));
        Ok(self)
    }

    /// Build the package
    #[allow(clippy::too_many_lines)]
    pub fn build(self) -> ImfResult<ImfPackage> {
        // Create root directory if it doesn't exist
        fs::create_dir_all(&self.root_path)?;

        // Generate IDs
        let asset_map_id = Uuid::new_v4();
        let pkl_id = Uuid::new_v4();
        let cpl_id = Uuid::new_v4();

        // Create AssetMap
        let mut asset_map = AssetMap::new(asset_map_id);
        if let Some(ref creator) = self.creator {
            asset_map.set_creator(creator.clone());
        }
        if let Some(ref issuer) = self.issuer {
            asset_map.set_issuer(issuer.clone());
        }

        // Create PackingList
        let mut pkl = PackingList::new(pkl_id);
        if let Some(ref creator) = self.creator {
            pkl.set_creator(creator.clone());
        }
        if let Some(ref issuer) = self.issuer {
            pkl.set_issuer(issuer.clone());
        }

        // Create CPL
        let mut cpl = CompositionPlaylist::new(cpl_id, self.title.clone(), self.edit_rate);
        if let Some(ref creator) = self.creator {
            cpl.set_creator(creator.clone());
        }
        if let Some(ref issuer) = self.issuer {
            cpl.set_issuer(issuer.clone());
        }

        // Process assets
        for (asset_path, asset_type) in &self.assets {
            let asset_id = Uuid::new_v4();

            // Copy file to package
            let filename = asset_path
                .file_name()
                .ok_or_else(|| ImfError::InvalidStructure("Invalid filename".to_string()))?;
            let dest_path = self.root_path.join(filename);

            // Copy if not already in destination
            if asset_path != &dest_path {
                fs::copy(asset_path, &dest_path)?;
            }

            // Create PKL asset
            let pkl_asset = PklAsset::from_file(
                asset_id,
                &dest_path,
                self.hash_algorithm,
                asset_type.clone(),
            )?;
            pkl.add_asset(pkl_asset);

            // Create AssetMap asset
            let mut am_asset = AssetMapAsset::new(asset_id, false);
            am_asset.add_chunk(Chunk::new(PathBuf::from(filename)));
            asset_map.add_asset(am_asset);
        }

        // Add PKL to AssetMap
        let pkl_filename = format!("PKL_{pkl_id}.xml");
        let pkl_path = self.root_path.join(&pkl_filename);

        // Write PKL
        {
            let file = fs::File::create(&pkl_path)?;
            let writer = std::io::BufWriter::new(file);
            pkl.to_xml(writer)?;
        }

        // Add PKL asset to AssetMap
        let pkl_asset_id = pkl_id;
        let mut pkl_am_asset = AssetMapAsset::new(pkl_asset_id, true);
        pkl_am_asset.add_chunk(Chunk::new(PathBuf::from(&pkl_filename)));
        asset_map.add_asset(pkl_am_asset);

        // Add CPL to PKL and AssetMap
        let cpl_filename = format!("CPL_{cpl_id}.xml");
        let cpl_path = self.root_path.join(&cpl_filename);

        // Write CPL
        {
            let file = fs::File::create(&cpl_path)?;
            let writer = std::io::BufWriter::new(file);
            cpl.to_xml(writer)?;
        }

        // Add CPL to PKL
        let cpl_pkl_asset = PklAsset::from_file(
            cpl_id,
            &cpl_path,
            self.hash_algorithm,
            "application/xml".to_string(),
        )?;
        pkl.add_asset(cpl_pkl_asset);

        // Re-write PKL with CPL included
        {
            let file = fs::File::create(&pkl_path)?;
            let writer = std::io::BufWriter::new(file);
            pkl.to_xml(writer)?;
        }

        // Add CPL to AssetMap
        let mut cpl_am_asset = AssetMapAsset::new(cpl_id, false);
        cpl_am_asset.add_chunk(Chunk::new(PathBuf::from(cpl_filename)));
        asset_map.add_asset(cpl_am_asset);

        // Write AssetMap
        let assetmap_path = self.root_path.join("ASSETMAP.xml");
        {
            let file = fs::File::create(&assetmap_path)?;
            let writer = std::io::BufWriter::new(file);
            asset_map.to_xml(writer)?;
        }

        // Create package
        let package = ImfPackage {
            root_path: self.root_path,
            asset_map,
            packing_lists: vec![pkl],
            composition_playlists: vec![cpl],
            output_profile_lists: Vec::new(),
            essence_files: HashMap::new(),
        };

        Ok(package)
    }
}

/// Package version tracker for managing IMF package versions
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PackageVersion {
    version_number: u32,
    previous_version: Option<Uuid>,
    changes: Vec<String>,
}

#[allow(dead_code)]
impl PackageVersion {
    /// Create a new package version
    pub fn new(version_number: u32) -> Self {
        Self {
            version_number,
            previous_version: None,
            changes: Vec::new(),
        }
    }

    /// Get version number
    pub fn version_number(&self) -> u32 {
        self.version_number
    }

    /// Get previous version ID
    pub fn previous_version(&self) -> Option<Uuid> {
        self.previous_version
    }

    /// Set previous version
    pub fn set_previous_version(&mut self, id: Uuid) {
        self.previous_version = Some(id);
    }

    /// Get changes
    pub fn changes(&self) -> &[String] {
        &self.changes
    }

    /// Add a change
    pub fn add_change(&mut self, change: String) {
        self.changes.push(change);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_version() {
        let mut version = PackageVersion::new(1);
        version.add_change("Initial release".to_string());
        version.add_change("Added audio track".to_string());

        assert_eq!(version.version_number(), 1);
        assert_eq!(version.changes().len(), 2);
        assert!(version.previous_version().is_none());
    }

    #[test]
    fn test_package_version_chain() {
        let mut v1 = PackageVersion::new(1);
        v1.add_change("Initial release".to_string());

        let prev_id = Uuid::new_v4();
        let mut v2 = PackageVersion::new(2);
        v2.set_previous_version(prev_id);
        v2.add_change("Updated video".to_string());

        assert_eq!(v2.version_number(), 2);
        assert_eq!(v2.previous_version(), Some(prev_id));
    }

    #[test]
    fn test_open_reads_back_composition_playlist() {
        // Real write-then-read-back round trip: build a package on disk with
        // ImfPackageBuilder (which writes a real CPL_<uuid>.xml via
        // CompositionPlaylist::to_xml), then reopen the SAME directory with
        // ImfPackage::open and confirm the CPL survives the round trip
        // instead of coming back as an empty Vec (the bug being fixed here).
        let root = std::env::temp_dir().join(format!(
            "oximedia-imf-package-open-cpl-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);

        let builder = ImfPackageBuilder::new(&root)
            .with_title("Round Trip Title".to_string())
            .with_creator("OxiMedia Test".to_string());
        let built = builder.build().expect("build should succeed");
        assert_eq!(
            built.composition_playlists().len(),
            1,
            "builder must return the CPL it just created"
        );

        let reopened = ImfPackage::open(&root).expect("open should succeed");
        assert_eq!(
            reopened.composition_playlists().len(),
            1,
            "ImfPackage::open must populate composition_playlists from the on-disk package, not return an empty Vec"
        );
        assert_eq!(
            reopened
                .primary_cpl()
                .expect("primary_cpl must be Some after a successful open")
                .content_title(),
            "Round Trip Title"
        );
        assert_eq!(
            reopened.primary_cpl().expect("primary cpl").id(),
            built.primary_cpl().expect("built cpl").id(),
            "reopened CPL must be the SAME composition (same UUID), not a fabricated stand-in"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_open_errors_on_malformed_explicit_cpl_asset() {
        // A PKL asset that is unambiguously a CPL (matches this crate's own
        // CPL_<uuid>.xml naming convention) but whose file contents are not
        // valid XML must fail ImfPackage::open with a real Err, never a
        // silent empty-Ok composition_playlists list.
        let root = std::env::temp_dir().join(format!(
            "oximedia-imf-package-open-malformed-cpl-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create root dir should succeed");

        let cpl_id = Uuid::new_v4();
        let pkl_id = Uuid::new_v4();

        let cpl_filename = format!("CPL_{cpl_id}.xml");
        fs::write(root.join(&cpl_filename), b"not valid xml at all")
            .expect("writing the corrupt CPL stand-in should succeed");

        let mut pkl = PackingList::new(pkl_id);
        let cpl_pkl_asset = PklAsset::from_file(
            cpl_id,
            &root.join(&cpl_filename),
            HashAlgorithm::Sha1,
            "application/xml".to_string(),
        )
        .expect("PklAsset::from_file should succeed even for non-XML bytes");
        pkl.add_asset(cpl_pkl_asset);

        let pkl_filename = format!("PKL_{pkl_id}.xml");
        {
            let file =
                fs::File::create(root.join(&pkl_filename)).expect("create PKL file should succeed");
            pkl.to_xml(std::io::BufWriter::new(file))
                .expect("writing PKL XML should succeed");
        }

        let mut asset_map = AssetMap::new(Uuid::new_v4());
        let mut pkl_am_asset = AssetMapAsset::new(pkl_id, true);
        pkl_am_asset.add_chunk(Chunk::new(PathBuf::from(&pkl_filename)));
        asset_map.add_asset(pkl_am_asset);
        let mut cpl_am_asset = AssetMapAsset::new(cpl_id, false);
        cpl_am_asset.add_chunk(Chunk::new(PathBuf::from(&cpl_filename)));
        asset_map.add_asset(cpl_am_asset);

        {
            let file = fs::File::create(root.join("ASSETMAP.xml"))
                .expect("create ASSETMAP file should succeed");
            asset_map
                .to_xml(std::io::BufWriter::new(file))
                .expect("writing ASSETMAP XML should succeed");
        }

        let result = ImfPackage::open(&root);
        assert!(
            result.is_err(),
            "opening a package whose explicitly-named CPL asset is malformed XML must be a hard error, not an empty-Ok composition_playlists"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_open_ignores_unrelated_xml_asset_that_is_not_a_cpl() {
        // A generically-typed XML asset that is NOT a CPL (no asdcpKind=CPL,
        // no CPL_ filename) and fails to parse as one must be silently
        // skipped rather than failing the whole package open.
        let root = std::env::temp_dir().join(format!(
            "oximedia-imf-package-open-nonfatal-xml-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create root dir should succeed");

        let sidecar_id = Uuid::new_v4();
        let pkl_id = Uuid::new_v4();

        let sidecar_filename = "SIDECAR_NOTES.xml".to_string();
        fs::write(
            root.join(&sidecar_filename),
            b"<Notes>not a composition playlist</Notes>",
        )
        .expect("writing the sidecar XML should succeed");

        let mut pkl = PackingList::new(pkl_id);
        let sidecar_pkl_asset = PklAsset::from_file(
            sidecar_id,
            &root.join(&sidecar_filename),
            HashAlgorithm::Sha1,
            "text/xml".to_string(),
        )
        .expect("PklAsset::from_file should succeed");
        pkl.add_asset(sidecar_pkl_asset);

        let pkl_filename = format!("PKL_{pkl_id}.xml");
        {
            let file =
                fs::File::create(root.join(&pkl_filename)).expect("create PKL file should succeed");
            pkl.to_xml(std::io::BufWriter::new(file))
                .expect("writing PKL XML should succeed");
        }

        let mut asset_map = AssetMap::new(Uuid::new_v4());
        let mut pkl_am_asset = AssetMapAsset::new(pkl_id, true);
        pkl_am_asset.add_chunk(Chunk::new(PathBuf::from(&pkl_filename)));
        asset_map.add_asset(pkl_am_asset);
        let mut sidecar_am_asset = AssetMapAsset::new(sidecar_id, false);
        sidecar_am_asset.add_chunk(Chunk::new(PathBuf::from(&sidecar_filename)));
        asset_map.add_asset(sidecar_am_asset);

        {
            let file = fs::File::create(root.join("ASSETMAP.xml"))
                .expect("create ASSETMAP file should succeed");
            asset_map
                .to_xml(std::io::BufWriter::new(file))
                .expect("writing ASSETMAP XML should succeed");
        }

        let reopened = ImfPackage::open(&root)
            .expect("a non-CPL XML sidecar that fails to parse must not fail the whole open");
        assert!(
            reopened.composition_playlists().is_empty(),
            "the sidecar is not a CPL, so no composition playlists should be populated"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_supplemental_package() {
        // Create a dummy base package (would use builder in real code)
        let root = std::env::temp_dir().join("oximedia-imf-package-test_imp");
        let asset_map = AssetMap::new(Uuid::new_v4());
        let base_package = ImfPackage {
            root_path: root,
            asset_map,
            packing_lists: Vec::new(),
            composition_playlists: Vec::new(),
            output_profile_lists: Vec::new(),
            essence_files: HashMap::new(),
        };

        let mut supp = SupplementalPackage::new(base_package);
        let asset_id = Uuid::new_v4();
        supp.add_supplemental_asset(asset_id);

        assert!(supp.is_supplemental(asset_id));
        assert!(!supp.is_supplemental(Uuid::new_v4()));
    }
}
