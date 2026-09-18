//! Export package creation for media asset delivery
//!
//! Provides structures and utilities for creating delivery packages from
//! MAM assets, targeting broadcast, OTT, archive, festival and social
//! delivery destinations.

#![allow(dead_code)]

/// The type of delivery package to create
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExportPackageType {
    /// Full broadcast delivery package (MXF, captions, etc.)
    Broadcast,
    /// OTT / streaming service package (adaptive bitrate)
    OTT,
    /// Long-term archival package (lossless, full metadata)
    Archive,
    /// Film festival delivery package
    Festival,
    /// Social media delivery package (short clips, thumbnails)
    Social,
}

impl ExportPackageType {
    /// Returns `true` if this package type includes a proxy video component
    #[must_use]
    pub fn includes_proxy(&self) -> bool {
        matches!(self, Self::OTT | Self::Social)
    }

    /// Human-readable label
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Broadcast => "broadcast",
            Self::OTT => "ott",
            Self::Archive => "archive",
            Self::Festival => "festival",
            Self::Social => "social",
        }
    }
}

/// Types of components that can be included in a package
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComponentType {
    /// Full-resolution master video file
    MasterVideo,
    /// Low-resolution proxy video file
    ProxyVideo,
    /// Audio track file
    Audio,
    /// Subtitle / caption file
    Subtitle,
    /// Thumbnail image
    Thumbnail,
    /// Metadata XML / JSON sidecar
    Metadata,
    /// Arbitrary sidecar file (EDL, LUT, etc.)
    Sidecar,
}

impl ComponentType {
    /// Human-readable label for the component type
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::MasterVideo => "master_video",
            Self::ProxyVideo => "proxy_video",
            Self::Audio => "audio",
            Self::Subtitle => "subtitle",
            Self::Thumbnail => "thumbnail",
            Self::Metadata => "metadata",
            Self::Sidecar => "sidecar",
        }
    }
}

/// A single file component within an export package
#[derive(Debug, Clone)]
pub struct PackageComponent {
    /// Type of this component
    pub component_type: ComponentType,
    /// Path to the component file
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
}

impl PackageComponent {
    /// Create a new package component
    #[must_use]
    pub fn new(component_type: ComponentType, path: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            component_type,
            path: path.into(),
            size_bytes,
        }
    }
}

/// An export package grouping one or more delivery components
#[derive(Debug, Clone)]
pub struct ExportPackage {
    /// Unique package ID
    pub id: String,
    /// ID of the source asset in the MAM
    pub asset_id: String,
    /// Delivery type for this package
    pub package_type: ExportPackageType,
    /// All files included in this package
    pub components: Vec<PackageComponent>,
    /// Manifest string (JSON-like) describing the package
    pub manifest: String,
}

impl ExportPackage {
    /// Create a new (empty) export package
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        package_type: ExportPackageType,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            package_type,
            components: Vec::new(),
            manifest: String::new(),
        }
    }

    /// Add a component to the package
    pub fn add_component(&mut self, component: PackageComponent) {
        self.components.push(component);
    }

    /// Return the total size of all components in bytes
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.components.iter().map(|c| c.size_bytes).sum()
    }

    /// Return all components of a given type
    #[must_use]
    pub fn components_of_type(&self, ct: &ComponentType) -> Vec<&PackageComponent> {
        self.components
            .iter()
            .filter(|c| &c.component_type == ct)
            .collect()
    }

    /// Return `true` if the package contains a proxy video
    #[must_use]
    pub fn has_proxy(&self) -> bool {
        self.components
            .iter()
            .any(|c| c.component_type == ComponentType::ProxyVideo)
    }
}

/// Generates manifest strings for export packages
pub struct PackageManifest;

impl PackageManifest {
    /// Generate a JSON-like manifest string for a package.
    ///
    /// The manifest lists the package ID, asset ID, type, total size,
    /// and all component paths.
    #[must_use]
    pub fn generate(package: &ExportPackage) -> String {
        let components_json: Vec<String> = package
            .components
            .iter()
            .map(|c| {
                format!(
                    r#"{{"type":"{}","path":"{}","size":{}}}"#,
                    c.component_type.label(),
                    c.path,
                    c.size_bytes
                )
            })
            .collect();

        format!(
            r#"{{"id":"{}","asset_id":"{}","package_type":"{}","total_size":{},"components":[{}]}}"#,
            package.id,
            package.asset_id,
            package.package_type.label(),
            package.total_size(),
            components_json.join(","),
        )
    }
}

/// Type of delivery destination
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DestType {
    /// Local disk or mounted volume
    LocalDisk,
    /// Network share (NFS/SMB)
    Network,
    /// Cloud object storage (S3, GCS, Azure)
    Cloud,
    /// FTP server
    FTP,
    /// SFTP server
    SFTP,
    /// Aspera high-speed transfer
    ASPERA,
}

impl DestType {
    /// Human-readable label
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::LocalDisk => "local_disk",
            Self::Network => "network",
            Self::Cloud => "cloud",
            Self::FTP => "ftp",
            Self::SFTP => "sftp",
            Self::ASPERA => "aspera",
        }
    }
}

/// A destination that an export package can be delivered to
#[derive(Debug, Clone)]
pub struct DeliveryDestination {
    /// Human-readable name for this destination
    pub name: String,
    /// Type of destination
    pub dest_type: DestType,
    /// Path or URL at the destination
    pub path: String,
}

impl DeliveryDestination {
    /// Create a new delivery destination
    #[must_use]
    pub fn new(name: impl Into<String>, dest_type: DestType, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dest_type,
            path: path.into(),
        }
    }

    /// Return `true` if this destination uses an encrypted transport
    #[must_use]
    pub fn is_secure(&self) -> bool {
        matches!(
            self.dest_type,
            DestType::Cloud | DestType::SFTP | DestType::ASPERA
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_package() -> ExportPackage {
        let mut pkg = ExportPackage::new("pkg-001", "asset-abc", ExportPackageType::Broadcast);
        pkg.add_component(PackageComponent::new(
            ComponentType::MasterVideo,
            "/out/master.mxf",
            5_000_000,
        ));
        pkg.add_component(PackageComponent::new(
            ComponentType::Audio,
            "/out/audio.wav",
            500_000,
        ));
        pkg.add_component(PackageComponent::new(
            ComponentType::Subtitle,
            "/out/subs.srt",
            10_000,
        ));
        pkg.add_component(PackageComponent::new(
            ComponentType::Thumbnail,
            "/out/thumb.jpg",
            50_000,
        ));
        pkg
    }

    #[test]
    fn test_export_package_type_includes_proxy() {
        assert!(ExportPackageType::OTT.includes_proxy());
        assert!(ExportPackageType::Social.includes_proxy());
        assert!(!ExportPackageType::Broadcast.includes_proxy());
        assert!(!ExportPackageType::Archive.includes_proxy());
        assert!(!ExportPackageType::Festival.includes_proxy());
    }

    #[test]
    fn test_export_package_type_labels() {
        assert_eq!(ExportPackageType::Broadcast.label(), "broadcast");
        assert_eq!(ExportPackageType::OTT.label(), "ott");
        assert_eq!(ExportPackageType::Archive.label(), "archive");
        assert_eq!(ExportPackageType::Festival.label(), "festival");
        assert_eq!(ExportPackageType::Social.label(), "social");
    }

    #[test]
    fn test_component_type_labels() {
        assert_eq!(ComponentType::MasterVideo.label(), "master_video");
        assert_eq!(ComponentType::ProxyVideo.label(), "proxy_video");
        assert_eq!(ComponentType::Audio.label(), "audio");
        assert_eq!(ComponentType::Subtitle.label(), "subtitle");
        assert_eq!(ComponentType::Thumbnail.label(), "thumbnail");
        assert_eq!(ComponentType::Metadata.label(), "metadata");
        assert_eq!(ComponentType::Sidecar.label(), "sidecar");
    }

    #[test]
    fn test_package_total_size() {
        let pkg = make_package();
        assert_eq!(pkg.total_size(), 5_560_000);
    }

    #[test]
    fn test_package_components_of_type() {
        let pkg = make_package();
        let audio = pkg.components_of_type(&ComponentType::Audio);
        assert_eq!(audio.len(), 1);
        assert_eq!(audio[0].path, "/out/audio.wav");
    }

    #[test]
    fn test_package_has_proxy() {
        let pkg = make_package();
        assert!(!pkg.has_proxy());

        let mut pkg2 = ExportPackage::new("p2", "asset-xyz", ExportPackageType::OTT);
        pkg2.add_component(PackageComponent::new(
            ComponentType::ProxyVideo,
            "/out/proxy.mp4",
            100_000,
        ));
        assert!(pkg2.has_proxy());
    }

    #[test]
    fn test_package_manifest_generate() {
        let pkg = make_package();
        let manifest = PackageManifest::generate(&pkg);
        assert!(manifest.contains("pkg-001"));
        assert!(manifest.contains("asset-abc"));
        assert!(manifest.contains("broadcast"));
        assert!(manifest.contains("master_video"));
        assert!(manifest.contains("5560000"));
    }

    #[test]
    fn test_dest_type_labels() {
        assert_eq!(DestType::LocalDisk.label(), "local_disk");
        assert_eq!(DestType::Network.label(), "network");
        assert_eq!(DestType::Cloud.label(), "cloud");
        assert_eq!(DestType::FTP.label(), "ftp");
        assert_eq!(DestType::SFTP.label(), "sftp");
        assert_eq!(DestType::ASPERA.label(), "aspera");
    }

    #[test]
    fn test_delivery_destination_is_secure() {
        let secure = DeliveryDestination::new("cloud", DestType::Cloud, "s3://bucket/path");
        assert!(secure.is_secure());

        let insecure = DeliveryDestination::new("ftp", DestType::FTP, "ftp://server/path");
        assert!(!insecure.is_secure());
    }

    #[test]
    fn test_package_manifest_empty_components() {
        let pkg = ExportPackage::new("p-empty", "a-empty", ExportPackageType::Archive);
        let manifest = PackageManifest::generate(&pkg);
        assert!(manifest.contains("p-empty"));
        assert!(manifest.contains("archive"));
        assert!(manifest.contains("\"components\":[]"));
    }

    #[test]
    fn test_package_add_multiple_components() {
        let mut pkg = ExportPackage::new("p", "a", ExportPackageType::Festival);
        for i in 0..5 {
            pkg.add_component(PackageComponent::new(
                ComponentType::Sidecar,
                format!("/out/sidecar_{i}.xml"),
                1_000,
            ));
        }
        assert_eq!(pkg.components.len(), 5);
        assert_eq!(pkg.total_size(), 5_000);
    }
}
