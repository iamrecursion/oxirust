//! IMF sidecar asset management.
//!
//! Sidecar assets are supplementary files associated with an IMF package,
//! such as subtitles, closed captions, audio descriptions, and immersive
//! audio tracks. This module provides structures for declaring and managing
//! such assets.

#![allow(dead_code)]

/// The type of a sidecar asset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SidecarType {
    /// Subtitle track (e.g., SRT, TTML).
    Subtitle,
    /// Closed caption track.
    ClosedCaption,
    /// Audio description track for visually impaired viewers.
    AudioDescription,
    /// Visually impaired narrative track.
    VisuallyImpairedNarrative,
    /// Hearing impaired subtitle/caption track.
    HearingImpaired,
    /// Dolby Atmos immersive audio.
    DolbyAtmos,
    /// Generic immersive / spatial audio.
    ImmersiveAudio,
}

impl SidecarType {
    /// Returns a human-readable label for this sidecar type.
    pub fn label(&self) -> &'static str {
        match self {
            SidecarType::Subtitle => "Subtitle",
            SidecarType::ClosedCaption => "Closed Caption",
            SidecarType::AudioDescription => "Audio Description",
            SidecarType::VisuallyImpairedNarrative => "Visually Impaired Narrative",
            SidecarType::HearingImpaired => "Hearing Impaired",
            SidecarType::DolbyAtmos => "Dolby Atmos",
            SidecarType::ImmersiveAudio => "Immersive Audio",
        }
    }

    /// Returns whether this sidecar type carries audio content.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            SidecarType::AudioDescription
                | SidecarType::VisuallyImpairedNarrative
                | SidecarType::DolbyAtmos
                | SidecarType::ImmersiveAudio
        )
    }

    /// Returns whether this sidecar type carries text/caption content.
    pub fn is_text(&self) -> bool {
        matches!(
            self,
            SidecarType::Subtitle | SidecarType::ClosedCaption | SidecarType::HearingImpaired
        )
    }
}

/// A single sidecar asset in an IMF package.
#[derive(Debug, Clone)]
pub struct SidecarAsset {
    /// Unique asset identifier.
    pub asset_id: String,
    /// The type of this sidecar asset.
    pub sidecar_type: SidecarType,
    /// BCP-47 language code, if applicable.
    pub language: Option<String>,
    /// URI or file path for the asset.
    pub uri: String,
    /// Size of the asset in bytes.
    pub size_bytes: u64,
}

impl SidecarAsset {
    /// Creates a new sidecar asset.
    pub fn new(id: &str, stype: SidecarType, uri: &str) -> Self {
        Self {
            asset_id: id.to_string(),
            sidecar_type: stype,
            language: None,
            uri: uri.to_string(),
            size_bytes: 0,
        }
    }

    /// Sets the language for this sidecar asset (builder-style).
    pub fn with_language(mut self, lang: &str) -> Self {
        self.language = Some(lang.to_string());
        self
    }

    /// Sets the file size for this sidecar asset (builder-style).
    pub fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = size_bytes;
        self
    }

    /// Returns a human-readable description of this asset.
    pub fn description(&self) -> String {
        let lang_suffix = self
            .language
            .as_deref()
            .map_or(String::new(), |l| format!(" [{}]", l));
        format!(
            "{}{} — {}",
            self.sidecar_type.label(),
            lang_suffix,
            self.uri
        )
    }

    /// Returns true if this asset has an associated language.
    pub fn has_language(&self) -> bool {
        self.language.is_some()
    }
}

/// A manifest listing all sidecar assets for a given IMF package.
#[derive(Debug, Clone)]
pub struct SidecarManifest {
    /// The IMF package identifier this manifest belongs to.
    pub package_id: String,
    /// All sidecar assets registered in this manifest.
    pub assets: Vec<SidecarAsset>,
}

impl SidecarManifest {
    /// Creates a new, empty sidecar manifest for the given package.
    pub fn new(package_id: &str) -> Self {
        Self {
            package_id: package_id.to_string(),
            assets: Vec::new(),
        }
    }

    /// Adds a sidecar asset to this manifest.
    pub fn add(&mut self, asset: SidecarAsset) {
        self.assets.push(asset);
    }

    /// Returns all assets of the given sidecar type.
    pub fn assets_by_type(&self, stype: &SidecarType) -> Vec<&SidecarAsset> {
        self.assets
            .iter()
            .filter(|a| &a.sidecar_type == stype)
            .collect()
    }

    /// Returns a sorted, deduplicated list of languages across all assets that have one.
    pub fn languages(&self) -> Vec<String> {
        let mut langs: Vec<String> = self
            .assets
            .iter()
            .filter_map(|a| a.language.clone())
            .collect();
        langs.sort();
        langs.dedup();
        langs
    }

    /// Returns the total number of assets in the manifest.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Returns all audio-bearing sidecar assets.
    pub fn audio_assets(&self) -> Vec<&SidecarAsset> {
        self.assets
            .iter()
            .filter(|a| a.sidecar_type.is_audio())
            .collect()
    }

    /// Returns all text-bearing sidecar assets.
    pub fn text_assets(&self) -> Vec<&SidecarAsset> {
        self.assets
            .iter()
            .filter(|a| a.sidecar_type.is_text())
            .collect()
    }

    /// Finds an asset by its ID.
    pub fn find_by_id(&self, asset_id: &str) -> Option<&SidecarAsset> {
        self.assets.iter().find(|a| a.asset_id == asset_id)
    }

    /// Returns the total size of all sidecar assets in bytes.
    pub fn total_size_bytes(&self) -> u64 {
        self.assets.iter().map(|a| a.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sidecar_type_label() {
        assert_eq!(SidecarType::Subtitle.label(), "Subtitle");
        assert_eq!(SidecarType::ClosedCaption.label(), "Closed Caption");
        assert_eq!(SidecarType::AudioDescription.label(), "Audio Description");
        assert_eq!(SidecarType::DolbyAtmos.label(), "Dolby Atmos");
        assert_eq!(SidecarType::ImmersiveAudio.label(), "Immersive Audio");
        assert_eq!(SidecarType::HearingImpaired.label(), "Hearing Impaired");
        assert_eq!(
            SidecarType::VisuallyImpairedNarrative.label(),
            "Visually Impaired Narrative"
        );
    }

    #[test]
    fn test_sidecar_type_is_audio() {
        assert!(SidecarType::AudioDescription.is_audio());
        assert!(SidecarType::DolbyAtmos.is_audio());
        assert!(SidecarType::ImmersiveAudio.is_audio());
        assert!(SidecarType::VisuallyImpairedNarrative.is_audio());
        assert!(!SidecarType::Subtitle.is_audio());
        assert!(!SidecarType::ClosedCaption.is_audio());
    }

    #[test]
    fn test_sidecar_type_is_text() {
        assert!(SidecarType::Subtitle.is_text());
        assert!(SidecarType::ClosedCaption.is_text());
        assert!(SidecarType::HearingImpaired.is_text());
        assert!(!SidecarType::DolbyAtmos.is_text());
        assert!(!SidecarType::ImmersiveAudio.is_text());
    }

    #[test]
    fn test_sidecar_asset_new() {
        let asset = SidecarAsset::new("asset-001", SidecarType::Subtitle, "subs/en.ttml");
        assert_eq!(asset.asset_id, "asset-001");
        assert_eq!(asset.sidecar_type, SidecarType::Subtitle);
        assert_eq!(asset.uri, "subs/en.ttml");
        assert!(asset.language.is_none());
        assert_eq!(asset.size_bytes, 0);
    }

    #[test]
    fn test_sidecar_asset_with_language() {
        let asset = SidecarAsset::new("asset-002", SidecarType::ClosedCaption, "cc/fr.srt")
            .with_language("fr");
        assert_eq!(asset.language, Some("fr".to_string()));
        assert!(asset.has_language());
    }

    #[test]
    fn test_sidecar_asset_description_no_language() {
        let asset = SidecarAsset::new("asset-003", SidecarType::DolbyAtmos, "audio/atmos.mxf");
        let desc = asset.description();
        assert!(desc.contains("Dolby Atmos"));
        assert!(desc.contains("audio/atmos.mxf"));
        assert!(!desc.contains('['));
    }

    #[test]
    fn test_sidecar_asset_description_with_language() {
        let asset = SidecarAsset::new("asset-004", SidecarType::Subtitle, "subs/de.ttml")
            .with_language("de");
        let desc = asset.description();
        assert!(desc.contains("Subtitle"));
        assert!(desc.contains("[de]"));
    }

    #[test]
    fn test_sidecar_manifest_add_and_count() {
        let mut manifest = SidecarManifest::new("pkg-001");
        assert_eq!(manifest.asset_count(), 0);
        manifest.add(SidecarAsset::new("a1", SidecarType::Subtitle, "s1.ttml").with_language("en"));
        manifest.add(SidecarAsset::new("a2", SidecarType::Subtitle, "s2.ttml").with_language("fr"));
        assert_eq!(manifest.asset_count(), 2);
    }

    #[test]
    fn test_sidecar_manifest_assets_by_type() {
        let mut manifest = SidecarManifest::new("pkg-002");
        manifest.add(SidecarAsset::new("a1", SidecarType::Subtitle, "s.ttml").with_language("en"));
        manifest.add(SidecarAsset::new(
            "a2",
            SidecarType::DolbyAtmos,
            "atmos.mxf",
        ));
        manifest.add(SidecarAsset::new("a3", SidecarType::Subtitle, "s2.ttml").with_language("de"));

        let subs = manifest.assets_by_type(&SidecarType::Subtitle);
        assert_eq!(subs.len(), 2);

        let atmos = manifest.assets_by_type(&SidecarType::DolbyAtmos);
        assert_eq!(atmos.len(), 1);
    }

    #[test]
    fn test_sidecar_manifest_languages() {
        let mut manifest = SidecarManifest::new("pkg-003");
        manifest.add(SidecarAsset::new("a1", SidecarType::Subtitle, "s.ttml").with_language("en"));
        manifest.add(SidecarAsset::new("a2", SidecarType::Subtitle, "s2.ttml").with_language("fr"));
        manifest.add(SidecarAsset::new("a3", SidecarType::Subtitle, "s3.ttml").with_language("en")); // dup
        manifest.add(SidecarAsset::new(
            "a4",
            SidecarType::DolbyAtmos,
            "atmos.mxf",
        )); // no language

        let langs = manifest.languages();
        assert_eq!(langs, vec!["en".to_string(), "fr".to_string()]);
    }

    #[test]
    fn test_sidecar_manifest_audio_and_text_assets() {
        let mut manifest = SidecarManifest::new("pkg-004");
        manifest.add(SidecarAsset::new("a1", SidecarType::Subtitle, "s.ttml"));
        manifest.add(SidecarAsset::new(
            "a2",
            SidecarType::AudioDescription,
            "ad.mxf",
        ));
        manifest.add(SidecarAsset::new(
            "a3",
            SidecarType::ClosedCaption,
            "cc.srt",
        ));
        manifest.add(SidecarAsset::new(
            "a4",
            SidecarType::DolbyAtmos,
            "atmos.mxf",
        ));

        let audio = manifest.audio_assets();
        assert_eq!(audio.len(), 2);

        let text = manifest.text_assets();
        assert_eq!(text.len(), 2);
    }

    #[test]
    fn test_sidecar_manifest_find_by_id() {
        let mut manifest = SidecarManifest::new("pkg-005");
        manifest.add(SidecarAsset::new(
            "asset-xyz",
            SidecarType::HearingImpaired,
            "hi.ttml",
        ));
        assert!(manifest.find_by_id("asset-xyz").is_some());
        assert!(manifest.find_by_id("nonexistent").is_none());
    }

    #[test]
    fn test_sidecar_manifest_total_size() {
        let mut manifest = SidecarManifest::new("pkg-006");
        manifest.add(SidecarAsset::new("a1", SidecarType::Subtitle, "s1.ttml").with_size(1024));
        manifest.add(SidecarAsset::new("a2", SidecarType::Subtitle, "s2.ttml").with_size(2048));
        assert_eq!(manifest.total_size_bytes(), 3072);
    }
}
