//! Camera-specific metadata for video clips.
//!
//! Stores lens, ISO, aperture, shutter speed and focal-length information
//! captured at the time of recording. This data is typically sourced from
//! camera roll XML exports (e.g. Arri ALEXAMetadata, REDCODE companion XML)
//! or from camera-embedded MXF/MOV metadata atoms.

#![allow(dead_code)]

use crate::clip::ClipId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Camera-specific technical metadata for a single clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraMetadata {
    /// Lens make and model, e.g. `"Zeiss Supreme Prime 50mm T1.5"`.
    pub lens: Option<String>,

    /// ISO sensitivity used during recording, e.g. `800`.
    pub iso: Option<u32>,

    /// Aperture as a floating-point f-stop value, e.g. `2.8` for f/2.8.
    pub aperture: Option<f32>,

    /// Shutter speed expressed as a fraction or angle string,
    /// e.g. `"1/48"` or `"172.8°"`.
    pub shutter_speed: Option<String>,

    /// Recorded focal length in millimetres, e.g. `50.0`.
    pub focal_length_mm: Option<f32>,

    /// Camera body make, e.g. `"ARRI"`.
    pub camera_make: Option<String>,

    /// Camera body model, e.g. `"ALEXA Mini LF"`.
    pub camera_model: Option<String>,
}

impl CameraMetadata {
    /// Creates an empty `CameraMetadata`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lens: None,
            iso: None,
            aperture: None,
            shutter_speed: None,
            focal_length_mm: None,
            camera_make: None,
            camera_model: None,
        }
    }

    /// Builder-style setter for `lens`.
    #[must_use]
    pub fn with_lens(mut self, lens: impl Into<String>) -> Self {
        self.lens = Some(lens.into());
        self
    }

    /// Builder-style setter for `iso`.
    #[must_use]
    pub fn with_iso(mut self, iso: u32) -> Self {
        self.iso = Some(iso);
        self
    }

    /// Builder-style setter for `aperture`.
    #[must_use]
    pub fn with_aperture(mut self, aperture: f32) -> Self {
        self.aperture = Some(aperture);
        self
    }

    /// Builder-style setter for `shutter_speed`.
    #[must_use]
    pub fn with_shutter_speed(mut self, speed: impl Into<String>) -> Self {
        self.shutter_speed = Some(speed.into());
        self
    }

    /// Builder-style setter for `focal_length_mm`.
    #[must_use]
    pub fn with_focal_length_mm(mut self, mm: f32) -> Self {
        self.focal_length_mm = Some(mm);
        self
    }

    /// Builder-style setter for `camera_make`.
    #[must_use]
    pub fn with_camera_make(mut self, make: impl Into<String>) -> Self {
        self.camera_make = Some(make.into());
        self
    }

    /// Builder-style setter for `camera_model`.
    #[must_use]
    pub fn with_camera_model(mut self, model: impl Into<String>) -> Self {
        self.camera_model = Some(model.into());
        self
    }

    /// Returns `true` if no fields are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lens.is_none()
            && self.iso.is_none()
            && self.aperture.is_none()
            && self.shutter_speed.is_none()
            && self.focal_length_mm.is_none()
            && self.camera_make.is_none()
            && self.camera_model.is_none()
    }

    /// Returns an exposure-value (EV) hint based on ISO and aperture when both
    /// are present.  EV = log2(aperture² / (ISO / 100)).  This is a coarse
    /// indication; shutter speed is not included here because it is stored as
    /// a string and would require additional parsing.
    #[must_use]
    pub fn approximate_ev(&self) -> Option<f32> {
        let iso = self.iso? as f32;
        let ap = self.aperture?;
        // Prevent log of non-positive values
        if iso <= 0.0 || ap <= 0.0 {
            return None;
        }
        let ev = (ap * ap / (iso / 100.0)).log2();
        Some(ev)
    }
}

impl Default for CameraMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait that allows attaching and retrieving `CameraMetadata` to any type
/// that is indexed by `ClipId`.
///
/// Implementors provide storage for an arbitrary number of clip camera records.
pub trait CameraMetadataExt {
    /// Attaches camera metadata to a clip.
    fn set_camera_metadata(&mut self, clip_id: ClipId, meta: CameraMetadata);

    /// Retrieves camera metadata for a clip.
    fn camera_metadata(&self, clip_id: &ClipId) -> Option<&CameraMetadata>;

    /// Removes and returns the camera metadata for a clip.
    fn remove_camera_metadata(&mut self, clip_id: &ClipId) -> Option<CameraMetadata>;
}

/// In-memory store implementing `CameraMetadataExt` keyed by `ClipId`.
#[derive(Debug, Default)]
pub struct CameraMetadataStore {
    inner: HashMap<ClipId, CameraMetadata>,
}

impl CameraMetadataStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the store has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns an iterator over all `(ClipId, CameraMetadata)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&ClipId, &CameraMetadata)> {
        self.inner.iter()
    }
}

impl CameraMetadataExt for CameraMetadataStore {
    fn set_camera_metadata(&mut self, clip_id: ClipId, meta: CameraMetadata) {
        self.inner.insert(clip_id, meta);
    }

    fn camera_metadata(&self, clip_id: &ClipId) -> Option<&CameraMetadata> {
        self.inner.get(clip_id)
    }

    fn remove_camera_metadata(&mut self, clip_id: &ClipId) -> Option<CameraMetadata> {
        self.inner.remove(clip_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipId;

    #[test]
    fn test_camera_metadata_default_is_empty() {
        let meta = CameraMetadata::new();
        assert!(meta.is_empty());
    }

    #[test]
    fn test_camera_metadata_builder() {
        let meta = CameraMetadata::new()
            .with_lens("Zeiss 50mm T1.5")
            .with_iso(800)
            .with_aperture(2.8)
            .with_shutter_speed("1/48")
            .with_focal_length_mm(50.0)
            .with_camera_make("ARRI")
            .with_camera_model("ALEXA Mini LF");

        assert_eq!(meta.lens.as_deref(), Some("Zeiss 50mm T1.5"));
        assert_eq!(meta.iso, Some(800));
        assert!((meta.aperture.expect("aperture should be set") - 2.8).abs() < 1e-5);
        assert_eq!(meta.shutter_speed.as_deref(), Some("1/48"));
        assert!((meta.focal_length_mm.expect("focal_length_mm should be set") - 50.0).abs() < 1e-5);
        assert_eq!(meta.camera_make.as_deref(), Some("ARRI"));
        assert_eq!(meta.camera_model.as_deref(), Some("ALEXA Mini LF"));
        assert!(!meta.is_empty());
    }

    #[test]
    fn test_camera_metadata_approximate_ev_some() {
        // aperture 2.8, iso 800 → EV = log2(7.84 / 8) ≈ −0.028
        let meta = CameraMetadata::new().with_aperture(2.8).with_iso(800);
        let ev = meta.approximate_ev();
        assert!(ev.is_some());
        let ev_val = ev.expect("EV should be computed when aperture and ISO are set");
        assert!(ev_val > -1.0 && ev_val < 1.0);
    }

    #[test]
    fn test_camera_metadata_approximate_ev_none_when_missing_fields() {
        let meta_no_iso = CameraMetadata::new().with_aperture(2.8);
        assert!(meta_no_iso.approximate_ev().is_none());

        let meta_no_ap = CameraMetadata::new().with_iso(800);
        assert!(meta_no_ap.approximate_ev().is_none());
    }

    #[test]
    fn test_camera_metadata_store_set_get() {
        let mut store = CameraMetadataStore::new();
        let id = ClipId::new();
        let meta = CameraMetadata::new().with_iso(400);

        store.set_camera_metadata(id, meta.clone());
        assert_eq!(store.len(), 1);

        let retrieved = store.camera_metadata(&id).expect("should be present");
        assert_eq!(retrieved.iso, Some(400));
    }

    #[test]
    fn test_camera_metadata_store_remove() {
        let mut store = CameraMetadataStore::new();
        let id = ClipId::new();
        store.set_camera_metadata(id, CameraMetadata::new().with_iso(100));

        let removed = store.remove_camera_metadata(&id);
        assert!(removed.is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_camera_metadata_store_missing_key() {
        let store = CameraMetadataStore::new();
        let id = ClipId::new();
        assert!(store.camera_metadata(&id).is_none());
    }

    #[test]
    fn test_camera_metadata_overwrite() {
        let mut store = CameraMetadataStore::new();
        let id = ClipId::new();

        store.set_camera_metadata(id, CameraMetadata::new().with_iso(200));
        store.set_camera_metadata(id, CameraMetadata::new().with_iso(3200));

        assert_eq!(store.len(), 1);
        assert_eq!(
            store
                .camera_metadata(&id)
                .expect("overwritten metadata should be present")
                .iso,
            Some(3200)
        );
    }
}
