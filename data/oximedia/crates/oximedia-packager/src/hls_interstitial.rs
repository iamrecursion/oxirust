// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! HLS interstitial support — `EXT-X-DATERANGE` for ad insertion and timed
//! events.
//!
//! HLS interstitials are described in:
//! - RFC 8216 §4.3.2.7 (`EXT-X-DATERANGE`)
//! - Apple WWDC 2022 — HLS Interstitial Scheduling
//!
//! An **interstitial** is a piece of content (typically an ad or a promo clip)
//! that is spliced into the primary stream at a specific wall-clock time.
//! Unlike SCTE-35, interstitials are self-describing at the manifest level
//! and do not require in-band signalling.
//!
//! # Key Concepts
//!
//! | Concept | Description |
//! |---------|-------------|
//! | `EXT-X-DATERANGE` | An HLS tag that annotates a calendar date range with arbitrary key–value attributes. |
//! | `X-ASSET-URI` | Apple extension — URI of the interstitial asset playlist. |
//! | `X-ASSET-LIST` | Apple extension — URI of a JSON asset list for dynamic selection. |
//! | `X-RESUME-OFFSET` | How many seconds into the primary content to resume after the interstitial. |
//! | `X-RESTRICT` | Comma-separated list of feature restrictions (e.g. `SKIP,JUMP`). |
//!
//! # Example
//!
//! ```
//! use oximedia_packager::hls_interstitial::{
//!     DateRangeClass, HlsDateRange, HlsInterstitial, InterstitialRestriction,
//! };
//! use std::time::Duration;
//!
//! // Build a 30-second pre-roll ad interstitial starting at T=0.
//! let interstitial = HlsInterstitial::builder("ad-preroll")
//!     .start_date("2024-01-01T00:00:00Z")
//!     .duration(Duration::from_secs(30))
//!     .asset_uri("https://ads.example.com/preroll.m3u8")
//!     .resume_offset(Duration::ZERO)
//!     .restrict(&[InterstitialRestriction::Skip, InterstitialRestriction::Jump])
//!     .build()
//!     .expect("interstitial build should succeed");
//!
//! let tag = interstitial.to_m3u8_tag();
//! assert!(tag.contains("EXT-X-DATERANGE"));
//! assert!(tag.contains("X-ASSET-URI"));
//! ```

use crate::error::{PackagerError, PackagerResult};
use std::collections::BTreeMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// InterstitialRestriction
// ---------------------------------------------------------------------------

/// Feature restrictions applied to an interstitial.
///
/// These map to the `X-RESTRICT` attribute of `EXT-X-DATERANGE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterstitialRestriction {
    /// The user must not be able to skip the interstitial.
    Skip,
    /// The user must not be able to jump past the interstitial.
    Jump,
}

impl InterstitialRestriction {
    /// Return the string representation used in the HLS tag.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "SKIP",
            Self::Jump => "JUMP",
        }
    }
}

// ---------------------------------------------------------------------------
// DateRangeClass
// ---------------------------------------------------------------------------

/// The `CLASS` attribute of an `EXT-X-DATERANGE` tag.
///
/// Well-known classes include Apple interstitials and SCTE-35.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DateRangeClass {
    /// Apple HLS interstitial class URI.
    AppleInterstitial,
    /// SCTE-35 CUE-OUT / CUE-IN class URI.
    Scte35,
    /// A custom class URI supplied by the caller.
    Custom(String),
}

impl DateRangeClass {
    /// Return the URI string for this class.
    #[must_use]
    pub fn as_uri(&self) -> &str {
        match self {
            Self::AppleInterstitial => "com.apple.hls.interstitial",
            Self::Scte35 => "com.apple.hls.scte35",
            Self::Custom(uri) => uri.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// HlsDateRange
// ---------------------------------------------------------------------------

/// A parsed/constructed `EXT-X-DATERANGE` tag.
///
/// This lower-level type mirrors the raw attribute set of the tag.  For the
/// higher-level interstitial abstraction, use [`HlsInterstitial`].
#[derive(Debug, Clone)]
pub struct HlsDateRange {
    /// Unique identifier for this date range.
    pub id: String,
    /// Optional class URI.
    pub class: Option<DateRangeClass>,
    /// ISO 8601 start date-time string.
    pub start_date: String,
    /// Optional ISO 8601 end date-time string.
    pub end_date: Option<String>,
    /// Optional declared duration.
    pub duration: Option<Duration>,
    /// Optional planned duration (upper bound estimate).
    pub planned_duration: Option<Duration>,
    /// Whether this date range is an end marker for a previous range.
    pub end_on_next: bool,
    /// Arbitrary `X-` prefixed client attributes (key → value).
    pub client_attributes: BTreeMap<String, String>,
}

impl HlsDateRange {
    /// Create a new date range with the given ID and start date.
    #[must_use]
    pub fn new(id: impl Into<String>, start_date: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            class: None,
            start_date: start_date.into(),
            end_date: None,
            duration: None,
            planned_duration: None,
            end_on_next: false,
            client_attributes: BTreeMap::new(),
        }
    }

    /// Set a client attribute.
    pub fn set_client_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.client_attributes.insert(key.into(), value.into());
    }

    /// Render as an `#EXT-X-DATERANGE:…` tag line (without trailing newline).
    #[must_use]
    pub fn to_m3u8_tag(&self) -> String {
        let mut attrs = Vec::new();

        attrs.push(format!("ID=\"{}\"", self.id));

        if let Some(class) = &self.class {
            attrs.push(format!("CLASS=\"{}\"", class.as_uri()));
        }

        attrs.push(format!("START-DATE=\"{}\"", self.start_date));

        if let Some(end) = &self.end_date {
            attrs.push(format!("END-DATE=\"{end}\""));
        }

        if let Some(dur) = self.duration {
            attrs.push(format!("DURATION={:.6}", dur.as_secs_f64()));
        }

        if let Some(planned) = self.planned_duration {
            attrs.push(format!("PLANNED-DURATION={:.6}", planned.as_secs_f64()));
        }

        if self.end_on_next {
            attrs.push("END-ON-NEXT=YES".to_string());
        }

        // Client attributes: sort for deterministic output
        for (key, val) in &self.client_attributes {
            attrs.push(format!("{key}=\"{val}\""));
        }

        format!("#EXT-X-DATERANGE:{}", attrs.join(","))
    }
}

// ---------------------------------------------------------------------------
// HlsInterstitial
// ---------------------------------------------------------------------------

/// A high-level HLS interstitial (ad / promo / chapter break).
///
/// Use [`HlsInterstitialBuilder`] (via [`HlsInterstitial::builder`]) for
/// construction.
#[derive(Debug, Clone)]
pub struct HlsInterstitial {
    /// Unique identifier for this interstitial.
    pub id: String,
    /// ISO 8601 start date-time.
    pub start_date: String,
    /// Duration of the interstitial content.
    pub duration: Option<Duration>,
    /// URI of the interstitial asset playlist (`X-ASSET-URI`).
    pub asset_uri: Option<String>,
    /// URI of a JSON asset list (`X-ASSET-LIST`).
    pub asset_list_uri: Option<String>,
    /// How far into the primary timeline to resume after the interstitial.
    /// `Some(Duration::ZERO)` means resume at the exact splice point.
    pub resume_offset: Option<Duration>,
    /// Feature restrictions.
    pub restrictions: Vec<InterstitialRestriction>,
    /// Content may snap to the previous segment boundary if `true`.
    pub snap: bool,
    /// Additional arbitrary client attributes.
    pub extra_attributes: BTreeMap<String, String>,
}

impl HlsInterstitial {
    /// Create a builder for a new interstitial.
    #[must_use]
    pub fn builder(id: impl Into<String>) -> HlsInterstitialBuilder {
        HlsInterstitialBuilder::new(id)
    }

    /// Render this interstitial as an `#EXT-X-DATERANGE` tag line.
    #[must_use]
    pub fn to_m3u8_tag(&self) -> String {
        let mut date_range = HlsDateRange::new(self.id.clone(), self.start_date.clone());
        date_range.class = Some(DateRangeClass::AppleInterstitial);
        date_range.duration = self.duration;

        if let Some(uri) = &self.asset_uri {
            date_range.set_client_attribute("X-ASSET-URI", uri.clone());
        }

        if let Some(list) = &self.asset_list_uri {
            date_range.set_client_attribute("X-ASSET-LIST", list.clone());
        }

        if let Some(offset) = self.resume_offset {
            date_range
                .set_client_attribute("X-RESUME-OFFSET", format!("{:.6}", offset.as_secs_f64()));
        }

        if !self.restrictions.is_empty() {
            let restriction_str = self
                .restrictions
                .iter()
                .map(|r| r.as_str())
                .collect::<Vec<_>>()
                .join(",");
            date_range.set_client_attribute("X-RESTRICT", restriction_str);
        }

        if self.snap {
            date_range.set_client_attribute("X-SNAP", "IN,OUT");
        }

        for (k, v) in &self.extra_attributes {
            date_range.set_client_attribute(k.clone(), v.clone());
        }

        date_range.to_m3u8_tag()
    }

    /// Validate the interstitial.
    ///
    /// # Errors
    ///
    /// Returns an error if neither `asset_uri` nor `asset_list_uri` is set,
    /// or if the ID is empty.
    pub fn validate(&self) -> PackagerResult<()> {
        if self.id.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "interstitial ID must not be empty".into(),
            ));
        }
        if self.asset_uri.is_none() && self.asset_list_uri.is_none() {
            return Err(PackagerError::InvalidConfig(
                "interstitial must have either asset_uri or asset_list_uri".into(),
            ));
        }
        if self.start_date.is_empty() {
            return Err(PackagerError::InvalidConfig(
                "interstitial start_date must not be empty".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HlsInterstitialBuilder
// ---------------------------------------------------------------------------

/// Builder for [`HlsInterstitial`].
pub struct HlsInterstitialBuilder {
    id: String,
    start_date: String,
    duration: Option<Duration>,
    asset_uri: Option<String>,
    asset_list_uri: Option<String>,
    resume_offset: Option<Duration>,
    restrictions: Vec<InterstitialRestriction>,
    snap: bool,
    extra_attributes: BTreeMap<String, String>,
}

impl HlsInterstitialBuilder {
    /// Create a new builder with the given interstitial ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            start_date: String::new(),
            duration: None,
            asset_uri: None,
            asset_list_uri: None,
            resume_offset: None,
            restrictions: Vec::new(),
            snap: false,
            extra_attributes: BTreeMap::new(),
        }
    }

    /// Set the ISO 8601 start date-time.
    #[must_use]
    pub fn start_date(mut self, date: impl Into<String>) -> Self {
        self.start_date = date.into();
        self
    }

    /// Set the duration of the interstitial content.
    #[must_use]
    pub fn duration(mut self, dur: Duration) -> Self {
        self.duration = Some(dur);
        self
    }

    /// Set the asset URI (`X-ASSET-URI`).
    #[must_use]
    pub fn asset_uri(mut self, uri: impl Into<String>) -> Self {
        self.asset_uri = Some(uri.into());
        self
    }

    /// Set the asset list URI (`X-ASSET-LIST`).
    #[must_use]
    pub fn asset_list_uri(mut self, uri: impl Into<String>) -> Self {
        self.asset_list_uri = Some(uri.into());
        self
    }

    /// Set the resume offset (`X-RESUME-OFFSET`).
    #[must_use]
    pub fn resume_offset(mut self, offset: Duration) -> Self {
        self.resume_offset = Some(offset);
        self
    }

    /// Set feature restrictions (`X-RESTRICT`).
    #[must_use]
    pub fn restrict(mut self, restrictions: &[InterstitialRestriction]) -> Self {
        self.restrictions = restrictions.to_vec();
        self
    }

    /// Enable content snapping.
    #[must_use]
    pub fn snap(mut self, snap: bool) -> Self {
        self.snap = snap;
        self
    }

    /// Add an extra client attribute.
    #[must_use]
    pub fn extra_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_attributes.insert(key.into(), value.into());
        self
    }

    /// Build the [`HlsInterstitial`].
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn build(self) -> PackagerResult<HlsInterstitial> {
        let interstitial = HlsInterstitial {
            id: self.id,
            start_date: self.start_date,
            duration: self.duration,
            asset_uri: self.asset_uri,
            asset_list_uri: self.asset_list_uri,
            resume_offset: self.resume_offset,
            restrictions: self.restrictions,
            snap: self.snap,
            extra_attributes: self.extra_attributes,
        };
        interstitial.validate()?;
        Ok(interstitial)
    }
}

// ---------------------------------------------------------------------------
// InterstitialSchedule
// ---------------------------------------------------------------------------

/// A collection of [`HlsInterstitial`] entries forming a complete interstitial
/// schedule for a single media playlist.
///
/// The schedule can be serialised as a block of `EXT-X-DATERANGE` tags that
/// are typically inserted near the top of an HLS media playlist.
#[derive(Debug, Clone, Default)]
pub struct InterstitialSchedule {
    interstitials: Vec<HlsInterstitial>,
}

impl InterstitialSchedule {
    /// Create an empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an interstitial to the schedule.
    ///
    /// # Errors
    ///
    /// Returns an error if the interstitial fails validation.
    pub fn add(&mut self, interstitial: HlsInterstitial) -> PackagerResult<()> {
        interstitial.validate()?;
        self.interstitials.push(interstitial);
        Ok(())
    }

    /// Return the number of interstitials in the schedule.
    #[must_use]
    pub fn len(&self) -> usize {
        self.interstitials.len()
    }

    /// Return `true` if the schedule has no interstitials.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interstitials.is_empty()
    }

    /// Render all interstitials as `EXT-X-DATERANGE` lines.
    ///
    /// Lines are joined by newlines and terminated with a trailing newline.
    #[must_use]
    pub fn to_m3u8_tags(&self) -> String {
        let mut out = String::new();
        for item in &self.interstitials {
            out.push_str(&item.to_m3u8_tag());
            out.push('\n');
        }
        out
    }

    /// Return all interstitials.
    #[must_use]
    pub fn interstitials(&self) -> &[HlsInterstitial] {
        &self.interstitials
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- InterstitialRestriction --------------------------------------------

    #[test]
    fn test_restriction_skip_str() {
        assert_eq!(InterstitialRestriction::Skip.as_str(), "SKIP");
    }

    #[test]
    fn test_restriction_jump_str() {
        assert_eq!(InterstitialRestriction::Jump.as_str(), "JUMP");
    }

    // --- DateRangeClass -----------------------------------------------------

    #[test]
    fn test_date_range_class_apple_uri() {
        assert_eq!(
            DateRangeClass::AppleInterstitial.as_uri(),
            "com.apple.hls.interstitial"
        );
    }

    #[test]
    fn test_date_range_class_scte35_uri() {
        assert_eq!(DateRangeClass::Scte35.as_uri(), "com.apple.hls.scte35");
    }

    #[test]
    fn test_date_range_class_custom_uri() {
        let cls = DateRangeClass::Custom("urn:example:event".to_string());
        assert_eq!(cls.as_uri(), "urn:example:event");
    }

    // --- HlsDateRange -------------------------------------------------------

    #[test]
    fn test_date_range_minimal_tag() {
        let dr = HlsDateRange::new("dr-1", "2024-01-01T00:00:00Z");
        let tag = dr.to_m3u8_tag();
        assert!(tag.starts_with("#EXT-X-DATERANGE:"));
        assert!(tag.contains("ID=\"dr-1\""));
        assert!(tag.contains("START-DATE=\"2024-01-01T00:00:00Z\""));
    }

    #[test]
    fn test_date_range_with_duration() {
        let mut dr = HlsDateRange::new("dr-2", "2024-01-01T00:00:00Z");
        dr.duration = Some(Duration::from_secs(30));
        let tag = dr.to_m3u8_tag();
        assert!(tag.contains("DURATION=30"));
    }

    #[test]
    fn test_date_range_with_class() {
        let mut dr = HlsDateRange::new("dr-3", "2024-01-01T00:00:00Z");
        dr.class = Some(DateRangeClass::AppleInterstitial);
        let tag = dr.to_m3u8_tag();
        assert!(tag.contains("CLASS=\"com.apple.hls.interstitial\""));
    }

    #[test]
    fn test_date_range_with_end_date() {
        let mut dr = HlsDateRange::new("dr-4", "2024-01-01T00:00:00Z");
        dr.end_date = Some("2024-01-01T00:00:30Z".to_string());
        let tag = dr.to_m3u8_tag();
        assert!(tag.contains("END-DATE=\"2024-01-01T00:00:30Z\""));
    }

    #[test]
    fn test_date_range_end_on_next() {
        let mut dr = HlsDateRange::new("dr-5", "2024-01-01T00:00:00Z");
        dr.end_on_next = true;
        let tag = dr.to_m3u8_tag();
        assert!(tag.contains("END-ON-NEXT=YES"));
    }

    #[test]
    fn test_date_range_client_attribute() {
        let mut dr = HlsDateRange::new("dr-6", "2024-01-01T00:00:00Z");
        dr.set_client_attribute("X-FOO", "bar");
        let tag = dr.to_m3u8_tag();
        assert!(tag.contains("X-FOO=\"bar\""));
    }

    // --- HlsInterstitialBuilder / HlsInterstitial ---------------------------

    #[test]
    fn test_interstitial_build_minimal() {
        let result = HlsInterstitial::builder("ad-1")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_interstitial_build_missing_id_fails() {
        let result = HlsInterstitial::builder("")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_interstitial_build_missing_asset_fails() {
        let result = HlsInterstitial::builder("ad-2")
            .start_date("2024-01-01T00:00:00Z")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_interstitial_asset_list_uri_accepted() {
        let result = HlsInterstitial::builder("ad-3")
            .start_date("2024-01-01T00:00:00Z")
            .asset_list_uri("https://ads.example.com/list.json")
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_interstitial_tag_contains_ext_x_daterange() {
        let ad = HlsInterstitial::builder("preroll")
            .start_date("2024-01-01T00:00:00Z")
            .duration(Duration::from_secs(30))
            .asset_uri("https://ads.example.com/preroll.m3u8")
            .build()
            .expect("build should succeed");

        let tag = ad.to_m3u8_tag();
        assert!(tag.contains("EXT-X-DATERANGE"));
        assert!(tag.contains("X-ASSET-URI"));
        assert!(tag.contains("preroll.m3u8"));
        assert!(tag.contains("DURATION=30"));
        assert!(tag.contains("com.apple.hls.interstitial"));
    }

    #[test]
    fn test_interstitial_tag_resume_offset() {
        let ad = HlsInterstitial::builder("midroll")
            .start_date("2024-01-01T00:01:30Z")
            .asset_uri("https://ads.example.com/mid.m3u8")
            .resume_offset(Duration::ZERO)
            .build()
            .expect("build should succeed");

        let tag = ad.to_m3u8_tag();
        assert!(tag.contains("X-RESUME-OFFSET"));
    }

    #[test]
    fn test_interstitial_tag_restrictions() {
        let ad = HlsInterstitial::builder("unskippable")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .restrict(&[InterstitialRestriction::Skip, InterstitialRestriction::Jump])
            .build()
            .expect("build should succeed");

        let tag = ad.to_m3u8_tag();
        assert!(tag.contains("X-RESTRICT"));
        assert!(tag.contains("SKIP"));
        assert!(tag.contains("JUMP"));
    }

    #[test]
    fn test_interstitial_tag_snap() {
        let ad = HlsInterstitial::builder("snap-ad")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .snap(true)
            .build()
            .expect("build should succeed");

        let tag = ad.to_m3u8_tag();
        assert!(tag.contains("X-SNAP"));
    }

    #[test]
    fn test_interstitial_extra_attribute() {
        let ad = HlsInterstitial::builder("custom-ad")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .extra_attribute("X-AD-CAMPAIGN", "summer2024")
            .build()
            .expect("build should succeed");

        let tag = ad.to_m3u8_tag();
        assert!(tag.contains("X-AD-CAMPAIGN=\"summer2024\""));
    }

    // --- InterstitialSchedule -----------------------------------------------

    #[test]
    fn test_schedule_empty() {
        let schedule = InterstitialSchedule::new();
        assert!(schedule.is_empty());
        assert_eq!(schedule.len(), 0);
        assert_eq!(schedule.to_m3u8_tags(), "");
    }

    #[test]
    fn test_schedule_add_and_render() {
        let mut schedule = InterstitialSchedule::new();

        let ad1 = HlsInterstitial::builder("pre")
            .start_date("2024-01-01T00:00:00Z")
            .duration(Duration::from_secs(15))
            .asset_uri("https://ads.example.com/pre.m3u8")
            .build()
            .expect("should succeed");

        let ad2 = HlsInterstitial::builder("mid")
            .start_date("2024-01-01T00:05:00Z")
            .duration(Duration::from_secs(30))
            .asset_uri("https://ads.example.com/mid.m3u8")
            .build()
            .expect("should succeed");

        schedule.add(ad1).expect("add should succeed");
        schedule.add(ad2).expect("add should succeed");

        assert_eq!(schedule.len(), 2);

        let tags = schedule.to_m3u8_tags();
        assert_eq!(tags.lines().count(), 2);
        assert!(tags.contains("pre"));
        assert!(tags.contains("mid"));
    }

    #[test]
    fn test_schedule_add_invalid_interstitial_fails() {
        let mut schedule = InterstitialSchedule::new();

        // Missing asset URI — should fail validation
        let bad = HlsInterstitial {
            id: "bad".to_string(),
            start_date: "2024-01-01T00:00:00Z".to_string(),
            duration: None,
            asset_uri: None,
            asset_list_uri: None,
            resume_offset: None,
            restrictions: Vec::new(),
            snap: false,
            extra_attributes: BTreeMap::new(),
        };

        assert!(schedule.add(bad).is_err());
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_schedule_interstitials_accessor() {
        let mut schedule = InterstitialSchedule::new();
        let ad = HlsInterstitial::builder("ad")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .build()
            .expect("should succeed");
        schedule.add(ad).expect("add should succeed");
        assert_eq!(schedule.interstitials().len(), 1);
        assert_eq!(schedule.interstitials()[0].id, "ad");
    }

    #[test]
    fn test_schedule_tags_end_with_newlines() {
        let mut schedule = InterstitialSchedule::new();
        let ad = HlsInterstitial::builder("nl-test")
            .start_date("2024-01-01T00:00:00Z")
            .asset_uri("https://ads.example.com/ad.m3u8")
            .build()
            .expect("should succeed");
        schedule.add(ad).expect("add should succeed");
        let tags = schedule.to_m3u8_tags();
        assert!(tags.ends_with('\n'));
    }
}
