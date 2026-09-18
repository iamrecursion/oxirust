//! Minting stable IRIs for chart resources.
//!
//! # Why a deterministic slug
//!
//! Linked Data lives or dies on *identity*: the same chart, computed
//! twice — on two machines, in two years — must carry the same IRI, or
//! nothing can be joined. So a chart's IRI is derived from the chart's
//! own defining inputs rather than from a clock or a counter:
//!
//! ```text
//! canonical = "{kind}|{jd_tt:.9}|{lat:.6}|{lon:.6}|{system}|{zodiac}"
//! slug      = first 32 hex chars of SHA-256(canonical)
//! chart IRI = {instance_base}chart/{slug}
//! ```
//!
//! The fixed decimal precisions matter: they are the granularity at which
//! two charts are declared "the same". `1e-9` days is ~86 µs of epoch and
//! `1e-6` degrees is ~11 cm of ground position — far finer than any birth
//! record, and coarse enough that float noise cannot split one chart into
//! two identities.
//!
//! 128 bits of SHA-256 is far beyond collision reach for any plausible
//! corpus of charts, and the full defining inputs are *also* emitted as
//! data properties on the chart, so a consumer never has to trust the
//! slug — it can always re-derive it.
//!
//! Every sub-resource of a chart is **skolemized** under the chart's own
//! IRI (`{chart}/position/Sun`) rather than left as a blank node: blank
//! nodes cannot be referenced across documents, which defeats the point
//! of publishing.

use oxrdf::NamedNode;
use sha2::{Digest, Sha256};

use crate::error::RdfError;
use crate::vocab::ns;

/// Number of hex characters of the SHA-256 digest kept in a chart slug.
const SLUG_HEX_LEN: usize = 32;

/// The first [`SLUG_HEX_LEN`] hex characters of `SHA-256(canonical)`.
fn slug_of(canonical: &str) -> String {
    let digest = Sha256::digest(canonical.as_bytes());
    let mut hex = String::with_capacity(SLUG_HEX_LEN);
    for byte in digest.iter().take(SLUG_HEX_LEN / 2) {
        use core::fmt::Write as _;
        // Writing to a `String` cannot fail; the `Result` is discarded
        // deliberately rather than unwrapped.
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// The defining inputs of a chart, from which its stable IRI is derived.
///
/// # What is *not* here
///
/// The rulership scheme is deliberately absent. A chart is defined by
/// **when and where** it was cast, and in which zodiac and house system its
/// degrees are read — those are facts about the sky. Whether its dignities
/// are then scored against the traditional or the modern domicile table is
/// an *interpretive lens applied to* the chart, not part of its identity.
///
/// Putting the scheme in the key would mint two IRIs for one chart, so a
/// natal chart emitted by `oxieph chart --rulership modern` would never
/// merge with the same person's chart appearing as one side of a synastry.
/// Instead the scheme qualifies each
/// [`DignityAssessment`](crate::vocab::oxa::DIGNITY_ASSESSMENT) resource,
/// which is where it actually belongs.
#[derive(Debug, Clone, Copy)]
pub struct ChartKey<'a> {
    /// Chart kind (`"natal"`, `"composite"`, `"progressed"`, `"transit"`).
    pub kind: &'a str,
    /// Epoch as a Julian Date in TT.
    pub jd_tt: f64,
    /// Observer geodetic latitude, degrees north.
    pub lat_deg: f64,
    /// Observer geodetic longitude, degrees east.
    pub lon_deg: f64,
    /// House-system name (kebab-case).
    pub house_system: &'a str,
    /// `"tropical"`, or the ayanamsha name for a sidereal chart.
    pub zodiac: &'a str,
}

impl ChartKey<'_> {
    /// The canonical string the slug digests. Documented and stable:
    /// changing it changes every chart IRI this crate mints.
    #[must_use]
    pub fn canonical(&self) -> String {
        format!(
            "{}|{:.9}|{:.6}|{:.6}|{}|{}",
            self.kind, self.jd_tt, self.lat_deg, self.lon_deg, self.house_system, self.zodiac,
        )
    }

    /// The chart slug: the first 32 hex characters (128 bits) of
    /// `SHA-256(canonical())`.
    #[must_use]
    pub fn slug(&self) -> String {
        slug_of(&self.canonical())
    }
}

/// The defining inputs of a comparison between two charts.
///
/// The two sides are identified by their own (already deterministic) chart
/// IRIs, so a comparison's identity follows from its charts' identities.
/// The order matters: a synastry read "A's Sun to B's Moon" is not the
/// same document as its mirror, and transits/progressions are directional
/// by construction.
#[derive(Debug, Clone, Copy)]
pub struct ComparisonKey<'a> {
    /// Comparison kind (`"synastry"`, `"transit"`, `"progression"`).
    pub kind: &'a str,
    /// IRI of the first (moving/left) chart.
    pub chart_a: &'a str,
    /// IRI of the second (fixed/right) chart.
    pub chart_b: &'a str,
}

impl ComparisonKey<'_> {
    /// The canonical string the slug digests.
    #[must_use]
    pub fn canonical(&self) -> String {
        format!("{}|{}|{}", self.kind, self.chart_a, self.chart_b)
    }

    /// The comparison slug: 32 hex characters of `SHA-256(canonical())`.
    #[must_use]
    pub fn slug(&self) -> String {
        slug_of(&self.canonical())
    }
}

/// Replaces every character outside `[A-Za-z0-9._-]` with `-`, so a local
/// name is always IRI-safe without percent-encoding.
fn slugify(local: &str) -> String {
    local
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Mints instance IRIs under a caller-chosen base.
#[derive(Debug, Clone)]
pub struct IriMinter {
    base: String,
}

impl Default for IriMinter {
    fn default() -> Self {
        Self {
            base: ns::DEFAULT_INSTANCE_BASE.to_owned(),
        }
    }
}

impl IriMinter {
    /// Builds a minter over `base`, appending a trailing `/` if absent.
    ///
    /// # Errors
    ///
    /// [`RdfError::InvalidBaseIri`] if `base` is not a valid absolute IRI.
    pub fn new(base: &str) -> Result<Self, RdfError> {
        let base = if base.ends_with('/') {
            base.to_owned()
        } else {
            format!("{base}/")
        };
        if NamedNode::new(base.clone()).is_err() {
            return Err(RdfError::InvalidBaseIri(base));
        }
        Ok(Self { base })
    }

    /// The base IRI, with its trailing `/`.
    #[must_use]
    pub fn base(&self) -> &str {
        &self.base
    }

    /// The IRI of the chart identified by `key`.
    #[must_use]
    pub fn chart(&self, key: &ChartKey<'_>) -> NamedNode {
        // `base` is a validated IRI and `slug` is pure lowercase hex, so
        // the concatenation is a valid IRI. `tests` re-parses it.
        NamedNode::new_unchecked(format!("{}chart/{}", self.base, key.slug()))
    }

    /// The chart IRI for an already-computed slug. Used by the composite
    /// chart, whose identity comes from its two source charts rather than
    /// from an epoch and a place of its own.
    #[must_use]
    pub fn chart_from_slug(&self, slug: &str) -> NamedNode {
        NamedNode::new_unchecked(format!("{}chart/{}", self.base, slugify(slug)))
    }

    /// Uses a caller-supplied chart IRI verbatim.
    ///
    /// # Errors
    ///
    /// [`RdfError::InvalidChartIri`] if `iri` is not a valid absolute IRI.
    pub fn chart_explicit(iri: &str) -> Result<NamedNode, RdfError> {
        NamedNode::new(iri.to_owned()).map_err(|_| RdfError::InvalidChartIri(iri.to_owned()))
    }

    /// A skolemized sub-resource of `chart`: `{chart}/{path}/{local}`.
    ///
    /// `local` is slugified, so any body or angle name is IRI-safe.
    #[must_use]
    pub fn child(chart: &NamedNode, path: &str, local: &str) -> NamedNode {
        NamedNode::new_unchecked(format!(
            "{}/{}/{}",
            chart.as_str(),
            slugify(path),
            slugify(local)
        ))
    }

    /// A skolemized sub-resource under an arbitrary number of path
    /// segments: `{parent}/{seg0}/{seg1}/…`.
    ///
    /// Used where one segment is not enough to keep two resources apart —
    /// a dignity assessment is `{chart}/dignity/{scheme}/{planet}`, because
    /// the same planet in the same chart has one assessment per rulership
    /// scheme and they must not collide.
    #[must_use]
    pub fn child_at(parent: &NamedNode, segments: &[&str]) -> NamedNode {
        let mut iri = parent.as_str().to_owned();
        for segment in segments {
            iri.push('/');
            iri.push_str(&slugify(segment));
        }
        NamedNode::new_unchecked(iri)
    }

    /// The IRI of the comparison identified by `key`.
    #[must_use]
    pub fn comparison(&self, key: &ComparisonKey<'_>) -> NamedNode {
        NamedNode::new_unchecked(format!("{}comparison/{}", self.base, key.slug()))
    }

    /// A direct sub-resource of `chart` with no path segment:
    /// `{chart}/{local}` (used for `activity`, `distribution`, `epoch`,
    /// `observer`).
    #[must_use]
    pub fn part(chart: &NamedNode, local: &str) -> NamedNode {
        NamedNode::new_unchecked(format!("{}/{}", chart.as_str(), slugify(local)))
    }

    /// The IRI of the software agent that computed a chart.
    #[must_use]
    pub fn agent(&self, name: &str, version: &str) -> NamedNode {
        NamedNode::new_unchecked(format!(
            "{}agent/{}-{}",
            self.base,
            slugify(name),
            slugify(version)
        ))
    }

    /// The IRI of an ephemeris data set (e.g. `"DE440"`).
    #[must_use]
    pub fn ephemeris(&self, label: &str) -> NamedNode {
        NamedNode::new_unchecked(format!("{}ephemeris/{}", self.base, slugify(label)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ChartKey<'static> {
        ChartKey {
            kind: "natal",
            jd_tt: 2_440_587.500_465_196,
            lat_deg: 51.4779,
            lon_deg: 0.0,
            house_system: "placidus",
            zodiac: "tropical",
        }
    }

    #[test]
    fn slug_is_deterministic_and_hex() {
        let a = key().slug();
        let b = key().slug();
        assert_eq!(a, b, "slug must be deterministic");
        assert_eq!(a.len(), SLUG_HEX_LEN);
        assert!(a
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }

    #[test]
    fn slug_changes_when_any_defining_input_changes() {
        let base = key().slug();
        let mut moved = key();
        moved.lat_deg += 0.000_01; // above the 1e-6 canonical precision
        assert_ne!(base, moved.slug());

        let mut other_system = key();
        other_system.house_system = "koch";
        assert_ne!(base, other_system.slug());

        let mut sidereal = key();
        sidereal.zodiac = "lahiri";
        assert_ne!(base, sidereal.slug());
    }

    #[test]
    fn slug_is_stable_below_the_canonical_precision() {
        // 1e-9 deg is far below the 1e-6 formatting precision, so the two
        // charts are deliberately the same identity.
        let mut jittered = key();
        jittered.lat_deg += 1e-9;
        assert_eq!(key().slug(), jittered.slug());
    }

    /// The whole reason the rulership scheme is not in the key: two runs
    /// that differ only in how dignities are scored describe the *same*
    /// chart and must share one IRI, or nothing will ever merge.
    #[test]
    fn chart_identity_ignores_interpretive_choices() {
        let canonical = key().canonical();
        assert!(
            !canonical.contains("traditional") && !canonical.contains("modern"),
            "rulership must not enter the chart's identity: {canonical}"
        );
        // It is the sky-facts that define a chart.
        assert!(canonical.contains("placidus"));
        assert!(canonical.contains("tropical"));
        assert!(canonical.starts_with("natal|"));
    }

    #[test]
    fn nested_children_do_not_collide_across_schemes() {
        let chart = IriMinter::default().chart(&key());
        let traditional = IriMinter::child_at(&chart, &["dignity", "traditional", "Mars"]);
        let modern = IriMinter::child_at(&chart, &["dignity", "modern", "Mars"]);
        assert_ne!(traditional, modern);
        assert!(traditional.as_str().ends_with("/dignity/traditional/Mars"));
        assert!(NamedNode::new(modern.as_str().to_owned()).is_ok());
    }

    #[test]
    fn comparison_iri_is_deterministic_and_directional() {
        let minter = IriMinter::default();
        let a = "https://example.org/id/chart/aaa";
        let b = "https://example.org/id/chart/bbb";
        let forward = ComparisonKey {
            kind: "synastry",
            chart_a: a,
            chart_b: b,
        };
        let mirror = ComparisonKey {
            kind: "synastry",
            chart_a: b,
            chart_b: a,
        };
        assert_eq!(minter.comparison(&forward), minter.comparison(&forward));
        assert_ne!(
            minter.comparison(&forward),
            minter.comparison(&mirror),
            "A->B and B->A are different documents"
        );
        let transit = ComparisonKey {
            kind: "transit",
            chart_a: a,
            chart_b: b,
        };
        assert_ne!(minter.comparison(&forward), minter.comparison(&transit));
        assert!(NamedNode::new(minter.comparison(&forward).as_str().to_owned()).is_ok());
    }

    #[test]
    fn minted_iris_are_valid() {
        let minter = IriMinter::default();
        let chart = minter.chart(&key());
        assert!(NamedNode::new(chart.as_str().to_owned()).is_ok());
        assert!(chart.as_str().starts_with(ns::DEFAULT_INSTANCE_BASE));

        let child = IriMinter::child(&chart, "position", "Sun");
        assert!(NamedNode::new(child.as_str().to_owned()).is_ok());
        assert!(child.as_str().ends_with("/position/Sun"));

        let part = IriMinter::part(&chart, "activity");
        assert!(NamedNode::new(part.as_str().to_owned()).is_ok());

        assert!(NamedNode::new(minter.agent("oxiephemeris", "0.1.1").as_str().to_owned()).is_ok());
        assert!(NamedNode::new(minter.ephemeris("DE440").as_str().to_owned()).is_ok());
    }

    #[test]
    fn slugify_makes_awkward_names_iri_safe() {
        let chart = IriMinter::default().chart(&key());
        let child = IriMinter::child(&chart, "angle", "East Point");
        assert!(child.as_str().ends_with("/angle/East-Point"));
        assert!(NamedNode::new(child.as_str().to_owned()).is_ok());
    }

    #[test]
    fn base_gets_a_trailing_slash_and_is_validated() {
        let Ok(minter) = IriMinter::new("https://example.org/id") else {
            panic!("valid IRI must be accepted");
        };
        assert_eq!(minter.base(), "https://example.org/id/");
        assert!(IriMinter::new("not an iri").is_err());
    }

    #[test]
    fn explicit_chart_iri_round_trips_and_rejects_garbage() {
        let Ok(node) = IriMinter::chart_explicit("https://example.org/c/1") else {
            panic!("valid IRI must be accepted");
        };
        assert_eq!(node.as_str(), "https://example.org/c/1");
        assert!(IriMinter::chart_explicit("nope nope").is_err());
    }
}
