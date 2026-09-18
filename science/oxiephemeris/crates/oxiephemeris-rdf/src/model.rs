//! The neutral chart data model the RDF layer consumes.
//!
//! These are plain structs, deliberately decoupled from both the CLI's
//! JSON types and `oxiephemeris_astro`'s internals: a caller assembles a
//! [`ChartResource`] once, and can then render it as text, JSON, or RDF
//! without three parallel copies of the chart drifting apart.
//!
//! All angles are **degrees** (not radians) here — the unit the RDF
//! literals carry, so that no conversion hides between the model and the
//! graph.

use oxiephemeris_astro::aspects::AspectKind;
use oxiephemeris_astro::ayanamsha::Ayanamsha;
use oxiephemeris_astro::dignities::Planet;
use oxiephemeris_astro::houses::HouseSystem;
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::parts::Sect;
use oxiephemeris_astro::zodiac::Sign;

/// Which kind of chart a [`ChartResource`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartKind {
    /// A natal (birth) chart.
    Natal,
    /// A midpoint composite of two charts.
    Composite,
    /// A secondary-progressed chart.
    Progressed,
    /// A chart of transiting bodies.
    Transit,
}

impl ChartKind {
    /// All chart kinds, in a stable order.
    pub const ALL: [Self; 4] = [
        Self::Natal,
        Self::Composite,
        Self::Progressed,
        Self::Transit,
    ];

    /// A short, stable lower-case name — also the `kind` field of the
    /// chart's [`crate::ChartKey`].
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Natal => "natal",
            Self::Composite => "composite",
            Self::Progressed => "progressed",
            Self::Transit => "transit",
        }
    }
}

/// One of the classical essential-dignity tiers (dignities and
/// debilities), with its Lilly point weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DignityTier {
    /// Rules the sign. `+5`.
    Domicile,
    /// Exalted in the sign. `+4`.
    Exaltation,
    /// The sect's triplicity ruler. `+3`.
    Triplicity,
    /// Rules the Egyptian term. `+2`.
    Term,
    /// Rules the face (decan). `+1`.
    Face,
    /// Opposite its domicile. `-5`.
    Detriment,
    /// Opposite its exaltation. `-4`.
    Fall,
    /// No dignity and no debility. `-5`.
    Peregrine,
}

impl DignityTier {
    /// All eight tiers, strongest dignity first.
    pub const ALL: [Self; 8] = [
        Self::Domicile,
        Self::Exaltation,
        Self::Triplicity,
        Self::Term,
        Self::Face,
        Self::Detriment,
        Self::Fall,
        Self::Peregrine,
    ];

    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Domicile => "domicile",
            Self::Exaltation => "exaltation",
            Self::Triplicity => "triplicity",
            Self::Term => "term",
            Self::Face => "face",
            Self::Detriment => "detriment",
            Self::Fall => "fall",
            Self::Peregrine => "peregrine",
        }
    }

    /// The Lilly point weight of this tier.
    #[must_use]
    pub const fn score(self) -> i32 {
        match self {
            Self::Domicile => 5,
            Self::Exaltation => 4,
            Self::Triplicity => 3,
            Self::Term => 2,
            Self::Face => 1,
            Self::Detriment | Self::Peregrine => -5,
            Self::Fall => -4,
        }
    }
}

/// Which of the four chart angles a point is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleKind {
    /// The Ascendant (rising degree).
    Ascendant,
    /// The Midheaven (culminating degree).
    Midheaven,
    /// The Vertex.
    Vertex,
    /// The East Point (equatorial ascendant).
    EastPoint,
}

impl AngleKind {
    /// All four angles, in a stable order.
    pub const ALL: [Self; 4] = [
        Self::Ascendant,
        Self::Midheaven,
        Self::Vertex,
        Self::EastPoint,
    ];

    /// A short, stable kebab-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ascendant => "ascendant",
            Self::Midheaven => "mc",
            Self::Vertex => "vertex",
            Self::EastPoint => "east-point",
        }
    }
}

/// Which lunar node or apogee a point is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// The mean lunar node.
    MeanNode,
    /// The mean lunar apogee ("mean Lilith").
    MeanApogee,
    /// The true (osculating) lunar node.
    TrueNode,
    /// The true (osculating) lunar apogee.
    TrueApogee,
}

impl NodeKind {
    /// All four, in a stable order.
    pub const ALL: [Self; 4] = [
        Self::MeanNode,
        Self::MeanApogee,
        Self::TrueNode,
        Self::TrueApogee,
    ];

    /// A short, stable kebab-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MeanNode => "mean-node",
            Self::MeanApogee => "mean-apogee",
            Self::TrueNode => "true-node",
            Self::TrueApogee => "true-apogee",
        }
    }
}

/// Which Arabic Part (Lot) a point is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LotKind {
    /// The Lot of Fortune.
    Fortune,
    /// The Lot of Spirit.
    Spirit,
}

impl LotKind {
    /// Both lots, in a stable order.
    pub const ALL: [Self; 2] = [Self::Fortune, Self::Spirit];

    /// A short, stable lower-case name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fortune => "fortune",
            Self::Spirit => "spirit",
        }
    }
}

/// Which zodiac the chart's longitudes are expressed in.
#[derive(Debug, Clone, Copy)]
pub enum Zodiac {
    /// Equinox-anchored (Western) zodiac.
    Tropical,
    /// Star-anchored zodiac, with its ayanamsha and that ayanamsha's
    /// value at the chart epoch, degrees.
    Sidereal {
        /// The ayanamsha in use.
        ayanamsha: Ayanamsha,
        /// Its value at the chart epoch, degrees.
        degrees: f64,
    },
}

impl Zodiac {
    /// The stable name used in the chart's [`crate::ChartKey`]:
    /// `"tropical"`, or the ayanamsha's own name.
    ///
    /// Returns `"sidereal-custom"` for [`Ayanamsha::Custom`], which has no
    /// published name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Tropical => "tropical",
            Self::Sidereal { ayanamsha, .. } => ayanamsha.name().unwrap_or("sidereal-custom"),
        }
    }
}

/// The software and data that produced a chart — the PROV-O inputs.
#[derive(Debug, Clone)]
pub struct Provenance {
    /// Software name, e.g. `"oxiephemeris"`.
    pub software_name: String,
    /// Software version, e.g. `"0.1.1"`.
    pub software_version: String,
    /// Ephemeris label, e.g. `"DE440"`.
    pub ephemeris_label: String,
}

/// The observer's geodetic position.
#[derive(Debug, Clone, Copy)]
pub struct Observer {
    /// Latitude, degrees north.
    pub lat_deg: f64,
    /// Longitude, degrees east.
    pub lon_deg: f64,
    /// Height above the ellipsoid, metres.
    pub alt_m: f64,
}

/// One body's placement in a chart.
#[derive(Debug, Clone)]
pub struct BodyPlacement {
    /// Body display name (e.g. `"Sun"`).
    pub name: String,
    /// The dignity-bearing planet, when the body is one of the ten.
    pub planet: Option<Planet>,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
    /// Ecliptic latitude, degrees.
    pub lat_deg: f64,
    /// Daily longitude speed, degrees per day.
    pub lon_speed_deg_per_day: f64,
    /// Daily latitude speed, degrees per day.
    pub lat_speed_deg_per_day: f64,
    /// Geocentric distance, AU.
    pub distance_au: f64,
    /// One-way light time, days.
    pub light_time_days: f64,
    /// The sign the longitude falls in.
    pub sign: Sign,
    /// Position within the sign, degrees in `[0, 30)`.
    pub degrees_in_sign: f64,
    /// House occupied, `1..=12`.
    pub house: usize,
    /// Equatorial declination, degrees.
    pub declination_deg: f64,
    /// Direct / retrograde / stationary.
    pub motion: MotionState,
    /// Whether `|declination|` exceeds the obliquity.
    pub out_of_bounds: bool,
}

/// One house cusp.
#[derive(Debug, Clone, Copy)]
pub struct CuspRecord {
    /// Cusp ordinal, `1..=12`.
    pub number: usize,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
}

/// One chart angle.
#[derive(Debug, Clone, Copy)]
pub struct AngleRecord {
    /// Which angle.
    pub kind: AngleKind,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
}

/// One lunar node or apogee.
#[derive(Debug, Clone, Copy)]
pub struct NodeRecord {
    /// Which node/apogee.
    pub kind: NodeKind,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
}

/// One Arabic Part.
#[derive(Debug, Clone, Copy)]
pub struct LotRecord {
    /// Which lot.
    pub kind: LotKind,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
}

/// One aspect between two chart bodies.
#[derive(Debug, Clone)]
pub struct AspectRecord {
    /// First body's name.
    pub body1: String,
    /// Second body's name.
    pub body2: String,
    /// Which aspect.
    pub kind: AspectKind,
    /// The aspect's exact separation, degrees.
    pub exact_angle_deg: f64,
    /// Signed offset from exactness, degrees.
    pub offset_deg: f64,
    /// Whether the orb is shrinking.
    pub applying: bool,
}

/// One planet's essential-dignity assessment.
#[derive(Debug, Clone)]
pub struct DignityRecord {
    /// The planet assessed.
    pub planet: Planet,
    /// Every tier that applies.
    pub tiers: Vec<DignityTier>,
    /// The Lilly point sum.
    pub score: i32,
}

/// Which kind of two-chart comparison a [`ChartComparison`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonKind {
    /// Cross-aspects between two natal charts.
    Synastry,
    /// Transiting bodies against a natal chart.
    Transit,
    /// A secondary-progressed chart against its natal chart.
    Progression,
}

impl ComparisonKind {
    /// All comparison kinds, in a stable order.
    pub const ALL: [Self; 3] = [Self::Synastry, Self::Transit, Self::Progression];

    /// A short, stable lower-case name — also the `kind` of the
    /// comparison's [`crate::iri::ComparisonKey`].
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Synastry => "synastry",
            Self::Transit => "transit",
            Self::Progression => "progression",
        }
    }
}

/// One named point of a chart: either a body placement or a chart angle.
///
/// This is the currency of the dual-chart commands, which compute the ten
/// planets plus the Ascendant and Midheaven but neither houses nor
/// dignities.
#[derive(Debug, Clone)]
pub struct ChartPointRecord {
    /// Display name (`"Sun"`, `"ASC"`, `"MC"`).
    pub name: String,
    /// The planet, when the point is one of the ten.
    pub planet: Option<Planet>,
    /// The angle, when the point is an Ascendant or Midheaven.
    pub angle: Option<AngleKind>,
    /// Ecliptic longitude, degrees.
    pub lon_deg: f64,
    /// Daily longitude speed, degrees per day (zero for angles).
    pub speed_deg_per_day: f64,
    /// The sign the longitude falls in.
    pub sign: Sign,
    /// Position within the sign, degrees in `[0, 30)`.
    pub degrees_in_sign: f64,
    /// Direct / retrograde / stationary.
    pub motion: MotionState,
}

/// One side of a [`ChartComparison`]: a chart, described by the points the
/// comparison actually needed.
///
/// RDF is open-world, so describing a chart by only its points asserts
/// nothing about its houses or dignities *not* existing — it merely does
/// not mention them. That is why a natal chart appearing here carries the
/// same IRI as the fully-described one `oxieph chart` emits, and the two
/// merge in a triple store.
#[derive(Debug, Clone)]
pub struct ChartSide {
    /// Which chart class this side is (natal, transit, progressed, …).
    pub kind: ChartKind,
    /// The epoch as an ISO 8601 UTC timestamp, when the side has one.
    ///
    /// A composite chart is a midpoint construction, not a moment: it has
    /// no epoch, and `None` says so rather than inventing one.
    pub epoch_utc: Option<String>,
    /// The epoch as a Julian Date in TT, when the side has one.
    pub jd_tt: Option<f64>,
    /// The observer's position, when the side has one.
    pub observer: Option<Observer>,
    /// House system used for this side's angles.
    pub house_system: HouseSystem,
    /// The points computed for this side.
    pub points: Vec<ChartPointRecord>,
}

/// One aspect between a point of chart A and a point of chart B.
#[derive(Debug, Clone)]
pub struct CrossAspectRecord {
    /// Name of the chart-A point (`"Sun"`, `"ASC"`, …).
    pub from_name: String,
    /// Name of the chart-B point.
    pub to_name: String,
    /// Which aspect.
    pub kind: AspectKind,
    /// Signed offset from exactness, degrees.
    pub offset_deg: f64,
    /// Whether the orb is shrinking.
    pub applying: bool,
}

/// A midpoint composite chart, ready to be turned into an RDF graph.
///
/// Unlike a [`ChartComparison`], a composite is a *chart in its own right*
/// — its points are the midpoints of two other charts' points — so it is
/// modelled as an [`ChartKind::Composite`] chart that
/// `prov:wasDerivedFrom` the two source charts, not as a relation between
/// them.
#[derive(Debug, Clone)]
pub struct CompositeChartResource {
    /// The composite chart itself (`kind` is [`ChartKind::Composite`],
    /// `epoch_utc`/`jd_tt`/`observer` are `None`).
    pub side: ChartSide,
    /// Aspects among the composite's own points.
    pub aspects: Vec<CrossAspectRecord>,
    /// IRIs of the two charts this composite was derived from.
    pub source_charts: Vec<String>,
    /// What computed this composite, and from what data.
    pub provenance: Provenance,
}

/// A comparison of two charts, ready to be turned into an RDF graph.
#[derive(Debug, Clone)]
pub struct ChartComparison {
    /// Synastry, transit, or progression.
    pub kind: ComparisonKind,
    /// The moving (left) chart.
    pub chart_a: ChartSide,
    /// The fixed (right) chart.
    pub chart_b: ChartSide,
    /// The cross-aspects between them.
    pub cross_aspects: Vec<CrossAspectRecord>,
    /// Tropical years elapsed, for a progression only.
    pub elapsed_years: Option<f64>,
    /// What computed this comparison, and from what data.
    pub provenance: Provenance,
}

/// A chart's element/modality balance.
#[derive(Debug, Clone, Copy)]
pub struct DistributionRecord {
    /// Counts in Fire, Earth, Air, Water order.
    pub elements: [usize; 4],
    /// Counts in Cardinal, Fixed, Mutable order.
    pub modalities: [usize; 3],
}

/// A complete chart, ready to be turned into an RDF graph.
#[derive(Debug, Clone)]
pub struct ChartResource {
    /// Natal / composite / progressed / transit.
    pub kind: ChartKind,
    /// The epoch as an ISO 8601 UTC timestamp, e.g.
    /// `"1970-01-01T00:00:00Z"` — serialized as `xsd:dateTime`.
    pub epoch_utc: String,
    /// The epoch as a Julian Date in TT.
    pub jd_tt: f64,
    /// The observer's position, when the chart has one.
    pub observer: Option<Observer>,
    /// House system used for the cusps.
    pub house_system: HouseSystem,
    /// Tropical or sidereal.
    pub zodiac: Zodiac,
    /// Rulership scheme used for the dignities (`"traditional"` /
    /// `"modern"`).
    pub rulership: String,
    /// Whether the chart is of the day or of the night.
    pub sect: Sect,
    /// The bodies.
    pub bodies: Vec<BodyPlacement>,
    /// The twelve house cusps.
    pub cusps: Vec<CuspRecord>,
    /// The chart angles.
    pub angles: Vec<AngleRecord>,
    /// The lunar nodes and apogees.
    pub nodes: Vec<NodeRecord>,
    /// The Arabic Parts.
    pub lots: Vec<LotRecord>,
    /// The aspects between bodies.
    pub aspects: Vec<AspectRecord>,
    /// Per-planet essential dignity.
    pub dignities: Vec<DignityRecord>,
    /// Element/modality balance.
    pub distribution: DistributionRecord,
    /// Years elapsed, for a progressed chart only.
    pub elapsed_years: Option<f64>,
    /// What computed this chart, and from what data.
    pub provenance: Provenance,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dignity_tier_scores_match_lilly() {
        assert_eq!(DignityTier::Domicile.score(), 5);
        assert_eq!(DignityTier::Exaltation.score(), 4);
        assert_eq!(DignityTier::Triplicity.score(), 3);
        assert_eq!(DignityTier::Term.score(), 2);
        assert_eq!(DignityTier::Face.score(), 1);
        assert_eq!(DignityTier::Detriment.score(), -5);
        assert_eq!(DignityTier::Fall.score(), -4);
        assert_eq!(DignityTier::Peregrine.score(), -5);
    }

    #[test]
    fn every_enum_name_is_unique_within_its_scheme() {
        fn all_unique(names: &[&str]) -> bool {
            let mut sorted = names.to_vec();
            sorted.sort_unstable();
            let total = sorted.len();
            sorted.dedup();
            sorted.len() == total
        }
        assert!(all_unique(&DignityTier::ALL.map(DignityTier::name)));
        assert!(all_unique(&AngleKind::ALL.map(AngleKind::name)));
        assert!(all_unique(&NodeKind::ALL.map(NodeKind::name)));
        assert!(all_unique(&LotKind::ALL.map(LotKind::name)));
        assert!(all_unique(&ChartKind::ALL.map(ChartKind::name)));
    }

    #[test]
    fn zodiac_names_follow_the_ayanamsha() {
        assert_eq!(Zodiac::Tropical.name(), "tropical");
        assert_eq!(
            Zodiac::Sidereal {
                ayanamsha: Ayanamsha::Lahiri,
                degrees: 23.85,
            }
            .name(),
            "lahiri"
        );
        assert_eq!(
            Zodiac::Sidereal {
                ayanamsha: Ayanamsha::Custom {
                    t0_jd_tt: 2_451_545.0,
                    value_at_t0_rad: 0.0,
                },
                degrees: 0.0,
            }
            .name(),
            "sidereal-custom"
        );
    }
}
