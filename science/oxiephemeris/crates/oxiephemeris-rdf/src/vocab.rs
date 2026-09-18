//! The `OxiEphemeris` astrology vocabulary and the external vocabularies it
//! aligns to.
//!
//! # Two namespaces, two lifetimes
//!
//! * The **vocabulary** namespaces ([`ns::OXA`] terms, [`ns::OXC`]
//!   concepts, [`ns::OXS`] concept schemes) are **fixed constants**. A
//!   published vocabulary IRI must never move: two graphs minted by
//!   different publishers still have to agree that
//!   `oxa:eclipticLongitude` means the same thing.
//! * The **instance** namespace (chart resources) defaults to
//!   [`ns::DEFAULT_INSTANCE_BASE`] but is caller-configurable (see
//!   [`crate::iri`]), because instance IRIs belong to whoever publishes
//!   the data.
//!
//! # External alignment
//!
//! | Vocabulary | Used for |
//! |---|---|
//! | RDF / RDFS / OWL | class & property declarations |
//! | SKOS | the concept schemes (signs, planets, aspects, …) |
//! | PROV-O | which ephemeris and which software produced a chart |
//! | QUDT | units on quantity properties |
//! | WGS84 `geo` | the observer's position |
//! | OWL-Time | the chart epoch as a `time:Instant` |
//! | Dublin Core Terms | software version |
//! | Wikidata | `skos:exactMatch` for planets and signs |
//!
//! ## Evidence for the external IRIs used here
//!
//! Only QUDT units that actually resolve are asserted: `unit:DEG`,
//! `unit:AU` and `unit:RAD` return HTTP 200; there is **no**
//! `unit:DEG-PER-DAY` (HTTP 404), so daily-speed properties carry a
//! documented `rdfs:comment` instead of a wrong `qudt:hasUnit`.
//!
//! Wikidata alignment is asserted **only** for the ten planets and the
//! twelve signs, whose QIDs were verified against the Wikidata API /
//! query service. Wikidata has no consistent class for astrological
//! aspects, elements or modalities, so those concepts get **no**
//! `skos:exactMatch` rather than a guessed one.

use oxrdf::NamedNode;

/// Namespace IRI strings.
pub mod ns {
    /// Ontology terms (classes and properties).
    pub const OXA: &str = "https://cooljapan.tech/ns/oxiephemeris/astro#";
    /// SKOS concepts.
    pub const OXC: &str = "https://cooljapan.tech/ns/oxiephemeris/concept/";
    /// SKOS concept schemes.
    pub const OXS: &str = "https://cooljapan.tech/ns/oxiephemeris/scheme/";
    /// Default base for *instance* (chart) IRIs; caller-overridable.
    pub const DEFAULT_INSTANCE_BASE: &str = "https://cooljapan.tech/id/oxiephemeris/";

    /// RDF.
    pub const RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
    /// RDF Schema.
    pub const RDFS: &str = "http://www.w3.org/2000/01/rdf-schema#";
    /// OWL.
    pub const OWL: &str = "http://www.w3.org/2002/07/owl#";
    /// XML Schema datatypes.
    pub const XSD: &str = "http://www.w3.org/2001/XMLSchema#";
    /// SKOS core.
    pub const SKOS: &str = "http://www.w3.org/2004/02/skos/core#";
    /// PROV-O.
    pub const PROV: &str = "http://www.w3.org/ns/prov#";
    /// QUDT schema.
    pub const QUDT: &str = "http://qudt.org/schema/qudt/";
    /// QUDT unit vocabulary.
    pub const UNIT: &str = "http://qudt.org/vocab/unit/";
    /// WGS84 geo positioning.
    pub const GEO: &str = "http://www.w3.org/2003/01/geo/wgs84_pos#";
    /// OWL-Time.
    pub const TIME: &str = "http://www.w3.org/2006/time#";
    /// Dublin Core Terms.
    pub const DCT: &str = "http://purl.org/dc/terms/";
    /// Wikidata entities.
    pub const WD: &str = "http://www.wikidata.org/entity/";
}

/// Prefix bindings registered on every Turtle/TriG serialization.
///
/// # One prefix per concept scheme, and no `oxc:`
///
/// A concept IRI is `{OXC}{scheme-path}/{local}`. Binding `oxc:` to
/// [`ns::OXC`] alone would force the serializer to escape the inner slash
/// (`oxc:sign\/Scorpio`) — legal Turtle, but unreadable. Binding one
/// prefix per scheme namespace instead yields `sign:Scorpio`,
/// `planet:Mars`, `aspect:trine`.
///
/// `oxc:` is deliberately **not** bound alongside them: it is a proper
/// prefix of every scheme namespace, so both would match a concept IRI and
/// the serializer's choice between them would be an implementation detail.
/// [`ns::OXC`] remains the constant that *builds* those IRIs.
pub const PREFIXES: &[(&str, &str)] = &[
    ("oxa", ns::OXA),
    ("oxs", ns::OXS),
    // One namespace per SKOS concept scheme.
    (
        "sign",
        "https://cooljapan.tech/ns/oxiephemeris/concept/sign/",
    ),
    (
        "planet",
        "https://cooljapan.tech/ns/oxiephemeris/concept/planet/",
    ),
    (
        "aspect",
        "https://cooljapan.tech/ns/oxiephemeris/concept/aspect/",
    ),
    (
        "element",
        "https://cooljapan.tech/ns/oxiephemeris/concept/element/",
    ),
    (
        "modality",
        "https://cooljapan.tech/ns/oxiephemeris/concept/modality/",
    ),
    (
        "hsys",
        "https://cooljapan.tech/ns/oxiephemeris/concept/house-system/",
    ),
    (
        "sect",
        "https://cooljapan.tech/ns/oxiephemeris/concept/sect/",
    ),
    (
        "motion",
        "https://cooljapan.tech/ns/oxiephemeris/concept/motion/",
    ),
    (
        "decl",
        "https://cooljapan.tech/ns/oxiephemeris/concept/declination-aspect/",
    ),
    (
        "ayan",
        "https://cooljapan.tech/ns/oxiephemeris/concept/ayanamsha/",
    ),
    (
        "dignity",
        "https://cooljapan.tech/ns/oxiephemeris/concept/dignity/",
    ),
    (
        "angle",
        "https://cooljapan.tech/ns/oxiephemeris/concept/angle/",
    ),
    (
        "node",
        "https://cooljapan.tech/ns/oxiephemeris/concept/node/",
    ),
    ("lot", "https://cooljapan.tech/ns/oxiephemeris/concept/lot/"),
    // External vocabularies.
    ("rdf", ns::RDF),
    ("rdfs", ns::RDFS),
    ("owl", ns::OWL),
    ("xsd", ns::XSD),
    ("skos", ns::SKOS),
    ("prov", ns::PROV),
    ("qudt", ns::QUDT),
    ("unit", ns::UNIT),
    ("geo", ns::GEO),
    ("time", ns::TIME),
    ("dct", ns::DCT),
    ("wd", ns::WD),
];

/// Builds a term IRI in a namespace. `new_unchecked` is sound here
/// because every call site concatenates a constant namespace with an
/// ASCII local name; `tests::every_constant_iri_is_valid` and
/// `tests::built_iris_are_valid` re-parse them strictly.
fn iri(namespace: &str, local: &str) -> NamedNode {
    NamedNode::new_unchecked(format!("{namespace}{local}"))
}

/// A SKOS concept IRI: `oxc:{path}/{local}` (e.g. `oxc:planet/Mars`).
#[must_use]
pub fn concept(path: &str, local: &str) -> NamedNode {
    iri(ns::OXC, &format!("{path}/{local}"))
}

/// A SKOS concept-scheme IRI: `oxs:{name}` (e.g. `oxs:ZodiacSigns`).
#[must_use]
pub fn scheme(name: &str) -> NamedNode {
    iri(ns::OXS, name)
}

/// A Wikidata entity IRI from a QID (e.g. `"Q111"` → `wd:Q111`).
#[must_use]
pub fn wikidata(qid: &str) -> NamedNode {
    iri(ns::WD, qid)
}

/// Concept-scheme path segments (the `{path}` of [`concept`]).
pub mod path {
    /// Zodiac signs.
    pub const SIGN: &str = "sign";
    /// Planets (the astrological seven plus the three modern).
    pub const PLANET: &str = "planet";
    /// Aspect kinds.
    pub const ASPECT: &str = "aspect";
    /// Classical elements.
    pub const ELEMENT: &str = "element";
    /// Modalities.
    pub const MODALITY: &str = "modality";
    /// House systems.
    pub const HOUSE_SYSTEM: &str = "house-system";
    /// Chart sect (diurnal/nocturnal).
    pub const SECT: &str = "sect";
    /// Motion states.
    pub const MOTION: &str = "motion";
    /// Declination aspects.
    pub const DECLINATION_ASPECT: &str = "declination-aspect";
    /// Ayanamshas.
    pub const AYANAMSHA: &str = "ayanamsha";
    /// Essential-dignity tiers.
    pub const DIGNITY: &str = "dignity";
    /// Chart angles (Ascendant, MC, …).
    pub const ANGLE: &str = "angle";
    /// Lunar nodes and apogees.
    pub const NODE: &str = "node";
    /// Arabic Parts (Lots).
    pub const LOT: &str = "lot";
}

/// Macro defining a module of `NamedNodeRef` constants over one namespace.
macro_rules! terms {
    ($module:ident, $base:expr, $( $(#[$doc:meta])* $name:ident => $local:literal ),* $(,)?) => {
        pub mod $module {
            use oxrdf::NamedNodeRef;
            $(
                $(#[$doc])*
                pub const $name: NamedNodeRef<'_> =
                    NamedNodeRef::new_unchecked(concat!($base, $local));
            )*
            /// Every constant in this module, for validation and closure
            /// tests.
            pub const ALL: &[NamedNodeRef<'_>] = &[$($name),*];
        }
    };
}

terms!(
    oxa,
    "https://cooljapan.tech/ns/oxiephemeris/astro#",
    // --- classes ---
    /// A computed astrological chart.
    CHART => "Chart",
    /// A natal (birth) chart.
    NATAL_CHART => "NatalChart",
    /// A midpoint composite chart.
    COMPOSITE_CHART => "CompositeChart",
    /// A secondary-progressed chart.
    PROGRESSED_CHART => "ProgressedChart",
    /// A chart of transiting bodies.
    TRANSIT_CHART => "TransitChart",
    /// A comparison of two charts (synastry, transit, progression).
    CHART_COMPARISON => "ChartComparison",
    /// Cross-aspects between two natal charts.
    SYNASTRY_COMPARISON => "SynastryComparison",
    /// Aspects of transiting bodies to a natal chart.
    TRANSIT_COMPARISON => "TransitComparison",
    /// Aspects of a secondary-progressed chart to its natal chart.
    PROGRESSION_COMPARISON => "ProgressionComparison",
    /// Any named point of a chart: a body placement or a chart angle.
    CHART_POINT => "ChartPoint",
    /// An aspect whose two endpoints lie in *different* charts.
    CROSS_ASPECT => "CrossAspect",
    /// One body's placement within a chart.
    BODY_POSITION => "BodyPosition",
    /// One house cusp.
    HOUSE_CUSP => "HouseCusp",
    /// A chart angle (Ascendant, MC, Vertex, East Point).
    CHART_ANGLE => "ChartAngle",
    /// A lunar node or apogee.
    LUNAR_NODE => "LunarNode",
    /// An Arabic Part (Lot).
    LOT => "Lot",
    /// One aspect between two chart points (an n-ary relation).
    ASPECT_OCCURRENCE => "AspectOccurrence",
    /// One planet's essential-dignity assessment.
    DIGNITY_ASSESSMENT => "DignityAssessment",
    /// A chart's element/modality balance.
    DISTRIBUTION => "Distribution",
    /// The observer's geographic position.
    OBSERVER => "Observer",
    // --- concept classes (all rdfs:subClassOf skos:Concept) ---
    /// A zodiac sign concept.
    ZODIAC_SIGN => "ZodiacSign",
    /// A planet concept.
    PLANET => "Planet",
    /// An aspect-kind concept.
    ASPECT_KIND => "AspectKind",
    /// A classical-element concept.
    ELEMENT => "Element",
    /// A modality concept.
    MODALITY => "Modality",
    /// A house-system concept.
    HOUSE_SYSTEM => "HouseSystem",
    /// A sect concept.
    SECT => "Sect",
    /// A motion-state concept.
    MOTION_STATE => "MotionState",
    /// A declination-aspect concept.
    DECLINATION_ASPECT => "DeclinationAspect",
    /// An ayanamsha concept.
    AYANAMSHA => "Ayanamsha",
    /// An essential-dignity tier concept.
    DIGNITY_TIER => "DignityTier",
    /// A chart-angle-kind concept.
    ANGLE_KIND => "AngleKind",
    /// A lunar-node-kind concept.
    NODE_KIND => "NodeKind",
    /// A lot-kind concept.
    LOT_KIND => "LotKind",
    // --- object properties ---
    /// Links a chart to one of its body positions.
    HAS_BODY_POSITION => "hasBodyPosition",
    /// Links a chart to one of its house cusps.
    HAS_HOUSE_CUSP => "hasHouseCusp",
    /// Links a chart to one of its angles.
    HAS_ANGLE => "hasAngle",
    /// Links a chart to one of its lunar nodes/apogees.
    HAS_LUNAR_NODE => "hasLunarNode",
    /// Links a chart to one of its Lots.
    HAS_LOT => "hasLot",
    /// Links a chart to one of its aspect occurrences.
    HAS_ASPECT => "hasAspect",
    /// Links a chart to one planet's dignity assessment.
    HAS_DIGNITY => "hasDignity",
    /// Links a chart to its element/modality distribution.
    HAS_DISTRIBUTION => "hasDistribution",
    /// Links a chart to the observer's position.
    HAS_OBSERVER => "hasObserver",
    /// Links a chart to its epoch instant.
    AT_EPOCH => "atEpoch",
    /// The body a position or dignity assessment is about.
    BODY => "body",
    /// The sign a point falls in.
    IN_SIGN => "inSign",
    /// The motion state of a body position.
    MOTION => "motion",
    /// The first body of an aspect occurrence.
    FIRST_BODY => "firstBody",
    /// The second body of an aspect occurrence.
    SECOND_BODY => "secondBody",
    /// The originating chart point of an aspect. Unlike `firstBody`, which
    /// names the *planet concept*, this names the concrete placement, so an
    /// aspect can involve an Ascendant and can say which chart it came from.
    FROM_POINT => "fromPoint",
    /// The receiving chart point of an aspect.
    TO_POINT => "toPoint",
    /// The moving (left) chart of a comparison.
    CHART_A => "chartA",
    /// The fixed (right) chart of a comparison.
    CHART_B => "chartB",
    /// Links a comparison to one of its cross-aspects.
    HAS_CROSS_ASPECT => "hasCrossAspect",
    /// The kind of an aspect occurrence.
    ASPECT_KIND_OF => "aspectKind",
    /// The kind of a chart angle.
    ANGLE_KIND_OF => "angleKind",
    /// The kind of a lunar node.
    NODE_KIND_OF => "nodeKind",
    /// The kind of a Lot.
    LOT_KIND_OF => "lotKind",
    /// The chart's sect.
    HAS_SECT => "hasSect",
    /// The house system the cusps were computed with.
    HOUSE_SYSTEM_OF => "houseSystem",
    /// The ayanamsha, when the chart is sidereal.
    AYANAMSHA_OF => "ayanamsha",
    /// A sign's classical element.
    ELEMENT_OF => "element",
    /// A sign's modality.
    MODALITY_OF => "modality",
    /// A sign's traditional (seven-planet) domicile ruler.
    TRADITIONAL_RULER => "traditionalRuler",
    /// A sign's modern domicile ruler.
    MODERN_RULER => "modernRuler",
    /// The planet exalted in a sign.
    EXALTED_PLANET => "exaltedPlanet",
    /// The day triplicity ruler of a sign.
    DAY_TRIPLICITY_RULER => "dayTriplicityRuler",
    /// The night triplicity ruler of a sign.
    NIGHT_TRIPLICITY_RULER => "nightTriplicityRuler",
    /// The participating triplicity ruler of a sign.
    PARTICIPATING_TRIPLICITY_RULER => "participatingTriplicityRuler",
    /// A dignity tier held by a dignity assessment.
    HOLDS_TIER => "holdsTier",
    // --- datatype properties ---
    /// Ecliptic longitude, degrees.
    ECLIPTIC_LONGITUDE => "eclipticLongitude",
    /// Ecliptic latitude, degrees.
    ECLIPTIC_LATITUDE => "eclipticLatitude",
    /// Position within the sign, degrees in `[0, 30)`.
    DEGREES_IN_SIGN => "degreesInSign",
    /// Geocentric distance, astronomical units.
    DISTANCE_AU => "distanceAu",
    /// Daily longitude speed, degrees per day.
    LONGITUDE_SPEED => "longitudeSpeed",
    /// Daily latitude speed, degrees per day.
    LATITUDE_SPEED => "latitudeSpeed",
    /// Equatorial declination, degrees.
    DECLINATION => "declination",
    /// One-way light time, days.
    LIGHT_TIME_DAYS => "lightTimeDays",
    /// The house a body occupies, `1..=12`.
    HOUSE_NUMBER => "houseNumber",
    /// The ordinal of a house cusp, `1..=12`.
    CUSP_NUMBER => "cuspNumber",
    /// Signed offset from aspect exactness, degrees.
    ORB => "orb",
    /// An aspect kind's exact separation, degrees.
    EXACT_ANGLE => "exactAngle",
    /// An aspect kind's default orb, degrees.
    DEFAULT_ORB => "defaultOrb",
    /// Whether an aspect is applying (orb shrinking).
    IS_APPLYING => "isApplying",
    /// Whether a body is retrograde.
    IS_RETROGRADE => "isRetrograde",
    /// Whether a body is out of bounds in declination.
    IS_OUT_OF_BOUNDS => "isOutOfBounds",
    /// Lilly point sum of one planet's dignity assessment.
    DIGNITY_SCORE => "dignityScore",
    /// The Lilly point weight a single dignity tier contributes. Distinct
    /// from `dignityScore`, which is the *sum* over an assessment.
    TIER_WEIGHT => "tierWeight",
    /// The chart epoch as a Julian Date in TT.
    JULIAN_DATE_TT => "julianDateTT",
    /// The rulership scheme the dignities were scored under.
    RULERSHIP_SCHEME => "rulershipScheme",
    /// The ayanamsha value at the epoch, degrees.
    AYANAMSHA_DEGREES => "ayanamshaDegrees",
    /// Years elapsed, for a progressed chart.
    ELAPSED_YEARS => "elapsedYears",
    /// A sign's 0-based index (Aries = 0).
    SIGN_INDEX => "signIndex",
    /// The degree of greatest exaltation within a sign.
    EXALTATION_DEGREE => "exaltationDegree",
    /// Count of bodies in Fire signs.
    FIRE_COUNT => "fireCount",
    /// Count of bodies in Earth signs.
    EARTH_COUNT => "earthCount",
    /// Count of bodies in Air signs.
    AIR_COUNT => "airCount",
    /// Count of bodies in Water signs.
    WATER_COUNT => "waterCount",
    /// Count of bodies in Cardinal signs.
    CARDINAL_COUNT => "cardinalCount",
    /// Count of bodies in Fixed signs.
    FIXED_COUNT => "fixedCount",
    /// Count of bodies in Mutable signs.
    MUTABLE_COUNT => "mutableCount",
);

terms!(
    skos,
    "http://www.w3.org/2004/02/skos/core#",
    /// `skos:Concept`.
    CONCEPT => "Concept",
    /// `skos:ConceptScheme`.
    CONCEPT_SCHEME => "ConceptScheme",
    /// `skos:inScheme`.
    IN_SCHEME => "inScheme",
    /// `skos:hasTopConcept`.
    HAS_TOP_CONCEPT => "hasTopConcept",
    /// `skos:topConceptOf`.
    TOP_CONCEPT_OF => "topConceptOf",
    /// `skos:prefLabel`.
    PREF_LABEL => "prefLabel",
    /// `skos:altLabel`.
    ALT_LABEL => "altLabel",
    /// `skos:notation`.
    NOTATION => "notation",
    /// `skos:definition`.
    DEFINITION => "definition",
    /// `skos:exactMatch`.
    EXACT_MATCH => "exactMatch",
);

terms!(
    owl,
    "http://www.w3.org/2002/07/owl#",
    /// `owl:Ontology`.
    ONTOLOGY => "Ontology",
    /// `owl:Class`.
    CLASS => "Class",
    /// `owl:ObjectProperty`.
    OBJECT_PROPERTY => "ObjectProperty",
    /// `owl:DatatypeProperty`.
    DATATYPE_PROPERTY => "DatatypeProperty",
);

terms!(
    prov,
    "http://www.w3.org/ns/prov#",
    /// `prov:Entity`.
    ENTITY => "Entity",
    /// `prov:Activity`.
    ACTIVITY => "Activity",
    /// `prov:SoftwareAgent`.
    SOFTWARE_AGENT => "SoftwareAgent",
    /// `prov:wasGeneratedBy`.
    WAS_GENERATED_BY => "wasGeneratedBy",
    /// `prov:wasDerivedFrom`.
    WAS_DERIVED_FROM => "wasDerivedFrom",
    /// `prov:wasAssociatedWith`.
    WAS_ASSOCIATED_WITH => "wasAssociatedWith",
    /// `prov:used`.
    USED => "used",
);

terms!(
    qudt,
    "http://qudt.org/schema/qudt/",
    /// `qudt:hasUnit`.
    HAS_UNIT => "hasUnit",
);

terms!(
    unit,
    "http://qudt.org/vocab/unit/",
    /// Degree of arc. Resolves (HTTP 200).
    DEG => "DEG",
    /// Astronomical unit. Resolves (HTTP 200).
    AU => "AU",
);

terms!(
    geo,
    "http://www.w3.org/2003/01/geo/wgs84_pos#",
    /// `geo:Point`.
    POINT => "Point",
    /// `geo:lat`.
    LAT => "lat",
    /// `geo:long`.
    LONG => "long",
    /// `geo:alt`.
    ALT => "alt",
);

terms!(
    time,
    "http://www.w3.org/2006/time#",
    /// `time:Instant`.
    INSTANT => "Instant",
    /// `time:inXSDDateTimeStamp`.
    IN_XSD_DATE_TIME_STAMP => "inXSDDateTimeStamp",
);

terms!(
    dct,
    "http://purl.org/dc/terms/",
    /// `dct:hasVersion`.
    HAS_VERSION => "hasVersion",
    /// `dct:title`.
    TITLE => "title",
    /// `dct:license`.
    LICENSE => "license",
);

// The XSD datatype IRIs, in the *borrowed* form the ontology's `const`
// property table needs. This is a local copy of the subset this crate
// emits, kept so the table does not depend on which spelling the model's
// own `oxrdf::vocab::xsd` uses. `tests::xsd_constants_match_the_model`
// pins the two sets against each other, so they cannot drift.
terms!(
    xsd,
    "http://www.w3.org/2001/XMLSchema#",
    /// `xsd:boolean`.
    BOOLEAN => "boolean",
    /// `xsd:dateTimeStamp`.
    DATE_TIME_STAMP => "dateTimeStamp",
    /// `xsd:double`.
    DOUBLE => "double",
    /// `xsd:integer`.
    INTEGER => "integer",
    /// `xsd:string`.
    STRING => "string",
);

/// The verified Wikidata QID for each planet, by
/// [`oxiephemeris_astro::dignities::Planet::name`].
///
/// Confirmed against the Wikidata API (`wbgetentities`, English labels).
pub const PLANET_WIKIDATA: &[(&str, &str)] = &[
    ("Sun", "Q525"),
    ("Moon", "Q405"),
    ("Mercury", "Q308"),
    ("Venus", "Q313"),
    ("Mars", "Q111"),
    ("Jupiter", "Q319"),
    ("Saturn", "Q193"),
    ("Uranus", "Q324"),
    ("Neptune", "Q332"),
    ("Pluto", "Q339"),
];

/// The verified Wikidata QID for each zodiac sign, by
/// [`oxiephemeris_astro::zodiac::Sign::name`].
///
/// Confirmed by querying the Wikidata Query Service for every instance of
/// `wd:Q1795024` ("occidental astrological sign"); the thirteenth
/// instance, Ophiuchus, is deliberately excluded — it is not one of the
/// twelve equal signs of the tropical zodiac this crate models.
pub const SIGN_WIKIDATA: &[(&str, &str)] = &[
    ("Aries", "Q32067"),
    ("Taurus", "Q164016"),
    ("Gemini", "Q129214"),
    ("Cancer", "Q161701"),
    ("Leo", "Q159816"),
    ("Virgo", "Q134061"),
    ("Libra", "Q134394"),
    ("Scorpio", "Q134398"),
    ("Sagittarius", "Q2194186"),
    ("Capricorn", "Q164272"),
    ("Aquarius", "Q162119"),
    ("Pisces", "Q1254190"),
];

/// Looks up a verified Wikidata QID by planet name.
#[must_use]
pub fn planet_qid(name: &str) -> Option<&'static str> {
    PLANET_WIKIDATA
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, q)| *q)
}

/// Looks up a verified Wikidata QID by sign name.
#[must_use]
pub fn sign_qid(name: &str) -> Option<&'static str> {
    SIGN_WIKIDATA
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, q)| *q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiephemeris_astro::dignities::Planet;
    use oxiephemeris_astro::zodiac::Sign;

    /// Every `new_unchecked` constant must survive a strict re-parse.
    /// This is what makes `new_unchecked` safe to use above: a typo in any
    /// IRI fails here rather than emitting a corrupt graph.
    #[test]
    fn every_constant_iri_is_valid() {
        let groups: [&[oxrdf::NamedNodeRef<'_>]; 10] = [
            oxa::ALL,
            skos::ALL,
            owl::ALL,
            prov::ALL,
            qudt::ALL,
            unit::ALL,
            geo::ALL,
            time::ALL,
            dct::ALL,
            xsd::ALL,
        ];
        for group in groups {
            for term in group {
                let s = term.as_str();
                assert!(
                    NamedNode::new(s).is_ok(),
                    "constant IRI is not a valid IRI: {s}"
                );
            }
        }
    }

    /// The borrowed `xsd::` constants above must denote exactly the IRIs
    /// the RDF model's own vocabulary does, or the ontology would declare
    /// ranges that no reasoner recognizes.
    #[test]
    fn xsd_constants_match_the_model() {
        use oxrdf::vocab::xsd as model_xsd;
        let pairs = [
            (xsd::BOOLEAN, &model_xsd::BOOLEAN),
            (xsd::DATE_TIME_STAMP, &model_xsd::DATE_TIME_STAMP),
            (xsd::DOUBLE, &model_xsd::DOUBLE),
            (xsd::INTEGER, &model_xsd::INTEGER),
            (xsd::STRING, &model_xsd::STRING),
        ];
        for (ours, theirs) in pairs {
            assert_eq!(ours.as_str(), theirs.as_str(), "xsd constant drifted");
        }
    }

    #[test]
    fn built_iris_are_valid() {
        for sign in Sign::ALL {
            assert!(NamedNode::new(concept(path::SIGN, sign.name()).as_str().to_owned()).is_ok());
        }
        assert!(NamedNode::new(scheme("ZodiacSigns").as_str().to_owned()).is_ok());
        assert!(NamedNode::new(wikidata("Q111").as_str().to_owned()).is_ok());
    }

    #[test]
    fn oxa_terms_are_unique() {
        let mut seen: Vec<&str> = oxa::ALL.iter().map(|term| term.as_str()).collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "duplicate IRI among oxa:: constants");
    }

    #[test]
    fn every_planet_has_a_verified_qid() {
        let planets = [
            Planet::Sun,
            Planet::Moon,
            Planet::Mercury,
            Planet::Venus,
            Planet::Mars,
            Planet::Jupiter,
            Planet::Saturn,
            Planet::Uranus,
            Planet::Neptune,
            Planet::Pluto,
        ];
        for planet in planets {
            assert!(
                planet_qid(planet.name()).is_some(),
                "no verified QID for {}",
                planet.name()
            );
        }
    }

    #[test]
    fn every_sign_has_a_verified_qid() {
        for sign in Sign::ALL {
            assert!(
                sign_qid(sign.name()).is_some(),
                "no verified QID for {}",
                sign.name()
            );
        }
        assert_eq!(SIGN_WIKIDATA.len(), 12, "Ophiuchus must not be included");
    }

    #[test]
    fn prefixes_cover_the_namespaces_we_emit() {
        for (_, iri) in PREFIXES {
            assert!(iri.ends_with(['#', '/']), "namespace must end in # or /");
        }
    }

    /// The scheme namespaces in [`PREFIXES`] are hand-written literals
    /// (Rust cannot `concat!` a `const &str`). This pins each one against
    /// the IRI that [`concept`] actually builds, so a typo in a literal is
    /// a test failure rather than a silently un-abbreviated IRI.
    #[test]
    fn every_scheme_path_has_a_matching_prefix_namespace() {
        let paths = [
            path::SIGN,
            path::PLANET,
            path::ASPECT,
            path::ELEMENT,
            path::MODALITY,
            path::HOUSE_SYSTEM,
            path::SECT,
            path::MOTION,
            path::DECLINATION_ASPECT,
            path::AYANAMSHA,
            path::DIGNITY,
            path::ANGLE,
            path::NODE,
            path::LOT,
        ];
        for p in paths {
            let expected = format!("{}{p}/", ns::OXC);
            assert!(
                PREFIXES.iter().any(|(_, iri)| *iri == expected),
                "no prefix bound to {expected}"
            );
            // And a concept in that scheme really lives under it.
            assert!(concept(p, "X").as_str().starts_with(&expected));
        }
    }

    /// `oxc:` must not be bound: it is a proper prefix of every scheme
    /// namespace, so binding both would make the serializer's choice
    /// ambiguous.
    #[test]
    fn bare_concept_namespace_is_not_bound() {
        assert!(PREFIXES.iter().all(|(_, iri)| *iri != ns::OXC));
    }

    #[test]
    fn prefix_names_are_unique() {
        let mut names: Vec<&str> = PREFIXES.iter().map(|(p, _)| *p).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate prefix name");
    }
}
