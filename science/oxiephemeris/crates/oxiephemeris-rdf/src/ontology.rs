//! The `oxa:` ontology: every class and property this crate emits,
//! declared in RDFS/OWL with labels, ranges, and QUDT units.
//!
//! # `rdfs:domain` is asserted only when it is true
//!
//! `rdfs:domain` is an *entailment*, not a hint: stating
//! `oxa:eclipticLongitude rdfs:domain oxa:BodyPosition` licenses a
//! reasoner to infer that every house cusp is a `BodyPosition`. Because
//! several properties are shared across resource kinds (a longitude
//! belongs to bodies, cusps, angles, nodes and lots alike), those
//! properties carry **no** `rdfs:domain` rather than a false one. Ranges,
//! which are unambiguous, are always given.
//!
//! # Units
//!
//! Angles carry `qudt:hasUnit unit:DEG` and distances `unit:AU`; both IRIs
//! were checked to resolve. QUDT publishes no `unit:DEG-PER-DAY`, so the
//! daily-speed properties describe their unit in an `rdfs:comment`
//! instead of pointing at an IRI that does not exist.

use oxrdf::vocab::{rdf, rdfs};
use oxrdf::{Graph, NamedNode, NamedNodeRef};

use crate::builder::{add, add_labels, lang, Lang};
// `xsd` is this crate's own copy of the XSD datatype IRIs, in the
// `NamedNodeRef` form the `const` property table below needs.
use crate::vocab::{dct, geo, ns, owl, oxa, prov, qudt, skos, time, unit, xsd};

/// The ontology's own IRI (the `oxa:` namespace without its `#`).
#[must_use]
pub fn ontology_iri() -> NamedNode {
    NamedNode::new_unchecked(ns::OXA.trim_end_matches('#').to_owned())
}

/// A class declaration.
struct ClassDef {
    term: NamedNodeRef<'static>,
    en: &'static str,
    ja: &'static str,
    super_class: Option<NamedNodeRef<'static>>,
}

/// A property declaration. `domain` is `None` when the property is shared
/// across resource kinds (see the module doc).
struct PropDef {
    term: NamedNodeRef<'static>,
    en: &'static str,
    ja: &'static str,
    object_property: bool,
    domain: Option<NamedNodeRef<'static>>,
    range: NamedNodeRef<'static>,
    unit: Option<NamedNodeRef<'static>>,
    comment: Option<&'static str>,
}

const CLASSES: &[ClassDef] = &[
    ClassDef {
        term: oxa::CHART,
        en: "Chart",
        ja: "チャート",
        super_class: Some(prov::ENTITY),
    },
    ClassDef {
        term: oxa::NATAL_CHART,
        en: "Natal chart",
        ja: "出生図",
        super_class: Some(oxa::CHART),
    },
    ClassDef {
        term: oxa::COMPOSITE_CHART,
        en: "Composite chart",
        ja: "コンポジットチャート",
        super_class: Some(oxa::CHART),
    },
    ClassDef {
        term: oxa::PROGRESSED_CHART,
        en: "Progressed chart",
        ja: "プログレスチャート",
        super_class: Some(oxa::CHART),
    },
    ClassDef {
        term: oxa::TRANSIT_CHART,
        en: "Transit chart",
        ja: "トランジットチャート",
        super_class: Some(oxa::CHART),
    },
    ClassDef {
        term: oxa::CHART_COMPARISON,
        en: "Chart comparison",
        ja: "チャート比較",
        super_class: Some(prov::ENTITY),
    },
    ClassDef {
        term: oxa::SYNASTRY_COMPARISON,
        en: "Synastry comparison",
        ja: "シナストリー",
        super_class: Some(oxa::CHART_COMPARISON),
    },
    ClassDef {
        term: oxa::TRANSIT_COMPARISON,
        en: "Transit comparison",
        ja: "トランジット比較",
        super_class: Some(oxa::CHART_COMPARISON),
    },
    ClassDef {
        term: oxa::PROGRESSION_COMPARISON,
        en: "Progression comparison",
        ja: "プログレッション比較",
        super_class: Some(oxa::CHART_COMPARISON),
    },
    ClassDef {
        term: oxa::CHART_POINT,
        en: "Chart point",
        ja: "チャートポイント",
        super_class: None,
    },
    ClassDef {
        term: oxa::BODY_POSITION,
        en: "Body position",
        ja: "天体の位置",
        super_class: Some(oxa::CHART_POINT),
    },
    ClassDef {
        term: oxa::HOUSE_CUSP,
        en: "House cusp",
        ja: "ハウスカスプ",
        super_class: None,
    },
    ClassDef {
        term: oxa::CHART_ANGLE,
        en: "Chart angle",
        ja: "チャートアングル",
        super_class: Some(oxa::CHART_POINT),
    },
    ClassDef {
        term: oxa::LUNAR_NODE,
        en: "Lunar node or apogee",
        ja: "月のノード／アポジー",
        super_class: None,
    },
    ClassDef {
        term: oxa::LOT,
        en: "Arabic Part (Lot)",
        ja: "アラビックパーツ",
        super_class: None,
    },
    ClassDef {
        term: oxa::ASPECT_OCCURRENCE,
        en: "Aspect occurrence",
        ja: "アスペクトの成立",
        super_class: None,
    },
    ClassDef {
        term: oxa::CROSS_ASPECT,
        en: "Cross-aspect",
        ja: "クロスアスペクト",
        super_class: Some(oxa::ASPECT_OCCURRENCE),
    },
    ClassDef {
        term: oxa::DIGNITY_ASSESSMENT,
        en: "Essential dignity assessment",
        ja: "ディグニティ評価",
        super_class: None,
    },
    ClassDef {
        term: oxa::DISTRIBUTION,
        en: "Element/modality distribution",
        ja: "エレメント／モダリティ配分",
        super_class: None,
    },
    ClassDef {
        term: oxa::OBSERVER,
        en: "Observer position",
        ja: "観測地",
        super_class: Some(geo::POINT),
    },
    // Concept classes.
    ClassDef {
        term: oxa::ZODIAC_SIGN,
        en: "Zodiac sign",
        ja: "星座（サイン）",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::PLANET,
        en: "Planet",
        ja: "天体",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::ASPECT_KIND,
        en: "Aspect kind",
        ja: "アスペクトの種類",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::ELEMENT,
        en: "Classical element",
        ja: "四元素",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::MODALITY,
        en: "Modality",
        ja: "三区分",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::HOUSE_SYSTEM,
        en: "House system",
        ja: "ハウスシステム",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::SECT,
        en: "Sect",
        ja: "セクト",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::MOTION_STATE,
        en: "Motion state",
        ja: "運行状態",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::DECLINATION_ASPECT,
        en: "Declination aspect",
        ja: "赤緯アスペクト",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::AYANAMSHA,
        en: "Ayanamsha",
        ja: "アヤナムシャ",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::DIGNITY_TIER,
        en: "Essential dignity tier",
        ja: "ディグニティの階梯",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::ANGLE_KIND,
        en: "Chart angle kind",
        ja: "アングルの種類",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::NODE_KIND,
        en: "Lunar node kind",
        ja: "ノードの種類",
        super_class: Some(skos::CONCEPT),
    },
    ClassDef {
        term: oxa::LOT_KIND,
        en: "Lot kind",
        ja: "ロットの種類",
        super_class: Some(skos::CONCEPT),
    },
];

/// Shorthand for an object property with a domain.
const fn op(
    term: NamedNodeRef<'static>,
    en: &'static str,
    ja: &'static str,
    domain: Option<NamedNodeRef<'static>>,
    range: NamedNodeRef<'static>,
) -> PropDef {
    PropDef {
        term,
        en,
        ja,
        object_property: true,
        domain,
        range,
        unit: None,
        comment: None,
    }
}

/// Shorthand for a datatype property.
const fn dp(
    term: NamedNodeRef<'static>,
    en: &'static str,
    ja: &'static str,
    domain: Option<NamedNodeRef<'static>>,
    range: NamedNodeRef<'static>,
    unit: Option<NamedNodeRef<'static>>,
) -> PropDef {
    PropDef {
        term,
        en,
        ja,
        object_property: false,
        domain,
        range,
        unit,
        comment: None,
    }
}

/// Shorthand for a datatype property whose unit has no QUDT IRI.
const fn dp_commented(
    term: NamedNodeRef<'static>,
    en: &'static str,
    ja: &'static str,
    domain: Option<NamedNodeRef<'static>>,
    range: NamedNodeRef<'static>,
    comment: &'static str,
) -> PropDef {
    PropDef {
        term,
        en,
        ja,
        object_property: false,
        domain,
        range,
        unit: None,
        comment: Some(comment),
    }
}

const PROPERTIES: &[PropDef] = &[
    // --- chart composition (object) ---
    op(oxa::HAS_BODY_POSITION, "has body position", "天体位置を持つ", Some(oxa::CHART), oxa::BODY_POSITION),
    op(oxa::HAS_HOUSE_CUSP, "has house cusp", "ハウスカスプを持つ", Some(oxa::CHART), oxa::HOUSE_CUSP),
    op(oxa::HAS_ANGLE, "has angle", "アングルを持つ", Some(oxa::CHART), oxa::CHART_ANGLE),
    op(oxa::HAS_LUNAR_NODE, "has lunar node", "ノードを持つ", Some(oxa::CHART), oxa::LUNAR_NODE),
    op(oxa::HAS_LOT, "has lot", "ロットを持つ", Some(oxa::CHART), oxa::LOT),
    op(oxa::HAS_ASPECT, "has aspect", "アスペクトを持つ", Some(oxa::CHART), oxa::ASPECT_OCCURRENCE),
    op(oxa::HAS_DIGNITY, "has dignity assessment", "ディグニティ評価を持つ", Some(oxa::CHART), oxa::DIGNITY_ASSESSMENT),
    op(oxa::HAS_DISTRIBUTION, "has distribution", "配分を持つ", Some(oxa::CHART), oxa::DISTRIBUTION),
    op(oxa::HAS_OBSERVER, "has observer", "観測地を持つ", Some(oxa::CHART), oxa::OBSERVER),
    op(oxa::AT_EPOCH, "at epoch", "エポック", Some(oxa::CHART), time::INSTANT),
    op(oxa::HAS_SECT, "has sect", "セクト", Some(oxa::CHART), oxa::SECT),
    op(oxa::HOUSE_SYSTEM_OF, "house system", "ハウスシステム", Some(oxa::CHART), oxa::HOUSE_SYSTEM),
    op(oxa::AYANAMSHA_OF, "ayanamsha", "アヤナムシャ", Some(oxa::CHART), oxa::AYANAMSHA),
    // --- resource-level object properties ---
    // `body` is shared by BodyPosition and DignityAssessment; `inSign` by
    // every positioned resource: no rdfs:domain (see the module doc).
    op(oxa::BODY, "body", "天体", None, oxa::PLANET),
    op(oxa::IN_SIGN, "in sign", "在住サイン", None, oxa::ZODIAC_SIGN),
    // Angles carry a (zero) speed and a motion state too, so these three
    // belong to every ChartPoint, not only to body positions.
    op(oxa::MOTION, "motion", "運行状態", Some(oxa::CHART_POINT), oxa::MOTION_STATE),
    op(oxa::FIRST_BODY, "first body", "第1天体", Some(oxa::ASPECT_OCCURRENCE), oxa::PLANET),
    op(oxa::SECOND_BODY, "second body", "第2天体", Some(oxa::ASPECT_OCCURRENCE), oxa::PLANET),
    op(oxa::FROM_POINT, "from point", "起点", Some(oxa::ASPECT_OCCURRENCE), oxa::CHART_POINT),
    op(oxa::TO_POINT, "to point", "終点", Some(oxa::ASPECT_OCCURRENCE), oxa::CHART_POINT),
    // --- comparisons ---
    op(oxa::CHART_A, "chart A", "チャートA", Some(oxa::CHART_COMPARISON), oxa::CHART),
    op(oxa::CHART_B, "chart B", "チャートB", Some(oxa::CHART_COMPARISON), oxa::CHART),
    op(oxa::HAS_CROSS_ASPECT, "has cross-aspect", "クロスアスペクトを持つ", Some(oxa::CHART_COMPARISON), oxa::CROSS_ASPECT),
    op(oxa::ASPECT_KIND_OF, "aspect kind", "アスペクトの種類", Some(oxa::ASPECT_OCCURRENCE), oxa::ASPECT_KIND),
    op(oxa::ANGLE_KIND_OF, "angle kind", "アングルの種類", Some(oxa::CHART_ANGLE), oxa::ANGLE_KIND),
    op(oxa::NODE_KIND_OF, "node kind", "ノードの種類", Some(oxa::LUNAR_NODE), oxa::NODE_KIND),
    op(oxa::LOT_KIND_OF, "lot kind", "ロットの種類", Some(oxa::LOT), oxa::LOT_KIND),
    op(oxa::HOLDS_TIER, "holds dignity tier", "ディグニティ階梯を持つ", Some(oxa::DIGNITY_ASSESSMENT), oxa::DIGNITY_TIER),
    // --- sign ↔ concept relations ---
    op(oxa::ELEMENT_OF, "element", "エレメント", Some(oxa::ZODIAC_SIGN), oxa::ELEMENT),
    op(oxa::MODALITY_OF, "modality", "モダリティ", Some(oxa::ZODIAC_SIGN), oxa::MODALITY),
    op(oxa::TRADITIONAL_RULER, "traditional ruler", "伝統的支配星", Some(oxa::ZODIAC_SIGN), oxa::PLANET),
    op(oxa::MODERN_RULER, "modern ruler", "現代的支配星", Some(oxa::ZODIAC_SIGN), oxa::PLANET),
    op(oxa::EXALTED_PLANET, "exalted planet", "高揚する天体", Some(oxa::ZODIAC_SIGN), oxa::PLANET),
    op(oxa::DAY_TRIPLICITY_RULER, "day triplicity ruler", "昼のトリプリシティ支配星", Some(oxa::ZODIAC_SIGN), oxa::PLANET),
    op(oxa::NIGHT_TRIPLICITY_RULER, "night triplicity ruler", "夜のトリプリシティ支配星", Some(oxa::ZODIAC_SIGN), oxa::PLANET),
    op(oxa::PARTICIPATING_TRIPLICITY_RULER, "participating triplicity ruler", "協力するトリプリシティ支配星", Some(oxa::ZODIAC_SIGN), oxa::PLANET),
    // --- angles, degrees ---
    dp(oxa::ECLIPTIC_LONGITUDE, "ecliptic longitude", "黄経", None, xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::ECLIPTIC_LATITUDE, "ecliptic latitude", "黄緯", Some(oxa::BODY_POSITION), xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::DEGREES_IN_SIGN, "degrees in sign", "サイン内度数", None, xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::DECLINATION, "declination", "赤緯", Some(oxa::BODY_POSITION), xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::ORB, "orb", "オーブ", Some(oxa::ASPECT_OCCURRENCE), xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::EXACT_ANGLE, "exact angle", "正確角", Some(oxa::ASPECT_KIND), xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::DEFAULT_ORB, "default orb", "既定オーブ", Some(oxa::ASPECT_KIND), xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::EXALTATION_DEGREE, "exaltation degree", "高揚の度数", Some(oxa::ZODIAC_SIGN), xsd::DOUBLE, Some(unit::DEG)),
    dp(oxa::AYANAMSHA_DEGREES, "ayanamsha value", "アヤナムシャ値", Some(oxa::CHART), xsd::DOUBLE, Some(unit::DEG)),
    // --- distances, times ---
    dp(oxa::DISTANCE_AU, "distance", "距離", Some(oxa::BODY_POSITION), xsd::DOUBLE, Some(unit::AU)),
    dp_commented(oxa::LIGHT_TIME_DAYS, "light time", "光行時間", Some(oxa::BODY_POSITION), xsd::DOUBLE, "One-way light time, in days."),
    dp_commented(oxa::JULIAN_DATE_TT, "Julian Date (TT)", "ユリウス日（TT）", Some(oxa::CHART), xsd::DOUBLE, "Julian Date in Terrestrial Time, in days."),
    dp_commented(oxa::RULERSHIP_SCHEME, "rulership scheme", "支配星の体系", Some(oxa::DIGNITY_ASSESSMENT), xsd::STRING, "Either \"traditional\" (the seven classical planets) or \"modern\"; selects the domicile table this assessment was scored against. It qualifies the assessment, not the chart: a chart may carry one assessment per scheme."),
    // Stated by both a progressed chart and a progression comparison, so no
    // single rdfs:domain would be true.
    dp_commented(oxa::ELAPSED_YEARS, "elapsed years", "経過年数", None, xsd::DOUBLE, "Tropical years elapsed between the natal and the target epoch."),
    // --- speeds: QUDT publishes no deg/day unit IRI ---
    dp_commented(oxa::LONGITUDE_SPEED, "longitude speed", "黄経速度", Some(oxa::CHART_POINT), xsd::DOUBLE, "Daily longitude speed, in degrees per day; negative means retrograde. QUDT has no unit IRI for degrees per day."),
    dp_commented(oxa::LATITUDE_SPEED, "latitude speed", "黄緯速度", Some(oxa::BODY_POSITION), xsd::DOUBLE, "Daily latitude speed, in degrees per day. QUDT has no unit IRI for degrees per day."),
    // --- counts and ordinals ---
    dp(oxa::HOUSE_NUMBER, "house number", "在住ハウス", Some(oxa::BODY_POSITION), xsd::INTEGER, None),
    dp(oxa::CUSP_NUMBER, "cusp number", "カスプ番号", Some(oxa::HOUSE_CUSP), xsd::INTEGER, None),
    dp(oxa::SIGN_INDEX, "sign index", "サイン番号", Some(oxa::ZODIAC_SIGN), xsd::INTEGER, None),
    dp(oxa::DIGNITY_SCORE, "dignity score", "ディグニティ得点", Some(oxa::DIGNITY_ASSESSMENT), xsd::INTEGER, None),
    dp(oxa::TIER_WEIGHT, "tier weight", "階梯の重み", Some(oxa::DIGNITY_TIER), xsd::INTEGER, None),
    dp(oxa::FIRE_COUNT, "fire count", "火のエレメント数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    dp(oxa::EARTH_COUNT, "earth count", "地のエレメント数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    dp(oxa::AIR_COUNT, "air count", "風のエレメント数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    dp(oxa::WATER_COUNT, "water count", "水のエレメント数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    dp(oxa::CARDINAL_COUNT, "cardinal count", "活動宮の数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    dp(oxa::FIXED_COUNT, "fixed count", "不動宮の数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    dp(oxa::MUTABLE_COUNT, "mutable count", "柔軟宮の数", Some(oxa::DISTRIBUTION), xsd::INTEGER, None),
    // --- booleans ---
    dp(oxa::IS_APPLYING, "is applying", "接近中", Some(oxa::ASPECT_OCCURRENCE), xsd::BOOLEAN, None),
    dp(oxa::IS_RETROGRADE, "is retrograde", "逆行中", Some(oxa::CHART_POINT), xsd::BOOLEAN, None),
    dp(oxa::IS_OUT_OF_BOUNDS, "is out of bounds", "アウトオブバウンズ", Some(oxa::BODY_POSITION), xsd::BOOLEAN, None),
];

/// The `oxa:` ontology as an RDF graph.
#[must_use]
pub fn ontology_graph() -> Graph {
    let mut graph = Graph::new();
    let iri = ontology_iri();
    add(&mut graph, iri.clone(), rdf::TYPE, owl::ONTOLOGY);
    add_labels(
        &mut graph,
        &iri,
        dct::TITLE,
        "OxiEphemeris astrology ontology",
        Some("OxiEphemeris 占星術オントロジー"),
    );
    add(
        &mut graph,
        iri,
        dct::LICENSE,
        NamedNode::new_unchecked("https://www.apache.org/licenses/LICENSE-2.0"),
    );

    for class in CLASSES {
        let term = class.term.into_owned();
        add(&mut graph, term.clone(), rdf::TYPE, owl::CLASS);
        add(
            &mut graph,
            term.clone(),
            rdfs::LABEL,
            lang(class.en, Lang::En),
        );
        add(
            &mut graph,
            term.clone(),
            rdfs::LABEL,
            lang(class.ja, Lang::Ja),
        );
        if let Some(parent) = class.super_class {
            add(&mut graph, term, rdfs::SUB_CLASS_OF, parent);
        }
    }

    for prop in PROPERTIES {
        let term = prop.term.into_owned();
        let kind = if prop.object_property {
            owl::OBJECT_PROPERTY
        } else {
            owl::DATATYPE_PROPERTY
        };
        add(&mut graph, term.clone(), rdf::TYPE, kind);
        add(&mut graph, term.clone(), rdf::TYPE, rdf::PROPERTY);
        add(
            &mut graph,
            term.clone(),
            rdfs::LABEL,
            lang(prop.en, Lang::En),
        );
        add(
            &mut graph,
            term.clone(),
            rdfs::LABEL,
            lang(prop.ja, Lang::Ja),
        );
        if let Some(domain) = prop.domain {
            add(&mut graph, term.clone(), rdfs::DOMAIN, domain);
        }
        add(&mut graph, term.clone(), rdfs::RANGE, prop.range);
        if let Some(u) = prop.unit {
            add(&mut graph, term.clone(), qudt::HAS_UNIT, u);
        }
        if let Some(comment) = prop.comment {
            add(&mut graph, term, rdfs::COMMENT, lang(comment, Lang::En));
        }
    }
    graph
}

/// Every `oxa:` term the ontology declares, as strings — the reference set
/// for the closure test in `tests/closure.rs`.
#[must_use]
pub fn declared_terms() -> Vec<String> {
    CLASSES
        .iter()
        .map(|c| c.term.as_str().to_owned())
        .chain(PROPERTIES.iter().map(|p| p.term.as_str().to_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxrdf::TripleRef;

    #[test]
    fn every_oxa_constant_is_declared() {
        let declared = declared_terms();
        for term in oxa::ALL {
            assert!(
                declared.iter().any(|d| d == term.as_str()),
                "{} is defined in vocab.rs but never declared in the ontology",
                term.as_str()
            );
        }
    }

    #[test]
    fn no_term_is_declared_twice() {
        let mut declared = declared_terms();
        let total = declared.len();
        declared.sort_unstable();
        declared.dedup();
        assert_eq!(declared.len(), total, "a term is declared twice");
    }

    #[test]
    fn shared_properties_carry_no_false_domain() {
        let g = ontology_graph();
        // `eclipticLongitude`, `inSign`, `degreesInSign` and `body` are
        // used by several resource kinds, so a domain would be a false
        // entailment.
        for shared in [
            oxa::ECLIPTIC_LONGITUDE,
            oxa::IN_SIGN,
            oxa::DEGREES_IN_SIGN,
            oxa::BODY,
        ] {
            let domains = g
                .objects_for_subject_predicate(
                    oxrdf::NamedOrBlankNodeRef::NamedNode(shared),
                    rdfs::DOMAIN,
                )
                .count();
            assert_eq!(domains, 0, "{} must not declare a domain", shared.as_str());
        }
    }

    #[test]
    fn angle_properties_carry_the_degree_unit() {
        let g = ontology_graph();
        assert!(g.contains(TripleRef::new(
            oxa::ECLIPTIC_LONGITUDE,
            qudt::HAS_UNIT,
            unit::DEG
        )));
        assert!(g.contains(TripleRef::new(oxa::DISTANCE_AU, qudt::HAS_UNIT, unit::AU)));
    }

    #[test]
    fn speed_properties_have_no_unit_but_do_have_a_comment() {
        let g = ontology_graph();
        for speed in [oxa::LONGITUDE_SPEED, oxa::LATITUDE_SPEED] {
            let units = g
                .objects_for_subject_predicate(
                    oxrdf::NamedOrBlankNodeRef::NamedNode(speed),
                    qudt::HAS_UNIT,
                )
                .count();
            assert_eq!(units, 0, "QUDT has no degrees-per-day unit IRI");
            let comments = g
                .objects_for_subject_predicate(
                    oxrdf::NamedOrBlankNodeRef::NamedNode(speed),
                    rdfs::COMMENT,
                )
                .count();
            assert_eq!(comments, 1, "the unit must be documented instead");
        }
    }

    #[test]
    fn chart_subclasses_and_prov_entity() {
        let g = ontology_graph();
        assert!(g.contains(TripleRef::new(oxa::CHART, rdfs::SUB_CLASS_OF, prov::ENTITY)));
        assert!(g.contains(TripleRef::new(
            oxa::NATAL_CHART,
            rdfs::SUB_CLASS_OF,
            oxa::CHART
        )));
        assert!(g.contains(TripleRef::new(
            oxa::ZODIAC_SIGN,
            rdfs::SUB_CLASS_OF,
            skos::CONCEPT
        )));
    }

    #[test]
    fn ontology_iri_has_no_hash() {
        assert!(!ontology_iri().as_str().ends_with('#'));
        assert!(NamedNode::new(ontology_iri().as_str().to_owned()).is_ok());
    }
}
