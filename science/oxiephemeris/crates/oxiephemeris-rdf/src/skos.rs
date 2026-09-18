//! The SKOS concept schemes: signs, planets, aspects, elements,
//! modalities, house systems, sect, motion states, declination aspects,
//! ayanamshas, dignity tiers, chart angles, lunar nodes and lots.
//!
//! # Generated, never hand-written
//!
//! Every concept is derived from the corresponding `oxiephemeris_astro`
//! enum (`Sign::ALL`, `Aspect::ALL`, `Planet::ALL`, …) and its own
//! `name()`. The published vocabulary therefore *cannot* drift from the
//! code that computes charts: a sign that exists in the engine has a
//! concept, with the same notation the JSON output uses, or the build
//! fails.
//!
//! The domain relations between concepts are likewise read from the
//! engine's own dignity tables ([`oxiephemeris_astro::dignities`]), so
//! `oxc:sign/Scorpio oxa:traditionalRuler oxc:planet/Mars` is a statement
//! about the very table that scores a chart, not a second transcription
//! of Ptolemy.
//!
//! # Flat schemes
//!
//! None of these vocabularies is hierarchical (no sign is broader than
//! another), so every concept is a top concept of its scheme and there are
//! no `skos:broader`/`skos:narrower` links.

use oxrdf::vocab::rdf;
use oxrdf::{Graph, NamedNode, NamedNodeRef};

use oxiephemeris_astro::aspects::{Aspect, OrbPolicy};
use oxiephemeris_astro::ayanamsha::Ayanamsha;
use oxiephemeris_astro::declination::DeclinationAspect;
use oxiephemeris_astro::dignities::{
    domicile_ruler, sign_exaltation, triplicity_rulers, Planet, RulershipScheme,
};
use oxiephemeris_astro::houses::HouseSystem;
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::parts::Sect;
use oxiephemeris_astro::zodiac::{Element, Modality, Sign};

use crate::builder::{
    add, add_double, add_labels, integer, lang, plain, round_definitional_deg, Lang,
};
use crate::labels;
use crate::model::{AngleKind, DignityTier, LotKind, NodeKind};
use crate::vocab::{concept, dct, oxa, path, planet_qid, scheme, sign_qid, skos, wikidata};

/// Metadata for one flat concept scheme.
struct SchemeDef {
    /// Local name of the scheme IRI, e.g. `"ZodiacSigns"`.
    name: &'static str,
    /// Concept IRI path segment, e.g. `"sign"`.
    path: &'static str,
    /// The `oxa:` class every concept in this scheme also belongs to.
    class: NamedNodeRef<'static>,
    /// English scheme title.
    title_en: &'static str,
    /// Japanese scheme title.
    title_ja: &'static str,
}

/// Every concept scheme this crate publishes, in a stable order.
const SCHEMES: [SchemeDef; 14] = [
    SchemeDef {
        name: "ZodiacSigns",
        path: path::SIGN,
        class: oxa::ZODIAC_SIGN,
        title_en: "Zodiac signs",
        title_ja: "十二星座",
    },
    SchemeDef {
        name: "Planets",
        path: path::PLANET,
        class: oxa::PLANET,
        title_en: "Planets",
        title_ja: "天体",
    },
    SchemeDef {
        name: "AspectKinds",
        path: path::ASPECT,
        class: oxa::ASPECT_KIND,
        title_en: "Aspect kinds",
        title_ja: "アスペクトの種類",
    },
    SchemeDef {
        name: "Elements",
        path: path::ELEMENT,
        class: oxa::ELEMENT,
        title_en: "Classical elements",
        title_ja: "四元素",
    },
    SchemeDef {
        name: "Modalities",
        path: path::MODALITY,
        class: oxa::MODALITY,
        title_en: "Modalities",
        title_ja: "三区分",
    },
    SchemeDef {
        name: "HouseSystems",
        path: path::HOUSE_SYSTEM,
        class: oxa::HOUSE_SYSTEM,
        title_en: "House systems",
        title_ja: "ハウスシステム",
    },
    SchemeDef {
        name: "Sects",
        path: path::SECT,
        class: oxa::SECT,
        title_en: "Chart sect",
        title_ja: "セクト（昼夜）",
    },
    SchemeDef {
        name: "MotionStates",
        path: path::MOTION,
        class: oxa::MOTION_STATE,
        title_en: "Motion states",
        title_ja: "運行状態",
    },
    SchemeDef {
        name: "DeclinationAspects",
        path: path::DECLINATION_ASPECT,
        class: oxa::DECLINATION_ASPECT,
        title_en: "Declination aspects",
        title_ja: "赤緯アスペクト",
    },
    SchemeDef {
        name: "Ayanamshas",
        path: path::AYANAMSHA,
        class: oxa::AYANAMSHA,
        title_en: "Ayanamshas",
        title_ja: "アヤナムシャ",
    },
    SchemeDef {
        name: "DignityTiers",
        path: path::DIGNITY,
        class: oxa::DIGNITY_TIER,
        title_en: "Essential dignity tiers",
        title_ja: "エッセンシャル・ディグニティ",
    },
    SchemeDef {
        name: "AngleKinds",
        path: path::ANGLE,
        class: oxa::ANGLE_KIND,
        title_en: "Chart angles",
        title_ja: "チャートアングル",
    },
    SchemeDef {
        name: "NodeKinds",
        path: path::NODE,
        class: oxa::NODE_KIND,
        title_en: "Lunar nodes and apogees",
        title_ja: "月のノードとアポジー",
    },
    SchemeDef {
        name: "LotKinds",
        path: path::LOT,
        class: oxa::LOT_KIND,
        title_en: "Arabic Parts (Lots)",
        title_ja: "アラビックパーツ（ロット）",
    },
];

/// Looks up a scheme definition by its local name. Panics never: the
/// callers below pass literals that are present in [`SCHEMES`], and
/// `tests::every_scheme_name_resolves` pins that.
fn def(name: &str) -> &'static SchemeDef {
    SCHEMES
        .iter()
        .find(|s| s.name == name)
        .unwrap_or(&SCHEMES[0])
}

/// Declares a scheme resource and returns its IRI.
fn declare_scheme(graph: &mut Graph, def: &SchemeDef) -> NamedNode {
    let iri = scheme(def.name);
    add(graph, iri.clone(), rdf::TYPE, skos::CONCEPT_SCHEME);
    add(graph, iri.clone(), dct::TITLE, lang(def.title_en, Lang::En));
    add(graph, iri.clone(), dct::TITLE, lang(def.title_ja, Lang::Ja));
    iri
}

/// Declares one concept in a flat scheme (every concept is a top concept)
/// and returns its IRI.
fn declare_concept(
    graph: &mut Graph,
    scheme_iri: &NamedNode,
    def: &SchemeDef,
    notation: &str,
    label_en: &str,
    label_ja: Option<&str>,
) -> NamedNode {
    let iri = concept(def.path, notation);
    add(graph, iri.clone(), rdf::TYPE, skos::CONCEPT);
    add(graph, iri.clone(), rdf::TYPE, def.class);
    add(graph, iri.clone(), skos::IN_SCHEME, scheme_iri.clone());
    add(graph, iri.clone(), skos::TOP_CONCEPT_OF, scheme_iri.clone());
    add(
        graph,
        scheme_iri.clone(),
        skos::HAS_TOP_CONCEPT,
        iri.clone(),
    );
    add(graph, iri.clone(), skos::NOTATION, plain(notation));
    add_labels(graph, &iri, skos::PREF_LABEL, label_en, label_ja);
    iri
}

/// The complete SKOS vocabulary as one graph.
#[must_use]
pub fn concept_scheme_graph() -> Graph {
    let mut graph = Graph::new();
    signs(&mut graph);
    planets(&mut graph);
    aspect_kinds(&mut graph);
    elements(&mut graph);
    modalities(&mut graph);
    house_systems(&mut graph);
    sects(&mut graph);
    motion_states(&mut graph);
    declination_aspects(&mut graph);
    ayanamshas(&mut graph);
    dignity_tiers(&mut graph);
    angle_kinds(&mut graph);
    node_kinds(&mut graph);
    lot_kinds(&mut graph);
    graph
}

/// The IRI of the concept for a [`Sign`].
#[must_use]
pub fn sign_concept(sign: Sign) -> NamedNode {
    concept(path::SIGN, sign.name())
}

/// The IRI of the concept for a [`Planet`].
#[must_use]
pub fn planet_concept(planet: Planet) -> NamedNode {
    concept(path::PLANET, planet.name())
}

/// The IRI of the concept for an aspect kind.
#[must_use]
pub fn aspect_concept(kind: oxiephemeris_astro::aspects::AspectKind) -> NamedNode {
    concept(path::ASPECT, kind.name())
}

/// The IRI of the concept for an [`Element`].
#[must_use]
pub fn element_concept(element: Element) -> NamedNode {
    concept(path::ELEMENT, element.name())
}

/// The IRI of the concept for a [`Modality`].
#[must_use]
pub fn modality_concept(modality: Modality) -> NamedNode {
    concept(path::MODALITY, modality.name())
}

/// The IRI of the concept for a [`HouseSystem`].
#[must_use]
pub fn house_system_concept(system: HouseSystem) -> NamedNode {
    concept(path::HOUSE_SYSTEM, system.name())
}

/// The IRI of the concept for a [`Sect`].
#[must_use]
pub fn sect_concept(sect: Sect) -> NamedNode {
    concept(path::SECT, sect.name())
}

/// The IRI of the concept for a [`MotionState`].
#[must_use]
pub fn motion_concept(motion: MotionState) -> NamedNode {
    concept(path::MOTION, motion.name())
}

/// The IRI of the concept for an [`Ayanamsha`], or `None` for
/// [`Ayanamsha::Custom`], which has no published name.
#[must_use]
pub fn ayanamsha_concept(ayanamsha: Ayanamsha) -> Option<NamedNode> {
    ayanamsha.name().map(|n| concept(path::AYANAMSHA, n))
}

/// The IRI of the concept for a [`DignityTier`].
#[must_use]
pub fn dignity_tier_concept(tier: DignityTier) -> NamedNode {
    concept(path::DIGNITY, tier.name())
}

/// The IRI of the concept for an [`AngleKind`].
#[must_use]
pub fn angle_concept(kind: AngleKind) -> NamedNode {
    concept(path::ANGLE, kind.name())
}

/// The IRI of the concept for a [`NodeKind`].
#[must_use]
pub fn node_concept(kind: NodeKind) -> NamedNode {
    concept(path::NODE, kind.name())
}

/// The IRI of the concept for a [`LotKind`].
#[must_use]
pub fn lot_concept(kind: LotKind) -> NamedNode {
    concept(path::LOT, kind.name())
}

fn signs(graph: &mut Graph) {
    let def = def("ZodiacSigns");
    let scheme_iri = declare_scheme(graph, def);
    for sign in Sign::ALL {
        let iri = declare_concept(
            graph,
            &scheme_iri,
            def,
            sign.name(),
            sign.name(),
            labels::sign_ja(sign),
        );
        if let Some(alt) = labels::sign_ja_classical(sign) {
            add(graph, iri.clone(), skos::ALT_LABEL, lang(alt, Lang::Ja));
        }
        if let Some(qid) = sign_qid(sign.name()) {
            add(graph, iri.clone(), skos::EXACT_MATCH, wikidata(qid));
        }
        // Index and the element/modality triplicity groupings.
        add(
            graph,
            iri.clone(),
            oxa::SIGN_INDEX,
            integer(i64::try_from(sign.index()).unwrap_or(0)),
        );
        add(
            graph,
            iri.clone(),
            oxa::ELEMENT_OF,
            element_concept(sign.element()),
        );
        add(
            graph,
            iri.clone(),
            oxa::MODALITY_OF,
            modality_concept(sign.modality()),
        );
        // Rulerships, read from the engine's own tables.
        add(
            graph,
            iri.clone(),
            oxa::TRADITIONAL_RULER,
            planet_concept(domicile_ruler(sign, RulershipScheme::Traditional)),
        );
        add(
            graph,
            iri.clone(),
            oxa::MODERN_RULER,
            planet_concept(domicile_ruler(sign, RulershipScheme::Modern)),
        );
        if let Some((planet, degree)) = sign_exaltation(sign) {
            add(
                graph,
                iri.clone(),
                oxa::EXALTED_PLANET,
                planet_concept(planet),
            );
            add_double(graph, iri.clone(), oxa::EXALTATION_DEGREE, degree);
        }
        let triplicity = triplicity_rulers(sign);
        add(
            graph,
            iri.clone(),
            oxa::DAY_TRIPLICITY_RULER,
            planet_concept(triplicity.day),
        );
        add(
            graph,
            iri.clone(),
            oxa::NIGHT_TRIPLICITY_RULER,
            planet_concept(triplicity.night),
        );
        add(
            graph,
            iri,
            oxa::PARTICIPATING_TRIPLICITY_RULER,
            planet_concept(triplicity.participating),
        );
    }
}

fn planets(graph: &mut Graph) {
    let def = def("Planets");
    let scheme_iri = declare_scheme(graph, def);
    for planet in Planet::ALL {
        let iri = declare_concept(
            graph,
            &scheme_iri,
            def,
            planet.name(),
            planet.name(),
            labels::planet_ja(planet),
        );
        if let Some(qid) = planet_qid(planet.name()) {
            add(graph, iri, skos::EXACT_MATCH, wikidata(qid));
        }
    }
}

fn aspect_kinds(graph: &mut Graph) {
    let def = def("AspectKinds");
    let scheme_iri = declare_scheme(graph, def);
    let policy = OrbPolicy::default();
    for aspect in Aspect::ALL {
        let kind = aspect.kind;
        let iri = declare_concept(
            graph,
            &scheme_iri,
            def,
            kind.name(),
            kind.name(),
            labels::aspect_ja(kind),
        );
        add_double(
            graph,
            iri.clone(),
            oxa::EXACT_ANGLE,
            round_definitional_deg(aspect.exact_angle_rad.to_degrees()),
        );
        add_double(
            graph,
            iri,
            oxa::DEFAULT_ORB,
            round_definitional_deg(policy.orb_rad(kind).to_degrees()),
        );
    }
}

fn elements(graph: &mut Graph) {
    let def = def("Elements");
    let scheme_iri = declare_scheme(graph, def);
    for element in Element::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            element.name(),
            element.name(),
            labels::element_ja(element),
        );
    }
}

fn modalities(graph: &mut Graph) {
    let def = def("Modalities");
    let scheme_iri = declare_scheme(graph, def);
    for modality in Modality::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            modality.name(),
            modality.name(),
            labels::modality_ja(modality),
        );
    }
}

fn house_systems(graph: &mut Graph) {
    let def = def("HouseSystems");
    let scheme_iri = declare_scheme(graph, def);
    for system in HouseSystem::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            system.name(),
            labels::house_system_en(system).unwrap_or(system.name()),
            labels::house_system_ja(system),
        );
    }
}

fn sects(graph: &mut Graph) {
    let def = def("Sects");
    let scheme_iri = declare_scheme(graph, def);
    for sect in Sect::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            sect.name(),
            sect.name(),
            labels::sect_ja(sect),
        );
    }
}

fn motion_states(graph: &mut Graph) {
    let def = def("MotionStates");
    let scheme_iri = declare_scheme(graph, def);
    for motion in MotionState::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            motion.name(),
            motion.name(),
            labels::motion_ja(motion),
        );
    }
}

fn declination_aspects(graph: &mut Graph) {
    let def = def("DeclinationAspects");
    let scheme_iri = declare_scheme(graph, def);
    for aspect in DeclinationAspect::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            aspect.name(),
            aspect.name(),
            labels::declination_aspect_ja(aspect),
        );
    }
}

fn ayanamshas(graph: &mut Graph) {
    let def = def("Ayanamshas");
    let scheme_iri = declare_scheme(graph, def);
    for ayanamsha in Ayanamsha::ALL_NAMED {
        let Some(notation) = ayanamsha.name() else {
            continue;
        };
        declare_concept(
            graph,
            &scheme_iri,
            def,
            notation,
            labels::ayanamsha_en(ayanamsha).unwrap_or(notation),
            labels::ayanamsha_ja(ayanamsha),
        );
    }
}

fn dignity_tiers(graph: &mut Graph) {
    let def = def("DignityTiers");
    let scheme_iri = declare_scheme(graph, def);
    for tier in DignityTier::ALL {
        let iri = declare_concept(
            graph,
            &scheme_iri,
            def,
            tier.name(),
            tier.name(),
            Some(labels::dignity_tier_ja(tier)),
        );
        add(
            graph,
            iri,
            oxa::TIER_WEIGHT,
            integer(i64::from(tier.score())),
        );
    }
}

fn angle_kinds(graph: &mut Graph) {
    let def = def("AngleKinds");
    let scheme_iri = declare_scheme(graph, def);
    for kind in AngleKind::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            kind.name(),
            labels::angle_kind_en(kind),
            Some(labels::angle_kind_ja(kind)),
        );
    }
}

fn node_kinds(graph: &mut Graph) {
    let def = def("NodeKinds");
    let scheme_iri = declare_scheme(graph, def);
    for kind in NodeKind::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            kind.name(),
            labels::node_kind_en(kind),
            Some(labels::node_kind_ja(kind)),
        );
    }
}

fn lot_kinds(graph: &mut Graph) {
    let def = def("LotKinds");
    let scheme_iri = declare_scheme(graph, def);
    for kind in LotKind::ALL {
        declare_concept(
            graph,
            &scheme_iri,
            def,
            kind.name(),
            labels::lot_kind_en(kind),
            Some(labels::lot_kind_ja(kind)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::{PLANET_WIKIDATA, SIGN_WIKIDATA};
    use oxrdf::{NamedNodeRef as Nnr, NamedOrBlankNodeRef, Term, TermRef, TripleRef};

    fn graph() -> Graph {
        concept_scheme_graph()
    }

    /// The objects of `(s, p, ?o)`, owned out of the graph so the result
    /// outlives the query's borrow.
    fn objects_of(g: &Graph, s: Nnr<'_>, p: Nnr<'_>) -> Vec<Term> {
        g.objects_for_subject_predicate(NamedOrBlankNodeRef::NamedNode(s), p)
            .map(TermRef::into_owned)
            .collect()
    }

    #[test]
    fn every_scheme_name_resolves() {
        for s in &SCHEMES {
            assert_eq!(def(s.name).name, s.name);
        }
    }

    #[test]
    fn every_sign_is_a_concept_in_its_scheme() {
        let g = graph();
        let scheme_iri = scheme("ZodiacSigns");
        for sign in Sign::ALL {
            let iri = sign_concept(sign);
            assert!(
                g.contains(TripleRef::new(
                    iri.as_ref(),
                    rdf::TYPE,
                    TermRef::NamedNode(skos::CONCEPT)
                )),
                "{} is not a skos:Concept",
                sign.name()
            );
            assert!(g.contains(TripleRef::new(
                iri.as_ref(),
                skos::IN_SCHEME,
                scheme_iri.as_ref()
            )));
        }
    }

    #[test]
    fn concept_counts_match_the_engine_enums() {
        let g = graph();
        let count = |s: &str, p: &str, n: usize| {
            let scheme_iri = scheme(s);
            let found = g
                .objects_for_subject_predicate(
                    NamedOrBlankNodeRef::NamedNode(scheme_iri.as_ref()),
                    skos::HAS_TOP_CONCEPT,
                )
                .count();
            assert_eq!(
                found, n,
                "scheme {s} (path {p}) has {found} concepts, want {n}"
            );
        };
        count("ZodiacSigns", path::SIGN, Sign::ALL.len());
        count("Planets", path::PLANET, Planet::ALL.len());
        count("AspectKinds", path::ASPECT, Aspect::ALL.len());
        count("Elements", path::ELEMENT, Element::ALL.len());
        count("Modalities", path::MODALITY, Modality::ALL.len());
        count("HouseSystems", path::HOUSE_SYSTEM, HouseSystem::ALL.len());
        count("Sects", path::SECT, Sect::ALL.len());
        count("MotionStates", path::MOTION, MotionState::ALL.len());
        count(
            "DeclinationAspects",
            path::DECLINATION_ASPECT,
            DeclinationAspect::ALL.len(),
        );
        count("Ayanamshas", path::AYANAMSHA, Ayanamsha::ALL_NAMED.len());
        count("DignityTiers", path::DIGNITY, DignityTier::ALL.len());
        count("AngleKinds", path::ANGLE, AngleKind::ALL.len());
        count("NodeKinds", path::NODE, NodeKind::ALL.len());
        count("LotKinds", path::LOT, LotKind::ALL.len());
    }

    #[test]
    fn every_concept_has_english_and_japanese_pref_labels() {
        let g = graph();
        for triple in &g {
            let TermRef::NamedNode(class) = triple.object else {
                continue;
            };
            if triple.predicate != rdf::TYPE || class != skos::CONCEPT {
                continue;
            }
            let NamedOrBlankNodeRef::NamedNode(subject) = triple.subject else {
                continue;
            };
            let labels = objects_of(&g, subject, skos::PREF_LABEL);
            let mut has_en = false;
            let mut has_ja = false;
            for term in labels {
                if let Term::Literal(lit) = term {
                    match lit.language() {
                        Some("en") => has_en = true,
                        Some("ja") => has_ja = true,
                        _ => {}
                    }
                }
            }
            assert!(has_en && has_ja, "{subject} missing en/ja prefLabel");
        }
    }

    #[test]
    fn rulerships_agree_with_the_dignity_tables() {
        let g = graph();
        // Scorpio: traditional ruler Mars, modern ruler Pluto.
        let scorpio = sign_concept(Sign::Scorpio);
        assert!(g.contains(TripleRef::new(
            scorpio.as_ref(),
            oxa::TRADITIONAL_RULER,
            planet_concept(Planet::Mars).as_ref()
        )));
        assert!(g.contains(TripleRef::new(
            scorpio.as_ref(),
            oxa::MODERN_RULER,
            planet_concept(Planet::Pluto).as_ref()
        )));
        // Aries: Sun exalted.
        let aries = sign_concept(Sign::Aries);
        assert!(g.contains(TripleRef::new(
            aries.as_ref(),
            oxa::EXALTED_PLANET,
            planet_concept(Planet::Sun).as_ref()
        )));
    }

    #[test]
    fn wikidata_alignment_only_for_verified_concepts() {
        let g = graph();
        let matches = g
            .iter()
            .filter(|t| t.predicate == skos::EXACT_MATCH)
            .count();
        assert_eq!(
            matches,
            PLANET_WIKIDATA.len() + SIGN_WIKIDATA.len(),
            "exactMatch must be emitted only for the verified planets and signs"
        );
    }

    #[test]
    fn aspect_concepts_carry_exact_angle_and_default_orb() {
        let g = graph();
        let trine = aspect_concept(oxiephemeris_astro::aspects::AspectKind::Trine);
        let angles = objects_of(&g, trine.as_ref(), oxa::EXACT_ANGLE);
        assert_eq!(angles.len(), 1);
        if let Some(Term::Literal(lit)) = angles.first() {
            assert_eq!(lit.value(), "120");
        } else {
            panic!("trine must have an xsd:double exactAngle");
        }
        assert_eq!(objects_of(&g, trine.as_ref(), oxa::DEFAULT_ORB).len(), 1);
    }

    #[test]
    fn custom_ayanamsha_has_no_concept() {
        assert!(ayanamsha_concept(Ayanamsha::Custom {
            t0_jd_tt: 2_451_545.0,
            value_at_t0_rad: 0.0,
        })
        .is_none());
    }
}
