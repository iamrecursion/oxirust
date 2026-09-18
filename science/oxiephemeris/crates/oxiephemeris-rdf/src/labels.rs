//! Japanese labels for the SKOS concepts.
//!
//! English `skos:prefLabel`s come straight from each enum's `name()`, so
//! they cannot drift. Japanese labels live here.
//!
//! For the twelve signs, the **preferred** Japanese label is the modern
//! 「〜座」form in everyday use (牡羊座, 蠍座, …), and the classical
//! 「〜宮」form (白羊宮, 天蝎宮, …) is attached as a `skos:altLabel` — the
//! latter is what Wikidata carries as its Japanese label, so a consumer
//! joining on either form finds the same concept.
//!
//! Every function returns `Option`, and the tests assert `Some` for every
//! member of each enum's `ALL`. That is what keeps this table from
//! silently falling behind a newly-added variant: adding one to
//! `oxiephemeris_astro` (whose enums are `#[non_exhaustive]`, so we cannot
//! match exhaustively from here) turns into a test failure, not a missing
//! label in published data.

use oxiephemeris_astro::aspects::AspectKind;
use oxiephemeris_astro::ayanamsha::Ayanamsha;
use oxiephemeris_astro::declination::DeclinationAspect;
use oxiephemeris_astro::dignities::Planet;
use oxiephemeris_astro::houses::HouseSystem;
use oxiephemeris_astro::motion::MotionState;
use oxiephemeris_astro::parts::Sect;
use oxiephemeris_astro::zodiac::{Element, Modality, Sign};

use crate::model::{AngleKind, DignityTier, LotKind, NodeKind};

/// Preferred Japanese label for a zodiac sign (modern 「座」 form).
#[must_use]
pub fn sign_ja(sign: Sign) -> Option<&'static str> {
    Some(match sign {
        Sign::Aries => "牡羊座",
        Sign::Taurus => "牡牛座",
        Sign::Gemini => "双子座",
        Sign::Cancer => "蟹座",
        Sign::Leo => "獅子座",
        Sign::Virgo => "乙女座",
        Sign::Libra => "天秤座",
        Sign::Scorpio => "蠍座",
        Sign::Sagittarius => "射手座",
        Sign::Capricorn => "山羊座",
        Sign::Aquarius => "水瓶座",
        Sign::Pisces => "魚座",
        _ => return None,
    })
}

/// Classical Japanese label for a zodiac sign (「宮」 form) — emitted as a
/// `skos:altLabel`, and the form Wikidata uses.
#[must_use]
pub fn sign_ja_classical(sign: Sign) -> Option<&'static str> {
    Some(match sign {
        Sign::Aries => "白羊宮",
        Sign::Taurus => "金牛宮",
        Sign::Gemini => "双児宮",
        Sign::Cancer => "巨蟹宮",
        Sign::Leo => "獅子宮",
        Sign::Virgo => "処女宮",
        Sign::Libra => "天秤宮",
        Sign::Scorpio => "天蝎宮",
        Sign::Sagittarius => "人馬宮",
        Sign::Capricorn => "磨羯宮",
        Sign::Aquarius => "宝瓶宮",
        Sign::Pisces => "双魚宮",
        _ => return None,
    })
}

/// Japanese label for a planet.
#[must_use]
pub fn planet_ja(planet: Planet) -> Option<&'static str> {
    Some(match planet {
        Planet::Sun => "太陽",
        Planet::Moon => "月",
        Planet::Mercury => "水星",
        Planet::Venus => "金星",
        Planet::Mars => "火星",
        Planet::Jupiter => "木星",
        Planet::Saturn => "土星",
        Planet::Uranus => "天王星",
        Planet::Neptune => "海王星",
        Planet::Pluto => "冥王星",
        _ => return None,
    })
}

/// Japanese label for a classical element.
#[must_use]
pub fn element_ja(element: Element) -> Option<&'static str> {
    Some(match element {
        Element::Fire => "火",
        Element::Earth => "地",
        Element::Air => "風",
        Element::Water => "水",
        _ => return None,
    })
}

/// Japanese label for a modality.
#[must_use]
pub fn modality_ja(modality: Modality) -> Option<&'static str> {
    Some(match modality {
        Modality::Cardinal => "活動宮",
        Modality::Fixed => "不動宮",
        Modality::Mutable => "柔軟宮",
        _ => return None,
    })
}

/// Japanese label for an aspect kind.
#[must_use]
pub fn aspect_ja(kind: AspectKind) -> Option<&'static str> {
    Some(match kind {
        AspectKind::Conjunction => "コンジャンクション",
        AspectKind::Opposition => "オポジション",
        AspectKind::Trine => "トライン",
        AspectKind::Square => "スクエア",
        AspectKind::Sextile => "セクスタイル",
        AspectKind::Semisextile => "セミセクスタイル",
        AspectKind::Semisquare => "セミスクエア",
        AspectKind::Sesquiquadrate => "セスキコードレート",
        AspectKind::Quincunx => "クインカンクス",
        AspectKind::Quintile => "クインタイル",
        AspectKind::Biquintile => "バイクインタイル",
        _ => return None,
    })
}

/// Japanese label for a house system.
#[must_use]
pub fn house_system_ja(system: HouseSystem) -> Option<&'static str> {
    Some(match system {
        HouseSystem::Placidus => "プラシーダス",
        HouseSystem::Koch => "コッホ",
        HouseSystem::WholeSign => "ホールサイン",
        HouseSystem::Equal => "イコール",
        HouseSystem::Porphyry => "ポルフィリー",
        HouseSystem::Regiomontanus => "レギオモンタヌス",
        HouseSystem::Campanus => "カンパヌス",
        _ => return None,
    })
}

/// Japanese label for a chart sect.
#[must_use]
pub fn sect_ja(sect: Sect) -> Option<&'static str> {
    Some(match sect {
        Sect::Diurnal => "昼のチャート",
        Sect::Nocturnal => "夜のチャート",
        _ => return None,
    })
}

/// Japanese label for a motion state.
#[must_use]
pub fn motion_ja(motion: MotionState) -> Option<&'static str> {
    Some(match motion {
        MotionState::Direct => "順行",
        MotionState::Retrograde => "逆行",
        MotionState::Stationary => "留",
        _ => return None,
    })
}

/// Japanese label for a declination aspect.
#[must_use]
pub fn declination_aspect_ja(aspect: DeclinationAspect) -> Option<&'static str> {
    Some(match aspect {
        DeclinationAspect::Parallel => "パラレル",
        DeclinationAspect::Contraparallel => "コントラパラレル",
        _ => return None,
    })
}

/// Japanese label for an ayanamsha.
#[must_use]
pub fn ayanamsha_ja(ayanamsha: Ayanamsha) -> Option<&'static str> {
    Some(match ayanamsha {
        Ayanamsha::FaganBradley => "フェイガン/ブラッドリー",
        Ayanamsha::Lahiri => "ラヒリ",
        Ayanamsha::Krishnamurti => "クリシュナムルティ",
        Ayanamsha::Raman => "ラマン",
        Ayanamsha::J2000Zero => "J2000ゼロ",
        _ => return None,
    })
}

/// Japanese label for an essential-dignity tier.
#[must_use]
pub const fn dignity_tier_ja(tier: DignityTier) -> &'static str {
    match tier {
        DignityTier::Domicile => "ドミサイル（支配）",
        DignityTier::Exaltation => "エグザルテーション（高揚）",
        DignityTier::Triplicity => "トリプリシティ",
        DignityTier::Term => "ターム（バウンド）",
        DignityTier::Face => "フェイス（デカン）",
        DignityTier::Detriment => "デトリメント（障害）",
        DignityTier::Fall => "フォール（転落）",
        DignityTier::Peregrine => "ペレグリン",
    }
}

/// Japanese label for a chart angle.
#[must_use]
pub const fn angle_kind_ja(kind: AngleKind) -> &'static str {
    match kind {
        AngleKind::Ascendant => "アセンダント",
        AngleKind::Midheaven => "MC（南中点）",
        AngleKind::Vertex => "ヴァーテックス",
        AngleKind::EastPoint => "イーストポイント",
    }
}

/// Japanese label for a lunar node or apogee.
#[must_use]
pub const fn node_kind_ja(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::MeanNode => "平均ノード",
        NodeKind::MeanApogee => "平均アポジー（平均リリス）",
        NodeKind::TrueNode => "真ノード",
        NodeKind::TrueApogee => "真アポジー（真リリス）",
    }
}

/// Japanese label for an Arabic Part.
#[must_use]
pub const fn lot_kind_ja(kind: LotKind) -> &'static str {
    match kind {
        LotKind::Fortune => "フォーチュン（福点）",
        LotKind::Spirit => "スピリット",
    }
}

/// English display label for a chart angle (its `name()` is kebab-case).
#[must_use]
pub const fn angle_kind_en(kind: AngleKind) -> &'static str {
    match kind {
        AngleKind::Ascendant => "Ascendant",
        AngleKind::Midheaven => "Midheaven",
        AngleKind::Vertex => "Vertex",
        AngleKind::EastPoint => "East Point",
    }
}

/// English display label for a lunar node or apogee.
#[must_use]
pub const fn node_kind_en(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::MeanNode => "Mean Node",
        NodeKind::MeanApogee => "Mean Apogee",
        NodeKind::TrueNode => "True Node",
        NodeKind::TrueApogee => "True Apogee",
    }
}

/// English display label for an Arabic Part.
#[must_use]
pub const fn lot_kind_en(kind: LotKind) -> &'static str {
    match kind {
        LotKind::Fortune => "Lot of Fortune",
        LotKind::Spirit => "Lot of Spirit",
    }
}

/// English display label for a house system.
#[must_use]
pub fn house_system_en(system: HouseSystem) -> Option<&'static str> {
    Some(match system {
        HouseSystem::Placidus => "Placidus",
        HouseSystem::Koch => "Koch",
        HouseSystem::WholeSign => "Whole Sign",
        HouseSystem::Equal => "Equal",
        HouseSystem::Porphyry => "Porphyry",
        HouseSystem::Regiomontanus => "Regiomontanus",
        HouseSystem::Campanus => "Campanus",
        _ => return None,
    })
}

/// English display label for an ayanamsha.
#[must_use]
pub fn ayanamsha_en(ayanamsha: Ayanamsha) -> Option<&'static str> {
    Some(match ayanamsha {
        Ayanamsha::FaganBradley => "Fagan/Bradley",
        Ayanamsha::Lahiri => "Lahiri (Chitrapaksha)",
        Ayanamsha::Krishnamurti => "Krishnamurti (KP)",
        Ayanamsha::Raman => "Raman",
        Ayanamsha::J2000Zero => "J2000 Zero",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiephemeris_astro::aspects::Aspect;

    /// The whole point of the `Option` returns: a new `#[non_exhaustive]`
    /// variant upstream must break a test here, not ship unlabelled.
    #[test]
    fn every_sign_has_both_japanese_forms() {
        for sign in Sign::ALL {
            assert!(sign_ja(sign).is_some(), "{}", sign.name());
            assert!(sign_ja_classical(sign).is_some(), "{}", sign.name());
        }
    }

    #[test]
    fn every_planet_has_a_japanese_label() {
        for planet in Planet::ALL {
            assert!(planet_ja(planet).is_some(), "{}", planet.name());
        }
    }

    #[test]
    fn every_element_and_modality_has_a_japanese_label() {
        for element in Element::ALL {
            assert!(element_ja(element).is_some(), "{}", element.name());
        }
        for modality in Modality::ALL {
            assert!(modality_ja(modality).is_some(), "{}", modality.name());
        }
    }

    #[test]
    fn every_aspect_kind_has_a_japanese_label() {
        for aspect in Aspect::ALL {
            assert!(aspect_ja(aspect.kind).is_some(), "{}", aspect.kind.name());
        }
    }

    #[test]
    fn every_house_system_has_both_labels() {
        for system in HouseSystem::ALL {
            assert!(house_system_ja(system).is_some(), "{}", system.name());
            assert!(house_system_en(system).is_some(), "{}", system.name());
        }
    }

    #[test]
    fn every_ayanamsha_has_both_labels() {
        for ayanamsha in Ayanamsha::ALL_NAMED {
            assert!(ayanamsha_ja(ayanamsha).is_some());
            assert!(ayanamsha_en(ayanamsha).is_some());
        }
    }

    #[test]
    fn every_sect_motion_and_declination_aspect_has_a_label() {
        for sect in Sect::ALL {
            assert!(sect_ja(sect).is_some(), "{}", sect.name());
        }
        for motion in MotionState::ALL {
            assert!(motion_ja(motion).is_some(), "{}", motion.name());
        }
        for aspect in DeclinationAspect::ALL {
            assert!(declination_aspect_ja(aspect).is_some(), "{}", aspect.name());
        }
    }

    #[test]
    fn japanese_labels_are_distinct_within_a_scheme() {
        let signs: Vec<_> = Sign::ALL.iter().filter_map(|s| sign_ja(*s)).collect();
        let mut sorted = signs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), signs.len(), "duplicate Japanese sign label");
    }
}
