//! CLI value-enum wrappers around `oxiephemeris_astro`'s house-system and
//! sidereal-zodiac (ayanamsha) selectors, shared by `houses` and `chart`.

use clap::ValueEnum;
use oxiephemeris_astro::ayanamsha::Ayanamsha;
use oxiephemeris_astro::houses::HouseSystem;

/// House-system selector (`--system`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum HouseSystemArg {
    /// Placidus.
    Placidus,
    /// Koch ("birthplace houses").
    Koch,
    /// Whole Sign.
    WholeSign,
    /// Equal.
    Equal,
    /// Porphyry.
    Porphyry,
    /// Regiomontanus.
    Regiomontanus,
    /// Campanus.
    Campanus,
}

impl HouseSystemArg {
    /// Maps to the `oxiephemeris_astro` house system.
    #[must_use]
    pub const fn to_house_system(self) -> HouseSystem {
        match self {
            Self::Placidus => HouseSystem::Placidus,
            Self::Koch => HouseSystem::Koch,
            Self::WholeSign => HouseSystem::WholeSign,
            Self::Equal => HouseSystem::Equal,
            Self::Porphyry => HouseSystem::Porphyry,
            Self::Regiomontanus => HouseSystem::Regiomontanus,
            Self::Campanus => HouseSystem::Campanus,
        }
    }

    /// Display/message name (title case, matching the system's common
    /// English name).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Placidus => "Placidus",
            Self::Koch => "Koch",
            Self::WholeSign => "Whole Sign",
            Self::Equal => "Equal",
            Self::Porphyry => "Porphyry",
            Self::Regiomontanus => "Regiomontanus",
            Self::Campanus => "Campanus",
        }
    }

    /// JSON `"system"` field value — identical to the `--system` value
    /// string (kebab-case, lowercase).
    #[must_use]
    pub const fn json_name(self) -> &'static str {
        match self {
            Self::Placidus => "placidus",
            Self::Koch => "koch",
            Self::WholeSign => "whole-sign",
            Self::Equal => "equal",
            Self::Porphyry => "porphyry",
            Self::Regiomontanus => "regiomontanus",
            Self::Campanus => "campanus",
        }
    }
}

/// Sidereal-zodiac (ayanamsha) selector (`--sidereal`); omitting the flag
/// keeps tropical (Western) longitudes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SiderealArg {
    /// Fagan/Bradley (Western siderealist).
    FaganBradley,
    /// N.C. Lahiri / Chitrapaksha.
    Lahiri,
    /// K.S. Krishnamurti ("KP").
    Krishnamurti,
    /// B.V. Raman.
    Raman,
}

impl SiderealArg {
    /// Maps to the `oxiephemeris_astro` ayanamsha.
    #[must_use]
    pub const fn to_ayanamsha(self) -> Ayanamsha {
        match self {
            Self::FaganBradley => Ayanamsha::FaganBradley,
            Self::Lahiri => Ayanamsha::Lahiri,
            Self::Krishnamurti => Ayanamsha::Krishnamurti,
            Self::Raman => Ayanamsha::Raman,
        }
    }

    /// JSON `"sidereal"` field value — identical to the `--sidereal`
    /// value string.
    #[must_use]
    pub const fn json_name(self) -> &'static str {
        match self {
            Self::FaganBradley => "fagan-bradley",
            Self::Lahiri => "lahiri",
            Self::Krishnamurti => "krishnamurti",
            Self::Raman => "raman",
        }
    }
}
