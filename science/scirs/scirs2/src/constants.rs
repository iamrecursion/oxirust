//! Physical and mathematical constants
//!
//! This module provides a comprehensive collection of physical and mathematical constants,
//! mirroring SciPy's constants module.

/// Mathematical constants
pub mod math {
    /// The mathematical constant π (pi)
    pub const PI: f64 = std::f64::consts::PI;
    
    /// 2 times π (2π)
    pub const TAU: f64 = 2.0 * PI;

    /// The mathematical constant e (Euler's number)
    pub const E: f64 = std::f64::consts::E;

    /// Euler-Mascheroni constant (γ)
    pub const EULER: f64 = 0.57721566490153286060;

    /// Golden ratio
    pub const GOLDEN: f64 = 1.618033988749894848;

    /// Square root of 2
    pub const SQRT2: f64 = std::f64::consts::SQRT_2;

    /// Square root of π
    pub const SQRTPI: f64 = 1.7724538509055160272;
}

/// Physical constants
pub mod physical {
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

    /// Gravitational constant (N m^2 / kg^2)
    pub const GRAVITATIONAL_CONSTANT: f64 = 6.67430e-11;

    /// Planck constant (J s)
    pub const PLANCK: f64 = 6.62607015e-34;

    /// Reduced Planck constant (J s)
    pub const HBAR: f64 = 1.0545718176461565e-34;

    /// Elementary charge (C)
    pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

    /// Electron mass (kg)
    pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

    /// Proton mass (kg)
    pub const PROTON_MASS: f64 = 1.67262192369e-27;

    /// Neutron mass (kg)
    pub const NEUTRON_MASS: f64 = 1.67492749804e-27;

    /// Boltzmann constant (J/K)
    pub const BOLTZMANN: f64 = 1.380649e-23;

    /// Avogadro constant (1/mol)
    pub const AVOGADRO: f64 = 6.02214076e23;

    /// Gas constant (J/(mol K))
    pub const GAS_CONSTANT: f64 = 8.31446261815324;

    /// Fine-structure constant
    pub const FINE_STRUCTURE: f64 = 7.2973525693e-3;

    /// Rydberg constant (1/m)
    pub const RYDBERG: f64 = 10973731.568160;

    /// Standard acceleration of gravity (m/s^2)
    pub const STANDARD_GRAVITY: f64 = 9.80665;

    /// Vacuum electric permittivity (F/m)
    pub const VACUUM_PERMITTIVITY: f64 = 8.8541878128e-12;

    /// Vacuum magnetic permeability (H/m)
    pub const VACUUM_PERMEABILITY: f64 = 1.25663706212e-6;

    /// Stefan-Boltzmann constant (W/(m^2 K^4))
    pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
}

/// Astronomical constants
pub mod astronomical {
    /// Astronomical unit (m)
    pub const ASTRONOMICAL_UNIT: f64 = 1.495978707e11;

    /// Light year (m)
    pub const LIGHT_YEAR: f64 = 9.460730472580800e15;

    /// Parsec (m)
    pub const PARSEC: f64 = 3.0856775814671916e16;

    /// Solar mass (kg)
    pub const SOLAR_MASS: f64 = 1.9884e30;

    /// Earth mass (kg)
    pub const EARTH_MASS: f64 = 5.9722e24;

    /// Earth radius (m) - Equatorial
    pub const EARTH_RADIUS: f64 = 6.3781e6;

    /// Earth-Moon distance (m) - Mean
    pub const EARTH_MOON_DISTANCE: f64 = 3.84399e8;

    /// Jupiter mass (kg)
    pub const JUPITER_MASS: f64 = 1.8982e27;

    /// Jupiter radius (m) - Equatorial
    pub const JUPITER_RADIUS: f64 = 7.1492e7;
}

/// Unit conversion factors
pub mod convert {
    /// Degrees to radians conversion factor
    pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

    /// Radians to degrees conversion factor
    pub const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

    /// Inch to meter conversion factor
    pub const INCH_TO_METER: f64 = 0.0254;

    /// Foot to meter conversion factor
    pub const FOOT_TO_METER: f64 = 0.3048;

    /// Yard to meter conversion factor
    pub const YARD_TO_METER: f64 = 0.9144;

    /// Mile to meter conversion factor
    pub const MILE_TO_METER: f64 = 1609.344;

    /// Pound to kilogram conversion factor
    pub const POUND_TO_KG: f64 = 0.45359237;

    /// Ounce to kilogram conversion factor
    pub const OUNCE_TO_KG: f64 = 0.028349523125;

    /// Gallon (US) to cubic meter conversion factor
    pub const GALLON_TO_CUBIC_METER: f64 = 0.003785411784;

    /// Electron volt to joule conversion factor
    pub const EV_TO_JOULE: f64 = 1.602176634e-19;

    /// Calorie to joule conversion factor
    pub const CALORIE_TO_JOULE: f64 = 4.184;

    /// Atmosphere to pascal conversion factor
    pub const ATM_TO_PASCAL: f64 = 101325.0;
}

/// Common unit prefixes (as powers of 10)
pub mod prefix {
    /// yotta (Y) = 10^24
    pub const YOTTA: f64 = 1.0e24;
    
    /// zetta (Z) = 10^21
    pub const ZETTA: f64 = 1.0e21;
    
    /// exa (E) = 10^18
    pub const EXA: f64 = 1.0e18;
    
    /// peta (P) = 10^15
    pub const PETA: f64 = 1.0e15;
    
    /// tera (T) = 10^12
    pub const TERA: f64 = 1.0e12;
    
    /// giga (G) = 10^9
    pub const GIGA: f64 = 1.0e9;
    
    /// mega (M) = 10^6
    pub const MEGA: f64 = 1.0e6;
    
    /// kilo (k) = 10^3
    pub const KILO: f64 = 1.0e3;
    
    /// hecto (h) = 10^2
    pub const HECTO: f64 = 1.0e2;
    
    /// deca (da) = 10^1
    pub const DECA: f64 = 1.0e1;
    
    /// deci (d) = 10^-1
    pub const DECI: f64 = 1.0e-1;
    
    /// centi (c) = 10^-2
    pub const CENTI: f64 = 1.0e-2;
    
    /// milli (m) = 10^-3
    pub const MILLI: f64 = 1.0e-3;
    
    /// micro (μ) = 10^-6
    pub const MICRO: f64 = 1.0e-6;
    
    /// nano (n) = 10^-9
    pub const NANO: f64 = 1.0e-9;
    
    /// pico (p) = 10^-12
    pub const PICO: f64 = 1.0e-12;
    
    /// femto (f) = 10^-15
    pub const FEMTO: f64 = 1.0e-15;
    
    /// atto (a) = 10^-18
    pub const ATTO: f64 = 1.0e-18;
    
    /// zepto (z) = 10^-21
    pub const ZEPTO: f64 = 1.0e-21;
    
    /// yocto (y) = 10^-24
    pub const YOCTO: f64 = 1.0e-24;
}

// Re-export for ease of use
pub use math::*;
pub use physical::*;
pub use astronomical::*;
pub use convert::*;
pub use prefix::*;
