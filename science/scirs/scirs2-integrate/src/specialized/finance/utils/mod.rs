//! Financial utilities and helper functions
//!
//! This module provides common utilities, mathematical functions, and helper tools
//! used throughout the finance module.
//!
//! # Modules
//! - `math`: Mathematical utilities for finance
//! - `calibration`: Model calibration tools
//! - `simulation`: Simulation utilities and random generators
//! - `financial_utils`: Day count conventions, calendars, and compounding

pub mod calibration;
pub mod financial_utils;
pub mod math;
pub mod simulation;

// Re-export commonly used utilities
pub use calibration::{
    CalibrationResult, HestonCalibrator, ImpliedVolatilitySurface, LossFunction, OptionQuote,
};
pub use financial_utils::{
    validate_date, BusinessDayConvention, Calendar, CompoundingConvention, DayCountConvention,
    USFederalCalendar, WeekendCalendar,
};
pub use math::{
    bachelier_call, bachelier_put, black_scholes_call, black_scholes_put, d1, d2, delta_call,
    delta_put, gamma, implied_volatility_brent, implied_volatility_newton, norm_cdf, norm_pdf,
    rho_call, rho_put, safe_log, safe_sqrt, theta_call, theta_put, vega, Greeks, SABRParameters,
};
// Deprecated functions
#[allow(deprecated)]
pub use math::{interpolate_smile, vol_surface_arbitrage_free};
pub use simulation::{
    PathConfig, PathGenerator, RandomNumberGenerator, StandardRng, VarianceReduction,
};
