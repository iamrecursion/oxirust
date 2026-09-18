//! Core types and data structures for financial modeling

use std::fmt::Debug;

/// Market quote for calibration and pricing
#[derive(Debug, Clone)]
pub struct MarketQuote {
    /// Asset symbol
    pub symbol: String,
    /// Current price
    pub price: f64,
    /// Bid price
    pub bid: f64,
    /// Ask price
    pub ask: f64,
    /// Volume
    pub volume: f64,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Option type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    /// Call option (right to buy)
    Call,
    /// Put option (right to sell)
    Put,
}

/// Option style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionStyle {
    /// European option (exercise only at maturity)
    European,
    /// American option (exercise any time before maturity)
    American,
    /// Asian option (payoff depends on average price)
    Asian,
    /// Barrier option (activated/deactivated by price level)
    Barrier {
        barrier: f64,
        is_up: bool,
        is_knock_in: bool,
    },
}

/// Financial option specification
#[derive(Debug, Clone)]
pub struct FinancialOption {
    /// Option type (call/put)
    pub option_type: OptionType,
    /// Option style (European/American/etc)
    pub option_style: OptionStyle,
    /// Strike price
    pub strike: f64,
    /// Time to maturity
    pub maturity: f64,
    /// Initial asset price
    pub spot: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
}

/// Available methods for solving financial PDEs
#[derive(Debug, Clone, Copy)]
pub enum FinanceMethod {
    /// Finite difference method
    FiniteDifference,
    /// Monte Carlo simulation
    MonteCarlo { n_paths: usize, antithetic: bool },
    /// Fourier transform methods
    FourierTransform,
    /// Tree methods (binomial/trinomial)
    Tree { n_steps: usize },
}
