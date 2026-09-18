//! NTP (Network Time Protocol) implementation - RFC 5905.
//!
//! Provides NTP v4 client functionality with server pool support.

#[cfg(not(target_arch = "wasm32"))]
pub mod client;
pub mod filter;
pub mod nts;
pub mod packet;
pub mod pool;
pub mod stratum;

#[cfg(not(target_arch = "wasm32"))]
pub use client::NtpClient;
pub use packet::{NtpPacket, NtpTimestamp};
pub use pool::ServerPool;
pub use stratum::Stratum;

/// NTP leap indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LeapIndicator {
    /// No warning
    NoWarning = 0,
    /// Last minute has 61 seconds
    Leap61 = 1,
    /// Last minute has 59 seconds
    Leap59 = 2,
    /// Clock not synchronized
    NotSynchronized = 3,
}

impl LeapIndicator {
    /// Convert from u8.
    #[must_use]
    pub fn from_u8(value: u8) -> Self {
        match value & 0x03 {
            0 => Self::NoWarning,
            1 => Self::Leap61,
            2 => Self::Leap59,
            _ => Self::NotSynchronized,
        }
    }

    /// Convert to u8.
    #[must_use]
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// NTP mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// Reserved
    Reserved = 0,
    /// Symmetric active
    SymmetricActive = 1,
    /// Symmetric passive
    SymmetricPassive = 2,
    /// Client
    Client = 3,
    /// Server
    Server = 4,
    /// Broadcast
    Broadcast = 5,
    /// NTP control message
    Control = 6,
    /// Reserved for private use
    Private = 7,
}

impl Mode {
    /// Convert from u8.
    #[must_use]
    pub fn from_u8(value: u8) -> Self {
        match value & 0x07 {
            1 => Self::SymmetricActive,
            2 => Self::SymmetricPassive,
            3 => Self::Client,
            4 => Self::Server,
            5 => Self::Broadcast,
            6 => Self::Control,
            7 => Self::Private,
            _ => Self::Reserved,
        }
    }

    /// Convert to u8.
    #[must_use]
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leap_indicator() {
        assert_eq!(LeapIndicator::from_u8(0), LeapIndicator::NoWarning);
        assert_eq!(LeapIndicator::from_u8(1), LeapIndicator::Leap61);
        assert_eq!(LeapIndicator::from_u8(2), LeapIndicator::Leap59);
        assert_eq!(LeapIndicator::from_u8(3), LeapIndicator::NotSynchronized);
    }

    #[test]
    fn test_mode() {
        assert_eq!(Mode::from_u8(3), Mode::Client);
        assert_eq!(Mode::from_u8(4), Mode::Server);
        assert_eq!(Mode::Client.to_u8(), 3);
        assert_eq!(Mode::Server.to_u8(), 4);
    }
}
