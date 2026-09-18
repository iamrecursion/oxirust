//! Platform-specific encoding presets.
//!
//! This module contains presets optimized for various video platforms:
//! - YouTube: Multiple quality tiers, HDR support
//! - Vimeo: Professional quality presets
//! - Facebook: Feed videos and ads
//! - Instagram: Feed, stories, reels
//! - TikTok: Vertical video optimization
//! - Twitter: Video posts and ads
//! - LinkedIn: Feed, story, and cover video presets
//! - Twitch: Low-latency and quality streaming tiers
//! - DCP: Digital Cinema Package for theatrical distribution

pub mod dcp;
pub mod facebook;
pub mod instagram;
pub mod linkedin;
pub mod ott;
pub mod tiktok;
pub mod twitch;
pub mod twitter;
pub mod vimeo;
pub mod youtube;
