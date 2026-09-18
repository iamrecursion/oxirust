//! Screen and game capture functionality.
//!
//! This module provides efficient screen capture capabilities optimized for
//! game streaming, including:
//!
//! - Monitor capture (entire screen or specific monitor)
//! - Window capture (specific application window)
//! - Region capture (arbitrary screen region)
//! - Game-specific optimizations
//! - Cursor capture and overlay

pub mod cursor;
pub mod game;
pub mod hooks;
pub mod screen;

pub use cursor::{CursorCapture, CursorInfo};
pub use game::{GameCapture, GameProfile};
pub use hooks::{CaptureHook, CaptureRect, FpsController, HookRegistry};
pub use screen::{CaptureRegion, MonitorInfo, ScreenCapture};
