//! Overlay system for stream graphics.

pub mod alert;
pub mod hud;
pub mod scoreboard;
pub mod system;
pub mod widget;

pub use alert::{Alert, AlertManager, AlertType};
pub use hud::{BannerQueue, HudOverlay, StatsPanel};
pub use scoreboard::{Scoreboard, ScoreboardConfig};
pub use system::{
    FpsCounterOverlay, ImageOverlay, OverlayContent, OverlayLayer, OverlaySystem, PerfPanelOverlay,
    RectOverlay, TextOverlay,
};
pub use widget::{Widget, WidgetConfig, WidgetType};
