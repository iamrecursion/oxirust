//! Channel routing and signal distribution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Route configuration for signal routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteConfig {
    /// Source channel ID.
    pub source: String,

    /// Destination output ID.
    pub destination: String,

    /// Whether this route is active.
    pub active: bool,

    /// Audio routing (source track -> destination track).
    pub audio_routes: Vec<(u32, u32)>,

    /// Video routing enabled.
    pub video_enabled: bool,

    /// Processing to apply.
    pub processing: Vec<ProcessingStep>,
}

/// Processing step for routed signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStep {
    /// Audio level adjustment.
    AudioLevel {
        /// Adjustment in dB.
        db: f64,
    },

    /// Video scaling.
    VideoScale {
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
    },

    /// Frame rate conversion.
    FrameRateConvert {
        /// Target frame rate.
        fps: f64,
    },

    /// Color space conversion.
    ColorSpaceConvert {
        /// Target color space.
        target: String,
    },

    /// Delay/offset.
    Delay {
        /// Delay in milliseconds.
        ms: u32,
    },
}

impl RouteConfig {
    /// Creates a new route configuration.
    #[must_use]
    pub fn new<S: Into<String>>(source: S, destination: S) -> Self {
        Self {
            source: source.into(),
            destination: destination.into(),
            active: true,
            audio_routes: Vec::new(),
            video_enabled: true,
            processing: Vec::new(),
        }
    }

    /// Adds an audio route.
    #[must_use]
    pub fn with_audio_route(mut self, source_track: u32, dest_track: u32) -> Self {
        self.audio_routes.push((source_track, dest_track));
        self
    }

    /// Disables video routing.
    #[must_use]
    pub const fn without_video(mut self) -> Self {
        self.video_enabled = false;
        self
    }

    /// Adds a processing step.
    #[must_use]
    pub fn with_processing(mut self, step: ProcessingStep) -> Self {
        self.processing.push(step);
        self
    }

    /// Activates this route.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivates this route.
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}

/// Channel router for managing signal routes.
#[derive(Debug, Default)]
pub struct ChannelRouter {
    routes: HashMap<String, RouteConfig>,
}

impl ChannelRouter {
    /// Creates a new channel router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a route.
    pub fn add_route<S: Into<String>>(&mut self, id: S, route: RouteConfig) {
        self.routes.insert(id.into(), route);
    }

    /// Removes a route.
    pub fn remove_route(&mut self, route_id: &str) {
        self.routes.remove(route_id);
    }

    /// Gets a route by ID.
    #[must_use]
    pub fn get_route(&self, route_id: &str) -> Option<&RouteConfig> {
        self.routes.get(route_id)
    }

    /// Gets a mutable route by ID.
    pub fn get_route_mut(&mut self, route_id: &str) -> Option<&mut RouteConfig> {
        self.routes.get_mut(route_id)
    }

    /// Activates a route.
    pub fn activate_route(&mut self, route_id: &str) {
        if let Some(route) = self.routes.get_mut(route_id) {
            route.activate();
        }
    }

    /// Deactivates a route.
    pub fn deactivate_route(&mut self, route_id: &str) {
        if let Some(route) = self.routes.get_mut(route_id) {
            route.deactivate();
        }
    }

    /// Gets all active routes.
    #[must_use]
    pub fn get_active_routes(&self) -> Vec<&RouteConfig> {
        self.routes.values().filter(|r| r.active).collect()
    }

    /// Gets routes for a specific source.
    #[must_use]
    pub fn get_routes_for_source(&self, source_id: &str) -> Vec<&RouteConfig> {
        self.routes
            .values()
            .filter(|r| r.source == source_id)
            .collect()
    }

    /// Gets routes for a specific destination.
    #[must_use]
    pub fn get_routes_for_destination(&self, dest_id: &str) -> Vec<&RouteConfig> {
        self.routes
            .values()
            .filter(|r| r.destination == dest_id)
            .collect()
    }

    /// Clears all routes.
    pub fn clear(&mut self) {
        self.routes.clear();
    }

    /// Returns the number of routes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Returns true if there are no routes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

/// Pre-configured routing matrices.
pub mod presets {
    use super::{ProcessingStep, RouteConfig};

    /// Creates a simple 1:1 route.
    #[must_use]
    pub fn simple_route(source: &str, destination: &str) -> RouteConfig {
        RouteConfig::new(source, destination)
            .with_audio_route(0, 0)
            .with_audio_route(1, 1)
    }

    /// Creates a route with downscaling.
    #[must_use]
    pub fn downscale_route(
        source: &str,
        destination: &str,
        width: u32,
        height: u32,
    ) -> RouteConfig {
        RouteConfig::new(source, destination)
            .with_audio_route(0, 0)
            .with_audio_route(1, 1)
            .with_processing(ProcessingStep::VideoScale { width, height })
    }

    /// Creates an audio-only route.
    #[must_use]
    pub fn audio_only_route(source: &str, destination: &str) -> RouteConfig {
        RouteConfig::new(source, destination)
            .with_audio_route(0, 0)
            .with_audio_route(1, 1)
            .without_video()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_config() {
        let route = RouteConfig::new("channel1", "output1")
            .with_audio_route(0, 0)
            .with_audio_route(1, 1)
            .with_processing(ProcessingStep::AudioLevel { db: -3.0 });

        assert!(route.active);
        assert_eq!(route.audio_routes.len(), 2);
        assert_eq!(route.processing.len(), 1);
    }

    #[test]
    fn test_channel_router() {
        let mut router = ChannelRouter::new();
        let route = RouteConfig::new("channel1", "output1");

        router.add_route("route1", route);
        assert_eq!(router.len(), 1);

        let active = router.get_active_routes();
        assert_eq!(active.len(), 1);

        router.deactivate_route("route1");
        let active = router.get_active_routes();
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_route_filtering() {
        let mut router = ChannelRouter::new();

        router.add_route("route1", RouteConfig::new("channel1", "output1"));
        router.add_route("route2", RouteConfig::new("channel1", "output2"));
        router.add_route("route3", RouteConfig::new("channel2", "output1"));

        let routes = router.get_routes_for_source("channel1");
        assert_eq!(routes.len(), 2);

        let routes = router.get_routes_for_destination("output1");
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn test_presets() {
        let route = presets::simple_route("ch1", "out1");
        assert_eq!(route.audio_routes.len(), 2);

        let route = presets::audio_only_route("ch1", "out1");
        assert!(!route.video_enabled);
    }
}
