//! NDI audio channel mapping and routing.
//!
//! Maps logical audio channels (e.g. "Left", "Right", "Center") to physical
//! NDI stream channel indices. Supports re-ordering, down-mixing, and
//! up-mixing between different channel layouts.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// Standard audio channel identifiers following broadcast conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Front left
    Left,
    /// Front right
    Right,
    /// Center / dialogue
    Center,
    /// Low-frequency effects (subwoofer)
    Lfe,
    /// Rear / surround left
    SurroundLeft,
    /// Rear / surround right
    SurroundRight,
    /// Side left (7.1 layouts)
    SideLeft,
    /// Side right (7.1 layouts)
    SideRight,
    /// Auxiliary channel with numeric index
    Aux(u8),
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => write!(f, "L"),
            Self::Right => write!(f, "R"),
            Self::Center => write!(f, "C"),
            Self::Lfe => write!(f, "LFE"),
            Self::SurroundLeft => write!(f, "Ls"),
            Self::SurroundRight => write!(f, "Rs"),
            Self::SideLeft => write!(f, "Lss"),
            Self::SideRight => write!(f, "Rss"),
            Self::Aux(n) => write!(f, "Aux{n}"),
        }
    }
}

/// A well-known channel layout preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Single mono channel
    Mono,
    /// Standard left/right stereo
    Stereo,
    /// 5.1 surround (L R C LFE Ls Rs)
    Surround51,
    /// 7.1 surround (L R C LFE Ls Rs Lss Rss)
    Surround71,
}

impl ChannelLayout {
    /// Return the ordered list of channels in this layout.
    #[must_use]
    pub fn channels(self) -> Vec<Channel> {
        match self {
            Self::Mono => vec![Channel::Center],
            Self::Stereo => vec![Channel::Left, Channel::Right],
            Self::Surround51 => vec![
                Channel::Left,
                Channel::Right,
                Channel::Center,
                Channel::Lfe,
                Channel::SurroundLeft,
                Channel::SurroundRight,
            ],
            Self::Surround71 => vec![
                Channel::Left,
                Channel::Right,
                Channel::Center,
                Channel::Lfe,
                Channel::SurroundLeft,
                Channel::SurroundRight,
                Channel::SideLeft,
                Channel::SideRight,
            ],
        }
    }

    /// Number of channels in this layout.
    #[must_use]
    pub fn count(self) -> usize {
        self.channels().len()
    }
}

/// A single routing entry: maps a source channel to a destination channel
/// with a linear gain coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteEntry {
    /// Source channel identifier
    pub src: Channel,
    /// Destination channel identifier
    pub dst: Channel,
    /// Linear gain applied to routed signal (1.0 = unity)
    pub gain: f64,
}

impl RouteEntry {
    /// Create a unity-gain route from `src` to `dst`.
    #[must_use]
    pub fn new(src: Channel, dst: Channel) -> Self {
        Self {
            src,
            dst,
            gain: 1.0,
        }
    }

    /// Create a route with a specific gain.
    #[must_use]
    pub fn with_gain(src: Channel, dst: Channel, gain: f64) -> Self {
        Self { src, dst, gain }
    }
}

/// Manages a mapping table from source channels to destination channels.
///
/// Used to remap NDI audio channel layouts when the sender and receiver
/// use different arrangements.
#[derive(Debug, Clone)]
pub struct ChannelMap {
    /// Source layout (what the sender provides)
    source_layout: ChannelLayout,
    /// Destination layout (what the receiver expects)
    dest_layout: ChannelLayout,
    /// Explicit routes; if empty, defaults are computed.
    routes: Vec<RouteEntry>,
    /// Name for debug display
    name: String,
}

impl ChannelMap {
    /// Create a new channel map between two layouts.
    #[must_use]
    pub fn new(source: ChannelLayout, dest: ChannelLayout) -> Self {
        let routes = Self::default_routes(source, dest);
        Self {
            source_layout: source,
            dest_layout: dest,
            routes,
            name: format!("{source:?} -> {dest:?}"),
        }
    }

    /// Source layout.
    #[must_use]
    pub fn source_layout(&self) -> ChannelLayout {
        self.source_layout
    }

    /// Destination layout.
    #[must_use]
    pub fn dest_layout(&self) -> ChannelLayout {
        self.dest_layout
    }

    /// All active routes.
    #[must_use]
    pub fn routes(&self) -> &[RouteEntry] {
        &self.routes
    }

    /// Display name of the mapping.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Replace all routes with custom ones.
    pub fn set_routes(&mut self, routes: Vec<RouteEntry>) {
        self.routes = routes;
    }

    /// Add a single route.
    pub fn add_route(&mut self, route: RouteEntry) {
        self.routes.push(route);
    }

    /// Remove all routes whose destination matches `dst`.
    pub fn remove_dest(&mut self, dst: Channel) {
        self.routes.retain(|r| r.dst != dst);
    }

    /// Build a gain matrix keyed by `(src_index, dst_index)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn gain_matrix(&self) -> HashMap<(usize, usize), f64> {
        let src_ch = self.source_layout.channels();
        let dst_ch = self.dest_layout.channels();
        let mut matrix = HashMap::new();
        for route in &self.routes {
            if let Some(si) = src_ch.iter().position(|c| *c == route.src) {
                if let Some(di) = dst_ch.iter().position(|c| *c == route.dst) {
                    let entry = matrix.entry((si, di)).or_insert(0.0);
                    *entry += route.gain;
                }
            }
        }
        matrix
    }

    /// Apply the channel map to interleaved f32 samples.
    ///
    /// `input` contains interleaved samples with `source_layout.count()` channels.
    /// Returns interleaved samples with `dest_layout.count()` channels.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_f32(&self, input: &[f32]) -> Vec<f32> {
        let src_count = self.source_layout.count();
        let dst_count = self.dest_layout.count();
        if src_count == 0 || dst_count == 0 {
            return Vec::new();
        }
        let frame_count = input.len() / src_count;
        let matrix = self.gain_matrix();
        let mut output = vec![0.0f32; frame_count * dst_count];
        for frame in 0..frame_count {
            for (&(si, di), &gain) in &matrix {
                let g = gain as f32;
                output[frame * dst_count + di] += input[frame * src_count + si] * g;
            }
        }
        output
    }

    /// Compute sensible default routes when converting between layouts.
    fn default_routes(source: ChannelLayout, dest: ChannelLayout) -> Vec<RouteEntry> {
        use Channel::{Center, Left, Lfe, Right, SideLeft, SideRight, SurroundLeft, SurroundRight};
        match (source, dest) {
            (a, b) if a == b => a
                .channels()
                .into_iter()
                .map(|ch| RouteEntry::new(ch, ch))
                .collect(),
            // Stereo -> Mono: sum L+R at -3 dB each
            (ChannelLayout::Stereo, ChannelLayout::Mono) => {
                vec![
                    RouteEntry::with_gain(Left, Center, 0.707),
                    RouteEntry::with_gain(Right, Center, 0.707),
                ]
            }
            // Mono -> Stereo: duplicate center to both
            (ChannelLayout::Mono, ChannelLayout::Stereo) => {
                vec![
                    RouteEntry::new(Center, Left),
                    RouteEntry::new(Center, Right),
                ]
            }
            // 5.1 -> Stereo: standard ITU down-mix
            (ChannelLayout::Surround51, ChannelLayout::Stereo) => {
                vec![
                    RouteEntry::new(Left, Left),
                    RouteEntry::new(Right, Right),
                    RouteEntry::with_gain(Center, Left, 0.707),
                    RouteEntry::with_gain(Center, Right, 0.707),
                    RouteEntry::with_gain(Lfe, Left, 0.5),
                    RouteEntry::with_gain(Lfe, Right, 0.5),
                    RouteEntry::with_gain(SurroundLeft, Left, 0.707),
                    RouteEntry::with_gain(SurroundRight, Right, 0.707),
                ]
            }
            // 7.1 -> Stereo
            (ChannelLayout::Surround71, ChannelLayout::Stereo) => {
                vec![
                    RouteEntry::new(Left, Left),
                    RouteEntry::new(Right, Right),
                    RouteEntry::with_gain(Center, Left, 0.707),
                    RouteEntry::with_gain(Center, Right, 0.707),
                    RouteEntry::with_gain(Lfe, Left, 0.5),
                    RouteEntry::with_gain(Lfe, Right, 0.5),
                    RouteEntry::with_gain(SurroundLeft, Left, 0.707),
                    RouteEntry::with_gain(SurroundRight, Right, 0.707),
                    RouteEntry::with_gain(SideLeft, Left, 0.5),
                    RouteEntry::with_gain(SideRight, Right, 0.5),
                ]
            }
            // Stereo -> 5.1: phantom center from L+R
            (ChannelLayout::Stereo, ChannelLayout::Surround51) => {
                vec![
                    RouteEntry::new(Left, Left),
                    RouteEntry::new(Right, Right),
                    RouteEntry::with_gain(Left, Center, 0.5),
                    RouteEntry::with_gain(Right, Center, 0.5),
                ]
            }
            // fallback: pass-through matching channels
            _ => {
                let src_ch = source.channels();
                let dst_ch = dest.channels();
                let mut routes = Vec::new();
                for s in &src_ch {
                    if dst_ch.contains(s) {
                        routes.push(RouteEntry::new(*s, *s));
                    }
                }
                routes
            }
        }
    }
}

/// Summary statistics for a channel map.
#[derive(Debug, Clone)]
pub struct ChannelMapStats {
    /// Number of active routes.
    pub route_count: usize,
    /// Source channel count.
    pub source_channels: usize,
    /// Destination channel count.
    pub dest_channels: usize,
    /// Average gain across all routes.
    pub avg_gain: f64,
}

impl ChannelMap {
    /// Compute summary statistics.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> ChannelMapStats {
        let avg = if self.routes.is_empty() {
            0.0
        } else {
            self.routes.iter().map(|r| r.gain).sum::<f64>() / self.routes.len() as f64
        };
        ChannelMapStats {
            route_count: self.routes.len(),
            source_channels: self.source_layout.count(),
            dest_channels: self.dest_layout.count(),
            avg_gain: avg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_display() {
        assert_eq!(Channel::Left.to_string(), "L");
        assert_eq!(Channel::SurroundLeft.to_string(), "Ls");
        assert_eq!(Channel::Aux(3).to_string(), "Aux3");
    }

    #[test]
    fn test_layout_channel_counts() {
        assert_eq!(ChannelLayout::Mono.count(), 1);
        assert_eq!(ChannelLayout::Stereo.count(), 2);
        assert_eq!(ChannelLayout::Surround51.count(), 6);
        assert_eq!(ChannelLayout::Surround71.count(), 8);
    }

    #[test]
    fn test_identity_map_stereo() {
        let map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Stereo);
        assert_eq!(map.routes().len(), 2);
        assert!((map.routes()[0].gain - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stereo_to_mono_routes() {
        let map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Mono);
        assert_eq!(map.routes().len(), 2);
        for r in map.routes() {
            assert_eq!(r.dst, Channel::Center);
            assert!((r.gain - 0.707).abs() < 0.001);
        }
    }

    #[test]
    fn test_mono_to_stereo_routes() {
        let map = ChannelMap::new(ChannelLayout::Mono, ChannelLayout::Stereo);
        assert_eq!(map.routes().len(), 2);
    }

    #[test]
    fn test_apply_f32_identity() {
        let map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Stereo);
        let input = vec![0.5f32, -0.5, 1.0, -1.0];
        let output = map.apply_f32(&input);
        assert_eq!(output.len(), 4);
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_f32_stereo_to_mono() {
        let map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Mono);
        // L=1.0, R=1.0 => center = 1.0*0.707 + 1.0*0.707 = 1.414
        let input = vec![1.0f32, 1.0];
        let output = map.apply_f32(&input);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_gain_matrix_entries() {
        let map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Stereo);
        let matrix = map.gain_matrix();
        assert_eq!(matrix.len(), 2);
        assert!((matrix[&(0, 0)] - 1.0).abs() < f64::EPSILON);
        assert!((matrix[&(1, 1)] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_route() {
        let mut map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Stereo);
        map.set_routes(vec![RouteEntry::with_gain(
            Channel::Left,
            Channel::Right,
            0.5,
        )]);
        assert_eq!(map.routes().len(), 1);
    }

    #[test]
    fn test_add_route() {
        let mut map = ChannelMap::new(ChannelLayout::Mono, ChannelLayout::Mono);
        let before = map.routes().len();
        map.add_route(RouteEntry::with_gain(Channel::Aux(0), Channel::Center, 0.1));
        assert_eq!(map.routes().len(), before + 1);
    }

    #[test]
    fn test_remove_dest() {
        let mut map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Stereo);
        map.remove_dest(Channel::Right);
        assert!(map.routes().iter().all(|r| r.dst != Channel::Right));
    }

    #[test]
    fn test_stats() {
        let map = ChannelMap::new(ChannelLayout::Stereo, ChannelLayout::Stereo);
        let s = map.stats();
        assert_eq!(s.route_count, 2);
        assert_eq!(s.source_channels, 2);
        assert_eq!(s.dest_channels, 2);
        assert!((s.avg_gain - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_surround51_to_stereo_routes() {
        let map = ChannelMap::new(ChannelLayout::Surround51, ChannelLayout::Stereo);
        // Should have 8 routes: L->L, R->R, C->L, C->R, LFE->L, LFE->R, Ls->L, Rs->R
        assert_eq!(map.routes().len(), 8);
    }

    #[test]
    fn test_name() {
        let map = ChannelMap::new(ChannelLayout::Mono, ChannelLayout::Stereo);
        assert!(map.name().contains("Mono"));
        assert!(map.name().contains("Stereo"));
    }
}
