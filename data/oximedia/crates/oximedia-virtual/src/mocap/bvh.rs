//! BVH (Biovision Hierarchy) motion capture file format parser.
//!
//! BVH is the de-facto standard interchange format for motion capture data,
//! used by Vicon, OptiTrack, Rokoko, and virtually every major NLE / DCC tool.
//!
//! # Format overview
//! A BVH file consists of two sections:
//! - **HIERARCHY** – a tree of joints, each with channel definitions
//!   (`Xposition`, `Yposition`, `Zposition`, `Zrotation`, `Yrotation`, `Xrotation`)
//! - **MOTION** – `Frames:` count, `Frame Time:` interval, then one line of
//!   space-separated floats per frame (values ordered by channel definition)
//!
//! This parser produces a [`BvhClip`] containing the skeleton and all frames.

use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// Channel type in a BVH joint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BvhChannel {
    Xposition,
    Yposition,
    Zposition,
    Xrotation,
    Yrotation,
    Zrotation,
}

impl BvhChannel {
    /// Parse from a keyword string (case-insensitive).
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "xposition" => Some(Self::Xposition),
            "yposition" => Some(Self::Yposition),
            "zposition" => Some(Self::Zposition),
            "xrotation" => Some(Self::Xrotation),
            "yrotation" => Some(Self::Yrotation),
            "zrotation" => Some(Self::Zrotation),
            _ => None,
        }
    }

    /// Whether this is a translation channel.
    #[must_use]
    pub fn is_translation(self) -> bool {
        matches!(self, Self::Xposition | Self::Yposition | Self::Zposition)
    }
}

/// A joint in the BVH hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BvhJoint {
    /// Joint name.
    pub name: String,
    /// Parent joint index (None for root).
    pub parent: Option<usize>,
    /// Offset from parent in rest pose [x, y, z] (in cm by convention).
    pub offset: [f64; 3],
    /// Ordered list of channels for this joint.
    pub channels: Vec<BvhChannel>,
    /// Indices of child joints.
    pub children: Vec<usize>,
    /// Index of this joint's first channel in the per-frame channel array.
    pub channel_offset: usize,
}

impl BvhJoint {
    /// Whether this joint has translation channels.
    #[must_use]
    pub fn has_translation(&self) -> bool {
        self.channels.iter().any(|c| c.is_translation())
    }
}

/// One frame of motion data: all channel values in joint/channel definition order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BvhFrame {
    /// All channel values for this frame.
    pub channels: Vec<f64>,
}

impl BvhFrame {
    /// Get the value for joint `joint_idx`, channel `chan_idx_in_joint`.
    #[must_use]
    pub fn get_channel(&self, joint: &BvhJoint, chan_idx_in_joint: usize) -> Option<f64> {
        let global = joint.channel_offset + chan_idx_in_joint;
        self.channels.get(global).copied()
    }

    /// Get translation [x, y, z] for a joint (if it has translation channels).
    #[must_use]
    pub fn translation(&self, joint: &BvhJoint) -> Option<[f64; 3]> {
        if !joint.has_translation() {
            return None;
        }
        // Find Xposition, Yposition, Zposition channels
        let mut tx = None;
        let mut ty = None;
        let mut tz = None;
        for (i, ch) in joint.channels.iter().enumerate() {
            let v = joint.channel_offset + i;
            match ch {
                BvhChannel::Xposition => tx = self.channels.get(v).copied(),
                BvhChannel::Yposition => ty = self.channels.get(v).copied(),
                BvhChannel::Zposition => tz = self.channels.get(v).copied(),
                _ => {}
            }
        }
        Some([tx.unwrap_or(0.0), ty.unwrap_or(0.0), tz.unwrap_or(0.0)])
    }

    /// Get Euler rotation angles [x, y, z] in degrees for a joint.
    #[must_use]
    pub fn rotation_deg(&self, joint: &BvhJoint) -> [f64; 3] {
        let mut rx = 0.0;
        let mut ry = 0.0;
        let mut rz = 0.0;
        for (i, ch) in joint.channels.iter().enumerate() {
            let v = joint.channel_offset + i;
            match ch {
                BvhChannel::Xrotation => rx = self.channels.get(v).copied().unwrap_or(0.0),
                BvhChannel::Yrotation => ry = self.channels.get(v).copied().unwrap_or(0.0),
                BvhChannel::Zrotation => rz = self.channels.get(v).copied().unwrap_or(0.0),
                _ => {}
            }
        }
        [rx, ry, rz]
    }
}

/// A complete BVH motion clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BvhClip {
    /// All joints in definition order (root at index 0).
    pub joints: Vec<BvhJoint>,
    /// All frames.
    pub frames: Vec<BvhFrame>,
    /// Frame time in seconds.
    pub frame_time_s: f64,
    /// Total number of channels.
    pub channel_count: usize,
    /// Quick lookup: joint name → joint index.
    pub joint_map: HashMap<String, usize>,
}

impl BvhClip {
    /// Duration of the clip in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        self.frames.len() as f64 * self.frame_time_s
    }

    /// Frame count.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Look up a joint by name.
    #[must_use]
    pub fn joint_by_name(&self, name: &str) -> Option<&BvhJoint> {
        let idx = self.joint_map.get(name)?;
        self.joints.get(*idx)
    }

    /// Root joint (index 0).
    #[must_use]
    pub fn root(&self) -> Option<&BvhJoint> {
        self.joints.first()
    }

    /// Get the frame at the given index.
    #[must_use]
    pub fn frame(&self, idx: usize) -> Option<&BvhFrame> {
        self.frames.get(idx)
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// BVH file parser.
pub struct BvhParser;

impl BvhParser {
    /// Parse BVH content from a string.
    pub fn parse(content: &str) -> Result<BvhClip> {
        let tokens: Vec<&str> = content.split_whitespace().collect();

        let mut pos = 0usize;

        // ---- HIERARCHY section ----
        Self::expect(&tokens, &mut pos, "HIERARCHY")?;

        let mut joints: Vec<BvhJoint> = Vec::new();
        let mut joint_stack: Vec<usize> = Vec::new(); // parent indices
        let mut total_channels = 0usize;

        // Parse joints recursively via an iterative stack
        Self::parse_joints(
            &tokens,
            &mut pos,
            &mut joints,
            &mut joint_stack,
            &mut total_channels,
        )?;

        // Build joint name map
        let mut joint_map = HashMap::new();
        for (i, j) in joints.iter().enumerate() {
            joint_map.insert(j.name.clone(), i);
        }

        // ---- MOTION section ----
        Self::expect(&tokens, &mut pos, "MOTION")?;
        Self::expect(&tokens, &mut pos, "Frames:")?;

        let frame_count: usize = Self::next_token(&tokens, &mut pos)?.parse().map_err(|e| {
            VirtualProductionError::MotionCapture(format!("Invalid frame count: {e}"))
        })?;

        Self::expect(&tokens, &mut pos, "Frame")?;
        Self::expect(&tokens, &mut pos, "Time:")?;

        let frame_time_s: f64 = Self::next_token(&tokens, &mut pos)?.parse().map_err(|e| {
            VirtualProductionError::MotionCapture(format!("Invalid frame time: {e}"))
        })?;

        // Parse frame data
        let mut frames = Vec::with_capacity(frame_count);
        for f in 0..frame_count {
            let mut channels = Vec::with_capacity(total_channels);
            for _ in 0..total_channels {
                let v: f64 = Self::next_token(&tokens, &mut pos).and_then(|t| {
                    t.parse().map_err(|e| {
                        VirtualProductionError::MotionCapture(format!(
                            "Frame {f}: invalid float '{t}': {e}"
                        ))
                    })
                })?;
                channels.push(v);
            }
            frames.push(BvhFrame { channels });
        }

        Ok(BvhClip {
            joints,
            frames,
            frame_time_s,
            channel_count: total_channels,
            joint_map,
        })
    }

    /// Iteratively parse JOINT / ROOT blocks.
    fn parse_joints(
        tokens: &[&str],
        pos: &mut usize,
        joints: &mut Vec<BvhJoint>,
        stack: &mut Vec<usize>, // indices of ancestor joints
        total_channels: &mut usize,
    ) -> Result<()> {
        loop {
            if *pos >= tokens.len() {
                break;
            }

            match tokens[*pos] {
                "ROOT" | "JOINT" => {
                    *pos += 1;
                    let name = Self::next_token(tokens, pos)?.to_string();

                    // Opening brace
                    Self::expect(tokens, pos, "{")?;

                    // OFFSET
                    Self::expect(tokens, pos, "OFFSET")?;
                    let ox: f64 = Self::next_token(tokens, pos)?.parse().map_err(|e| {
                        VirtualProductionError::MotionCapture(format!("offset x: {e}"))
                    })?;
                    let oy: f64 = Self::next_token(tokens, pos)?.parse().map_err(|e| {
                        VirtualProductionError::MotionCapture(format!("offset y: {e}"))
                    })?;
                    let oz: f64 = Self::next_token(tokens, pos)?.parse().map_err(|e| {
                        VirtualProductionError::MotionCapture(format!("offset z: {e}"))
                    })?;

                    // CHANNELS
                    Self::expect(tokens, pos, "CHANNELS")?;
                    let num_channels: usize =
                        Self::next_token(tokens, pos)?.parse().map_err(|e| {
                            VirtualProductionError::MotionCapture(format!("channel count: {e}"))
                        })?;

                    let mut channels = Vec::with_capacity(num_channels);
                    for _ in 0..num_channels {
                        let ch_str = Self::next_token(tokens, pos)?;
                        let ch = BvhChannel::from_str(ch_str).ok_or_else(|| {
                            VirtualProductionError::MotionCapture(format!(
                                "Unknown channel '{ch_str}'"
                            ))
                        })?;
                        channels.push(ch);
                    }

                    let parent = stack.last().copied();
                    let joint_idx = joints.len();
                    let channel_offset = *total_channels;

                    *total_channels += num_channels;

                    // Register as child of parent
                    if let Some(parent_idx) = parent {
                        if let Some(pj) = joints.get_mut(parent_idx) {
                            pj.children.push(joint_idx);
                        }
                    }

                    joints.push(BvhJoint {
                        name,
                        parent,
                        offset: [ox, oy, oz],
                        channels,
                        children: Vec::new(),
                        channel_offset,
                    });

                    stack.push(joint_idx);
                }
                "End" => {
                    // End Site: skip "End" then "Site" then "{ OFFSET x y z }"
                    *pos += 1; // skip "End" (we matched on it but didn't advance)
                    Self::expect(tokens, pos, "Site")?;
                    Self::expect(tokens, pos, "{")?;
                    Self::expect(tokens, pos, "OFFSET")?;
                    *pos += 3; // skip x y z
                    Self::expect(tokens, pos, "}")?;
                }
                "}" => {
                    *pos += 1;
                    stack.pop();
                    if stack.is_empty() {
                        break; // finished hierarchy
                    }
                }
                "MOTION" => break,
                _ => {
                    *pos += 1; // skip unknown token
                }
            }
        }
        Ok(())
    }

    fn expect<'a>(tokens: &'a [&str], pos: &mut usize, expected: &str) -> Result<&'a str> {
        if *pos >= tokens.len() {
            return Err(VirtualProductionError::MotionCapture(format!(
                "Expected '{expected}' but reached end of input"
            )));
        }
        let t = tokens[*pos];
        if t != expected {
            return Err(VirtualProductionError::MotionCapture(format!(
                "Expected '{expected}' but got '{t}'"
            )));
        }
        *pos += 1;
        Ok(t)
    }

    fn next_token<'a>(tokens: &'a [&str], pos: &mut usize) -> Result<&'a str> {
        if *pos >= tokens.len() {
            return Err(VirtualProductionError::MotionCapture(
                "Unexpected end of BVH input".to_string(),
            ));
        }
        let t = tokens[*pos];
        *pos += 1;
        Ok(t)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal single-joint BVH file with 2 frames.
    fn minimal_bvh() -> &'static str {
        "HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    End Site
    {
        OFFSET 0.00 10.00 0.00
    }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.00 90.00 0.00 10.00 0.00 5.00
1.00 91.00 0.00 11.00 1.00 6.00"
    }

    /// Two-joint BVH: Hip with one child (Spine).
    fn two_joint_bvh() -> &'static str {
        "HIERARCHY
ROOT Hips
{
    OFFSET 0.00 0.00 0.00
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.00 10.00 0.00
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.00 10.00 0.00
        }
    }
}
MOTION
Frames: 1
Frame Time: 0.033333
0.00 90.00 0.00 5.00 0.00 0.00 2.00 1.00 0.00"
    }

    #[test]
    fn test_parse_minimal_bvh() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        assert_eq!(clip.joints.len(), 1);
        assert_eq!(clip.frames.len(), 2);
        assert!((clip.frame_time_s - 0.033333).abs() < 1e-5);
    }

    #[test]
    fn test_root_joint_name() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let root = clip.root().expect("should have root");
        assert_eq!(root.name, "Hips");
    }

    #[test]
    fn test_root_has_no_parent() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        assert!(root.parent.is_none());
    }

    #[test]
    fn test_root_offset_zero() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        assert!((root.offset[0]).abs() < 1e-9);
        assert!((root.offset[1]).abs() < 1e-9);
        assert!((root.offset[2]).abs() < 1e-9);
    }

    #[test]
    fn test_channel_count() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        assert_eq!(clip.channel_count, 6); // 6 root channels
    }

    #[test]
    fn test_frame_channel_values() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        let frame0 = clip.frame(0).expect("frame 0");
        let trans = frame0.translation(root).expect("translation");
        assert!((trans[0] - 0.0).abs() < 1e-9, "tx: {}", trans[0]);
        assert!((trans[1] - 90.0).abs() < 1e-9, "ty: {}", trans[1]);
        assert!((trans[2] - 0.0).abs() < 1e-9, "tz: {}", trans[2]);
    }

    #[test]
    fn test_frame1_channel_values() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        let frame1 = clip.frame(1).expect("frame 1");
        let trans = frame1.translation(root).expect("translation");
        assert!((trans[0] - 1.0).abs() < 1e-9);
        assert!((trans[1] - 91.0).abs() < 1e-9);
    }

    #[test]
    fn test_rotation_degrees_frame0() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        let frame0 = clip.frame(0).expect("frame 0");
        let rot = frame0.rotation_deg(root);
        // Zrotation=10, Xrotation=0, Yrotation=5
        // rotation_deg returns [rx, ry, rz] → index 1 is Yrotation
        assert!((rot[1] - 5.0).abs() < 1e-9, "yrot: {}", rot[1]); // Yrotation
    }

    #[test]
    fn test_joint_by_name() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        assert!(clip.joint_by_name("Hips").is_some());
        assert!(clip.joint_by_name("Nonexistent").is_none());
    }

    #[test]
    fn test_duration_s() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        let expected = 2.0 * 0.033333;
        assert!((clip.duration_s() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_two_joint_hierarchy() {
        let clip = BvhParser::parse(two_joint_bvh()).expect("should parse");
        assert_eq!(clip.joints.len(), 2, "should have 2 joints");

        let spine = clip.joint_by_name("Spine").expect("Spine joint");
        assert_eq!(spine.parent, Some(0), "Spine parent should be Hips (idx 0)");
        assert!((spine.offset[1] - 10.0).abs() < 1e-9, "Spine y offset");
    }

    #[test]
    fn test_two_joint_channel_count() {
        let clip = BvhParser::parse(two_joint_bvh()).expect("should parse");
        assert_eq!(clip.channel_count, 9, "6 root + 3 spine");
    }

    #[test]
    fn test_root_has_child_spine() {
        let clip = BvhParser::parse(two_joint_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        assert!(
            root.children.contains(&1),
            "root should list Spine as child"
        );
    }

    #[test]
    fn test_bvh_joint_has_translation() {
        let clip = BvhParser::parse(two_joint_bvh()).expect("should parse");
        let root = clip.root().expect("root");
        let spine = clip.joint_by_name("Spine").expect("Spine");
        assert!(root.has_translation(), "root has translation");
        assert!(!spine.has_translation(), "Spine only has rotation");
    }

    #[test]
    fn test_no_translation_returns_none() {
        let clip = BvhParser::parse(two_joint_bvh()).expect("should parse");
        let spine = clip.joint_by_name("Spine").expect("Spine");
        let frame = clip.frame(0).expect("frame");
        assert!(
            frame.translation(spine).is_none(),
            "Spine has no translation channels"
        );
    }

    #[test]
    fn test_parse_bad_input_fails() {
        let result = BvhParser::parse("GARBAGE DATA THAT IS NOT BVH");
        assert!(result.is_err(), "bad input should fail");
    }

    #[test]
    fn test_frame_out_of_bounds_returns_none() {
        let clip = BvhParser::parse(minimal_bvh()).expect("should parse");
        assert!(clip.frame(100).is_none());
    }
}
