//! Camera tracking for virtual production — FreeD protocol simulation.
//!
//! Provides [`CameraTransform`] for representing full 6-DOF camera pose plus
//! lens data, [`FreeDPacket`] for encoding/decoding the industry-standard
//! FreeD D1 UDP packet, and [`CameraTracker`] for history management with
//! latency compensation and motion prediction.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ---------------------------------------------------------------------------
// CameraTransform
// ---------------------------------------------------------------------------

/// Full camera pose and lens state for virtual production.
///
/// Angular values are in degrees; position values are in millimetres.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraTransform {
    /// Horizontal pan in degrees.
    pub pan_deg: f32,
    /// Vertical tilt in degrees.
    pub tilt_deg: f32,
    /// Lens roll in degrees.
    pub roll_deg: f32,
    /// Lateral (X) position in mm.
    pub x_mm: f32,
    /// Vertical (Y) position in mm.
    pub y_mm: f32,
    /// Depth (Z) position in mm.
    pub z_mm: f32,
    /// Zoom / focal length in mm.
    pub focal_length_mm: f32,
    /// Focus distance in mm.
    pub focus_distance_mm: f32,
    /// Capture timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

impl CameraTransform {
    /// Identity transform: zero position, zero orientation, 50 mm lens, 1 m focus.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            pan_deg: 0.0,
            tilt_deg: 0.0,
            roll_deg: 0.0,
            x_mm: 0.0,
            y_mm: 0.0,
            z_mm: 0.0,
            focal_length_mm: 50.0,
            focus_distance_mm: 1000.0,
            timestamp_ns: 0,
        }
    }

    /// Linearly interpolate between `a` and `b` at factor `t` ∈ [0, 1].
    ///
    /// The timestamp is also interpolated.
    #[must_use]
    pub fn interpolate(a: &Self, b: &Self, t: f32) -> Self {
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        let lerp_u64 = |x: u64, y: u64| {
            let xf = x as f64;
            let yf = y as f64;
            (xf + (yf - xf) * f64::from(t)) as u64
        };
        Self {
            pan_deg: lerp(a.pan_deg, b.pan_deg),
            tilt_deg: lerp(a.tilt_deg, b.tilt_deg),
            roll_deg: lerp(a.roll_deg, b.roll_deg),
            x_mm: lerp(a.x_mm, b.x_mm),
            y_mm: lerp(a.y_mm, b.y_mm),
            z_mm: lerp(a.z_mm, b.z_mm),
            focal_length_mm: lerp(a.focal_length_mm, b.focal_length_mm),
            focus_distance_mm: lerp(a.focus_distance_mm, b.focus_distance_mm),
            timestamp_ns: lerp_u64(a.timestamp_ns, b.timestamp_ns),
        }
    }

    /// Encode to a FreeD D1 packet.
    #[must_use]
    pub fn to_free_d(&self) -> FreeDPacket {
        // FreeD angle scaling: integer value = degrees × 32768
        let angle_scale = 32768.0_f32;
        // FreeD position scaling: integer value = mm × 64
        let pos_scale = 64.0_f32;
        // FreeD zoom/focus: integer value = mm × 1000
        let lens_scale = 1000.0_f32;

        FreeDPacket {
            device_id: 1,
            pan: (self.pan_deg * angle_scale) as i32,
            tilt: (self.tilt_deg * angle_scale) as i32,
            roll: (self.roll_deg * angle_scale) as i32,
            x_pos: (self.x_mm * pos_scale) as i32,
            y_pos: (self.y_mm * pos_scale) as i32,
            z_pos: (self.z_mm * pos_scale) as i32,
            zoom: (self.focal_length_mm * lens_scale) as u32,
            focus: (self.focus_distance_mm * lens_scale) as u32,
            user_data: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// FreeDPacket
// ---------------------------------------------------------------------------

/// FreeD D1 packet (29 bytes) as used in broadcast camera tracking systems.
///
/// Byte layout (big-endian):
/// ```text
/// [0]       message type (0xD1)
/// [1]       device id
/// [2..4]    pan    (3 bytes, signed, big-endian)
/// [5..7]    tilt   (3 bytes, signed, big-endian)
/// [8..10]   roll   (3 bytes, signed, big-endian)
/// [11..13]  x_pos  (3 bytes, signed, big-endian)
/// [14..16]  y_pos  (3 bytes, signed, big-endian)
/// [17..19]  z_pos  (3 bytes, signed, big-endian)
/// [20..23]  zoom   (4 bytes, unsigned, big-endian)
/// [24..27]  focus  (4 bytes, unsigned, big-endian)
/// [28]      checksum (XOR of bytes [0..28])
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FreeDPacket {
    /// Device identifier (1-byte).
    pub device_id: u8,
    /// Pan in 1/32768 degrees.
    pub pan: i32,
    /// Tilt in 1/32768 degrees.
    pub tilt: i32,
    /// Roll in 1/32768 degrees.
    pub roll: i32,
    /// X position in mm × 64.
    pub x_pos: i32,
    /// Y position in mm × 64.
    pub y_pos: i32,
    /// Z position in mm × 64.
    pub z_pos: i32,
    /// Focal length in mm × 1000.
    pub zoom: u32,
    /// Focus distance in mm × 1000.
    pub focus: u32,
    /// User-defined payload (not encoded in standard 29-byte packet).
    pub user_data: u32,
}

/// Read a 3-byte big-endian signed integer from `data[offset..]`.
fn read_i24_be(data: &[u8], offset: usize) -> Option<i32> {
    if data.len() < offset + 3 {
        return None;
    }
    let b0 = data[offset] as u32;
    let b1 = data[offset + 1] as u32;
    let b2 = data[offset + 2] as u32;
    let raw = (b0 << 16) | (b1 << 8) | b2;
    // Sign-extend from 24 bits.
    let signed = if raw & 0x0080_0000 != 0 {
        (raw | 0xFF00_0000) as i32
    } else {
        raw as i32
    };
    Some(signed)
}

/// Read a 4-byte big-endian unsigned integer from `data[offset..]`.
fn read_u32_be(data: &[u8], offset: usize) -> Option<u32> {
    if data.len() < offset + 4 {
        return None;
    }
    Some(
        (data[offset] as u32) << 24
            | (data[offset + 1] as u32) << 16
            | (data[offset + 2] as u32) << 8
            | data[offset + 3] as u32,
    )
}

impl FreeDPacket {
    /// Parse a 29-byte FreeD D1 packet.
    ///
    /// Returns `None` if `data` is shorter than 29 bytes, has the wrong
    /// message type, or fails the XOR checksum.
    #[must_use]
    pub fn decode(data: &[u8]) -> Option<FreeDPacket> {
        if data.len() < 29 {
            return None;
        }
        // Message type must be 0xD1.
        if data[0] != 0xD1 {
            return None;
        }
        // Verify XOR checksum (byte 28 = XOR of bytes 0..=27).
        let expected_checksum: u8 = data[..28].iter().fold(0u8, |acc, &b| acc ^ b);
        if data[28] != expected_checksum {
            return None;
        }

        let device_id = data[1];
        let pan = read_i24_be(data, 2)?;
        let tilt = read_i24_be(data, 5)?;
        let roll = read_i24_be(data, 8)?;
        let x_pos = read_i24_be(data, 11)?;
        let y_pos = read_i24_be(data, 14)?;
        let z_pos = read_i24_be(data, 17)?;
        let zoom = read_u32_be(data, 20)?;
        let focus = read_u32_be(data, 24)?;

        Some(FreeDPacket {
            device_id,
            pan,
            tilt,
            roll,
            x_pos,
            y_pos,
            z_pos,
            zoom,
            focus,
            user_data: 0,
        })
    }

    /// Encode the packet to a 29-byte buffer.
    #[must_use]
    pub fn encode(&self) -> [u8; 29] {
        let mut buf = [0u8; 29];
        buf[0] = 0xD1;
        buf[1] = self.device_id;

        // Helper: write signed 24-bit big-endian.
        let write_i24 = |buf: &mut [u8; 29], offset: usize, val: i32| {
            let raw = (val as u32) & 0x00FF_FFFF;
            buf[offset] = ((raw >> 16) & 0xFF) as u8;
            buf[offset + 1] = ((raw >> 8) & 0xFF) as u8;
            buf[offset + 2] = (raw & 0xFF) as u8;
        };

        write_i24(&mut buf, 2, self.pan);
        write_i24(&mut buf, 5, self.tilt);
        write_i24(&mut buf, 8, self.roll);
        write_i24(&mut buf, 11, self.x_pos);
        write_i24(&mut buf, 14, self.y_pos);
        write_i24(&mut buf, 17, self.z_pos);

        buf[20] = ((self.zoom >> 24) & 0xFF) as u8;
        buf[21] = ((self.zoom >> 16) & 0xFF) as u8;
        buf[22] = ((self.zoom >> 8) & 0xFF) as u8;
        buf[23] = (self.zoom & 0xFF) as u8;

        buf[24] = ((self.focus >> 24) & 0xFF) as u8;
        buf[25] = ((self.focus >> 16) & 0xFF) as u8;
        buf[26] = ((self.focus >> 8) & 0xFF) as u8;
        buf[27] = (self.focus & 0xFF) as u8;

        // Compute and store XOR checksum.
        buf[28] = buf[..28].iter().fold(0u8, |acc, &b| acc ^ b);
        buf
    }

    /// Convert a decoded FreeD packet back to a [`CameraTransform`].
    #[must_use]
    pub fn to_transform(&self) -> CameraTransform {
        let angle_scale = 32768.0_f32;
        let pos_scale = 64.0_f32;
        let lens_scale = 1000.0_f32;

        CameraTransform {
            pan_deg: self.pan as f32 / angle_scale,
            tilt_deg: self.tilt as f32 / angle_scale,
            roll_deg: self.roll as f32 / angle_scale,
            x_mm: self.x_pos as f32 / pos_scale,
            y_mm: self.y_pos as f32 / pos_scale,
            z_mm: self.z_pos as f32 / pos_scale,
            focal_length_mm: self.zoom as f32 / lens_scale,
            focus_distance_mm: self.focus as f32 / lens_scale,
            timestamp_ns: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CameraTracker
// ---------------------------------------------------------------------------

/// Ring-buffer camera tracker with latency compensation and motion prediction.
pub struct CameraTracker {
    /// Recorded transform history (oldest first, most recent last).
    pub history: std::collections::VecDeque<CameraTransform>,
    /// Maximum number of frames to keep.
    pub max_history: usize,
    /// Number of frames of latency to compensate for when returning the
    /// "current" transform.
    pub latency_compensation_frames: u32,
}

impl CameraTracker {
    /// Create a new tracker.  `max_history` must be ≥ 1.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: std::collections::VecDeque::new(),
            max_history: max_history.max(1),
            latency_compensation_frames: 0,
        }
    }

    /// Record a new transform sample.  Evicts the oldest frame when at capacity.
    pub fn record(&mut self, transform: CameraTransform) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(transform);
    }

    /// Return a reference to the most recently recorded transform.
    #[must_use]
    pub fn latest(&self) -> Option<&CameraTransform> {
        self.history.back()
    }

    /// Predict the next frame transform by linear extrapolation from the last
    /// two frames.
    ///
    /// Returns `None` if fewer than two frames are stored.
    #[must_use]
    pub fn predict_next(&self) -> Option<CameraTransform> {
        let len = self.history.len();
        if len < 2 {
            return None;
        }
        let a = &self.history[len - 2];
        let b = &self.history[len - 1];
        // Extrapolate: b + (b - a) = 2b - a
        let lerp2 = |x: f32, y: f32| 2.0 * y - x;
        let ts_delta = b.timestamp_ns.saturating_sub(a.timestamp_ns);
        Some(CameraTransform {
            pan_deg: lerp2(a.pan_deg, b.pan_deg),
            tilt_deg: lerp2(a.tilt_deg, b.tilt_deg),
            roll_deg: lerp2(a.roll_deg, b.roll_deg),
            x_mm: lerp2(a.x_mm, b.x_mm),
            y_mm: lerp2(a.y_mm, b.y_mm),
            z_mm: lerp2(a.z_mm, b.z_mm),
            focal_length_mm: lerp2(a.focal_length_mm, b.focal_length_mm),
            focus_distance_mm: lerp2(a.focus_distance_mm, b.focus_distance_mm),
            timestamp_ns: b.timestamp_ns.saturating_add(ts_delta),
        })
    }

    /// Return the rate of change as a delta transform (latest − previous).
    ///
    /// Returns `None` if fewer than two frames are stored.
    #[must_use]
    pub fn velocity(&self) -> Option<CameraTransform> {
        let len = self.history.len();
        if len < 2 {
            return None;
        }
        let a = &self.history[len - 2];
        let b = &self.history[len - 1];
        let diff = |x: f32, y: f32| y - x;
        Some(CameraTransform {
            pan_deg: diff(a.pan_deg, b.pan_deg),
            tilt_deg: diff(a.tilt_deg, b.tilt_deg),
            roll_deg: diff(a.roll_deg, b.roll_deg),
            x_mm: diff(a.x_mm, b.x_mm),
            y_mm: diff(a.y_mm, b.y_mm),
            z_mm: diff(a.z_mm, b.z_mm),
            focal_length_mm: diff(a.focal_length_mm, b.focal_length_mm),
            focus_distance_mm: diff(a.focus_distance_mm, b.focus_distance_mm),
            timestamp_ns: b.timestamp_ns.saturating_sub(a.timestamp_ns),
        })
    }

    /// Return the moving average over the last `window` frames.
    ///
    /// Returns `None` if the history is empty or `window` is zero.
    #[must_use]
    pub fn smooth(&self, window: usize) -> Option<CameraTransform> {
        if window == 0 || self.history.is_empty() {
            return None;
        }
        let len = self.history.len();
        let start = len.saturating_sub(window);
        let frames: Vec<&CameraTransform> = self.history.range(start..).collect();
        let n = frames.len() as f32;
        if n == 0.0 {
            return None;
        }

        let sum_f32 = |f: &dyn Fn(&CameraTransform) -> f32| -> f32 {
            frames.iter().map(|t| f(t)).sum::<f32>() / n
        };
        let avg_ts: u64 =
            (frames.iter().map(|t| t.timestamp_ns as f64).sum::<f64>() / n as f64) as u64;

        Some(CameraTransform {
            pan_deg: sum_f32(&|t| t.pan_deg),
            tilt_deg: sum_f32(&|t| t.tilt_deg),
            roll_deg: sum_f32(&|t| t.roll_deg),
            x_mm: sum_f32(&|t| t.x_mm),
            y_mm: sum_f32(&|t| t.y_mm),
            z_mm: sum_f32(&|t| t.z_mm),
            focal_length_mm: sum_f32(&|t| t.focal_length_mm),
            focus_distance_mm: sum_f32(&|t| t.focus_distance_mm),
            timestamp_ns: avg_ts,
        })
    }

    /// Return the latest transform adjusted for `latency_compensation_frames`
    /// by linearly extrapolating that many frames into the future.
    ///
    /// Returns `None` if fewer than two frames are stored.
    #[must_use]
    pub fn compensated_transform(&self) -> Option<CameraTransform> {
        if self.latency_compensation_frames == 0 {
            return self.latest().cloned();
        }
        let len = self.history.len();
        if len < 2 {
            return None;
        }
        let a = &self.history[len - 2];
        let b = &self.history[len - 1];
        let t = self.latency_compensation_frames as f32;
        // Extrapolate: b + t * (b - a)
        let extrap = |x: f32, y: f32| y + t * (y - x);
        let ts_delta = b.timestamp_ns.saturating_sub(a.timestamp_ns);
        Some(CameraTransform {
            pan_deg: extrap(a.pan_deg, b.pan_deg),
            tilt_deg: extrap(a.tilt_deg, b.tilt_deg),
            roll_deg: extrap(a.roll_deg, b.roll_deg),
            x_mm: extrap(a.x_mm, b.x_mm),
            y_mm: extrap(a.y_mm, b.y_mm),
            z_mm: extrap(a.z_mm, b.z_mm),
            focal_length_mm: extrap(a.focal_length_mm, b.focal_length_mm),
            focus_distance_mm: extrap(a.focus_distance_mm, b.focus_distance_mm),
            timestamp_ns: b
                .timestamp_ns
                .saturating_add(ts_delta * self.latency_compensation_frames as u64),
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transform(x: f32, pan: f32, ts_ns: u64) -> CameraTransform {
        CameraTransform {
            x_mm: x,
            pan_deg: pan,
            timestamp_ns: ts_ns,
            ..CameraTransform::identity()
        }
    }

    #[test]
    fn test_camera_transform_identity() {
        let t = CameraTransform::identity();
        assert_eq!(t.pan_deg, 0.0);
        assert_eq!(t.x_mm, 0.0);
        assert_eq!(t.timestamp_ns, 0);
    }

    #[test]
    fn test_camera_transform_interpolate_midpoint() {
        let a = make_transform(0.0, 0.0, 0);
        let b = make_transform(100.0, 90.0, 1000);
        let mid = CameraTransform::interpolate(&a, &b, 0.5);
        assert!((mid.x_mm - 50.0).abs() < 1e-4);
        assert!((mid.pan_deg - 45.0).abs() < 1e-4);
        assert_eq!(mid.timestamp_ns, 500);
    }

    #[test]
    fn test_camera_transform_interpolate_t0() {
        let a = make_transform(10.0, 30.0, 100);
        let b = make_transform(200.0, 180.0, 200);
        let r = CameraTransform::interpolate(&a, &b, 0.0);
        assert!((r.x_mm - 10.0).abs() < 1e-4);
        assert!((r.pan_deg - 30.0).abs() < 1e-4);
    }

    #[test]
    fn test_free_d_encode_decode_roundtrip() {
        let t = CameraTransform {
            pan_deg: 10.0,
            tilt_deg: -5.0,
            roll_deg: 1.5,
            x_mm: 200.0,
            y_mm: -100.0,
            z_mm: 50.0,
            focal_length_mm: 35.0,
            focus_distance_mm: 2500.0,
            timestamp_ns: 0,
        };
        let pkt = t.to_free_d();
        let encoded = pkt.encode();
        let decoded = FreeDPacket::decode(&encoded);
        assert!(decoded.is_some());
        let decoded_pkt = decoded.expect("should succeed in test");
        assert_eq!(decoded_pkt.device_id, 1);
        // Check roundtrip to transform (tolerance due to integer quantisation).
        let back = decoded_pkt.to_transform();
        assert!((back.pan_deg - t.pan_deg).abs() < 0.01);
        assert!((back.x_mm - t.x_mm).abs() < 0.1);
    }

    #[test]
    fn test_free_d_decode_wrong_length() {
        let buf = [0xD1u8; 10];
        assert!(FreeDPacket::decode(&buf).is_none());
    }

    #[test]
    fn test_free_d_decode_wrong_type() {
        let mut buf = [0u8; 29];
        buf[0] = 0xAA; // not D1
        assert!(FreeDPacket::decode(&buf).is_none());
    }

    #[test]
    fn test_camera_tracker_empty() {
        let tracker = CameraTracker::new(10);
        assert!(tracker.latest().is_none());
        assert!(tracker.predict_next().is_none());
        assert!(tracker.velocity().is_none());
        assert!(tracker.smooth(3).is_none());
    }

    #[test]
    fn test_camera_tracker_record_evicts() {
        let mut tracker = CameraTracker::new(3);
        for i in 0..5u32 {
            tracker.record(make_transform(i as f32 * 10.0, 0.0, i as u64 * 33_000_000));
        }
        assert_eq!(tracker.history.len(), 3);
    }

    #[test]
    fn test_camera_tracker_predict_next() {
        let mut tracker = CameraTracker::new(10);
        tracker.record(make_transform(0.0, 0.0, 0));
        tracker.record(make_transform(10.0, 5.0, 1_000_000));
        let pred = tracker.predict_next().expect("should succeed in test");
        assert!((pred.x_mm - 20.0).abs() < 1e-4);
        assert!((pred.pan_deg - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_camera_tracker_velocity() {
        let mut tracker = CameraTracker::new(10);
        tracker.record(make_transform(0.0, 0.0, 0));
        tracker.record(make_transform(5.0, 2.0, 100_000));
        let vel = tracker.velocity().expect("should succeed in test");
        assert!((vel.x_mm - 5.0).abs() < 1e-4);
        assert!((vel.pan_deg - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_camera_tracker_smooth() {
        let mut tracker = CameraTracker::new(10);
        for i in 0..5u32 {
            tracker.record(make_transform(i as f32 * 10.0, 0.0, i as u64 * 1_000_000));
        }
        // Average of last 3: 20, 30, 40 → 30
        let smoothed = tracker.smooth(3).expect("should succeed in test");
        assert!((smoothed.x_mm - 30.0).abs() < 1e-3);
    }

    #[test]
    fn test_camera_tracker_compensated_transform_zero_latency() {
        let mut tracker = CameraTracker::new(10);
        tracker.record(make_transform(10.0, 5.0, 0));
        tracker.record(make_transform(20.0, 10.0, 1_000_000));
        let comp = tracker
            .compensated_transform()
            .expect("should succeed in test");
        // Zero latency → just the latest
        assert!((comp.x_mm - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_camera_tracker_compensated_transform_with_latency() {
        let mut tracker = CameraTracker::new(10);
        tracker.latency_compensation_frames = 2;
        tracker.record(make_transform(0.0, 0.0, 0));
        tracker.record(make_transform(10.0, 5.0, 1_000_000));
        // Extrapolate 2 frames: 10 + 2*(10-0) = 30
        let comp = tracker
            .compensated_transform()
            .expect("should succeed in test");
        assert!((comp.x_mm - 30.0).abs() < 1e-4);
    }
}
