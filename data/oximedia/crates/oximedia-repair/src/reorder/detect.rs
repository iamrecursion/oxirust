//! Frame order detection.
//!
//! This module provides functions to detect frames that are out of order.

/// Frame with metadata.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Frame number/sequence.
    pub sequence: u32,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Frame data.
    pub data: Vec<u8>,
}

/// Detect out-of-order frames.
pub fn detect_frame_order_issues(frames: &[Frame]) -> Vec<usize> {
    let mut issues = Vec::new();

    for i in 1..frames.len() {
        // Check if PTS is out of order
        if frames[i].pts < frames[i - 1].pts {
            issues.push(i);
        }
    }

    issues
}

/// Detect frames with DTS/PTS mismatch.
pub fn detect_dts_pts_mismatch(frames: &[Frame]) -> Vec<usize> {
    let mut issues = Vec::new();

    for (i, frame) in frames.iter().enumerate() {
        // DTS should be <= PTS
        if frame.dts > frame.pts {
            issues.push(i);
        }
    }

    issues
}

/// Check if frames are in presentation order.
pub fn is_presentation_order(frames: &[Frame]) -> bool {
    frames.windows(2).all(|w| w[0].pts <= w[1].pts)
}

/// Check if frames are in decode order.
pub fn is_decode_order(frames: &[Frame]) -> bool {
    frames.windows(2).all(|w| w[0].dts <= w[1].dts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_frame_order_issues() {
        let frames = vec![
            Frame {
                sequence: 0,
                pts: 0,
                dts: 0,
                data: vec![],
            },
            Frame {
                sequence: 1,
                pts: 200,
                dts: 100,
                data: vec![],
            },
            Frame {
                sequence: 2,
                pts: 100,
                dts: 200,
                data: vec![],
            },
        ];

        let issues = detect_frame_order_issues(&frames);
        assert_eq!(issues, vec![2]);
    }

    #[test]
    fn test_is_presentation_order() {
        let frames = vec![
            Frame {
                sequence: 0,
                pts: 0,
                dts: 0,
                data: vec![],
            },
            Frame {
                sequence: 1,
                pts: 100,
                dts: 100,
                data: vec![],
            },
        ];

        assert!(is_presentation_order(&frames));
    }
}
