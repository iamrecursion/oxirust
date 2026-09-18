//! Frame-accurate timecode interpolation conformance tests.
//!
//! Verifies that `TcInterpolator` derives exact intermediate timecodes from
//! sparse reference points across the supported interpolation modes.

use oximedia_timecode::tc_interpolate::{InterpolationMode, TcInterpolator, TcRefPoint};
use oximedia_timecode::{FrameRate, Timecode, TimecodeError};

/// Builds a 25 fps timecode, surfacing any construction error to the caller.
fn tc(h: u8, m: u8, s: u8, f: u8) -> Result<Timecode, TimecodeError> {
    Timecode::new(h, m, s, f, FrameRate::Fps25)
}

/// Linear interpolation between two reference points spanning 50 frames
/// (`00:00:00:00` @ pos 0 and `00:00:02:00` @ pos 50) must count frames
/// exactly: position `p` maps to `p` frames from the start.
#[test]
fn linear_interpolation_exact_frames() -> Result<(), TimecodeError> {
    let mut it = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
    it.add_ref(TcRefPoint::new(tc(0, 0, 0, 0)?, 0));
    it.add_ref(TcRefPoint::new(tc(0, 0, 2, 0)?, 50));

    let p10 = it
        .interpolate(10)
        .expect("position in range")
        .expect("valid timecode");
    assert_eq!((p10.seconds, p10.frames), (0, 10));

    let p25 = it
        .interpolate(25)
        .expect("position in range")
        .expect("valid timecode");
    assert_eq!((p25.seconds, p25.frames), (1, 0));

    let p37 = it
        .interpolate(37)
        .expect("position in range")
        .expect("valid timecode");
    assert_eq!((p37.seconds, p37.frames), (1, 12));

    Ok(())
}

/// Querying a position that lands exactly on a reference point must return that
/// reference's timecode unchanged.
#[test]
fn exact_ref_position_returns_ref() -> Result<(), TimecodeError> {
    let mut it = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear);
    it.add_ref(TcRefPoint::new(tc(0, 0, 0, 0)?, 0));
    it.add_ref(TcRefPoint::new(tc(0, 0, 2, 0)?, 50));

    let got = it
        .interpolate(50)
        .expect("position in range")
        .expect("valid timecode");
    assert_eq!(got, tc(0, 0, 2, 0)?);

    Ok(())
}

/// With a single reference at position 0 and a max-gap of 10 frames, a query at
/// position 20 exceeds the gap and must return `None`.
#[test]
fn max_gap_exceeded_returns_none() -> Result<(), TimecodeError> {
    let mut it = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Linear).with_max_gap(10);
    it.add_ref(TcRefPoint::new(tc(0, 0, 0, 0)?, 0));
    assert!(it.interpolate(20).is_none());

    Ok(())
}

/// Nearest-neighbour mode snaps a query to the closest reference point:
/// position 20 is closer to ref@0, position 30 is closer to ref@50.
#[test]
fn nearest_mode_snaps() -> Result<(), TimecodeError> {
    let mut it = TcInterpolator::new(FrameRate::Fps25, InterpolationMode::Nearest);
    let ref0 = tc(0, 0, 0, 0)?;
    let ref50 = tc(0, 0, 2, 0)?;
    it.add_ref(TcRefPoint::new(ref0, 0));
    it.add_ref(TcRefPoint::new(ref50, 50));

    let near0 = it
        .interpolate(20)
        .expect("position in range")
        .expect("valid timecode");
    assert_eq!(near0, ref0);

    let near50 = it
        .interpolate(30)
        .expect("position in range")
        .expect("valid timecode");
    assert_eq!(near50, ref50);

    Ok(())
}
