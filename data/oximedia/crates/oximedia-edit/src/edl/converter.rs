//! EDL format conversion utilities.

use super::{Edl, EdlFormat, EdlResult};

/// Convert an EDL from one format to another.
pub fn convert(edl: &Edl, target: EdlFormat) -> EdlResult<String> {
    edl.to_format(target)
}
