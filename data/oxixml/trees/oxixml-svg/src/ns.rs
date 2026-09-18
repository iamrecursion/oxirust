//! Namespace constants used by SVG documents.

/// The SVG element namespace, `http://www.w3.org/2000/svg`.
pub const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

/// The XLink namespace, `http://www.w3.org/1999/xlink`, used by the legacy
/// `xlink:href` attribute that SVG 2 replaces with a plain `href`.
pub const XLINK_NAMESPACE: &str = "http://www.w3.org/1999/xlink";

pub use oxixml_dom::{XML_NAMESPACE, XMLNS_NAMESPACE};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_constants_match_the_specifications() {
        assert_eq!(SVG_NAMESPACE, "http://www.w3.org/2000/svg");
        assert_eq!(XLINK_NAMESPACE, "http://www.w3.org/1999/xlink");
        assert_eq!(XML_NAMESPACE, "http://www.w3.org/XML/1998/namespace");
        assert_eq!(XMLNS_NAMESPACE, "http://www.w3.org/2000/xmlns/");
    }
}
