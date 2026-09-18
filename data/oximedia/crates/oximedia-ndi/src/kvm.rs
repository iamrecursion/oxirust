//! KVM (Keyboard/Video/Mouse) metadata transport over the NDI metadata channel.
//!
//! NDI metadata frames carry arbitrary XML payloads.  This module serialises
//! keyboard, mouse, and pointer events into compact XML fragments that can be
//! embedded in an NDI stream, enabling thin-client / KVM-over-NDI workflows.
//!
//! # Wire format (examples)
//!
//! ```xml
//! <ndi_kvm type="key" action="press" keycode="65" modifiers="shift"/>
//! <ndi_kvm type="mouse_move" x="960" y="540"/>
//! <ndi_kvm type="mouse_btn" button="left" action="click"/>
//! <ndi_kvm type="mouse_scroll" delta_x="0" delta_y="-3"/>
//! <ndi_kvm type="clipboard" encoding="utf8">Hello world</ndi_kvm>
//! ```

#![allow(dead_code)]

use crate::{NdiError, Result};

// ---------------------------------------------------------------------------
// Keyboard events
// ---------------------------------------------------------------------------

/// Keyboard modifier flags (bit field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyModifiers(pub u8);

impl KeyModifiers {
    pub const NONE: Self = Self(0x00);
    pub const SHIFT: Self = Self(0x01);
    pub const CTRL: Self = Self(0x02);
    pub const ALT: Self = Self(0x04);
    pub const META: Self = Self(0x08);

    /// Returns `true` if the Shift modifier is active.
    pub fn shift(self) -> bool {
        (self.0 & 0x01) != 0
    }
    /// Returns `true` if the Ctrl modifier is active.
    pub fn ctrl(self) -> bool {
        (self.0 & 0x02) != 0
    }
    /// Returns `true` if the Alt modifier is active.
    pub fn alt(self) -> bool {
        (self.0 & 0x04) != 0
    }
    /// Returns `true` if the Meta/Super modifier is active.
    pub fn meta(self) -> bool {
        (self.0 & 0x08) != 0
    }

    /// Combine two modifier sets (bitwise OR).
    pub fn combine(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// A keyboard event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyEvent {
    /// X11 / Win32 virtual key code.
    pub keycode: u32,
    /// Whether this is a key-press (`true`) or key-release (`false`).
    pub pressed: bool,
    /// Active modifier keys.
    pub modifiers: KeyModifiers,
}

impl KeyEvent {
    /// Create a new key press event.
    pub fn press(keycode: u32, modifiers: KeyModifiers) -> Self {
        Self {
            keycode,
            pressed: true,
            modifiers,
        }
    }

    /// Create a new key release event.
    pub fn release(keycode: u32, modifiers: KeyModifiers) -> Self {
        Self {
            keycode,
            pressed: false,
            modifiers,
        }
    }
}

// ---------------------------------------------------------------------------
// Mouse events
// ---------------------------------------------------------------------------

/// Mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    Button4,
    Button5,
}

impl MouseButton {
    fn label(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Middle => "middle",
            Self::Right => "right",
            Self::Button4 => "btn4",
            Self::Button5 => "btn5",
        }
    }

    fn from_label(s: &str) -> Option<Self> {
        match s {
            "left" => Some(Self::Left),
            "middle" => Some(Self::Middle),
            "right" => Some(Self::Right),
            "btn4" => Some(Self::Button4),
            "btn5" => Some(Self::Button5),
            _ => None,
        }
    }
}

/// A mouse move event (absolute or relative coordinates).
#[derive(Debug, Clone, PartialEq)]
pub struct MouseMoveEvent {
    /// Horizontal position (pixels from left edge, or delta if relative).
    pub x: f32,
    /// Vertical position (pixels from top edge, or delta if relative).
    pub y: f32,
    /// `true` = absolute coordinates; `false` = relative/delta.
    pub absolute: bool,
}

/// A mouse button press/release event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MouseButtonEvent {
    pub button: MouseButton,
    /// `true` = pressed; `false` = released.
    pub pressed: bool,
}

/// A mouse scroll-wheel event.
#[derive(Debug, Clone, PartialEq)]
pub struct MouseScrollEvent {
    /// Horizontal scroll (positive = right).
    pub delta_x: f32,
    /// Vertical scroll (positive = down).
    pub delta_y: f32,
}

// ---------------------------------------------------------------------------
// KvmEvent enum
// ---------------------------------------------------------------------------

/// A KVM event that can be transported over NDI metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum KvmEvent {
    /// Keyboard key press or release.
    Key(KeyEvent),
    /// Mouse pointer movement.
    MouseMove(MouseMoveEvent),
    /// Mouse button press or release.
    MouseButton(MouseButtonEvent),
    /// Mouse scroll wheel.
    MouseScroll(MouseScrollEvent),
    /// Clipboard text content.
    Clipboard(String),
}

// ---------------------------------------------------------------------------
// Serialisation / deserialisation
// ---------------------------------------------------------------------------

/// Serialise a [`KvmEvent`] to an NDI metadata XML string.
pub fn encode_kvm_event(event: &KvmEvent) -> String {
    match event {
        KvmEvent::Key(k) => {
            let action = if k.pressed { "press" } else { "release" };
            let mods = modifier_string(k.modifiers);
            format!(
                "<ndi_kvm type=\"key\" action=\"{}\" keycode=\"{}\" modifiers=\"{}\"/>",
                action, k.keycode, mods
            )
        }
        KvmEvent::MouseMove(m) => {
            let coord_type = if m.absolute { "absolute" } else { "relative" };
            format!(
                "<ndi_kvm type=\"mouse_move\" coord_type=\"{}\" x=\"{:.3}\" y=\"{:.3}\"/>",
                coord_type, m.x, m.y
            )
        }
        KvmEvent::MouseButton(b) => {
            let action = if b.pressed { "press" } else { "release" };
            format!(
                "<ndi_kvm type=\"mouse_btn\" button=\"{}\" action=\"{}\"/>",
                b.button.label(),
                action
            )
        }
        KvmEvent::MouseScroll(s) => {
            format!(
                "<ndi_kvm type=\"mouse_scroll\" delta_x=\"{:.3}\" delta_y=\"{:.3}\"/>",
                s.delta_x, s.delta_y
            )
        }
        KvmEvent::Clipboard(text) => {
            format!(
                "<ndi_kvm type=\"clipboard\" encoding=\"utf8\">{}</ndi_kvm>",
                xml_escape(text)
            )
        }
    }
}

/// Deserialise a [`KvmEvent`] from an NDI metadata XML string.
///
/// Returns `Err` if the XML cannot be parsed or the event type is unknown.
pub fn decode_kvm_event(xml: &str) -> Result<KvmEvent> {
    let event_type = extract_attr(xml, "type")
        .ok_or_else(|| NdiError::Protocol("KVM: missing 'type' attribute".to_string()))?;

    match event_type.as_str() {
        "key" => {
            let action = extract_attr(xml, "action").unwrap_or_default();
            let keycode: u32 = extract_attr(xml, "keycode")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mods_str = extract_attr(xml, "modifiers").unwrap_or_default();
            let modifiers = parse_modifier_string(&mods_str);
            let pressed = action == "press";
            Ok(KvmEvent::Key(KeyEvent {
                keycode,
                pressed,
                modifiers,
            }))
        }
        "mouse_move" => {
            let x: f32 = extract_attr(xml, "x")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let y: f32 = extract_attr(xml, "y")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let coord_type = extract_attr(xml, "coord_type").unwrap_or_default();
            let absolute = coord_type != "relative";
            Ok(KvmEvent::MouseMove(MouseMoveEvent { x, y, absolute }))
        }
        "mouse_btn" => {
            let btn_str = extract_attr(xml, "button").unwrap_or_default();
            let button = MouseButton::from_label(&btn_str).ok_or_else(|| {
                NdiError::Protocol(format!("KVM: unknown mouse button '{btn_str}'"))
            })?;
            let action = extract_attr(xml, "action").unwrap_or_default();
            let pressed = action == "press";
            Ok(KvmEvent::MouseButton(MouseButtonEvent { button, pressed }))
        }
        "mouse_scroll" => {
            let dx: f32 = extract_attr(xml, "delta_x")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let dy: f32 = extract_attr(xml, "delta_y")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            Ok(KvmEvent::MouseScroll(MouseScrollEvent {
                delta_x: dx,
                delta_y: dy,
            }))
        }
        "clipboard" => {
            // Extract text between tags
            let text = extract_content(xml).unwrap_or_default();
            Ok(KvmEvent::Clipboard(xml_unescape(&text)))
        }
        other => Err(NdiError::Protocol(format!(
            "KVM: unknown event type '{other}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// KvmChannel — serialisation queue
// ---------------------------------------------------------------------------

/// A channel that queues KVM events for transmission over NDI metadata.
#[derive(Debug, Default)]
pub struct KvmChannel {
    /// Pending events (oldest first).
    pending: std::collections::VecDeque<KvmEvent>,
}

impl KvmChannel {
    /// Create an empty `KvmChannel`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a KVM event into the queue.
    pub fn push(&mut self, event: KvmEvent) {
        self.pending.push_back(event);
    }

    /// Pop the oldest event from the queue.
    pub fn pop(&mut self) -> Option<KvmEvent> {
        self.pending.pop_front()
    }

    /// Return the number of pending events.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Drain all pending events as serialised XML strings.
    pub fn drain_as_xml(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(ev) = self.pending.pop_front() {
            out.push(encode_kvm_event(&ev));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the value of an XML attribute `name="value"` from `xml`.
fn extract_attr(xml: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = xml.find(&needle)? + needle.len();
    let end = xml[start..].find('"')?;
    Some(xml[start..start + end].to_string())
}

/// Extract the text content between the outermost XML tags.
fn extract_content(xml: &str) -> Option<String> {
    let start = xml.find('>')? + 1;
    let end = xml.rfind('<')?;
    if end > start {
        Some(xml[start..end].to_string())
    } else {
        None
    }
}

/// Convert `KeyModifiers` to a comma-separated label string.
fn modifier_string(mods: KeyModifiers) -> String {
    let mut parts = Vec::new();
    if mods.shift() {
        parts.push("shift");
    }
    if mods.ctrl() {
        parts.push("ctrl");
    }
    if mods.alt() {
        parts.push("alt");
    }
    if mods.meta() {
        parts.push("meta");
    }
    parts.join(",")
}

/// Parse a comma-separated modifier string back to `KeyModifiers`.
fn parse_modifier_string(s: &str) -> KeyModifiers {
    let mut m = KeyModifiers::NONE;
    for part in s.split(',') {
        match part.trim() {
            "shift" => m = KeyModifiers(m.0 | KeyModifiers::SHIFT.0),
            "ctrl" => m = KeyModifiers(m.0 | KeyModifiers::CTRL.0),
            "alt" => m = KeyModifiers(m.0 | KeyModifiers::ALT.0),
            "meta" => m = KeyModifiers(m.0 | KeyModifiers::META.0),
            _ => {}
        }
    }
    m
}

/// Minimal XML character escaping.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Reverse of [`xml_escape`].
fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- KeyModifiers --

    #[test]
    fn test_key_modifiers_flags() {
        let m = KeyModifiers::SHIFT.combine(KeyModifiers::CTRL);
        assert!(m.shift());
        assert!(m.ctrl());
        assert!(!m.alt());
        assert!(!m.meta());
    }

    #[test]
    fn test_modifier_string_roundtrip() {
        let m = KeyModifiers::SHIFT.combine(KeyModifiers::ALT);
        let s = modifier_string(m);
        let back = parse_modifier_string(&s);
        assert_eq!(m, back);
    }

    // -- KeyEvent --

    #[test]
    fn test_encode_key_press() {
        let ev = KvmEvent::Key(KeyEvent::press(65, KeyModifiers::SHIFT));
        let xml = encode_kvm_event(&ev);
        assert!(xml.contains("type=\"key\""));
        assert!(xml.contains("action=\"press\""));
        assert!(xml.contains("keycode=\"65\""));
        assert!(xml.contains("shift"));
    }

    #[test]
    fn test_decode_key_press_roundtrip() {
        let ev = KvmEvent::Key(KeyEvent::press(65, KeyModifiers::CTRL));
        let xml = encode_kvm_event(&ev);
        let decoded = decode_kvm_event(&xml).expect("decode should succeed");
        assert_eq!(ev, decoded);
    }

    #[test]
    fn test_decode_key_release() {
        let ev = KvmEvent::Key(KeyEvent::release(27, KeyModifiers::NONE));
        let xml = encode_kvm_event(&ev);
        let decoded = decode_kvm_event(&xml).expect("decode should succeed");
        if let KvmEvent::Key(k) = decoded {
            assert!(!k.pressed);
            assert_eq!(k.keycode, 27);
        } else {
            panic!("expected Key event");
        }
    }

    // -- MouseMove --

    #[test]
    fn test_encode_mouse_move() {
        let ev = KvmEvent::MouseMove(MouseMoveEvent {
            x: 960.0,
            y: 540.0,
            absolute: true,
        });
        let xml = encode_kvm_event(&ev);
        assert!(xml.contains("type=\"mouse_move\""));
        assert!(xml.contains("960"));
        assert!(xml.contains("540"));
    }

    #[test]
    fn test_decode_mouse_move_roundtrip() {
        let ev = KvmEvent::MouseMove(MouseMoveEvent {
            x: 100.5,
            y: 200.75,
            absolute: false,
        });
        let xml = encode_kvm_event(&ev);
        let decoded = decode_kvm_event(&xml).expect("decode should succeed");
        if let KvmEvent::MouseMove(m) = decoded {
            assert!((m.x - 100.5).abs() < 0.01);
            assert!(!m.absolute);
        } else {
            panic!("expected MouseMove event");
        }
    }

    // -- MouseButton --

    #[test]
    fn test_encode_mouse_button() {
        let ev = KvmEvent::MouseButton(MouseButtonEvent {
            button: MouseButton::Right,
            pressed: true,
        });
        let xml = encode_kvm_event(&ev);
        assert!(xml.contains("button=\"right\""));
        assert!(xml.contains("action=\"press\""));
    }

    #[test]
    fn test_decode_mouse_button_roundtrip() {
        let ev = KvmEvent::MouseButton(MouseButtonEvent {
            button: MouseButton::Left,
            pressed: false,
        });
        let xml = encode_kvm_event(&ev);
        let decoded = decode_kvm_event(&xml).expect("decode should succeed");
        assert_eq!(ev, decoded);
    }

    // -- MouseScroll --

    #[test]
    fn test_encode_mouse_scroll() {
        let ev = KvmEvent::MouseScroll(MouseScrollEvent {
            delta_x: 0.0,
            delta_y: -3.0,
        });
        let xml = encode_kvm_event(&ev);
        assert!(xml.contains("mouse_scroll"));
        assert!(xml.contains("delta_y=\"-3.000\""));
    }

    #[test]
    fn test_decode_mouse_scroll_roundtrip() {
        let ev = KvmEvent::MouseScroll(MouseScrollEvent {
            delta_x: 1.5,
            delta_y: -2.5,
        });
        let xml = encode_kvm_event(&ev);
        let decoded = decode_kvm_event(&xml).expect("decode should succeed");
        if let KvmEvent::MouseScroll(s) = decoded {
            assert!((s.delta_x - 1.5).abs() < 0.01);
            assert!((s.delta_y - (-2.5)).abs() < 0.01);
        } else {
            panic!("expected MouseScroll event");
        }
    }

    // -- Clipboard --

    #[test]
    fn test_encode_clipboard() {
        let ev = KvmEvent::Clipboard("Hello <World>".to_string());
        let xml = encode_kvm_event(&ev);
        assert!(xml.contains("clipboard"));
        assert!(xml.contains("&lt;World&gt;"));
    }

    #[test]
    fn test_decode_clipboard_roundtrip() {
        let ev = KvmEvent::Clipboard("Test & \"paste\"".to_string());
        let xml = encode_kvm_event(&ev);
        let decoded = decode_kvm_event(&xml).expect("decode should succeed");
        assert_eq!(ev, decoded);
    }

    // -- Unknown type --

    #[test]
    fn test_decode_unknown_type_error() {
        let xml = r#"<ndi_kvm type="touchpad" x="0" y="0"/>"#;
        assert!(decode_kvm_event(xml).is_err());
    }

    // -- KvmChannel --

    #[test]
    fn test_kvm_channel_push_pop() {
        let mut ch = KvmChannel::new();
        ch.push(KvmEvent::Key(KeyEvent::press(10, KeyModifiers::NONE)));
        ch.push(KvmEvent::Key(KeyEvent::press(11, KeyModifiers::NONE)));
        assert_eq!(ch.len(), 2);
        let _ = ch.pop();
        assert_eq!(ch.len(), 1);
    }

    #[test]
    fn test_kvm_channel_drain_as_xml() {
        let mut ch = KvmChannel::new();
        ch.push(KvmEvent::MouseScroll(MouseScrollEvent {
            delta_x: 0.0,
            delta_y: 1.0,
        }));
        ch.push(KvmEvent::Key(KeyEvent::press(32, KeyModifiers::NONE)));
        let xmls = ch.drain_as_xml();
        assert_eq!(xmls.len(), 2);
        assert!(ch.is_empty());
    }

    // -- xml escape --

    #[test]
    fn test_xml_escape_roundtrip() {
        let s = "a<b>&\"c";
        let escaped = xml_escape(s);
        let back = xml_unescape(&escaped);
        assert_eq!(back, s);
    }
}
