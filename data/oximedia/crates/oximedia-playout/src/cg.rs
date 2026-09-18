//! Character Generator (CG) integration
//!
//! Manages CG elements (lower-thirds, tickers, full-screen graphics) and
//! provides a command-based API for controlling their lifecycle.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single CG element that can be rendered over the video output
#[derive(Debug, Clone)]
pub struct CgElement {
    /// Unique identifier for this element
    pub id: u64,
    /// Human-readable name
    pub name: String,
    /// Template identifier used by the CG renderer
    pub template: String,
    /// Key/value data fields for populating the template
    pub data: HashMap<String, String>,
    /// Render layer (higher values appear on top)
    pub layer: u8,
    /// Whether the element is currently visible on output
    pub visible: bool,
}

impl CgElement {
    /// Create a new CG element with empty data
    pub fn new(id: u64, name: &str, template: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            template: template.to_string(),
            data: HashMap::new(),
            layer: 0,
            visible: false,
        }
    }

    /// Set a data field value
    pub fn set_field(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }

    /// Get a data field value
    pub fn get_field(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(String::as_str)
    }

    /// Make the element visible on output
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Remove the element from output
    pub fn hide(&mut self) {
        self.visible = false;
    }
}

/// Commands that can be sent to the CG controller
#[derive(Debug, Clone)]
pub enum CgCommand {
    /// Load an element into memory (does not make it visible yet)
    Load { element_id: u64 },
    /// Take an element to air (make it visible)
    Take { element_id: u64 },
    /// Take an element off air (hide it)
    Out { element_id: u64 },
    /// Update a single field on an element
    Update {
        element_id: u64,
        field: String,
        value: String,
    },
    /// Clear all elements from output
    Clear,
}

/// Controller for managing CG elements and executing commands
#[derive(Debug, Default)]
pub struct CgController {
    elements: Vec<CgElement>,
    command_log: Vec<CgCommand>,
}

impl CgController {
    /// Create a new empty CG controller
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an element to the controller's registry
    pub fn add_element(&mut self, el: CgElement) {
        self.elements.push(el);
    }

    /// Execute a CG command
    ///
    /// # Errors
    ///
    /// Returns an error if the target element does not exist.
    pub fn execute(&mut self, cmd: CgCommand) -> Result<(), String> {
        match &cmd {
            CgCommand::Load { element_id } => {
                self.find_element_mut(*element_id)
                    .ok_or_else(|| format!("Element {element_id} not found"))?;
                // Load is a no-op in this simulation; visibility unchanged
            }
            CgCommand::Take { element_id } => {
                self.find_element_mut(*element_id)
                    .ok_or_else(|| format!("Element {element_id} not found"))?
                    .show();
            }
            CgCommand::Out { element_id } => {
                self.find_element_mut(*element_id)
                    .ok_or_else(|| format!("Element {element_id} not found"))?
                    .hide();
            }
            CgCommand::Update {
                element_id,
                field,
                value,
            } => {
                let el = self
                    .find_element_mut(*element_id)
                    .ok_or_else(|| format!("Element {element_id} not found"))?;
                el.set_field(field, value);
            }
            CgCommand::Clear => {
                for el in &mut self.elements {
                    el.hide();
                }
            }
        }
        self.command_log.push(cmd);
        Ok(())
    }

    /// Return references to all currently visible elements
    pub fn visible_elements(&self) -> Vec<&CgElement> {
        self.elements.iter().filter(|e| e.visible).collect()
    }

    /// Total number of registered elements
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Number of commands executed so far
    pub fn command_count(&self) -> usize {
        self.command_log.len()
    }

    // ── private ──────────────────────────────────────────────────────────────

    fn find_element_mut(&mut self, id: u64) -> Option<&mut CgElement> {
        self.elements.iter_mut().find(|e| e.id == id)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_element(id: u64) -> CgElement {
        CgElement::new(id, &format!("el_{id}"), "lower_third")
    }

    #[test]
    fn test_element_new_hidden() {
        let el = make_element(1);
        assert!(!el.visible);
    }

    #[test]
    fn test_element_show_hide() {
        let mut el = make_element(1);
        el.show();
        assert!(el.visible);
        el.hide();
        assert!(!el.visible);
    }

    #[test]
    fn test_element_set_get_field() {
        let mut el = make_element(1);
        el.set_field("headline", "Breaking News");
        assert_eq!(el.get_field("headline"), Some("Breaking News"));
    }

    #[test]
    fn test_element_missing_field_is_none() {
        let el = make_element(1);
        assert!(el.get_field("nonexistent").is_none());
    }

    #[test]
    fn test_controller_add_and_count() {
        let mut ctrl = CgController::new();
        ctrl.add_element(make_element(1));
        ctrl.add_element(make_element(2));
        assert_eq!(ctrl.element_count(), 2);
    }

    #[test]
    fn test_execute_take_makes_visible() {
        let mut ctrl = CgController::new();
        ctrl.add_element(make_element(10));
        ctrl.execute(CgCommand::Take { element_id: 10 })
            .expect("should succeed in test");
        assert_eq!(ctrl.visible_elements().len(), 1);
    }

    #[test]
    fn test_execute_out_hides_element() {
        let mut ctrl = CgController::new();
        ctrl.add_element(make_element(10));
        ctrl.execute(CgCommand::Take { element_id: 10 })
            .expect("should succeed in test");
        ctrl.execute(CgCommand::Out { element_id: 10 })
            .expect("should succeed in test");
        assert_eq!(ctrl.visible_elements().len(), 0);
    }

    #[test]
    fn test_execute_update_changes_field() {
        let mut ctrl = CgController::new();
        ctrl.add_element(make_element(5));
        ctrl.execute(CgCommand::Update {
            element_id: 5,
            field: "name".to_string(),
            value: "Alice".to_string(),
        })
        .expect("should succeed in test");
        let el = ctrl
            .elements
            .iter()
            .find(|e| e.id == 5)
            .expect("should succeed in test");
        assert_eq!(el.get_field("name"), Some("Alice"));
    }

    #[test]
    fn test_execute_clear_hides_all() {
        let mut ctrl = CgController::new();
        ctrl.add_element(make_element(1));
        ctrl.add_element(make_element(2));
        ctrl.execute(CgCommand::Take { element_id: 1 })
            .expect("should succeed in test");
        ctrl.execute(CgCommand::Take { element_id: 2 })
            .expect("should succeed in test");
        ctrl.execute(CgCommand::Clear)
            .expect("should succeed in test");
        assert_eq!(ctrl.visible_elements().len(), 0);
    }

    #[test]
    fn test_execute_missing_element_returns_err() {
        let mut ctrl = CgController::new();
        let result = ctrl.execute(CgCommand::Take { element_id: 99 });
        assert!(result.is_err());
    }

    #[test]
    fn test_command_log_grows() {
        let mut ctrl = CgController::new();
        ctrl.add_element(make_element(1));
        ctrl.execute(CgCommand::Load { element_id: 1 })
            .expect("should succeed in test");
        ctrl.execute(CgCommand::Take { element_id: 1 })
            .expect("should succeed in test");
        assert_eq!(ctrl.command_count(), 2);
    }

    #[test]
    fn test_element_layer_default_zero() {
        let el = make_element(3);
        assert_eq!(el.layer, 0);
    }
}
