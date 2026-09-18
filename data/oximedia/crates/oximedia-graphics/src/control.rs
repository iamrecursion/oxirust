//! Control interface (REST API and WebSocket)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Graphics control command
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command")]
pub enum ControlCommand {
    /// Show graphic
    Show {
        /// Graphic ID
        id: String,
        /// Data for template variables
        data: HashMap<String, String>,
    },
    /// Hide graphic
    Hide {
        /// Graphic ID
        id: String,
    },
    /// Update graphic data
    Update {
        /// Graphic ID
        id: String,
        /// New data
        data: HashMap<String, String>,
    },
    /// Trigger animation
    Animate {
        /// Graphic ID
        id: String,
        /// Animation name
        animation: String,
    },
    /// Clear all graphics
    ClearAll,
}

/// Control command response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResponse {
    /// Success
    pub success: bool,
    /// Message
    pub message: Option<String>,
    /// Error
    pub error: Option<String>,
}

impl CommandResponse {
    /// Create success response
    #[must_use]
    pub fn success(message: Option<String>) -> Self {
        Self {
            success: true,
            message,
            error: None,
        }
    }

    /// Create error response
    #[must_use]
    pub fn error(error: String) -> Self {
        Self {
            success: false,
            message: None,
            error: Some(error),
        }
    }
}

/// Graphic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphicState {
    /// Graphic ID
    pub id: String,
    /// Template name
    pub template: String,
    /// Data
    pub data: HashMap<String, String>,
    /// Visible
    pub visible: bool,
}

impl GraphicState {
    /// Create a new graphic state
    #[must_use]
    pub fn new(id: String, template: String) -> Self {
        Self {
            id,
            template,
            data: HashMap::new(),
            visible: false,
        }
    }
}

/// Graphics controller
pub struct GraphicsController {
    graphics: Arc<RwLock<HashMap<String, GraphicState>>>,
}

impl GraphicsController {
    /// Create a new graphics controller
    #[must_use]
    pub fn new() -> Self {
        Self {
            graphics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Execute command
    pub async fn execute(&self, command: ControlCommand) -> CommandResponse {
        match command {
            ControlCommand::Show { id, data } => {
                let mut graphics = self.graphics.write().await;
                if let Some(graphic) = graphics.get_mut(&id) {
                    graphic.data = data;
                    graphic.visible = true;
                    CommandResponse::success(Some(format!("Graphic {id} shown")))
                } else {
                    CommandResponse::error(format!("Graphic {id} not found"))
                }
            }
            ControlCommand::Hide { id } => {
                let mut graphics = self.graphics.write().await;
                if let Some(graphic) = graphics.get_mut(&id) {
                    graphic.visible = false;
                    CommandResponse::success(Some(format!("Graphic {id} hidden")))
                } else {
                    CommandResponse::error(format!("Graphic {id} not found"))
                }
            }
            ControlCommand::Update { id, data } => {
                let mut graphics = self.graphics.write().await;
                if let Some(graphic) = graphics.get_mut(&id) {
                    graphic.data = data;
                    CommandResponse::success(Some(format!("Graphic {id} updated")))
                } else {
                    CommandResponse::error(format!("Graphic {id} not found"))
                }
            }
            ControlCommand::Animate { id, animation } => {
                let graphics = self.graphics.read().await;
                if graphics.contains_key(&id) {
                    CommandResponse::success(Some(format!(
                        "Animation {animation} triggered for graphic {id}"
                    )))
                } else {
                    CommandResponse::error(format!("Graphic {id} not found"))
                }
            }
            ControlCommand::ClearAll => {
                let mut graphics = self.graphics.write().await;
                for graphic in graphics.values_mut() {
                    graphic.visible = false;
                }
                CommandResponse::success(Some("All graphics cleared".to_string()))
            }
        }
    }

    /// Add graphic
    pub async fn add_graphic(&self, id: String, template: String) {
        let mut graphics = self.graphics.write().await;
        graphics.insert(id.clone(), GraphicState::new(id, template));
    }

    /// Get graphic state
    pub async fn get_graphic(&self, id: &str) -> Option<GraphicState> {
        let graphics = self.graphics.read().await;
        graphics.get(id).cloned()
    }

    /// List all graphics
    pub async fn list_graphics(&self) -> Vec<GraphicState> {
        let graphics = self.graphics.read().await;
        graphics.values().cloned().collect()
    }
}

impl Default for GraphicsController {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro command (sequence of commands)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Macro {
    /// Macro name
    pub name: String,
    /// Commands
    pub commands: Vec<ControlCommand>,
}

impl Macro {
    /// Create a new macro
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            commands: Vec::new(),
        }
    }

    /// Add command
    pub fn add_command(&mut self, command: ControlCommand) {
        self.commands.push(command);
    }

    /// Execute macro
    pub async fn execute(&self, controller: &GraphicsController) -> Vec<CommandResponse> {
        let mut responses = Vec::new();
        for command in &self.commands {
            let response = controller.execute(command.clone()).await;
            responses.push(response);
        }
        responses
    }
}

/// Playout schedule item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleItem {
    /// Item ID
    pub id: String,
    /// Start time (milliseconds from start)
    pub start_time: u64,
    /// Duration (milliseconds)
    pub duration: Option<u64>,
    /// Command to execute
    pub command: ControlCommand,
}

impl ScheduleItem {
    /// Create a new schedule item
    #[must_use]
    pub fn new(id: String, start_time: u64, command: ControlCommand) -> Self {
        Self {
            id,
            start_time,
            duration: None,
            command,
        }
    }

    /// Set duration
    #[must_use]
    pub fn with_duration(mut self, duration: u64) -> Self {
        self.duration = Some(duration);
        self
    }
}

/// Playout scheduler
pub struct Scheduler {
    items: Vec<ScheduleItem>,
    current_time: u64,
}

impl Scheduler {
    /// Create a new scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            current_time: 0,
        }
    }

    /// Add item to schedule
    pub fn add_item(&mut self, item: ScheduleItem) {
        self.items.push(item);
        self.items.sort_by_key(|i| i.start_time);
    }

    /// Update time and get commands to execute
    pub fn update(&mut self, time: u64) -> Vec<ControlCommand> {
        self.current_time = time;

        let mut commands = Vec::new();
        for item in &self.items {
            if item.start_time <= time {
                if let Some(duration) = item.duration {
                    if item.start_time + duration > time {
                        commands.push(item.command.clone());
                    }
                } else {
                    commands.push(item.command.clone());
                }
            }
        }

        commands
    }

    /// Clear schedule
    pub fn clear(&mut self) {
        self.items.clear();
        self.current_time = 0;
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graphics_controller() {
        let controller = GraphicsController::new();
        controller
            .add_graphic("lower_third".to_string(), "lt_template".to_string())
            .await;

        let graphic = controller.get_graphic("lower_third").await;
        assert!(graphic.is_some());
    }

    #[tokio::test]
    async fn test_show_graphic() {
        let controller = GraphicsController::new();
        controller
            .add_graphic("test".to_string(), "template".to_string())
            .await;

        let mut data = HashMap::new();
        data.insert("name".to_string(), "John".to_string());

        let response = controller
            .execute(ControlCommand::Show {
                id: "test".to_string(),
                data,
            })
            .await;

        assert!(response.success);
    }

    #[tokio::test]
    async fn test_hide_graphic() {
        let controller = GraphicsController::new();
        controller
            .add_graphic("test".to_string(), "template".to_string())
            .await;

        let response = controller
            .execute(ControlCommand::Hide {
                id: "test".to_string(),
            })
            .await;

        assert!(response.success);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let controller = GraphicsController::new();
        controller
            .add_graphic("test1".to_string(), "template".to_string())
            .await;
        controller
            .add_graphic("test2".to_string(), "template".to_string())
            .await;

        let response = controller.execute(ControlCommand::ClearAll).await;
        assert!(response.success);
    }

    #[tokio::test]
    async fn test_macro() {
        let controller = GraphicsController::new();
        controller
            .add_graphic("test".to_string(), "template".to_string())
            .await;

        let mut macro_cmd = Macro::new("show_lower_third".to_string());
        macro_cmd.add_command(ControlCommand::Show {
            id: "test".to_string(),
            data: HashMap::new(),
        });

        let responses = macro_cmd.execute(&controller).await;
        assert_eq!(responses.len(), 1);
        assert!(responses[0].success);
    }

    #[test]
    fn test_scheduler() {
        let mut scheduler = Scheduler::new();

        let item = ScheduleItem::new(
            "item1".to_string(),
            1000,
            ControlCommand::Show {
                id: "test".to_string(),
                data: HashMap::new(),
            },
        );

        scheduler.add_item(item);

        let commands = scheduler.update(500);
        assert_eq!(commands.len(), 0);

        let commands = scheduler.update(1500);
        assert_eq!(commands.len(), 1);
    }

    #[test]
    fn test_schedule_item_with_duration() {
        let item = ScheduleItem::new(
            "item1".to_string(),
            1000,
            ControlCommand::Show {
                id: "test".to_string(),
                data: HashMap::new(),
            },
        )
        .with_duration(5000);

        assert_eq!(item.duration, Some(5000));
    }

    #[test]
    fn test_command_response() {
        let success = CommandResponse::success(Some("OK".to_string()));
        assert!(success.success);
        assert_eq!(success.message, Some("OK".to_string()));

        let error = CommandResponse::error("Error".to_string());
        assert!(!error.success);
        assert_eq!(error.error, Some("Error".to_string()));
    }
}
