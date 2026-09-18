//! Automation API exposed to Lua scripts.

use crate::{AutomationError, Result};
use std::collections::HashMap;
use tracing::{debug, info};

/// Automation API for Lua scripts.
pub struct AutomationApi {
    functions: HashMap<String, String>,
}

impl AutomationApi {
    /// Create a new automation API.
    pub fn new() -> Self {
        info!("Creating automation API");

        let mut api = Self {
            functions: HashMap::new(),
        };

        api.register_functions();
        api
    }

    /// Register API functions.
    fn register_functions(&mut self) {
        // Register all available API functions
        self.functions
            .insert("log".to_string(), "Log a message".to_string());
        self.functions
            .insert("play".to_string(), "Start playback".to_string());
        self.functions
            .insert("stop".to_string(), "Stop playback".to_string());
        self.functions
            .insert("cut".to_string(), "Perform switcher cut".to_string());
        self.functions.insert(
            "trigger_failover".to_string(),
            "Trigger failover".to_string(),
        );
        self.functions
            .insert("send_alert".to_string(), "Send EAS alert".to_string());
    }

    /// Get function documentation.
    pub fn get_function_docs(&self, name: &str) -> Option<&String> {
        self.functions.get(name)
    }

    /// List all available functions.
    pub fn list_functions(&self) -> Vec<&String> {
        self.functions.keys().collect()
    }

    /// Execute API function.
    pub fn execute(&self, name: &str, args: Vec<String>) -> Result<String> {
        debug!("Executing API function: {} with {} args", name, args.len());

        match name {
            "log" => self.api_log(&args),
            "play" => self.api_play(&args),
            "stop" => self.api_stop(&args),
            "cut" => self.api_cut(&args),
            "trigger_failover" => self.api_trigger_failover(&args),
            "send_alert" => self.api_send_alert(&args),
            _ => Err(AutomationError::Scripting(format!(
                "Unknown function: {name}"
            ))),
        }
    }

    /// API: Log a message.
    fn api_log(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Err(AutomationError::Scripting(
                "log() requires a message".to_string(),
            ));
        }

        info!("Script: {}", args[0]);
        Ok("OK".to_string())
    }

    /// API: Start playback.
    fn api_play(&self, _args: &[String]) -> Result<String> {
        info!("API: Starting playback");
        Ok("OK".to_string())
    }

    /// API: Stop playback.
    fn api_stop(&self, _args: &[String]) -> Result<String> {
        info!("API: Stopping playback");
        Ok("OK".to_string())
    }

    /// API: Perform switcher cut.
    fn api_cut(&self, _args: &[String]) -> Result<String> {
        info!("API: Performing cut");
        Ok("OK".to_string())
    }

    /// API: Trigger failover.
    fn api_trigger_failover(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Err(AutomationError::Scripting(
                "trigger_failover() requires channel_id".to_string(),
            ));
        }

        info!("API: Triggering failover for channel: {}", args[0]);
        Ok("OK".to_string())
    }

    /// API: Send EAS alert.
    fn api_send_alert(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Err(AutomationError::Scripting(
                "send_alert() requires message".to_string(),
            ));
        }

        info!("API: Sending EAS alert: {}", args[0]);
        Ok("OK".to_string())
    }
}

impl Default for AutomationApi {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automation_api() {
        let api = AutomationApi::new();
        assert!(!api.list_functions().is_empty());
    }

    #[test]
    fn test_execute_log() {
        let api = AutomationApi::new();
        let result = api.execute("log", vec!["test message".to_string()]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_unknown_function() {
        let api = AutomationApi::new();
        let result = api.execute("unknown", vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_function_docs() {
        let api = AutomationApi::new();
        let docs = api.get_function_docs("log");
        assert!(docs.is_some());
    }
}
