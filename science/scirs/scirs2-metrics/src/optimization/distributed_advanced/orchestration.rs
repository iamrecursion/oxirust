//! Orchestration Module
//!
//! Provides orchestration capabilities for distributed optimization systems.

use crate::error::{MetricsError, Result};
use std::collections::HashMap;

/// Orchestration manager
#[derive(Debug, Clone)]
pub struct OrchestrationManager {
    node_id: String,
    services: HashMap<String, ServiceInfo>,
}

#[derive(Debug, Clone)]
pub struct ServiceInfo {
    pub name: String,
    pub status: ServiceStatus,
    pub endpoint: String,
}

#[derive(Debug, Clone)]
pub enum ServiceStatus {
    Running,
    Stopped,
    Failed,
}

impl OrchestrationManager {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            services: HashMap::new(),
        }
    }

    pub fn register_service(&mut self, name: String, endpoint: String) -> Result<()> {
        let service = ServiceInfo {
            name: name.clone(),
            status: ServiceStatus::Stopped,
            endpoint,
        };
        self.services.insert(name, service);
        Ok(())
    }

    pub fn start_service(&mut self, name: &str) -> Result<()> {
        if let Some(service) = self.services.get_mut(name) {
            service.status = ServiceStatus::Running;
            Ok(())
        } else {
            Err(MetricsError::InvalidOperation(format!(
                "Service {} not found",
                name
            )))
        }
    }

    pub fn stop_service(&mut self, name: &str) -> Result<()> {
        if let Some(service) = self.services.get_mut(name) {
            service.status = ServiceStatus::Stopped;
            Ok(())
        } else {
            Err(MetricsError::InvalidOperation(format!(
                "Service {} not found",
                name
            )))
        }
    }

    pub fn get_service_status(&self, name: &str) -> Option<&ServiceStatus> {
        self.services.get(name).map(|s| &s.status)
    }
}
