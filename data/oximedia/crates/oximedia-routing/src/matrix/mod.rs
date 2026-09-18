//! Matrix routing module for crosspoint routing and connection management.

pub mod connection;
pub mod crosspoint;
pub mod mix_minus;
pub mod solver;

pub use connection::{
    Connection, ConnectionId, ConnectionManager, ConnectionType, RoutingPriority,
};
pub use crosspoint::{CrosspointId, CrosspointMatrix, CrosspointState, MatrixError};
pub use mix_minus::{MixMinusBus, MixMinusRouter};
pub use solver::{RoutingPath, RoutingPathSolver};
