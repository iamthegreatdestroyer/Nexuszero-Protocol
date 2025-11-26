//! NexusZero Database - Database abstractions and utilities
//!
//! This crate provides common database patterns for PostgreSQL and Redis.

pub mod postgres;
pub mod redis;
pub mod error;
pub mod pool;

pub use error::DbError;
pub use postgres::*;
pub use pool::*;
