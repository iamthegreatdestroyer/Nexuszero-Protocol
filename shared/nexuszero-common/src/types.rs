//! Common types used across NexusZero services

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Standard API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<ErrorInfo>,
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            request_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
        }
    }

    pub fn error(code: &str, message: &str) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(ErrorInfo {
                code: code.to_string(),
                message: message.to_string(),
                details: None,
            }),
            request_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
        }
    }
}

/// Error information for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub page: u32,
    pub limit: u32,
    pub total: u64,
    pub total_pages: u32,
}

impl Pagination {
    pub fn new(page: u32, limit: u32, total: u64) -> Self {
        let total_pages = ((total as f64) / (limit as f64)).ceil() as u32;
        Self {
            page,
            limit,
            total,
            total_pages,
        }
    }
}

/// Paginated response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub pagination: Pagination,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub version: String,
    pub uptime_seconds: u64,
    pub dependencies: Vec<DependencyHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyHealth {
    pub name: String,
    pub status: ServiceStatus,
    pub latency_ms: Option<u64>,
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionStatus {
    Pending,
    Validating,
    Processing,
    Confirmed,
    Failed,
    Cancelled,
}

/// Network type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkType {
    Mainnet,
    Testnet,
    Devnet,
}

/// Chain identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainId {
    pub network: NetworkType,
    pub chain: String,
    pub chain_id: u64,
}

impl ChainId {
    pub fn ethereum_mainnet() -> Self {
        Self {
            network: NetworkType::Mainnet,
            chain: "ethereum".to_string(),
            chain_id: 1,
        }
    }

    pub fn ethereum_sepolia() -> Self {
        Self {
            network: NetworkType::Testnet,
            chain: "ethereum".to_string(),
            chain_id: 11155111,
        }
    }
}

/// Address wrapper for multi-chain support
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address {
    pub chain_id: ChainId,
    pub address: String,
}

impl Address {
    pub fn new(chain_id: ChainId, address: String) -> Self {
        Self { chain_id, address }
    }
}

/// Amount with precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Amount {
    pub value: String,
    pub decimals: u8,
    pub symbol: Option<String>,
}

impl Amount {
    pub fn new(value: &str, decimals: u8) -> Self {
        Self {
            value: value.to_string(),
            decimals,
            symbol: None,
        }
    }

    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_response_success() {
        let response: ApiResponse<String> = ApiResponse::success("test".to_string());
        assert!(response.success);
        assert_eq!(response.data.unwrap(), "test");
    }

    #[test]
    fn test_pagination() {
        let pagination = Pagination::new(1, 10, 95);
        assert_eq!(pagination.total_pages, 10);
    }
}
