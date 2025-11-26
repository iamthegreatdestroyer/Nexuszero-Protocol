//! WebSocket Handlers
//!
//! Real-time communication for proof status updates and notifications

use crate::auth::{Claims, JwtManager};
use crate::state::AppState;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Extension, Query,
    },
    response::Response,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

/// WebSocket query parameters
#[derive(Debug, Deserialize)]
pub struct WsQuery {
    pub token: String,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsMessage {
    /// Authentication response
    #[serde(rename = "auth")]
    Auth { success: bool, user_id: Option<String> },

    /// Proof status update
    #[serde(rename = "proof_status")]
    ProofStatus {
        proof_id: String,
        status: String,
        progress: Option<u8>,
        error: Option<String>,
    },

    /// Transaction status update
    #[serde(rename = "transaction_status")]
    TransactionStatus {
        transaction_id: String,
        status: String,
        chain_tx_hash: Option<String>,
    },

    /// Bridge transfer update
    #[serde(rename = "bridge_status")]
    BridgeStatus {
        transfer_id: String,
        status: String,
        source_tx_hash: Option<String>,
        target_tx_hash: Option<String>,
    },

    /// Ping/Pong for keepalive
    #[serde(rename = "ping")]
    Ping { timestamp: i64 },

    #[serde(rename = "pong")]
    Pong { timestamp: i64 },

    /// Error message
    #[serde(rename = "error")]
    Error { code: String, message: String },

    /// Subscription request
    #[serde(rename = "subscribe")]
    Subscribe { channel: String, id: Option<String> },

    /// Subscription confirmation
    #[serde(rename = "subscribed")]
    Subscribed { channel: String, id: Option<String> },

    /// Unsubscription request
    #[serde(rename = "unsubscribe")]
    Unsubscribe { channel: String, id: Option<String> },
}

/// WebSocket handler for general real-time updates
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Extension(state): Extension<Arc<AppState>>,
    Query(params): Query<WsQuery>,
) -> Response {
    // Validate token before upgrading
    let jwt_manager = JwtManager::new(&state.config.jwt);
    let claims = jwt_manager.validate_access_token(&params.token);

    ws.on_upgrade(move |socket| handle_ws_connection(socket, state, claims.ok()))
}

/// WebSocket handler for proof status updates
pub async fn proof_status_ws(
    ws: WebSocketUpgrade,
    Extension(state): Extension<Arc<AppState>>,
    Query(params): Query<WsQuery>,
) -> Response {
    let jwt_manager = JwtManager::new(&state.config.jwt);
    let claims = jwt_manager.validate_access_token(&params.token);

    ws.on_upgrade(move |socket| handle_proof_status_connection(socket, state, claims.ok()))
}

/// Handle general WebSocket connection
async fn handle_ws_connection(socket: WebSocket, state: Arc<AppState>, claims: Option<Claims>) {
    let (mut sender, mut receiver) = socket.split();

    // Track connection
    state.increment_ws_connections();
    crate::handlers::metrics::ACTIVE_WEBSOCKET_CONNECTIONS
        .set(state.get_ws_connection_count() as f64);

    // Send auth result
    let auth_msg = if let Some(ref c) = claims {
        WsMessage::Auth {
            success: true,
            user_id: Some(c.sub.clone()),
        }
    } else {
        WsMessage::Auth {
            success: false,
            user_id: None,
        }
    };

    let _ = sender
        .send(Message::Text(serde_json::to_string(&auth_msg).unwrap()))
        .await;

    // If not authenticated, close connection
    if claims.is_none() {
        let _ = sender.close().await;
        state.decrement_ws_connections();
        return;
    }

    let user_id = claims.as_ref().map(|c| c.sub.clone()).unwrap_or_default();

    // Create channels for this user
    let (tx, mut rx) = broadcast::channel::<WsMessage>(100);

    // Spawn task to handle incoming messages
    let state_clone = state.clone();
    let tx_clone = tx.clone();
    let user_id_clone = user_id.clone();

    let receive_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        handle_incoming_message(&state_clone, &user_id_clone, ws_msg, &tx_clone)
                            .await;
                    }
                }
                Ok(Message::Ping(data)) => {
                    // Respond with pong
                    let _ = tx_clone.send(WsMessage::Pong {
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    });
                }
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {}
            }
        }
    });

    // Spawn task to send outgoing messages
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if let Ok(text) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(text)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = receive_task => {}
        _ = send_task => {}
    }

    // Clean up
    state.decrement_ws_connections();
    crate::handlers::metrics::ACTIVE_WEBSOCKET_CONNECTIONS
        .set(state.get_ws_connection_count() as f64);

    tracing::debug!(user_id = %user_id, "WebSocket connection closed");
}

/// Handle proof status WebSocket connection
async fn handle_proof_status_connection(
    socket: WebSocket,
    state: Arc<AppState>,
    claims: Option<Claims>,
) {
    let (mut sender, mut receiver) = socket.split();

    state.increment_ws_connections();

    // Auth check
    let auth_msg = if let Some(ref c) = claims {
        WsMessage::Auth {
            success: true,
            user_id: Some(c.sub.clone()),
        }
    } else {
        WsMessage::Auth {
            success: false,
            user_id: None,
        }
    };

    let _ = sender
        .send(Message::Text(serde_json::to_string(&auth_msg).unwrap()))
        .await;

    if claims.is_none() {
        let _ = sender.close().await;
        state.decrement_ws_connections();
        return;
    }

    let user_id = claims.as_ref().map(|c| c.sub.clone()).unwrap_or_default();

    // Subscribe to proof updates for this user
    let (tx, mut rx) = broadcast::channel::<WsMessage>(100);

    // Poll for proof updates
    let state_clone = state.clone();
    let user_id_clone = user_id.clone();
    let tx_clone = tx.clone();

    let poll_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));

        loop {
            interval.tick().await;

            // Check for proof updates from Redis
            if let Ok(updates) = get_proof_updates(&state_clone, &user_id_clone).await {
                for update in updates {
                    let _ = tx_clone.send(update);
                }
            }
        }
    });

    // Handle incoming messages
    let receive_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(WsMessage::Subscribe { channel, id }) =
                        serde_json::from_str::<WsMessage>(&text)
                    {
                        if channel == "proof" {
                            // Subscribe to specific proof updates
                            tracing::debug!(proof_id = ?id, "Subscribed to proof updates");
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(_) => break,
                _ => {}
            }
        }
    });

    // Send updates to client
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if let Ok(text) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(text)).await.is_err() {
                    break;
                }
            }
        }
    });

    tokio::select! {
        _ = receive_task => {}
        _ = send_task => {}
        _ = poll_task => {}
    }

    state.decrement_ws_connections();
    tracing::debug!(user_id = %user_id, "Proof status WebSocket closed");
}

/// Handle incoming WebSocket message
async fn handle_incoming_message(
    state: &AppState,
    user_id: &str,
    msg: WsMessage,
    tx: &broadcast::Sender<WsMessage>,
) {
    match msg {
        WsMessage::Ping { timestamp } => {
            let _ = tx.send(WsMessage::Pong { timestamp });
        }
        WsMessage::Subscribe { channel, id } => {
            tracing::debug!(
                user_id = %user_id,
                channel = %channel,
                id = ?id,
                "Subscription request"
            );

            // Store subscription in Redis
            if let Ok(mut conn) = state.redis_conn().await {
                let key = format!("ws:sub:{}:{}", user_id, channel);
                let _: Result<(), _> = redis::cmd("SADD")
                    .arg(&key)
                    .arg(id.clone().unwrap_or_default())
                    .query_async(&mut conn)
                    .await;
            }

            let _ = tx.send(WsMessage::Subscribed { channel, id });
        }
        WsMessage::Unsubscribe { channel, id } => {
            if let Ok(mut conn) = state.redis_conn().await {
                let key = format!("ws:sub:{}:{}", user_id, channel);
                let _: Result<(), _> = redis::cmd("SREM")
                    .arg(&key)
                    .arg(id.unwrap_or_default())
                    .query_async(&mut conn)
                    .await;
            }
        }
        _ => {}
    }
}

/// Get pending proof updates from Redis
async fn get_proof_updates(state: &AppState, user_id: &str) -> Result<Vec<WsMessage>, ()> {
    let mut conn = state.redis_conn().await.map_err(|_| ())?;

    // Check for proof updates queue
    let key = format!("ws:updates:proof:{}", user_id);
    let updates: Vec<String> = redis::cmd("LPOP")
        .arg(&key)
        .arg(10) // Get up to 10 updates at once
        .query_async(&mut conn)
        .await
        .unwrap_or_default();

    let mut messages = Vec::new();
    for update in updates {
        if let Ok(msg) = serde_json::from_str::<WsMessage>(&update) {
            messages.push(msg);
        }
    }

    Ok(messages)
}

/// Publish a proof status update (called from other services)
pub async fn publish_proof_update(
    state: &AppState,
    user_id: &str,
    proof_id: &str,
    status: &str,
    progress: Option<u8>,
    error: Option<String>,
) -> Result<(), ()> {
    let mut conn = state.redis_conn().await.map_err(|_| ())?;

    let msg = WsMessage::ProofStatus {
        proof_id: proof_id.to_string(),
        status: status.to_string(),
        progress,
        error,
    };

    let key = format!("ws:updates:proof:{}", user_id);
    let _: Result<(), _> = redis::cmd("RPUSH")
        .arg(&key)
        .arg(serde_json::to_string(&msg).unwrap())
        .query_async(&mut conn)
        .await;

    // Set TTL to clean up old updates
    let _: Result<(), _> = redis::cmd("EXPIRE")
        .arg(&key)
        .arg(3600) // 1 hour
        .query_async(&mut conn)
        .await;

    Ok(())
}

/// Publish a transaction status update
pub async fn publish_transaction_update(
    state: &AppState,
    user_id: &str,
    transaction_id: &str,
    status: &str,
    chain_tx_hash: Option<String>,
) -> Result<(), ()> {
    let mut conn = state.redis_conn().await.map_err(|_| ())?;

    let msg = WsMessage::TransactionStatus {
        transaction_id: transaction_id.to_string(),
        status: status.to_string(),
        chain_tx_hash,
    };

    let key = format!("ws:updates:tx:{}", user_id);
    let _: Result<(), _> = redis::cmd("RPUSH")
        .arg(&key)
        .arg(serde_json::to_string(&msg).unwrap())
        .query_async(&mut conn)
        .await;

    Ok(())
}

/// Publish a bridge transfer update
pub async fn publish_bridge_update(
    state: &AppState,
    user_id: &str,
    transfer_id: &str,
    status: &str,
    source_tx_hash: Option<String>,
    target_tx_hash: Option<String>,
) -> Result<(), ()> {
    let mut conn = state.redis_conn().await.map_err(|_| ())?;

    let msg = WsMessage::BridgeStatus {
        transfer_id: transfer_id.to_string(),
        status: status.to_string(),
        source_tx_hash,
        target_tx_hash,
    };

    let key = format!("ws:updates:bridge:{}", user_id);
    let _: Result<(), _> = redis::cmd("RPUSH")
        .arg(&key)
        .arg(serde_json::to_string(&msg).unwrap())
        .query_async(&mut conn)
        .await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_message_serialization() {
        let msg = WsMessage::ProofStatus {
            proof_id: "test-123".to_string(),
            status: "generating".to_string(),
            progress: Some(50),
            error: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("proof_status"));
        assert!(json.contains("test-123"));

        let parsed: WsMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            WsMessage::ProofStatus { proof_id, progress, .. } => {
                assert_eq!(proof_id, "test-123");
                assert_eq!(progress, Some(50));
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_auth_message() {
        let msg = WsMessage::Auth {
            success: true,
            user_id: Some("user-123".to_string()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("auth"));
        assert!(json.contains("user-123"));
    }
}
