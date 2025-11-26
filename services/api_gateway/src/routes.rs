//! Route definitions for the API Gateway
//!
//! This module re-exports all route modules for organization

pub mod v1 {
    //! Version 1 API routes

    /// Transaction-related routes
    pub mod transactions {
        use axum::{routing::get, Router};

        pub fn routes() -> Router {
            Router::new()
                .route("/", get(super::super::super::handlers::transaction::list_transactions))
        }
    }

    /// Privacy-related routes
    pub mod privacy {
        use axum::{routing::get, Router};

        pub fn routes() -> Router {
            Router::new()
                .route("/levels", get(super::super::super::handlers::privacy::list_privacy_levels))
        }
    }

    /// Compliance-related routes
    pub mod compliance {
        use axum::{routing::post, Router};

        pub fn routes() -> Router {
            Router::new()
                .route("/verify", post(super::super::super::handlers::compliance::verify_compliance))
        }
    }

    /// Bridge-related routes
    pub mod bridge {
        use axum::{routing::get, Router};

        pub fn routes() -> Router {
            Router::new()
                .route("/supported-chains", get(super::super::super::handlers::bridge::get_supported_chains))
        }
    }

    /// Proof-related routes
    pub mod proofs {
        use axum::{routing::post, Router};

        pub fn routes() -> Router {
            Router::new()
                .route("/generate", post(super::super::super::handlers::proof::generate_proof))
        }
    }
}
