//! Privacy Morphing Engine Core

use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::anonymity::AnonymitySetManager;
use crate::compliance::ComplianceIntegration;
use crate::config::MorphingConfig;
use crate::error::{MorphingError, MorphingResult};
use crate::schedule::ScheduleExecutor;
use crate::types::*;

/// The main Privacy Morphing Engine
pub struct PrivacyMorphingEngine {
    config: MorphingConfig,
    anonymity_manager: Arc<AnonymitySetManager>,
    compliance: Arc<ComplianceIntegration>,
    schedule_executor: Arc<RwLock<ScheduleExecutor>>,
}

impl PrivacyMorphingEngine {
    /// Create a new Privacy Morphing Engine
    pub fn new(config: MorphingConfig) -> Self {
        let anonymity_manager = Arc::new(AnonymitySetManager::new(
            config.min_anonymity_set_size,
            config.performance.anonymity_cache_size,
        ));

        let compliance = Arc::new(ComplianceIntegration::new(
            config.compliance.clone(),
        ));

        let schedule_executor = Arc::new(RwLock::new(ScheduleExecutor::new()));

        Self {
            config,
            anonymity_manager,
            compliance,
            schedule_executor,
        }
    }

    /// Calculate the optimal privacy level for a transaction
    pub async fn calculate_privacy_level(
        &self,
        context: &TransactionContext,
    ) -> crate::error::MorphingResult<PrivacyCalculationResult> {
        info!("Calculating privacy level for transaction");
        
        // Start with requested or default level
        let mut level = context.requested_level
            .unwrap_or(self.config.default_privacy_level);

        // Apply compliance ceiling if required
        if context.compliance_required {
            let ceiling = self.compliance.get_privacy_ceiling(&context.recipient_type).await?;
            if level > ceiling {
                debug!("Applying compliance ceiling: {} -> {}", level.value(), ceiling.value());
                level = ceiling;
            }
        }

        // Check chain-specific limits
        if let Some(ref chain) = context.source_chain {
            if let Some(chain_config) = self.config.get_chain_config(chain) {
                if level > chain_config.max_privacy_level {
                    debug!("Applying chain ceiling: {} -> {}", level.value(), chain_config.max_privacy_level.value());
                    level = chain_config.max_privacy_level;
                }
            }
        }

        // Get anonymity set
        let required_size = level.required_anonymity_set_size();
        let anonymity_set = self.anonymity_manager
            .get_or_create_set(required_size, context.source_chain.as_deref())
            .await?;

        // Calculate mixing delay
        let mixing_delay = self.calculate_mixing_delay(&context.urgency, level);

        Ok(PrivacyCalculationResult {
            privacy_level: level,
            anonymity_set_id: anonymity_set.id,
            anonymity_set_size: anonymity_set.size,
            ring_size: level.ring_size(),
            mixing_delay_seconds: mixing_delay,
            compliance_attached: context.compliance_required,
            privacy_ceiling: if context.compliance_required {
                Some(self.compliance.get_privacy_ceiling(&context.recipient_type).await?)
            } else {
                None
            },
            calculated_at: Utc::now(),
        })
    }

    /// Apply a morphing schedule to an account
    pub async fn apply_schedule(
        &self,
        profile: &mut PrivacyProfile,
        schedule: crate::schedule::MorphingSchedule,
    ) -> MorphingResult<()> {
        let mut executor = self.schedule_executor.write().await;
        executor.register_schedule(profile.account_id.clone(), schedule.clone());
        profile.active_schedule = Some(schedule.id);
        info!("Applied morphing schedule {} to account {}", schedule.id, profile.account_id);
        Ok(())
    }

    /// Manually adjust privacy level
    pub async fn adjust_privacy_level(
        &self,
        profile: &mut PrivacyProfile,
        new_level: PrivacyLevel,
        reason: AdjustmentReason,
    ) -> MorphingResult<PrivacyAdjustment> {
        // Validate the new level is within profile bounds
        if !profile.is_level_allowed(new_level) {
            return Err(MorphingError::InvalidPrivacyLevel(new_level.value()));
        }

        let adjustment = PrivacyAdjustment {
            id: Uuid::new_v4(),
            from_level: profile.default_level,
            to_level: new_level,
            reason,
            timestamp: Utc::now(),
        };

        profile.default_level = new_level;
        profile.updated_at = Utc::now();

        info!(
            "Adjusted privacy level for {}: {} -> {}",
            profile.account_id,
            adjustment.from_level.value(),
            adjustment.to_level.value()
        );

        Ok(adjustment)
    }

    /// Process pending morphing operations
    pub async fn process_pending_morphs(&self) -> MorphingResult<Vec<PrivacyAdjustment>> {
        let mut executor = self.schedule_executor.write().await;
        let pending = executor.get_pending_morphs().await;
        
        let mut adjustments = Vec::new();
        for (account_id, new_level) in pending {
            debug!("Processing pending morph for account {}", account_id);
            // In a real implementation, this would update the account profile
            // For now, just record it
            adjustments.push(PrivacyAdjustment {
                id: Uuid::new_v4(),
                from_level: PrivacyLevel::default(),
                to_level: new_level,
                reason: AdjustmentReason::ScheduledMorph { 
                    schedule_id: Uuid::nil() 
                },
                timestamp: Utc::now(),
            });
        }

        Ok(adjustments)
    }

    /// Get current engine statistics
    pub async fn get_stats(&self) -> EngineStats {
        let anonymity_stats = self.anonymity_manager.get_stats().await;
        let executor = self.schedule_executor.read().await;
        let active_schedules = executor.active_schedule_count();

        EngineStats {
            active_anonymity_sets: anonymity_stats.active_sets,
            total_anonymity_entries: anonymity_stats.total_entries,
            active_schedules,
            morphing_enabled: self.config.auto_morphing_enabled,
        }
    }

    fn calculate_mixing_delay(&self, urgency: &TransactionUrgency, level: PrivacyLevel) -> u64 {
        let base_delay = urgency.mixing_time_seconds();
        
        // Higher privacy levels may need more mixing time
        let level_multiplier = match level.value() {
            1..=3 => 0.5,
            4..=6 => 1.0,
            7..=8 => 1.5,
            9..=10 => 2.0,
            _ => 1.0,
        };

        (base_delay as f64 * level_multiplier) as u64
    }
}

/// Engine statistics
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub active_anonymity_sets: usize,
    pub total_anonymity_entries: usize,
    pub active_schedules: usize,
    pub morphing_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let config = MorphingConfig::default();
        let engine = PrivacyMorphingEngine::new(config);
        
        let stats = engine.get_stats().await;
        assert!(stats.morphing_enabled);
    }

    #[tokio::test]
    async fn test_privacy_calculation() {
        let config = MorphingConfig::default();
        let engine = PrivacyMorphingEngine::new(config);

        let context = TransactionContext {
            amount: 1000,
            recipient_type: RecipientType::Normal,
            compliance_required: false,
            urgency: TransactionUrgency::Normal,
            source_chain: Some("ethereum".to_string()),
            dest_chain: None,
            requested_level: Some(PrivacyLevel::new(7)),
        };

        let result = engine.calculate_privacy_level(&context).await.unwrap();
        assert_eq!(result.privacy_level.value(), 7);
    }

    #[tokio::test]
    async fn test_privacy_adjustment() {
        let config = MorphingConfig::default();
        let engine = PrivacyMorphingEngine::new(config);

        let mut profile = PrivacyProfile::new("test".to_string());
        let adjustment = engine.adjust_privacy_level(
            &mut profile,
            PrivacyLevel::new(8),
            AdjustmentReason::UserRequest,
        ).await.unwrap();

        assert_eq!(adjustment.to_level.value(), 8);
        assert_eq!(profile.default_level.value(), 8);
    }
}
