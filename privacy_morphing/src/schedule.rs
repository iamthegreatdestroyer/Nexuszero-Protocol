//! Morphing Schedule Management

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::PrivacyLevel;

/// A morphing schedule defines how privacy levels change over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphingSchedule {
    /// Unique schedule ID
    pub id: Uuid,
    /// Schedule name
    pub name: String,
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Schedule steps
    pub steps: Vec<ScheduleStep>,
    /// Whether the schedule repeats
    pub repeating: bool,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last executed step index
    pub last_executed_step: Option<usize>,
}

/// Types of morphing schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    /// Gradually increase privacy over time
    GradualIncrease {
        start_level: PrivacyLevel,
        end_level: PrivacyLevel,
        duration_hours: u64,
    },
    /// Gradually decrease privacy over time
    GradualDecrease {
        start_level: PrivacyLevel,
        end_level: PrivacyLevel,
        duration_hours: u64,
    },
    /// Cycle between levels
    Cyclic {
        levels: Vec<PrivacyLevel>,
        cycle_duration_hours: u64,
    },
    /// Time-of-day based
    TimeOfDay {
        day_level: PrivacyLevel,
        night_level: PrivacyLevel,
        day_start_hour: u8,
        night_start_hour: u8,
    },
    /// Custom step-based
    Custom,
}

/// A single step in a morphing schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleStep {
    /// Step index
    pub index: usize,
    /// Target privacy level
    pub target_level: PrivacyLevel,
    /// When to execute (relative to schedule start or previous step)
    pub delay: Duration,
    /// Optional condition
    pub condition: Option<StepCondition>,
}

/// Conditions for schedule step execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepCondition {
    /// Only execute if anonymity set size is above threshold
    MinAnonymitySet(usize),
    /// Only execute if network is not congested
    NetworkNotCongested,
    /// Only execute during certain hours
    TimeRange { start_hour: u8, end_hour: u8 },
}

impl MorphingSchedule {
    /// Create a new gradual increase schedule
    pub fn gradual_increase(
        name: &str,
        start: PrivacyLevel,
        end: PrivacyLevel,
        duration_hours: u64,
        steps: usize,
    ) -> Self {
        let step_duration = Duration::hours(duration_hours as i64 / steps as i64);
        let level_increment = (end.value() - start.value()) as f64 / steps as f64;

        let schedule_steps: Vec<ScheduleStep> = (0..=steps)
            .map(|i| {
                let level = (start.value() as f64 + level_increment * i as f64).round() as u8;
                ScheduleStep {
                    index: i,
                    target_level: PrivacyLevel::new(level),
                    delay: step_duration * i as i32,
                    condition: None,
                }
            })
            .collect();

        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            schedule_type: ScheduleType::GradualIncrease {
                start_level: start,
                end_level: end,
                duration_hours,
            },
            steps: schedule_steps,
            repeating: false,
            created_at: Utc::now(),
            last_executed_step: None,
        }
    }

    /// Create a time-of-day schedule
    pub fn time_of_day(
        name: &str,
        day_level: PrivacyLevel,
        night_level: PrivacyLevel,
        day_start: u8,
        night_start: u8,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            schedule_type: ScheduleType::TimeOfDay {
                day_level,
                night_level,
                day_start_hour: day_start,
                night_start_hour: night_start,
            },
            steps: vec![
                ScheduleStep {
                    index: 0,
                    target_level: day_level,
                    delay: Duration::zero(),
                    condition: Some(StepCondition::TimeRange {
                        start_hour: day_start,
                        end_hour: night_start,
                    }),
                },
                ScheduleStep {
                    index: 1,
                    target_level: night_level,
                    delay: Duration::zero(),
                    condition: Some(StepCondition::TimeRange {
                        start_hour: night_start,
                        end_hour: day_start,
                    }),
                },
            ],
            repeating: true,
            created_at: Utc::now(),
            last_executed_step: None,
        }
    }

    /// Get the next step to execute
    pub fn next_step(&self) -> Option<&ScheduleStep> {
        let next_index = self.last_executed_step.map_or(0, |i| i + 1);
        
        if next_index < self.steps.len() {
            Some(&self.steps[next_index])
        } else if self.repeating {
            Some(&self.steps[0])
        } else {
            None
        }
    }

    /// Mark a step as executed
    pub fn mark_step_executed(&mut self, step_index: usize) {
        self.last_executed_step = Some(step_index);
    }
}

/// Executor for morphing schedules
pub struct ScheduleExecutor {
    /// Active schedules by account ID
    schedules: HashMap<String, MorphingSchedule>,
    /// Schedule start times
    start_times: HashMap<Uuid, DateTime<Utc>>,
}

impl ScheduleExecutor {
    /// Create a new schedule executor
    pub fn new() -> Self {
        Self {
            schedules: HashMap::new(),
            start_times: HashMap::new(),
        }
    }

    /// Register a schedule for an account
    pub fn register_schedule(&mut self, account_id: String, schedule: MorphingSchedule) {
        self.start_times.insert(schedule.id, Utc::now());
        self.schedules.insert(account_id, schedule);
    }

    /// Remove a schedule
    pub fn remove_schedule(&mut self, account_id: &str) -> Option<MorphingSchedule> {
        if let Some(schedule) = self.schedules.remove(account_id) {
            self.start_times.remove(&schedule.id);
            Some(schedule)
        } else {
            None
        }
    }

    /// Get pending morphs that should be executed
    pub async fn get_pending_morphs(&mut self) -> Vec<(String, PrivacyLevel)> {
        let now = Utc::now();
        let mut pending = Vec::new();
        let mut executed_steps: Vec<(String, usize)> = Vec::new();

        for (account_id, schedule) in self.schedules.iter() {
            if let Some(step) = schedule.next_step() {
                let start_time = self.start_times.get(&schedule.id).copied()
                    .unwrap_or(now);
                
                let execute_time = start_time + step.delay;
                
                if now >= execute_time {
                    // Check conditions
                    let condition_met = step.condition.as_ref().map_or(true, |c| {
                        Self::check_condition_static(c)
                    });
                    
                    if condition_met {
                        pending.push((account_id.clone(), step.target_level));
                        executed_steps.push((account_id.clone(), step.index));
                    }
                }
            }
        }

        // Mark executed steps after iteration is complete
        for (account_id, step_index) in executed_steps {
            if let Some(schedule) = self.schedules.get_mut(&account_id) {
                schedule.mark_step_executed(step_index);
            }
        }

        pending
    }

    /// Get count of active schedules
    pub fn active_schedule_count(&self) -> usize {
        self.schedules.len()
    }

    fn check_condition_static(condition: &StepCondition) -> bool {
        match condition {
            StepCondition::MinAnonymitySet(min_size) => {
                // In real implementation, check actual anonymity set size
                *min_size <= 100
            }
            StepCondition::NetworkNotCongested => {
                // In real implementation, check network status
                true
            }
            StepCondition::TimeRange { start_hour, end_hour } => {
                let current_hour = Utc::now().format("%H").to_string().parse::<u8>().unwrap_or(0);
                if start_hour <= end_hour {
                    current_hour >= *start_hour && current_hour < *end_hour
                } else {
                    // Wraps around midnight
                    current_hour >= *start_hour || current_hour < *end_hour
                }
            }
        }
    }
}

impl Default for ScheduleExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradual_increase_schedule() {
        let schedule = MorphingSchedule::gradual_increase(
            "test",
            PrivacyLevel::new(3),
            PrivacyLevel::new(7),
            4,
            4,
        );

        assert_eq!(schedule.steps.len(), 5);
        assert_eq!(schedule.steps[0].target_level.value(), 3);
        assert_eq!(schedule.steps[4].target_level.value(), 7);
    }

    #[test]
    fn test_time_of_day_schedule() {
        let schedule = MorphingSchedule::time_of_day(
            "work hours",
            PrivacyLevel::new(5),
            PrivacyLevel::new(8),
            9,
            21,
        );

        assert_eq!(schedule.steps.len(), 2);
        assert!(schedule.repeating);
    }

    #[test]
    fn test_schedule_executor() {
        let mut executor = ScheduleExecutor::new();
        
        let schedule = MorphingSchedule::gradual_increase(
            "test",
            PrivacyLevel::new(3),
            PrivacyLevel::new(7),
            0, // Immediate
            4,
        );

        executor.register_schedule("account1".to_string(), schedule);
        assert_eq!(executor.active_schedule_count(), 1);
    }
}
