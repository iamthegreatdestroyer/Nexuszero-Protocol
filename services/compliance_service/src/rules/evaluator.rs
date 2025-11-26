//! Rule Evaluator - Expression evaluation for custom rules

use crate::error::{ComplianceError, Result};
use crate::models::*;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

/// Rule evaluator for complex expressions
pub struct RuleEvaluator {
    /// Cached regex patterns
    regex_cache: HashMap<String, Regex>,
}

impl Default for RuleEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleEvaluator {
    pub fn new() -> Self {
        Self {
            regex_cache: HashMap::new(),
        }
    }

    /// Evaluate a rule condition against data
    pub fn evaluate(
        &mut self,
        condition: &Value,
        data: &HashMap<String, Value>,
    ) -> Result<bool> {
        match condition {
            Value::Object(obj) => {
                // Check for logical operators
                if let Some(and_conditions) = obj.get("$and") {
                    return self.evaluate_and(and_conditions, data);
                }
                if let Some(or_conditions) = obj.get("$or") {
                    return self.evaluate_or(or_conditions, data);
                }
                if let Some(not_condition) = obj.get("$not") {
                    return self.evaluate_not(not_condition, data);
                }
                
                // Evaluate field conditions
                for (field, condition) in obj {
                    if !self.evaluate_field_condition(field, condition, data)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Err(ComplianceError::InvalidRule(
                "Condition must be an object".to_string()
            )),
        }
    }

    /// Evaluate AND conditions
    fn evaluate_and(
        &mut self,
        conditions: &Value,
        data: &HashMap<String, Value>,
    ) -> Result<bool> {
        match conditions {
            Value::Array(arr) => {
                for condition in arr {
                    if !self.evaluate(condition, data)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Err(ComplianceError::InvalidRule(
                "$and must be an array".to_string()
            )),
        }
    }

    /// Evaluate OR conditions
    fn evaluate_or(
        &mut self,
        conditions: &Value,
        data: &HashMap<String, Value>,
    ) -> Result<bool> {
        match conditions {
            Value::Array(arr) => {
                for condition in arr {
                    if self.evaluate(condition, data)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Err(ComplianceError::InvalidRule(
                "$or must be an array".to_string()
            )),
        }
    }

    /// Evaluate NOT condition
    fn evaluate_not(
        &mut self,
        condition: &Value,
        data: &HashMap<String, Value>,
    ) -> Result<bool> {
        Ok(!self.evaluate(condition, data)?)
    }

    /// Evaluate single field condition
    fn evaluate_field_condition(
        &mut self,
        field: &str,
        condition: &Value,
        data: &HashMap<String, Value>,
    ) -> Result<bool> {
        let field_value = data.get(field);
        
        match condition {
            Value::Object(ops) => {
                for (op, expected) in ops {
                    let result = self.apply_operator(op, field_value, expected)?;
                    if !result {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            // Direct equality check
            expected => self.apply_operator("$eq", field_value, expected),
        }
    }

    /// Apply comparison operator
    fn apply_operator(
        &mut self,
        op: &str,
        actual: Option<&Value>,
        expected: &Value,
    ) -> Result<bool> {
        match op {
            "$eq" => Ok(actual == Some(expected)),
            "$ne" => Ok(actual != Some(expected)),
            "$gt" => self.compare_numeric(actual, expected, |a, b| a > b),
            "$gte" => self.compare_numeric(actual, expected, |a, b| a >= b),
            "$lt" => self.compare_numeric(actual, expected, |a, b| a < b),
            "$lte" => self.compare_numeric(actual, expected, |a, b| a <= b),
            "$in" => self.check_in(actual, expected),
            "$nin" => Ok(!self.check_in(actual, expected)?),
            "$exists" => {
                let should_exist = expected.as_bool().unwrap_or(true);
                Ok(actual.is_some() == should_exist)
            }
            "$regex" => self.check_regex(actual, expected),
            "$contains" => self.check_contains(actual, expected),
            "$startsWith" => self.check_starts_with(actual, expected),
            "$endsWith" => self.check_ends_with(actual, expected),
            "$between" => self.check_between(actual, expected),
            _ => Err(ComplianceError::InvalidRule(
                format!("Unknown operator: {}", op)
            )),
        }
    }

    /// Numeric comparison
    fn compare_numeric<F>(
        &self,
        actual: Option<&Value>,
        expected: &Value,
        compare: F,
    ) -> Result<bool>
    where
        F: Fn(f64, f64) -> bool,
    {
        let actual_num = actual
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ComplianceError::InvalidRule(
                "Field must be numeric".to_string()
            ))?;
        
        let expected_num = expected
            .as_f64()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "Expected value must be numeric".to_string()
            ))?;
        
        Ok(compare(actual_num, expected_num))
    }

    /// Check if value is in array
    fn check_in(&self, actual: Option<&Value>, expected: &Value) -> Result<bool> {
        let arr = expected
            .as_array()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "$in requires an array".to_string()
            ))?;
        
        match actual {
            Some(val) => Ok(arr.contains(val)),
            None => Ok(false),
        }
    }

    /// Check regex pattern
    fn check_regex(&mut self, actual: Option<&Value>, expected: &Value) -> Result<bool> {
        let pattern = expected
            .as_str()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "$regex requires a string pattern".to_string()
            ))?;
        
        let actual_str = actual
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        // Get or compile regex
        let regex = if let Some(r) = self.regex_cache.get(pattern) {
            r
        } else {
            let compiled = Regex::new(pattern).map_err(|e| {
                ComplianceError::InvalidRule(format!("Invalid regex: {}", e))
            })?;
            self.regex_cache.insert(pattern.to_string(), compiled);
            self.regex_cache.get(pattern).unwrap()
        };
        
        Ok(regex.is_match(actual_str))
    }

    /// Check if string contains substring
    fn check_contains(&self, actual: Option<&Value>, expected: &Value) -> Result<bool> {
        let needle = expected
            .as_str()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "$contains requires a string".to_string()
            ))?;
        
        let haystack = actual
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        Ok(haystack.contains(needle))
    }

    /// Check if string starts with prefix
    fn check_starts_with(&self, actual: Option<&Value>, expected: &Value) -> Result<bool> {
        let prefix = expected
            .as_str()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "$startsWith requires a string".to_string()
            ))?;
        
        let value = actual
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        Ok(value.starts_with(prefix))
    }

    /// Check if string ends with suffix
    fn check_ends_with(&self, actual: Option<&Value>, expected: &Value) -> Result<bool> {
        let suffix = expected
            .as_str()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "$endsWith requires a string".to_string()
            ))?;
        
        let value = actual
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        Ok(value.ends_with(suffix))
    }

    /// Check if value is between range
    fn check_between(&self, actual: Option<&Value>, expected: &Value) -> Result<bool> {
        let range = expected
            .as_array()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "$between requires [min, max] array".to_string()
            ))?;
        
        if range.len() != 2 {
            return Err(ComplianceError::InvalidRule(
                "$between requires exactly 2 values".to_string()
            ));
        }
        
        let min = range[0].as_f64()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "Min value must be numeric".to_string()
            ))?;
        
        let max = range[1].as_f64()
            .ok_or_else(|| ComplianceError::InvalidRule(
                "Max value must be numeric".to_string()
            ))?;
        
        let value = actual
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ComplianceError::InvalidRule(
                "Field must be numeric".to_string()
            ))?;
        
        Ok(value >= min && value <= max)
    }

    /// Test a rule against sample data
    pub fn test_rule(
        &mut self,
        rule: &ComplianceRule,
        test_data: &Value,
    ) -> Result<RuleTestResult> {
        let start = std::time::Instant::now();
        
        // Convert test data to HashMap
        let data: HashMap<String, Value> = if let Value::Object(obj) = test_data {
            obj.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        } else {
            HashMap::new()
        };
        
        let triggered = self.evaluate(&rule.conditions, &data)?;
        
        let matched_conditions = if triggered {
            self.get_matched_conditions(&rule.conditions, &data)?
        } else {
            vec![]
        };
        
        let actions_to_execute = if triggered {
            rule.actions.clone()
        } else {
            vec![]
        };
        
        let elapsed = start.elapsed();

        Ok(RuleTestResult {
            rule_id: rule.id,
            triggered,
            matched_conditions,
            actions_to_execute,
            execution_time_ms: elapsed.as_millis() as u64,
        })
    }

    /// Get list of matched conditions for debugging
    fn get_matched_conditions(
        &mut self,
        condition: &Value,
        data: &HashMap<String, Value>,
    ) -> Result<Vec<String>> {
        let mut matched = Vec::new();
        
        if let Value::Object(obj) = condition {
            for (field, cond) in obj {
                if field.starts_with('$') {
                    continue;
                }
                
                if self.evaluate_field_condition(field, cond, data)? {
                    matched.push(format!("{}: {:?}", field, cond));
                }
            }
        }
        
        Ok(matched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equality() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("status".to_string(), Value::String("active".to_string()));
        
        let condition = serde_json::json!({
            "status": "active"
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
        
        data.insert("status".to_string(), Value::String("inactive".to_string()));
        assert!(!evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_numeric_comparison() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("amount".to_string(), Value::Number(serde_json::Number::from(15000)));
        
        let condition = serde_json::json!({
            "amount": { "$gt": 10000 }
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
        
        let condition = serde_json::json!({
            "amount": { "$lt": 10000 }
        });
        
        assert!(!evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_in_operator() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("country".to_string(), Value::String("US".to_string()));
        
        let condition = serde_json::json!({
            "country": { "$in": ["US", "CA", "GB"] }
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
        
        data.insert("country".to_string(), Value::String("KP".to_string()));
        assert!(!evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_and_operator() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("amount".to_string(), Value::Number(serde_json::Number::from(15000)));
        data.insert("currency".to_string(), Value::String("USD".to_string()));
        
        let condition = serde_json::json!({
            "$and": [
                { "amount": { "$gt": 10000 } },
                { "currency": "USD" }
            ]
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
        
        data.insert("currency".to_string(), Value::String("EUR".to_string()));
        assert!(!evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_or_operator() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("amount".to_string(), Value::Number(serde_json::Number::from(5000)));
        data.insert("risk".to_string(), Value::String("high".to_string()));
        
        let condition = serde_json::json!({
            "$or": [
                { "amount": { "$gt": 10000 } },
                { "risk": "high" }
            ]
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_regex() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("email".to_string(), Value::String("user@example.com".to_string()));
        
        let condition = serde_json::json!({
            "email": { "$regex": r"^[a-z]+@[a-z]+\.[a-z]+$" }
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_between() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("amount".to_string(), Value::Number(serde_json::Number::from(5000)));
        
        let condition = serde_json::json!({
            "amount": { "$between": [1000, 10000] }
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
        
        data.insert("amount".to_string(), Value::Number(serde_json::Number::from(15000)));
        assert!(!evaluator.evaluate(&condition, &data).unwrap());
    }

    #[test]
    fn test_exists() {
        let mut evaluator = RuleEvaluator::new();
        let mut data = HashMap::new();
        data.insert("name".to_string(), Value::String("John".to_string()));
        
        let condition = serde_json::json!({
            "name": { "$exists": true },
            "email": { "$exists": false }
        });
        
        assert!(evaluator.evaluate(&condition, &data).unwrap());
    }
}
