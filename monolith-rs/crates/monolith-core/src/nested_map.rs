//! NestedMap: a dict-like tree with dot-path access and list traversal.
//!
//! This mirrors the behavior of `monolith/core/py_utils.py::NestedMap` with
//! explicit methods for nested access, flattening, packing, and filtering.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::dyn_value::DynValue;
use crate::error::{MonolithError, Result};

static KEY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").expect("valid key regex"));

static RESERVED_KEYS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "keys", "values", "items", "get", "insert", "remove", "clear", "len", "is_empty",
        "contains_key", "flatten", "pack", "filter", "map", "debug_string",
    ]
});

/// NestedMap leaf or container.
#[derive(Clone)]
pub enum NestedValue {
    Map(NestedMap),
    List(Vec<NestedValue>),
    Value(Arc<dyn DynValue>),
}

impl fmt::Debug for NestedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NestedValue::Map(m) => write!(f, "{:?}", m),
            NestedValue::List(v) => write!(f, "{:?}", v),
            NestedValue::Value(v) => write!(f, "{:?}", v),
        }
    }
}

impl NestedValue {
    fn is_container(&self) -> bool {
        matches!(self, NestedValue::Map(_) | NestedValue::List(_))
    }
}

/// A map that supports attribute-style keys and recursive operations.
#[derive(Clone, Default)]
pub struct NestedMap {
    map: BTreeMap<String, NestedValue>,
}

impl fmt::Debug for NestedMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.map.iter()).finish()
    }
}

impl NestedMap {
    /// Creates an empty NestedMap.
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Validates a key for NestedMap usage.
    pub fn check_key(key: &str) -> Result<()> {
        if !KEY_RE.is_match(key) {
            return Err(MonolithError::ConfigError {
                message: format!("Invalid NestedMap key '{}'", key),
            });
        }
        if RESERVED_KEYS.contains(&key) {
            return Err(MonolithError::ConfigError {
                message: format!("'{}' is a reserved key", key),
            });
        }
        Ok(())
    }

    /// Inserts a value with a simple (non-nested) key.
    pub fn insert(&mut self, key: &str, value: NestedValue) -> Result<()> {
        Self::check_key(key)?;
        self.map.insert(key.to_string(), value);
        Ok(())
    }

    /// Returns a reference to the value for a simple key.
    pub fn get(&self, key: &str) -> Option<&NestedValue> {
        self.map.get(key)
    }

    /// Returns a mutable reference to the value for a simple key.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut NestedValue> {
        self.map.get_mut(key)
    }

    /// Returns a deep copy of this NestedMap.
    pub fn deep_copy(&self) -> Self {
        self.clone()
    }

    /// Converts a nested `BTreeMap`/`Vec` structure into a NestedMap recursively.
    pub fn from_nested(value: NestedValue) -> NestedValue {
        match value {
            NestedValue::Map(m) => NestedValue::Map(m),
            NestedValue::List(v) => {
                NestedValue::List(v.into_iter().map(NestedMap::from_nested).collect())
            }
            NestedValue::Value(v) => NestedValue::Value(v),
        }
    }

    /// Gets the value for a nested key with dot notation.
    pub fn get_item(&self, key: &str) -> Result<&NestedValue> {
        let mut current = self;
        let parts = key.split('.').collect::<Vec<_>>();
        for part in parts.iter().take(parts.len().saturating_sub(1)) {
            let next = current
                .map
                .get(*part)
                .ok_or_else(|| MonolithError::ConfigError {
                    message: format!("Key '{}' not found", part),
                })?;
            match next {
                NestedValue::Map(map) => current = map,
                _ => {
                    return Err(MonolithError::ConfigError {
                        message: format!("Sub key '{}' is not a map", part),
                    })
                }
            }
        }
        let last = parts.last().unwrap_or(&key);
        current.map.get(*last).ok_or_else(|| MonolithError::ConfigError {
            message: format!("Key '{}' not found", last),
        })
    }

    /// Gets the value for a nested key, returns None if missing.
    pub fn get_or_none(&self, key: &str) -> Option<&NestedValue> {
        self.get_item(key).ok()
    }

    /// Sets a nested key with dot notation.
    pub fn set(&mut self, key: &str, value: NestedValue) -> Result<()> {
        let parts = key.split('.').collect::<Vec<_>>();
        let mut current = self;
        for part in parts.iter().take(parts.len().saturating_sub(1)) {
            Self::check_key(part)?;
            if !current.map.contains_key(*part) {
                current
                    .map
                    .insert((*part).to_string(), NestedValue::Map(NestedMap::new()));
            }
            let next = current.map.get_mut(*part).unwrap();
            match next {
                NestedValue::Map(map) => current = map,
                _ => {
                    return Err(MonolithError::ConfigError {
                        message: format!(
                            "Error while setting key {}. Sub key '{}' is not a map.",
                            key, part
                        ),
                    })
                }
            }
        }
        let last = parts.last().unwrap_or(&key);
        Self::check_key(last)?;
        current.map.insert((*last).to_string(), value);
        Ok(())
    }

    /// Returns flattened values in traversal order (maps + lists).
    pub fn flatten(&self) -> Vec<NestedValue> {
        self.flatten_items()
            .into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    /// Returns flattened (key, value) pairs.
    pub fn flatten_items(&self) -> Vec<(String, NestedValue)> {
        let mut out = Vec::new();
        self.recursive_flatten("", &NestedValue::Map(self.clone()), &mut out);
        out
    }

    fn recursive_flatten(
        &self,
        prefix: &str,
        value: &NestedValue,
        out: &mut Vec<(String, NestedValue)>,
    ) {
        match value {
            NestedValue::Map(map) => {
                for (k, v) in &map.map {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{}.{}", prefix, k)
                    };
                    self.recursive_flatten(&key, v, out);
                }
            }
            NestedValue::List(list) => {
                for (idx, v) in list.iter().enumerate() {
                    let key = format!("{}[{}]", prefix, idx);
                    self.recursive_flatten(&key, v, out);
                }
            }
            _ => out.push((prefix.to_string(), value.clone())),
        }
    }

    /// Returns a copy with values replaced by `values`, in flatten order.
    pub fn pack(&self, values: Vec<NestedValue>) -> Result<NestedMap> {
        let mut iter = values.into_iter();
        Ok(self.recursive_pack(&mut iter))
    }

    fn recursive_pack(&self, iter: &mut dyn Iterator<Item = NestedValue>) -> NestedMap {
        let mut out = NestedMap::new();
        for (k, v) in &self.map {
            let packed = pack_value(v, iter);
            let _ = out.insert(k, packed);
        }
        out
    }

    /// Applies a transform to each leaf value, preserving structure.
    pub fn transform<F>(&self, mut f: F) -> NestedMap
    where
        F: FnMut(&NestedValue) -> NestedValue,
    {
        self.recursive_transform(&mut f)
    }

    fn recursive_transform<F>(&self, f: &mut F) -> NestedMap
    where
        F: FnMut(&NestedValue) -> NestedValue,
    {
        let mut out = NestedMap::new();
        for (k, v) in &self.map {
            let value = match v {
                NestedValue::Map(map) => NestedValue::Map(map.recursive_transform(f)),
                NestedValue::List(list) => NestedValue::List(
                    list.iter()
                        .map(|item| match item {
                            NestedValue::Map(map) => NestedValue::Map(map.recursive_transform(f)),
                            NestedValue::List(_) => f(item),
                            _ => f(item),
                        })
                        .collect(),
                ),
                _ => f(v),
            };
            let _ = out.insert(k, value);
        }
        out
    }

    /// Checks whether this NestedMap is compatible with another.
    pub fn is_compatible(&self, other: &NestedMap) -> bool {
        let keys_self = self
            .flatten_items()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<Vec<_>>();
        let keys_other = other
            .flatten_items()
            .into_iter()
            .map(|(k, _)| k)
            .collect::<Vec<_>>();
        keys_self == keys_other
    }

    /// Filters entries based on a predicate over values.
    pub fn filter<F>(&self, mut f: F) -> NestedMap
    where
        F: FnMut(&NestedValue) -> bool,
    {
        self.filter_key_val(|_, v| f(v))
    }

    /// Filters entries based on a predicate over (key, value).
    pub fn filter_key_val<F>(&self, mut f: F) -> NestedMap
    where
        F: FnMut(&str, &NestedValue) -> bool,
    {
        self.recursive_filter("", &mut f)
    }

    fn recursive_filter<F>(&self, prefix: &str, f: &mut F) -> NestedMap
    where
        F: FnMut(&str, &NestedValue) -> bool,
    {
        let mut out = NestedMap::new();
        for (k, v) in &self.map {
            let key = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{}.{}", prefix, k)
            };
            let keep = match v {
                NestedValue::Map(map) => {
                    let filtered = map.recursive_filter(&key, f);
                    if filtered.map.is_empty() {
                        None
                    } else {
                        Some(NestedValue::Map(filtered))
                    }
                }
                NestedValue::List(list) => {
                    let mut filtered_list = Vec::new();
                    for (idx, item) in list.iter().enumerate() {
                        let list_key = format!("{}[{}]", key, idx);
                        if f(&list_key, item) {
                            filtered_list.push(item.clone());
                        }
                    }
                    if filtered_list.is_empty() {
                        None
                    } else {
                        Some(NestedValue::List(filtered_list))
                    }
                }
                _ => {
                    if f(&key, v) {
                        Some(v.clone())
                    } else {
                        None
                    }
                }
            };
            if let Some(val) = keep {
                let _ = out.insert(k, val);
            }
        }
        out
    }

    /// Returns a debug string listing keys and values.
    pub fn debug_string(&self) -> String {
        let items = self.flatten_items();
        let max_len = items.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let mut lines = Vec::new();
        for (k, v) in items {
            lines.push(format!("{:width$}    {:?}", k, v, width = max_len));
        }
        lines.join("\n")
    }
}

fn pack_value(template: &NestedValue, iter: &mut dyn Iterator<Item = NestedValue>) -> NestedValue {
    match template {
        NestedValue::Map(map) => NestedValue::Map(map.recursive_pack(iter)),
        NestedValue::List(list) => {
            let packed = list
                .iter()
                .map(|item| pack_value(item, iter))
                .collect();
            NestedValue::List(packed)
        }
        _ => iter.next().unwrap_or_else(|| template.clone()),
    }
}
