//! Dynamic hyperparameter containers mirroring Python `Params` and `InstantiableParams`.
//!
//! This module provides a flexible, runtime-configurable parameter store that
//! supports nested parameters, immutability, dotted-path access, and deep copy
//! semantics similar to the Python implementation in `monolith/core/hyperparams.py`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::dyn_value::DynValue;
use crate::error::{MonolithError, Result};

static PARAM_NAME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[a-z][a-z0-9_]*$").expect("valid param regex"));

/// A dynamically typed value stored in [`Params`].
#[derive(Clone)]
pub enum ParamValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<ParamValue>),
    Map(BTreeMap<String, ParamValue>),
    Params(Box<Params>),
    External(Arc<dyn DynValue>),
}

impl ParamValue {
    fn to_debug_string(&self) -> String {
        match self {
            ParamValue::None => "None".to_string(),
            ParamValue::Bool(v) => v.to_string(),
            ParamValue::Int(v) => v.to_string(),
            ParamValue::Float(v) => v.to_string(),
            ParamValue::String(v) => format!("\"{}\"", v),
            ParamValue::List(v) => format!("{:?}", v.iter().map(|x| x.to_debug_string()).collect::<Vec<_>>()),
            ParamValue::Map(v) => {
                let items = v
                    .iter()
                    .map(|(k, val)| format!("{:?}: {}", k, val.to_debug_string()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{}}}", items)
            }
            ParamValue::Params(p) => p.to_string(),
            ParamValue::External(v) => format!("{:?}", v),
        }
    }

    /// Wraps an external value (non-serializable) into a ParamValue.
    pub fn external<T>(value: T) -> Self
    where
        T: DynValue,
    {
        ParamValue::External(Arc::new(value))
    }
}

impl fmt::Debug for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_debug_string())
    }
}

macro_rules! impl_from {
    ($t:ty, $variant:ident) => {
        impl From<$t> for ParamValue {
            fn from(value: $t) -> Self {
                ParamValue::$variant(value)
            }
        }
    };
}

impl_from!(bool, Bool);
impl_from!(i64, Int);
impl_from!(f64, Float);
impl_from!(String, String);

impl From<&str> for ParamValue {
    fn from(value: &str) -> Self {
        ParamValue::String(value.to_string())
    }
}

impl From<Params> for ParamValue {
    fn from(value: Params) -> Self {
        ParamValue::Params(Box::new(value))
    }
}

impl<T> From<Vec<T>> for ParamValue
where
    T: Into<ParamValue>,
{
    fn from(value: Vec<T>) -> Self {
        ParamValue::List(value.into_iter().map(|v| v.into()).collect())
    }
}

/// A single parameter entry.
#[derive(Clone, Debug)]
pub struct Param {
    name: String,
    value: ParamValue,
    description: String,
}

impl Param {
    fn new(name: impl Into<String>, value: ParamValue, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            description: description.into(),
        }
    }
}

/// Parameter container with dotted-path access and immutability.
#[derive(Clone, Debug, Default)]
pub struct Params {
    immutable: bool,
    params: BTreeMap<String, Param>,
}

impl Params {
    /// Creates an empty Params object.
    pub fn new() -> Self {
        Self {
            immutable: false,
            params: BTreeMap::new(),
        }
    }

    /// Defines a parameter with default value and description.
    pub fn define(
        &mut self,
        name: &str,
        default_value: impl Into<ParamValue>,
        description: &str,
    ) -> Result<()> {
        if self.immutable {
            return Err(MonolithError::ConfigError {
                message: "Params instance is immutable".to_string(),
            });
        }
        if !PARAM_NAME_RE.is_match(name) {
            return Err(MonolithError::ConfigError {
                message: format!("Invalid param name: {}", name),
            });
        }
        if self.params.contains_key(name) {
            return Err(MonolithError::ConfigError {
                message: format!("Parameter {} is already defined", name),
            });
        }
        self.params
            .insert(name.to_string(), Param::new(name, default_value.into(), description));
        Ok(())
    }

    /// Returns true if the parameter is defined.
    pub fn contains(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Freezes the Params to make it immutable.
    pub fn freeze(&mut self) {
        self.immutable = true;
    }

    /// Returns whether the Params is immutable.
    pub fn is_immutable(&self) -> bool {
        self.immutable
    }

    /// Returns a deep copy of this Params.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Sets a parameter using dotted path notation.
    pub fn set(&mut self, name: &str, value: impl Into<ParamValue>) -> Result<()> {
        if self.immutable {
            return Err(MonolithError::ConfigError {
                message: format!("Params instance is immutable: {}", self),
            });
        }
        let (parent, key) = self.get_nested_params_mut(name)?;
        match parent.params.get_mut(&key) {
            Some(param) => {
                param.value = value.into();
                Ok(())
            }
            None => Err(MonolithError::ConfigError {
                message: parent.key_error_string(name),
            }),
        }
    }

    /// Gets a parameter using dotted path notation.
    pub fn get(&self, name: &str) -> Result<&ParamValue> {
        let (parent, key) = self.get_nested_params(name)?;
        parent
            .params
            .get(&key)
            .map(|p| &p.value)
            .ok_or_else(|| MonolithError::ConfigError {
                message: parent.key_error_string(name),
            })
    }

    /// Deletes a parameter using dotted path notation.
    pub fn delete(&mut self, name: &str) -> Result<()> {
        if self.immutable {
            return Err(MonolithError::ConfigError {
                message: "Params instance is immutable".to_string(),
            });
        }
        let (parent, key) = self.get_nested_params_mut(name)?;
        if parent.params.remove(&key).is_none() {
            return Err(MonolithError::ConfigError {
                message: parent.key_error_string(name),
            });
        }
        Ok(())
    }

    /// Returns an iterator over parameters (name, value).
    pub fn iter_params(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.params.iter().map(|(k, v)| (k.as_str(), &v.value))
    }

    fn get_nested_params<'a>(&'a self, name: &str) -> Result<(&'a Params, String)> {
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = self;
        for part in &parts[..parts.len().saturating_sub(1)] {
            let (base, index) = parse_list_index(part)?;
            let param = current.params.get(base).ok_or_else(|| MonolithError::ConfigError {
                message: current.key_error_string(base),
            })?;
            current = match (&param.value, index) {
                (ParamValue::Params(p), None) => p.as_ref(),
                (ParamValue::List(list), Some(idx)) => match list.get(idx) {
                    Some(ParamValue::Params(p)) => p.as_ref(),
                    _ => {
                        return Err(MonolithError::ConfigError {
                            message: format!("Invalid list index {} for {}", idx, base),
                        })
                    }
                },
                (ParamValue::List(_), None) => {
                    return Err(MonolithError::ConfigError {
                        message: format!("Expected list index for {}", base),
                    })
                }
                _ => {
                    return Err(MonolithError::ConfigError {
                        message: format!(
                            "Cannot introspect {} for {}",
                            part,
                            parts[..parts.len().saturating_sub(1)].join(".")
                        ),
                    })
                }
            };
        }
        let key = parts.last().unwrap_or(&name).to_string();
        Ok((current, key))
    }

    fn get_nested_params_mut<'a>(&'a mut self, name: &str) -> Result<(&'a mut Params, String)> {
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = self as *mut Params;
        for part in &parts[..parts.len().saturating_sub(1)] {
            let (base, index) = parse_list_index(part)?;
            // SAFETY: we only create one mutable reference at a time during traversal.
            let curr = unsafe { &mut *current };
            let err_msg = curr.key_error_string(base);
            let param = curr.params.get_mut(base).ok_or_else(|| MonolithError::ConfigError {
                message: err_msg,
            })?;
            let next = match (&mut param.value, index) {
                (ParamValue::Params(p), None) => p.as_mut(),
                (ParamValue::List(list), Some(idx)) => match list.get_mut(idx) {
                    Some(ParamValue::Params(p)) => p.as_mut(),
                    _ => {
                        return Err(MonolithError::ConfigError {
                            message: format!("Invalid list index {} for {}", idx, base),
                        })
                    }
                },
                (ParamValue::List(_), None) => {
                    return Err(MonolithError::ConfigError {
                        message: format!("Expected list index for {}", base),
                    })
                }
                _ => {
                    return Err(MonolithError::ConfigError {
                        message: format!(
                            "Cannot introspect {} for {}",
                            part,
                            parts[..parts.len().saturating_sub(1)].join(".")
                        ),
                    })
                }
            };
            current = next as *mut Params;
        }
        let key = parts.last().unwrap_or(&name).to_string();
        // SAFETY: current is valid and unique at this point.
        Ok((unsafe { &mut *current }, key))
    }

    fn key_error_string(&self, name: &str) -> String {
        let similar = self.similar_keys(name);
        if similar.is_empty() {
            name.to_string()
        } else {
            format!("{} (did you mean: [{}])", name, similar.join(","))
        }
    }

    fn similar_keys(&self, name: &str) -> Vec<String> {
        fn overlap(name: &str, key: &str) -> f32 {
            let mut matches = 0;
            let mut trials = 0;
            if name.len() < 3 || key.len() < 3 {
                return 0.0;
            }
            for i in 0..(name.len() - 2) {
                trials += 1;
                if key.contains(&name[i..i + 3]) {
                    matches += 1;
                }
            }
            if trials == 0 {
                0.0
            } else {
                matches as f32 / trials as f32
            }
        }
        self.params
            .keys()
            .filter(|k| overlap(name, k) > 0.5)
            .cloned()
            .collect()
    }
}

impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indent = "  ";
        let mut lines = Vec::new();
        for (key, param) in &self.params {
            let val_str = match &param.value {
                ParamValue::Params(p) => format!("{}", p),
                _ => param.value.to_debug_string(),
            };
            lines.push(format!("{}{}: {}", indent, key, val_str));
        }
        writeln!(f, "{{")?;
        for line in lines {
            writeln!(f, "{}", line)?;
        }
        write!(f, "}}")
    }
}

/// Copies parameters from one Params to another, skipping optional keys.
pub fn copy_params_to(from_p: &Params, to_p: &mut Params, skip: Option<&[&str]>) {
    for (name, value) in from_p.iter_params() {
        if skip.map(|s| s.contains(&name)).unwrap_or(false) {
            continue;
        }
        match value {
            ParamValue::Params(p) => {
                let _ = to_p.set(name, ParamValue::Params(Box::new(p.as_ref().copy())));
            }
            _ => {
                let _ = to_p.set(name, value.clone());
            }
        }
    }
}

/// Updates params with values from a flat map.
pub fn update_params(params: &mut Params, values: &mut BTreeMap<String, ParamValue>) {
    let keys: Vec<String> = params.params.keys().cloned().collect();
    for key in keys {
        let current = params.params.get(&key).map(|p| p.value.clone());
        if let Some(ParamValue::Params(mut nested)) = current {
            update_params(&mut nested, values);
            let _ = params.set(&key, ParamValue::Params(nested));
        } else if let Some(value) = values.remove(&key) {
            let _ = params.set(&key, value);
        }
    }
}

/// Factory trait for dynamically instantiating objects from Params.
pub trait ParamsFactory: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn create(&self, params: &Params) -> Result<Arc<dyn DynValue>>;
}

/// Params that can instantiate a concrete object via a factory.
#[derive(Clone)]
pub struct InstantiableParams {
    params: Params,
    factory: Option<Arc<dyn ParamsFactory>>,
}

impl InstantiableParams {
    /// Creates a new InstantiableParams with optional factory.
    pub fn new(factory: Option<Arc<dyn ParamsFactory>>) -> Self {
        let mut params = Params::new();
        let _ = params.define("cls", ParamValue::None, "Class associated with these params.");
        Self { params, factory }
    }

    /// Returns a mutable reference to the underlying Params.
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    /// Returns a reference to the underlying Params.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Instantiates the object using the stored factory.
    pub fn instantiate(&self) -> Result<Arc<dyn DynValue>> {
        let factory = self.factory.as_ref().ok_or_else(|| MonolithError::ConfigError {
            message: "InstantiableParams has no factory".to_string(),
        })?;
        factory.create(&self.params)
    }
}

fn parse_list_index(part: &str) -> Result<(&str, Option<usize>)> {
    if let Some(start) = part.find('[') {
        let end = part
            .find(']')
            .ok_or_else(|| MonolithError::ConfigError {
                message: format!("Invalid list index in {}", part),
            })?;
        let base = &part[..start];
        let idx: usize = part[start + 1..end].parse().map_err(|_| MonolithError::ConfigError {
            message: format!("Invalid list index in {}", part),
        })?;
        Ok((base, Some(idx)))
    } else {
        Ok((part, None))
    }
}
