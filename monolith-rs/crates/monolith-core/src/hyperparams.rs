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
    fn python_str_value(&self) -> String {
        match self {
            ParamValue::String(v) => format!("\"{}\"", v),
            ParamValue::Params(p) => p.to_string(),
            _ => self.python_repr(),
        }
    }

    /// Python-like `repr()` used for values inside lists/dicts and for non-string
    /// leaf values when formatting params.
    fn python_repr(&self) -> String {
        match self {
            ParamValue::None => "None".to_string(),
            ParamValue::Bool(v) => {
                if *v {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            ParamValue::Int(v) => v.to_string(),
            ParamValue::Float(v) => {
                if v.is_nan() {
                    "nan".to_string()
                } else if *v == f64::INFINITY {
                    "inf".to_string()
                } else if *v == f64::NEG_INFINITY {
                    "-inf".to_string()
                } else if v.is_finite() && v.fract() == 0.0 {
                    // Python prints `1.0` rather than `1` for floats.
                    format!("{:.1}", v)
                } else {
                    v.to_string()
                }
            }
            ParamValue::String(v) => python_single_quoted_string(v),
            ParamValue::List(v) => format!(
                "[{}]",
                v.iter()
                    .map(|x| x.python_repr())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ParamValue::Map(v) => {
                let items = v
                    .iter()
                    .map(|(k, val)| {
                        format!("{}: {}", python_single_quoted_string(k), val.python_repr())
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{{{}}}", items)
            }
            // When a Params appears in a list/dict, Python renders it as a sorted dict,
            // not as the multi-line Params string.
            ParamValue::Params(p) => p.python_dict_repr(),
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

    pub fn as_external(&self) -> Option<&Arc<dyn DynValue>> {
        match self {
            ParamValue::External(v) => Some(v),
            _ => None,
        }
    }
}

impl fmt::Debug for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.python_repr())
    }
}

impl PartialEq for ParamValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ParamValue::None, ParamValue::None) => true,
            (ParamValue::Bool(a), ParamValue::Bool(b)) => a == b,
            (ParamValue::Int(a), ParamValue::Int(b)) => a == b,
            (ParamValue::Float(a), ParamValue::Float(b)) => a.to_bits() == b.to_bits(),
            (ParamValue::String(a), ParamValue::String(b)) => a == b,
            (ParamValue::List(a), ParamValue::List(b)) => a == b,
            (ParamValue::Map(a), ParamValue::Map(b)) => a == b,
            (ParamValue::Params(a), ParamValue::Params(b)) => a == b,
            // Python `object()` comparison is identity-based; mirror by pointer equality.
            (ParamValue::External(a), ParamValue::External(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
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

    fn to_string(&self, nested_depth: usize) -> String {
        let nested_indent = "  ".repeat(nested_depth);
        let value_str = match &self.value {
            ParamValue::Params(p) => p.to_string_depth(nested_depth),
            _ => self.value.python_str_value(),
        };
        format!("{}{}: {}", nested_indent, self.name, value_str)
    }
}

impl PartialEq for Param {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.value == other.value
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
            return Err(MonolithError::PyTypeError {
                message: "This Params instance is immutable.".to_string(),
            });
        }
        if !PARAM_NAME_RE.is_match(name) {
            return Err(MonolithError::PyAssertionError {
                message: String::new(),
            });
        }
        if self.params.contains_key(name) {
            return Err(MonolithError::PyAttributeError {
                message: format!("Parameter {} is already defined", name),
            });
        }
        self.params.insert(
            name.to_string(),
            Param::new(name, default_value.into(), description),
        );
        Ok(())
    }

    /// Returns true if the parameter is defined.
    pub fn contains(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Returns number of defined params.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Freezes the Params to make it immutable.
    pub fn freeze(&mut self) {
        self.immutable = true;
    }

    /// Returns whether the Params is immutable.
    pub fn is_immutable(&self) -> bool {
        self.immutable
    }

    /// Returns the list of defined param keys in sorted order.
    pub fn keys(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }

    /// Returns a deep copy of this Params.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Sets a parameter using dotted path notation.
    pub fn set(&mut self, name: &str, value: impl Into<ParamValue>) -> Result<()> {
        if self.immutable {
            return Err(MonolithError::PyTypeError {
                message: format!("This Params instance is immutable: {}", self),
            });
        }
        let (parent, key) = self.get_nested_params_mut(name)?;
        match parent.params.get_mut(&key) {
            Some(param) => {
                param.value = value.into();
                Ok(())
            }
            None => Err(MonolithError::PyAttributeError {
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
            .ok_or_else(|| MonolithError::PyAttributeError {
                message: parent.key_error_string(name),
            })
    }

    /// Deletes a parameter using dotted path notation.
    pub fn delete(&mut self, name: &str) -> Result<()> {
        if self.immutable {
            return Err(MonolithError::PyTypeError {
                message: "This Params instance is immutable.".to_string(),
            });
        }
        let (parent, key) = self.get_nested_params_mut(name)?;
        if parent.params.remove(&key).is_none() {
            return Err(MonolithError::PyAttributeError {
                message: parent.key_error_string(name),
            });
        }
        Ok(())
    }

    /// Returns a reference to a nested Params value.
    pub fn get_params(&self, name: &str) -> Result<&Params> {
        let v = self.get(name)?;
        match v {
            ParamValue::Params(p) => Ok(p.as_ref()),
            _ => Err(MonolithError::ConfigError {
                message: format!("Expected Params for {}", name),
            }),
        }
    }

    /// Returns a mutable reference to a nested Params value.
    pub fn get_params_mut(&mut self, name: &str) -> Result<&mut Params> {
        let (parent, key) = self.get_nested_params_mut(name)?;
        let err_msg = parent.key_error_string(name);
        let param = parent
            .params
            .get_mut(&key)
            .ok_or_else(|| MonolithError::ConfigError { message: err_msg })?;
        match &mut param.value {
            ParamValue::Params(p) => Ok(p.as_mut()),
            _ => Err(MonolithError::ConfigError {
                message: format!("Expected Params for {}", name),
            }),
        }
    }

    /// Returns an iterator over parameters (name, value).
    pub fn iter_params(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.params.iter().map(|(k, v)| (k.as_str(), &v.value))
    }

    fn to_string_depth(&self, nested_depth: usize) -> String {
        let sorted_param_strs = self
            .params
            .values()
            .map(|p| p.to_string(nested_depth + 1))
            .collect::<Vec<_>>();
        let nested_indent = "  ".repeat(nested_depth);
        format!("{{\n{}\n{}}}", sorted_param_strs.join("\n"), nested_indent)
    }

    fn python_dict_repr(&self) -> String {
        let items = self
            .params
            .iter()
            .map(|(k, p)| {
                format!(
                    "{}: {}",
                    python_single_quoted_string(k),
                    p.value.python_repr()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{{}}}", items)
    }

    fn get_nested_params<'a>(&'a self, name: &str) -> Result<(&'a Params, String)> {
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = self;
        for (i, part) in parts[..parts.len().saturating_sub(1)].iter().enumerate() {
            let (base, index) = parse_list_index(part)?;
            let param = current
                .params
                .get(base)
                .ok_or_else(|| MonolithError::PyAttributeError {
                    message: base.to_string(),
                })?;
            current = match (&param.value, index) {
                (ParamValue::Params(p), None) => p.as_ref(),
                (ParamValue::List(list), Some(idx)) => match list.get(idx) {
                    Some(ParamValue::Params(p)) => p.as_ref(),
                    _ => {
                        return Err(MonolithError::PyAttributeError {
                            message: parts[..=i].join("."),
                        });
                    }
                },
                (ParamValue::List(_), None) => {
                    return Err(MonolithError::PyAttributeError {
                        message: parts[..=i].join("."),
                    });
                }
                _ => {
                    return Err(MonolithError::PyAssertionError {
                        message: format!(
                            "Cannot introspect {} for {}",
                            param_value_type_name(&param.value),
                            parts[..=i].join(".")
                        ),
                    });
                }
            };
        }
        let key = parts.last().unwrap_or(&name).to_string();
        Ok((current, key))
    }

    fn get_nested_params_mut<'a>(&'a mut self, name: &str) -> Result<(&'a mut Params, String)> {
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = self as *mut Params;
        for (i, part) in parts[..parts.len().saturating_sub(1)].iter().enumerate() {
            let (base, index) = parse_list_index(part)?;
            // SAFETY: we only create one mutable reference at a time during traversal.
            let curr = unsafe { &mut *current };
            let param = curr
                .params
                .get_mut(base)
                .ok_or_else(|| MonolithError::PyAttributeError {
                    message: parts[..=i].join("."),
                })?;
            let next = match (&mut param.value, index) {
                (ParamValue::Params(p), None) => p.as_mut(),
                (ParamValue::List(list), Some(idx)) => match list.get_mut(idx) {
                    Some(ParamValue::Params(p)) => p.as_mut(),
                    _ => {
                        return Err(MonolithError::PyAttributeError {
                            message: parts[..=i].join("."),
                        });
                    }
                },
                (ParamValue::List(_), None) => {
                    return Err(MonolithError::PyAttributeError {
                        message: parts[..=i].join("."),
                    });
                }
                _ => {
                    return Err(MonolithError::PyAssertionError {
                        message: format!(
                            "Cannot introspect {} for {}",
                            param_value_type_name(&param.value),
                            parts[..=i].join(".")
                        ),
                    });
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
            // Mirror Python: `for i in range(len(name) - 3): ... name[i:i+3]`.
            for i in 0..name.len().saturating_sub(3) {
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
        f.write_str(&self.to_string_depth(0))
    }
}

impl PartialEq for Params {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params
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
        let _ = params.define(
            "cls",
            ParamValue::None,
            "Class associated with these params.",
        );
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
        let factory = self
            .factory
            .as_ref()
            .ok_or_else(|| MonolithError::ConfigError {
                message: "InstantiableParams has no factory".to_string(),
            })?;
        factory.create(&self.params)
    }
}

fn parse_list_index(part: &str) -> Result<(&str, Option<usize>)> {
    if let Some(start) = part.find('[') {
        let end = part.find(']').ok_or_else(|| MonolithError::ConfigError {
            message: format!("Invalid list index in {}", part),
        })?;
        let base = &part[..start];
        let idx: usize = part[start + 1..end]
            .parse()
            .map_err(|_| MonolithError::ConfigError {
                message: format!("Invalid list index in {}", part),
            })?;
        Ok((base, Some(idx)))
    } else {
        Ok((part, None))
    }
}

fn python_single_quoted_string(value: &str) -> String {
    // Minimal escaping consistent with Python's repr() for simple strings.
    let escaped = value.replace('\\', "\\\\").replace('\'', "\\'");
    format!("'{}'", escaped)
}

fn param_value_type_name(value: &ParamValue) -> &'static str {
    match value {
        ParamValue::None => "NoneType",
        ParamValue::Bool(_) => "bool",
        ParamValue::Int(_) => "int",
        ParamValue::Float(_) => "float",
        ParamValue::String(_) => "str",
        ParamValue::List(_) => "list",
        ParamValue::Map(_) => "dict",
        ParamValue::Params(_) => "Params",
        ParamValue::External(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Debug)]
    struct TestObj(&'static str);

    struct TestEnumB;

    impl fmt::Debug for TestEnumB {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            // Mirror Python's `str(enum.EnumMember)` output in `test_to_string`.
            f.write_str("TestEnum.B")
        }
    }

    #[test]
    fn test_equals() {
        let mut params1 = Params::new();
        let mut params2 = Params::new();
        assert_eq!(params1, params2);

        params1.define("first", "firstvalue", "").unwrap();
        assert_ne!(params1, params2);

        params2.define("first", "firstvalue", "").unwrap();
        assert_eq!(params1, params2);

        let some_object = ParamValue::external(TestObj("x"));
        let other_object = ParamValue::external(TestObj("y"));
        params1.define("second", some_object.clone(), "").unwrap();
        params2.define("second", other_object, "").unwrap();
        assert_ne!(params1, params2);

        params2.set("second", some_object.clone()).unwrap();
        assert_eq!(params1, params2);

        params1.define("third", Params::new(), "").unwrap();
        params2.define("third", Params::new(), "").unwrap();
        assert_eq!(params1, params2);

        params1
            .get_params_mut("third")
            .unwrap()
            .define("fourth", "x", "")
            .unwrap();
        params2
            .get_params_mut("third")
            .unwrap()
            .define("fourth", "y", "")
            .unwrap();
        assert_ne!(params1, params2);

        params2.set("third.fourth", "x").unwrap();
        assert_eq!(params1, params2);
    }

    #[test]
    fn test_deep_copy() {
        let shared_tensor = ParamValue::external(TestObj("tensor"));

        let mut inner = Params::new();
        inner.define("alpha", 2_i64, "").unwrap();
        inner.define("tensor", shared_tensor.clone(), "").unwrap();

        let mut outer = Params::new();
        outer.define("beta", 1_i64, "").unwrap();
        outer.define("inner", inner, "").unwrap();

        let outer_copy = outer.copy();
        assert_ne!((&outer as *const _), (&outer_copy as *const _));
        assert_eq!(outer, outer_copy);

        let inner_ptr = outer.get_params("inner").unwrap() as *const _;
        let inner_copy_ptr = outer_copy.get_params("inner").unwrap() as *const _;
        assert_ne!(inner_ptr, inner_copy_ptr);

        let tensor_a = outer.get("inner.tensor").unwrap().as_external().unwrap();
        let tensor_b = outer_copy
            .get("inner.tensor")
            .unwrap()
            .as_external()
            .unwrap();
        assert!(Arc::ptr_eq(tensor_a, tensor_b));
    }

    #[test]
    fn test_copy_params_to() {
        let mut source = Params::new();
        let mut dest = Params::new();
        source.define("a", "a", "").unwrap();
        source.define("b", "b", "").unwrap();
        source.define("c", "c", "").unwrap();
        dest.define("a", "", "").unwrap();

        copy_params_to(&source, &mut dest, Some(&["b", "c"]));
        assert_eq!(source.get("a").unwrap(), dest.get("a").unwrap());
        assert!(!dest.contains("b"));
        assert!(!dest.contains("c"));
    }

    #[test]
    fn test_define_existing() {
        let mut p = Params::new();
        p.define("foo", 1_i64, "").unwrap();
        let err = p.define("foo", 1_i64, "").unwrap_err();
        assert!(err.to_string().contains("already defined"));
    }

    #[test]
    fn test_legal_param_names() {
        let mut p = Params::new();
        assert!(p.define("", 1_i64, "").is_err());
        assert!(p.define("_foo", 1_i64, "").is_err());
        assert!(p.define("Foo", 1_i64, "").is_err());
        assert!(p.define("1foo", 1_i64, "").is_err());
        assert!(p.define("foo$", 1_i64, "").is_err());
        p.define("foo_bar", 1_i64, "").unwrap();
        p.define("foo9", 1_i64, "").unwrap();
    }

    #[test]
    fn test_set_and_get() {
        let mut p = Params::new();
        assert!(p.set("foo", 4_i64).unwrap_err().to_string().contains("foo"));

        p.define("foo", 1_i64, "").unwrap();
        assert_eq!(p.get("foo").unwrap(), &ParamValue::Int(1));
        assert!(p.contains("foo"));
        assert!(!p.contains("bar"));

        p.set("foo", 2_i64).unwrap();
        assert_eq!(p.get("foo").unwrap(), &ParamValue::Int(2));

        p.set("foo", 3_i64).unwrap();
        assert_eq!(p.get("foo").unwrap(), &ParamValue::Int(3));

        p.delete("foo").unwrap();
        assert!(!p.contains("foo"));
        assert!(p.get("foo").is_err());
    }

    #[test]
    fn test_set_and_get_nested_param() {
        let mut innermost = Params::new();
        innermost.define("delta", 22_i64, "").unwrap();
        innermost.define("zeta", 5_i64, "").unwrap();

        let mut inner = Params::new();
        inner.define("alpha", 2_i64, "").unwrap();
        inner.define("innermost", innermost, "").unwrap();

        let mut outer = Params::new();
        outer.define("beta", 1_i64, "").unwrap();
        outer.define("inner", inner, "").unwrap();
        let mut d = BTreeMap::new();
        d.insert("foo".to_string(), ParamValue::from("bar"));
        outer.define("d", ParamValue::Map(d), "").unwrap();

        assert_eq!(outer.get("inner.alpha").unwrap(), &ParamValue::Int(2));
        assert_eq!(
            outer.get("inner.innermost.delta").unwrap(),
            &ParamValue::Int(22)
        );
        assert_eq!(
            outer.get("inner.innermost.zeta").unwrap(),
            &ParamValue::Int(5)
        );

        outer.set("inner.alpha", 3_i64).unwrap();
        let mut d2 = BTreeMap::new();
        d2.insert("foo".to_string(), ParamValue::from("baq"));
        outer.set("d", ParamValue::Map(d2)).unwrap();
        outer.delete("beta").unwrap();
        outer.delete("inner.innermost.zeta").unwrap();

        assert_eq!(outer.get("inner.alpha").unwrap(), &ParamValue::Int(3));
        assert!(outer.get("beta").is_err());
        assert!(outer.get("inner.innermost.zeta").is_err());

        assert!(outer
            .set("inner.gamma", 5_i64)
            .unwrap_err()
            .to_string()
            .contains("inner.gamma"));
        assert!(outer
            .set("inner.innermost.bad", 5_i64)
            .unwrap_err()
            .to_string()
            .contains("inner.innermost.bad"));
        assert!(outer
            .set("d.foo", "baz")
            .unwrap_err()
            .to_string()
            .contains("Cannot introspect"));
    }

    #[test]
    fn test_freeze() {
        let mut p = Params::new();
        assert!(p.set("foo", 4_i64).is_err());
        p.define("foo", 1_i64, "").unwrap();
        p.define("nested", p.copy(), "").unwrap();
        assert_eq!(p.get("foo").unwrap(), &ParamValue::Int(1));
        assert_eq!(p.get("nested.foo").unwrap(), &ParamValue::Int(1));

        p.freeze();
        assert!(p
            .set("foo", 2_i64)
            .unwrap_err()
            .to_string()
            .contains("immutable"));
        assert!(p
            .delete("foo")
            .unwrap_err()
            .to_string()
            .contains("immutable"));
        assert!(p
            .define("bar", 1_i64, "")
            .unwrap_err()
            .to_string()
            .contains("immutable"));
        assert!(p.get("bar").is_err());

        // Nested params remain mutable.
        p.get_params_mut("nested")
            .unwrap()
            .set("foo", 2_i64)
            .unwrap();
        assert_eq!(p.get("foo").unwrap(), &ParamValue::Int(1));
        assert_eq!(p.get("nested.foo").unwrap(), &ParamValue::Int(2));

        let q = p.copy();
        assert!(q.is_immutable());
    }

    #[test]
    fn test_to_string() {
        let mut outer = Params::new();
        outer.define("foo", 1_i64, "").unwrap();
        let mut inner = Params::new();
        inner.define("bar", 2_i64, "").unwrap();
        outer.define("inner", inner.clone(), "").unwrap();
        outer
            .define(
                "list",
                ParamValue::List(vec![1_i64.into(), inner.clone().into(), 2_i64.into()]),
                "",
            )
            .unwrap();
        let mut dict = BTreeMap::new();
        dict.insert("a".to_string(), 1_i64.into());
        dict.insert("b".to_string(), ParamValue::from(inner));
        outer.define("dict", ParamValue::Map(dict), "").unwrap();
        outer
            .define("enum", ParamValue::external(TestEnumB), "")
            .unwrap();

        let expected = r#"
{
  dict: {'a': 1, 'b': {'bar': 2}}
  enum: TestEnum.B
  foo: 1
  inner: {
    bar: 2
  }
  list: [1, {'bar': 2}, 2]
}"#;
        assert_eq!(format!("\n{}", outer), expected);
    }

    #[test]
    fn test_iter_params() {
        let keys = ["a", "b", "c", "d", "e"];
        let values = [
            ParamValue::Bool(true),
            ParamValue::None,
            ParamValue::from("zippidy"),
            ParamValue::Float(78.5),
            ParamValue::Int(5),
        ];

        let mut p = Params::new();
        for (k, v) in keys.iter().zip(values.iter().cloned()) {
            p.define(k, v, "").unwrap();
        }
        let mut count = 0;
        for (k, _v) in p.iter_params() {
            assert!(keys.contains(&k));
            count += 1;
        }
        assert_eq!(count, keys.len());
    }

    #[test]
    fn test_similar_keys() {
        let mut p = Params::new();
        p.define("activation", "RELU", "").unwrap();
        p.define("activations", "RELU", "").unwrap();
        p.define("cheesecake", ParamValue::None, "").unwrap();
        p.define("tofu", ParamValue::None, "").unwrap();

        let err = p.set("actuvation", 1_i64).unwrap_err();
        assert!(err
            .to_string()
            .contains("actuvation (did you mean: [activation,activations])"));
    }
}
