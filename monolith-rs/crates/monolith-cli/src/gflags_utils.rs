//! Python `monolith.native_training.gflags_utils` parity helpers.
//!
//! The upstream Python implementation dynamically extracts help from dataclass docstrings and
//! defines absl flags for annotated fields. In Rust we keep the observable logic used by the
//! Python tests:
//! - Parse `:param name: ...` blocks from a doc string into a help map.
//! - Update a config struct field from a "flags" source *only if* the field is still at its
//!   default value and the flag value is different from its default.
//! - Support linking one flag to multiple fields and inheritance-like merging of links.
//!
//! We do not attempt to replicate absl's global flag registry. Instead tests use a local
//! `FlagRegistry` with explicit default values.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Init,
    Open,
    Extend,
    Closed,
}

fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract `:param <name>: <help>` from a doc string, joining continued lines.
pub fn extract_help_info(doc: &str) -> HashMap<String, String> {
    let mut help_info: HashMap<String, Vec<String>> = HashMap::new();
    let mut key_stack: Vec<String> = Vec::new();

    let lines: Vec<String> = doc
        .lines()
        .map(|l| normalize_ws(l.trim()))
        .filter(|l| !l.is_empty())
        .collect();

    let mut status = Status::Init;

    for (i, line) in lines.iter().enumerate() {
        if let Some(rest) = line.strip_prefix(":param") {
            // `:param name: blah`
            let rest = rest.trim_start();
            let Some((key, info)) = rest.split_once(':') else {
                continue;
            };
            let key = key.trim().to_string();
            let info = info.trim().to_string();

            match status {
                Status::Init => {
                    help_info.insert(key.clone(), vec![info]);
                    key_stack.push(key);
                    status = Status::Open;
                }
                Status::Open | Status::Extend => {
                    let old_key = key_stack.pop().expect("key stack");
                    assert_ne!(old_key, key);
                    help_info.insert(key.clone(), vec![info]);
                    key_stack.push(key);
                    status = Status::Open;
                }
                Status::Closed => break,
            }
        } else {
            match status {
                Status::Init => {}
                Status::Open | Status::Extend => {
                    let key = key_stack.last().expect("key stack");
                    help_info
                        .get_mut(key)
                        .expect("help entry")
                        .push(line.clone());
                    status = Status::Extend;
                }
                Status::Closed => break,
            }
        }

        if i + 1 == lines.len() {
            status = Status::Closed;
        }
    }

    assert_eq!(status, Status::Closed);
    help_info
        .into_iter()
        .map(|(k, v)| (k, v.join(" ")))
        .collect()
}

/// A tiny flag registry used by tests.
#[derive(Debug, Clone, Default)]
pub struct FlagRegistry {
    // name -> (current, default)
    strings: HashMap<String, (String, String)>,
    ints: HashMap<String, (i64, i64)>,
}

impl FlagRegistry {
    pub fn define_string(&mut self, name: &str, default: &str) {
        self.strings
            .entry(name.to_string())
            .or_insert_with(|| (default.to_string(), default.to_string()));
    }

    pub fn define_int(&mut self, name: &str, default: i64) {
        self.ints
            .entry(name.to_string())
            .or_insert_with(|| (default, default));
    }

    pub fn set_string(&mut self, name: &str, value: &str) {
        if let Some((cur, _)) = self.strings.get_mut(name) {
            *cur = value.to_string();
        } else {
            self.define_string(name, "");
            self.set_string(name, value);
        }
    }

    pub fn set_int(&mut self, name: &str, value: i64) {
        if let Some((cur, _)) = self.ints.get_mut(name) {
            *cur = value;
        } else {
            self.define_int(name, 0);
            self.set_int(name, value);
        }
    }

    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.strings.get(name).map(|(cur, _)| cur.as_str())
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.ints.get(name).map(|(cur, _)| *cur)
    }

    pub fn default_string(&self, name: &str) -> Option<&str> {
        self.strings.get(name).map(|(_, def)| def.as_str())
    }

    pub fn default_int(&self, name: &str) -> Option<i64> {
        self.ints.get(name).map(|(_, def)| *def)
    }
}

/// Update helper mirroring Python `gflags_utils.update(config)` at the field level.
///
/// In the Python code the comparison is against the *config field default*:
/// - If config field equals its default *and* flag value != that default, update.
pub fn update_if_default<T: PartialEq + Clone>(
    field_value: &mut T,
    field_default: &T,
    flag_value: &T,
) {
    if field_value == field_default && flag_value != field_default {
        *field_value = flag_value.clone();
    }
}

/// Metadata for linking fields to flags (Rust replacement for `_MonolithGflagMeta`).
#[derive(Debug, Clone, Default)]
pub struct GflagMeta {
    pub linked_map: HashMap<String, String>,
}

impl GflagMeta {
    pub fn merged_from_mro(mros: &[&GflagMeta]) -> Self {
        let mut out = GflagMeta::default();
        for m in mros {
            out.linked_map.extend(m.linked_map.clone());
        }
        out
    }
}

/// Links one or more fields to flags.
#[derive(Debug, Clone, Default)]
pub struct LinkDataclassToFlags {
    linked_map: HashMap<String, String>,
}

impl LinkDataclassToFlags {
    pub fn new(linked_list: &[&str], linked_map: &[(&str, &str)]) -> Self {
        let mut m = HashMap::new();
        for &name in linked_list {
            m.insert(name.to_string(), name.to_string());
        }
        for (field, flag) in linked_map {
            m.insert((*field).to_string(), (*flag).to_string());
        }
        Self { linked_map: m }
    }

    /// Validates and produces metadata for a struct.
    pub fn build_meta(
        &self,
        struct_fields: &HashSet<&'static str>,
        flags: &FlagRegistry,
    ) -> Result<GflagMeta, String> {
        for (field, flag) in &self.linked_map {
            if !struct_fields.contains(field.as_str()) {
                return Err(format!("{field} is not a valid attribute"));
            }
            let ok = flags.strings.contains_key(flag) || flags.ints.contains_key(flag);
            if !ok {
                return Err(format!("{flag} is not defined in flags"));
            }
        }
        Ok(GflagMeta {
            linked_map: self.linked_map.clone(),
        })
    }
}
