//! Feature list definitions and parsing utilities (Python parity).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

const BOOL_FLAGS: [&str; 5] = ["true", "yes", "t", "y", "1"];
const FID_MASK: u64 = (1_u64 << 64) - 1;

static FEATURE_LIST_CACHE: Lazy<RwLock<HashMap<String, Arc<FeatureList>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static VALID_FEATURES: Lazy<RwLock<HashSet<String>>> = Lazy::new(|| RwLock::new(HashSet::new()));
static USED_FEATURE_NAMES: Lazy<RwLock<HashMap<String, i32>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static NAME_TO_SLOT: Lazy<RwLock<HashMap<String, i32>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TOB_ENV: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));
static DATA_TYPE: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));
static FEATURE_LIST_PATH: Lazy<RwLock<Option<String>>> = Lazy::new(|| RwLock::new(None));

/// Enable TOB env naming for slots.
pub fn enable_tob_env() {
    *TOB_ENV.write().unwrap_or_else(|e| e.into_inner()) = true;
}

/// Sets the current data type (used by `is_example_batch`).
pub fn set_data_type(value: impl Into<String>) {
    *DATA_TYPE.write().unwrap_or_else(|e| e.into_inner()) = Some(value.into());
}

/// Sets the default feature list path used by `FeatureList::parse_default`.
pub fn set_feature_list_path(path: impl Into<String>) {
    *FEATURE_LIST_PATH
        .write()
        .unwrap_or_else(|e| e.into_inner()) = Some(path.into());
}

fn get_feature_list_path() -> Option<String> {
    if let Ok(val) = std::env::var("MONOLITH_FEATURE_LIST") {
        return Some(val);
    }
    FEATURE_LIST_PATH
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Returns a feature name for a slot ID.
pub fn get_slot_feature_name(slot: i32) -> String {
    let tob = *TOB_ENV.read().unwrap_or_else(|e| e.into_inner());
    if tob {
        format!("fc_slot_{}", slot)
    } else {
        format!("slot_{}", slot)
    }
}

/// Returns slot id from feature name or assigns a new one.
pub fn get_slot_from_feature_name(feature_name: &str) -> Option<i32> {
    let name_to_slot = NAME_TO_SLOT.read().unwrap_or_else(|e| e.into_inner());
    if let Some(slot) = name_to_slot.get(feature_name) {
        return Some(*slot);
    }

    if feature_name.starts_with("slot_") || feature_name.starts_with("fc_slot_") {
        return feature_name
            .split('_')
            .last()
            .and_then(|s| s.parse::<i32>().ok());
    }

    drop(name_to_slot);
    let mut used = USED_FEATURE_NAMES.write().unwrap_or_else(|e| e.into_inner());
    if let Some(slot) = used.get(feature_name) {
        return Some(*slot);
    }
    let next = used.len() as i32 + 1;
    used.insert(feature_name.to_string(), next);
    Some(next)
}

/// Input type for registering slots.
pub trait RegisterSlotsInput {
    /// Register slots into global mapping.
    fn register(self);
}

impl RegisterSlotsInput for &[i32] {
    fn register(self) {
        let mut map = NAME_TO_SLOT.write().unwrap_or_else(|e| e.into_inner());
        for slot in self {
            map.insert(get_slot_feature_name(*slot), *slot);
        }
    }
}

impl RegisterSlotsInput for Vec<i32> {
    fn register(self) {
        self.as_slice().register();
    }
}

impl RegisterSlotsInput for &HashMap<String, i32> {
    fn register(self) {
        let mut map = NAME_TO_SLOT.write().unwrap_or_else(|e| e.into_inner());
        for (name, slot) in self {
            map.insert(name.clone(), *slot);
        }
    }
}

impl RegisterSlotsInput for HashMap<String, i32> {
    fn register(self) {
        (&self).register();
    }
}

/// Register slots by list or mapping.
pub fn register_slots<T: RegisterSlotsInput>(sparse_features: T) {
    sparse_features.register();
}

#[derive(Debug, Clone, Default)]
pub struct Feed {
    pub feed_name: Option<String>,
    pub shared: bool,
    pub feature_id: Option<i64>,
}

impl Feed {
    fn from_params(params: &HashMap<String, String>) -> Self {
        let mut feed = Feed::default();
        feed.feed_name = params
            .get("feed_name")
            .cloned()
            .or_else(|| params.get("feed").cloned());
        if let Some(shared) = params.get("shared") {
            feed.shared = BOOL_FLAGS.contains(&shared.to_lowercase().as_str());
        }
        if let Some(feature_id) = params.get("feature_id") {
            feed.feature_id = feature_id.parse::<i64>().ok();
        }
        feed
    }

    pub fn name(&self) -> Option<&str> {
        self.feed_name.as_deref()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Cache {
    pub cache_column: Option<String>,
    pub cache_name: Option<String>,
    pub capacity: Option<i64>,
    pub timeout: Option<i64>,
    pub cache_type: Option<String>,
    pub cache_key_class: Option<String>,
}

impl Cache {
    fn from_params(params: &HashMap<String, String>) -> Self {
        Cache {
            cache_column: params.get("cache_column").cloned(),
            cache_name: params.get("cache_name").cloned(),
            capacity: params.get("capacity").and_then(|v| v.parse::<i64>().ok()),
            timeout: params.get("timeout").and_then(|v| v.parse::<i64>().ok()),
            cache_type: params.get("cache_type").cloned(),
            cache_key_class: params.get("cache_key_class").cloned(),
        }
    }

    pub fn name(&self) -> Result<String, String> {
        if let Some(name) = &self.cache_name {
            Ok(name.clone())
        } else if let Some(name) = &self.cache_key_class {
            Ok(name.clone())
        } else if let Some(_) = &self.cache_column {
            Ok("cache_column".to_string())
        } else {
            Err("no name for cache".to_string())
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Feature {
    pub feature_name: Option<String>,
    pub depend: Vec<String>,
    pub method: Option<String>,
    pub slot: Option<i32>,
    pub args: Vec<String>,
    pub feature_version: Option<i32>,
    pub shared: bool,
    pub cache_keys: Vec<String>,
    pub need_raw: bool,
    pub feature_id: Option<i64>,
    pub input_optional: Vec<bool>,
    pub feature_group: Vec<String>,
}

impl Feature {
    fn from_params(params: &HashMap<String, String>) -> Self {
        let mut feature = Feature::default();
        feature.feature_name = params
            .get("feature_name")
            .cloned()
            .or_else(|| params.get("feature").cloned());
        if let Some(dep) = params.get("depend") {
            feature.depend = split_list(dep);
        }
        feature.method = params.get("method").cloned();
        feature.slot = params.get("slot").and_then(|v| v.parse::<i32>().ok());
        if let Some(args) = params.get("args") {
            feature.args = split_list(args);
        }
        feature.feature_version = params
            .get("feature_version")
            .and_then(|v| v.parse::<i32>().ok());
        if let Some(shared) = params.get("shared") {
            feature.shared = BOOL_FLAGS.contains(&shared.to_lowercase().as_str());
        }
        if let Some(keys) = params.get("cache_keys") {
            feature.cache_keys = split_list(keys);
        }
        if let Some(need_raw) = params.get("need_raw") {
            feature.need_raw = BOOL_FLAGS.contains(&need_raw.to_lowercase().as_str());
        }
        feature.feature_id = params.get("feature_id").and_then(|v| v.parse::<i64>().ok());
        if let Some(opts) = params.get("input_optional") {
            feature.input_optional = split_list(opts)
                .into_iter()
                .map(|v| BOOL_FLAGS.contains(&v.to_lowercase().as_str()))
                .collect();
        }
        if let Some(groups) = params.get("feature_group") {
            feature.feature_group = split_list(groups);
        }
        feature
    }

    pub fn name(&self) -> Option<String> {
        let feature_name = self.feature_name.as_ref()?;
        let mut term_list = Vec::new();
        for term in feature_name.split('-') {
            let mut t = term.to_string();
            if t.starts_with("fc_") {
                t = t.trim_start_matches("fc_").to_string();
            } else if feature_name.starts_with("f_") {
                t = t.trim_start_matches("f_").to_string();
            }
            term_list.push(t);
        }
        Some(term_list.join("-").to_lowercase())
    }

    pub fn depend_strip_prefix(&self) -> Vec<String> {
        let mut out = Vec::new();
        for dep in &self.depend {
            let mut term_list = Vec::new();
            for term in dep.split('-') {
                let mut t = term.to_string();
                if t.starts_with("fc_") {
                    t = t.trim_start_matches("fc_").to_string();
                } else if dep.starts_with("f_") {
                    t = t.trim_start_matches("f_").to_string();
                }
                term_list.push(t);
            }
            out.push(term_list.join("-").to_lowercase());
        }
        out
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        if let Some(name) = &self.feature_name {
            terms.push(format!("feature_name={}", name));
        }
        if !self.depend.is_empty() {
            terms.push(format!("depend={}", self.depend.join(",")));
        }
        if let Some(method) = &self.method {
            terms.push(format!("method={}", method));
        }
        if let Some(slot) = self.slot {
            terms.push(format!("slot={}", slot));
        }
        if !self.args.is_empty() {
            terms.push(format!("args={}", self.args.join(",")));
        }
        if let Some(ver) = self.feature_version {
            terms.push(format!("feature_version={}", ver));
        }
        if self.shared {
            terms.push("shared=true".to_string());
        }
        if !self.cache_keys.is_empty() {
            terms.push(format!("cache_keys={}", self.cache_keys.join(",")));
        }
        if self.need_raw {
            terms.push("need_raw=true".to_string());
        }
        if let Some(fid) = self.feature_id {
            terms.push(format!("feature_id={}", fid));
        }
        if !self.input_optional.is_empty() {
            let vals = self
                .input_optional
                .iter()
                .map(|v| v.to_string().to_lowercase())
                .collect::<Vec<_>>()
                .join(",");
            terms.push(format!("input_optional={}", vals));
        }
        if !self.feature_group.is_empty() {
            terms.push(format!("feature_group={}", self.feature_group.join(",")));
        }
        write!(f, "{}", terms.join(";"))
    }
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|item| item.trim().trim_matches('"').trim_matches('\''))
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}

#[derive(Debug, Clone)]
pub struct FeatureList {
    pub column_name: Option<HashSet<String>>,
    pub feeds: HashMap<String, Feed>,
    pub caches: HashMap<String, Cache>,
    pub features: HashMap<String, Feature>,
    slots: HashMap<i32, Vec<Feature>>,
}

impl FeatureList {
    fn new(
        column_name: Option<HashSet<String>>,
        feeds: HashMap<String, Feed>,
        caches: HashMap<String, Cache>,
        features: HashMap<String, Feature>,
    ) -> Self {
        let mut slots = HashMap::new();
        for feature in features.values() {
            if let Some(slot) = feature.slot {
                slots.entry(slot).or_insert_with(Vec::new).push(feature.clone());
            }
        }
        FeatureList {
            column_name,
            feeds,
            caches,
            features,
            slots,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Feature> {
        self.get_internal(key)
    }

    pub fn get_with_slot(&self, slot: i32) -> Vec<Feature> {
        self.slots.get(&slot).cloned().unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.features.contains_key(key)
            || self.features.contains_key(&format!("f_{}", key))
            || self.features.contains_key(&format!("fc_{}", key))
            || key.parse::<i32>().map(|slot| self.slots.contains_key(&slot)).unwrap_or(false)
    }

    fn get_internal(&self, item: &str) -> Option<&Feature> {
        if let Ok(slot) = item.parse::<i32>() {
            return self.slots.get(&slot).and_then(|v| v.first());
        }
        let key = item.trim();
        if let Some(feature) = self.features.get(key) {
            return Some(feature);
        }
        let f_key = format!("f_{}", key);
        if let Some(feature) = self.features.get(&f_key) {
            return Some(feature);
        }
        let fc_key = format!("fc_{}", key);
        if let Some(feature) = self.features.get(&fc_key) {
            return Some(feature);
        }
        if key.contains('-') {
            let fc_item = key
                .split('-')
                .map(|t| format!("fc_{}", t))
                .collect::<Vec<_>>()
                .join("-");
            if let Some(feature) = self.features.get(&fc_item) {
                return Some(feature);
            }
            let f_item = key
                .split('-')
                .map(|t| format!("f_{}", t))
                .collect::<Vec<_>>()
                .join("-");
            if let Some(feature) = self.features.get(&f_item) {
                return Some(feature);
            }
        }
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = &Feature> {
        self.features.values()
    }

    /// Parses a feature list file and returns a cached FeatureList.
    pub fn parse<P: AsRef<Path>>(path: P, use_old_name: bool) -> Result<Arc<FeatureList>, String> {
        let fname = path.as_ref().to_string_lossy().to_string();
        if fname.is_empty() {
            return FeatureList::parse_default(use_old_name);
        }
        FeatureList::parse_from_path(&fname, use_old_name)
    }

    /// Parses using the default path (set via `set_feature_list_path` or env).
    pub fn parse_default(use_old_name: bool) -> Result<Arc<FeatureList>, String> {
        let path = get_feature_list_path().ok_or_else(|| {
            "feature list path not set (set_feature_list_path or MONOLITH_FEATURE_LIST)".to_string()
        })?;
        FeatureList::parse_from_path(&path, use_old_name)
    }

    fn parse_from_path(path: &str, use_old_name: bool) -> Result<Arc<FeatureList>, String> {
        let fname = path.to_string();
        {
            let cache = FEATURE_LIST_CACHE.read().unwrap_or_else(|e| e.into_inner());
            if let Some(list) = cache.get(&fname) {
                return Ok(Arc::clone(list));
            }
        }
        let content = fs::read_to_string(&fname).map_err(|e| e.to_string())?;
        let mut column_name: Option<HashSet<String>> = None;
        let mut feeds = HashMap::new();
        let mut caches = HashMap::new();
        let mut features = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("column_name") {
                let cols_part = line
                    .split_once(':')
                    .map(|(_, v)| v)
                    .unwrap_or("")
                    .trim();
                let cols = cols_part
                    .split(',')
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                    .collect::<HashSet<_>>();
                column_name = Some(cols);
                continue;
            }
            if line.starts_with("cache_column") {
                let mut params = HashMap::new();
                let value = line
                    .split_once(':')
                    .map(|(_, v)| v)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                params.insert("cache_column".to_string(), value);
                let cache = Cache::from_params(&params);
                let name = cache.name().map_err(|e| e.to_string())?;
                caches.insert(name, cache);
                continue;
            }

            let params = parse_params(line)?;
            if line.starts_with("feed") {
                let feed = Feed::from_params(&params);
                if let Some(name) = feed.name() {
                    feeds.insert(name.to_string(), feed);
                }
            } else if line.starts_with("cache") {
                let cache = Cache::from_params(&params);
                let name = cache.name().map_err(|e| e.to_string())?;
                caches.insert(name, cache);
            } else {
                let feature = Feature::from_params(&params);
                let name = if use_old_name {
                    feature.feature_name.clone()
                } else {
                    feature.name()
                };
                if let Some(name) = name {
                    features.insert(name, feature);
                }
            }
        }

        let list = Arc::new(FeatureList::new(column_name, feeds, caches, features));
        let mut cache = FEATURE_LIST_CACHE.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(fname, Arc::clone(&list));
        Ok(list)
    }
}

fn parse_params(line: &str) -> Result<HashMap<String, String>, String> {
    let mut params = HashMap::new();
    let items: Vec<&str> = line.split('=').collect();
    if items.len() < 2 {
        return Ok(params);
    }
    for i in 0..items.len() - 1 {
        let key = if i == 0 {
            items[i].trim().to_string()
        } else {
            let start = items[i]
                .rfind(' ')
                .map(|idx| idx + 1)
                .unwrap_or(0);
            items[i][start..].trim().to_string()
        };
        let value = if i == items.len() - 2 {
            items[i + 1].to_string()
        } else {
            let end = items[i + 1]
                .rfind(' ')
                .unwrap_or(items[i + 1].len());
            items[i + 1][0..end].to_string()
        };
        let value = value
            .trim()
            .trim_end_matches(',')
            .trim_end_matches(';')
            .to_string();
        params.insert(key, value);
    }
    Ok(params)
}

/// Returns feature name and slot for input.
/// Input type for resolving feature name and slot.
pub trait FeatureNameOrSlot {
    /// Resolve into (feature_name, slot).
    fn resolve(self) -> (String, Option<i32>);
}

impl FeatureNameOrSlot for i32 {
    fn resolve(self) -> (String, Option<i32>) {
        if let Ok(list) = FeatureList::parse_default(true) {
            if let Some(feature) = list.get_internal(&self.to_string()) {
                return (
                    feature
                        .feature_name
                        .clone()
                        .unwrap_or_else(|| self.to_string()),
                    Some(self),
                );
            }
        }
        (get_slot_feature_name(self), Some(self))
    }
}

impl FeatureNameOrSlot for &str {
    fn resolve(self) -> (String, Option<i32>) {
        if let Ok(list) = FeatureList::parse_default(true) {
            if list.contains(self) {
                if let Some(feature) = list.get_internal(self) {
                    return (self.to_string(), feature.slot);
                }
            }
        }
        (self.to_string(), get_slot_from_feature_name(self))
    }
}

impl FeatureNameOrSlot for String {
    fn resolve(self) -> (String, Option<i32>) {
        self.as_str().resolve()
    }
}

/// Returns feature name and slot for input.
pub fn get_feature_name_and_slot<T: FeatureNameOrSlot>(item: T) -> (String, Option<i32>) {
    item.resolve()
}

/// Returns whether the data type indicates example_batch.
pub fn is_example_batch() -> bool {
    let dt = DATA_TYPE.read().unwrap_or_else(|e| e.into_inner());
    if let Some(dt) = dt.as_ref() {
        let dt = dt.to_lowercase();
        return dt == "example_batch" || dt == "examplebatch";
    }
    false
}

/// Adds one or more feature names to the valid list.
/// Input type for adding feature names.
pub trait AddFeatureInput {
    /// Convert into feature names.
    fn into_feature_names(self) -> Vec<String>;
}

impl AddFeatureInput for i32 {
    fn into_feature_names(self) -> Vec<String> {
        vec![get_slot_feature_name(self)]
    }
}

impl AddFeatureInput for u64 {
    fn into_feature_names(self) -> Vec<String> {
        if self <= i32::MAX as u64 {
            vec![get_slot_feature_name(self as i32)]
        } else {
            vec![self.to_string()]
        }
    }
}

impl AddFeatureInput for &str {
    fn into_feature_names(self) -> Vec<String> {
        vec![self.to_string()]
    }
}

impl AddFeatureInput for String {
    fn into_feature_names(self) -> Vec<String> {
        vec![self]
    }
}

impl<T: AddFeatureInput> AddFeatureInput for Vec<T> {
    fn into_feature_names(self) -> Vec<String> {
        let mut out = Vec::new();
        for item in self {
            out.extend(item.into_feature_names());
        }
        out
    }
}

impl<T: AddFeatureInput + Clone> AddFeatureInput for &[T] {
    fn into_feature_names(self) -> Vec<String> {
        let mut out = Vec::new();
        for item in self {
            out.extend(item.clone().into_feature_names());
        }
        out
    }
}

/// Adds one or more feature names to the valid list.
pub fn add_feature<T: AddFeatureInput>(feature: T) {
    let mut valid = VALID_FEATURES.write().unwrap_or_else(|e| e.into_inner());
    for name in feature.into_feature_names() {
        valid.insert(name);
    }
}

/// Adds features by fid list, using a feature list if available.
pub fn add_feature_by_fids(fids: &[u64], feature_list: Option<Arc<FeatureList>>) -> Result<(), String> {
    if !is_example_batch() {
        return Ok(());
    }
    let list = if let Some(list) = feature_list {
        list
    } else {
        FeatureList::parse("", true)?
    };

    for fid in fids {
        let fid = fid & FID_MASK;
        let mut found = false;
        let slot_v1 = (fid >> 54) as i32;
        for feature in list.get_with_slot(slot_v1) {
            if feature.feature_version.is_none() || feature.feature_version == Some(1) {
                if let Some(name) = feature.feature_name.clone() {
                    add_feature(name);
                    found = true;
                }
            }
        }
        let slot_v2 = (fid >> 48) as i32;
        for feature in list.get_with_slot(slot_v2) {
            if feature.feature_version == Some(2) {
                if let Some(name) = feature.feature_name.clone() {
                    add_feature(name);
                    found = true;
                }
            }
        }
        if !found {
            return Err(format!("Cannot find feature name for fid: {}", fid));
        }
    }
    Ok(())
}

/// Returns the valid features collected so far.
pub fn get_valid_features() -> Vec<String> {
    VALID_FEATURES
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .cloned()
        .collect()
}
