//! Local model copy manager (Python parity for `monolith/agent_service/model_manager.py`).
//!
//! The Python agent copies "ready" models from a source directory (e.g. a P2P download
//! location) into a receive directory that TFServing loads from.
//!
//! Readiness is signaled by a `"{model}@{version}.write.done"` file at the source root,
//! and the model contents are under `"{model}@{version}/{model}/.../{version}"`.
//!
//! This module ports the behavior needed by parity tests in
//! `monolith/agent_service/model_manager_test.py`, including:
//! - wait-for-ready behavior and retry loop in `start()`
//! - version selection and "ignore old versions"
//! - temp directory copy + atomic rename into place
//! - read-lock files under the source directory

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn is_dir(p: &Path) -> bool {
    fs::metadata(p).map(|m| m.is_dir()).unwrap_or(false)
}

fn is_file(p: &Path) -> bool {
    fs::metadata(p).map(|m| m.is_file()).unwrap_or(false)
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    // Recursive copy without extra dependencies.
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&from, &to)?;
        } else if file_type.is_file() {
            if let Some(parent) = to.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&from, &to)?;
        } else {
            // Symlinks/special files are not expected in parity tests; skip them.
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct SourceModelData {
    version: String,
    version_data: Vec<(String, PathBuf)>, // (sub_model_name/version, absolute path)
    #[allow(dead_code)]
    real_path: PathBuf,
}

#[derive(Debug)]
struct ModelManagerInner {
    model_name: Option<String>,
    source_path: PathBuf,
    receive_path: PathBuf,

    // model_name -> [(version, [dst_paths...])]
    models: Mutex<HashMap<String, Vec<(String, Vec<PathBuf>)>>>,
    latest_models: Mutex<HashMap<String, (String, i64)>>,

    /// Max duration to wait for the source path and a `.write.done` file (seconds).
    wait_timeout_secs: AtomicU64,
    /// Background loop interval (seconds).
    loop_interval_secs: AtomicU64,
    /// How many versions to keep under receive_path.
    remain_version_num: AtomicUsize,

    lock_files: Mutex<HashSet<PathBuf>>,
    exit: AtomicBool,
    loop_thread: Mutex<Option<JoinHandle<()>>>,

    #[allow(dead_code)]
    use_metrics: bool,
}

/// Copy/refresh models from a source directory to a receive directory.
///
/// This is intentionally blocking/thread-based, matching the Python implementation.
#[derive(Debug, Clone)]
pub struct ModelManager {
    inner: Arc<ModelManagerInner>,
}

impl ModelManager {
    /// Marker suffix for a fully-written model version (Python parity).
    pub const WRITE_DONE: &'static str = ".write.done";
    /// Lock file suffix used while reading a model version (Python parity).
    pub const READ_LOCK: &'static str = ".read.lock";

    /// Create a model manager for a single model (or no-op when `model_name` is None/empty).
    pub fn new(
        model_name: Option<String>,
        source_path: impl Into<PathBuf>,
        receive_path: impl Into<PathBuf>,
        use_metrics: bool,
    ) -> Self {
        Self {
            inner: Arc::new(ModelManagerInner {
                model_name,
                source_path: source_path.into(),
                receive_path: receive_path.into(),
                models: Mutex::new(HashMap::new()),
                latest_models: Mutex::new(HashMap::new()),
                wait_timeout_secs: AtomicU64::new(1200),
                loop_interval_secs: AtomicU64::new(30),
                remain_version_num: AtomicUsize::new(5),
                lock_files: Mutex::new(HashSet::new()),
                exit: AtomicBool::new(false),
                loop_thread: Mutex::new(None),
                use_metrics,
            }),
        }
    }

    /// Start the blocking initialization and spawn the background copy loop.
    pub fn start(&self) -> bool {
        // Python catches generic Exception; Rust doesn't treat panics as recoverable error
        // semantics here, so just execute and let panics crash tests (fail-fast).
        self.start_inner()
    }

    fn start_inner(&self) -> bool {
        let inner = &self.inner;
        if inner.model_name.as_deref().unwrap_or_default().is_empty() {
            info!("ModelManager is not needed");
            return true;
        }

        // Delete receive path first.
        if !self.delete(&inner.receive_path) {
            return false;
        }

        // Wait for the source path and a ready marker.
        if !self.wait_for_download() {
            return false;
        }

        // Do loop-once until copy succeeds (Python retries every 10s).
        loop {
            match self.loop_once() {
                true => break,
                false => {
                    info!("loop once failed, wait for ready model");
                    thread::sleep(Duration::from_secs(10));
                }
            }
        }

        self.remove_read_lock();

        // Spawn the background loop.
        let mut guard = inner.loop_thread.lock();
        if guard.is_none() {
            let this = self.clone();
            *guard = Some(
                thread::Builder::new()
                    .name("thread-model_manager".to_string())
                    .spawn(move || this.run_loop())
                    .expect("spawn model manager loop"),
            );
        }
        true
    }

    /// Stop the background copy loop and join its thread.
    pub fn stop(&self) {
        self.inner.exit.store(true, Ordering::Relaxed);
        if let Some(h) = self.inner.loop_thread.lock().take() {
            let _ = h.join();
        }
    }

    /// Configure how long to wait for a ready marker on startup.
    pub fn set_wait_timeout(&self, d: Duration) {
        self.inner
            .wait_timeout_secs
            .store(d.as_secs(), Ordering::Relaxed);
    }

    /// Configure how often the background loop scans for new versions.
    pub fn set_loop_interval(&self, d: Duration) {
        self.inner
            .loop_interval_secs
            .store(d.as_secs(), Ordering::Relaxed);
    }

    /// Configure how many versions to keep under `receive_path`.
    pub fn set_remain_version_num(&self, n: usize) {
        self.inner.remain_version_num.store(n, Ordering::Relaxed);
    }

    fn run_loop(self) {
        let inner = self.inner.clone();
        while !inner.exit.load(Ordering::Relaxed) {
            let ret = self.loop_once();
            self.remove_read_lock();
            if !ret {
                error!("model manager loop once failed");
            }

            let interval = Duration::from_secs(inner.loop_interval_secs.load(Ordering::Relaxed));
            thread::sleep(interval);
            self.remove_old_file();
        }
    }

    fn touch(&self, file: &Path) -> bool {
        match fs::File::create(file) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    fn create_read_lock(&self, name: &Path) -> PathBuf {
        let lock_name = PathBuf::from(format!("{}{}", name.display(), Self::READ_LOCK));
        if !self.touch(&lock_name) {
            error!("create lock {} failed", lock_name.display());
        }
        lock_name
    }

    fn remove_read_lock(&self) {
        // Remove locks we created during the last scan.
        let mut locks = self.inner.lock_files.lock();
        for lock in locks.iter() {
            let _ = self.delete(lock);
        }
        locks.clear();
        drop(locks);

        // Remove other lock files at the source root.
        let Ok(mut it) = fs::read_dir(&self.inner.source_path) else {
            return;
        };
        while let Some(Ok(ent)) = it.next() {
            let p = ent.path();
            if p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.ends_with(Self::READ_LOCK))
                .unwrap_or(false)
            {
                // Python bug uses `os.join`; in Rust just remove the file.
                info!("delete lock file: {}", p.display());
                let _ = self.delete(&p);
            }
        }
    }

    /// Perform one scan/copy iteration.
    ///
    /// Returns true when the scan completed successfully (even if there were no updates).
    pub fn loop_once(&self) -> bool {
        let source_data = match self.get_source_data() {
            Ok(v) => v,
            Err(e) => {
                error!("get download data failed: {}", e);
                return false;
            }
        };

        let mut result = true;
        for (model_name, smd) in source_data {
            let new_version = smd.version.clone();

            // Compare versions as strings (Python parity).
            let old_version = {
                let models = self.inner.models.lock();
                models
                    .get(&model_name)
                    .and_then(|l| l.last().map(|(v, _)| v.clone()))
            };
            if let Some(old) = old_version {
                if old >= new_version {
                    continue;
                }
            }

            let (ok, file_list) = self.copy_model(&model_name, &new_version, &smd.version_data);
            if ok {
                {
                    let mut models = self.inner.models.lock();
                    models
                        .entry(model_name.clone())
                        .or_default()
                        .push((new_version.clone(), file_list.clone()));
                }
                {
                    let mut latest = self.inner.latest_models.lock();
                    latest.insert(model_name.clone(), (new_version.clone(), now_ts()));
                }
                info!("{} update to {}", model_name, new_version);
            } else {
                error!("copy {} failed", model_name);
                result = false;
            }
        }

        result
    }

    fn copy_model(
        &self,
        model_name: &str,
        _version: &str,
        model_data: &[(String, PathBuf)],
    ) -> (bool, Vec<PathBuf>) {
        let sub_model_num = model_data.len();
        let mut ready_data: Vec<(PathBuf, PathBuf)> = Vec::new();
        let mut result: Vec<PathBuf> = Vec::new();
        let mut ready_num: usize = 0;

        for (sub_model_name, sub_model_data) in model_data.iter() {
            // sub_model_name: ps_0/version
            // sub_model_data: /xxx/model_name@version/model_name/ps_0/version
            let src_file = sub_model_data;
            let dst_file = self
                .inner
                .receive_path
                .join(model_name)
                .join(sub_model_name.as_str());
            let temp_dst_file = PathBuf::from(format!("{}-temp", dst_file.display()));

            result.push(dst_file.clone());

            if is_dir(&dst_file) {
                error!("{} exist", dst_file.display());
                ready_num += 1;
                continue;
            }

            if is_dir(&temp_dst_file) {
                error!("{} exist", temp_dst_file.display());
                ready_num += 1;
                ready_data.push((temp_dst_file, dst_file));
                continue;
            }

            if let Some(parent) = temp_dst_file.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    error!("create dir {} failed: {}", parent.display(), e);
                    let _ = self.delete(&temp_dst_file);
                    break;
                }
            }

            match copy_dir_all(src_file, &temp_dst_file) {
                Ok(_) => {
                    ready_data.push((temp_dst_file, dst_file));
                    ready_num += 1;
                }
                Err(e) => {
                    error!(
                        "copy model {} -> {} failed: {}",
                        src_file.display(),
                        temp_dst_file.display(),
                        e
                    );
                    let _ = self.delete(&temp_dst_file);
                    break;
                }
            }
        }

        if ready_num != sub_model_num {
            error!(
                "copy model failed, ready_num={}, expect_num={}",
                ready_num, sub_model_num
            );
            for (tmp, _) in ready_data {
                let _ = self.delete(&tmp);
            }
            return (false, Vec::new());
        }

        for (tmp, dst) in ready_data {
            if let Err(e) = fs::rename(&tmp, &dst) {
                error!(
                    "rename {} -> {} failed: {}",
                    tmp.display(),
                    dst.display(),
                    e
                );
                let _ = self.delete(&tmp);
                return (false, Vec::new());
            }
        }

        (true, result)
    }

    fn wait_for_download(&self) -> bool {
        let Some(model_name) = self.inner.model_name.as_deref() else {
            return true;
        };

        let mut duration = Duration::from_secs(0);
        let mut download_path_ready = is_dir(&self.inner.source_path);
        let wait_timeout =
            Duration::from_secs(self.inner.wait_timeout_secs.load(Ordering::Relaxed));

        while !download_path_ready && duration < wait_timeout {
            info!("wait {} created", self.inner.source_path.display());
            thread::sleep(Duration::from_secs(10));
            duration += Duration::from_secs(10);
            download_path_ready = is_dir(&self.inner.source_path);
        }

        if !download_path_ready {
            error!("{} is not ready", self.inner.source_path.display());
            return false;
        }

        while duration < wait_timeout {
            let Ok(mut it) = fs::read_dir(&self.inner.source_path) else {
                error!("{} is empty", self.inner.source_path.display());
                return false;
            };
            while let Some(Ok(ent)) = it.next() {
                let p = ent.path();
                if is_file(&p) {
                    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if name.starts_with(model_name) && name.ends_with(Self::WRITE_DONE) {
                        info!("{} is ready", name);
                        return true;
                    }
                }
            }

            info!("no ready model found");
            thread::sleep(Duration::from_secs(10));
            duration += Duration::from_secs(10);
        }

        error!("no ready model found");
        false
    }

    fn get_source_data(&self) -> io::Result<HashMap<String, SourceModelData>> {
        let Some(model_name_filter) = self.inner.model_name.as_deref() else {
            return Ok(HashMap::new());
        };

        let mut source_data: HashMap<String, SourceModelData> = HashMap::new();
        if !is_dir(&self.inner.source_path) {
            warn!("{} is empty", self.inner.source_path.display());
            return Ok(source_data);
        }

        let mut done_file_set: HashSet<String> = HashSet::new();
        let mut dirs: Vec<String> = Vec::new();

        for ent in fs::read_dir(&self.inner.source_path)? {
            let ent = ent?;
            let p = ent.path();
            let name = ent.file_name().to_string_lossy().to_string();
            if is_file(&p)
                && name.starts_with(model_name_filter)
                && name.ends_with(Self::WRITE_DONE)
            {
                done_file_set.insert(name);
            } else if is_dir(&p) {
                dirs.push(name);
            }
        }

        for model_data in dirs {
            // Create a read lock for this download directory.
            let lock_file = self.create_read_lock(&self.inner.source_path.join(&model_data));
            self.inner.lock_files.lock().insert(lock_file);

            if !done_file_set.contains(&self.get_done_file(&model_data)) {
                continue;
            }

            let parts: Vec<&str> = model_data.split('@').collect();
            if parts.len() != 2 {
                error!("{} is not valid", model_data);
                continue;
            }
            let model_name = parts[0].to_string();
            let version = parts[1].to_string();

            // real_path: /xxx/model_name@version/model_name
            let real_path = self.inner.source_path.join(&model_data).join(&model_name);
            let version_data = self.get_version_data(&real_path, &version)?;
            if version_data.is_empty() {
                continue;
            }

            match source_data.get(&model_name) {
                None => {
                    source_data.insert(
                        model_name.clone(),
                        SourceModelData {
                            version,
                            version_data,
                            real_path,
                        },
                    );
                }
                Some(old) => {
                    // Compare versions as strings (Python parity).
                    if old.version < version {
                        source_data.insert(
                            model_name.clone(),
                            SourceModelData {
                                version,
                                version_data,
                                real_path,
                            },
                        );
                    }
                }
            }
        }

        Ok(source_data)
    }

    fn get_version_data(&self, path: &Path, version: &str) -> io::Result<Vec<(String, PathBuf)>> {
        if !is_dir(path) {
            error!("get version data [{}] failed", path.display());
            return Ok(Vec::new());
        }

        let mut sub_dirs: Vec<String> = Vec::new();
        for ent in fs::read_dir(path)? {
            let ent = ent?;
            let p = ent.path();
            if is_dir(&p) {
                sub_dirs.push(ent.file_name().to_string_lossy().to_string());
            }
        }
        if sub_dirs.is_empty() {
            return Ok(Vec::new());
        }

        let mut res = Vec::new();
        for sub_dir in sub_dirs {
            let version_dir = path.join(&sub_dir).join(version);
            if !is_dir(&version_dir) {
                error!("{} not exist", version_dir.display());
                return Ok(Vec::new());
            }
            res.push((format!("{sub_dir}/{version}"), version_dir));
        }
        Ok(res)
    }

    fn get_done_file(&self, dir_name: &str) -> String {
        format!("{dir_name}{}", Self::WRITE_DONE)
    }

    fn delete(&self, path: &Path) -> bool {
        if !path.exists() {
            return true;
        }
        let res = if is_file(path) {
            fs::remove_file(path)
        } else {
            fs::remove_dir_all(path)
        };
        match res {
            Ok(_) => true,
            Err(e) => {
                error!("delete [{}] failed: {}", path.display(), e);
                false
            }
        }
    }

    fn remove_old_file(&self) {
        let mut models = self.inner.models.lock();
        for (_model_name, versions) in models.iter_mut() {
            let keep = self.inner.remain_version_num.load(Ordering::Relaxed);
            while versions.len() > keep {
                let (_ver, files) = versions.remove(0);
                for p in files {
                    let _ = self.delete(&p);
                }
            }
        }
    }
}
