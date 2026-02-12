//! Python `monolith.native_training.file_ops` parity.
//!
//! The upstream Python code wraps TensorFlow custom ops to write to files from
//! inside a TF graph. In the Rust port we provide a direct I/O implementation
//! that matches the observable side effects of the Python tests:
//! - `WritableFile.append()` appends raw bytes to a file, creating parent dirs.
//! - `WritableFile.append_entry_dump()` writes TFRecord-framed `EntryDump` protos.
//! - `FileCloseHook` closes files in the training hook `end()` callback.

use monolith_data::tfrecord::TFRecordWriter;
use monolith_proto::monolith::hash_table::EntryDump;
use prost::Message;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct Inner {
    file: Option<File>,
}

/// A writable file handle mirroring Python's TF custom-op wrapper.
#[derive(Debug, Clone)]
pub struct WritableFile {
    path: PathBuf,
    inner: Arc<Mutex<Inner>>,
}

impl WritableFile {
    /// Create (truncate) a file and prepare it for appends.
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;

        Ok(Self {
            path,
            inner: Arc::new(Mutex::new(Inner { file: Some(file) })),
        })
    }

    /// Append raw bytes to the file.
    pub fn append(&self, content: impl AsRef<[u8]>) -> io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .expect("writable file mutex should not be poisoned during append");
        let file = guard
            .file
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "file is closed"))?;
        file.write_all(content.as_ref())
    }

    /// Append entry dumps as TFRecord records.
    ///
    /// This mirrors the TF custom op `MonolithEntryDumpFileAppend`, which writes
    /// one `monolith.hash_table.EntryDump` proto per batch element as a TFRecord.
    pub fn append_entry_dump(
        &self,
        item_id: &[i64],
        bias: &[f32],
        embedding: &[f32],
    ) -> io::Result<()> {
        if item_id.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "item_id must be non-empty",
            ));
        }
        if bias.len() != item_id.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "bias must have the same length as item_id",
            ));
        }
        if embedding.len() % item_id.len() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "embedding length must be divisible by batch size",
            ));
        }
        let emb_len = embedding.len() / item_id.len();

        let mut guard = self
            .inner
            .lock()
            .expect("writable file mutex should not be poisoned during entry dump append");
        let file = guard
            .file
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "file is closed"))?;

        // Use the existing TFRecord writer implementation from monolith-data for byte-for-byte
        // compatibility with TF's RecordWriter framing.
        let mut writer = TFRecordWriter::new(&mut *file);
        for (batch_id, id) in item_id.iter().copied().enumerate() {
            let mut dump = EntryDump::default();
            dump.id = Some(id);
            dump.num.push(bias[batch_id]);
            let base = batch_id * emb_len;
            dump.num.extend_from_slice(&embedding[base..base + emb_len]);
            writer
                .write_record(&dump.encode_to_vec())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        }
        Ok(())
    }

    /// Close the underlying file.
    pub fn close(&self) -> io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .expect("writable file mutex should not be poisoned during close");
        if let Some(mut file) = guard.file.take() {
            file.flush()?;
            // Best-effort sync for durability (not required by Python, but helpful for tests).
            let _ = file.sync_all();
        }
        Ok(())
    }

    /// Path of the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WritableFile {
    fn drop(&mut self) {
        // Mirror TF resource behavior: close on drop if not explicitly closed.
        let _ = self.close();
    }
}

/// A training hook that closes a list of files at the end of training.
#[derive(Debug, Clone)]
pub struct FileCloseHook {
    files: Vec<WritableFile>,
}

impl FileCloseHook {
    pub fn new(files: Vec<WritableFile>) -> Self {
        Self { files }
    }
}

impl crate::hooks::Hook for FileCloseHook {
    fn name(&self) -> &str {
        "file_close_hook"
    }

    fn end(
        &mut self,
        _step: u64,
        _metrics: Option<&crate::metrics::Metrics>,
    ) -> crate::hooks::HookResult<()> {
        for f in &self.files {
            f.close()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::Hook;

    #[test]
    fn writable_file_basic() {
        let tmp = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = tmp.path().join("test_basic").join("test_name");

        let f = WritableFile::new(&path).expect("writable file creation should succeed");
        for _ in 0..1000 {
            f.append("1234")
                .expect("appending raw content should succeed");
        }
        f.close().expect("explicit file close should succeed");

        let content = std::fs::read_to_string(&path)
            .expect("reading written file content should succeed");
        assert_eq!(content, "1234".repeat(1000));
    }

    #[test]
    fn file_close_hook_runs() {
        let tmp = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = tmp.path().join("test_hook").join("test_name");

        let f = WritableFile::new(&path).expect("writable file creation should succeed");
        f.append("1234")
            .expect("appending raw content should succeed");

        let mut hook = FileCloseHook::new(vec![f.clone()]);
        hook.end(0, None)
            .expect("file close hook end callback should succeed");

        let content = std::fs::read_to_string(&path)
            .expect("reading written file content should succeed");
        assert_eq!(content, "1234");
    }
}
