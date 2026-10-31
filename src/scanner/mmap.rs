//! Zero-copy memory-mapped file loader.
//!
//! Uses `memmap2::Mmap` to map a file directly into virtual memory.
//! The OS page cache serves data on demand — the CPU never eagerly
//! copies file contents into application RAM.

use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// A memory-mapped file ready for GPU consumption.
///
/// Holds both the `File` handle (to keep the fd alive) and
/// the `Mmap` mapping. The mapped bytes are accessible via
/// [`MappedFile::as_bytes`].
pub struct MappedFile {
    // The Mmap borrows from the kernel mapping, not from `_file`,
    // but we keep the handle alive so the fd isn't closed.
    _file: File,
    mmap: Mmap,
}

impl MappedFile {
    /// Opens and memory-maps the file at the given `path`.
    ///
    /// # Errors
    /// Returns `io::Error` if the file cannot be opened or mapped.
    ///
    /// # Safety Note
    /// `memmap2::Mmap` is inherently unsafe because another process
    /// could modify the file while we hold the mapping. We accept
    /// this risk in exchange for zero-copy performance — this is a
    /// read-only search tool operating on static datasets.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: We treat the file as read-only and immutable for
        // the lifetime of this search. External modification would
        // produce undefined behavior, which is an accepted trade-off
        // for zero-copy performance on static datasets.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { _file: file, mmap })
    }

    /// Returns the mapped file contents as a byte slice.
    ///
    /// This slice is backed by virtual memory — accessing it
    /// triggers page faults that the OS resolves from disk/cache.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Returns the total size of the mapped file in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns `true` if the mapped file is empty (0 bytes).
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}
