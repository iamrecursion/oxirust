//! Microsoft Structured Storage (Compound File) implementation
//!
//! This module implements the Microsoft Compound File Binary Format used by AAF.
//! It provides:
//! - Directory entry parsing
//! - FAT (File Allocation Table) chain traversal
//! - `MiniFAT` for small streams
//! - Red-black tree directory structure
//! - Storage and stream navigation
//! - Little-endian data handling

use crate::{AafError, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Seek, SeekFrom, Write};

/// Sector size for regular sectors (512 bytes)
const SECTOR_SIZE: usize = 512;

/// Signature for compound files
const SIGNATURE: &[u8; 8] = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1";

/// Maximum regular sector shift (512 bytes = 2^9)
const MAX_REG_SECT_SHIFT: u16 = 9;

/// Mini sector shift (64 bytes = 2^6)
const MINI_SECT_SHIFT: u16 = 6;

/// Mini sector size
const MINI_SECTOR_SIZE: usize = 64;

/// End of chain marker
const END_OF_CHAIN: u32 = 0xFFFFFFFE;

/// Free sector marker
const FREE_SECTOR: u32 = 0xFFFFFFFF;

/// FAT sector marker
const FAT_SECTOR: u32 = 0xFFFFFFFD;

/// DIFAT sector marker
const DIFAT_SECTOR: u32 = 0xFFFFFFFC;

/// Directory entry size
const DIR_ENTRY_SIZE: usize = 128;

/// Maximum number of DIFAT entries in header
const HEADER_DIFAT_ENTRIES: usize = 109;

/// Object type in directory entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    /// Unallocated/unknown
    Unknown,
    /// Storage object (directory)
    Storage,
    /// Stream object (file)
    Stream,
    /// Root storage entry
    Root,
}

impl ObjectType {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => ObjectType::Unknown,
            1 => ObjectType::Storage,
            2 => ObjectType::Stream,
            5 => ObjectType::Root,
            _ => ObjectType::Unknown,
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            ObjectType::Unknown => 0,
            ObjectType::Storage => 1,
            ObjectType::Stream => 2,
            ObjectType::Root => 5,
        }
    }
}

/// Node color for red-black tree
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeColor {
    Red,
    Black,
}

impl NodeColor {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => NodeColor::Red,
            1 => NodeColor::Black,
            _ => NodeColor::Black,
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            NodeColor::Red => 0,
            NodeColor::Black => 1,
        }
    }
}

/// Directory entry in structured storage
#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    /// Entry name (UTF-16)
    pub name: String,
    /// Object type
    pub object_type: ObjectType,
    /// Node color (red-black tree)
    pub color: NodeColor,
    /// Left sibling directory entry ID
    pub left_sibling_id: u32,
    /// Right sibling directory entry ID
    pub right_sibling_id: u32,
    /// Child directory entry ID
    pub child_id: u32,
    /// CLSID for storage objects
    pub clsid: [u8; 16],
    /// State bits
    pub state_bits: u32,
    /// Creation time
    pub creation_time: u64,
    /// Modified time
    pub modified_time: u64,
    /// Starting sector
    pub starting_sector: u32,
    /// Stream size
    pub size: u64,
}

impl DirectoryEntry {
    /// Parse a directory entry from bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < DIR_ENTRY_SIZE {
            return Err(AafError::InvalidStructuredStorage(
                "Directory entry too small".to_string(),
            ));
        }

        let mut cursor = std::io::Cursor::new(data);

        // Read name (64 bytes, UTF-16LE)
        let mut name_buf = [0u16; 32];
        for item in &mut name_buf {
            *item = cursor.read_u16::<LittleEndian>()?;
        }
        let name_len = cursor.read_u16::<LittleEndian>()? as usize;
        let name =
            String::from_utf16_lossy(&name_buf[..((name_len.saturating_sub(2)) / 2).min(32)]);

        let object_type = ObjectType::from_byte(cursor.read_u8()?);
        let color = NodeColor::from_byte(cursor.read_u8()?);
        let left_sibling_id = cursor.read_u32::<LittleEndian>()?;
        let right_sibling_id = cursor.read_u32::<LittleEndian>()?;
        let child_id = cursor.read_u32::<LittleEndian>()?;

        let mut clsid = [0u8; 16];
        cursor.read_exact(&mut clsid)?;

        let state_bits = cursor.read_u32::<LittleEndian>()?;
        let creation_time = cursor.read_u64::<LittleEndian>()?;
        let modified_time = cursor.read_u64::<LittleEndian>()?;
        let starting_sector = cursor.read_u32::<LittleEndian>()?;
        let size = cursor.read_u64::<LittleEndian>()?;

        Ok(DirectoryEntry {
            name,
            object_type,
            color,
            left_sibling_id,
            right_sibling_id,
            child_id,
            clsid,
            state_bits,
            creation_time,
            modified_time,
            starting_sector,
            size,
        })
    }

    /// Serialize directory entry to bytes
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut data = Vec::with_capacity(DIR_ENTRY_SIZE);

        // Write name (UTF-16LE, max 32 characters)
        let name_utf16: Vec<u16> = self.name.encode_utf16().take(31).collect();
        for &ch in &name_utf16 {
            data.write_u16::<LittleEndian>(ch)?;
        }
        // Pad to 64 bytes
        for _ in name_utf16.len()..32 {
            data.write_u16::<LittleEndian>(0)?;
        }

        // Name length in bytes (including null terminator)
        data.write_u16::<LittleEndian>((name_utf16.len() * 2 + 2) as u16)?;

        data.write_u8(self.object_type.to_byte())?;
        data.write_u8(self.color.to_byte())?;
        data.write_u32::<LittleEndian>(self.left_sibling_id)?;
        data.write_u32::<LittleEndian>(self.right_sibling_id)?;
        data.write_u32::<LittleEndian>(self.child_id)?;
        data.write_all(&self.clsid)?;
        data.write_u32::<LittleEndian>(self.state_bits)?;
        data.write_u64::<LittleEndian>(self.creation_time)?;
        data.write_u64::<LittleEndian>(self.modified_time)?;
        data.write_u32::<LittleEndian>(self.starting_sector)?;
        data.write_u64::<LittleEndian>(self.size)?;

        Ok(data)
    }

    /// Check if this is a stream entry
    #[must_use]
    pub fn is_stream(&self) -> bool {
        self.object_type == ObjectType::Stream
    }

    /// Check if this is a storage entry
    #[must_use]
    pub fn is_storage(&self) -> bool {
        self.object_type == ObjectType::Storage || self.object_type == ObjectType::Root
    }

    /// Check if this entry uses mini stream
    #[must_use]
    pub fn uses_mini_stream(&self) -> bool {
        self.size < 4096 && self.is_stream()
    }
}

/// Compound file header
#[derive(Debug, Clone)]
pub struct Header {
    /// Sector shift (power of 2 for sector size)
    pub sector_shift: u16,
    /// Mini sector shift
    pub mini_sector_shift: u16,
    /// Total sectors
    pub total_sectors: u32,
    /// FAT sectors
    pub fat_sectors: u32,
    /// First directory sector
    pub first_dir_sector: u32,
    /// Transaction signature
    pub transaction_signature: u32,
    /// Mini stream cutoff size
    pub mini_stream_cutoff_size: u32,
    /// First mini FAT sector
    pub first_mini_fat_sector: u32,
    /// Number of mini FAT sectors
    pub mini_fat_sectors: u32,
    /// First DIFAT sector
    pub first_difat_sector: u32,
    /// Number of DIFAT sectors
    pub difat_sectors: u32,
    /// DIFAT array (first 109 entries in header)
    pub difat: [u32; HEADER_DIFAT_ENTRIES],
}

impl Header {
    /// Parse header from reader
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        reader.seek(SeekFrom::Start(0))?;

        // Verify signature
        let mut sig = [0u8; 8];
        reader.read_exact(&mut sig)?;
        if &sig != SIGNATURE {
            return Err(AafError::InvalidStructuredStorage(
                "Invalid signature".to_string(),
            ));
        }

        // Skip CLSID (16 bytes)
        reader.seek(SeekFrom::Current(16))?;

        // Minor version (2 bytes)
        let _minor_version = reader.read_u16::<LittleEndian>()?;

        // Major version (2 bytes) - should be 3 or 4
        let major_version = reader.read_u16::<LittleEndian>()?;
        if major_version != 3 && major_version != 4 {
            return Err(AafError::InvalidStructuredStorage(format!(
                "Unsupported version: {major_version}"
            )));
        }

        // Byte order (2 bytes) - should be 0xFFFE (little-endian)
        let byte_order = reader.read_u16::<LittleEndian>()?;
        if byte_order != 0xFFFE {
            return Err(AafError::InvalidStructuredStorage(
                "Invalid byte order".to_string(),
            ));
        }

        let sector_shift = reader.read_u16::<LittleEndian>()?;
        let mini_sector_shift = reader.read_u16::<LittleEndian>()?;

        // Reserved (6 bytes)
        reader.seek(SeekFrom::Current(6))?;

        let total_sectors = reader.read_u32::<LittleEndian>()?;
        let fat_sectors = reader.read_u32::<LittleEndian>()?;
        let first_dir_sector = reader.read_u32::<LittleEndian>()?;
        let transaction_signature = reader.read_u32::<LittleEndian>()?;
        let mini_stream_cutoff_size = reader.read_u32::<LittleEndian>()?;
        let first_mini_fat_sector = reader.read_u32::<LittleEndian>()?;
        let mini_fat_sectors = reader.read_u32::<LittleEndian>()?;
        let first_difat_sector = reader.read_u32::<LittleEndian>()?;
        let difat_sectors = reader.read_u32::<LittleEndian>()?;

        // Read DIFAT array
        let mut difat = [FREE_SECTOR; HEADER_DIFAT_ENTRIES];
        for item in &mut difat {
            *item = reader.read_u32::<LittleEndian>()?;
        }

        Ok(Header {
            sector_shift,
            mini_sector_shift,
            total_sectors,
            fat_sectors,
            first_dir_sector,
            transaction_signature,
            mini_stream_cutoff_size,
            first_mini_fat_sector,
            mini_fat_sectors,
            first_difat_sector,
            difat_sectors,
            difat,
        })
    }

    /// Get sector size
    #[must_use]
    pub fn sector_size(&self) -> usize {
        1 << self.sector_shift
    }

    /// Get mini sector size
    #[must_use]
    pub fn mini_sector_size(&self) -> usize {
        1 << self.mini_sector_shift
    }

    /// Create a new header with default values
    #[must_use]
    pub fn new() -> Self {
        Self {
            sector_shift: MAX_REG_SECT_SHIFT,
            mini_sector_shift: MINI_SECT_SHIFT,
            total_sectors: 0,
            fat_sectors: 0,
            first_dir_sector: 0,
            transaction_signature: 0,
            mini_stream_cutoff_size: 4096,
            first_mini_fat_sector: END_OF_CHAIN,
            mini_fat_sectors: 0,
            first_difat_sector: END_OF_CHAIN,
            difat_sectors: 0,
            difat: [FREE_SECTOR; HEADER_DIFAT_ENTRIES],
        }
    }

    /// Write header to writer
    pub fn write<W: Write + Seek>(&self, writer: &mut W) -> Result<()> {
        writer.seek(SeekFrom::Start(0))?;

        // Signature
        writer.write_all(SIGNATURE)?;

        // CLSID (zeros)
        writer.write_all(&[0u8; 16])?;

        // Minor version
        writer.write_u16::<LittleEndian>(0x003E)?;

        // Major version (3 for 512-byte sectors)
        writer.write_u16::<LittleEndian>(3)?;

        // Byte order (little-endian)
        writer.write_u16::<LittleEndian>(0xFFFE)?;

        writer.write_u16::<LittleEndian>(self.sector_shift)?;
        writer.write_u16::<LittleEndian>(self.mini_sector_shift)?;

        // Reserved (6 bytes)
        writer.write_all(&[0u8; 6])?;

        writer.write_u32::<LittleEndian>(self.total_sectors)?;
        writer.write_u32::<LittleEndian>(self.fat_sectors)?;
        writer.write_u32::<LittleEndian>(self.first_dir_sector)?;
        writer.write_u32::<LittleEndian>(self.transaction_signature)?;
        writer.write_u32::<LittleEndian>(self.mini_stream_cutoff_size)?;
        writer.write_u32::<LittleEndian>(self.first_mini_fat_sector)?;
        writer.write_u32::<LittleEndian>(self.mini_fat_sectors)?;
        writer.write_u32::<LittleEndian>(self.first_difat_sector)?;
        writer.write_u32::<LittleEndian>(self.difat_sectors)?;

        // Write DIFAT array
        for &entry in &self.difat {
            writer.write_u32::<LittleEndian>(entry)?;
        }

        Ok(())
    }
}

impl Default for Header {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured storage reader
pub struct StorageReader<R: Read + Seek> {
    reader: R,
    header: Header,
    fat: Vec<u32>,
    mini_fat: Vec<u32>,
    directory_entries: Vec<DirectoryEntry>,
    mini_stream_data: Vec<u8>,
}

impl<R: Read + Seek> StorageReader<R> {
    /// Create a new storage reader
    pub fn new(mut reader: R) -> Result<Self> {
        let header = Header::parse(&mut reader)?;
        let fat = Self::read_fat(&mut reader, &header)?;
        let directory_entries = Self::read_directory(&mut reader, &header, &fat)?;

        // Read mini stream from root entry
        let mini_stream_data = if directory_entries.is_empty() {
            Vec::new()
        } else {
            let root = &directory_entries[0];
            Self::read_stream_data(&mut reader, &header, &fat, root)?
        };

        let mini_fat = Self::read_mini_fat(&mut reader, &header, &fat)?;

        Ok(Self {
            reader,
            header,
            fat,
            mini_fat,
            directory_entries,
            mini_stream_data,
        })
    }

    /// Read FAT (File Allocation Table)
    fn read_fat<T: Read + Seek>(reader: &mut T, header: &Header) -> Result<Vec<u32>> {
        let sector_size = header.sector_size();
        let entries_per_sector = sector_size / 4;
        let mut fat = Vec::new();

        // Collect all FAT sector numbers from DIFAT
        let mut fat_sectors = Vec::new();
        for &sector in &header.difat {
            if sector != FREE_SECTOR {
                fat_sectors.push(sector);
            }
        }

        // Read additional DIFAT sectors if needed
        let mut difat_sector = header.first_difat_sector;
        while difat_sector != END_OF_CHAIN && difat_sector != FREE_SECTOR {
            let offset = 512 + (u64::from(difat_sector) * sector_size as u64);
            reader.seek(SeekFrom::Start(offset))?;

            for _ in 0..entries_per_sector - 1 {
                let sector = reader.read_u32::<LittleEndian>()?;
                if sector != FREE_SECTOR {
                    fat_sectors.push(sector);
                }
            }

            difat_sector = reader.read_u32::<LittleEndian>()?;
        }

        // Read FAT sectors
        for &sector in &fat_sectors {
            let offset = 512 + (u64::from(sector) * sector_size as u64);
            reader.seek(SeekFrom::Start(offset))?;

            for _ in 0..entries_per_sector {
                fat.push(reader.read_u32::<LittleEndian>()?);
            }
        }

        Ok(fat)
    }

    /// Read Mini FAT
    fn read_mini_fat<T: Read + Seek>(
        reader: &mut T,
        header: &Header,
        fat: &[u32],
    ) -> Result<Vec<u32>> {
        if header.first_mini_fat_sector == END_OF_CHAIN
            || header.first_mini_fat_sector == FREE_SECTOR
        {
            return Ok(Vec::new());
        }

        let sector_size = header.sector_size();
        let entries_per_sector = sector_size / 4;
        let mut mini_fat = Vec::new();

        let mut sector = header.first_mini_fat_sector;
        while sector != END_OF_CHAIN && sector != FREE_SECTOR {
            let offset = 512 + (u64::from(sector) * sector_size as u64);
            reader.seek(SeekFrom::Start(offset))?;

            for _ in 0..entries_per_sector {
                mini_fat.push(reader.read_u32::<LittleEndian>()?);
            }

            sector = fat.get(sector as usize).copied().unwrap_or(END_OF_CHAIN);
        }

        Ok(mini_fat)
    }

    /// Read directory entries
    fn read_directory<T: Read + Seek>(
        reader: &mut T,
        header: &Header,
        fat: &[u32],
    ) -> Result<Vec<DirectoryEntry>> {
        let sector_size = header.sector_size();
        let entries_per_sector = sector_size / DIR_ENTRY_SIZE;
        let mut entries = Vec::new();

        let mut sector = header.first_dir_sector;
        while sector != END_OF_CHAIN && sector != FREE_SECTOR {
            let offset = 512 + (u64::from(sector) * sector_size as u64);
            reader.seek(SeekFrom::Start(offset))?;

            for _ in 0..entries_per_sector {
                let mut entry_data = vec![0u8; DIR_ENTRY_SIZE];
                reader.read_exact(&mut entry_data)?;
                let entry = DirectoryEntry::parse(&entry_data)?;
                entries.push(entry);
            }

            sector = fat.get(sector as usize).copied().unwrap_or(END_OF_CHAIN);
        }

        Ok(entries)
    }

    /// Read stream data
    fn read_stream_data<T: Read + Seek>(
        reader: &mut T,
        header: &Header,
        fat: &[u32],
        entry: &DirectoryEntry,
    ) -> Result<Vec<u8>> {
        let sector_size = header.sector_size();
        let mut data = Vec::new();

        let mut sector = entry.starting_sector;
        let mut remaining = entry.size as usize;

        while sector != END_OF_CHAIN && sector != FREE_SECTOR && remaining > 0 {
            let offset = 512 + (u64::from(sector) * sector_size as u64);
            reader.seek(SeekFrom::Start(offset))?;

            let to_read = remaining.min(sector_size);
            let mut buffer = vec![0u8; to_read];
            reader.read_exact(&mut buffer)?;
            data.extend_from_slice(&buffer);

            remaining -= to_read;
            sector = fat.get(sector as usize).copied().unwrap_or(END_OF_CHAIN);
        }

        Ok(data)
    }

    /// Get header
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Get directory entries
    pub fn directory_entries(&self) -> &[DirectoryEntry] {
        &self.directory_entries
    }

    /// Find directory entry by path
    pub fn find_entry(&self, path: &str) -> Option<&DirectoryEntry> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        if parts.is_empty() {
            return self.directory_entries.first();
        }

        let mut current_id = 0u32;
        for part in parts {
            let children = self.get_children(current_id);
            let found = children.iter().find(|&&id| {
                self.directory_entries
                    .get(id as usize)
                    .is_some_and(|e| e.name == part)
            });

            if let Some(&id) = found {
                current_id = id;
            } else {
                return None;
            }
        }

        self.directory_entries.get(current_id as usize)
    }

    /// Get children of a directory entry
    fn get_children(&self, parent_id: u32) -> Vec<u32> {
        let parent = match self.directory_entries.get(parent_id as usize) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let mut children = Vec::new();
        if parent.child_id != 0xFFFFFFFF {
            self.collect_children(parent.child_id, &mut children);
        }

        children
    }

    /// Recursively collect children using red-black tree
    fn collect_children(&self, entry_id: u32, children: &mut Vec<u32>) {
        if entry_id == 0xFFFFFFFF {
            return;
        }

        let entry = match self.directory_entries.get(entry_id as usize) {
            Some(e) => e,
            None => return,
        };

        if entry.left_sibling_id != 0xFFFFFFFF {
            self.collect_children(entry.left_sibling_id, children);
        }

        children.push(entry_id);

        if entry.right_sibling_id != 0xFFFFFFFF {
            self.collect_children(entry.right_sibling_id, children);
        }
    }

    /// Read stream contents
    pub fn read_stream(&mut self, entry: &DirectoryEntry) -> Result<Vec<u8>> {
        if entry.uses_mini_stream() {
            self.read_mini_stream(entry)
        } else {
            Self::read_stream_data(&mut self.reader, &self.header, &self.fat, entry)
        }
    }

    /// Read mini stream contents
    fn read_mini_stream(&self, entry: &DirectoryEntry) -> Result<Vec<u8>> {
        let mini_sector_size = self.header.mini_sector_size();
        let mut data = Vec::new();

        let mut sector = entry.starting_sector;
        let mut remaining = entry.size as usize;

        while sector != END_OF_CHAIN && sector != FREE_SECTOR && remaining > 0 {
            let offset = sector as usize * mini_sector_size;
            let to_read = remaining.min(mini_sector_size);

            if offset + to_read <= self.mini_stream_data.len() {
                data.extend_from_slice(&self.mini_stream_data[offset..offset + to_read]);
            }

            remaining -= to_read;
            sector = self
                .mini_fat
                .get(sector as usize)
                .copied()
                .unwrap_or(END_OF_CHAIN);
        }

        Ok(data)
    }

    /// Read stream by path
    pub fn read_stream_by_path(&mut self, path: &str) -> Result<Vec<u8>> {
        let entry = self
            .find_entry(path)
            .ok_or_else(|| AafError::ObjectNotFound(path.to_string()))?
            .clone();

        if !entry.is_stream() {
            return Err(AafError::InvalidStructuredStorage(format!(
                "Not a stream: {path}"
            )));
        }

        self.read_stream(&entry)
    }
}

/// Structured storage writer
///
/// Builds a Microsoft Compound File Binary container. Streams and nested
/// storages are added via [`StorageWriter::write_stream_in`] /
/// [`StorageWriter::create_storage_in`]; the parent storage id is the
/// previously-returned entry id (or `0` for the root). At
/// [`StorageWriter::finalize`] time the writer links each parent's children
/// into a left-leaning chain so the standard red-black tree traversal
/// recovers them.
pub struct StorageWriter<W: Write + Seek> {
    writer: W,
    header: Header,
    fat: Vec<u32>,
    mini_fat: Vec<u32>,
    directory_entries: Vec<DirectoryEntry>,
    mini_stream_data: Vec<u8>,
    current_sector: u32,
    /// Parent→children mapping built incrementally; consumed at finalize time.
    children: std::collections::HashMap<u32, Vec<u32>>,
    /// CLSID for the root entry (AAF-specific in normal usage).
    root_clsid: [u8; 16],
}

impl<W: Write + Seek> StorageWriter<W> {
    /// Create a new storage writer
    pub fn new(writer: W) -> Result<Self> {
        let header = Header::new();

        // Create root entry
        let root = DirectoryEntry {
            name: "Root Entry".to_string(),
            object_type: ObjectType::Root,
            color: NodeColor::Black,
            left_sibling_id: 0xFFFFFFFF,
            right_sibling_id: 0xFFFFFFFF,
            child_id: 0xFFFFFFFF,
            clsid: [0u8; 16],
            state_bits: 0,
            creation_time: 0,
            modified_time: 0,
            starting_sector: END_OF_CHAIN,
            size: 0,
        };

        Ok(Self {
            writer,
            header,
            fat: Vec::new(),
            mini_fat: Vec::new(),
            directory_entries: vec![root],
            mini_stream_data: Vec::new(),
            current_sector: 0,
            children: std::collections::HashMap::new(),
            root_clsid: [0u8; 16],
        })
    }

    /// Override the CLSID written into the root entry.
    ///
    /// AAF clients use this to mark the root with the AAF storage class id.
    pub fn set_root_clsid(&mut self, clsid: [u8; 16]) {
        self.root_clsid = clsid;
    }

    /// Allocate a new sector
    fn allocate_sector(&mut self) -> u32 {
        let sector = self.current_sector;
        self.current_sector += 1;
        self.fat.push(END_OF_CHAIN);
        sector
    }

    /// Write a top-level stream into the root storage (back-compat wrapper).
    pub fn write_stream(&mut self, name: &str, data: &[u8]) -> Result<u32> {
        self.write_stream_in(0, name, data)
    }

    /// Write a stream as a child of the given parent storage id.
    pub fn write_stream_in(&mut self, parent_id: u32, name: &str, data: &[u8]) -> Result<u32> {
        let entry_id = self.directory_entries.len() as u32;

        let (starting_sector, size) = if data.is_empty() {
            (END_OF_CHAIN, 0u64)
        } else if data.len() < self.header.mini_stream_cutoff_size as usize {
            // Use mini stream
            let mini_sector = (self.mini_stream_data.len() / MINI_SECTOR_SIZE) as u32;
            self.mini_stream_data.extend_from_slice(data);
            // Pad to mini sector boundary
            let padding = (MINI_SECTOR_SIZE - (data.len() % MINI_SECTOR_SIZE)) % MINI_SECTOR_SIZE;
            self.mini_stream_data
                .resize(self.mini_stream_data.len() + padding, 0);

            // Update mini FAT
            let sectors_needed = data.len().div_ceil(MINI_SECTOR_SIZE);
            for i in 0..sectors_needed {
                let sector = mini_sector + i as u32;
                if i < sectors_needed - 1 {
                    self.mini_fat.push(sector + 1);
                } else {
                    self.mini_fat.push(END_OF_CHAIN);
                }
            }

            (mini_sector, data.len() as u64)
        } else {
            // Use regular stream — pre-allocate FAT chain so we never grow
            // `self.fat` while iterating chunks.
            let sector_size = self.header.sector_size();
            let total_chunks = data.len().div_ceil(sector_size);
            let starting_sector = self.allocate_sector();
            let mut current_sector = starting_sector;

            for (idx, chunk) in data.chunks(sector_size).enumerate() {
                let offset = 512 + (u64::from(current_sector) * sector_size as u64);
                self.writer.seek(SeekFrom::Start(offset))?;
                self.writer.write_all(chunk)?;

                if chunk.len() < sector_size {
                    let padding = vec![0u8; sector_size - chunk.len()];
                    self.writer.write_all(&padding)?;
                }

                if idx + 1 < total_chunks {
                    let next_sector = self.allocate_sector();
                    self.fat[current_sector as usize] = next_sector;
                    current_sector = next_sector;
                }
            }

            (starting_sector, data.len() as u64)
        };

        let entry = DirectoryEntry {
            name: name.to_string(),
            object_type: ObjectType::Stream,
            color: NodeColor::Black,
            left_sibling_id: 0xFFFFFFFF,
            right_sibling_id: 0xFFFFFFFF,
            child_id: 0xFFFFFFFF,
            clsid: [0u8; 16],
            state_bits: 0,
            creation_time: 0,
            modified_time: 0,
            starting_sector,
            size,
        };

        self.directory_entries.push(entry);
        self.children.entry(parent_id).or_default().push(entry_id);
        Ok(entry_id)
    }

    /// Create a nested storage under the root (back-compat wrapper).
    pub fn create_storage(&mut self, name: &str) -> Result<u32> {
        self.create_storage_in(0, name, [0u8; 16])
    }

    /// Create a nested storage as a child of `parent_id`.
    ///
    /// `clsid` is written into the directory entry's CLSID field — for AAF
    /// objects this is the class AUID.
    pub fn create_storage_in(
        &mut self,
        parent_id: u32,
        name: &str,
        clsid: [u8; 16],
    ) -> Result<u32> {
        let entry_id = self.directory_entries.len() as u32;

        let entry = DirectoryEntry {
            name: name.to_string(),
            object_type: ObjectType::Storage,
            color: NodeColor::Black,
            left_sibling_id: 0xFFFFFFFF,
            right_sibling_id: 0xFFFFFFFF,
            child_id: 0xFFFFFFFF,
            clsid,
            state_bits: 0,
            creation_time: 0,
            modified_time: 0,
            starting_sector: 0,
            size: 0,
        };

        self.directory_entries.push(entry);
        self.children.entry(parent_id).or_default().push(entry_id);
        Ok(entry_id)
    }

    /// Link each parent's children list into a left-leaning chain so the
    /// CFB tree-walking reader can recover them.
    ///
    /// A truly conformant CFB requires names sorted by (length, lexical) and
    /// a balanced red-black tree — that's future-wave; this revision focuses
    /// on byte-symmetric round-trip within the crate, which the existing
    /// `StorageReader::collect_children` recovers via in-order DFS.
    fn link_children(&mut self) {
        let keys: Vec<u32> = self.children.keys().copied().collect();
        for parent_id in keys {
            let children = match self.children.get(&parent_id) {
                Some(c) if !c.is_empty() => c.clone(),
                _ => continue,
            };
            if let Some(parent) = self.directory_entries.get_mut(parent_id as usize) {
                parent.child_id = children[0];
            }
            for window in children.windows(2) {
                let cur = window[0];
                let next = window[1];
                if let Some(e) = self.directory_entries.get_mut(cur as usize) {
                    e.left_sibling_id = 0xFFFFFFFF;
                    e.right_sibling_id = next;
                }
            }
            if let Some(&last) = children.last() {
                if let Some(e) = self.directory_entries.get_mut(last as usize) {
                    e.left_sibling_id = 0xFFFFFFFF;
                    e.right_sibling_id = 0xFFFFFFFF;
                }
            }
        }
    }

    /// Finalize and write all structures
    pub fn finalize(&mut self) -> Result<()> {
        // Link the directory tree before serialization so children are
        // reachable from each parent.
        self.link_children();
        // Apply user-set root CLSID
        if let Some(root) = self.directory_entries.get_mut(0) {
            root.clsid = self.root_clsid;
        }

        // Update root entry with mini stream data
        if !self.mini_stream_data.is_empty() {
            let mini_stream_sector = self.allocate_sector();
            let sector_size = self.header.sector_size();
            let mini_stream_data = self.mini_stream_data.clone();
            let total_chunks = mini_stream_data.len().div_ceil(sector_size);

            let mut current_sector = mini_stream_sector;
            for (idx, chunk) in mini_stream_data.chunks(sector_size).enumerate() {
                let offset = 512 + (u64::from(current_sector) * sector_size as u64);
                self.writer.seek(SeekFrom::Start(offset))?;
                self.writer.write_all(chunk)?;
                if chunk.len() < sector_size {
                    let padding = vec![0u8; sector_size - chunk.len()];
                    self.writer.write_all(&padding)?;
                }

                if idx + 1 < total_chunks {
                    let next_sector = self.allocate_sector();
                    self.fat[current_sector as usize] = next_sector;
                    current_sector = next_sector;
                }
            }

            if let Some(root) = self.directory_entries.get_mut(0) {
                root.starting_sector = mini_stream_sector;
                root.size = mini_stream_data.len() as u64;
            }
        }

        // Write mini FAT if needed
        if !self.mini_fat.is_empty() {
            let sector_size = self.header.sector_size();
            let entries_per_sector = sector_size / 4;
            let mini_fat = self.mini_fat.clone();
            let needed_sectors = mini_fat.len().div_ceil(entries_per_sector);

            // Pre-allocate all mini-FAT sectors so we can chain them in the
            // FAT correctly (avoids growing self.fat while we iterate).
            let mut mini_fat_sectors = Vec::with_capacity(needed_sectors);
            for _ in 0..needed_sectors {
                mini_fat_sectors.push(self.allocate_sector());
            }
            if let Some(&first) = mini_fat_sectors.first() {
                self.header.first_mini_fat_sector = first;
            }

            for (i, chunk) in mini_fat.chunks(entries_per_sector).enumerate() {
                let sector = mini_fat_sectors[i];
                let offset = 512 + (u64::from(sector) * sector_size as u64);
                self.writer.seek(SeekFrom::Start(offset))?;

                for &entry in chunk {
                    self.writer.write_u32::<LittleEndian>(entry)?;
                }
                for _ in chunk.len()..entries_per_sector {
                    self.writer.write_u32::<LittleEndian>(FREE_SECTOR)?;
                }

                // Chain in FAT to next mini-FAT sector
                if i + 1 < mini_fat_sectors.len() {
                    self.fat[sector as usize] = mini_fat_sectors[i + 1];
                } else {
                    self.fat[sector as usize] = END_OF_CHAIN;
                }
            }

            self.header.mini_fat_sectors = needed_sectors as u32;
        }

        // Write directory
        self.write_directory()?;

        // Write FAT
        self.write_fat()?;

        // Update total sectors before header is written
        self.header.total_sectors = self.current_sector;

        // Write header
        self.header.write(&mut self.writer)?;

        // Ensure file is padded to last sector boundary so readers don't
        // hit EOF mid-sector when chunks happened to align exactly.
        if self.current_sector > 0 {
            let sector_size = self.header.sector_size();
            let target_end = 512u64 + u64::from(self.current_sector) * sector_size as u64;
            self.writer.seek(SeekFrom::Start(target_end - 1))?;
            self.writer.write_all(&[0u8])?;
        }

        self.writer.flush()?;
        Ok(())
    }

    /// Write directory entries
    fn write_directory(&mut self) -> Result<()> {
        let sector_size = self.header.sector_size();
        let entries_per_sector = sector_size / DIR_ENTRY_SIZE;
        let directory_entries = self.directory_entries.clone();
        let needed_sectors = directory_entries.len().div_ceil(entries_per_sector).max(1);

        // Pre-allocate directory sectors
        let mut dir_sectors = Vec::with_capacity(needed_sectors);
        for _ in 0..needed_sectors {
            dir_sectors.push(self.allocate_sector());
        }
        if let Some(&first) = dir_sectors.first() {
            self.header.first_dir_sector = first;
        }

        for (i, chunk) in directory_entries.chunks(entries_per_sector).enumerate() {
            let sector = dir_sectors[i];
            let offset = 512 + (u64::from(sector) * sector_size as u64);
            self.writer.seek(SeekFrom::Start(offset))?;

            for entry in chunk {
                let data = entry.serialize()?;
                self.writer.write_all(&data)?;
            }
            // Pad with empty entries
            for _ in chunk.len()..entries_per_sector {
                self.writer.write_all(&[0u8; DIR_ENTRY_SIZE])?;
            }

            if i + 1 < dir_sectors.len() {
                self.fat[sector as usize] = dir_sectors[i + 1];
            } else {
                self.fat[sector as usize] = END_OF_CHAIN;
            }
        }
        Ok(())
    }

    /// Write FAT
    ///
    /// NOTE: only supports up to `HEADER_DIFAT_ENTRIES` (109) FAT sectors —
    /// i.e. up to ~14 MiB of compound-file payload. Going beyond this needs
    /// DIFAT sectors which are a future-wave concern; AAF compositions
    /// typically fit comfortably.
    fn write_fat(&mut self) -> Result<()> {
        let sector_size = self.header.sector_size();
        let entries_per_sector = sector_size / 4;

        // Fixed-point iteration so that allocating FAT sectors (which extends
        // self.fat) does not change the required sector count.
        // For our scale this converges in ≤ 2 iterations.
        let mut fat_sectors;
        loop {
            fat_sectors = self.fat.len().div_ceil(entries_per_sector);
            let allocated = self
                .header
                .difat
                .iter()
                .take(HEADER_DIFAT_ENTRIES)
                .filter(|&&s| s != FREE_SECTOR)
                .count();
            if allocated >= fat_sectors {
                break;
            }
            let new_sector = self.allocate_sector();
            self.header.difat[allocated] = new_sector;
            // mark as FAT_SECTOR in the FAT itself
            self.fat[new_sector as usize] = FAT_SECTOR;
        }

        self.header.fat_sectors = fat_sectors as u32;

        // Write FAT sectors
        let fat_clone = self.fat.clone();
        for (i, chunk) in fat_clone.chunks(entries_per_sector).enumerate() {
            if i >= HEADER_DIFAT_ENTRIES {
                break;
            }
            let sector = self.header.difat[i];
            if sector == FREE_SECTOR {
                break;
            }
            let offset = 512 + (u64::from(sector) * sector_size as u64);
            self.writer.seek(SeekFrom::Start(offset))?;
            for &entry in chunk {
                self.writer.write_u32::<LittleEndian>(entry)?;
            }
            for _ in chunk.len()..entries_per_sector {
                self.writer.write_u32::<LittleEndian>(FREE_SECTOR)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_type() {
        assert_eq!(ObjectType::from_byte(0), ObjectType::Unknown);
        assert_eq!(ObjectType::from_byte(1), ObjectType::Storage);
        assert_eq!(ObjectType::from_byte(2), ObjectType::Stream);
        assert_eq!(ObjectType::from_byte(5), ObjectType::Root);
        assert_eq!(ObjectType::Storage.to_byte(), 1);
    }

    #[test]
    fn test_node_color() {
        assert_eq!(NodeColor::from_byte(0), NodeColor::Red);
        assert_eq!(NodeColor::from_byte(1), NodeColor::Black);
        assert_eq!(NodeColor::Red.to_byte(), 0);
    }

    #[test]
    fn test_header_creation() {
        let header = Header::new();
        assert_eq!(header.sector_shift, MAX_REG_SECT_SHIFT);
        assert_eq!(header.mini_sector_shift, MINI_SECT_SHIFT);
        assert_eq!(header.sector_size(), SECTOR_SIZE);
        assert_eq!(header.mini_sector_size(), MINI_SECTOR_SIZE);
    }
}
