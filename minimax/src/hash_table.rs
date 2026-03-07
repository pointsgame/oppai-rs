use oppai_field::field::Pos;
use std::{
  convert::From,
  default::Default,
  iter,
  sync::atomic::{AtomicU64, Ordering},
};

const HASH_TYPE_MASK: u64 = 0xFF; // 8 bits
const DEPTH_MASK: u64 = 0xFF00; // 8 bits
const POS_MASK: u64 = 0xFFFF_0000; // 16 bits
const ESTIMATION_MASK: u64 = 0xFFFF_FFFF_0000_0000; // 32 bits

const HASH_TYPE_SHIFT: usize = 0;
const DEPTH_SHIFT: usize = 8;
const POS_SHIFT: usize = 16;
const ESTIMATION_SHIFT: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HashType {
  Empty = 0,
  Alpha = 1,
  Exact = 2,
  Beta = 3,
}

impl From<u64> for HashType {
  fn from(value: u64) -> HashType {
    match value {
      1 => HashType::Alpha,
      2 => HashType::Exact,
      3 => HashType::Beta,
      _ => HashType::Empty,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct HashData {
  data: u64,
}

impl HashData {
  fn pack_hash_type(hash_type: HashType) -> u64 {
    ((hash_type as u64) << HASH_TYPE_SHIFT) & HASH_TYPE_MASK
  }

  fn pack_depth(depth: u32) -> u64 {
    ((depth as u64) << DEPTH_SHIFT) & DEPTH_MASK
  }

  fn pack_pos(pos: Pos) -> u64 {
    ((pos as u64) << POS_SHIFT) & POS_MASK
  }

  fn pack_estimation(estimation: i32) -> u64 {
    ((estimation as u64) << ESTIMATION_SHIFT) & ESTIMATION_MASK
  }

  pub fn new(depth: u32, hash_type: HashType, pos: Pos, estimation: i32) -> HashData {
    HashData {
      data: HashData::pack_hash_type(hash_type)
        | HashData::pack_depth(depth)
        | HashData::pack_pos(pos)
        | HashData::pack_estimation(estimation),
    }
  }

  pub fn hash_type(self) -> HashType {
    HashType::from((self.data & HASH_TYPE_MASK) >> HASH_TYPE_SHIFT)
  }

  pub fn depth(self) -> u32 {
    ((self.data & DEPTH_MASK) >> DEPTH_SHIFT) as u32
  }

  pub fn pos(self) -> Pos {
    ((self.data & POS_MASK) >> POS_SHIFT) as Pos
  }

  pub fn estimation(self) -> i32 {
    ((self.data & ESTIMATION_MASK) >> ESTIMATION_SHIFT) as i32
  }
}

#[derive(Debug)]
struct HashEntry {
  hash: AtomicU64,
  data: AtomicU64,
}

impl Clone for HashEntry {
  fn clone(&self) -> Self {
    Self {
      hash: AtomicU64::new(self.hash.load(Ordering::SeqCst)),
      data: AtomicU64::new(self.data.load(Ordering::SeqCst)),
    }
  }
}

impl Default for HashEntry {
  fn default() -> HashEntry {
    HashEntry {
      hash: AtomicU64::new(0),
      data: AtomicU64::new(0),
    }
  }
}

impl HashEntry {
  fn verified(&self, hash: u64) -> HashData {
    let xored_hash = self.hash.load(Ordering::Relaxed);
    let data = self.data.load(Ordering::Relaxed);
    if xored_hash ^ data == hash {
      HashData { data }
    } else {
      HashData::default()
    }
  }
}

#[derive(Debug, Clone)]
pub struct HashTable {
  entries: Vec<HashEntry>,
}

impl HashTable {
  #[inline]
  fn index(length: usize, hash: u64) -> usize {
    (hash % (length as u64)) as usize
  }

  pub fn new(length: usize) -> HashTable {
    HashTable {
      entries: iter::repeat_with(HashEntry::default).take(length).collect(),
    }
  }

  pub fn len(&self) -> usize {
    self.entries.len()
  }

  pub fn is_empty(&self) -> bool {
    self.entries.is_empty()
  }

  fn choose_best(data1: HashData, data2: HashData) -> HashData {
    if data1.hash_type() != HashType::Alpha && data2.hash_type() == HashType::Alpha {
      data1
    } else if data1.hash_type() == HashType::Alpha && data2.hash_type() != HashType::Alpha {
      data2
    } else if data1.depth() > data2.depth() {
      data1
    } else if data1.depth() < data2.depth() {
      data2
    } else if data1.estimation() > data2.estimation() {
      data1
    } else {
      data2
    }
  }

  pub fn put(&self, hash: u64, hash_data: HashData) {
    let idx = HashTable::index(self.len(), hash);
    let cur_data = self.entries[idx].verified(hash);
    let new_data = if cur_data.hash_type() == HashType::Empty {
      hash_data
    } else {
      HashTable::choose_best(cur_data, hash_data)
    };
    if cur_data != new_data {
      let xored_hash = hash ^ new_data.data;
      let entry = &self.entries[idx];
      entry.hash.store(xored_hash, Ordering::Relaxed);
      entry.data.store(new_data.data, Ordering::Relaxed);
    }
  }

  pub fn get(&self, hash: u64) -> HashData {
    let idx = HashTable::index(self.len(), hash);
    self.entries[idx].verified(hash)
  }

  pub fn clear(&mut self) {
    for entry in self.entries.iter_mut() {
      *entry = HashEntry::default();
    }
  }
}
