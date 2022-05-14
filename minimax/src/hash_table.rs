use oppai_field::field::Pos;
use std::{
  convert::From,
  default::Default,
  iter,
  sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(target_pointer_width = "32")]
const HASH_TYPE_MASK: usize = 0b00000000000000000000000000000011; // 2 bits
#[cfg(target_pointer_width = "32")]
const DEPTH_MASK: usize = 0b00000000000000000000000000111100; // 4 bits
#[cfg(target_pointer_width = "32")]
const POS_MASK: usize = 0b00000000000000111111111111000000; // 12 bits
#[cfg(target_pointer_width = "32")]
const ESTIMATION_MASK: usize = 0b01111111111111000000000000000000; // 13 bits
#[cfg(target_pointer_width = "32")]
const ESTIMATION_SIGN_MASK: usize = 0b10000000000000000000000000000000; // 1 bit

#[cfg(target_pointer_width = "32")]
const HASH_TYPE_SHIFT: usize = 0;
#[cfg(target_pointer_width = "32")]
const DEPTH_SHIFT: usize = 2;
#[cfg(target_pointer_width = "32")]
const POS_SHIFT: usize = 6;
#[cfg(target_pointer_width = "32")]
const ESTIMATION_SHIFT: usize = 18;

#[cfg(target_pointer_width = "64")]
const HASH_TYPE_MASK: usize = 0xFF; // 8 bits
#[cfg(target_pointer_width = "64")]
const DEPTH_MASK: usize = 0xFF00; // 8 bits
#[cfg(target_pointer_width = "64")]
const POS_MASK: usize = 0xFFFF_0000; // 16 bits
#[cfg(target_pointer_width = "64")]
const ESTIMATION_MASK: usize = 0xFFFF_FFFF_0000_0000; // 32 bits

#[cfg(target_pointer_width = "64")]
const HASH_TYPE_SHIFT: usize = 0;
#[cfg(target_pointer_width = "64")]
const DEPTH_SHIFT: usize = 8;
#[cfg(target_pointer_width = "64")]
const POS_SHIFT: usize = 16;
#[cfg(target_pointer_width = "64")]
const ESTIMATION_SHIFT: usize = 32;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HashType {
  Empty = 0,
  Alpha = 1,
  Exact = 2,
  Beta = 3,
}

impl From<usize> for HashType {
  fn from(value: usize) -> HashType {
    match value {
      1 => HashType::Alpha,
      2 => HashType::Exact,
      3 => HashType::Beta,
      _ => HashType::Empty,
    }
  }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct HashData {
  data: usize,
}

impl HashData {
  fn pack_hash_type(hash_type: HashType) -> usize {
    ((hash_type as usize) << HASH_TYPE_SHIFT) & HASH_TYPE_MASK
  }

  fn pack_depth(depth: u32) -> usize {
    ((depth as usize) << DEPTH_SHIFT) & DEPTH_MASK
  }

  fn pack_pos(pos: Pos) -> usize {
    ((pos as usize) << POS_SHIFT) & POS_MASK
  }

  #[cfg(target_pointer_width = "64")]
  fn pack_estimation(estimation: i32) -> usize {
    ((estimation as usize) << ESTIMATION_SHIFT) & ESTIMATION_MASK
  }

  #[cfg(target_pointer_width = "32")]
  fn pack_estimation(estimation: i32) -> usize {
    (((estimation as usize) << ESTIMATION_SHIFT) & ESTIMATION_MASK) | ((estimation as usize) & ESTIMATION_SIGN_MASK)
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

  #[cfg(target_pointer_width = "64")]
  pub fn estimation(self) -> i32 {
    ((self.data & ESTIMATION_MASK) >> ESTIMATION_SHIFT) as i32
  }

  #[cfg(target_pointer_width = "32")]
  pub fn estimation(self) -> i32 {
    (((self.data & ESTIMATION_MASK) >> ESTIMATION_SHIFT) | (self.data & ESTIMATION_SIGN_MASK)) as i32
  }
}

#[derive(Debug)]
struct HashEntry {
  hash: AtomicUsize,
  data: AtomicUsize,
}

impl Default for HashEntry {
  fn default() -> HashEntry {
    HashEntry {
      hash: AtomicUsize::new(0),
      data: AtomicUsize::new(0),
    }
  }
}

impl HashEntry {
  fn verified(&self, hash: u64) -> HashData {
    let xored_hash = self.hash.load(Ordering::Relaxed);
    let data = self.data.load(Ordering::Relaxed);
    if xored_hash ^ data == hash as usize {
      HashData { data }
    } else {
      HashData::default()
    }
  }
}

#[derive(Debug)]
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
      let xored_hash = hash as usize ^ new_data.data;
      let entry = &self.entries[idx];
      entry.hash.store(xored_hash, Ordering::Relaxed);
      entry.data.store(new_data.data, Ordering::Relaxed);
    }
  }

  pub fn get(&self, hash: u64) -> HashData {
    let idx = HashTable::index(self.len(), hash);
    self.entries[idx].verified(hash)
  }
}
