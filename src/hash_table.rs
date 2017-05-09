use std::{iter, ptr};
use std::sync::atomic::{AtomicPtr, Ordering};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HashEntryType {
  Alpha,
  Exact,
  Beta
}

#[repr(packed)]
#[derive(Clone, PartialEq, Debug)]
pub struct HashEntry {
  hash: u64,
  depth: u8,
  entry_type: HashEntryType,
  pos: u16,
  estimation: i32
}

pub struct HashTable {
  entries: Vec<AtomicPtr<HashEntry>>
}

impl HashTable {
  #[inline]
  fn index(length: usize, hash: u64) -> usize {
    (hash % (length as u64)) as usize
  }

  pub fn new(length: usize) -> HashTable {
    HashTable {
      entries: iter::repeat(ptr::null_mut()).map(AtomicPtr::new).take(length).collect()
    }
  }

  pub fn len(&self) -> usize {
    self.entries.len()
  }

  #[cfg_attr(feature="clippy", allow(if_same_then_else))]
  fn choose_best(entry1: Box<HashEntry>, entry2: Box<HashEntry>) -> Box<HashEntry> {
    if entry1.entry_type != HashEntryType::Alpha && entry2.entry_type == HashEntryType::Alpha {
      entry1
    } else if entry1.entry_type == HashEntryType::Alpha && entry2.entry_type != HashEntryType::Alpha {
      entry2
    } else if entry1.depth > entry2.depth {
      entry1
    } else if entry1.depth < entry2.depth {
      entry2
    } else if entry1.estimation > entry2.estimation {
      entry1
    } else {
      entry2
    }
  }

  fn insert_to(&self, idx: usize, mut entry: Box<HashEntry>, mut prefer_new: bool) {
    loop {
      let entry_ptr = Box::into_raw(entry);
      if self.entries[idx].compare_and_swap(ptr::null_mut(), entry_ptr, Ordering::Relaxed).is_null() {
        return;
      }
      let cur_entry_ptr = self.entries[idx].swap(ptr::null_mut(), Ordering::Relaxed);
      if cur_entry_ptr.is_null() {
        entry = unsafe {
          Box::from_raw(entry_ptr)
        };
        continue;
      }
      let cur_entry = unsafe {
        Box::from_raw(cur_entry_ptr)
      };
      let new_entry = unsafe {
        Box::from_raw(entry_ptr)
      };
      if cur_entry.hash != new_entry.hash {
        entry = if prefer_new { new_entry } else { cur_entry };
      } else {
        entry = HashTable::choose_best(cur_entry, new_entry);
      }
      prefer_new = false;
    }
  }

  pub fn insert(&self, entry: Box<HashEntry>) {
    let idx = HashTable::index(self.len(), entry.hash);
    self.insert_to(idx, entry, true);
  }

  pub fn get(&self, hash: u64) -> Option<HashEntry> {
    let idx = HashTable::index(self.len(), hash);
    let entry_ptr = self.entries[idx].swap(ptr::null_mut(), Ordering::Relaxed);
    if entry_ptr.is_null() {
      return None;
    }
    let entry = unsafe {
      Box::from_raw(entry_ptr)
    };
    if entry.hash == hash {
      let cloned = (*entry).clone();
      self.insert_to(idx, entry, false);
      Some(cloned)
    } else {
      self.insert_to(idx, entry, false);
      None
    }
  }
}

pub struct HashTablePair<'l> {
  pub cur_hash_table: &'l HashTable,
  pub enemy_hash_table: &'l HashTable
}

impl<'l> HashTablePair<'l> {
  pub fn swap(&self) -> HashTablePair<'l> {
    HashTablePair {
      cur_hash_table: self.enemy_hash_table,
      enemy_hash_table: self.cur_hash_table
    }
  }
}
