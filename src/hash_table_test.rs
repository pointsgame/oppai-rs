use quickcheck;
use quickcheck::{Arbitrary, Gen};
use field::Pos;
use hash_table::{HashType, HashData, HashTable};

#[derive(Clone, Copy, PartialEq, Debug)]
struct HashTypeArbitrary {
  hash_type: HashType
}

impl Arbitrary for HashTypeArbitrary {
  fn arbitrary<G: Gen>(gen: &mut G) -> HashTypeArbitrary {
    HashTypeArbitrary {
      hash_type: HashType::from((gen.next_u32() % 4) as usize)
    }
  }
}

#[test]
fn hash_data_check() {
  #[cfg_attr(feature="clippy", allow(needless_pass_by_value))]
  fn prop(depth: u32, hash_type_arbitrary: HashTypeArbitrary, pos: Pos, estimation: i32) -> bool {
    let hash_data = HashData::new(depth, hash_type_arbitrary.hash_type, pos, estimation);
    hash_data.depth() == depth &&
      hash_data.hash_type() == hash_type_arbitrary.hash_type &&
      hash_data.pos() == pos &&
      hash_data.estimation() == estimation
  }
  quickcheck::quickcheck(prop as fn(u32, HashTypeArbitrary, Pos, i32) -> bool);
}

#[test]
fn hash_table_put_get_one_entry() {
  let hash_table = HashTable::new(100);
  let hash = 1_234_567_890u64;
  let data = HashData::new(3, HashType::Exact, 17, 1234);
  hash_table.put(hash, data);
  assert_eq!(hash_table.get(hash), data);
}

#[test]
fn hash_table_collision() {
  let hash_table = HashTable::new(100);
  let hash1 = 123u64;
  let hash2 = 723u64;
  let data1 = HashData::new(3, HashType::Exact, 17, 1234);
  let data2 = HashData::new(7, HashType::Alpha, 23, -4321);
  hash_table.put(hash1, data1);
  hash_table.put(hash2, data2);
  assert_eq!(hash_table.get(hash1).hash_type(), HashType::Empty);
  assert_eq!(hash_table.get(hash2), data2);
}

#[test]
fn hash_table_priority_replace() {
  let hash_table = HashTable::new(100);
  let hash = 1_234_567_890u64;
  let data1 = HashData::new(3, HashType::Alpha, 17, 1234);
  let data2 = HashData::new(3, HashType::Exact, 19, 1237);
  hash_table.put(hash, data1);
  hash_table.put(hash, data2);
  assert_eq!(hash_table.get(hash), data2);
}

#[test]
fn hash_table_priority_remain() {
let hash_table = HashTable::new(100);
  let hash = 1_234_567_890u64;
  let data1 = HashData::new(3, HashType::Exact, 17, 1234);
  let data2 = HashData::new(3, HashType::Alpha, 19, 1233);
  hash_table.put(hash, data1);
  hash_table.put(hash, data2);
  assert_eq!(hash_table.get(hash), data1);
}
