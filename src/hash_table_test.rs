use quickcheck;
use quickcheck::{Arbitrary, Gen, TestResult};
use field::Pos;
use hash_table::{HashType, HashData};

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
  fn prop(depth: u32, hash_type_arbitrary: HashTypeArbitrary, pos: Pos, estimation: i32) -> TestResult {
    let hash_data = HashData::new(depth, hash_type_arbitrary.hash_type, pos, estimation);
    if hash_data.depth() != depth ||
      hash_data.hash_type() != hash_type_arbitrary.hash_type ||
      hash_data.pos() != pos ||
      hash_data.estimation() != estimation {
      TestResult::failed()
    } else {
      TestResult::passed()
    }
  }
  quickcheck::quickcheck(prop as fn(u32, HashTypeArbitrary, Pos, i32) -> TestResult);
}
