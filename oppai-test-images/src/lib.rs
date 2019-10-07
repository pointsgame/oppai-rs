pub struct TestImage {
  pub image: &'static str,
  pub solution: (u32, u32),
}

// 8 is the minimum depth value to detect correct move in this test.
// With depth 7 after 3 moves we might have this position:
// ........
// .....a..
// ...a....
// ..AaAAa.
// ...Aaa?.
// ..A.A?..
// ........
// ........
// Question marks here indicate trajectory that will be excluded because
// it doesn't intersect any other trajectory with length 2.
// Without this trajectory black player won't be able to find the escape.
// So red player will think that he wins with move (5, 1).
pub const IMAGE_1: TestImage = TestImage {
  image: "
  ........
  ........
  ...a....
  ..AaA...
  ...Aaa..
  ..A.A...
  ........
  ........
  ",
  solution: (5, 2),
};

pub const IMAGE_2: TestImage = TestImage {
  image: "
  ........
  ........
  ...a.a..
  ...AAa..
  ...aAa..
  ....Aa..
  ...aaA..
  ........
  ........
  ",
  solution: (2, 3),
};

pub const IMAGE_3: TestImage = TestImage {
  image: "
  ........
  ........
  ...a....
  ..aA.a..
  ..aAA...
  ..aa....
  ........
  ........
  ",
  solution: (5, 5),
};

pub const IMAGE_4: TestImage = TestImage {
  image: "
  .........
  ....a....
  .........
  ...Aa.A..
  ..A...A..
  ..AaaaA..
  ...AAAa..
  ......a..
  .........
  ",
  solution: (5, 3),
};

pub const IMAGE_5: TestImage = TestImage {
  image: "
  ...........
  ....aaa....
  ..AAa.A.A..
  .A.aAA...A.
  ...a.......
  ...a..a....
  ....aa.....
  ...........
  ",
  solution: (6, 3),
};

pub const IMAGE_6: TestImage = TestImage {
  image: "
  ............
  ............
  ..A.a.......
  ...Aa..aa...
  ...aAaaaAA..
  ...aAAaA....
  ...a.A......
  ............
  ............
  ",
  solution: (7, 6),
};

pub const IMAGE_7: TestImage = TestImage {
  image: "
  ............
  .......aa...
  .a...AaA.a..
  ..a.A.A.Aa..
  ..a..A.A.a..
  ...aaaaaa...
  ............
  ............
  ",
  solution: (4, 1),
};

pub const IMAGE_8: TestImage = TestImage {
  image: "
  ............
  ............
  .......AA...
  .....AAaaa..
  .....Aa.....
  ..A.Aa.a....
  ...Aa.A..a..
  ..Aa.a......
  ..Aa.a..A...
  ...AAAAA....
  ............
  ............
  ",
  solution: (6, 7),
};

pub const IMAGE_9: TestImage = TestImage {
  image: "
  ...........
  ...........
  ...aA...a..
  ..aA...a...
  ..aAA.a....
  ..aAAAAa...
  ..aaAaaA...
  ..AAaaAA...
  ....a......
  ...AaA.....
  ....A......
  ...........
  ",
  solution: (5, 3),
};

pub const IMAGE_10: TestImage = TestImage {
  image: "
  ..........
  ..........
  ....aaaA..
  .....AAa..
  ..A..A.a..
  ...A..a...
  ....A.a...
  .....Aa...
  ....Aa.a..
  ....Aa....
  ..........
  ..........
  ",
  solution: (5, 6),
};

pub const IMAGE_11: TestImage = TestImage {
  image: "
  ...........
  ...........
  ..A........
  ..A........
  ..A...Aaa..
  ...AaaaA...
  ....AAA....
  ...........
  ...........
  ",
  solution: (5, 3),
};

pub const IMAGE_12: TestImage = TestImage {
  image: "
  ...........
  ...........
  ...a..a....
  ...AA.aAA..
  ...a.AAa...
  ...aaAaa...
  ..AAAa.....
  .....a.....
  ...........
  ...........
  ",
  solution: (5, 3),
};

pub const IMAGE_13: TestImage = TestImage {
  image: "
  .........
  .........
  ...AA.A..
  ...Aaa...
  ...Aa.A..
  ..aaAA...
  ....aa...
  .........
  .........
  ",
  solution: (6, 5),
};

pub const IMAGE_14: TestImage = TestImage {
  image: "
  ..........
  ..........
  ...aa.....
  ..a..a....
  ..a...a...
  ..aAA.Aa..
  ..Aa..Aa..
  .....A.a..
  ...AA..a..
  ......a...
  ..........
  ..........
  ",
  solution: (4, 7),
};