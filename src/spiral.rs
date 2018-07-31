pub struct Spiral<T> {
  x: T,
  y: T,
}

impl Spiral<i32> {
  pub fn new() -> Spiral<i32> {
    Spiral { x: 0, y: 0 }
  }

  fn next_state(&mut self) {
    if self.y == 0 {
      if self.x <= 0 {
        self.y += 1;
      } else {
        self.x -= 1;
        self.y -= 1;
      }
    } else if self.x == 0 {
      if self.y <= 0 {
        self.x -= 1;
        self.y += 1;
      } else {
        self.x += 1;
        self.y -= 1;
      }
    } else if self.y < 0 {
      if self.x < 0 {
        self.y += 1;
      } else {
        self.y -= 1;
      }
      self.x -= 1;
    } else {
      if self.x < 0 {
        self.y += 1;
      } else {
        self.y -= 1;
      }
      self.x += 1;
    }
  }
}

impl Iterator for Spiral<i32> {
  type Item = (i32, i32);

  fn next(&mut self) -> Option<(i32, i32)> {
    let result = Some((self.x, self.y));
    self.next_state();
    result
  }
}
