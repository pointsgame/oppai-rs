use std::io::{BufReader, BufRead};
use std::str::FromStr;
use std::fs::File;
use tar::Archive;
use zigzag::Zigzag;
use dfa::{Dfa, DfaState};

struct Move {
    x: u32,
    y: u32,
    p: f64 // probability
}

struct Pattern {
    p: f64, // priority (probability = p / sum(p))
    moves: Vec<Move>
}

struct Patterns {
    dfa: Dfa,
    patterns: Vec<Pattern>
}

impl Patterns {
    fn read_sizes<T: BufRead>(input: &mut T, s: &mut String) -> (u32, u32, u32) {
        s.clear();
        input.read_line(s).ok();
        let mut split = s.split(' ').fuse();
        let width = u32::from_str(split.next().expect("???")).expect("???");
        let height = u32::from_str(split.next().expect("???")).expect("???");
        let moves_count = u32::from_str(split.next().expect("???")).expect("???");
        (width, height, moves_count)
    }

    fn read_pattern<T: BufRead>(input: &mut T, s: &mut String, width: u32, height: u32) {
        for y in 0 .. height {
            input.read_line(s).ok();
        }
    }

    fn build_dfa(width: u32, height: u32, pattern: u32, s: &str) -> Dfa {
        let center_x = width / 2;
        let center_y = height / 2;
        let mut states = Vec::new(); //TODO: capacity
        let mut i = 0;
        for (shift_x, shift_y) in Zigzag::new().into_iter().take(10) {
            i += 1;
            let x = center_x as i32 + shift_x;
            let y = center_y as i32 + shift_y;
            let state = if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let pos = y as u32 * width + x as u32;
                match s.char_at(pos as usize) {
                    '.' => DfaState::new(i, -1, -1, -1, -1),
                    '?' => DfaState::new(i, i, i, i, -1),
                    'R' => DfaState::new(-1, i, -1, -1, -1),
                    'B' => DfaState::new(-1, -1, i, -1, -1),
                    'r' => DfaState::new(i, i, -1, -1, -1),
                    'b' => DfaState::new(i, -1, i, -1, -1),
                    '*' => DfaState::new(-1, -1, -1, i, -1),
                    c   => panic!("Invalid character in pattern: {}", c)
                }
                
            } else {
                DfaState::new(-1, -1, -1, -1, -1)
            };
            states.push(state);
        }
        states.push(DfaState::new(-1, -1, -1, -1, pattern as i32));
        Dfa::new(states)
    }

    pub fn load(file: File) -> Patterns {
        let archive = Archive::new(file);
        let mut s = String::new();
        for file in archive.files().expect("Reading of tar archive is failed.").into_iter().map(|file| file.expect("Reading of file in tar archive is failed.")) {
            let mut input = BufReader::new(file);
            let (width, height, moves_count) = Patterns::read_sizes(&mut input, &mut s);
        }
        unimplemented!()
    }
}
