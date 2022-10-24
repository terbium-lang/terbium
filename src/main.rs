use grammar::parser::Peeker;
use std::ops::{Deref, DerefMut};

fn main() {
    let mut p = Peeker::new("abcdefg".chars());
    println!("{:?}", p.peek()); // a
    println!("{:?}", p.peek_nth(3)); // c
    println!("{:?}", p.next()); // a
    println!("{:?}", p.next()); // b
    println!("{:?}", p.next()); // c
    println!("{:?}", p.peek()); // d
    println!("{:?}", p.next()); // d
    println!("{:?}", p.peek_nth(2)); // f
    println!("{:?}", p.next()); // e
}
