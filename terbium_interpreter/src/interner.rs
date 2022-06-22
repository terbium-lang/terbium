// Implementation partially taken from https://matklad.github.io/2020/03/22/fast-simple-rust-interner.html

use std::{collections::HashMap, mem};

pub type StringId = usize;

#[derive(Debug)]
pub struct Interner {
    map: HashMap<&'static str, StringId>,
    vec: Vec<&'static str>,
    buf: String,
    full: Vec<String>,
}

impl Interner {
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.next_power_of_two();

        Self {
            map: HashMap::default(),
            vec: Vec::new(),
            buf: String::with_capacity(cap),
            full: Vec::new(),
        }
    }

    pub fn intern(&mut self, name: &str) -> StringId {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let name = unsafe { self.alloc(name) };
        let id = self.map.len();

        self.map.insert(name, id);
        self.vec.push(name);

        id
    }

    #[must_use]
    pub fn lookup(&self, id: StringId) -> &str {
        self.vec[id]
    }

    unsafe fn alloc(&mut self, name: &str) -> &'static str {
        let cap = self.buf.capacity();
        if cap < self.buf.len() + name.len() {
            let new_cap = (cap.max(name.len()) + 1).next_power_of_two();

            let new_buf = String::with_capacity(new_cap);
            let old_buf = mem::replace(&mut self.buf, new_buf);
            self.full.push(old_buf);
        }

        let interned = {
            let start = self.buf.len();
            self.buf.push_str(name);
            &self.buf[start..]
        };

        &*(interned as *const str)
    }
}
