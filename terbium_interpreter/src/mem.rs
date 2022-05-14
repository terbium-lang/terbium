use std::{
    alloc::{alloc, dealloc, Layout},
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    mem::{replace, size_of},
    ops::Deref,
    ptr::{write, NonNull},
};

pub struct Block {
    ptr: NonNull<u8>,
    pub size: usize,
}

#[derive(Debug, PartialEq)]
pub enum BlockAllocError {
    /// An invalid block size was specified.
    InvalidSize,

    /// Out of memory to allocate this block.
    OutOfMemory,
}

impl Block {
    pub fn new(size: usize) -> Result<Self, BlockAllocError> {
        if !size.is_power_of_two() {
            return Err(BlockAllocError::InvalidSize);
        }

        Ok(Self {
            ptr: Self::alloc(size)?,
            size,
        })
    }

    pub fn alloc(size: usize) -> Result<NonNull<u8>, BlockAllocError> {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, size);
            let ptr = alloc(layout);

            if ptr.is_null() {
                Err(BlockAllocError::OutOfMemory)
            } else {
                Ok(NonNull::new_unchecked(ptr))
            }
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn into_mut_ptr(self) -> NonNull<u8> {
        self.ptr
    }

    pub fn dealloc(ptr: NonNull<u8>, size: usize) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, size);

            dealloc(ptr.as_ptr(), layout);
        }
    }
}

pub struct BlockMeta {
    line_mark: [bool; BlockBuffer::LINE_COUNT],
    block_mark: bool,
}

impl BlockMeta {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            line_mark: [false; BlockBuffer::LINE_COUNT],
            block_mark: false,
        })
    }

    pub fn mark_line(&mut self, line: usize) {
        self.line_mark[line] = true;
    }

    pub fn mark_block(&mut self) {
        self.block_mark = true;
    }

    pub fn find_next_available_hole(&self, starting_at: usize) -> Option<(usize, usize)> {
        let mut count = 0_usize;
        let mut start: Option<usize> = None;
        let mut stop = 0_usize;

        let starting_line = starting_at / BlockBuffer::LINE_SIZE;

        for (i, marked) in self.line_mark[starting_line..].iter().enumerate() {
            let index = starting_line + i;

            if !*marked {
                count += 1;

                // If this is the first line in a hole (and not the zeroth line), consider it
                // conservatively marked and skip to the next line
                if count == 1 && index > 0 {
                    continue;
                }

                // record the first hole index
                if start.is_none() {
                    start = Some(index);
                }

                stop = index + 1;
            }

            // If we reached a marked line or the end of the block, see if we have
            // a valid hole to work with
            if count > 0 && (*marked || stop >= BlockBuffer::LINE_COUNT) {
                if let Some(start) = start {
                    let cursor = start * BlockBuffer::LINE_SIZE;
                    let limit = stop * BlockBuffer::LINE_SIZE;

                    return Some((cursor, limit));
                }
            }

            // If this line is marked and we didn't return a new cursor/limit pair by now,
            // reset the hole state
            if *marked {
                count = 0;
                start = None;
            }
        }

        None
    }
}

pub struct BlockBuffer {
    block: Block,
    cursor: usize,
    limit: usize,
    meta: Box<BlockMeta>,
}

impl BlockBuffer {
    pub const BLOCK_SIZE_BITS: usize = 15;
    pub const BLOCK_SIZE: usize = 1 << Self::BLOCK_SIZE_BITS;
    pub const BLOCK_OFFSET: usize = size_of::<usize>() * 2;

    pub const LINE_SIZE_BITS: usize = 7;
    pub const LINE_SIZE: usize = 1 << Self::LINE_SIZE_BITS;
    pub const LINE_COUNT: usize = Self::BLOCK_SIZE / Self::LINE_SIZE;

    pub fn new() -> Result<Self, BlockAllocError> {
        let mut block = Self {
            block: Block::new(Self::BLOCK_SIZE)?,
            cursor: Self::BLOCK_SIZE - Self::BLOCK_OFFSET,
            limit: Self::BLOCK_SIZE,
            meta: BlockMeta::new(),
        };

        unsafe {
            let p: *const BlockMeta = &*block.meta;
            block.write(p, 0);
        }

        Ok(block)
    }

    pub unsafe fn write<T>(&mut self, o: T, offset: usize) -> *const T {
        let ptr = self.block.as_ptr().add(offset) as *mut T;
        write(ptr, o);

        ptr
    }

    pub fn inner_alloc(&mut self, size: usize) -> Option<*const u8> {
        let next_bump = self.cursor + size;

        if next_bump > self.limit {
            if self.limit < Self::BLOCK_SIZE {
                if let Some((cursor, limit)) = self.meta.find_next_available_hole(self.limit) {
                    self.cursor = cursor;
                    self.limit = limit;
                    return self.inner_alloc(size);
                }
            }
            None
        } else {
            let offset = self.cursor;
            self.cursor = next_bump;

            unsafe { Some(self.block.as_ptr().add(offset) as *const u8) }
        }
    }

    pub fn hole_size(&self) -> usize {
        self.limit - self.cursor
    }
}

struct BlockList {
    head: Option<BlockBuffer>,
    overflow: Option<BlockBuffer>,
    rest: Vec<BlockBuffer>,
}

impl BlockList {
    pub fn new() -> Self {
        Self {
            head: None,
            overflow: None,
            rest: Vec::new(),
        }
    }

    pub fn overflow_alloc(&mut self, alloc_size: usize) -> Result<*const u8, BlockAllocError> {
        assert!(alloc_size <= BlockBuffer::BLOCK_SIZE - BlockBuffer::BLOCK_OFFSET);

        Ok(match self.overflow {
            // Use the existing overflow block
            Some(ref mut overflow) => match overflow.inner_alloc(alloc_size) {
                // the block has a suitable hole
                Some(space) => space,
                None => {
                    let previous = replace(overflow, BlockBuffer::new()?);
                    self.rest.push(previous);

                    overflow
                        .inner_alloc(alloc_size)
                        .expect("error allocating memory")
                }
            },
            None => {
                let mut overflow = BlockBuffer::new()?;

                // Assertion above allows us to unwrap this safely
                let space = overflow.inner_alloc(alloc_size).unwrap();
                self.overflow = Some(overflow);

                space
            }
        } as *const u8)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BlockSize {
    Small,
    Medium,
    Large,
}

impl BlockSize {
    pub fn from_size(size: usize) -> Self {
        if size <= BlockBuffer::LINE_SIZE {
            Self::Small
        } else if size <= BlockBuffer::BLOCK_SIZE - BlockBuffer::BLOCK_OFFSET {
            Self::Medium
        } else {
            Self::Large
        }
    }
}

/// Implements a "stricky-immix heap" allocator.
pub struct RawHeap<H> {
    blocks: UnsafeCell<BlockList>,
    header: PhantomData<*const H>,
}

impl<H> RawHeap<H> {
    pub fn new() -> Self {
        Self {
            blocks: UnsafeCell::new(BlockList::new()),
            header: PhantomData,
        }
    }

    pub(crate) fn find_space(
        &self,
        alloc_size: usize,
        size: BlockSize,
    ) -> Result<*const u8, BlockAllocError> {
        let blocks = unsafe { &mut *self.blocks.get() };

        // TODO handle large objects
        if size == BlockSize::Large {
            // simply fail for objects larger than the block size
            return Err(BlockAllocError::InvalidSize);
        }

        Ok(match blocks.head {
            Some(ref mut head) => {
                // If this is a medium object that doesn't fit in the hole, use overflow
                if size == BlockSize::Medium && alloc_size > head.hole_size() {
                    return blocks.overflow_alloc(alloc_size);
                }

                match head.inner_alloc(alloc_size) {
                    Some(space) => space,
                    None => {
                        let previous = replace(head, BlockBuffer::new()?);
                        blocks.rest.push(previous);

                        head.inner_alloc(alloc_size).unwrap()
                    }
                }
            }
            // Make a new block if one doesn't already exist
            None => {
                let mut head = BlockBuffer::new()?;

                // Assertion in overflow_alloc allows us to unwrap this safely,
                // this should only ever be called after the assertion.
                let space = head.inner_alloc(alloc_size).unwrap();
                blocks.head = Some(head);

                space
            }
        } as *const u8)
    }
}

pub struct Ptr<T: Sized> {
    ptr: NonNull<T>,
}

impl<T: Sized> Ptr<T> {
    pub fn new(ptr: *const T) -> Self {
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
        }
    }

    pub fn ptr(self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn untyped(self) -> NonNull<()> {
        self.ptr.cast()
    }

    unsafe fn as_ref(&self) -> &T {
        self.ptr.as_ref()
    }

    unsafe fn as_mut(&mut self) -> &mut T {
        self.ptr.as_mut()
    }
}

impl<T: Sized> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T: Sized> Copy for Ptr<T> {}

impl<T: Sized> PartialEq for Ptr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mark {
    Allocated,
    Unmarked,
    Marked,
}

pub trait AllocObject<T: Copy + Clone> {
    const TYPE_ID: T;
}

pub trait AllocHeader: Sized {
    type TypeId: Copy + Clone;

    fn new<O: AllocObject<Self::TypeId>>(size: usize, block_size: BlockSize, mark: Mark) -> Self;
    fn mark(&mut self);
    fn is_marked(&self) -> bool;
    fn size(&self) -> usize;
    fn block_size(&self) -> BlockSize;
    fn type_id(&self) -> Self::TypeId;
}

pub trait RawAllocator {
    type Header: AllocHeader;

    fn alloc<T>(&self, o: T) -> Result<Ptr<T>, BlockAllocError>
    where
        T: AllocObject<<Self::Header as AllocHeader>::TypeId>;

    // fn alloc_array(&self, size: usize) -> Result<Ptr<u8>, BlockAllocError>;
    fn get_header(o: NonNull<()>) -> NonNull<Self::Header>;
    fn get_object(header: NonNull<Self::Header>) -> NonNull<()>;
}

fn get_alloc_size(size: usize) -> usize {
    let align = size_of::<usize>();
    (size + (align - 1)) & !(align - 1)
}

impl<H: AllocHeader> RawAllocator for RawHeap<H> {
    type Header = H;

    fn alloc<T>(&self, o: T) -> Result<Ptr<T>, BlockAllocError>
    where
        T: AllocObject<<Self::Header as AllocHeader>::TypeId>,
    {
        let header_size = size_of::<Self::Header>();
        let object_size = size_of::<T>();
        let total_size = header_size + object_size;

        let alloc_size = get_alloc_size(total_size);
        let size_t = BlockSize::from_size(alloc_size);

        // Allocate enough space for the header and object
        let space = self.find_space(alloc_size, size_t)?;

        let header = Self::Header::new::<T>(object_size, size_t, Mark::Allocated);

        unsafe {
            write(space as *mut Self::Header, header);
        }

        let object_space = unsafe { space.offset(header_size as isize) };
        unsafe {
            write(object_space as *mut T, o);
        }

        Ok(Ptr::new(object_space as *const T))
    }

    fn get_header(o: NonNull<()>) -> NonNull<Self::Header> {
        unsafe { NonNull::new_unchecked(o.cast::<Self::Header>().as_ptr().offset(-1)) }
    }

    fn get_object(header: NonNull<Self::Header>) -> NonNull<()> {
        unsafe { NonNull::new_unchecked(header.as_ptr().offset(1).cast::<()>()) }
    }
}

/// Anchors lifetimes of mutators
pub trait MutatorScope {}

pub struct ScopedPtr<'s, T: Sized> {
    o: &'s T,
}

impl<'s, T: Sized> ScopedPtr<'s, T> {
    // The 's lifetime acts as a "guard" for the lifetime of o
    pub fn new(_: &'s dyn MutatorScope, o: &'s T) -> Self {
        ScopedPtr { o }
    }

    pub fn inner(&self) -> &'s T {
        self.o
    }
}

impl<'s, T: Sized> MutatorScope for ScopedPtr<'s, T> {}

impl<'s, T: Sized> Clone for ScopedPtr<'s, T> {
    fn clone(&self) -> Self {
        ScopedPtr { o: self.o }
    }
}

impl<'s, T: Sized> Copy for ScopedPtr<'s, T> {}

impl<'s, T: Sized> Deref for ScopedPtr<'s, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.o
    }
}

impl<'s, T: Sized + PartialEq> PartialEq for ScopedPtr<'s, T> {
    fn eq(&self, other: &Self) -> bool {
        self.o == other.o
    }
}

#[derive(Clone)]
pub struct CellPtr<T: Sized> {
    o: Cell<Ptr<T>>,
}

impl<T: Sized> CellPtr<T> {
    pub fn get<'s>(&self, guard: &'s dyn MutatorScope) -> ScopedPtr<'s, T> {
        ScopedPtr::new(guard, self.o.get().scoped_ref(guard))
    }
}

pub trait ScopedRef<T> {
    fn scoped_ref<'s>(&self, scope: &'s dyn MutatorScope) -> &'s T;
}

impl<T> ScopedRef<T> for Ptr<T> {
    fn scoped_ref<'s>(&self, _: &'s dyn MutatorScope) -> &'s T {
        unsafe { &*self.ptr() }
    }
}
