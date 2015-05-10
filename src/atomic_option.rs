use std::ptr::*;
use std::sync::atomic::*;
use std::boxed::*;

#[unsafe_no_drop_flag]
pub struct AtomicOption<T> {
    atomic_ptr: AtomicPtr<T>
}

impl<T> Drop for AtomicOption<T> {
  fn drop(&mut self) {
    self.clear(Ordering::Relaxed);
  }
}

impl<T> AtomicOption<T> {
  pub fn new(b: Box<T>) -> AtomicOption<T> {
    AtomicOption {
      atomic_ptr: AtomicPtr::new( unsafe { into_raw(b) } )
    }
  }

  pub fn empty() -> AtomicOption<T> {
    AtomicOption {
      atomic_ptr: AtomicPtr::new(null_mut())
    }
  }

  pub unsafe fn load<'a>(&'a self, order: Ordering) -> Option<&'a T> {
    let ptr = self.atomic_ptr.load(order);
    if !ptr.is_null() {
      Some(&*ptr)
    } else {
      None
    }
  }

  pub fn take(&self, order: Ordering) -> Option<Box<T>> {
    let ptr = self.atomic_ptr.swap(null_mut(), order);
    if !ptr.is_null() {
      Some(unsafe { Box::from_raw(ptr) })
    } else {
      None
    }
  }

  pub fn store(&self, b: Box<T>, order: Ordering) {
    let ptr = self.atomic_ptr.swap(unsafe { into_raw(b) }, order);
    if !ptr.is_null() {
      let b = unsafe { Box::from_raw(ptr) };
      drop(b);
    }
  }

  pub fn clear(&self, order: Ordering) {
    let ptr = self.atomic_ptr.swap(null_mut(), order);
    if !ptr.is_null() {
      let b = unsafe { Box::from_raw(ptr) };
      drop(b);
    }
  }

  pub fn store_option(&self, b_option: Option<Box<T>>, order: Ordering) {
    match b_option {
      Some(b) => self.store(b, order),
      None => self.clear(order)
    }
  }

  pub fn fill(&self, b: Box<T>, order: Ordering) -> Option<Box<T>> {
    let ptr = unsafe { into_raw(b) };
    let old = self.atomic_ptr.compare_and_swap(null_mut(), ptr, order);
    if !old.is_null() {
      Some(unsafe { Box::from_raw(ptr) })
    } else {
      None
    }
  }

  pub fn swap(&self, b_option: Option<Box<T>>, order: Ordering) -> Option<Box<T>> {
    let ptr = match b_option {
      Some(b) => unsafe { into_raw(b) },
      None => null_mut()
    };
    let old_ptr = self.atomic_ptr.swap(ptr, order);
    if !old_ptr.is_null() {
      Some(unsafe { Box::from_raw(old_ptr) })
    } else {
      None
    }
  }
}
