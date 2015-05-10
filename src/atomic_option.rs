use std::ptr::*;
use std::sync::atomic::*;
use std::boxed::*;
use std::marker::Send;

#[unsafe_no_drop_flag]
pub struct AtomicOption<T: Send> {
    atomic_ptr: AtomicPtr<T>
}

impl<T: Send> Drop for AtomicOption<T> {
  fn drop(&mut self) {
    self.clear(Ordering::Relaxed);
  }
}

impl<T: Send> AtomicOption<T> {
  /// Create a new `AtomicOption`.
  #[inline]
  pub fn new(b: Box<T>) -> AtomicOption<T> {
    AtomicOption {
      atomic_ptr: AtomicPtr::new( unsafe { into_raw(b) } )
    }
  }

  /// Create a new `AtomicOption` that doesn't contain a value.
  #[inline]
  pub fn empty() -> AtomicOption<T> {
    AtomicOption {
      atomic_ptr: AtomicPtr::new(null_mut())
    }
  }

  #[inline]
  pub unsafe fn load<'a>(&'a self, order: Ordering) -> Option<&'a T> {
    let ptr = self.atomic_ptr.load(order);
    if !ptr.is_null() {
      Some(&*ptr)
    } else {
      None
    }
  }

  /// Remove the value, leaving the `AtomicOption` empty.
  #[inline]
  pub fn take(&self, order: Ordering) -> Option<Box<T>> {
    let ptr = self.atomic_ptr.swap(null_mut(), order);
    if !ptr.is_null() {
      Some(unsafe { Box::from_raw(ptr) })
    } else {
      None
    }
  }

  #[inline]
  pub fn store(&self, b: Box<T>, order: Ordering) {
    let ptr = self.atomic_ptr.swap(unsafe { into_raw(b) }, order);
    if !ptr.is_null() {
      let b = unsafe { Box::from_raw(ptr) };
      drop(b);
    }
  }

  #[inline]
  pub fn clear(&self, order: Ordering) {
    let ptr = self.atomic_ptr.swap(null_mut(), order);
    if !ptr.is_null() {
      let b = unsafe { Box::from_raw(ptr) };
      drop(b);
    }
  }

  #[inline]
  pub fn store_option(&self, b_option: Option<Box<T>>, order: Ordering) {
    match b_option {
      Some(b) => self.store(b, order),
      None => self.clear(order)
    }
  }

  /// Replace an empty value with a non-empty value.
  ///
  /// Succeeds if the option is `None` and returns `None` if so. If
  /// the option was already `Some`, returns `Some` of the rejected
  /// value.
  #[inline]
  pub fn fill(&self, b: Box<T>, order: Ordering) -> Option<Box<T>> {
    let ptr = unsafe { into_raw(b) };
    let old = self.atomic_ptr.compare_and_swap(null_mut(), ptr, order);
    if !old.is_null() {
      Some(unsafe { Box::from_raw(ptr) })
    } else {
      None
    }
  }

  /// Store a value, returning the old value.
  #[inline]
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
