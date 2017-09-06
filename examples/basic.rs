#[macro_use(af_print)]
extern crate arrayfire as af;

use af::*;

#[allow(unused_must_use)]
fn main() {
  // set_backend(Backend::CPU);
  set_device(0);
  info();
}