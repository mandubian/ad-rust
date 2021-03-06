extern crate arrayfire as af;
extern crate time;

use super::*;
use self::af::*;
use self::trigo;

#[test]
fn test_conditional_op() {
  // choose a ArrayFire Backend (could be CPU too or any GPU you have in your computer)
  af::set_backend(af::Backend::OPENCL);
  af::info();

  let d_dims = Dim4::new(&[1, 6, 1, 1]);
  let d_input: [i32; 6] = [1, 2, 3, 4, 5, 6];
  let d_input2: [i32; 6] = [2, 4, 6, 8, 10, 12];
  let d = Array::new(&d_input, d_dims);
  let d2 = Array::new(&d_input2, d_dims);
  let d3 = Array::new(&d_input, d_dims);

  /// Here is the program to differentiate (with a conditional expression in it)
  /// This is equivalent to:
  /// ```
  /// x = d;
  /// y = d2;
  /// z = d3;
  /// a = &x + &y;
  /// t = trigo::sin(&z);
  /// u = &x + &t;
  /// v = if y < z then { &x } else { &u };
  /// ```
  /// Macros are required because we can't override assignment operator (=) & if/then/else constructions
  /// which is sad because many operators are overridable...
  var!(x = d);
  var!(y = d2);
  var!(z = d3);
  var!(a = &x + &y);
  var!(t = trigo::sin(&a));
  var!(u = &x + &t);
  var!(v = cond!(if y < z then { &x } else { &u }));

  // The Context of execution containing the mutable graph of evaluated values & gradients (aka Wengert Tape in theory)
  let ctx = Ctx::new();

  // let's evaluate in forward mode
  let ev = v.eval(&ctx);

  // let's compute partial gradients per variable x (by reverse diff aka back propagation)
  timed!(let dx  = d!(ev / d(x)).unwrap());
  af_print!("d/dx:", dx); 

  // produces
  // d/dx:
  // [1 6 1 1]
  // 0.0100     1.9602     0.0889     1.8439     0.2403     1.6603

  // let's compute partial gradients per variable y (should reuse previously computed values)
  timed!(let dy  = d!(ev / d(y)).unwrap());
  af_print!("d/dy:", dy); 

  // produces
  // d/dy:
  // [1 6 1 1]
  // -0.9900     0.9602    -0.9111     0.8439    -0.7597     0.6603

  // let's compute partial gradients per variable z that should be 0 as u doesn't depend on z
  timed!(let dz  = d!(ev / d(z)).unwrap());
  af_print!("d/dz:", dz); 

  // Produces:
  // d/dz:
  // [1 6 1 1]
  //          0          0          0          0          0          0
}

/*
#[test]
fn test_basic_op() {
  af::set_backend(af::Backend::OPENCL);
  af::info();

  let d_dims = Dim4::new(&[1, 6, 1, 1]);
  let d_input: [i32; 6] = [1, 2, 3, 4, 5, 6];
  let d_input2: [i32; 6] = [2, 4, 6, 8, 10, 12];
  let d = Array::new(&d_input, d_dims);
  let d2 = Array::new(&d_input2, d_dims);

  let ctx = Ctx::new();

  var!(x = d);
  var!(y = d2);
  var!(z = &x + &y);
  var!(t = trigo::sin(&z));
  var!(u = &x + &t);

  let ev = u.eval(&ctx);

  timed!(let dz  = d!(ev / d(z)).unwrap());
  af_print!("d/dz:", dz);    

  // second time, it should be much faster
  timed!(let dz  = d!(ev / d(z)).unwrap());
  af_print!("d/dz:", dz); 

  timed!(let dz  = d!(ev / d(z)).unwrap());
  af_print!("d/dz:", dz); 

  let dx = d!(ev / d(x)).unwrap();
  af_print!("d/dx:", dx);    

  let dy = d!(ev / d(y)).unwrap();
  af_print!("d/dy:", dy);    

  let dt = d!(ev / d(t)).unwrap();
  af_print!("d/dt:", dt);

}
*/