extern crate libc;
extern crate arrayfire as af;
extern crate time;

// use std::marker::PhantomData;

use self::af::*;
use std::rc::Rc;
use std::cell::RefCell;

use af::autograd::*;

#[derive(Clone)]
pub struct Sin<'a, E : 'a> {
  x: &'a E,
}

impl<'a, E : 'a> Sin<'a, E> {
  fn new(x: &'a E) -> Sin<'a, E> {
    Sin { x : x }
  }
}

impl<'a, E : 'a + Expr<SharedCtx, SharedValue>> Expr<SharedCtx, SharedValue> for Sin<'a, E> {
  fn eval(&self, ctx: & SharedCtx) -> SharedValue {
    let xe = self.x.eval(ctx);

    let xev = &xe.borrow().value;
    Ctx::new_val(
      ctx,
      ValueType::Expr,
      af::sin(&xev.clone()),
      vec![xe.clone()],
      vec![],
      Rc::new(|e| {
        let ev = e.borrow();
        let grad = &ev.grads[0].clone();
        let input = &ev.inputs[0].borrow().value.clone();
        Value::add_grad(&ev.inputs[0], &(grad * cos(input)));
      })
    )
  } 
}

/// Sine function
pub fn sin<'a, E : 'a + Clone>(s: &'a E) -> Sin<'a, E> {
  Sin::new(&s)
}