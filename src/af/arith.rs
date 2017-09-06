extern crate libc;
extern crate arrayfire as af;
extern crate time;

// use std::marker::PhantomData;

use self::af::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::ops;

use af::autograd::*;

#[derive(Clone)]
pub struct Add<'a, E1 : 'a, E2 : 'a> {
  lhs: &'a E1,
  rhs: &'a E2,
}

impl<'a, E1 : 'a + Expr<SharedCtx, SharedValue>, E2: 'a + Expr<SharedCtx, SharedValue>> Expr<SharedCtx, SharedValue> for Add<'a, E1, E2> {
  fn eval(&self, ctx: & SharedCtx) -> SharedValue {
    let l = self.lhs.eval(ctx);
    let r = self.rhs.eval(ctx);

    let lv = &l.borrow().value;
    let rv = &r.borrow().value;
    Ctx::new_val(
      ctx,
      ValueType::Expr,
      add(lv, rv, false),
      vec![l.clone(), r.clone()],
      vec![],
      Rc::new(|e| {
        let ev = e.borrow();
        Value::add_grad(&ev.inputs[0], &ev.grads[0]);
        Value::add_grad(&ev.inputs[1], &ev.grads[0]);
      })
    )
  } 
}

impl<'a, E1 : 'a + Clone, E2: 'a + Clone> ops::Add<&'a Var<E2>> for &'a Var<E1> {
    type Output = Add<'a, Var<E1>, Var<E2>>;

    fn add(self, other: &'a Var<E2>) -> Add<'a, Var<E1>, Var<E2>> {
      Add { lhs: self, rhs: other }
    }
}

#[derive(Clone)]
pub struct LT<'a, E1 : 'a, E2 : 'a> {
  pub lhs: &'a E1,
  pub rhs: &'a E2,
}

impl<'a, E1 : 'a + Expr<SharedCtx, SharedValue>, E2: 'a + Expr<SharedCtx, SharedValue>> Expr<SharedCtx, SharedValue> for LT<'a, E1, E2> {
  fn eval(&self, ctx: & SharedCtx) -> SharedValue {
    let l = self.lhs.eval(ctx);
    let r = self.rhs.eval(ctx);

    let lv = &l.borrow().value;
    let rv = &r.borrow().value;
    Ctx::new_val(
      ctx,
      ValueType::Expr,
      lt(lv, rv, false),
      vec![l.clone(), r.clone()],
      vec![],
      Rc::new(|e| { })
    )
  } 
}

#[derive(Clone)]
pub struct IfThenElse<'a, If : 'a, Then : 'a, Else : 'a> {
  pub if_: &'a If,
  pub then_: &'a Then,
  pub else_: &'a Else,  
}

impl< 'a,
      If : 'a + Expr<SharedCtx, SharedValue>,
      Then: 'a + Expr<SharedCtx, SharedValue>,
      Else: 'a + Expr<SharedCtx, SharedValue>
    > Expr<SharedCtx, SharedValue> for IfThenElse<'a, If, Then, Else> {
  fn eval(&self, ctx: & SharedCtx) -> SharedValue {
    let if_ = self.if_.eval(ctx);
    let ifv = &if_.borrow().value;
    // if true, eval then_ only & propagate only then_ grads

    if all_true_all(ifv).0 == 1f64 {
      let then_ = self.then_.eval(ctx);
      let thenv = &then_.borrow().value;
      af_print!("thenv:", thenv);
      Ctx::new_val(
        ctx,
        ValueType::Expr,
        thenv.clone(),
        vec![then_.clone()],
        vec![],
        Rc::new(|e| {
          let ev = e.borrow();
          Value::add_grads(&ev.inputs[0], &ev.grads);
        })
      )
    }
    // else propagate else_
    else {
      let else_ = self.else_.eval(ctx);
      let elsev = &else_.borrow().value;
      af_print!("elsev:", elsev);

      Ctx::new_val(
        ctx,
        ValueType::Expr,
        elsev.clone(),
        vec![else_.clone()],
        vec![],
        Rc::new(|e| {
          let ev = e.borrow();
          Value::add_grads(&ev.inputs[0], &ev.grads);
        })
      )
    }

  } 
}
