extern crate libc;
extern crate arrayfire as af;
extern crate time;

// use std::marker::PhantomData;

use self::af::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops;

// required to implement typeclasses due to orphan typeclass rule
#[derive(Clone)]
struct Wrapper<E: Clone + Expr> {
  inner: E
}

impl<E: Clone + Expr> Wrapper<E> {
  pub fn new(e: E) -> Self {
    Wrapper {
      inner: e,
    }
  }
}

// a simple Array can be made a variable
// let d = Array::new(&d_input, d_dims);
// var!(x = array(d));
// Array can be cloned without cloning deep data
#[derive(Clone)]
struct ArrayVar {
  name: &'static str,
  value: Array,
}

// an expression can be made a variable
// var!(e = expr(&a + &b));
#[derive(Clone)]
struct ExprVar<E : Clone + Expr> {
  name: &'static str,
  value: E,
}

// Expr typeclass representing the expression evaluation in a given shared context
trait Expr {
  fn eval(& self, ctx: & SharedCtx) -> SharedValue;
}

impl Expr for ArrayVar {
  fn eval(& self, ctx: & SharedCtx) -> SharedValue {
    Ctx::eval_arrayvar(ctx, self)
  }
}

impl<E : Clone + Expr> Expr for ExprVar<E> {
  fn eval(& self, ctx: & SharedCtx) -> SharedValue {
    Ctx::eval_exprvar(ctx, self)
  }
}

type SharedValue = Rc<RefCell<Value>>;
type SharedCtx = Rc<RefCell<Ctx>>;

// The context of execution
struct Ctx {
  // the variables mapped to instantiated values by their name
  vars: HashMap<&'static str, SharedValue>,
  // all the instantiated values as ref-counted mutable refcells (vars are also values)
  vals: Vec<SharedValue>,
  count: usize,
}

impl Ctx {
  fn new() -> SharedCtx {
    Rc::new(RefCell::new(Ctx {
      vars : HashMap::new(),
      vals : vec![],
      count : 0,
    }))
  }

  fn get_next_idx(
    this: &SharedCtx,
  ) -> usize {
    let mut r = this.borrow_mut();
    let v = r.count;
    r.count += 1;
    v
  }

  fn find_var(
    this: &SharedCtx,
    name: &str,
  ) -> Option<SharedValue> {
    this.borrow().vars.get(name).map(|x| x.clone())
  }

  fn eval_arrayvar(
    this: &SharedCtx,
    var: &ArrayVar,
  ) -> SharedValue {
    Ctx::new_var(
      this,
      VarType::VarV(var.name),
      var.value.clone(),
      vec![],
      vec![],
      Rc::new(|e| { () }),
    )
  }

  fn eval_exprvar<E : Clone + Expr>(
    this: &SharedCtx,
    var: &ExprVar<E>,
  ) -> SharedValue {
    let ev = var.value.eval(this);
    let r = ev.borrow();
    Ctx::new_var(
      this,
      VarType::VarE(var.name, r.idx),
      r.value.clone(),
      vec![ev.clone()],
      vec![],
      Rc::new(|e| {
        let ev = e.borrow();
        println!("grad_fun add {} {:?} {}", ev.idx, ev.tpe, ev.grads.len());
        Value::add_grads(&ev.inputs[0], &ev.grads);
      }),
    )
  }

  fn new_val(
    this: &SharedCtx,
    tpe: ValueType,
    value: Array,
    inputs: Vec<SharedValue>,
    grads: Vec<Array>,
    grad_fun: Rc<Fn(&SharedValue) -> ()>,
  ) -> SharedValue {
    let idx = Ctx::get_next_idx(this);
    let v = Rc::new(RefCell::new(Value {
      value : value,
      idx : idx,
      tpe : tpe,
      inputs: inputs,
      ctx : this.clone(),
      grads: grads,
      grad_fun: grad_fun,
    }));

    let mut rthis = this.borrow_mut();
    rthis.vals.push(v.clone());
    v
  }

  fn new_var(
    this: &SharedCtx,
    tpe: VarType,
    value: Array,
    inputs: Vec<SharedValue>,
    grads: Vec<Array>,
    grad_fun: Rc<Fn(&SharedValue) -> ()>,
  ) -> SharedValue {
    let name = match tpe {
      VarType::VarV(name) => name,
      VarType::VarE(name, _) => name
    };
    // ugly way to use immutable & mutable &self :(
    let mut result: Option<SharedValue> = None;
    if let Some(v) = this.borrow().vars.get(name) {
      println!("already found variable {}", name);
      result = Some(v.clone())
    }
    if let Some(v) = result {
      v
    } else {
      println!("Inserting variable {}", name);
      let v = Ctx::new_val(this, ValueType::Var(tpe), value, inputs, grads, grad_fun);
      this.borrow_mut().vars.insert(name, v.clone());
      v.clone()
    }
  }
}

#[derive(Debug)]
enum VarType {
  VarV(&'static str),
  VarE(&'static str, usize),  
}

#[derive(Debug)]
enum ValueType {
  Var(VarType),
  Expr, 
}

struct Value {
  value: Array,
  idx: usize,
  tpe: ValueType,
  inputs: Vec<SharedValue>,
  ctx: SharedCtx,
  grads: Vec<Array>,
  grad_fun: Rc<Fn(&SharedValue) -> ()>
}

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "[idx:{}, tpe:{:?}]", self.idx, self.tpe)
  }
}

impl Value {
  fn add_grad(this: &SharedValue, grad: &Array) -> () {
    this.borrow_mut().grads.push(grad.clone());
  }

  fn add_grads(this: &SharedValue, grad: &Vec<Array>) -> () {
    this.borrow_mut().grads.extend(grad.clone());
  }

  fn dag(this: &SharedValue) -> Vec<SharedValue> {
    let mut m = HashMap::new();
    let mut dag = Vec::new();
    {
      let mut stack: Vec<SharedValue> = vec![this.clone()];
      while let Some(e) = stack.pop() {
        let ev = e.borrow();
        if !m.contains_key(&ev.idx) {      
          for input in ev.inputs.iter() {
            stack.push(input.clone());
          }
          m.insert(ev.idx, true);
          // println!("dag add {} {:?}", ev.idx, ev.tpe);
          dag.push(e.clone());
        } else {
          // println!("dag already found {} {:?}", ev.idx, ev.tpe);
        }
      }
    }

    dag
  }

  fn set_grad(this: &SharedValue, grads: Vec<Array>) -> () {
    this.borrow_mut().grads = grads;
  }

  fn compute_grad(this: &SharedValue) -> () {
    let mut grad = this.borrow().grads[0].clone();
    if this.borrow().grads.len() > 1 {
      for arr in this.borrow().grads.split_at(1).1.iter() {
        grad = add(&grad, arr, false);
      }
    }
    // println!("compute_grad {} {:?}", this.borrow().idx, this.borrow().tpe);
    // af_print!("compute_grad grad:", grad);
    Value::set_grad(this, vec![grad]);
  }

  // computes backprop on mutable graph
  fn backprop(this: &SharedValue, force: bool) -> () {
    
    if !this.borrow().grads.is_empty() && !force { return; }

    let dims = this.borrow().value.dims();
    let ones = af::constant(1, dims);
    Value::add_grad(this, &ones);

    let dag = Value::dag(this);
    for e in dag.iter() {
      // println!("backprop {} {:?}", e.borrow().idx, e.borrow().tpe);
      Value::compute_grad(e);
      (e.borrow().grad_fun)(e);
    }
  }

  fn grad_by_var(this: &SharedValue, var_name: &str) -> Option<Array> {
    Value::backprop(this, false);
    Ctx::find_var(&this.borrow().ctx, var_name).map(|var| {
      var.borrow().grads[0].clone()
    })
  }
}


trait Gradient<T> {
  fn grad(&self, var: &T) -> Option<Array>;
}

impl Gradient<String> for SharedValue {
  fn grad(&self, var: &String) -> Option<Array> {
    Value::grad_by_var(self, var)
  }
}

impl Gradient<ArrayVar> for SharedValue {
  fn grad(&self, var: &ArrayVar) -> Option<Array> {
    Value::grad_by_var(self, var.name)
  }
}

impl<E : Clone + Expr> Gradient<ExprVar<E>> for SharedValue {
  fn grad(&self, var: &ExprVar<E>) -> Option<Array> {
    Value::grad_by_var(self, var.name)
  }
}

impl Gradient<Wrapper<ArrayVar>> for SharedValue {
  fn grad(&self, var: &Wrapper<ArrayVar>) -> Option<Array> {
    Value::grad_by_var(self, var.inner.name)
  }
}

impl<E : Clone + Expr> Gradient<Wrapper<ExprVar<E>>> for SharedValue {
  fn grad(&self, var: &Wrapper<ExprVar<E>>) -> Option<Array> {
    Value::grad_by_var(self, var.inner.name)
  }
}

impl<E, T : Clone + Expr + Gradient<E>> Gradient<E> for Wrapper<T> {
  fn grad(&self, var: &E) -> Option<Array> {
    self.inner.grad(var)
  }
}

#[derive(Clone)]
struct Add<'a, E1 : 'a + Expr, E2: 'a + Expr> {
  lhs: &'a E1,
  rhs: &'a E2,
}

impl<'a, E1 : 'a + Expr, E2: 'a + Expr> Expr for Add<'a, E1, E2> {
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

#[derive(Clone)]
struct Sin<'a, E : 'a + Expr> {
  x: &'a E,
}

impl<'a, E : 'a + Expr> Sin<'a, E> {
  fn new(x: &'a E) -> Sin<'a, E> {
    Sin { x : x }
  }
}

impl<'a, E : 'a + Expr> Expr for Sin<'a, E> {
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
fn Sin<'a, E : 'a + Clone + Expr>(s: &'a Wrapper<E>) -> Wrapper<Sin<'a, E>> {
  Wrapper::new(Sin::new(&s.inner))
}

impl <E: Clone + Expr> Expr for Wrapper<E> {
  fn eval(&self, ctx: & SharedCtx) -> SharedValue {
    self.inner.eval(ctx)
  }
}

impl<'a, E1 : 'a + Clone + Expr, E2: 'a + Clone + Expr> ops::Add<&'a Wrapper<E2>> for &'a Wrapper<E1> {
    type Output = Wrapper<Add<'a, E1, E2>>;

    fn add(self, other: &'a Wrapper<E2>) -> Wrapper<Add<'a, E1, E2>> {
      Wrapper { inner : Add { lhs: &self.inner, rhs: &other.inner } }
    }
}

macro_rules! var {
  ($var:ident = array($value:expr)) => (
    let $var = Wrapper::new(ArrayVar { name: stringify!($var), value: $value });
  );

  ($var:ident = expr($value:expr)) => (
    let $var = Wrapper::new(ExprVar { name: stringify!($var), value: $value });
  );
}

macro_rules! d {
  ($value:ident /d($var:ident)) => (
    $value.grad(&$var)
  );
}

macro_rules! timed {

  ($e : stmt)=> (
    let before = time::precise_time_ns();
    $e;
    let after = time::precise_time_ns();
    println!("duration: {}", (after - before));
  );

  ($e : expr)=> (
    let before = time::precise_time_ns();
    $e;
    let after = time::precise_time_ns();
    println!("duration: {}", (after - before));
  );

  ($e : block)=> (
    let before = time::precise_time_ns();
    $e;
    let after = time::precise_time_ns();
    println!("duration: {}ns", (after - before));
  );
}

#[cfg(test)]
mod tests {
  use super::*;
  use super::af::*;

  // use super::functions::*;

  // use num::Float;

  // #[test]
  fn test_basic_op() {
    af::set_backend(af::Backend::OPENCL);
    af::info();

    let d_dims = Dim4::new(&[1, 6, 1, 1]);
    let d_input: [i32; 6] = [1, 2, 3, 4, 5, 6];
    let d = Array::new(&d_input, d_dims);
    let d2 = Array::new(&d_input, d_dims);

    let mut ctx = Ctx::new();
    
    var!(x = array(d));
    var!(y = array(d2));
    var!(z = expr(&x + &y));
    var!(t = expr(Sin(&z)));
    var!(u = expr(&x + &t));

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
}
