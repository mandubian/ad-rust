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


// a simple Array can be made a variable
// let d = Array::new(&d_input, d_dims);
// var!(x = array(d));
// Array can be cloned without cloning deep data
#[derive(Clone)]
pub struct Var<E> {
  pub name: &'static str,
  pub value: E,
}

// Expr typeclass representing the expression evaluation in a given shared context
pub trait Expr<Ctx, Value> {
  fn eval(& self, ctx: & Ctx) -> Value;
}

impl Expr<SharedCtx, SharedValue> for Var<Array> {
  fn eval(& self, ctx: & SharedCtx) -> SharedValue {
    Ctx::eval_arrayvar(ctx, self)
  }
}

impl<E : Clone + Expr<SharedCtx, SharedValue>> Expr<SharedCtx, SharedValue> for Var<E> {
  fn eval(& self, ctx: & SharedCtx) -> SharedValue {
    Ctx::eval_exprvar(ctx, self)
  }
}

pub type SharedValue = Rc<RefCell<Value>>;
pub type SharedCtx = Rc<RefCell<Ctx>>;

// The context of execution
pub struct Ctx {
  // the variables mapped to instantiated values by their name
  vars: HashMap<&'static str, SharedValue>,
  // all the instantiated values as ref-counted mutable refcells (vars are also values)
  vals: Vec<SharedValue>,
  count: usize,
}

impl Ctx {
  pub fn new() -> SharedCtx {
    Rc::new(RefCell::new(Ctx {
      vars : HashMap::new(),
      vals : vec![],
      count : 0,
    }))
  }

  pub fn get_next_idx(
    this: &SharedCtx,
  ) -> usize {
    let mut r = this.borrow_mut();
    let v = r.count;
    r.count += 1;
    v
  }

  pub fn find_var(
    this: &SharedCtx,
    name: &str,
  ) -> Option<SharedValue> {
    this.borrow().vars.get(name).map(|x| x.clone())
  }

  pub fn eval_arrayvar(
    this: &SharedCtx,
    var: &Var<Array>,
  ) -> SharedValue {
    match Self::find_var(this, var.name) {
      Some(v) => v.clone(),
      None    =>
        Self::new_var(
          this,
          VarType::VarV(var.name),
          var.value.clone(),
          vec![],
          vec![],
          Rc::new(|e| { () }),
        )
    }
  }

  pub fn eval_exprvar<E : Clone + Expr<SharedCtx, SharedValue>>(
    this: &SharedCtx,
    var: &Var<E>,
  ) -> SharedValue {
    match Self::find_var(this, var.name) {
      Some(v) => v.clone(),
      None    => {
        let ev = var.value.eval(this);
        let r = ev.borrow();
        Self::new_var(
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
    }
  }

  pub fn new_val(
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

  pub fn new_var(
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
pub enum VarType {
  VarV(&'static str),
  VarE(&'static str, usize),  
}

#[derive(Debug)]
pub enum ValueType {
  Var(VarType),
  Expr, 
}

pub struct Value {
  pub  value: Array,
  pub  idx: usize,
  pub tpe: ValueType,
  pub inputs: Vec<SharedValue>,
  ctx: SharedCtx,
  pub grads: Vec<Array>,
  pub grad_fun: Rc<Fn(&SharedValue) -> ()>
}

impl fmt::Display for Value {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "[idx:{}, tpe:{:?}]", self.idx, self.tpe)
  }
}

impl Value {
  pub fn add_grad(this: &SharedValue, grad: &Array) -> () {
    this.borrow_mut().grads.push(grad.clone());
  }

  pub fn add_grads(this: &SharedValue, grad: &Vec<Array>) -> () {
    this.borrow_mut().grads.extend(grad.clone());
  }

  pub fn dag(this: &SharedValue) -> Vec<SharedValue> {
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

  pub fn set_grad(this: &SharedValue, grads: Vec<Array>) -> () {
    this.borrow_mut().grads = grads;
  }

  pub fn compute_grad(this: &SharedValue) -> () {
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
  pub fn backprop(this: &SharedValue, force: bool) -> () {
    
    if !this.borrow().grads.is_empty() && !force { return; }

    let dims = this.borrow().value.dims();
    let ones = af::constant(1, dims);
    Value::add_grad(this, &ones);

    let dag = Value::dag(this);
    for e in dag.iter() {
      println!("backprop {} {:?}", e.borrow().idx, e.borrow().tpe);
      Value::compute_grad(e);
      (e.borrow().grad_fun)(e);
    }
  }

  pub fn grad_by_var(this: &SharedValue, var_name: &str) -> Option<Array> {
    Value::backprop(this, false);
    Ctx::find_var(&this.borrow().ctx, var_name).map(|var| {
      if var.borrow().grads.len() > 0 {
        var.borrow().grads[0].clone()
      }
      else {
        let dims = var.borrow().value.dims();
        af::constant(0, dims)
      }
    })
  }
}


pub trait Gradient<T> {
  fn grad(&self, var: &T) -> Option<Array>;
}

impl Gradient<String> for SharedValue {
  fn grad(&self, var: &String) -> Option<Array> {
    Value::grad_by_var(self, var)
  }
}

impl<E : Clone> Gradient<Var<E>> for SharedValue {
  fn grad(&self, var: &Var<E>) -> Option<Array> {
    Value::grad_by_var(self, var.name)
  }
}

