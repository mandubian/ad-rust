
macro_rules! var {
  ($var:ident = $value:expr) => (
    let $var = Var { name: stringify!($var), value: $value };
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
    $e
    let after = time::precise_time_ns();
    println!("duration: {}", (after - before));
  );

  ($e : expr)=> (
    let before = time::precise_time_ns();
    $e
    let after = time::precise_time_ns();
    println!("duration: {}", (after - before));
  );

  ($e : block)=> (
    let before = time::precise_time_ns();
    $e
    let after = time::precise_time_ns();
    println!("duration: {}ns", (after - before));
  );
}

macro_rules! cond {
  (if $lhs:tt < $rhs:tt then $then:block else $else:block) => ({
    IfThenElse { if_ : &LT { lhs: &$lhs, rhs: &$rhs }, then_ : $then, else_ : $else }
  });
}