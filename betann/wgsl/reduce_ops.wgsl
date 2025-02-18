// include constants.wgsl

// The initial values for reduction ops.
const initial_value_and = output_dtype(1);
const initial_value_or = output_dtype(0);
const initial_value_sum = output_dtype(0);
const initial_value_product = output_dtype(1);
const initial_value_min = max_value_$output_dtype;
const initial_value_max = min_value_$output_dtype;

fn get_initial_value_$op() -> output_dtype {
  return initial_value_$op;
}

// The reduction ops.
fn reduce_op_and(a: output_dtype, b: output_dtype) -> output_dtype {
  return output_dtype(bool(a) && bool(b));
}

fn reduce_op_or(a: output_dtype, b: output_dtype) -> output_dtype {
  return output_dtype(bool(a) || bool(b));
}

fn reduce_op_sum(a: output_dtype, b: output_dtype) -> output_dtype {
  return a + b;
}

fn reduce_op_product(a: output_dtype, b: output_dtype) -> output_dtype {
  return a * b;
}

fn reduce_op_min(a: output_dtype, b: output_dtype) -> output_dtype {
  return min(a, b);
}

fn reduce_op_max(a: output_dtype, b: output_dtype) -> output_dtype {
  return max(a, b);
}
