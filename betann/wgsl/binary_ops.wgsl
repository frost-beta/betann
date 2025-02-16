fn add(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a + b);
}

if (!$input_is_floating) {
  fn bitwise_and(a: input_dtype, b: input_dtype) -> output_dtype {
    return output_dtype(a & b);
  }

  fn bitwise_or(a: input_dtype, b: input_dtype) -> output_dtype {
    return output_dtype(a | b);
  }

  fn bitwise_xor(a: input_dtype, b: input_dtype) -> output_dtype {
    return output_dtype(a ^ b);
  }
}

fn divide(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a / b);
}

fn equal(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a == b);
}

fn greater(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a > b);
}

fn greater_equal(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a >= b);
}

if ($input_is_integer) {
  fn left_shift(a: input_dtype, b: input_dtype) -> output_dtype {
    return output_dtype(a << u32(b));
  }
}

fn less(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a < b);
}

fn less_equal(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a <= b);
}

fn log_add_exp(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(log(1 + exp(f32(a + b))));
}

fn logical_and(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(bool(a) && bool(b));
}

fn logical_or(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(bool(a) || bool(b));
}

fn maximum(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(max(a, b));
}

fn minimum(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(min(a, b));
}

fn multiply(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a * b);
}

fn not_equal(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a != b);
}

fn power(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(pow(f32(a), f32(b)));
}

fn subtract(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a - b);
}

if ($input_is_integer) {
  fn right_shift(a: input_dtype, b: input_dtype) -> output_dtype {
    return output_dtype(a >> u32(b));
  }
}

fn remainder(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(a % b);
}

fn arc_tan2(a: input_dtype, b: input_dtype) -> output_dtype {
  return output_dtype(atan2(f32(a), f32(b)));
}
