if ($input_is_floating) {
  alias return_dtype = input_dtype;
} else {
  alias return_dtype = f32;
}

fn betann_abs(input: input_dtype) -> input_dtype {
  return abs(input);
}

fn betann_acos(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return acos(input);
  } else {
    return acos(f32(input));
  }
}

fn betann_acosh(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return acosh(input);
  } else {
    return acosh(f32(input));
  }
}

fn betann_asin(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return asin(input);
  } else {
    return asin(f32(input));
  }
}

fn betann_asinh(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return asinh(input);
  } else {
    return asinh(f32(input));
  }
}

fn betann_atan(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return atan(input);
  } else {
    return atan(f32(input));
  }
}

fn betann_atanh(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return atanh(input);
  } else {
    return atanh(f32(input));
  }
}

if (!$input_is_floating) {
  fn betann_bitwise_invert(input: input_dtype) -> return_dtype {
    if ($input_is_bool) {
      return return_dtype(!bool(input));
    } else {
      return return_dtype(~input);
    }
  }
}

fn betann_ceil(input: input_dtype) -> input_dtype {
  if ($input_is_floating) {
    return ceil(input);
  } else {
    return input;
  }
}

fn betann_cos(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return cos(input);
  } else {
    return cos(f32(input));
  }
}

fn betann_cosh(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return cosh(input);
  } else {
    return cosh(f32(input));
  }
}

const r0 =  0.3275911;
const r1 =  0.254829592;
const r2 = -0.284496736;
const r3 =  1.421413741;
const r4 = -1.453152027;
const r5 =  1.0615429;

fn betann_erf(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    let v = input;
  } else {
    let v = f32(input);
  }
  let absv = abs(v);
  let x = 1 / (1 + r0 * absv);
  return sign(v) * (1 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-absv * absv));
}

fn betann_erf_inv(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    let v = input;
  } else {
    let v = f32(input);
  }
  let w = -log(( 1 - v) * ( 1 + v));
  let p = select((((r5 * w + r4) * w + r3) * w + r2) * w + r1,
                 ((((r0 * w + r1) * w + r2) * w + r3) * w + r4) * w + r5,
                 w < 6.5);
  return sign(v) * sqrt(p);
}

fn betann_exp(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return exp(input);
  } else {
    return exp(f32(input));
  }
}

fn betann_expm1(input: input_dtype) -> return_dtype {
  return betann_exp(input) - 1;
}

fn betann_floor(input: input_dtype) -> input_dtype {
  if ($input_is_floating) {
    return floor(input);
  } else {
    return input;
  }
}

fn betann_log(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return log(input);
  } else {
    return log(f32(input));
  }
}

fn betann_log2(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return log2(input);
  } else {
    return log2(f32(input));
  }
}

fn betann_log10(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return log(input) / log(input_dtype(10));
  } else {
    return log(f32(input)) / log(10f);
  }
}

fn betann_log1p(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return log(1 + input);
  } else {
    return log(f32(1 + input));
  }
}

fn betann_logical_not(input: input_dtype) -> bool {
  return !bool(input);
}

fn betann_negative(input: input_dtype) -> input_dtype {
  return 0 - input;
}

fn betann_round(input: input_dtype) -> input_dtype {
  if ($input_is_floating) {
    return round(input);
  } else {
    return input;
  }
}

fn betann_rsqrt(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return inverseSqrt(input);
  } else {
    return inverseSqrt(f32(input));
  }
}

fn betann_sigmoid(input: input_dtype) -> return_dtype {
  let y = 1 / (1 + betann_exp(betann_negative(input)));
  return select(y, 1 - y, input < 0);
}

fn betann_sign(input: input_dtype) -> input_dtype {
  if ($input_is_unsigned) {
    return input_dtype(select(1, 0, input == 0));
  } else {
    return sign(input);
  }
}

fn betann_sin(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return sin(input);
  } else {
    return sin(f32(input));
  }
}

fn betann_sinh(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return sinh(input);
  } else {
    return sinh(f32(input));
  }
}

fn betann_sqrt(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return sqrt(input);
  } else {
    return sqrt(f32(input));
  }
}

fn betann_square(input: input_dtype) -> input_dtype {
  return input * input;
}

fn betann_tan(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return tan(input);
  } else {
    return tan(f32(input));
  }
}

fn betann_tanh(input: input_dtype) -> return_dtype {
  if ($input_is_floating) {
    return tanh(input);
  } else {
    return tanh(f32(input));
  }
}
