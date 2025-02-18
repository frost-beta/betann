const max_value_i32 = 0x7fffffffi;
const min_value_i32 = -2147483648;
const max_value_u32 = 0xffffffffu;
const min_value_u32 = 0u;
const max_value_f32 = 0x1.fffffep+127f;
const min_value_f32 = -0x0.fffffep+127f;
if ($enable_f16) {
  const max_value_f16 = 0x1.ffcp+15h;
  const min_value_f16 = 0x0.ffcp+15h;
}

fn get_max_value_$dtype() -> $dtype {
  return max_value_$dtype;
}
