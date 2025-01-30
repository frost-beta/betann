const rotations = array<vec4<u32>, 2>(vec4<u32>(13, 15, 26, 6),
                                      vec4<u32>(17, 29, 16, 24));

fn threefry2x32_hash(key: vec2<u32>, count: vec2<u32>) -> vec2<u32> {
  var ks = vec4<u32>(key.x, key.y, key.x ^ key.y ^ 0x1BD11BDAu, 0u);

  var v: vec2<u32>;
  v.x = count.x + ks[0];
  v.y = count.y + ks[1];

  for (var i: u32 = 0; i < 5; i++) {
    for (var j: u32 = 0; j < 4; j++) {
      let r = rotations[i % 2][j];
      v.x += v.y;
      v.y = (v.y << r) | (v.y >> (32 - r));
      v.y ^= v.x;
    }
    v.x += ks[(i + 1) % 3];
    v.y += ks[(i + 2) % 3] + i + 1;
  }

  return v;
}

fn nth_byte(value: u32, n: u32) -> u32 {
  return (value >> (8 * n)) & 0xFFu;
}

fn set_nth_byte_in_u32(original: u32, n: u32, byte: u32) -> u32 {
  let shift = n * 8;
  let mask = ~(0xFFu << shift);
  return (original & mask) | ((byte & 0xFFu) << shift);
}

fn set_nth_byte_in_out(n: u32, byte: u32) {
  let index = n / 4;
  out[index] = set_nth_byte_in_u32(out[index], n % 4, byte);
}
