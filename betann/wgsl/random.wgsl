override num_threads: u32 = 8;

// WGSL does not have byte type so |out| is represented as an array of 4-bytes,
// which requires the caller to align |out| to 4-bytes.
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> bytes_per_key: u32;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
if (!$contiguous) {
  @group(0) @binding(3) var<storage, read> keys_shape: array<u32>;
  @group(0) @binding(4) var<storage, read> keys_strides: array<u32>;
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn rbits(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&keys)) {
    return;
  }
  let out_per_key = (bytes_per_key + 4 - 1) / 4;
  let half_size = out_per_key / 2;
  let odd = out_per_key % 2;
  if (gid.y >= half_size + odd) {
    return;
  }
  if ($contiguous) {
    let key = vec2<u32>(keys[2 * gid.x], keys[2 * gid.x + 1]);
  } else {
    let k1_idx = coord_to_index(2 * gid.x, &keys_shape, &keys_strides);
    let k2_idx = coord_to_index(2 * gid.x + 1, &keys_shape, &keys_strides);
    let key = vec2<u32>(keys[k1_idx], keys[k2_idx]);
  }
  let drop_last = odd == 1 && gid.y == half_size;
  let bits = threefry2x32_hash(
      key,
      vec2<u32>(gid.y, select(gid.y + half_size + odd, 0, drop_last)));
  if (bytes_per_key % 4 == 0) {
    out[gid.x * bytes_per_key / 4 + gid.y] = bits[0];
  } else {
    let idx = gid.x * bytes_per_key + gid.y * 4;
    for (var i: u32 = 0; i < 4; i++) {
      set_nth_byte_in_out(idx + i, nth_byte(bits[0], i));
    }
  }
  if (!drop_last) {
    if ((gid.y == half_size - 1) && (bytes_per_key % 4 > 0)) {
      let idx = gid.x * bytes_per_key + (gid.y + half_size + odd) * 4;
      let edge_bytes = bytes_per_key % 4;
      for (var i: u32 = 0; i < edge_bytes; i++) {
        set_nth_byte_in_out(idx + i, nth_byte(bits[1], i));
      }
    } else {
      out[gid.x * bytes_per_key / 4 + gid.y + half_size + odd] = bits[1];
    }
  }
}

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

// include utils.wgsl
