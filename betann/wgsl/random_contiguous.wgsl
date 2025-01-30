override num_threads: u32 = 8;

// WGSL does not have byte type so |out| is represented as an array of 4-bytes,
// which requires the caller to align |out| to 4-bytes.
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> bytes_per_key: u32;

@compute @workgroup_size(num_threads, num_threads, 1)
fn rbits_c(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= arrayLength(&keys)) {
    return;
  }
  let out_per_key = (bytes_per_key + 4 - 1) / 4;
  let half_size = out_per_key / 2;
  let odd = out_per_key % 2;
  if (gid.y >= half_size + odd) {
    return;
  }
  let key = vec2<u32>(keys[2 * gid.x], keys[2 * gid.x + 1]);
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

// include random_ops.wgsl
