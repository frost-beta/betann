alias dst_dtype = $0;
alias src_dtype = $1;

override num_threads: u32 = 8;

@group(0) @binding(0) var<storage, read_write> dst: array<dst_dtype>;
@group(0) @binding(1) var<storage, read> src: array<src_dtype>;
@group(0) @binding(2) var<storage, read> src_shape: array<u32>;
@group(0) @binding(3) var<storage, read> src_strides: array<u32>;

@compute @workgroup_size(num_threads, 1, 1)
fn copy_g1(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[0]) {
    return;
  }
  let src_idx = gid.x * src_strides[0];
  dst[gid.x] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn copy_g2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[1] || gid.y >= src_shape[0]) {
    return;
  }
  let src_idx = gid.x * src_strides[1] + gid.y * src_strides[0];
  let dst_idx = gid.x + src_shape[1] * gid.y;
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_g3(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[2] || gid.y >= src_shape[1] || gid.z >= src_shape[0]) {
    return;
  }
  let src_idx = gid.x * src_strides[2] + gid.y * src_strides[1] + gid.z * src_strides[0];
  let dst_idx = gid.x + src_shape[1] * (gid.y + src_shape[2] * gid.z);
  dst[dst_idx] = dst_dtype(src[src_idx]);
}
