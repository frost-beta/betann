enable f16;

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
  let dst_idx = gid.x + src_shape[1] * gid.y;
  let src_idx = gid.x * src_strides[1] + gid.y * src_strides[0];
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_g3(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[2] || gid.y >= src_shape[1] || gid.z >= src_shape[0]) {
    return;
  }
  let dst_idx = gid.x + src_shape[1] * (gid.y + src_shape[2] * gid.z);
  let src_idx = gid.x * src_strides[2] + gid.y * src_strides[1] + gid.z * src_strides[0];
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_g_n2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  const work_per_thread: u32 = 2;
  let ndim = i32(arrayLength(&src_shape));
  let dim0 = src_shape[ndim - 1];
  let dim1 = src_shape[ndim - 2];
  var rest: u32 = 1;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    rest *= src_shape[d];
  }
  if (work_per_thread * gid.x >= dim0 || gid.y >= dim1 || gid.z >= rest) {
    return;
  }
  // Get index in src and dst.
  let dst_idx = work_per_thread * gid.x + dim0 * (gid.y + dim1 * gid.z);
  var src_idx = work_per_thread * gid.x * src_strides[ndim - 1] + gid.y * src_strides[ndim - 2];
  var elem_z = gid.z;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    let l = elem_z % src_shape[d];
    src_idx += l * src_strides[d];
    elem_z /= src_shape[d];
  }
  // Iterate and assign.
  let src_xstride = src_strides[ndim - 1];
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dim0;
       i++) {
    dst[dst_idx + i] = dst_dtype(src[src_idx]);
    src_idx += src_xstride;
  }
}
