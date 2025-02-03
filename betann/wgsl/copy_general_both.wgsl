if ($enable_f16) {
  enable f16;
}

alias dst_dtype = $dst_dtype;
alias src_dtype = $src_dtype;

override num_threads: u32 = 8;

@group(0) @binding(0) var<storage, read_write> dst: array<dst_dtype>;
@group(0) @binding(1) var<storage, read> dst_strides: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<src_dtype>;
@group(0) @binding(3) var<storage, read> src_shape: array<u32>;
@group(0) @binding(4) var<storage, read> src_strides: array<u32>;
@group(0) @binding(5) var<uniform> src_rest_dims: u32;

@compute @workgroup_size(num_threads, 1, 1)
fn copy_gg1(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[0]) {
    return;
  }
  let dst_idx = gid.x * src_strides[0];
  let src_idx = gid.x * src_strides[0];
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn copy_gg2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[1] || gid.y >= src_shape[0]) {
    return;
  }
  let dst_idx = gid.x * dst_strides[1] + gid.y * dst_strides[0];
  let src_idx = gid.x * src_strides[1] + gid.y * src_strides[0];
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_gg3(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[2] || gid.y >= src_shape[1] || gid.z >= src_shape[0]) {
    return;
  }
  let dst_idx = gid.x * dst_strides[2] + gid.y * dst_strides[1] + gid.z * dst_strides[0];
  let src_idx = gid.x * src_strides[2] + gid.y * src_strides[1] + gid.z * src_strides[0];
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_gg_n2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  const work_per_thread: u32 = 2;
  let ndim = i32(arrayLength(&src_shape));
  let dim0 = src_shape[ndim - 1];
  let dim1 = src_shape[ndim - 2];
  if (work_per_thread * gid.x >= dim0 || gid.y >= dim1 || gid.z >= src_rest_dims) {
    return;
  }
  // Get index in src and dst.
  var dst_idx = work_per_thread * gid.x * dst_strides[ndim - 1] + gid.y * dst_strides[ndim - 2];
  var src_idx = work_per_thread * gid.x * src_strides[ndim - 1] + gid.y * src_strides[ndim - 2];
  var elem_z = gid.z;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    let l = elem_z % src_shape[d];
    dst_idx += l * dst_strides[d];
    src_idx += l * src_strides[d];
    elem_z /= src_shape[d];
  }
  // Iterate and assign.
  let dst_xstride = dst_strides[ndim - 1];
  let src_xstride = src_strides[ndim - 1];
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dim0;
       i++) {
    dst[dst_idx] = dst_dtype(src[src_idx]);
    dst_idx += dst_xstride;
    src_idx += src_xstride;
  }
}
