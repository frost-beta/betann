if ($enable_f16) {
  enable f16;
}

alias dst_dtype = $dst_dtype;
alias src_dtype = $src_dtype;

override num_threads: u32 = 8;

@group(0) @binding(0) var<storage, read_write> dst: array<dst_dtype>;
@group(0) @binding(1) var<storage, read> src: array<src_dtype>;
@group(0) @binding(2) var<storage, read> src_shape: array<u32>;
@group(0) @binding(3) var<storage, read> src_strides: array<u32>;
@group(0) @binding(4) var<uniform> src_rest_dims: u32;

@compute @workgroup_size(num_threads, 1, 1)
fn copy_g1(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[0]) {
    return;
  }
  let src_idx = coords_to_index_d1(gid.x, &src_strides);
  dst[gid.x] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn copy_g2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[1] || gid.y >= src_shape[0]) {
    return;
  }
  let dst_idx = gid.x + src_shape[1] * gid.y;
  let src_idx = coords_to_index_d2(gid.xy, &src_strides);
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_g3(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[2] || gid.y >= src_shape[1] || gid.z >= src_shape[0]) {
    return;
  }
  let dst_idx = gid.x + src_shape[1] * (gid.y + src_shape[2] * gid.z);
  let src_idx = coords_to_index_d3(gid, &src_strides);
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_g_n2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  const work_per_thread: u32 = 2;
  let ndim = i32(arrayLength(&src_shape));
  let dim0 = src_shape[ndim - 1];
  let dim1 = src_shape[ndim - 2];
  if (work_per_thread * gid.x >= dim0 || gid.y >= dim1 || gid.z >= src_rest_dims) {
    return;
  }
  // Get index in src and dst.
  let dst_idx = work_per_thread * gid.x + dim0 * (gid.y + dim1 * gid.z);
  var src_idx = coords_to_index(vec3(work_per_thread * gid.x, gid.yz),
                                &src_shape,
                                &src_strides);
  // Iterate and assign.
  let src_xstride = src_strides[ndim - 1];
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dim0;
       i++) {
    dst[dst_idx + i] = dst_dtype(src[src_idx]);
    src_idx += src_xstride;
  }
}

// include utils.wgsl
