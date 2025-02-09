if ($enable_f16) {
  enable f16;
}

alias dst_dtype = $dst_dtype;
alias src_dtype = $src_dtype;

const num_threads: u32 = 8;
const work_per_thread: u32 = 2;

@group(0) @binding(0) var<storage, read_write> dst: array<dst_dtype>;
@group(0) @binding(1) var<storage, read> dst_strides: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<src_dtype>;
@group(0) @binding(3) var<storage, read> src_shape: array<u32>;
@group(0) @binding(4) var<storage, read> src_strides: array<u32>;
@group(0) @binding(5) var<uniform> dims: dims_t;

@compute @workgroup_size(num_threads, 1, 1)
fn copy_gg1(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[0]) {
    return;
  }
  let dst_idx = coords_to_index_d1(gid.x, &dst_strides);
  let src_idx = coords_to_index_d1(gid.x, &src_strides);
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn copy_gg2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[1] || gid.y >= src_shape[0]) {
    return;
  }
  let dst_idx = coords_to_index_d2(gid.xy, &dst_strides);
  let src_idx = coords_to_index_d2(gid.xy, &src_strides);
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_gg3(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= src_shape[2] || gid.y >= src_shape[1] || gid.z >= src_shape[0]) {
    return;
  }
  let dst_idx = coords_to_index_d3(gid, &dst_strides);
  let src_idx = coords_to_index_d3(gid, &src_strides);
  dst[dst_idx] = dst_dtype(src[src_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn copy_gg_n2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  if (work_per_thread * gid.x >= dims.dim0 ||
      gid.y >= dims.dim1 ||
      gid.z >= dims.rest) {
    return;
  }
  // Get index in src and dst.
  var idx = coords_to_indices(vec3(work_per_thread * gid.x, gid.yz),
                              &src_shape,
                              &dst_strides,
                              &src_strides);
  // Iterate and assign.
  let dst_xstride = dst_strides[dims.ndim - 1];
  let src_xstride = src_strides[dims.ndim - 1];
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dims.dim0;
       i++) {
    dst[idx.x] = dst_dtype(src[idx.y]);
    idx.x += dst_xstride;
    idx.y += src_xstride;
  }
}

// include utils.wgsl
