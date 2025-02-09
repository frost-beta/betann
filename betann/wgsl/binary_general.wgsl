if ($enable_f16) {
  enable f16;
}

alias output_dtype = $output_dtype;
alias input_dtype = $input_dtype;

const num_threads: u32 = 8;
const work_per_thread: u32 = 2;

@group(0) @binding(0) var<storage, read_write> c: array<output_dtype>;
@group(0) @binding(1) var<storage, read> shape: array<u32>;
@group(0) @binding(2) var<storage, read> a: array<input_dtype>;
@group(0) @binding(3) var<storage, read> a_strides: array<u32>;
@group(0) @binding(4) var<storage, read> b: array<input_dtype>;
@group(0) @binding(5) var<storage, read> b_strides: array<u32>;
@group(0) @binding(6) var<uniform> dims: dims_t;

@compute @workgroup_size(num_threads, 1, 1)
fn binary_g1_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[0]) {
    return;
  }
  let a_idx = coords_to_index_d1(gid.x, &a_strides);
  let b_idx = coords_to_index_d1(gid.x, &b_strides);
  c[gid.x] = $op(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn binary_g2_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[1] || gid.y >= shape[0]) {
    return;
  }
  let a_idx = coords_to_index_d2(gid.xy, &a_strides);
  let b_idx = coords_to_index_d2(gid.xy, &b_strides);
  let out_idx = gid.x + shape[1] * gid.y;
  c[out_idx] = $op(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn binary_g3_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[2] || gid.y >= shape[1] || gid.z >= shape[0]) {
    return;
  }
  let a_idx = coords_to_index_d3(gid, &a_strides);
  let b_idx = coords_to_index_d3(gid, &b_strides);
  let out_idx = gid.x + shape[1] * (gid.y + shape[2] * gid.z);
  c[out_idx] = $op(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn binary_g_n2_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  if (work_per_thread * gid.x >= dims.dim0 ||
      gid.y >= dims.dim1 ||
      gid.z >= dims.rest) {
    return;
  }
  // Get index in a and b.
  var idx = coords_to_indices(vec3(work_per_thread * gid.x, gid.yz),
                              &shape,
                              &a_strides,
                              &b_strides);
  // Iterate and assign.
  let a_xstride = a_strides[dims.ndim - 1];
  let b_xstride = b_strides[dims.ndim - 1];
  let out_idx = work_per_thread * gid.x + dims.dim0 * (gid.y + dims.dim1 * gid.z);
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dims.dim0;
       i++) {
    c[out_idx + i] = $op(a[idx.x], b[idx.y]);
    idx.x += a_xstride;
    idx.y += b_xstride;
  }
}

// include binary_ops.wgsl
// include utils.wgsl
