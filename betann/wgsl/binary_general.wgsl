alias output_dtype = $0;
alias input_dtype = $1;

override num_threads: u32 = 8;

@group(0) @binding(0) var<storage, read_write> c: array<output_dtype>;
@group(0) @binding(1) var<storage, read> shape: array<u32>;
@group(0) @binding(2) var<storage, read> a: array<input_dtype>;
@group(0) @binding(3) var<storage, read> a_strides: array<u32>;
@group(0) @binding(4) var<storage, read> b: array<input_dtype>;
@group(0) @binding(5) var<storage, read> b_strides: array<u32>;

@compute @workgroup_size(num_threads, 1, 1)
fn binary_g1_$2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[0]) {
    return;
  }
  let a_idx = gid.x * a_strides[0];
  let b_idx = gid.x * b_strides[0];
  c[gid.x] = $2(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn binary_g2_$2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[1] || gid.y >= shape[0]) {
    return;
  }
  let a_idx = gid.x * a_strides[1] + gid.y * a_strides[0];
  let b_idx = gid.x * b_strides[1] + gid.y * b_strides[0];
  let out_idx = gid.x + shape[1] * gid.y;
  c[out_idx] = $2(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn binary_g3_$2(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[2] || gid.y >= shape[1] || gid.z >= shape[0]) {
    return;
  }
  let a_idx = gid.x * a_strides[2] + gid.y * a_strides[1] + gid.z * a_strides[0];
  let b_idx = gid.x * b_strides[2] + gid.y * b_strides[1] + gid.z * b_strides[0];
  let out_idx = gid.x + shape[1] * (gid.y + shape[2] * gid.z);
  c[out_idx] = $2(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn binary_g_n2_$2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  const work_per_thread: u32 = 2;
  let ndim = i32(arrayLength(&shape));
  let dim0 = shape[ndim - 1];
  let dim1 = shape[ndim - 2];
  var rest: u32 = 1;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    rest *= shape[d];
  }
  if (work_per_thread * gid.x >= dim0 || gid.y >= dim1 || gid.z >= rest) {
    return;
  }
  // Get index in a and b.
  var a_idx = work_per_thread * gid.x * a_strides[ndim - 1] + gid.y * a_strides[ndim - 2];
  var b_idx = work_per_thread * gid.x * b_strides[ndim - 1] + gid.y * b_strides[ndim - 2];
  var z_idx = gid.z;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    let l = z_idx % shape[d];
    a_idx += l * a_strides[d];
    b_idx += l * b_strides[d];
    z_idx /= shape[d];
  }
  // Iterate and assign.
  var out_idx = work_per_thread * gid.x + dim0 * (gid.y + dim1 * gid.z);
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dim0;
       i++) {
    c[out_idx] = $2(a[a_idx], b[b_idx]);
    a_idx += a_strides[ndim - 1];
    b_idx += b_strides[ndim - 1];
    out_idx++;
  }
}

// include binary_ops.wgsl
