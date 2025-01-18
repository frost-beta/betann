alias input_dtype = $1;
alias output_dtype = $2;

override num_threads: u32 = 8;

@group(0) @binding(0) var<storage, read> a: array<input_dtype>;
@group(0) @binding(1) var<storage, read> a_strides: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<input_dtype>;
@group(0) @binding(3) var<storage, read> b_strides: array<u32>;
@group(0) @binding(4) var<storage, read> shape: array<u32>;
@group(0) @binding(5) var<storage, read_write> c: array<output_dtype>;

@compute @workgroup_size(num_threads, 1, 1)
fn binary_g1_$0(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= shape[0]) {
    return;
  }
  let a_idx = gid.x * a_strides[0];
  let b_idx = gid.x * b_strides[0];
  c[gid.x] = $0(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, 1)
fn binary_g2_$0(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  if (gid.x >= shape[1] || gid.y >= shape[0]) {
    return;
  }
  let a_idx = gid.x * a_strides[1] + gid.y * a_strides[0];
  let b_idx = gid.x * b_strides[1] + gid.y * b_strides[0];
  let out_idx = gid.x + shape[1] * gid.y;
  c[out_idx] = $0(a[a_idx], b[b_idx]);
}

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn binary_g3_$0(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  if (gid.x >= shape[2] || gid.y >= shape[1] || gid.z >= shape[0]) {
    return;
  }
  let a_idx = gid.x * a_strides[2] + gid.y * a_strides[1] + gid.z * a_strides[0];
  let b_idx = gid.x * b_strides[2] + gid.y * b_strides[1] + gid.z * b_strides[0];
  let out_idx = gid.x + shape[1] * (gid.y + shape[2] * gid.z);
  c[out_idx] = $0(a[a_idx], b[b_idx]);
}

// include binary_ops.wgsl
