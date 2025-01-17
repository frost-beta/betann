alias input_dtype = $1;
alias output_dtype = $2;

@group(0) @binding(0) var<storage, read> a: array<input_dtype>;
@group(0) @binding(1) var<storage, read> b: array<input_dtype>;
@group(0) @binding(2) var<storage, read_write> c: array<output_dtype>;

override num_threads: u32 = 256;

@compute @workgroup_size(num_threads, 1, 1)
fn binary_ss_$0(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $0(a[0], b[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_sv_$0(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $0(a[0], b[gid.x]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vs_$0(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $0(a[gid.x], b[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vv_$0(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $0(a[gid.x], b[gid.x]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_sv2_$0(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&c)) {
    c[offset] = $0(a[0], b[offset]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vs2_$0(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&c)) {
    c[offset] = $0(a[offset], b[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vv2_$0(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&c)) {
    c[offset] = $0(a[offset], b[offset]);
  }
}

// include binary_ops.wgsl
