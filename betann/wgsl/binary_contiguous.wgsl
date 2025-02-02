if ($enable_f16) {
  enable f16;
}

alias output_dtype = $output_dtype;
alias input_dtype = $input_dtype;

override num_threads: u32 = 64;

@group(0) @binding(0) var<storage, read_write> c: array<output_dtype>;
@group(0) @binding(1) var<storage, read> a: array<input_dtype>;
@group(0) @binding(2) var<storage, read> b: array<input_dtype>;

@compute @workgroup_size(num_threads, 1, 1)
fn binary_ss_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $op(a[0], b[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_sv_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $op(a[0], b[gid.x]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vs_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $op(a[gid.x], b[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vv_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&c)) {
    c[gid.x] = $op(a[gid.x], b[gid.x]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_sv2_$op(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&c)) {
    c[offset] = $op(a[0], b[offset]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vs2_$op(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&c)) {
    c[offset] = $op(a[offset], b[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn binary_vv2_$op(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&c)) {
    c[offset] = $op(a[offset], b[offset]);
  }
}

// include binary_ops.wgsl
