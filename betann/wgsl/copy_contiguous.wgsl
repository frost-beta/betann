alias input_dtype = $0;
alias output_dtype = $1;

override num_threads: u32 = 256;

@group(0) @binding(0) var<storage, read> src: array<input_dtype>;
@group(0) @binding(1) var<storage, read_write> dst: array<output_dtype>;

@compute @workgroup_size(num_threads, 1, 1)
fn copy_s(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&dst)) {
    dst[gid.x] = output_dtype(src[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn copy_v(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&dst)) {
    dst[gid.x] = output_dtype(src[gid.x]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn copy_s2(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&dst)) {
    dst[offset] = output_dtype(src[0]);
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn copy_v2(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&dst)) {
    dst[offset] = output_dtype(src[offset]);
  }
}
