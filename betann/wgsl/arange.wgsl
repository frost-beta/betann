if ($enable_f16) {
  enable f16;
}

alias dtype = $dtype;

override num_threads: u32 = 64;

@group(0) @binding(0) var<uniform> start: dtype;
@group(0) @binding(1) var<uniform> step: dtype;
@group(0) @binding(2) var<storage, read_write> out: array<dtype>;

@compute @workgroup_size(num_threads, 1, 1)
fn arange(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&out)) {
    out[gid.x] = start + dtype(gid.x) * step;
  }
}
