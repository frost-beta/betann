if ($enable_f16) {
  enable f16;
}

alias output_dtype = $output_dtype;

const workgroup_size: u32 = $workgroup_size;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<uniform> num_outputs: u32;

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_none_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < num_outputs) {
    output[gid.x] = get_initial_value_$op();
  }
}

// include reduce_ops.wgsl
