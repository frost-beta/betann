if ($enable_f16) {
  enable f16;
}

alias output_dtype = $output_dtype;
alias input_dtype = $input_dtype;

override num_threads: u32 = 64;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<storage, read> input: array<input_dtype>;

@compute @workgroup_size(num_threads, 1, 1)
fn unary_v_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&output)) {
    output[gid.x] = output_dtype(betann_$op(input[gid.x]));
  }
}

@compute @workgroup_size(num_threads, 1, 1)
fn unary_v2_$op(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let offset = gid.x + gid.y * (num_threads * num_workgroups.x);
  if (offset < arrayLength(&output)) {
    output[offset] = output_dtype(betann_$op(input[offset]));
  }
}

// include unary_ops.wgsl
