if ($enable_f16) {
  enable f16;
}

alias output_dtype = $output_dtype;
alias input_dtype = $input_dtype;

override num_threads: u32 = 8;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<storage, read> input: array<input_dtype>;
@group(0) @binding(2) var<storage, read> shape: array<u32>;
@group(0) @binding(3) var<storage, read> strides: array<u32>;
@group(0) @binding(4) var<uniform> rest_dims: u32;

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn unary_g_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  const work_per_thread: u32 = 1;
  let ndim = i32(arrayLength(&shape));
  let dim0 = shape[ndim - 1];
  let dim1 = shape[ndim - 2];
  if (work_per_thread * gid.x >= dim0 || gid.y >= dim1 || gid.z >= rest_dims) {
    return;
  }
  // Get index in inputs.
  var input_idx = coords_to_index(vec3(work_per_thread * gid.x, gid.yz),
                                  &shape,
                                  &strides);
  // Iterate and assign.
  let xstride = strides[ndim - 1];
  let out_idx = work_per_thread * gid.x + dim0 * (gid.y + dim1 * gid.z);
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dim0;
       i++) {
    output[out_idx + i] = output_dtype(betann_$op(input[input_idx]));
    input_idx += xstride;
  }
}

// include unary_ops.wgsl
// include utils.wgsl
