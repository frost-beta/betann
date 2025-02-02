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

@compute @workgroup_size(num_threads, num_threads, num_threads)
fn unary_g_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Check boundries.
  const work_per_thread: u32 = 1;
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
  // Get index in inputs.
  var idx = work_per_thread * gid.x * strides[ndim - 1] + gid.y * strides[ndim - 2];
  var elem = gid.z;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    idx += (elem % shape[d]) * strides[d];
    elem /= shape[d];
  }
  // Iterate and assign.
  let xstride = strides[ndim - 1];
  let out_idx = work_per_thread * gid.x + dim0 * (gid.y + dim1 * gid.z);
  for (var i: u32 = 0;
       i < work_per_thread && (work_per_thread * gid.x + i) < dim0;
       i++) {
    output[out_idx + i] = output_dtype(betann_$op(input[idx]));
    idx += xstride;
  }
}

// include unary_ops.wgsl
