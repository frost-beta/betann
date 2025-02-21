if ($enable_f16) {
  enable f16;
}
if ($enable_subgroups) {
  enable subgroups;
}
if ($enable_subgroups_f16) {
  enable subgroups_f16;
}

alias output_dtype = $output_dtype;
alias input_dtype = $input_dtype;

const workgroup_size: u32 = $workgroup_size;
const work_per_thread: u32 = 4;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<uniform> output_num_elements: u32;
@group(0) @binding(2) var<storage, read> input: array<input_dtype>;
@group(0) @binding(3) var<uniform> row_size: u32;
@group(0) @binding(4) var<uniform> non_row_reductions: u32;
@group(0) @binding(5) var<storage, read> non_reduction_shape: array<u32>;
@group(0) @binding(6) var<storage, read> non_reduction_strides: array<u32>;
@group(0) @binding(7) var<storage, read> reduction_shape: array<u32>;
@group(0) @binding(8) var<storage, read> reduction_strides: array<u32>;

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_row_small_$op(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= output_num_elements) {
    return;
  }

  var total = get_initial_value_$op();
  var input_offset = coord_to_index(gid.x, &non_reduction_shape, &non_reduction_strides);
  for (var r = 0u; r < non_row_reductions; r++) {
    let reduction_index = coord_to_index(r, &reduction_shape, &reduction_strides);;
    let row_offset = input_offset + reduction_index;
    thread_reduce_$op(&total, &input, row_offset, row_size, work_per_thread);
  }

  output[gid.x] = total;
}

fn thread_reduce_$op(total: ptr<function, output_dtype>,
                     input: ptr<storage, array<input_dtype>>,
                     row_offset: u32,
                     row_size: u32,
                     block_size: u32) {
  for (var block = 0u; block < row_size / block_size; block++) {
    var vals: array<input_dtype, work_per_thread>;
    for (var i = 0u; i < block_size; i++) {
      vals[i] = input[row_offset + block * block_size + i];
    }
    for (var i = 0u; i < block_size; i++) {
      *total = reduce_op_$op(vals[i], *total);
    }
  }

  let leftover = row_size % block_size;
  if (leftover != 0) {
    let idx = row_size - leftover;
    for (var i = 0u; i < block_size && idx + i < row_size; i++) {
      *total = reduce_op_$op(output_dtype(input[row_offset + idx + i]), *total);
    }
  }
}

// include reduce_ops.wgsl
// include utils.wgsl
