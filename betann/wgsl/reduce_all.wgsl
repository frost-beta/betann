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
@group(0) @binding(1) var<storage, read> input: array<input_dtype>;
@group(0) @binding(2) var<uniform> row_size: u32;

if ($enable_subgroups) {
  var<workgroup> workgroup_vals: array<output_dtype, workgroup_size / $subgroup_min_size>;
} else {
  var<workgroup> workgroup_vals: array<output_dtype, workgroup_size>;
}

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_all_$op(if ($enable_subgroups) {
                    @builtin(subgroup_size) subgroup_size: u32,
                    @builtin(subgroup_invocation_id) subgroup_gid: u32,
                  }
                  @builtin(global_invocation_id) gid: vec3<u32>) {
  let initial_value = get_initial_value_$op();
  var total = initial_value;

  // How much work to do.
  let input_size = arrayLength(&input);
  let input_offset = gid.y * row_size;
  let current_row_size = select(
      row_size,
      select(0, input_size - input_offset, input_size > input_offset),
      input_offset + row_size > input_size);

  // Loop over input.
  const block_size = workgroup_size * work_per_thread;
  for (var block = 0u; block < current_row_size / block_size; block++) {
    let idx = block * block_size + gid.x * work_per_thread;
    for (var i = 0u; i < work_per_thread; i++) {
      total = reduce_op_$op(output_dtype(input[input_offset + idx + i]), total);
    }
  }

  // Leftover in input.
  let leftover = current_row_size % block_size;
  if (leftover != 0) {
    let idx = (current_row_size - leftover) + gid.x * work_per_thread;
    for (var i = 0u; i < work_per_thread && idx + i < current_row_size; i++) {
      total = reduce_op_$op(output_dtype(input[input_offset + idx + i]), total);
    }
  }

  if ($enable_subgroups) {
    // Subgroup reduction.
    total = reduce_subgroup_op_$op(total);

    // Workgroup reduction.
    // FIXME(zcbenz): Must convert delta to f32 for comparison, possible Metal bug.
    for (var delta = workgroup_size / subgroup_size; f32(delta) > 1f; delta /= subgroup_size) {
      // Write first lane's result to shared memory.
      if (subgroup_gid == 0) {
        workgroup_vals[gid.x / subgroup_size] = total;
      }

      // Subgroup reduction.
      workgroupBarrier();
      total = select(initial_value, workgroup_vals[gid.x], gid.x < delta);
      total = reduce_subgroup_op_$op(total);
    }
  } else {
    // Write to shared memory.
    workgroup_vals[gid.x] = total;

    // Workgroup reduction.
    workgroupBarrier();
    for (var delta = workgroup_size / 2; delta >= 1; delta >>= 1) {
      if (gid.x < delta) {
        workgroup_vals[gid.x] = reduce_op_$op(workgroup_vals[gid.x],
                                              workgroup_vals[gid.x + delta]);
      }
      workgroupBarrier();
    }
  }

  // Write output.
  if (gid.x == 0) {
    if ($enable_subgroups) {
      output[gid.y] = total;
    } else {
      output[gid.y] = workgroup_vals[0];
    }
  }
}

// include reduce_ops.wgsl
