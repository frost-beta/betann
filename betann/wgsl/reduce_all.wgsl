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

const workgroup_size_x: u32 = 32;
const work_per_thread: u32 = 4;
const block_size = workgroup_size_x * work_per_thread;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<storage, read> input: array<input_dtype>;

if ($enable_subgroups) {
  var<workgroup> workgroup_vals: array<output_dtype, workgroup_size_x / $subgroup_min_size>;
} else {
  var<workgroup> workgroup_vals: array<output_dtype, workgroup_size_x>;
}

@compute @workgroup_size(workgroup_size_x, 1, 1)
fn reduce_all_$op(if ($enable_subgroups) {
                    @builtin(subgroup_size) subgroup_size: u32,
                    @builtin(subgroup_invocation_id) subgroup_gid: u32,
                  }
                  @builtin(global_invocation_id) gid: vec3<u32>) {
  let initial_value = get_initial_value_$op();
  var total = initial_value;

  // Loop over input.
  let input_size = arrayLength(&input);
  for (var block = 0u; block < input_size / block_size; block++) {
    let idx = gid.x * work_per_thread + block * block_size;
    for (var i = 0u; i < work_per_thread; i++) {
      total = reduce_op_$op(output_dtype(input[idx + i]), total);
    }
  }

  // Leftover in input.
  let leftover = input_size % block_size;
  if (leftover != 0) {
    let idx = (input_size - leftover) + gid.x * work_per_thread;
    for (var i = 0u; i < work_per_thread && idx + i < input_size; i++) {
      total = reduce_op_$op(output_dtype(input[idx + i]), total);
    }
  }

  if ($enable_subgroups) {
    // Subgroup reduction.
    total = reduce_subgroup_op_$op(total);

    // Workgroup reduction.
    for (var delta = workgroup_size_x / subgroup_size; delta > 1; delta /= subgroup_size) {
      if (subgroup_gid == 0) {
        workgroup_vals[gid.x / subgroup_size] = total;
      }

      workgroupBarrier();
      total = select(initial_value, workgroup_vals[gid.x], gid.x < delta);
      total = reduce_subgroup_op_$op(total);
    }
  } else {
    // Write to shared memory.
    workgroup_vals[gid.x] = total;

    // Workgroup reduction.
    workgroupBarrier();
    for (var delta = workgroup_size_x / 2; delta >= 1; delta >>= 1) {
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
      output[0] = total;
    } else {
      output[0] = workgroup_vals[0];
    }
  }
}

// include reduce_ops.wgsl
