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
const read_per_thread: u32 = 4;
const write_per_thread: u32 = 4;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<uniform> num_outputs: u32;
@group(0) @binding(2) var<storage, read> input: array<input_dtype>;
@group(0) @binding(3) var<uniform> row_size: u32;

var<workgroup> workgroup_vals: array<output_dtype,
                                     workgroup_size * write_per_thread>;

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_last_$op(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = gid.y * write_per_thread;
  let input_offset = row * row_size;
  let block_size = workgroup_size * read_per_thread;

  // Initialize accumulator registers.
  let initial_value = get_initial_value_$op();
  var totals: array<output_dtype, write_per_thread>;
  for (var w = 0u; w < write_per_thread; w++) {
    totals[w] = initial_value;
  }

  // Loop over rows.
  for (var block = 0u; block < row_size / block_size; block++) {
    for (var w = 0u; w < write_per_thread && row + w < num_outputs; w++) {
      let idx = input_offset + w * row_size + block * block_size + lid.x * read_per_thread;
      for (var r = 0u; r < read_per_thread; r++) {
        totals[w] = reduce_op_$op(input[idx + r], totals[w]);
      }
    }
  }

  // Leftovers in rows.
  let leftover = row_size % block_size;
  if (leftover != 0) {
    for (var w = 0u; w < write_per_thread && row + w < num_outputs; w++) {
      let idx = (row_size - leftover) + lid.x * read_per_thread;
      for (var r = 0u; r < read_per_thread && idx + r < row_size; r++) {
        totals[w] = reduce_op_$op(input[input_offset + w * row_size + idx + r],
                                  totals[w]);
      }
    }
  }

  // Write to shared memory.
  for (var w = 0u; w < write_per_thread; w++) {
    workgroup_vals[w * workgroup_size + lid.x] = totals[w];
  }

  // Workgroup reduction.
  workgroupBarrier();
  for (var w = 0u; w < write_per_thread; w++) {
    let idx = w * workgroup_size + lid.x;
    for (var delta = workgroup_size / 2; delta >= 1; delta >>= 1) {
      if (lid.x < delta) {
        workgroup_vals[idx] = reduce_op_$op(workgroup_vals[idx],
                                            workgroup_vals[idx + delta]);
      }
      workgroupBarrier();
    }
  }

  // Write output.
  if (lid.x == 0) {
    for (var w = 0u; w < write_per_thread && row + w < num_outputs; w++) {
      output[row + w] = workgroup_vals[w * workgroup_size];
    }
  }
}

// include reduce_ops.wgsl
// include utils.wgsl
