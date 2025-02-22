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

const rows_threads: u32 = $workgroup_size;
const read_per_thread: u32 = 4;
const write_per_thread: u32 = 4;

@group(0) @binding(0) var<storage, read_write> output: array<output_dtype>;
@group(0) @binding(1) var<uniform> num_outputs: u32;
@group(0) @binding(2) var<storage, read> input: array<input_dtype>;
@group(0) @binding(3) var<uniform> row_size: u32;

if ($enable_subgroups) {
  const workgroup_totals_size = rows_threads * write_per_thread / $subgroup_min_size;
} else {
  const workgroup_totals_size = rows_threads * write_per_thread;
}
var<workgroup> workgroup_totals: array<output_dtype, workgroup_totals_size>;

@compute @workgroup_size(rows_threads, 1, 1)
fn reduce_last_$op(if ($enable_subgroups) {
                     @builtin(subgroup_size) subgroup_size: u32,
                     @builtin(subgroup_invocation_id) subgroup_gid: u32,
                   }
                   @builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = gid.y * write_per_thread;
  let input_offset = row * row_size;
  let block_size = rows_threads * read_per_thread;

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

  if ($enable_subgroups) {
    // Subgroup reduction.
    for (var w = 0u; w < write_per_thread; w++) {
      totals[w] = reduce_subgroup_op_$op(totals[w]);
    }

    // Workgroup reduction.
    // FIXME(zcbenz): Must convert delta to f32 for comparison, possible Metal bug.
    for (var delta = rows_threads / subgroup_size; f32(delta) > 1f; delta /= subgroup_size) {
      // Write first lane's result to shared memory.
      if (subgroup_gid == 0) {
        for (var w = 0u; w < write_per_thread; w++) {
          workgroup_totals[w * workgroup_totals_size + lid.x / subgroup_size] = totals[w];
        }
      }

      // Subgroup reduction.
      workgroupBarrier();
      for (var w = 0u; w < write_per_thread; w++) {
        totals[w] = select(initial_value,
                           workgroup_totals[w * workgroup_totals_size + lid.x],
                           lid.x < delta);
        totals[w] = reduce_subgroup_op_$op(totals[w]);
      }
    }
  } else {
    // Write to shared memory.
    for (var w = 0u; w < write_per_thread; w++) {
      workgroup_totals[w * rows_threads + lid.x] = totals[w];
    }

    // Workgroup reduction.
    workgroupBarrier();
    for (var w = 0u; w < write_per_thread; w++) {
      let idx = w * rows_threads + lid.x;
      for (var delta = rows_threads / 2; delta >= 1; delta >>= 1) {
        if (lid.x < delta) {
          workgroup_totals[idx] = reduce_op_$op(workgroup_totals[idx],
                                                workgroup_totals[idx + delta]);
        }
        workgroupBarrier();
      }
    }
  }

  // Write output.
  if (lid.x == 0) {
    for (var w = 0u; w < write_per_thread && row + w < num_outputs; w++) {
      if ($enable_subgroups) {
        output[row + w] = totals[w];
      } else {
        output[row + w] = workgroup_totals[w * rows_threads];
      }
    }
  }
}

// include reduce_ops.wgsl
// include utils.wgsl
