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
  // Initialize accumulator registers.
  let initial_value = get_initial_value_$op();
  var totals: array<output_dtype, write_per_thread>;
  for (var w = 0u; w < write_per_thread; w++) {
    totals[w] = initial_value;
  }

  // Reduce rows per thread.
  let row = gid.y * write_per_thread;
  for (var w = 0u; w < write_per_thread && row + w < num_outputs; w++) {
    row_reduce(&totals[w], lid.x, &input, (row + w) * row_size, row_size, rows_threads, read_per_thread);
  }

  // Reduce across the workgroup.
  for (var w = 0u; w < write_per_thread; w++) {
    if ($enable_subgroups) {
      workgroup_reduce(&totals[w], lid.x, rows_threads, subgroup_gid, subgroup_size);
    } else {
      workgroup_reduce(&totals[w], lid.x, rows_threads, 0, 0);
    }
  }

  // Write output.
  if (lid.x == 0) {
    for (var w = 0u; w < write_per_thread && row + w < num_outputs; w++) {
      output[row + w] = totals[w];
    }
  }
}

// include reduce_ops.wgsl
// include utils.wgsl
