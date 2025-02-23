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
  var<workgroup> workgroup_totals: array<output_dtype, workgroup_size / $subgroup_min_size>;
} else {
  var<workgroup> workgroup_totals: array<output_dtype, workgroup_size>;
}

@compute @workgroup_size(workgroup_size, 1, 1)
fn reduce_all_$op(if ($enable_subgroups) {
                    @builtin(subgroup_size) subgroup_size: u32,
                    @builtin(subgroup_invocation_id) subgroup_gid: u32,
                  }
                  @builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {
  // How much work to do.
  let input_size = arrayLength(&input);
  let row = gid.y * row_size;
  let current_row_size = select(
      row_size,
      select(0, input_size - row, input_size > row),
      row + row_size > input_size);

  // Reduce current row.
  var total = get_initial_value_$op();
  row_reduce(&total, lid.x, &input, row, current_row_size, workgroup_size, work_per_thread);

  // Reduce across the workgroup.
  if ($enable_subgroups) {
    workgroup_reduce(&total, lid.x, workgroup_size, subgroup_gid, subgroup_size);
  } else {
    workgroup_reduce(&total, lid.x, workgroup_size, 0, 0);
  }

  // Write output.
  if (lid.x == 0) {
    output[gid.y] = total;
  }
}

// include reduce_ops.wgsl
