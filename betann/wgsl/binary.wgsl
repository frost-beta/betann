@group(0) @binding(0) var<storage, read> a: array<$1>;
@group(0) @binding(1) var<storage, read> b: array<$1>;
@group(0) @binding(2) var<storage, read_write> c: array<$2>;

@compute @workgroup_size(64, 1, 1)
fn binary_ss_$0(@builtin(global_invocation_id) gid: vec3<u32>) {
  c[gid.x] = a[0] + b[0];
}
