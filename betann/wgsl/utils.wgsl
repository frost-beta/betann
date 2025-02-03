fn coords_to_index_d1(coords_x: u32,
                      strides: ptr<storage, array<u32>>) -> u32 {
  return coords_x * strides[0];
}

fn coords_to_index_d2(coords: vec2<u32>,
                      strides: ptr<storage, array<u32>>) -> u32 {
  return coords.x * strides[1] + coords.y * strides[0];
}

fn coords_to_index_d3(coords: vec3<u32>,
                      strides: ptr<storage, array<u32>>) -> u32 {
  return coords.x * strides[2] + coords.y * strides[1] + coords.z * strides[0];
}

fn coord_to_index(coord: u32,
                  shape: ptr<storage, array<u32>>,
                  strides: ptr<storage, array<u32>>) -> u32 {
  let ndim = i32(arrayLength(shape));
  var idx: u32 = 0;
  var elem = coord;
  for (var d: i32 = ndim - 1; d >= 0 && elem > 0; d--) {
    idx += (elem % shape[d]) * strides[d];
    elem /= shape[d];
  }
  return idx;
}

fn coords_to_index(coords: vec3<u32>,
                   shape: ptr<storage, array<u32>>,
                   strides: ptr<storage, array<u32>>) -> u32 {
  let ndim = i32(arrayLength(shape));
  var idx = coords.x * strides[ndim - 1] + coords.y * strides[ndim - 2];
  var coords_z = coords.z;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    idx += (coords_z % shape[d]) * strides[d];
    coords_z /= shape[d];
  }
  return idx;
}

fn coords_to_indices(coords: vec3<u32>,
                     shape: ptr<storage, array<u32>>,
                     a_strides: ptr<storage, array<u32>>,
                     b_strides: ptr<storage, array<u32>>) -> vec2<u32> {
  let ndim = i32(arrayLength(shape));
  var indices = vec2(coords.x * a_strides[ndim - 1] + coords.y * a_strides[ndim - 2],
                     coords.x * b_strides[ndim - 1] + coords.y * b_strides[ndim - 2]);
  var coords_z = coords.z;
  for (var d: i32 = ndim - 3; d >= 0; d--) {
    let l = coords_z % shape[d];
    indices.x += l * a_strides[d];
    indices.y += l * b_strides[d];
    coords_z /= shape[d];
  }
  return indices;
}
