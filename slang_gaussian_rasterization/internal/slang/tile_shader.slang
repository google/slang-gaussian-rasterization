// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import utils;

[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(TensorView<float> xyz_vs,
                   TensorView<int32_t> rect_tile_space,
                   TensorView<int32_t> index_buffer_offset,
                   TensorView<int64_t> out_unsorted_keys,
                   TensorView<int32_t> out_unsorted_gauss_idx,
                   uint grid_height,
                   uint grid_width)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= xyz_vs.size(0))
        return;

    float3 ndc_xyz = {
        xyz_vs[uint2(globalIdx, 0)],
        xyz_vs[uint2(globalIdx, 1)],
        xyz_vs[uint2(globalIdx, 2)]
    };

    // Find this Gaussian's offset in buffer for writing keys/values.
    int32_t offset;
    if (globalIdx == 0)
        offset = 0;
    else
        offset = index_buffer_offset[globalIdx - 1];

    int32_t rect_min_x = rect_tile_space[uint2(globalIdx, 0)];
    int32_t rect_min_y = rect_tile_space[uint2(globalIdx, 1)];
    int32_t rect_max_x = rect_tile_space[uint2(globalIdx, 2)];
    int32_t rect_max_y = rect_tile_space[uint2(globalIdx, 3)];
    
    for (int32_t y = rect_min_y; y < rect_max_y; y++)
    {
        for (int32_t x = rect_min_x; x < rect_max_x; x++)
        {
            uint64_t key = y * grid_width + x;
            key <<= 32;
            key = key | reinterpret<int32_t>(ndc_xyz.z);
            out_unsorted_keys[offset] = key;
            out_unsorted_gauss_idx[offset] = globalIdx;
            offset++;
        }
    }

}

[AutoPyBindCUDA]
[CUDAKernel]
void compute_tile_ranges(TensorView<int64_t> sorted_keys,
                         TensorView<int32_t> out_tile_ranges)
{
    int32_t globalIdx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (globalIdx >= sorted_keys.size(0))
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint32_t currtile = uint32_t(uint64_t(sorted_keys[globalIdx]) >> 32);
    if (globalIdx == 0)
        out_tile_ranges[uint2(currtile, 0)] = 0;
    else
    {
        uint32_t prevtile = uint32_t(uint64_t(sorted_keys[globalIdx - 1]) >> 32);
        if (currtile != prevtile)
        {
            out_tile_ranges[uint2(prevtile, 1)] = globalIdx;
            out_tile_ranges[uint2(currtile, 0)] = globalIdx;
        }
    }
    if (globalIdx == sorted_keys.size(0) - 1)
        out_tile_ranges[uint2(currtile, 1)] = sorted_keys.size(0);
}