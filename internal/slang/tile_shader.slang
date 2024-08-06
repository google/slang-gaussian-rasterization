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
import spherical_harmonics;

struct rectangle {
    float min_x;
    float min_y;
    float max_x;
    float max_y;
};


rectangle get_rectangle_tile_space(
    float2 ndc_xy,
    float radius,
    uint grid_height,
    uint grid_width,
    uint tile_height,
    uint tile_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = clip((ndc_xy.x - radius) / tile_width, 0.0, grid_width);
    rect_tile_space.min_y = clip((ndc_xy.y - radius) / tile_height, 0.0, grid_height);
    // rect_tile_space.max_x = clip((ndc_xy.x + radius) / tile_width, 0.0, grid_width);
    // rect_tile_space.max_y = clip((ndc_xy.y + radius) / tile_height, 0.0, grid_height);
    rect_tile_space.max_x = clip((ndc_xy.x + radius + tile_width - 1)/tile_width, 0.0, grid_width);
    rect_tile_space.max_y = clip((ndc_xy.y + radius + tile_height - 1)/tile_height, 0.0, grid_height);

    return rect_tile_space;
}

[Differentiable]
float3x3 get_covariance_from_quat_scales(float4 q, float3 s) {
    float r=q[0], x=q[1], y=q[2], z=q[3];

    float3x3 rotation_matrix = float3x3(
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z)    , 2 * (x * z + r * y),
        2 * (x * y + r * z)    , 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y)    , 2 * (y * z + r * x)    , 1 - 2 * (x * x + y * y));

    float3x3 scales_matrix = float3x3(s[0], 0   , 0,
                                     0   , s[1], 0,
                                     0   , 0   , s[2]);

    float3x3 L = mul(rotation_matrix, scales_matrix);

    return mul(L, transpose(L));
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void tiles_touched_per_gaussian(DiffTensorView xyz_ws,
                                DiffTensorView rotations,
                                DiffTensorView scales,
                                DiffTensorView sh_coeffs,
                                uint active_sh,
                                TensorView<float> world_view_transform,
                                TensorView<float> proj_mat,
                                TensorView<float> cam_pos,
                                TensorView<int32_t> out_tiles_touched,
                                TensorView<float> out_rect_tile_space,
                                TensorView<int32_t> out_radii,
                                DiffTensorView out_xyz_vs,
                                DiffTensorView out_inv_cov_vs,
                                DiffTensorView out_rgb,
                                no_diff float fovy,
                                no_diff float fovx,
                                uint image_height,
                                uint image_width,
                                uint grid_height,
                                uint grid_width,
                                uint tile_height,
                                uint tile_width)
{
    uint32_t g_idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;

    if (g_idx >= xyz_ws.size(0))
        return;

    Camera cam = no_diff load_camera(world_view_transform, proj_mat, fovy, fovx, image_height, image_width);

    float3 g_xyz_ws = float3(xyz_ws[uint2(g_idx, 0)],
                             xyz_ws[uint2(g_idx, 1)],
                             xyz_ws[uint2(g_idx, 2)]);
    float3 g_xyz_vs = project_point(g_xyz_ws, cam);
    if (g_xyz_vs.z <= 0.2) {
        return;
    }
    float4 g_rotations = float4(rotations[uint2(g_idx, 0)], rotations[uint2(g_idx, 1)],
                                rotations[uint2(g_idx, 2)], rotations[uint2(g_idx, 3)]);
    float3 g_scales = float3(scales[(uint2(g_idx, 0))], scales[(uint2(g_idx, 1))], scales[(uint2(g_idx, 2))]);
    float3 g_cam_pos = float3(no_diff cam_pos[0], no_diff cam_pos[1], no_diff cam_pos[2]);

    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh);
    float3 rgb = compute_color_from_sh_coeffs(g_sh_coeffs, g_xyz_ws, g_cam_pos, active_sh);

    float3x3 g_cov_ws = get_covariance_from_quat_scales(g_rotations, g_scales);
    float2x2 g_cov_vs = covariance_3d_to_2d(cam, g_xyz_ws, g_cov_ws);

    float det = (g_cov_vs[0][0] * g_cov_vs[1][1] - g_cov_vs[0][1] * g_cov_vs[1][0]);
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute a bounding rectangle
    // of screen-space tiles that this Gaussian overlaps with. Quit if
    // rectangle covers 0 tiles.
    float mid = 0.5f * (g_cov_vs[0][0] + g_cov_vs[1][1]);
    float eigen_val_1 = mid + sqrt(max(0.1f, mid * mid - det));
    float eigen_val_2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(eigen_val_1, eigen_val_2)));

    float2 pixelspace_xy = { ndc2pix(g_xyz_vs.x, image_width), ndc2pix(g_xyz_vs.y, image_height) };

    rectangle rect_tile_space = get_rectangle_tile_space(pixelspace_xy,
                                                         radius, grid_height, grid_width, tile_height, tile_width);
    int32_t n_tiles = (int32_t(rect_tile_space.max_x) - int32_t(rect_tile_space.min_x)) *
                      (int32_t(rect_tile_space.max_y) - int32_t(rect_tile_space.min_y));

    if (n_tiles == 0) {
        return;
    }

    float2x2 g_inv_cov_vs = float2x2(g_cov_vs[1][1], -g_cov_vs[0][1], -g_cov_vs[1][0], g_cov_vs[0][0]) * det_inv;

    out_radii[g_idx] = (uint32_t)radius;
    out_tiles_touched[g_idx] = n_tiles;
    out_rect_tile_space[uint2(g_idx, 0)] = rect_tile_space.min_x;
    out_rect_tile_space[uint2(g_idx, 1)] = rect_tile_space.min_y;
    out_rect_tile_space[uint2(g_idx, 2)] = rect_tile_space.max_x;
    out_rect_tile_space[uint2(g_idx, 3)] = rect_tile_space.max_y;

    out_xyz_vs.storeOnce(uint2(g_idx, 0), g_xyz_vs.x);
    out_xyz_vs.storeOnce(uint2(g_idx, 1), g_xyz_vs.y);
    out_xyz_vs.storeOnce(uint2(g_idx, 2), g_xyz_vs.z);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 0), g_inv_cov_vs[0][0]);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 0, 1), g_inv_cov_vs[0][1]);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 0), g_inv_cov_vs[1][0]);
    out_inv_cov_vs.storeOnce(uint3(g_idx, 1, 1), g_inv_cov_vs[1][1]);
    out_rgb.storeOnce(uint2(g_idx, 0), rgb.r);
    out_rgb.storeOnce(uint2(g_idx, 1), rgb.g);
    out_rgb.storeOnce(uint2(g_idx, 2), rgb.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
void generate_keys(TensorView<float> xyz_vs,
                   TensorView<float> rect_tile_space,
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

    int rect_min_x = int((rect_tile_space[uint2(globalIdx, 0)]));
    int rect_min_y = int((rect_tile_space[uint2(globalIdx, 1)]));
    int rect_max_x = int((rect_tile_space[uint2(globalIdx, 2)]));
    int rect_max_y = int((rect_tile_space[uint2(globalIdx, 3)]));
    
    for (int y = rect_min_y; y < rect_max_y; y++)
    {
        for (int x = rect_min_x; x < rect_max_x; x++)
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