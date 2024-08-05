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

#define MAX_ITERS_TILED 200

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat(DiffTensorView xyz,
           DiffTensorView inv_cov,
           DiffTensorView opacity,
           DiffTensorView rgb,
           DiffTensorView output_img)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (globalIdx.x >= output_img.size(1) || globalIdx.y >= output_img.size(0))
        return;

    float2 pix_coord = float2(globalIdx.x + 0.5, globalIdx.y + 0.5);
    uint2 resolution = uint2(output_img.size(0), output_img.size(1));

    float T = 1.0f;
    float3 local_output_rgb = float3(0., 0., 0.);
    [MaxIters(MAX_ITERS_TILED)]
    for (uint g_idx = 0; g_idx < xyz.size(0); g_idx++) {
        float3 g_xyz = float3(xyz[uint2(g_idx, 0)], 
                              xyz[uint2(g_idx, 1)],
                              xyz[uint2(g_idx, 2)]);
        float3 g_rgb = float3(rgb[uint2(g_idx, 0)],
                              rgb[uint2(g_idx, 1)],
                              rgb[uint2(g_idx, 2)]);
        float g_opacity = opacity[uint2(g_idx, 0)];
        float3 g_inv_cov = float3(inv_cov[uint3(g_idx, 0, 0)],
                                  inv_cov[uint3(g_idx, 1, 0)],
                                  inv_cov[uint3(g_idx, 1, 1)]);
        float2 d = { ndc2pix(g_xyz.x, output_img.size(1)) - pix_coord.x, 
                     ndc2pix(g_xyz.y, output_img.size(0)) - pix_coord.y };
        float power = -0.5f * (g_inv_cov.x * d.x * d.x +
                               g_inv_cov.z * d.y * d.y) - g_inv_cov.y * d.x * d.y;
        if (power > 0.0)
            continue;

        float alpha = min(0.99f, g_opacity * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;
        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f)
            continue;

        local_output_rgb.r += g_rgb.r * alpha * T;
        local_output_rgb.g += g_rgb.g * alpha * T;
        local_output_rgb.b += g_rgb.b * alpha * T;

        T = test_T;
    }
    output_img.storeOnce(uint3(globalIdx.y, globalIdx.x, 0), local_output_rgb.r);
    output_img.storeOnce(uint3(globalIdx.y, globalIdx.x, 1), local_output_rgb.g);
    output_img.storeOnce(uint3(globalIdx.y, globalIdx.x, 2), local_output_rgb.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 DiffTensorView xyz,
                 DiffTensorView inv_cov,
                 DiffTensorView opacity,
                 DiffTensorView rgb,
                 DiffTensorView output_img,
                 int grid_height,
                 int grid_width,
                 int tile_height,
                 int tile_width)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= output_img.size(1) || globalIdx.y >= output_img.size(0))
        return;

    float2 pix_coord = float2(globalIdx.x + 0.5, globalIdx.y + 0.5);

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    float T = 1.0f;
    float3 local_output_rgb = float3(0., 0., 0.);
    uint32_t iter_counter=0;

    [MaxIters(MAX_ITERS_TILED)]
    for (uint32_t i = tile_idx_start; i < tile_idx_end; i++) {
        uint32_t g_idx = uint32_t(sorted_gauss_idx[i]);
        float3 g_xyz = float3(xyz[uint2(g_idx, 0)],
                              xyz[uint2(g_idx, 1)],
                              xyz[uint2(g_idx, 2)]);
        float3 g_rgb = float3(rgb[uint2(g_idx, 0)],
                              rgb[uint2(g_idx, 1)],
                              rgb[uint2(g_idx, 2)]);
        float g_opacity = opacity[uint2(g_idx, 0)];
        float3 g_inv_cov = float3(inv_cov[uint3(g_idx, 0, 0)],
                                  inv_cov[uint3(g_idx, 1, 0)],
                                  inv_cov[uint3(g_idx, 1, 1)]);

        float2 d = { ndc2pix(g_xyz.x, output_img.size(1)) - pix_coord.x,
                     ndc2pix(g_xyz.y, output_img.size(0)) - pix_coord.y };
        float power = -0.5f * (g_inv_cov.x * d.x * d.x +
                               g_inv_cov.z * d.y * d.y) - g_inv_cov.y * d.x * d.y;
        if (power > 0.0)
            continue;

        float alpha = min(0.99f, g_opacity * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;
        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f)
            break;

        local_output_rgb.r += g_rgb.r * alpha * T;
        local_output_rgb.g += g_rgb.g * alpha * T;
        local_output_rgb.b += g_rgb.b * alpha * T;

        T = test_T;
        iter_counter++;
    }

    output_img.storeOnce(uint3(globalIdx.y, globalIdx.x, 0), local_output_rgb.r);
    output_img.storeOnce(uint3(globalIdx.y, globalIdx.x, 1), local_output_rgb.g);
    output_img.storeOnce(uint3(globalIdx.y, globalIdx.x, 2), local_output_rgb.b);
}
