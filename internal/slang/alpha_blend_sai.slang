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

[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (1 - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}

float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = pixel_state_t_n.a / (1 - gauss_rgba_t_n.a);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, transmittance_t_nm1);
}

groupshared Gaussian collected_gaussians[16 * 16];
groupshared uint32_t collected_idx[16 * 16];

[BackwardDerivative(bwd_alpha_blend)] // Use a custom derivative so that we can hand-write the structure of the reverse loop
float4 alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                   DiffTensorView xyz_vs,
                   DiffTensorView inv_cov_vs,
                   DiffTensorView opacity,
                   DiffTensorView rgb,
                   DiffTensorView final_pixel_state,
                   TensorView<int32_t> last_contributor,
                   uint32_t2 pix_coord,
                   uint32_t tile_idx_start,
                   uint32_t tile_idx_end,
                   uint32_t tile_height,
                   uint32_t tile_width,
                   uint32_t H,
                   uint32_t W)
{
    float2 center_pix_coord = pix_coord;
    float4 curr_pixel_state = float4(0.f, 0.f, 0.f, 1.f);
    uint32_t block_size = tile_height * tile_width;

    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool done = !is_inside;

    uint32_t contributor = 0;
    uint32_t n_contributors = 0;

    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    int toDo = tile_idx_end - tile_idx_start;

    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++, toDo -= block_size)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + progress]);
            collected_gaussians[thread_rank] = load_gaussian(coll_id, xyz_vs, inv_cov_vs, opacity, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        for (int j = 0; !done && j < min(block_size, toDo); j++)
        {
            contributor++;
            Gaussian g = collected_gaussians[j];
            float4 gauss_rgba = evaluate_gaussian(g, center_pix_coord, H, W); // [Differentiable] function with all the relevant logic for
                                                                              // computing this blob's contribution. This method will be auto-diffed

            if (gauss_rgba.a < 1.0f / 255.0f)
                continue;
            // Combine the contribution into the throughput in an undo-able way.
            float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba); // [Differentiable] function
            if (new_pixel_state.a < 0.0001f) {
                done = true;
                break;
            }
            n_contributors = contributor;
            curr_pixel_state = new_pixel_state;
        }
    }

    if (is_inside)
        last_contributor[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)] = n_contributors;

    return curr_pixel_state;
}

void bwd_alpha_blend(TensorView<int32_t> sorted_gauss_idx,
                     DiffTensorView xyz_vs,
                     DiffTensorView inv_cov_vs,
                     DiffTensorView opacity,
                     DiffTensorView rgb,
                     DiffTensorView final_pixel_state,
                     TensorView<int32_t> last_contributor,
                     uint32_t2 pix_coord,
                     uint32_t tile_idx_start,
                     uint32_t tile_idx_end,
                     uint32_t tile_height,
                     uint32_t tile_width,
                     uint32_t H,
                     uint32_t W,
                     float4 d_current_pixel_state)
{
    // Load the final pixel state.
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    bool done = !is_inside;
    uint32_t block_size = tile_height * tile_width;
    const int rounds = ((tile_idx_end - tile_idx_start + block_size - 1) / block_size);
    int toDo = tile_idx_end - tile_idx_start;
    uint32_t contributor = toDo;

    float4 current_pixel_state;
    int32_t last_idx;
    if (is_inside) {
        current_pixel_state = float4(final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2)],
                                     final_pixel_state[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3)]);
        last_idx = last_contributor[uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0)];
    }
    const int last_contributor = is_inside ? last_idx : 0;

    float2 center_pix_coord = pix_coord;

    DifferentialPair<float2> dp_center_pix_coord = diffPair(center_pix_coord);


    uint32_t thread_rank = cudaThreadIdx().y * cudaBlockDim().x + cudaThreadIdx().x;
    for (int i = 0; i < rounds; i++, toDo -= block_size)
    {
        // Collectively fetch per-Gaussian data from global to shared
        AllMemoryBarrierWithGroupSync();
        int progress = i * block_size + thread_rank;
        if (tile_idx_start + progress < tile_idx_end)
        {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_end - progress - 1]);
            collected_idx[thread_rank] = coll_id;
            collected_gaussians[thread_rank] = load_gaussian(coll_id, xyz_vs, inv_cov_vs, opacity, rgb);
        }
        AllMemoryBarrierWithGroupSync();
        for (int j = 0; !done && j < min(block_size, toDo); j++)
        {
            contributor--;
            if (contributor >= last_contributor)
                continue;
            uint32_t g_idx = collected_idx[j];
            Gaussian g = collected_gaussians[j];

            float4 gauss_rgba = evaluate_gaussian(g, center_pix_coord, H, W);

            if (gauss_rgba.a < 1.0f / 255.0f)
                continue;

            // Undo pixel state
            current_pixel_state = undo_pixel_state(current_pixel_state, gauss_rgba);

            // Back-prop automatically through blending and gaussian evaluation.
            DifferentialPair<Gaussian> dp_g = diffPair(g);
            DifferentialPair<float4> dp_gauss_rgba = diffPair(gauss_rgba);
            DifferentialPair<float4> dp_current_pixel_state = diffPair(current_pixel_state);

            bwd_diff(update_pixel_state)(dp_current_pixel_state, dp_gauss_rgba, d_current_pixel_state);
            d_current_pixel_state = dp_current_pixel_state.getDifferential();
            bwd_diff(evaluate_gaussian)(dp_g, dp_center_pix_coord, H, W, dp_gauss_rgba.d);
            bwd_diff(load_gaussian)(g_idx, xyz_vs, inv_cov_vs, opacity, rgb, dp_g.d);
        }
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void splat_tiled(TensorView<int32_t> sorted_gauss_idx,
                 TensorView<int32_t> tile_ranges,
                 DiffTensorView xyz_vs,
                 DiffTensorView inv_cov_vs,
                 DiffTensorView opacity,
                 DiffTensorView rgb,
                 DiffTensorView output_img,
                 TensorView<int32_t> last_contributor,
                 int grid_height,
                 int grid_width,
                 int tile_height,
                 int tile_width)
{
    uint32_t3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint32_t2 pix_coord = globalIdx.xy;

    uint32_t tile_idx = cudaBlockIdx().y * grid_width + cudaBlockIdx().x;
    uint32_t tile_idx_start = uint32_t(tile_ranges[uint2(tile_idx, 0)]);
    uint32_t tile_idx_end = uint32_t(tile_ranges[uint2(tile_idx, 1)]);

    bool is_inside = (pix_coord.x < output_img.size(1) && pix_coord.y < output_img.size(0));

    float4 pixel_state = alpha_blend(sorted_gauss_idx,
                                     xyz_vs,
                                     inv_cov_vs,
                                     opacity,
                                     rgb,
                                     output_img,
                                     last_contributor,
                                     pix_coord,
                                     tile_idx_start,
                                     tile_idx_end,
                                     tile_height,
                                     tile_width,
                                     output_img.size(0),
                                     output_img.size(1));

    if (is_inside) {
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 0), pixel_state.r);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 1), pixel_state.g);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 2), pixel_state.b);
      output_img.storeOnce(uint3(uint32_t(pix_coord.y), uint32_t(pix_coord.x), 3), pixel_state.a);
    }
}
