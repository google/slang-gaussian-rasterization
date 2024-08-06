# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import submodules.slang_gaussian_rasterization.internal.utils as gs_render_utils
import submodules.slang_gaussian_rasterization.internal.slang.slang_modules as slang_modules
from submodules.slang_gaussian_rasterization.internal.tile_shader_slang import tile_shader

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def render_alpha_blend_tiles_slang_raw(xyz_ws, rotations, scales, opacity, 
                                       sh_coeffs, active_sh,
                                       world_view_transform, proj_mat, cam_pos,
                                       fovy, fovx, height, width, tile_size=16):
    
    render_grid = gs_render_utils.RenderGrid(height,
                                             width,
                                             tile_height=tile_size,
                                             tile_width=tile_size)
    sorted_gauss_idx, tile_ranges, radii, xyz_vs, inv_cov_vs, rgb = tile_shader(xyz_ws,
                                                                                rotations,
                                                                                scales,
                                                                                sh_coeffs,
                                                                                active_sh,
                                                                                world_view_transform,
                                                                                proj_mat,
                                                                                cam_pos,
                                                                                fovy,
                                                                                fovx,
                                                                                render_grid)
   
    # retain_grad fails if called with torch.no_grad() under evaluation
    try:
        xyz_vs.retain_grad()
    except:
        pass

    image_rgb = AlphaBlendTiledRender.apply(
        sorted_gauss_idx,
        tile_ranges,
        xyz_vs,
        inv_cov_vs,
        opacity,
        rgb,
        render_grid)
    
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...],
        'viewspace_points': xyz_vs,
        'visibility_filter': radii > 0,
        'radii': radii,
    }

    return render_pkg


class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                sorted_gauss_idx, tile_ranges,
                xyz_vs, inv_cov_vs, opacity, rgb, render_grid, device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 4), 
                                  device=device)
        last_contributor = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 1),
                                  dtype=torch.int32, device=device)

        splat_kernel_with_args = slang_modules.alpha_blend_sai.splat_tiled(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz_vs=xyz_vs, inv_cov_vs=inv_cov_vs, 
            opacity=opacity, rgb=rgb, 
            output_img=output_img,
            last_contributor=last_contributor,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        ctx.save_for_backward(sorted_gauss_idx, tile_ranges,
                              xyz_vs, inv_cov_vs, opacity, rgb, 
                              output_img, last_contributor)
        ctx.render_grid = render_grid

        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (sorted_gauss_idx, tile_ranges, 
         xyz_vs, inv_cov_vs, opacity, rgb, 
         output_img, last_contributor) = ctx.saved_tensors
        render_grid = ctx.render_grid

        xyz_vs_grad = torch.zeros_like(xyz_vs)
        inv_cov_vs_grad = torch.zeros_like(inv_cov_vs)
        opacity_grad = torch.zeros_like(opacity)
        rgb_grad = torch.zeros_like(rgb)
        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        #
        kernel_with_args = slang_modules.alpha_blend_sai.splat_tiled.bwd(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz_vs=(xyz_vs, xyz_vs_grad),
            inv_cov_vs=(inv_cov_vs, inv_cov_vs_grad),
            opacity=(opacity, opacity_grad),
            rgb=(rgb, rgb_grad),
            output_img=(output_img, grad_output_img),
            last_contributor=last_contributor,
            grid_height=render_grid.grid_height,
            grid_width=render_grid.grid_width,
            tile_height=render_grid.tile_height,
            tile_width=render_grid.tile_width)
        
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        
        return None, None, xyz_vs_grad, inv_cov_vs_grad, opacity_grad, rgb_grad, None, None
