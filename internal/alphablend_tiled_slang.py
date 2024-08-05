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
import internal.splatter.render_utils as gs_render_utils
import internal.splatter.slang_shaders.slang_modules as slang_modules
from internal.splatter.vertex_shader import vertex_shader
from internal.splatter.tile_shader_slang import tile_shader

def render_alphablend_tiled_slang(camera, splats, tile_size):
    render_grid = gs_render_utils.RenderGrid(camera.height,
                                             camera.width,
                                             tile_height=tile_size,
                                             tile_width=tile_size)

    cov_worldspace, rgb, opacity = gs_render_utils.get_cov_rgb_opacity(splats)

    xyz_viewspace, cov_viewspace = vertex_shader(camera, splats.xyz, cov_worldspace)
    inv_cov_viewspace = cov_viewspace.inverse()
    sorted_gauss_idx, tile_ranges, radii = tile_shader(
                        xyz_viewspace,
                        cov_viewspace,
                        render_grid)
    tile_size = (tile_ranges[:,1] - tile_ranges[:,0])
    assert torch.all(tile_size<200), f"{tile_size.max()}"
    image_rgb = AlphaBlendTiledRender.apply(
        sorted_gauss_idx,
        tile_ranges,
        xyz_viewspace,
        inv_cov_viewspace,
        opacity,
        rgb,
        render_grid)
    
    render_pkg = {
        #'screenspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
    }
    return image_rgb.permute(2,0,1), render_pkg

class AlphaBlendTiledRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sorted_gauss_idx, tile_ranges, xyz, inv_cov, opacity, rgb, render_grid, device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 3), 
                                  device=device)

        splat_kernel_with_args = slang_modules.alpha_blend.splat_tiled(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz=xyz, inv_cov=inv_cov, 
            opacity=opacity, rgb=rgb, 
            output_img=output_img,
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
                              xyz, inv_cov, opacity, rgb, 
                              output_img)
        ctx.render_grid = render_grid
        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (sorted_gauss_idx, tile_ranges, xyz, inv_cov, opacity, rgb, output_img) = ctx.saved_tensors
        render_grid = ctx.render_grid

        xyz_grad = torch.zeros_like(xyz)
        inv_cov_grad = torch.zeros_like(inv_cov)
        opacity_grad = torch.zeros_like(opacity)
        rgb_grad = torch.zeros_like(rgb)
        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        #
        kernel_with_args = slang_modules.alpha_blend.splat_tiled.bwd(
            sorted_gauss_idx=sorted_gauss_idx,
            tile_ranges=tile_ranges,
            xyz=(xyz, xyz_grad),
            inv_cov=(inv_cov, inv_cov_grad),
            opacity=(opacity, opacity_grad),
            rgb=(rgb, rgb_grad),
            output_img=(output_img, grad_output_img),
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
        
        return None, None, xyz_grad, inv_cov_grad, opacity_grad, rgb_grad, None, None
