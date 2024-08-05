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


def render_alphablend_slang(camera, splats):
    render_grid = gs_render_utils.RenderGrid(camera.height,
                                             camera.width,
                                             tile_height=16,
                                             tile_width=16)

    cov_worldspace, rgb, opacity = gs_render_utils.get_cov_rgb_opacity(splats)

    xyz_viewspace, cov_viewspace = vertex_shader(camera, splats.xyz, cov_worldspace)
    inv_cov_viewspace = cov_viewspace.inverse()

    sorted_xyz_viewspace, sorted_inv_cov_viewspace, sorted_rgb, sorted_opacity = gs_render_utils.z_sort_gaussians(xyz_viewspace, inv_cov_viewspace, rgb, opacity)
        
    image_rgb = AlphaBlendRender.apply(sorted_xyz_viewspace,
                                       sorted_inv_cov_viewspace,
                                       sorted_opacity,
                                       sorted_rgb,
                                       render_grid)
    return image_rgb.permute(2,0,1), None


class AlphaBlendRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, inv_cov, opacity, rgb, render_grid, device="cuda"):
        output_img = torch.zeros((render_grid.image_height, 
                                  render_grid.image_width, 3), 
                                  device=device)

        splat_kernel_with_args = slang_modules.alpha_blend.splat(
            xyz=xyz, inv_cov=inv_cov, opacity=opacity, rgb=rgb, output_img=output_img
        )
        splat_kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )

        ctx.save_for_backward(xyz, inv_cov, opacity, rgb, output_img)
        ctx.render_grid = render_grid
        return output_img

    @staticmethod
    def backward(ctx, grad_output_img):
        (xyz, inv_cov, opacity, rgb, output_img) = ctx.saved_tensors    
        render_grid = ctx.render_grid

        xyz_grad = torch.zeros_like(xyz)
        inv_cov_grad = torch.zeros_like(inv_cov)
        opacity_grad = torch.zeros_like(opacity)
        rgb_grad = torch.zeros_like(rgb)

        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        #
        kernel_with_args = slang_modules.alpha_blend.splat.bwd(xyz=(xyz, xyz_grad),
                                        inv_cov=(inv_cov, inv_cov_grad),
                                        opacity=(opacity, opacity_grad),
                                        rgb=(rgb, rgb_grad),
                                        output_img=(output_img, grad_output_img))
        kernel_with_args.launchRaw(
            blockSize=(render_grid.tile_width, 
                       render_grid.tile_height, 1),
            gridSize=(render_grid.grid_width, 
                      render_grid.grid_height, 1)
        )
        
        return xyz_grad, inv_cov_grad, opacity_grad, rgb_grad, None, None, None
    