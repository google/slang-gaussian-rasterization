import torch
import submodules.slang_gaussian_rasterization.internal.utils as utils
import submodules.slang_gaussian_rasterization.internal.slang.slang_modules as slang_modules
import math
import numpy as np

def tile_shader(xyz_ws,
                rotations,
                scales,
                sh_coeffs,
                active_sh,
                world_view_transform,
                proj_mat,
                cam_pos,
                fovy,
                fovx,
                render_grid):
    n_points = xyz_ws.shape[0]
    tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb = TilesTouchedPerGaussian.apply(xyz_ws, 
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
    with torch.no_grad():
      index_buffer_offset = torch.cumsum(tiles_touched, dim=0, dtype=tiles_touched.dtype)
      total_size_index_buffer = index_buffer_offset[-1]
      unsorted_keys = torch.zeros((total_size_index_buffer,), 
                                  device="cuda", 
                                  dtype=torch.int64)
      unsorted_gauss_idx = torch.zeros((total_size_index_buffer,), 
                                      device="cuda", 
                                      dtype=torch.int32)
      slang_modules.tile_shader.generate_keys(xyz_vs=xyz_vs,
                                                rect_tile_space=rect_tile_space,
                                                index_buffer_offset=index_buffer_offset,
                                                out_unsorted_keys=unsorted_keys,
                                                out_unsorted_gauss_idx=unsorted_gauss_idx,
                                                grid_height=render_grid.grid_height,
                                                grid_width=render_grid.grid_width).launchRaw(
            blockSize=(256, 1, 1),
            gridSize=(math.ceil(n_points/256), 1, 1)
      )    


      sorted_gauss_idx, sorted_keys = utils.sort_values_by_keys(unsorted_keys, unsorted_gauss_idx)

      tile_ranges = torch.zeros((render_grid.grid_height*render_grid.grid_width, 2), 
                                device="cuda",
                                dtype=torch.int32)
      slang_modules.tile_shader.compute_tile_ranges(sorted_keys=sorted_keys,
                                                    out_tile_ranges=tile_ranges).launchRaw(
              blockSize=(256, 1, 1),
              gridSize=(math.ceil(total_size_index_buffer/256), 1, 1)
      )

    return sorted_gauss_idx, tile_ranges, radii, xyz_vs, inv_cov_vs, rgb


class TilesTouchedPerGaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                xyz_ws, rotations, scales,
                sh_coeffs, active_sh,
                world_view_transform, proj_mat, cam_pos,
                fovy, fovx,
                render_grid, device="cuda"):
      n_points = xyz_ws.shape[0]
      tiles_touched = torch.zeros((n_points), 
                                  device="cuda", 
                                  dtype=torch.int32)
      rect_tile_space = torch.zeros((n_points, 4), 
                                    device="cuda", 
                                    dtype=torch.float)
      radii = torch.zeros((n_points),
                          device="cuda",
                          dtype=torch.int32)
      
      xyz_vs = torch.zeros((n_points, 3),
                          device="cuda",
                          dtype=torch.float)
      inv_cov_vs = torch.zeros((n_points, 2, 2),
                                device="cuda",
                                dtype=torch.float)
      rgb = torch.zeros((n_points, 3),
                        device="cuda",
                        dtype=torch.float)
      
      slang_modules.tile_shader.tiles_touched_per_gaussian(xyz_ws=xyz_ws,
                                                           rotations=rotations,
                                                           scales=scales,
                                                           sh_coeffs=sh_coeffs,
                                                           active_sh=active_sh,
                                                           world_view_transform=world_view_transform,
                                                           proj_mat=proj_mat,
                                                           cam_pos=cam_pos,
                                                           out_tiles_touched=tiles_touched,
                                                           out_rect_tile_space=rect_tile_space,
                                                           out_radii=radii,
                                                           out_xyz_vs=xyz_vs,
                                                           out_inv_cov_vs=inv_cov_vs,
                                                           out_rgb=rgb,
                                                           fovy=fovy,
                                                           fovx=fovx,
                                                           image_height=render_grid.image_height,
                                                           image_width=render_grid.image_width,
                                                           grid_height=render_grid.grid_height,
                                                           grid_width=render_grid.grid_width,
                                                           tile_height=render_grid.tile_height,
                                                           tile_width=render_grid.tile_width).launchRaw(
              blockSize=(256, 1, 1),
              gridSize=(math.ceil(n_points/256), 1, 1)
      )

      ctx.save_for_backward(xyz_ws, rotations, scales, sh_coeffs, world_view_transform, proj_mat, cam_pos,
                            tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb)
      ctx.render_grid = render_grid
      ctx.fovy = fovy
      ctx.fovx = fovx
      ctx.active_sh = active_sh

      return tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb
    
    @staticmethod
    def backward(ctx, grad_tiles_touched, grad_rect_tile_space, grad_radii, grad_xyz_vs, grad_inv_cov_vs, grad_rgb):
        (xyz_ws, rotations, scales, sh_coeffs, world_view_transform, proj_mat, cam_pos,
         tiles_touched, rect_tile_space, radii, xyz_vs, inv_cov_vs, rgb) = ctx.saved_tensors
        render_grid = ctx.render_grid
        fovy = ctx.fovy
        fovx = ctx.fovx
        active_sh = ctx.active_sh

        n_points = xyz_ws.shape[0]

        grad_xyz_ws = torch.zeros_like(xyz_ws)
        grad_rotations = torch.zeros_like(rotations)
        grad_scales = torch.zeros_like(scales)
        grad_sh_coeffs = torch.zeros_like(sh_coeffs)

        slang_modules.tile_shader.tiles_touched_per_gaussian.bwd(xyz_ws=(xyz_ws, grad_xyz_ws),
                                                                 rotations=(rotations, grad_rotations),
                                                                 scales=(scales, grad_scales),
                                                                 sh_coeffs=(sh_coeffs, grad_sh_coeffs),
                                                                 active_sh=active_sh,
                                                                 world_view_transform=world_view_transform,
                                                                 proj_mat=proj_mat,
                                                                 cam_pos=cam_pos,
                                                                 out_tiles_touched=tiles_touched,
                                                                 out_rect_tile_space=rect_tile_space,
                                                                 out_radii=radii,
                                                                 out_xyz_vs=(xyz_vs, grad_xyz_vs),
                                                                 out_inv_cov_vs=(inv_cov_vs, grad_inv_cov_vs),
                                                                 out_rgb=(rgb, grad_rgb),
                                                                 fovy=fovy,
                                                                 fovx=fovx,
                                                                 image_height=render_grid.image_height,
                                                                 image_width=render_grid.image_width,
                                                                 grid_height=render_grid.grid_height,
                                                                 grid_width=render_grid.grid_width,
                                                                 tile_height=render_grid.tile_height,
                                                                 tile_width=render_grid.tile_width).launchRaw(
              blockSize=(256, 1, 1),
              gridSize=(math.ceil(n_points/256), 1, 1)
        )
        return grad_xyz_ws, grad_rotations, grad_scales, grad_sh_coeffs, None, None, None, None, None, None, None