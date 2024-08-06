from submodules.slang_gaussian_rasterization.internal.utils import common_properties_from_inria_GaussianModel
from submodules.slang_gaussian_rasterization.internal.utils import common_properties_from_inria_Camera
from submodules.slang_gaussian_rasterization.internal.alphablend_tiled_slang_sai import render_alpha_blend_tiles_slang_raw

def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier = 1.0, override_color = None):
  assert scaling_modifier == 1.0, "scaling_modifier is not supported in the slang-gaussian-rasterization."
  assert override_color is None, "override_color is not support in the slang-gaussian-rasterization."

  active_sh = pc.active_sh_degree
  xyz_ws, rotations, scales, sh_coeffs, opacity = common_properties_from_inria_GaussianModel(pc)
  world_view_transform, proj_mat, cam_pos, fovy, fovx, height, width = common_properties_from_inria_Camera(viewpoint_camera)  


  render_pkg = render_alpha_blend_tiles_slang_raw(xyz_ws, rotations, scales, opacity, 
                                                  sh_coeffs, active_sh,
                                                  world_view_transform, proj_mat, cam_pos,
                                                  fovy, fovx, height, width)
  
  return render_pkg
