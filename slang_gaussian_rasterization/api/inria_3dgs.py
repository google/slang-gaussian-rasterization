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
"""Provides a thin wrapper to conviently interface with the official
Inria 3DGS code-base https://github.com/graphdeco-inria/gaussian-splatting"""

import torch
from slang_gaussian_rasterization.internal.alphablend_tiled_slang import render_alpha_blend_tiles_slang_raw

def common_properties_from_inria_GaussianModel(gaussian_model):
  """ Fetches all the Gaussian properties from the inria defined Gaussian Model object"""
  xyz_ws = gaussian_model.get_xyz
  opacity = gaussian_model.get_opacity
  rotations = gaussian_model.get_rotation
  scales = gaussian_model.get_scaling
  sh_coeffs = gaussian_model.get_features
  
  return xyz_ws, rotations, scales, sh_coeffs, opacity

def common_properties_from_inria_Camera(camera):
  """ Fetches all the Camera properties from the inria defined object"""
  world_view_transform = camera.world_view_transform.T
  projection_matrix = camera.projection_matrix.T
  fovy = camera.FoVy
  fovx = camera.FoVx
  height = camera.image_height
  width = camera.image_width
  cam_pos = camera.camera_center

  return world_view_transform, projection_matrix, cam_pos, fovy, fovx, height, width 

def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier = 1.0, override_color = None):
  """ Implements the Interface defined in the inria code-base."""
  assert scaling_modifier == 1.0, "scaling_modifier is not supported in the slang-gaussian-rasterization."
  assert override_color is None, "override_color is not support in the slang-gaussian-rasterization."
  assert pipe.convert_SHs_python is False, "convert_SHs_python is not supported."
  assert pipe.compute_cov3D_python is False, "compute_cov3D_python is not supported."
  assert pipe.debug is False, "debug mode is not supported."
  assert torch.equal(bg_color, torch.zeros_like(bg_color)), "only black background is supported currently."

  active_sh = pc.active_sh_degree
  xyz_ws, rotations, scales, sh_coeffs, opacity = common_properties_from_inria_GaussianModel(pc)
  world_view_transform, proj_mat, cam_pos, fovy, fovx, height, width = common_properties_from_inria_Camera(viewpoint_camera)  


  render_pkg = render_alpha_blend_tiles_slang_raw(xyz_ws, rotations, scales, opacity, 
                                                  sh_coeffs, active_sh,
                                                  world_view_transform, proj_mat, cam_pos,
                                                  fovy, fovx, height, width)
  
  return render_pkg
 