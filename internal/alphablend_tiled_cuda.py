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

import math

import torch

import diff_gaussian_rasterization as dgr
import internal.camera as gs_camera
import internal.sh as gs_sh


def render(
    camera: gs_camera.Camera,
    xyz: torch.Tensor,
    shs: torch.Tensor,
    active_sh_degree: int,
    rgb_precomputed: torch.Tensor,
    opacity: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    cov3d_precomputed: torch.Tensor,
    bg_color: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Renders a 3D image from a set of Gaussian Splats.

  Args:
    camera: Camera to render.
    xyz: The means of the Gaussian Splats.
    shs: The spherical harmonics of the Gaussian Splats or None, if None the
      rgb_precomputed needs to have values.
    active_sh_degree: Number of active sh bands.
    rgb_precomputed: The rgb colors of the splats or None, if None shs needs to
      be set.
    opacity: The opacity of the splats.
    scale: The scales of the splats.
    rotation: The rotations of the splats as quaternions.
    cov3d_precomputed: Precomputed values of the covariance matrix or None.
      Should be defined only if scale and rotation is None.
    bg_color: The background color.


  Returns:
  A tuple that consists of:
   * The rendered image.
   * A tensor that holds the gradients of the Loss wrt the screenspace xyz.
   * The maximum screenspace radius of each gaussian.
  """
  # screnspace_points is used as a vessel to carry the viewpsace gradients
  screenspace_points = torch.zeros_like(xyz, requires_grad=True)

  tanfovx = math.tan(camera.fovx * 0.5)
  tanfovy = math.tan(camera.fovy * 0.5)

  raster_settings = dgr.GaussianRasterizationSettings(
      image_height=int(camera.height),
      image_width=int(camera.width),
      tanfovx=tanfovx,
      tanfovy=tanfovy,
      bg=bg_color,
      sh_degree=active_sh_degree,
      scale_modifier=1.0,
      viewmatrix=camera.world_view_transform.transpose(0, 1),
      projmatrix=camera.full_proj_transform.transpose(0, 1),
      campos=camera.camera_center,
      prefiltered=False,
      debug=False
  )

  rasterizer = dgr.GaussianRasterizer(raster_settings=raster_settings)
  rendered_image, radii = rasterizer(
      means3D=xyz,
      means2D=screenspace_points,
      shs=shs,
      colors_precomp=rgb_precomputed,
      opacities=opacity,
      scales=scale,
      rotations=rotation,
      cov3D_precomp=cov3d_precomputed,
  )

  return rendered_image, screenspace_points, radii


def render_alphablend_tiled_cuda(camera, splats):
    assert splats.active_sh_degree==0, f"active_sh_degree is {splats.active_sh_degree}"
    assert splats.mip_splatting_enabled is False

    bg_color = torch.tensor([0., 0, 0,], device="cuda")

    #shs = splats.get_concat_sh()
    rgb = gs_sh.sh_dc_to_rgb(splats.sh_dc).squeeze(1)

    scale, opacity = splats.get_scale_and_opacity_for_rendering()
    rendered_image, screenspace_points, radii= render(
        camera=camera,
        xyz=splats.xyz,
        shs=None,
        active_sh_degree=splats.active_sh_degree,
        rgb_precomputed=rgb,
        opacity=opacity,
        scale=scale,
        rotation=torch.nn.functional.normalize(splats.rotation),
        cov3d_precomputed=None,
        bg_color=bg_color
    )

    render_pkg = {
        'screenspace_points': screenspace_points,
        'visibility_filter': radii > 0,
        'radii': radii,
    }
    return rendered_image, render_pkg