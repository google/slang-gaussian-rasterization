from typing import Dict, Optional, Tuple
from typing_extensions import Literal
import math
import torch
from torch import Tensor
from slang_gaussian_rasterization.internal.alphablend_tiled_slang import render_alpha_blend_tiles_slang_raw


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_slang_projection_matrix(znear, zfar, fy, fx, height, width, device):
    tanHalfFovX = width/(2*fx)
    tanHalfFovY = height/(2*fy)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    P = torch.tensor([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0 ],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0 ],
       [0.0,                              0.0,                          z_sign * zfar / (zfar - znear),  -(zfar * znear) / (zfar - znear) ],
       [0.0,                              0.0,                          z_sign,                          0.0 ]
    ], device=device)

    return P

def common_camera_properties_from_gsplat(viewmats, Ks, height, width):
  """ Fetches all the Camera properties from the inria defined object"""
  zfar = 100.0
  znear = 0.01
  
  world_view_transform = viewmats
  fx = Ks[0,0]
  fy = Ks[1,1]
  projection_matrix = get_slang_projection_matrix(znear, zfar, fy, fx, height, width, Ks.device)
  fovx = focal2fov(fx, width)
  fovy = focal2fov(fy, height)

  cam_pos = viewmats.inverse()[:, 3]

  return world_view_transform, projection_matrix, cam_pos, fovy, fovx
 


def rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
) -> Tuple[Tensor, Tensor, Dict]:

  assert viewmats.shape[0] == 1, "Camera Batching is not support in the slang-gaussian-rasterization"
  assert Ks.shape[0] == 1, "Camera Batching is not support in the slang-gaussian-rasterization"
  assert not(len(colors.shape) == 4 and colors.shape[0] == 1), "Camera Batching is not support in the slang-gaussian-rasterization"
  assert render_mode == "RGB", "Currently only render_mode=\"RGB\" is supported."
  assert rasterize_mode == "classic", "Currently only rasterize_mode=\"classic\" is supported."
  assert absgrad == False, "Currently only absgrd=False is supported."
  assert backgrounds is None
  assert packed == False, "Currently only packed=False is supported."
  assert sparse_grad == False, "Currently only sparce_grad=False is supported."
  assert distributed == False, "Currently ony distributed=False is supported."

  world_view_transform, projection_matrix, cam_pos, fovy, fovx = common_camera_properties_from_gsplat(viewmats[0], Ks[0], height, width)

  render_pkg = render_alpha_blend_tiles_slang_raw(means, quats, scales, opacities, 
                                                  colors, sh_degree,
                                                  world_view_transform, projection_matrix, cam_pos,
                                                  fovy, fovx, height, width, tile_size=tile_size)


  meta = {"radii": render_pkg["radii"][None, ...],
          "means2d": render_pkg["viewspace_points"]}

  return render_pkg["render"].permute(1,2,0)[None,...], None, meta
