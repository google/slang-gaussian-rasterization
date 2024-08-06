import math
import torch

class RenderGrid():    
  def __init__(self, image_height, image_width, tile_width, tile_height):
    self.image_height = image_height
    self.image_width = image_width
    if tile_width is not None:
      self.tile_height = tile_height
      self.tile_width = tile_width
      self.grid_height = math.ceil(image_height / tile_height)
      self.grid_width = math.ceil(image_width  / tile_width)

def qvec2rotmat(r):
  """Converts a quaternion to a rotation matrix."""
  q = r / r.norm(dim=-1, keepdim=True)

  rot_mat = torch.zeros(q.shape[:-1] + (3, 3), device=q.device)

  # Assumption: quaternions are stored as [r, x, y, z].
  r = q[..., 0]
  x = q[..., 1]
  y = q[..., 2]
  z = q[..., 3]

  rot_mat[..., 0, 0] = 1 - 2 * (y * y + z * z)
  rot_mat[..., 0, 1] = 2 * (x * y - r * z)
  rot_mat[..., 0, 2] = 2 * (x * z + r * y)
  rot_mat[..., 1, 0] = 2 * (x * y + r * z)
  rot_mat[..., 1, 1] = 1 - 2 * (x * x + z * z)
  rot_mat[..., 1, 2] = 2 * (y * z - r * x)
  rot_mat[..., 2, 0] = 2 * (x * z - r * y)
  rot_mat[..., 2, 1] = 2 * (y * z + r * x)
  rot_mat[..., 2, 2] = 1 - 2 * (x * x + y * y)

  return rot_mat

C0 = 0.28209479177387814

def sh_dc_to_rgb(sh: torch.Tensor) -> torch.Tensor:
  return sh * C0 + 0.5

def get_covariance_from_scale_quat(scales, quats):
  """Returns the covariance matrix corresponding to a scale and a quaternion."""
  rotation_mat = qvec2rotmat(quats)
  # hidden matmul since scales is diagonal
  L = rotation_mat * scales[..., None, :]  # pylint: disable=invalid-name

  return L @ L.transpose(1, 2)


def common_properties_from_inria_GaussianModel(gaussian_model):
  xyz_ws = gaussian_model.get_xyz
  opacity = gaussian_model.get_opacity
  rotations = gaussian_model.get_rotation
  scales = gaussian_model.get_scaling
  sh_coeffs = gaussian_model.get_features
  
  return xyz_ws, rotations, scales, sh_coeffs, opacity

def common_properties_from_inria_Camera(camera):
  world_view_transform = camera.world_view_transform.T
  projection_matrix = camera.projection_matrix.T
  fovy = camera.FoVy
  fovx = camera.FoVx
  height = camera.image_height
  width = camera.image_width
  cam_pos = camera.camera_center

  return world_view_transform, projection_matrix, cam_pos, fovy, fovx, height, width 


def sort_values_by_keys(keys, values):
  sorted_keys, idxs = torch.sort(keys)
  sorted_val = values[idxs]
  return sorted_val, sorted_keys
