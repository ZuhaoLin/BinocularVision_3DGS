import torch
from plyfile import PlyData
import numpy as np
from typing import Union

def lookat_matrix(lookat_point, up, position):
   position = -position
   lookat_mat = torch.eye(4)

   lookat = position - lookat_point
   lookat, up = normalize(lookat), normalize(up)

   right = torch.cross(up, lookat)
   cam_up = torch.cross(lookat, right)

   rot = torch.stack([right, cam_up, lookat], 0)

   pos_c2w = position @ rot.T

   lookat_mat[:-1, -1] = pos_c2w
   lookat_mat[:-1, :-1] = rot

   return lookat_mat

def normalize(vec):
   return vec / torch.linalg.norm(vec)

def get_Rz(theta, shape=3, as_type=torch.Tensor):

   assert shape >= 3, 'shape must be larger or equal to 3'
   R = torch.eye(shape)

   if np.isscalar(theta):
      R[:3, :3] = torch.Tensor([
         [np.cos(theta), -np.sin(theta), 0],
         [np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]
         ])
   elif isinstance(theta, (torch.Tensor, np.ndarray)):
      assert theta.ndim == 1, 'theta is not one dimensional'
      R = R.repeat(theta.size, 1, 1)
      cs = torch.Tensor(np.cos(theta))
      sn = torch.Tensor(np.sin(theta))

      R[:, [0, 1], [0, 1]] = cs[:, None].repeat(1, 2)
      R[:, 0, 1] = -sn
      R[:, 1, 0] = sn
   else:
      raise TypeError('theta is not a scalar, torch.Tensor, or np.ndarray')
   
   return as_type(R)

def get_Ry(
      theta: Union[float, torch.Tensor, np.ndarray],
      shape=3,
      as_type=torch.Tensor
      ):
   
   assert shape >= 3, 'shape must be larger or equal to 3'
   R = torch.eye(shape)

   if np.isscalar(theta):
      R[:3, :3] = torch.Tensor([
         [np.cos(theta), 0, np.sin(theta)],
         [0, 1, 0],
         [-np.sin(theta), 0, np.cos(theta)]
         ])
   elif isinstance(theta, (torch.Tensor, np.ndarray)):
      assert theta.ndim == 1, 'theta is not one dimensional'
      R = R.repeat(theta.size, 1, 1)
      cs = torch.Tensor(np.cos(theta))
      sn = torch.Tensor(np.sin(theta))

      R[:, [0, 2], [0, 2]] = cs[:, None].repeat(1, 2)
      R[:, 0, 2] = sn
      R[:, 2, 0] = -sn
   else:
      raise TypeError('theta is not a scalar, torch.Tensor, or np.ndarray')
   
   return as_type(R)

def get_Rx(theta, shape=3, as_type=torch.Tensor):
   
   assert shape >= 3, 'shape must be larger or equal to 3'
   R = torch.eye(shape)

   if np.isscalar(theta):
      R[:3, :3] = torch.Tensor([
         [1, 0, 0],
         [0, np.cos(theta), -np.sin(theta)],
         [0, np.sin(theta), np.cos(theta)]
         ])
   elif isinstance(theta, (torch.Tensor, np.ndarray)):
      assert theta.ndim == 1, 'theta is not one dimensional'
      R = R.repeat(theta.size, 1, 1)
      cs = torch.Tensor(np.cos(theta))
      sn = torch.Tensor(np.sin(theta))

      R[:, [1, 2], [1, 2]] = cs[:, None].repeat(1, 2)
      R[:, 1, 2] = -sn
      R[:, 2, 1] = sn
   else:
      raise TypeError('theta is not a scalar, torch.Tensor, or np.ndarray')
   
   return as_type(R)

def quick_viewmat_inv(viewmat: torch.Tensor):
   '''
   Does a quick and computationally less expensive inverse of viewmats
   '''
   viewmat_inv = torch.eye(4)
   R = viewmat[:3, :3]                                                                         # Rotation component
   T = viewmat[:3, 3][:, None]

   # if gsplat_convention:
   #     R_convert = torch.diag(torch.tensor([1, -1, -1])).type(torch.float)
   #     R = R @ R_convert

   R_inv = R.T                                                                                 # inv of rotation == transpose of rotation
   T_inv = -R_inv @ T                                                                          # inv of translation

   viewmat_inv[:3, :3] = R_inv
   viewmat_inv[:3, 3:4] = T_inv
   return viewmat_inv

def quick_multiviewmat_inv(viewmats: torch.Tensor):
   if viewmats.ndim != 3:
      raise ValueError('viewmat must be three dimensional (ndim==3)')
   
   Rs = viewmats[:, :3, :3]
   Ts = viewmats[:, :3, 3].reshape(-1, 3, 1)

   R_inv = Rs.transpose(1, 2)
   T_inv = -R_inv @ Ts

   viewmats_inv = torch.eye(4).repeat(viewmats.shape[0], 1, 1)
   viewmats_inv[:, :3, :3] = R_inv
   viewmats_inv[:, :3, -1] = T_inv[:, :, 0]

   return viewmats_inv

def process_ply(ply_file_path):
   plydata = PlyData.read(ply_file_path)
   vert = plydata["vertex"]
   sorted_indices = np.argsort(
      -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
      / (1 + np.exp(-vert["opacity"]))
   )

   N = len(sorted_indices)
   values = dict()
   values.update({'position': np.zeros((N, 3)),
                  'scales': np.zeros((N, 3)),
                  'rot': np.zeros((N, 4)),
                  'color': np.zeros((N, 3)),
                  'opacity': np.zeros(N)})

   for i, idx in enumerate(sorted_indices):
      v = plydata["vertex"][idx]
      position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
      scales = np.exp(
         np.array(
               [v["scale_0"], v["scale_1"], v["scale_2"]],
               dtype=np.float32,
         )
      )
      rot = np.array(
         [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
         dtype=np.float32,
      )
      SH_C0 = 0.28209479177387814
      color = np.array(
         [
               0.5 + SH_C0 * v["f_dc_0"],
               0.5 + SH_C0 * v["f_dc_1"],
               0.5 + SH_C0 * v["f_dc_2"],
               # 1 / (1 + np.exp(-v["opacity"])),
         ]
      )
      # color = np.clip(color, 0, 1)
      # color = np.array([v['red'], v['green'], v['blue']])
      # opacity = np.array(
      #    v["opacity"])
      
      opacity = 1 / (1 + np.exp(-v["opacity"]))

      # Insert into dictionary
      values['position'][i, :] = position
      values['scales'][i, :] = scales
      values['rot'][i, :] = rot
      values['color'][i, :] = color
      values['opacity'][i] = opacity

   return values