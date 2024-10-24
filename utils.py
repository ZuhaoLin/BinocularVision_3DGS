import torch
from plyfile import PlyData
import numpy as np

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

   R = torch.eye(shape)
   R[:3, :3] = torch.Tensor([
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta), np.cos(theta), 0],
      [0, 0, 1]
      ])

   return as_type(R)

def get_Ry(theta, shape=3, as_type=torch.Tensor):
   R = torch.eye(shape)
   R[:3, :3] = torch.Tensor([
      [np.cos(theta), 0, np.sin(theta)],
      [0, 1, 0],
      [-np.sin(theta), 0, np.cos(theta)]
      ])
   
   return as_type(R)

def get_Rx(theta, shape=3, as_type=torch.Tensor):
   R = torch.eye(shape)
   R[:3, :3] = torch.Tensor([
      [1, 0, 0],
      [0, np.cos(theta), -np.sin(theta)],
      [0, np.sin(theta), np.cos(theta)]
      ])

   return as_type(R)

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