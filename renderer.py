import numpy as np
import torch
import torch.nn.functional as F

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization
from nerfstudio.cameras import camera_utils

from plyfile import PlyData
import sklearn.preprocessing
import cv2 as cv
from binocular_vision import UP

import utils
import simworld

import os

def main():
   print(os.getcwd())
   splat_filepath = r'./exports/IMG_5435/splat.ply'                                                     # Specify ply file of splat
   c2w = torch.eye(4)                                                                                    # Initial camera transform
   w2c = torch.eye(4)
   height, width = 1080, 1920                                                                         # Image height and width
   camera_intrinsic = torch.Tensor([[500, 0, int(width/2)], [0, 500, int(height/2)], [0, 0, 1]])          # Arbitrary camera intrinsics matrix

   t = 0.01                                                                                           # Rate to adjust camera

   torch.manual_seed(42)
   device = "cuda"
   data = utils.process_ply(splat_filepath)                                                                 # Read ply file

   # # load .ply file data for processing
   # means = torch.from_numpy(data['position']).float().to(device)
   # quats = torch.from_numpy(data['rot']).float().to(device)
   # scales = torch.from_numpy(data['scales']).float().to(device)
   # opacities = torch.from_numpy(data['opacity']).float().to(device)
   # colors = torch.from_numpy(data['color']).float().to(device)
   
   # # Camera intrinsics
   # Ks = torch.from_numpy(camera_intrinsic)[None, :, :].float().to(device)

   # # Initialize the camera properties
   # x, y, z = 0, 0, 0

   # Plot means for viz purposes
   # import matplotlib.pyplot as plt
   # ax = plt.figure().add_subplot(projection='3d')
   # ax.plot(data['position'][:, 0], data['position'][:, 1], data['position'][:, 2], 'b.')
   # plt.show()



   world = simworld.simworld(data)

   red_dot = {
      'means': torch.Tensor([0, 0, 0]),
      'quats': torch.Tensor([1, 0, 0, 0]),
      'scales': torch.Tensor([0.01, 0.01, 0.01]),
      'opacities': torch.Tensor([1]),
      'colors': torch.Tensor([1, 0, 0])
   }

   # eye_loc = torch.Tensor([0, 0, 0]).float()
   # look_pt = red_dot['means'].float()
   # look = look_pt - eye_loc
   # c2w[:-1, :] = camera_utils.viewmatrix(look.flatten(), UP.flatten(), eye_loc.flatten())
   # w2c = utils.quick_viewmat_inv(c2w)

   ind = world.add_splats(**red_dot)

   camera_intrinsic = camera_intrinsic.float().reshape(-1, 3, 3)

   while True:
      # Camera loop
      # w2c = np.linalg.inv(c2w)
      # w2c[:-1, -1] = [x, y, z]

      # look_direction = c2w[2, :-1]/np.linalg.norm(c2w[2, :-1])

      # print(f'c2w:\n {c2w}')

      viewmats = w2c[None, ...].float().to(device)
      # viewmats = torch.linalg.inv(c2w)[None, :, :].float().to(device)
      # viewmats = lookat[None, :, :].float().to(device)

      # print(f'w2c:\n {viewmats}')
      # print(f'c2w:\n {torch.linalg.inv(viewmats)}')

      # C = len(viewmats)
      # N = len(means)

      # render_colors, render_alphas, meta = rasterization(
      #    means,
      #    quats,
      #    scales,
      #    opacities,
      #    colors,
      #    viewmats,
      #    Ks,
      #    width=width,
      #    height=height,
      #    near_plane=0.2,
      #    far_plane=5,
      #    render_mode='RGB+D'
      # )

      img = world.render(viewmats, camera_intrinsic, width, height)
      img = np.take(img, [2, 1, 0], axis=3).squeeze()

      # Show Images
      # canvas = (
      #    torch.cat(
      #       [
      #             render_rgbs.reshape(C * height, width, 3),
      #             render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
      #             render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
      #       ],
      #       dim=1,
      #    )
      #    .cpu()
      #    .numpy()
      # )
      cv.imshow('Image', np.array(img, dtype=np.uint8))
      key = cv.waitKey(0)

      if key == ord('p'):
         break
      elif key == ord('w'):
         w2c[2, -1]-=t
      elif key == ord('s'):
         w2c[2, -1]+=t
      elif key == ord('d'):
         w2c[0, -1]-=t
      elif key == ord('a'):
         w2c[0, -1]+=t
      elif key == ord('r'):
         w2c[1, -1]+=t
      elif key == ord('f'):
         w2c[1, -1]-=t
      elif key == ord('q'):
         w2c =   get_Rz(t, shape=4, as_type=torch.FloatTensor) @ w2c
      elif key == ord('e'):
         w2c =   get_Rz(-t, shape=4, as_type=torch.FloatTensor) @ w2c
      elif key == ord('t'):
         w2c =   get_Ry(t, shape=4, as_type=torch.FloatTensor) @ w2c
      elif key == ord('g'):
         w2c =   get_Ry(-t, shape=4, as_type=torch.FloatTensor) @ w2c
      elif key == ord('z'):
         w2c =   get_Rx(t, shape=4, as_type=torch.FloatTensor) @ w2c
      elif key == ord('x'):
         w2c =   get_Rx(-t, shape=4, as_type=torch.FloatTensor) @ w2c
      elif key == ord('['):
         world.means[ind, 0] += t
      elif key == ord(']'):
         world.means[ind, 0] -= t
      elif key == ord('k'):
         world.means[ind, 1] += t
      elif key == ord('l'):
         world.means[ind, 1] -= t
      elif key == ord('n'):
         world.means[ind, 2] += t
      elif key == ord('m'):
         world.means[ind, 2] -= t
      elif key == ord('j'):
         print(world.means[ind, :])
      else:
         continue

# def process_ply(ply_file_path):
#    plydata = PlyData.read(ply_file_path)
#    vert = plydata["vertex"]
#    sorted_indices = np.argsort(
#       -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
#       / (1 + np.exp(-vert["opacity"]))
#    )

#    N = len(sorted_indices)
#    values = dict()
#    values.update({'position': np.zeros((N, 3)),
#                   'scales': np.zeros((N, 3)),
#                   'rot': np.zeros((N, 4)),
#                   'color': np.zeros((N, 3)),
#                   'opacity': np.zeros(N)})

#    for i, idx in enumerate(sorted_indices):
#       v = plydata["vertex"][idx]
#       position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
#       scales = np.exp(
#          np.array(
#                [v["scale_0"], v["scale_1"], v["scale_2"]],
#                dtype=np.float32,
#          )
#       )
#       rot = np.array(
#          [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
#          dtype=np.float32,
#       )
#       SH_C0 = 0.28209479177387814
#       color = np.array(
#          [
#                0.5 + SH_C0 * v["f_dc_0"],
#                0.5 + SH_C0 * v["f_dc_1"],
#                0.5 + SH_C0 * v["f_dc_2"],
#                # 1 / (1 + np.exp(-v["opacity"])),
#          ]
#       )
#       # color = np.clip(color, 0, 1)
#       # color = np.array([v['red'], v['green'], v['blue']])
#       # opacity = np.array(
#       #    v["opacity"])
      
#       opacity = 1 / (1 + np.exp(-v["opacity"]))

#       # Insert into dictionary
#       values['position'][i, :] = position
#       values['scales'][i, :] = scales
#       values['rot'][i, :] = rot
#       values['color'][i, :] = color
#       values['opacity'][i] = opacity

#    return values

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

def lookat_matrix(lookat_point, up, position):
   # position = -position
   lookat_mat = torch.eye(4)

   lookat = position - lookat_point
   # lookat = lookat_point - position
   lookat, up = normalize(lookat), normalize(up)

   right = normalize(torch.cross(up, lookat))
   # right = normalize(torch.cross(lookat, up))
   cam_up = normalize(torch.cross(lookat, right))
   # cam_up = torch.cross(right, lookat)

   rot = torch.stack([right, cam_up, lookat], 0)

   pos_c2w = position @ rot.T
   # pos_c2w = -position

   lookat_mat[:-1, -1] = -pos_c2w
   lookat_mat[:-1, :-1] = rot

   return lookat_mat
   

def normalize(vec):
   return vec / torch.linalg.norm(vec)
if __name__ == "__main__":
   # splat_filepath = r'exports/splat/splat.ply'
   # camera_transform = np.array([
   #    [0.9963047752847375,
   #    0.06803213582128569,
   #    0.052425406440226086,
   #    0.055874174184152296],
   #    [0.06660707195329088,
   #    -0.22664817654182923,
   #    -0.9716965071646974,
   #    0.16383521518157287],
   #    [-0.05422446597835606,
   #    0.9715977730346327,
   #    -0.2303420819639871,
   #    -2.6714806182674176],
   #    [0.0,
   #    0.0,
   #    0.0,
   #    1.0]
   # ])
   # camera_transform = np.eye(4)
   # camera_transform[:-1] = camera_transform[:-1] * 0.5
   # camera_transform[0, :-1] = [1, 1, 0]
   # camera_transform[-1, :-1] = [1000, 100, 10000]
   # print(camera_transform)
   main()
