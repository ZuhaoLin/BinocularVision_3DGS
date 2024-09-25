import numpy as np
import torch
import torch.nn.functional as F

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization

from plyfile import PlyData
import sklearn.preprocessing
import cv2 as cv

import os

def main():
   print(os.getcwd())
   splat_filepath = r'./exports/splat/splat.ply'                                                      # Specify ply file of splat
   c2w = np.eye(4)                                                                                    # Initial camera transform
   w2c = np.eye(4)
   camera_intrinsic = np.array([[480.613, 0, 324.1875], [0, 481.5445, 210.0625], [0, 0, 1]])          # Arbitrary camera intrinsics matrix
   height, width = 1080, 1920                                                                         # Image height and width

   t = 0.05                                                                                           # Rate to adjust camera

   torch.manual_seed(42)
   device = "cuda"
   data = process_ply(splat_filepath)                                                                 # Read ply file

   # load .ply file data for processing
   means = torch.from_numpy(data['position']).float().to(device)
   quats = torch.from_numpy(data['rot']).float().to(device)
   scales = torch.from_numpy(data['scales']).float().to(device)
   opacities = torch.from_numpy(data['opacity']).float().to(device)
   colors = torch.from_numpy(data['color']).float().to(device)
   
   # Camera intrinsics
   Ks = torch.from_numpy(camera_intrinsic)[None, :, :].float().to(device)

   # Initialize the camera properties
   x, y, z = 0, 0, 0

   # Example data
   # (
   #    means,
   #    quats,
   #    scales, 
   #    opacities,
   #    colors,
   #    viewmats1,
   #    Ks1,
   #    width1,
   #    height1,
   # ) = load_test_data(device=device)

   while True:
      # Camera loop
      c2w[:-1, -1] = [x, y, z]

      # look_direction = c2w[2, :-1]/np.linalg.norm(c2w[2, :-1])

      # print(f'c2w:\n {c2w}')

      # w2c[:-1, :-1] = np.linalg.inv(c2w[:-1, :-1])
      w2c = np.linalg.inv(c2w)
      look_direction = w2c[2, :-1]/np.linalg.norm(w2c[2, :-1])
      # w2c[:-1, -1] = [x, y, z]
      viewmats = torch.from_numpy(w2c)[None, :, :].float().to(device)

      print(f'w2c:\n {w2c}')

      C = len(viewmats)
      N = len(means)

      render_colors, render_alphas, meta = rasterization(
         means,
         quats,
         scales,
         opacities,
         colors,
         viewmats,
         Ks,
         width=width,
         height=height,
         render_mode='RGB+D'
      )



      C = render_colors.shape[0]
      assert render_colors.shape == (C, height, width, 4)
      assert render_alphas.shape == (C, height, width, 1)

      render_rgbs = render_colors[..., 0:3]
      render_depths = render_colors[..., 3:4]
      render_depths = render_depths / render_depths.max()

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

      rgbs = render_rgbs.reshape(height, width, 3).cpu().numpy()
      img = (rgbs*255).astype(np.uint8)

      cv.imshow('Image', img)
      key = cv.waitKey(0)

      if key == ord('p'):
         break
      elif key == ord('w'):
         x+=t*look_direction[0]
         y+=t*look_direction[1]
         z+=t*look_direction[2]
      elif key == ord('s'):
         x-=t*look_direction[0]
         y-=t*look_direction[1]
         z-=t*look_direction[2]
      elif key == ord('d'):
         y+=t
      elif key == ord('a'):
         y-=t
      elif key == ord('r'):
         z+=t
      elif key == ord('f'):
         z-=t
      elif key == ord('q'):
         c2w[:-1, :-1] =  get_yaw(t, as_type=np.array) @ c2w[:-1, :-1]
      elif key == ord('e'):
         c2w[:-1, :-1] =  get_yaw(-t, as_type=np.array) @ c2w[:-1, :-1]
      elif key == ord('t'):
         c2w[:-1, :-1] =  get_pitch(t, as_type=np.array) @ c2w[:-1, :-1]
      elif key == ord('g'):
         c2w[:-1, :-1] =  get_pitch(-t, as_type=np.array) @ c2w[:-1, :-1]
      elif key == ord('z'):
         c2w[:-1, :-1] =  get_roll(t, as_type=np.array) @ c2w[:-1, :-1]
      elif key == ord('x'):
         c2w[:-1, :-1] =  get_roll(-t, as_type=np.array) @ c2w[:-1, :-1]
      else:
         continue

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
      color = np.clip(color, 0, 1)
      # color = np.array([v['red'], v['green'], v['blue']])
      opacity = np.array(
         v["opacity"])
      
      # Insert into dictionary
      values['position'][i, :] = position
      values['scales'][i, :] = scales
      values['rot'][i, :] = rot
      values['color'][i, :] = color
      values['opacity'][i] = opacity

   return values

def get_yaw(theta, as_type=list):

   R = [
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta), np.cos(theta), 0],
      [0, 0, 1]
      ]

   if as_type != list:
      R = as_type(R)
   return R

def get_pitch(theta, as_type=list):
   R = [
      [np.cos(theta), 0, np.sin(theta)],
      [0, 1, 0],
      [-np.sin(theta), 0, np.cos(theta)]
      ]
   
   if as_type != list:
      R = as_type(R)
   return R

def get_roll(theta, as_type=list):
   R = [
      [1, 0, 0],
      [0, np.cos(theta), -np.sin(theta)],
      [0, np.sin(theta), np.cos(theta)]
      ]

   if as_type != list:
      R = as_type(R)
   return R

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