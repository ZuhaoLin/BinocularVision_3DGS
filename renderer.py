import numpy as np
import torch
import torch.nn.functional as F

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization

from plyfile import PlyData
import cv2 as cv


def main():
   splat_filepath = r'../exports/splat/splat.ply'                                                        # Specify ply file of splat
   camera_transform = np.eye(4)                                                                       # Initial camera transform
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

   print(np.max(data['color']))
   print(np.min(data['color']))
   
   # Camera intrinsics
   Ks = torch.from_numpy(camera_intrinsic)[None, :, :].float().to(device)

   # Initialize the camera properties
   x, y, z = 0, 0, 0
   yaw, pitch, roll = 0, 0, 0

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
      Ryaw = get_yaw(yaw)
      Rpitch = get_pitch(pitch)
      Rroll = get_roll(roll)
      
      camera_transform[:-1, :-1] = Ryaw @ Rpitch @ Rroll
      camera_transform[:-1, -1] = [x, y, z]
      viewmats = torch.from_numpy(camera_transform)[None, :, :].float().to(device)  

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

      print(render_rgbs.shape)

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
         x+=t
      elif key == ord('s'):
         x-=t
      elif key == ord('d'):
         y+=t
      elif key == ord('a'):
         y-=t
      elif key == ord('r'):
         z+=t
      elif key == ord('f'):
         z-=t
      elif key == ord('q'):
         yaw+=t
      elif key == ord('e'):
         yaw-=t
      elif key == ord('t'):
         pitch+=t
      elif key == ord('g'):
         pitch-=t
      elif key == ord('z'):
         roll-=t
      elif key == ord('x'):
         roll+=t
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

def get_yaw(theta):
   R = torch.Tensor([
      [np.cos(theta), -np.sin(theta), 0],
      [np.sin(theta), np.cos(theta), 0],
      [0, 0, 1]
   ])

   return R

def get_pitch(theta):
   R = torch.Tensor([
      [np.cos(theta), 0, np.sin(theta)],
      [0, 1, 0],
      [-np.sin(theta), 0, np.cos(theta)]
   ])
   
   return R

def get_roll(theta):
   R = torch.Tensor([
      [1, 0, 0],
      [0, np.cos(theta), -np.sin(theta)],
      [0, np.sin(theta), np.cos(theta)]
   ])

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