import numpy as np
import torch
import torch.nn.functional as F

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization
import binocular_vision

from plyfile import PlyData
import cv2 as cv

import nerfstudio.cameras.cameras as cameras
import nerfstudio.cameras.camera_utils as camera_utils

import utils
import image_processing_utils as img_utils
import simworld

import matplotlib.pyplot as plt

import os

def main():
   print(os.getcwd())
   splat_filepath = r'./exports/IMG_5435/splat.ply'                                                   # Specify ply file of splat
   height, width = 1080, 1200                                                                         # Image size
   fx, fy = 500.0, 500.0                                                                              # Cam intrinsics
   cx, cy = width/2, height/2                                                                         # Camera intrinsics


   eye_cams = cameras.Cameras(                                                                        # Eye cam
      torch.eye(4),
      fx,
      fy,
      cx,
      cy,
      width,
      height
   )

   beyes = binocular_vision.binocular_eyes(0.25, torch.eye(4), eye_cams)                               # Create binocular eyes

   t = 0.05                                                                                           # Rate to adjust camera

   torch.manual_seed(42)
   device = "cuda"
   data = utils.process_ply(splat_filepath)                                                           # Read ply file

   # Detection window size
   wh, wl = 100, 100

   # Point visualization
   # ax = plt.figure().add_subplot(projection='3d')

   # cvec = ['r', 'g', 'b']
   # for i in range(3):
   #    vec = [0, 0, 0]
   #    vec[i] = 1
   #    dir = tuple(vec)
   #    ax.quiver(
   #       0, 0, 0,
   #       *dir,
   #       length=1,
   #       color=cvec[i])
   
   # centers = data['position']
   # ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 'om')

   # load .ply file data for processing
   # means = torch.from_numpy(data['position']).float().to(device)
   # quats = torch.from_numpy(data['rot']).float().to(device)
   # scales = torch.from_numpy(data['scales']).float().to(device)
   # opacities = torch.from_numpy(data['opacity']).float().to(device)
   # colors = torch.from_numpy(data['color']).float().to(device)

   world = simworld.simworld(data)
   
   # Camera intrinsics
   Ks1 = beyes.left_eye.get_intrinsics_matrices().reshape((1, 3, 3)).float().to(device)
   Ks2 = beyes.right_eye.get_intrinsics_matrices().reshape((1, 3, 3)).float().to(device)
   Ks = torch.cat((Ks1, Ks2), 0).float().to(device)

   # Test eye location
   eye_loc = torch.Tensor([-0.43, -0.4, -0.19]).float()
   # look_pt = torch.from_numpy(centers[0, :]).float()
   look_pt = torch.Tensor([-0.1, 0, -0.3]).float()

   beyes.set_position(eye_loc)
   beyes.face_lookat(look_pt)
   beyes.eye_lookat(look_pt)


   while True:
      # beyes.set_position(torch.tensor([eye_loc[0], eye_loc[2], eye_loc[1]]))
      # print(beyes.get_position())

      leye_w2c = beyes.get_left_eye_w2c().reshape((1, 4, 4))
      reye_w2c = beyes.get_right_eye_w2c().reshape((1, 4, 4))
      # print(beyes.get_eyes_c2w())

      viewmats = torch.cat((leye_w2c, reye_w2c), 0).float().to(device)
      
      # print(viewmats)

      # Plotting stuff for reference
      # Z = torch.Tensor([0, 0, 1, 0])[:, None]
      # left_look = beyes.get_left_eye_w2c() @ Z
      # right_look = beyes.get_right_eye_w2c() @ Z

      # left_pos = beyes.get_left_eye_position()
      # right_pos = beyes.get_right_eye_position()

      # ax.plot(*tuple(look_pt), 'xr')

      # ax.quiver(
      #    0, 0, 0,
      #    *tuple(Z[:-1, :]),
      #    length=10,
      #    color='m'
      # )

      # ax.quiver(
      #    *tuple(left_pos),
      #    *tuple(left_look[:-1].flatten()),
      #    length=10,
      #    color='g'
      # )

      # ax.quiver(
      #    *tuple(right_pos),
      #    *tuple(right_look[:-1].flatten()),
      #    length=10,
      #    color='b'
      # )

      # plt.show()

      # render_colors, render_alphas, meta = rasterization(
      #       means,
      #       quats,
      #       scales,
      #       opacities,
      #       colors,
      #       viewmats,
      #       Ks,
      #       width=width,
      #       height=height,
      #       near_plane=0,
      #       render_mode='RGB+D'
      #    )
      
      # C = render_colors.shape[0]
      # assert render_colors.shape == (C, height, width, 4)
      # assert render_alphas.shape == (C, height, width, 1)

      # render_rgbs = render_colors[..., 0:3]
      # render_depths = render_colors[..., 3:4]
      # render_depths = render_depths / render_depths.max()

      # rgbs = torch.clamp(render_rgbs, 0.0, 1.0).reshape(C, height, width, 3).cpu().numpy()
      # img = (rgbs*255).astype(np.uint8)
      img = world.render(viewmats, Ks, width, height)
      img = np.take(img, [2, 1, 0], axis=3)

      left_img, right_img = img[0, ...], img[1, ...]

      combined_img = img_utils.combine_images(left_img, right_img)

      # Blur detection
      blur = []
      for img in [left_img, right_img, combined_img]:
         blur.append(img_utils.blur_detection(img))
      # print(blur)
      # print(img_utils.image_distance_error(
      #    img_utils.get_center_patch(left_img, wh, wl),
      #    img_utils.get_center_patch(right_img, wh, wl)
      #    ))
      
      left_img = img_utils.draw_center_patch(left_img, wh, wl)
      right_img = img_utils.draw_center_patch(right_img, wh, wl)
      combined_img = img_utils.draw_center_patch(combined_img, wh, wl)

      cv.imshow('Left eye', left_img)
      cv.imshow('Right eye', right_img)
      cv.imshow('Combined', combined_img)

      key = cv.waitKey(0)
      if key == ord('p'):
         break
      elif key == ord('w'):
         beyes.move_forward(t)
      elif key == ord('s'):
         beyes.move_backwards(t)
      elif key == ord('a'):
         beyes.move_left(t)
      elif key == ord('d'):
         beyes.move_right(t)
      elif key == ord('r'):
         beyes.move_up(t)
      elif key == ord('f'):
         beyes.move_down(t)
      elif key == ord('q'):
         beyes.yaw(t)
      elif key == ord('e'):
         beyes.yaw(-t)
      elif key == ord('t'):
         beyes.pitch(t)
      elif key == ord('g'):
         beyes.pitch(-t)
      elif key == ord('['):
         beyes.left_eye.yaw(t)
      elif key == ord(']'):
         beyes.left_eye.yaw(-t)
      elif key == ord('j'):
         print(beyes.get_position())
      # print(eye_loc)

if __name__ == "__main__":
   main()