import os
import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from nerfstudio.cameras import cameras

import simworld
import binocular_vision
import hill_climbing
import utils
import image_processing_utils as img_utils

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
    world = simworld.simworld(data)

    # Test eye location
    eye_loc = torch.Tensor([-0.43, -0.4, -0.19]).float()
    look_pt = torch.Tensor([-0.1, 0, -0.3]).float()

    beyes.set_position(eye_loc)
    beyes.face_lookat(look_pt)
    beyes.eye_lookat(look_pt)

    gen_neigh_params = {
        'min_x': -0.05,
        'max_x': 0.05
    }

    x_result, info = hill_climbing.hill_climbing(
        0,
        world,
        beyes,
        n_neighbors=40,
        threshold=120,
        save_values=True,
        generate_neighbors_params=gen_neigh_params
    )

    beyes.left_eye.yaw(x_result)
    leye_w2c = beyes.get_left_eye_w2c().reshape((1, 4, 4))
    reye_w2c = beyes.get_right_eye_w2c().reshape((1, 4, 4))
    # print(beyes.get_eyes_c2w())

    viewmats = torch.cat((leye_w2c, reye_w2c), 0).float().to(device)

    img = world.render(
        viewmats,
        beyes.get_intrinsics().float().to(device),
        beyes.width,
        beyes.height
    )

    img = np.take(img, [2, 1, 0], axis=3)
    left_img, right_img = img[0, ...], img[1, ...]
    combined_img = img_utils.combine_images(left_img, right_img)

    left_img = img_utils.draw_center_patch(left_img, wh, wl)
    right_img = img_utils.draw_center_patch(right_img, wh, wl)

    plt.figure()
    ax1 = plt.plot(np.arange(info['iters']+1), info['blur_vals'], 'b.')
    plt.title('Blur Values')
    plt.xlabel('Iterations')
    plt.ylabel('Blur')
    plt.figure()
    ax2 = plt.plot(np.arange(info['iters']+1), info['x_vals'], 'r.')
    plt.title('Adjustment Values (rad)')
    plt.xlabel('Iterations')
    plt.ylabel('Adjustment Values (rad)')
    plt.show()

    cv.imshow('Left eye', left_img)
    cv.imshow('Right eye', right_img)
    cv.imshow('Combined', combined_img.numpy().astype(np.uint8))

    key = cv.waitKey(0)

if __name__ == "__main__":
   main()