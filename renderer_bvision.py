import numpy as np
import torch
import cv2 as cv
import nerfstudio.cameras.cameras as cameras
import torchvision.transforms.functional
from ultralytics import YOLO
import matplotlib.pyplot as plt

import utils
import simworld
import binocular_vision
import image_processing_utils as img_utils

import os

def main():
    print(os.getcwd())
    # splat_filepath = r'./exports/IMG_5435/splat.ply'
    splat_filepath = r'./exports/IMG_5463/splat.ply'                                                   # Specify ply file of splat
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
    wh, wl = 50, 50

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
    # eye_loc = torch.Tensor([-0.43, -0.4, -0.19]).float()
    eye_loc = torch.Tensor([-0.4271, 0.2371, -0.1328]).float()
    # look_pt = torch.Tensor([-0.1473, -0.0927, -0.2796]).float()
    look_pt = torch.Tensor([-0.1800, 0.1700, -0.3300]).float()

    beyes.set_position(eye_loc)
    beyes.face_lookat(look_pt)
    beyes.eye_lookat(look_pt)

    red_dot = {
        'means': torch.Tensor([-0.1473, -0.0927, -0.2796]),
        'quats': torch.Tensor([1, 0, 0, 0]),
        'scales': torch.Tensor([0.01, 0.01, 0.01]),
        'opacities': torch.Tensor([0]),
        'colors': torch.Tensor([1, 0, 0])
    }

    ind = world.add_splats(**red_dot)

    object_focusing(world, beyes)
    # diverging(world, beyes, wh, wl, t)

    return

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
        print(f'Combined blur: {img_utils.blur_detection(combined_img)}')
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
        
        # plt.figure()
        # plt.imshow(left_img)
        # plt.title('Left eye')
        # plt.figure()
        # plt.imshow(right_img)
        # plt.title('Right eye')
        # plt.figure()
        # plt.imshow(combined_img)
        # plt.title('Combined')
        # plt.show()

        cv.imshow('Left eye', left_img)
        cv.imshow('Right eye', right_img)
        cv.imshow('Combined', combined_img)

        key = cv.waitKey(0)
        if not keybindings(key, beyes, t):
            break

def keybindings(key, eyes, t):
    if key == ord('p'):
        return False
    elif key == ord('w'):
        eyes.move_forward(t)
    elif key == ord('s'):
        eyes.move_backwards(t)
    elif key == ord('a'):
        eyes.move_left(t)
    elif key == ord('d'):
        eyes.move_right(t)
    elif key == ord('r'):
        eyes.move_up(t)
    elif key == ord('f'):
        eyes.move_down(t)
    elif key == ord('q'):
        eyes.yaw(t)
    elif key == ord('e'):
        eyes.yaw(-t)
    elif key == ord('t'):
        eyes.pitch(t)
    elif key == ord('g'):
        eyes.pitch(-t)
    elif key == ord('['):
        eyes.right_eye.yaw(t)
    elif key == ord(']'):
        eyes.right_eye.yaw(-t)
    elif key == ord('j'):
        print(eyes.get_position())

    return True

def diverging(
    world: simworld.simworld,
    eyes: binocular_vision.binocular_eyes,
    wh: int,
    wl: int,
    t,
    device='cuda'
    ):
    # Camera intrinsics
    Ks1 = eyes.left_eye.get_intrinsics_matrices().reshape((1, 3, 3)).float().to(device)
    Ks2 = eyes.right_eye.get_intrinsics_matrices().reshape((1, 3, 3)).float().to(device)
    Ks = torch.cat((Ks1, Ks2), 0).float().to(device)

    width = eyes.width
    height = eyes.height

    while True:
        leye_w2c = eyes.get_left_eye_w2c().reshape((1, 4, 4))
        reye_w2c = eyes.get_right_eye_w2c().reshape((1, 4, 4))
        viewmats = torch.cat((leye_w2c, reye_w2c), 0).float().to(device)

        img = world.render(viewmats, Ks, width, height)
        img = np.take(img, [2, 1, 0], axis=3)
        left_img, right_img = img[0, ...], img[1, ...]

        combined_img = img_utils.combine_images(left_img, right_img)

        # Blur detection
        print(f'Combined blur: {img_utils.blur_detection(combined_img)}')

        left_img = img_utils.draw_center_patch(left_img, wh, wl)
        right_img = img_utils.draw_center_patch(right_img, wh, wl)
        combined_img = img_utils.draw_center_patch(combined_img, wh, wl)

        cv.imshow('Left eye', left_img)
        cv.imshow('Right eye', right_img)
        cv.imshow('Combined', combined_img)

        key = cv.waitKey(0)
        if not keybindings(key, eyes, t):
            break

def object_focusing(
    world: simworld.simworld,
    eyes: binocular_vision.binocular_eyes,
    device='cuda'
):
    model = YOLO('yolo11n.yaml')
    model = YOLO('yolo11n.pt')

    viewmats = eyes.get_eyes_w2c().float().to(device)
    Ks = eyes.get_intrinsics().float().to(device)
    width = eyes.width
    height = eyes.height

    img = world.render(viewmats, Ks, width, height)
    # img = np.take(img, [2, 1, 0], axis=3)
    left_img, right_img = img[0, ...], img[1, ...]

    # Restructure tensors for model compatibility in BCHW format
    source = img.permute((0, 3, 1, 2))                                                  # Was in BHWC format
    source = torchvision.transforms.functional.resize(source, [1120, 1024])             # Try to preserve as much data as possible
    source = source / 255                                                               # Normalize to [0, 1]

    results = model.predict(source=source)

    left_result, right_result = results[0], results[1]
    left_labels = left_result.boxes.cls
    right_labels = right_result.boxes.cls

    # Search for specified class
    search_cls = 41
    left_searched = left_labels == search_cls
    right_searched = right_labels == search_cls
    left_boxes = left_result.boxes[left_searched]
    right_boxes = right_result.boxes[right_searched]

    # Crop all the objects of the specified class detected
    left_cropped = img_utils.image_crop(source[0]*255, left_boxes.xyxy)
    right_cropped = img_utils.image_crop(source[1]*255, right_boxes.xyxy)
    # Match objects in left to objects in right
    matched_imgs, inds = img_utils.match_images_SIFT(left_cropped, right_cropped)
    print(inds)

    # Show matched objects
    for [left_img, right_img] in matched_imgs:
        left_img = img_utils.convert_CHW2HWC(left_img)/255
        right_img = img_utils.convert_CHW2HWC(right_img)/255
        plt.figure()
        plt.imshow(left_img)
        plt.title('Left Image Object')
        plt.figure()
        plt.imshow(right_img)
        plt.title('Right Image Object')
        plt.show()

    

    return
    # Show the detected objects in both cameras
    print('Done Predicting')
    img_left = results[0].plot()
    img_right = results[1].plot()
    plt.figure()
    plt.imshow(img_left)
    plt.title('left eye detection')
    plt.figure()
    plt.imshow(img_right)
    plt.title('right eye detection')
    plt.show()



if __name__ == "__main__":
    main()