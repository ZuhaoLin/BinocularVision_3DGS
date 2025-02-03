import numpy as np
import torch
import cv2 as cv
import nerfstudio.cameras.cameras as cameras
import torchvision.transforms.functional
from ultralytics import YOLO
import matplotlib.pyplot as plt
from simple_pid import PID
import copy

import utils
import simworld
import binocular_vision
import image_processing_utils as img_utils

import os

def main():
    print(os.getcwd())
    # splat_filepath = r'./exports/IMG_5435/splat.ply'
    splat_filepath = r'./exports/IMG_5463/splat.ply'                                                   # Specify ply file of splat
    height, width = 1120, 1024                                                                         # Image size
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
    eye_loc = torch.Tensor([-0.43, -0.4, -0.19]).float()
    # eye_loc = torch.Tensor([-0.4271, 0.2371, -0.1328]).float()
    # eye_loc = torch.Tensor([-0.0302, 0.50, -0.1552]).float()
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

def keybindings(key, eyes: binocular_vision.binocular_eyes, t):
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
        eye_pos = eyes.get_position()
        intersect_pos, _ = eyes.get_look_point()
        dist = torch.sqrt(torch.sum((eye_pos - intersect_pos) ** 2))

        print(f'Binocular Eye Position: {eye_pos}')
        print(f'Intersect Position: {intersect_pos}')
        print(f'Distance: {dist}')

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
        # print(f'Combined blur: {img_utils.blur_detection(combined_img)}')

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
    # YOLO detection models
    model_left = YOLO('yolo11n.pt')
    model_right = YOLO('yolo11n.pt')

    # Camera/image properties
    Ks = eyes.get_intrinsics().float().to(device)
    width = eyes.width
    height = eyes.height
    center = (int(width/2), int(height/2))

    acquired_id = None

    # PID Controllers
    pidxl = PID(0.0008, 0.00001, 0.00000, center[0])
    pidyl = PID(0.0008, 0.00001, 0.00000, center[1])
    pidxr = copy.deepcopy(pidxl)
    pidyr = copy.deepcopy(pidyl)

    while True:
        viewmats = eyes.get_eyes_w2c().float().to(device)
        img = world.render(viewmats, Ks, width, height)
        left_img, right_img = img[0, ...], img[1, ...]

        # Restructure tensors for model compatibility in BCHW format
        source = img.permute((0, 3, 1, 2))                                                  # Was in BHWC format
        source = torchvision.transforms.functional.resize(source, [1120, 1024])             # Try to preserve as much data as possible
        source = np.array(source.permute((0, 2, 3, 1))).astype(np.uint8)
        source[:, :, :, [0, 2]] = source[:, :, :, [2, 0]]

        left_result = model_left.track(
            source=source[0, ...],
            conf=0.3,
            persist=True,
            verbose=False
        )[0]

        right_result = model_right.track(
            source=source[1, ...],
            conf=0.3,
            persist=True,
            verbose=False
        )[0]

        if acquired_id is None:
            # If the object has not been determined yet, this will obtain the matching ID
            search_cls = 41

            # Search for specified class
            left_labels = left_result.boxes.cls
            right_labels = right_result.boxes.cls
            left_searched = left_labels == search_cls
            right_searched = right_labels == search_cls
            left_boxes = left_result.boxes[left_searched]
            right_boxes = right_result.boxes[right_searched]
            
            # Crop all the objects of the specified class detected
            left_img = img_utils.convert_HWC2CHW(torch.tensor(source[0, ...]))
            right_img = img_utils.convert_HWC2CHW(torch.tensor(source[1, ...]))
            left_cropped = img_utils.image_crop(left_img, left_boxes.xyxy)
            right_cropped = img_utils.image_crop(right_img, right_boxes.xyxy)
            # Match objects in left to objects in right
            matched_imgs, inds = img_utils.match_images_SIFT(left_cropped, right_cropped)
            left_ind, right_ind = inds[0]
            acquired_id = (left_boxes.id[left_ind], right_boxes.id[right_ind])

            if len(matched_imgs) != 0:
                # Show the first matched class
                left_img, right_img = matched_imgs[0]
                left_img = np.array(img_utils.convert_CHW2HWC(left_img), dtype=np.uint8)
                right_img = np.array(img_utils.convert_CHW2HWC(right_img), dtype=np.uint8)
                cv.imshow('Left Object', left_img)
                cv.imshow('Right Object', right_img)
                print(f'Left ID: {acquired_id[0]}\nRight ID: {acquired_id[1]}')

            # Show whole image with all boxes
            left_det = left_result.plot()
            right_det = right_result.plot()

        else:
            # Look for the acquired id
            left_ind = np.where(left_result.boxes.id == acquired_id[0])[0]
            right_ind = np.where(right_result.boxes.id == acquired_id[1])[0]

            if left_ind.size == 0 or right_ind.size == 0:
                print('Cannot find object')
                left_det = left_result.orig_img
                right_det = right_result.orig_img
            else:
                left_obj_box = left_result.boxes[left_ind]
                right_obj_box = right_result.boxes[right_ind]

                left_det = left_result.orig_img
                right_det = right_result.orig_img

                # # Draw blue rectangle around selected object
                # left_det = cv.rectangle(
                #     left_result.orig_img,
                #     tuple(np.array(np.rint(left_obj_box.xyxy[0, :2]), dtype=int)),
                #     tuple(np.array(np.rint(left_obj_box.xyxy[0, 2:]), dtype=int)),
                #     (255, 0, 0)
                # )

                # right_det = cv.rectangle(
                #     right_result.orig_img,
                #     tuple(np.array(np.rint(right_obj_box.xyxy[0, :2]), dtype=int)),
                #     tuple(np.array(np.rint(right_obj_box.xyxy[0, 2:]), dtype=int)),
                #     (255, 0, 0)
                # )

                pos_xl, pos_yl = left_obj_box.xywh[0, :2]
                pos_xr, pos_yr = right_obj_box.xywh[0, :2]
                # print(f'pos_xl: {pos_xl}\npos_yl: {pos_yl}')
                signal_xl, signal_yl = pidxl(pos_xl), -pidyl(pos_yl)
                signal_xr, signal_yr = pidxr(pos_xr), -pidyr(pos_yr)

                # Draw xl and yl for reference
                # left_det = cv.circle(left_det, (int(pos_xl), int(pos_yl)), 5, (0, 255, 0), 2)
                # right_det = cv.circle(right_det, (int(pos_xr), int(pos_yr)), 5, (0, 255, 0), 2)

            # print(f'xl signal: {signal_xl}\nyl signal: {signal_yl}')
            # print(f'xr signal: {signal_xr}\nyr signal: {signal_yr}')
            eyes.left_eye.yaw(float(signal_xl))
            eyes.left_eye.pitch(float(signal_yl))
            eyes.right_eye.yaw(float(signal_xr))
            eyes.right_eye.pitch(float(signal_yr))


        left_det = cv.circle(left_det, center, 5, (0, 0, 255), 2)
        right_det = cv.circle(right_det, center, 5, (0, 0, 255), 2)
        combined = img_utils.combine_images(left_det, right_det)
        cv.imshow('Left Tracking', left_det)
        cv.imshow('Right Tracking', right_det)
        cv.imshow('Combined', combined)

        key = cv.waitKey(1)
        if not keybindings(key, eyes, 0.01):
            cv.imwrite('leye_cup_3.jpg', left_det)
            cv.imwrite('reye_cup_3.jpg', right_det)
            cv.imwrite('combined_cup_3.jpg', combined)
            break

    return

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

    

    # return
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