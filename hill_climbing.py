import simworld
import binocular_vision
import image_processing_utils as img_utils

import torch
import numpy as np

def hill_climbing(
   x0,
   world: simworld.simworld,
   eyes: binocular_vision.binocular_eyes,
   n_neighbors: int = 10,
   max_iter: int = 1000,
   save_values=False,
   verbose=True,
   generate_neighbors_params: dict = {},
   get_blur_params: dict = {}
):
    blurs = []
    xs = []

    blur_val0 = evaluate_neighbors(x0, world, eyes, **get_blur_params)[0]
    for i in range(max_iter):
        if save_values:
            blurs.append(blur_val0)
            xs.append(x0)


        neighbors = generate_neighbors(
            x0,
            n_neighbors,
            **generate_neighbors_params
        )
        blur_vals = evaluate_neighbors(
            neighbors,
            world,
            eyes,
            **get_blur_params
        )
        best_neighbor = np.argmax(blur_vals)
        
        if verbose:
            print('================================================================')
            print(f'iter {i}')
            print(f'iter blur: {blur_vals[best_neighbor]} \nbest blur: {blur_val0}')
            print('================================================================')

        if blur_vals[best_neighbor] <= blur_val0:
            break
        else:
            x0 = neighbors[best_neighbor]
            blur_val0 = blur_vals[best_neighbor]

    if save_values:
        info = {
            'blur_vals': blurs,
            'x_vals': xs,
            'iters': i,
            }
        return x0, info
    else:
        return x0

def generate_neighbors(
    x,
    n,
    min_x=-1,
    max_x=1,
):
   
    rng = max_x - min_x
    rand_change = np.random.rand(n) * rng + min_x
    return x + rand_change

def evaluate_neighbors(
    x,
    world: simworld.simworld,
    eyes: binocular_vision.binocular_eyes,
    wh: int = 100,
    ww: int = 100,
):
    left_viewmats = eyes.left_eye.generate_yaw_peeks(x, mat_type='w2c')
    if np.isscalar(x):
        viewmats = torch.eye(4).repeat(2, 1, 1)
    else:
        viewmats = torch.eye(4).repeat(x.size*2, 1, 1)
    viewmats[::2, :, :] = left_viewmats
    viewmats[1::2, :, :] = eyes.right_eye.get_w2c()

    if np.isscalar(x):
        imgs = world.render(
            viewmats,
            eyes.get_intrinsics(),
            eyes.width,
            eyes.height
        )
    else:
        imgs = world.render(
            viewmats,
            eyes.get_intrinsics().repeat(x.size, 1, 1),
            eyes.width,
            eyes.height
        )
    
    patches = img_utils.get_center_patch(imgs, wh, ww)
    img_combined = img_utils.combine_images(
        patches[::2, :, :],
        patches[1::2, :, :]
        )
    blurs = img_utils.blur_detection(img_combined)

    return blurs