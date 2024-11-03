import simworld
import binocular_vision
import image_processing_utils as img_utils

import torch
from torchvision.utils import save_image
import numpy as np
from typing import Union

def hill_climbing(
   x0,
   world: simworld.simworld,
   eyes: binocular_vision.binocular_eyes,
   n_neighbors: int = 10,
   threshold: float = 0.0,
   max_iter: int = 1000,
   save_values=False,
   verbose=True,
   generate_neighbors_params: dict = {},
   evaluate_neighbors_params: dict = {}
):
    blurs = []
    xs = []

    blur_val0 = evaluate_neighbors(x0, world, eyes, **evaluate_neighbors_params)[0]
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
            **evaluate_neighbors_params
        )
        best_neighbor = np.argmax(blur_vals)
        
        if verbose:
            print('================================================================')
            print(f'iter {i}')
            print(f'iter blur: {blur_vals[best_neighbor]} \nbest blur: {blur_val0}')
            print('================================================================')

        if (
            blur_vals[best_neighbor] <= blur_val0 \
            and blur_val0 > threshold
            ):
            break
        elif blur_vals[best_neighbor] >= blur_val0:
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
    imgs = generate_images_from_peeks(x, world, eyes)
    patches = img_utils.get_center_patch(imgs, wh, ww)
    img_combined = img_utils.combine_images(
        patches[:-1, ...],
        patches[-1, ...]
        )
    blurs = img_utils.blur_detection(img_combined)
    blurs_norm = blurs / (wh*ww)
    return blurs_norm

def generate_images_from_peeks(
    x: Union[torch.Tensor, np.ndarray, np.number],
    world: simworld.simworld,
    eyes: binocular_vision.binocular_eyes
) -> torch.Tensor:
    left_viewmats = eyes.left_eye.generate_yaw_peeks(x, mat_type='w2c')
    if np.isscalar(x):
        repeat_val = 1
    else:
        repeat_val = x.size

    viewmats = torch.eye(4).repeat(repeat_val+1, 1, 1)
    viewmats[:-1, ...] = left_viewmats
    viewmats[-1, ...] = eyes.right_eye.get_w2c()

    intrinsics = eyes.left_eye \
        .get_intrinsics_matrices() \
        .reshape(1, 3, 3) \
        .repeat(repeat_val, 1, 1)
    
    intrinsics = torch.cat(
        (intrinsics,
         eyes.right_eye \
            .get_intrinsics_matrices() \
            .reshape(1, 3, 3)
        ),
        0
        )
    
    imgs = world.render(
        viewmats,
        intrinsics,
        eyes.width,
        eyes.height
    )

    return imgs

def save_images_from_peeks(
    x: Union[torch.Tensor, np.ndarray, np.number],
    world: simworld.simworld,
    eyes: binocular_vision.binocular_eyes,
    save_dir: str,
    save_combined: bool = True,
):
    imgs = generate_images_from_peeks(x, world, eyes)
    if save_combined:
        combo = img_utils.combine_images(
            imgs[:-1, ...],
            imgs[-1, ...]
            ).permute(0, 3, 1, 2)
    imgs = imgs.permute(0, 3, 1, 2)
    for i in range(imgs.shape[0]):
        if i != imgs.shape[0]-1:
            rev_num = imgs.shape[0] - 1 - i
            name = f'left_eye_{rev_num}'
            if save_combined:
                combo_name = f'combined_{rev_num}'
                save_image(combo[i, ...]/255, save_dir+'/'+combo_name+'.png')
        else:
            name = f'right_eye_0'
        save_image(imgs[i, ...]/255, save_dir+'/'+name+'.png')