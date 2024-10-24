import numpy as np
import torch
import cv2 as cv

def combine_images(image1, image2):
    assert image1.shape == image2.shape, \
    f'image1 and image2 must be the same shape.' + \
        'image1 is of shape {image1.shape} and' + \
        'image2 is of shape {image2.shape}'

    combined_img = (image1.astype(np.float32) + image2.astype(np.float32)) / 2

    return combined_img.astype(np.uint8)

def blur_detection(image):
    return cv.Laplacian(image, cv.CV_64F).var()

def image_distance_error(image1, image2):
    '''

    '''
    assert image1.shape == image2.shape, \
    f'image1 and image2 must be the same shape.' + \
        'image1 is of shape {image1.shape} and' + \
        'image2 is of shape {image2.shape}'
    
    diff_img = image1 - image2
    error = np.sum(diff_img * diff_img)/np.product(image1.shape)

    return error

def get_center_patch(images, win_height, win_width, image_type='rgb'):

    half_height, half_width = int(win_height/2), int(win_width/2)

    if images.ndim == 4 and image_type == 'rgb':

        # Multiple RGB Images
        center = (int(images.shape[1]/2), int(images.shape[2]/2))
        return images[
            :,
            center[0]-half_height:center[0]+half_height,
            center[1]-half_width:center[1]+half_width,
            :
            ]
    elif images.ndim == 3 and image_type == 'rgb':
        # One RGB Image
        center = (int(images.shape[0]/2), int(images.shape[1]/2))
        return images[
            center[0]-half_height:center[0]+half_height,
            center[1]-half_width:center[1]+half_width,
            :
        ]
    
def draw_center_patch(images, win_height, win_width, color=(255, 0, 0), thickness=1, image_type='rgb'):
    half_len = np.array([int(win_height/2), int(win_width/2)])

    if images.ndim == 4 and image_type == 'rgb':
        # Multiple RGB Images
        center = np.array([int(images.shape[1]/2), int(images.shape[2]/2)])
        up_left, low_right = center - half_len, center + half_len
        for i in range(images.shape[0]):
            images[i, ...] = cv.rectangle(
                images[i, ...],
                up_left[[1, 0]],
                low_right[[1, 0]],
                color,
                thickness=thickness
                )
    elif images.ndim == 3 and image_type == 'rgb':
        # One RGB Image
        center = [int(images.shape[0]/2), int(images.shape[1]/2)]
        up_left, low_right = center - half_len, center + half_len
        images = cv.rectangle(
            images,
            up_left[[1, 0]],
            low_right[[1, 0]],
            color,
            thickness=thickness
            )

    return images