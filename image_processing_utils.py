import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale
import cv2 as cv
from typing import Union
import matplotlib.pyplot as plt

def match_images_SIFT(imgs1, imgs2, threshold=1000, grey=False):
    '''
    Images must be in CHW order
    '''
    sift = cv.SIFT.create()
    matcher = cv.BFMatcher()

    ## Keypoint detection on both sets of images
    kp1, des1 = [], []
    for img1 in imgs1:
        # plt.imshow(convert_CHW2HWC(img1/255))
        # plt.show()
        if not grey:
            img1 = rgb_to_grayscale(img1)
        img1 = convert_CHW2HWC(img1).squeeze().numpy().astype(np.uint8)

        kp, des = sift.detectAndCompute(img1, None)
        if des is None:
            continue
        else:
            kp1.append(kp)
            des1.append(des)

    kp2, des2 = [], []
    for img2 in imgs2:
        # plt.imshow(convert_CHW2HWC(img2/255))
        # plt.show()
        if not grey:
            img2 = rgb_to_grayscale(img2)
        img2 = convert_CHW2HWC(img2).squeeze().numpy().astype(np.uint8)

        kp, des = sift.detectAndCompute(img2, None)
        if des is None:
            continue
        else:
            kp2.append(kp)
            des2.append(des)

    if len(kp1) == 0 or len(kp2) == 0:
        return [], []

    ## Matching between both sets of images
    distances = torch.full([len(kp1), len(kp2)], float('Inf'), dtype=float)
    for im1, (keypoints1, descriptors1) in enumerate(zip(kp1, des1)):
        if descriptors1.shape[0] < 2:
            continue
        for im2, (keypoints2, descriptors2) in enumerate(zip(kp2, des2)):
            if descriptors2.shape[0] < 2:
                continue
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            good = []
            total_dist = 0
            total = 0
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
                    total_dist+=m.distance
                    total+=1
            if total == 0:
                continue
            else:
                distances[im1, im2] = total_dist / total
    
    ## Finding which matches passed the threshold
    img_matches = torch.argmin(distances, dim=1)
    passed = distances[torch.arange(0, distances.shape[0]), img_matches] < threshold
    passed1 = torch.where(passed)[0]
    passed2 = img_matches[passed]

    matched_imgs = list(zip([imgs1[x] for x in passed1], [imgs2[x] for x in passed2]))
    passed_inds = list(zip(passed1, passed2))

    return matched_imgs, passed_inds



def image_crop(image, crop_bounds):
    '''
    Crops the image to the given crop_bounds in (x1, y1, x2, y2) format.
    Image must be in CHW format
    '''

    cropped = []
    for i in range(crop_bounds.shape[0]):
        x1, y1, x2, y2 = torch.round(crop_bounds[i, :]).type(torch.int)
        cropped.append(image[:, y1:y2, x1:x2])
    return cropped

def convert_HWC2CHW(images: torch.Tensor):
    if images.ndim == 4:
        return images.permute((0, 3, 1, 2))                                                  # Was in BHWC format
    elif images.ndim == 3:
        return images.permute((2, 0, 1))
    
def convert_CHW2HWC(images: torch.Tensor):
    if images.ndim == 4:
        return images.permute((0, 2, 3, 1))                                                  # Was in BCHW format
    elif images.ndim == 3:
        return images.permute((1, 2, 0))

def combine_images(image1, image2):
    # if (image1.shape != image2.shape) and (image1.ndim != image2.ndim), \
    # f'image1 and image2 must be the same shape.' + \
    #     'image1 is of shape {image1.shape} and' + \
    #     'image2 is of shape {image2.shape}'

    if isinstance(image1, torch.Tensor):
        combined_img = (image1.type(torch.FloatTensor) + image2.type(torch.FloatTensor)) / 2
        combined_img = combined_img.type(torch.IntTensor)
    elif isinstance(image1, np.ndarray):
        combined_img = (image1.astype(np.float64) + image2.astype(np.float64)) / 2
        combined_img = np.rint(combined_img).astype(np.uint8)

    return combined_img

def blur_detection(
        image: Union[np.ndarray, torch.Tensor],
        image_type='rgb'
        ):
    if isinstance(image, torch.Tensor):
        image = image.numpy().astype(np.uint8)

    if image_type == 'rgb' and image.ndim == 3:
        return cv.Laplacian(image, cv.CV_64F).var()
    elif image_type == 'rgb' and image.ndim == 4:
        blurs = np.zeros(image.shape[0])
        for i in range(image.shape[0]):
            blurs[i] = cv.Laplacian(
                image[i, ...],
                cv.CV_64F
            ).var()
        return blurs
    else:
        raise ValueError(f'Does not support image type {image_type} with dimension {image.ndim}')

def image_distance_error(image1, image2):
    '''
    Bad function
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
    if isinstance(images, torch.Tensor):
        images = images.numpy().astype(np.uint8)
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