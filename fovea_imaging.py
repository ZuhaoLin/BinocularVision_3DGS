import retina_transform
import glob
import numpy as np
import os
import re
import cv2 as cv
import time
from PIL import Image
from image_processing_utils import combine_images


def main():
    # folderpath = './images/detection_tests/'
    
    # foveate_eyes(folderpath + 'leye*', folderpath + 'reye*', save_imgs=True)

    movement_detection_foveated('./Data/Videos/cow_moving.mp4')

def foveate_folder(folderpath):
    image_paths = glob.glob(folderpath + '*')

    for image_path in image_paths:
        image = Image.open(image_path)
        w, h = image.size
        cx, cy = w // 2, h // 2
        fimg = retina_transform.foveat_img(np.array(image), [(cx, cy)])
        fimg = Image.fromarray(fimg)
        fimg.show()

def foveate_eyes(
        leye_search,
        reye_search,
        combined=True,
        save_imgs=False
    ):
    l_images = glob.glob(leye_search)
    r_images = glob.glob(reye_search)

    l_images.sort(), r_images.sort()

    for l_img_path, r_img_path in zip(l_images, r_images):
        l_img = Image.open(l_img_path)
        r_img = Image.open(r_img_path)

        lw, lh = l_img.size
        rw, rh = r_img.size
        lcx, lcy = lw // 2, lh // 2
        rcx, rcy = rw //2, rh // 2

        l_fimg = retina_transform.foveat_img(np.array(l_img), [(lcx, lcy)])
        r_fimg = retina_transform.foveat_img(np.array(r_img), [(rcx, rcy)])

        limg = Image.fromarray(l_fimg)
        rimg = Image.fromarray(r_fimg)

        if combined:
            c_img = combine_images(l_fimg, r_fimg)
            cimg = Image.fromarray(c_img)
            cimg.show()

        limg.show()
        rimg.show()

        if save_imgs:
            limg.save(l_img_path[:-4] + '_fove.jpg')
            rimg.save(r_img_path[:-4] + '_fove.jpg')
            if combined:
                folder_path, filename = os.path.split(os.path.abspath(l_img_path))
                numbering = re.findall(r'\d+', filename)[-1]
                cimg.save(folder_path + f'/combined_{int(numbering)}_fove.jpg')

        # input('Press enter')

def movement_detection_foveated(
        video_file,
        pix_change_thresh=0.5,
        write_video=False,
    ):
    vidcap = cv.VideoCapture(video_file)
    success, img = vidcap.read()
    prev = np.zeros(img[:, :, 0].shape)
    if write_video:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output = cv.VideoWriter(video_file[:-4] + '_fove.mp4', fourcc, 30, (img.shape[1], img.shape[0]))
    count = 0
    while success:
        while_start = time.time()
        # img_pil = Image.fromarray(img)
        # img_pil.show()

        w = img.shape[1]
        h = img.shape[0]
        cx, cy = w // 2, h // 2
        f_start_time = time.time()
        fimg = retina_transform.foveat_img(
            np.array(img),
            [(cx, cy)],
            sigma=0.8,
            p=15,
            k=10
        )
        ftime = time.time() - f_start_time

        # Difference
        grey = cv.cvtColor(fimg, cv.COLOR_BGR2GRAY) 
        diff = (grey - prev).astype(np.uint8)
        diff = cv.medianBlur(diff, 3)
        _, diff = cv.threshold(diff, int(pix_change_thresh*255), 255, cv.THRESH_BINARY)
        diff = cv.erode(diff, np.ones(20))

        cv.imshow('Original', cv.pyrDown(img, dstsize=(cx, cy)))
        cv.imshow('Foveated', cv.pyrDown(fimg, dstsize=(cx, cy)))
        cv.imshow('Difference', cv.pyrDown(diff, dstsize=(cx, cy)))

        # fimg_pil = Image.fromarray(fimg)
        # fimg_pil.show()

        cv.waitKey(1)
        if write_video:
            output.write(fimg)
        prev = grey
        success, img = vidcap.read()
        count += 1
        while_time = time.time() - while_start
        # print(f'Percentage of while loop function takes: {ftime/while_time}')
        # print(f'Function time: {ftime}')


    if write_video:
        output.release()



if __name__ == "__main__":
    main()