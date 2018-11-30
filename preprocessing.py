import sys, cv2

# from skimage.color import yiq2rgb, rgb2yiq
from util import yiq2rgb, rgb2yiq
import random
import numpy as np
import math



def generate(input_dir, sketch_dir, gray_dir, small=5000, big=3000):

    im_rgb = cv2.imread(input_dir)
    im_yiq = rgb2yiq(im_rgb)
    x, y = im_yiq.shape[:2]

    im_gray = im_yiq.copy()
    im_gray[:, :, 1:] = np.zeros(shape=(x, y, 2))
    im_yiq_mark = im_gray.copy()

    im_gray_output = yiq2rgb(im_gray)
    cv2.imwrite(gray_dir, im_gray_output)


    for i in range(small):
        d = int(math.sqrt(x * y) / 200)
        xi = random.randint(0, x - 1)
        yi = random.randint(0, y - 1)
        x1 = max(xi - d, 0)
        x2 = min(xi + d, x)
        y1 = max(yi - d, 0)
        y2 = min(yi + d, y)
        if(np.std(im_yiq[x1:x2, y1:y2, 1:2])<0.03):
            color1 = np.mean(im_yiq[x1:x2, y1:y2, 1])
            color2 = np.mean(im_yiq[x1:x2, y1:y2, 2])
            im_yiq_mark[x1:x2, y1:y2, 1] = 2 * color1 * np.ones(shape=(x2-x1, y2-y1))
            im_yiq_mark[x1:x2, y1:y2, 2] = 2 * color2 * np.ones(shape=(x2-x1, y2-y1))
    for i in range(big):
        d = int(math.sqrt(x * y) / 100)
        xi = random.randint(0, x-1)
        yi = random.randint(0, y-1)
        x1 = max(xi - d, 0)
        x2 = min(xi + d, x)
        y1 = max(yi - d, 0)
        y2 = min(yi + d, y)
        if (np.std(im_yiq[x1:x2, y1:y2, 1:2]) < 0.01):
            color1 = np.mean(im_yiq[x1:x2, y1:y2, 1])
            color2 = np.mean(im_yiq[x1:x2, y1:y2, 2])
            im_yiq_mark[x1:x2, y1:y2, 1] = 2 * color1 * np.ones(shape=(x2 - x1, y2 - y1))
            im_yiq_mark[x1:x2, y1:y2, 2] = 2 * color2 * np.ones(shape=(x2 - x1, y2 - y1))
    im_output = yiq2rgb(im_yiq_mark)
    cv2.imwrite(sketch_dir, im_output)


if __name__ == "__main__":
    try:
        imgname = sys.argv[1]
    except:
        imgname = "5"

    origin_dir = 'images/origin_{}.png'.format(imgname)
    sketch_dir = 'images/sketch_{}.png'.format(imgname)
    gray_dir = 'images/gray_{}.png'.format(imgname)

    generate(origin_dir, sketch_dir, gray_dir)