import imageio, os
from util import *
from skimage.color import yiq2rgb, rgb2yiq
import pickle
import time
from multiprocessing import Pool
import random
import math
import frame


dir = "videos/bird/"


def transfer(last_dir, gray_dir, sketch_dir, small=5000, big=3000):
    last_image_rgb = imageio.imread(last_dir)
    gray_rgb = imageio.imread(gray_dir)

    im_yiq = rgb2yiq(last_image_rgb)
    gray_yiq = rgb2yiq(gray_rgb)

    gray_yiq[:, :, 1:] = np.zeros(shape=(gray_yiq.shape[0], gray_yiq.shape[1], 2))
    im_yiq_mark = gray_yiq.copy()

    x, y = gray_yiq.shape[:2]
    d = int(math.sqrt(x * y) / 100)
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
            im_yiq_mark[x1:x2, y1:y2, 1] = color1 * np.ones(shape=(x2-x1, y2-y1))
            im_yiq_mark[x1:x2, y1:y2, 2] = color2 * np.ones(shape=(x2-x1, y2-y1))
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
            im_yiq_mark[x1:x2, y1:y2, 1] = color1 * np.ones(shape=(x2 - x1, y2 - y1))
            im_yiq_mark[x1:x2, y1:y2, 2] = color2 * np.ones(shape=(x2 - x1, y2 - y1))

    sumI = np.sum(im_yiq[:, :, 1])
    sumQ = np.sum(im_yiq[:, :, 2])

    gray_rgb = yiq2rgb(gray_yiq)
    im_output = yiq2rgb(im_yiq_mark)
    imageio.imwrite(gray_dir, gray_rgb)
    imageio.imwrite(sketch_dir, im_output)
    return sumI, sumQ

def run(i, sumI=None, sumQ=None):
    #if(not os.path.exists("{}seq_result/frame{}.png".format(dir, str(i)))):
    if(True):
        start_time = time.time()
        origin_rgb = imageio.imread("{}gray/frame{}.png".format(dir, str(i)))
        origin_yiq = rgb2yiq(origin_rgb)
        sketch_rgb = imageio.imread("{}sketch/frame{}.png".format(dir, str(i)))
        sketch_yiq = rgb2yiq(sketch_rgb)
        Y = np.array(origin_yiq[:, :, 0], dtype='float64')

        s_origin_yiq = compress2(origin_yiq)
        s_sketch_yiq = compress2(sketch_yiq)

        curFrame = frame.StaticFrame(s_sketch_yiq, s_origin_yiq)
        if (not os.path.exists("{}weight/{}.pickle".format(dir, str(i)))):
            curFrame.build_weights_matrix()
            with open("{}weight/{}.pickle".format(dir, str(i)), "wb") as f:
                pickle.dump(curFrame.Wn, f)
        else:
            with open("{}weight/{}.pickle".format(dir, str(i)), "rb") as f:
                curFrame.Wn = pickle.load(f)

        print("Finish loading weights of {}".format(str(i)))

        s_sol_yiq = curFrame.color()

        sol_yiq = decompress2(s_sol_yiq, origin_rgb.shape)
        sol_yiq[:, :, 0] = Y
        if(sumI!=None):
            scaleI = sumI / np.sum(sol_yiq[:, :, 1])
            sol_yiq[:, :, 1] = sol_yiq[:, :, 1] * scaleI

        if (sumQ != None):
            scaleI = sumQ / np.sum(sol_yiq[:, :, 2])
            sol_yiq[:, :, 2] = sol_yiq[:, :, 2] * scaleI

        sol_rgb = yiq2rgb(sol_yiq)

        if SEQUENTIAL:
            imageio.imsave("{}seq_result/frame{}.png".format(dir, str(i)), sol_rgb)
        else:
            imageio.imsave("{}result/frame{}.png".format(dir, str(i)), sol_rgb)
        print("Finish all {} {}".format(str(i), time.time() - start_time))
    else:
        print("Result of {}result/frame{}.png exists.".format(dir, str(i)))


def seq_run(t):
    run(t*10)
    for i in range(10*t + 1, 10*t + 10):
        sumI, sumQ = transfer("{}seq_result/frame{}.png".format(dir, str(i - 1)),
                              "{}gray/frame{}.png".format(dir, str(i)),
                              "{}sketch/frame{}.png".format(dir, str(i)))
        print("Generated sketch of frame {}".format(i))
        run(i, sumI, sumQ)


if __name__ == '__main__':
    pool = Pool(10)                     # Create a multiprocessing Pool
    if (not os.path.exists("{}result".format(dir))):
        os.mkdir("{}result".format(dir))
    if (not os.path.exists("{}weight".format(dir))):
        os.mkdir("{}weight".format(dir))

    pool.map(seq_run, range(18))


