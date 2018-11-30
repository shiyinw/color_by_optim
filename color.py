import time
import imageio
from util import *
start_time = time.time()
import numpy as np
import sys
import pickle, os
import frame


# read and convert image


if __name__ == "__main__":

    try:
        filename = sys.argv[1]
    except:
        filename = "5"

    origin_rgb = imageio.imread("images/gray_{}.png".format(filename))
    sketch_rgb = imageio.imread("images/sketch_{}.png".format(filename))
    output_rgb = "images/result_{}.png".format(filename)
    weight_dir = "images/weights/"

    print(sketch_rgb.shape)
    print(origin_rgb.shape)

    print(output_rgb)
    origin_yiq = rgb2yiq(origin_rgb)
    sketch_yiq = rgb2yiq(sketch_rgb)
    Y = np.array(origin_yiq[:, :, 0], dtype='float64')

    s_origin_yiq = compress2(origin_yiq)
    s_sketch_yiq = compress2(sketch_yiq)
    print("Finish converting RGB to YIQ {}".format(time.time() - start_time))

    curFrame = frame.StaticFrame(s_sketch_yiq, s_origin_yiq)

    try:
        with open(weight_dir + "{}.pickle".format(filename), "rb") as f:
            Wn = pickle.load(f)
            curFrame.load_weight(Wn)
        print("Finish loading weight matrix {}".format(time.time() - start_time))
    except:
        curFrame.build_weights_matrix()
        with open(weight_dir + "{}.pickle".format(filename), "wb") as f:
            pickle.dump(curFrame.Wn, f)
        print("Finish building weight matrix {}".format(time.time() - start_time))

    s_sol_yiq = curFrame.color()

    sol_yiq = decompress2(s_sol_yiq, origin_rgb.shape)
    sol_yiq[:, :, 0] = Y

    sol_rgb = yiq2rgb(sol_yiq)


    imageio.imsave(output_rgb, sol_rgb)
    print("Finish all {}".format(time.time() - start_time))

