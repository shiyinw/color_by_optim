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
        gray_dir = sys.argv[1]
        sketch_dir = sys.argv[2]
        output_dir = sys.argv[3]
        weight_dir = sys.argv[4]
    except:
        filename = "34"
        gray_dir = "images/gray_{}.png".format(filename)
        sketch_dir = "images/sketch_{}.png".format(filename)
        output_dir = "images/result_{}.png".format(filename)
        weight_dir = "images/weights/{}.pickle".format(filename)


    origin_rgb = imageio.imread(gray_dir)
    sketch_rgb = imageio.imread(sketch_dir)
    print(origin_rgb.shape)
    print(sketch_rgb.shape)
    sketch_rgb = sketch_rgb[:, :, :3]
    origin_rgb = origin_rgb[:, :, :3]
    assert origin_rgb.shape==sketch_rgb.shape, "The origin picture and sketch should have the same sizes."+str(sketch_rgb.shape)+str(origin_rgb.shape)

    origin_yiq = rgb2yiq(origin_rgb)
    sketch_yiq = rgb2yiq(sketch_rgb)
    Y = np.array(origin_yiq[:, :, 0], dtype='float64')

    s_origin_yiq = compress2(origin_yiq)
    s_sketch_yiq = compress2(sketch_yiq)
    print("Finish converting RGB to YIQ in time {}".format(time.time() - start_time))

    curFrame = frame.StaticFrame(s_sketch_yiq, s_origin_yiq)

    print("marks", curFrame.idx_marks)
    print("whites", curFrame.idx_white)

    try:
        with open(weight_dir, "rb") as f:
            Wn = pickle.load(f)
            curFrame.load_weight(Wn)
        print("Finish loading weight matrix {}".format(time.time() - start_time))
    except:
        curFrame.build_weights_matrix()
        with open(weight_dir, "wb") as f:
            pickle.dump(curFrame.Wn, f)
        print("Finish building weight matrix {}".format(time.time() - start_time))

    s_sol_yiq = curFrame.color()

    sol_yiq = decompress2(s_sol_yiq, origin_rgb.shape)
    sol_yiq[:, :, 0] = Y

    sol_rgb = yiq2rgb(sol_yiq)


    imageio.imsave(output_dir, sol_rgb)
    print("Finish all {}".format(time.time() - start_time))

