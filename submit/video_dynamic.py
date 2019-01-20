import imageio, os
from util import *
import pickle
import time
from multiprocessing import Pool
import frame


try:
    dir = sys.argv[1]
except:
    dir = "videos/eye/"

def run(i, sumI=None, sumQ=None):
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

    imageio.imsave("{}dynamic_result/frame{}.png".format(dir, str(i)), sol_rgb)

    print("Finish all {} {}".format(str(i), time.time() - start_time))

    return curFrame


def seq_run(t):
    if(os.path.exists("{}frame/dynamic_frame{}.pickle".format(dir, str(t*10)))):
        with open("{}frame/dynamic_frame{}.pickle".format(dir, str(t * 10)), "rb") as f:
            pre = pickle.load(f)
        print("Using existing frame {}".format(str(t*10)))
    else:
        pre = run(t*10)
        with open("{}frame/dynamic_frame{}.pickle".format(dir, str(t*10)), "wb") as f:
            pickle.dump(pre, f)
        print("Creating and saving frame {}".format(str(t * 10)))
    for i in range(10*t + 1, 10*t + 10):
        origin_rgb = imageio.imread("{}gray/frame{}.png".format(dir, str(i)))
        origin_yiq = rgb2yiq(origin_rgb)
        sketch_rgb = imageio.imread("{}sketch/frame{}.png".format(dir, str(i)))
        sketch_yiq = rgb2yiq(sketch_rgb)
        Y = np.array(origin_yiq[:, :, 0], dtype='float64')
        s_origin_yiq = compress2(origin_yiq)
        s_sketch_yiq = compress2(sketch_yiq)

        curr = frame.DynamicFrame(s_sketch_yiq, s_origin_yiq, pre)
        curr.build_weights_matrix()
        s_sol_yiq = curr.color()

        sol_yiq = decompress2(s_sol_yiq, origin_rgb.shape)
        sol_yiq[:, :, 0] = Y
        sol_rgb = yiq2rgb(sol_yiq)

        imageio.imsave("{}dynamic_result/frame{}.png".format(dir, str(i)), sol_rgb)
        with open("{}frame/dynamic_frame{}.pickle".format(dir, str(i)), "wb") as f:
            pickle.dump(curr, f)
        pre = curr

if __name__ == '__main__':
    if (not os.path.exists("{}dynamic_result".format(dir))):
        os.mkdir("{}dynamic_result".format(dir))
    if (not os.path.exists("{}weight".format(dir))):
        os.mkdir("{}weight".format(dir))
    if (not os.path.exists("{}frame".format(dir))):
        os.mkdir("{}frame".format(dir))

    pool = Pool(18)                     # Create a multiprocessing Pool
    pool.map(seq_run, range(18))