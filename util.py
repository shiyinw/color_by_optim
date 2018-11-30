import numpy as np
import inspect, sys


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]
    print ("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def compress2(image):
    x , y, z = image.shape
    x2 = int((x+1)/2)
    y2 = int((y+1)/2)
    smaller = np.zeros(shape=(x2, y2, z))
    for i in range(x2):
        for j in range(y2):
            smaller[i, j, :] = image[i*2, j*2, :]
    return smaller

def decompress2(image, shape):
    x, y, z = image.shape
    x2, y2 = shape[:2]
    larger = np.zeros(shape=shape)
    x = x-1
    y = y-1
    for i in range(x2):
        for j in range(y2):
            i = int(i)
            j = int(j)
            if(i%2==0 and j%2==0):
                larger[i, j, :] = image[int(i/2), int(j/2), :]
            elif(i%2!=0 and j%2==0):
                larger[i, j, :] = (image[int(i/2), int(j/2), :] + image[min(int(i/2) + 1, x), int(j/2), :])/2
            elif (i % 2 == 0 and j % 2 != 0):
                larger[i, j, :] = (image[int(i/2), int(j/2), :] + image[int(i/2), min(int(j/2)+1, y), :]) / 2
            elif (i % 2 != 0 and j % 2 != 0):
                larger[i, j, :] = (image[int(i/2), int(j/2), :] + image[min(int(i/2) + 1, x), int(j/2), :] +
                                   image[int(i/2), min(int(j/2)+1, y), :] + image[min(int(i/2) + 1, x), min(int(j/2)+1, y), :]) / 4
            else:
                assert 1
    return larger


def rgb2yiq(rgb):
    rgb = rgb / 255.0
    y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.144])),             0,   1)
    i = np.clip(np.dot(rgb, np.array([0.595716, -0.274453, -0.321263])), -0.5957, 0.5957)
    q = np.clip(np.dot(rgb, np.array([0.211456, -0.522591, 0.311135])),  -0.5226, 0.5226)
    yiq = rgb
    yiq[..., 0] = y
    yiq[..., 1] = i
    yiq[..., 2] = q
    return yiq


def yiq2rgb(yiq):
    r = np.dot(yiq, np.array([1.0,  0.956295719758948,  0.621024416465261]))
    g = np.dot(yiq, np.array([1.0, -0.272122099318510, -0.647380596825695]))
    b = np.dot(yiq, np.array([1.0, -1.106989016736491,  1.704614998364648]))
    rgb = yiq
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    out = np.clip(rgb, 0.0, 1.0) * 255.0
    out = out.astype(np.uint8)
    return out