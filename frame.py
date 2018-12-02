import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import util, imageio, pickle, frame, scipy
from util import *


class Frame:
    def __init__(self):
        util.raiseNotDefined()

    def load_sketch(self, sketch):
        self.sketch = sketch

    def load_gray(self, gray):
        util.raiseNotDefined()

    def load_weight(self, Wn):
        self.Wn = Wn

    def neighbors(self, p):
        util.raiseNotDefined()

    def idx(self, p):
        util.raiseNotDefined()

    def weight(self, p):
        N = self.neighbors(p)
        idx = [self.idx(p) for p in N]
        ys = [self.Y[n] for n in N]
        S = np.std(ys)
        if (S <= 0):
            return np.zeros(shape=len(N)), idx
        else:
            return [np.exp(-1 * np.square(self.Y[tuple(p)] - y) / 2 / S / S) for y in ys], idx

    def build_weights_matrix(self):
        start_time = time.time()
        x, y = self.shape[:2]
        for i in range(x):
            for j in range(y):
                weights, idx = self.weight(i, j, self.Y)
                self.Wn[self.idx([i, j]), idx] = -1 * np.asmatrix(weights)
            if (i % 10 == 0):
                print(i, self.Y.shape[0], time.time() - start_time)
        self.Wn = normalize(self.Wn, norm='l1', axis=1).tolil()
        self.Wn[np.arange(x * y), np.arange(x * y)] = 1

    def color(self):
        util.raiseNotDefined()


class StaticFrame(Frame):
    def __init__(self, sketch, gray):
        self.shape = sketch.shape
        self.sketch = sketch
        self.gray = gray
        self.Y = np.array(gray[:, :, 0], dtype='float64')
        self.Wn = sparse.lil_matrix((self.shape[0] * self.shape[1], self.shape[0] * self.shape[1]))
        self.solution = np.zeros(shape=self.shape)

        colored = abs(self.sketch[:, :, 1]-self.gray[:, :, 1]) + abs(self.sketch[:, :, 2]-self.gray[:, :, 2]) > 0
        self.idx_marks = np.nonzero(colored)
        self.idx_marks = self.idx(self.idx_marks)

        white = (abs(self.sketch[:, :, 0]-np.ones(shape=(sketch.shape[:2]))) + abs(self.sketch[:, :, 1]) + abs(self.sketch[:, :, 2]))<1e-8
        self.idx_white = np.nonzero(white)
        self.idx_white = self.idx(self.idx_white)
        self.idx_white = [i for i in self.idx_white if i in self.idx_marks]

        self.idx_marks = [i for i in self.idx_marks if i not in self.idx_white]


    def load_weight(self, Wn):
        self.Wn = Wn

    def neighbors(self, p, d=3):
        i, j = p[0], p[1]
        x, y = self.Y.shape[:2]
        x1 = max(i - d, 0)
        x2 = min(i + d+1, x)
        y1 = max(j - d, 0)
        y2 = min(j + d+1, y)
        N = []
        for a in range(x1, x2):
            for b in range(y1, y2):
                if (a != i or b != j):
                    N.append(tuple([a, b]))
        return N

    def idx(self, p):
        return p[0]*self.shape[1]+p[1]

    def build_weights_matrix(self):
        print("Starting calculating weight matrix......")
        start_time = time.time()
        x, y = self.shape[:2]
        for i in range(x):
            for j in range(y):
                weights, idxthis = self.weight([i, j])
                self.Wn[self.idx([i, j]), idxthis] = -1 * np.asarray(weights)
            if (i % 10 == 0):
                print(i, self.Y.shape[0], time.time() - start_time)
        self.Wn = normalize(self.Wn, norm='l1', axis=1).tolil()
        self.Wn[np.arange(x * y), np.arange(x * y)] = 1

    def color(self):
        start_time = time.time()
        ## set rows in colored indices
        Wn = self.Wn.tocsc()

        for p in list(self.idx_marks):
            Wn[p] = sparse.csr_matrix(([1.0], ([0], [p])), shape=(1, self.shape[0]*self.shape[1]))
        for p in list(self.idx_white):
            Wn[p] = sparse.csr_matrix(([1.0], ([0], [p])), shape=(1, self.shape[0]*self.shape[1]))

        print("Finish adding colored to Wn {}".format(time.time() - start_time))


        b1 = np.zeros(shape=(self.shape[0]*self.shape[1]))
        b2 = np.zeros(shape=(self.shape[0]*self.shape[1]))
        b1[self.idx_marks] = (self.sketch[:, :, 1]).flatten()[self.idx_marks]
        b2[self.idx_marks] = (self.sketch[:, :, 2]).flatten()[self.idx_marks]
        b1[self.idx_white] = (self.gray[:, :, 1]).flatten()[self.idx_white]
        b2[self.idx_white] = (self.gray[:, :, 2]).flatten()[self.idx_white]

        x1 = sparse.linalg.spsolve(Wn, b1)
        x2 = sparse.linalg.spsolve(Wn, b2)

        print("Finish solving LU {}".format(time.time() - start_time))

        self.solution[:, :, 0] = self.Y
        self.solution[:, :, 1] = x1.reshape(self.shape[:2])
        self.solution[:, :, 2] = x2.reshape(self.shape[:2])

        return self.solution


class DynamicFrame(Frame):
    def __init__(self, sketch, gray, previous):
        self.shape = sketch.shape
        self.sketch = sketch
        self.gray = gray
        self.previous = previous
        self.Y = np.zeros(shape=(self.shape[0], self.shape[1]*2))
        self.Y[:, :self.shape[1]] = gray[:, :, 0]
        self.Y[:, self.shape[1]:] = self.previous.Y
        self.Wn = sparse.csc_matrix((self.shape[0] * self.shape[1] * 2, self.shape[0] * self.shape[1] * 2))
        self.Ix, self.Iy = np.gradient(self.Y)
        self.It = gray[:, :, 0] - self.previous.Y
        self.solution = np.zeros(shape=self.shape)

        colored = abs(self.sketch[:, :, 1] - self.gray[:, :, 1]) + abs(self.sketch[:, :, 2] - self.gray[:, :, 2]) > 0
        self.idx_marks = np.nonzero(colored)
        self.idx_marks = self.idx(self.idx_marks)
        white = (abs(self.sketch[:, :, 0] - np.ones(shape=(sketch.shape[:2]))) + abs(self.sketch[:, :, 1]) + abs(
            self.sketch[:, :, 2])) < 1e-8
        self.idx_white = np.nonzero(white)
        self.idx_white = self.idx(self.idx_white)
        self.idx_white = [i for i in self.idx_white if i in self.idx_marks]
        self.idx_marks = [i for i in self.idx_marks if i not in self.idx_white]

    def idx(self, p):
        return p[0]*self.shape[1]+p[1]

    def build_weights_matrix(self):
        start_time = time.time()
        print("call build_weights_matrix")
        x, y = self.shape[:2]
        xs, ys = self.previous.Wn.nonzero()
        self.Wn.tolil()
        self.previous.Wn.tolil()
        for i in range(x):
            self.Wn[i+x*y, x*y:] = self.previous.Wn[i, :]
        print("copy weights from previous {}".format(time.time()-start_time))
        for i in range(x):
            for j in range(y):
                weights, idxthis = self.weight([i, j])
                self.Wn[self.idx([i, j]), idxthis] = -1 * np.asarray(weights)
            if (i % 10 == 0):
                print(i, self.Y.shape[0], time.time() - start_time)
        self.Wn = normalize(self.Wn, norm='l1', axis=1).tolil()
        self.Wn[np.arange(x * y * 2), np.arange(x * y * 2)] = 1



    def velocity(self, p):
        x1 = max(0, p[0]-2)
        x2 = min(self.shape[0]-1, p[0]+2)
        y1 = max(0, p[1] - 2)
        y2 = min(self.shape[1] - 1, p[1] + 2)
        A = []
        b = []
        for i in range(x1, x2+1):
            for j in range(y1, y2+1):
                A.append([self.Ix[i, j], self.Iy[i, j]])
                b.append([self.It[i, j]])
        try:
            v = scipy.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
        except:
            return [0, 0]

        if(v[0]==None):
            v[0]=0
            print(p, v)
        if (v[1] == None):
            v[1] = 0
            print(p, v)
        return v

    def neighbors(self, p):
        i, j = p[0], p[1]
        x, y = self.shape[:2]
        x1 = max(i - 1, 0)
        x2 = min(i + 2, x)
        y1 = max(j - 1, 0)
        y2 = min(j + 2, y)
        N = []
        for a in range(x1, x2):
            for b in range(y1, y2):
                if (a != i or b != j):
                    N.append(tuple([a, b]))

        vx, vy = self.velocity(p)
        i, j = int(p[0]-vx), int(p[1]-vy)
        x1 = max(i - 1, 0)
        x2 = min(i + 2, x)
        y1 = max(j - 1, 0)
        y2 = min(j + 2, y)
        for a in range(x1, x2):
            for b in range(y1, y2):
                if (a != i or b != j):
                    N.append(tuple([a, y+b]))
        return N

    def color(self):
        start_time = time.time()
        ## set rows in colored indices
        Wn = self.Wn.tocsc()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (self.sketch[i, j, 1] != self.gray[i, j, 1] or self.sketch[i, j, 2] != self.gray[i, j, 2]):
                    Wn[self.idx([i, j])] = sparse.csr_matrix(([1.0], ([0], [self.idx([i, j])])), shape=(1, self.shape[0]*self.shape[1]*2))
        print("Finish adding colored to Wn {}".format(time.time() - start_time))

        print(self.sketch.shape)
        print(self.previous.solution.shape)
        b10 = np.zeros(shape=(self.shape[1]*self.shape[0]))
        b20 = np.zeros(shape=(self.shape[1]*self.shape[0]))

        b10[self.idx_marks] = (self.sketch[:, :, 1]).flatten()[self.idx_marks]
        b20[self.idx_marks] = (self.sketch[:, :, 2]).flatten()[self.idx_marks]
        b10[self.idx_white] = (self.gray[:, :, 1]).flatten()[self.idx_white]
        b20[self.idx_white] = (self.gray[:, :, 2]).flatten()[self.idx_white]

        b1 = np.concatenate((b10, self.previous.solution[:, :, 1].flatten()), axis=None)
        b2 = np.concatenate((b20, self.previous.solution[:, :, 2].flatten()), axis=None)
        print(b1.shape)

        x1 = sparse.linalg.spsolve(Wn, b1)[:self.shape[1]*self.shape[0]]
        x2 = sparse.linalg.spsolve(Wn, b2)[:self.shape[1]*self.shape[0]]

        print("Finish solving LU {}".format(time.time() - start_time))

        self.solution[:, :, 0] = self.Y[:, :self.shape[1]]
        self.solution[:, :, 1] = x1.reshape(self.shape[:2])
        self.solution[:, :, 2] = x2.reshape(self.shape[:2])
        self.Y = self.Y[:, :self.shape[1]]
        self.Wn = self.Wn[:self.shape[0]*self.shape[1], :self.shape[1]*self.shape[0]]
        return self.solution


if __name__ == "__main__":
    dir = "videos/butterfly/"
    start_time = time.time()
    origin_rgb = imageio.imread("{}gray/frame1.png".format(dir, str(31)))
    origin_yiq = rgb2yiq(origin_rgb)
    sketch_rgb = imageio.imread("{}sketch/frame1.png".format(dir, str(31)))
    sketch_yiq = rgb2yiq(sketch_rgb)
    Y = np.array(origin_yiq[:, :, 0], dtype='float64')

    s_origin_yiq = compress2(origin_yiq)
    s_sketch_yiq = compress2(sketch_yiq)

    with open("{}frame/dynamic_frame{}.pickle".format(dir, str(30)), "rb") as f:
        pre = pickle.load(f)

    curr = frame.DynamicFrame(s_sketch_yiq, s_origin_yiq, pre)
    print("Starting building Wn")
    curr.build_weights_matrix()
    print(curr.Wn.shape)
    sol = curr.color()




