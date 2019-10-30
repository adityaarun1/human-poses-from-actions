from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import cv2
import ref
import matplotlib.pyplot as plt


def show2D(img, points, c):
    points = ((points.reshape(ref.nJoints, -1))).astype(np.int32)
    for j in range(ref.nJoints):
        cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
    for e in ref.edges:
        cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                 (points[e[1], 0], points[e[1], 1]), c, 2)
    return img


class Debugger(object):
    def __init__(self):
        self.plt = plt
        self.fig = self.plt.figure()
        self.ax = self.fig.add_subplot((111))
        self.ax.grid(False)

    def addImg(self, img, imgId='default'):
        self.imgs[imgId] = img.copy()

    def addPoint2D(self, point, c, imgId='default'):
        self.imgs[imgId] = show2D(self.imgs[imgId], point, c)

    def showImg(self, pause=False, imgId='default'):
        cv2.imshow('{}'.format(imgId), self.imgs[imgId])
        if pause:
            cv2.waitKey()

    def showAllImg(self, pause=False):
        for i, v in list(self.imgs.items()):
            cv2.imshow('{}'.format(i), v)
        if pause:
            cv2.waitKey()

    def saveImg(self, imgId='default', path='../debug/'):
        cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])

    def saveAllImg(self, path='../debug/'):
        for i, v in list(self.imgs.items()):
            cv2.imwrite(path + '/{}.png'.format(i), v)
